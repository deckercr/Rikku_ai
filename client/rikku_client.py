import requests
import base64
import time
import subprocess
import json
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write
import io
import tempfile
import os

# Optional imports for capture functionality
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

# Optional whisper import
try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False

# --- AUDIO CONFIG ---
WHISPER_MODEL_SIZE = "base.en"  # Options: tiny.en, base.en, small.en, medium.en, large
SAMPLE_RATE = 16000
RECORD_DURATION = 5  # Max record time per segment in seconds

# Initialize Whisper model (lazy loading)
stt_model = None


def get_input_device():
    """Find the best input device: Blue Yeti if available, otherwise internal mic.
    Returns (device_index, sample_rate) tuple.
    """
    devices = sd.query_devices()
    blue_yeti = None
    internal_mic = None

    for i, dev in enumerate(devices):
        if dev['max_input_channels'] > 0:  # Has input capability
            name = dev['name'].lower()
            # Prefer ALSA devices (hw:X,X) over JACK for direct access
            if 'blue' in name and 'hw:' in name:
                blue_yeti = i
            elif 'alc256' in name:
                internal_mic = i

    if blue_yeti is not None:
        dev_info = sd.query_devices(blue_yeti)
        rate = int(dev_info['default_samplerate'])
        print(f"Using Blue Yeti (device {blue_yeti}, {rate} Hz)")
        return blue_yeti, rate
    elif internal_mic is not None:
        dev_info = sd.query_devices(internal_mic)
        rate = int(dev_info['default_samplerate'])
        print(f"Using internal mic (device {internal_mic}, {rate} Hz)")
        return internal_mic, rate
    else:
        print("Using default input device")
        return None, SAMPLE_RATE


def load_whisper_model():
    """Load Whisper model on first use."""
    global stt_model
    if stt_model is None and WHISPER_AVAILABLE:
        print("Loading Whisper model...")
        stt_model = whisper.load_model(WHISPER_MODEL_SIZE)
        print("Whisper model loaded.")
    return stt_model

def get_preferred_monitor():
    """Get preferred monitor: DP-11 if available, otherwise eDP-2."""
    try:
        result = subprocess.run(
            ["hyprctl", "monitors", "-j"],
            capture_output=True, text=True, timeout=5
        )
        monitors = json.loads(result.stdout)
        monitor_names = [m["name"] for m in monitors]
        if "DP-11" in monitor_names:
            return "DP-11"
        if "eDP-2" in monitor_names:
            return "eDP-2"
        # Fallback to first available monitor
        return monitor_names[0] if monitor_names else None
    except Exception:
        return None

def hyprshot_available():
    """Check if hyprshot is available."""
    try:
        subprocess.run(["which", "hyprshot"], capture_output=True, check=True)
        return True
    except Exception:
        return False

HYPRSHOT_AVAILABLE = hyprshot_available()

# Your server's IP
CHAT_URL = "http://10.0.0.5:8080/api/chat/"


def capture_vision(capture_type="webcam"):
    if capture_type == "webcam":
        if not CV2_AVAILABLE:
            print("Error: OpenCV not available for webcam capture")
            return None
        cam = cv2.VideoCapture(1)
        time.sleep(0.5)  # Let camera warm up
        # Discard a few frames to allow auto-exposure to adjust
        for _ in range(5):
            cam.read()
        ret, frame = cam.read()
        if ret:
            _, buffer = cv2.imencode('.jpg', frame)
            content = base64.b64encode(buffer).decode('utf-8')
            cam.release()
            return content
        cam.release()
        print("Error: Failed to capture from webcam")
        return None
    elif capture_type == "screenshot":
        if not HYPRSHOT_AVAILABLE:
            print("Error: hyprshot not available for screenshots")
            return None
        monitor = get_preferred_monitor()
        if not monitor:
            print("Error: No monitors found")
            return None
        print(f"Capturing monitor: {monitor}")
        try:
            result = subprocess.run(
                ["hyprshot", "-m", "output", "-m", monitor, "-r", "--silent"],
                capture_output=True, timeout=10
            )
            if result.stdout:
                return base64.b64encode(result.stdout).decode('utf-8')
            print(f"Error: hyprshot produced no output (code {result.returncode})")
            return None
        except subprocess.TimeoutExpired:
            print("Error: Screenshot timed out")
            return None
    return None


def record_audio(device=None, samplerate=SAMPLE_RATE):
    """Records audio from the mic for RECORD_DURATION seconds."""
    print("[Listening...]")
    recording = sd.rec(
        int(RECORD_DURATION * samplerate),
        samplerate=samplerate,
        channels=1,
        dtype='float32',
        device=device
    )
    sd.wait()
    return recording.flatten(), samplerate


def transcribe_audio(audio_data, samplerate=SAMPLE_RATE):
    """Converts audio array to text using Whisper."""
    model = load_whisper_model()
    if model is None:
        print("Error: Whisper not available")
        return ""

    # Write audio to a temporary WAV file (whisper requires a file)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        temp_path = f.name
        # Convert float32 to int16 for WAV
        audio_int16 = (audio_data * 32767).astype(np.int16)
        write(f.name, samplerate, audio_int16)

    try:
        # Whisper will resample internally to 16kHz
        result = model.transcribe(temp_path, fp16=False)
        text = result["text"].strip()
        return text
    finally:
        os.unlink(temp_path)


def send_to_rikku(prompt, vision_type=None):
    """Send prompt to Rikku server with optional vision."""
    if not prompt:
        return None

    # Auto-trigger vision based on keywords
    if vision_type is None:
        lower_prompt = prompt.lower()
        if any(word in lower_prompt for word in ["look", "webcam", "see me", "camera"]):
            vision_type = "webcam"
        elif any(word in lower_prompt for word in ["screen", "screenshot", "this", "display"]):
            vision_type = "screenshot"

    image_data = None
    if vision_type:
        print(f"Capturing {vision_type}...")
        image_data = capture_vision(vision_type)
        if image_data:
            print(f"Captured image: {len(image_data)} bytes (base64)")
        else:
            print("Continuing without image...")

    payload = {
        "prompt": prompt,
        "image": image_data
    }

    try:
        response = requests.post(CHAT_URL, json=payload, timeout=60)
        response.raise_for_status()
        reply = response.json().get('response')
        print("Rikku:", reply)
        return reply
    except requests.exceptions.RequestException as e:
        print(f"Error connecting to Rikku: {e}")
        return None


if __name__ == "__main__":
    print("Rikku Voice Client Active.")
    print(f"  Webcam: {CV2_AVAILABLE}")
    print(f"  Screenshot: {HYPRSHOT_AVAILABLE}")
    print(f"  Whisper STT: {WHISPER_AVAILABLE}")

    # Detect input device
    input_device, device_samplerate = get_input_device()

    if WHISPER_AVAILABLE:
        # Pre-load the model
        load_whisper_model()

    print("\nPress Enter to talk, Ctrl+C to quit.")

    try:
        while True:
            input("\n[Press Enter to Talk]")

            # 1. Record audio
            audio_buffer, rec_rate = record_audio(device=input_device, samplerate=device_samplerate)

            # 2. Transcribe
            text_query = transcribe_audio(audio_buffer, samplerate=rec_rate)

            if text_query:
                print(f"You: {text_query}")
                # 3. Send to Rikku
                send_to_rikku(text_query)
            else:
                print("... I didn't catch that.")

    except KeyboardInterrupt:
        print("\nClosing Rikku Client.")
