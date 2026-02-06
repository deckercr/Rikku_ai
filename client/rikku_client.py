import argparse
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
import re
from dotenv import load_dotenv

# Load .env from project root (one level up from client/)
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

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

# --- GROQ STT CONFIG ---
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_STT_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
GROQ_STT_MODEL = "whisper-large-v3-turbo"  # Fast + accurate

# --- AUDIO CONFIG ---
WHISPER_MODEL_SIZE = "small.en"  # Local fallback: tiny.en, base.en, small.en, medium.en, large
SAMPLE_RATE = 16000
MAX_RECORD_DURATION = 30  # Max record time in seconds (safety cap)
SILENCE_THRESHOLD = None  # Auto-calibrated at startup; set manually to override
SILENCE_DURATION = 1.5  # Seconds of silence before stopping
CHUNK_DURATION = 0.1  # Size of each audio chunk in seconds

# --- WAKE WORD CONFIG ---
# Multiple variations to handle Whisper's transcription of "Rikku"
WAKE_WORDS = [
    "rikku", "riku", "rico", "reeku", "ricku", "rikou", "reekou",
    "riku's", "rikku's", "rico's", "reeko", "ricco", "riko",
    "reku", "reiku", "ryku", "ryko", "roku", "riku,",
    "rika", "rika's", "rikka", "rekka",
]
CANONICAL_NAME = "Rikku"  # The correct name to substitute for wake word variants
ACTIVE_LISTEN_DURATION = 60  # Seconds to stay active after wake word

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


def calibrate_silence(device=None, samplerate=SAMPLE_RATE, duration=2):
    """Record ambient noise and set silence threshold automatically."""
    global SILENCE_THRESHOLD
    if SILENCE_THRESHOLD is not None:
        print(f"Using manual silence threshold: {SILENCE_THRESHOLD}")
        return

    print("Calibrating silence... (stay quiet for 2 seconds)")
    ambient = sd.rec(
        int(duration * samplerate),
        samplerate=samplerate,
        channels=1,
        dtype='float32',
        device=device
    )
    sd.wait()
    ambient = ambient.flatten()

    # Measure the noise floor RMS, then set threshold above it
    ambient_rms = np.sqrt(np.mean(ambient ** 2))
    SILENCE_THRESHOLD = ambient_rms * 3
    print(f"Ambient noise RMS: {ambient_rms:.4f}, silence threshold set to: {SILENCE_THRESHOLD:.4f}")


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
SERVER_BASE = "http://10.0.0.5:8080"
CHAT_URL = f"{SERVER_BASE}/api/chat/"
ENROLL_URL = f"{SERVER_BASE}/api/identity/enroll/"
IDENTIFY_URL = f"{SERVER_BASE}/api/identity/identify/"
PROFILES_URL = f"{SERVER_BASE}/api/identity/profiles/"


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
    """Records audio with voice activity detection.
    Waits for speech, then records until silence is detected.
    """
    chunk_samples = int(CHUNK_DURATION * samplerate)
    max_chunks = int(MAX_RECORD_DURATION / CHUNK_DURATION)
    silence_chunks_needed = int(SILENCE_DURATION / CHUNK_DURATION)

    chunks = []
    silence_count = 0
    speech_detected = False

    print("[Listening...]")

    for _ in range(max_chunks):
        chunk = sd.rec(
            chunk_samples,
            samplerate=samplerate,
            channels=1,
            dtype='float32',
            device=device
        )
        sd.wait()
        chunk = chunk.flatten()
        chunks.append(chunk)

        # Measure energy (RMS)
        rms = np.sqrt(np.mean(chunk ** 2))

        if rms >= SILENCE_THRESHOLD:
            speech_detected = True
            silence_count = 0
        else:
            if speech_detected:
                silence_count += 1

        # Stop after sustained silence following speech
        if speech_detected and silence_count >= silence_chunks_needed:
            break

    if not chunks:
        return np.array([], dtype='float32'), samplerate

    return np.concatenate(chunks), samplerate


def audio_to_wav_bytes(audio_data, samplerate):
    """Convert float32 audio array to WAV bytes in memory."""
    audio_int16 = (audio_data * 32767).astype(np.int16)
    buf = io.BytesIO()
    write(buf, samplerate, audio_int16)
    buf.seek(0)
    return buf.read()


def transcribe_audio_groq(wav_bytes):
    """Transcribe audio using Groq's Whisper API."""
    response = requests.post(
        GROQ_STT_URL,
        headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
        files={"file": ("audio.wav", wav_bytes, "audio/wav")},
        data={
            "model": GROQ_STT_MODEL,
            "language": "en",
            "response_format": "json",
        },
        timeout=15,
    )
    response.raise_for_status()
    return response.json().get("text", "").strip()


def transcribe_audio_local(temp_path):
    """Transcribe audio using local Whisper model."""
    model = load_whisper_model()
    if model is None:
        return ""
    result = model.transcribe(temp_path, fp16=False)
    return result["text"].strip()


def transcribe_audio(audio_data, samplerate=SAMPLE_RATE):
    """Converts audio to text. Uses Groq API if available, falls back to local Whisper."""
    wav_bytes = audio_to_wav_bytes(audio_data, samplerate)

    # Try Groq API first
    if GROQ_API_KEY:
        try:
            text = transcribe_audio_groq(wav_bytes)
            return text
        except Exception as e:
            print(f"Groq STT failed, falling back to local: {e}")

    # Fall back to local Whisper
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(wav_bytes)
        temp_path = f.name

    try:
        return transcribe_audio_local(temp_path)
    finally:
        os.unlink(temp_path)


def contains_wake_word(text):
    """Check if text contains any wake word variant.
    Returns (found, corrected_text) where any matched variant is replaced with CANONICAL_NAME.
    """
    if not text:
        return False, text

    lower_text = text.lower()
    for wake_word in WAKE_WORDS:
        if wake_word in lower_text:
            # Replace the variant with the correct name (case-insensitive)
            corrected = re.sub(re.escape(wake_word), CANONICAL_NAME, text, count=1, flags=re.IGNORECASE)
            return True, corrected
    return False, text


def is_active_listening(last_wake_time):
    """Check if we're still in the active listening window."""
    if last_wake_time is None:
        return False
    return (time.time() - last_wake_time) < ACTIVE_LISTEN_DURATION


def get_remaining_active_time(last_wake_time):
    """Get remaining seconds in active listening mode."""
    if last_wake_time is None:
        return 0
    remaining = ACTIVE_LISTEN_DURATION - (time.time() - last_wake_time)
    return max(0, int(remaining))


def enroll_face(name):
    """Capture a webcam frame and enroll the face under the given name."""
    if not CV2_AVAILABLE:
        print("Error: OpenCV required for face enrollment")
        return False

    print(f"Capturing face for enrollment as '{name}'...")
    image_data = capture_vision("webcam")
    if not image_data:
        print("Error: Failed to capture webcam image")
        return False

    try:
        response = requests.post(
            ENROLL_URL,
            json={"name": name, "image": image_data},
            timeout=30,
        )
        response.raise_for_status()
        result = response.json()
        print(f"Enrolled: {result.get('user')} "
              f"(face #{result.get('face_id')}, "
              f"{result.get('total_embeddings')} total embeddings)")
        return True
    except requests.exceptions.RequestException as e:
        print(f"Enrollment failed: {e}")
        try:
            print(f"  Detail: {response.json().get('error', '')}")
        except Exception:
            pass
        return False


def list_profiles():
    """List all enrolled user profiles."""
    try:
        response = requests.get(PROFILES_URL, timeout=10)
        response.raise_for_status()
        profiles = response.json().get("profiles", [])
        if not profiles:
            print("No enrolled profiles.")
            return
        print("Enrolled profiles:")
        for p in profiles:
            print(f"  [{p['id']}] {p['name']} — {p['embeddings']} face(s)")
    except requests.exceptions.RequestException as e:
        print(f"Error listing profiles: {e}")


def send_to_rikku(prompt, vision_type=None, auto_identify=False):
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

    # In auto-identify mode, always include a webcam frame so Rikku can identify you
    if auto_identify and image_data is None and CV2_AVAILABLE:
        image_data = capture_vision("webcam")
        if image_data:
            print("(auto-identify: webcam frame attached)")

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
    parser = argparse.ArgumentParser(description="Rikku Voice Client")
    parser.add_argument(
        "--enroll", metavar="NAME",
        help="Enroll your face under the given name (captures from webcam)"
    )
    parser.add_argument(
        "--profiles", action="store_true",
        help="List all enrolled user profiles"
    )
    parser.add_argument(
        "--no-identify", action="store_true",
        help="Disable auto-identification (webcam frame not sent with every request)"
    )
    args = parser.parse_args()

    auto_identify = not args.no_identify

    # Handle one-shot commands
    if args.enroll:
        enroll_face(args.enroll)
        exit(0)

    if args.profiles:
        list_profiles()
        exit(0)

    # Normal voice client mode
    print("Rikku Voice Client Active.")
    print(f"  Webcam: {CV2_AVAILABLE}")
    print(f"  Screenshot: {HYPRSHOT_AVAILABLE}")
    print(f"  Groq STT: {'configured' if GROQ_API_KEY else 'no API key (set GROQ_API_KEY)'}")
    print(f"  Local Whisper fallback: {WHISPER_AVAILABLE}")
    print(f"  Auto-Identify: {'ON' if auto_identify else 'OFF'}")
    print(f"  Wake words: {', '.join(WAKE_WORDS[:5])}...")

    # Detect input device
    input_device, device_samplerate = get_input_device()

    if not GROQ_API_KEY and WHISPER_AVAILABLE:
        # Only pre-load local model if Groq isn't configured
        load_whisper_model()

    # Calibrate silence threshold for this mic/environment
    calibrate_silence(device=input_device, samplerate=device_samplerate)

    print("\nListening for wake word 'Rikku'... (Ctrl+C to quit)")

    # Track when we last heard the wake word
    last_wake_time = None

    try:
        while True:
            # Show status
            if is_active_listening(last_wake_time):
                remaining = get_remaining_active_time(last_wake_time)
                print(f"\n[Active listening: {remaining}s remaining]")
            else:
                print("\n[Waiting for wake word...]")

            # 1. Record audio
            audio_buffer, rec_rate = record_audio(device=input_device, samplerate=device_samplerate)

            # 2. Transcribe
            text_query = transcribe_audio(audio_buffer, samplerate=rec_rate)

            if not text_query:
                continue

            # Check for wake word or active listening mode
            has_wake_word, processed_text = contains_wake_word(text_query)

            if has_wake_word:
                # Wake word detected - activate and process the full sentence
                last_wake_time = time.time()
                print(f"[Wake word detected!]")
                print(f"You: {processed_text}")
                send_to_rikku(processed_text, auto_identify=auto_identify)

            elif is_active_listening(last_wake_time):
                # In active listening mode - process without wake word
                print(f"You: {text_query}")
                send_to_rikku(text_query, auto_identify=auto_identify)
                # Reset the timer on each interaction to keep conversation going
                last_wake_time = time.time()

            else:
                # Not active and no wake word - ignore (but show for debugging)
                print(f"(Ignored: {text_query})")

    except KeyboardInterrupt:
        print("\nClosing Rikku Client.")
