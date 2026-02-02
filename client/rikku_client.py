import requests
import base64
import time
import subprocess
import json

# Optional imports for capture functionality
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

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


def send_to_rikku(prompt, vision_type=None):
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
        print("Rikku:", response.json().get('response'))
    except requests.exceptions.RequestException as e:
        print(f"Error connecting to Rikku: {e}")


if __name__ == "__main__":
    # Check what's available
    print(f"Webcam available: {CV2_AVAILABLE}")
    print(f"Screenshot available: {HYPRSHOT_AVAILABLE}")
    print()

    # Example usage:
    send_to_rikku("Hey Rikku, what time is it?")
#    send_to_rikku("Look at me through my webcam!", vision_type="webcam")
    send_to_rikku("What's on my screen?", vision_type="screenshot")
