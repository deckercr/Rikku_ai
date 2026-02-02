import requests
import base64

# Optional imports for capture functionality
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    import pyautogui
    PYAUTOGUI_AVAILABLE = True
except Exception:
    # Can fail due to missing DISPLAY, Xauthority, or import issues
    PYAUTOGUI_AVAILABLE = False

# Your server's IP
CHAT_URL = "http://10.0.0.5:8080/api/chat/"


def capture_vision(capture_type="webcam"):
    if capture_type == "webcam":
        if not CV2_AVAILABLE:
            print("Error: OpenCV not available for webcam capture")
            return None
        cam = cv2.VideoCapture(0)
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
        if not PYAUTOGUI_AVAILABLE:
            print("Error: pyautogui not available for screenshots")
            return None
        screenshot = pyautogui.screenshot()
        import io
        buffer = io.BytesIO()
        screenshot.save(buffer, format='PNG')
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    return None


def send_to_rikku(prompt, vision_type=None):
    image_data = None
    if vision_type:
        print(f"Capturing {vision_type}...")
        image_data = capture_vision(vision_type)
        if image_data is None and vision_type:
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
    print(f"Screenshot available: {PYAUTOGUI_AVAILABLE}")
    print()

    # Example usage:
    send_to_rikku("Hey Rikku, what time is it?")
    # send_to_rikku("Look at me through my webcam!", vision_type="webcam")
    # send_to_rikku("What's on my screen?", vision_type="screenshot")
