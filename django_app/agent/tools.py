import datetime
import os

# Optional imports for hardware-dependent tools
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    import pyautogui
    PYAUTOGUI_AVAILABLE = True
except (ImportError, KeyError):
    # KeyError happens when DISPLAY env var is missing
    PYAUTOGUI_AVAILABLE = False

TOOL_REGISTRY = {}


def register_tool(name, description, parameters):
    def decorator(func):
        TOOL_REGISTRY[name] = {
            "name": name,
            "description": description,
            "parameters": parameters,
            "callable": func
        }
        return func
    return decorator


@register_tool(
    name="get_current_time",
    description="Returns the current time and date on the server.",
    parameters={"type": "object", "properties": {}}
)
def get_current_time():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@register_tool(
    name="take_screenshot",
    description="Captures the current screen of the host machine.",
    parameters={"type": "object", "properties": {}}
)
def take_screenshot():
    if not PYAUTOGUI_AVAILABLE:
        return "Screenshot unavailable: No display connected (headless server)."

    os.makedirs('media/screenshots', exist_ok=True)
    path = f"media/screenshots/screen_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    screenshot = pyautogui.screenshot()
    screenshot.save(path)
    return f"Screenshot saved at {path}. I can see your screen now!"


@register_tool(
    name="capture_webcam",
    description="Takes a photo from the webcam.",
    parameters={"type": "object", "properties": {}}
)
def capture_webcam():
    if not CV2_AVAILABLE:
        return "Webcam unavailable: OpenCV not installed."

    os.makedirs('media/webcam', exist_ok=True)
    path = f"media/webcam/photo_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"

    cam = cv2.VideoCapture(0)
    ret, frame = cam.read()
    if ret:
        cv2.imwrite(path, frame)
        cam.release()
        return f"Photo captured at {path}. I see you!"
    cam.release()
    return "Failed to access webcam. Device may not be connected."
