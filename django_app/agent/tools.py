import datetime
import os
from zoneinfo import ZoneInfo

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
    tz_name = os.environ.get("TZ", "America/Chicago")
    tz = ZoneInfo(tz_name)
    now = datetime.datetime.now(tz)
    return now.strftime("%Y-%m-%d %H:%M:%S %Z")


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


@register_tool(
    name="enroll_user",
    description=(
        "Enrolls a new user's face for recognition. Use this when you see someone "
        "you don't recognize and they have told you their name. "
        "Usage: [TOOL: enroll_user:TheirName]"
    ),
    parameters={"type": "object", "properties": {"name": {"type": "string"}}}
)
def enroll_user(args=None):
    from .identity import get_current_image, enroll_face

    if not args or not args.strip():
        return "Error: No name provided. Usage: [TOOL: enroll_user:TheirName]"

    name = args.strip()
    image_b64 = get_current_image()
    if not image_b64:
        return "Error: No webcam image available to enroll from."

    try:
        profile, _ = enroll_face(name, image_b64)
        count = profile.face_embeddings.count()
        return (
            f"Successfully enrolled {name}! "
            f"I now have {count} face reference(s) for them. "
            f"I will recognize {name} from now on."
        )
    except ValueError as e:
        return f"Enrollment failed: {e}"


@register_tool(
    name="rename_user",
    description=(
        "Renames an enrolled user's profile. Use this when someone tells you "
        "their name is wrong or they want to be called something different. "
        "Usage: [TOOL: rename_user:OldName>NewName]"
    ),
    parameters={"type": "object", "properties": {"names": {"type": "string"}}}
)
def rename_user(args=None):
    from .models import UserProfile

    if not args or ">" not in args:
        return "Error: Usage: [TOOL: rename_user:OldName>NewName]"

    old_name, new_name = args.split(">", 1)
    old_name = old_name.strip()
    new_name = new_name.strip()

    if not old_name or not new_name:
        return "Error: Both old and new names are required."

    try:
        profile = UserProfile.objects.get(name=old_name)
        profile.name = new_name
        profile.save()
        return f"Done! Renamed '{old_name}' to '{new_name}'. I'll call them {new_name} from now on."
    except UserProfile.DoesNotExist:
        return f"Error: No profile found with the name '{old_name}'."
