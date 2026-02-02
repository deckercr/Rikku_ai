import cv2
import pyautogui
import requests
import base64
import time

# Your 3090's Tailscale IP
REMOTE_URL = "http://10.0.0.5:8080/api/upload_vision/"
CHAT_URL = "http://10.0.0.5:8080/api/chat/"

def capture_vision(type="webcam"):
    if type == "webcam":
        cam = cv2.VideoCapture(0)
        ret, frame = cam.read()
        if ret:
            _, buffer = cv2.imencode('.jpg', frame)
            content = base64.b64encode(buffer).decode('utf-8')
            cam.release()
            return content
    else: # screenshot
        screenshot = pyautogui.screenshot()
        screenshot.save("temp.png")
        with open("temp.png", "rb") as f:
            return base64.b64encode(f.read()).decode('utf-8')
    return None

def send_to_rikku(prompt, vision_type=None):
    image_data = None
    if vision_type:
        print(f"Capturing {vision_type}...")
        image_data = capture_vision(vision_type)

    payload = {
        "prompt": prompt,
        "image": image_data # Send image data if user asked to "see"
    }
    
    response = requests.post(CHAT_URL, json=payload)
    print("Rikku:", response.json().get('response'))

if __name__ == "__main__":
    # Example: Manually trigger a vision-aware prompt
    send_to_rikku("Rikku, look at me through my webcam and say hi!", vision_type="webcam")
    # send_to_rikku("What is on my screen right now?", vision_type="screenshot")