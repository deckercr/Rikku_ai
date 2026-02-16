import base64
import json
import os
import re
import logging
import requests
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from openai import OpenAI
from .tools import TOOL_REGISTRY
from .identity import (
    enroll_face, recognize_face, strengthen_embedding,
    set_current_image, clear_current_image,
)
from .models import UserProfile

logger = logging.getLogger(__name__)

# Text model (GPT-OSS-20B via llama.cpp)
TEXT_MODEL_URL = os.environ.get("LLM_TEXT_URL", "http://localhost:8081")
TEXT_MODEL_NAME = os.environ.get("LLM_TEXT_MODEL", "gpt-oss-20b")

# Vision model (LLaVA-1.6 via llama.cpp)
VISION_MODEL_URL = os.environ.get("LLM_VISION_URL", "http://localhost:8082")
VISION_MODEL_NAME = os.environ.get("LLM_VISION_MODEL", "llava-v1.6-mistral-7b")

# TTS config (Qwen3-TTS)
TTS_URL = os.environ.get("TTS_URL", "http://localhost:8083")
TTS_ENABLED = os.environ.get("TTS_ENABLED", "false").lower() == "true"

# STT config (Groq Whisper)
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_STT_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
GROQ_STT_MODEL = "whisper-large-v3-turbo"

text_client = OpenAI(base_url=f"{TEXT_MODEL_URL}/v1", api_key="none")
vision_client = OpenAI(base_url=f"{VISION_MODEL_URL}/v1", api_key="none")


def transcribe_audio(audio_b64: str) -> str:
    """Transcribe base64-encoded audio using Groq Whisper API."""
    if not GROQ_API_KEY:
        logger.warning("GROQ_API_KEY not set, cannot transcribe audio")
        return ""

    try:
        audio_bytes = base64.b64decode(audio_b64)
        response = requests.post(
            GROQ_STT_URL,
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            files={"file": ("audio.wav", audio_bytes, "audio/wav")},
            data={
                "model": GROQ_STT_MODEL,
                "language": "en",
                "response_format": "json",
            },
            timeout=30,
        )
        response.raise_for_status()
        text = response.json().get("text", "").strip()
        logger.info(f"Transcribed audio: {text}")
        return text
    except Exception as e:
        logger.error(f"Audio transcription failed: {e}")
        return ""


def build_tool_descriptions():
    """Build a text description of available tools for the system prompt."""
    if not TOOL_REGISTRY:
        return ""

    lines = ["You have access to the following tools:", ""]
    for name, tool in TOOL_REGISTRY.items():
        lines.append(f"- {name}: {tool['description']}")

    lines.append("")
    lines.append("To use a tool, respond ONLY with: [TOOL: tool_name]")
    lines.append("For tools that take a parameter: [TOOL: tool_name:parameter_value]")
    lines.append("Examples: [TOOL: get_current_time] or [TOOL: enroll_user:John]")
    lines.append("Include the brackets. Only use ONE tool per response.")
    return "\n".join(lines)


def parse_tool_call(response_text):
    """Extract tool call and optional args from response if present.

    Returns (tool_name, args) tuple, or (None, None) if no tool call found.
    Supports: [TOOL: name] and [TOOL: name:args]
    """
    if not response_text:
        return None, None

    # Try bracketed format with optional args: [TOOL: name:args] or [TOOL: name]
    match = re.search(r'\[TOOL:\s*(\w+)(?::(.+?))?\]', response_text, re.IGNORECASE)
    if match:
        return match.group(1), match.group(2)

    # Fallback: TOOL: name (without brackets, no args)
    match = re.search(r'(?:^|\s)TOOL:\s*(\w+)', response_text, re.IGNORECASE)
    if match:
        return match.group(1), None

    return None, None


def execute_tool(tool_name, args=None):
    """Execute a registered tool and return the result."""
    if tool_name not in TOOL_REGISTRY:
        return False, f"Unknown tool: {tool_name}"

    tool_func = TOOL_REGISTRY[tool_name]["callable"]
    try:
        if args is not None:
            result = tool_func(args=args)
        else:
            result = tool_func()
        return True, str(result)
    except Exception as e:
        return False, f"Error executing {tool_name}: {str(e)}"


def generate_speech(text: str) -> bytes | None:
    """Generate speech audio from text using Qwen3-TTS."""
    if not TTS_ENABLED:
        return None
    try:
        resp = requests.post(
            f"{TTS_URL}/v1/audio/speech",
            json={"input": text, "voice": "vivian"},
            timeout=60
        )
        if resp.status_code == 200:
            return resp.content  # WAV/MP3 bytes
        logger.warning(f"TTS returned status {resp.status_code}: {resp.text[:200]}")
    except Exception as e:
        logger.warning(f"TTS failed: {e}")
    return None


def build_user_message(prompt, image_data=None):
    """Build user message, optionally with an image for vision models."""
    if image_data:
        logger.info(f"Building message with image ({len(image_data)} bytes base64)")
        # Multimodal message with image for Ollama vision
        return {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_data}"
                    }
                },
                {
                    "type": "text",
                    "text": prompt
                }
            ]
        }
    else:
        return {"role": "user", "content": prompt}


@csrf_exempt
@require_http_methods(["POST"])
def chat(request):
    """Handle chat requests with tool calling, vision, and face identification.

    Accepts:
        prompt: Text prompt (optional if audio provided)
        audio: Base64-encoded WAV audio to transcribe (optional)
        image: Base64-encoded image for vision (optional)
    """
    try:
        data = json.loads(request.body)
        prompt = data.get("prompt", "")
        audio_data = data.get("audio")
        image_data = data.get("image")

        # Transcribe audio if provided (overrides text prompt)
        if audio_data:
            prompt = transcribe_audio(audio_data)
            if not prompt:
                return JsonResponse({"error": "Audio transcription failed"}, status=400)

        if not prompt:
            return JsonResponse({"error": "No prompt or audio provided"}, status=400)

        # Store image in thread-local so tools (like enroll_user) can access it
        if image_data:
            set_current_image(image_data)

        # --- Face identification ---
        identified_user = None
        face_detected = False
        if image_data:
            try:
                username, distance, face_detected, embedding = recognize_face(image_data)
                if username:
                    identified_user = username
                    logger.info(f"Identified user: {username} (distance={distance:.4f})")
                    # Silently strengthen recognition with this new angle/lighting
                    if embedding:
                        strengthen_embedding(username, embedding)
                elif face_detected:
                    logger.info("Unknown face detected — prompting for enrollment")
            except Exception:
                logger.exception("Face recognition failed, continuing without identity")

        # --- Build system prompt ---
        if image_data:
            # Vision requests: only offer enroll_user tool, not capture/screenshot
            # (the client already captured and sent the image)
            if identified_user:
                system_prompt = (
                    "You are Rikku, a friendly AI waifu assistant. "
                    f"You are currently talking to {identified_user} — you recognize them. "
                    "A live webcam image is attached. Respond conversationally to what they said. "
                    "Do NOT describe the image unless they ask you to. "
                    "Do NOT use any tools unless explicitly asked.\n\n"
                    "If they say their name is wrong or ask to be called something else, "
                    f"use: [TOOL: rename_user:{identified_user}>NewName]"
                )
            elif face_detected:
                system_prompt = (
                    "You are Rikku, a friendly AI waifu assistant. "
                    "A live webcam image is attached. You can see a person but you do NOT recognize them. "
                    "They are not in your memory yet.\n\n"
                    "IMPORTANT: Look at what the person said. "
                    "If they are telling you their name (e.g. 'I'm John', 'My name is Sarah', 'It's Mike', etc.), "
                    "respond ONLY with: [TOOL: enroll_user:TheirName]\n"
                    "Replace TheirName with the actual name they gave you.\n\n"
                    "If they have NOT told you their name yet, greet them warmly and ask what their name is "
                    "so you can remember them. Do NOT describe their appearance. Do NOT use any other tools."
                )
            else:
                system_prompt = (
                    "You are Rikku, a friendly AI waifu assistant with vision capabilities. "
                    "An image has been shared with you. Describe what you see and respond helpfully. "
                    "Do NOT use any tools unless explicitly asked."
                )
        else:
            tool_descriptions = build_tool_descriptions()
            system_prompt = (
                "You are Rikku, a friendly and helpful AI waifu assistant. "
                "You run on a home server and can interact with the host system.\n\n"
                f"{tool_descriptions}"
            )

        messages = [
            {"role": "system", "content": system_prompt},
            build_user_message(prompt, image_data)
        ]

        # Route to appropriate model based on image presence
        if image_data:
            logger.info("Sending request to vision model (LLaVA)")
            response = vision_client.chat.completions.create(
                model=VISION_MODEL_NAME,
                messages=messages,
            )
        else:
            logger.info("Sending request to text model (GPT-OSS)")
            response = text_client.chat.completions.create(
                model=TEXT_MODEL_NAME,
                messages=messages,
            )

        assistant_response = response.choices[0].message.content
        logger.info(f"LLM response: {assistant_response[:100]}...")

        # Check for tool calls (always, even with images — needed for enroll_user)
        tool_name, tool_args = parse_tool_call(assistant_response)

        if tool_name:
            logger.info(f"Tool call detected: {tool_name} (args={tool_args})")
            success, tool_result = execute_tool(tool_name, tool_args)

            messages.append({"role": "assistant", "content": assistant_response})
            messages.append({
                "role": "user",
                "content": f"[Tool Result: {tool_result}]\nNow respond naturally to the user using this information."
            })

            # Use same model routing for follow-up
            if image_data:
                final_response = vision_client.chat.completions.create(
                    model=VISION_MODEL_NAME,
                    messages=messages,
                )
            else:
                final_response = text_client.chat.completions.create(
                    model=TEXT_MODEL_NAME,
                    messages=messages,
                )
            reply = final_response.choices[0].message.content
        else:
            reply = assistant_response

        # Generate TTS audio if enabled
        audio_b64 = None
        if TTS_ENABLED:
            audio_bytes = generate_speech(reply)
            if audio_bytes:
                audio_b64 = base64.b64encode(audio_bytes).decode()

        return JsonResponse({
            "response": reply,
            "audio": audio_b64,
            "transcript": prompt if audio_data else None,
        })

    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON in request body"}, status=400)
    except Exception as e:
        logger.exception("Error in chat view")
        return JsonResponse({"error": str(e)}, status=500)
    finally:
        clear_current_image()


@csrf_exempt
@require_http_methods(["POST"])
def enroll(request):
    """Enroll a face for a user profile.

    POST: {"name": "Username", "image": "<base64 JPEG>"}
    """
    try:
        data = json.loads(request.body)
        name = data.get("name", "").strip()
        image_data = data.get("image")

        if not name:
            return JsonResponse({"error": "No name provided"}, status=400)
        if not image_data:
            return JsonResponse({"error": "No image provided"}, status=400)

        profile, face = enroll_face(name, image_data)
        embedding_count = profile.face_embeddings.count()

        return JsonResponse({
            "status": "enrolled",
            "user": profile.name,
            "face_id": face.pk,
            "total_embeddings": embedding_count,
        })

    except ValueError as e:
        return JsonResponse({"error": str(e)}, status=400)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON in request body"}, status=400)
    except Exception as e:
        logger.exception("Error in enroll view")
        return JsonResponse({"error": str(e)}, status=500)


@csrf_exempt
@require_http_methods(["GET", "DELETE"])
def profiles(request, profile_id=None):
    """List enrolled user profiles or delete one.

    GET  /api/identity/profiles/          — List all profiles
    DELETE /api/identity/profiles/<id>/    — Delete a profile and its embeddings
    """
    try:
        if request.method == "DELETE":
            if profile_id is None:
                return JsonResponse({"error": "Profile ID required"}, status=400)
            try:
                profile = UserProfile.objects.get(pk=profile_id)
                name = profile.name
                profile.delete()
                return JsonResponse({"status": "deleted", "user": name})
            except UserProfile.DoesNotExist:
                return JsonResponse({"error": "Profile not found"}, status=404)

        # GET — list all profiles
        all_profiles = UserProfile.objects.all().order_by("name")
        result = []
        for p in all_profiles:
            result.append({
                "id": p.pk,
                "name": p.name,
                "embeddings": p.face_embeddings.count(),
                "created_at": p.created_at.isoformat(),
            })
        return JsonResponse({"profiles": result})

    except Exception as e:
        logger.exception("Error in profiles view")
        return JsonResponse({"error": str(e)}, status=500)


@csrf_exempt
@require_http_methods(["POST"])
def identify(request):
    """Standalone face identification (without chat).

    POST: {"image": "<base64 JPEG>"}
    Returns: {"identified": true, "user": "Name", "distance": 0.123}
             or {"identified": false}
    """
    try:
        data = json.loads(request.body)
        image_data = data.get("image")

        if not image_data:
            return JsonResponse({"error": "No image provided"}, status=400)

        username, distance, face_found, _ = recognize_face(image_data)

        if username:
            return JsonResponse({
                "identified": True,
                "user": username,
                "distance": distance,
            })
        return JsonResponse({"identified": False, "face_detected": face_found})

    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON in request body"}, status=400)
    except Exception as e:
        logger.exception("Error in identify view")
        return JsonResponse({"error": str(e)}, status=500)
