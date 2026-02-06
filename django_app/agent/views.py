import json
import os
import re
import logging
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

OLLAMA_HOST = os.environ.get("OLLAMA_URL", "http://localhost:11434")
MODEL_NAME = os.environ.get("OLLAMA_MODEL", "llama3.2-vision")

client = OpenAI(
    base_url=f"{OLLAMA_HOST}/v1",
    api_key="ollama",
)


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
    """Handle chat requests with tool calling, vision, and face identification."""
    try:
        data = json.loads(request.body)
        prompt = data.get("prompt", "")
        image_data = data.get("image")

        if not prompt:
            return JsonResponse({"error": "No prompt provided"}, status=400)

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
        tool_descriptions = build_tool_descriptions()

        if image_data:
            if identified_user:
                # Known user with image
                system_prompt = (
                    "You are Rikku, a friendly AI assistant with vision capabilities. "
                    f"You are currently talking to {identified_user}. "
                    "An image has been shared with you. Respond helpfully.\n\n"
                    f"{tool_descriptions}"
                )
            elif face_detected:
                # Unknown face — ask who they are and offer to enroll
                system_prompt = (
                    "You are Rikku, a friendly AI assistant with vision capabilities. "
                    "You can see a person in the image but you do NOT recognize them. "
                    "They are not in your memory yet. "
                    "Greet them warmly and ask them their name. "
                    "Once they tell you their name, use the enroll_user tool to remember their face: "
                    "[TOOL: enroll_user:TheirName]\n\n"
                    f"{tool_descriptions}"
                )
            else:
                # Image but no face (screenshot, object, etc.)
                system_prompt = (
                    "You are Rikku, a friendly AI assistant with vision capabilities. "
                    "An image has been shared with you. Describe what you see and respond helpfully.\n\n"
                    f"{tool_descriptions}"
                )
        else:
            system_prompt = (
                "You are Rikku, a friendly and helpful AI assistant. "
                "You run on a home server and can interact with the host system.\n\n"
                f"{tool_descriptions}"
            )

        messages = [
            {"role": "system", "content": system_prompt},
            build_user_message(prompt, image_data)
        ]

        logger.info(f"Sending request to Ollama (image: {image_data is not None})")

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
        )

        assistant_response = response.choices[0].message.content
        logger.info(f"Ollama response: {assistant_response[:100]}...")

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

            final_response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
            )
            reply = final_response.choices[0].message.content
        else:
            reply = assistant_response

        return JsonResponse({"response": reply})

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
