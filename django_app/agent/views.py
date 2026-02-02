import json
import os
import re
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from openai import OpenAI
from .tools import TOOL_REGISTRY


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
    lines.append("To use a tool, respond with: [TOOL: tool_name]")
    lines.append("For example: [TOOL: get_current_time]")
    lines.append("Only use ONE tool per response. After the tool runs, you'll get the result.")
    return "\n".join(lines)


def parse_tool_call(response_text):
    """Extract tool call from response if present."""
    match = re.search(r'\[TOOL:\s*(\w+)\]', response_text)
    if match:
        return match.group(1)
    return None


def execute_tool(tool_name):
    """Execute a registered tool and return the result."""
    if tool_name not in TOOL_REGISTRY:
        return None, f"Unknown tool: {tool_name}"

    tool_func = TOOL_REGISTRY[tool_name]["callable"]
    try:
        result = tool_func()
        return True, str(result)
    except Exception as e:
        return False, f"Error executing {tool_name}: {str(e)}"


def build_user_message(prompt, image_data=None):
    """Build user message, optionally with an image for vision models."""
    if image_data:
        # Multimodal message with image
        return {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_data}"
                    }
                }
            ]
        }
    else:
        # Text-only message
        return {"role": "user", "content": prompt}


@csrf_exempt
@require_http_methods(["POST"])
def chat(request):
    """Handle chat requests with prompt-based tool calling and vision support."""
    try:
        data = json.loads(request.body)
        prompt = data.get("prompt", "")
        image_data = data.get("image")  # Base64-encoded image from client

        if not prompt:
            return JsonResponse({"error": "No prompt provided"}, status=400)

        tool_descriptions = build_tool_descriptions()
        system_prompt = (
            "You are Rikku, a friendly and helpful AI assistant. "
            "You run on a home server and can interact with the host system.\n\n"
            f"{tool_descriptions}"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            build_user_message(prompt, image_data)
        ]

        # First call
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
        )

        assistant_response = response.choices[0].message.content

        # Check if the model wants to use a tool
        tool_name = parse_tool_call(assistant_response)

        if tool_name:
            success, tool_result = execute_tool(tool_name)

            # Add the assistant's tool request and result to messages
            messages.append({"role": "assistant", "content": assistant_response})
            messages.append({
                "role": "user",
                "content": f"[Tool Result: {tool_result}]\nNow respond naturally to the user using this information."
            })

            # Get final response
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
        return JsonResponse({"error": str(e)}, status=500)
