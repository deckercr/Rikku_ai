import base64
import logging
import threading
import numpy as np
import cv2

from pgvector.django import CosineDistance

from .models import UserProfile, FaceEmbedding

logger = logging.getLogger(__name__)

# DeepFace is heavy — lazy-load to avoid slowing Django startup
_deepface = None

DEEPFACE_MODEL = "ArcFace"  # 512-dim embeddings, good accuracy/speed balance
# "ssd" uses OpenCV's DNN SSD detector — much more robust than "opencv" (Haar cascade)
# which fails with glasses, angles, and dim lighting. No extra dependencies needed.
DETECTOR_BACKEND = "ssd"
RECOGNITION_THRESHOLD = 0.40  # Cosine distance; lower = stricter matching
MAX_EMBEDDINGS_PER_USER = 10  # Cap stored faces to avoid DB bloat

# Thread-local storage so tools can access the current request's image
_context = threading.local()


def set_current_image(image_b64):
    """Store the current request's image for tool access."""
    _context.current_image = image_b64


def get_current_image():
    """Retrieve the current request's image (set by the view)."""
    return getattr(_context, 'current_image', None)


def clear_current_image():
    """Clean up after request processing."""
    _context.current_image = None


def _get_deepface():
    """Lazy-load DeepFace on first use."""
    global _deepface
    if _deepface is None:
        from deepface import DeepFace
        _deepface = DeepFace
        logger.info("DeepFace loaded successfully")
    return _deepface


def _b64_to_numpy(image_b64):
    """Decode a base64 JPEG/PNG string into a BGR numpy array for OpenCV/DeepFace."""
    image_bytes = base64.b64decode(image_b64)
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # Returns BGR — what DeepFace expects
    return img


def extract_embedding(image_b64):
    """Extract a 512-dim face embedding from a base64-encoded image.

    Returns the embedding list, or None if no face is detected.
    """
    DeepFace = _get_deepface()
    img_array = _b64_to_numpy(image_b64)

    try:
        results = DeepFace.represent(
            img_path=img_array,
            model_name=DEEPFACE_MODEL,
            enforce_detection=True,
            detector_backend=DETECTOR_BACKEND,
        )
        if results:
            logger.info("Face detected (confidence: %.2f)", results[0].get("face_confidence", -1))
            return results[0]["embedding"]
    except ValueError as e:
        # "Face could not be detected" — normal when no face is in frame
        logger.warning("No face detected by %s: %s", DETECTOR_BACKEND, e)
    except Exception:
        logger.exception("Error extracting face embedding")

    return None


def enroll_face(user_name, image_b64):
    """Enroll a face for a user. Creates the profile if it doesn't exist.

    Returns (UserProfile, FaceEmbedding) on success, or raises ValueError.
    """
    embedding = extract_embedding(image_b64)
    if embedding is None:
        raise ValueError("No face detected in the provided image.")

    profile, _ = UserProfile.objects.get_or_create(name=user_name)
    face = FaceEmbedding.objects.create(user=profile, embedding=embedding)
    logger.info("Enrolled face #%d for user '%s'", face.pk, user_name)
    return profile, face


def recognize_face(image_b64):
    """Try to identify a face in the image against enrolled profiles.

    Returns (username, distance, face_detected, embedding):
      - ("John", 0.25, True, [...]) — matched a known user
      - (None, None, True, [...])   — face found but no match in DB
      - (None, None, False, None)   — no face detected in the image

    The embedding is returned so callers can reuse it (e.g., for strengthening)
    without re-running DeepFace extraction.
    """
    embedding = extract_embedding(image_b64)
    if embedding is None:
        return None, None, False, None

    # No enrolled faces? Face exists but nothing to match against.
    if not FaceEmbedding.objects.exists():
        return None, None, True, embedding

    nearest = (
        FaceEmbedding.objects
        .annotate(distance=CosineDistance("embedding", embedding))
        .order_by("distance")
        .select_related("user")
        .first()
    )

    if nearest and nearest.distance < RECOGNITION_THRESHOLD:
        logger.info(
            "Recognized '%s' (distance=%.4f)", nearest.user.name, nearest.distance
        )
        return nearest.user.name, float(nearest.distance), True, embedding

    logger.debug(
        "No match found (nearest distance=%.4f, threshold=%.4f)",
        nearest.distance if nearest else -1,
        RECOGNITION_THRESHOLD,
    )
    return None, None, True, embedding


def strengthen_embedding(username, embedding):
    """Add a pre-extracted embedding for a known user to improve future recognition.

    Silently skips if the user is at the embedding cap.
    Accepts the embedding directly (not an image) to avoid redundant extraction.
    """
    try:
        profile = UserProfile.objects.get(name=username)
    except UserProfile.DoesNotExist:
        return

    if profile.face_embeddings.count() >= MAX_EMBEDDINGS_PER_USER:
        return

    FaceEmbedding.objects.create(user=profile, embedding=embedding)
    logger.info(
        "Strengthened recognition for '%s' (%d embeddings)",
        username, profile.face_embeddings.count(),
    )
