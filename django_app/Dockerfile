FROM python:3.11-slim

# Install system dependencies for OpenCV, Postgres, and DeepFace
RUN apt-get update && apt-get install -y \
    libpq-dev \
    gcc \
    libgl1 \
    libglib2.0-0 \
    cmake \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python requirements
RUN pip install --no-cache-dir \
    django \
    djangorestframework \
    psycopg2-binary \
    openai \
    requests \
    pgvector \
    opencv-python-headless \
    pyautogui \
    deepface \
    tf-keras

# We use the volume mount for code, so no COPY needed here
