FROM python:3.11-slim

# Install system dependencies for OpenCV and Postgres
# Changed libgl1-mesa-glx to libgl1
RUN apt-get update && apt-get install -y \
    libpq-dev \
    gcc \
    libgl1 \
    libglib2.0-0 \
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
    pyautogui

# We use the volume mount for code, so no COPY needed here
