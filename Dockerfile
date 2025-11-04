# Dockerfile (CPU variant)

FROM python:3.11-slim-bullseye

ENV PYTHONUNBUFFERED=1 \
    MODEL_PATH=/app/model \
    OCR_LANGS=en,ch_sim \
    OCR_CONF_THRESHOLD=0.80 \
    PORT=8080 \
    MAX_IMAGE_SIZE=2048

WORKDIR /app

# System deps for easyocr & pillow
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirement and install. Adjust torch in requirements if you want CPU-only wheel.
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy app
#COPY app.py /app/app.py

# Copy model into image if you have it at build time
# If not, mount the model path at runtime as a volume and set MODEL_PATH.
# COPY model /app/model

EXPOSE 8080
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
