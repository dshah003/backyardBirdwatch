FROM python:3.11-slim

# System packages needed by OpenCV and ultralytics
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# YOLOv8n and TFHub models are downloaded on first run and cached here.
# Mount a host volume to /root/.cache to persist across container restarts.
VOLUME ["/root/.cache", "/data"]

CMD ["python", "pipeline.py"]
