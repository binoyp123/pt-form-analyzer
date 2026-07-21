FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

COPY app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ .

ENV PYTHONUNBUFFERED=1
ENV PORT=8000
ENV POSE_MODEL_COMPLEXITY=0
ENV POSE_TARGET_FPS=10
ENV POSE_MAX_WIDTH=480
ENV POSE_MAX_FRAMES=90
ENV POSE_OVERLAY_FRAMES=60

EXPOSE 8000

CMD uvicorn main:app --host 0.0.0.0 --port ${PORT} --workers 1
