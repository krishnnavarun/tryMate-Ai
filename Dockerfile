# tryMate AI service
#   docker build -t trymate-ai .
#   docker run --rm -p 8000:8000 --env-file .env trymate-ai

FROM python:3.13-slim

# System libraries, needed even though we never open a window or use a GPU:
#   libgl1, libglib2.0-0  opencv-contrib-python (pulled in by mediapipe) is the non-headless build
#   libegl1, libgles2     MediaPipe's native library (libmediapipe.so) links against libEGL.so.1
#                         and libGLESv2.so.2; without them loading a model fails with
#                         "OSError: libGLESv2.so.2: cannot open shared object file"
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 libegl1 libgles2 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /srv

# Dependencies first: this layer is reused as long as requirements.txt doesn't change
COPY requirements.txt .
RUN pip install -r requirements.txt

# Models are downloaded at build time, so containers start fast and work offline.
# Only the ones the service uses by default (pose heavy + face landmarker, ~34 MB).
COPY scripts/ scripts/
RUN python scripts/download_models.py --only pose_landmarker_heavy.task face_landmarker.task

COPY app/ app/

# Run as a normal user, not root
RUN useradd --create-home --uid 1000 appuser && chown -R appuser /srv
USER appuser

ENV PORT=8000 \
    LOG_FORMAT=json \
    POSE_MODEL=heavy
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\", \"8000\")}/health', timeout=4)"

# --no-access-log: the app writes its own access log (JSON, with request ids).
# One worker: the models use ~0.5 GB per process; scale with more containers instead.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --no-access-log --workers 1"]
