# FinalRound — Railway app image (Gemini + Groq + local Piper TTS).
# The coding round's code execution runs on a separate hardened Piston droplet
# (see config.yaml `executor.base_url`), so this image needs no Docker/runtimes.
FROM python:3.11-slim

# libgomp1: onnxruntime (used by Piper TTS) needs it at runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install CPU-only torch + torchaudio FIRST, from the same index, so they're
# matched builds and pip won't later pull the multi-GB CUDA torch or a
# mismatched torchaudio. silero-vad imports torchaudio, whose native extension
# fails to load against a mismatched torch — so both must come from here.
RUN pip install --no-cache-dir --timeout 300 --retries 10 \
        torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install the package from pyproject (NOT requirements.txt — that lists
# llama-cpp-python, which needs a C toolchain and isn't used on the Gemini path).
COPY pyproject.toml README.md ./
COPY ai_interviewer ./ai_interviewer
COPY backend ./backend
COPY config.yaml ./
RUN pip install --no-cache-dir .

# Static frontend (vanilla JS — no build step) and the voice downloader.
COPY frontend ./frontend
COPY scripts ./scripts

# Bake the Piper voices declared in config.yaml into the image so startup has
# no network dependency for TTS.
RUN python -m scripts.download_voices

# Runtime scratch dirs (ephemeral on Railway; fine).
RUN mkdir -p uploads history

# Railway injects $PORT; bind it (default 8000 for local `docker run`).
EXPOSE 8000
CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
