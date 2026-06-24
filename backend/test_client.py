"""
End-to-end test client for the WebSocket interview backend.

Mimics what the browser frontend will do, but uses sounddevice for audio
(mic in / speaker out) so you can run a full interview from the terminal.

Usage:
    # 1. start the server first:
    #    ./venv/Scripts/python.exe -m uvicorn backend.main:app --port 8000
    # 2. then run this client (from project root):
    #    ./venv/Scripts/python.exe -m backend.test_client docs/Ravi_AI.pdf Ravi
"""
import asyncio
import io
import json
import sys
import wave

import numpy as np
import requests  # type: ignore
import sounddevice as sd  # type: ignore
import websockets  # type: ignore

HTTP_BASE = "http://127.0.0.1:8000"
WS_BASE = "ws://127.0.0.1:8000"

SAMPLE_RATE = 16000
CHUNK = 512  # samples per mic read (matches VAD chunk_size)


def upload_resume(pdf_path: str) -> str:
    """POST the resume, return the session_id."""
    with open(pdf_path, "rb") as f:
        files = {"file": (pdf_path, f, "application/pdf")}
        resp = requests.post(f"{HTTP_BASE}/upload", files=files)
    resp.raise_for_status()
    session_id = resp.json()["session_id"]
    print(f"[upload] session_id = {session_id}")
    return session_id


def play_wav(wav_bytes: bytes) -> None:
    """Play TTS WAV bytes through the speakers (blocking)."""
    with wave.open(io.BytesIO(wav_bytes)) as wf:
        sr = wf.getframerate()
        audio = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
    sd.play(audio, samplerate=sr)
    sd.wait()


async def stream_mic(ws) -> None:
    """Capture mic and stream float32 PCM chunks until cancelled."""
    loop = asyncio.get_event_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def callback(indata, frames, time_info, status):
        # called from sounddevice's thread -> hand bytes to the event loop
        loop.call_soon_threadsafe(queue.put_nowait, indata.copy().tobytes())

    with sd.InputStream(
        samplerate=SAMPLE_RATE, channels=1, dtype="float32",
        blocksize=CHUNK, callback=callback,
    ):
        try:
            while True:
                chunk = await queue.get()
                await ws.send(chunk)
        except asyncio.CancelledError:
            pass


async def run(pdf_path: str, candidate_name: str) -> None:
    session_id = upload_resume(pdf_path)

    async with websockets.connect(f"{WS_BASE}/ws/{session_id}", max_size=None) as ws:
        await ws.send(json.dumps({
            "type": "start",
            "candidate_name": candidate_name,
        }))

        mic_task: asyncio.Task | None = None

        async for message in ws:
            # binary frame = TTS audio
            if isinstance(message, (bytes, bytearray)):
                print("[speak] playing question audio...")
                play_wav(message)
                continue

            data = json.loads(message)
            mtype = data.get("type")

            if mtype == "status":
                print(f"[status] {data['message']}")

            elif mtype == "listening":
                print("[listening] streaming mic... (speak now)")
                mic_task = asyncio.create_task(stream_mic(ws))

            elif mtype == "listening_stop":
                print("[listening] stopped")
                if mic_task:
                    mic_task.cancel()
                    mic_task = None

            elif mtype == "transcribed":
                print(f"[transcribed] {data['text']}")

            elif mtype == "report":
                print("\n===== REPORT =====\n")
                print(data["markdown"])
                break

            elif mtype == "error":
                print(f"[error] {data['message']}")
                break


if __name__ == "__main__":
    pdf = sys.argv[1] if len(sys.argv) > 1 else "docs/Ravi_AI.pdf"
    name = sys.argv[2] if len(sys.argv) > 2 else "Ravi"
    asyncio.run(run(pdf, name))
