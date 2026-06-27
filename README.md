# FOSS AI Interviewer

An open-source, voice-based AI interviewer. It parses a candidate's resume,
conducts a spoken interview over a **STT → LLM → TTS** pipeline, and produces a
structured hiring report at the end.

Runs locally on CPU (GPU optional). Pluggable backends — the **LLM** is local
`llama.cpp` or Gemini, and **STT** is local `faster-whisper` or hosted Groq
(`whisper-large-v3`) with automatic fallback to local Whisper.

---

## How it works

```
Resume (PDF) ──► parse ──► generate questions
                                  │
                                  ▼
        ┌─────────────  interview loop  ─────────────┐
        │  TTS speaks question  (Piper)               │
        │  mic audio ──► VAD (Silero) ──► STT (Groq/   │
        │                                  Whisper)    │
        │  answer ──► LLM evaluates ──► score+feedback │
        │  grounded follow-up: re-transcribe the answer│
        │  and ask a deeper question (filler masks the │
        │  latency while it's produced)                │
        └──────────────────────────────────────────────┘
                                  │
                                  ▼
                  LLM summary + recommendation
                                  │
                                  ▼
                     Markdown interview report
```

Two ways to run it:

- **CLI pipeline** (`backend/pipeline.py`) — local mic/speaker via `sounddevice`
- **WebSocket server** (`backend/main.py`) — browser streams audio over a socket

---

## Install

Requires Python 3.10+.

```bash
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS/Linux

pip install -e .
```

CPU-only PyTorch (recommended unless you have CUDA set up):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

For the **local** `llama.cpp` LLM backend (needs a C/C++ toolchain to build):

```bash
pip install -e ".[local]"     # or ".[cuda]" / ".[metal]"
```

By default the project uses the Gemini backend, which doesn't need `llama-cpp-python`.

### Quick start

After installing, set up your LLM key and launch:

```bash
echo "GEMINI_API_KEY=your-key-here" > .env
# optional: hosted STT (fast + accurate). Falls back to local Whisper if unset/down.
echo "GROQ_API_KEY=your-key-here" >> .env

ai-interviewer --start
```

> The `.env` is read relative to the working directory — launch from the repo root.

`ai-interviewer --start` runs the WebSocket server in the current terminal and opens the
frontend in Chrome (`http://localhost:8000`). Use the browser to enter your name,
upload a resume, and take the interview. Stop with `Ctrl+C`.

```
ai-interviewer --start [--host HOST] [--port PORT] [--no-browser]
```

### Models

- **STT** — hosted Groq `whisper-large-v3` (`provider: groq`, needs `GROQ_API_KEY`),
  or local `faster-whisper` (`provider: local`, `medium` by default, downloaded
  automatically). Groq automatically falls back to local Whisper on any failure.
- **VAD** — `silero-vad`, downloaded automatically
- **TTS** — Piper voice files under `models/en/en_US/lessac/medium/`
- **LLM** — either a local GGUF model under `models/`, or set `provider: gemini`
  in `config.yaml` and export `GEMINI_API_KEY`

---

## Configuration

All settings live in `config.yaml`:

```yaml
interview:
  max_questions: 2
  follow_up_enabled: true
llm:
  provider: gemini            # gemini | local
  model_name: gemini-2.5-flash
stt:
  provider: groq              # groq | local
  model_name: whisper-large-v3  # groq model id
  model_path: medium          # local faster-whisper model (fallback)
vad:
  chunk_size: 512             # must be 512 at 16kHz on Windows
  silence_duration_ms: 2000
tts:
  voice: en_US-lessac-medium
```

---

## Running

### Browser (recommended)

```bash
ai-interviewer --start
```

Boots the server and opens the frontend in Chrome. See [Quick start](#quick-start).

### WebSocket server (manual)

```bash
ai-interviewer --start --no-browser
# or directly:
venv\Scripts\python.exe -m uvicorn backend.main:app --port 8000
```

- Swagger UI: http://127.0.0.1:8000/docs (REST routes only; WS routes aren't listed)
- `POST /upload` — multipart PDF, returns `{ "session_id": "..." }`
- `WebSocket /ws/{session_id}` — drives the interview

### CLI pipeline (local mic/speaker)

```bash
venv\Scripts\python.exe -m backend.pipeline
```

### Terminal test client (end-to-end WS test)

Drives the full WebSocket flow from the terminal using your mic/speakers —
useful for testing the backend without a browser:

```bash
venv\Scripts\python.exe -m backend.test_client docs/Ravi_AI.pdf Ravi
```

---

## WebSocket protocol

**Client → Server**

| Frame | Payload |
|-------|---------|
| JSON `start` | `{ "type": "start", "candidate_name": "..." }` |
| binary | float32 PCM, 16kHz mono (mic audio while listening) |

**Server → Client**

| Frame | Payload |
|-------|---------|
| binary | TTS WAV bytes (spoken question) |
| JSON `status` | `{ "message": "..." }` progress updates |
| JSON `listening` | start streaming mic audio |
| JSON `listening_stop` | VAD detected end; stop streaming |
| JSON `transcribed` | `{ "text": "..." }` what was heard |
| JSON `report` | `{ "markdown": "..." }` final report |
| JSON `error` | `{ "message": "..." }` |

---

## Project layout

```
ai_interviewer/        core pipeline (pip package)
  parser.py            PDF → ResumeData
  question_gen.py      questions + follow-ups
  vad.py               Silero VAD
  stt.py               local Whisper / Groq client + fallback
  llm.py               local / Gemini client
  tts.py               Piper TTS
  report.py            evaluation + markdown report
  config.py            config.yaml loader
backend/
  main.py              FastAPI app, /upload, /ws
  session.py           per-connection InterviewSession state
  ws_handler.py        interview loop over WebSocket
  pipeline.py          CLI mic/speaker pipeline
  test_client.py       terminal WS test client
docs/
  DESIGN.md            full design document
  ws_backend_plan.md   WebSocket backend plan
config.yaml
```

---

## Status

- [x] Resume parsing, question generation
- [x] VAD + STT + TTS + LLM pipeline (CLI)
- [x] Report generation + markdown export
- [x] WebSocket backend (tested end-to-end)
- [x] Browser frontend (`ai-interviewer --start`)
- [x] Silence timeout while listening
- [x] Background transcription/evaluation (overlapped with the interview)
- [x] Hosted Groq STT with automatic local-Whisper fallback (provider toggle)
- [x] Grounded follow-up questions (filler phrase masks the produce latency)

See `handover.md` for current working notes.

---

## License

MIT
