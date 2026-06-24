# FOSS AI Interviewer

An open-source, voice-based AI interviewer. It parses a candidate's resume,
conducts a spoken interview over a **STT → LLM → TTS** pipeline, and produces a
structured hiring report at the end.

Runs locally on CPU (GPU optional). Pluggable LLM backend — local `llama.cpp`
or Gemini.

---

## How it works

```
Resume (PDF) ──► parse ──► generate questions
                                  │
                                  ▼
        ┌─────────────  interview loop  ─────────────┐
        │  TTS speaks question  (Piper)               │
        │  mic audio ──► VAD (Silero) ──► STT (Whisper)│
        │  answer ──► LLM evaluates ──► score+feedback │
        │  optional follow-up question                 │
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

### Models

- **STT** — `faster-whisper`, downloaded automatically (`medium` by default)
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
  model_path: medium
vad:
  chunk_size: 512             # must be 512 at 16kHz on Windows
  silence_duration_ms: 2000
tts:
  voice: en_US-lessac-medium
```

---

## Running

### WebSocket server

```bash
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
  stt.py               faster-whisper
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

- [x] Resume parsing, question generation, follow-ups
- [x] VAD + STT + TTS + LLM pipeline (CLI)
- [x] Report generation + markdown export
- [x] WebSocket backend (tested end-to-end)
- [ ] Browser frontend
- [ ] Silence timeout while listening

See `handover.md` for current working notes.

---

## License

MIT
