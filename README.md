# FOSS AI Interviewer

An open-source, voice-based AI interviewer. It parses a candidate's resume,
conducts a spoken interview over a **STT → LLM → TTS** pipeline, and produces a
structured hiring report at the end.

It also runs a **coding round**: the candidate solves a problem in an in-browser
editor, runs it against visible test cases (executed in a sandboxed
[Piston](https://github.com/engineer-man/piston) engine), and the interviewer
asks a spoken follow-up about their code.

Runs locally on CPU (GPU optional). Pluggable backends — the **LLM** is local
`llama.cpp` or Gemini, and **STT** is local `faster-whisper` or hosted Groq
(`whisper-large-v3`) with automatic fallback to local Whisper.

---

## How it works

```
Resume (PDF) ──► parse ──► generate questions   (in the background)
       │
       ▼
  greeting + self-intro
       │
       ▼
┌──────────────  coding round  ───────────────┐
│  editor in browser (CodeMirror)             │
│  Run / Run tests ──► Piston sandbox ──►      │
│                       stdout + pass/fail     │
│  submit ──► grounded spoken follow-up        │
│  graded on code + test results (coding rubric)│
└──────────────────────────────────────────────┘
       │
       ▼
┌─────────────  verbal questions  ────────────┐
│  TTS speaks question  (Piper)               │
│  mic audio ──► VAD (Silero) ──► STT (Groq/   │
│                                  Whisper)    │
│  conversational follow-up: acknowledge the   │
│  answer, then ask deeper. LLM tokens are     │
│  streamed sentence-by-sentence into TTS so   │
│  playback starts almost immediately          │
└──────────────────────────────────────────────┘
       │
       ▼
  LLM summary + recommendation (coding + verbal combined)
       │
       ▼
  Markdown interview report (with evidence quotes)
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

### Code execution engine (for the coding round)

The coding round runs candidate code in a self-hosted **Piston** container
(Docker, Linux engine). Set it up once — see **[docs/piston_setup.md](docs/piston_setup.md)**
for the full guide. In short:

```bash
docker volume create piston_packages
docker run -d --name piston_api -p 2000:2000 \
  -v piston_packages:/piston/packages --privileged ghcr.io/engineer-man/piston
# then install the Python and C++ (gcc) runtimes — see the doc
```

If Piston isn't reachable, set `interview.coding_enabled: false` in `config.yaml`
to skip the coding round.

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
  coding_enabled: true        # run the coding round (needs Piston)
  coding_questions: 1         # how many coding problems to pose
  code_time_limit: 300        # coding round time cap (seconds) — also the UI timer
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
executor:                     # Piston code-execution engine
  base_url: http://localhost:2000
  python_version: 3.12.0
  cpp_version: 10.2.0
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
| JSON `run_code` | `{ "type": "run_code", "language": "python", "code": "..." }` |
| JSON `run_tests` | `{ "type": "run_tests", "language": "...", "code": "..." }` (runs visible tests) |
| JSON `code_submit` | `{ "type": "code_submit", "language": "...", "code": "..." }` |

**Server → Client**

| Frame | Payload |
|-------|---------|
| binary | TTS WAV bytes (spoken question) |
| JSON `status` | `{ "message": "..." }` progress updates |
| JSON `listening` | start streaming mic audio |
| JSON `listening_stop` | VAD detected end; stop streaming |
| JSON `transcribed` | `{ "text": "..." }` what was heard |
| JSON `coding_question` | `{ "title", "prompt", "languages", "starter", "tests", "time_limit" }` show the editor |
| JSON `run_result` | `{ "stdout", "stderr", "compile_error", "exit_code", "timed_out" }` |
| JSON `test_results` | `{ "results": [{ "name", "passed", "expected", "actual", "error" }] }` |
| JSON `report` | `{ "markdown": "..." }` final report |
| JSON `error` | `{ "message": "..." }` |

---

## Project layout

```
ai_interviewer/        core pipeline (pip package)
  parser.py            PDF → ResumeData
  question_gen.py      questions, follow-ups, coding-question bank
  sentence_splitter.py incremental splitter for streaming TTS
  vad.py               Silero VAD
  stt.py               local Whisper / Groq client + fallback
  llm.py               local / Gemini client (with .stream())
  tts.py               Piper TTS
  executor.py          Piston code-execution client
  report.py            evaluation (verbal + coding rubric) + markdown report
  config.py            config.yaml loader
  data/
    coding_questions.json  coding problem bank (prompt, starter, tests)
backend/
  main.py              FastAPI app, /upload, /ws
  session.py           per-connection InterviewSession state
  ws_handler.py        interview loop over WebSocket (verbal + coding)
  pipeline.py          CLI mic/speaker pipeline
  test_client.py       terminal WS test client
frontend/
  index.html, app.js, style.css   browser UI (vanilla JS + CodeMirror editor)
scripts/
  bench_tts.py, bench_latency.py  latency benchmarks
docs/
  DESIGN.md            full design document
  piston_setup.md      code-execution engine setup
  benchmarks.md        latency results
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
- [x] Grounded, conversational follow-ups (acknowledge then ask)
- [x] Streaming follow-up: LLM tokens → per-sentence TTS (~4x faster to first audio)
- [x] Coding round: in-browser editor, sandboxed Piston execution, visible test cases
- [x] Coding graded on correctness (test results) via a dedicated coding rubric
- [x] Combined report (coding + verbal) with evidence quotes

See `handover.md` for current working notes.

---

## License

MIT
