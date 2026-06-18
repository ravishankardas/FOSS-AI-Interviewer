# AI Interviewer — Design Document

## Overview

A fully open source, pip-installable AI Interviewer that parses a candidate's resume,
conducts a real-time voice-based interview using a STT → LLM → TTS pipeline, and
generates a structured report rendered in the browser at the end.

---

## Goals

- Fully open source (no mandatory paid APIs)
- Runs locally on CPU (GPU optional for speed)
- Pip installable: `pip install ai-interviewer`
- Launch with: `ai-interviewer --resume path/to/resume.pdf`
- Modular: swap STT / LLM / TTS backends independently
- Real-time: VAD-gated STT, streaming LLM → sentence-by-sentence TTS
- Web UI: React frontend + FastAPI backend over WebSocket

---

## Non-Goals

- Barge-in / interruption handling (v1)
- Multi-candidate batch processing (single session per run)
- Cloud deployment

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Browser (React)                         │
│                                                                 │
│  ┌──────────┐  ┌─────────────────────────────────────────────┐ │
│  │  Upload  │  │              Interview UI                   │ │
│  │  Resume  │  │  - Live transcript (candidate speech)       │ │
│  └────┬─────┘  │  - Interviewer text (LLM streaming)        │ │
│       │        │  - Progress indicator (Q3 of 10)           │ │
│       │        │  - Mic activity indicator                  │ │
│       │        └─────────────────────────────────────────────┘ │
│       │                          │                             │
│       │        ┌─────────────────────────────────────────────┐ │
│       │        │              Report View                    │ │
│       │        │  - Rendered markdown report in browser      │ │
│       │        │  - Per-skill scores, strengths, gaps        │ │
│       │        │  - Download as markdown button              │ │
│       │        └─────────────────────────────────────────────┘ │
└───────┼──────────────────────────┼──────────────────────────────┘
        │ HTTP POST /upload        │ WebSocket /ws/interview
        ▼                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                      FastAPI Backend                            │
│                                                                 │
│  POST /upload        → parse resume → return ResumeData JSON   │
│  POST /start         → generate questions → return question list│
│  WebSocket /ws/interview → drives full interview session        │
│  GET  /report/{id}   → return generated report as JSON         │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                       Core Pipeline                             │
│                                                                 │
│   audio_queue ← [Browser mic PCM over WebSocket]               │
│        │                                                        │
│        ▼                                                        │
│   Silero VAD → speech_end → faster-whisper → transcript        │
│        │                                                        │
│        ▼                                                        │
│   LLM (llama-cpp-python, streaming)                            │
│        │                                                        │
│        ├──► token stream → WebSocket → browser (live text)     │
│        │                                                        │
│        └──► SentenceSplitter → piper-tts → PCM                 │
│                                    │                            │
│                                    ├──► WebSocket → browser    │
│                                    │    (plays in AudioContext) │
│                                    └──► (optional local speaker)│
└─────────────────────────────────────────────────────────────────┘
```

---

## WebSocket Message Protocol

All messages are JSON (except raw audio which is binary frames).

### Client → Server

| Type | Payload | Description |
|------|---------|-------------|
| `audio_chunk` | binary PCM (16kHz, mono, int16) | Raw mic audio |
| `control` | `{"action": "start" \| "end_session"}` | Session control |

### Server → Client

| Type | Payload | Description |
|------|---------|-------------|
| `transcript` | `{"text": "...", "is_final": bool}` | Candidate speech transcript |
| `llm_token` | `{"token": "..."}` | Streaming interviewer token |
| `llm_done` | `{}` | LLM turn complete |
| `tts_audio` | binary PCM (22050Hz, mono, int16) | TTS audio chunk |
| `tts_done` | `{}` | TTS playback complete — mic opens |
| `state` | `{"state": "listening" \| "thinking" \| "speaking"}` | Pipeline state |
| `progress` | `{"current": 3, "total": 10, "category": "Technical"}` | Question progress |
| `report` | `{"markdown": "..."}` | Final report |
| `error` | `{"message": "..."}` | Error event |

---

## HTTP REST Endpoints

```
POST /upload
  Body: multipart/form-data (resume PDF)
  Response: { "session_id": "...", "resume": ResumeData }

POST /start
  Body: { "session_id": "...", "config": {...} }
  Response: { "questions": [...], "total": 10 }

GET /report/{session_id}
  Response: { "markdown": "...", "scores": {...} }

GET /health
  Response: { "status": "ok", "model_loaded": bool }
```

---

## Real-Time Pipeline Detail

### Audio Flow

```
Browser mic (getUserMedia)
  → PCM chunks (16kHz, mono, int16, 512 samples)
  → WebSocket binary frames
  → FastAPI WebSocket handler
  → audio_queue (asyncio.Queue)
  → VAD worker (Silero)
       ├─ SPEECH_START → send state: "listening" to client
       └─ SPEECH_END   → flush audio → faster-whisper
                              → send transcript to client
                              → LLM worker
                                   → stream tokens to client (llm_token)
                                   → SentenceSplitter
                                        → piper-tts → PCM
                                             → send tts_audio binary frames
                                             → send tts_done when complete
```

### Async Worker Model (FastAPI)

```
WebSocket handler (per session)
  ├─ receive_loop: reads binary audio → audio_queue
  ├─ vad_worker:   audio_queue → VAD → transcript_queue
  ├─ llm_worker:   transcript_queue → LLM stream → sentence_queue + ws send tokens
  └─ tts_worker:   sentence_queue → TTS → ws send binary PCM
```

All workers are `asyncio` tasks sharing a session state object.

### State Guard (no barge-in v1)

- `pipeline_state` enum: `LISTENING | THINKING | SPEAKING`
- VAD worker only enqueues audio when state is `LISTENING`
- State transitions:
  - `LISTENING` → `THINKING`: on VAD speech_end
  - `THINKING` → `SPEAKING`: on first TTS chunk sent
  - `SPEAKING` → `LISTENING`: on `tts_done`

---

## Frontend (React)

### Pages / Views

```
/                   → Landing: upload resume, configure interview
/interview          → Live interview UI
/report             → Report view (post-interview)
```

### Interview UI Components

```
<InterviewPage>
  ├─ <ProgressBar />          question index + category
  ├─ <InterviewerPanel>
  │    └─ <StreamingText />   LLM tokens rendered word by word
  ├─ <CandidatePanel>
  │    └─ <TranscriptText />  live STT transcript
  ├─ <MicIndicator />         pulsing when VAD active / listening
  └─ <StateLabel />           "Listening..." / "Thinking..." / "Speaking..."
```

### Audio in Browser

- `getUserMedia` → `AudioWorklet` → PCM chunks → WebSocket
- TTS audio: receive binary PCM → `AudioContext` → `AudioBufferSourceNode` → play
- No barge-in: mic `AudioWorklet` stops sending when state is `SPEAKING`

### Report View

- Receives `report` WebSocket event (markdown string)
- Rendered with `react-markdown` + `remark-gfm`
- Sections: summary, scores table, strengths, gaps, recommendation
- Download button: saves raw markdown to disk

---

## Component Breakdown

### Backend (`backend/`)

| File | Responsibility |
|------|---------------|
| `main.py` | FastAPI app, routes, WebSocket endpoint |
| `session.py` | Session state, worker lifecycle management |
| `ws_handler.py` | WebSocket message routing + binary audio receive |
| `pipeline.py` | Orchestrates VAD → STT → LLM → TTS workers |

### Core (`ai_interviewer/`)

| File | Responsibility |
|------|---------------|
| `parser.py` | PDF → ResumeData |
| `question_gen.py` | ResumeData → Question list |
| `vad.py` | Silero VAD — speech segmentation |
| `stt.py` | faster-whisper — audio → transcript |
| `llm.py` | llama-cpp-python — streaming inference |
| `sentence_splitter.py` | token stream → sentence boundaries |
| `tts.py` | piper-tts — sentence → PCM audio |
| `report.py` | transcript + resume → markdown report |
| `config.py` | config.yaml loader |

---

## Project Structure

```
ai-interviewer/
├── ai_interviewer/             # core pipeline (pip package)
│   ├── __init__.py
│   ├── parser.py
│   ├── question_gen.py
│   ├── vad.py
│   ├── stt.py
│   ├── llm.py
│   ├── sentence_splitter.py
│   ├── tts.py
│   ├── report.py
│   └── config.py
├── backend/                    # FastAPI server
│   ├── main.py
│   ├── session.py
│   ├── ws_handler.py
│   └── pipeline.py
├── frontend/                   # React app (vibe coded)
│   ├── src/
│   │   ├── pages/
│   │   │   ├── Landing.jsx
│   │   │   ├── Interview.jsx
│   │   │   └── Report.jsx
│   │   ├── components/
│   │   │   ├── ProgressBar.jsx
│   │   │   ├── StreamingText.jsx
│   │   │   ├── TranscriptText.jsx
│   │   │   ├── MicIndicator.jsx
│   │   │   └── StateLabel.jsx
│   │   ├── hooks/
│   │   │   ├── useWebSocket.js
│   │   │   └── useAudio.js
│   │   └── App.jsx
│   ├── package.json
│   └── vite.config.js
├── models/                     # GGUF models (gitignored)
├── voices/                     # piper voice files (gitignored)
├── tests/
│   ├── test_parser.py
│   ├── test_question_gen.py
│   ├── test_sentence_splitter.py
│   └── test_report.py
├── config.yaml
├── pyproject.toml
├── README.md
└── docs/
    └── DESIGN.md
```

---

## Packaging (`pyproject.toml`)

```toml
[project]
name = "ai-interviewer"
version = "0.1.0"
requires-python = ">=3.10"

dependencies = [
    "pdfminer.six",
    "faster-whisper",
    "llama-cpp-python",
    "piper-tts",
    "silero-vad",
    "torch",
    "numpy",
    "pyyaml",
    "fastapi",
    "uvicorn[standard]",
    "python-multipart",
    "websockets",
]

[project.optional-dependencies]
cuda = ["llama-cpp-python[cuda]"]
metal = ["llama-cpp-python[metal]"]

[project.scripts]
ai-interviewer = "ai_interviewer.cli:main"
```

---

## Interview Flow (State Machine)

```
INIT → UPLOAD_RESUME → PARSE → GENERATE_QUESTIONS → START_SESSION
  → [
      SPEAKING (TTS interviewer question)
        → LISTENING (VAD active, mic open)
        → THINKING (STT → LLM streaming)
        → SPEAKING (TTS answer/follow-up)
     ]*
  → WRAP_UP → GENERATE_REPORT → REPORT_VIEW
```

---

## Latency Budget (target on CPU)

| Stage | Target |
|-------|--------|
| VAD speech end detection | ~600ms silence |
| faster-whisper (base, 10s audio) | ~800ms |
| LLM first token (llama.cpp) | ~500ms |
| First sentence TTS + audio to browser | ~300ms |
| **Total turn latency** | **~2.2s** |

GPU on any stage cuts total significantly.

---

## Open Questions / Future Work

- [ ] Barge-in / interruption handling (v2)
- [ ] Domain-specific interview modes (ML, backend, frontend, PM)
- [ ] Multilingual support (Hindi + English)
- [ ] OpenAI / Anthropic API as optional LLM backend
- [ ] Export report as PDF
- [ ] Streaming STT (word-by-word display while candidate speaks)
- [ ] Auth + session persistence for multiple candidates
