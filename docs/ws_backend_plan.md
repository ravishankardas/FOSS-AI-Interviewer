# WebSocket Backend Plan

## Design Decisions

- **VAD** — synchronous but fast (microseconds/chunk), call directly on event loop. STT and TTS are slow — always `run_in_executor`.
- **AI components** — LLM, STT, TTS instantiated once at startup on `app.state`. VAD is per-session (stateful iterator).
- **Session** — created during `POST /upload`, retrieved in `WS /ws/{session_id}`.

---

## `backend/main.py`

Two routes:

**`POST /upload`** — saves PDF, creates session, returns `session_id`:
```python
# validate PDF, save to uploads/{uuid}.pdf
# create InterviewSession, store in sessions dict
# return {"session_id": uuid}
```

**`WebSocket /ws/{session_id}`** — retrieves session, drives interview, cleans up in `finally`.

Startup `lifespan`: load config, create shared `llm`, `stt`, `tts`. Store on `app.state`.

---

## `backend/session.py`

```python
class SessionState(Enum):
    CREATED | STARTED | SPEAKING | LISTENING | PROCESSING | DONE

@dataclass
class InterviewSession:
    session_id: str
    resume_path: str
    cfg: AppConfig
    llm: Any
    stt: LocalSTTClient         # shared
    tts: LocalTTSClient         # shared
    vad: LocalVADModel          # per-session (stateful iterator)
    candidate_name: str
    questions: list[Question]
    evals: list[AnswerEval]
    audio_buf: bytearray        # raw incoming bytes
    speech_samples: list[np.ndarray]
    speech_started: bool

    def cleanup(self):          # delete resume file, called in WS finally block
```

---

## `backend/ws_handler.py`

Main entry point: `handle_interview(ws, session)`

```
1. _wait_for_start()       — receive {"type":"start", "candidate_name": ...}
2. run_in_executor → parse_resume()
3. run_in_executor → generate_questions()
4. for each question:
     _ask_question()       — TTS in executor → send binary WAV → send {"type":"listening"}
     _listen_for_answer()  — receive binary chunks → VAD → STT in executor
     run_in_executor → evaluate_answer()
     if follow_up_enabled: repeat with followup question
5. run_in_executor → generate_report()
6. send {"type":"report", "markdown": ...}
```

### `_listen_for_answer` — critical part

```python
CHUNK_BYTES = chunk_size * 4  # float32 = 4 bytes per sample

while True:
    msg = await ws.receive()
    if "bytes" in msg:
        audio_buf.extend(msg["bytes"])
        while len(audio_buf) >= CHUNK_BYTES:
            chunk = np.frombuffer(bytes(audio_buf[:CHUNK_BYTES]), dtype=np.float32).copy()
            del audio_buf[:CHUNK_BYTES]
            result = vad.iterator(torch.from_numpy(chunk), return_seconds=False)
            if result is not None:
                if "start" in result:
                    speech_started = True
                if "end" in result and speech_started:
                    await ws.send_json({"type": "listening_stop"})
                    # break out of both loops (use while/else + continue/break pattern)
        if speech_ended:
            break
    if speech_started:
        speech_samples.append(chunk)

audio = np.concatenate(speech_samples)
text = await run_in_executor(session.stt.transcribe, audio)
await ws.send_json({"type": "transcribed", "text": text})
return text
```

---

## Message Protocol

| Direction | Type | Payload |
|---|---|---|
| C→S | JSON `start` | `candidate_name`, `resume_id` |
| C→S | binary | float32 PCM chunks at 16kHz mono |
| S→C | binary | TTS WAV bytes |
| S→C | JSON `listening` | — |
| S→C | JSON `listening_stop` | — |
| S→C | JSON `transcribed` | `text` |
| S→C | JSON `status` | `message` |
| S→C | JSON `report` | `markdown` |
| S→C | JSON `error` | `message` |

---

## Key Gotchas

1. **Break out of nested loops** — use Python's `while/else` + `continue/break` pattern for the VAD drain loop
2. **`np.frombuffer` doesn't copy** — always call `.copy()` before storing chunks
3. **Timeout on silence** — wrap `ws.receive()` with `asyncio.wait_for(..., timeout=cfg.interview.answer_time_limit)` so a silent user doesn't hang forever
4. **Use `ws.receive()` not `receive_bytes()`** — in the listen loop since frames can be mixed binary + text
5. **VAD chunk size must be 512 on Windows** — already set in `config.yaml`, don't change it
6. **STT on CPU is thread-safe** — shared `LocalSTTClient` instance is fine. If you ever switch to CUDA, add a per-session lock before STT executor call
