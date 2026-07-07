from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.concurrency import run_in_threadpool
from datetime import date
import os, uuid, shutil, random, threading

from ai_interviewer.config import load_config
from ai_interviewer.parser import check_is_resume
from backend import history
from ai_interviewer.llm import create_llm
from ai_interviewer.stt import create_stt
from ai_interviewer.tts import LocalTTSClient
from ai_interviewer.executor import PistonExecutor
from backend.session import InterviewSession
from backend.ws_handler import handle_interview

# resolve paths relative to the repo root (parent of this backend package)
# so the server works no matter which directory it's launched from
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
CONFIG_PATH = os.path.join(BASE_DIR, "config.yaml")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
sessions = {}

# --- simple daily rate limiting for the public demo -------------------------
# In-memory is fine: the app runs as a single instance (like `sessions` above)
# and a reset on redeploy/rollover is harmless. Caps the number of interviews
# started per IP and in total per day, so the demo can't be looped to drain the
# Gemini spend cap or hit Groq's free-tier limits.
_RL_LOCK = threading.Lock()
_rl_state = {"day": date.today(), "per_ip": {}, "total": 0}

PER_IP_DAILY = 5     # interviews per IP per day
GLOBAL_DAILY = 40    # interviews per day, all users combined

# IPs that skip the cap entirely (the developer's own machine). Set in Railway
# as a comma-separated list, e.g. RL_EXEMPT_IPS="203.0.113.7,198.51.100.4".
# Your public IP can change between networks — update this when it does.
# Find your current one at https://api.ipify.org
RL_EXEMPT_IPS = {
    ip.strip() for ip in os.environ.get("RL_EXEMPT_IPS", "").split(",") if ip.strip()
}
# loopback is always the developer's own machine (on Railway real visitors
# arrive via the proxy with X-Forwarded-For, never as loopback) — never cap it.
RL_EXEMPT_IPS |= {"127.0.0.1", "::1"}


def _client_ip(request: Request) -> str:
    # Railway sits in front of the app, so request.client.host is the proxy.
    # The original client is the first hop in X-Forwarded-For.
    xff = request.headers.get("x-forwarded-for")
    if xff:
        return xff.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _rate_limit_check(ip: str):
    """Read-only: roll the day and check caps WITHOUT counting. Returns None if
    allowed, or a reason string ("busy" / "ip") if already at the limit."""
    if ip in RL_EXEMPT_IPS:          # developer's own machine — never capped
        return None
    with _RL_LOCK:
        today = date.today()
        if today != _rl_state["day"]:
            _rl_state.update(day=today, per_ip={}, total=0)
        if _rl_state["total"] >= GLOBAL_DAILY:
            return "busy"
        if _rl_state["per_ip"].get(ip, 0) >= PER_IP_DAILY:
            return "ip"
        return None


def _rate_limit_commit(ip: str):
    """Count one successful interview start against the caps."""
    if ip in RL_EXEMPT_IPS:
        return
    with _RL_LOCK:
        today = date.today()
        if today != _rl_state["day"]:
            _rl_state.update(day=today, per_ip={}, total=0)
        _rl_state["per_ip"][ip] = _rl_state["per_ip"].get(ip, 0) + 1
        _rl_state["total"] += 1

@asynccontextmanager
async def lifespan(app: FastAPI):
    config = load_config(CONFIG_PATH)
    llm = create_llm(config.llm)
    stt = create_stt(config.stt)
    tts = LocalTTSClient(config.tts)
    executor = PistonExecutor(config.executor)

    app.state.cfg = config
    app.state.llm = llm
    app.state.stt = stt
    app.state.tts = tts
    app.state.executor = executor
    os.makedirs(UPLOAD_DIR, exist_ok=True)

    yield



app = FastAPI(lifespan=lifespan)


@app.post("/upload")
async def upload(file: UploadFile, request: Request):
    # rate-limit the public demo before doing any expensive work (the resume
    # check below already costs a Gemini call), then validate PDF, save to
    # uploads/{uuid}.pdf, create InterviewSession, and return session_id.

    client_ip = _client_ip(request)
    blocked = _rate_limit_check(client_ip)
    if blocked == "busy":
        return JSONResponse(
            status_code=429,
            content={"error": "The demo has hit its daily limit — please try again tomorrow."},
        )
    if blocked == "ip":
        return JSONResponse(
            status_code=429,
            content={"error": "You've reached the daily interview limit for this demo. Please try again tomorrow."},
        )

    if file.content_type != "application/pdf":
        return JSONResponse(status_code=400, content={"error": "PDF only"})
    
    header = await file.read(5)
    await file.seek(0)
    if header != b"%PDF-":
        return JSONResponse(status_code=400, content={"error": "PDF only"})
    

    session_id = uuid.uuid4().hex
    path = os.path.abspath(os.path.join(UPLOAD_DIR, f"{session_id}.pdf"))

    with open(path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # reject anything that isn't actually a resume before starting a session.
    # runs off the event loop; the parse is cached so the later parse is free.
    try:
        looks_like_resume = await run_in_threadpool(
            check_is_resume, path, app.state.llm
        )
    except Exception:
        looks_like_resume = False
    if not looks_like_resume:
        os.remove(path)
        return JSONResponse(
            status_code=400,
            content={
                "error": "This PDF doesn't look like a resume. Please upload your resume/CV.",
                "reason": "not_resume",
            },
        )

    # a real resume → this counts as an interview start; charge it to the caps.
    _rate_limit_commit(client_ip)

    # pick a random voice for this interview and bind it to the session so every
    # synth call uses it; the interviewer's name/gender persona follows the voice.
    voice = random.choice(app.state.tts.voice_names())

    sessions[session_id] = InterviewSession(
        session_id=session_id,
        resume_path=path,
        cfg=app.state.cfg,
        llm=app.state.llm,
        stt=app.state.stt,
        tts=app.state.tts.bound(voice),
        executor=app.state.executor,
        voice=voice,
    )

    return {'session_id': session_id}


@app.get("/history")
async def history_list():
    return {"interviews": history.list_interviews()}


@app.get("/history/{hid}")
async def history_get(hid: str):
    record = history.get_interview(hid)
    if record is None:
        return JSONResponse(status_code=404, content={"error": "not found"})
    return record


@app.websocket("/ws/{session_id}")
async def ws_endpoint(ws: WebSocket, session_id: str):
    # accept connection
    # lookup session; if None → send error json + close + return
    # try: await handle_interview(ws, session)
    # except WebSocketDisconnect: pass
    # finally: session.cleanup(); sessions.pop(session_id, None)

    await ws.accept()

    session = sessions.get(session_id)
    if session is None:
        await ws.send_json({"type": "error", "message": "invalid session"})
        await ws.close()
        return
    
    try:
        await handle_interview(ws, session)
    except WebSocketDisconnect:
        pass
    finally:
        session.cleanup()
        sessions.pop(session_id, None)


# Serve the browser frontend. Mounted last so /upload and /ws win.
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
