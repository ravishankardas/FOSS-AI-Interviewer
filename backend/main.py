from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.concurrency import run_in_threadpool
import os, uuid, shutil, random

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
async def upload(file: UploadFile):
    # validate PDF, save to uploads/{uuid}.pdf
    # create InterviewSession, store in sessions
    # return session_id

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
            content={"error": "This PDF doesn't look like a resume. Please upload your resume/CV."},
        )

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
