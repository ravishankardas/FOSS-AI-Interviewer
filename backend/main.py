from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import os, uuid, shutil

from ai_interviewer.config import load_config
from ai_interviewer.llm import create_llm
from ai_interviewer.stt import LocalSTTClient
from ai_interviewer.tts import LocalTTSClient
from backend.session import InterviewSession
from backend.ws_handler import handle_interview

UPLOAD_DIR = "uploads"
sessions = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    config = load_config("config.yaml")
    llm = create_llm(config.llm)
    stt = LocalSTTClient(config.stt)
    tts = LocalTTSClient(config.tts)

    app.state.cfg = config
    app.state.llm = llm
    app.state.stt = stt
    app.state.tts = tts
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


    sessions[session_id] = InterviewSession(
        session_id=session_id,
        resume_path=path,
        cfg=app.state.cfg,
        llm=app.state.llm,
        stt=app.state.stt,
        tts=app.state.tts
    )

    return {'session_id': session_id}


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
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
