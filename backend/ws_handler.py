import asyncio
from fastapi import WebSocket, WebSocketDisconnect

from ai_interviewer.config import AppConfig
from ai_interviewer.parser import parse_resume
from ai_interviewer.question_gen import Question, generate_followup, generate_questions
from ai_interviewer.report import evaluate_answer, generate_report, to_markdown
from .session import InterviewSession, SessionState
import json
import numpy as np
import torch
from fastapi import FastAPI


async def run_in_executor(fn, *args):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, fn, *args)



async def _wait_for_start(ws: WebSocket, session: InterviewSession):
    # receive first text msg, parse JSON
    # if type != "start" → send error, raise WebSocketDisconnect
    # store candidate_name on session
    # set state = STARTED

    raw = await ws.receive_text()
    data = json.loads(raw)

    if data.get("type") != "start":
        await ws.send_json({"type": "error", "message": "exptected start"})
        raise WebSocketDisconnect(code=500)
    
    session.candidate_name = data.get("candidate_name", "Candidate")
    session.state = SessionState.STARTED

async def _ask_question(ws: WebSocket, session: InterviewSession, question: Question):
    # set state = SPEAKING
    # synthesize TTS in executor (tts.synthesize, question.text) → wav bytes
    # send wav bytes as binary
    # send {"type": "listening"} json


    session.state = SessionState.SPEAKING
    wav_bytes = await run_in_executor(session.tts.synthesize, question.text)
    await ws.send_bytes(data=wav_bytes)

    await ws.send_json({"type": "listening"})

async def _listen_for_answer(ws: WebSocket, session: InterviewSession):
    # state = LISTENING
    # session.vad.reset()
    # clear audio_buf, speech_samples; speech_started = False
    # CHUNK_BYTES = cfg.vad.chunk_size * 4

    session.state = SessionState.LISTENING
    session.vad.reset()
    audio_buf = bytearray()
    speech_samples = []
    CHUNK_BYTES = session.cfg.vad.chunk_size * 4
    speech_started = False


    while True:
        msg = await ws.receive()
        if "bytes" not in msg:
            continue

        audio_buf.extend(msg["bytes"])
        speech_ended = False

        while len(audio_buf) >= CHUNK_BYTES:
            raw = audio_buf[:CHUNK_BYTES]
            audio_buf = audio_buf[CHUNK_BYTES: ]

            chunk = np.frombuffer(raw, np.float32).copy()
            result = session.vad.iterator(torch.from_numpy(chunk), return_seconds = False)
            if result is not None:
                if "start" in result:
                    speech_started = True
                if "end" in result and speech_started:
                    speech_ended = True

            if speech_started:
                speech_samples.append(chunk)
            
            if speech_ended:
                break

        if speech_ended:
            break

    await ws.send_json({"type": "listening_stop"})

    audio = np.concatenate(speech_samples)
    text = await run_in_executor(session.stt.transcribe, audio)
    await ws.send_json({"type": "transcribed", "text": text})
    return text
    

async def handle_interview(ws: WebSocket, session: InterviewSession):
    await _wait_for_start(ws, session)

    await ws.send_json({"type": "status", "message": "Parsing resume..."})
    resume = await run_in_executor(parse_resume, session.resume_path, session.llm)

    await ws.send_json({"type": "status", "message": "Generating questions..."})
    questions = await run_in_executor(generate_questions, resume, session.cfg.interview, session.llm)

    for question in questions:
        await _ask_question(ws, session, question)
        answer = await _listen_for_answer(ws, session)

        eval = await run_in_executor(evaluate_answer, question, answer, session.llm)

        session.evals.append(eval)

        if session.cfg.interview.follow_up_enabled:
            followup = await run_in_executor(generate_followup, question, answer, session.llm)
            await _ask_question(ws, session, followup)
            followup_answer = await _listen_for_answer(ws, session)
            followup_eval = await run_in_executor(evaluate_answer, followup, followup_answer, session.llm)

            session.evals.append(followup_eval)

    report = await run_in_executor(generate_report, session.candidate_name, session.evals, session.llm)

    await ws.send_json({"type": "report", "markdown": to_markdown(report)})



    







    

