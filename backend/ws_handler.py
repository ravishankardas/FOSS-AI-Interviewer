import asyncio
from fastapi import WebSocket, WebSocketDisconnect

from ai_interviewer.config import AppConfig
from ai_interviewer.parser import parse_resume
from ai_interviewer.question_gen import Question, generate_questions, generate_followup_stream
from ai_interviewer.report import evaluate_answer, generate_report, to_markdown
from ai_interviewer.sentence_splitter import SentenceSplitter
from .session import InterviewSession, SessionState
import json
import numpy as np
import torch
from fastapi import FastAPI
from loguru import logger  # type: ignore


async def run_in_executor(fn, *args):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, fn, *args)


async def _aiter_blocking(make_gen):
    # bridge a blocking generator (the LLM token stream) onto the event loop:
    # run it in a thread and hand tokens back through an asyncio.Queue so we can
    # synthesize+send each sentence while the LLM is still producing the next.
    loop = asyncio.get_event_loop()
    queue: asyncio.Queue = asyncio.Queue()
    SENTINEL = object()

    def worker():
        try:
            for token in make_gen():
                loop.call_soon_threadsafe(queue.put_nowait, token)
        except Exception as exc:
            loop.call_soon_threadsafe(queue.put_nowait, exc)
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, SENTINEL)

    loop.run_in_executor(None, worker)
    while True:
        item = await queue.get()
        if item is SENTINEL:
            break
        if isinstance(item, Exception):
            raise item
        yield item


async def _speak_stream(ws: WebSocket, session: InterviewSession, make_gen) -> str:
    # consume an LLM token stream, synthesizing and sending TTS one sentence at a
    # time so playback starts on the first sentence instead of the whole answer.
    # returns the full spoken text (for scoring).
    session.state = SessionState.SPEAKING
    splitter = SentenceSplitter()
    parts: list[str] = []

    async def _say(sentence: str):
        wav = await run_in_executor(session.tts.synthesize, sentence)
        await ws.send_bytes(data=wav)

    async for token in _aiter_blocking(make_gen):
        parts.append(token)
        for sentence in splitter.feed(token):
            await _say(sentence)
    for sentence in splitter.flush():
        await _say(sentence)

    return "".join(parts).strip()



async def _wait_for_start(ws: WebSocket, session: InterviewSession):
    # receive first text msg, parse JSON
    # if type != "start" → send error, raise WebSocketDisconnect
    # store candidate_name on session
    # set state = STARTED

    raw = await ws.receive_text()
    data = json.loads(raw)

    if data.get("type") != "start":
        await ws.send_json({"type": "error", "message": "expected start"})
        raise WebSocketDisconnect(code=500)
    
    session.candidate_name = data.get("candidate_name", "Candidate")
    session.state = SessionState.STARTED
    logger.info(f"[{session.session_id[:8]}] interview started for '{session.candidate_name}'")

async def _ask_question(ws: WebSocket, session: InterviewSession, question: Question, wav_bytes: bytes = None, turn: int = None):
    # set state = SPEAKING
    # synthesize TTS in executor (unless pre-rendered wav_bytes provided)
    # send wav bytes as binary
    # send {"type": "listening", "turn": turn} json

    session.state = SessionState.SPEAKING
    logger.info(f"[{session.session_id[:8]}] asking: {question.text}")
    if wav_bytes is None:
        wav_bytes = await run_in_executor(session.tts.synthesize, question.text)
    await ws.send_bytes(data=wav_bytes)

    await ws.send_json({"type": "listening", "turn": turn})

async def _capture_answer(ws: WebSocket, session: InterviewSession):
    # capture the spoken answer via VAD and return the raw audio (no transcription).
    # transcription happens later in the background STT worker so the candidate
    # never waits on Whisper between questions.
    sid = session.session_id[:8]
    session.state = SessionState.LISTENING
    logger.info(f"[{sid}] listening for answer...")
    session.vad.reset()
    audio_buf = bytearray()
    speech_samples = []
    CHUNK_BYTES = session.cfg.vad.chunk_size * 4
    speech_started = False
    speech_ended = False

    # cap the whole answer turn so a silent candidate or a stalled stream
    # can't hang the interview forever
    loop = asyncio.get_event_loop()
    deadline = loop.time() + session.cfg.interview.answer_time_limit

    while not speech_ended:
        remaining = deadline - loop.time()
        if remaining <= 0:
            logger.info(f"[{sid}] answer timed out after {session.cfg.interview.answer_time_limit}s")
            break

        try:
            msg = await asyncio.wait_for(ws.receive(), timeout=remaining)
        except asyncio.TimeoutError:
            logger.info(f"[{sid}] answer timed out after {session.cfg.interview.answer_time_limit}s")
            break

        if msg["type"] == "websocket.disconnect":
            raise WebSocketDisconnect(msg.get("code", 1000))
        if msg.get("bytes") is None:
            continue

        audio_buf.extend(msg["bytes"])

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

    await ws.send_json({"type": "listening_stop"})

    if not speech_samples:
        logger.info(f"[{sid}] no speech captured")
        return None

    return np.concatenate(speech_samples)


async def _stt_worker(ws: WebSocket, session: InterviewSession, queue: asyncio.Queue, evals: dict):
    # serialized worker: transcribe captured answers and evaluate them off the
    # critical path. One at a time, since each transcription saturates the CPU.
    sid = session.session_id[:8]
    while True:
        item = await queue.get()
        try:
            if item is None:
                break  # sentinel: interview finished asking

            turn, question, audio, text = item
            if text is not None:
                # already transcribed on the critical path (to ground a follow-up)
                pass
            elif audio is None:
                text = ""
            else:
                secs = len(audio) / session.cfg.vad.sample_rate
                logger.info(f"[{sid}] (bg) transcribing turn {turn} ({secs:.1f}s)...")
                text = await run_in_executor(session.stt.transcribe, audio)
                logger.info(f"[{sid}] (bg) turn {turn} transcribed: {text!r}")

            await ws.send_json({"type": "transcribed", "turn": turn, "text": text})

            # question is None for the warm-up intro — transcribe but don't score
            if question is not None and text.strip():
                eval = await run_in_executor(evaluate_answer, question, text, session.llm)
                logger.info(f"[{sid}] (bg) turn {turn} score: {eval.score}/10")
                evals[turn] = eval
        except Exception:
            logger.exception(f"[{sid}] (bg) transcription/eval failed")
        finally:
            queue.task_done()

GREETING = (
    "Hi {name}, I am Vanya, your AI interviewer. "
    "Could you please give me a brief introduction about yourself?"
)

FAREWELL = "Thank you for the interview, {name}. Best of luck with your results."


async def _run_interview(ws: WebSocket, session: InterviewSession):
    await _wait_for_start(ws, session)

    sid = session.session_id[:8]

    # prepare the interview (resume parse + question gen + first-question TTS)
    # in the background while the candidate gives their introduction
    async def _prepare():
        logger.info(f"[{sid}] (bg) parsing resume: {session.resume_path}")
        resume = await run_in_executor(parse_resume, session.resume_path, session.llm)
        logger.info(f"[{sid}] (bg) generating questions...")
        questions = await run_in_executor(generate_questions, resume, session.cfg.interview, session.llm)
        logger.info(f"[{sid}] (bg) generated {len(questions)} questions")
        first_wav = None
        if questions:
            first_wav = await run_in_executor(session.tts.synthesize, questions[0].text)
            logger.info(f"[{sid}] (bg) first question audio ready")
        return questions, first_wav

    prep_task = asyncio.create_task(_prepare())

    # background worker transcribes + evaluates answers off the critical path,
    # keyed by turn so order is preserved for the report
    evals: dict = {}
    queue: asyncio.Queue = asyncio.Queue()
    worker = asyncio.create_task(_stt_worker(ws, session, queue, evals))
    turn = 0

    try:
        # greet and capture the introduction (overlaps with prep_task)
        logger.info(f"[{sid}] === introduction ===")
        greeting = GREETING.format(name=session.candidate_name)
        await _ask_question(ws, session, Question(text=greeting, topic="introduction"), turn=turn)
        intro_audio = await _capture_answer(ws, session)
        await queue.put((turn, None, intro_audio, None))  # warm-up: transcribe, don't score
        turn += 1

        # by now the background prep is usually finished
        questions, first_wav = await prep_task

        answered = False
        for idx, question in enumerate(questions, 1):
            logger.info(f"[{sid}] === question {idx}/{len(questions)} ===")
            await _ask_question(ws, session, question, wav_bytes=first_wav if idx == 1 else None, turn=turn)
            audio = await _capture_answer(ws, session)
            if audio is not None:
                answered = True

            do_followup = session.cfg.interview.follow_up_enabled and audio is not None
            answer_text = None
            if do_followup:
                # transcribe on the critical path to ground the follow-up. Groq is
                # near-instant; the streamed follow-up below masks the rest.
                answer_text = await run_in_executor(session.stt.transcribe, audio)
                logger.info(f"[{sid}] grounding follow-up on: {answer_text!r}")

            # hand off eval (reuse answer_text if we already transcribed it)
            await queue.put((turn, question, audio, answer_text))
            turn += 1

            if do_followup and answer_text and answer_text.strip():
                # stream the follow-up: LLM tokens → sentences → TTS, sentence by
                # sentence, so the candidate hears the first sentence while the
                # rest is still being generated. Replaces the old filler hack.
                logger.info(f"[{sid}] --- streaming follow-up...")
                followup_text = await _speak_stream(
                    ws, session,
                    lambda q=question, a=answer_text: generate_followup_stream(q, a, session.llm),
                )
                logger.info(f"[{sid}] --- follow-up spoken: {followup_text!r}")
                await ws.send_json({"type": "listening", "turn": turn})
                fu_audio = await _capture_answer(ws, session)
                followup = Question(text=followup_text, topic=question.topic)
                await queue.put((turn, followup, fu_audio, None))
                turn += 1

        # speak the farewell now so its playback masks the final transcription,
        # evaluation, and report generation that still have to finish
        await ws.send_json({"type": "status", "message": "Wrapping up — scoring your answers…"})
        if answered:
            logger.info(f"[{sid}] speaking farewell")
            farewell_wav = await run_in_executor(session.tts.synthesize, FAREWELL.format(name=session.candidate_name))
            await ws.send_bytes(data=farewell_wav)

        # drain the worker so all transcriptions/evals finish
        logger.info(f"[{sid}] all questions asked, waiting for transcription/eval to finish...")
        await queue.put(None)
        await worker
    finally:
        # don't leak background tasks if the interview errors out mid-way
        for task in (worker, prep_task):
            if not task.done():
                task.cancel()

    ordered = [evals[k] for k in sorted(evals)]
    if not ordered:
        # every answer was empty / failed — nothing to report on
        logger.info(f"[{sid}] no answers recorded, skipping report")
        await ws.send_json({
            "type": "error",
            "message": "No answers were recorded, so there's nothing to evaluate. Please try again.",
        })
        return

    # report generation also overlaps with the farewell playback on the client
    logger.info(f"[{sid}] generating report from {len(ordered)} evals...")
    report = await run_in_executor(generate_report, session.candidate_name, ordered, session.llm)

    await ws.send_json({"type": "report", "markdown": to_markdown(report)})
    logger.info(f"[{sid}] interview complete, report sent")


async def handle_interview(ws: WebSocket, session: InterviewSession):
    try:
        await _run_interview(ws, session)
    except WebSocketDisconnect:
        raise  # client went away; let the endpoint clean up
    except Exception:
        # any LLM/STT/TTS failure: tell the browser and end cleanly instead
        # of crashing the socket with an unhandled traceback.
        logger.exception("interview failed")
        try:
            await ws.send_json({
                "type": "error",
                "message": "The interview hit an unexpected error and had to stop. Please try again.",
            })
        except Exception:
            pass



    







    

