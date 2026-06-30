import asyncio
from fastapi import WebSocket, WebSocketDisconnect

from ai_interviewer.config import AppConfig
from ai_interviewer.parser import parse_resume
from ai_interviewer.question_gen import (
    Question,
    Turn,
    generate_adaptive_question,
    generate_followup_stream,
    generate_code_followup_stream,
    pick_coding_questions,
)
from ai_interviewer.report import evaluate_answer, evaluate_code, generate_report, to_markdown
from ai_interviewer.sentence_splitter import SentenceSplitter
from .session import InterviewSession, SessionState
import json
import struct
import numpy as np
import torch
from fastapi import FastAPI
from loguru import logger  # type: ignore


class EngineUnavailable(Exception):
    """The code-execution engine (Piston) couldn't be reached for a test run."""


# shown to the candidate when the execution engine is down — makes clear it's
# infrastructure, not their code
ENGINE_DOWN_MSG = (
    "The code execution service is temporarily unavailable, so tests couldn't "
    "run. This is on our end, not your code."
)


class CandidateAbandoned(Exception):
    """The candidate went silent through repeated nudges; end the interview."""


# if the candidate doesn't start speaking within this many seconds, nudge them;
# after IDLE_MAX_NUDGES unanswered nudges, end the interview gracefully.
IDLE_NUDGE_SECONDS = 5.0
IDLE_MAX_NUDGES = 2
IDLE_NUDGES = [
    "Sorry, are you still there?",
    "I still can't hear you — are you on the line?",
]
ABANDON_MSG = (
    "It looks like I've lost you, so I'll end the interview here. "
    "Feel free to start again whenever you're ready."
)


async def run_in_executor(fn, *args):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, fn, *args)


def _wav_seconds(wav: bytes) -> float:
    # playback duration of a PCM WAV from its byte-rate. Piper writes a streaming
    # header whose declared data-chunk size can be a 0/placeholder, so prefer the
    # ACTUAL bytes after the data chunk and only trust the declared size if sane.
    try:
        if len(wav) < 44 or wav[:4] != b"RIFF":
            return 0.0
        byte_rate = struct.unpack_from("<I", wav, 28)[0]
        idx = wav.find(b"data", 12)
        if idx == -1 or byte_rate == 0:
            return 0.0
        actual = len(wav) - (idx + 8)            # payload bytes physically present
        declared = struct.unpack_from("<I", wav, idx + 4)[0]
        size = declared if 0 < declared <= actual else actual
        return size / byte_rate
    except Exception:
        return 0.0


async def _send_audio(ws: WebSocket, session: InterviewSession, wav: bytes):
    # send a TTS clip and advance session.speaking_until — the server's estimate of
    # when the client will FINISH playing everything queued so far. The silence
    # timer in _capture_answer starts from that point, so we never nudge while the
    # bot is still talking. Clips queue back-to-back client-side (playbackChain).
    loop = asyncio.get_event_loop()
    base = max(loop.time(), getattr(session, "speaking_until", 0.0))
    session.speaking_until = base + _wav_seconds(wav)
    await ws.send_bytes(data=wav)


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
        await _send_audio(ws, session, wav)

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
    await _send_audio(ws, session, wav_bytes)

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

    # nudge a silent candidate a couple of times, then give up. The countdown only
    # starts after the bot has finished SPEAKING: session.speaking_until is the
    # server's estimate of when the client finishes playing the question (or a
    # nudge), so we never nudge over our own audio. _speak_line below advances it.
    nudges_used = 0
    silence_deadline = max(loop.time(), getattr(session, "speaking_until", 0.0)) + IDLE_NUDGE_SECONDS

    while not speech_ended:
        now = loop.time()
        if now >= deadline:
            logger.info(f"[{sid}] answer timed out after {session.cfg.interview.answer_time_limit}s")
            break

        # silence nudging is driven by wall-clock time, NOT by ws.receive timing
        # out — the client streams mic audio continuously (silent frames included),
        # so receive() keeps returning and never times out. Check the elapsed time
        # against the nudge deadline each pass, while no speech has started yet.
        if not speech_started and now >= silence_deadline:
            if nudges_used >= IDLE_MAX_NUDGES:
                logger.info(f"[{sid}] no response after {IDLE_MAX_NUDGES} nudges — ending interview")
                await ws.send_json({"type": "listening_stop"})
                raise CandidateAbandoned()
            logger.info(f"[{sid}] silence — nudging candidate ({nudges_used + 1}/{IDLE_MAX_NUDGES})")
            await _speak_line(ws, session, IDLE_NUDGES[nudges_used])  # advances speaking_until
            session.state = SessionState.LISTENING
            session.vad.reset()  # drop any frames captured during the nudge playback
            nudges_used += 1
            # re-arm after the nudge finishes playing, not while it's still talking
            silence_deadline = max(loop.time(), session.speaking_until) + IDLE_NUDGE_SECONDS
            continue

        # wake up by the next nudge checkpoint even if the stream stalls entirely
        wait = (deadline if speech_started else min(deadline, silence_deadline)) - now
        try:
            msg = await asyncio.wait_for(ws.receive(), timeout=max(0.0, wait))
        except asyncio.TimeoutError:
            # stream stalled (no frames at all) — loop back; the checks at the top
            # handle the overall deadline and the silence nudges
            continue

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

            # grader is an optional callable(text) -> AnswerEval for coding turns;
            # verbal turns leave it None and fall back to evaluate_answer
            turn, question, audio, text, grader = item
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

            if grader is not None:
                # coding turn: grade with the dedicated coding rubric (the spoken
                # explanation may be empty; correctness still scores)
                eval = await run_in_executor(grader, text)
                logger.info(f"[{sid}] (bg) turn {turn} coding score: {eval.score}/10")
                evals[turn] = eval
            elif question is not None and text.strip():
                # question is None for the warm-up intro — transcribe but don't score
                eval = await run_in_executor(evaluate_answer, question, text, session.llm)
                logger.info(f"[{sid}] (bg) turn {turn} score: {eval.score}/10")
                evals[turn] = eval
        except Exception:
            logger.exception(f"[{sid}] (bg) transcription/eval failed")
        finally:
            queue.task_done()

GREETING = (
    "Hi {name}, great to meet you! I'm Vanya, and I'll be your interviewer today. "
    "We'll start with a short coding exercise, then chat through your background. "
    "But first, to break the ice — tell me a little about yourself."
)

FAREWELL = "Thank you for the interview, {name}. Best of luck with your results."

CODING_LANGUAGES = ["python", "c++"]

# spoken when moving from the coding round into the verbal questions
TRANSITION_TO_QUESTIONS = (
    "Great, thanks for working through that. Let's switch gears now and talk "
    "through your background for a bit."
)


async def _speak_line(ws: WebSocket, session: InterviewSession, text: str):
    # speak a line without listening afterwards (used for transitions/segues)
    session.state = SessionState.SPEAKING
    wav = await run_in_executor(session.tts.synthesize, text)
    await _send_audio(ws, session, wav)


def _result_payload(r) -> dict:
    return {
        "type": "run_result",
        "stdout": r.stdout,
        "stderr": r.stderr,
        "compile_error": r.compile_error,
        "exit_code": r.exit_code,
        "timed_out": r.timed_out,
        "error": r.error,
    }


async def _run_tests(session: InterviewSession, language: str, code: str, tests: list) -> list:
    # run the candidate's code against each visible test case and compare stdout
    # (trimmed). One execution per case, serialized.
    #
    # raises EngineUnavailable if the execution engine can't be reached, so the
    # caller can say "engine down" instead of reporting every case as a failure
    # (an engine fault is not the candidate's code being wrong).
    results = []
    for i, t in enumerate(tests):
        name = t.get("name", f"test {i+1}")
        expected = t.get("expected", "")
        r = await run_in_executor(session.executor.run, language, code, t.get("stdin", ""))
        if r.error:
            # transport/engine failure (ExecResult.error is never set by user code)
            raise EngineUnavailable(r.error)
        actual = r.stdout
        err = r.compile_error or (r.stderr if r.exit_code != 0 else "")
        passed = (not err) and actual.strip() == expected.strip()
        results.append({
            "name": name,
            "passed": passed,
            "expected": expected,
            "actual": actual,
            "error": err,
            "timed_out": r.timed_out,
        })
    return results


async def _coding_turn(ws: WebSocket, session: InterviewSession, cq, queue: asyncio.Queue, turn: int):
    # show the problem, let the candidate run code freely, then ask a grounded
    # spoken follow-up about what they wrote. Returns (next_turn, answered).
    sid = session.session_id[:8]
    session.state = SessionState.CODING
    logger.info(f"[{sid}] coding problem: {cq.title}")

    await ws.send_json({
        "type": "coding_question",
        "turn": turn,
        "id": cq.id,
        "title": cq.title,
        "prompt": cq.prompt,
        "languages": CODING_LANGUAGES,
        "starter": cq.starter,
        "time_limit": session.cfg.interview.code_time_limit,
        # expose the visible test cases (without the trailing-newline noise)
        "tests": [{"name": t.get("name", f"test {i+1}"), "stdin": t.get("stdin", ""),
                   "expected": t.get("expected", "")} for i, t in enumerate(cq.tests)],
    })

    # speak the intro (no listening — the candidate types, doesn't talk yet)
    intro_wav = await run_in_executor(session.tts.synthesize, cq.spoken_intro)
    await _send_audio(ws, session, intro_wav)

    last_lang, last_code, last_output = "python", "", ""

    loop = asyncio.get_event_loop()
    deadline = loop.time() + session.cfg.interview.code_time_limit
    while True:
        remaining = deadline - loop.time()
        if remaining <= 0:
            logger.info(f"[{sid}] coding turn timed out")
            break
        try:
            msg = await asyncio.wait_for(ws.receive(), timeout=remaining)
        except asyncio.TimeoutError:
            break
        if msg["type"] == "websocket.disconnect":
            raise WebSocketDisconnect(msg.get("code", 1000))
        if msg.get("text") is None:
            continue  # ignore stray binary frames during coding

        data = json.loads(msg["text"])
        mtype = data.get("type")
        if mtype == "run_code":
            last_lang = data.get("language", "python")
            last_code = data.get("code", "")
            result = await run_in_executor(session.executor.run, last_lang, last_code)
            if result.error:
                logger.warning(f"[{sid}] run skipped — engine down: {result.error}")
                await ws.send_json({"type": "engine_error", "message": ENGINE_DOWN_MSG})
                continue
            last_output = result.stdout or result.compile_error or result.stderr or ""
            logger.info(f"[{sid}] ran {last_lang} (exit {result.exit_code})")
            await ws.send_json(_result_payload(result))
        elif mtype == "run_tests":
            last_lang = data.get("language", "python")
            last_code = data.get("code", "")
            try:
                results = await _run_tests(session, last_lang, last_code, cq.tests)
            except EngineUnavailable as exc:
                logger.warning(f"[{sid}] tests skipped — engine down: {exc}")
                await ws.send_json({"type": "engine_error", "message": ENGINE_DOWN_MSG})
                continue
            last_output = "\n".join(
                f"{r['name']}: {'PASS' if r['passed'] else 'FAIL'}" for r in results
            )
            passed = sum(1 for r in results if r["passed"])
            logger.info(f"[{sid}] ran tests: {passed}/{len(results)} passed")
            await ws.send_json({"type": "test_results", "results": results})
        elif mtype == "code_submit":
            last_lang = data.get("language", last_lang)
            last_code = data.get("code", last_code)
            logger.info(f"[{sid}] code submitted ({last_lang}, {len(last_code)} chars)")
            break

    if not last_code.strip():
        logger.info(f"[{sid}] no code submitted, skipping coding follow-up")
        return turn, False

    # grounded spoken follow-up about their code, then capture the voice answer
    followup_text = await _speak_stream(
        ws, session,
        lambda: generate_code_followup_stream(cq, last_lang, last_code, last_output, session.llm),
    )
    logger.info(f"[{sid}] coding follow-up: {followup_text!r}")
    await ws.send_json({"type": "listening", "turn": turn})
    audio = await _capture_answer(ws, session)

    # run the FULL suite (visible + hidden) against the submitted code as the
    # authoritative correctness signal for scoring — hidden cases never reached
    # the browser, so they can't be gamed. Off the critical path; the candidate
    # already answered.
    scoring_tests = list(cq.tests) + list(cq.hidden_tests)
    try:
        results = await _run_tests(session, last_lang, last_code, scoring_tests) if scoring_tests else []
    except EngineUnavailable as exc:
        # engine down at submit: don't fabricate a 0/N. Grade on code + explanation
        # only (evaluate_code treats empty results as "no automated tests").
        logger.warning(f"[{sid}] scoring without tests — engine down: {exc}")
        results = []
    # results align with scoring_tests order: visible first, then hidden
    visible_results = results[:len(cq.tests)]
    hidden_results = results[len(cq.tests):]
    passed = sum(1 for r in results if r["passed"])
    logger.info(f"[{sid}] submitted code scored {passed}/{len(results)} tests "
                f"({len(cq.tests)} visible + {len(cq.hidden_tests)} hidden)")

    # grade with the dedicated coding rubric: problem + code + test outcome +
    # the follow-up Q and the candidate's spoken answer (filled in by the worker)
    def grader(answer_text, _cq=cq, _lang=last_lang, _code=last_code,
               _vis=visible_results, _hid=hidden_results, _fq=followup_text):
        return evaluate_code(_cq.title, _cq.prompt, _lang, _code, _vis, _hid, _fq,
                             answer_text, session.llm)

    await queue.put((turn, None, audio, None, grader))
    return turn + 1, True


async def _run_interview(ws: WebSocket, session: InterviewSession):
    await _wait_for_start(ws, session)

    sid = session.session_id[:8]

    # prepare the interview (resume parse + first adaptive question + its TTS)
    # in the background while the candidate gives their introduction. The rest
    # of the questions are chosen on the fly, one per turn, from how they answer.
    async def _prepare():
        logger.info(f"[{sid}] (bg) parsing resume: {session.resume_path}")
        resume = await run_in_executor(parse_resume, session.resume_path, session.llm)
        logger.info(f"[{sid}] (bg) generating first question...")
        first_q, _ = await run_in_executor(generate_adaptive_question, resume, [], session.llm)
        first_wav = await run_in_executor(session.tts.synthesize, first_q.text)
        logger.info(f"[{sid}] (bg) first question audio ready")
        return resume, first_q, first_wav

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
        await queue.put((turn, None, intro_audio, None, None))  # warm-up: transcribe, don't score
        turn += 1

        answered = False

        # coding round first — it runs while the resume parse + question
        # generation (prep_task) finish in the background, so that time isn't idle
        if session.cfg.interview.coding_enabled and session.executor is not None:
            for cq in pick_coding_questions(session.cfg.interview.coding_questions):
                logger.info(f"[{sid}] === coding: {cq.title} ===")
                turn, coded = await _coding_turn(ws, session, cq, queue, turn)
                if coded:
                    answered = True

        # then the verbal questions — chosen adaptively, one per turn, from how
        # the candidate answers (prep gave us the resume + the first question)
        resume, question, first_wav = await prep_task

        # natural segue out of the coding round into the conversation
        if answered:
            await _speak_line(ws, session, TRANSITION_TO_QUESTIONS)

        max_q = session.cfg.interview.max_questions
        history: list[Turn] = []
        for idx in range(1, max_q + 1):
            logger.info(f"[{sid}] === question {idx}/{max_q} [{question.topic}] ===")
            await _ask_question(ws, session, question, wav_bytes=first_wav if idx == 1 else None, turn=turn)
            audio = await _capture_answer(ws, session)
            if audio is not None:
                answered = True

            # transcribe on the critical path: the adaptive picker needs the
            # answer to choose the next question. Groq is near-instant.
            answer_text = ""
            if audio is not None:
                answer_text = await run_in_executor(session.stt.transcribe, audio)
                logger.info(f"[{sid}] answer: {answer_text!r}")

            # hand off the rich report scoring to the background worker
            await queue.put((turn, question, audio, answer_text, None))
            history.append(Turn(question=question, answer=answer_text))
            turn += 1

            # adaptively pick the next question. The same call rates the answer
            # just given (cheap steering signal) and targets the next question
            # to it — struggled → easier/pivot, nailed it → deeper/harder.
            if idx < max_q:
                question, mastery = await run_in_executor(
                    generate_adaptive_question, resume, history, session.llm
                )
                history[-1].mastery = mastery
                first_wav = None
                logger.info(f"[{sid}] last answer mastery {mastery}/10 → next [{question.topic}]")

        # speak the farewell now so its playback masks the final transcription,
        # evaluation, and report generation that still have to finish
        await ws.send_json({"type": "status", "message": "Wrapping up — scoring your answers…"})
        if answered:
            logger.info(f"[{sid}] speaking farewell")
            farewell_wav = await run_in_executor(session.tts.synthesize, FAREWELL.format(name=session.candidate_name))
            await _send_audio(ws, session, farewell_wav)

        # drain the worker so all transcriptions/evals finish
        logger.info(f"[{sid}] all questions asked, waiting for transcription/eval to finish...")
        await queue.put(None)
        await worker
    except CandidateAbandoned:
        # the candidate went silent through repeated nudges; say goodbye and tell
        # the client to reset itself — no report, no error banner
        logger.info(f"[{sid}] candidate unresponsive — ending interview early")
        try:
            await _speak_line(ws, session, ABANDON_MSG)
        except Exception:
            pass
        # send the reload independently so a TTS hiccup above can't swallow it
        try:
            await ws.send_json({"type": "reload"})
        except Exception:
            pass
        return
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



    







    

