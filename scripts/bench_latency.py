"""Benchmark time-to-first-audio (TTFA) for the follow-up question:
batch path (full LLM + full TTS) vs streaming path (first sentence pipelined).

Run from the repo root in the project venv (needs GEMINI_API_KEY + Piper model):

    python -m scripts.bench_latency [runs]

Prints per-run TTFA and total, plus medians and the speedup, so you can pull a
real number for the resume.
"""
import statistics
import sys
import time

from ai_interviewer.config import load_config
from ai_interviewer.llm import create_llm
from ai_interviewer.tts import LocalTTSClient
from ai_interviewer.sentence_splitter import SentenceSplitter
from ai_interviewer.question_gen import Question, generate_followup_stream

# a representative grounded follow-up prompt
QUESTION = Question(
    text="Walk me through the architecture of the speech-to-speech voice bot you built with Pipecat.",
    topic="experience",
)
ANSWER = (
    "I used FastAPI with WebSockets for real-time audio, Silero VAD for "
    "endpointing, Whisper for STT, an LLM for the dialog, and Piper for TTS, "
    "and I pipelined the stages to cut latency."
)


def bench_batch(llm, tts) -> tuple[float, float]:
    """Old path: collect the whole follow-up, then synthesize it whole."""
    t0 = time.perf_counter()
    text = "".join(generate_followup_stream(QUESTION, ANSWER, llm))
    tts.synthesize(text)            # first audio only exists once this returns
    ttfa = time.perf_counter() - t0
    return ttfa, ttfa               # batch: first audio == done


def bench_stream(llm, tts) -> tuple[float, float]:
    """New path: synthesize per sentence; first audio is the first sentence."""
    t0 = time.perf_counter()
    ttfa = None
    splitter = SentenceSplitter()

    def say(sentence: str):
        nonlocal ttfa
        tts.synthesize(sentence)
        if ttfa is None:
            ttfa = time.perf_counter() - t0

    for token in generate_followup_stream(QUESTION, ANSWER, llm):
        for sentence in splitter.feed(token):
            say(sentence)
    for sentence in splitter.flush():
        say(sentence)

    total = time.perf_counter() - t0
    return ttfa or total, total


def main():
    runs = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    cfg = load_config("config.yaml")
    llm = create_llm(cfg.llm)
    tts = LocalTTSClient(cfg.tts)

    print(f"warming up...")
    bench_stream(llm, tts)          # warm caches / connections, discard

    batch_ttfa, stream_ttfa = [], []
    for i in range(1, runs + 1):
        b_ttfa, b_total = bench_batch(llm, tts)
        s_ttfa, s_total = bench_stream(llm, tts)
        batch_ttfa.append(b_ttfa)
        stream_ttfa.append(s_ttfa)
        print(f"run {i}: batch TTFA={b_ttfa:.2f}s | stream TTFA={s_ttfa:.2f}s "
              f"(stream total={s_total:.2f}s)")

    mb, ms = statistics.median(batch_ttfa), statistics.median(stream_ttfa)
    print("\n--- medians over {} runs ---".format(runs))
    print(f"batch  TTFA: {mb:.2f}s")
    print(f"stream TTFA: {ms:.2f}s")
    print(f"speedup:     {mb / ms:.1f}x faster to first audio  "
          f"({(mb - ms):.2f}s lower latency)")


if __name__ == "__main__":
    main()
