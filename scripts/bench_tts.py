"""Measure the TTS pipelining win deterministically (no LLM noise):
time-to-first-audio for a multi-sentence utterance, batch vs per-sentence.

    python -m scripts.bench_tts [runs]
"""
import statistics
import sys
import time

from ai_interviewer.config import load_config
from ai_interviewer.tts import LocalTTSClient
from ai_interviewer.sentence_splitter import SentenceSplitter

# a realistic multi-sentence interviewer turn (e.g. a richer follow-up / intro)
TEXT = (
    "That's a solid overview, thanks for walking me through it. "
    "I'd like to dig into the latency side a bit more. "
    "How did you decide where to put the streaming boundaries in the pipeline? "
    "And were there trade-offs you had to make between latency and accuracy?"
)


def bench_batch(tts) -> float:
    t0 = time.perf_counter()
    tts.synthesize(TEXT)                 # first audio only after the whole render
    return time.perf_counter() - t0


def bench_stream(tts) -> float:
    t0 = time.perf_counter()
    splitter = SentenceSplitter()
    for sentence in splitter.feed(TEXT) + splitter.flush():
        tts.synthesize(sentence)
        return time.perf_counter() - t0   # first sentence = first audio
    return time.perf_counter() - t0


def main():
    runs = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    cfg = load_config("config.yaml")
    tts = LocalTTSClient(cfg.tts)

    bench_stream(tts); bench_batch(tts)   # warm up

    batch, stream = [], []
    for i in range(1, runs + 1):
        b, s = bench_batch(tts), bench_stream(tts)
        batch.append(b); stream.append(s)
        print(f"run {i}: batch TTFA={b:.3f}s | stream TTFA={s:.3f}s")

    mb, ms = statistics.median(batch), statistics.median(stream)
    print(f"\n--- medians over {runs} runs ---")
    print(f"batch  TTFA: {mb:.3f}s")
    print(f"stream TTFA: {ms:.3f}s")
    print(f"speedup:     {mb / ms:.1f}x to first audio ({(mb - ms):.2f}s lower)")


if __name__ == "__main__":
    main()
