# Benchmarks

Latency measurements for the streaming pipeline. Reproduce with the scripts in
`scripts/` (run from the repo root in the project venv).

## Time-to-first-audio (TTFA): per-sentence TTS vs whole-utterance TTS

`python -m scripts.bench_tts 12` — measures how long until the first spoken
audio is ready for a multi-sentence interviewer turn. Deterministic (Piper TTS
only, no LLM/network in the loop).

Machine: local CPU (Windows, Python 3.11). Voice: `en_US-lessac-medium`.

| Path                          | Median TTFA |
|-------------------------------|-------------|
| Batch (synthesize whole text) | 1.84 s      |
| Streaming (per sentence)      | 0.46 s      |

**~4.0x faster to first audio (~1.4 s lower latency)** on a 4-sentence turn.
Variance across 12 runs was tight (batch 1.81–1.97 s, stream 0.41–0.49 s).

The win scales with utterance length: it comes from starting playback on the
first sentence while later sentences are still synthesizing, so it's largest for
multi-sentence turns (rich follow-ups, intros) and negligible for single-sentence
ones.

## Follow-up generation (LLM + TTS end-to-end)

`python -m scripts.bench_latency 10` — real grounded follow-up via Gemini
(`thinking_budget=0`) then TTS.

No measurable TTFA win here: generated follow-ups are typically a single
sentence, so there is nothing to overlap and LLM/network jitter dominates
(~1.7 s median both paths). The streaming machinery only pays off once a turn
spans multiple sentences (see above).
