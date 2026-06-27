from faster_whisper import WhisperModel # type: ignore
import yaml # type: ignore
from .config import STTConfig
import numpy as np
import io
import os
import wave
from dotenv import load_dotenv # type: ignore
from loguru import logger # type: ignore

load_dotenv(".env")


def _to_wav_bytes(audio: np.ndarray, sample_rate: int = 16000) -> bytes:
    # encode float32 [-1, 1] mono PCM into an in-memory 16-bit WAV for upload
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes((np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16).tobytes())
    return buf.getvalue()


class LocalSTTClient():
    def __init__(self, cfg: STTConfig) -> None:
        cpu_threads = cfg.cpu_threads or (os.cpu_count() or 0)
        self.model = WhisperModel(cfg.model_path, device = cfg.device, compute_type="int8", cpu_threads=cpu_threads)

    def transcribe(self, audio: np.ndarray) -> str:

        segments, _ = self.model.transcribe(audio, language = "en")

        return "".join(segment.text for segment in segments)


class GroqSTTClient():
    def __init__(self, cfg: STTConfig) -> None:
        # imported lazily so the default (local) path doesn't require the groq SDK
        from groq import Groq  # type: ignore
        self._client = Groq(api_key=os.environ["GROQ_API_KEY"])
        self.model_name = cfg.model_name

    def transcribe(self, audio: np.ndarray) -> str:
        resp = self._client.audio.transcriptions.create(
            file=("audio.wav", _to_wav_bytes(audio)),
            model=self.model_name,
            language="en",
        )
        return resp.text


class FallbackSTTClient():
    """Try the primary (Groq) client; on any failure fall back to local Whisper.
    The local model is loaded lazily on first failure so we don't pay its RAM/load
    cost unless we actually need it."""

    def __init__(self, primary, cfg: STTConfig) -> None:
        self._primary = primary
        self._cfg = cfg
        self._fallback = None

    def _get_fallback(self) -> "LocalSTTClient":
        if self._fallback is None:
            logger.warning("loading local whisper fallback (this may take a moment)...")
            self._fallback = LocalSTTClient(self._cfg)
        return self._fallback

    def transcribe(self, audio: np.ndarray) -> str:
        try:
            return self._primary.transcribe(audio)
        except Exception as exc:
            logger.warning(f"groq stt failed ({exc!r}), falling back to local whisper")
            return self._get_fallback().transcribe(audio)


def create_stt(cfg: STTConfig):
    if cfg.provider == "groq":
        try:
            primary = GroqSTTClient(cfg)
        except Exception as exc:
            # missing key / SDK / etc. at startup — don't even use Groq
            logger.warning(f"groq stt init failed ({exc!r}), using local whisper")
            return LocalSTTClient(cfg)
        logger.info("using groq stt (local whisper fallback)")
        return FallbackSTTClient(primary, cfg)
    logger.info("using local stt")
    return LocalSTTClient(cfg)


if __name__ == "__main__":
    from .config import load_config
    cfg = load_config("config.yaml")

    client = LocalSTTClient(cfg.stt)

    import sys
    args = sys.argv[1]
    if args == "--audio-file":
        import librosa # type: ignore

        chunk_size = 16000 * 30 # 30 seconds at 16khz
        audio, _ = librosa.load("ai_interviewer/test_audio.mp3", sr=16000, mono=True)

        chunks = [audio[i:i+chunk_size] for i in range(0, len(audio), chunk_size)]

        for i, chunk in enumerate(chunks):
            result = client.transcribe(chunk)
            print(f"chunk: {i}: {result}")

    elif args == "--mic":
        import sounddevice as sd # type: ignore
        import numpy

        duration = 5
        sample_rate = 16000

        print("Recording...")
        audio = sd.rec(int(duration * sample_rate), samplerate = sample_rate, channels=1, dtype='float32', device=1)
        sd.wait(timeout = duration + 2)
        audio = audio.flatten()
        print("Done")

        result = client.transcribe(audio)
        print(f"Result: {result}")
