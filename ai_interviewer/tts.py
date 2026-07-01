from piper import PiperVoice # type: ignore
from .config import TTSConfig
import wave
import io

class LocalTTSClient():
    def __init__(self, cfg: TTSConfig) -> None:
        # load every registered voice up front so a session can use any of them
        # instantly (e.g. a random pick per interview). Falls back to the single
        # model_path if no registry is configured.
        registry = cfg.voices or {cfg.voice: cfg.model_path}
        self.voices = {name: PiperVoice.load(path) for name, path in registry.items()}
        self.default_name = cfg.voice if cfg.voice in self.voices else next(iter(self.voices))

    def voice_names(self) -> list:
        return list(self.voices)

    def synthesize(self, text: str, voice: str | None = None) -> bytes:
        v = self.voices.get(voice) or self.voices[self.default_name]

        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(v.config.sample_rate)
            v.synthesize_wav(text, wav_file)

        return buffer.getvalue()

    def bound(self, voice: str | None = None) -> "BoundVoice":
        """A lightweight per-session view that always synthesizes with `voice`,
        so callers can keep calling .synthesize(text) with no voice argument."""
        return BoundVoice(self, voice or self.default_name)


class BoundVoice:
    def __init__(self, client: "LocalTTSClient", voice: str) -> None:
        self._client = client
        self.voice = voice

    def synthesize(self, text: str) -> bytes:
        return self._client.synthesize(text, self.voice)



if __name__ == "__main__":
    from .config import load_config
    cfg = load_config("config.yaml")

    client = LocalTTSClient(cfg.tts)

    audio = client.synthesize("hello this is a test")

    with open("test_output.wav", "wb") as f:
        f.write(audio)

    print("done")




