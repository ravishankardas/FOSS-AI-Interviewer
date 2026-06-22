from piper import PiperVoice # type: ignore
from .config import TTSConfig
import wave
import io

class LocalTTSClient():
    def __init__(self, cfg: TTSConfig) -> None:
        self.voice = PiperVoice.load(cfg.model_path)
    
    def synthesize(self, text: str) -> bytes:

        buffer = io.BytesIO()
        
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(self.voice.config.sample_rate)    
            self.voice.synthesize_wav(text, wav_file)
        
        return buffer.getvalue()



if __name__ == "__main__":
    from .config import load_config
    cfg = load_config("config.yaml")

    client = LocalTTSClient(cfg.tts)

    audio = client.synthesize("hello this is a test")

    with open("test_output.wav", "wb") as f:
        f.write(audio)

    print("done")




