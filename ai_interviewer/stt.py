from faster_whisper import WhisperModel # type: ignore
import yaml # type: ignore
from .config import STTConfig
import numpy as np

class LocalSTTClient():
    def __init__(self, cfg: STTConfig) -> None:
        self.model = WhisperModel(cfg.model_path, device = cfg.device, compute_type="int8")
    
    def transcribe(self, audio: np.ndarray) -> str:

        segments, _ = self.model.transcribe(audio, language = "en")

        return "".join(segment.text for segment in segments)

        

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

