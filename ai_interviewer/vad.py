from .config import VADConfig
from silero_vad import load_silero_vad, VADIterator, get_speech_timestamps # type: ignore
import numpy as np
import torch # type: ignore


class LocalVADModel():
    def __init__(self, cfg: VADConfig) -> None:
        self.threshold = cfg.threshold
        self.silence_duration_ms = cfg.silence_duration_ms
        self.sample_rate = cfg.sample_rate
        self.chunk_size = cfg.chunk_size
        self.model = load_silero_vad()
        self.iterator = VADIterator(self.model, sampling_rate = self.sample_rate, threshold = self.threshold)

    def is_speech(self, chunk: np.ndarray) -> bool:

        tensor = torch.from_numpy(chunk)
        # prob = self.model(tensor.unsqueeze(0), self.sample_rate).item()
        result = self.iterator(tensor, return_seconds=False)
        return result is not None
        # print(f"prob: {prob}")
        # return prob >= self.threshold
    

    def reset(self):
        # self.model.reset_states()
        self.iterator.reset_states()
    
    


if __name__ == "__main__":
    from .config import load_config
    cfg = load_config("config.yaml")


    vad = LocalVADModel(cfg.vad)

    import librosa # type: ignore
    
    audio, _ = librosa.load("test_output.wav", sr = 16000, mono = True)

    print(f"audio_shape: {audio.shape}")
    chunks = [audio[i:i+cfg.vad.chunk_size] for i in range(0, len(audio), cfg.vad.chunk_size)]

    vad.reset()


    for i, chunk in enumerate(chunks):
        if len(chunk) < cfg.vad.chunk_size:
            break
        # tensor = torch.from_numpy(chunk)
        # print(f"tensor shape: {tensor.shape}, dtype: {tensor.dtype}")
        result = vad.is_speech(chunk)
        if result:
            print(f"chunk: {i}: {result}")


    import librosa
    import torch
    audio, _ = librosa.load("test_output.wav", sr=16000,mono=True)
    wav = torch.from_numpy(audio)
    timestamps = get_speech_timestamps(wav, vad.model, sampling_rate=16000)
    print(timestamps)


    # mid = len(chunks) // 2
    # indices = range(mid, mid + 10)
    # for i in indices:
    #     result = vad.is_speech(chunks[i])
    #     print(f"result: {result}")
    