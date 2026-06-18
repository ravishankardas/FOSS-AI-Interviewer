import yaml
from dataclasses import dataclass, field


@dataclass
class LLMConfig:
    model_path: str
    n_ctx: int = 2048
    n_gpu_layers: int = 0
    temperature: float = 0.7


@dataclass
class STTConfig:
    model_size: str = "base"
    device: str = "cpu"


@dataclass
class TTSConfig:
    voice: str = "en_US-lessac-medium"
    speed: float = 1.0


@dataclass
class VADConfig:
    threshold: float = 0.5
    min_speech_duration_ms: int = 250
    silence_duration_ms: int = 600
    sample_rate: int = 16000
    chunk_size: int = 512


@dataclass
class InterviewConfig:
    max_questions: int = 10
    answer_time_limit: int = 120
    follow_up_enabled: bool = True


@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 8000


@dataclass
class AppConfig:
    llm: LLMConfig
    stt: STTConfig = field(default_factory=STTConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    vad: VADConfig = field(default_factory=VADConfig)
    interview: InterviewConfig = field(default_factory=InterviewConfig)
    server: ServerConfig = field(default_factory=ServerConfig)


def load_config(path: str = "config.yaml") -> AppConfig:
    with open(path) as f:
        data = yaml.safe_load(f)

    return AppConfig(
        llm=LLMConfig(**data["llm"]),
        stt=STTConfig(**data.get("stt", {})),
        tts=TTSConfig(**data.get("tts", {})),
        vad=VADConfig(**data.get("vad", {})),
        interview=InterviewConfig(**data.get("interview", {})),
        server=ServerConfig(**data.get("server", {})),
    )
