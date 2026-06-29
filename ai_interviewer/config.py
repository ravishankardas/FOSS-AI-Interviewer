import yaml # type: ignore
from dataclasses import dataclass, field


@dataclass
class LLMConfig:
    model_path: str
    n_ctx: int = 2048
    n_gpu_layers: int = 0
    temperature: float = 0.7
    provider: str = "local"
    model_name: str = "gemini-2.5-flash"


@dataclass
class STTConfig:
    provider: str = "local"            # local | groq
    model_path: str = "base"           # local faster-whisper model
    device: str = "cpu"
    cpu_threads: int = 0               # 0 = use all available cores
    model_name: str = "whisper-large-v3"  # groq model id


@dataclass
class TTSConfig:
    voice: str = "en_US-lessac-medium"
    speed: float = 1.0
    model_path: str = "models/en/en_US/lessac/medium/en_US-lessac-medium.onnx"


@dataclass
class VADConfig:
    threshold: float = 0.5
    min_speech_duration_ms: int = 250
    silence_duration_ms: int = 600
    sample_rate: int = 16000
    chunk_size: int = 512  # Silero VAD on Windows only accepts 512 @ 16kHz


@dataclass
class InterviewConfig:
    max_questions: int = 2
    answer_time_limit: int = 120
    follow_up_enabled: bool = True
    coding_enabled: bool = True
    coding_questions: int = 1          # how many coding problems to pose
    code_time_limit: int = 600         # cap on the whole coding turn (seconds)


@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 8000


@dataclass
class ExecutorConfig:
    # self-hosted Piston code-execution engine (see docs/piston_setup.md)
    base_url: str = "http://localhost:2000"
    python_version: str = "3.12.0"
    cpp_version: str = "10.2.0"
    run_timeout_ms: int = 3000      # wall-clock cap on the run stage
    compile_timeout_ms: int = 10000  # wall-clock cap on the compile stage (C++)


@dataclass
class AppConfig:
    llm: LLMConfig
    stt: STTConfig = field(default_factory=STTConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    vad: VADConfig = field(default_factory=VADConfig)
    interview: InterviewConfig = field(default_factory=InterviewConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    executor: ExecutorConfig = field(default_factory=ExecutorConfig)


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
        executor=ExecutorConfig(**data.get("executor", {})),
    )
