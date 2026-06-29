from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any
import os
import numpy as np


from ai_interviewer.vad import LocalVADModel
from ai_interviewer.question_gen import Question
from ai_interviewer.report import AnswerEval

class SessionState(Enum):
    CREATED = auto()
    STARTED = auto()
    SPEAKING = auto()
    LISTENING = auto()
    CODING = auto()
    PROCESSING = auto()
    DONE = auto()


@dataclass
class InterviewSession:
    session_id: str
    resume_path: str
    cfg: Any
    llm: Any
    stt: Any
    tts: Any
    executor: Any = None

    vad: LocalVADModel = field(init=False)
    state: SessionState = SessionState.CREATED
    candidate_name: str = ""
    questions: list[Question] = field(default_factory=list)
    evals: list[AnswerEval] = field(default_factory=list)
    audio_buf: bytearray = field(default_factory=bytearray)
    speech_samples: list[np.ndarray] = field(default_factory=list)
    speech_started: bool = False


    def __post_init__(self):
        self.vad = LocalVADModel(self.cfg.vad)
    
    def cleanup(self):
        try:
            os.remove(self.resume_path)
        except FileNotFoundError:
            pass

