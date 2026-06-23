from ai_interviewer.llm import GeminiLLMClient, create_llm
from ai_interviewer.config import AppConfig, load_config
from ai_interviewer.report import InterviewReport, evaluate_answer, generate_report
from ai_interviewer.tts import LocalTTSClient
from ai_interviewer.stt import LocalSTTClient
from ai_interviewer.vad import LocalVADModel
from ai_interviewer.parser import parse_resume
from ai_interviewer.question_gen import generate_questions, Question
from typing import Any
from loguru import logger # type: ignore

import sounddevice as sd # type: ignore
import numpy as np
import torch # type: ignore
import io
import wave
import json
import os

QUESTIONS_CACHE = "questions_cache.json"



class InterviewPipeline:
    def __init__(self, cfg, llm, stt, tts, vad ) :
        self.cfg: AppConfig = cfg
        self.llm: Any   = llm
        self.stt: LocalSTTClient  = stt
        self.tts: LocalTTSClient = tts
        self.vad: LocalVADModel = vad

    def listen(self) -> str:
      sample_rate = self.cfg.vad.sample_rate
      chunk_size = self.cfg.vad.chunk_size
      silence_duration_ms = self.cfg.vad.silence_duration_ms


      speech_chunks = []
      speaking = False

      self.vad.reset()
      logger.info("Listening... (speak now)")

      with sd.InputStream(samplerate=sample_rate, channels=1, dtype='float32') as stream:
          while True:
              chunk, _ = stream.read(chunk_size)
              chunk = chunk.flatten()

              tensor = torch.from_numpy(chunk)
              result = self.vad.iterator(tensor, return_seconds=False)

              if result is not None:
                  if 'start' in result:
                      speaking = True
                  elif 'end' in result:
                      if speaking:
                          speech_chunks.append(chunk)
                          break

              if speaking:
                  speech_chunks.append(chunk)

      logger.info("Done listening, transcribing...")
      audio = np.concatenate(speech_chunks)
      return self.stt.transcribe(audio)

    def speak(self, text:str):
        logger.info(f"Speaking: {text}")
        wav_bytes = self.tts.synthesize(text)
        with wave.open(io.BytesIO(wav_bytes)) as wf:
            sample_rate = wf.getframerate()
            audio = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)

        sd.play(audio, samplerate = sample_rate)
        sd.wait()
        logger.info("Done speaking.")
    

    def run(self, resume_path: str, candidate_name: str) -> InterviewReport:
        logger.info("Parsing resume...")
        resume = parse_resume(resume_path, self.llm)
        logger.info("Resume parsed.")

        if os.path.exists(QUESTIONS_CACHE):
            logger.info("Loading questions from cache...")
            with open(QUESTIONS_CACHE) as f:
                data = json.load(f)
            questions = [Question(text=q['text'], topic=q['topic']) for q in data]
            logger.info(f"Loaded {len(questions)} questions from cache.")
        else:
            logger.info("Generating questions...")
            questions = generate_questions(resume, self.cfg.interview, self.llm)
            logger.info(f"Generated {len(questions)} questions.")
            with open(QUESTIONS_CACHE, 'w') as f:
                json.dump([{'text': q.text, 'topic': q.topic} for q in questions], f)

        answers = []
        for i, question in enumerate(questions):
            logger.info(f"Question {i+1}/{len(questions)} [{question.topic}]: {question.text}")
            self.speak(question.text)
            answer = self.listen()
            logger.info(f"Answer received: {answer}")
            eval = evaluate_answer(question, answer, self.llm)
            answers.append(eval)

        logger.info("Generating final report...")
        report = generate_report(candidate_name, answers, self.llm)
        logger.info(f"Report generated. Recommendation: {report.recommendation}")

        return report



if __name__ == "__main__":
    
    cfg = load_config("config.yaml")
    llm = create_llm(cfg.llm)
    stt = LocalSTTClient(cfg.stt)
    tts = LocalTTSClient(cfg.tts)
    vad = LocalVADModel(cfg.vad)


    pipeline = InterviewPipeline(cfg, llm, stt, tts, vad)

    report = pipeline.run("docs/Ravi_AI.pdf", "Ravi")
    from ai_interviewer.report import to_markdown
    print(to_markdown(report))





