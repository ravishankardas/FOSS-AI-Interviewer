from dataclasses import dataclass
import json
import os
import random
import time

from .llm import create_llm
from .parser import ResumeData, parse_resume
from .config import InterviewConfig
from typing import Any, List
from pprint import pprint
from pydantic import BaseModel # type: ignore
from loguru import logger # type: ignore


@dataclass
class Question:
    text: str
    topic: str


# Schemas for Gemini structured output (native JSON mode).
class _QuestionSchema(BaseModel):
    text: str
    topic: str

SYSTEM_PROMPT = """
  You are an expert technical interviewer conducting a real job interview.
  Given a candidate's resume, generate exactly {n} interview questions.

  Rules:
  - Generate only simple questions
  - Questions must be specific to the candidate's resume, not generic
  - Cover a mix of skills, experience, and projects
  - Ask about real things on their resume (specific technologies, companies,
  projects)
  - Questions should be conversational, as if spoken out loud
  - Do not number the questions
  - Do not repeat similar questions

  Output format rules (strictly follow):
  - Return ONLY a JSON array, nothing else
  - Start your response with [ and end with ]
  - Do NOT return a JSON object {{}}
  - Do NOT wrap in markdown or code fences
  - Do NOT add any explanation before or after the JSON
  - Every question must be complete, do not cut off mid-sentence
  - Generate all {n} questions, do not stop early

  Format: [{{"text": "question here", "topic": "skills|experience|projects|education"}}]
  """

def _resume_to_text(resume: ResumeData):

    alphabet_list = [chr(ord('a') + i) for i in range(0, 26)]

    lines = []
    lines.append(f"Name: {resume.name}")
    lines.append(f"Skills: {', '.join(resume.skills)}")
    lines.append("Experience: ")

    for idx, exp in enumerate(resume.experience):
        lines.append(f"  {idx+1}. {exp['title']} at {exp['company']}({exp['duration']})")
        for i, b in enumerate(exp.get('bullets', [])):
            lines.append(f"    ({alphabet_list[i]}) {b}")

    
    lines.append("Projects:")
    for idx, proj in enumerate(resume.projects):
        lines.append(f"  ({alphabet_list[idx]}) {proj['name']} using {', '.join(proj.get('tech',[]))}")

    lines.append("Education:")
    for idx, edu in enumerate(resume.education):
        lines.append(f"  ({alphabet_list[idx]}) {edu['degree']} from {edu['institution']}")

    return "\n".join(lines)


def generate_questions(resume: ResumeData, cfg: InterviewConfig, llm: Any) ->List[Question]:
    resume_text = _resume_to_text(resume)

    system = SYSTEM_PROMPT.format(n = cfg.max_questions)
    response = llm.complete(prompt = resume_text, system = system, response_schema = list[_QuestionSchema])

    data = json.loads(response)
    questions = [Question(text=q['text'], topic=q['topic']) for q in data]
    random.shuffle(questions)
    return questions


def generate_followup(question: Question, answer: str, llm: Any) -> Question:

    system = f"""
        You are a technical interviewer. Given a question and the candidate's answer,
        generate one follow-up question that digs deeper into their response.
        Return only JSON: {{"text": "...", "topic": "skills|experience|projects|education"}}
        No markdown, no explanation.
    """
    prompt = f"""
        The original question was: {question.text} on the topic: {question.topic}.
        The candidate's answer is: {answer}
    """

    response = llm.complete(prompt=prompt, system= system, response_schema = _QuestionSchema)

    data = json.loads(response)

    return Question(text=data['text'], topic=data['topic'])



if __name__ == "__main__":
    from .config import load_config
    cfg = load_config("config.yaml")
    pdf_path = os.path.join(os.getcwd(), "docs/Ravi_AI.pdf")
    
    llm = create_llm(cfg=cfg.llm)


    # start_time = time.time()
    # resume = parse_resume(pdf_path, llm)
    # logger.info(f"parsed resume in: {time.time() - start_time} seconds")


    
    # start_time = time.time()

    question = Question(text = 'you architected and deployed a production speech-to-speech voice bot using Pipecat. Could you walk me through the overall architecture of this system and highlight some of the key design decisions you made to achieve the reported call reduction and KYC turnaround improvements?', topic= 'experience')
    
    follow = generate_followup(question, "I used FastAPI with WebSockets for real-time communication", llm)
    print(follow)
    
    # questions = generate_questions(resume, cfg.interview, llm)
    # logger.info(f"generated questions in: {time.time() - start_time} seconds")
    
    # print(questions)



