from dataclasses import dataclass
import json
import os
import time

from .llm import create_llm
from .parser import ResumeData, parse_resume
from .config import InterviewConfig
from typing import Any, List
from pprint import pprint
from loguru import logger # type: ignore


@dataclass
class Question:
    text: str
    topic: str

SYSTEM_PROMPT = """
  You are an expert technical interviewer conducting a real job interview.
  Given a candidate's resume, generate exactly {n} interview questions.

  Rules:
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

  Format: [{{"text": "question here", "topic":
  "skills|experience|projects|education"}}]
  """

def _resume_to_text(resume: ResumeData):

    lines = []
    lines.append(f"Name: {resume.name}")
    lines.append(f"Skills: {', '.join(resume.skills)}")
    lines.append("Experience: ")

    for idx, exp in enumerate(resume.experience):
        lines.append(f"  {idx+1}. {exp['title']} at {exp['company']}({exp['duration']})")
        for i, b in enumerate(exp.get('bullets', [])):
            lines.append(f"    ({chr(ord('a') + i)}) {b}")

    
    lines.append("Projects:")
    for idx, proj in enumerate(resume.projects):
        lines.append(f"  ({chr(ord('a') + idx)}) {proj['name']} using {', '.join(proj.get('tech',[]))}")

    lines.append("Education:")
    for idx, edu in enumerate(resume.education):
        lines.append(f"  ({chr(ord('a') + idx)}) {edu['degree']} from {edu['institution']}")

    return "\n".join(lines)


def generate_questions(resume: ResumeData, cfg: InterviewConfig, llm: Any) ->List[Question]:
    resume_text = _resume_to_text(resume)

    system = SYSTEM_PROMPT.format(n = cfg.max_questions)
    response = llm.complete(prompt = resume_text, system = system)

    cleaned = response.strip()
    cleaned = cleaned[cleaned.find("["):cleaned.rfind("]") + 1]

    data = json.loads(cleaned)
    return [Question(text=q['text'], topic=q['topic']) for q in data]



if __name__ == "__main__":
    from .config import load_config
    cfg = load_config("config.yaml")
    pdf_path = os.path.join(os.getcwd(), "docs/Ravi_AI.pdf")
    
    llm = create_llm(cfg=cfg.llm)

    start_time = time.time()
    resume = parse_resume(pdf_path, llm)
    logger.info(f"parsed resume in: {time.time() - start_time} seconds")


    
    start_time = time.time()
    questions = generate_questions(resume, cfg.interview, llm)
    logger.info(f"generated questions in: {time.time() - start_time} seconds")
    
    print(questions)



