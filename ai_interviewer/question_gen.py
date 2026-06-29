from dataclasses import dataclass, field
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


@dataclass
class CodingQuestion:
    id: str
    title: str
    difficulty: str
    prompt: str                          # shown in the editor panel
    spoken_intro: str                    # read aloud by TTS (shorter, conversational)
    tests: list = field(default_factory=list)  # [{name, stdin, expected}] visible cases


# curated coding-question bank lives next to this module
_CODING_BANK = os.path.join(os.path.dirname(__file__), "data", "coding_questions.json")


def load_coding_questions(path: str = _CODING_BANK) -> List[CodingQuestion]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return [CodingQuestion(**q) for q in data]


def pick_coding_questions(n: int = 1, path: str = _CODING_BANK) -> List[CodingQuestion]:
    """Return n random questions from the bank (or all of them if n is larger)."""
    bank = load_coding_questions(path)
    return random.sample(bank, min(n, len(bank)))


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
        You are a warm, engaged technical interviewer in a real spoken conversation.
        Briefly react to the candidate's answer in a few natural words, then ask one
        follow-up question that digs deeper into something specific they said.
        Sound like a person talking, not a form.
        Return only JSON: {{"text": "...", "topic": "skills|experience|projects|education"}}
        The "text" is exactly what you'd say out loud (acknowledgment + question).
        No markdown, no explanation.
    """
    prompt = f"""
        The original question was: {question.text} on the topic: {question.topic}.
        The candidate's answer is: {answer}
    """

    response = llm.complete(prompt=prompt, system= system, response_schema = _QuestionSchema)

    data = json.loads(response)

    return Question(text=data['text'], topic=data['topic'])


def generate_followup_stream(question: Question, answer: str, llm: Any):
    """Stream a follow-up question's text token by token.

    Plain text, not JSON — structured output can't be streamed cleanly, and the
    spoken question only needs the text. The caller reuses the parent question's
    topic for scoring.
    """
    system = (
        "You are a warm, engaged technical interviewer in a real spoken "
        "conversation. First react to the candidate's answer in a few natural "
        "words (e.g. 'Got it', 'Nice, that makes sense'), then ask one follow-up "
        "question that digs deeper into something specific they said. Keep it to "
        "one or two sentences and sound like a person talking, not a form. Reply "
        "with ONLY what you'd say out loud — no labels, no JSON, no markdown."
    )
    prompt = f"""
        The original question was: {question.text} on the topic: {question.topic}.
        The candidate's answer is: {answer}
    """
    return llm.stream(prompt=prompt, system=system)


def generate_code_followup_stream(coding_q: "CodingQuestion", language: str, code: str, output: str, llm: Any):
    """Stream a spoken follow-up grounded on the candidate's submitted code."""
    system = (
        "You are a warm, engaged technical interviewer in a real spoken "
        "conversation. The candidate just solved a coding problem. Briefly react "
        "to their solution in a few natural words, then ask ONE follow-up about "
        "THEIR code — e.g. its time/space complexity, an edge case it might miss, "
        "or a design choice they made. Be specific to the code shown, keep it to "
        "one or two sentences, and sound like a person talking. Reply with ONLY "
        "what you'd say out loud — no labels, no markdown."
    )
    prompt = f"""
        Problem: {coding_q.prompt}
        Language: {language}
        The candidate's code:
        {code}
        Program output:
        {output}
    """
    return llm.stream(prompt=prompt, system=system)



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



