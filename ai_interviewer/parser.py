from pdfminer.high_level import extract_text # type: ignore
from dataclasses import dataclass, field
from typing import List, Any
from .llm import create_llm
from pydantic import BaseModel # type: ignore
import os

import json
import time
import hashlib

# parse cache keyed by the PDF's content hash (not its path) — every upload
# lands at a fresh uploads/{uuid}.pdf, so a path key would miss on re-upload of
# the same file and re-hit the LLM. Content hash makes the same bytes free.
store = {
}

def get_text_from_pdf(pdf_path):
    text = extract_text(pdf_path)
    return text


def _file_hash(pdf_path: str) -> str:
    with open(pdf_path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


@dataclass
class ResumeData:
    name: str
    email: str
    phone: str
    skills: List[str]
    experience: List[dict]
    projects: List[dict]
    education: List[dict]


# Schemas for Gemini structured output (native JSON mode).
class _ExperienceSchema(BaseModel):
    title: str
    company: str
    duration: str
    bullets: List[str]


class _ProjectSchema(BaseModel):
    name: str
    tech: List[str]
    bullets: List[str]


class _EducationSchema(BaseModel):
    degree: str
    institution: str


class _ResumeSchema(BaseModel):
    is_resume: bool
    name: str
    email: str
    phone: str
    skills: List[str]
    experience: List[_ExperienceSchema]
    projects: List[_ProjectSchema]
    education: List[_EducationSchema]


SYSTEM_PROMPT = """
    You are a resume parser. Extract structured information from the resume
    text and return valid JSON only.
    Do not include any explanation, markdown, or code fences. Return only the
    raw JSON object.

    First decide whether the text is actually a person's resume / CV (i.e. it
    describes an individual's work experience, skills, education, or projects).
    Set "is_resume" to true if so, or false for anything else (an invoice, a
    report, a random article, a blank/garbled document, etc.). If it is not a
    resume, still return the other fields as empty strings / empty lists.

    The JSON must follow this exact structure:
    {
        "is_resume": true,
        "name": "string",
        "email": "string",
        "phone": "string",
        "skills": ["string"],
        "experience": [
            {
                "title": "string",
                "company": "string",
                "duration": "string",
                "bullets": ["string"]
            }
        ],
        "projects": [
            {
                "name": "string",
                "tech": ["string"],
                "bullets": ["string"]
            }
        ],
        "education": [
            {
                "degree": "string",
                "institution": "string"
            }
        ]
    }
"""


# below this, an extracted PDF has too little text to be a real resume
# (e.g. a scanned/image-only PDF that yields no selectable text)
MIN_RESUME_TEXT_CHARS = 80


def _parse_raw(pdf_path: str, llm: Any, text: str | None = None) -> dict:
    """Run (and cache) the structured parse, returning the raw JSON dict.
    Cache is keyed by content hash so the same PDF re-uploaded under a new
    path is free."""
    key = _file_hash(pdf_path)
    if key in store:
        return store[key]
    if text is None:
        text = get_text_from_pdf(pdf_path)
    response = llm.complete(prompt=text, system=SYSTEM_PROMPT, response_schema=_ResumeSchema)
    data = json.loads(response)
    store[key] = data
    return data


def check_is_resume(pdf_path: str, llm: Any) -> bool:
    """True if the PDF looks like a real resume. Rejects near-empty PDFs
    without an LLM call, then trusts the parser's own is_resume verdict.
    Caches the parse so a later parse_resume() on the same path is free."""
    text = get_text_from_pdf(pdf_path)
    if len(text.strip()) < MIN_RESUME_TEXT_CHARS:
        return False
    data = _parse_raw(pdf_path, llm, text=text)
    return bool(data.get("is_resume", True))


def parse_resume(pdf_path: str, llm: Any) -> ResumeData:
    data = _parse_raw(pdf_path, llm)

    return ResumeData(
        name=data.get("name", ""),
        email=data.get("email", ""),
        phone=data.get("phone", ""),
        skills=data.get("skills", []),
        experience=data.get("experience", []),
        projects=data.get("projects", []),
        education=data.get("education", []),
    )


if __name__ == "__main__":
    from .config import load_config
    cfg = load_config("config.yaml")
    pdf_path = os.path.join(os.getcwd(), "docs/Ravi_AI.pdf")
    
    llm = create_llm(cfg=cfg.llm)
    start_time = time.time()
    resp = parse_resume(pdf_path, llm)
    end_time = time.time()
    print(f"time_taken: {round(end_time - start_time, 2)} seconds\n")
    from pprint import pprint
    pprint(resp)





    