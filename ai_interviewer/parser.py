from pdfminer.high_level import extract_text # type: ignore
from dataclasses import dataclass, field
from typing import List, Any
from .llm import create_llm
import os

import json
import time
store = {
}

def get_text_from_pdf(pdf_path):
    text = extract_text(pdf_path)
    return text


@dataclass
class ResumeData:
    name: str
    email: str
    phone: str
    skills: List[str]
    experience: List[dict]
    projects: List[dict]
    education: List[dict]


SYSTEM_PROMPT = """
    You are a resume parser. Extract structured information from the resume
    text and return valid JSON only.
    Do not include any explanation, markdown, or code fences. Return only the
    raw JSON object.

    The JSON must follow this exact structure:
    {
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


def parse_resume(pdf_path: str, llm: Any) -> ResumeData:
    if pdf_path in store:
        data = store[pdf_path]
    else:
        text = get_text_from_pdf(pdf_path)
        response = llm.complete(prompt=text, system=SYSTEM_PROMPT)
        cleaned = response.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned[cleaned.find("{"):cleaned.rfind("}") + 1]
        data = json.loads(cleaned)
        store[pdf_path] = data

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





    