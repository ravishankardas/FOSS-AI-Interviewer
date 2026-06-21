import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from unittest.mock import MagicMock
from ai_interviewer.question_gen import Question, generate_questions, generate_followup, _resume_to_text
from ai_interviewer.parser import ResumeData
from ai_interviewer.config import InterviewConfig

# reusable fake resume
def make_resume():
    return ResumeData(
        name="John Doe",
        email="john@example.com",
        phone="1234567890",
        skills=["Python", "FastAPI"],
        experience=[{"title": "Engineer", "company": "Acme", "duration":
"2023-2024", "bullets": ["Built stuff"]}],
        projects=[{"name": "MyProject", "tech": ["Python"], "bullets":
["Did things"]}],
        education=[{"degree": "B.Tech", "institution": "MIT"}]
    )


def make_mock_llm(response: str):
    llm = MagicMock()
    llm.complete.return_value = response
    return llm


def test_generate_questions_returns_correct_count():
    fake_response = '[{"text": "Tell me about Python", "topic": "skills"}]'

    llm = make_mock_llm(fake_response)
    cfg = InterviewConfig(max_questions=1)
    questions = generate_questions(make_resume(), cfg, llm)

    assert len(questions) == 1
    assert isinstance(questions[0], Question)

def test_generate_questions_returns_question_objects():
    fake_response = '[{"text": "Tell me about Python", "topic": "skills"}, {"text": "Tell me about FastAPI", "topic": "skills"}]'
    llm = make_mock_llm(fake_response)
    cfg = InterviewConfig(max_questions=2)
    questions = generate_questions(make_resume(), cfg, llm)

    assert all(isinstance(q, Question) for q in questions)


def test_resume_to_text_contains_name():
    resume = make_resume()
    text = _resume_to_text(resume)

    assert "John Doe" in text
    assert "Python" in text
    assert "Acme" in text


def test_generate_followup_returns_question():
    fake_response = '{"text": "Can you elaborate?", "topic":"experience"}'
    llm = make_mock_llm(fake_response)

    q = Question(text = "Tell me about yourself", topic="experience")

    followup = generate_followup(q, "I built voice bots", llm)

    assert isinstance(followup, Question)
    assert followup.text == "Can you elaborate?"




