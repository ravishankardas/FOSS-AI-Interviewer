import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from unittest.mock import MagicMock
from ai_interviewer.question_gen import Question
from ai_interviewer.report import (
    evaluate_answer,
    evaluate_code,
    generate_report,
    AnswerEval,
)


def make_mock_llm(response: str):
    llm = MagicMock()
    llm.complete.return_value = response
    return llm


# The scoring calls MUST run at temperature 0 so a given answer/solution scores
# the same on every run. Gemini otherwise samples at ~1.0 and scores drift.

def test_evaluate_answer_scores_deterministically():
    llm = make_mock_llm('{"score": 7, "feedback": "ok", "evidence": "built voice bots"}')
    q = Question(text="Tell me about yourself", topic="experience")

    ev = evaluate_answer(q, "I built voice bots", llm)

    assert isinstance(ev, AnswerEval) and ev.score == 7
    assert llm.complete.call_args.kwargs.get("temperature") == 0.0


def test_evaluate_code_scores_deterministically():
    llm = make_mock_llm('{"score": 9, "feedback": "clean", "evidence": "all tests pass"}')
    visible = [{"name": "example", "passed": True}]
    hidden = [{"name": "hidden 1", "passed": True}]

    ev = evaluate_code(
        title="Two Sum", problem="add two numbers", language="python",
        code="print(1)", visible_results=visible, hidden_results=hidden,
        followup_q="why?", followup_a="because", llm=llm,
    )

    assert ev.score == 9
    assert llm.complete.call_args.kwargs.get("temperature") == 0.0


def test_generate_report_scores_deterministically():
    llm = make_mock_llm('{"overall_summary": "solid", "recommendation": "LEAN_HIRE"}')
    evals = [AnswerEval(question="Q", topic="skills", answer="A", score=7, feedback="ok")]

    report = generate_report("Ravi", evals, llm)

    assert report.recommendation == "LEAN_HIRE"
    assert llm.complete.call_args.kwargs.get("temperature") == 0.0
