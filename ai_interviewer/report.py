from dataclasses import dataclass
from typing import List, Any
from .llm import create_llm
from .question_gen import Question
from pydantic import BaseModel # type: ignore
import json

@dataclass
class AnswerEval:
    question: str
    topic: str
    answer: str
    score: int
    feedback: str


@dataclass
class InterviewReport:
    candidate_name: str
    evaluations: List[AnswerEval]
    overall_summary: str
    recommendation: str


# Schemas for Gemini structured output (native JSON mode).
class _EvalSchema(BaseModel):
    score: int
    feedback: str


class _ReportSchema(BaseModel):
    overall_summary: str
    recommendation: str

def evaluate_answer(question: Question, answer: str, llm: Any) -> AnswerEval:
    prompt = f"""
        question is : {question.text}
        topic is: {question.topic}
        answer is: {answer}
    """
    system = """
        You are an expert answer evaluator. Given the question, topic and the answer
        evaluate the answer based on the question. Give the score and your feedback

        Scoring guidance (be lenient and encouraging):
        - If the answer is relevant to the question and on-topic, the score must be
          at least 6, even if it is brief or lacks detail.
        - Reserve scores below 6 only for answers that are off-topic, empty, or do
          not actually address the question.
        - Reward concrete examples, metrics, and clear reasoning with higher scores
          (8-10), but do not penalise an otherwise relevant answer harshly for
          missing depth.

        Output format rules (strictly follow):
        - Return ONLY a JSON Dictionary, nothing else
        - Start your response with { and end with }
        - Do NOT wrap in markdown or code fences
        - Do NOT add any explanation before or after the JSON
        Format: {"score": 0-10, "feedback": your feedback}

        Below is an example of expected output. DO NOT parrot it in the actual output

        Example:
            Input:
                question: Tell me about a time you optimised a system for performance.
                topic: experience
                answer: I reduced API latency by 40% by caching frequent DB queries with
            Redis.

            Output:
            {"score": 8, "feedback": "Strong answer with a concrete metric and clear
            technical solution. Could mention trade-offs considered."}
    """

    response = llm.complete(prompt = prompt, system = system, response_schema = _EvalSchema)
    data = json.loads(response)
    return AnswerEval(question=question.text, topic=question.topic, answer=answer, score=data['score'], feedback=data['feedback'])


def evals_to_string(evals: List[AnswerEval]) -> str:
    answer = []
    for i, eval in enumerate(evals):
        curr = f"""
        ### Q{i+1}: {eval.question} (Topic: {eval.topic})
        - **Answer:** {eval.answer}
        - **Score:** {eval.score}
        - **Feedback:** {eval.feedback}
        """
        answer.append(curr)
    return "\n".join(answer)



def generate_report(candidate_name: str, evals: List[AnswerEval], llm: Any) -> InterviewReport:
    evals_text = evals_to_string(evals)
    prompt = f"""
        Following are the {len(evals)} evaluations:
        {evals_text}
    """
    system = f"""
        You are an expert summary generator and final recommendation master.
        Given the evaluation of a candidate for an interview, give me an overall summary and
        final recommendation: STRONG_HIRE|LEAN_HIRE|NO_HIRE

        Base the recommendation on the average of the per-question scores. Use this
        rubric and do NOT be harsher than it:
        - average score >= 8           -> STRONG_HIRE
        - average score between 6 and 8 -> LEAN_HIRE
        - average score < 6            -> NO_HIRE
        A candidate scoring 6 or 7 across questions is a LEAN_HIRE, not a NO_HIRE.
        Reserve NO_HIRE for genuinely weak interviews (mostly off-topic or low scores).
        The summary should be fair and constructive, consistent with this recommendation.

        Output format rules (strictly follow):
        - Return ONLY a JSON Dictionary, nothing else
        - Start your response with {{ and end with }}
        - Do NOT wrap in markdown or code fences
        - Do NOT add any explanation before or after the JSON
        Format: {{"overall_summary": summary, "recommendation": STRONG_HIRE|LEAN_HIRE|NO_HIRE}}
        
    """
    

    response = llm.complete(prompt = prompt, system = system, response_schema = _ReportSchema)
    data = json.loads(response)
    return InterviewReport(candidate_name = candidate_name, evaluations=evals, overall_summary=data['overall_summary'], recommendation=data['recommendation'])



def to_markdown(report: InterviewReport):
    evals = report.evaluations
    candidate_name = report.candidate_name

    question_text = evals_to_string(evals)
    markdown = f"""# Interview Report - {candidate_name}
## Per-Question Breakdown
{question_text}

## Overall Summary
{report.overall_summary}


## Recommendation
{report.recommendation}

"""

    return markdown


if __name__ == "__main__":
    from .config import load_config
    cfg = load_config("config.yaml")
    client = create_llm(cfg.llm)

    # eval = evaluate_answer(Question(text="What was your most challenging project", topic="experience"), "optimising llm inference optimisation", client)
    # print(eval)

    dummy_evals = [
      AnswerEval(
          question="Walk me through your experience with LLM inference optimization.",
          topic="experience",
          answer="I used quantization and batching to reduce latency by 40% on our production model.",
          score=8,
          feedback="Strong answer with concrete metrics. Could mention specific tools like vLLM or TGI."
      ),
      AnswerEval(
          question="Tell me about a project where you used FastAPI.",
          topic="projects",
          answer="I built a REST API for our KYC pipeline using FastAPI with async endpoints.",
          score=7,
          feedback="Good answer but lacks detail on how async improved throughput."
      ),
      AnswerEval(
          question="How do you handle model drift in production?",
          topic="skills",
          answer="I monitor metrics and retrain periodically but haven't set up automated drift detection yet.",
          score=5,
          feedback="Honest but shows a gap in MLOps maturity. No mention of monitoring tools."
      ),
  ]
    

    report = generate_report("Ravi", dummy_evals, client)
    # print(report)


    markdown = to_markdown(report)

    print(markdown)

    



    