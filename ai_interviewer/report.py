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
    evidence: str = ""
    ideal_answer: str = ""   # what a strong answer/approach looks like


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
    evidence: str
    ideal_answer: str


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
        evaluate the answer based on the question. Give the score, your feedback, the
        evidence, and an ideal_answer.
        - evidence: a short VERBATIM quote from the candidate's answer that best
          justifies the score. Quote the answer exactly — do not paraphrase or invent
          text. If the answer is empty or off-topic, return an empty evidence string.
        - ideal_answer: 2-3 sentences describing what a strong answer to THIS question
          would cover — the key points, structure, or concrete examples the candidate
          could have given. This is teaching material, so write it even for good
          answers (what would make it a 10).

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
        Format: {"score": 0-10, "feedback": your feedback, "evidence": verbatim quote from the answer, "ideal_answer": what a strong answer would cover}

        Below is an example of expected output. DO NOT parrot it in the actual output

        Example:
            Input:
                question: Tell me about a time you optimised a system for performance.
                topic: experience
                answer: I reduced API latency by 40% by caching frequent DB queries with
            Redis.

            Output:
            {"score": 8, "feedback": "Strong answer with a concrete metric and clear
            technical solution. Could mention trade-offs considered.", "evidence":
            "I reduced API latency by 40% by caching frequent DB queries with Redis.",
            "ideal_answer": "A strong answer names the bottleneck and how it was found
            (profiling/metrics), the fix and why it worked, the measured impact, and a
            trade-off considered such as cache invalidation or staleness."}
    """

    # temperature=0.0 → the same answer scores the same on every run
    response = llm.complete(prompt = prompt, system = system, response_schema = _EvalSchema, temperature = 0.0)
    data = json.loads(response)
    return AnswerEval(question=question.text, topic=question.topic, answer=answer, score=data['score'], feedback=data['feedback'], evidence=data.get('evidence', ''), ideal_answer=data.get('ideal_answer', ''))


def evaluate_code(title: str, problem: str, language: str, code: str,
                  visible_results: list, hidden_results: list,
                  followup_q: str, followup_a: str, llm: Any,
                  dialogue: str = "", optimal: dict = None,
                  optimize_asked: bool = False) -> AnswerEval:
    # dedicated coding grader: correctness-first, not bound by the lenient verbal
    # rubric. Judges the problem + actual code + test outcome + spoken explanation.
    #
    # hidden_results never reach the candidate's report — they sharpen the score
    # (anti-gaming) but are fed to the grader only as an aggregate pass ratio, and
    # the report's answer line shows the VISIBLE count alone.
    vpass, vtotal = sum(1 for r in visible_results if r["passed"]), len(visible_results)
    hpass, htotal = sum(1 for r in hidden_results if r["passed"]), len(hidden_results)
    total = vtotal + htotal
    passed = vpass + hpass

    if total:
        vlines = "\n".join(f"  - {r['name']}: {'PASS' if r['passed'] else 'FAIL'}"
                           for r in visible_results)
        test_block = f"{vpass}/{vtotal} visible tests passed\n{vlines}"
        if htotal:
            # aggregate only — never the hidden cases' names or I/O
            test_block += f"\nHidden tests (not shown to candidate): {hpass}/{htotal} passed"
    else:
        test_block = "No automated tests available."

    optimal = optimal or {}
    _optimal_text = (
        f"time {optimal.get('time', '?')}, space {optimal.get('space', '?')}"
        if optimal else "not specified"
    )
    _optimize_asked_text = "yes" if optimize_asked else "no"

    system = """
        You are an expert technical interviewer grading a candidate's solution to a
        coding problem. Score from 0 to 10 based primarily on CORRECTNESS, then on
        approach/efficiency, then on how clearly they explained it.

        Rubric (do NOT be more lenient than this):
        - Passes all tests, clean approach, clear explanation: 8-10
        - Passes all tests but brute-force or weakly explained: 6-7
        - Partially correct (passes some tests): 3-5
        - Fails most tests or fundamentally wrong: 0-2
        Do not inflate the score for failing code just because the explanation sounds
        plausible. Correctness is shown by the automated test results.

        Interviewer assistance also affects the score. Below is the spoken exchange
        during the round; judge how much help the candidate needed. MORE help = LOWER
        score:
        - Solved it on their own with little or no help: no penalty.
        - Needed a nudge or two: small reduction.
        - Needed repeated or increasingly direct hints, or the approach spelled out:
          significant reduction — a solution they largely couldn't reach on their own
          should not score 8-10.
        Ignore small talk and clarifying questions about the problem; only count real
        hints/guidance toward the solution.
        If the candidate was asked to optimize a working solution, judge efficiency
        against the optimal complexity given below: improving it is a plus; leaving
        it clearly sub-optimal after being asked should temper the score.

        Also return ideal_answer: 2-3 sentences describing an optimal, clean approach
        to THIS problem (the key idea and its time/space complexity), so the candidate
        can learn what a strong solution looks like — even if theirs already passed.

        Output rules (strict):
        - Return ONLY a JSON object, starting with { and ending with }
        - No markdown, no code fences, no text before or after
        Format: {"score": 0-10, "feedback": "...", "evidence": "short note citing the test outcome or a specific part of the code", "ideal_answer": "the optimal approach + complexity"}
    """
    prompt = f"""
        Problem: {title}
        {problem}

        Candidate's {language} solution:
        {code}

        Automated test results:
        {test_block}

        Interviewer's follow-up question: {followup_q}
        Candidate's spoken answer: {followup_a}

        Spoken exchange during the round:
        {dialogue or '(the candidate did not speak during coding)'}

        Optimal complexity for this problem: {_optimal_text}
        Was the candidate asked to optimize their solution? {_optimize_asked_text}
    """
    response = llm.complete(prompt=prompt, system=system, response_schema=_EvalSchema, temperature=0.0)
    data = json.loads(response)
    # report shows only what the candidate could see; hidden cases stay out of it
    answer = f"[{language}] solution — {vpass}/{vtotal} tests passed.\nExplanation: {followup_a}"
    return AnswerEval(question=f"{title} (coding)", topic="coding", answer=answer,
                      score=data["score"], feedback=data["feedback"], evidence=data.get("evidence", ""),
                      ideal_answer=data.get("ideal_answer", ""))


def evals_to_string(evals: List[AnswerEval]) -> str:
    answer = []
    for i, eval in enumerate(evals):
        evidence_line = f"\n        - **Evidence:** \"{eval.evidence}\"" if eval.evidence else ""
        ideal_line = f"\n        - **Model answer:** {eval.ideal_answer}" if eval.ideal_answer else ""
        curr = f"""
        ### Q{i+1}: {eval.question} (Topic: {eval.topic})
        - **Answer:** {eval.answer}
        - **Score:** {eval.score}
        - **Feedback:** {eval.feedback}{evidence_line}{ideal_line}
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
    

    response = llm.complete(prompt = prompt, system = system, response_schema = _ReportSchema, temperature = 0.0)
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

    



    