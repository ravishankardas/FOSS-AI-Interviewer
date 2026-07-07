from dataclasses import dataclass, field
import hashlib
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
    starter: dict = field(default_factory=dict)  # {python, c++} input-reading scaffold
    tests: list = field(default_factory=list)  # [{name, stdin, expected}] visible cases
    hidden_tests: list = field(default_factory=list)  # never shown; scoring-only, anti-gaming
    optimal: dict = field(default_factory=dict)  # {time, space} best complexity, for the optimize step


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


@dataclass
class Turn:
    """One asked-and-answered exchange, fed back in as adaptive history."""
    question: Question
    answer: str
    mastery: int | None = None  # the judge's read of THIS answer, filled in next turn


class _AdaptiveSchema(BaseModel):
    last_mastery: int          # 0-10 read of the most recent answer
    next_text: str             # the next question, spoken verbatim
    next_topic: str            # skills|experience|projects|education
    next_difficulty: str       # easy|medium|hard

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


ADAPTIVE_SYSTEM_PROMPT = """
  You are an expert technical interviewer running a live, adaptive interview.
  You are given the candidate's resume and the exchange so far (the questions
  you've already asked and how they answered). Do two things, in one JSON object:

  1. Rate the MOST RECENT answer from 0 to 10 ("last_mastery"):
     - 0-3: struggled, vague, wrong, or didn't really answer
     - 4-6: partial — correct shape but shallow or missing depth
     - 7-10: strong, specific, confident, technically sound

  2. Choose the NEXT question, adapting to that rating:
     - If they struggled (low mastery): make it EASIER, or pivot to a different
       topic where they may be stronger. Don't pile on.
     - If they did okay (mid): stay on a similar level, probe a nearby area.
     - If they nailed it (high mastery): go DEEPER or HARDER on what they know.

  Rules for the next question:
  - Specific to THIS resume, never generic
  - Do not repeat or closely echo any question already asked
  - Conversational, as if spoken out loud; a single question
  - "next_difficulty" must honestly reflect how hard the question is

  Output ONLY a JSON object with keys:
  last_mastery (int 0-10), next_text (string), next_topic
  (skills|experience|projects|education), next_difficulty (easy|medium|hard).
  No markdown, no code fences, no extra text.
  """


def _history_to_text(history: List["Turn"]) -> str:
    """Render the exchange so far for the adaptive prompt."""
    if not history:
        return "(no questions asked yet)"
    lines = []
    for i, t in enumerate(history, 1):
        lines.append(f"Q{i} [{t.question.topic}]: {t.question.text}")
        lines.append(f"A{i}: {t.answer or '(no answer)'}")
    return "\n".join(lines)


# ── dev-only persistent cache for the adaptive-question LLM call ──────────
# Keyed by the resume + exchange-so-far ONLY (persona is excluded, since the
# random interviewer voice changes per run and would bust every hit). On disk
# so it survives the frequent restarts of `--reload` during development. Enabled
# only when LITMUS_QGEN_CACHE is set — it must never be on in production, where
# every interview should generate fresh.
_QGEN_CACHE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), ".qgen_cache"
)


def _qgen_cache_enabled() -> bool:
    return bool(os.environ.get("LITMUS_QGEN_CACHE"))


def _qgen_cache_key(resume: ResumeData, history: List["Turn"]) -> str:
    blob = _resume_to_text(resume) + "\n---\n" + _history_to_text(history)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _qgen_cache_get(key: str):
    path = os.path.join(_QGEN_CACHE_DIR, f"{key}.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def _qgen_cache_put(key: str, data: dict) -> None:
    os.makedirs(_QGEN_CACHE_DIR, exist_ok=True)
    with open(os.path.join(_QGEN_CACHE_DIR, f"{key}.json"), "w", encoding="utf-8") as f:
        json.dump(data, f)


def generate_adaptive_question(
    resume: ResumeData, history: List["Turn"], llm: Any, persona: str = ""
) -> tuple[Question, int]:
    """Pick the next question on the fly from the resume + exchange so far.

    A single structured call does double duty: it rates the most recent answer
    (the cheap steering signal) and produces the next question targeted to that
    rating. Returns (next_question, last_mastery); last_mastery is 0 when there's
    no prior answer to rate (the first adaptive turn).

    `persona` (optional) is a sentence identifying the interviewer (name +
    gender) prepended to the system prompt so the voice and persona line up.
    """
    cache_on = _qgen_cache_enabled()
    key = _qgen_cache_key(resume, history) if cache_on else None
    if cache_on:
        data = _qgen_cache_get(key)
        if data is not None:
            logger.info("[qgen] cache hit — skipping Gemini call (dev cache)")
            question = Question(text=data["next_text"], topic=data["next_topic"])
            mastery = int(data["last_mastery"]) if history else 0
            return question, mastery

    prompt = (
        f"Resume:\n{_resume_to_text(resume)}\n\n"
        f"Exchange so far:\n{_history_to_text(history)}"
    )
    system = f"{persona}\n{ADAPTIVE_SYSTEM_PROMPT}" if persona else ADAPTIVE_SYSTEM_PROMPT
    response = llm.complete(
        prompt=prompt, system=system, response_schema=_AdaptiveSchema
    )
    data = json.loads(response)
    if cache_on:
        _qgen_cache_put(key, data)
    question = Question(text=data["next_text"], topic=data["next_topic"])
    mastery = int(data["last_mastery"]) if history else 0
    return question, mastery


ADAPTIVE_STREAM_SYSTEM_PROMPT = """
  You are an expert technical interviewer running a live, adaptive interview. You
  are given the candidate's resume and the exchange so far. Silently judge how the
  candidate handled their MOST RECENT answer, then ask the NEXT question, adapting:
  - struggled -> make it easier, or pivot to a different area they may be stronger in
  - did okay  -> probe a nearby area at a similar level
  - nailed it -> go deeper or harder on what they know

  Rules:
  - Specific to THIS resume, never generic
  - Do not repeat or closely echo any question already asked
  - Conversational, as if spoken out loud; a single question
  - First briefly acknowledge their previous answer in a few natural words, then ask.
  Reply with ONLY what you'd say out loud — no labels, no JSON, no markdown.
  """


def generate_adaptive_question_stream(resume: ResumeData, history: List["Turn"],
                                      llm: Any, persona: str = ""):
    """Stream the next adaptive question's text token by token, so TTS can start on
    the first sentence before the whole question is written. Same adaptation as
    generate_adaptive_question, but plain-text/streamed — the model still steers off
    the most recent answer internally; we just don't surface the mastery integer.
    """
    prompt = (
        f"Resume:\n{_resume_to_text(resume)}\n\n"
        f"Exchange so far:\n{_history_to_text(history)}"
    )
    system = (f"{persona}\n{ADAPTIVE_STREAM_SYSTEM_PROMPT}"
              if persona else ADAPTIVE_STREAM_SYSTEM_PROMPT)
    return llm.stream(prompt=prompt, system=system)


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


def generate_interruption_response_stream(question: Question, interjection: str, llm: Any, persona: str = ""):
    """Stream the interviewer's spoken reaction when the candidate barges in.

    The interviewer was cut off partway through asking `question`; `interjection`
    is what the candidate blurted over it. Respond naturally so the conversation
    doesn't lose the thread — then the caller listens for the real answer.
    """
    system = (
        (persona + " " if persona else "")
        + "You are a warm, engaged technical interviewer in a real spoken "
        "conversation. You were partway through asking a question when the "
        "candidate cut in. In ONE or two natural sentences: acknowledge what they "
        "said, and — if they sound confused or asked you to repeat — briefly "
        "restate the gist of your question so it isn't lost; if they've already "
        "started answering, just encourage them to keep going. Do NOT answer the "
        "question yourself. Sound like a person talking, not a form. Reply with "
        "ONLY what you'd say out loud — no labels, no markdown."
    )
    prompt = f"""
        The question you were asking: {question.text} (topic: {question.topic})
        The candidate interrupted you and said: {interjection}
    """
    return llm.stream(prompt=prompt, system=system)


CODING_REPLY_SYSTEM = (
    "You are a warm, engaged technical interviewer in a real spoken conversation, "
    "watching the candidate solve a coding problem live. You can see their current "
    "code, their latest program/test output, and the conversation so far. Respond "
    "naturally to what they JUST said, choosing how in context:\n"
    "- If they ask about the PROBLEM (constraints, input/output format, examples, "
    "edge cases): answer, but stay GUARDED — never reveal or confirm the approach/"
    "algorithm. If they're really fishing for how to solve it, gently deflect "
    "('that part's for you to figure out') and restate what the problem asks.\n"
    "- If they're STUCK or asking for a hint: give only as much as they need, and "
    "ESCALATE GRADUALLY across the conversation — a gentle nudge first; get more "
    "direct only if the history shows you've already nudged and they're still "
    "stuck. Never give the full solution or write their code.\n"
    "- If they ask about THEIR OWN code or an error they're seeing: look at their "
    "code and output and point them toward the failing case or suspect logic as a "
    "GUARDED nudge — don't rewrite it or hand them the fix.\n"
    "- If they're just thinking out loud or making small talk: give a brief, warm "
    "acknowledgement that keeps them going; don't reveal anything.\n"
    "Keep it to one or two natural sentences, sound like a person talking, and "
    "reply with ONLY what you'd say out loud — no labels, no markdown, no code."
)


def generate_coding_reply_stream(coding_q: "CodingQuestion", language: str, code: str,
                                 output: str, utterance: str, history: str,
                                 llm: Any, persona: str = ""):
    """Stream the interviewer's spoken reply to something the candidate said while
    coding — one call that decides in-context (guarded clarify / graduated hint /
    debug nudge / encouragement) so there's no extra classify round-trip before
    audio. `history` is the recent spoken exchange, which drives hint escalation.
    """
    system = (persona + " " if persona else "") + CODING_REPLY_SYSTEM
    prompt = f"""
        Problem: {coding_q.prompt}
        Language: {language}
        Their code so far:
        {code or '(nothing written yet)'}
        Latest program output:
        {output or '(none)'}
        Conversation so far:
        {history or '(this is their first remark)'}
        The candidate just said: {utterance}
    """
    return llm.stream(prompt=prompt, system=system)


class _OptimalitySchema(BaseModel):
    can_improve: bool
    reason: str


def assess_optimizability(coding_q: "CodingQuestion", language: str, code: str, llm: Any) -> bool:
    """Judge whether the candidate's (working) solution is meaningfully worse than
    the problem's optimal complexity and could realistically be improved. Only
    returns True when there's worthwhile headroom, so we don't nag on already-
    optimal solutions. No stored optimum → False (nothing to compare against).
    """
    optimal = coding_q.optimal or {}
    if not optimal:
        return False
    system = (
        "You assist a technical interviewer. Given a coding problem, its OPTIMAL "
        "time and space complexity, and the candidate's WORKING solution, infer "
        "the candidate's actual time and space complexity from their code and "
        "decide whether it is meaningfully WORSE than optimal with worthwhile "
        "headroom to improve (e.g. O(n^2) where O(n) is possible, or O(n) space "
        "where O(1) is possible). If it already matches the optimal, or is only "
        "trivially different, return can_improve=false. "
        'Return ONLY JSON: {"can_improve": bool, "reason": "one short phrase"}.'
    )
    prompt = f"""
        Problem: {coding_q.prompt}
        Optimal complexity: time {optimal.get('time', '?')}, space {optimal.get('space', '?')}
        Language: {language}
        Candidate's solution:
        {code}
    """
    try:
        resp = llm.complete(prompt=prompt, system=system,
                            response_schema=_OptimalitySchema, temperature=0.0)
        return bool(json.loads(resp).get("can_improve", False))
    except Exception:
        return False


def generate_optimize_prompt_stream(coding_q: "CodingQuestion", language: str, code: str, llm: Any, persona: str = ""):
    """Stream the interviewer asking the candidate to improve a working-but-
    suboptimal solution — nudging at which of time/space has headroom, without
    handing over the optimal approach."""
    optimal = coding_q.optimal or {}
    system = (
        (persona + " " if persona else "")
        + "You are a warm, engaged technical interviewer. The candidate's solution "
        "works and passes the tests, but it can be more efficient. In one or two "
        "natural spoken sentences: briefly congratulate them that it works, then "
        "ask them to improve its time or space complexity. Point at WHICH one has "
        "room to improve, but do NOT reveal the optimal approach or write code. "
        "Reply with ONLY what you'd say out loud — no labels, no markdown."
    )
    prompt = f"""
        Problem: {coding_q.prompt}
        Optimal complexity: time {optimal.get('time', '?')}, space {optimal.get('space', '?')}
        Language: {language}
        Their current solution:
        {code}
    """
    return llm.stream(prompt=prompt, system=system)


def generate_code_followup_stream(coding_q: "CodingQuestion", language: str, code: str, output: str, llm: Any, persona: str = ""):
    """Stream a spoken follow-up grounded on the candidate's submitted code.

    `persona` (optional) prepends the interviewer's name/gender to the prompt.
    """
    system = (
        (persona + " " if persona else "")
        + "You are a warm, engaged technical interviewer in a real spoken "
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



