"""Persist finished interviews to disk so candidates can revisit them.

One JSON file per interview under history/ (repo root, gitignored). Small and
dependency-free — good enough for a local, single-user practice tool.
"""
import json
import os
import re
import time
import uuid

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HISTORY_DIR = os.path.join(BASE_DIR, "history")

# ids we generate + accept back from the URL: date-time + short random suffix
_ID_RE = re.compile(r"^[0-9]{8}-[0-9]{6}-[0-9a-f]{6}$")

# fields returned in the list view (the full record also carries `markdown`)
_SUMMARY_FIELDS = ("id", "candidate_name", "created_at", "recommendation", "avg_score")


def save_interview(candidate_name: str, recommendation: str,
                   avg_score: float, markdown: str) -> str:
    os.makedirs(HISTORY_DIR, exist_ok=True)
    hid = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
    record = {
        "id": hid,
        "candidate_name": candidate_name,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "recommendation": recommendation,
        "avg_score": round(avg_score, 1),
        "markdown": markdown,
    }
    with open(os.path.join(HISTORY_DIR, hid + ".json"), "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
    return hid


def list_interviews() -> list:
    if not os.path.isdir(HISTORY_DIR):
        return []
    out = []
    for fn in os.listdir(HISTORY_DIR):
        if not fn.endswith(".json"):
            continue
        try:
            with open(os.path.join(HISTORY_DIR, fn), encoding="utf-8") as f:
                r = json.load(f)
            out.append({k: r.get(k) for k in _SUMMARY_FIELDS})
        except (OSError, json.JSONDecodeError):
            continue
    out.sort(key=lambda r: r.get("created_at") or "", reverse=True)
    return out


def get_interview(hid: str) -> dict | None:
    # validate the id so a crafted URL can't escape HISTORY_DIR
    if not _ID_RE.match(hid or ""):
        return None
    path = os.path.join(HISTORY_DIR, hid + ".json")
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)
