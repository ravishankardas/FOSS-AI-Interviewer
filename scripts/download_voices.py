"""Download the Piper voices listed in config.yaml's `tts.voices` registry.

    python -m scripts.download_voices           # fetch any missing voices
    python -m scripts.download_voices --force    # re-download everything

Each Piper voice is two files from HuggingFace (rhasspy/piper-voices): the
`.onnx` model and its `.onnx.json` config — Piper needs both, side by side. The
local target path comes straight from the registry; the remote path is the same
minus the leading `models/` segment.
"""
import argparse
import os
import urllib.request

HF_BASE = "https://huggingface.co/rhasspy/piper-voices/resolve/main/"
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _remote_url(local_path: str) -> str:
    # models/en/en_US/amy/medium/en_US-amy-medium.onnx
    #   ->  <HF>/en/en_US/amy/medium/en_US-amy-medium.onnx
    rel = local_path.replace("\\", "/")
    if rel.startswith("models/"):
        rel = rel[len("models/"):]
    return HF_BASE + rel


def _download(url: str, dest: str, force: bool) -> None:
    if os.path.exists(dest) and not force:
        print(f"    have {os.path.relpath(dest, BASE_DIR)}")
        return
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    print(f"    get  {url}")
    urllib.request.urlretrieve(url, dest)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="re-download even if present")
    args = ap.parse_args()

    from ai_interviewer.config import load_config

    cfg = load_config(os.path.join(BASE_DIR, "config.yaml"))
    voices = cfg.tts.voices or {cfg.tts.voice: cfg.tts.model_path}

    for name, path in voices.items():
        local = path if os.path.isabs(path) else os.path.join(BASE_DIR, path)
        print(name)
        _download(_remote_url(path), local, args.force)              # .onnx
        _download(_remote_url(path) + ".json", local + ".json", args.force)  # .onnx.json

    print(f"\n{len(voices)} voice(s) ready.")


if __name__ == "__main__":
    main()
