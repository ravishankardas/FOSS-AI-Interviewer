from dataclasses import dataclass
from typing import Optional

import httpx  # type: ignore
from loguru import logger  # type: ignore
import os
from .config import ExecutorConfig


# languages we support -> (piston language id, source file name)
_LANGS = {
    "python": ("python", "main.py"),
    "c++": ("c++", "main.cpp"),
}


@dataclass
class ExecResult:
    stdout: str
    stderr: str
    exit_code: int
    compile_error: str = ""   # non-empty only when a C++ build fails
    timed_out: bool = False
    error: str = ""           # engine/transport failure, not the user's code

    @property
    def ok(self) -> bool:
        """True when the code compiled and ran to completion (exit 0)."""
        return not self.error and not self.compile_error and self.exit_code == 0


class PistonExecutor:
    """Thin client over a self-hosted Piston engine. See docs/piston_setup.md."""

    def __init__(self, cfg: ExecutorConfig) -> None:
        self.cfg = cfg
        self._url = cfg.base_url.rstrip("/") + "/api/v2/execute"
        token = os.environ.get("PISTON_TOKEN")
        self._headers = {"Authorization": f"Bearer {token}"} if token else {}
        self._versions = {"python": cfg.python_version, "c++": cfg.cpp_version}

    def run(self, language: str, code: str, stdin: str = "") -> ExecResult:
        if language not in _LANGS:
            return ExecResult("", "", 1, error=f"unsupported language: {language!r}")

        piston_lang, filename = _LANGS[language]
        payload = {
            "language": piston_lang,
            "version": self._versions[language],
            "files": [{"name": filename, "content": code}],
            "stdin": stdin,
            "run_timeout": self.cfg.run_timeout_ms,
            "compile_timeout": self.cfg.compile_timeout_ms,
        }

        try:
            resp = httpx.post(self._url, json=payload, headers=self._headers, timeout=30.0)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            logger.warning(f"piston execute failed: {exc!r}")
            return ExecResult("", "", 1, error=f"execution engine error: {exc}")

        return self._parse(data)

    def _parse(self, data: dict) -> ExecResult:
        run = data.get("run", {}) or {}
        compile_ = data.get("compile", {}) or {}

        # a non-zero compile stage means the build failed; the run stage is empty
        compile_error = ""
        if compile_ and compile_.get("code", 0) != 0:
            compile_error = compile_.get("stderr", "") or compile_.get("output", "")

        # Piston reports a killed (timed-out) run via the `signal` field (e.g. SIGKILL)
        timed_out = bool(run.get("signal")) and run.get("code") is None

        return ExecResult(
            stdout=run.get("stdout", ""),
            stderr=run.get("stderr", ""),
            exit_code=run.get("code") if run.get("code") is not None else 1, # type: ignore
            compile_error=compile_error,
            timed_out=timed_out,
        )


if __name__ == "__main__":
    from .config import load_config

    cfg = load_config("config.yaml")
    ex = PistonExecutor(cfg.executor)

    print("python:", ex.run("python", "print(2 + 2)"))
    print("c++:", ex.run("c++", "#include <iostream>\nint main(){std::cout<<2+2;}"))
    print("py error:", ex.run("python", "import sys; sys.exit(3)"))
    print("cpp build fail:", ex.run("c++", "int main(){ this is not valid }"))
