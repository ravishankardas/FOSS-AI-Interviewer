"""Command-line entry point for FinalRound, the AI interviewer.

Installed as the `ai-interviewer` command (see pyproject.toml). Run:

    ai-interviewer --start

to launch the backend server in this terminal and open the frontend in Chrome.
Add --reload to auto-restart the server whenever code changes (dev mode).
"""
import argparse
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser

# repo root = parent of this package, used to find config.yaml
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _open_chrome(url: str) -> None:
    """Open url in Chrome if we can find it, else the default browser."""
    # 1) browsers registered with the webbrowser module
    try:
        webbrowser.get("chrome").open(url)
        return
    except webbrowser.Error:
        pass

    # 2) chrome/chromium on PATH
    for name in ("google-chrome", "chrome", "chromium", "chromium-browser"):
        path = shutil.which(name)
        if path:
            subprocess.Popen([path, url])
            return

    # 3) common Windows install locations
    if sys.platform == "win32":
        candidates = [
            r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
            os.path.expandvars(r"%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe"),
        ]
        for path in candidates:
            if os.path.exists(path):
                subprocess.Popen([path, url])
                return

    # 4) fall back to whatever the OS default browser is
    webbrowser.open(url)


def _open_when_ready(url: str) -> None:
    """Poll the server until it responds, then open the browser."""
    for _ in range(240):  # up to ~2 min while models load
        try:
            urllib.request.urlopen(url, timeout=1)
            break
        except Exception:
            time.sleep(0.5)
    _open_chrome(url)


def _load_server_settings():
    """Read host/port from config.yaml, falling back to sensible defaults."""
    try:
        from ai_interviewer.config import load_config

        cfg = load_config(os.path.join(BASE_DIR, "config.yaml"))
        return cfg.server.host, cfg.server.port
    except Exception:
        return "0.0.0.0", 8000


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="ai-interviewer", description="FinalRound — AI interviewer"
    )
    parser.add_argument(
        "--start", action="store_true",
        help="start the server and open the frontend in Chrome",
    )
    parser.add_argument("--host", default=None, help="override server host")
    parser.add_argument("--port", type=int, default=None, help="override server port")
    parser.add_argument(
        "--no-browser", action="store_true", help="don't open the browser",
    )
    parser.add_argument(
        "--reload", action="store_true",
        help="auto-restart the server when code changes (dev mode)",
    )
    args = parser.parse_args()

    if not args.start:
        parser.print_help()
        return

    host, port = _load_server_settings()
    host = args.host or host
    port = args.port or port

    url = f"http://localhost:{port}/"
    if not args.no_browser:
        threading.Thread(target=_open_when_ready, args=(url,), daemon=True).start()

    mode = " [reload]" if args.reload else ""
    print(f"Starting FinalRound on {url}{mode} (Ctrl+C to stop)")
    import uvicorn

    # reload watches the whole repo; restarts the server on any .py change, plus
    # config.yaml (so voice/config edits take effect without a manual restart).
    # note: a restart re-runs startup (models reload), so keep this dev-only.
    uvicorn.run(
        "backend.main:app", host=host, port=port,
        reload=args.reload,
        reload_dirs=[BASE_DIR] if args.reload else None,
        reload_includes=["*.py", "*.yaml"] if args.reload else None,
    )


if __name__ == "__main__":
    main()
