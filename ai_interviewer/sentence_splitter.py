import re

# match a run of text ending in sentence punctuation, followed by whitespace
# or the end of the buffer. DOTALL so sentences spanning token newlines match.
_BOUNDARY = re.compile(r"(.+?[.!?])(\s+|$)", re.S)


class SentenceSplitter:
    """Incremental sentence splitter for streaming LLM output.

    Feed it text deltas as they arrive; it returns whole sentences as soon as
    their terminating punctuation shows up, buffering the trailing partial until
    more text (or flush) completes it.
    """

    def __init__(self) -> None:
        self._buf = ""

    def feed(self, text: str) -> list[str]:
        self._buf += text
        out: list[str] = []
        while True:
            m = _BOUNDARY.match(self._buf)
            if not m:
                break
            sentence = m.group(1).strip()
            if sentence:
                out.append(sentence)
            self._buf = self._buf[m.end():]
        return out

    def flush(self) -> list[str]:
        """Return any trailing text not terminated by punctuation."""
        tail = self._buf.strip()
        self._buf = ""
        return [tail] if tail else []
