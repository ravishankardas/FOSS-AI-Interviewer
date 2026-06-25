from typing import Iterator
from .config import LLMConfig
import os
import sys
import time
import httpx # type: ignore
from google import genai # type: ignore
from dotenv import load_dotenv # type: ignore
from loguru import logger # type: ignore

load_dotenv(".env")


class LocalLLMClient:
    def __init__(self, cfg: LLMConfig):
        # imported lazily so the default (gemini) path doesn't require
        # llama-cpp-python to be installed
        from llama_cpp import Llama  # type: ignore
        self._llm = Llama(
            model_path=cfg.model_path,
            n_ctx=cfg.n_ctx,
            n_gpu_layers=cfg.n_gpu_layers,
            verbose=False,
        )
        self._temperature = cfg.temperature

    def complete(self, prompt: str, system: str = "", response_schema=None) -> str:
        # response_schema is a no-op here: llama_cpp has no native structured
        # output, so JSON callers must keep relying on prompt instructions.
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = self._llm.create_chat_completion(
            messages=messages,
            temperature=self._temperature,
        )
        return response["choices"][0]["message"]["content"]   # type: ignore

    def stream(self, prompt: str, system: str = "") -> Iterator[str]:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        for chunk in self._llm.create_chat_completion(
            messages=messages,
            temperature=self._temperature,
            stream=True,
        ):
            delta = chunk["choices"][0]["delta"]  # type: ignore
            if "content" in delta:
                yield delta["content"]  # type: ignore


# transport-level failures the genai SDK does NOT retry (it only retries on
# HTTP status codes). These happen when an idle pooled socket is dropped.
_TRANSIENT = (httpx.RemoteProtocolError, httpx.ConnectError, httpx.ReadError)


def _with_retry(fn, *, attempts: int = 3, base_delay: float = 0.5):
    """Call fn(), retrying transport-level connection drops with backoff."""
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except _TRANSIENT as exc:
            if attempt == attempts:
                raise
            delay = base_delay * (2 ** (attempt - 1))
            logger.warning(f"gemini transient error ({exc!r}), retry {attempt}/{attempts - 1} in {delay}s")
            time.sleep(delay)


class GeminiLLMClient:
    def __init__(self, cfg: LLMConfig):
        # disable keep-alive reuse: each call opens a fresh connection so we
        # never reuse a socket the server dropped during an idle pause.
        self._client = genai.Client(
            api_key=os.environ["GEMINI_API_KEY"],
            http_options=genai.types.HttpOptions(
                client_args={"limits": httpx.Limits(max_keepalive_connections=0)},
            ),
        )
        self.model_name = cfg.model_name

    def complete(self, prompt: str, system: str = "", response_schema=None) -> str:
        # disable model "thinking" — these are simple structured tasks and the
        # thinking phase adds several seconds of latency for no benefit.
        config_kwargs = {"thinking_config": genai.types.ThinkingConfig(thinking_budget=0)}
        if system:
            config_kwargs["system_instruction"] = system
        if response_schema is not None:
            # native JSON mode: API guarantees valid, parseable JSON
            config_kwargs["response_mime_type"] = "application/json"
            config_kwargs["response_schema"] = response_schema
        config = genai.types.GenerateContentConfig(**config_kwargs)
        response = _with_retry(
            lambda: self._client.models.generate_content(model=self.model_name, contents=prompt, config=config)
        )
        return response.text  # type: ignore

    def stream(self, prompt: str, system: str = "") -> Iterator[str]:
        config_kwargs = {"thinking_config": genai.types.ThinkingConfig(thinking_budget=0)}
        if system:
            config_kwargs["system_instruction"] = system
        config = genai.types.GenerateContentConfig(**config_kwargs)
        stream = _with_retry(
            lambda: self._client.models.generate_content_stream(model=self.model_name, contents=prompt, config=config)
        )
        for chunk in stream:
            if chunk.text:
                yield chunk.text


def create_llm(cfg: LLMConfig):
    if cfg.provider == "gemini":
        logger.info("using gemini llm")
        return GeminiLLMClient(cfg)
    logger.info("using local llm")
    return LocalLLMClient(cfg) # type: ignore


if __name__ == "__main__":
    from .config import load_config
    cfg = load_config("config.yaml")
    client = create_llm(cfg.llm)
    print(client.complete("What is the capital of France?"))