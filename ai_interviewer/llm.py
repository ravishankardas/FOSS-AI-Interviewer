from typing import Iterator
from llama_cpp import Llama
from .config import LLMConfig
import os
from google import genai # type: ignore
from dotenv import load_dotenv # type: ignore
from loguru import logger # type: ignore

load_dotenv(".env")

class LocalLLMClient:
    def __init__(self, cfg: LLMConfig):
        self._llm = Llama(
            model_path=cfg.model_path,
            n_ctx=cfg.n_ctx,
            n_gpu_layers=cfg.n_gpu_layers,
            verbose=False,
        )
        self._temperature = cfg.temperature

    def complete(self, prompt: str, system: str = "") -> str:
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


class GeminiLLMClient:
    def __init__(self, cfg: LLMConfig):
        self._client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        self.model_name = cfg.model_name

    def complete(self, prompt: str, system: str = "") -> str:
        config = genai.types.GenerateContentConfig(system_instruction=system) if system else None
        response = self._client.models.generate_content(model=self.model_name, contents=prompt, config=config)
        return response.text  # type: ignore

    def stream(self, prompt: str, system: str = "") -> Iterator[str]:
        config = genai.types.GenerateContentConfig(system_instruction=system) if system else None
        for chunk in self._client.models.generate_content_stream(model=self.model_name, contents=prompt, config=config):
            if chunk.text:
                yield chunk.text


def create_llm(cfg: LLMConfig):
    if cfg.provider == "gemini":
        logger.info("using gemini llm")
        return GeminiLLMClient(cfg)
    return LocalLLMClient(cfg)


if __name__ == "__main__":
    from .config import load_config
    cfg = load_config("config.yaml")
    client = create_llm(cfg.llm)
    print(client.complete("What is the capital of France?"))