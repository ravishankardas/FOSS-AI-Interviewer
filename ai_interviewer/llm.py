from typing import Iterator
from llama_cpp import Llama
from .config import LLMConfig


class LLMClient:
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
        return response["choices"][0]["message"]["content"] # type: ignore

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
            delta = chunk["choices"][0]["delta"] # type: ignore
            if "content" in delta:
                yield delta["content"] # type: ignore


if __name__ == "__main__":
    from .config import load_config
    cfg = load_config("config.yaml")
    client = LLMClient(cfg.llm)
    print(client.complete("What is the capital of France?"))
