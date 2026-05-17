"""Ollama HTTP chat backend — swap via `config/providers.yaml` only."""

from __future__ import annotations

from typing import Sequence

import httpx

from core.llm.base import ChatMessage, GenerationParameters


class OllamaLLMProvider:
    def __init__(
        self,
        base_url: str,
        model: str,
        generation_parameters: GenerationParameters,
        timeout_s: float = 120.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._generation_parameters = generation_parameters
        self._timeout = timeout_s

    def complete(self, messages: Sequence[ChatMessage]) -> str:
        params = self._generation_parameters
        payload = {
            "model": self._model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "options": {
                "temperature": params.temperature,
                "top_p": params.top_p,
                "num_ctx": params.num_ctx,
            },
            "stream": False,
        }
        url = f"{self._base_url}/api/chat"
        try:
            with httpx.Client(timeout=self._timeout) as client:
                response = client.post(url, json=payload)
                response.raise_for_status()
                data = response.json()
        except httpx.ConnectError:
            raise RuntimeError(
                f"Ollama not reachable at {self._base_url}. "
                "Is it running? Start with: ollama serve"
            )
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                raise RuntimeError(
                    f"Model '{self._model}' not found. "
                    f"Pull it with: ollama pull {self._model}"
                )
            raise RuntimeError(f"Ollama returned HTTP {exc.response.status_code}")
        message = data.get("message") or {}
        content = message.get("content")
        if not isinstance(content, str):
            raise RuntimeError("Unexpected Ollama response shape")
        return content
