"""Ollama HTTP chat backend — swap via `config/providers.yaml` only."""

from __future__ import annotations

from typing import Sequence

import httpx

from core.llm.base import ChatMessage


class OllamaLLMProvider:
    def __init__(self, base_url: str, model: str, timeout_s: float = 120.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = timeout_s

    def complete(self, messages: Sequence[ChatMessage]) -> str:
        payload = {
            "model": self._model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": False,
        }
        url = f"{self._base_url}/api/chat"
        with httpx.Client(timeout=self._timeout) as client:
            response = client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
        message = data.get("message") or {}
        content = message.get("content")
        if not isinstance(content, str):
            raise RuntimeError("Unexpected Ollama response shape")
        return content
