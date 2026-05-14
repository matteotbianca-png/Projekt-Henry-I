"""OpenAI Chat Completions backend — configure via routing candidates + OPENAI_API_KEY."""

from __future__ import annotations

import os
from typing import Sequence

import httpx

from core.llm.base import ChatMessage


class OpenAILLMProvider:
    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout_s: float = 120.0,
    ) -> None:
        key = (api_key or os.environ.get("OPENAI_API_KEY") or "").strip()
        if not key:
            raise ValueError("OpenAI provider requires OPENAI_API_KEY")
        self._api_key = key
        self._model = model
        self._base = (base_url or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip(
            "/"
        )
        self._timeout = timeout_s

    def complete(self, messages: Sequence[ChatMessage]) -> str:
        payload = {
            "model": self._model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
        }
        url = f"{self._base}/chat/completions"
        headers = {"Authorization": f"Bearer {self._api_key}"}
        with httpx.Client(timeout=self._timeout) as client:
            response = client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError("OpenAI response missing choices")
        message = choices[0].get("message") or {}
        content = message.get("content")
        if not isinstance(content, str):
            raise RuntimeError("Unexpected OpenAI response shape")
        return content
