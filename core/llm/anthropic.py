"""Anthropic Messages API — configure via routing candidates + ANTHROPIC_API_KEY."""

from __future__ import annotations

import os
from typing import Sequence

import httpx

from core.llm.base import ChatMessage, GenerationParameters


class AnthropicLLMProvider:
    def __init__(
        self,
        model: str,
        generation_parameters: GenerationParameters,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout_s: float = 120.0,
    ) -> None:
        key = (api_key or os.environ.get("ANTHROPIC_API_KEY") or "").strip()
        if not key:
            raise ValueError("Anthropic provider requires ANTHROPIC_API_KEY")
        self._api_key = key
        self._model = model
        self._generation_parameters = generation_parameters
        self._base = (base_url or os.environ.get("ANTHROPIC_BASE_URL") or "https://api.anthropic.com").rstrip(
            "/"
        )
        self._timeout = timeout_s

    def complete(self, messages: Sequence[ChatMessage]) -> str:
        params = self._generation_parameters
        system_parts: list[str] = []
        api_messages: list[dict[str, str]] = []
        for m in messages:
            if m.role == "system":
                system_parts.append(m.content)
            else:
                api_messages.append({"role": m.role, "content": m.content})
        payload: dict[str, object] = {
            "model": self._model,
            "temperature": params.temperature,
            "top_p": params.top_p,
            "max_tokens": params.num_ctx,
            "messages": api_messages,
        }
        if system_parts:
            payload["system"] = "\n".join(system_parts)
        url = f"{self._base}/v1/messages"
        headers = {
            "x-api-key": self._api_key,
            "anthropic-version": "2023-06-01",
        }
        with httpx.Client(timeout=self._timeout) as client:
            response = client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
        blocks = data.get("content") or []
        texts = [b.get("text") for b in blocks if isinstance(b, dict) and b.get("type") == "text"]
        if not texts or not all(isinstance(t, str) for t in texts):
            raise RuntimeError("Unexpected Anthropic response shape")
        return "".join(str(t) for t in texts)
