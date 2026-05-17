"""Abstract LLM interface — implementations live beside this module."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence, runtime_checkable


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str


@dataclass(frozen=True)
class GenerationParameters:
    """Explicit generation controls loaded from config/routing_preferences.yaml."""

    temperature: float
    top_p: float
    num_ctx: int


@runtime_checkable
class LLMProvider(Protocol):
    """Any chat-capable model backend (Ollama, HTTP APIs, etc.)."""

    def complete(self, messages: Sequence[ChatMessage]) -> str:
        """Return the assistant text for the given conversation slice."""
        ...
