"""Abstract LLM interface — implementations live beside this module."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence, runtime_checkable


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str


@runtime_checkable
class LLMProvider(Protocol):
    """Any chat-capable model backend (Ollama, HTTP APIs, etc.)."""

    def complete(self, messages: Sequence[ChatMessage]) -> str:
        """Return the assistant text for the given conversation slice."""
        ...
