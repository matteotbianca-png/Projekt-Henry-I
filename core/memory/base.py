"""Abstract memory interface — swap filesystem vs vector DB via config."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class MemoryStore(Protocol):
    """Long-term memory backend (encrypted volume, remote store, etc.)."""

    def is_available(self) -> bool:
        """False when the secure volume is not mounted or path is unusable."""
        ...

    def put(self, key: str, value: str) -> None:
        ...

    def get(self, key: str) -> str | None:
        ...
