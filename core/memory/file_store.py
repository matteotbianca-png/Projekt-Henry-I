"""Filesystem-backed memory under a configurable root (e.g. /mnt/secure_memory)."""

from __future__ import annotations

import re
from pathlib import Path

_SAFE_KEY = re.compile(r"[^a-zA-Z0-9._-]+")


def _normalize_key(key: str) -> str:
    cleaned = _SAFE_KEY.sub("_", key.strip())
    if not cleaned:
        raise ValueError("memory key must contain at least one safe character")
    if ".." in cleaned or "/" in cleaned or "\\" in cleaned:
        raise ValueError("memory key must not contain path separators")
    return cleaned[:200]


class FileMemoryStore:
    def __init__(self, base_path: Path) -> None:
        self._base = base_path
        self._kv = base_path / "kv"

    def is_available(self) -> bool:
        try:
            self._base.mkdir(parents=True, exist_ok=True)
            self._kv.mkdir(parents=True, exist_ok=True)
        except OSError:
            return False
        return self._base.is_dir() and self._kv.is_dir() and self._is_writable(self._kv)

    @staticmethod
    def _is_writable(directory: Path) -> bool:
        probe = directory / ".henry_write_probe"
        try:
            probe.write_text("", encoding="utf-8")
            probe.unlink(missing_ok=True)
            return True
        except OSError:
            return False

    def put(self, key: str, value: str) -> None:
        safe = _normalize_key(key)
        if not self.is_available():
            raise RuntimeError("memory store is not available (mount / permissions?)")
        path = self._kv / f"{safe}.txt"
        path.write_text(value, encoding="utf-8")

    def get(self, key: str) -> str | None:
        safe = _normalize_key(key)
        path = self._kv / f"{safe}.txt"
        if not path.is_file():
            return None
        return path.read_text(encoding="utf-8")
