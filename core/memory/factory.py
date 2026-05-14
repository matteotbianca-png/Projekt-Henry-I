"""Construct the configured memory backend."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

import yaml

from core.memory.base import MemoryStore
from core.memory.file_store import FileMemoryStore


def _expand_placeholders(value: str) -> str:
    return os.path.expandvars(value)


def _load_yaml(path: Path) -> Mapping[str, Any]:
    raw = path.read_text(encoding="utf-8")
    data = yaml.safe_load(os.path.expandvars(raw)) or {}
    if not isinstance(data, Mapping):
        raise ValueError("providers.yaml must be a mapping at the root")
    return data


def build_memory_store(config_path: Path | None = None) -> MemoryStore:
    root = Path(__file__).resolve().parents[2]
    cfg_file = config_path or (root / "config" / "providers.yaml")
    cfg = _load_yaml(cfg_file)
    mem_cfg = cfg.get("memory") or {}
    provider = str(mem_cfg.get("provider") or "file").lower()

    if provider == "file":
        fc = mem_cfg.get("file") or {}
        raw = str(fc.get("base_path") or "").strip()
        base = Path(_expand_placeholders(raw) if raw else "").expanduser()
        if not str(base):
            env_path = os.environ.get("HENRY_MEMORY_PATH", "/mnt/secure_memory")
            base = Path(env_path)
        return FileMemoryStore(base_path=base)

    raise ValueError(f"Unsupported memory provider: {provider!r}")
