"""Construct the configured LLM provider."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

import yaml

from core.llm.base import LLMProvider
from core.llm.ollama import OllamaLLMProvider


def _expand_placeholders(value: str) -> str:
    return os.path.expandvars(value)


def _load_yaml(path: Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, Mapping):
        raise ValueError("providers.yaml must be a mapping at the root")
    return data


def build_llm_provider(config_path: Path | None = None) -> LLMProvider:
    root = Path(__file__).resolve().parents[2]
    cfg_file = config_path or (root / "config" / "providers.yaml")
    cfg = _load_yaml(cfg_file)
    llm_cfg = cfg.get("llm") or {}
    provider = str(llm_cfg.get("provider") or "ollama").lower()

    if provider == "ollama":
        oc = llm_cfg.get("ollama") or {}
        base_url = _expand_placeholders(str(oc.get("base_url") or "")).strip()
        model = _expand_placeholders(str(oc.get("model") or "")).strip()
        if not base_url:
            base_url = os.environ.get("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
        if not model:
            model = os.environ.get("OLLAMA_MODEL", "llama3.2")
        return OllamaLLMProvider(base_url=base_url, model=model)

    raise ValueError(f"Unsupported LLM provider: {provider!r}")
