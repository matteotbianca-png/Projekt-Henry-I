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
    raw = path.read_text(encoding="utf-8")
    data = yaml.safe_load(os.path.expandvars(raw)) or {}
    if not isinstance(data, Mapping):
        raise ValueError(f"{path.name} must be a mapping at the root")
    return data


def build_llm_provider(config_path: Path | None = None) -> LLMProvider:
    root = Path(__file__).resolve().parents[2]
    henry_path = root / "config" / "config.yaml"
    if henry_path.is_file():
        henry = _load_yaml(henry_path)
        routing = henry.get("routing") or {}
        if isinstance(routing, Mapping) and routing.get("enabled"):
            from core.llm_manager import build_routed_llm

            return build_routed_llm(root)

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
