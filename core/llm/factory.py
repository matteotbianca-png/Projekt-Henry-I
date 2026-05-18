"""Construct the configured LLM provider."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

import yaml

from core.llm.base import GenerationParameters, LLMProvider
from core.llm.ollama import OllamaLLMProvider


def _expand_placeholders(value: str) -> str:
    return os.path.expandvars(value)


def _load_yaml(path: Path) -> Mapping[str, Any]:
    raw = path.read_text(encoding="utf-8")
    data = yaml.safe_load(os.path.expandvars(raw)) or {}
    if not isinstance(data, Mapping):
        raise ValueError(f"{path.name} must be a mapping at the root")
    return data


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _generation_parameters_for_model(root: Path, model: str) -> GenerationParameters:
    prefs_path = root / "config" / "routing_preferences.yaml"
    prefs = _load_yaml(prefs_path)
    params = prefs.get("model_parameters")
    if not isinstance(params, Mapping):
        raise ValueError("routing_preferences.yaml requires model_parameters")

    selected = params.get("defaults")
    model_key = model.strip().lower()
    for label, block in params.items():
        if not isinstance(block, Mapping):
            continue
        aliases = block.get("aliases") or []
        alias_values = [str(alias).strip().lower() for alias in aliases] if isinstance(aliases, list) else []
        if model_key == str(label).strip().lower() or model_key in alias_values:
            selected = block
            break

    if not isinstance(selected, Mapping):
        raise ValueError("model_parameters.defaults is required")
    return GenerationParameters(
        temperature=max(0.0, min(1.0, _as_float(selected.get("temperature"), 0.3))),
        top_p=max(0.0, min(1.0, _as_float(selected.get("top_p"), 0.9))),
        num_ctx=max(1, _as_int(selected.get("num_ctx"), 4096)),
    )


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
            base_url = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
        if not model:
            model = os.environ.get("OLLAMA_MODEL", "llama3.2")
        return OllamaLLMProvider(
            base_url=base_url,
            model=model,
            generation_parameters=_generation_parameters_for_model(root, model),
        )

    raise ValueError(f"Unsupported LLM provider: {provider!r}")
