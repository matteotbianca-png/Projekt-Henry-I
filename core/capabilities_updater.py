"""Deterministic local capability cache updater for Henry's LLM router."""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path
from typing import Any, Mapping

import httpx

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CAPABILITIES_CACHE_PATH = PROJECT_ROOT / "config" / "model_capabilities_cache.json"

_STATIC_CLOUD_BASELINES: dict[str, dict[str, float]] = {
    "gpt_4o": {"base_skill": 0.90, "base_speed": 0.57, "base_cost": 0.72},
    "claude_3_5_sonnet": {"base_skill": 0.98, "base_speed": 0.52, "base_cost": 0.86},
}

_LOCAL_BLUEPRINTS: dict[str, dict[str, float]] = {
    "phi4": {"base_skill": 0.90, "base_speed": 0.45, "base_cost": 0.0},
    "mistral-nemo": {"base_skill": 0.75, "base_speed": 0.60, "base_cost": 0.0},
    "llama3.1": {"base_skill": 0.70, "base_speed": 0.80, "base_cost": 0.0},
    "qwen2.5": {"base_skill": 0.85, "base_speed": 0.60, "base_cost": 0.0},
}


def _cache_key(model_name: str) -> str:
    return model_name.strip().lower()


def _canonical_cache_key(model_name: str) -> str | None:
    base_name = _strip_minor_tag(model_name)
    if "qwen2.5" in base_name:
        return "qwen2_5"
    if "qwen3" in base_name:
        return "qwen3"
    if "llama3.2" in base_name:
        return "llama3_2"
    return None


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _strip_minor_tag(model_name: str) -> str:
    return model_name.strip().lower().split(":", 1)[0]


def _blueprint_for_model(model_name: str) -> dict[str, float] | None:
    base_name = _strip_minor_tag(model_name)
    for marker, values in _LOCAL_BLUEPRINTS.items():
        if marker in base_name:
            return dict(values)
    return None


def _fallback_for_model(model_name: str) -> dict[str, float]:
    text = model_name.lower()

    if "70b" in text:
        skill, speed = 0.88, 0.35
    elif "14b" in text or "13b" in text:
        skill, speed = 0.80, 0.50
    elif "7b" in text or "8b" in text or "9b" in text:
        skill, speed = 0.70, 0.65
    elif "1b" in text or "3b" in text:
        skill, speed = 0.60, 0.90
    else:
        skill, speed = 0.60, 0.70

    if "embed" in text:
        skill = 0.10
    return {
        "base_skill": _clamp(skill),
        "base_speed": _clamp(speed),
        "base_cost": 0.0,
    }


def _capability_for_local_model(model_name: str) -> dict[str, float]:
    return _blueprint_for_model(model_name) or _fallback_for_model(model_name)


def _read_cache() -> dict[str, Any]:
    if not CAPABILITIES_CACHE_PATH.is_file():
        return {}
    data = json.loads(CAPABILITIES_CACHE_PATH.read_text(encoding="utf-8"))
    return dict(data) if isinstance(data, Mapping) else {}


def _write_cache(data: Mapping[str, Any]) -> None:
    CAPABILITIES_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=str(CAPABILITIES_CACHE_PATH.parent),
        delete=False,
    ) as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")
        tmp_path = Path(handle.name)
    tmp_path.replace(CAPABILITIES_CACHE_PATH)


def _discover_local_ollama_models(base_url: str, timeout_s: float) -> list[str]:
    try:
        with httpx.Client(timeout=timeout_s) as client:
            response = client.get(f"{base_url.rstrip('/')}/api/tags")
            response.raise_for_status()
            data = response.json()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Local Ollama capability inventory unavailable at %s: %s", base_url, exc)
        return []

    models_raw = data.get("models") if isinstance(data, Mapping) else None
    if not isinstance(models_raw, list):
        logger.warning("Local Ollama capability inventory returned unexpected response shape.")
        return []

    models: list[str] = []
    for row in models_raw:
        if not isinstance(row, Mapping):
            continue
        model = str(row.get("model") or row.get("name") or "").strip()
        if model:
            models.append(model)
    return sorted(set(models))


def refresh_capabilities_from_local_inventory(
    *,
    base_url: str = "http://localhost:11434",
    cache_path: Path | None = None,
    timeout_s: float = 5.0,
) -> dict[str, Any]:
    """Run a one-shot local Ollama inventory pass and rewrite the capability cache."""
    global CAPABILITIES_CACHE_PATH

    original_path = CAPABILITIES_CACHE_PATH
    if cache_path is not None:
        CAPABILITIES_CACHE_PATH = cache_path
    try:
        existing = _read_cache()
        live_models = _discover_local_ollama_models(base_url, timeout_s)
        if not live_models:
            if existing:
                logger.warning("Keeping existing capability cache because no local Ollama inventory was available.")
                return existing
            cache = {key: dict(value) for key, value in _STATIC_CLOUD_BASELINES.items()}
            _write_cache(cache)
            return cache

        cache: dict[str, Any] = {
            key: dict(value)
            for key, value in _STATIC_CLOUD_BASELINES.items()
        }
        for key, value in existing.items():
            if str(key).strip().lower() in _STATIC_CLOUD_BASELINES and isinstance(value, Mapping):
                cache[str(key)] = dict(value)

        for model in live_models:
            values = _capability_for_local_model(model)
            cache[_cache_key(model)] = values
            canonical_key = _canonical_cache_key(model)
            if canonical_key is not None:
                cache[canonical_key] = values

        _write_cache(cache)
        logger.info("Capability cache refreshed from local Ollama inventory (%d models).", len(live_models))
        print(
            f"[Henry Router] Capability cache refreshed from local Ollama inventory ({len(live_models)} models).",
            flush=True,
        )
        return cache
    finally:
        CAPABILITIES_CACHE_PATH = original_path
