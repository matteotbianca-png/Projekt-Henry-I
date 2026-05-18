"""Dynamic multi-factor LLM routing for Henry (Presidio, scoring, benchmark refresh)."""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, Sequence

import httpx
import yaml

from core.capabilities_updater import refresh_capabilities_from_local_inventory
from core.health import (
    HealthProbe,
    READINESS_PROBE_TIMEOUT_S,
    chroma_persist_path,
    probe_chroma_vector_memory,
    probe_embeddings,
    probe_ollama,
    probe_satellite_status,
    probe_sqlite_personal_memory,
    run_with_timeout,
)
from core.llm.anthropic import AnthropicLLMProvider
from core.llm.base import ChatMessage, GenerationParameters, LLMProvider
from core.llm.ollama import OllamaLLMProvider
from core.llm.openai import OpenAILLMProvider
from core.queue_manager import TaskPayload, get_default_queue_manager

logger = logging.getLogger(__name__)

SECURITY_FLIGHT_MODE: bool = False
presidio_online: bool = False
_PRESIDIO_OFFLINE_REASON = ""

_ANSI_RESET = "\033[0m"
_ANSI_BOLD = "\033[1m"
_ANSI_RED = "\033[31m"
_ANSI_GREEN = "\033[32m"
_ANSI_YELLOW = "\033[33m"
_ANSI_CYAN = "\033[36m"
_ANSI_DIM = "\033[2m"

_MEMORY_INTENT_SYSTEM = (
    "You classify whether a single user message contains information Henry should remember "
    "long-term about this user: stable preferences, standing instructions, recurring lists, "
    "their own name or role, names of people/pets they identify as theirs, or facts they "
    "clearly state as true about themselves. Casual chat, one-off questions, or hypotheticals "
    "with no durable fact should not be remembered.\n"
    "Reply with one JSON object only, no markdown or code fences. Schema:\n"
    '{"remember": <boolean>, "facts": [{"category": "<short label>", "fact": "<one concise sentence>"}]}\n'
    "If remember is false, facts must be []. Each fact must be self-contained and safe to store."
)

# --- Types -----------------------------------------------------------------


@dataclass(frozen=True)
class ModelCandidate:
    """One routable model endpoint (local or cloud)."""

    id: str
    provider: str
    deployment: str
    model: str
    capability: float
    privacy_impact: float
    cost: float
    latency_ms: float
    generation_parameters: GenerationParameters
    base_skill: float
    base_speed: float
    base_cost: float
    benchmark_aliases: tuple[str, ...] = ()


@dataclass
class PIIScanResult:
    has_pii: bool
    analyzer_hits: int
    method: str


@dataclass
class PresidioRoutingContext:
    """Structured output of Presidio analysis used for privacy scoring."""

    has_pii: bool
    hit_count: int
    counts_by_type: dict[str, int]
    severity: float
    method: str


@dataclass(frozen=True)
class PrivacyFromPresidioConfig:
    local_severity_boost: float
    cloud_severity_discount: float
    cloud_anonymization_trust: float
    severity_cap: float
    entity_weights: Mapping[str, float]


@dataclass(frozen=True)
class SupervisorConfig:
    enabled: bool
    provider: str
    model: str
    timeout_seconds: float
    blend_weight: float


@dataclass
class RoutingWeights:
    w1_capability: float
    w2_privacy: float
    w3_cost: float
    w4_latency: float
    pii_privacy_weight_multiplier: float


@dataclass
class AutonomousIntelConfig:
    enabled: bool
    refresh_interval_seconds: float
    cache_path: Path
    request_timeout_seconds: float
    sources: tuple[dict[str, str], ...]


@dataclass(frozen=True)
class RoutingPreferenceSliders:
    skill_importance: float
    speed_priority: float
    cost_saving: float
    privacy_priority: float


@dataclass(frozen=True)
class RoutingPreferences:
    global_weights: RoutingPreferenceSliders
    cloud_allowance: float
    tier2_local_efficiency_threshold: float
    tier2_lookup_efficiency_threshold: float
    tier3_local_efficiency_threshold: float
    model_parameters: Mapping[str, GenerationParameters]
    model_aliases: Mapping[str, str]


@dataclass(frozen=True)
class ModelCapability:
    base_skill: float
    base_speed: float
    base_cost: float


PrivacyTier = Literal["tier1_strict_local", "tier2_anonymized", "tier3_public"]
EscalationAction = Literal[
    "strict_local_runtime_safety",
    "tier2_local_only",
    "tier2_micro_query",
    "tier2_full_context_cloud",
    "tier3_local_only",
    "tier3_open_competition",
]
ExecutionMode = Literal["auto", "sync", "background"]


@dataclass(frozen=True)
class CriticReview:
    approved: bool
    corrected_text: str | None
    raw_response: str


@dataclass(frozen=True)
class RoutingEscalationDecision:
    privacy_tier: PrivacyTier
    action: EscalationAction
    allowed_candidate_ids: frozenset[str]
    cloud_score_multiplier: float
    local_efficiency_ratio: float
    lookup_efficiency_ratio: float
    reason: str


@dataclass(frozen=True)
class EnvironmentLedger:
    """Boot-time discovery snapshot used for routing safety decisions."""

    runtime_probes: Mapping[str, HealthProbe]
    ollama_base_url: str
    ollama_online: bool
    local_models: tuple[str, ...]
    cloud_keys: Mapping[str, bool]
    memory_mount_path: str
    encrypted_storage_online: bool
    memory_collections: Mapping[str, bool]
    presidio_online: bool
    security_flight_mode: bool
    usable_candidate_ids: tuple[str, ...]
    degraded_reasons: tuple[str, ...]


# --- Config loading --------------------------------------------------------


def _expand_placeholders(value: str) -> str:
    return os.path.expandvars(value)


def _is_unresolved_placeholder(value: str) -> bool:
    return "${" in value or "$" in value


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_slider(value: Any, default: float) -> float:
    return max(0.0, min(1.0, _as_float(value, default)))


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _load_yaml(path: Path) -> Mapping[str, Any]:
    raw = path.read_text(encoding="utf-8")
    data = yaml.safe_load(os.path.expandvars(raw)) or {}
    if not isinstance(data, Mapping):
        raise ValueError(f"{path} must be a mapping at the root")
    return data


def _load_json(path: Path) -> Mapping[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        raise ValueError(f"{path} must be a mapping at the root")
    return data


def load_routing_config(
    root: Path,
) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    """Return Henry config, providers, routing preferences, and model capability cache."""
    cfg_path = root / "config" / "config.yaml"
    prov_path = root / "config" / "providers.yaml"
    prefs_path = root / "config" / "routing_preferences.yaml"
    caps_path = root / "config" / "model_capabilities_cache.json"
    henry = _load_yaml(cfg_path) if cfg_path.is_file() else {}
    providers = _load_yaml(prov_path) if prov_path.is_file() else {}
    if not prefs_path.is_file():
        raise FileNotFoundError("Missing required config/routing_preferences.yaml")
    if not caps_path.is_file():
        raise FileNotFoundError("Missing required config/model_capabilities_cache.json")
    preferences = _load_yaml(prefs_path)
    capability_cache = _load_json(caps_path)
    return henry, providers, preferences, capability_cache


def _normalize_model_key(value: str) -> str:
    return value.strip().lower()


def _parse_generation_parameters(raw: Mapping[str, Any], *, label: str) -> GenerationParameters:
    missing = [
        key
        for key in ("temperature", "top_p", "num_ctx")
        if key not in raw or raw.get(key) is None
    ]
    if missing:
        raise ValueError(f"model_parameters.{label} is missing: {', '.join(missing)}")
    return GenerationParameters(
        temperature=_as_slider(raw.get("temperature"), 0.3),
        top_p=_as_slider(raw.get("top_p"), 0.9),
        num_ctx=max(1, _as_int(raw.get("num_ctx"), 4096)),
    )


def parse_routing_preferences(raw: Mapping[str, Any]) -> RoutingPreferences:
    weights_raw = raw.get("global_weights")
    if not isinstance(weights_raw, Mapping):
        raise ValueError("routing_preferences.yaml requires global_weights")
    tier_raw = raw.get("tier_overrides")
    tier2 = tier_raw.get("tier2_anonymized") if isinstance(tier_raw, Mapping) else {}
    if not isinstance(tier2, Mapping):
        tier2 = {}
    tier3 = tier_raw.get("tier3_public") if isinstance(tier_raw, Mapping) else {}
    if not isinstance(tier3, Mapping):
        tier3 = {}
    params_raw = raw.get("model_parameters")
    if not isinstance(params_raw, Mapping):
        raise ValueError("routing_preferences.yaml requires model_parameters")

    model_parameters: dict[str, GenerationParameters] = {}
    model_aliases: dict[str, str] = {}
    for label, block in params_raw.items():
        if not isinstance(block, Mapping):
            continue
        key = str(label).strip()
        params = _parse_generation_parameters(block, label=key)
        model_parameters[key] = params
        model_aliases[_normalize_model_key(key)] = key
        aliases = block.get("aliases") or []
        if isinstance(aliases, list):
            for alias in aliases:
                alias_text = str(alias).strip()
                if alias_text:
                    model_aliases[_normalize_model_key(alias_text)] = key

    if "defaults" not in model_parameters:
        raise ValueError("model_parameters.defaults is required")

    return RoutingPreferences(
        global_weights=RoutingPreferenceSliders(
            skill_importance=_as_slider(weights_raw.get("skill_importance"), 0.8),
            speed_priority=_as_slider(weights_raw.get("speed_priority"), 0.45),
            cost_saving=_as_slider(weights_raw.get("cost_saving"), 0.65),
            privacy_priority=_as_slider(weights_raw.get("privacy_priority"), 0.9),
        ),
        cloud_allowance=_as_slider(tier2.get("cloud_allowance"), 0.85),
        tier2_local_efficiency_threshold=_as_slider(tier2.get("local_efficiency_threshold"), 0.9),
        tier2_lookup_efficiency_threshold=_as_slider(tier2.get("lookup_efficiency_threshold"), 0.9),
        tier3_local_efficiency_threshold=_as_slider(tier3.get("local_efficiency_threshold"), 0.9),
        model_parameters=model_parameters,
        model_aliases=model_aliases,
    )


def generation_parameters_for_model(model: str, preferences: RoutingPreferences) -> GenerationParameters:
    key = preferences.model_aliases.get(_normalize_model_key(model))
    if key is None:
        key = "defaults"
    return preferences.model_parameters[key]


def parse_model_capabilities_cache(raw: Mapping[str, Any]) -> dict[str, ModelCapability]:
    required_keys = {"base_skill", "base_speed", "base_cost"}
    capabilities: dict[str, ModelCapability] = {}
    for label, entry in raw.items():
        key = str(label).strip()
        if not key:
            raise ValueError("model_capabilities_cache.json contains an empty model key")
        if not isinstance(entry, Mapping):
            raise ValueError(f"Capability cache entry {key!r} must be an object")
        entry_keys = set(str(k) for k in entry.keys())
        if entry_keys != required_keys:
            raise ValueError(
                f"Capability cache entry {key!r} must contain exactly: "
                f"{', '.join(sorted(required_keys))}"
            )
        parsed_values: dict[str, float] = {}
        for field_name, value in {
            "base_skill": entry.get("base_skill"),
            "base_speed": entry.get("base_speed"),
            "base_cost": entry.get("base_cost"),
        }.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"Capability cache {key}.{field_name} must be numeric")
            parsed = float(value)
            if parsed < 0.0 or parsed > 1.0:
                raise ValueError(f"Capability cache {key}.{field_name} must be between 0.0 and 1.0")
            parsed_values[field_name] = parsed
        capabilities[key] = ModelCapability(
            base_skill=parsed_values["base_skill"],
            base_speed=parsed_values["base_speed"],
            base_cost=parsed_values["base_cost"],
        )
    return capabilities


def model_capability_for_model(
    model: str,
    preferences: RoutingPreferences,
    capabilities: Mapping[str, ModelCapability],
) -> ModelCapability:
    model_key = preferences.model_aliases.get(_normalize_model_key(model))
    if model_key is None:
        model_key = _normalize_model_key(model)
    if model_key not in capabilities:
        raise ValueError(
            f"No capability cache entry for model {model!r}; "
            "add it to config/model_capabilities_cache.json or routing_preferences.yaml aliases"
        )
    return capabilities[model_key]


def parse_candidates(
    raw: Mapping[str, Any],
    preferences: RoutingPreferences,
    capabilities: Mapping[str, ModelCapability],
) -> list[ModelCandidate]:
    items = raw.get("candidates")
    if not isinstance(items, list):
        return []
    out: list[ModelCandidate] = []
    for entry in items:
        if not isinstance(entry, Mapping):
            continue
        cid = str(entry.get("id") or "").strip()
        provider = str(entry.get("provider") or "").strip().lower()
        deployment = str(entry.get("deployment") or "local").strip().lower()
        model = _expand_placeholders(str(entry.get("model") or "")).strip()
        if _is_unresolved_placeholder(model):
            model = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b").strip()
        if not cid or not provider or not model:
            continue
        aliases = entry.get("benchmark_aliases") or []
        alias_tuple = tuple(str(a).strip() for a in aliases if str(a).strip())
        capability_baseline = model_capability_for_model(model, preferences, capabilities)
        out.append(
            ModelCandidate(
                id=cid,
                provider=provider,
                deployment=deployment,
                model=model,
                capability=_as_float(entry.get("capability"), 0.5),
                privacy_impact=_as_float(entry.get("privacy_impact"), 0.5),
                cost=_as_float(entry.get("cost"), 0.5),
                latency_ms=_as_float(entry.get("latency_ms"), 500.0),
                generation_parameters=generation_parameters_for_model(model, preferences),
                base_skill=capability_baseline.base_skill,
                base_speed=capability_baseline.base_speed,
                base_cost=capability_baseline.base_cost,
                benchmark_aliases=alias_tuple,
            )
        )
    return out


def weights_from_preferences(preferences: RoutingPreferences) -> RoutingWeights:
    sliders = preferences.global_weights
    return RoutingWeights(
        w1_capability=sliders.skill_importance,
        w2_privacy=sliders.privacy_priority,
        w3_cost=sliders.cost_saving,
        w4_latency=sliders.speed_priority,
        pii_privacy_weight_multiplier=1.0 + (2.5 * sliders.privacy_priority),
    )


def parse_privacy_from_presidio(routing: Mapping[str, Any]) -> PrivacyFromPresidioConfig:
    raw = routing.get("privacy_from_presidio") or {}
    if not isinstance(raw, Mapping):
        raw = {}
    ew = raw.get("entity_weights") or {}
    entity_weights: dict[str, float] = {}
    if isinstance(ew, Mapping):
        for k, v in ew.items():
            entity_weights[str(k)] = _as_float(v, 1.0)
    return PrivacyFromPresidioConfig(
        local_severity_boost=_as_float(raw.get("local_severity_boost"), 0.35),
        cloud_severity_discount=_as_float(raw.get("cloud_severity_discount"), 0.6),
        cloud_anonymization_trust=_as_float(raw.get("cloud_anonymization_trust"), 0.78),
        severity_cap=max(1.0, _as_float(raw.get("severity_cap"), 12.0)),
        entity_weights=entity_weights,
    )


def parse_supervisor(routing: Mapping[str, Any]) -> SupervisorConfig:
    raw = routing.get("supervisor") or {}
    if not isinstance(raw, Mapping):
        raw = {}
    blend = _as_float(raw.get("blend_weight"), 0.65)
    blend = max(0.0, min(1.0, blend))
    return SupervisorConfig(
        enabled=bool(raw.get("enabled", True)),
        provider=str(raw.get("provider") or "ollama").strip().lower() or "ollama",
        model=str(raw.get("model") or "qwen2.5:7b").strip() or "qwen2.5:7b",
        timeout_seconds=max(10.0, _as_float(raw.get("timeout_seconds"), 120.0)),
        blend_weight=blend,
    )


def parse_autonomous_intel(root: Path, routing: Mapping[str, Any]) -> AutonomousIntelConfig:
    ai = routing.get("autonomous_intelligence") or {}
    sources_raw = ai.get("sources") or []
    sources: list[dict[str, str]] = []
    if isinstance(sources_raw, list):
        for s in sources_raw:
            if isinstance(s, Mapping):
                url = str(s.get("url") or "").strip()
                fmt = str(s.get("format") or "wulong_arena_v1").strip()
                if url:
                    sources.append({"url": url, "format": fmt})
    rel = str(ai.get("cache_path") or "data/capability_benchmarks.json").strip()
    cache_path = (root / rel).resolve() if not os.path.isabs(rel) else Path(rel)
    return AutonomousIntelConfig(
        enabled=bool(ai.get("enabled", False)),
        refresh_interval_seconds=max(60.0, _as_float(ai.get("refresh_interval_seconds"), 86_400.0)),
        cache_path=cache_path,
        request_timeout_seconds=max(5.0, _as_float(ai.get("request_timeout_seconds"), 45.0)),
        sources=tuple(sources),
    )


def _set_presidio_online(value: bool, reason: str = "") -> None:
    """Update the runtime privacy boundary state and log transitions loudly."""
    global _PRESIDIO_OFFLINE_REASON, presidio_online

    previous = presidio_online
    presidio_online = value
    if value:
        _PRESIDIO_OFFLINE_REASON = ""
        return

    _PRESIDIO_OFFLINE_REASON = reason or "Presidio unavailable"
    if previous:
        logger.critical(
            "DEGRADED SYSTEM STATE / CRITICAL RISK: %s. Routing locked to Tier 1 local only.",
            _PRESIDIO_OFFLINE_REASON,
        )


# --- Presidio + lightweight fallback ---------------------------------------


@dataclass
class _PresidioEngines:
    analyzer: Any
    anonymizer: Any
    language: str


def _build_presidio_engines(language: str, timeout_s: float = 120.0) -> _PresidioEngines | None:
    """Initialize Presidio off the main startup path and fail closed on timeout."""
    result: dict[str, _PresidioEngines | None] = {"engines": None}
    error: dict[str, BaseException | None] = {"exc": None}

    def target() -> None:
        try:
            from presidio_analyzer import AnalyzerEngine
            from presidio_anonymizer import AnonymizerEngine

            result["engines"] = _PresidioEngines(
                analyzer=AnalyzerEngine(),
                anonymizer=AnonymizerEngine(),
                language=language,
            )
        except BaseException as exc:  # noqa: BLE001 - dependency import can raise SystemExit
            error["exc"] = exc

    thread = threading.Thread(
        target=target,
        name="henry-presidio-init",
        daemon=True,
    )
    thread.start()
    thread.join(timeout_s)
    if thread.is_alive():
        logger.warning(
            "Presidio init exceeded %.1fs; continuing in strict local mode.",
            timeout_s,
        )
        print(
            f"[Henry Router] Presidio init exceeded {timeout_s:.1f}s; strict local mode active.",
            flush=True,
        )
        return None
    if error["exc"] is not None:
        logger.warning(
            "Presidio init failed (%s); strict local mode will be used.",
            error["exc"],
        )
        return None
    return result["engines"]


def _presidio_init_timeout() -> float:
    raw = os.environ.get("HENRY_PRESIDIO_INIT_TIMEOUT_S", "3.0").strip()
    try:
        return max(0.1, float(raw))
    except ValueError:
        return 3.0


class _PresidioFacade:
    """Microsoft Presidio analyzer/anonymizer with hard local-only lockdown on failure."""

    def __init__(self, language: str, enabled: bool) -> None:
        self._language = language
        self._requested = enabled
        self._engines: _PresidioEngines | None = None
        self._init_error: BaseException | None = None
        self._init_thread: threading.Thread | None = None
        if enabled:
            self._start_background_init()
        else:
            logger.warning("Presidio disabled by routing config; strict local mode will be used.")

        self._email = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b", re.I)
        self._phone = re.compile(r"\b\+?\d[\d\s().-]{7,}\d\b")
        self._date = re.compile(
            r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{4}-\d{2}-\d{2}\b",
            re.I,
        )

    def _start_background_init(self) -> None:
        def target() -> None:
            try:
                from presidio_analyzer import AnalyzerEngine
                from presidio_anonymizer import AnonymizerEngine

                self._engines = _PresidioEngines(
                    analyzer=AnalyzerEngine(),
                    anonymizer=AnonymizerEngine(),
                    language=self._language,
                )
            except BaseException as exc:  # noqa: BLE001 - dependency import can raise SystemExit
                self._init_error = exc

        self._init_thread = threading.Thread(
            target=target,
            name="henry-presidio-init",
            daemon=True,
        )
        self._init_thread.start()

    @property
    def presidio_active(self) -> bool:
        return self._engines is not None

    def health_probe(self) -> HealthProbe:
        started = time.monotonic()
        if not self._requested:
            return HealthProbe("presidio_analyzer", "disabled", "disabled by routing config", 0.0)
        if self._engines is None:
            if self._init_thread is not None and self._init_thread.is_alive():
                return HealthProbe(
                    "presidio_analyzer",
                    "initializing",
                    "background engine load still running",
                    round((time.monotonic() - started) * 1000, 1),
                )
            if self._init_error is not None:
                return HealthProbe("presidio_analyzer", "fail", str(self._init_error), 0.0)
            return HealthProbe("presidio_analyzer", "fail", "engine unavailable", 0.0)

        def analyze_probe() -> HealthProbe:
            probe_started = time.monotonic()
            try:
                results = self._engines.analyzer.analyze(
                    text="Henry boot privacy probe for John Doe at john.doe@example.com",
                    language=self._engines.language,
                )
            except Exception as exc:  # noqa: BLE001
                self._engines = None
                return HealthProbe("presidio_analyzer", "fail", str(exc), round((time.monotonic() - probe_started) * 1000, 1))
            if not results:
                return HealthProbe(
                    "presidio_analyzer",
                    "warn",
                    "analyzer returned no entities for known PII probe",
                    round((time.monotonic() - probe_started) * 1000, 1),
                )
            return HealthProbe(
                "presidio_analyzer",
                "ok",
                f"{len(results)} entity span(s) detected",
                round((time.monotonic() - probe_started) * 1000, 1),
            )

        return run_with_timeout(
            "presidio_analyzer",
            analyze_probe,
            timeout_s=READINESS_PROBE_TIMEOUT_S,
            timeout_status="initializing",
        )

    def ping(self) -> bool:
        """Run a live analyzer probe; regex fallback is not considered online."""
        return self.health_probe().status == "ok"

    def analyze_results(self, text: str) -> tuple[list[Any], str]:
        """Return (recognizer_results, method_label)."""
        if not text.strip():
            return [], "empty"
        if self._engines is not None:
            try:
                results = self._engines.analyzer.analyze(text=text, language=self._engines.language)
                return list(results), "presidio"
            except Exception as exc:  # noqa: BLE001
                logger.warning("Presidio analyze failed: %s", exc)
                self._engines = None
                _set_presidio_online(False, f"Presidio analyze failed at runtime: {exc}")
        hits: list[Any] = []
        if self._email.search(text):
            hits.append("EMAIL")
        if self._phone.search(text):
            hits.append("PHONE")
        if self._date.search(text):
            hits.append("DATE")
        return hits, "regex_fallback"

    def analyze(self, text: str) -> PIIScanResult:
        results, method = self.analyze_results(text)
        if method == "regex_fallback":
            n = len(results)
            return PIIScanResult(n > 0, n, method)
        return PIIScanResult(len(results) > 0, len(results), method)

    def build_routing_context(self, text: str) -> PresidioRoutingContext:
        results, method = self.analyze_results(text)
        counts: Counter[str] = Counter()
        if method == "presidio":
            for r in results:
                et = getattr(r, "entity_type", None) or type(r).__name__
                counts[str(et)] += 1
        else:
            for token in results:
                counts[str(token)] += 1
        severity = _severity_from_counts(counts, method)
        hit_count = int(sum(counts.values())) if counts else len(results)
        return PresidioRoutingContext(
            has_pii=hit_count > 0,
            hit_count=hit_count,
            counts_by_type=dict(counts),
            severity=severity,
            method=method,
        )

    def anonymize(self, text: str) -> str:
        if not text.strip():
            return text
        if self._engines is not None:
            try:
                results = self._engines.analyzer.analyze(text=text, language=self._engines.language)
                if not results:
                    return self._regex_anonymize(text)
                out = self._engines.anonymizer.anonymize(text=text, analyzer_results=results)
                return out.text
            except Exception as exc:  # noqa: BLE001
                logger.warning("Presidio anonymize failed: %s", exc)
                self._engines = None
                _set_presidio_online(False, f"Presidio anonymize failed at runtime: {exc}")
        return self._regex_anonymize(text)

    def _regex_anonymize(self, text: str) -> str:
        text = self._email.sub("<EMAIL>", text)
        text = self._phone.sub("<PHONE>", text)
        text = self._date.sub("<DATE>", text)
        return text


def _severity_from_counts(counts: Mapping[str, int], method: str) -> float:
    """Map Presidio hit mass to [0,1] for weighting (dynamic, not a hard ban)."""
    total = float(sum(counts.values())) if counts else 0.0
    if total <= 0.0:
        return 0.0
    if method != "presidio":
        return max(0.0, min(1.0, total / 4.0))
    return max(0.0, min(1.0, total / 12.0))


def presidio_summary_dict(ctx: PresidioRoutingContext) -> dict[str, Any]:
    return {
        "method": ctx.method,
        "hit_count": ctx.hit_count,
        "severity": round(ctx.severity, 4),
        "counts_by_type": dict(ctx.counts_by_type),
    }


def weighted_presidio_mass(
    counts: Mapping[str, int],
    entity_weights: Mapping[str, float],
    severity_cap: float,
) -> float:
    weighted = 0.0
    for entity, n in counts.items():
        w = float(entity_weights.get(entity, 1.0))
        weighted += float(n) * w
    if severity_cap <= 0:
        return 0.0
    return max(0.0, min(1.0, weighted / severity_cap))


def privacy_impact_from_presidio(
    candidate: ModelCandidate,
    ctx: PresidioRoutingContext,
    cfg: PrivacyFromPresidioConfig,
    *,
    anonymize_cloud_enabled: bool,
) -> float:
    """
    Privacy_Impact in [0,1] derived from Presidio: higher means better privacy posture for routing.

    Local routes gain as PII mass increases (data stays on-device). Cloud routes are discounted
    with severity unless anonymization is enabled, in which case a trust floor applies.
    """
    base = max(0.0, min(1.0, float(candidate.privacy_impact)))
    severity = max(0.0, min(1.0, float(ctx.severity)))
    mass = weighted_presidio_mass(ctx.counts_by_type, cfg.entity_weights, cfg.severity_cap)
    severity = max(severity, mass)

    if candidate.deployment == "local":
        return max(0.0, min(1.0, base + cfg.local_severity_boost * severity))

    if anonymize_cloud_enabled:
        trust = max(0.0, min(1.0, cfg.cloud_anonymization_trust))
        discounted = base * (1.0 - cfg.cloud_severity_discount * severity * (1.0 - trust))
        floor = base * trust * (1.0 - 0.35 * severity)
        return max(0.0, min(1.0, max(discounted, floor)))

    return max(0.0, min(1.0, base * (1.0 - cfg.cloud_severity_discount * severity)))


def anonymize_for_cloud(text: str, facade: _PresidioFacade) -> str:
    """Replace names/dates/sensitive spans before any cloud provider call."""
    return facade.anonymize(text)


# --- Scoring ---------------------------------------------------------------

LatencyFactor = float  # documented: latency_ms / 1000


def total_score(
    capability: float,
    privacy_impact: float,
    cost: float,
    latency_ms: float,
    weights: RoutingWeights,
    *,
    pii_severity: float,
) -> float:
    """Score = Capability*W1 + Privacy_Impact*W2 - Cost*W3 - Latency*W4 (W2 scales with PII severity)."""
    sev = max(0.0, min(1.0, float(pii_severity)))
    pii_factor = 1.0 + max(0.0, (weights.pii_privacy_weight_multiplier - 1.0)) * sev
    w2 = weights.w2_privacy * pii_factor
    latency_term = (latency_ms / 1000.0) * weights.w4_latency
    return (
        capability * weights.w1_capability
        + privacy_impact * w2
        - cost * weights.w3_cost
        - latency_term
    )


def infer_privacy_tier(ctx: PresidioRoutingContext) -> PrivacyTier:
    if not presidio_online:
        return "tier1_strict_local"
    if ctx.has_pii:
        return "tier2_anonymized"
    return "tier3_public"


def _available_candidates(
    candidates: Sequence[ModelCandidate],
    is_available: Callable[[ModelCandidate], bool],
) -> list[ModelCandidate]:
    return [candidate for candidate in candidates if is_available(candidate)]


def _task_skill_multiplier(user_blob: str) -> float:
    """Small local assessment of task difficulty; no network or cloud dependency."""
    text = user_blob.lower()
    hard_markers = (
        "architecture",
        "architectural",
        "refactor",
        "debug",
        "security",
        "legal",
        "financial",
        "medical",
        "multi-step",
        "production",
        "design",
        "analyze",
    )
    if any(marker in text for marker in hard_markers):
        return 1.08
    if len(text) > 1800:
        return 1.05
    return 1.0


def _lookup_efficiency_ratio(user_blob: str) -> float:
    """Estimate whether cloud can answer an isolated abstract lookup without private context."""
    text = user_blob.lower()
    lookup_markers = (
        "what is",
        "who is",
        "when did",
        "define",
        "explain",
        "lookup",
        "search",
        "current",
        "latest",
        "compare",
        "general knowledge",
        "documentation",
    )
    private_context_markers = (
        "my document",
        "this document",
        "attached",
        "contract",
        "invoice",
        "letter",
        "case",
        "account",
        "personal",
    )
    if any(marker in text for marker in lookup_markers) and not any(
        marker in text for marker in private_context_markers
    ):
        return 0.95
    if any(marker in text for marker in lookup_markers):
        return 0.9
    return 0.45


def _efficiency_ratio(
    *,
    local_candidates: Sequence[ModelCandidate],
    cloud_candidates: Sequence[ModelCandidate],
    user_blob: str,
) -> float:
    if not local_candidates:
        return 0.0
    top_local = max(candidate.base_skill for candidate in local_candidates)
    frontier = max((candidate.base_skill for candidate in cloud_candidates), default=top_local)
    frontier = max(frontier * _task_skill_multiplier(user_blob), 0.01)
    return max(0.0, min(1.0, top_local / frontier))


def decide_privacy_escalation(
    *,
    privacy_tier: PrivacyTier,
    candidates: Sequence[ModelCandidate],
    is_available: Callable[[ModelCandidate], bool],
    user_blob: str,
    preferences: RoutingPreferences,
) -> RoutingEscalationDecision:
    available = _available_candidates(candidates, is_available)
    local = [candidate for candidate in available if candidate.deployment == "local"]
    cloud = [candidate for candidate in available if candidate.deployment == "cloud"]
    local_ids = frozenset(candidate.id for candidate in local)
    all_ids = frozenset(candidate.id for candidate in available)
    local_efficiency = _efficiency_ratio(
        local_candidates=local,
        cloud_candidates=cloud,
        user_blob=user_blob,
    )

    if privacy_tier == "tier1_strict_local":
        return RoutingEscalationDecision(
            privacy_tier=privacy_tier,
            action="strict_local_runtime_safety",
            allowed_candidate_ids=local_ids,
            cloud_score_multiplier=0.0,
            local_efficiency_ratio=local_efficiency,
            lookup_efficiency_ratio=0.0,
            reason="Presidio is offline; strict local-only routing is active.",
        )

    if privacy_tier == "tier2_anonymized":
        if local_efficiency >= preferences.tier2_local_efficiency_threshold:
            return RoutingEscalationDecision(
                privacy_tier=privacy_tier,
                action="tier2_local_only",
                allowed_candidate_ids=local_ids,
                cloud_score_multiplier=0.0,
                local_efficiency_ratio=local_efficiency,
                lookup_efficiency_ratio=0.0,
                reason="Tier 2 Step 1 passed: local pool meets frontier efficiency threshold.",
            )
        lookup_efficiency = _lookup_efficiency_ratio(user_blob)
        if lookup_efficiency >= preferences.tier2_lookup_efficiency_threshold:
            return RoutingEscalationDecision(
                privacy_tier=privacy_tier,
                action="tier2_micro_query",
                allowed_candidate_ids=local_ids,
                cloud_score_multiplier=0.0,
                local_efficiency_ratio=local_efficiency,
                lookup_efficiency_ratio=lookup_efficiency,
                reason="Tier 2 Step 2 passed: private context stays local; cloud is limited to an abstract micro-query.",
            )
        return RoutingEscalationDecision(
            privacy_tier=privacy_tier,
            action="tier2_full_context_cloud",
            allowed_candidate_ids=all_ids,
            cloud_score_multiplier=preferences.cloud_allowance,
            local_efficiency_ratio=local_efficiency,
            lookup_efficiency_ratio=lookup_efficiency,
            reason="Tier 2 Step 3 active: cloud pool allowed with sovereignty penalty.",
        )

    if local_efficiency >= preferences.tier3_local_efficiency_threshold:
        return RoutingEscalationDecision(
            privacy_tier=privacy_tier,
            action="tier3_local_only",
            allowed_candidate_ids=local_ids,
            cloud_score_multiplier=0.0,
            local_efficiency_ratio=local_efficiency,
            lookup_efficiency_ratio=0.0,
            reason="Tier 3 90% gate passed: cloud stripped to save cost.",
        )
    return RoutingEscalationDecision(
        privacy_tier=privacy_tier,
        action="tier3_open_competition",
        allowed_candidate_ids=all_ids,
        cloud_score_multiplier=1.0,
        local_efficiency_ratio=local_efficiency,
        lookup_efficiency_ratio=0.0,
        reason="Tier 3 90% gate failed: local and cloud compete freely.",
    )


def _extract_json_object(text: str) -> dict[str, Any] | None:
    raw = text.strip()
    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, Mapping) else None
    except json.JSONDecodeError:
        pass
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        try:
            obj = json.loads(raw[start : end + 1])
            return obj if isinstance(obj, Mapping) else None
        except json.JSONDecodeError:
            return None
    return None


def call_supervisor_scores_ollama(
    *,
    base_url: str,
    model: str,
    generation_parameters: GenerationParameters,
    timeout_s: float,
    user_snippet: str,
    presidio_summary: dict[str, Any],
    candidate_rows: list[dict[str, Any]],
) -> dict[str, float] | None:
    """Ask the local Qwen supervisor (Ollama) for per-candidate routing scores."""
    params = generation_parameters
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are Henry's routing supervisor. Output one JSON object only, no markdown, "
                    'no code fences. Schema: {"scores":{"<candidate_id>": <number>, ...}}. '
                    "Scores are 0-10 (float). Higher means a better choice for this user text given "
                    "Presidio PII findings, deployment (local vs cloud), cost, latency, and capability. "
                    "Prefer local routes when PII severity is high unless cloud capability is clearly required."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "presidio": presidio_summary,
                        "candidates": candidate_rows,
                        "user_text_excerpt": user_snippet,
                    },
                    ensure_ascii=False,
                ),
            },
        ],
        "options": {
            "temperature": params.temperature,
            "top_p": params.top_p,
            "num_ctx": params.num_ctx,
        },
        "stream": False,
    }
    url = f"{base_url.rstrip('/')}/api/chat"
    try:
        with httpx.Client(timeout=timeout_s) as client:
            response = client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
        message = data.get("message") or {}
        content = message.get("content")
        if not isinstance(content, str):
            return None
        obj = _extract_json_object(content)
        if not isinstance(obj, Mapping):
            return None
        scores_raw = obj.get("scores")
        if not isinstance(scores_raw, Mapping):
            return None
        out: dict[str, float] = {}
        for key, val in scores_raw.items():
            if isinstance(val, (int, float)):
                out[str(key)] = float(val)
        return out or None
    except Exception as exc:  # noqa: BLE001
        logger.warning("Supervisor Qwen call failed: %s", exc)
        return None


def blend_formula_and_supervisor(
    formula_scores: dict[str, float],
    supervisor_scores: dict[str, float] | None,
    blend_weight: float,
) -> dict[str, float]:
    """Blend deterministic routing score with supervisor scores (see config supervisor.blend_weight)."""
    if not supervisor_scores or blend_weight <= 0:
        return dict(formula_scores)
    common = [k for k in formula_scores if k in supervisor_scores]
    if not common:
        return dict(formula_scores)
    f_lo = min(formula_scores[k] for k in common)
    f_hi = max(formula_scores[k] for k in common)
    s_lo = min(supervisor_scores[k] for k in common)
    s_hi = max(supervisor_scores[k] for k in common)
    out: dict[str, float] = {}
    for cid, fv in formula_scores.items():
        if cid not in supervisor_scores:
            out[cid] = fv
            continue
        sv = supervisor_scores[cid]
        if s_hi > s_lo:
            s_norm = (sv - s_lo) / (s_hi - s_lo)
        else:
            s_norm = 0.5
        if f_hi > f_lo:
            sv_scaled = f_lo + s_norm * (f_hi - f_lo)
        else:
            sv_scaled = fv
        out[cid] = (1.0 - blend_weight) * fv + blend_weight * sv_scaled
    return out


# --- Benchmark refresh (HTTP JSON) -----------------------------------------


def _parse_wulong_arena_v1(payload: Mapping[str, Any]) -> dict[str, float]:
    models = payload.get("models")
    if not isinstance(models, list):
        return {}
    raw_scores: dict[str, float] = {}
    for row in models:
        if not isinstance(row, Mapping):
            continue
        name = str(row.get("model") or "").strip()
        score = row.get("score")
        if not name or not isinstance(score, (int, float)):
            continue
        raw_scores[name] = float(score)
    if not raw_scores:
        return {}
    vals = list(raw_scores.values())
    lo, hi = min(vals), max(vals)
    if hi <= lo:
        return {k: 0.5 for k in raw_scores}
    return {k: (v - lo) / (hi - lo) for k, v in raw_scores.items()}


def fetch_benchmark_capability_table(
    sources: Sequence[Mapping[str, str]],
    timeout_s: float,
) -> dict[str, float]:
    """Fetch and merge capability hints keyed by public benchmark model name."""
    merged: dict[str, float] = {}
    for src in sources:
        url = str(src.get("url") or "").strip()
        fmt = str(src.get("format") or "wulong_arena_v1").strip()
        if not url:
            continue
        try:
            with httpx.Client(timeout=timeout_s) as client:
                response = client.get(url)
                response.raise_for_status()
                data = response.json()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Benchmark fetch failed for %s: %s", url, exc)
            continue
        if not isinstance(data, Mapping):
            continue
        if fmt == "wulong_arena_v1":
            table = _parse_wulong_arena_v1(data)
        else:
            # Generic flat mapping: {"gpt-4o": 0.93, ...}
            table = {str(k): float(v) for k, v in data.items() if isinstance(v, (int, float))}
        merged.update(table)
    return merged


def match_benchmark_capability(
    candidate: ModelCandidate,
    benchmark: Mapping[str, float],
) -> float | None:
    """Return normalized capability if a benchmark row matches this candidate."""
    if not benchmark:
        return None
    cmodel = candidate.model.lower()
    aliases = [a.lower() for a in candidate.benchmark_aliases] + [cmodel]
    for bench_name, cap in benchmark.items():
        bn = bench_name.lower()
        for a in aliases:
            if a == bn or a in bn or bn in a:
                return float(cap)
    return None


def refresh_capability_scores_from_benchmarks(
    root: Path,
    candidates: Sequence[ModelCandidate],
    intel: AutonomousIntelConfig,
) -> dict[str, float]:
    """
    Pull latest public benchmark JSON and persist per-candidate capability hints.

    Returns mapping candidate_id -> capability used for subsequent scoring until refresh.
    """
    table = fetch_benchmark_capability_table(intel.sources, intel.request_timeout_seconds)
    per_id: dict[str, float] = {}
    for c in candidates:
        matched = match_benchmark_capability(c, table)
        if matched is not None:
            per_id[c.id] = matched
    intel.cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "updated_at": time.time(),
        "benchmark_keys": sorted(table.keys()),
        "capabilities_by_candidate_id": per_id,
    }
    intel.cache_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return per_id


def load_cached_capabilities(intel: AutonomousIntelConfig) -> dict[str, float]:
    if not intel.cache_path.is_file():
        return {}
    try:
        data = json.loads(intel.cache_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(data, Mapping):
        return {}
    raw = data.get("capabilities_by_candidate_id")
    if not isinstance(raw, Mapping):
        return {}
    out: dict[str, float] = {}
    for k, v in raw.items():
        if isinstance(v, (int, float)):
            out[str(k)] = float(v)
    return out


# --- Provider wiring -------------------------------------------------------


def _ollama_base_url(providers: Mapping[str, Any]) -> str:
    llm = providers.get("llm") or {}
    oc = llm.get("ollama") or {}
    base = _expand_placeholders(str(oc.get("base_url") or "")).strip()
    if not base or _is_unresolved_placeholder(base):
        base = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    return base.rstrip("/")


def _discover_ollama_models(base_url: str, timeout_s: float) -> tuple[bool, tuple[str, ...], str]:
    """Return live Ollama model names from /api/tags."""
    try:
        with httpx.Client(timeout=timeout_s) as client:
            response = client.get(f"{base_url.rstrip('/')}/api/tags")
            response.raise_for_status()
            data = response.json()
    except Exception as exc:  # noqa: BLE001
        return False, (), str(exc)

    models_raw = data.get("models") if isinstance(data, Mapping) else None
    if not isinstance(models_raw, list):
        return True, (), "unexpected /api/tags response shape"

    names: set[str] = set()
    for row in models_raw:
        if not isinstance(row, Mapping):
            continue
        for key in ("model", "name"):
            value = str(row.get(key) or "").strip()
            if value:
                names.add(value)
    return True, tuple(sorted(names)), ""


def _cloud_key_status() -> dict[str, bool]:
    return {
        "openai": bool(os.environ.get("OPENAI_API_KEY", "").strip()),
        "gemini": bool(
            os.environ.get("GEMINI_API_KEY", "").strip()
            or os.environ.get("GOOGLE_API_KEY", "").strip()
        ),
        "anthropic": bool(os.environ.get("ANTHROPIC_API_KEY", "").strip()),
    }


def _memory_mount_path() -> Path:
    raw = os.environ.get("MEMORY_MOUNT_PATH", "/Volumes/HenryData").strip()
    return Path(raw).expanduser()


def _memory_collection_status(memory_manager: Any | None) -> dict[str, bool]:
    if memory_manager is None:
        return {
            "archive_chroma": False,
            "chat_history_chroma": False,
            "working_memory_chroma": False,
            "personal_sqlite": False,
        }

    def chroma_ready(ready_attr: str, store_attr: str) -> bool:
        store = getattr(memory_manager, store_attr, None)
        collection = getattr(store, "_collection", None)
        return bool(getattr(memory_manager, ready_attr, False) and collection is not None)

    return {
        "archive_chroma": chroma_ready("archive_ready", "_archive_vectorstore"),
        "chat_history_chroma": chroma_ready("chat_history_ready", "_chat_history_vectorstore"),
        "working_memory_chroma": chroma_ready("working_memory_ready", "_working_memory_vectorstore"),
        "personal_sqlite": bool(getattr(memory_manager, "personal_memory_ready", False)),
    }


def _candidate_available_from_ledger(
    candidate: ModelCandidate,
    *,
    local_models: set[str],
    ollama_online: bool,
    cloud_keys: Mapping[str, bool],
) -> bool:
    if candidate.provider == "ollama":
        return bool(ollama_online and candidate.model in local_models)
    if candidate.provider in {"openai", "anthropic", "gemini"}:
        return bool(cloud_keys.get(candidate.provider, False))
    return False


def _status_icon(ok: bool) -> str:
    return f"{_ANSI_GREEN}OK{_ANSI_RESET}" if ok else f"{_ANSI_RED}FAIL{_ANSI_RESET}"


def _probe_status_text(probe: HealthProbe) -> str:
    color = {
        "ok": _ANSI_GREEN,
        "warn": _ANSI_YELLOW,
        "fail": _ANSI_RED,
        "missing": _ANSI_RED,
        "initializing": _ANSI_YELLOW,
        "disabled": _ANSI_DIM,
    }.get(probe.status, _ANSI_DIM)
    return f"{color}{probe.status.upper()}{_ANSI_RESET}"


def _probe_latency_text(probe: HealthProbe) -> str:
    return "" if probe.latency_ms is None else f" ({probe.latency_ms:.0f}ms)"


def _print_environment_dashboard(
    ledger: EnvironmentLedger,
    candidates: Sequence[ModelCandidate],
) -> None:
    """Emit an operator-visible startup dashboard to terminal logs."""
    line = f"{_ANSI_CYAN}{'=' * 78}{_ANSI_RESET}"
    print(line, flush=True)
    print(f"{_ANSI_BOLD}{_ANSI_CYAN}Henry Environment Ledger - Boot Discovery{_ANSI_RESET}", flush=True)
    print(line, flush=True)
    print("Runtime readiness probes:", flush=True)
    for probe in ledger.runtime_probes.values():
        print(
            f"  {probe.name}: {_probe_status_text(probe)}{_probe_latency_text(probe)} - {probe.detail}",
            flush=True,
        )

    print(f"Ollama endpoint: {ledger.ollama_base_url}", flush=True)
    if ledger.local_models:
        for model in ledger.local_models:
            print(f"  {_ANSI_GREEN}local model{_ANSI_RESET}: {model}", flush=True)
    else:
        print(f"  {_ANSI_RED}no live local models discovered via /api/tags{_ANSI_RESET}", flush=True)

    configured_local = [c for c in candidates if c.deployment == "local"]
    if configured_local:
        print("Configured local routes:", flush=True)
        live = set(ledger.local_models)
        for cand in configured_local:
            ok = cand.model in live and ledger.ollama_online
            print(f"  {_status_icon(ok)} {cand.id}: {cand.model}", flush=True)

    print("Cloud keys:", flush=True)
    for name, present in ledger.cloud_keys.items():
        label = "present" if present else "missing"
        color = _ANSI_GREEN if present else _ANSI_DIM
        print(f"  {color}{name}{_ANSI_RESET}: {label}", flush=True)

    print("Memory and privacy boundary:", flush=True)
    print(
        f"  encrypted volume: {_status_icon(ledger.encrypted_storage_online)} "
        f"{ledger.memory_mount_path}",
        flush=True,
    )
    print(f"  security flight mode: {_status_icon(not ledger.security_flight_mode)}", flush=True)
    for name, ok in ledger.memory_collections.items():
        print(f"  {_status_icon(ok)} {name}", flush=True)
    print(f"  Presidio analyzer: {_probe_status_text(ledger.runtime_probes['presidio_analyzer'])}", flush=True)
    print(f"Routable candidates: {', '.join(ledger.usable_candidate_ids) or 'none'}", flush=True)

    if ledger.degraded_reasons:
        print(f"{_ANSI_RED}{_ANSI_BOLD}{'!' * 78}{_ANSI_RESET}", flush=True)
        print(
            f"{_ANSI_RED}{_ANSI_BOLD}DEGRADED SYSTEM STATE / CRITICAL RISK{_ANSI_RESET}",
            flush=True,
        )
        for reason in ledger.degraded_reasons:
            print(f"{_ANSI_RED}  - {reason}{_ANSI_RESET}", flush=True)
        print(
            f"{_ANSI_YELLOW}Cloud routing is blocked whenever Presidio is offline; "
            f"Tier 1 strict local mode is enforced.{_ANSI_RESET}",
            flush=True,
        )
        print(f"{_ANSI_RED}{_ANSI_BOLD}{'!' * 78}{_ANSI_RESET}", flush=True)
    print(line, flush=True)


def initialize_environment_ledger(
    *,
    candidates: Sequence[ModelCandidate],
    providers: Mapping[str, Any],
    memory_manager: Any | None,
    presidio_facade: _PresidioFacade,
    timeout_s: float = READINESS_PROBE_TIMEOUT_S,
) -> EnvironmentLedger:
    """Actively discover boot environment state before the router accepts traffic."""
    global SECURITY_FLIGHT_MODE

    ollama_base_url = _ollama_base_url(providers)
    configured_local_models = tuple(
        candidate.model for candidate in candidates if candidate.provider == "ollama" and candidate.model
    )
    ollama_probe, local_models = probe_ollama(
        ollama_base_url,
        required_models=configured_local_models,
        timeout_s=timeout_s,
    )
    cloud_keys = _cloud_key_status()

    mount_path = _memory_mount_path()
    encrypted_storage_online = mount_path.exists() and mount_path.is_dir()
    if memory_manager is not None:
        encrypted_storage_online = bool(
            encrypted_storage_online
            and getattr(memory_manager, "is_encrypted_storage_available", False)
        )
    SECURITY_FLIGHT_MODE = not encrypted_storage_online
    os.environ["SECURITY_FLIGHT_MODE"] = "1" if SECURITY_FLIGHT_MODE else "0"

    embed_model = os.environ.get("HENRY_EMBED_MODEL", "nomic-embed-text").strip() or "nomic-embed-text"
    archive_dir = chroma_persist_path()
    personal_db_path = Path(os.environ.get("PERSONAL_MEMORY_PATH", "")).expanduser()
    embeddings_probe = probe_embeddings(ollama_base_url, embed_model, timeout_s=READINESS_PROBE_TIMEOUT_S)
    chroma_probe = probe_chroma_vector_memory(
        archive_dir,
        base_url=ollama_base_url,
        embed_model=embed_model,
        timeout_s=READINESS_PROBE_TIMEOUT_S,
    )
    sqlite_probe = probe_sqlite_personal_memory(personal_db_path, timeout_s=READINESS_PROBE_TIMEOUT_S)
    presidio_probe = presidio_facade.health_probe()
    worker_probe = probe_satellite_status(
        "worker_satellite",
        os.environ.get("HENRY_WORKER_API_URL", "http://127.0.0.1:8001"),
        timeout_s=READINESS_PROBE_TIMEOUT_S,
    )
    ui_probe = probe_satellite_status(
        "ui_satellite",
        os.environ.get("HENRY_UI_API_URL", "http://127.0.0.1:8002"),
        timeout_s=READINESS_PROBE_TIMEOUT_S,
    )
    runtime_probes = {
        probe.name: probe
        for probe in (
            ollama_probe,
            embeddings_probe,
            chroma_probe,
            sqlite_probe,
            presidio_probe,
            worker_probe,
            ui_probe,
        )
    }

    memory_collections = {
        "archive_chroma": chroma_probe.status == "ok",
        "chat_history_chroma": chroma_probe.status == "ok",
        "working_memory_chroma": chroma_probe.status == "ok",
        "personal_sqlite": sqlite_probe.status == "ok",
    }
    presidio_ok = presidio_probe.status == "ok"
    _set_presidio_online(presidio_ok, "Presidio boot probe failed")

    local_model_set = set(local_models)
    usable_ids = tuple(
        candidate.id
        for candidate in candidates
        if (presidio_ok or candidate.deployment == "local")
        and _candidate_available_from_ledger(
            candidate,
            local_models=local_model_set,
            ollama_online=ollama_probe.status == "ok",
            cloud_keys=cloud_keys,
        )
    )

    degraded: list[str] = []
    if ollama_probe.status != "ok":
        degraded.append(f"Ollama readiness failed: {ollama_probe.detail}")
    missing_local = [
        f"{candidate.id} ({candidate.model})"
        for candidate in candidates
        if candidate.deployment == "local" and candidate.model not in local_model_set
    ]
    if missing_local:
        degraded.append("Configured local models missing from disk: " + ", ".join(missing_local))
    if len(usable_ids) <= 1:
        degraded.append("Routable choices collapsed to a single model or none")
    if not presidio_ok:
        degraded.append(f"Presidio analyzer {presidio_probe.status}; routing locked to Tier 1 strict local only")
    if SECURITY_FLIGHT_MODE:
        degraded.append("Encrypted Volume Shield offline; memory writes remain disabled")
    for probe in (embeddings_probe, chroma_probe, sqlite_probe):
        if probe.status not in {"ok", "disabled"}:
            degraded.append(f"{probe.name} readiness {probe.status}: {probe.detail}")

    ledger = EnvironmentLedger(
        runtime_probes=runtime_probes,
        ollama_base_url=ollama_base_url,
        ollama_online=ollama_probe.status == "ok",
        local_models=local_models,
        cloud_keys=cloud_keys,
        memory_mount_path=str(mount_path),
        encrypted_storage_online=encrypted_storage_online,
        memory_collections=memory_collections,
        presidio_online=presidio_ok,
        security_flight_mode=SECURITY_FLIGHT_MODE,
        usable_candidate_ids=usable_ids,
        degraded_reasons=tuple(degraded),
    )
    _print_environment_dashboard(ledger, candidates)
    return ledger


def build_provider_for_candidate(
    candidate: ModelCandidate,
    providers: Mapping[str, Any],
    timeout_s: float = 120.0,
) -> LLMProvider:
    if candidate.provider == "ollama":
        base_url = _ollama_base_url(providers)
        return OllamaLLMProvider(
            base_url=base_url,
            model=candidate.model,
            generation_parameters=candidate.generation_parameters,
            timeout_s=timeout_s,
        )
    if candidate.provider == "openai":
        return OpenAILLMProvider(
            model=candidate.model,
            generation_parameters=candidate.generation_parameters,
            timeout_s=timeout_s,
        )
    if candidate.provider == "anthropic":
        return AnthropicLLMProvider(
            model=candidate.model,
            generation_parameters=candidate.generation_parameters,
            timeout_s=timeout_s,
        )
    raise ValueError(f"Unsupported provider on candidate {candidate.id!r}: {candidate.provider!r}")


def provider_is_usable(candidate: ModelCandidate) -> bool:
    if candidate.provider == "ollama":
        return bool(candidate.model)
    if candidate.provider == "openai":
        return bool(os.environ.get("OPENAI_API_KEY", "").strip())
    if candidate.provider == "anthropic":
        return bool(os.environ.get("ANTHROPIC_API_KEY", "").strip())
    return False


# --- Routed facade ---------------------------------------------------------


class RoutedLLMProvider:
    """
    Presidio-backed PII context, Presidio-derived Privacy_Impact, weighted formula scores,
    optional Qwen2.5:7b (Ollama) supervisor blend, Presidio anonymization for cloud winners.
    """

    def __init__(
        self,
        root: Path,
        candidates: Sequence[ModelCandidate],
        weights: RoutingWeights,
        providers: Mapping[str, Any],
        presidio: _PresidioFacade,
        *,
        anonymize_cloud: bool,
        intel: AutonomousIntelConfig,
        privacy_cfg: PrivacyFromPresidioConfig,
        supervisor_cfg: SupervisorConfig,
        supervisor_base_url: str,
        supervisor_parameters: GenerationParameters,
        cloud_allowance: float,
        routing_preferences: RoutingPreferences,
        environment_ledger: EnvironmentLedger,
        memory_manager: Any | None = None,
    ) -> None:
        self._root = root
        self._candidates = list(candidates)
        self._weights = weights
        self._providers_map = providers
        self._presidio = presidio
        self._anonymize_cloud = anonymize_cloud
        self._intel = intel
        self._privacy_cfg = privacy_cfg
        self._supervisor_cfg = supervisor_cfg
        self._supervisor_base_url = supervisor_base_url.rstrip("/")
        self._supervisor_parameters = supervisor_parameters
        self._cloud_allowance = cloud_allowance
        self._routing_preferences = routing_preferences
        self._environment_ledger = environment_ledger
        self._live_local_models = set(environment_ledger.local_models)
        self._ollama_online = environment_ledger.ollama_online
        self._memory_manager = memory_manager
        self._provider_cache: dict[str, LLMProvider] = {}
        self._critic_provider_cache: dict[str, LLMProvider] = {}
        self._queue_manager = get_default_queue_manager()
        self._queue_manager.start_background_worker(self._process_background_task)
        self._cap_overrides: dict[str, float] = {}
        self._last_refresh_monotonic = 0.0
        self._lock = threading.Lock()

        if self._intel.enabled:
            self._cap_overrides = load_cached_capabilities(self._intel)

    def runtime_health_probe(self) -> list[HealthProbe]:
        """Return live readiness probes for `/api/status`; green requires active checks."""
        configured_local_models = tuple(
            candidate.model for candidate in self._candidates if candidate.provider == "ollama" and candidate.model
        )
        base_url = _ollama_base_url(self._providers_map)
        embed_model = os.environ.get("HENRY_EMBED_MODEL", "nomic-embed-text").strip() or "nomic-embed-text"
        archive_dir = chroma_persist_path()
        personal_db_path = Path(os.environ.get("PERSONAL_MEMORY_PATH", "")).expanduser()

        ollama_probe, local_models = probe_ollama(
            base_url,
            required_models=configured_local_models,
            timeout_s=READINESS_PROBE_TIMEOUT_S,
        )
        self._ollama_online = ollama_probe.status == "ok"
        self._live_local_models = set(local_models)

        return [
            ollama_probe,
            probe_embeddings(base_url, embed_model, timeout_s=READINESS_PROBE_TIMEOUT_S),
            probe_chroma_vector_memory(
                archive_dir,
                base_url=base_url,
                embed_model=embed_model,
                timeout_s=READINESS_PROBE_TIMEOUT_S,
            ),
            probe_sqlite_personal_memory(personal_db_path, timeout_s=READINESS_PROBE_TIMEOUT_S),
            self._presidio.health_probe(),
        ]

    @classmethod
    def from_project_root(
        cls,
        root: Path | None = None,
        *,
        memory_manager: Any | None = None,
    ) -> RoutedLLMProvider:
        root = root or Path(__file__).resolve().parents[1]
        henry, providers, preferences_raw, _capabilities_raw = load_routing_config(root)
        preferences = parse_routing_preferences(preferences_raw)
        capabilities_raw = refresh_capabilities_from_local_inventory(
            base_url=_ollama_base_url(providers),
            cache_path=root / "config" / "model_capabilities_cache.json",
        )
        model_capabilities = parse_model_capabilities_cache(capabilities_raw)
        routing = henry.get("routing") or {}
        candidates = parse_candidates(henry, preferences, model_capabilities)
        if not candidates:
            raise ValueError("routing is enabled but config/config.yaml has no candidates")
        weights = weights_from_preferences(preferences)
        intel = parse_autonomous_intel(root, routing if isinstance(routing, Mapping) else {})
        privacy_cfg = parse_privacy_from_presidio(routing if isinstance(routing, Mapping) else {})
        supervisor_cfg = parse_supervisor(routing if isinstance(routing, Mapping) else {})
        supervisor_parameters = generation_parameters_for_model(supervisor_cfg.model, preferences)
        supervisor_base_url = _ollama_base_url(providers)
        presidio_cfg = routing.get("presidio") if isinstance(routing, Mapping) else {}
        presidio_on = bool((presidio_cfg or {}).get("enabled", True))
        language = str((presidio_cfg or {}).get("language") or "en")
        anonym_cfg = routing.get("anonymization") if isinstance(routing, Mapping) else {}
        anonym_cloud = bool((anonym_cfg or {}).get("enabled", True))
        facade = _PresidioFacade(language=language, enabled=presidio_on)
        environment_ledger = initialize_environment_ledger(
            candidates=candidates,
            providers=providers,
            memory_manager=memory_manager,
            presidio_facade=facade,
        )
        return cls(
            root,
            candidates,
            weights,
            providers,
            facade,
            anonymize_cloud=anonym_cloud,
            intel=intel,
            privacy_cfg=privacy_cfg,
            supervisor_cfg=supervisor_cfg,
            supervisor_base_url=supervisor_base_url,
            supervisor_parameters=supervisor_parameters,
            cloud_allowance=preferences.cloud_allowance,
            routing_preferences=preferences,
            environment_ledger=environment_ledger,
            memory_manager=memory_manager,
        )

    def _maybe_refresh_benchmarks_locked(self) -> None:
        if not self._intel.enabled or not self._intel.sources:
            return
        now = time.monotonic()
        if self._last_refresh_monotonic and (
            now - self._last_refresh_monotonic
        ) < self._intel.refresh_interval_seconds:
            return
        try:
            updated = refresh_capability_scores_from_benchmarks(
                self._root,
                self._candidates,
                self._intel,
            )
            self._cap_overrides.update(updated)
            self._last_refresh_monotonic = now
        except Exception:  # noqa: BLE001 — keep routing alive if benchmarks are down
            logger.exception("Benchmark refresh failed; using cached or static capabilities.")

    def _effective_capability(self, c: ModelCandidate) -> float:
        return float(self._cap_overrides.get(c.id, c.capability))

    def _candidate_is_available(self, c: ModelCandidate) -> bool:
        if c.provider == "ollama":
            return bool(self._ollama_online and c.model in self._live_local_models)
        return provider_is_usable(c)

    def _candidate_by_id(self, candidate_id: str) -> ModelCandidate:
        for candidate in self._candidates:
            if candidate.id == candidate_id:
                return candidate
        raise KeyError(f"Unknown candidate id: {candidate_id}")

    def _available_candidates(self) -> list[ModelCandidate]:
        return [candidate for candidate in self._candidates if self._candidate_is_available(candidate)]

    def _available_local_candidates(self) -> list[ModelCandidate]:
        return [
            candidate
            for candidate in self._available_candidates()
            if candidate.deployment == "local" and candidate.provider == "ollama"
        ]

    def _should_bypass_critic(self) -> tuple[bool, str]:
        available = self._available_candidates()
        local = self._available_local_candidates()
        if len(self._candidates) <= 1:
            return True, "single-endpoint configuration"
        if len(available) <= 1:
            return True, "only one functional endpoint is available"
        if len(local) <= 1:
            return True, "only one operational local Ollama model is available"
        return False, ""

    def _select_critic_candidate(self, proposer: ModelCandidate) -> ModelCandidate | None:
        local = sorted(
            self._available_local_candidates(),
            key=lambda candidate: (candidate.id == proposer.id, -candidate.base_skill),
        )
        for candidate in local:
            if candidate.id != proposer.id:
                return candidate
        return local[0] if local else None

    def _get_critic_provider(self, candidate: ModelCandidate) -> LLMProvider:
        key = f"critic:{candidate.provider}:{candidate.model}"
        if key not in self._critic_provider_cache:
            base_url = _ollama_base_url(self._providers_map)
            params = GenerationParameters(
                temperature=0.0,
                top_p=min(1.0, max(0.1, candidate.generation_parameters.top_p)),
                num_ctx=candidate.generation_parameters.num_ctx,
            )
            self._critic_provider_cache[key] = OllamaLLMProvider(
                base_url=base_url,
                model=candidate.model,
                generation_parameters=params,
            )
        return self._critic_provider_cache[key]

    @staticmethod
    def _critic_system_prompt(user_blob: str) -> str:
        text = user_blob.lower()
        context = "general assistant response"
        if any(marker in text for marker in ("code", "python", "yaml", "json", "shell", "api", "refactor")):
            context = "coding or system automation payload"
        return (
            "You are Henry's local Critic verifier. Validate the proposer output as a "
            f"{context}. Check that it is structurally sound, contains zero PII leaks, "
            "does not expose secrets, and has correctly escaped syntax where syntax is present. "
            "Return exactly one of these forms, with no markdown and no commentary:\n"
            "APPROVED\n"
            "CORRECTED:\n"
            "<raw corrected payload>"
        )

    @staticmethod
    def _parse_critic_review(raw_response: str) -> CriticReview:
        stripped = raw_response.strip()
        if stripped.upper() == "APPROVED":
            return CriticReview(approved=True, corrected_text=None, raw_response=raw_response)
        if stripped.upper().startswith("APPROVED\n"):
            return CriticReview(approved=True, corrected_text=None, raw_response=raw_response)
        marker = "CORRECTED:"
        if stripped.upper().startswith(marker):
            corrected = stripped[len(marker):].lstrip("\n\r ")
            if corrected:
                return CriticReview(approved=False, corrected_text=corrected, raw_response=raw_response)
        return CriticReview(approved=False, corrected_text=None, raw_response=raw_response)

    @staticmethod
    def _log_critic_correction(
        *,
        proposer: ModelCandidate,
        critic: ModelCandidate,
        original_len: int,
        corrected_len: int,
    ) -> None:
        message = (
            "Dual-pass verification: Critic corrected proposer output "
            f"(proposer={proposer.id}, critic={critic.id}, "
            f"original_chars={original_len}, corrected_chars={corrected_len})."
        )
        logger.warning(message)
        print(f"[Henry Router] {message}", flush=True)

    def _verify_with_local_critic(
        self,
        *,
        proposer_text: str,
        proposer: ModelCandidate,
        user_blob: str,
    ) -> str:
        bypass, reason = self._should_bypass_critic()
        if bypass:
            logger.info("Dual-pass verification bypassed: %s.", reason)
            return proposer_text

        critic = self._select_critic_candidate(proposer)
        if critic is None:
            logger.info("Dual-pass verification bypassed: no local critic candidate available.")
            return proposer_text

        with self._lock:
            critic_provider = self._get_critic_provider(critic)

        critic_messages = [
            ChatMessage(role="system", content=self._critic_system_prompt(user_blob)),
            ChatMessage(
                role="user",
                content=(
                    "Original user/task context:\n"
                    f"{user_blob[:4000]}\n\n"
                    "Proposer output to validate:\n"
                    f"{proposer_text}"
                ),
            ),
        ]
        try:
            raw_review = critic_provider.complete(critic_messages)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Dual-pass verification failed; returning proposer output: %s", exc)
            return proposer_text

        review = self._parse_critic_review(raw_review)
        if review.approved:
            logger.info("Dual-pass verification approved by critic=%s for proposer=%s.", critic.id, proposer.id)
            return proposer_text
        if review.corrected_text is not None:
            self._log_critic_correction(
                proposer=proposer,
                critic=critic,
                original_len=len(proposer_text),
                corrected_len=len(review.corrected_text),
            )
            return review.corrected_text

        logger.warning(
            "Dual-pass verification returned an unrecognized critic response; "
            "returning proposer output unchanged.",
        )
        return proposer_text

    def _messages_for_route(self, messages: Sequence[ChatMessage], winner: ModelCandidate) -> list[ChatMessage]:
        if winner.deployment != "cloud" or not self._anonymize_cloud:
            return list(messages)
        if not presidio_online:
            raise RuntimeError(
                "Cloud route blocked because Presidio is offline; "
                "Tier 1 strict local mode is active."
            )
        out: list[ChatMessage] = []
        for m in messages:
            if m.role == "user":
                text = anonymize_for_cloud(m.content, self._presidio)
                if not presidio_online:
                    raise RuntimeError(
                        "Cloud route blocked because Presidio failed during anonymization."
                    )
                out.append(ChatMessage(role=m.role, content=text))
            else:
                out.append(ChatMessage(role=m.role, content=m.content))
        return out

    def _compute_routing(self, user_blob: str) -> tuple[ModelCandidate, dict[str, Any]]:
        ctx = self._presidio.build_routing_context(user_blob)
        strict_local_only = not presidio_online
        privacy_tier = infer_privacy_tier(ctx)
        escalation = decide_privacy_escalation(
            privacy_tier=privacy_tier,
            candidates=self._candidates,
            is_available=self._candidate_is_available,
            user_blob=user_blob,
            preferences=self._routing_preferences,
        )
        mass = weighted_presidio_mass(
            ctx.counts_by_type,
            self._privacy_cfg.entity_weights,
            self._privacy_cfg.severity_cap,
        )
        pii_severity = max(float(ctx.severity), float(mass))
        summary = presidio_summary_dict(ctx)
        summary["severity_effective"] = round(pii_severity, 4)
        summary["weighted_mass"] = round(mass, 4)

        formula_scores: dict[str, float] = {}
        candidate_rows: list[dict[str, Any]] = []
        privacy_by_id: dict[str, float] = {}

        for cand in self._candidates:
            if cand.id not in escalation.allowed_candidate_ids:
                continue
            if not self._candidate_is_available(cand):
                continue
            cap = self._effective_capability(cand)
            privacy = privacy_impact_from_presidio(
                cand,
                ctx,
                self._privacy_cfg,
                anonymize_cloud_enabled=self._anonymize_cloud,
            )
            privacy_by_id[cand.id] = privacy
            fs = total_score(
                cap,
                privacy,
                cand.cost,
                cand.latency_ms,
                self._weights,
                pii_severity=pii_severity,
            )
            formula_scores[cand.id] = fs
            candidate_rows.append(
                {
                    "id": cand.id,
                    "deployment": cand.deployment,
                    "provider": cand.provider,
                    "model": cand.model,
                    "capability": cap,
                    "cost": cand.cost,
                    "latency_ms": cand.latency_ms,
                    "privacy_impact_effective": privacy,
                }
            )

        sup_scores: dict[str, float] | None = None
        if formula_scores and self._supervisor_cfg.enabled and self._supervisor_cfg.provider == "ollama":
            snippet = user_blob.strip().replace("\n", " ")[:900]
            sup_scores = call_supervisor_scores_ollama(
                base_url=self._supervisor_base_url,
                model=self._supervisor_cfg.model,
                generation_parameters=self._supervisor_parameters,
                timeout_s=self._supervisor_cfg.timeout_seconds,
                user_snippet=snippet,
                presidio_summary=summary,
                candidate_rows=candidate_rows,
            )

        blend_w = self._supervisor_cfg.blend_weight if self._supervisor_cfg.enabled else 0.0
        final_scores = blend_formula_and_supervisor(formula_scores, sup_scores, blend_w)
        if escalation.cloud_score_multiplier < 1.0:
            final_scores = {
                cid: (
                    score * escalation.cloud_score_multiplier
                    if self._candidate_by_id(cid).deployment == "cloud"
                    else score
                )
                for cid, score in final_scores.items()
            }

        scored: list[tuple[ModelCandidate, float]] = []
        for cand in self._candidates:
            if cand.id not in escalation.allowed_candidate_ids:
                continue
            if not self._candidate_is_available(cand):
                continue
            if cand.id not in final_scores:
                continue
            scored.append((cand, final_scores[cand.id]))

        if not scored:
            if strict_local_only:
                raise RuntimeError(
                    "No local Ollama model is available while Presidio is offline; "
                    "strict local-only routing cannot proceed."
                )
            raise RuntimeError("No routable LLM candidates (check API keys / config).")

        def tie_key(item: tuple[ModelCandidate, float]) -> tuple[float, int]:
            cand, score = item
            local_bonus = 1 if cand.deployment == "local" else 0
            return (score, local_bonus)

        winner = max(scored, key=tie_key)[0]
        details: dict[str, Any] = {
            "presidio": summary,
            "presidio_engine_active": presidio_online,
            "strict_local_only": strict_local_only,
            "strict_local_only_reason": _PRESIDIO_OFFLINE_REASON if strict_local_only else "",
            "privacy_tier": privacy_tier,
            "escalation": {
                "action": escalation.action,
                "allowed_candidate_ids": sorted(escalation.allowed_candidate_ids),
                "cloud_score_multiplier": escalation.cloud_score_multiplier,
                "local_efficiency_ratio": round(escalation.local_efficiency_ratio, 4),
                "lookup_efficiency_ratio": round(escalation.lookup_efficiency_ratio, 4),
                "reason": escalation.reason,
            },
            "privacy_impact_effective_by_candidate": dict(privacy_by_id),
            "formula_scores": dict(formula_scores),
            "supervisor_scores": dict(sup_scores) if sup_scores else None,
            "final_scores": dict(final_scores),
            "winner": {
                "id": winner.id,
                "provider": winner.provider,
                "deployment": winner.deployment,
                "model": winner.model,
            },
        }
        return winner, details

    def _pick_candidate(self, user_blob: str) -> ModelCandidate:
        return self._compute_routing(user_blob)[0]

    def preview_routing(self, user_blob: str) -> dict[str, Any]:
        """Diagnostics: Presidio summary, per-candidate privacy scores, blended scores, chosen model."""
        with self._lock:
            self._maybe_refresh_benchmarks_locked()
            _, details = self._compute_routing(user_blob)
        return details

    def route_query(self, raw_text: str) -> dict[str, Any]:
        """Presidio-backed multi-model routing analysis for a text blob (documents or chat)."""
        return self.preview_routing(raw_text)

    def process_memory_intent(self, user_text: str) -> dict[str, Any]:
        """
        Ask a local Ollama model whether the user message contains durable personal facts;
        if so, persist rows into SQLite via the dual-memory manager.
        """
        result: dict[str, Any] = {"saved": 0, "facts": []}
        mm = self._memory_manager
        if mm is None:
            result["skipped"] = "no_memory_manager"
            return result
        if not getattr(mm, "is_encrypted_storage_available", False):
            result["skipped"] = "mount_unavailable"
            return result
        if not getattr(mm, "personal_memory_ready", False):
            result["skipped"] = "personal_db_unavailable"
            return result

        stripped = user_text.strip()
        if len(stripped) < 8:
            return result

        local_provider: LLMProvider | None = None
        with self._lock:
            for cand in self._candidates:
                if cand.deployment == "local" and cand.provider == "ollama" and self._candidate_is_available(cand):
                    local_provider = self._get_provider(cand)
                    break

        if local_provider is None:
            result["skipped"] = "no_local_ollama"
            return result

        messages = [
            ChatMessage(role="system", content=_MEMORY_INTENT_SYSTEM),
            ChatMessage(
                role="user",
                content=json.dumps({"user_message": stripped}, ensure_ascii=False),
            ),
        ]
        try:
            raw = local_provider.complete(messages)
        except Exception as exc:  # noqa: BLE001
            logger.warning("process_memory_intent LLM call failed: %s", exc)
            result["error"] = str(exc)
            return result

        obj = _extract_json_object(raw)
        if not isinstance(obj, Mapping):
            result["skipped"] = "unparseable_json"
            return result

        remember = bool(obj.get("remember"))
        if not remember:
            result["skipped"] = "no_durable_facts"
            return result

        facts_raw = obj.get("facts")
        if not isinstance(facts_raw, list) or not facts_raw:
            result["skipped"] = "empty_facts"
            return result

        saved_ids: list[int] = []
        for row in facts_raw:
            if not isinstance(row, Mapping):
                continue
            cat = str(row.get("category") or "general").strip()
            fact = str(row.get("fact") or "").strip()
            if not fact:
                continue
            row_id = mm.save_user_fact(cat, fact)
            if row_id is not None:
                saved_ids.append(row_id)

        result["saved"] = len(saved_ids)
        result["fact_ids"] = saved_ids
        return result

    def _get_provider(self, c: ModelCandidate) -> LLMProvider:
        key = f"{c.provider}:{c.model}"
        if key not in self._provider_cache:
            self._provider_cache[key] = build_provider_for_candidate(c, self._providers_map)
        return self._provider_cache[key]

    @staticmethod
    def _messages_to_payload(messages: Sequence[ChatMessage]) -> list[dict[str, str]]:
        return [{"role": message.role, "content": message.content} for message in messages]

    @staticmethod
    def _messages_from_payload(raw_messages: Any) -> list[ChatMessage]:
        if not isinstance(raw_messages, list):
            raise ValueError("queued LLM task payload must include a messages list")
        messages: list[ChatMessage] = []
        for row in raw_messages:
            if not isinstance(row, Mapping):
                raise ValueError("queued message rows must be objects")
            role = str(row.get("role") or "").strip()
            content = str(row.get("content") or "")
            if not role:
                raise ValueError("queued message row missing role")
            messages.append(ChatMessage(role=role, content=content))
        return messages

    @staticmethod
    def _infer_execution_mode(
        *,
        execution_mode: ExecutionMode,
        intent: str,
        payload_data: Mapping[str, Any],
    ) -> ExecutionMode:
        if execution_mode != "auto":
            return execution_mode
        normalized = intent.strip().lower()
        heavy_intents = {
            "document_processing",
            "document_classification",
            "archive_ingest",
            "batch_ocr",
            "long_context_generation",
            "tool_execution",
        }
        if normalized in heavy_intents:
            return "background"
        raw_text = str(payload_data.get("raw_text") or payload_data.get("text") or "")
        if len(raw_text) > 8_000:
            return "background"
        return "sync"

    def _process_background_task(self, task: TaskPayload) -> dict[str, Any]:
        """Generic heavy-task processor; frontend-specific work stays outside the router."""
        logger.info("Processing queued task id=%s intent=%s", task.task_id, task.intent)
        print(
            f"[Henry Router] Processing queued task id={task.task_id} intent={task.intent}",
            flush=True,
        )
        if task.intent in {"chat", "short_query", "long_context_generation", "llm_generation"}:
            messages = self._messages_from_payload(task.payload_data.get("messages"))
            reply = self.complete(messages)
            return {"status": "completed", "reply": reply}
        return {
            "status": "completed",
            "message": "Task accepted by queue; no router-local processor is registered for this intent.",
            "intent": task.intent,
        }

    def complete_or_queue(
        self,
        messages: Sequence[ChatMessage],
        *,
        client_id: str = "core",
        intent: str = "chat",
        execution_mode: ExecutionMode = "auto",
        payload_data: Mapping[str, Any] | None = None,
    ) -> str | dict[str, str]:
        """Run fast requests immediately; enqueue heavy work and return a task id."""
        payload: dict[str, Any] = dict(payload_data or {})
        payload.setdefault("messages", self._messages_to_payload(messages))
        mode = self._infer_execution_mode(
            execution_mode=execution_mode,
            intent=intent,
            payload_data=payload,
        )
        if mode == "sync":
            return self.complete(messages)

        task = self._queue_manager.enqueue(
            client_id=client_id,
            intent=intent,
            payload_data=payload,
        )
        return {"status": "queued", "task_id": task.task_id}

    def complete(self, messages: Sequence[ChatMessage]) -> str:
        user_parts = [m.content for m in messages if m.role == "user"]
        user_blob = "\n".join(user_parts)

        with self._lock:
            self._maybe_refresh_benchmarks_locked()
            winner = self._pick_candidate(user_blob)
            provider = self._get_provider(winner)
            to_send = self._messages_for_route(messages, winner)

        proposer_text = provider.complete(to_send)
        return self._verify_with_local_critic(
            proposer_text=proposer_text,
            proposer=winner,
            user_blob=user_blob,
        )


def build_routed_llm(root: Path | None = None) -> RoutedLLMProvider:
    """Factory helper for `core.llm.factory` when dynamic routing is enabled."""
    return RoutedLLMProvider.from_project_root(root)


LLMManager = RoutedLLMProvider
