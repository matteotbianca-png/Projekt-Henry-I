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
from typing import Any, Mapping, Sequence

import httpx
import yaml

from core.llm.anthropic import AnthropicLLMProvider
from core.llm.base import ChatMessage, LLMProvider
from core.llm.ollama import OllamaLLMProvider
from core.llm.openai import OpenAILLMProvider

logger = logging.getLogger(__name__)

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

try:
    from presidio_analyzer import AnalyzerEngine
    from presidio_anonymizer import AnonymizerEngine
except ImportError:  # pragma: no cover — optional in minimal installs
    AnalyzerEngine = None  # type: ignore[misc, assignment]
    AnonymizerEngine = None  # type: ignore[misc, assignment]


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


# --- Config loading --------------------------------------------------------


def _expand_placeholders(value: str) -> str:
    return os.path.expandvars(value)


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _load_yaml(path: Path) -> Mapping[str, Any]:
    raw = path.read_text(encoding="utf-8")
    data = yaml.safe_load(os.path.expandvars(raw)) or {}
    if not isinstance(data, Mapping):
        raise ValueError(f"{path} must be a mapping at the root")
    return data


def load_routing_config(root: Path) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    """Return (henry_config, providers_yaml) mappings."""
    cfg_path = root / "config" / "config.yaml"
    prov_path = root / "config" / "providers.yaml"
    henry = _load_yaml(cfg_path) if cfg_path.is_file() else {}
    providers = _load_yaml(prov_path) if prov_path.is_file() else {}
    return henry, providers


def parse_candidates(raw: Mapping[str, Any]) -> list[ModelCandidate]:
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
        if not cid or not provider or not model:
            continue
        aliases = entry.get("benchmark_aliases") or []
        alias_tuple = tuple(str(a).strip() for a in aliases if str(a).strip())
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
                benchmark_aliases=alias_tuple,
            )
        )
    return out


def parse_weights(routing: Mapping[str, Any]) -> RoutingWeights:
    w = routing.get("weights") or {}
    return RoutingWeights(
        w1_capability=_as_float(w.get("w1_capability"), 1.0),
        w2_privacy=_as_float(w.get("w2_privacy"), 1.0),
        w3_cost=_as_float(w.get("w3_cost"), 1.0),
        w4_latency=_as_float(w.get("w4_latency"), 0.001),
        pii_privacy_weight_multiplier=_as_float(routing.get("pii_privacy_weight_multiplier"), 3.0),
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


# --- Presidio + lightweight fallback ---------------------------------------


@dataclass
class _PresidioEngines:
    analyzer: Any
    anonymizer: Any
    language: str


class _PresidioFacade:
    """Microsoft Presidio analyzer/anonymizer; regex only if Presidio is unavailable."""

    def __init__(self, language: str, enabled: bool) -> None:
        self._language = language
        self._requested = enabled
        self._engines: _PresidioEngines | None = None
        if enabled and AnalyzerEngine is not None and AnonymizerEngine is not None:
            try:
                self._engines = _PresidioEngines(
                    analyzer=AnalyzerEngine(),
                    anonymizer=AnonymizerEngine(),
                    language=language,
                )
            except SystemExit as exc:  # spaCy may sys.exit when models / pip are missing
                logger.warning("Presidio init exited early (%s); using regex PII fallback.", exc)
                self._engines = None
            except Exception as exc:  # noqa: BLE001 — stay up without full NLP stack
                logger.warning("Presidio init failed (%s); using regex PII fallback.", exc)
                self._engines = None
        elif enabled:
            logger.warning("Presidio packages not importable; using regex PII fallback.")

        self._email = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b", re.I)
        self._phone = re.compile(r"\b\+?\d[\d\s().-]{7,}\d\b")
        self._date = re.compile(
            r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{4}-\d{2}-\d{2}\b",
            re.I,
        )

    @property
    def presidio_active(self) -> bool:
        return self._engines is not None

    def analyze_results(self, text: str) -> tuple[list[Any], str]:
        """Return (recognizer_results, method_label)."""
        if not text.strip():
            return [], "empty"
        if self._engines is not None:
            try:
                results = self._engines.analyzer.analyze(text=text, language=self._engines.language)
                return list(results), "presidio"
            except Exception as exc:  # noqa: BLE001
                logger.debug("Presidio analyze failed: %s", exc)
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
                logger.debug("Presidio anonymize failed: %s", exc)
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
    timeout_s: float,
    user_snippet: str,
    presidio_summary: dict[str, Any],
    candidate_rows: list[dict[str, Any]],
) -> dict[str, float] | None:
    """Ask the local Qwen supervisor (Ollama) for per-candidate routing scores."""
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
    if not base:
        base = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    return base.rstrip("/")


def build_provider_for_candidate(
    candidate: ModelCandidate,
    providers: Mapping[str, Any],
    timeout_s: float = 120.0,
) -> LLMProvider:
    if candidate.provider == "ollama":
        base_url = _ollama_base_url(providers)
        return OllamaLLMProvider(base_url=base_url, model=candidate.model, timeout_s=timeout_s)
    if candidate.provider == "openai":
        return OpenAILLMProvider(model=candidate.model, timeout_s=timeout_s)
    if candidate.provider == "anthropic":
        return AnthropicLLMProvider(model=candidate.model, timeout_s=timeout_s)
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
        self._memory_manager = memory_manager
        self._provider_cache: dict[str, LLMProvider] = {}
        self._cap_overrides: dict[str, float] = {}
        self._last_refresh_monotonic = 0.0
        self._lock = threading.Lock()

        if self._intel.enabled:
            self._cap_overrides = load_cached_capabilities(self._intel)

    @classmethod
    def from_project_root(
        cls,
        root: Path | None = None,
        *,
        memory_manager: Any | None = None,
    ) -> RoutedLLMProvider:
        root = root or Path(__file__).resolve().parents[1]
        henry, providers = load_routing_config(root)
        routing = henry.get("routing") or {}
        candidates = parse_candidates(henry)
        if not candidates:
            raise ValueError("routing is enabled but config/config.yaml has no candidates")
        weights = parse_weights(routing if isinstance(routing, Mapping) else {})
        intel = parse_autonomous_intel(root, routing if isinstance(routing, Mapping) else {})
        privacy_cfg = parse_privacy_from_presidio(routing if isinstance(routing, Mapping) else {})
        supervisor_cfg = parse_supervisor(routing if isinstance(routing, Mapping) else {})
        supervisor_base_url = _ollama_base_url(providers)
        presidio_cfg = routing.get("presidio") if isinstance(routing, Mapping) else {}
        presidio_on = bool((presidio_cfg or {}).get("enabled", True))
        language = str((presidio_cfg or {}).get("language") or "en")
        anonym_cfg = routing.get("anonymization") if isinstance(routing, Mapping) else {}
        anonym_cloud = bool((anonym_cfg or {}).get("enabled", True))
        facade = _PresidioFacade(language=language, enabled=presidio_on)
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

    def _messages_for_route(self, messages: Sequence[ChatMessage], winner: ModelCandidate) -> list[ChatMessage]:
        if winner.deployment != "cloud" or not self._anonymize_cloud:
            return list(messages)
        if not self._presidio.presidio_active:
            logger.warning(
                "Cloud route selected but Presidio engines are not active; "
                "using regex-based anonymization before the cloud call.",
            )
        out: list[ChatMessage] = []
        for m in messages:
            if m.role == "user":
                text = anonymize_for_cloud(m.content, self._presidio)
                out.append(ChatMessage(role=m.role, content=text))
            else:
                out.append(ChatMessage(role=m.role, content=m.content))
        return out

    def _compute_routing(self, user_blob: str) -> tuple[ModelCandidate, dict[str, Any]]:
        ctx = self._presidio.build_routing_context(user_blob)
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
            if not provider_is_usable(cand):
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
        if self._supervisor_cfg.enabled and self._supervisor_cfg.provider == "ollama":
            snippet = user_blob.strip().replace("\n", " ")[:900]
            sup_scores = call_supervisor_scores_ollama(
                base_url=self._supervisor_base_url,
                model=self._supervisor_cfg.model,
                timeout_s=self._supervisor_cfg.timeout_seconds,
                user_snippet=snippet,
                presidio_summary=summary,
                candidate_rows=candidate_rows,
            )

        blend_w = self._supervisor_cfg.blend_weight if self._supervisor_cfg.enabled else 0.0
        final_scores = blend_formula_and_supervisor(formula_scores, sup_scores, blend_w)

        scored: list[tuple[ModelCandidate, float]] = []
        for cand in self._candidates:
            if not provider_is_usable(cand):
                continue
            if cand.id not in final_scores:
                continue
            scored.append((cand, final_scores[cand.id]))

        if not scored:
            raise RuntimeError("No routable LLM candidates (check API keys / config).")

        def tie_key(item: tuple[ModelCandidate, float]) -> tuple[float, int]:
            cand, score = item
            local_bonus = 1 if cand.deployment == "local" else 0
            return (score, local_bonus)

        winner = max(scored, key=tie_key)[0]
        details: dict[str, Any] = {
            "presidio": summary,
            "presidio_engine_active": self._presidio.presidio_active,
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
                if cand.deployment == "local" and cand.provider == "ollama" and provider_is_usable(cand):
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

    def complete(self, messages: Sequence[ChatMessage]) -> str:
        user_parts = [m.content for m in messages if m.role == "user"]
        user_blob = "\n".join(user_parts)

        with self._lock:
            self._maybe_refresh_benchmarks_locked()
            winner = self._pick_candidate(user_blob)
            provider = self._get_provider(winner)
            to_send = self._messages_for_route(messages, winner)

        return provider.complete(to_send)


def build_routed_llm(root: Path | None = None) -> RoutedLLMProvider:
    """Factory helper for `core.llm.factory` when dynamic routing is enabled."""
    return RoutedLLMProvider.from_project_root(root)


LLMManager = RoutedLLMProvider
