"""Install preflight and live runtime readiness probes for Henry."""

from __future__ import annotations

import os
import queue
import sqlite3
import threading
import time
from dataclasses import asdict, dataclass
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Callable, Literal, TypeVar
from uuid import uuid4

ProbeStatus = Literal["ok", "warn", "fail", "missing", "initializing", "disabled"]
READINESS_PROBE_TIMEOUT_S = 15.0


@dataclass(frozen=True)
class HealthProbe:
    name: str
    status: ProbeStatus
    detail: str
    latency_ms: float | None = None


class OllamaEmbeddingAdapter:
    """Minimal Chroma-compatible embedding adapter backed by Ollama HTTP."""

    def __init__(self, *, model: str, base_url: str, timeout_s: float = READINESS_PROBE_TIMEOUT_S) -> None:
        self._model = model
        self._url = f"{base_url.rstrip('/')}/api/embeddings"
        self._timeout = timeout_s

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        import httpx

        payload = {"model": self._model, "prompt": text}
        with httpx.Client(timeout=self._timeout) as client:
            response = client.post(self._url, json=payload)
            response.raise_for_status()
            data = response.json()

        embedding = data.get("embedding")
        if not isinstance(embedding, list) or not embedding:
            raise RuntimeError("unexpected Ollama embedding response shape")
        return [float(value) for value in embedding]


T = TypeVar("T")


def health_probe_to_dict(probe: HealthProbe) -> dict[str, Any]:
    return asdict(probe)


def install_preflight(required: list[tuple[str, str]]) -> dict[str, str]:
    """Return package install presence without importing executable module code."""
    return {
        package: "installed" if find_spec(import_name) is not None else "missing"
        for package, import_name in required
    }


def chroma_persist_path() -> Path:
    """Return Henry's Chroma persistence directory on the encrypted volume."""
    raw = os.environ.get("ARCHIVE_DB_PATH", "").strip()
    if raw:
        return Path(raw).expanduser()
    mount = Path(os.environ.get("MEMORY_MOUNT_PATH", "/Volumes/HenryData").strip()).expanduser()
    return mount / "chroma"


def disable_chroma_telemetry_env() -> None:
    os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
    os.environ.setdefault("CHROMA_ANONYMIZED_TELEMETRY", "False")
    os.environ.setdefault("CHROMA_TELEMETRY_BOARD", "False")


def chroma_settings() -> Any:
    from chromadb.config import Settings

    candidates = [
        {
            "is_persistent": True,
            "anonymized_telemetry": False,
            "chroma_telemetry_board": False,
            "chroma_mode": "lightweight",
        },
        {
            "is_persistent": True,
            "anonymized_telemetry": False,
            "chroma_telemetry_board": False,
        },
        {
            "is_persistent": True,
            "anonymized_telemetry": False,
        },
    ]
    for kwargs in candidates:
        try:
            return Settings(**kwargs)
        except Exception:  # noqa: BLE001 - Chroma settings vary by version
            continue
    return Settings()


def run_with_timeout(
    name: str,
    fn: Callable[[], HealthProbe],
    *,
    timeout_s: float = READINESS_PROBE_TIMEOUT_S,
    timeout_status: ProbeStatus = "fail",
) -> HealthProbe:
    """Run a potentially heavy probe on a daemon thread and bound caller wait time."""
    started = time.monotonic()
    results: queue.Queue[HealthProbe | BaseException] = queue.Queue(maxsize=1)

    def target() -> None:
        try:
            results.put(fn())
        except BaseException as exc:  # noqa: BLE001 - probes must never crash boot
            results.put(exc)

    thread = threading.Thread(target=target, name=f"henry-health-{name}", daemon=True)
    thread.start()
    thread.join(timeout_s)

    latency_ms = (time.monotonic() - started) * 1000
    if thread.is_alive():
        return HealthProbe(
            name=name,
            status=timeout_status,
            detail=f"probe exceeded {timeout_s:.1f}s timeout",
            latency_ms=latency_ms,
        )

    result = results.get_nowait()
    if isinstance(result, HealthProbe):
        return result
    return HealthProbe(name=name, status="fail", detail=str(result), latency_ms=latency_ms)


def probe_ollama(
    base_url: str,
    *,
    required_models: tuple[str, ...] = (),
    timeout_s: float = READINESS_PROBE_TIMEOUT_S,
) -> tuple[HealthProbe, tuple[str, ...]]:
    import httpx

    started = time.monotonic()
    try:
        with httpx.Client(timeout=timeout_s) as client:
            response = client.get(f"{base_url.rstrip('/')}/api/tags")
            response.raise_for_status()
            data = response.json()
    except Exception as exc:  # noqa: BLE001
        return (
            HealthProbe("ollama", "fail", f"/api/tags unreachable: {exc}", _elapsed_ms(started)),
            (),
        )

    models_raw = data.get("models") if isinstance(data, dict) else None
    if not isinstance(models_raw, list):
        return (
            HealthProbe("ollama", "fail", "unexpected /api/tags response shape", _elapsed_ms(started)),
            (),
        )

    names: set[str] = set()
    for row in models_raw:
        if not isinstance(row, dict):
            continue
        for key in ("model", "name"):
            value = str(row.get(key) or "").strip()
            if value:
                names.add(value)

    missing = [model for model in required_models if model and model not in names]
    if missing:
        return (
            HealthProbe(
                "ollama",
                "fail",
                "missing configured model(s): " + ", ".join(missing),
                _elapsed_ms(started),
            ),
            tuple(sorted(names)),
        )

    return (
        HealthProbe("ollama", "ok", f"{len(names)} model(s) live via /api/tags", _elapsed_ms(started)),
        tuple(sorted(names)),
    )


def probe_embeddings(base_url: str, model: str, *, timeout_s: float = READINESS_PROBE_TIMEOUT_S) -> HealthProbe:
    started = time.monotonic()
    try:
        adapter = OllamaEmbeddingAdapter(model=model, base_url=base_url, timeout_s=timeout_s)
        vector = adapter.embed_query("henry health probe")
    except Exception as exc:  # noqa: BLE001
        return HealthProbe("embeddings", "fail", str(exc), _elapsed_ms(started))

    if not vector or not all(isinstance(value, float) for value in vector):
        return HealthProbe("embeddings", "fail", "embedding endpoint returned invalid vector", _elapsed_ms(started))
    return HealthProbe("embeddings", "ok", f"vector dimension {len(vector)}", _elapsed_ms(started))


def probe_chroma_vector_memory(
    persist_directory: Path,
    *,
    base_url: str,
    embed_model: str,
    timeout_s: float = READINESS_PROBE_TIMEOUT_S,
) -> HealthProbe:
    def do_probe() -> HealthProbe:
        started = time.monotonic()
        if not str(persist_directory):
            return HealthProbe("chroma_vector_memory", "disabled", "ARCHIVE_DB_PATH is empty", _elapsed_ms(started))
        try:
            os.makedirs(persist_directory, exist_ok=True)
        except OSError as exc:
            return HealthProbe("chroma_vector_memory", "fail", f"could not create {persist_directory}: {exc}", _elapsed_ms(started))

        disable_chroma_telemetry_env()
        import chromadb

        collection_name = f"henry_health_{os.getpid()}_{uuid4().hex[:8]}"
        embeddings = OllamaEmbeddingAdapter(model=embed_model, base_url=base_url, timeout_s=timeout_s)
        client = chromadb.PersistentClient(
            path=str(persist_directory),
            settings=chroma_settings(),
        )
        collection = client.get_or_create_collection(collection_name)
        try:
            doc_id = uuid4().hex
            text = "henry non-pii vector health probe"
            collection.upsert(
                ids=[doc_id],
                documents=[text],
                embeddings=[embeddings.embed_query(text)],
                metadatas=[{"tier": "health_probe"}],
            )
            query = collection.query(
                query_embeddings=[embeddings.embed_query("henry vector health")],
                n_results=1,
                include=["documents", "metadatas"],
            )
            hits = (query.get("documents") or [[]])[0] or []
            if not hits:
                return HealthProbe("chroma_vector_memory", "fail", "write succeeded but read returned no hits", _elapsed_ms(started))
            return HealthProbe("chroma_vector_memory", "ok", "temporary write/read/delete transaction succeeded", _elapsed_ms(started))
        finally:
            client.delete_collection(collection_name)

    return run_with_timeout("chroma_vector_memory", do_probe, timeout_s=timeout_s)


def probe_sqlite_personal_memory(db_path: Path, *, timeout_s: float = READINESS_PROBE_TIMEOUT_S) -> HealthProbe:
    started = time.monotonic()
    if not str(db_path):
        return HealthProbe("sqlite_personal_memory", "disabled", "PERSONAL_MEMORY_PATH is empty", _elapsed_ms(started))
    if not db_path.parent.exists():
        return HealthProbe("sqlite_personal_memory", "missing", f"{db_path.parent} does not exist", _elapsed_ms(started))
    if not db_path.exists():
        return HealthProbe("sqlite_personal_memory", "missing", f"{db_path} does not exist", _elapsed_ms(started))

    probe_file = db_path.parent / f".henry_sqlite_write_probe_{os.getpid()}"
    try:
        with sqlite3.connect(str(db_path), timeout=timeout_s) as conn:
            row = conn.execute("PRAGMA quick_check;").fetchone()
        if not row or row[0] != "ok":
            return HealthProbe("sqlite_personal_memory", "fail", f"PRAGMA quick_check returned {row}", _elapsed_ms(started))

        probe_file.write_bytes(b"x")
        if probe_file.stat().st_size != 1:
            return HealthProbe("sqlite_personal_memory", "fail", "1-byte write probe failed", _elapsed_ms(started))
    except Exception as exc:  # noqa: BLE001
        return HealthProbe("sqlite_personal_memory", "fail", str(exc), _elapsed_ms(started))
    finally:
        try:
            probe_file.unlink()
        except FileNotFoundError:
            pass
        except OSError:
            pass

    return HealthProbe("sqlite_personal_memory", "ok", "quick_check and 1-byte write succeeded", _elapsed_ms(started))


def probe_satellite_status(name: str, base_url: str, *, timeout_s: float = READINESS_PROBE_TIMEOUT_S) -> HealthProbe:
    import httpx

    started = time.monotonic()
    try:
        with httpx.Client(timeout=timeout_s) as client:
            response = client.get(f"{base_url.rstrip('/')}/status")
            response.raise_for_status()
            data = response.json()
    except httpx.ConnectError:
        return HealthProbe(name, "missing", f"{base_url.rstrip('/')}/status did not respond", _elapsed_ms(started))
    except Exception as exc:  # noqa: BLE001
        return HealthProbe(name, "fail", str(exc), _elapsed_ms(started))

    service = data.get("service") if isinstance(data, dict) else None
    detail = f"{service or 'status endpoint'} responded"
    return HealthProbe(name, "ok", detail, _elapsed_ms(started))


def _elapsed_ms(started: float) -> float:
    return round((time.monotonic() - started) * 1000, 1)
