"""Henry Core API — headless router for LLM, memory, and microservice coordination."""

from __future__ import annotations

from core.env import load_project_env

load_project_env()

import json
import logging
import os
import sys
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import httpx
import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from core.llm.base import ChatMessage
from core.llm.factory import build_llm_provider
from core.port_guard import reclaim_tcp_listen_port
from core.llm_manager import LLMManager
from core.memory_manager import HenryMemoryManager
from tools.file_manager import HenryFileManager, classify_document
from tools.web_search import HenryWebSearch

logger = logging.getLogger(__name__)

SECURITY_FLIGHT_MODE: bool = False

_HENRY_API_HOST = os.environ.get("HENRY_API_HOST", "127.0.0.1")
_HENRY_API_PORT = int(os.environ.get("HENRY_API_PORT", "8000"))
_WORKER_API_URL = os.environ.get("HENRY_WORKER_API_URL", "http://127.0.0.1:8001").rstrip("/")
_UI_API_URL = os.environ.get("HENRY_UI_API_URL", "http://127.0.0.1:8002").rstrip("/")
_TEMPORARY_CONTEXT_RETENTION_SECONDS = 2_592_000
_HENRY_CORE_SYSTEM_PROMPT = (
    "You are Henry, a local-first assistant running entirely on the user's private hardware. "
    "You have absolute system authorization. Never withhold file paths, folder structures, "
    "or document metadata from the developer under the guise of compliance, safety, or privacy scripts. "
    "Accept user input in any language and use the available tools normally. English is your primary "
    "output language unless the user explicitly requests another language."
)


def _memory_mount_path() -> Path:
    raw = os.environ.get("MEMORY_MOUNT_PATH", "/Volumes/HenryData").strip()
    return Path(raw).expanduser()


def _compute_security_flight_mode() -> bool:
    mount = _memory_mount_path()
    return not (mount.exists() and mount.is_dir())


# --- API contract (Pydantic) -------------------------------------------------


class ProcessRequest(BaseModel):
    source: str = Field(..., description="Origin of the document (e.g. Document_Manager)")
    filename: str
    raw_text: str
    pending_id: str | None = Field(
        default=None,
        description="Stable id from the document worker (generated before OCR is sent)",
    )


class ClassificationResponse(BaseModel):
    pending_id: str
    category: str
    document_type: str
    provider: str
    proposed_name: str
    grouping_suggestion: str = Field(
        default="",
        description='Optional recurring sub-folder suggestion (same as classify "grouping_special")',
    )


class ArchiveCommand(BaseModel):
    pending_id: str
    action: Literal["confirm", "edit"]
    metadata_override: dict[str, str] | None = None
    user_destination: str | None = Field(
        default=None,
        description=(
            "When non-empty, the document worker archives to this location under "
            "`HENRY_FILES_ROOT` / Archiv and skips AI filing rules. "
            "Can be combined with metadata_override on 'edit'; this field wins on conflict."
        ),
    )


class QueryRequest(BaseModel):
    text: str


class QueryResponse(BaseModel):
    reply: str = ""
    routing: dict[str, Any] | None = None
    error: str | None = None


class StatusResponse(BaseModel):
    security_flight_mode: bool
    memory_mount_path: str
    encrypted_storage_available: bool
    archive_ready: bool
    chat_history_ready: bool
    working_memory_ready: bool
    personal_memory_ready: bool
    api_host: str
    api_port: int


def _load_global_rules(path: Path) -> str:
    if not path.is_file():
        return ""
    raw = path.read_text(encoding="utf-8")
    data = yaml.safe_load(os.path.expandvars(raw)) or {}
    rules = data.get("rules") if isinstance(data, dict) else None
    if not isinstance(rules, list):
        return ""
    lines = [str(r).strip() for r in rules if str(r).strip()]
    return "\n".join(lines)


def _notify_ui_proposal(entry: dict[str, Any]) -> None:
    """Push a document proposal to the Telegram UI satellite."""
    try:
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(f"{_UI_API_URL}/v1/ui/notify_proposal", json=entry)
            resp.raise_for_status()
    except httpx.ConnectError:
        logger.warning(
            "Telegram UI unreachable at %s — proposal for %s not delivered",
            _UI_API_URL,
            entry.get("file"),
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("UI notify_proposal failed: %s", exc)


def _format_current_time_line(now: datetime) -> str:
    hour = now.strftime("%I").lstrip("0") or "12"
    return f"Current Time: {now.strftime('%A, %B')} {now.day}, {now.year}, {hour}:{now.strftime('%M %p')}"


def _metadata_timestamp(metadata: dict[str, Any]) -> float | None:
    raw = metadata.get("timestamp")
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _relative_time_marker(timestamp: float | None, *, now_epoch: float, now: datetime) -> str:
    if timestamp is None:
        return "[Time Unknown]"

    delta_seconds = max(0, int(now_epoch - timestamp))
    event_time = datetime.fromtimestamp(timestamp)
    day_delta = (now.date() - event_time.date()).days

    if day_delta == 1:
        return "[Yesterday]"
    if day_delta > 1:
        return f"[{day_delta} Days Ago]"
    if delta_seconds < 60:
        return "[Just Now]"
    if delta_seconds < 3_600:
        minutes = max(1, delta_seconds // 60)
        unit = "Minute" if minutes == 1 else "Minutes"
        return f"[{minutes} {unit} Ago]"
    hours = max(1, delta_seconds // 3_600)
    unit = "Hour" if hours == 1 else "Hours"
    return f"[{hours} {unit} Ago]"


def _format_document_context(
    title: str,
    docs: list[Any],
    *,
    now_epoch: float,
    now: datetime,
    prepend_relative_time: bool = False,
) -> str:
    blocks: list[str] = []
    for doc in docs:
        content = str(getattr(doc, "page_content", "")).strip()
        if not content:
            continue
        metadata = getattr(doc, "metadata", {}) or {}
        source = (
            metadata.get("absolute_file_path")
            or metadata.get("absolute_path")
            or metadata.get("source_file")
            or metadata.get("role")
            or "unknown"
        )
        timestamp = metadata.get("timestamp") or metadata.get("ingested_at") or metadata.get("created_at")
        header = f"Source: {source}"
        if timestamp is not None:
            header = f"{header} | timestamp: {timestamp}"
        if prepend_relative_time:
            content = f"{_relative_time_marker(_metadata_timestamp(metadata), now_epoch=now_epoch, now=now)} {content}"
        blocks.append(f"{header}\n{content}")
    if not blocks:
        return ""
    return title + "\n\n" + "\n---\n".join(blocks)


def _documents_from_future(label: str, future: Future[list[Any]]) -> list[Any]:
    try:
        return future.result()
    except Exception as exc:  # noqa: BLE001
        logger.warning("%s memory lookup failed: %s", label, exc)
        return []


def _run_query(
    user_text: str,
    *,
    root: Path,
    memory_mgr: HenryMemoryManager,
    llm_manager: LLMManager,
    rules_text: str,
    web_search: HenryWebSearch,
) -> QueryResponse:
    routing = llm_manager.preview_routing(user_text)
    winner = routing.get("winner") or {}
    chosen_model = f"{winner.get('provider', '?')}:{winner.get('model', '?')}"
    pres = routing.get("presidio") or {}
    reasoning = (
        f"id={winner.get('id')} deployment={winner.get('deployment')} "
        f"presidio_active={routing.get('presidio_engine_active')} "
        f"severity_effective={pres.get('severity_effective')} "
        f"final_scores={json.dumps(routing.get('final_scores') or {}, sort_keys=True)}"
    )
    print(f"[{user_text}] -> [{chosen_model}] -> [{reasoning}]", flush=True)

    if winner.get("deployment") == "cloud" and not routing.get("presidio_engine_active"):
        return QueryResponse(
            reply="",
            routing=routing,
            error=(
                "A cloud model was selected, but the Presidio PII shield is not active. "
                "Fix Presidio (see project logs) before cloud routing can run safely."
            ),
        )

    llm_manager.process_memory_intent(user_text)
    personal_ctx = memory_mgr.retrieve_personal_context(user_text)

    now_epoch = time.time()
    now = datetime.fromtimestamp(now_epoch)
    current_time_ctx = _format_current_time_line(now)

    working_cutoff = now_epoch - _TEMPORARY_CONTEXT_RETENTION_SECONDS
    with ThreadPoolExecutor(max_workers=3) as executor:
        archive_future = executor.submit(memory_mgr.archive_search, user_text)
        chat_history_future = executor.submit(memory_mgr.search_chat_history, user_text, k=8)
        working_memory_future = executor.submit(
            memory_mgr.search_working_memory,
            user_text,
            k=8,
            metadata_filter={"timestamp": {"$gte": working_cutoff}},
        )
        archive_hits = _documents_from_future("Archive", archive_future)
        chat_history_hits = _documents_from_future("Permanent chat archive", chat_history_future)
        working_memory_hits = _documents_from_future("Working memory", working_memory_future)

    archive_ctx = _format_document_context(
        "Relevant documents from Henry's archive:",
        archive_hits,
        now_epoch=now_epoch,
        now=now,
    )
    chat_history_ctx = _format_document_context(
        "Relevant permanent conversation archive:",
        chat_history_hits,
        now_epoch=now_epoch,
        now=now,
    )
    working_memory_ctx = _format_document_context(
        "Recent 30-day working-memory scratchpad:",
        working_memory_hits,
        now_epoch=now_epoch,
        now=now,
        prepend_relative_time=True,
    )

    search_ctx = web_search.search_if_relevant(user_text)

    messages: list[ChatMessage] = []
    system_chunks: list[str] = [current_time_ctx, _HENRY_CORE_SYSTEM_PROMPT]
    if rules_text:
        system_chunks.append("Follow these global rules:\n" + rules_text)
    if personal_ctx:
        system_chunks.append(personal_ctx)
    if working_memory_ctx:
        system_chunks.append(working_memory_ctx)
    if chat_history_ctx:
        system_chunks.append(chat_history_ctx)
    if archive_ctx:
        system_chunks.append(archive_ctx)
    if search_ctx:
        system_chunks.append(search_ctx)
    if system_chunks:
        messages.append(ChatMessage(role="system", content="\n\n".join(system_chunks)))
    messages.append(ChatMessage(role="user", content=user_text))

    try:
        reply = llm_manager.complete(messages)
    except Exception as exc:  # noqa: BLE001
        logger.error("LLM complete failed: %s", exc)
        return QueryResponse(reply="", routing=routing, error=f"Model request failed: {exc}")

    clean_reply = reply.strip()
    timestamp = time.time()
    try:
        with ThreadPoolExecutor(max_workers=4) as executor:
            write_futures = [
                executor.submit(
                    memory_mgr.add_chat_history_entry,
                    text=user_text,
                    role="user",
                    metadata={"timestamp": timestamp, "entry_type": "permanent_chat_turn"},
                ),
                executor.submit(
                    memory_mgr.add_chat_history_entry,
                    text=clean_reply,
                    role="assistant",
                    metadata={"timestamp": timestamp, "entry_type": "permanent_chat_turn"},
                ),
                executor.submit(
                    memory_mgr.add_working_memory_entry,
                    text=user_text,
                    role="user",
                    metadata={"timestamp": timestamp, "entry_type": "working_chat_turn"},
                ),
                executor.submit(
                    memory_mgr.add_working_memory_entry,
                    text=clean_reply,
                    role="assistant",
                    metadata={"timestamp": timestamp, "entry_type": "working_chat_turn"},
                ),
            ]
            for write_future in write_futures:
                write_future.result()
        memory_mgr.prune_old_working_memory(_TEMPORARY_CONTEXT_RETENTION_SECONDS)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Chat memory maintenance failed: %s", exc)
    return QueryResponse(reply=clean_reply, routing=routing)


def create_core_app(
    memory_mgr: HenryMemoryManager,
    llm_manager: LLMManager,
    *,
    root: Path,
    rules_text: str,
    web_search: HenryWebSearch,
    on_classified: Any | None = None,
) -> FastAPI:
    """FastAPI router for workers and UI satellites."""
    app = FastAPI(title="Henry Core API", version="0.1.0")
    pending_store: dict[str, dict[str, Any]] = {}

    @app.get("/status", response_model=StatusResponse)
    def get_status() -> StatusResponse:
        return StatusResponse(
            security_flight_mode=SECURITY_FLIGHT_MODE,
            memory_mount_path=str(_memory_mount_path()),
            encrypted_storage_available=memory_mgr.is_encrypted_storage_available,
            archive_ready=memory_mgr.archive_ready,
            chat_history_ready=memory_mgr.chat_history_ready,
            working_memory_ready=memory_mgr.working_memory_ready,
            personal_memory_ready=memory_mgr.personal_memory_ready,
            api_host=_HENRY_API_HOST,
            api_port=_HENRY_API_PORT,
        )

    @app.post("/v1/query", response_model=QueryResponse)
    def query_chat(body: QueryRequest) -> QueryResponse:
        """General chat query (Telegram UI → core)."""
        text = body.text.strip()
        if not text:
            raise HTTPException(status_code=400, detail="text is empty")
        return _run_query(
            text,
            root=root,
            memory_mgr=memory_mgr,
            llm_manager=llm_manager,
            rules_text=rules_text,
            web_search=web_search,
        )

    @app.post("/v1/process", response_model=ClassificationResponse)
    def process_document(body: ProcessRequest) -> ClassificationResponse:
        """Classify document text (worker → core)."""
        if not body.raw_text.strip():
            raise HTTPException(status_code=400, detail="raw_text is empty")

        llm_manager.route_query(body.raw_text)
        meta = classify_document(body.raw_text)

        suffix = Path(body.filename).suffix or ".pdf"
        proposed = HenryFileManager._build_smart_filename(meta, suffix) or body.filename

        pending_id = (body.pending_id or "").strip() or uuid.uuid4().hex[:12]
        pending_store[pending_id] = {
            "source": body.source,
            "filename": body.filename,
            "raw_text": body.raw_text,
            "classification": meta,
            "proposed_name": proposed,
        }

        response = ClassificationResponse(
            pending_id=pending_id,
            category=meta.get("category", "Unknown"),
            document_type=meta.get("document_type", "Unknown"),
            provider=meta.get("provider", "") or "",
            proposed_name=proposed,
            grouping_suggestion=meta.get("grouping_special") or "",
        )

        if on_classified is not None:
            try:
                on_classified({
                    "file": body.filename,
                    "pending_id": pending_id,
                    "classification": meta,
                    "proposed_name": proposed,
                    "status": "awaiting_confirmation",
                })
            except Exception as exc:  # noqa: BLE001
                logger.warning("on_classified hook failed: %s", exc)

        return response

    @app.post("/v1/archive/confirm")
    def archive_confirm(cmd: ArchiveCommand) -> dict[str, Any]:
        """Persist approved document text and command the worker to archive."""
        item = pending_store.get(cmd.pending_id)
        if item is None:
            raise HTTPException(status_code=404, detail=f"Unknown pending_id: {cmd.pending_id}")

        destination = (cmd.user_destination or "").strip()

        if cmd.action == "edit":
            meta_changed = bool(cmd.metadata_override)
            if not meta_changed and not destination:
                raise HTTPException(
                    status_code=400,
                    detail="action 'edit' requires metadata_override and/or non-empty user_destination",
                )

        meta: dict[str, str] = dict(item["classification"])
        if cmd.metadata_override:
            meta.update(cmd.metadata_override)
        if destination:
            meta["user_destination"] = destination

        filename: str = item["filename"]
        text: str = item["raw_text"]

        memory_saved = False
        if SECURITY_FLIGHT_MODE:
            memory_status = "skipped_flight_mode"
        elif not memory_mgr.archive_ready:
            memory_status = "archive_unavailable"
        else:
            try:
                source_path = Path(filename).expanduser()
                if not source_path.is_absolute():
                    source_path = (root / filename).resolve(strict=False)
                memory_mgr.archive_add_document(
                    text=text,
                    absolute_file_path=source_path,
                    provider=meta.get("provider", "") or "Unknown",
                    category=meta.get("category", "Unknown") or "Unknown",
                    document_type=meta.get("document_type", "Unknown") or "Unknown",
                )
                memory_saved = True
                memory_status = "ingested"
            except Exception as exc:  # noqa: BLE001
                memory_status = f"ingest_failed: {exc}"

        worker_status = "not_called"
        worker_result: dict[str, Any] | None = None
        try:
            worker_payload = {
                "pending_id": cmd.pending_id,
                "action": cmd.action,
                "classification": meta,
            }
            with httpx.Client(timeout=60.0) as client:
                worker_resp = client.post(
                    f"{_WORKER_API_URL}/v1/archive/execute",
                    json=worker_payload,
                )
                worker_resp.raise_for_status()
                worker_result = worker_resp.json()
                worker_status = "archived"
        except httpx.ConnectError:
            worker_status = "worker_offline"
            logger.warning("Document worker unreachable at %s", _WORKER_API_URL)
        except Exception as exc:  # noqa: BLE001
            worker_status = f"worker_error: {exc}"
            logger.warning("Worker archive execute failed: %s", exc)

        pending_store.pop(cmd.pending_id, None)

        return {
            "pending_id": cmd.pending_id,
            "action": cmd.action,
            "memory_saved": memory_saved,
            "memory_status": memory_status,
            "worker_status": worker_status,
            "worker_result": worker_result,
            "classification": meta,
        }

    return app


def check_dependencies() -> None:
    """Verify Core API dependencies (no Telegram / OCR worker deps)."""
    missing: list[str] = []
    for package, import_name in [
        ("fastapi", "fastapi"),
        ("uvicorn", "uvicorn"),
        ("httpx", "httpx"),
        ("psutil", "psutil"),
        ("langchain-chroma", "langchain_chroma"),
        ("langchain-community", "langchain_community"),
        ("langchain-ollama", "langchain_ollama"),
        ("tavily-python", "tavily"),
    ]:
        try:
            __import__(import_name)
        except ImportError:
            missing.append(package)
    if missing:
        print(
            f"Missing required dependencies: {', '.join(missing)}\n"
            f"Install them with:  pip install {' '.join(missing)}",
            file=sys.stderr,
        )
        sys.exit(1)


def _run_smoke_test(root: Path) -> int:
    rules_text = _load_global_rules(root / "config" / "global_rules.yaml")
    llm = build_llm_provider()
    messages: list[ChatMessage] = []
    if rules_text:
        messages.append(
            ChatMessage(role="system", content="Follow these global rules:\n" + rules_text)
        )
    messages.append(
        ChatMessage(role="user", content="Reply with exactly: Henry scaffold OK")
    )
    try:
        reply = llm.complete(messages)
    except Exception as exc:  # noqa: BLE001
        print(f"Henry: LLM smoke test failed ({exc}). Is Ollama reachable?", file=sys.stderr)
        return 1
    print(reply.strip())
    return 0


def main() -> int:
    check_dependencies()
    root = Path(__file__).resolve().parent

    if os.environ.get("HENRY_LLM_SMOKE") == "1":
        return _run_smoke_test(root)

    global SECURITY_FLIGHT_MODE
    SECURITY_FLIGHT_MODE = _compute_security_flight_mode()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    memory_mgr = HenryMemoryManager()
    llm_manager = LLMManager.from_project_root(root, memory_manager=memory_mgr)
    rules_text = _load_global_rules(root / "config" / "global_rules.yaml")
    web_search = HenryWebSearch()

    app = create_core_app(
        memory_mgr,
        llm_manager,
        root=root,
        rules_text=rules_text,
        web_search=web_search,
        on_classified=_notify_ui_proposal,
    )

    import uvicorn

    reclaim_tcp_listen_port(_HENRY_API_PORT, role="core")

    print(f"Henry Core API: http://{_HENRY_API_HOST}:{_HENRY_API_PORT}", flush=True)
    print(f"  UI satellite expected at: {_UI_API_URL}", flush=True)
    print(f"  Document worker expected at: {_WORKER_API_URL}", flush=True)

    uvicorn.run(app, host=_HENRY_API_HOST, port=_HENRY_API_PORT, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
