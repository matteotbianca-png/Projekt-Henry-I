"""In-memory background task queue for heavy Henry workloads."""

from __future__ import annotations

import asyncio
import inspect
import logging
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Literal

logger = logging.getLogger(__name__)

TaskStatus = Literal["queued", "running", "completed", "failed"]
TaskProcessor = Callable[["TaskPayload"], Any]


@dataclass(frozen=True)
class TaskPayload:
    task_id: str
    client_id: str
    intent: str
    payload_data: dict[str, Any]
    created_at: float


@dataclass
class TaskRecord:
    payload: TaskPayload
    status: TaskStatus
    result: Any | None = None
    error: str | None = None
    updated_at: float = 0.0


class QueueManager:
    """Owns a single asyncio.Queue and sequential background worker."""

    def __init__(self) -> None:
        self._loop: asyncio.AbstractEventLoop | None = None
        self._queue: asyncio.Queue[TaskPayload] | None = None
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._lock = threading.Lock()
        self._records: dict[str, TaskRecord] = {}
        self._processor: TaskProcessor | None = None

    def start_background_worker(self, processor: TaskProcessor | None = None) -> None:
        """Start the worker loop if needed and update the task processor."""
        if processor is not None:
            self._processor = processor
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._ready.clear()
            self._thread = threading.Thread(
                target=self._run_loop,
                name="henry-background-task-worker",
                daemon=True,
            )
            self._thread.start()
        self._ready.wait(timeout=5.0)

    def enqueue(
        self,
        *,
        client_id: str,
        intent: str,
        payload_data: dict[str, Any],
        task_id: str | None = None,
    ) -> TaskPayload:
        """Submit a task from any thread without blocking the caller."""
        self.start_background_worker()
        if self._loop is None or self._queue is None:
            raise RuntimeError("background queue is not ready")

        payload = TaskPayload(
            task_id=task_id or uuid.uuid4().hex,
            client_id=client_id,
            intent=intent,
            payload_data=dict(payload_data),
            created_at=time.time(),
        )
        self._records[payload.task_id] = TaskRecord(
            payload=payload,
            status="queued",
            updated_at=payload.created_at,
        )
        future = asyncio.run_coroutine_threadsafe(self._queue.put(payload), self._loop)
        future.result(timeout=2.0)
        logger.info("Queued background task id=%s intent=%s client=%s", payload.task_id, intent, client_id)
        return payload

    def get_status(self, task_id: str) -> dict[str, Any] | None:
        record = self._records.get(task_id)
        if record is None:
            return None
        return {
            "task_id": record.payload.task_id,
            "client_id": record.payload.client_id,
            "intent": record.payload.intent,
            "status": record.status,
            "result": record.result,
            "error": record.error,
            "created_at": record.payload.created_at,
            "updated_at": record.updated_at,
        }

    def _run_loop(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        self._queue = asyncio.Queue()
        self._ready.set()
        loop.create_task(self._worker_loop())
        loop.run_forever()

    async def _worker_loop(self) -> None:
        if self._queue is None:
            raise RuntimeError("background queue missing")
        while True:
            payload = await self._queue.get()
            await self._process_payload(payload)
            self._queue.task_done()

    async def _process_payload(self, payload: TaskPayload) -> None:
        record = self._records[payload.task_id]
        record.status = "running"
        record.updated_at = time.time()
        processor = self._processor
        if processor is None:
            record.status = "failed"
            record.error = "no background task processor registered"
            record.updated_at = time.time()
            logger.warning("Background task %s failed: no processor registered", payload.task_id)
            return

        try:
            if inspect.iscoroutinefunction(processor):
                result = await processor(payload)
            else:
                result = await asyncio.to_thread(processor, payload)
            record.status = "completed"
            record.result = result
            record.updated_at = time.time()
            logger.info("Background task completed id=%s intent=%s", payload.task_id, payload.intent)
        except Exception as exc:  # noqa: BLE001
            record.status = "failed"
            record.error = str(exc)
            record.updated_at = time.time()
            logger.exception("Background task failed id=%s intent=%s", payload.task_id, payload.intent)


_DEFAULT_QUEUE_MANAGER = QueueManager()


def get_default_queue_manager() -> QueueManager:
    return _DEFAULT_QUEUE_MANAGER


def start_background_worker(processor: TaskProcessor | None = None) -> None:
    _DEFAULT_QUEUE_MANAGER.start_background_worker(processor)
