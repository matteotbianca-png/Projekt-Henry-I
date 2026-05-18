"""Henry Telegram UI satellite — outbound Core API client + inbound proposal notifications."""

from __future__ import annotations

print("!!! HENRY IS LIVE !!! Telegram UI entrypoint reached", flush=True)

import asyncio
import contextlib
import logging
import os
import re
import signal
import sys
import threading
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI
from pydantic import BaseModel, Field
from telegram import ForceReply, InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.error import TelegramError
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from core.env import load_project_env

logging.getLogger("httpx").setLevel(logging.WARNING)

load_project_env()

logger = logging.getLogger(__name__)

_CORE_API_URL = os.environ.get("HENRY_CORE_API_URL", "http://127.0.0.1:8000").rstrip("/")
_UI_API_HOST = os.environ.get("HENRY_UI_API_HOST", "127.0.0.1")
_UI_API_PORT = int(os.environ.get("HENRY_UI_API_PORT", "8002"))
_TELEGRAM_POLL_INTERVAL = float(os.environ.get("HENRY_TELEGRAM_POLL_INTERVAL", "0.0"))
_TELEGRAM_LONG_POLL_TIMEOUT = int(os.environ.get("HENRY_TELEGRAM_LONG_POLL_TIMEOUT", "50"))


class ProposalNotifyPayload(BaseModel):
    """Inbound notification from Core when a document is classified."""

    file: str
    pending_id: str
    classification: dict[str, str] = Field(default_factory=dict)
    proposed_name: str = ""
    status: str = "awaiting_confirmation"
    error: str | None = None


class SweepSummaryPayload(BaseModel):
    """Inbound notification from Core when lazy archive self-cleaning runs."""

    trigger_file: str = ""
    archived_to: str = ""
    result: dict[str, Any] = Field(default_factory=dict)


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
        force=True,
    )


def _check_telegram_dependencies() -> None:
    try:
        __import__("telegram")
    except ImportError:
        print(
            "Missing python-telegram-bot. Install with: pip install python-telegram-bot",
            file=sys.stderr,
        )
        sys.exit(1)


class TelegramUISatellite:
    """Telegram bot + inbound notification API for document proposals."""

    def __init__(self) -> None:
        token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
        auth_raw = os.environ.get("AUTHORIZED_USER_ID", "").strip()
        if not token:
            raise RuntimeError("TELEGRAM_BOT_TOKEN is not set.")
        if not auth_raw:
            raise RuntimeError("AUTHORIZED_USER_ID is not set.")
        try:
            self.authorized_user_id = int(auth_raw)
        except ValueError as exc:
            raise RuntimeError("AUTHORIZED_USER_ID must be an integer.") from exc

        self.token = token
        self.bot_app_ref: list[Application] = []
        self.event_loop: asyncio.AbstractEventLoop | None = None
        self.awaiting_correction: dict[int, str] = {}
        self.reply_prompt_documents: dict[int, str] = {}

    @staticmethod
    def _document_id_from_prompt_text(text: str | None) -> str:
        if not text:
            return ""
        match = re.search(r"Document ID:\s*([A-Za-z0-9_-]+)", text)
        return match.group(1).strip() if match else ""

    async def _send_force_reply_prompt(
        self,
        *,
        document_id: str,
        reason: str,
    ) -> None:
        if not self.bot_app_ref:
            return
        prompt = (
            f"{reason}\n\n"
            f"Document ID: {document_id}\n"
            "Reply to this message with the correction text.\n\n"
            "Examples:\n"
            "date should be 31.12.2025\n"
            "provider should be AcmeCorp\n"
            "this is a contract"
        )
        try:
            sent = await self.bot_app_ref[0].bot.send_message(
                chat_id=self.authorized_user_id,
                text=prompt,
                reply_markup=ForceReply(
                    selective=True,
                    input_field_placeholder="date should be 31.12.2025",
                ),
            )
        except TelegramError as exc:
            logger.warning("Could not send ForceReply correction prompt: %s", exc)
            return
        self.reply_prompt_documents[sent.message_id] = document_id

    async def send_proposal(self, entry: dict[str, Any]) -> None:
        if not self.bot_app_ref:
            return

        if entry.get("error"):
            if entry.get("error") == "no_category":
                try:
                    await self.bot_app_ref[0].bot.send_message(
                        chat_id=self.authorized_user_id,
                        text=(
                            "\u26a0\ufe0f Henry could not classify:\n"
                            f"{entry.get('file', '?')}\n\n"
                            "Moved to manual_review."
                        ),
                    )
                except TelegramError:
                    pass
            return

        pending_id = entry.get("pending_id")
        if not pending_id:
            return

        meta = entry.get("classification") or {}
        category = meta.get("category", "Unknown")
        doc_type = meta.get("document_type", "Unknown")
        provider = meta.get("provider", "") or "\u2014"
        proposed_name = entry.get("proposed_name", entry.get("file", "?"))
        group_hint = (entry.get("classification") or {}).get("grouping_special", "").strip()

        text = (
            "\U0001F4C4 Document Detected\n\n"
            f"File: {entry.get('file', '?')}\n"
            f"Proposed Name: {proposed_name}\n"
            f"Category: {category}\n"
            f"Type: {doc_type}\n"
            f"Provider: {provider}"
        )
        if group_hint:
            text += f"\nSuggested grouping (optional): {group_hint}"
        text += f"\nDocument ID: {pending_id}"
        keyboard = InlineKeyboardMarkup([
            [
                InlineKeyboardButton(
                    "\u2705 Best\u00e4tigen",
                    callback_data=f"confirm:{pending_id}",
                ),
                InlineKeyboardButton(
                    "\u274c Korrigieren",
                    callback_data=f"edit:{pending_id}",
                ),
            ]
        ])
        try:
            await self.bot_app_ref[0].bot.send_message(
                chat_id=self.authorized_user_id,
                text=text,
                reply_markup=keyboard,
            )
        except TelegramError as exc:
            logger.warning("Could not send proposal: %s", exc)
            return
        await self._send_force_reply_prompt(
            document_id=pending_id,
            reason="Document correction prompt",
        )

    async def send_sweep_summary(self, entry: dict[str, Any]) -> None:
        """Send a concise lazy archive maintenance summary to the authorized user."""
        if not self.bot_app_ref:
            return

        result = entry.get("result") or {}
        moved = int(result.get("moved") or 0)
        purged = int(result.get("purged") or 0)
        vector_updates = int(result.get("vector_updates") or 0)
        threshold = int(result.get("threshold") or 0)

        text = (
            "Archive Self-Cleaning Sweep\n\n"
            f"Trigger: {entry.get('trigger_file') or 'archive write'}\n"
            f"Moved files: {moved}\n"
            f"Updated vector paths: {vector_updates}\n"
            f"Removed empty folders: {purged}"
        )
        if threshold:
            text += f"\nThreshold: every {threshold} archive writes"

        archived_to = str(entry.get("archived_to") or "").strip()
        if archived_to:
            text += f"\nLatest archive path: {archived_to}"

        try:
            await self.bot_app_ref[0].bot.send_message(
                chat_id=self.authorized_user_id,
                text=text,
            )
        except TelegramError as exc:
            logger.warning("Could not send sweep summary: %s", exc)

    async def _post_archive_confirm(self, payload: dict[str, Any]) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(f"{_CORE_API_URL}/v1/archive/confirm", json=payload)
            resp.raise_for_status()
            return resp.json()

    async def _post_query(self, text: str) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=300.0) as client:
            resp = await client.post(f"{_CORE_API_URL}/v1/query", json={"text": text})
            resp.raise_for_status()
            return resp.json()

    async def _post_query_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=300.0) as client:
            resp = await client.post(f"{_CORE_API_URL}/v1/query", json=payload)
            resp.raise_for_status()
            return resp.json()

    async def _reset_telegram_transport(self, application: Application) -> None:
        """Clear webhook mode and drop queued updates before long-polling."""
        bot = application.bot
        try:
            deleted = await bot.delete_webhook(drop_pending_updates=True)
            if deleted:
                logger.info("Telegram: webhook removed; pending updates dropped.")
            else:
                logger.info("Telegram: pending updates dropped (no webhook was active).")
        except TelegramError as exc:
            logger.warning("Telegram transport reset failed: %s", exc)

    async def post_init(self, application: Application) -> None:
        await self._reset_telegram_transport(application)

        self.bot_app_ref.append(application)
        self.event_loop = asyncio.get_running_loop()
        print(
            f"Henry Telegram UI: polling active (Core API: {_CORE_API_URL})",
            flush=True,
        )

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                status_resp = await client.get(f"{_CORE_API_URL}/status")
                status_resp.raise_for_status()
                status = status_resp.json()
            if not status.get("encrypted_storage_available"):
                await application.bot.send_message(
                    chat_id=self.authorized_user_id,
                    text=(
                        "\u26a0\ufe0f Security Alert: Encrypted volume not found.\n\n"
                        "Document memory is disabled to protect your privacy.\n"
                        "Please mount HenryData and restart the Core."
                    ),
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not fetch Core status at startup: %s", exc)

    async def handle_start(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        _ = context
        if update.effective_user is None or update.message is None:
            return
        if update.effective_user.id != self.authorized_user_id:
            return
        await update.message.reply_text(
            "Henry is online.\n\n"
            "Send a message to chat, or drop documents in the inbox "
            "(document worker + Core must be running)."
        )

    async def handle_callback(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        _ = context
        query = update.callback_query
        if query is None:
            return
        await query.answer()

        user = query.from_user
        if user is None or user.id != self.authorized_user_id:
            return

        data = query.data or ""

        if data.startswith("confirm_") or data.startswith("confirm:"):
            pending_id = data.split("_", 1)[1] if data.startswith("confirm_") else data.split(":", 1)[1]
            try:
                result = await self._post_archive_confirm(
                    {"pending_id": pending_id, "action": "confirm"},
                )
            except Exception as exc:  # noqa: BLE001
                await query.edit_message_text(f"\u26a0\ufe0f Could not archive: {exc}")
                return

            if result.get("error"):
                await query.edit_message_text(f"\u26a0\ufe0f Could not archive: {result['error']}")
                return

            worker_result = result.get("worker_result") or {}
            if result.get("worker_status") != "archived" or worker_result.get("error"):
                err = worker_result.get("error") or result.get("worker_status", "unknown")
                await query.edit_message_text(f"\u26a0\ufe0f Could not archive: {err}")
            else:
                meta = result.get("classification") or {}
                cat = meta.get("category", "?")
                dest = worker_result.get("archived_to", "?")
                await query.edit_message_text(
                    f"\u2705 Archiviert in {cat}\n\n"
                    f"Type: {meta.get('document_type', '?')}\n"
                    f"Path: {dest}"
                )
            return

        if data.startswith("edit_") or data.startswith("edit:"):
            pending_id = data.split("_", 1)[1] if data.startswith("edit_") else data.split(":", 1)[1]
            self.awaiting_correction[self.authorized_user_id] = pending_id
            await self._send_force_reply_prompt(
                document_id=pending_id,
                reason="Correction mode",
            )
            await query.edit_message_reply_markup(reply_markup=None)

    async def handle_text(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        _ = context
        if update.effective_user is None or update.message is None:
            return
        user_text = (update.message.text or "").strip()
        if not user_text:
            return
        uid = update.effective_user.id

        if uid != self.authorized_user_id:
            logger.info("Ignored message from unauthorized user_id=%s", uid)
            return

        reply_doc_id = ""
        reply_to_text = ""
        if update.message.reply_to_message is not None:
            reply_msg = update.message.reply_to_message
            reply_to_text = reply_msg.text or reply_msg.caption or ""
            reply_doc_id = self.reply_prompt_documents.get(reply_msg.message_id, "")
            if not reply_doc_id:
                reply_doc_id = self._document_id_from_prompt_text(reply_to_text)

        if reply_doc_id:
            self.awaiting_correction.pop(uid, None)
            payload = {
                "text": user_text,
                "reply_to_document_id": reply_doc_id,
                "reply_to_message_text": reply_to_text,
                "source": "telegram_force_reply",
            }
            try:
                result = await self._post_query_payload(payload)
            except Exception as exc:  # noqa: BLE001
                await update.message.reply_text(f"\u26a0\ufe0f Could not submit correction: {exc}")
                return
            if result.get("error"):
                await update.message.reply_text(f"\u26a0\ufe0f Could not submit correction: {result['error']}")
                return
            await update.message.reply_text((result.get("reply") or "Correction received.").strip())
            return

        if uid in self.awaiting_correction:
            pending_id = self.awaiting_correction.pop(uid)
            payload = {
                "text": user_text,
                "reply_to_document_id": pending_id,
                "reply_to_message_text": f"Document ID: {pending_id}",
                "source": "telegram_correction_mode",
            }
            try:
                result = await self._post_query_payload(payload)
            except Exception as exc:  # noqa: BLE001
                await update.message.reply_text(f"\u26a0\ufe0f Could not override: {exc}")
                return

            if result.get("error"):
                await update.message.reply_text(f"\u26a0\ufe0f Could not override: {result['error']}")
                return

            await update.message.reply_text((result.get("reply") or "Correction received.").strip())
            return

        try:
            result = await self._post_query(user_text)
        except httpx.ConnectError:
            await update.message.reply_text(
                f"Henry Core is offline at {_CORE_API_URL}. Start it with: python main.py"
            )
            return
        except Exception as exc:  # noqa: BLE001
            await update.message.reply_text(f"Sorry, the request failed: {exc}")
            return

        if result.get("error"):
            await update.message.reply_text(result["error"])
            return

        await update.message.reply_text((result.get("reply") or "").strip())

    async def handle_repair(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        _ = context
        if update.effective_user is None or update.message is None:
            return
        if update.effective_user.id != self.authorized_user_id:
            return
        await update.message.reply_text(
            "\U0001F527 Archive repair will be exposed on the Core API in a future step."
        )

    async def handle_purge(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        _ = context
        if update.effective_user is None or update.message is None:
            return
        if update.effective_user.id != self.authorized_user_id:
            return
        files_root = os.environ.get("HENRY_FILES_ROOT", "").strip()
        if not files_root:
            await update.message.reply_text("HENRY_FILES_ROOT is not set.")
            return

        import shutil as _shutil

        root = Path(files_root)
        stale_paths = [
            root / "internal" / "chroma_db",
            root / "internal" / "document_memory",
            root / "internal" / "debug_texte",
            root / "internal" / "extracted_texts",
            root / "internal" / "processing",
            root / "internal" / "temp_backup",
            root / "knowledge_base.json",
        ]
        removed: list[str] = []
        for p in stale_paths:
            if not p.exists():
                continue
            try:
                if p.is_dir():
                    _shutil.rmtree(str(p))
                else:
                    p.unlink()
                removed.append(p.name)
            except OSError as exc:
                await update.message.reply_text(f"\u26a0\ufe0f Could not delete {p.name}: {exc}")

        if not removed:
            await update.message.reply_text(
                "\u2705 No stale local data found. Everything is clean."
            )
        else:
            await update.message.reply_text(
                "\u2705 Purged local data:\n"
                + "\n".join(f"  \u2022 {name}" for name in removed)
                + "\n\nAll sensitive data now lives on HenryData only."
            )

    def build_telegram_application(self) -> Application:
        application = (
            Application.builder()
            .token(self.token)
            .post_init(self.post_init)
            .build()
        )
        application.add_handler(CommandHandler("start", self.handle_start))
        application.add_handler(CommandHandler("repair", self.handle_repair))
        application.add_handler(CommandHandler("purge", self.handle_purge))
        application.add_handler(CallbackQueryHandler(self.handle_callback))
        application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_text))
        return application

    def notify_proposal_sync(self, entry: dict[str, Any]) -> None:
        """Thread-safe bridge from the inbound FastAPI handler to the bot loop."""
        loop = self.event_loop
        if loop is None or not loop.is_running():
            logger.warning("Telegram loop not ready — dropping proposal for %s", entry.get("file"))
            return
        asyncio.run_coroutine_threadsafe(self.send_proposal(entry), loop)

    def notify_sweep_summary_sync(self, entry: dict[str, Any]) -> None:
        """Thread-safe bridge for lazy archive maintenance summaries."""
        loop = self.event_loop
        if loop is None or not loop.is_running():
            logger.warning("Telegram loop not ready — dropping sweep summary")
            return
        asyncio.run_coroutine_threadsafe(self.send_sweep_summary(entry), loop)


def create_ui_app(satellite: TelegramUISatellite) -> FastAPI:
    app = FastAPI(title="Henry Telegram UI API", version="0.1.0")

    @app.get("/status")
    def ui_status() -> dict[str, Any]:
        return {
            "service": "henry-telegram-ui",
            "core_api_url": _CORE_API_URL,
            "telegram_ready": bool(satellite.bot_app_ref),
        }

    @app.post("/v1/ui/notify_proposal")
    def notify_proposal(body: ProposalNotifyPayload) -> dict[str, str]:
        satellite.notify_proposal_sync(body.model_dump())
        return {"status": "queued", "pending_id": body.pending_id}

    @app.post("/v1/ui/notify_sweep_summary")
    def notify_sweep_summary(body: SweepSummaryPayload) -> dict[str, str]:
        satellite.notify_sweep_summary_sync(body.model_dump())
        return {"status": "queued"}

    return app


def _run_ui_api(satellite: TelegramUISatellite) -> None:
    import uvicorn

    ui_app = create_ui_app(satellite)
    uvicorn.run(
        ui_app,
        host=_UI_API_HOST,
        port=_UI_API_PORT,
        log_level="info",
        log_config=None,
    )


async def _wait_for_shutdown_signal() -> None:
    shutdown = asyncio.Event()
    loop = asyncio.get_running_loop()

    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(sig, shutdown.set)

    await shutdown.wait()


async def run_telegram_ui_service_async() -> None:
    _configure_logging()
    _check_telegram_dependencies()

    satellite = TelegramUISatellite()
    api_thread = threading.Thread(
        target=_run_ui_api,
        args=(satellite,),
        name="henry-ui-api",
        daemon=True,
    )
    api_thread.start()
    print(
        f"Henry Telegram UI API: http://{_UI_API_HOST}:{_UI_API_PORT}",
        flush=True,
    )

    application = satellite.build_telegram_application()
    print(
        "Henry Telegram UI: starting bot polling "
        f"(poll_interval={_TELEGRAM_POLL_INTERVAL}, "
        f"long_poll_timeout={_TELEGRAM_LONG_POLL_TIMEOUT}s).",
        flush=True,
    )

    await application.initialize()
    await satellite.post_init(application)
    if application.updater is None:
        raise RuntimeError("Telegram application has no updater; polling cannot start.")

    try:
        await application.updater.start_polling(
            poll_interval=_TELEGRAM_POLL_INTERVAL,
            timeout=_TELEGRAM_LONG_POLL_TIMEOUT,
            allowed_updates=Update.ALL_TYPES,
            drop_pending_updates=True,
        )
        await application.start()
        await _wait_for_shutdown_signal()
    finally:
        if application.updater.running:
            await application.updater.stop()
        if application.running:
            await application.stop()
        await application.shutdown()


def run_telegram_ui_service() -> None:
    asyncio.run(
        run_telegram_ui_service_async(),
    )


if __name__ == "__main__":
    run_telegram_ui_service()
