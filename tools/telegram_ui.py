"""Henry Telegram UI satellite — outbound Core API client + inbound proposal notifications."""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import threading
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
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
from tools.file_manager import CATEGORY_FOLDERS

logging.getLogger("httpx").setLevel(logging.WARNING)

load_project_env()

logger = logging.getLogger(__name__)

_CORE_API_URL = os.environ.get("HENRY_CORE_API_URL", "http://127.0.0.1:8000").rstrip("/")
_UI_API_HOST = os.environ.get("HENRY_UI_API_HOST", "127.0.0.1")
_UI_API_PORT = int(os.environ.get("HENRY_UI_API_PORT", "8002"))


class ProposalNotifyPayload(BaseModel):
    """Inbound notification from Core when a document is classified."""

    file: str
    pending_id: str
    classification: dict[str, str] = Field(default_factory=dict)
    proposed_name: str = ""
    status: str = "awaiting_confirmation"
    error: str | None = None


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
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
            f"Category: {CATEGORY_FOLDERS.get(category, category)}\n"
            f"Type: {doc_type}\n"
            f"Provider: {provider}"
        )
        if group_hint:
            text += f"\nSuggested sub-folder (optional): {group_hint}"
        keyboard = InlineKeyboardMarkup([
            [
                InlineKeyboardButton(
                    "\u2705 Best\u00e4tigen",
                    callback_data=f"confirm_{pending_id}",
                ),
                InlineKeyboardButton(
                    "\u274c Korrigieren",
                    callback_data=f"edit_{pending_id}",
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

        if data.startswith("confirm_"):
            pending_id = data[len("confirm_"):]
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
                    f"\u2705 Archiviert in {CATEGORY_FOLDERS.get(cat, cat)}\n\n"
                    f"Type: {meta.get('document_type', '?')}\n"
                    f"Path: {dest}"
                )
            return

        if data.startswith("edit_"):
            pending_id = data[len("edit_"):]
            self.awaiting_correction[self.authorized_user_id] = pending_id
            await query.edit_message_text(
                "\u270f\ufe0f Correction mode\n\n"
                "Category + rename:\n"
                "02_Finanzen: Neuer_Dateiname\n\n"
                "Category only:\n"
                "03_Versicherung\n\n"
                "Exact archive path (overrides AI filing):\n"
                "Archiv/04_Arbeit/Contracts/AcmeCorp\n"
                "Archiv/02_Finanzen/2026/03/01"
            )

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

        if uid in self.awaiting_correction:
            pending_id = self.awaiting_correction.pop(uid)
            ut = user_text.strip()

            payload: dict[str, Any] = {
                "pending_id": pending_id,
                "action": "edit",
            }

            if "/" in ut or "\\" in ut or ut.lower().startswith("archiv"):
                payload["user_destination"] = ut
            elif ":" in ut:
                segment = ut.split(":", 1)
                override: dict[str, str] = {"category": segment[0].strip()}
                if segment[1].strip():
                    override["document_type"] = segment[1].strip()
                payload["metadata_override"] = override
            else:
                payload["metadata_override"] = {"category": ut}

            try:
                result = await self._post_archive_confirm(payload)
            except Exception as exc:  # noqa: BLE001
                await update.message.reply_text(f"\u26a0\ufe0f Could not override: {exc}")
                return

            if result.get("error"):
                await update.message.reply_text(f"\u26a0\ufe0f Could not override: {result['error']}")
                return

            worker_result = result.get("worker_result") or {}
            if result.get("worker_status") != "archived" or worker_result.get("error"):
                err = worker_result.get("error") or result.get("worker_status", "unknown")
                await update.message.reply_text(f"\u26a0\ufe0f Could not override: {err}")
            else:
                meta = result.get("classification") or {}
                cat = meta.get("category", "?")
                dest = worker_result.get("archived_to", "?")
                await update.message.reply_text(
                    f"\u2705 Korrigiert & archiviert in {CATEGORY_FOLDERS.get(cat, cat)}\n\n"
                    f"Type: {meta.get('document_type', '?')}\n"
                    f"Path: {dest}"
                )
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

    return app


def _run_ui_api(satellite: TelegramUISatellite) -> None:
    import uvicorn

    ui_app = create_ui_app(satellite)
    uvicorn.run(ui_app, host=_UI_API_HOST, port=_UI_API_PORT, log_level="info")


def run_telegram_ui_service() -> None:
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
        "(webhook cleared, pending updates dropped).",
        flush=True,
    )
    application.run_polling(
        allowed_updates=Update.ALL_TYPES,
        drop_pending_updates=True,
    )


if __name__ == "__main__":
    run_telegram_ui_service()
