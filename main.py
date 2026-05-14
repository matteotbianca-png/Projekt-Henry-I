"""Henry entrypoint — Telegram bot with routed LLM, or legacy scaffold / smoke test."""

from __future__ import annotations

from dotenv import load_dotenv

load_dotenv()

import asyncio
import json
import os
import sys
from pathlib import Path

import yaml

from core.llm.base import ChatMessage
from core.llm.factory import build_llm_provider
from core.llm_manager import LLMManager
from core.memory.factory import build_memory_store
from core.memory_manager import HenryDualMemoryManager
from tools.file_manager import CATEGORY_FOLDERS, HenryFileManager
from tools.web_search import HenryWebSearch


def check_dependencies() -> None:
    """Verify that required third-party packages are installed."""
    missing: list[str] = []
    for package, import_name in [
        ("langchain-chroma", "langchain_chroma"),
        ("langchain-community", "langchain_community"),
        ("langchain-ollama", "langchain_ollama"),
        ("python-telegram-bot", "telegram"),
        ("tavily-python", "tavily"),
        ("pytesseract", "pytesseract"),
        ("pdf2image", "pdf2image"),
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


def _run_telegram_bot(root: Path) -> int:
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

    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    auth_raw = os.environ.get("AUTHORIZED_USER_ID", "").strip()
    if not token:
        print("TELEGRAM_BOT_TOKEN is not set.", file=sys.stderr)
        return 1
    if not auth_raw:
        print("AUTHORIZED_USER_ID is not set.", file=sys.stderr)
        return 1
    try:
        authorized_user_id = int(auth_raw)
    except ValueError:
        print("AUTHORIZED_USER_ID must be an integer Telegram user id.", file=sys.stderr)
        return 1

    rules_text = _load_global_rules(root / "config" / "global_rules.yaml")
    memory_mgr = HenryDualMemoryManager()
    llm_manager = LLMManager.from_project_root(root, memory_manager=memory_mgr)
    web_search = HenryWebSearch()

    bot_app_ref: list[Application] = []
    awaiting_correction: dict[int, str] = {}

    async def _send_proposal(entry: dict) -> None:
        """Send a classification proposal with inline buttons to Telegram."""
        if not bot_app_ref:
            return
        if entry.get("error"):
            if entry.get("error") == "no_category":
                try:
                    await bot_app_ref[0].bot.send_message(
                        chat_id=authorized_user_id,
                        text=(
                            f"\u26a0\ufe0f Henry could not classify:\n"
                            f"{entry.get('file', '?')}\n\n"
                            f"Moved to manual_review."
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

        text = (
            f"\U0001F4C4 Document Detected\n\n"
            f"File: {entry.get('file', '?')}\n"
            f"Proposed Name: {proposed_name}\n"
            f"Category: {CATEGORY_FOLDERS.get(category, category)}\n"
            f"Type: {doc_type}\n"
            f"Provider: {provider}"
        )
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
            await bot_app_ref[0].bot.send_message(
                chat_id=authorized_user_id,
                text=text,
                reply_markup=keyboard,
            )
        except TelegramError as exc:
            print(f"Henry: could not send proposal: {exc}", file=sys.stderr, flush=True)

    file_manager = HenryFileManager(
        on_file_processed=_send_proposal,
        memory_manager=memory_mgr,
    )
    file_manager.start_watching()
    print("Henry File Watcher active.", flush=True)

    async def post_init(application: Application) -> None:
        bot_app_ref.append(application)
        file_manager.set_event_loop(asyncio.get_running_loop())

        if memory_mgr.is_encrypted_storage_available:
            return
        try:
            await application.bot.send_message(
                chat_id=authorized_user_id,
                text=(
                    "Warning: Memory volume not found. Running in volatile mode. "
                    "Please mount HenryData for long-term memory access."
                ),
            )
        except TelegramError as exc:
            print(f"Henry: could not send startup memory warning: {exc}", file=sys.stderr, flush=True)

    # --- Inline-button callback handler ------------------------------------

    async def handle_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        query = update.callback_query
        if query is None:
            return
        await query.answer()

        user = query.from_user
        if user is None or user.id != authorized_user_id:
            return

        data = query.data or ""

        if data.startswith("confirm_"):
            pending_id = data[len("confirm_"):]
            result = await asyncio.to_thread(file_manager.confirm_pending, pending_id)

            if result is None or result.get("error"):
                err = (result or {}).get("error", "file not found")
                await query.edit_message_text(f"\u26a0\ufe0f Could not archive: {err}")
            else:
                meta = result.get("classification") or {}
                cat = meta.get("category", "?")
                dest = result.get("archived_to", "?")
                await query.edit_message_text(
                    f"\u2705 Archiviert in {CATEGORY_FOLDERS.get(cat, cat)}\n\n"
                    f"Type: {meta.get('document_type', '?')}\n"
                    f"Path: {dest}"
                )
            return

        if data.startswith("edit_"):
            pending_id = data[len("edit_"):]
            if pending_id not in file_manager._pending_items:
                await query.edit_message_text("\u26a0\ufe0f This document is no longer pending.")
                return

            awaiting_correction[authorized_user_id] = pending_id
            await query.edit_message_text(
                f"\u270f\ufe0f Correction mode\n\n"
                f"Send the correct category and name, e.g.:\n"
                f'"02_Finanzen: Neuer_Name"\n\n'
                f"Or just send the category to keep the current name:\n"
                f'"03_Versicherung"'
            )

    # --- Text message handler (corrections + normal chat) ------------------

    async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if update.effective_user is None or update.message is None:
            return
        user_text = (update.message.text or "").strip()
        if not user_text:
            return
        uid = update.effective_user.id

        if uid != authorized_user_id:
            print(
                f"[non-authorized user_id={uid}] -> [ignored] -> [not AUTHORIZED_USER_ID]",
                flush=True,
            )
            return

        if uid in awaiting_correction:
            pending_id = awaiting_correction.pop(uid)
            parts = user_text.split(":", 1)
            new_category = parts[0].strip()
            new_name = parts[1].strip() if len(parts) > 1 and parts[1].strip() else None

            result = await asyncio.to_thread(
                file_manager.override_pending, pending_id, new_category, new_name
            )
            if result is None or result.get("error"):
                err = (result or {}).get("error", "file not found")
                await update.message.reply_text(f"\u26a0\ufe0f Could not override: {err}")
            else:
                meta = result.get("classification") or {}
                cat = meta.get("category", "?")
                dest = result.get("archived_to", "?")
                await update.message.reply_text(
                    f"\u2705 Korrigiert & archiviert in {CATEGORY_FOLDERS.get(cat, cat)}\n\n"
                    f"Type: {meta.get('document_type', '?')}\n"
                    f"Path: {dest}"
                )
            return

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
            await update.message.reply_text(
                "A cloud model was selected, but the Presidio PII shield is not active. "
                "Fix Presidio (see project logs) before cloud routing can run safely."
            )
            return

        await asyncio.to_thread(llm_manager.process_memory_intent, user_text)
        personal_ctx = memory_mgr.retrieve_personal_context(user_text)
        search_ctx = await asyncio.to_thread(web_search.search_if_relevant, user_text)

        messages: list[ChatMessage] = []
        system_chunks: list[str] = []
        if rules_text:
            system_chunks.append("Follow these global rules:\n" + rules_text)
        if personal_ctx:
            system_chunks.append(personal_ctx)
        if search_ctx:
            system_chunks.append(search_ctx)
        if system_chunks:
            messages.append(
                ChatMessage(
                    role="system",
                    content="\n\n".join(system_chunks),
                )
            )
        messages.append(ChatMessage(role="user", content=user_text))

        try:
            reply = await asyncio.to_thread(llm_manager.complete, messages)
        except Exception as exc:  # noqa: BLE001 — surface to user and log
            print(f"LLM complete failed: {exc}", file=sys.stderr, flush=True)
            await update.message.reply_text(f"Sorry, the model request failed: {exc}")
            return

        await update.message.reply_text(reply.strip())

    async def handle_repair(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if update.effective_user is None or update.message is None:
            return
        if update.effective_user.id != authorized_user_id:
            return
        await update.message.reply_text("\U0001F527 Running archive repair\u2026")
        result = await asyncio.to_thread(file_manager.repair_knowledge_base)

        fixed = result.get("fixed_entries", 0)
        pruned = result.get("pruned", 0)
        repaired = result.get("repaired", 0)
        skipped = result.get("skipped", 0)
        total = fixed + pruned + repaired

        if total == 0:
            await update.message.reply_text(
                "\u2705 Knowledge base is healthy. Nothing to repair."
            )
            return

        lines = ["\u2705 Repair complete\n"]
        if fixed:
            lines.append(f"Fixed entries: {fixed}")
        if pruned:
            lines.append(f"Pruned stale entries: {pruned}")
        if repaired:
            files_list = "\n".join(f"  \u2022 {f}" for f in result.get("files", []))
            lines.append(f"Re-indexed orphans: {repaired}\n{files_list}")
        if skipped:
            lines.append(f"Skipped (OCR error): {skipped}")
        await update.message.reply_text("\n".join(lines))

    async def post_shutdown(application: Application) -> None:
        file_manager.stop_watching()
        print("Henry File Watcher stopped.", flush=True)

    application = (
        Application.builder()
        .token(token)
        .post_init(post_init)
        .post_shutdown(post_shutdown)
        .build()
    )
    application.add_handler(CommandHandler("repair", handle_repair))
    application.add_handler(CallbackQueryHandler(handle_callback))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    print("Henry: Telegram bot polling (authorized user only).", flush=True)
    application.run_polling(allowed_updates=Update.ALL_TYPES)
    return 0


def main() -> int:
    check_dependencies()
    root = Path(__file__).resolve().parent

    if os.environ.get("TELEGRAM_BOT_TOKEN", "").strip():
        return _run_telegram_bot(root)

    rules_text = _load_global_rules(root / "config" / "global_rules.yaml")

    memory = build_memory_store()

    if not memory.is_available():
        print(
            "Henry: memory path not ready (expected after manual decrypt/mount).",
            file=sys.stderr,
        )
    else:
        memory.put("bootstrap", "Henry scaffold online")

    if os.environ.get("HENRY_LLM_SMOKE") != "1":
        print("Henry: ready (set HENRY_LLM_SMOKE=1 to ping Ollama).")
        return 0

    llm = build_llm_provider()
    messages = []
    if rules_text:
        messages.append(
            ChatMessage(
                role="system",
                content="Follow these global rules:\n" + rules_text,
            )
        )
    messages.append(
        ChatMessage(
            role="user",
            content="Reply with exactly: Henry scaffold OK",
        )
    )
    try:
        reply = llm.complete(messages)
    except Exception as exc:  # noqa: BLE001 — bootstrap should stay informative
        print(f"Henry: LLM smoke test failed ({exc}). Is Ollama reachable?", file=sys.stderr)
        return 1

    print(reply.strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
