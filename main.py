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
    from telegram import Update
    from telegram.error import TelegramError
    from telegram.ext import Application, ContextTypes, MessageHandler, filters

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

    async def post_init(application: Application) -> None:
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

    async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if update.effective_user is None or update.message is None:
            return
        user_text = (update.message.text or "").strip()
        if not user_text:
            return

        if update.effective_user.id != authorized_user_id:
            print(
                f"[non-authorized user_id={update.effective_user.id}] -> [ignored] -> [not AUTHORIZED_USER_ID]",
                flush=True,
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

        messages: list[ChatMessage] = []
        system_chunks: list[str] = []
        if rules_text:
            system_chunks.append("Follow these global rules:\n" + rules_text)
        if personal_ctx:
            system_chunks.append(personal_ctx)
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

    application = (
        Application.builder()
        .token(token)
        .post_init(post_init)
        .build()
    )
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    print("Henry: Telegram bot polling (authorized user only).", flush=True)
    application.run_polling(allowed_updates=Update.ALL_TYPES)
    return 0


def main() -> int:
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
