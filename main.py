"""Henry entrypoint — wires provider-agnostic LLM and memory from config/ENV."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv

from core.llm.base import ChatMessage
from core.llm.factory import build_llm_provider
from core.memory.factory import build_memory_store


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


def main() -> int:
    root = Path(__file__).resolve().parent
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
    load_dotenv()
    raise SystemExit(main())
