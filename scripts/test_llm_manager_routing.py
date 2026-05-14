#!/usr/bin/env python3
"""
Initialize LLMManager and print Presidio-backed routing diagnostics per prompt.

Run from repository root:
  python scripts/test_llm_manager_routing.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Repository root (parent of scripts/)
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.llm_manager import LLMManager  # noqa: E402 — after sys.path


def _print_case(title: str, user_text: str, manager: LLMManager) -> None:
    info = manager.preview_routing(user_text)
    presidio = info.get("presidio") or {}
    privacy = info.get("privacy_impact_effective_by_candidate") or {}
    winner = info.get("winner") or {}

    print(f"\n{'=' * 72}\nCase: {title}\n{'=' * 72}")
    print("User text:\n", user_text[:500] + ("…" if len(user_text) > 500 else ""), sep="")

    print("\n--- Microsoft Presidio (analyzer summary) ---")
    print(
        json.dumps(
            {
                "method": presidio.get("method"),
                "presidio_engine_active": info.get("presidio_engine_active"),
                "hit_count": presidio.get("hit_count"),
                "severity": presidio.get("severity"),
                "severity_effective": presidio.get("severity_effective"),
                "weighted_mass": presidio.get("weighted_mass"),
                "counts_by_type": presidio.get("counts_by_type"),
            },
            indent=2,
        )
    )

    print("\n--- Privacy scores (routing Privacy_Impact from Presidio context) ---")
    print(json.dumps(privacy, indent=2, sort_keys=True))

    print("\n--- Routing decision (scores; higher is better after supervisor blend) ---")
    print(
        json.dumps(
            {
                "formula_scores": info.get("formula_scores"),
                "supervisor_scores": info.get("supervisor_scores"),
                "final_scores": info.get("final_scores"),
            },
            indent=2,
        )
    )

    print("\n--- Final model selected ---")
    print(json.dumps(winner, indent=2))


def main() -> int:
    manager = LLMManager.from_project_root(ROOT)

    cases: list[tuple[str, str]] = [
        (
            "simple_greeting",
            "Hello! Hope you're having a good day.",
        ),
        (
            "complex_coding",
            (
                "I need a production-grade design: implement a distributed tracing pipeline "
                "in Go that ingests OTLP, applies tail-based sampling with configurable rules, "
                "and exports to Tempo and S3 with backpressure handling and idempotent retries."
            ),
        ),
        (
            "fake_name_and_iban",
            (
                "Please confirm payment details for Max Mustermann; "
                "the IBAN is DE89370400440532013000 and the reference is INV-2026-042."
            ),
        ),
    ]

    for title, text in cases:
        _print_case(title, text, manager)

    print(f"\n{'=' * 72}\nDone.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
