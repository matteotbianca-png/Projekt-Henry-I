"""Load project configuration from the repository root."""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

# Repository root (parent of `core/`)
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_project_env() -> Path:
    """Load `.env` from the repo root regardless of the shell's current working directory."""
    load_dotenv(PROJECT_ROOT / ".env")
    return PROJECT_ROOT
