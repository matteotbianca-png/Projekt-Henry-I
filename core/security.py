"""Shared local filesystem safety helpers."""

from __future__ import annotations

from pathlib import Path


def validate_safe_path(path: str | Path, root: str | Path) -> Path:
    """Return resolved *path* only when it stays inside resolved *root*.

    Raises ``ValueError`` for path traversal or attempts to write outside the
    configured Henry storage root. The target does not need to exist yet.
    """
    root_path = Path(root).expanduser().resolve()
    target_path = Path(path).expanduser().resolve(strict=False)
    try:
        target_path.relative_to(root_path)
    except ValueError:
        raise ValueError(f"Unsafe path outside storage root: {target_path}") from None
    return target_path
