"""Recover Henry's TCP LISTEN sockets when a previous interpreter was orphaned."""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import time
from typing import Iterable, Literal

import psutil

logger = logging.getLogger(__name__)

HenryListenRole = Literal["core", "worker"]


def reclaim_tcp_listen_port(port: int, *, role: HenryListenRole) -> None:
    """Stop orphaned Henry listeners so Uvicorn can bind.

    - **Discovery:** ``psutil`` first (no ``lsof`` required); results are merged with
      ``lsof`` when present so permissive kernels / sandboxes still see the socket.
    - **Safety:** Only processes with the **same effective UID** as this interpreter
      and an argv pointing at Henry's ``main.py`` (core) or ``file_manager.py`` (worker)
      are signalled — not arbitrary services on shared ports.

    Disable: ``HENRY_RECLAIM_PORTS=0`` / ``false`` / ``no`` / ``off``.
    """
    flag = os.environ.get("HENRY_RECLAIM_PORTS", "1").strip().lower()
    if flag in ("0", "false", "no", "off"):
        return

    my_pid = os.getpid()
    my_euid = os.geteuid()

    listeners = _pids_via_psutil(port) | _pids_via_lsof(port)
    if not listeners:
        return

    try:
        current = psutil.Process()
    except psutil.Error as exc:
        logger.warning("Cannot inspect owning process (%s); skipping reclaim.", exc)
        return

    for pid in sorted(listeners):
        if pid <= 1 or pid == my_pid:
            continue

        try:
            proc = psutil.Process(pid)
            ok, detail = _should_reclaim(proc, current, role, my_euid)
        except (psutil.NoSuchProcess, psutil.AccessDenied) as exc:
            logger.debug("Port %s: skip PID %s (%s)", port, pid, exc)
            continue

        if not ok:
            logger.info(
                "Port %s held by PID %s — skipping reclaim (%s).",
                port,
                pid,
                detail,
            )
            continue

        cmd = _format_cmd(proc)
        logger.warning(
            "Port %s: reclaiming Henry %s PID %s (%s)",
            port,
            role,
            pid,
            cmd or "(no argv)",
        )
        _terminate_pid(pid)

    time.sleep(0.2)


def _pids_via_psutil(port: int) -> set[int]:
    pids: set[int] = set()
    try:
        conns = psutil.net_connections(kind="inet")
    except (psutil.AccessDenied, PermissionError) as exc:
        logger.warning("psutil socket scan limited: %s", exc)
        return pids

    for conn in conns:
        if conn.pid is None or conn.status != psutil.CONN_LISTEN:
            continue
        lip = conn.laddr
        if getattr(lip, "port", None) != port:
            continue
        pids.add(conn.pid)

    return pids


def _pids_via_lsof(port: int) -> set[int]:
    try:
        completed = subprocess.run(
            ["lsof", "-t", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN"],
            capture_output=True,
            text=True,
            timeout=6,
            check=False,
        )
    except FileNotFoundError:
        return set()
    except subprocess.TimeoutExpired:
        logger.warning("lsof timed out scanning port %s.", port)
        return set()

    text = (completed.stdout or "").strip()
    if not text:
        return set()

    pids: set[int] = set()
    for line in text.splitlines():
        token = line.strip().split(None, 1)[0]
        try:
            pids.add(int(token))
        except ValueError:
            continue
    return pids


def _effective_uid(proc: psutil.Process) -> int | None:
    try:
        return int(proc.uids().real)
    except (AttributeError, NotImplementedError, TypeError, ValueError):
        return None


def _should_reclaim(
    proc: psutil.Process,
    current: psutil.Process,
    role: HenryListenRole,
    my_euid: int,
) -> tuple[bool, str]:
    puid = _effective_uid(proc)

    if puid is None:
        try:
            if proc.username() != current.username():
                return False, "different POSIX user account"
        except (psutil.Error, KeyError, OSError):
            return False, "cannot resolve process owner"
    elif puid != my_euid:
        return False, f"different EUID ({puid} != {my_euid}; not reclaiming)"

    cmdline = _safe_cmdline(proc)

    if role == "core":
        if _argv_names_script(cmdline, "telegram_ui.py"):
            return False, "Telegram satellite (different program)"
        if not _argv_names_script(cmdline, "main.py"):
            return False, "not running main.py — leave listener alone"

        return True, "matching Henry Core"

    if role == "worker":
        if not _argv_names_script(cmdline, "file_manager.py"):
            return False, "not running file_manager.py — leave listener alone"
        return True, "matching Henry Worker"

    return False, "unknown role"


def _safe_cmdline(proc: psutil.Process) -> list[str]:
    try:
        cli = proc.cmdline()
        if cli:
            return cli
    except (psutil.AccessDenied, psutil.NoSuchProcess):
        pass
    try:
        return [proc.name()]
    except (psutil.AccessDenied, psutil.NoSuchProcess):
        return []


def _argv_names_script(argv: Iterable[str], filename: str) -> bool:
    """True iff some argv token resolves to exactly *filename* (not ``xmain.py``)."""
    target = filename.lower()
    for tok in argv:
        path = tok.replace("\\", "/").strip()
        if not path:
            continue
        seg = path.rstrip("/").split("/")[-1].lower()
        if seg == target:
            return True
    return False


def _format_cmd(proc: psutil.Process, max_len: int = 120) -> str:
    cli = " ".join(_safe_cmdline(proc))
    if len(cli) <= max_len:
        return cli
    return cli[: max_len - 3] + "..."


def _terminate_pid(pid: int) -> None:
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    except PermissionError:
        logger.error(
            "Permission denied signalling PID %s — cannot reclaim (try same user)",
            pid,
        )
        return

    for _ in range(40):
        time.sleep(0.05)
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return

    logger.warning("PID %s still alive after SIGTERM — SIGKILL.", pid)
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    except PermissionError:
        logger.error("Permission denied KILL PID %s", pid)
