"""Dual persistent memory: Chroma archive (RAG) + SQLite personal facts on an encrypted mount."""

from __future__ import annotations

import logging
import os
import re
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Sequence

logger = logging.getLogger(__name__)


def _env_path(name: str, default: str = "") -> Path:
    raw = os.environ.get(name, default).strip()
    return Path(raw).expanduser() if raw else Path()


class HenryDualMemoryManager:
    """
    Archive (Chroma + local Ollama embeddings) and personal facts (SQLite).

    When MEMORY_MOUNT_PATH is missing, ``is_encrypted_storage_available`` is False and
    no writes or Chroma initialization occur on HenryData paths.
    """

    def __init__(
        self,
        *,
        mount_path: Path | None = None,
        archive_path: Path | None = None,
        personal_db_path: Path | None = None,
        embed_model: str | None = None,
        ollama_base_url: str | None = None,
    ) -> None:
        self._mount = mount_path or _env_path("MEMORY_MOUNT_PATH")
        self._archive_dir = archive_path or _env_path("ARCHIVE_DB_PATH")
        self._personal_path = personal_db_path or _env_path("PERSONAL_MEMORY_PATH")
        self._embed_model = (embed_model or os.environ.get("HENRY_EMBED_MODEL", "nomic-embed-text")).strip()
        self._ollama_base = (
            (ollama_base_url or os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")).rstrip("/")
        )

        self.is_encrypted_storage_available = self._mount.exists() and self._mount.is_dir()
        self.personal_memory_ready = False
        self.archive_ready = False

        self._conn: sqlite3.Connection | None = None
        self._vectorstore: Any = None
        self._sqlite_lock = threading.Lock()

        if not self.is_encrypted_storage_available:
            logger.warning("HenryData mount missing at %s; dual memory disabled.", self._mount)
            return

        try:
            self._personal_path.parent.mkdir(parents=True, exist_ok=True)
            self._archive_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            logger.warning("Could not prepare HenryData paths: %s", exc)
            self.is_encrypted_storage_available = False
            return

        self._init_personal_sqlite()
        self._init_archive_chroma()

    def _init_personal_sqlite(self) -> None:
        if not str(self._personal_path):
            return
        try:
            self._conn = sqlite3.connect(str(self._personal_path), check_same_thread=False)
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS user_facts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    category TEXT NOT NULL,
                    fact TEXT NOT NULL,
                    timestamp REAL NOT NULL
                )
                """
            )
            self._conn.commit()
            self.personal_memory_ready = True
        except (OSError, sqlite3.Error) as exc:
            logger.warning("Personal SQLite init failed: %s", exc)
            self._conn = None
            self.personal_memory_ready = False

    def _init_archive_chroma(self) -> None:
        if not str(self._archive_dir):
            return
        try:
            from langchain_chroma import Chroma
            from langchain_ollama import OllamaEmbeddings

            embeddings = OllamaEmbeddings(
                model=self._embed_model,
                base_url=self._ollama_base,
            )
            self._vectorstore = Chroma(
                collection_name="henry_archive",
                embedding_function=embeddings,
                persist_directory=str(self._archive_dir),
            )
            self.archive_ready = True
        except Exception as exc:  # noqa: BLE001 — optional stack (missing deps / Ollama)
            logger.warning("Chroma / LangChain archive init failed: %s", exc)
            self._vectorstore = None
            self.archive_ready = False

    def close(self) -> None:
        with self._sqlite_lock:
            if self._conn is not None:
                try:
                    self._conn.close()
                except sqlite3.Error:
                    pass
                self._conn = None

    def save_user_fact(self, category: str, fact: str) -> int | None:
        if not self.personal_memory_ready or self._conn is None:
            return None
        cat = (category or "general").strip()[:200]
        body = (fact or "").strip()
        if not body:
            return None
        try:
            with self._sqlite_lock:
                cur = self._conn.execute(
                    "INSERT INTO user_facts (category, fact, timestamp) VALUES (?, ?, ?)",
                    (cat, body[:4000], time.time()),
                )
                self._conn.commit()
                return int(cur.lastrowid) if cur.lastrowid is not None else None
        except sqlite3.Error as exc:
            logger.warning("save_user_fact failed: %s", exc)
            return None

    def retrieve_personal_context(self, user_text: str, *, limit: int = 12) -> str:
        """Return a short block of matching personal facts for injection into the system prompt."""
        if not self.personal_memory_ready or self._conn is None:
            return ""
        tokens = [w for w in re.findall(r"\w+", user_text.lower()) if len(w) > 2]
        if not tokens:
            return ""
        tokens = tokens[:8]
        clauses: list[str] = []
        params: list[str] = []
        for t in tokens:
            p = f"%{t}%"
            clauses.append("(LOWER(fact) LIKE LOWER(?) OR LOWER(category) LIKE LOWER(?))")
            params.extend([p, p])
        where_sql = " OR ".join(clauses)
        sql = (
            f"SELECT category, fact FROM user_facts WHERE {where_sql} "
            f"ORDER BY timestamp DESC LIMIT ?"
        )
        try:
            with self._sqlite_lock:
                rows = self._conn.execute(sql, (*params, limit)).fetchall()
        except sqlite3.Error as exc:
            logger.debug("retrieve_personal_context query failed: %s", exc)
            return ""
        if not rows:
            return ""
        lines = [f"- ({c}) {f}" for c, f in rows]
        return "Known personal facts (from encrypted store; treat as user-provided context):\n" + "\n".join(lines)

    def archive_add_texts(
        self,
        texts: Sequence[str],
        metadatas: Sequence[dict[str, Any]] | None = None,
    ) -> None:
        """Ingest plain texts into the Chroma archive (optional RAG writes)."""
        if not self.archive_ready or self._vectorstore is None:
            return
        from langchain_core.documents import Document

        docs = [Document(page_content=t) for t in texts if str(t).strip()]
        if not docs:
            return
        if metadatas and len(metadatas) == len(docs):
            for d, meta in zip(docs, metadatas):
                d.metadata = dict(meta)
        self._vectorstore.add_documents(docs)

    def archive_search(self, query: str, k: int = 4) -> list[str]:
        if not self.archive_ready or self._vectorstore is None or not query.strip():
            return []
        try:
            hits = self._vectorstore.similarity_search(query, k=k)
        except Exception as exc:  # noqa: BLE001
            logger.debug("archive_search failed: %s", exc)
            return []
        return [d.page_content for d in hits if getattr(d, "page_content", None)]
