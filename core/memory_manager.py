"""Three Chroma memory collections plus SQLite personal facts."""

from __future__ import annotations

import logging
import os
import re
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

from langchain_core.documents import Document

logger = logging.getLogger(__name__)

_ARCHIVE_COLLECTION = "henry_archive"
_CHAT_HISTORY_COLLECTION = "henry_chat_history"
_WORKING_MEMORY_COLLECTION = "henry_working_memory"
_LEGACY_COLLECTION_NAMES = frozenset({
    "langchain",
    "default",
    "documents",
    "texts",
    "text_collection",
    "henry_documents",
})
_OLD_ARCHIVE_METADATA_KEYS = frozenset({
    "source_file",
    "provider",
    "category",
    "document_type",
    "date",
    "pending_id",
})


def _env_path(name: str, default: str = "") -> Path:
    raw = os.environ.get(name, default).strip()
    return Path(raw).expanduser() if raw else Path()


class HenryMemoryManager:
    """
    Encrypted three-tier memory for Henry.

    The three Chroma collections are ``henry_archive`` for permanent documents
    and raw OCR, ``henry_chat_history`` for permanent conversations, and
    ``henry_working_memory`` for the 30-day temporary scratchpad. SQLite
    ``user_facts`` remains the durable personal profile tier.

    When MEMORY_MOUNT_PATH is missing, ``is_encrypted_storage_available`` is False and
    no SQLite or Chroma initialization occurs on HenryData paths.
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
        self.chat_history_ready = False
        self.working_memory_ready = False

        self._conn: sqlite3.Connection | None = None
        self._archive_vectorstore: Any = None
        self._chat_history_vectorstore: Any = None
        self._working_memory_vectorstore: Any = None
        self._sqlite_lock = threading.Lock()

        if not self.is_encrypted_storage_available:
            logger.warning(
                "Encrypted volume not found at %s - all memory services disabled "
                "to protect privacy.",
                self._mount,
            )
            return

        try:
            self._personal_path.parent.mkdir(parents=True, exist_ok=True)
            self._archive_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            logger.warning("Could not prepare HenryData paths: %s", exc)
            self.is_encrypted_storage_available = False
            return

        self._init_personal_sqlite()
        self._init_chroma_collections()

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

    def _init_chroma_collections(self) -> None:
        if not str(self._archive_dir):
            return
        try:
            from langchain_chroma import Chroma
            from langchain_ollama import OllamaEmbeddings

            embeddings = OllamaEmbeddings(
                model=self._embed_model,
                base_url=self._ollama_base,
            )
            self._archive_vectorstore = Chroma(
                collection_name=_ARCHIVE_COLLECTION,
                embedding_function=embeddings,
                persist_directory=str(self._archive_dir),
            )
            self._chat_history_vectorstore = Chroma(
                collection_name=_CHAT_HISTORY_COLLECTION,
                embedding_function=embeddings,
                persist_directory=str(self._archive_dir),
            )
            self._working_memory_vectorstore = Chroma(
                collection_name=_WORKING_MEMORY_COLLECTION,
                embedding_function=embeddings,
                persist_directory=str(self._archive_dir),
            )
            self.archive_ready = True
            self.chat_history_ready = True
            self.working_memory_ready = True
            self._recover_and_purge_legacy_chroma_collections()
        except Exception as exc:  # noqa: BLE001 - optional stack (missing deps / Ollama)
            logger.warning("Chroma / LangChain memory init failed: %s", exc)
            self._archive_vectorstore = None
            self._chat_history_vectorstore = None
            self._working_memory_vectorstore = None
            self.archive_ready = False
            self.chat_history_ready = False
            self.working_memory_ready = False

    def _recover_and_purge_legacy_chroma_collections(self) -> None:
        if self._archive_vectorstore is None:
            return

        client = getattr(self._archive_vectorstore, "_client", None)
        if client is None:
            return

        try:
            collection_names = self._list_chroma_collection_names(client)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not inspect Chroma legacy collections: %s", exc)
            return

        for collection_name in collection_names:
            if collection_name in {_ARCHIVE_COLLECTION, _CHAT_HISTORY_COLLECTION, _WORKING_MEMORY_COLLECTION}:
                continue
            self._migrate_and_delete_legacy_collection(client, collection_name)

        self._optimize_chroma_storage()

    @staticmethod
    def _list_chroma_collection_names(client: Any) -> list[str]:
        raw_collections = client.list_collections()
        names: list[str] = []
        for collection in raw_collections:
            if isinstance(collection, str):
                name = collection
            else:
                name = str(getattr(collection, "name", "") or "")
            if name:
                names.append(name)
        return names

    def _migrate_and_delete_legacy_collection(self, client: Any, collection_name: str) -> None:
        try:
            collection = client.get_collection(collection_name)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not open legacy Chroma collection %s: %s", collection_name, exc)
            self._delete_chroma_collection(client, collection_name)
            return

        if not self._is_legacy_collection(collection_name, collection):
            return

        try:
            payload = collection.get(include=["documents", "metadatas"])
        except Exception as exc:  # noqa: BLE001
            logger.warning("Legacy Chroma collection %s is unreadable and will be dropped: %s", collection_name, exc)
            self._delete_chroma_collection(client, collection_name)
            return

        documents = list(payload.get("documents") or [])
        metadatas = list(payload.get("metadatas") or [])
        migrated_docs: list[Document] = []
        for index, text in enumerate(documents):
            body = str(text or "").strip()
            if not body:
                continue
            raw_metadata = metadatas[index] if index < len(metadatas) and isinstance(metadatas[index], dict) else {}
            metadata = self._migrate_legacy_archive_metadata(raw_metadata, collection_name)
            migrated_docs.append(Document(page_content=body, metadata=metadata))

        if migrated_docs:
            try:
                ids = [uuid4().hex for _ in migrated_docs]
                self._archive_vectorstore.add_documents(migrated_docs, ids=ids)
                logger.info(
                    "Migrated %d records from legacy Chroma collection %s into %s",
                    len(migrated_docs),
                    collection_name,
                    _ARCHIVE_COLLECTION,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Migration from legacy Chroma collection %s failed: %s", collection_name, exc)
                return

        self._delete_chroma_collection(client, collection_name)

    @staticmethod
    def _is_legacy_collection(collection_name: str, collection: Any) -> bool:
        if collection_name in _LEGACY_COLLECTION_NAMES:
            return True

        try:
            sample = collection.get(limit=5, include=["metadatas", "documents"])
        except Exception:
            return True

        documents = [str(item or "").strip() for item in sample.get("documents") or []]
        metadatas = [item for item in sample.get("metadatas") or [] if isinstance(item, dict)]
        if not documents and not metadatas:
            return False
        if any(meta.get("tier") in {"archive", "chat_history", "working_memory"} for meta in metadatas):
            return False
        return any(_OLD_ARCHIVE_METADATA_KEYS.intersection(meta.keys()) for meta in metadatas)

    @staticmethod
    def _migrate_legacy_archive_metadata(raw_metadata: dict[str, Any], collection_name: str) -> dict[str, Any]:
        metadata = HenryMemoryManager._normalize_chroma_metadata(raw_metadata)
        source_file = str(metadata.get("source_file") or "").strip()
        absolute_path = str(metadata.get("absolute_path") or metadata.get("absolute_file_path") or "").strip()
        if not absolute_path and source_file:
            absolute_path = source_file
        if absolute_path:
            metadata["absolute_path"] = absolute_path
            metadata["absolute_file_path"] = absolute_path
        if source_file and "source_file" not in metadata:
            metadata["source_file"] = source_file
        metadata["tier"] = "archive"
        metadata["migrated_from_collection"] = collection_name
        metadata["migrated_at"] = time.time()
        return metadata

    @staticmethod
    def _delete_chroma_collection(client: Any, collection_name: str) -> None:
        try:
            client.delete_collection(collection_name)
            logger.info("Deleted legacy Chroma collection %s", collection_name)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not delete legacy Chroma collection %s: %s", collection_name, exc)

    def _optimize_chroma_storage(self) -> None:
        sqlite_path = self._archive_dir / "chroma.sqlite3"
        if not sqlite_path.is_file():
            return

        try:
            with sqlite3.connect(str(sqlite_path), timeout=30.0, isolation_level=None) as conn:
                conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                conn.execute("PRAGMA optimize")
                conn.execute("VACUUM")
        except sqlite3.Error as exc:
            logger.warning("Chroma SQLite optimization failed: %s", exc)

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

    def archive_add_document(
        self,
        *,
        text: str,
        absolute_file_path: str | Path,
        provider: str,
        category: str,
        document_type: str,
    ) -> str | None:
        """Ingest one permanent document into the archival RAG collection."""
        if not self.archive_ready or self._archive_vectorstore is None:
            return None

        body = text.strip()
        if not body:
            raise ValueError("archive_add_document requires non-empty text")

        path = Path(absolute_file_path).expanduser()
        if not path.is_absolute():
            raise ValueError("archive_add_document requires an absolute_file_path")

        provider_value = provider.strip()
        category_value = category.strip()
        document_type_value = document_type.strip()
        if not provider_value:
            raise ValueError("archive_add_document requires a non-empty provider")
        if not category_value:
            raise ValueError("archive_add_document requires a non-empty category")
        if not document_type_value:
            raise ValueError("archive_add_document requires a non-empty document_type")

        metadata = {
            "tier": "archive",
            "absolute_file_path": str(path),
            "absolute_path": str(path),
            "source_file": path.name,
            "provider": provider_value,
            "category": category_value,
            "document_type": document_type_value,
            "ingested_at": time.time(),
        }
        document_id = uuid4().hex
        document = Document(page_content=body, metadata=metadata)
        self._archive_vectorstore.add_documents([document], ids=[document_id])
        return document_id

    def archive_search(self, query: str, k: int = 4) -> list[Document]:
        """Return full LangChain Document hits, including metadata and file paths."""
        if not self.archive_ready or self._archive_vectorstore is None or not query.strip():
            return []
        try:
            hits = self._archive_vectorstore.similarity_search(query, k=k)
        except Exception as exc:  # noqa: BLE001
            logger.debug("archive_search failed: %s", exc)
            return []
        return [d for d in hits if isinstance(d, Document) and d.page_content]

    def add_chat_history_entry(
        self,
        *,
        text: str,
        role: str,
        metadata: dict[str, Any] | None = None,
    ) -> str | None:
        """Store one permanent conversation archive entry."""
        if not self.chat_history_ready or self._chat_history_vectorstore is None:
            return None

        body = text.strip()
        speaker = role.strip()
        if not body:
            raise ValueError("add_chat_history_entry requires non-empty text")
        if not speaker:
            raise ValueError("add_chat_history_entry requires a non-empty role")

        payload = self._normalize_chroma_metadata(metadata or {})
        if "timestamp" not in payload:
            payload["timestamp"] = time.time()
        entry_type = str(payload.get("entry_type") or "permanent_chat_archive").strip() or "permanent_chat_archive"
        payload.update({
            "tier": "chat_history",
            "entry_type": entry_type,
            "role": speaker,
        })
        document_id = uuid4().hex
        document = Document(page_content=body, metadata=payload)
        self._chat_history_vectorstore.add_documents([document], ids=[document_id])
        return document_id

    def search_chat_history(
        self,
        query: str,
        *,
        k: int = 8,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[Document]:
        """Search the permanent chat archive and return full documents."""
        if not self.chat_history_ready or self._chat_history_vectorstore is None or not query.strip():
            return []

        filter_query = metadata_filter if metadata_filter else None
        try:
            if filter_query is None:
                hits = self._chat_history_vectorstore.similarity_search(query, k=k)
            else:
                hits = self._chat_history_vectorstore.similarity_search(query, k=k, filter=filter_query)
        except Exception as exc:  # noqa: BLE001
            logger.debug("search_chat_history failed: %s", exc)
            return []
        return [d for d in hits if isinstance(d, Document) and d.page_content]

    def add_working_memory_entry(
        self,
        *,
        text: str,
        role: str,
        metadata: dict[str, Any] | None = None,
    ) -> str | None:
        """Store one temporary scratchpad entry in ``henry_working_memory``."""
        if not self.working_memory_ready or self._working_memory_vectorstore is None:
            return None

        body = text.strip()
        speaker = role.strip()
        if not body:
            raise ValueError("add_working_memory_entry requires non-empty text")
        if not speaker:
            raise ValueError("add_working_memory_entry requires a non-empty role")

        payload = self._normalize_chroma_metadata(metadata or {})
        if "timestamp" not in payload:
            payload["timestamp"] = time.time()
        entry_type = str(payload.get("entry_type") or "working_memory").strip() or "working_memory"
        payload.update({
            "tier": "working_memory",
            "entry_type": entry_type,
            "role": speaker,
        })
        document_id = uuid4().hex
        document = Document(page_content=body, metadata=payload)
        self._working_memory_vectorstore.add_documents([document], ids=[document_id])
        return document_id

    def search_working_memory(
        self,
        query: str,
        *,
        k: int = 8,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[Document]:
        """Search the temporary working scratchpad and return full documents."""
        if not self.working_memory_ready or self._working_memory_vectorstore is None or not query.strip():
            return []

        filter_query = metadata_filter if metadata_filter else None
        try:
            if filter_query is None:
                hits = self._working_memory_vectorstore.similarity_search(query, k=k)
            else:
                hits = self._working_memory_vectorstore.similarity_search(query, k=k, filter=filter_query)
        except Exception as exc:  # noqa: BLE001
            logger.debug("search_working_memory failed: %s", exc)
            return []
        return [d for d in hits if isinstance(d, Document) and d.page_content]

    def chat_history_add_entry(
        self,
        *,
        text: str,
        entry_type: str,
        session_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> str | None:
        """Store permanent conversational context or active research/link metadata."""
        payload = dict(metadata or {})
        payload["entry_type"] = entry_type
        payload["session_id"] = session_id
        return self.add_chat_history_entry(
            text=text,
            role=str(payload.get("role") or entry_type),
            metadata=payload,
        )

    def chat_history_search(
        self,
        query: str,
        *,
        k: int = 6,
        session_id: str | None = None,
    ) -> list[Document]:
        """Search permanent chat/research/link context and return full documents."""
        filter_query = {"session_id": session_id.strip()} if session_id and session_id.strip() else None
        return self.search_chat_history(query, k=k, metadata_filter=filter_query)

    def add_temporary_chat_context(
        self,
        *,
        text: str,
        role: str,
        metadata: dict[str, Any] | None = None,
    ) -> str | None:
        """Compatibility wrapper for the 30-day working-memory scratchpad."""
        return self.add_working_memory_entry(text=text, role=role, metadata=metadata)

    def search_temporary_context(
        self,
        query: str,
        *,
        k: int = 6,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[Document]:
        """Compatibility wrapper for the 30-day working-memory scratchpad."""
        return self.search_working_memory(query, k=k, metadata_filter=metadata_filter)

    def prune_old_temporary_context(self, max_age_seconds: int | float = 2_592_000) -> int:
        """Compatibility wrapper for pruning the 30-day working-memory scratchpad."""
        return self.prune_old_working_memory(max_age_seconds=max_age_seconds)

    def prune_old_working_memory(self, max_age_seconds: int | float = 2_592_000) -> int:
        """Physically delete working-memory entries older than the retention window."""
        if not self.working_memory_ready or self._working_memory_vectorstore is None:
            return 0

        try:
            retention_seconds = float(max_age_seconds)
        except (TypeError, ValueError):
            raise ValueError("prune_old_working_memory requires a numeric max_age_seconds") from None
        if retention_seconds <= 0:
            raise ValueError("prune_old_working_memory requires max_age_seconds greater than zero")

        collection = getattr(self._working_memory_vectorstore, "_collection", None)
        if collection is None:
            return 0

        cutoff = time.time() - retention_seconds
        try:
            matches = collection.get(where={"timestamp": {"$lt": cutoff}}, include=[])
            ids = list(matches.get("ids") or [])
            if not ids:
                return 0
            collection.delete(ids=ids)
            return len(ids)
        except Exception as exc:  # noqa: BLE001
            logger.debug("prune_old_working_memory failed: %s", exc)
            return 0

    def chat_history_clear(self, *, session_id: str | None = None) -> int:
        """Permanent chat history is append-only and is never cleared by runtime maintenance."""
        if session_id and session_id.strip():
            logger.warning("Ignoring chat_history_clear for permanent session %s", session_id.strip())
        else:
            logger.warning("Ignoring chat_history_clear for permanent chat archive")
        return 0

    @staticmethod
    def _normalize_chroma_metadata(metadata: dict[str, Any]) -> dict[str, str | int | float | bool]:
        normalized: dict[str, str | int | float | bool] = {}
        for key, value in metadata.items():
            clean_key = str(key).strip()
            if not clean_key or value is None:
                continue
            if isinstance(value, (str, int, float, bool)):
                normalized[clean_key] = value
            else:
                normalized[clean_key] = str(value)
        return normalized
