"""File pipeline manager — watches inbox, OCR-extracts text, and archives into 7 category folders."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import time
from pathlib import Path
from typing import Any

import httpx
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

logger = logging.getLogger(__name__)

_TESSERACT_CMD = (
    os.environ.get("TESSERACT_CMD")
    or shutil.which("tesseract")
    or "/usr/bin/tesseract"
)
pytesseract.pytesseract.tesseract_cmd = _TESSERACT_CMD

if not Path(_TESSERACT_CMD).is_file():
    print(
        f"WARNING: Tesseract binary not found at {_TESSERACT_CMD}. "
        "OCR will fail. Install via: brew install tesseract (macOS) "
        "or apt-get install tesseract-ocr (Linux).",
        flush=True,
    )

_DEFAULT_ROOT = os.environ.get("HENRY_FILES_ROOT", "")

INBOX_FOLDER = "01_Eingang_OCR"

CATEGORY_FOLDERS: dict[str, str] = {
    "Wohnen": "01_Wohnen",
    "Finanzen": "02_Finanzen",
    "Versicherung": "03_Versicherung",
    "Arbeit": "04_Arbeit",
    "Gesundheit": "05_Gesundheit",
    "Mobilität": "06_Mobilität",
    "Korrespondenz": "07_Korrespondenz",
}

_CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "Wohnen": [
        "mietzins", "mietvertrag", "liegenschaft", "vermieter", "mieter",
        "mietobjekt", "nettomiete", "wohnung", "nebenkosten", "heizkosten",
        "hauswart", "stockwerkeigentum", "kündigung der wohnung",
    ],
    "Finanzen": [
        "valuta", "saldo", "kontoauszug", "iban", "zahlungseingang",
        "belastung", "gutschrift", "zins", "hypothek", "darlehen",
        "steuererklärung", "steuer", "veranlagung", "budget",
    ],
    "Versicherung": [
        "police", "prämie", "versicherung", "deckung", "versicherungsnehmer",
        "schadenfall", "franchise", "selbstbehalt", "leistungsabrechnung",
    ],
    "Arbeit": [
        "lohnabrechnung", "bruttolohn", "nettolohn", "ahv", "arbeitsvertrag",
        "arbeitszeugnis", "kündigung", "sozialversicherung", "bvg", "pensionskasse",
    ],
    "Gesundheit": [
        "arzt", "diagnose", "rezept", "patient", "spital", "krankenhaus",
        "therapie", "überweisung", "medikament", "krankenkasse",
    ],
    "Mobilität": [
        "fahrzeug", "auto", "halbtax", "general", "sbb", "zug",
        "fahrzeugausweis", "versicherungsnachweis", "parkplatz", "vignette",
    ],
    "Korrespondenz": [
        "brief", "einschreiben", "mitteilung", "benachrichtigung",
        "einladung", "bestätigung", "quittung",
    ],
}

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif"}
_PDF_EXT = ".pdf"
_SUPPORTED_EXTS = _IMAGE_EXTS | {_PDF_EXT}
_IGNORED_FILES = {".DS_Store", "Thumbs.db", "._.DS_Store"}


def _detect_category_by_keywords(text: str) -> str | None:
    """Scan text for category keywords. Returns the best matching category or None."""
    lower = text.lower()
    scores: dict[str, int] = {}
    for category, keywords in _CATEGORY_KEYWORDS.items():
        hits = sum(1 for kw in keywords if kw in lower)
        if hits:
            scores[category] = hits
    if not scores:
        return None
    return max(scores, key=scores.get)  # type: ignore[arg-type]


def _map_document_type_to_category(doc_type: str) -> str:
    """Map an LLM-returned document_type to one of the 7 main categories."""
    mapping: dict[str, str] = {
        "mietvertrag": "Wohnen",
        "nebenkosten": "Wohnen",
        "wohnung": "Wohnen",
        "rechnung": "Finanzen",
        "kontoauszug": "Finanzen",
        "steuerdokument": "Finanzen",
        "steuer": "Finanzen",
        "invoice": "Finanzen",
        "versicherung": "Versicherung",
        "police": "Versicherung",
        "insurance": "Versicherung",
        "lohnabrechnung": "Arbeit",
        "arbeitsvertrag": "Arbeit",
        "arbeitszeugnis": "Arbeit",
        "kündigung": "Arbeit",
        "medizinisch": "Gesundheit",
        "arztbericht": "Gesundheit",
        "rezept": "Gesundheit",
        "fahrzeug": "Mobilität",
        "sbb": "Mobilität",
        "brief": "Korrespondenz",
        "vertrag": "Korrespondenz",
        "letter": "Korrespondenz",
        "contract": "Korrespondenz",
    }
    return mapping.get(doc_type.lower(), "")

# --- Semantic OCR cleaning -------------------------------------------------

_CLEAN_OCR_SYSTEM_PROMPT = (
    "The following is a raw OCR output with potential errors "
    '(e.g., "Ziirich" instead of "Zürich", "Strasse" instead of "Strasse"). '
    "Clean this text semantically. Fix obvious typos in names, dates, addresses, "
    "and Swiss legal/business terms, but keep the original meaning 100% intact. "
    "Do NOT add any commentary, explanation, or markdown. "
    "Return ONLY the cleaned text, nothing else."
)


def _clean_ocr_text(
    raw_text: str,
    *,
    ollama_base_url: str | None = None,
    model: str | None = None,
    timeout_s: float = 180.0,
) -> str:
    """Send raw OCR output to Ollama for semantic cleaning."""
    base = (ollama_base_url or os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")).rstrip("/")
    chosen_model = model or os.environ.get("HENRY_CLASSIFY_MODEL") or os.environ.get("OLLAMA_MODEL", "qwen2.5:7b")

    payload = {
        "model": chosen_model,
        "messages": [
            {"role": "system", "content": _CLEAN_OCR_SYSTEM_PROMPT},
            {"role": "user", "content": raw_text},
        ],
        "stream": False,
    }
    try:
        with httpx.Client(timeout=timeout_s) as client:
            resp = client.post(f"{base}/api/chat", json=payload)
            resp.raise_for_status()
            data = resp.json()
    except httpx.ConnectError:
        logger.warning("OCR cleaning: Ollama not reachable at %s — using raw text", base)
        print(f"WARNING: Ollama not reachable at {base}. Is it running?", flush=True)
        return raw_text
    except httpx.HTTPStatusError as exc:
        logger.warning("OCR cleaning: Ollama returned HTTP %s — using raw text", exc.response.status_code)
        if exc.response.status_code == 404:
            print(f"WARNING: Model '{chosen_model}' not found in Ollama. Pull it with: ollama pull {chosen_model}", flush=True)
        return raw_text
    except Exception as exc:  # noqa: BLE001
        logger.warning("OCR text cleaning failed, using raw text: %s", exc)
        return raw_text

    content = (data.get("message") or {}).get("content")
    if isinstance(content, str) and content.strip():
        return content.strip()
    return raw_text


# --- Fuzzy provider matching -----------------------------------------------

KNOWN_PROVIDERS: list[str] = [
    "Luzius Sprüngli",
    "Peter Halter AG",
    "Swisscom",
    "CSS",
    "Migros",
    "Coop",
    "Helsana",
    "Visana",
    "Concordia",
    "SWICA",
    "Sanitas",
    "Baloise",
    "Zurich Versicherung",
    "AXA",
    "Mobiliar",
    "Helvetia",
    "PostFinance",
    "UBS",
    "Credit Suisse",
    "Raiffeisen",
    "SBB",
    "Sunrise",
    "Salt",
    "EWL",
    "EWZ",
    "Axpo",
    "BKW",
]


def _fuzzy_match_provider(provider: str, threshold: float = 0.90) -> str:
    """Snap a provider name to a known spelling if similarity >= threshold.

    Uses a simple character-level similarity (SequenceMatcher) so we don't
    need an extra dependency.
    """
    if not provider or provider == "Unknown":
        return provider

    from difflib import SequenceMatcher

    best_score = 0.0
    best_match = provider
    provider_lower = provider.lower()

    for known in KNOWN_PROVIDERS:
        score = SequenceMatcher(None, provider_lower, known.lower()).ratio()
        if score > best_score:
            best_score = score
            best_match = known

    if best_score >= threshold:
        if best_match != provider:
            logger.info("Fuzzy provider match: '%s' → '%s' (%.0f%%)", provider, best_match, best_score * 100)
        return best_match

    return provider

_CLASSIFY_SYSTEM_PROMPT = (
    "You are an expert Swiss archivist. The following is the FULL cleaned text of a document.\n\n"
    "STEP 1 — CATEGORY (most important): Assign the document to exactly ONE of these 7 categories:\n"
    '  "Wohnen"         — rent, apartment, landlord, Nebenkosten, Mietvertrag\n'
    '  "Finanzen"       — bank statements, invoices, taxes, Rechnung, Kontoauszug, Steuer\n'
    '  "Versicherung"   — insurance, Police, Prämie, Deckung, Franchise\n'
    '  "Arbeit"         — employment, Lohnabrechnung, Arbeitsvertrag, AHV, BVG\n'
    '  "Gesundheit"     — medical, doctor, Rezept, Diagnose, Spital\n'
    '  "Mobilität"      — transport, vehicle, SBB, Halbtax, Fahrzeugausweis\n'
    '  "Korrespondenz"  — general letters, contracts, confirmations, Quittung\n\n'
    "STEP 2 — DOCUMENT TYPE: A short label for the specific document "
    '(e.g. "Mietvertrag", "Rechnung", "Lohnabrechnung", "Kontoauszug").\n\n'
    "STEP 3 — PROVIDER: The company or person who sent/issued this document. "
    "Look in the header, letterhead, sender address, or signature block. "
    "If you see Swiss company names or specific addresses, use them. "
    'If you truly cannot identify the provider, use an empty string "".\n\n'
    "STEP 4 — DATE: The document date in YYYY-MM-DD format. "
    "Swiss DD.MM.YYYY must be converted. Use the document date, not due dates.\n\n"
    "Return ONLY a valid JSON object — no markdown, no code fences, no extra text.\n"
    "The JSON must have exactly these keys:\n"
    '  "category": one of the 7 categories above (e.g. "Wohnen", "Finanzen")\n'
    '  "document_type": the specific type (e.g. "Mietvertrag", "Rechnung")\n'
    '  "provider": company/person name, or "" if unsure\n'
    '  "date": YYYY-MM-DD or "Unknown"\n'
    '  "year": YYYY or "Unknown"\n'
    '  "month": MM or "Unknown"\n'
    '  "summary": one-sentence summary\n\n'
    "IMPORTANT: Even if you cannot identify the provider, you MUST still assign "
    "a category and document_type. Never fail the entire classification just "
    "because the provider is unclear."
)

_FALLBACK_META: dict[str, str] = {
    "category": "Unknown",
    "document_type": "Unknown",
    "provider": "",
    "date": "Unknown",
    "year": "Unknown",
    "month": "Unknown",
    "summary": "Unknown",
}

_META_KEYS = tuple(_FALLBACK_META.keys())


def _extract_json_object(text: str) -> dict[str, Any] | None:
    """Best-effort extraction of the first JSON object from LLM output."""
    stripped = text.strip()
    try:
        obj = json.loads(stripped)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        try:
            obj = json.loads(stripped[start : end + 1])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
    return None


def _call_ollama_classify(
    snippet: str,
    base_url: str,
    model: str,
    timeout_s: float,
) -> dict[str, str] | None:
    """Single Ollama classification attempt. Returns parsed metadata or None."""
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": _CLASSIFY_SYSTEM_PROMPT},
            {"role": "user", "content": snippet},
        ],
        "stream": False,
    }
    try:
        with httpx.Client(timeout=timeout_s) as client:
            resp = client.post(f"{base_url}/api/chat", json=payload)
            resp.raise_for_status()
            data = resp.json()
    except httpx.ConnectError:
        logger.warning("Classification: Ollama not reachable at %s", base_url)
        print(f"WARNING: Ollama not reachable at {base_url}. Is it running?", flush=True)
        return None
    except httpx.HTTPStatusError as exc:
        logger.warning("Classification: Ollama returned HTTP %s", exc.response.status_code)
        if exc.response.status_code == 404:
            print(f"WARNING: Model '{model}' not found in Ollama. Pull it with: ollama pull {model}", flush=True)
        return None
    except Exception as exc:  # noqa: BLE001
        logger.warning("classify_document LLM call failed: %s", exc)
        return None

    content = (data.get("message") or {}).get("content")
    if not isinstance(content, str):
        return None

    obj = _extract_json_object(content)
    if obj is None:
        logger.warning("classify_document: could not parse JSON from LLM output")
        return None

    result: dict[str, str] = {}
    for key in _META_KEYS:
        val = str(obj.get(key, "Unknown")).strip()
        result[key] = val if val else "Unknown"
    return result


def _has_category(meta: dict[str, str]) -> bool:
    """True when the classification has a valid category."""
    cat = meta.get("category", "Unknown")
    return cat != "Unknown" and cat in CATEGORY_FOLDERS


def _enrich_with_keywords(meta: dict[str, str], text: str) -> dict[str, str]:
    """Fill in missing category from keyword detection and type→category mapping."""
    if not _has_category(meta):
        mapped = _map_document_type_to_category(meta.get("document_type", "Unknown"))
        if mapped:
            meta["category"] = mapped

    if not _has_category(meta):
        detected = _detect_category_by_keywords(text)
        if detected:
            meta["category"] = detected
            logger.info("Category detected via keywords: %s", detected)

    return meta


def classify_document(
    text: str,
    *,
    ollama_base_url: str | None = None,
    model: str | None = None,
    timeout_s: float = 120.0,
) -> dict[str, str]:
    """Classify cleaned OCR text: LLM first, then keyword fallback.

    Category is required for filing. Provider is optional — an empty provider
    does NOT send the file to manual_review.
    """
    base = (ollama_base_url or os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")).rstrip("/")
    chosen_model = model or os.environ.get("HENRY_CLASSIFY_MODEL") or os.environ.get("OLLAMA_MODEL", "qwen2.5:7b")

    result = _call_ollama_classify(text, base, chosen_model, timeout_s)
    if result is None:
        result = dict(_FALLBACK_META)

    if not _has_category(result) and len(text) > 200:
        logger.info("classify_document: no category — retrying with header-focused snippet")
        retry = _call_ollama_classify(text[:1000], base, chosen_model, timeout_s)
        if retry is not None and _has_category(retry):
            result = retry

    result = _enrich_with_keywords(result, text)

    provider = result.get("provider", "")
    if provider and provider != "Unknown":
        result["provider"] = _fuzzy_match_provider(provider)
    elif provider == "Unknown":
        result["provider"] = ""

    return result


class _InboxHandler(FileSystemEventHandler):
    """Reacts to new files landing in the inbox."""

    def __init__(self, manager: HenryFileManager) -> None:
        super().__init__()
        self._manager = manager

    def on_created(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        src = Path(event.src_path)

        if src.name in _IGNORED_FILES or src.name.startswith("._"):
            try:
                src.unlink(missing_ok=True)
            except OSError:
                pass
            return

        if src.suffix.lower() not in _SUPPORTED_EXTS:
            print(f"Henry: ignoring unsupported file type: {src.name}", flush=True)
            return

        print(f"Henry detected a new file: {src.name}", flush=True)
        self._manager._stage_to_backup(src)
        self._manager._auto_process_pending()


class _ManualReviewHandler(FileSystemEventHandler):
    """Detects renames in manual_review so Henry can learn from user corrections."""

    def __init__(self, manager: HenryFileManager) -> None:
        super().__init__()
        self._manager = manager
        self._known: set[str] = set()
        if manager._manual_review.is_dir():
            self._known = {f.name for f in manager._manual_review.iterdir() if f.is_file()}

    def on_moved(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        dest = Path(event.dest_path)
        if dest.parent != self._manager._manual_review:
            return
        if dest.name in _IGNORED_FILES or dest.name.startswith("._"):
            return
        old_name = Path(event.src_path).name
        if old_name == dest.name:
            return
        print(f"Henry noticed a rename in manual_review: {old_name} → {dest.name}", flush=True)
        self._manager.learn_from_manual_review()


class HenryFileManager:
    """
    Manages Henry's category-based document pipeline.

    Visible folders (created automatically under *root*):
        01_Eingang_OCR   — incoming documents (watched)
        Archiv/          — long-term archive, contains category subfolders:
            01_Wohnen, 02_Finanzen, 03_Versicherung, 04_Arbeit,
            05_Gesundheit, 06_Mobilität, 07_Korrespondenz

    System folder (not for the user):
        internal/        — temp files, debug output, document memory
    """

    def __init__(
        self,
        root: Path | str | None = None,
        *,
        on_file_processed: Any | None = None,
        memory_manager: Any | None = None,
    ) -> None:
        self._root = Path(root or _DEFAULT_ROOT or ".").resolve()

        self.inbox = self._root / INBOX_FOLDER
        self._manual_review = self.inbox / "manual_review"

        self._archive = self._root / "Archiv"
        self._categories: dict[str, Path] = {}
        for cat, folder in CATEGORY_FOLDERS.items():
            self._categories[cat] = self._archive / folder

        self._internal = self._root / "internal"
        self._processing = self._internal / "processing"
        self._temp_backup = self._internal / "temp_backup"
        self._pending = self._internal / "pending"
        self._texts_dir = self._internal / "extracted_texts"
        self._debug_texts = self._internal / "debug_texte"
        self._doc_memory = self._internal / "document_memory"
        self._knowledge_base_path = self._root / "knowledge_base.json"

        self._ensure_dirs()

        self._observer: Observer | None = None
        self._extracted: dict[str, str] = {}
        self._on_file_processed = on_file_processed
        self._memory_manager = memory_manager
        self._event_loop: Any | None = None

        self._pending_items: dict[str, dict[str, Any]] = {}

    def _ensure_dirs(self) -> None:
        dirs = [
            self.inbox, self._manual_review,
            self._archive, *self._categories.values(),
            self._internal, self._processing, self._temp_backup, self._pending,
            self._texts_dir, self._debug_texts, self._doc_memory,
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)

    # --- Inbox watcher -----------------------------------------------------

    def start_watching(self) -> None:
        """Begin monitoring inbox and manual_review (non-blocking)."""
        if self._observer is not None:
            return

        inbox_abs = str(self.inbox.resolve())
        print(f"Henry is now watching: {inbox_abs}", flush=True)

        if not self.inbox.is_dir():
            print(f"WARNING: Inbox folder does not exist: {inbox_abs}", flush=True)
            self.inbox.mkdir(parents=True, exist_ok=True)
            print(f"  → Created it automatically.", flush=True)

        if not os.access(str(self.inbox), os.R_OK):
            print(
                f"WARNING: No read permission on {inbox_abs}. "
                "Grant Full Disk Access to your terminal in System Settings → Privacy & Security.",
                flush=True,
            )

        self._observer = Observer()
        self._observer.schedule(_InboxHandler(self), inbox_abs, recursive=False)
        self._observer.schedule(_ManualReviewHandler(self), str(self._manual_review), recursive=False)
        self._observer.start()
        logger.info("Watching %s for new files.", inbox_abs)
        logger.info("Watching %s for renames (learning loop).", self._manual_review)

    def stop_watching(self) -> None:
        if self._observer is None:
            return
        self._observer.stop()
        self._observer.join()
        self._observer = None

    def set_event_loop(self, loop: Any) -> None:
        """Store the running asyncio event loop so watchdog threads can schedule coroutines."""
        self._event_loop = loop

    def _auto_process_pending(self) -> None:
        """Called from the watchdog thread after a file is copied to temp_backup."""
        results = self.process_inbox()
        cb = self._on_file_processed
        if not cb or not results:
            return
        loop = self._event_loop
        for entry in results:
            if loop is not None and loop.is_running():
                loop.call_soon_threadsafe(asyncio.ensure_future, cb(entry))
            else:
                try:
                    cb(entry)
                except Exception:  # noqa: BLE001
                    pass

    # --- Processing stage --------------------------------------------------

    def _stage_to_backup(self, src: Path) -> Path | None:
        """Copy a newly detected file into temp_backup for safe processing.

        The original stays in the inbox until processing succeeds.
        """
        if not src.is_file():
            return None
        dest = self._temp_backup / src.name
        if dest.exists():
            stem, suffix = dest.stem, dest.suffix
            dest = self._temp_backup / f"{stem}_{int(time.time())}{suffix}"
        try:
            shutil.copy2(str(src), str(dest))
            logger.info("Backed up to temp_backup: %s", dest.name)
            return dest
        except OSError as exc:
            logger.warning("Could not copy %s to temp_backup: %s", src.name, exc)
            return None

    def _move_to_manual_review(self, src: Path) -> Path | None:
        """Move a file to manual_review when processing fails."""
        dest = self._manual_review / src.name
        if dest.exists():
            stem, suffix = dest.stem, dest.suffix
            dest = self._manual_review / f"{stem}_{int(time.time())}{suffix}"
        try:
            shutil.move(str(src), str(dest))
            logger.info("Moved to manual_review: %s", dest.name)
            return dest
        except OSError as exc:
            logger.warning("Could not move %s to manual_review: %s", src.name, exc)
            return None

    # --- Knowledge base ---------------------------------------------------

    def _load_knowledge_base(self) -> list[dict[str, Any]]:
        if not self._knowledge_base_path.is_file():
            return []
        try:
            data = json.loads(self._knowledge_base_path.read_text(encoding="utf-8"))
            return data if isinstance(data, list) else []
        except (json.JSONDecodeError, OSError):
            return []

    def _save_knowledge_base(self, entries: list[dict[str, Any]]) -> None:
        try:
            self._knowledge_base_path.write_text(
                json.dumps(entries, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError as exc:
            logger.warning("Could not write knowledge_base.json: %s", exc)

    def _add_to_knowledge_base(
        self,
        archived_filename: str,
        meta: dict[str, str],
        full_text_path: str,
    ) -> None:
        kb = self._load_knowledge_base()
        kb.append({
            "filename": archived_filename,
            "category": meta.get("category", "Unknown"),
            "document_type": meta.get("document_type", "Unknown"),
            "provider": meta.get("provider", ""),
            "date": meta.get("date", "Unknown"),
            "summary": meta.get("summary", ""),
            "full_text_path": full_text_path,
            "archived_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        })
        self._save_knowledge_base(kb)

    def _save_to_document_memory(self, smart_name: str, text: str) -> Path:
        """Persist cleaned text in internal/document_memory."""
        txt_name = f"{Path(smart_name).stem}.txt"
        out = self._doc_memory / txt_name
        if out.exists():
            stem = out.stem
            out = self._doc_memory / f"{stem}_{int(time.time())}.txt"
        out.write_text(text, encoding="utf-8")
        return out

    def _cleanup_temp_files(self, filename: str) -> None:
        """Remove transient copies from temp_backup and extracted_texts after archiving."""
        stem = Path(filename).stem
        for d, patterns in [
            (self._temp_backup, [filename]),
            (self._texts_dir, [f"{stem}.txt"]),
        ]:
            for pattern in patterns:
                target = d / pattern
                try:
                    if target.is_file():
                        target.unlink()
                        logger.info("Cleaned up %s", target)
                except OSError as exc:
                    logger.debug("Cleanup failed for %s: %s", target, exc)

    def _ingest_to_memory(
        self,
        filename: str,
        cleaned_text: str,
        meta: dict[str, str],
    ) -> None:
        """Push the cleaned document text + metadata into the RAG long-term memory."""
        mm = self._memory_manager
        if mm is None:
            return
        if not getattr(mm, "archive_ready", False):
            logger.info("Memory archive not ready — skipping ingestion for %s", filename)
            return
        try:
            metadata = {
                "category": meta.get("category", "Unknown"),
                "document_type": meta.get("document_type", "Unknown"),
                "provider": meta.get("provider", ""),
                "date": meta.get("date", "Unknown"),
                "source_file": filename,
            }
            mm.archive_add_texts([cleaned_text], [metadata])
            print(
                f"Henry: Content of {filename} successfully added to local long-term memory.",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Memory ingestion failed for %s: %s", filename, exc)

    # --- Learning loop (manual_review renames) -----------------------------

    def learn_from_manual_review(self) -> list[dict[str, str]]:
        """Detect manually renamed files in manual_review and update the knowledge base.

        Convention: if the user renames a file to follow the standard format
        ``YYYY-MM-DD_DocumentType_Provider.ext``, Henry extracts those fields
        and records them so future documents with similar layouts can be matched.
        """
        learned: list[dict[str, str]] = []
        if not self._manual_review.is_dir():
            return learned

        for f in sorted(self._manual_review.iterdir()):
            if not f.is_file() or f.name in _IGNORED_FILES or f.name.startswith("._"):
                continue
            parts = f.stem.split("_")
            if len(parts) < 2:
                continue

            date_candidate = parts[0] if re.match(r"\d{4}-\d{2}-\d{2}$", parts[0]) else "Unknown"
            doc_type = parts[1] if len(parts) >= 2 else "Unknown"
            provider = parts[2] if len(parts) >= 3 else "Unknown"

            if doc_type == "Unknown":
                continue

            category = _map_document_type_to_category(doc_type)
            if not category:
                category = "Korrespondenz"

            meta: dict[str, str] = {
                "category": category,
                "document_type": doc_type,
                "provider": provider if provider != "Unknown" else "",
                "date": date_candidate,
                "year": date_candidate[:4] if date_candidate != "Unknown" else "Unknown",
                "month": date_candidate[5:7] if date_candidate != "Unknown" else "Unknown",
                "summary": "Learned from manual rename",
            }

            text = ""
            cached = self._extracted.get(f.name)
            if cached:
                text = cached
            else:
                try:
                    text = self.extract_text(f)
                except Exception:  # noqa: BLE001
                    pass

            doc_mem_path: Path | None = None
            if text:
                smart = self._build_smart_filename(meta, f.suffix)
                doc_mem_path = self._save_to_document_memory(smart or f.name, text)

            dest = self.organize_file(f, meta)
            if dest is not None:
                self._add_to_knowledge_base(
                    dest.name,
                    meta,
                    str(doc_mem_path) if doc_mem_path else "",
                )
                learned.append({"file": f.name, "archived_as": dest.name, **meta})
                print(
                    f"Henry learned: {f.name} → type='{doc_type}' provider='{provider}'",
                    flush=True,
                )

        return learned

    # --- Archive organiser -------------------------------------------------

    @staticmethod
    def _safe_dirname(value: str) -> str:
        """Sanitise a string for use as a directory or file-name component."""
        cleaned = re.sub(r'[<>:"/\\|?*]', "_", value.strip())
        return cleaned or "Unknown"

    @staticmethod
    def _build_smart_filename(meta: dict[str, str], original_suffix: str) -> str:
        """Build ``YYYY-MM-DD_DocumentType_Provider.ext``, skipping Unknown parts."""
        date = meta.get("date", "Unknown")
        doc_type = meta.get("document_type", "Unknown")
        provider = meta.get("provider", "Unknown")

        parts: list[str] = []
        if date != "Unknown":
            parts.append(date)
        if doc_type != "Unknown":
            parts.append(doc_type)
        if provider != "Unknown":
            parts.append(provider)

        if not parts:
            return ""

        safe = re.sub(r'[<>:"/\\|?*]', "_", "_".join(parts))
        return f"{safe}{original_suffix}"

    def organize_file(
        self,
        file_path: Path | str,
        meta: dict[str, str],
    ) -> Path | None:
        """
        Rename and move *file_path* into a category folder:
        ``<category>/<year>/<month>/YYYY-MM-DD_Type_Provider<ext>``

        Provider is optional — skipped in the filename if empty.
        Only fails if category is missing.
        """
        src = Path(file_path)
        if not src.is_file():
            logger.warning("organize_file: source does not exist — %s", src)
            return None

        category = meta.get("category", "Unknown")
        cat_base = self._categories.get(category)
        if cat_base is None:
            logger.warning("organize_file: unknown category '%s'", category)
            return None

        year = self._safe_dirname(meta.get("year", "Unknown"))
        month = self._safe_dirname(meta.get("month", "Unknown"))

        target_dir = cat_base / year / month
        target_dir.mkdir(parents=True, exist_ok=True)

        smart_name = self._build_smart_filename(meta, src.suffix)
        dest = target_dir / (smart_name if smart_name else src.name)

        if dest.exists():
            stem, suffix = dest.stem, dest.suffix
            dest = target_dir / f"{stem}_{int(time.time())}{suffix}"

        try:
            shutil.move(str(src), str(dest))
            logger.info("Archived %s → %s", src.name, dest)
            return dest
        except OSError as exc:
            logger.warning("organize_file failed for %s: %s", src.name, exc)
            return None

    def list_processing(self) -> list[Path]:
        """Return files currently sitting in temp_backup awaiting processing."""
        if not self._temp_backup.is_dir():
            return []
        return sorted(
            p for p in self._temp_backup.iterdir()
            if p.is_file() and p.name not in _IGNORED_FILES and not p.name.startswith("._")
        )

    # --- OCR extraction ----------------------------------------------------

    @staticmethod
    def _ocr_image(path: Path) -> str:
        img = Image.open(path)
        return pytesseract.image_to_string(img).strip()

    @staticmethod
    def _ocr_pdf(path: Path) -> str:
        pages = convert_from_path(str(path))
        parts: list[str] = []
        for page_img in pages:
            text = pytesseract.image_to_string(page_img).strip()
            if text:
                parts.append(text)
        return "\n\n".join(parts)

    def extract_text(self, file_path: Path | str) -> str:
        """Run OCR on *file_path* (image or PDF) and return the extracted text."""
        src = Path(file_path)
        ext = src.suffix.lower()
        if ext == _PDF_EXT:
            return self._ocr_pdf(src)
        if ext in _IMAGE_EXTS:
            return self._ocr_image(src)
        logger.warning("Unsupported file type for OCR: %s", ext)
        return ""

    def _persist_extracted_text(self, source_name: str, text: str) -> Path:
        """Write extracted text to disk so it can be used for training or archiving."""
        out = self._texts_dir / f"{Path(source_name).stem}.txt"
        out.write_text(text, encoding="utf-8")
        return out

    # --- Inbox processing --------------------------------------------------

    def process_inbox(self) -> list[dict[str, Any]]:
        """
        Process every file in ``temp_backup``:

        1. OCR-extract text from the backup copy
        2. Persist the text for later training / archiving use
        3. Classify via ``classify_document`` (local Ollama LLM)
        4. Stage the file into ``internal/pending/`` and build a proposal
        5. On OCR/classification failure → move original to ``manual_review``

        Returns a list of proposal dicts (one per file) for Telegram confirmation.
        """
        results: list[dict[str, Any]] = []
        for backup_path in self.list_processing():
            filename = backup_path.name
            inbox_original = self.inbox / filename
            entry: dict[str, Any] = {"file": filename}

            failed = False

            try:
                text = self.extract_text(backup_path)
            except Exception as exc:  # noqa: BLE001
                logger.warning("OCR failed for %s: %s", filename, exc)
                entry["error"] = str(exc)
                failed = True
                text = ""

            if not failed and not text:
                entry["error"] = "no_text_extracted"
                failed = True

            if failed:
                if inbox_original.is_file():
                    review_dest = self._move_to_manual_review(inbox_original)
                    entry["moved_to"] = str(review_dest) if review_dest else None
                try:
                    backup_path.unlink(missing_ok=True)
                except OSError:
                    pass
                results.append(entry)
                continue

            self._extracted[filename] = text
            text_file = self._persist_extracted_text(filename, text)
            entry["text_file"] = str(text_file)

            debug_name = f"{Path(filename).stem}.txt"
            debug_path = self._debug_texts / debug_name
            debug_path.write_text(text, encoding="utf-8")
            print(f"DEBUG: Raw OCR saved to internal/debug_texte/{debug_name}", flush=True)
            print(f"DEBUG: Raw preview → {text[:200]}", flush=True)

            time.sleep(2)

            print(f"Henry cleaning OCR text for {filename}…", flush=True)
            cleaned_text = _clean_ocr_text(text)
            entry["text_cleaned"] = cleaned_text != text

            cleaned_debug = self._debug_texts / f"{Path(filename).stem}_cleaned.txt"
            cleaned_debug.write_text(cleaned_text, encoding="utf-8")
            print(f"DEBUG: Cleaned text saved to internal/debug_texte/{Path(filename).stem}_cleaned.txt", flush=True)
            print(f"DEBUG: Cleaned preview → {cleaned_text[:200]}", flush=True)

            time.sleep(2)

            meta = classify_document(cleaned_text)
            entry["classification"] = meta

            if not _has_category(meta):
                logger.warning("No category could be determined for %s", filename)
                entry["error"] = "no_category"
                if inbox_original.is_file():
                    review_dest = self._move_to_manual_review(inbox_original)
                    entry["moved_to"] = str(review_dest) if review_dest else None
                try:
                    backup_path.unlink(missing_ok=True)
                except OSError:
                    pass
                print(
                    f"Henry could not categorise {filename} → moved to manual_review",
                    flush=True,
                )
                results.append(entry)
                continue

            smart_name = self._build_smart_filename(meta, backup_path.suffix) or filename
            entry["proposed_name"] = smart_name

            pending_dest = self._pending / filename
            if pending_dest.exists():
                stem, suffix = pending_dest.stem, pending_dest.suffix
                pending_dest = self._pending / f"{stem}_{int(time.time())}{suffix}"
            try:
                shutil.move(str(backup_path), str(pending_dest))
            except OSError as exc:
                logger.warning("Could not stage %s to pending: %s", filename, exc)
                entry["error"] = "staging_failed"
                results.append(entry)
                continue

            pending_id = pending_dest.stem
            self._pending_items[pending_id] = {
                "pending_path": pending_dest,
                "inbox_original": inbox_original,
                "meta": meta,
                "proposed_name": smart_name,
                "cleaned_text": cleaned_text,
                "original_filename": filename,
            }
            entry["pending_id"] = pending_id
            entry["status"] = "awaiting_confirmation"

            print(
                f"Henry staged {filename} → pending (proposed: {smart_name}, "
                f"category: {meta.get('category')})",
                flush=True,
            )
            results.append(entry)

        return results

    # --- Confirmation / override from Telegram -----------------------------

    def confirm_pending(self, pending_id: str) -> dict[str, Any] | None:
        """User replied OK — finalize archiving for this pending item."""
        item = self._pending_items.pop(pending_id, None)
        if item is None:
            return None

        pending_path: Path = item["pending_path"]
        inbox_original: Path = item["inbox_original"]
        meta: dict[str, str] = item["meta"]
        cleaned_text: str = item["cleaned_text"]
        filename: str = item["original_filename"]

        if not pending_path.is_file():
            logger.warning("confirm_pending: file gone from pending — %s", pending_path)
            return None

        dest = self.organize_file(pending_path, meta)
        if dest is None:
            if inbox_original.is_file():
                self._move_to_manual_review(inbox_original)
            return {"file": filename, "error": "archive_failed"}

        doc_mem_path = self._save_to_document_memory(dest.name, cleaned_text)
        self._add_to_knowledge_base(dest.name, meta, str(doc_mem_path))
        self._ingest_to_memory(filename, cleaned_text, meta)

        if inbox_original.is_file():
            try:
                inbox_original.unlink()
            except OSError as exc:
                logger.warning("Could not delete inbox original %s: %s", filename, exc)

        self._cleanup_temp_files(filename)

        print(
            f"Henry confirmed {filename} → "
            f"category='{meta.get('category')}' "
            f"type='{meta.get('document_type')}' "
            f"provider='{meta.get('provider') or '—'}' "
            f"date='{meta.get('date')}'",
            flush=True,
        )
        return {
            "file": filename,
            "archived_to": str(dest),
            "classification": meta,
        }

    def override_pending(
        self,
        pending_id: str,
        new_category: str,
        new_name: str | None = None,
    ) -> dict[str, Any] | None:
        """User corrected the classification — re-classify, archive, and learn."""
        item = self._pending_items.pop(pending_id, None)
        if item is None:
            return None

        pending_path: Path = item["pending_path"]
        inbox_original: Path = item["inbox_original"]
        old_meta: dict[str, str] = item["meta"]
        cleaned_text: str = item["cleaned_text"]
        filename: str = item["original_filename"]

        if not pending_path.is_file():
            logger.warning("override_pending: file gone from pending — %s", pending_path)
            return None

        cat_key = new_category
        for key, folder in CATEGORY_FOLDERS.items():
            if new_category == folder or new_category.lower() == key.lower():
                cat_key = key
                break

        if cat_key not in CATEGORY_FOLDERS:
            logger.warning("override_pending: unknown category '%s'", cat_key)
            return {"file": filename, "error": f"unknown_category: {new_category}"}

        new_meta = dict(old_meta)
        new_meta["category"] = cat_key

        if new_name:
            new_meta["document_type"] = new_name

        smart_name = self._build_smart_filename(new_meta, pending_path.suffix) or filename
        if new_name and not self._build_smart_filename(new_meta, pending_path.suffix):
            safe = re.sub(r'[<>:"/\\|?*]', "_", new_name)
            smart_name = f"{safe}{pending_path.suffix}"

        dest = self.organize_file(pending_path, new_meta)
        if dest is None:
            if inbox_original.is_file():
                self._move_to_manual_review(inbox_original)
            return {"file": filename, "error": "archive_failed"}

        doc_mem_path = self._save_to_document_memory(dest.name, cleaned_text)
        self._add_to_knowledge_base(dest.name, new_meta, str(doc_mem_path))
        self._ingest_to_memory(filename, cleaned_text, new_meta)

        self._record_correction(filename, old_meta, new_meta)

        if inbox_original.is_file():
            try:
                inbox_original.unlink()
            except OSError as exc:
                logger.warning("Could not delete inbox original %s: %s", filename, exc)

        self._cleanup_temp_files(filename)

        print(
            f"Henry corrected {filename} → "
            f"category='{new_meta.get('category')}' "
            f"type='{new_meta.get('document_type')}' "
            f"(user override)",
            flush=True,
        )
        return {
            "file": filename,
            "archived_to": str(dest),
            "classification": new_meta,
            "was_corrected": True,
        }

    def _record_correction(
        self,
        filename: str,
        old_meta: dict[str, str],
        new_meta: dict[str, str],
    ) -> None:
        """Record a user correction in the knowledge base so Henry can learn."""
        kb = self._load_knowledge_base()
        kb.append({
            "type": "correction",
            "filename": filename,
            "original_classification": {
                "category": old_meta.get("category", "Unknown"),
                "document_type": old_meta.get("document_type", "Unknown"),
                "provider": old_meta.get("provider", ""),
            },
            "corrected_to": {
                "category": new_meta.get("category", "Unknown"),
                "document_type": new_meta.get("document_type", "Unknown"),
                "provider": new_meta.get("provider", ""),
            },
            "corrected_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        })
        self._save_knowledge_base(kb)
        print(
            f"Henry learned: {filename} should be "
            f"'{new_meta.get('category')}' / '{new_meta.get('document_type')}' "
            f"(was '{old_meta.get('category')}' / '{old_meta.get('document_type')}')",
            flush=True,
        )

    def get_pending_ids(self) -> list[str]:
        """Return all pending item IDs awaiting user confirmation."""
        return list(self._pending_items.keys())

    def get_extracted_text(self, filename: str) -> str | None:
        """Retrieve cached OCR text for a previously processed file."""
        return self._extracted.get(filename)

    # --- Repair / re-index -------------------------------------------------

    def repair_knowledge_base(self) -> dict[str, Any]:
        """Fix stale KB entries and re-index archive files missing from knowledge_base.json.

        Phase 1 — fix existing entries:
          - Add missing ``category`` field by locating the file in the archive
          - Fix ``full_text_path`` if the referenced file no longer exists
          - Remove entries whose archive file no longer exists

        Phase 2 — re-index orphans:
          - Scan Archiv/ for files not present in the KB
          - Derive metadata from folder path + filename
          - OCR + clean if no text file exists in document_memory
          - Ingest into the vector store

        Returns a summary dict.
        """
        kb = self._load_knowledge_base()

        archive_index: dict[str, tuple[str, Path]] = {}
        if self._archive.is_dir():
            for cat_key, cat_folder in CATEGORY_FOLDERS.items():
                cat_dir = self._archive / cat_folder
                if not cat_dir.is_dir():
                    continue
                for fp in cat_dir.rglob("*"):
                    if fp.is_file() and fp.name not in _IGNORED_FILES and not fp.name.startswith("._"):
                        archive_index[fp.name] = (cat_key, fp)

        fixed_entries = 0
        pruned = 0
        cleaned_kb: list[dict[str, Any]] = []
        for entry in kb:
            if entry.get("type") == "correction":
                cleaned_kb.append(entry)
                continue
            fn = entry.get("filename", "")
            if not fn:
                continue
            if fn not in archive_index:
                pruned += 1
                print(f"Henry repair: pruned stale KB entry for missing file {fn}", flush=True)
                continue

            cat_key, file_path = archive_index[fn]
            changed = False

            if not entry.get("category") or entry.get("category") == "Unknown":
                entry["category"] = cat_key
                changed = True

            text_path = entry.get("full_text_path", "")
            if text_path and not Path(text_path).is_file():
                new_path = self._doc_memory / f"{Path(fn).stem}.txt"
                if new_path.is_file():
                    entry["full_text_path"] = str(new_path)
                else:
                    entry["full_text_path"] = ""
                changed = True

            if changed:
                fixed_entries += 1
                print(f"Henry repair: fixed KB entry for {fn}", flush=True)

            cleaned_kb.append(entry)

        known_files = {e.get("filename") for e in cleaned_kb if e.get("filename")}

        repaired: list[str] = []
        skipped: list[str] = []

        for filename, (cat_key, file_path) in archive_index.items():
            if filename in known_files:
                continue

            cat_dir = self._archive / CATEGORY_FOLDERS[cat_key]
            rel = file_path.relative_to(cat_dir)
            parts = rel.parts
            year = parts[0] if len(parts) > 1 else "Unknown"
            month = parts[1] if len(parts) > 2 else "Unknown"

            doc_type = "Unknown"
            provider = ""
            date = "Unknown"
            stem_parts = file_path.stem.split("_")
            if stem_parts and re.match(r"\d{4}-\d{2}-\d{2}$", stem_parts[0]):
                date = stem_parts[0]
            if len(stem_parts) >= 2:
                doc_type = stem_parts[1]
            if len(stem_parts) >= 3:
                provider = stem_parts[2]

            meta: dict[str, str] = {
                "category": cat_key,
                "document_type": doc_type,
                "provider": provider,
                "date": date,
                "year": year,
                "month": month,
                "summary": "Re-indexed by repair function",
            }

            doc_mem_text = ""
            txt_name = f"{file_path.stem}.txt"
            doc_mem_file = self._doc_memory / txt_name
            if doc_mem_file.is_file():
                try:
                    doc_mem_text = doc_mem_file.read_text(encoding="utf-8")
                except OSError:
                    pass

            if not doc_mem_text and file_path.suffix.lower() in _SUPPORTED_EXTS:
                try:
                    raw = self.extract_text(file_path)
                    if raw:
                        doc_mem_text = _clean_ocr_text(raw)
                        self._save_to_document_memory(file_path.name, doc_mem_text)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Repair OCR failed for %s: %s", file_path.name, exc)
                    skipped.append(file_path.name)
                    continue

            full_text_path = str(doc_mem_file) if doc_mem_file.is_file() else ""
            cleaned_kb.append({
                "filename": filename,
                "category": cat_key,
                "document_type": doc_type,
                "provider": provider,
                "date": date,
                "summary": "Re-indexed by repair function",
                "full_text_path": full_text_path,
                "archived_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            })

            if doc_mem_text:
                self._ingest_to_memory(file_path.name, doc_mem_text, meta)

            repaired.append(file_path.name)
            print(
                f"Henry repair: re-indexed {file_path.name} "
                f"(category={cat_key}, type={doc_type})",
                flush=True,
            )

        self._save_knowledge_base(cleaned_kb)

        total_actions = fixed_entries + pruned + len(repaired)
        result = {
            "fixed_entries": fixed_entries,
            "pruned": pruned,
            "repaired": len(repaired),
            "skipped": len(skipped),
            "files": repaired,
            "skipped_files": skipped,
        }
        if total_actions:
            print(
                f"Henry repair complete: {fixed_entries} entries fixed, "
                f"{pruned} stale entries pruned, {len(repaired)} orphan file(s) re-indexed, "
                f"{len(skipped)} skipped.",
                flush=True,
            )
        else:
            print("Henry repair: knowledge base is healthy. Nothing to fix.", flush=True)
        return result
