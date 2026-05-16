"""Autonomous document worker — inbox watcher, OCR, and physical archiving.

Standalone service: POST OCR text to the Henry Core API; receive archive commands
via a local inbound API. No Telegram, LLM routing, or direct database access.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Literal

from core.env import load_project_env
from core.port_guard import reclaim_tcp_listen_port

load_project_env()

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
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

_DEFAULT_ROOT = os.environ.get(
    "HENRY_FILES_ROOT",
    str(Path.home() / "Desktop" / "Henry Files"),
).strip()

_CORE_API_URL = os.environ.get("HENRY_CORE_API_URL", "http://127.0.0.1:8000").rstrip("/")
_WORKER_API_HOST = os.environ.get("HENRY_WORKER_API_HOST", "127.0.0.1")
_WORKER_API_PORT = int(os.environ.get("HENRY_WORKER_API_PORT", "8001"))

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
    "STEP 3 — SMART GROUPING (optional): Populate key \"grouping_special\" ONLY when "
    "a recurring document type would clearly benefit from a dedicated folder "
    '(e.g. "Insurance Policies", "Steuerunterlagen"). Use empty string \"\" when '
    'the default date ladder [Year]/[Month]/[Day] under the Domain is appropriate.\n\n'
    "STEP 4 — PROVIDER / COUNTERPARTY: The company or person who sent/issued this document. "
    "Look in the header, letterhead, sender address, or signature block. "
    "If you see Swiss company names or specific addresses, use them. "
    'If you truly cannot identify the provider, use an empty string "".\n\n'
    "STEP 5 — DATE: The document date in YYYY-MM-DD format. "
    "Swiss DD.MM.YYYY must be converted. Use the document date, not due dates.\n\n"
    "Return ONLY a valid JSON object — no markdown, no code fences, no extra text.\n"
    "The JSON must have exactly these keys:\n"
    '  "category": one of the 7 categories above (e.g. "Wohnen", "Finanzen")\n'
    '  "document_type": the specific type (e.g. "Mietvertrag", "Rechnung")\n'
    '  "provider": company/person name, or "" if unsure\n'
    '  "date": YYYY-MM-DD or "Unknown"\n'
    '  "year": YYYY or "Unknown"\n'
    '  "month": MM or "Unknown"\n'
    '  "grouping_special": recurring-folder suggestion or "" — never invent new Domains '
    '("01_Wohnen" … "07_Korrespondenz");\n'
    '  "summary": one-sentence summary\n\n'
    "IMPORTANT:\n"
    "- Even if you cannot identify the provider, you MUST still assign "
    "a category and document_type.\n"
    '- For formal contracts ("Vertrag", "Contract"), classify correctly (e.g. '
    "\"Mietvertrag\"→Wohnen, \"Arbeitsvertrag\"→Arbeit) — Henry will archive under "
    "Contracts/<Entity> automatically.\n"
    "- NEVER assign a synthetic 08_* domain folder; if nothing fits reasonably, "
    "pick the closest of the seven and mention the uncertainty in summary."
)

_FALLBACK_META: dict[str, str] = {
    "category": "Unknown",
    "document_type": "Unknown",
    "provider": "",
    "date": "Unknown",
    "year": "Unknown",
    "month": "Unknown",
    "summary": "Unknown",
    "grouping_special": "",
}

_META_KEYS = tuple(_FALLBACK_META.keys())

# Smart grouping overrides (prior to default date ladder). Lohn beats generic "Vertrag"-like types.
_CONTRACT_DOCUMENT_MARKERS = (
    "vertrag",
    "contract",
    "mietvertrag",
    "arbeitsvertrag",
    "kaufvertrag",
    "leasingvertrag",
    "nebeneintragung",
    "nebeneinbarung",
    "unterzeichneter vertrag",
)


def _is_lohnabrechnung(meta: dict[str, str]) -> bool:
    """Payslips use 04_Arbeit/Lohnabrechnungen/[Year]/[Month] (no day layer)."""
    dt = meta.get("document_type", "").lower()
    if any(x in dt for x in ("lohnabrechnung", "gehaltsabrechnung", "salary slip", "payroll")):
        return True
    return "lohn" in dt and "abrechnung" in dt


def _is_contract_document(meta: dict[str, str]) -> bool:
    """Contracts use [Domain]/Contracts/[Entity] — no date path."""
    if _is_lohnabrechnung(meta):
        return False
    dt = meta.get("document_type", "").lower()
    return any(marker in dt for marker in _CONTRACT_DOCUMENT_MARKERS)


def _is_under_tree(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    else:
        return True


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
        fallback = "" if key == "grouping_special" else "Unknown"
        val = str(obj.get(key, fallback)).strip()
        if key == "grouping_special":
            result[key] = "" if val in ("Unknown", "") else val
        else:
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

    gs = result.get("grouping_special", "")
    if gs in ("Unknown", "", None):
        result["grouping_special"] = ""
    elif _is_contract_document(result):
        # Contracts use Contracts/<Entity>; date grouping is suppressed.
        result["grouping_special"] = ""

    return result


# --- Worker API models (inbound from Core) -----------------------------------


class ArchiveExecuteCommand(BaseModel):
    """Payload from Core after user approval — triggers physical file move."""

    pending_id: str
    action: Literal["confirm", "edit"]
    metadata_override: dict[str, str] | None = None
    classification: dict[str, str] | None = Field(
        default=None,
        description="Final metadata from Core (category, document_type, provider, date, year, month)",
    )


class _PendingRecord:
    """Operational memory entry for a file awaiting archive approval."""

    __slots__ = ("pending_id", "staged_path", "inbox_path", "filename", "raw_text")

    def __init__(
        self,
        pending_id: str,
        staged_path: Path,
        inbox_path: Path,
        filename: str,
        raw_text: str,
    ) -> None:
        self.pending_id = pending_id
        self.staged_path = staged_path
        self.inbox_path = inbox_path
        self.filename = filename
        self.raw_text = raw_text


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
    Autonomous document worker: Watchdog + OCR + operational memory + physical archive.

    Visible folders under *root* (default ``~/Desktop/Henry Files``):
        01_Eingang_OCR/  — watched inbox
        Archiv/          — category archive tree
        internal/        — transient temp + pending staging only
    """

    def __init__(self, root: Path | str | None = None) -> None:
        self._root = Path(root or _DEFAULT_ROOT or ".").resolve()

        self.inbox = self._root / INBOX_FOLDER
        self._manual_review = self.inbox / "manual_review"

        self._archive = self._root / "Archiv"
        self._categories: dict[str, Path] = {}
        for cat, folder in CATEGORY_FOLDERS.items():
            self._categories[cat] = self._archive / folder

        self._internal = self._root / "internal"
        self._temp = self._internal / "temp"
        self._pending = self._internal / "pending"

        self._ensure_dirs()

        self._observer: Observer | None = None
        self._pending_lock = threading.Lock()
        self._pending_files: dict[str, str] = {}
        self._pending_records: dict[str, _PendingRecord] = {}

    def _ensure_dirs(self) -> None:
        dirs = [
            self.inbox,
            self._manual_review,
            self._archive,
            *self._categories.values(),
            self._internal,
            self._temp,
            self._pending,
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)

    # --- Inbox watcher -----------------------------------------------------

    def start_watching(self) -> None:
        """Begin monitoring inbox and manual_review (non-blocking)."""
        if self._observer is not None:
            return

        os.makedirs(self._temp, exist_ok=True)

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

    def _auto_process_pending(self) -> None:
        """Called from the watchdog thread after a file is copied to internal/temp."""
        self.process_inbox()

    # --- Processing stage --------------------------------------------------

    def _stage_to_backup(self, src: Path) -> Path | None:
        """Copy a newly detected file into internal/temp for safe processing.

        The original stays in the inbox until processing succeeds.
        """
        if not src.is_file():
            return None
        dest = self._temp / src.name
        if dest.exists():
            stem, suffix = dest.stem, dest.suffix
            dest = self._temp / f"{stem}_{int(time.time())}{suffix}"
        try:
            shutil.copy2(str(src), str(dest))
            logger.info("Staged to temp: %s", dest.name)
            return dest
        except OSError as exc:
            logger.warning("Could not copy %s to temp: %s", src.name, exc)
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

    def _register_pending(self, record: _PendingRecord) -> None:
        with self._pending_lock:
            self._pending_files[record.pending_id] = str(record.staged_path)
            self._pending_records[record.pending_id] = record

    def _get_pending(self, pending_id: str) -> _PendingRecord | None:
        with self._pending_lock:
            return self._pending_records.get(pending_id)

    def _pop_pending(self, pending_id: str) -> _PendingRecord | None:
        with self._pending_lock:
            self._pending_files.pop(pending_id, None)
            return self._pending_records.pop(pending_id, None)

    def _post_to_core(self, pending_id: str, filename: str, raw_text: str) -> dict[str, Any] | None:
        """Send OCR text to the Core API for classification."""
        url = f"{_CORE_API_URL}/v1/process"
        payload = {
            "source": "Document_Manager",
            "filename": filename,
            "raw_text": raw_text,
            "pending_id": pending_id,
        }
        try:
            with httpx.Client(timeout=120.0) as client:
                resp = client.post(url, json=payload)
                resp.raise_for_status()
                return resp.json()
        except httpx.ConnectError:
            logger.error(
                "Core API unreachable at %s — file %s kept in inbox (pending_id=%s)",
                _CORE_API_URL,
                filename,
                pending_id,
            )
            print(
                f"ERROR: Henry Core offline at {_CORE_API_URL}. "
                f"'{filename}' remains in the inbox.",
                flush=True,
            )
            return None
        except httpx.HTTPStatusError as exc:
            logger.error(
                "Core API returned HTTP %s for %s: %s",
                exc.response.status_code,
                filename,
                exc.response.text[:500],
            )
            print(
                f"ERROR: Core rejected processing of '{filename}' "
                f"(HTTP {exc.response.status_code}). File kept in inbox.",
                flush=True,
            )
            return None
        except Exception as exc:  # noqa: BLE001
            logger.error("Core API call failed for %s: %s", filename, exc)
            print(f"ERROR: Could not reach Core for '{filename}': {exc}", flush=True)
            return None

    def _cleanup_temp_files(self, filename: str) -> None:
        """Remove transient copies from internal/temp after archiving."""
        target = self._temp / filename
        try:
            if target.is_file():
                target.unlink()
                logger.info("Cleaned up %s", target)
        except OSError as exc:
            logger.debug("Cleanup failed for %s: %s", target, exc)

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
                "grouping_special": "",
            }

            dest = self.organize_file(f, meta)
            if dest is not None:
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

    @staticmethod
    def _enrich_ymd_from_date(meta: dict[str, str]) -> None:
        """Infer year/month/day from ISO date string when segmentation is missing."""
        raw = meta.get("date", "").strip()
        m_iso = re.match(r"^(\d{4})-(\d{2})-(\d{2})$", raw)
        if not m_iso:
            return
        y, mo, d_part = m_iso.group(1), m_iso.group(2), m_iso.group(3)
        if meta.get("year") in ("", "Unknown", None):
            meta["year"] = y
        if meta.get("month") in ("", "Unknown", None):
            meta["month"] = mo
        if meta.get("day") in ("", "Unknown", None):
            meta["day"] = d_part

    def _contract_entity_dirname(self, meta: dict[str, str]) -> str:
        prov = meta.get("provider", "").strip()
        if not prov:
            return self._safe_dirname("Unknown_Counterparty")
        return self._safe_dirname(prov)

    def _derive_target_archive_dir(self, meta: dict[str, str], cat_base: Path) -> Path:
        """Date-default and smart-folder rules under ``Archiv/<Domain>/``.

        Payslips **always** anchor under ``Archiv/04_Arbeit/Lohnabrechnungen/``, per project rules.
        Other layouts use *cat_base*:

        - Payslips: ``04_Arbeit/Lohnabrechnungen/[Year]/[Month]``
        - Contracts: ``Contracts/[Entity]`` (flat; no calendar folders)
        - Optional AI grouping_special: ``<Folder>/[Year]/[Month]/[Day]``
        - Default: ``[Year]/[Month]/[Day]``
        """
        HenryFileManager._enrich_ymd_from_date(meta)

        year = self._safe_dirname(meta.get("year", "Unknown"))
        month = self._safe_dirname(meta.get("month", "Unknown"))
        day = self._safe_dirname(meta.get("day", "Unknown"))

        arbeit_base = self._categories["Arbeit"]
        if _is_lohnabrechnung(meta):
            return arbeit_base / "Lohnabrechnungen" / year / month

        if _is_contract_document(meta):
            return cat_base / "Contracts" / self._contract_entity_dirname(meta)

        gs_raw = meta.get("grouping_special", "").strip()
        if gs_raw and gs_raw.lower() not in ("unknown", ""):
            group_folder = self._safe_dirname(gs_raw.replace(" ", "_"))
            if group_folder and group_folder != "Unknown":
                return cat_base / group_folder / year / month / day

        return cat_base / year / month / day

    def _resolve_destination_from_user_path(
        self,
        raw: str,
        src: Path,
        *,
        smart_name: str,
    ) -> tuple[Path | None, str | None]:
        """Resolve a chat-typed archive path beneath ``Henry root`` / ``Archiv``.

        Returns ``(destination_file_path, error_message)``. When *error_message* is
        set, *destination_file_path* is ``None``.
        """
        cleaned = raw.strip().strip("`").strip('"').strip("'")
        if not cleaned:
            return None, "empty user_destination"

        suffix = src.suffix
        treats_as_file = cleaned.lower().endswith(suffix.lower())

        normalized = cleaned.replace("\\", "/")
        while normalized.startswith("Archiv/"):
            normalized = normalized[len("Archiv/") :]

        trial_paths: list[Path] = []
        if Path(cleaned).is_absolute():
            trial_paths.append(Path(cleaned))
        else:
            trial_paths.append(self._archive / normalized)
            trial_paths.append(self._root / normalized)
            trial_paths.append(self._root / "Archiv" / normalized)

        archive_r = self._archive.resolve()
        root_r = self._root.resolve()

        anchor: Path | None = None
        for cand in trial_paths:
            resolved = cand.resolve()
            if _is_under_tree(resolved, archive_r) or _is_under_tree(resolved, root_r):
                anchor = cand
                break

        if anchor is None:
            return None, "path must stay under Henry root or Archiv"

        if treats_as_file:
            dest_file = anchor
        else:
            os.makedirs(anchor, exist_ok=True)
            fname = smart_name if smart_name else src.name
            dest_file = anchor / fname

        os.makedirs(dest_file.parent, exist_ok=True)
        return dest_file, None

    def organize_file(
        self,
        file_path: Path | str,
        meta: dict[str, str],
    ) -> Path | None:
        """
        Move *file_path* into ``Archiv`` using date-based defaults and smart grouping.

        User override: when ``meta['user_destination']`` is set (from chat), all
        filing heuristics are skipped and the file is placed exactly there
        (directory or full file path under Henry root / Archiv).

        Otherwise:
        - Default: ``<Domain>/[Year]/[Month]/[Day]/``
        - Payslips: ``04_Arbeit/Lohnabrechnungen/[Year]/[Month]/``
        - Contracts: ``<Domain>/Contracts/[Entity]/``
        - ``grouping_special``: ``<Domain>/<Group>/[Year]/[Month]/[Day]/``
        """
        src = Path(file_path)
        if not src.is_file():
            logger.warning("organize_file: source does not exist — %s", src)
            return None

        meta = dict(meta)
        HenryFileManager._enrich_ymd_from_date(meta)
        user_dest = meta.get("user_destination", "").strip()

        smart_name = self._build_smart_filename(meta, src.suffix)

        if user_dest:
            dest, err = self._resolve_destination_from_user_path(
                user_dest,
                src,
                smart_name=smart_name,
            )
            if err:
                logger.warning("organize_file: user_destination rejected — %s", err)
                return None
        else:
            category = meta.get("category", "Unknown")
            cat_base = self._categories.get(category)
            if cat_base is None:
                logger.warning("organize_file: unknown category '%s'", category)
                return None

            target_dir = self._derive_target_archive_dir(meta, cat_base)
            os.makedirs(target_dir, exist_ok=True)
            dest = target_dir / (smart_name if smart_name else src.name)

        if dest.exists():
            stem, suffix = dest.stem, dest.suffix
            dest = dest.parent / f"{stem}_{int(time.time())}{suffix}"

        try:
            shutil.move(str(src), str(dest))
            logger.info("Archived %s → %s", src.name, dest)
            return dest
        except OSError as exc:
            logger.warning("organize_file failed for %s: %s", src.name, exc)
            return None

    def list_processing(self) -> list[Path]:
        """Return files currently sitting in internal/temp awaiting processing."""
        if not self._temp.is_dir():
            return []
        return sorted(
            p for p in self._temp.iterdir()
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


    # --- Inbox processing --------------------------------------------------

    def process_inbox(self) -> list[dict[str, Any]]:
        """
        Process files in ``internal/temp``:

        1. OCR-extract text (pytesseract / pdf2image)
        2. Assign ``pending_id`` and stage under ``internal/pending/``
        3. POST raw text to Core API ``/v1/process``
        4. On failure → keep inbox file; move to manual_review only when OCR fails
        """
        results: list[dict[str, Any]] = []
        for backup_path in self.list_processing():
            filename = backup_path.name
            inbox_original = self.inbox / filename
            entry: dict[str, Any] = {"file": filename}

            try:
                text = self.extract_text(backup_path)
            except Exception as exc:  # noqa: BLE001
                logger.warning("OCR failed for %s: %s", filename, exc)
                entry["error"] = str(exc)
                if inbox_original.is_file():
                    review_dest = self._move_to_manual_review(inbox_original)
                    entry["moved_to"] = str(review_dest) if review_dest else None
                try:
                    backup_path.unlink(missing_ok=True)
                except OSError:
                    pass
                results.append(entry)
                continue

            if not text.strip():
                entry["error"] = "no_text_extracted"
                if inbox_original.is_file():
                    review_dest = self._move_to_manual_review(inbox_original)
                    entry["moved_to"] = str(review_dest) if review_dest else None
                try:
                    backup_path.unlink(missing_ok=True)
                except OSError:
                    pass
                results.append(entry)
                continue

            print(f"Henry OCR preview ({filename}): {text[:200]}", flush=True)

            pending_id = uuid.uuid4().hex[:12]
            staged_name = f"{pending_id}_{filename}"
            pending_dest = self._pending / staged_name
            try:
                shutil.move(str(backup_path), str(pending_dest))
            except OSError as exc:
                logger.warning("Could not stage %s to pending: %s", filename, exc)
                entry["error"] = "staging_failed"
                results.append(entry)
                continue

            record = _PendingRecord(
                pending_id=pending_id,
                staged_path=pending_dest,
                inbox_path=inbox_original,
                filename=filename,
                raw_text=text,
            )
            self._register_pending(record)

            core_response = self._post_to_core(pending_id, filename, text)
            if core_response is None:
                entry["error"] = "core_unreachable"
                entry["pending_id"] = pending_id
                results.append(entry)
                continue

            entry["pending_id"] = pending_id
            entry["status"] = "awaiting_confirmation"
            entry["classification"] = {
                "category": core_response.get("category"),
                "document_type": core_response.get("document_type"),
                "provider": core_response.get("provider"),
                "proposed_name": core_response.get("proposed_name"),
                "grouping_suggestion": core_response.get("grouping_suggestion") or "",
            }
            print(
                f"Henry: sent {filename} to Core (pending_id={pending_id}, "
                f"category={core_response.get('category')})",
                flush=True,
            )
            results.append(entry)

        return results

    # --- Archive execution (inbound from Core) ------------------------------

    def execute_archive(self, cmd: ArchiveExecuteCommand) -> dict[str, Any]:
        """Physically move a staged file into Archiv/ after Core approval."""
        record = self._get_pending(cmd.pending_id)
        if record is None:
            raise ValueError(f"Unknown pending_id: {cmd.pending_id}")

        if not record.staged_path.is_file():
            raise ValueError(f"Staged file missing for pending_id={cmd.pending_id}")

        meta: dict[str, str] = dict(cmd.classification or {})
        if cmd.metadata_override:
            meta.update(cmd.metadata_override)

        user_requested_path = meta.get("user_destination", "").strip()

        if cmd.action == "edit" and not meta.get("category"):
            raise ValueError("edit action requires classification metadata")

        cat_key = meta.get("category", "Unknown")
        for key, folder in CATEGORY_FOLDERS.items():
            if cat_key == folder or cat_key.lower() == key.lower():
                cat_key = key
                break
        if cat_key not in CATEGORY_FOLDERS:
            if user_requested_path:
                cat_key = "Korrespondenz"
            else:
                raise ValueError(f"Unknown category: {meta.get('category')}")

        meta["category"] = cat_key
        for field in ("document_type", "provider", "date", "year", "month", "summary"):
            meta.setdefault(field, "Unknown")
        meta.setdefault("grouping_special", "")
        if meta.get("grouping_special") in ("Unknown", None):
            meta["grouping_special"] = ""
        meta.setdefault("day", "Unknown")
        if meta.get("provider") == "Unknown":
            meta["provider"] = ""

        dest = self.organize_file(record.staged_path, meta)
        if dest is None:
            if record.inbox_path.is_file():
                self._move_to_manual_review(record.inbox_path)
            return {"file": record.filename, "error": "archive_failed"}

        self._pop_pending(cmd.pending_id)

        if record.inbox_path.is_file():
            try:
                record.inbox_path.unlink()
            except OSError as exc:
                logger.warning("Could not delete inbox original %s: %s", record.filename, exc)

        self._cleanup_temp_files(record.filename)

        print(
            f"Henry archived {record.filename} → {dest} "
            f"(pending_id={cmd.pending_id})",
            flush=True,
        )
        return {
            "file": record.filename,
            "archived_to": str(dest),
            "classification": meta,
            "pending_id": cmd.pending_id,
        }

    def get_pending_ids(self) -> list[str]:
        with self._pending_lock:
            return list(self._pending_files.keys())


def create_worker_app(manager: HenryFileManager) -> FastAPI:
    """Inbound API for archive commands from the Core Router."""
    app = FastAPI(title="Henry Document Worker", version="0.1.0")

    @app.get("/status")
    def worker_status() -> dict[str, Any]:
        return {
            "service": "henry-document-worker",
            "inbox": str(manager.inbox.resolve()),
            "pending_count": len(manager.get_pending_ids()),
            "core_api_url": _CORE_API_URL,
        }

    @app.post("/v1/archive/execute")
    def archive_execute(cmd: ArchiveExecuteCommand) -> dict[str, Any]:
        try:
            return manager.execute_archive(cmd)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            logger.exception("archive_execute failed for %s", cmd.pending_id)
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    return app


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
    )


def _run_worker_service() -> None:
    _configure_logging()

    manager = HenryFileManager()
    worker_app = create_worker_app(manager)

    import uvicorn

    reclaim_tcp_listen_port(_WORKER_API_PORT, role="worker")

    def _serve_api() -> None:
        uvicorn.run(
            worker_app,
            host=_WORKER_API_HOST,
            port=_WORKER_API_PORT,
            log_level="info",
        )

    api_thread = threading.Thread(target=_serve_api, name="henry-worker-api", daemon=True)
    api_thread.start()
    print(
        f"Henry Document Worker API: http://{_WORKER_API_HOST}:{_WORKER_API_PORT}",
        flush=True,
    )

    manager.start_watching()
    print("Henry Document Worker: inbox watcher active.", flush=True)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Henry Document Worker: shutting down.", flush=True)
        manager.stop_watching()


if __name__ == "__main__":
    _run_worker_service()
