# Projekt Henry I

Henry is a local-first document assistant: OCR and filing via a document worker, LLM routing and memory on a headless Core API, and Telegram as a separate UI satellite.

## Architecture

| Service | Command | Port | Role |
|---------|---------|------|------|
| **Core** | `python main.py` | 8000 | LLM routing, Chroma/SQLite memory, document classification, chat |
| **Document worker** | `python tools/file_manager.py` | 8001 | Inbox watcher, OCR, physical archive moves |
| **Telegram UI** | `python tools/telegram_ui.py` | 8002 | Bot + inbound proposal notifications from Core |

All services communicate over HTTP on `localhost`.

## Quick start

Open **three terminals**, each from the **project root**:

```bash
cd /path/to/Projekt-Henry-I

# Terminal 1 — Core
python main.py

# Terminal 2 — Document worker
python tools/file_manager.py

# Terminal 3 — Telegram UI
python tools/telegram_ui.py
```

Mount the encrypted **HenryData** volume before starting Core if you want document and personal memory enabled.

Drop PDFs or images into `HENRY_FILES_ROOT/01_Eingang_OCR/` (default: `~/Desktop/Henry Files/01_Eingang_OCR/`).

---

## Environment (`.env`)

### Where is the file?

Configuration lives in a single file at the **repository root** (next to `main.py`):

```
Projekt-Henry-I/
├── .env              ← your secrets (never commit)
├── .env.example      ← template (safe to commit)
├── main.py
└── tools/
    ├── file_manager.py
    └── telegram_ui.py
```

`.env` is listed in `.gitignore`. If it is missing, create it from the template:

```bash
cp .env.example .env
```

Then edit `.env` with your real paths, tokens, and API keys.

### One file, three processes

Core, the document worker, and the Telegram UI all call `load_dotenv()` at startup. They read the **same** `.env` file. You do **not** need a separate env file per service for local development.

Each process only uses a subset of the variables (see table below).

### How `.env` is loaded

All entry points call `load_project_env()` from `core/env.py`, which loads:

```
Projekt-Henry-I/.env
```

using a path relative to the repository root—not the shell’s current working directory. You can start services from any folder, for example:

```bash
python /path/to/Projekt-Henry-I/main.py
python /path/to/Projekt-Henry-I/tools/file_manager.py
```

Starting from the project root is still recommended so relative paths in logs and tooling stay predictable.

### Variables by service

| Variable | Core | Worker | Telegram UI |
|----------|:----:|:------:|:-----------:|
| `MEMORY_MOUNT_PATH` | ✓ | | |
| `ARCHIVE_DB_PATH` | ✓ | | |
| `PERSONAL_MEMORY_PATH` | ✓ | | |
| `OLLAMA_MODEL`, `OLLAMA_BASE_URL` | ✓ | | |
| `HENRY_CLASSIFY_MODEL` | ✓ | | |
| `HENRY_EMBED_MODEL` | ✓ | | |
| `TAVILY_API_KEY` | ✓ | | |
| `OPENAI_API_KEY`, `ANTHROPIC_API_KEY` | ✓ (if used) | | |
| `HENRY_FILES_ROOT` | | ✓ | ✓ (`/purge` only) |
| `TESSERACT_CMD` | | ✓ | |
| `HENRY_CORE_API_URL` | | ✓ | ✓ |
| `HENRY_WORKER_API_URL` | ✓ | | |
| `HENRY_UI_API_URL` | ✓ | | |
| `TELEGRAM_BOT_TOKEN` | | | ✓ |
| `AUTHORIZED_USER_ID` | | | ✓ |
| `HENRY_API_HOST`, `HENRY_API_PORT` | ✓ | | |
| `HENRY_WORKER_API_HOST`, `HENRY_WORKER_API_PORT` | | ✓ | |
| `HENRY_UI_API_HOST`, `HENRY_UI_API_PORT` | | | ✓ |

### Local URL defaults

If these are omitted, services default to:

| Setting | Default |
|---------|---------|
| Core API | `http://127.0.0.1:8000` |
| Document worker API | `http://127.0.0.1:8001` |
| Telegram UI API | `http://127.0.0.1:8002` |
| Ollama | `http://127.0.0.1:11434` |
| HenryData mount | `/Volumes/HenryData` |
| Files root | `~/Desktop/Henry Files` |

### Minimum `.env` for the new setup

If you already had a `.env` from the monolith, keep your existing values and ensure these lines exist (see `.env.example`):

```env
# Telegram UI
TELEGRAM_BOT_TOKEN=...
AUTHORIZED_USER_ID=...

# Microservice URLs (single machine)
HENRY_CORE_API_URL=http://127.0.0.1:8000
HENRY_WORKER_API_URL=http://127.0.0.1:8001
HENRY_UI_API_URL=http://127.0.0.1:8002

# Files + memory
HENRY_FILES_ROOT=/path/to/Henry Files
MEMORY_MOUNT_PATH=/Volumes/HenryData
ARCHIVE_DB_PATH=/Volumes/HenryData/archive
PERSONAL_MEMORY_PATH=/Volumes/HenryData/personal_memory.db

# LLM
OLLAMA_MODEL=qwen2.5:7b
OLLAMA_BASE_URL=http://127.0.0.1:11434
```

### Security notes

- **Never commit** `.env` or put tokens in code.
- Document memory and personal facts are stored only on the encrypted **HenryData** volume when it is mounted. If the volume is missing, Core runs in flight mode and skips memory writes.
- The document worker does not open Chroma or SQLite directly; it only talks to Core over HTTP.

### Docker / OrbStack (later)

When you containerize services, pass the same variables via `env_file: .env` or orchestrator secrets. The loader in `core/env.py` can be extended to read from the container filesystem path you mount for `.env`.

---

## Install

```bash
pip install -r requirements.txt
```

Core and the worker need OCR tools on the host (`tesseract`, `poppler` for PDFs) when using the document pipeline.

---

## Health checks

```bash
curl http://127.0.0.1:8000/status   # Core
curl http://127.0.0.1:8001/status   # Document worker
curl http://127.0.0.1:8002/status   # Telegram UI
```
