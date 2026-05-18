#!/usr/bin/env bash
# Henry native process orchestrator — Core (8000), Worker (8001), Telegram UI (8002).
set -euo pipefail

# --- Paths (resolved from this script's location) ------------------------------
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_ROOT="${SCRIPT_DIR}"
readonly PID_FILE="${PROJECT_ROOT}/.henry.pids"
readonly LOG_CORE="${PROJECT_ROOT}/logs_core.log"
readonly LOG_WORKER="${PROJECT_ROOT}/logs_worker.log"
readonly LOG_UI="${PROJECT_ROOT}/logs_ui.log"

readonly MAIN_PY="${PROJECT_ROOT}/main.py"
readonly WORKER_PY="${PROJECT_ROOT}/tools/file_manager.py"
readonly UI_PY="${PROJECT_ROOT}/tools/telegram_ui.py"

readonly VENV_DIR="${PROJECT_ROOT}/.venv"
readonly VENV_PYTHON="${VENV_DIR}/bin/python3"
readonly REQUIREMENTS="${PROJECT_ROOT}/requirements.txt"

# --- Helpers -----------------------------------------------------------------
log_info()  { printf '📘 %s\n' "$*"; }
log_ok()    { printf '✅ %s\n' "$*"; }
log_warn()  { printf '⚠️  %s\n' "$*" >&2; }
log_err()   { printf '❌ %s\n' "$*" >&2; }

resolve_python() {
  if [[ -x "${VENV_PYTHON}" ]]; then
    echo "${VENV_PYTHON}"
    return 0
  fi
  if [[ -x "${VENV_DIR}/bin/python" ]]; then
    echo "${VENV_DIR}/bin/python"
    return 0
  fi
  log_err "Project virtualenv not found: ${VENV_PYTHON}"
  log_err "Create it and install deps with: ./henry.sh setup"
  return 1
}

verify_venv_deps() {
  local python_bin="$1"
  if "${python_bin}" -c "import httpx, dotenv" 2>/dev/null; then
    return 0
  fi
  log_err "Required packages missing in .venv (httpx, python-dotenv, …)."
  log_err "Install them with: ./henry.sh setup"
  return 1
}

pid_alive() {
  local pid="$1"
  [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null
}

# Populated by read_pid_file (bash 3.2 on macOS has no `local -n` namerefs).
HENRY_PID_CORE=""
HENRY_PID_WORKER=""
HENRY_PID_UI=""

read_pid_file() {
  HENRY_PID_CORE=""
  HENRY_PID_WORKER=""
  HENRY_PID_UI=""

  if [[ ! -f "${PID_FILE}" ]]; then
    return 1
  fi

  while IFS= read -r line || [[ -n "${line}" ]]; do
    line="${line%%#*}"
    line="$(echo "${line}" | tr -d '[:space:]')"
    [[ -z "${line}" ]] && continue
    case "${line}" in
      core:*|CORE=*)
        HENRY_PID_CORE="${line#*:}"
        HENRY_PID_CORE="${HENRY_PID_CORE#CORE=}"
        ;;
      worker:*|WORKER=*)
        HENRY_PID_WORKER="${line#*:}"
        HENRY_PID_WORKER="${HENRY_PID_WORKER#WORKER=}"
        ;;
      ui:*|UI=*)
        HENRY_PID_UI="${line#*:}"
        HENRY_PID_UI="${HENRY_PID_UI#UI=}"
        ;;
      *)
        if [[ -z "${HENRY_PID_CORE}" ]]; then HENRY_PID_CORE="${line}"
        elif [[ -z "${HENRY_PID_WORKER}" ]]; then HENRY_PID_WORKER="${line}"
        elif [[ -z "${HENRY_PID_UI}" ]]; then HENRY_PID_UI="${line}"
        fi
        ;;
    esac
  done < "${PID_FILE}"

  [[ -n "${HENRY_PID_CORE}" && -n "${HENRY_PID_WORKER}" && -n "${HENRY_PID_UI}" ]]
}

write_pid_file() {
  local core_pid="$1" worker_pid="$2" ui_pid="$3"
  cat > "${PID_FILE}" <<EOF
core:${core_pid}
worker:${worker_pid}
ui:${ui_pid}
EOF
}

require_project_files() {
  local missing=0
  for f in "${MAIN_PY}" "${WORKER_PY}" "${UI_PY}"; do
    if [[ ! -f "${f}" ]]; then
      log_err "Missing required file: ${f}"
      missing=1
    fi
  done
  return "${missing}"
}

warn_stray_henry_processes() {
  local known_pids=" "
  if read_pid_file; then
    known_pids=" ${HENRY_PID_CORE} ${HENRY_PID_WORKER} ${HENRY_PID_UI} "
  fi

  local matches stray line pid
  matches="$(pgrep -fl "${MAIN_PY}|${WORKER_PY}|${UI_PY}" 2>/dev/null || true)"
  stray=""

  while IFS= read -r line; do
    [[ -z "${line}" ]] && continue
    pid="${line%% *}"

    case "${known_pids}" in
      *" ${pid} "*) continue ;;
    esac

    stray+="${line}"$'\n'
  done <<< "${matches}"

  if [[ -n "${stray}" ]]; then
    log_warn "Stray Henry Python processes detected (not in ${PID_FILE}):"
    while IFS= read -r line; do
      [[ -n "${line}" ]] && printf '   %s\n' "${line}" >&2
    done <<< "${stray}"
    log_warn "These can cause Telegram getUpdates conflicts. Stop them with: kill <PID>"
    log_warn "Also check OrbStack/Docker containers using the same TELEGRAM_BOT_TOKEN."
  fi
}

terminate_pid() {
  local label="$1" pid="$2"

  if [[ -z "${pid}" ]]; then
    log_warn "${label}: no PID recorded — skipping."
    return 0
  fi

  if ! pid_alive "${pid}"; then
    log_warn "${label} (PID ${pid}): already stopped."
    return 0
  fi

  log_info "Stopping ${label} (PID ${pid})…"
  kill -TERM "${pid}" 2>/dev/null || true

  local i
  for i in {1..20}; do
    if ! pid_alive "${pid}"; then
      log_ok "${label} (PID ${pid}) terminated."
      return 0
    fi
    sleep 0.25
  done

  log_warn "${label} (PID ${pid}): still running — sending SIGKILL."
  kill -KILL "${pid}" 2>/dev/null || true
  sleep 0.5

  if pid_alive "${pid}"; then
    log_err "${label} (PID ${pid}): failed to stop."
    return 1
  fi
  log_ok "${label} (PID ${pid}) force-stopped."
}

# --- Commands ----------------------------------------------------------------
cmd_start() {
  cd "${PROJECT_ROOT}"

  if [[ -f "${PID_FILE}" ]]; then
    if read_pid_file; then
      if pid_alive "${HENRY_PID_CORE}" || pid_alive "${HENRY_PID_WORKER}" || pid_alive "${HENRY_PID_UI}"; then
        log_warn "Henry appears to be already running (found ${PID_FILE})."
        log_warn "Run: ./henry.sh stop   — or   ./henry.sh restart"
        exit 1
      fi
      log_warn "Stale ${PID_FILE} found (processes not running). Removing and continuing."
      rm -f "${PID_FILE}"
    else
      log_warn "Corrupt ${PID_FILE}. Remove it manually or run ./henry.sh stop"
      exit 1
    fi
  fi

  require_project_files || exit 1

  local python_bin
  python_bin="$(resolve_python)" || exit 1
  verify_venv_deps "${python_bin}" || exit 1
  log_info "Using Python: ${python_bin}"
  export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

  : > "${LOG_CORE}"
  : > "${LOG_WORKER}"
  : > "${LOG_UI}"

  log_info "🚀 Starting Henry Core API (port 8000)…"
  nohup env PYTHONUNBUFFERED=1 "${python_bin}" -u "${MAIN_PY}" >> "${LOG_CORE}" 2>&1 &
  local core_pid=$!

  log_info "🚀 Starting Document Worker (port 8001)…"
  nohup env PYTHONUNBUFFERED=1 "${python_bin}" -u "${WORKER_PY}" >> "${LOG_WORKER}" 2>&1 &
  local worker_pid=$!

  log_info "🚀 Starting Telegram UI (port 8002)…"
  nohup env PYTHONUNBUFFERED=1 "${python_bin}" -u "${UI_PY}" >> "${LOG_UI}" 2>&1 &
  local ui_pid=$!

  sleep 0.5

  local failed=0 entry label pid
  for entry in "Core:${core_pid}" "Worker:${worker_pid}" "UI:${ui_pid}"; do
    label="${entry%%:*}"
    pid="${entry#*:}"
    if ! pid_alive "${pid}"; then
      log_err "${label} exited immediately (PID ${pid}). Check logs."
      failed=1
    fi
  done

  if (( failed )); then
    terminate_pid "Core" "${core_pid}" || true
    terminate_pid "Worker" "${worker_pid}" || true
    terminate_pid "UI" "${ui_pid}" || true
    exit 1
  fi

  write_pid_file "${core_pid}" "${worker_pid}" "${ui_pid}"
  log_ok "Henry is running."
  log_info "  Core   PID ${core_pid}  → ${LOG_CORE}"
  log_info "  Worker PID ${worker_pid}  → ${LOG_WORKER}"
  log_info "  UI     PID ${ui_pid}  → ${LOG_UI}"
  log_info "Tail logs: ./henry.sh logs"
  warn_stray_henry_processes
}

cmd_stop() {
  cd "${PROJECT_ROOT}"

  if [[ ! -f "${PID_FILE}" ]]; then
    log_err "Henry is not running (no ${PID_FILE})."
    exit 1
  fi

  if ! read_pid_file; then
    log_err "Could not parse ${PID_FILE}. Remove it manually if processes are gone."
    exit 1
  fi

  log_info "🛑 Stopping Henry services…"
  local err=0
  terminate_pid "Core"   "${HENRY_PID_CORE}"   || err=1
  terminate_pid "Worker" "${HENRY_PID_WORKER}" || err=1
  terminate_pid "UI"     "${HENRY_PID_UI}"     || err=1

  if (( err != 0 )); then
    log_err "Some processes could not be stopped. ${PID_FILE} kept for inspection."
    exit 1
  fi

  rm -f "${PID_FILE}"
  log_ok "Henry stopped. ${PID_FILE} removed."
  warn_stray_henry_processes
}

cmd_restart() {
  log_info "🔄 Restarting Henry…"
  if [[ -f "${PID_FILE}" ]]; then
    cmd_stop
  else
    log_info "No running instance detected — starting fresh."
  fi
  log_info "Waiting 1s for ports 8000, 8001, 8002 to clear…"
  sleep 1
  cmd_start
}

cmd_setup() {
  cd "${PROJECT_ROOT}"

  if [[ ! -f "${REQUIREMENTS}" ]]; then
    log_err "Missing ${REQUIREMENTS}"
    exit 1
  fi

  if [[ ! -x "${VENV_PYTHON}" ]]; then
    log_info "Creating virtual environment at ${VENV_DIR}…"
    if ! command -v python3 >/dev/null 2>&1; then
      log_err "python3 is required to bootstrap .venv (PEP 668 blocks global pip)."
      exit 1
    fi
    python3 -m venv "${VENV_DIR}"
  fi

  local python_bin
  python_bin="$(resolve_python)" || exit 1

  log_info "Upgrading pip inside .venv…"
  "${python_bin}" -m pip install --upgrade pip

  log_info "Installing dependencies from requirements.txt…"
  "${python_bin}" -m pip install -r "${REQUIREMENTS}"

  verify_venv_deps "${python_bin}" || exit 1
  log_ok "Virtual environment ready at ${python_bin}"
  log_info "Start Henry with: ./henry.sh start"
}

cmd_logs() {
  cd "${PROJECT_ROOT}"

  touch "${LOG_CORE}" "${LOG_WORKER}" "${LOG_UI}"

  log_info "📜 Live unified logs (Ctrl+C to exit)…"
  log_info "  ${LOG_CORE}"
  log_info "  ${LOG_WORKER}"
  log_info "  ${LOG_UI}"
  echo ""

  # macOS/BSD tail prefixes each file section when following multiple files.
  tail -f "${LOG_CORE}" "${LOG_WORKER}" "${LOG_UI}"
}

cmd_doctor() {
  local status_url="http://127.0.0.1:8000/api/status"

  if ! command -v curl >/dev/null 2>&1; then
    log_err "curl is required for ./henry.sh doctor"
    exit 1
  fi

  log_info "🩺 Querying Henry health matrix: ${status_url}"

  local payload
  if ! payload="$(curl --fail --silent --show-error --max-time 5 "${status_url}")"; then
    log_err "Henry Core health endpoint is not reachable."
    log_err "Start Henry first with: ./henry.sh start"
    exit 1
  fi

  if command -v jq >/dev/null 2>&1; then
    printf '%s\n' "${payload}" | jq .
    return 0
  fi

  if command -v python3 >/dev/null 2>&1; then
    printf '%s\n' "${payload}" | python3 -m json.tool
    return 0
  fi

  log_warn "Neither jq nor python3 found; printing raw JSON."
  printf '%s\n' "${payload}"
}

usage() {
  cat <<EOF
Henry process control (native, no Docker)

Usage: $(basename "$0") <command>

Commands:
  setup    Create/update .venv and pip install -r requirements.txt
  start    Launch Core, Worker, and Telegram UI (uses ${VENV_PYTHON})
  stop     Stop the three recorded PIDs and remove ${PID_FILE}
  restart  stop → wait 1s → start
  logs     Follow logs_core.log, logs_worker.log, logs_ui.log together
  doctor   Query Core /api/status and pretty-print the health matrix

Python: services always run via ${VENV_PYTHON} (never global python3).

Project root: ${PROJECT_ROOT}
EOF
}

main() {
  local cmd="${1:-}"
  case "${cmd}" in
    setup)   cmd_setup ;;
    start)   cmd_start ;;
    stop)    cmd_stop ;;
    restart) cmd_restart ;;
    logs)    cmd_logs ;;
    doctor)  cmd_doctor ;;
    -h|--help|help)
      usage
      exit 0
      ;;
    "")
      usage
      exit 1
      ;;
    *)
      log_err "Unknown command: ${cmd}"
      usage
      exit 1
      ;;
  esac
}

main "$@"
