#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
BACKEND_DIR="$ROOT_DIR/backend"
FRONTEND_DIR="$ROOT_DIR/frontend"
BACKEND_LOG="$ROOT_DIR/backend_server.log"
BACKEND_PID_FILE="$ROOT_DIR/backend_server.pid"

cleanup() {
  if [ -f "$BACKEND_PID_FILE" ]; then
    PID_VALUE="$(cat "$BACKEND_PID_FILE")"
    if ps -p "$PID_VALUE" >/dev/null 2>&1; then
      echo "[all] stopping backend pid $PID_VALUE" && kill "$PID_VALUE" >/dev/null 2>&1 || true
    fi
    rm -f "$BACKEND_PID_FILE"
  fi
}

trap cleanup EXIT

cd "$BACKEND_DIR"
echo "[all] preparing backend" && python3 -m venv .venv
echo "[all] activating backend virtualenv" && source .venv/bin/activate
echo "[all] upgrading pip" && python -m pip install --upgrade pip
echo "[all] installing backend requirements" && pip install -r requirements.txt
echo "[all] starting backend in background on http://127.0.0.1:8431" && nohup uvicorn app.main:app --host 127.0.0.1 --port 8431 > "$BACKEND_LOG" 2>&1 &
BACKEND_PID="$!"
echo "$BACKEND_PID" > "$BACKEND_PID_FILE"

cd "$FRONTEND_DIR"
echo "[all] installing frontend packages" && npm install
echo "[all] opening frontend on http://127.0.0.1:4317" && VITE_API_BASE_URL="http://127.0.0.1:8431" npm run dev -- --host 127.0.0.1 --port 4317
