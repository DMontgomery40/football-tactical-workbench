#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
BACKEND_DIR="$ROOT_DIR/backend"
FRONTEND_DIR="$ROOT_DIR/frontend"
BACKEND_LOG="$ROOT_DIR/backend_server.log"
BACKEND_PID_FILE="$ROOT_DIR/backend_server.pid"
LAN_IP="${LAN_IP:-}"

if [ -z "$LAN_IP" ]; then
  LAN_IP="$(ipconfig getifaddr en0 2>/dev/null || true)"
fi

if [ -z "$LAN_IP" ]; then
  LAN_IP="$(ipconfig getifaddr en1 2>/dev/null || true)"
fi

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
echo "[all] starting backend in background on http://0.0.0.0:8431" && nohup uvicorn app.main:app --host 0.0.0.0 --port 8431 > "$BACKEND_LOG" 2>&1 &
BACKEND_PID="$!"
echo "$BACKEND_PID" > "$BACKEND_PID_FILE"

cd "$FRONTEND_DIR"
echo "[all] installing frontend packages" && npm install
if [ -n "$LAN_IP" ]; then
  echo "[all] open the UI from this Mac at http://127.0.0.1:4317"
  echo "[all] open the UI from another device at http://$LAN_IP:4317"
  echo "[all] backend will be reachable on this Mac at http://127.0.0.1:8431 and on the LAN at http://$LAN_IP:8431"
else
  echo "[all] open the UI locally at http://127.0.0.1:4317"
  echo "[all] backend will listen on port 8431 on all interfaces"
fi
npm run dev -- --host 0.0.0.0 --port 4317
