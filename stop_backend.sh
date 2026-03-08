#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
PID_FILE="$ROOT_DIR/backend_server.pid"
if [ ! -f "$PID_FILE" ]; then
  echo "[stop] no backend pid file found" && exit 0
fi
PID_VALUE="$(cat "$PID_FILE")"
if ps -p "$PID_VALUE" >/dev/null 2>&1; then
  echo "[stop] stopping backend pid $PID_VALUE" && kill "$PID_VALUE"
else
  echo "[stop] backend pid $PID_VALUE is not running"
fi
rm -f "$PID_FILE"
