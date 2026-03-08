#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"
echo "[backend] root: $ROOT_DIR" && python3 -m venv .venv
echo "[backend] activating virtualenv" && source .venv/bin/activate
echo "[backend] upgrading pip" && python -m pip install --upgrade pip
echo "[backend] installing requirements" && pip install -r requirements.txt
echo "[backend] starting FastAPI on http://0.0.0.0:8431 (LAN-accessible API server)" && uvicorn app.main:app --reload --host 0.0.0.0 --port 8431
