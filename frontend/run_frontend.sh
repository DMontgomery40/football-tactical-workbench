#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"
echo "[frontend] root: $ROOT_DIR"
echo "[frontend] installing packages" && npm install
echo "[frontend] starting Vite on http://127.0.0.1:4317" && VITE_API_BASE_URL="http://127.0.0.1:8431" npm run dev -- --host 127.0.0.1 --port 4317
