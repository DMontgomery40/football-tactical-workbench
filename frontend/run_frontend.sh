#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"
echo "[frontend] root: $ROOT_DIR"
echo "[frontend] installing packages" && npm install
echo "[frontend] starting Vite on http://0.0.0.0:4317 (LAN-accessible dev server)" && npm run dev -- --host 0.0.0.0 --port 4317
