#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

if [[ "${VITE_API_BASE_URL:-}" == "http://127.0.0.1:8431" || "${VITE_API_BASE_URL:-}" == "http://localhost:8431" ]]; then
  echo "[frontend] removing loopback VITE_API_BASE_URL so LAN clients use this machine's host on port 8431"
  unset VITE_API_BASE_URL
fi

echo "[frontend] root: $ROOT_DIR"
echo "[frontend] installing packages" && npm install
echo "[frontend] backend default: browser hostname on port 8431 unless VITE_API_BASE_URL is set"
echo "[frontend] starting Vite on http://0.0.0.0:4317 (LAN-accessible dev server)" && npm run dev -- --host 0.0.0.0 --port 4317
