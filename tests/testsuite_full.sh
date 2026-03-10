#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BACKEND_VENV_PY="$ROOT_DIR/backend/.venv/bin/python"

if [ ! -x "$BACKEND_VENV_PY" ]; then
  python3 -m venv "$ROOT_DIR/backend/.venv"
fi

if ! "$BACKEND_VENV_PY" -c "import fastapi, pytest, pydantic_ai" >/dev/null 2>&1; then
  "$BACKEND_VENV_PY" -m pip install --upgrade pip
  "$BACKEND_VENV_PY" -m pip install -r "$ROOT_DIR/backend/requirements.txt"
fi

cd "$ROOT_DIR"
"$BACKEND_VENV_PY" -m pytest -q
node --experimental-specifier-resolution=node --test tests/frontend/*.test.mjs

cd "$ROOT_DIR/backend"
"$BACKEND_VENV_PY" -m py_compile app/main.py app/wide_angle.py

cd "$ROOT_DIR/frontend"
npm run typecheck
npm run build
