#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BACKEND_VENV_PY="$ROOT_DIR/backend/.venv/bin/python"

if [ ! -x "$BACKEND_VENV_PY" ]; then
  python3 -m venv "$ROOT_DIR/backend/.venv"
fi

if ! "$BACKEND_VENV_PY" - <<'PY' >/dev/null 2>&1
from importlib.metadata import version

assert version("fastapi")
assert version("pytest")
assert version("pydantic-ai-slim") == "1.41.0"
assert version("pydantic-evals") == "1.41.0"
PY
then
  "$BACKEND_VENV_PY" -m pip install --upgrade pip
  "$BACKEND_VENV_PY" -m pip install -r "$ROOT_DIR/backend/requirements.txt"
fi

if [ ! -d "$ROOT_DIR/frontend/node_modules" ]; then
  (cd "$ROOT_DIR/frontend" && npm ci)
fi

cd "$ROOT_DIR"
"$BACKEND_VENV_PY" -m pytest -q
node --experimental-specifier-resolution=node --test tests/frontend/*.test.mjs

cd "$ROOT_DIR/backend"
"$BACKEND_VENV_PY" -m py_compile app/main.py app/wide_angle.py

cd "$ROOT_DIR/frontend"
npm run typecheck
npm run build
