from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = REPO_ROOT / "backend"
FRONTEND_DIR = REPO_ROOT / "frontend"
CONTRACTS_DIR = REPO_ROOT / "packages" / "contracts"
GENERATED_DIR = CONTRACTS_DIR / "generated"
OPENAPI_JSON_PATH = GENERATED_DIR / "openapi.json"
SCHEMA_TS_PATH = GENERATED_DIR / "schema.ts"
BACKEND_VENV_PYTHON = BACKEND_DIR / ".venv" / "bin" / "python"


def maybe_reexec_with_backend_python() -> None:
    if not BACKEND_VENV_PYTHON.exists():
        return
    current_python = Path(sys.executable)
    backend_python = BACKEND_VENV_PYTHON
    if current_python == backend_python or "--backend-python" in sys.argv:
        return
    subprocess.run(
        [str(backend_python), str(Path(__file__).resolve()), "--backend-python"],
        cwd=str(REPO_ROOT),
        check=True,
    )
    raise SystemExit(0)


def export_openapi() -> None:
    sys.path.insert(0, str(BACKEND_DIR))
    try:
        from app.main import app  # noqa: PLC0415
    finally:
        sys.path.pop(0)

    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    OPENAPI_JSON_PATH.write_text(
        json.dumps(app.openapi(), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def resolve_openapi_typescript_binary() -> Path | None:
    candidates = [
        FRONTEND_DIR / "node_modules" / ".bin" / "openapi-typescript",
        FRONTEND_DIR / "node_modules" / ".bin" / "openapi-typescript.cmd",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def generate_typescript_contract() -> None:
    binary = resolve_openapi_typescript_binary()
    if binary is None:
        raise RuntimeError(
            "Could not find openapi-typescript in frontend/node_modules. "
            "Run `cd frontend && npm install` first."
        )

    subprocess.run(
        [str(binary), str(OPENAPI_JSON_PATH), "-o", str(SCHEMA_TS_PATH)],
        cwd=str(REPO_ROOT),
        check=True,
    )


def main() -> int:
    maybe_reexec_with_backend_python()
    export_openapi()
    generate_typescript_contract()
    print(f"Wrote {OPENAPI_JSON_PATH.relative_to(REPO_ROOT)}")
    print(f"Wrote {SCHEMA_TS_PATH.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
