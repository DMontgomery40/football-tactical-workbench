from __future__ import annotations

import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

PROVENANCE_SCHEMA_VERSION = 1
PROVENANCE_FILENAME = "training_provenance.json"
BACKEND_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_ROOT.parent
PROMOTED_MODELS_DIR = BACKEND_ROOT / "models" / "promoted"


def utc_now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def normalize_path(value: str | Path | None) -> str | None:
    if value in {None, ""}:
        return None
    return str(Path(str(value)).expanduser().resolve())


def resolve_dvc_binary() -> str | None:
    direct = shutil.which("dvc")
    if direct:
        return direct
    repo_local = BACKEND_ROOT / ".venv" / "bin" / "dvc"
    if repo_local.exists() and repo_local.is_file():
        return str(repo_local)
    return None


def probe_dvc_runtime(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    binary = resolve_dvc_binary()
    config_path = repo_root / ".dvc" / "config"
    ignore_path = repo_root / ".dvcignore"
    hook_path = repo_root / ".githooks" / "post-checkout"
    version: str | None = None
    probe_error: str | None = None

    if binary:
        try:
            completed = subprocess.run(
                [binary, "version"],
                cwd=str(repo_root),
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            if completed.returncode == 0:
                first_line = next((line.strip() for line in completed.stdout.splitlines() if line.strip()), "")
                version = first_line or None
            else:
                probe_error = (completed.stderr or completed.stdout or "").strip() or "DVC version probe failed."
        except (OSError, subprocess.SubprocessError) as exc:
            probe_error = f"DVC probe failed: {exc}"

    repo_enabled = config_path.exists()
    status = "disabled"
    if repo_enabled and binary:
        status = "ready"
    elif repo_enabled:
        status = "repo_configured"
    elif binary:
        status = "cli_only"

    return {
        "status": status,
        "cli_available": bool(binary),
        "cli_path": binary,
        "version": version,
        "repo_enabled": repo_enabled,
        "repo_root": str(repo_root.resolve()),
        "config_path": str(config_path.resolve()) if config_path.exists() else None,
        "ignore_path": str(ignore_path.resolve()) if ignore_path.exists() else None,
        "checkout_hook_path": str(hook_path.resolve()) if hook_path.exists() else None,
        "probe_error": probe_error,
    }


def is_within_repo(path: Path, repo_root: Path = REPO_ROOT) -> bool:
    try:
        path.resolve().relative_to(repo_root.resolve())
        return True
    except ValueError:
        return False


def resolve_repo_relative_path(path: str | Path | None, repo_root: Path = REPO_ROOT) -> str | None:
    normalized = normalize_path(path)
    if not normalized:
        return None
    resolved = Path(normalized)
    try:
        return str(resolved.relative_to(repo_root.resolve()))
    except ValueError:
        return None


def resolve_dvc_tracking(path: str | Path | None, repo_root: Path = REPO_ROOT) -> dict[str, Any] | None:
    normalized = normalize_path(path)
    if not normalized:
        return None

    repo_root = repo_root.resolve()
    resolved_path = Path(normalized)
    tracked_pointer: Path | None = None
    tracked_root: Path | None = None

    if is_within_repo(resolved_path, repo_root):
        current = resolved_path
        while current != repo_root and is_within_repo(current, repo_root):
            candidate = current.parent / f"{current.name}.dvc"
            if candidate.exists() and candidate.is_file():
                tracked_pointer = candidate.resolve()
                tracked_root = current.resolve()
                break
            current = current.parent

    tracking_scope: str | None = None
    if tracked_root is not None:
        tracking_scope = "self" if tracked_root == resolved_path else "ancestor"

    exists = resolved_path.exists()
    return {
        "path": str(resolved_path),
        "repo_relative_path": resolve_repo_relative_path(resolved_path, repo_root),
        "exists": exists,
        "tracked": tracked_pointer is not None,
        "state": "tracked" if tracked_pointer is not None else ("present" if exists else "missing"),
        "pointer_path": str(tracked_pointer) if tracked_pointer else None,
        "pointer_relative_path": resolve_repo_relative_path(tracked_pointer, repo_root) if tracked_pointer else None,
        "tracked_root_path": str(tracked_root) if tracked_root else None,
        "tracked_root_relative_path": resolve_repo_relative_path(tracked_root, repo_root) if tracked_root else None,
        "tracking_scope": tracking_scope,
    }


def build_training_provenance(
    *,
    run_id: str,
    run_dir: str | Path,
    status: str,
    config: dict[str, Any] | None = None,
    dataset_path: str | None = None,
    dataset_scan_path: str | None = None,
    generated_dataset_yaml: str | None = None,
    generated_split_lists: dict[str, str] | None = None,
    summary_path: str | None = None,
    best_checkpoint: str | None = None,
    activation: dict[str, Any] | None = None,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    normalized_config = dict(config or {})
    normalized_run_dir = normalize_path(run_dir)
    normalized_dataset_path = normalize_path(dataset_path or normalized_config.get("dataset_path"))
    normalized_dataset_scan_path = normalize_path(dataset_scan_path)
    normalized_generated_dataset_yaml = normalize_path(generated_dataset_yaml)
    normalized_summary_path = normalize_path(summary_path)
    normalized_best_checkpoint = normalize_path(best_checkpoint)

    return {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "generated_at": utc_now_iso(),
        "run_id": run_id,
        "status": status,
        "run_dir": normalized_run_dir,
        "run_name": str(normalized_config.get("run_name") or run_id),
        "base_weights": str(normalized_config.get("base_weights") or "soccana"),
        "dvc_runtime": probe_dvc_runtime(repo_root),
        "dataset": {
            "source_path": normalized_dataset_path,
            "source_dvc": resolve_dvc_tracking(normalized_dataset_path, repo_root),
            "dataset_scan_path": normalized_dataset_scan_path,
            "generated_dataset_yaml": normalized_generated_dataset_yaml,
            "generated_split_lists": dict(generated_split_lists or {}),
        },
        "output": {
            "summary_path": normalized_summary_path,
            "summary_dvc": resolve_dvc_tracking(normalized_summary_path, repo_root),
            "best_checkpoint": normalized_best_checkpoint,
            "checkpoint_dvc": resolve_dvc_tracking(normalized_best_checkpoint, repo_root),
        },
        "activation": dict(activation or {}) or None,
    }


def write_training_provenance(target_path: str | Path, payload: dict[str, Any]) -> str:
    resolved_path = Path(str(target_path)).expanduser().resolve()
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return str(resolved_path)


def read_training_provenance(path: str | Path | None) -> dict[str, Any] | None:
    normalized = normalize_path(path)
    if not normalized:
        return None
    resolved_path = Path(normalized)
    if not resolved_path.exists() or not resolved_path.is_file():
        return None
    payload = json.loads(resolved_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Training provenance is not a JSON object: {resolved_path}")
    return payload


def promoted_detector_id(run_id: str) -> str:
    return f"custom_{run_id}"


def resolve_promoted_detector_dir(run_id: str) -> Path:
    return PROMOTED_MODELS_DIR / promoted_detector_id(run_id)


def resolve_promoted_checkpoint_path(run_id: str) -> Path:
    return resolve_promoted_detector_dir(run_id) / "best.pt"


def resolve_promoted_provenance_path(run_id: str) -> Path:
    return resolve_promoted_detector_dir(run_id) / PROVENANCE_FILENAME


def stage_promoted_detector_checkpoint(run_id: str, source_checkpoint: str | Path) -> str:
    source_path = Path(str(source_checkpoint)).expanduser().resolve()
    target_path = resolve_promoted_checkpoint_path(run_id)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, target_path)
    return str(target_path.resolve())
