from __future__ import annotations

from pathlib import Path
from typing import Any

from app.training_provenance import (
    normalize_path,
    probe_dvc_runtime,
    resolve_dvc_tracking,
    utc_now_iso,
)

PROVENANCE_SCHEMA_VERSION = 1


def build_benchmark_provenance(
    *,
    benchmark_id: str,
    suite: dict[str, Any],
    recipe: dict[str, Any],
    dataset_root: str | None,
    manifest_path: str | None,
    benchmark_dir: str | Path,
    artifacts_dir: str | Path,
) -> dict[str, Any]:
    normalized_benchmark_dir = normalize_path(benchmark_dir)
    normalized_artifacts_dir = normalize_path(artifacts_dir)
    normalized_dataset_root = normalize_path(dataset_root)
    normalized_manifest_path = normalize_path(manifest_path)
    return {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "generated_at": utc_now_iso(),
        "benchmark_id": benchmark_id,
        "suite_id": str(suite.get("id") or ""),
        "recipe_id": str(recipe.get("id") or ""),
        "benchmark_dir": normalized_benchmark_dir,
        "artifacts_dir": normalized_artifacts_dir,
        "dvc_runtime": probe_dvc_runtime(),
        "suite": {
            "dataset_root": normalized_dataset_root,
            "dataset_dvc": resolve_dvc_tracking(normalized_dataset_root),
            "manifest_path": normalized_manifest_path,
            "manifest_dvc": resolve_dvc_tracking(normalized_manifest_path),
        },
        "recipe": {
            "asset_id": recipe.get("asset_id"),
            "source_asset_ids": list(recipe.get("source_asset_ids") or []),
        },
        "artifacts_dvc": resolve_dvc_tracking(normalized_artifacts_dir),
    }
