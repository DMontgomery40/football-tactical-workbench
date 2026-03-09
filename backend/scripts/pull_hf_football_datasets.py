#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download

try:
    import yaml
except Exception:  # pragma: no cover - script stays usable without PyYAML
    yaml = None  # type: ignore[assignment]


ROOT_DIR = Path(__file__).resolve().parents[1]
DATASETS_DIR = ROOT_DIR / "datasets" / "huggingface"


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    repo_id: str
    local_dir_name: str
    purpose: str
    format_hint: str
    training_ready: bool
    allow_patterns: tuple[str, ...]
    notes: tuple[str, ...]
    recommended_scan_subpath: str | None = None
    max_workers: int = 4


DATASET_SPECS: dict[str, DatasetSpec] = {
    "smoke": DatasetSpec(
        key="smoke",
        repo_id="martinjolif/football-player-detection",
        local_dir_name="martinjolif_football_player_detection",
        purpose="Small real football detector dataset for Training Studio smoke tests.",
        format_hint="YOLO-style folders under data/train, data/valid, data/test",
        training_ready=True,
        allow_patterns=("README.md", "data/**"),
        notes=(
            "Real football images with YOLO labels for ball, goalkeeper, player, and referee.",
            "Good for quick end-to-end detector fine-tuning tests in Training Studio.",
            "License declared in dataset card README frontmatter: cc-by-4.0.",
        ),
        recommended_scan_subpath="data",
        max_workers=4,
    ),
    "real": DatasetSpec(
        key="real",
        repo_id="Voxel51/SoccerNet-V3",
        local_dir_name="voxel51_soccernet_v3",
        purpose="Broadcast-real football benchmark/reference dataset from Hugging Face.",
        format_hint="FiftyOne / metadata-driven object-detection dataset, not YOLO-native",
        training_ready=False,
        allow_patterns=("README.md", "fiftyone.yml", "metadata.json", "samples.json", "data/**"),
        notes=(
            "Professional broadcast football benchmark dataset with much stronger realism than a smoke set.",
            "Excellent reference pull for future conversion, evaluation, and serious data work.",
            "License declared in dataset card README frontmatter: MIT.",
        ),
        recommended_scan_subpath=None,
        max_workers=1,
    ),
}


def parse_readme_frontmatter(readme_path: Path) -> dict[str, Any]:
    try:
        text = readme_path.read_text(encoding="utf-8")
    except Exception:
        return {}
    if not text.startswith("---\n"):
        return {}
    closing = text.find("\n---", 4)
    if closing < 0:
        return {}
    frontmatter = text[4:closing]
    if yaml is not None:
        try:
            payload = yaml.safe_load(frontmatter) or {}
        except Exception:
            payload = {}
        return payload if isinstance(payload, dict) else {}
    metadata: dict[str, Any] = {}
    for raw_line in frontmatter.splitlines():
        if ":" not in raw_line:
            continue
        key, value = raw_line.split(":", 1)
        metadata[key.strip()] = value.strip()
    return metadata


def write_manifest(dataset_dir: Path, spec: DatasetSpec) -> dict[str, Any]:
    readme_path = dataset_dir / "README.md"
    readme_meta = parse_readme_frontmatter(readme_path)
    recommended_scan_path = (dataset_dir / spec.recommended_scan_subpath).resolve() if spec.recommended_scan_subpath else dataset_dir.resolve()
    manifest = {
        "source": "huggingface",
        "repo_id": spec.repo_id,
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "local_path": str(dataset_dir.resolve()),
        "recommended_scan_path": str(recommended_scan_path),
        "purpose": spec.purpose,
        "format_hint": spec.format_hint,
        "training_ready": spec.training_ready,
        "license": readme_meta.get("license"),
        "size_categories": readme_meta.get("size_categories"),
        "task_categories": readme_meta.get("task_categories"),
        "notes": list(spec.notes),
    }
    manifest_path = dataset_dir / "fpw_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def download_dataset(spec: DatasetSpec, force: bool = False) -> tuple[Path, dict[str, Any]]:
    target_dir = DATASETS_DIR / spec.local_dir_name
    if force and target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    try:
        snapshot_download(
            repo_id=spec.repo_id,
            repo_type="dataset",
            local_dir=str(target_dir),
            allow_patterns=list(spec.allow_patterns),
            max_workers=spec.max_workers,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Download failed for {spec.repo_id}. Hugging Face sometimes rate-limits larger pulls; rerun the same command to resume."
        ) from exc

    manifest = write_manifest(target_dir, spec)
    return target_dir, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Pull recommended football datasets from Hugging Face into backend/datasets/huggingface.")
    parser.add_argument(
        "--dataset",
        choices=["smoke", "real", "all"],
        default="all",
        help="Which preset to pull.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete and re-download the target directories first.",
    )
    args = parser.parse_args()

    selected = list(DATASET_SPECS.values()) if args.dataset == "all" else [DATASET_SPECS[args.dataset]]

    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    for spec in selected:
        target_dir, manifest = download_dataset(spec, force=args.force)
        print(f"[downloaded] {spec.key}: {target_dir}")
        print(f"  repo: {spec.repo_id}")
        print(f"  training_ready: {spec.training_ready}")
        print(f"  format: {spec.format_hint}")
        print(f"  recommended_scan_path: {manifest['recommended_scan_path']}")
        print(f"  license: {manifest.get('license') or 'unknown'}")


if __name__ == "__main__":
    main()
