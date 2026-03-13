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
        local_dir_name="voxel51_soccernet_v3_sample",
        purpose="Broadcast-real SoccerNet-V3 slice for serious local reference work without pulling the full benchmark.",
        format_hint="FiftyOne / metadata-driven object-detection dataset, not YOLO-native",
        training_ready=False,
        allow_patterns=(
            "README.md",
            "fiftyone.yml",
            "metadata.json",
            "samples.json",
            "data/data_0/**",
            "data/data_1/**",
            "data/data_2/**",
            "data/data_3/**",
            "data/data_4/**",
            "data/data_5/**",
            "data/data_6/**",
            "data/data_7/**",
            "data/data_8/**",
            "data/data_9/**",
        ),
        notes=(
            "Representative broadcast-real slice of SoccerNet-V3 with much stronger realism than a smoke set.",
            "Good local reference pull for future conversion, evaluation, and serious data work without forcing a full multi-gig download on day one.",
            "License declared in dataset card README frontmatter: MIT.",
        ),
        recommended_scan_subpath=None,
        max_workers=1,
    ),
    "benchmark_full": DatasetSpec(
        key="benchmark_full",
        repo_id="Voxel51/SoccerNet-V3",
        local_dir_name="voxel51_soccernet_v3",
        purpose="Full SoccerNet-V3 benchmark pull for offline archival and large-scale experimentation.",
        format_hint="FiftyOne / metadata-driven object-detection dataset, not YOLO-native",
        training_ready=False,
        allow_patterns=("README.md", "fiftyone.yml", "metadata.json", "samples.json", "data/**"),
        notes=(
            "Full SoccerNet-V3 benchmark pull.",
            "Much heavier than the curated `real` slice and best treated as an explicit long-running download.",
            "License declared in dataset card README frontmatter: MIT.",
        ),
        recommended_scan_subpath=None,
        max_workers=1,
    ),
    "gsr_medium": DatasetSpec(
        key="gsr_medium",
        repo_id="SoccerNet/SN-GSR-2025",
        local_dir_name="soccernet_sn_gsr_2025_medium",
        purpose="Game-state reconstruction medium tier with validation metadata plus a fixed 12-clip subset manifest.",
        format_hint="Official SoccerNet GSR zips; benchmark-ready after unpacking the validation assets referenced by the manifest",
        training_ready=False,
        allow_patterns=("README.md", "valid.zip"),
        notes=(
            "Pulls the validation archive for SN-GSR-2025 so Benchmark Lab can materialize the fixed 12-clip medium tier.",
            "Use together with backend/benchmarks/_manifests/gsr.medium_v1.json.",
            "License and usage terms follow the SoccerNet dataset card.",
        ),
        recommended_scan_subpath=None,
        max_workers=1,
    ),
    "gsr_long": DatasetSpec(
        key="gsr_long",
        repo_id="SoccerNet/SN-GSR-2025",
        local_dir_name="soccernet_sn_gsr_2025_long",
        purpose="Full SoccerNet GSR validation pull for the long benchmark tier.",
        format_hint="Official SoccerNet GSR validation zip for long-running benchmark evaluation",
        training_ready=False,
        allow_patterns=("README.md", "valid.zip"),
        notes=(
            "Pulls the full validation archive for SN-GSR-2025.",
            "Use for the long GS-HOTA benchmark tier.",
            "License and usage terms follow the SoccerNet dataset card.",
        ),
        recommended_scan_subpath=None,
        max_workers=1,
    ),
    "team_bas_val": DatasetSpec(
        key="team_bas_val",
        repo_id="SoccerNet/SN-BAS-2025",
        local_dir_name="soccernet_sn_bas_2025",
        purpose="Team BAS validation archive plus companion labels archive for Benchmark Lab Stage 2 materialization work.",
        format_hint="Official SoccerNet Ball action spotting validation ZIP with encrypted members that still require the SoccerNet password to extract",
        training_ready=False,
        allow_patterns=("README.md", "valid.zip", "ExtraLabelsActionSpotting500games/valid_labels.zip"),
        notes=(
            "Pulls the validation archive used by spot.team_bas_quick_v1.",
            "The archive itself is downloadable from Hugging Face, but the official valid.zip members are still password-protected at extraction time on this machine.",
            "The extra labels ZIP is not a substitute for the ball-spotting Labels-ball.json tree required by the Benchmark Lab evaluator.",
        ),
        recommended_scan_subpath=None,
        max_workers=1,
    ),
    "pcbas_val": DatasetSpec(
        key="pcbas_val",
        repo_id="SoccerNet/SN-PCBAS-2026",
        local_dir_name="soccernet_sn_pcbas_2026",
        purpose="PCBAS validation tactical-data archive for Benchmark Lab Stage 2 materialization work.",
        format_hint="Official FOOTPASS tactical-data validation ZIP; gated at the file-download level on Hugging Face",
        training_ready=False,
        allow_patterns=("tactical_data_VAL.zip", "tactical_data_format.txt", "README.md"),
        notes=(
            "Pulls the official validation tactical-data archive used by spot.pcbas_medium_v1.",
            "On this machine the Hugging Face repo is gated and returns a 401 until the operator is authenticated for SoccerNet/SN-PCBAS-2026.",
            "Benchmark Lab can still stage the vendored playbyplay_val.json ground-truth file locally, but the official tactical/video archives remain separate upstream artifacts.",
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
        choices=["smoke", "real", "benchmark_full", "gsr_medium", "gsr_long", "team_bas_val", "pcbas_val", "benchmark_stage2", "all"],
        default="all",
        help="Which preset to pull.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete and re-download the target directories first.",
    )
    args = parser.parse_args()

    if args.dataset == "all":
        selected = [DATASET_SPECS["smoke"], DATASET_SPECS["real"]]
    elif args.dataset == "benchmark_stage2":
        selected = [
            DATASET_SPECS["team_bas_val"],
            DATASET_SPECS["pcbas_val"],
            DATASET_SPECS["gsr_medium"],
            DATASET_SPECS["gsr_long"],
        ]
    else:
        selected = [DATASET_SPECS[args.dataset]]

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
