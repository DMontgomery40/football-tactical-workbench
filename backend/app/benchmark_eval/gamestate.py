from __future__ import annotations

import importlib.util
import tomllib
from pathlib import Path
from typing import Any

from .common import BenchmarkEvaluationUnavailable, metric_value
from .external_cli import run_external_json_command
from .runtime_profiles import probe_runtime_profile, runtime_unavailable_message

TRACKLAB_REPO_DIR = Path(__file__).resolve().parents[2] / "third_party" / "soccernet" / "tracklab"
SN_GAMESTATE_REPO_DIR = Path(__file__).resolve().parents[2] / "third_party" / "soccernet" / "sn-gamestate"
SN_GAMESTATE_PYPROJECT_PATH = SN_GAMESTATE_REPO_DIR / "pyproject.toml"
TRACKLAB_PYPROJECT_PATH = TRACKLAB_REPO_DIR / "pyproject.toml"
GAMESTATE_RUNTIME_KEY = "tracklab_gamestate_py39_np1"


def _load_pyproject_dependencies(path: Path) -> list[str]:
    if not path.exists():
        return []
    payload = tomllib.loads(path.read_text(encoding="utf-8"))
    project = payload.get("project")
    if not isinstance(project, dict):
        return []
    dependencies = project.get("dependencies")
    if not isinstance(dependencies, list):
        return []
    return [str(item) for item in dependencies]


def probe_gamestate_blockers(
    *,
    suite: dict[str, Any],
    dataset_root: str,
    manifest_payload: dict[str, Any] | None = None,
) -> list[str]:
    blockers: list[str] = []
    manifest_payload = manifest_payload or {}
    dataset_path = Path(dataset_root).expanduser().resolve() if dataset_root else None
    materialization = manifest_payload.get("materialization")
    manifest_materialization_blockers = materialization.get("blockers") if isinstance(materialization, dict) else None
    has_manifest_materialization_blockers = isinstance(manifest_materialization_blockers, list) and bool(manifest_materialization_blockers)

    if not TRACKLAB_REPO_DIR.exists():
        blockers.append(
            "Vendored TrackLab sources are missing under backend/third_party/soccernet/tracklab."
        )
    if not SN_GAMESTATE_REPO_DIR.exists():
        blockers.append(
            "Vendored sn-gamestate sources are missing under backend/third_party/soccernet/sn-gamestate."
        )
    if blockers:
        return blockers

    sn_gamestate_dependencies = _load_pyproject_dependencies(SN_GAMESTATE_PYPROJECT_PATH)
    tracklab_dependencies = _load_pyproject_dependencies(TRACKLAB_PYPROJECT_PATH)

    if not any("mmocr==1.0.1" in dependency for dependency in sn_gamestate_dependencies):
        blockers.append(
            "The vendored sn-gamestate baseline dependency contract could not be verified from pyproject.toml."
        )
    if not any("sn-trackeval" in dependency for dependency in tracklab_dependencies):
        blockers.append(
            "The vendored TrackLab dependency contract could not be verified from pyproject.toml."
        )

    runtime_probe = probe_runtime_profile(GAMESTATE_RUNTIME_KEY)
    if not runtime_probe.get("available"):
        blockers.append(runtime_unavailable_message(GAMESTATE_RUNTIME_KEY))

    if (dataset_path is None or not dataset_path.exists()) and not has_manifest_materialization_blockers:
        blockers.append(
            "Game-state dataset materialization is missing. Expected a SoccerNetGS tree under "
            f"{dataset_root or 'backend/benchmarks/_datasets/gsr.medium_v1/SoccerNetGS'} "
            "with a valid/ split and Labels-GameState.json for each benchmark clip."
        )

    manifest_items = manifest_payload.get("items")
    if isinstance(manifest_items, list) and any(str(item).startswith("SN-GSR-2025-valid-") for item in manifest_items):
        blockers.append(
            "The gsr.medium_v1 manifest still contains placeholder clip ids, so the fixed 12-clip benchmark subset "
            "is not durably pinned yet."
        )

    blockers.append(
        "Benchmark Lab does not yet translate recipe rows into the TrackLab/sn-gamestate Hydra configuration and "
        "tracker-state inputs required to run GS-HOTA evaluation from the vendored baseline."
    )
    return blockers


def evaluate_gamestate(
    *,
    suite: dict[str, Any],
    recipe: dict[str, Any],
    dataset_root: str,
    artifacts_dir: str | Path,
    benchmark_id: str,
) -> dict[str, Any]:
    blockers = probe_gamestate_blockers(
        suite=suite,
        dataset_root=dataset_root,
        manifest_payload={},
    )
    if blockers:
        raise BenchmarkEvaluationUnavailable(" ".join(dict.fromkeys(blockers)))

    payload = run_external_json_command(
        command=[
            "python",
            "-m",
            "tracklab.main",
            "dataset=soccernet_gs",
            f"dataset.dataset_path={dataset_root}",
            f"tracking.name={recipe.get('id') or 'recipe'}",
        ],
        cwd=TRACKLAB_REPO_DIR,
        artifacts_dir=artifacts_dir,
        runtime_key=GAMESTATE_RUNTIME_KEY,
    )
    external_result_path = payload.pop("_external_result_path", None)
    return {
        "metrics": {
            "gs_hota": metric_value(payload.get("gs_hota"), label="GS-HOTA"),
            "hota": metric_value(payload.get("hota"), label="HOTA"),
            "deta": metric_value(payload.get("deta"), label="DetA"),
            "assa": metric_value(payload.get("assa"), label="AssA"),
            "frames_per_second": metric_value(payload.get("frames_per_second"), label="Frames/s", precision=2),
        },
        "artifacts": {
            **({"external_result_json": external_result_path} if external_result_path else {}),
        },
        "raw_result": payload,
    }
