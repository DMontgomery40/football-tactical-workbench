from __future__ import annotations

from pathlib import Path
from typing import Any

from .common import BenchmarkEvaluationUnavailable, metric_value
from .external_cli import run_external_json_command
from .prediction_exports import ensure_team_spotting_prediction_export

TEAM_SPOTTING_WRAPPER_PATH = Path(__file__).resolve().with_name("run_team_spotting_eval.py")


def _normalized_team_spotting_split(suite: dict[str, Any]) -> str:
    raw_split = str(suite.get("dataset_split") or "test").strip().lower()
    if raw_split in {"validation", "valid"}:
        return "val"
    return raw_split or "test"


def evaluate_team_spotting(*, suite: dict[str, Any], recipe: dict[str, Any], dataset_root: str, artifacts_dir: str | Path, benchmark_id: str) -> dict[str, Any]:
    repo_dir = Path(__file__).resolve().parents[2] / "third_party" / "soccernet" / "sn-teamspotting"
    if not repo_dir.exists():
        raise BenchmarkEvaluationUnavailable("sn-teamspotting checkout is missing under backend/third_party/soccernet/sn-teamspotting.")
    export_info = ensure_team_spotting_prediction_export(
        dataset_root=dataset_root,
        artifacts_dir=artifacts_dir,
    )
    predictions_root = Path(str(export_info["prediction_root"])).expanduser().resolve()
    payload = run_external_json_command(
        command=[
            "python",
            str(TEAM_SPOTTING_WRAPPER_PATH),
            "--labels-root",
            str(Path(dataset_root).expanduser().resolve()),
            "--predictions-root",
            str(predictions_root),
            "--split",
            _normalized_team_spotting_split(suite),
            "--metric",
            "at1",
            "--prediction-file",
            "results_spotting.json",
        ],
        cwd=repo_dir,
        artifacts_dir=artifacts_dir,
        runtime_key="modern_action_spotting",
    )
    external_result_path = payload.pop("_external_result_path", None)
    return {
        "metrics": {
            "team_map_at_1": metric_value(payload.get("team_map_at_1"), label="Team-mAP@1"),
            "map_at_1": metric_value(payload.get("map_at_1"), label="mAP@1"),
            "clips_per_second": metric_value(payload.get("clips_per_second"), label="Clips/s", precision=2),
        },
        "artifacts": {
            "prediction_root": str(predictions_root),
            "prediction_export_summary_json": export_info.get("export_summary_json"),
            "source_prediction_artifact": export_info.get("source_prediction_artifact"),
            **({"external_result_json": external_result_path} if external_result_path else {}),
        },
        "raw_result": payload,
    }
