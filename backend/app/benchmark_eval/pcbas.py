from __future__ import annotations

from pathlib import Path
from typing import Any

from .common import BenchmarkEvaluationUnavailable, metric_value
from .external_cli import run_external_json_command
from .prediction_exports import ensure_footpass_prediction_export

FOOTPASS_WRAPPER_PATH = Path(__file__).resolve().with_name("run_footpass_eval.py")


def _resolve_ground_truth_file(dataset_root: str) -> Path:
    dataset_path = Path(dataset_root).expanduser().resolve()
    if dataset_path.is_file():
        return dataset_path
    candidates = [
        dataset_path / "playbyplay_GT" / "playbyplay_val.json",
        dataset_path / "playbyplay_val.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def evaluate_pcbas(*, suite: dict[str, Any], recipe: dict[str, Any], dataset_root: str, artifacts_dir: str | Path, benchmark_id: str) -> dict[str, Any]:
    repo_dir = Path(__file__).resolve().parents[2] / "third_party" / "soccernet" / "FOOTPASS"
    if not repo_dir.exists():
        raise BenchmarkEvaluationUnavailable("FOOTPASS checkout is missing under backend/third_party/soccernet/FOOTPASS.")
    ground_truth_file = _resolve_ground_truth_file(dataset_root)
    if not ground_truth_file.exists():
        raise BenchmarkEvaluationUnavailable(
            "FOOTPASS ground-truth JSON is missing. Expected "
            f"{ground_truth_file}."
        )
    export_info = ensure_footpass_prediction_export(
        artifacts_dir=artifacts_dir,
    )
    predictions_file = Path(str(export_info["predictions_json"])).expanduser().resolve()
    payload = run_external_json_command(
        command=[
            "python",
            str(FOOTPASS_WRAPPER_PATH),
            "--predictions-file",
            str(predictions_file),
            "--ground-truth-file",
            str(ground_truth_file),
            "--delta",
            "12",
            "--confidence-threshold",
            "0.15",
        ],
        cwd=repo_dir,
        artifacts_dir=artifacts_dir,
        runtime_key="footpass_eval",
    )
    external_result_path = payload.pop("_external_result_path", None)
    return {
        "metrics": {
            "f1_at_15": metric_value(payload.get("f1_at_15"), label="F1@15%"),
            "precision_at_15": metric_value(payload.get("precision_at_15"), label="Precision@15%"),
            "recall_at_15": metric_value(payload.get("recall_at_15"), label="Recall@15%"),
            "clips_per_second": metric_value(payload.get("clips_per_second"), label="Clips/s", precision=2),
        },
        "artifacts": {
            "predictions_json": str(predictions_file),
            "prediction_export_summary_json": export_info.get("export_summary_json"),
            "source_prediction_artifact": export_info.get("source_prediction_artifact"),
            **({"external_result_json": external_result_path} if external_result_path else {}),
        },
        "raw_result": payload,
    }
