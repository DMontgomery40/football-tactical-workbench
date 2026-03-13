from __future__ import annotations

from pathlib import Path
from typing import Any

from .common import BenchmarkEvaluationUnavailable, metric_value
from .external_cli import run_external_json_command
from .prediction_exports import ensure_synloc_prediction_export


SYNLOC_WRAPPER_PATH = Path(__file__).resolve().with_name("run_synloc_eval.py")
SSKIT_REPO_DIR = Path(__file__).resolve().parents[2] / "third_party" / "soccernet" / "sskit"


def evaluate_synloc(
    *,
    suite: dict[str, Any],
    recipe: dict[str, Any],
    dataset_root: str,
    artifacts_dir: str | Path,
    benchmark_id: str,
) -> dict[str, Any]:
    if not SSKIT_REPO_DIR.exists():
        raise BenchmarkEvaluationUnavailable(
            "SynLoc evaluator checkout is missing under backend/third_party/soccernet/sskit."
        )

    export_info = ensure_synloc_prediction_export(
        recipe=recipe,
        dataset_root=dataset_root,
        artifacts_dir=artifacts_dir,
    )
    predictions_json = Path(str(export_info["predictions_json"])).expanduser().resolve()
    metadata_json = Path(str(export_info["metadata_json"])).expanduser().resolve()
    ground_truth_json = Path(str(export_info["ground_truth_json"])).expanduser().resolve()

    payload = run_external_json_command(
        command=[
            "python",
            str(SYNLOC_WRAPPER_PATH),
            "--ground-truth-json",
            str(ground_truth_json),
            "--predictions-json",
            str(predictions_json),
            "--metadata-json",
            str(metadata_json),
        ],
        cwd=SSKIT_REPO_DIR,
        artifacts_dir=artifacts_dir,
        runtime_key="backend_default",
    )
    external_result_path = payload.pop("_external_result_path", None)
    return {
        "metrics": {
            "map_locsim": metric_value(payload.get("map_locsim"), label="mAP-LocSim"),
            "frames_per_second": metric_value(payload.get("frames_per_second"), label="Frames/s", precision=2),
            "avg_frame_latency_ms": metric_value(payload.get("avg_frame_latency_ms"), label="Latency", unit=" ms", precision=2),
        },
        "artifacts": {
            "predictions_json": str(predictions_json),
            "metadata_json": str(metadata_json),
            "ground_truth_json": str(ground_truth_json),
            "prediction_export_summary_json": export_info.get("export_summary_json"),
            "source_prediction_artifact": export_info.get("source_prediction_artifact"),
            **({"external_result_json": external_result_path} if external_result_path else {}),
        },
        "raw_result": payload,
    }
