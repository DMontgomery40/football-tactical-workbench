from __future__ import annotations

from pathlib import Path
from typing import Any

from .common import BenchmarkEvaluationUnavailable, metric_value
from .external_cli import run_external_json_command
from .prediction_exports import ensure_calibration_prediction_export

CALIBRATION_WRAPPER_PATH = Path(__file__).resolve().with_name("run_calibration_eval.py")


def evaluate_calibration(*, suite: dict[str, Any], recipe: dict[str, Any], dataset_root: str, artifacts_dir: str | Path, benchmark_id: str) -> dict[str, Any]:
    repo_dir = Path(__file__).resolve().parents[2] / "third_party" / "soccernet" / "sn-calibration"
    if not repo_dir.exists():
        raise BenchmarkEvaluationUnavailable("sn-calibration checkout is missing under backend/third_party/soccernet/sn-calibration.")
    export_info = ensure_calibration_prediction_export(
        recipe=recipe,
        dataset_root=dataset_root,
        artifacts_dir=artifacts_dir,
    )
    prediction_root = Path(str(export_info["prediction_root"])).expanduser().resolve()
    payload = run_external_json_command(
        command=[
            "python",
            str(CALIBRATION_WRAPPER_PATH),
            "--dataset-root",
            str(Path(dataset_root).expanduser().resolve()),
            "--prediction-root",
            str(prediction_root),
            "--split",
            "valid",
            "--threshold",
            "5",
            "--resolution-width",
            "960",
            "--resolution-height",
            "540",
            "--artifacts-dir",
            str(Path(artifacts_dir).expanduser().resolve()),
        ],
        cwd=repo_dir,
        artifacts_dir=artifacts_dir,
        runtime_key="sn_calibration_legacy",
    )
    external_result_path = payload.pop("_external_result_path", None)
    return {
        "metrics": {
            "completeness_x_jac_5": metric_value(payload.get("completeness_x_jac_5"), label="Completeness x JaC@5"),
            "completeness": metric_value(payload.get("completeness"), label="Completeness"),
            "jac_5": metric_value(payload.get("jac_5"), label="JaC@5"),
            "frames_per_second": metric_value(payload.get("frames_per_second"), label="Frames/s", precision=2),
        },
        "artifacts": {
            "prediction_root": str(prediction_root),
            "prediction_export_summary_json": export_info.get("export_summary_json"),
            **({"external_result_json": external_result_path} if external_result_path else {}),
        },
        "raw_result": payload,
    }
