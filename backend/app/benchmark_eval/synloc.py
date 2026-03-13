from __future__ import annotations

from pathlib import Path
from typing import Any

from .common import BenchmarkEvaluationUnavailable, metric_value
from .external_cli import run_external_json_command


def evaluate_synloc(*, suite: dict[str, Any], recipe: dict[str, Any], dataset_root: str, artifacts_dir: str | Path, benchmark_id: str) -> dict[str, Any]:
    repo_dir = Path(__file__).resolve().parents[2] / "third_party" / "soccernet" / "sskit"
    if not repo_dir.exists():
        raise BenchmarkEvaluationUnavailable("SynLoc evaluator checkout is missing under backend/third_party/soccernet/sskit.")
    payload = run_external_json_command(
        command=[
            "python",
            "-m",
            "sskit",
            "--dataset-root",
            str(dataset_root),
            "--model-path",
            str(recipe.get("artifact_path") or ""),
        ],
        cwd=repo_dir,
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
            **({"external_result_json": external_result_path} if external_result_path else {}),
        },
        "raw_result": payload,
    }
