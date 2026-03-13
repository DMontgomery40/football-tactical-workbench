from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from app.wide_angle import analyze_video as analyze_wide_angle_video, resolve_model_path

from .common import BenchmarkEvaluationUnavailable, metric_value

BASE_DIR = Path(__file__).resolve().parents[2]
RUNS_DIR = BASE_DIR / "runs"
BENCHMARK_RUNTIME_PROFILE: dict[str, Any] = {
    "pipeline": "classic",
    "keypoint_model": "soccana_keypoint",
    "tracker_mode": "hybrid_reid",
    "include_ball": True,
    "player_conf": 0.25,
    "ball_conf": 0.20,
    "iou": 0.50,
}


def _compute_operational_metrics(summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    frames = max(int(summary.get("frames_processed") or 0), 1)
    fps = float(summary.get("fps") or 0.0)
    avg_track_len = float(summary.get("average_track_length") or 0.0)
    churn = float(summary.get("player_track_churn_ratio") or 1.0)
    registered_ratio = float(summary.get("field_registered_ratio") or 0.0)
    avg_dets = float(summary.get("average_player_detections_per_frame") or 0.0)
    track_stability = min(avg_track_len / frames, 1.0) * (1.0 - min(churn, 1.0))
    coverage = min(avg_dets / 22.0, 1.0)
    return {
        "fps": metric_value(fps, label="FPS", precision=2),
        "track_stability": metric_value(track_stability * 100.0, label="Track stability", precision=2),
        "calibration": metric_value(registered_ratio * 100.0, label="Calibration", precision=2),
        "coverage": metric_value(coverage * 100.0, label="Coverage", precision=2),
    }


class _OperationalJobManager:
    def __init__(self, benchmark_dir: Path, recipe_id: str) -> None:
        self._log_path = benchmark_dir / "benchmark.log"
        self._recipe_id = recipe_id
        self._lock = threading.Lock()

    def log(self, job_id: str, message: str) -> None:
        stamp = datetime.utcnow().strftime("%H:%M:%S")
        line = f"[{stamp}] [{self._recipe_id}] {message}"
        with self._lock:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            with self._log_path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")

    def update(self, job_id: str, **kwargs: Any) -> None:
        return None


def evaluate_operational(
    *,
    suite: dict[str, Any],
    recipe: dict[str, Any],
    dataset_root: str,
    artifacts_dir: str | Path,
    benchmark_id: str,
) -> dict[str, Any]:
    clip_path = Path(dataset_root).expanduser().resolve()
    if not clip_path.exists():
        raise RuntimeError(f"Benchmark clip is missing: {clip_path}")

    run_id = f"bench_{benchmark_id}_{str(recipe.get('id') or 'recipe').replace(':', '_')}_{uuid.uuid4().hex[:4]}"
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    pipeline = str(recipe.get("pipeline") or BENCHMARK_RUNTIME_PROFILE["pipeline"])
    if pipeline == "soccermaster":
        player_model = resolve_model_path("soccana", "detector")
        keypoint_model = "soccermaster"
    else:
        player_model = str(recipe.get("artifact_path") or resolve_model_path("soccana", "detector"))
        keypoint_model = str(recipe.get("keypoint_model") or BENCHMARK_RUNTIME_PROFILE["keypoint_model"])

    config_payload = {
        "source_video_path": str(clip_path),
        "label_path": "",
        "pipeline": pipeline,
        "player_model": player_model,
        "ball_model": player_model,
        "keypoint_model": keypoint_model,
        "tracker_mode": str(recipe.get("requested_tracker_mode") or BENCHMARK_RUNTIME_PROFILE["tracker_mode"]),
        "include_ball": BENCHMARK_RUNTIME_PROFILE["include_ball"],
        "player_conf": BENCHMARK_RUNTIME_PROFILE["player_conf"],
        "ball_conf": BENCHMARK_RUNTIME_PROFILE["ball_conf"],
        "iou": BENCHMARK_RUNTIME_PROFILE["iou"],
    }

    job_manager = _OperationalJobManager(Path(artifacts_dir).expanduser().resolve().parent.parent, str(recipe.get("id") or "recipe"))
    try:
        summary = analyze_wide_angle_video(
            job_id=f"bench_{uuid.uuid4().hex[:8]}",
            run_dir=run_dir,
            config_payload=config_payload,
            job_manager=job_manager,
            job_control=None,
        )
    except ImportError as exc:
        if pipeline == "soccermaster":
            raise BenchmarkEvaluationUnavailable(
                "SoccerMaster operational review is blocked by a local dependency import mismatch while loading the "
                f"pipeline: {exc}"
            ) from exc
        raise
    except Exception as exc:
        if pipeline == "soccermaster" and "is_offline_mode" in str(exc):
            raise BenchmarkEvaluationUnavailable(
                "SoccerMaster operational review is blocked by the current backend dependency set. The pipeline still "
                "expects huggingface_hub.is_offline_mode, which is unavailable in this environment."
            ) from exc
        raise
    summary_path = Path(artifacts_dir).expanduser().resolve() / "summary_excerpt.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {
        "metrics": _compute_operational_metrics(summary),
        "artifacts": {
            "run_dir": str(run_dir),
            "summary_json": str(summary_path),
            "overlay_video": summary.get("overlay_video"),
        },
        "raw_result": {
            "run_id": run_id,
            "frames_processed": summary.get("frames_processed"),
            "fps": summary.get("fps"),
        },
    }
