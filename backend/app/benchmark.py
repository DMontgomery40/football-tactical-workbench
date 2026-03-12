"""Benchmark Lab -- detector comparison orchestration.

Runs multiple detector candidates against a canonical benchmark clip using
fixed runtime settings, then ranks them by a transparent composite score
derived from existing analysis summary signals.

CRITICAL: imported benchmark models are never activated in the main analysis
detector registry.  They live under ``backend/benchmarks/_imports/`` and are
only referenced during benchmark runs.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
from app.training_provenance import (
    probe_dvc_runtime,
    resolve_dvc_tracking,
    utc_now_iso,
)
from app.wide_angle import (
    analyze_video as analyze_wide_angle_video,
    resolve_detector_spec,
    resolve_model_path,
)

BASE_DIR = Path(__file__).resolve().parent.parent
BENCHMARKS_DIR = BASE_DIR / "benchmarks"
CLIP_CACHE_DIR = BENCHMARKS_DIR / "_clip_cache"
IMPORTS_DIR = BENCHMARKS_DIR / "_imports"
RUNS_DIR = BASE_DIR / "runs"  # analysis runs land here

BENCHMARKS_DIR.mkdir(parents=True, exist_ok=True)
CLIP_CACHE_DIR.mkdir(parents=True, exist_ok=True)
IMPORTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Fixed runtime profile -- every candidate is evaluated under identical
# conditions so the comparison is fair.
# ---------------------------------------------------------------------------
BENCHMARK_RUNTIME_PROFILE: dict[str, Any] = {
    "pipeline": "classic",
    "keypoint_model": "soccana_keypoint",
    "tracker_mode": "hybrid_reid",
    "include_ball": True,
    "player_conf": 0.25,
    "ball_conf": 0.20,
    "iou": 0.50,
}

BENCHMARK_CLIP_FILENAME = "benchmark_clip.mp4"

# ---------------------------------------------------------------------------
# Composite scoring weights  (sum to 1.0)
# ---------------------------------------------------------------------------
SCORE_WEIGHT_TRACK_STABILITY = 0.30
SCORE_WEIGHT_CALIBRATION = 0.25
SCORE_WEIGHT_COVERAGE = 0.25
SCORE_WEIGHT_THROUGHPUT = 0.20


# ---- Clip management ------------------------------------------------------


def _inspect_clip(path: Path) -> dict[str, Any]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open benchmark clip: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    duration_seconds = float(frame_count / fps) if fps > 0 and frame_count > 0 else 0.0
    return {
        "fps": round(fps, 4) if fps > 0 else 0.0,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "duration_seconds": round(duration_seconds, 4),
        "size_mb": round(path.stat().st_size / (1024 * 1024), 2),
    }


def sn_gamestate_reference() -> dict[str, Any]:
    return {
        "label": "SoccerNet Game State Reconstruction",
        "repo_url": "https://github.com/SoccerNet/sn-gamestate",
        "paper_url": "https://arxiv.org/abs/2404.11335",
        "task_url": "https://www.soccer-net.org/tasks/new-game-state-reconstruction",
        "dataset_version": "1.3",
        "summary": (
            "Official SoccerNet GSR devkit built on TrackLab. It is an external baseline and evaluation stack, "
            "not a drop-in detector candidate inside this workbench."
        ),
    }


def clip_status() -> dict[str, Any]:
    """Return benchmark clip readiness info, including DVC tracking state."""
    clip_path = CLIP_CACHE_DIR / BENCHMARK_CLIP_FILENAME
    exists = clip_path.exists() and clip_path.stat().st_size > 0
    metadata = _inspect_clip(clip_path) if exists else {}
    dvc_tracking = resolve_dvc_tracking(str(clip_path)) if exists else None
    return {
        "ready": exists,
        "path": str(clip_path) if exists else None,
        "size_mb": metadata.get("size_mb"),
        "duration": metadata.get("duration_seconds"),
        "duration_seconds": metadata.get("duration_seconds"),
        "fps": metadata.get("fps"),
        "frame_count": metadata.get("frame_count"),
        "width": metadata.get("width"),
        "height": metadata.get("height"),
        "cache_dir": str(CLIP_CACHE_DIR),
        "expected_filename": BENCHMARK_CLIP_FILENAME,
        "dvc": dvc_tracking,
        "note": (
            "Place a ~5 min HD benchmark clip at "
            f"{clip_path} or use /api/benchmark/ensure-clip to provide one."
        ) if not exists else None,
    }


def ensure_clip(source_path: str = "") -> dict[str, Any]:
    """Copy or download a benchmark clip into the cache.

    If *source_path* points to a local file it is copied directly.
    Otherwise we return instructions for the operator.
    """
    dest = CLIP_CACHE_DIR / BENCHMARK_CLIP_FILENAME

    source = Path(source_path.strip()).expanduser().resolve() if source_path.strip() else None
    if source and source.exists() and source.is_file():
        shutil.copy2(str(source), str(dest))
        return clip_status()

    # If the clip is already cached, report success.
    if dest.exists() and dest.stat().st_size > 0:
        return clip_status()

    return {
        "ready": False,
        "path": None,
        "cache_dir": str(CLIP_CACHE_DIR),
        "expected_filename": BENCHMARK_CLIP_FILENAME,
        "error": (
            "No benchmark clip available. Provide a local path via "
            "'source_path' or manually place a ~5 min HD football clip at "
            f"{dest}"
        ),
    }


# ---- Candidate management -------------------------------------------------

SOCCERMASTER_MODELS_DIR = BASE_DIR / "models" / "soccermaster"


def _pretrained_candidate() -> dict[str, Any]:
    """The built-in soccana detector."""
    path = resolve_model_path("soccana", "detector")
    return {
        "id": "soccana",
        "label": "soccana (pretrained)",
        "source": "pretrained",
        "path": path,
        "is_pretrained": True,
    }


def _soccermaster_candidate() -> dict[str, Any] | None:
    """SoccerMaster unified pipeline candidate.

    Only available when the SoccerMaster model weights are present locally.
    This candidate uses a completely different pipeline (unified backbone for
    detection + calibration), so it overrides the locked runtime profile's
    pipeline and keypoint_model settings.
    """
    backbone = SOCCERMASTER_MODELS_DIR / "backbone.pt"
    detection = SOCCERMASTER_MODELS_DIR / "SoccerNetGSR_Detection.pt"
    keypoints = SOCCERMASTER_MODELS_DIR / "KeypointsDetection.pt"
    if not (backbone.exists() and detection.exists() and keypoints.exists()):
        return None
    return {
        "id": "soccermaster",
        "label": "SoccerMaster (unified)",
        "source": "pretrained",
        "path": str(SOCCERMASTER_MODELS_DIR),
        "is_pretrained": True,
        "pipeline_override": "soccermaster",
        "keypoint_override": "soccermaster",
    }


def _registry_candidates() -> list[dict[str, Any]]:
    """Custom detectors registered via Training Studio."""
    from app.training_registry import REGISTRY_PATH

    if not REGISTRY_PATH.exists():
        return []
    try:
        payload = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    except Exception:
        return []
    candidates: list[dict[str, Any]] = []
    for entry in payload.get("detectors") or []:
        if entry.get("is_pretrained"):
            continue
        detector_id = str(entry.get("id") or "")
        if not detector_id:
            continue
        path = str(entry.get("path") or "")
        if not path or not Path(path).exists():
            continue
        candidates.append({
            "id": detector_id,
            "label": str(entry.get("label") or detector_id),
            "source": "registry",
            "path": path,
            "is_pretrained": False,
            "training_run_id": entry.get("training_run_id"),
            "metrics": entry.get("metrics"),
        })
    return candidates


def _import_candidates() -> list[dict[str, Any]]:
    """Ad hoc imports staged under _imports/."""
    candidates: list[dict[str, Any]] = []
    manifest_path = IMPORTS_DIR / "imports.json"
    if not manifest_path.exists():
        return candidates
    try:
        records = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return candidates
    if not isinstance(records, list):
        return candidates
    for record in records:
        path = str(record.get("path") or "")
        if path and Path(path).exists():
            candidates.append({
                "id": str(record.get("id") or ""),
                "label": str(record.get("label") or record.get("id") or ""),
                "source": "import",
                "path": path,
                "is_pretrained": False,
                "import_origin": record.get("origin"),
            })
    return candidates


def list_candidates() -> list[dict[str, Any]]:
    """All benchmark candidates: pretrained + soccermaster + registry + imports."""
    seen: set[str] = set()
    result: list[dict[str, Any]] = []
    base = [_pretrained_candidate()]
    sm = _soccermaster_candidate()
    if sm is not None:
        base.append(sm)
    for candidate in base + _registry_candidates() + _import_candidates():
        cid = candidate["id"]
        if cid in seen:
            continue
        seen.add(cid)
        result.append(candidate)
    return result


def _save_import_manifest(records: list[dict[str, Any]]) -> None:
    manifest_path = IMPORTS_DIR / "imports.json"
    manifest_path.write_text(json.dumps(records, indent=2), encoding="utf-8")


def _load_import_manifest() -> list[dict[str, Any]]:
    manifest_path = IMPORTS_DIR / "imports.json"
    if not manifest_path.exists():
        return []
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except Exception:
        return []


def validate_checkpoint(path: str) -> dict[str, Any]:
    """Validate that a checkpoint can be loaded as a detector."""
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists() or not resolved.is_file():
        return {"valid": False, "error": f"File not found: {resolved}"}
    try:
        spec = resolve_detector_spec(str(resolved))
        return {
            "valid": True,
            "path": str(resolved),
            "class_names_source": spec.get("class_names_source", ""),
            "player_class_ids": spec.get("player_class_ids", []),
            "ball_class_ids": spec.get("ball_class_ids", []),
        }
    except Exception as exc:
        return {"valid": False, "error": str(exc)}


def import_local_checkpoint(checkpoint_path: str, label: str = "") -> dict[str, Any]:
    """Stage a local checkpoint into the imports dir for benchmarking."""
    resolved = Path(checkpoint_path).expanduser().resolve()
    if not resolved.exists() or not resolved.is_file():
        raise ValueError(f"Checkpoint not found: {resolved}")

    validation = validate_checkpoint(str(resolved))
    if not validation["valid"]:
        raise ValueError(f"Checkpoint validation failed: {validation.get('error')}")

    import_id = f"import_{uuid.uuid4().hex[:8]}"
    dest_dir = IMPORTS_DIR / import_id
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / "best.pt"
    shutil.copy2(str(resolved), str(dest_path))

    records = _load_import_manifest()
    record = {
        "id": import_id,
        "label": label.strip() or import_id,
        "path": str(dest_path),
        "origin": str(resolved),
        "imported_at": utc_now_iso(),
        "origin_dvc": resolve_dvc_tracking(str(resolved)),
    }
    records.append(record)
    _save_import_manifest(records)

    return {
        "id": import_id,
        "label": record["label"],
        "source": "import",
        "path": str(dest_path),
        "is_pretrained": False,
        "import_origin": str(resolved),
        "origin_dvc": record["origin_dvc"],
    }


def import_hf_checkpoint(repo_id: str, filename: str = "best.pt", label: str = "") -> dict[str, Any]:
    """Pull a detector checkpoint from Hugging Face and stage it for benchmarking."""
    from huggingface_hub import hf_hub_download

    import_id = f"import_hf_{uuid.uuid4().hex[:8]}"
    dest_dir = IMPORTS_DIR / import_id
    dest_dir.mkdir(parents=True, exist_ok=True)

    downloaded = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=str(dest_dir),
    )
    dest_path = Path(downloaded).resolve()

    validation = validate_checkpoint(str(dest_path))
    if not validation["valid"]:
        shutil.rmtree(str(dest_dir), ignore_errors=True)
        raise ValueError(f"HF checkpoint validation failed: {validation.get('error')}")

    records = _load_import_manifest()
    record = {
        "id": import_id,
        "label": label.strip() or f"{repo_id}/{filename}",
        "path": str(dest_path),
        "origin": f"hf://{repo_id}/{filename}",
        "imported_at": datetime.utcnow().isoformat() + "Z",
    }
    records.append(record)
    _save_import_manifest(records)

    return {
        "id": import_id,
        "label": record["label"],
        "source": "import",
        "path": str(dest_path),
        "is_pretrained": False,
        "import_origin": record["origin"],
    }


# ---- Scoring ---------------------------------------------------------------

def compute_composite_score(summary: dict[str, Any]) -> dict[str, Any]:
    """Derive a transparent composite score from an analysis run summary.

    Returns individual metric scores (0-100) plus the weighted composite.
    """
    frames = max(int(summary.get("frames_processed") or 0), 1)
    fps = float(summary.get("fps") or 0)

    # Track stability: penalise high churn, reward long average tracks
    avg_track_len = float(summary.get("average_track_length") or 0)
    churn = float(summary.get("player_track_churn_ratio") or 1.0)
    # Ideal avg track length ~= total frames, churn ~= 0
    stability_raw = min(avg_track_len / frames, 1.0) * (1.0 - min(churn, 1.0))
    track_stability = round(stability_raw * 100, 2)

    # Calibration health: fraction of frames with usable homography
    registered_ratio = float(summary.get("field_registered_ratio") or 0)
    calibration_score = round(registered_ratio * 100, 2)

    # Coverage: average player detections per frame, normalised to ~22 max
    avg_dets = float(summary.get("average_player_detections_per_frame") or 0)
    coverage_score = round(min(avg_dets / 22.0, 1.0) * 100, 2)

    # Throughput: FPS normalised to 30 as reference
    throughput_score = round(min(fps / 30.0, 1.0) * 100, 2)

    composite = round(
        track_stability * SCORE_WEIGHT_TRACK_STABILITY
        + calibration_score * SCORE_WEIGHT_CALIBRATION
        + coverage_score * SCORE_WEIGHT_COVERAGE
        + throughput_score * SCORE_WEIGHT_THROUGHPUT,
        2,
    )

    return {
        "composite": composite,
        "track_stability": track_stability,
        "calibration": calibration_score,
        "coverage": coverage_score,
        "throughput": throughput_score,
        "weights": {
            "track_stability": SCORE_WEIGHT_TRACK_STABILITY,
            "calibration": SCORE_WEIGHT_CALIBRATION,
            "coverage": SCORE_WEIGHT_COVERAGE,
            "throughput": SCORE_WEIGHT_THROUGHPUT,
        },
    }


# ---- Benchmark orchestration -----------------------------------------------

class _BenchmarkJobManager:
    """Minimal job-manager interface expected by analyze_video."""

    def __init__(self, benchmark_id: str, candidate_id: str, benchmark_dir: Path) -> None:
        self._benchmark_id = benchmark_id
        self._candidate_id = candidate_id
        self._log_path = benchmark_dir / "benchmark.log"
        self._progress: float = 0.0
        self._lines: list[str] = []
        self._lock = threading.Lock()

    def log(self, job_id: str, message: str) -> None:
        stamp = datetime.utcnow().strftime("%H:%M:%S")
        line = f"[{stamp}] [{self._candidate_id}] {message}"
        with self._lock:
            self._lines.append(line)
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            with self._log_path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")

    def update(self, job_id: str, **kwargs: Any) -> None:
        with self._lock:
            if "progress" in kwargs:
                self._progress = float(kwargs["progress"])

    @property
    def progress(self) -> float:
        with self._lock:
            return self._progress

    @property
    def lines(self) -> list[str]:
        with self._lock:
            return list(self._lines)


class BenchmarkOrchestrator:
    """Manages benchmark lifecycle: creation, execution, and result persistence."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._active_benchmarks: dict[str, dict[str, Any]] = {}
        self._restore_completed()

    # -- public API --

    def create_benchmark(self, candidate_ids: list[str] | None = None) -> dict[str, Any]:
        """Create a new benchmark job.  Optionally restrict to specific candidates."""
        status = clip_status()
        if not status["ready"]:
            raise RuntimeError("Benchmark clip is not cached. Use /api/benchmark/ensure-clip first.")

        all_candidates = list_candidates()
        if candidate_ids:
            id_set = set(candidate_ids)
            selected = [c for c in all_candidates if c["id"] in id_set]
        else:
            selected = all_candidates

        if not selected:
            raise RuntimeError("No candidates available for benchmarking.")

        benchmark_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:6]
        benchmark_dir = BENCHMARKS_DIR / benchmark_id
        benchmark_dir.mkdir(parents=True, exist_ok=True)

        job_state = {
            "benchmark_id": benchmark_id,
            "status": "queued",
            "created_at": utc_now_iso(),
            "clip_path": status["path"],
            "clip_dvc": status.get("dvc"),
            "runtime_profile": dict(BENCHMARK_RUNTIME_PROFILE),
            "dvc_runtime": probe_dvc_runtime(),
            "candidates": selected,
            "candidate_results": {},
            "leaderboard": [],
            "progress": 0.0,
            "logs": [],
            "error": None,
        }
        self._persist_state(benchmark_dir, job_state)
        with self._lock:
            self._active_benchmarks[benchmark_id] = job_state

        thread = threading.Thread(
            target=self._run_benchmark,
            args=(benchmark_id, benchmark_dir, job_state),
            daemon=True,
        )
        thread.start()

        return job_state

    def get_benchmark(self, benchmark_id: str) -> dict[str, Any] | None:
        with self._lock:
            state = self._active_benchmarks.get(benchmark_id)
            if state:
                return dict(state)
        # Try loading from disk
        benchmark_dir = BENCHMARKS_DIR / benchmark_id
        summary_path = benchmark_dir / "benchmark_summary.json"
        if summary_path.exists():
            try:
                return json.loads(summary_path.read_text(encoding="utf-8"))
            except Exception:
                pass
        return None

    def list_benchmarks(self) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        if not BENCHMARKS_DIR.exists():
            return results
        for child in sorted(BENCHMARKS_DIR.iterdir(), reverse=True):
            if child.name.startswith("_") or not child.is_dir():
                continue
            summary_path = child / "benchmark_summary.json"
            if summary_path.exists():
                try:
                    data = json.loads(summary_path.read_text(encoding="utf-8"))
                    data["candidate_count"] = len(data.get("candidates") or [])
                    results.append(data)
                except Exception:
                    continue
            else:
                # In-progress
                with self._lock:
                    state = self._active_benchmarks.get(child.name)
                if state:
                    live_state = dict(state)
                    live_state["candidate_count"] = len(live_state.get("candidates") or [])
                    results.append(live_state)
        return results

    def history(self, limit: int = 20) -> list[dict[str, Any]]:
        benchmarks = self.list_benchmarks()
        return benchmarks[:limit]

    # -- internal --

    def _run_benchmark(self, benchmark_id: str, benchmark_dir: Path, state: dict[str, Any]) -> None:
        candidates = state["candidates"]
        clip_path = state["clip_path"]
        total = len(candidates)

        self._update_state(benchmark_id, benchmark_dir, status="running")

        for idx, candidate in enumerate(candidates):
            candidate_id = candidate["id"]
            self._append_log(benchmark_id, benchmark_dir, f"Starting candidate {idx + 1}/{total}: {candidate.get('label', candidate_id)}")

            try:
                result = self._run_candidate(benchmark_id, benchmark_dir, candidate, clip_path)
                with self._lock:
                    st = self._active_benchmarks[benchmark_id]
                    st["candidate_results"][candidate_id] = result
                self._append_log(benchmark_id, benchmark_dir, f"Candidate {candidate_id} completed. Composite={result['score']['composite']}")
            except Exception as exc:
                error_result = {
                    "candidate_id": candidate_id,
                    "status": "failed",
                    "error": str(exc),
                    "score": None,
                    "run_id": None,
                }
                with self._lock:
                    st = self._active_benchmarks[benchmark_id]
                    st["candidate_results"][candidate_id] = error_result
                self._append_log(benchmark_id, benchmark_dir, f"Candidate {candidate_id} failed: {exc}")

            progress = round(((idx + 1) / total) * 100, 2)
            self._update_state(benchmark_id, benchmark_dir, progress=progress)

        # Build leaderboard
        with self._lock:
            st = self._active_benchmarks[benchmark_id]
            leaderboard = self._build_leaderboard(st["candidate_results"], candidates)
            st["leaderboard"] = leaderboard
            st["status"] = "completed"
            st["progress"] = 100.0
        self._persist_state(benchmark_dir, self._active_benchmarks[benchmark_id])
        self._append_log(benchmark_id, benchmark_dir, "Benchmark completed.")

    def _run_candidate(
        self,
        benchmark_id: str,
        benchmark_dir: Path,
        candidate: dict[str, Any],
        clip_path: str,
    ) -> dict[str, Any]:
        candidate_id = candidate["id"]
        detector_path = candidate["path"]

        # Create a normal analysis run under backend/runs/
        run_id = f"bench_{benchmark_id}_{candidate_id}_{uuid.uuid4().hex[:4]}"
        run_dir = RUNS_DIR / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        # Respect per-candidate pipeline overrides (e.g. SoccerMaster uses
        # its own unified pipeline rather than classic + separate detector).
        pipeline = candidate.get("pipeline_override") or BENCHMARK_RUNTIME_PROFILE["pipeline"]
        keypoint_model = candidate.get("keypoint_override") or BENCHMARK_RUNTIME_PROFILE["keypoint_model"]

        # SoccerMaster doesn't use separate YOLO detector/ball models -- its
        # unified backbone handles everything.  But analyze_video still runs
        # resolve_detector_spec() on player_model before checking pipeline mode,
        # so we pass the standard soccana detector as a valid-but-unused default.
        if pipeline == "soccermaster":
            fallback_detector = resolve_model_path("soccana", "detector")
        else:
            fallback_detector = detector_path

        config_payload = {
            "source_video_path": clip_path,
            "label_path": "",
            "pipeline": pipeline,
            "player_model": fallback_detector,
            "ball_model": fallback_detector,
            "keypoint_model": keypoint_model,
            "tracker_mode": BENCHMARK_RUNTIME_PROFILE["tracker_mode"],
            "include_ball": BENCHMARK_RUNTIME_PROFILE["include_ball"],
            "player_conf": BENCHMARK_RUNTIME_PROFILE["player_conf"],
            "ball_conf": BENCHMARK_RUNTIME_PROFILE["ball_conf"],
            "iou": BENCHMARK_RUNTIME_PROFILE["iou"],
        }

        job_mgr = _BenchmarkJobManager(benchmark_id, candidate_id, benchmark_dir)
        job_id = f"bench_{candidate_id}_{uuid.uuid4().hex[:4]}"

        summary = analyze_wide_angle_video(
            job_id=job_id,
            run_dir=run_dir,
            config_payload=config_payload,
            job_manager=job_mgr,
            job_control=None,
        )

        score = compute_composite_score(summary)

        return {
            "candidate_id": candidate_id,
            "status": "completed",
            "error": None,
            "run_id": run_id,
            "run_dir": str(run_dir),
            "logs": job_mgr.lines,
            "score": score,
            "summary_excerpt": {
                "frames_processed": summary.get("frames_processed"),
                "fps": summary.get("fps"),
                "unique_player_track_ids": summary.get("unique_player_track_ids"),
                "average_player_detections_per_frame": summary.get("average_player_detections_per_frame"),
                "average_track_length": summary.get("average_track_length"),
                "player_track_churn_ratio": summary.get("player_track_churn_ratio"),
                "field_registered_ratio": summary.get("field_registered_ratio"),
                "tracklet_merges_applied": summary.get("tracklet_merges_applied"),
                "overlay_video": summary.get("overlay_video"),
                "summary_json": summary.get("summary_json"),
                "diagnostics_summary_line": summary.get("diagnostics_summary_line"),
                "diagnostics": summary.get("diagnostics") or [],
                "heuristic_diagnostics": summary.get("heuristic_diagnostics") or [],
                "diagnostics_source": summary.get("diagnostics_source"),
            },
        }

    def _build_leaderboard(
        self,
        candidate_results: dict[str, dict[str, Any]],
        candidates: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        candidate_map = {c["id"]: c for c in candidates}
        rows: list[dict[str, Any]] = []
        for cid, result in candidate_results.items():
            entry = candidate_map.get(cid, {})
            score = result.get("score")
            rows.append({
                "rank": 0,
                "candidate_id": cid,
                "label": entry.get("label", cid),
                "source": entry.get("source", "unknown"),
                "pipeline": entry.get("pipeline_override") or BENCHMARK_RUNTIME_PROFILE["pipeline"],
                "status": result.get("status", "unknown"),
                "composite": score["composite"] if score else None,
                "track_stability": score["track_stability"] if score else None,
                "calibration": score["calibration"] if score else None,
                "coverage": score["coverage"] if score else None,
                "throughput": score["throughput"] if score else None,
                "run_id": result.get("run_id"),
                "error": result.get("error"),
            })
        # Sort by composite descending, failed entries at bottom
        rows.sort(key=lambda r: (0 if r["composite"] is not None else 1, -(r["composite"] or 0)))
        for i, row in enumerate(rows):
            row["rank"] = i + 1
        return rows

    def _update_state(self, benchmark_id: str, benchmark_dir: Path, **kwargs: Any) -> None:
        with self._lock:
            state = self._active_benchmarks.get(benchmark_id)
            if state:
                state.update(kwargs)
                self._persist_state(benchmark_dir, state)

    def _append_log(self, benchmark_id: str, benchmark_dir: Path, message: str) -> None:
        stamp = datetime.utcnow().strftime("%H:%M:%S")
        line = f"[{stamp}] {message}"
        with self._lock:
            state = self._active_benchmarks.get(benchmark_id)
            if state:
                state.setdefault("logs", []).append(line)

    def _persist_state(self, benchmark_dir: Path, state: dict[str, Any]) -> None:
        summary_path = benchmark_dir / "benchmark_summary.json"
        benchmark_dir.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")

    def _restore_completed(self) -> None:
        """Load completed benchmark summaries from disk on startup."""
        if not BENCHMARKS_DIR.exists():
            return
        for child in BENCHMARKS_DIR.iterdir():
            if child.name.startswith("_") or not child.is_dir():
                continue
            summary_path = child / "benchmark_summary.json"
            if summary_path.exists():
                try:
                    data = json.loads(summary_path.read_text(encoding="utf-8"))
                    if data.get("status") in ("completed", "failed"):
                        with self._lock:
                            self._active_benchmarks[child.name] = data
                except Exception:
                    continue


# Module-level singleton
benchmark_orchestrator = BenchmarkOrchestrator()
