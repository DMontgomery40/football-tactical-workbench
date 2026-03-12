from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2

SN_GAMESTATE_ENV_VAR = "SN_GAMESTATE_REPO_PATH"
SN_GAMESTATE_REPO_URL = "https://github.com/SoccerNet/sn-gamestate"
SN_GAMESTATE_PAPER_URL = "https://arxiv.org/abs/2404.11335"
SN_GAMESTATE_TASK_URL = "https://www.soccer-net.org/tasks/new-game-state-reconstruction"
MACOS_CPU_TRACKLAB_BOOTSTRAP = (
    "import os, sys, torch; "
    "os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1'); "
    "torch.backends.mps.is_available = lambda: False; "
    "from tracklab.main import main; "
    "sys.argv = ['tracklab', *sys.argv[1:]]; "
    "raise SystemExit(main())"
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _candidate_repo_paths() -> list[Path]:
    repo_root = _repo_root()
    candidates: list[Path] = []
    raw_env = os.environ.get(SN_GAMESTATE_ENV_VAR, "").strip()
    if raw_env:
        candidates.append(Path(raw_env).expanduser())
    candidates.extend(
        [
            repo_root / "external" / "sn-gamestate",
            repo_root.parent / "sn-gamestate",
            Path.home() / "sn-gamestate",
        ]
    )
    deduped: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)
    return deduped


def resolve_sn_gamestate_repo() -> Path | None:
    for candidate in _candidate_repo_paths():
        if (candidate / "sn_gamestate" / "configs" / "soccernet.yaml").exists():
            return candidate.resolve()
    return None


@lru_cache(maxsize=8)
def _sn_gamestate_python_module_available(repo_path: str, module_name: str, uv_path: str | None = None) -> bool:
    resolved_uv = uv_path or shutil.which("uv")
    if not resolved_uv:
        return False
    command = [
        str(resolved_uv),
        "run",
        "python",
        "-c",
        (
            "import importlib.util, sys; "
            f"sys.exit(0 if importlib.util.find_spec({module_name!r}) else 1)"
        ),
    ]
    completed = subprocess.run(
        command,
        cwd=str(repo_path),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    return completed.returncode == 0


def _review_mode_jersey_overrides(repo_path: Path, uv_path: str) -> tuple[list[str], str]:
    if _sn_gamestate_python_module_available(str(repo_path), "mmcv", uv_path):
        return [], "mmocr"
    if _sn_gamestate_python_module_available(str(repo_path), "easyocr", uv_path):
        return ["modules/jersey_number_detect=easyocr"], "easyocr_fallback"
    return [], "mmocr_missing"


def sn_gamestate_status() -> dict[str, Any]:
    repo_path = resolve_sn_gamestate_repo()
    uv_path = shutil.which("uv")
    repo_present = repo_path is not None
    uv_available = bool(uv_path)
    available = repo_present and uv_available
    return {
        "available": available,
        "repo_present": repo_present,
        "uv_available": uv_available,
        "repo_path": str(repo_path) if repo_path else None,
        "uv_path": uv_path,
        "env_var": SN_GAMESTATE_ENV_VAR,
        "single_video_command": "uv run tracklab -cn soccernet dataset=youtube dataset.video_path=/abs/path/to/video.mp4",
        "dataset_eval_command": "uv run tracklab -cn soccernet",
        "weights": {
            "mode": "automatic_first_run",
            "note": "The official baseline auto-downloads SoccerNetGS data and model weights on first run.",
        },
        "evaluation": {
            "metric": "GS-HOTA",
            "available_for_single_video": False,
            "note": "Official GS-HOTA evaluation requires a labeled SoccerNetGS split, not an arbitrary external video.",
        },
        "note": (
            "sn-gamestate repo + uv detected. First run may still need to build the environment and auto-download dataset/model weights."
            if available
            else "Clone sn-gamestate and use uv. The official baseline will then auto-download dataset/model weights on first run."
        ),
        "links": {
            "repo": SN_GAMESTATE_REPO_URL,
            "paper": SN_GAMESTATE_PAPER_URL,
            "task": SN_GAMESTATE_TASK_URL,
        },
    }


def _inspect_video(path: Path) -> dict[str, Any]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    duration_seconds = float(frame_count / fps) if fps > 0 and frame_count > 0 else 0.0
    return {
        "fps": round(fps, 4) if fps > 0 else 0.0,
        "width": width,
        "height": height,
        "frame_count": frame_count,
        "duration_seconds": round(duration_seconds, 4),
    }


def _find_visualization_video(external_run_dir: Path) -> Path:
    videos = sorted(external_run_dir.rglob("*.mp4"))
    if not videos:
        raise RuntimeError("sn-gamestate completed without producing a visualization video")
    for candidate in videos:
        if "visualization" in candidate.parts:
            return candidate
    return videos[0]


def _build_sn_gamestate_command(
    *,
    uv_path: str,
    source_video_path: Path,
    external_run_dir: Path,
    jersey_overrides: list[str],
) -> tuple[list[str], dict[str, str], str]:
    common_args = [
        "-cn",
        "soccernet",
        "dataset=youtube",
        "dataset.eval_set=val",
        f"dataset.video_path={source_video_path}",
        "eval_tracking=False",
        "test_tracking=True",
        "visualization.cfg.save_videos=True",
        "experiment_name=fpw-sn-gamestate",
        f"hydra.run.dir={external_run_dir}",
    ]
    common_args.extend(jersey_overrides)

    process_env = os.environ.copy()
    process_env.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    if platform.system() == "Darwin":
        common_args.append("num_cores=0")
        return (
            [uv_path, "run", "python", "-c", MACOS_CPU_TRACKLAB_BOOTSTRAP, *common_args],
            process_env,
            "cpu_safe_macos",
        )

    return [uv_path, "run", "tracklab", *common_args], process_env, "default"


def run_sn_gamestate_analysis(
    *,
    job_id: str,
    run_dir: Path,
    source_video_path: Path,
    job_manager: Any,
) -> dict[str, Any]:
    status = sn_gamestate_status()
    if not status["available"]:
        raise RuntimeError(status["note"])

    repo_path = Path(str(status["repo_path"]))
    outputs_dir = run_dir / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    external_run_dir = run_dir / "sn_gamestate_run"
    external_run_dir.mkdir(parents=True, exist_ok=True)
    jersey_overrides, jersey_backend = _review_mode_jersey_overrides(repo_path, str(status["uv_path"]))
    command, process_env, execution_mode = _build_sn_gamestate_command(
        uv_path=str(status["uv_path"]),
        source_video_path=source_video_path,
        external_run_dir=external_run_dir,
        jersey_overrides=jersey_overrides,
    )

    job_manager.log(job_id, f"Running sn-gamestate from {repo_path}")
    job_manager.log(job_id, f"Command: {' '.join(command)}")
    job_manager.log(job_id, status["note"])
    if execution_mode == "cpu_safe_macos":
        job_manager.log(
            job_id,
            "macOS stability mode is enabled for sn-gamestate: forcing CPU inference and single-process loaders to avoid MPS crashes during arbitrary-clip review runs.",
        )
    if jersey_backend == "easyocr_fallback":
        job_manager.log(
            job_id,
            "MMOCR is unavailable in the local sn-gamestate environment, so this arbitrary-clip review run is using EasyOCR for jersey number detection instead.",
        )
    elif jersey_backend == "mmocr_missing":
        job_manager.log(
            job_id,
            "MMOCR dependencies are missing in the local sn-gamestate environment and no EasyOCR fallback was found. The external baseline may fail before producing an overlay.",
        )
    job_manager.log(
        job_id,
        "Official GS-HOTA evaluation is only available on labeled SoccerNetGS splits. Arbitrary external videos get overlay output but no official accuracy metric.",
    )

    process = subprocess.Popen(
        command,
        cwd=str(repo_path),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=process_env,
    )
    try:
        if process.stdout is not None:
            for line in process.stdout:
                text = line.rstrip()
                if text:
                    job_manager.log(job_id, f"[sn-gamestate] {text}")
        return_code = process.wait()
    finally:
        if process.stdout is not None:
            process.stdout.close()

    if return_code != 0:
        raise RuntimeError(f"sn-gamestate exited with code {return_code}")

    overlay_source = _find_visualization_video(external_run_dir)
    overlay_target = outputs_dir / "overlay.mp4"
    shutil.copy2(str(overlay_source), str(overlay_target))

    source_meta = _inspect_video(source_video_path)
    overlay_meta = _inspect_video(overlay_target)
    summary_json_path = outputs_dir / "summary.json"
    summary = {
        "summary_version": "2026.03.10",
        "pipeline": "sn_gamestate",
        "job_id": job_id,
        "run_dir": str(run_dir),
        "input_video": str(source_video_path),
        "overlay_video": f"/runs/{run_dir.name}/outputs/overlay.mp4",
        "summary_json": f"/runs/{run_dir.name}/outputs/summary.json",
        "device": "",
        "field_calibration_device": "",
        "player_model": "sn-gamestate external baseline",
        "requested_player_tracker_mode": None,
        "player_tracker_mode": "external_tracklab",
        "resolved_player_tracker_mode": "external_tracklab",
        "player_tracker_runtime": "tracklab_sn_gamestate",
        "player_tracker_backend": "tracklab",
        "ball_model": "sn-gamestate external baseline",
        "field_calibration_model": "sn-gamestate external baseline",
        "include_ball": True,
        "player_conf": 0.0,
        "ball_conf": 0.0,
        "iou": 0.0,
        "frames_processed": source_meta["frame_count"],
        "fps": overlay_meta["fps"],
        "diagnostics_source": "heuristic",
        "diagnostics_provider": None,
        "diagnostics_model": None,
        "diagnostics_status": "completed",
        "diagnostics_summary_line": (
            "sn-gamestate external baseline completed. Review the generated overlay and external run logs. "
            "Official GS-HOTA is not produced for arbitrary external videos."
        ),
        "diagnostics_error": "",
        "diagnostics": [
            {
                "level": "good",
                "title": "sn-gamestate baseline completed",
                "message": "The external TrackLab-based pipeline ran successfully and produced an overlay video for review.",
                "next_step": "Inspect the overlay and the external run directory before comparing against native pipeline outputs.",
                "implementation_diagnosis": "This run was produced by the external SoccerNet Game State Reconstruction baseline, not the native workbench analyzer.",
                "suggested_fix": "",
                "code_refs": ["backend/app/sn_gamestate.py::run_sn_gamestate_analysis"],
                "evidence_keys": [],
            }
        ],
        "heuristic_diagnostics": [],
        "created_at": datetime.utcnow().isoformat() + "Z",
        "learn_cards": [],
        "experiments": [],
        "top_tracks": [],
        "external_pipeline": {
            "id": "sn_gamestate",
            "repo_path": str(repo_path),
            "hydra_run_dir": str(external_run_dir),
            "command": command,
            "output_video": str(overlay_source),
            "source_duration_seconds": source_meta["duration_seconds"],
            "evaluation_metric": status["evaluation"]["metric"],
            "evaluation_enabled": False,
            "weights_mode": status["weights"]["mode"],
            "jersey_number_backend": jersey_backend,
            "execution_mode": execution_mode,
        },
    }
    summary_json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
