from __future__ import annotations

import csv
import json
import mimetypes
import platform as platform_module
import re
import ssl
import shutil
import subprocess
import threading
import time
import uuid
import zipfile
from collections import defaultdict
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from SoccerNet.Downloader import SoccerNetDownloader
from SoccerNet.utils import getListGames
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.ai_diagnostics import PROMPT_VERSION as CURRENT_DIAGNOSTICS_PROMPT_VERSION, generate_run_diagnostics, resolve_provider_config
from app.reid_tracker import DEFAULT_PLAYER_TRACKER_MODE, PLAYER_TRACKER_MODE_OPTIONS
from app.schemas import (
    ActiveExperimentResponse,
    AnalyzeAcceptedResponse,
    ConfigResponse,
    FolderScanResponse,
    JOB_STATE_SCHEMA_VERSION,
    JobStateResponse,
    RunSummary,
    SOURCE_SCHEMA_VERSION,
    SUMMARY_SCHEMA_VERSION,
    SoccerNetConfigResponse,
    SoccerNetGamesResponse,
    SourceResponse,
)
from app.training import (
    build_training_backend_config,
    inspect_training_dataset,
    prepare_training_run_inputs,
    scan_training_dataset_path,
)
from app.wide_angle import (
    AnalysisStoppedError,
    CALIBRATION_REFRESH_FRAMES,
    PLAYER_MODEL_OPTIONS,
    TACTICAL_HELP_CATALOG,
    TACTICAL_LEARN_CARDS,
    analyze_video as analyze_wide_angle_video,
    generate_live_preview_stream,
    prewarm_default_models,
    resolve_model_path,
)

TRAINING_IMPORT_ERROR: str | None = None
try:
    from app.training_manager import TrainingManager
    from app.training_registry import TrainingRegistry
except Exception as exc:
    TrainingManager = None  # type: ignore[assignment]
    TrainingRegistry = None  # type: ignore[assignment]
    TRAINING_IMPORT_ERROR = str(exc)

BASE_DIR = Path(__file__).resolve().parent.parent
RUNS_DIR = BASE_DIR / "runs"
TRAINING_RUNS_DIR = BASE_DIR / "training_runs"
UPLOADS_DIR = BASE_DIR / "uploads"
SOCCERNET_DIR = BASE_DIR / "datasets" / "soccernet"
EXPERIMENTS_DIR = BASE_DIR / "experiments"
SOURCE_REGISTRY_PATH = UPLOADS_DIR / "sources.json"
RUNS_DIR.mkdir(parents=True, exist_ok=True)
TRAINING_RUNS_DIR.mkdir(parents=True, exist_ok=True)
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
SOCCERNET_DIR.mkdir(parents=True, exist_ok=True)
EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
ACTIVE_EXPERIMENT_FRESHNESS_SECONDS = 300

PERSON_CLASS_ID = 0
BALL_CLASS_ID = 32
KEYPOINT_CONF_THRESHOLD = 0.35
DEFAULT_TRACKER = "bytetrack.yaml"
TEAM_DRAW_COLORS: dict[str, tuple[int, int, int]] = {
    "home": (255, 160, 70),
    "away": (80, 100, 255),
    "unassigned": (175, 175, 175),
    "ball": (0, 215, 255),
}
PITCH_MAP_SIZE = (360, 232)
SKELETON_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6),
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (5, 11), (6, 12),
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
]


class FolderScanRequest(BaseModel):
    folder_path: str


class SoccerNetDownloadRequest(BaseModel):
    split: str
    game: str
    password: str
    files: list[str]


class TrainDatasetScanRequest(BaseModel):
    path: str


class TrainingDetectJobRequest(BaseModel):
    base_weights: str = "soccana"
    dataset_path: str
    run_name: str = ""
    epochs: int = 50
    imgsz: int = 640
    batch: int = 16
    device: str = "auto"
    workers: int = 4
    patience: int = 20
    freeze: int | None = None
    cache: bool = False


class TrainingRegistryActivateRequest(BaseModel):
    detector_id: str


def dump_json_model(instance: BaseModel) -> dict[str, Any]:
    return instance.model_dump(mode="json", exclude_none=True)


def serialize_run_summary(summary: dict[str, Any] | None) -> dict[str, Any] | None:
    if summary is None:
        return None
    normalized = dict(summary)
    normalize_fn = globals().get("normalize_persisted_summary")
    if callable(normalize_fn):
        normalized = normalize_fn(normalized)
    normalized["summary_version"] = normalized.get("summary_version", SUMMARY_SCHEMA_VERSION)
    return dump_json_model(RunSummary.model_validate(normalized))


@dataclass
class SourceState:
    source_id: str
    path: str
    display_name: str
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    uploaded: bool = False

    def as_dict(self, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        payload = {
            "source_state_version": SOURCE_SCHEMA_VERSION,
            "source_id": self.source_id,
            "path": self.path,
            "display_name": self.display_name,
            "created_at": self.created_at,
            "uploaded": self.uploaded,
            "video_url": f"/api/source/{self.source_id}/video",
            **(metadata or {}),
        }
        return dump_json_model(SourceResponse.model_validate(payload))


@dataclass
class JobState:
    job_id: str
    status: str = "queued"
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    progress: float = 0.0
    logs: list[str] = field(default_factory=list)
    run_dir: str = ""
    summary: dict[str, Any] | None = None
    error: str | None = None
    restart_config: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        run_id = Path(self.run_dir).name if self.run_dir else ""
        return dump_json_model(
            JobStateResponse(
                job_state_version=JOB_STATE_SCHEMA_VERSION,
                job_id=self.job_id,
                run_id=run_id,
                status=self.status,
                created_at=self.created_at,
                progress=round(self.progress, 2),
                logs=self.logs[-250:],
                run_dir=self.run_dir,
                summary=serialize_run_summary(self.summary),
                error=self.error,
            )
        )

    def persistence_dict(self) -> dict[str, Any]:
        payload = self.as_dict()
        payload["restart_config"] = self.restart_config
        return payload


class JobManager:
    def __init__(self, runs_dir: Path) -> None:
        self._jobs: dict[str, JobState] = {}
        self._lock = threading.Lock()
        self._runs_dir = runs_dir
        self._restartable_job_ids: list[str] = []
        self._restore()

    def _job_state_path(self, run_dir: str) -> Path:
        return Path(run_dir) / "job_state.json"

    def _persist_locked(self, job: JobState) -> None:
        run_dir = Path(job.run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        self._job_state_path(job.run_dir).write_text(json.dumps(job.persistence_dict(), indent=2), encoding="utf-8")

    def _restore(self) -> None:
        if not self._runs_dir.exists():
            return

        restored_at = datetime.utcnow().strftime("%H:%M:%S")
        for run_dir in sorted(self._runs_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            state_path = run_dir / "job_state.json"
            if not state_path.exists():
                continue
            try:
                payload = json.loads(state_path.read_text(encoding="utf-8"))
            except Exception:
                continue

            job = JobState(
                job_id=str(payload.get("job_id", "")),
                status=str(payload.get("status", "queued")),
                created_at=str(payload.get("created_at", datetime.utcnow().isoformat() + "Z")),
                progress=float(payload.get("progress", 0.0) or 0.0),
                logs=list(payload.get("logs") or []),
                run_dir=str(payload.get("run_dir", run_dir)),
                summary=payload.get("summary"),
                error=payload.get("error"),
                restart_config=payload.get("restart_config") if isinstance(payload.get("restart_config"), dict) else None,
            )
            if not job.job_id:
                continue
            if job.status in {"queued", "running", "paused", "stopping"}:
                if job.restart_config:
                    job.status = "queued"
                    job.progress = 0.0
                    job.error = None
                    job.summary = None
                    job.logs.append(f"[{restored_at}] Backend restarted; restarting analysis from frame 0.")
                    self._restartable_job_ids.append(job.job_id)
                else:
                    job.status = "failed"
                    job.error = job.error or "Backend restarted before the job finished."
                    job.logs.append(f"[{restored_at}] Backend restarted before the job finished.")
            self._jobs[job.job_id] = job
            self._persist_locked(job)

    def create(self, run_dir: Path, restart_config: dict[str, Any] | None = None) -> JobState:
        job = JobState(job_id=uuid.uuid4().hex[:12], run_dir=str(run_dir), restart_config=restart_config)
        with self._lock:
            self._jobs[job.job_id] = job
            self._persist_locked(job)
        return job

    def get(self, job_id: str) -> JobState | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self) -> list[dict[str, Any]]:
        with self._lock:
            return [job.as_dict() for job in sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)]

    def update(self, job_id: str, **kwargs: Any) -> None:
        with self._lock:
            job = self._jobs[job_id]
            for key, value in kwargs.items():
                setattr(job, key, value)
            self._persist_locked(job)

    def log(self, job_id: str, message: str) -> None:
        stamp = datetime.utcnow().strftime("%H:%M:%S")
        with self._lock:
            job = self._jobs[job_id]
            job.logs.append(f"[{stamp}] {message}")
            self._persist_locked(job)

    def consume_restartable_jobs(self) -> list[tuple[str, Path, dict[str, Any]]]:
        with self._lock:
            recovered: list[tuple[str, Path, dict[str, Any]]] = []
            for job_id in self._restartable_job_ids:
                job = self._jobs.get(job_id)
                if job is None or not job.restart_config:
                    continue
                recovered.append((job.job_id, Path(job.run_dir), dict(job.restart_config)))
            self._restartable_job_ids = []
            return recovered


job_manager = JobManager(RUNS_DIR)


class JobControl:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pause_requested = False
        self._stop_requested = False

    def request_pause(self) -> None:
        with self._lock:
            self._pause_requested = True

    def request_resume(self) -> None:
        with self._lock:
            self._pause_requested = False

    def request_stop(self) -> None:
        with self._lock:
            self._stop_requested = True

    def is_pause_requested(self) -> bool:
        with self._lock:
            return self._pause_requested

    def is_stop_requested(self) -> bool:
        with self._lock:
            return self._stop_requested


class JobControlManager:
    def __init__(self) -> None:
        self._controls: dict[str, JobControl] = {}
        self._lock = threading.Lock()

    def create(self, job_id: str) -> JobControl:
        control = JobControl()
        with self._lock:
            self._controls[job_id] = control
        return control

    def get(self, job_id: str) -> JobControl | None:
        with self._lock:
            return self._controls.get(job_id)

    def clear(self, job_id: str) -> None:
        with self._lock:
            self._controls.pop(job_id, None)


job_control_manager = JobControlManager()


class SourceManager:
    def __init__(self, registry_path: Path) -> None:
        self._sources: dict[str, SourceState] = {}
        self._lock = threading.Lock()
        self._registry_path = registry_path
        self._restore()

    def _persist_locked(self) -> None:
        payload: list[dict[str, Any]] = []
        for source in self._sources.values():
            source_path = Path(source.path).expanduser()
            if not source_path.exists() or not source_path.is_file():
                continue
            payload.append(
                {
                    "source_id": source.source_id,
                    "path": source.path,
                    "display_name": source.display_name,
                    "created_at": source.created_at,
                    "uploaded": source.uploaded,
                }
            )
        self._registry_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _restore(self) -> None:
        if not self._registry_path.exists():
            return
        try:
            payload = json.loads(self._registry_path.read_text(encoding="utf-8"))
        except Exception:
            return
        if not isinstance(payload, list):
            return
        for item in payload:
            if not isinstance(item, dict):
                continue
            source_path = Path(str(item.get("path", ""))).expanduser()
            if not source_path.exists() or not source_path.is_file():
                continue
            source = SourceState(
                source_id=str(item.get("source_id", "")),
                path=str(source_path.resolve()),
                display_name=str(item.get("display_name") or source_path.name),
                created_at=str(item.get("created_at") or datetime.utcnow().isoformat() + "Z"),
                uploaded=bool(item.get("uploaded", False)),
            )
            if source.source_id:
                self._sources[source.source_id] = source

    def register(self, path: Path, display_name: str, uploaded: bool = False) -> SourceState:
        resolved_path = str(path.expanduser().resolve())
        with self._lock:
            for source in self._sources.values():
                try:
                    existing_path = str(Path(source.path).expanduser().resolve())
                except Exception:
                    existing_path = str(source.path)
                if existing_path == resolved_path:
                    source.display_name = display_name or source.display_name
                    source.uploaded = bool(source.uploaded or uploaded)
                    self._persist_locked()
                    return source

            source = SourceState(
                source_id=uuid.uuid4().hex[:12],
                path=resolved_path,
                display_name=display_name,
                uploaded=uploaded,
            )
            self._sources[source.source_id] = source
            self._persist_locked()
            return source

    def get(self, source_id: str) -> SourceState | None:
        with self._lock:
            return self._sources.get(source_id)


source_manager = SourceManager(SOURCE_REGISTRY_PATH)


training_manager = None
training_registry = None
if TrainingManager and TrainingRegistry:
    try:
        training_manager = TrainingManager(TRAINING_RUNS_DIR)
        training_registry = TrainingRegistry(BASE_DIR / "models" / "registry.json")
    except Exception as exc:
        TRAINING_IMPORT_ERROR = TRAINING_IMPORT_ERROR or str(exc)
        training_manager = None
        training_registry = None


@dataclass
class DownloadJobState:
    job_id: str
    status: str = "queued"
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    progress: float = 0.0
    logs: list[str] = field(default_factory=list)
    split: str = ""
    game: str = ""
    files: list[str] = field(default_factory=list)
    output_dir: str = ""
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "status": self.status,
            "created_at": self.created_at,
            "progress": round(self.progress, 2),
            "logs": self.logs[-250:],
            "split": self.split,
            "game": self.game,
            "files": self.files,
            "output_dir": self.output_dir,
            "error": self.error,
        }


class DownloadJobManager:
    def __init__(self) -> None:
        self._jobs: dict[str, DownloadJobState] = {}
        self._lock = threading.Lock()

    def create(self, split: str, game: str, files: list[str], output_dir: Path) -> DownloadJobState:
        job = DownloadJobState(
            job_id=uuid.uuid4().hex[:12],
            split=split,
            game=game,
            files=list(files),
            output_dir=str(output_dir),
        )
        with self._lock:
            self._jobs[job.job_id] = job
        return job

    def get(self, job_id: str) -> DownloadJobState | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self) -> list[dict[str, Any]]:
        with self._lock:
            return [job.as_dict() for job in sorted(self._jobs.values(), key=lambda item: item.created_at, reverse=True)]

    def update(self, job_id: str, **kwargs: Any) -> None:
        with self._lock:
            job = self._jobs[job_id]
            for key, value in kwargs.items():
                setattr(job, key, value)

    def log(self, job_id: str, message: str) -> None:
        stamp = datetime.utcnow().strftime("%H:%M:%S")
        with self._lock:
            job = self._jobs[job_id]
            job.logs.append(f"[{stamp}] {message}")


download_job_manager = DownloadJobManager()


def training_available() -> bool:
    return training_manager is not None and training_registry is not None and TRAINING_IMPORT_ERROR is None


def require_training_available() -> tuple[Any, Any]:
    if not training_available():
        detail = TRAINING_IMPORT_ERROR or "Training support is unavailable in this backend build."
        raise HTTPException(status_code=503, detail=detail)
    return training_manager, training_registry


@asynccontextmanager
async def lifespan(app: FastAPI):
    if training_available():
        try:
            training_registry.init_if_absent()
        except Exception as exc:
            print(f"[startup] training registry init skipped: {exc}")

    try:
        prewarm_default_models()
    except Exception as exc:
        print(f"[startup] model prewarm skipped: {exc}")

    for job_id, run_dir, config_payload in job_manager.consume_restartable_jobs():
        job_control_manager.create(job_id)
        job_manager.update(job_id, status="running", progress=0.0, error=None)
        job_manager.log(job_id, "Restarting analysis after backend restart")
        thread = threading.Thread(
            target=_run_analysis_job,
            args=(job_id, run_dir, config_payload),
            daemon=True,
        )
        thread.start()

    if training_available():
        for job_id, run_dir, config_payload in training_manager.consume_restartable_jobs():
            training_manager.launch(job_id, run_dir, config_payload)

    yield


app = FastAPI(title="Football Tactical Workbench API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/runs", StaticFiles(directory=RUNS_DIR), name="runs")


LEARN_CARDS = [
    {
        "title": "1. Detection plus tracking",
        "what_it_does": "Runs a regular player detector and ByteTrack so each visible player can keep an ID across frames.",
        "what_breaks": "Wide pans, zoom swings, occlusion, and tiny players still cause ID churn and false merges.",
        "what_to_try_next": "Start on short broadcast segments with one camera phase and tune detection confidence before adding more ambition.",
    },
    {
        "title": "2. Team ID by jersey color",
        "what_it_does": "Samples upper-body jersey colors from tracked players and clusters them into two team buckets.",
        "what_breaks": "Goalkeepers, referees, shadows, motion blur, and dominant grass pixels can poison the color split.",
        "what_to_try_next": "Use clips with strong shirt contrast first and trust track-level votes more than one noisy frame.",
    },
    {
        "title": "3. Ball tracking",
        "what_it_does": "Runs a second detector on the ball so you can see whether the sequence reads as actual match play rather than generic crowd motion.",
        "what_breaks": "The ball is small, blurry, and regularly disappears into feet, socks, and advertising boards.",
        "what_to_try_next": "Treat ball tracking as a supporting signal; stable player tracks are the main deliverable.",
    },
    {
        "title": "4. Optional pitch projection",
        "what_it_does": "Accepts four manual pitch points, builds a homography, and projects player foot-points onto a minimap inset.",
        "what_breaks": "Bad point order or weak line visibility produces nonsense top-down positions immediately.",
        "what_to_try_next": "Use one clean frame with visible touchline geometry and enter corners in a consistent order.",
    },
    {
        "title": "5. Debug first",
        "what_it_does": "Summarizes tracker churn, team-separation quality, ball hit-rate, and homography status in plain language.",
        "what_breaks": "It is easy to confuse a flashy overlay with a pipeline that actually survives wide-angle football footage.",
        "what_to_try_next": "If the diagnostics look weak, fix the first broken stage instead of piling on more model complexity.",
    },
]


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "backend_port": 8431,
        "frontend_port": 4317,
        "runs_dir": str(RUNS_DIR),
        "training_available": training_available(),
    }


def available_runtime_devices() -> list[str]:
    devices: list[str] = []
    try:
        import torch

        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            devices.append("mps")
        if torch.cuda.is_available():
            devices.append("cuda")
    except Exception:
        pass
    devices.append("cpu")
    ordered: list[str] = []
    for device in devices:
        if device not in ordered:
            ordered.append(device)
    return ordered


def host_platform_name() -> str:
    system_name = platform_module.system().lower()
    if system_name == "darwin":
        return "macos"
    if system_name == "windows":
        return "windows"
    if system_name == "linux":
        return "linux"
    return system_name or "unknown"


def build_runtime_profile() -> dict[str, Any]:
    backend_config = build_training_backend_config()
    available_devices = available_runtime_devices()
    preferred_device = available_devices[0] if available_devices else "cpu"
    runtime_notes = [
        "Detector inference currently uses the Ultralytics/PyTorch stack already present in this repo.",
        "The planned future inference runway is ONNX Runtime with CoreML on Apple Silicon and ONNX Runtime with CUDA on GPU hosts.",
    ]
    if "mps" in available_devices:
        runtime_notes.insert(
            1,
            "Apple Silicon is the primary local target. Detector inference prefers MPS, while field calibration currently falls back to CPU when detector inference uses MPS.",
        )
    elif "cuda" in available_devices:
        runtime_notes.insert(
            1,
            "CUDA is available on this host for detector workloads. ONNX Runtime CUDA remains planned for future inference, not the active default today.",
        )
    else:
        runtime_notes.insert(
            1,
            "This host is currently CPU-first. ONNX Runtime remains the planned future path for CoreML and CUDA targets.",
        )
    return {
        "backend": "ultralytics_torch",
        "backend_label": "Ultralytics YOLO / PyTorch",
        "backend_version": backend_config.get("backend_version"),
        "host_platform": host_platform_name(),
        "host_arch": platform_module.machine().lower() or "unknown",
        "preferred_device": preferred_device,
        "available_devices": available_devices,
        "field_calibration_device_policy": "cpu_when_detector_uses_mps" if "mps" in available_devices else "same_as_detector",
        "detector_export_formats": ["onnx", "coreml"],
        "planned_backends": [
            {"id": "onnxruntime_coreml", "label": "ONNX Runtime + CoreML EP"},
            {"id": "onnxruntime_cuda", "label": "ONNX Runtime + CUDA EP"},
        ],
        "runtime_notes": runtime_notes,
        "license_notes": [backend_config.get("license_note")] if backend_config.get("license_note") else [],
    }


def build_training_device_options(runtime_profile: dict[str, Any]) -> list[dict[str, str]]:
    preferred_device = str(runtime_profile.get("preferred_device") or "cpu").upper()
    labels = {
        "auto": f"Auto ({preferred_device} recommended)" if preferred_device != "CPU" else "Auto (recommended)",
        "mps": "Apple Silicon MPS",
        "cuda": "CUDA GPU",
        "cpu": "CPU only",
    }
    option_ids = ["auto", *(runtime_profile.get("available_devices") or ["cpu"])]
    ordered: list[str] = []
    for option_id in option_ids:
        if option_id not in ordered:
            ordered.append(option_id)
    return [{"id": option_id, "label": labels.get(option_id, option_id.upper())} for option_id in ordered]


def build_training_device_guidance(runtime_profile: dict[str, Any]) -> str:
    available_devices = set(runtime_profile.get("available_devices") or [])
    if "mps" in available_devices:
        return (
            "Mac-first default: use Auto or Apple Silicon MPS for detector fine-tuning. "
            "Live analysis still keeps field calibration on CPU when detector inference uses MPS."
        )
    if "cuda" in available_devices:
        return "Use Auto or CUDA for detector fine-tuning on GPU hosts. ONNX Runtime CUDA is planned for future inference, not active yet."
    return "Use Auto unless you have a specific reason to pin CPU. ONNX Runtime CoreML/CUDA remains a planned future inference path."


@app.get("/api/config")
def config() -> ConfigResponse:
    diagnostics_config = resolve_provider_config()
    active_detector_entry = training_registry.get_active_entry() if training_available() else {"id": "soccana", "label": "soccana (football-pretrained)"}
    runtime_profile = build_runtime_profile()
    return ConfigResponse.model_validate({
        "detector_models": PLAYER_MODEL_OPTIONS,
        "player_models": PLAYER_MODEL_OPTIONS,
        "tracker": DEFAULT_PLAYER_TRACKER_MODE,
        "player_tracker_modes": PLAYER_TRACKER_MODE_OPTIONS,
        "default_player_tracker_mode": DEFAULT_PLAYER_TRACKER_MODE,
        "learn_cards": TACTICAL_LEARN_CARDS,
        "help_catalog": TACTICAL_HELP_CATALOG,
        "field_calibration_refresh_frames": CALIBRATION_REFRESH_FRAMES,
        "field_calibration_mode": "automatic_keypoints",
        "soccernet_dataset_dir": str(SOCCERNET_DIR),
        "soccernet_video_files": ["1_720p.mkv", "2_720p.mkv", "1_224p.mkv", "2_224p.mkv"],
        "soccernet_label_files": ["Labels-v2.json", "Labels.json"],
        "diagnostics_provider": diagnostics_config.provider if diagnostics_config else None,
        "diagnostics_model": diagnostics_config.model if diagnostics_config else None,
        "training_available": training_available(),
        "training_error": TRAINING_IMPORT_ERROR,
        "active_detector": active_detector_entry.get("id", "soccana"),
        "active_detector_label": active_detector_entry.get("label", "soccana (football-pretrained)"),
        "active_detector_is_custom": str(active_detector_entry.get("id", "soccana")) != "soccana",
        "runtime_profile": runtime_profile,
    })


@app.get("/api/train/config")
def training_config() -> dict[str, Any]:
    _training_manager, registry = require_training_available()
    backend_config = build_training_backend_config()
    runtime_profile = build_runtime_profile()
    return {
        "training_families": ["detector"],
        "enabled_families": ["detector"],
        **backend_config,
        "active_detector": registry.get_active_detector_id(),
        "device_options": build_training_device_options(runtime_profile),
        "device_guidance": build_training_device_guidance(runtime_profile),
        "runtime_profile": runtime_profile,
    }


@app.post("/api/train/datasets/scan")
def scan_training_dataset(request: TrainDatasetScanRequest) -> dict[str, Any]:
    require_training_available()
    return scan_training_dataset_path(Path(request.path))


@app.post("/api/train/jobs/detect")
def start_training_job(request: TrainingDetectJobRequest) -> dict[str, Any]:
    manager, _registry = require_training_available()
    dataset_path = Path(request.dataset_path).expanduser().resolve()
    inspection = inspect_training_dataset(dataset_path)
    scan = inspection.to_dict()
    if not inspection.can_start:
        raise HTTPException(status_code=400, detail=scan["errors"][0] if scan["errors"] else "Dataset is invalid for YOLO training")
    base_weights = request.base_weights.strip() or "soccana"
    if base_weights != "soccana":
        raise HTTPException(status_code=400, detail="Only soccana is currently available as a training base weight")

    run_name = request.run_name.strip()
    backend_config = build_training_backend_config()
    config_payload = {
        "base_weights": base_weights,
        "dataset_path": str(dataset_path),
        "run_name": run_name or "",
        "epochs": max(int(request.epochs), 1),
        "imgsz": max(int(request.imgsz), 32),
        "batch": max(int(request.batch), 1),
        "device": request.device.strip() or "auto",
        "workers": max(int(request.workers), 0),
        "patience": max(int(request.patience), 0),
        "freeze": request.freeze,
        "cache": bool(request.cache),
        "backend": backend_config["backend"],
        "backend_version": backend_config["backend_version"],
    }

    job = manager.create(config_payload)
    run_dir = Path(job.run_dir)
    config_payload["run_name"] = str(job.config.get("run_name") or job.run_id)
    try:
        runtime_inputs = prepare_training_run_inputs(dataset_path, run_dir)
    except Exception as exc:
        manager.update(
            job.job_id,
            status="failed",
            error=str(exc),
            finished_at=datetime.utcnow().isoformat() + "Z",
        )
        raise
    config_payload["dataset_scan"] = runtime_inputs["scan"]
    config_payload["generated_dataset_yaml"] = runtime_inputs["generated_dataset_yaml"]
    config_payload["generated_split_lists"] = runtime_inputs["generated_split_lists"]
    config_payload["validation_strategy"] = runtime_inputs["validation_strategy"]

    manager.append_log(job.job_id, f"Dataset scan tier: {scan['tier']}.")
    for warning in scan["warnings"]:
        manager.append_log(job.job_id, warning)
    manager.append_log(job.job_id, f"Validation strategy: {runtime_inputs['validation_strategy']}.")
    manager.append_log(job.job_id, f"Runtime dataset manifest: {runtime_inputs['generated_dataset_yaml']}")
    manager.update(
        job.job_id,
        config=config_payload,
        dataset_scan=runtime_inputs["scan"],
        generated_dataset_yaml=runtime_inputs["generated_dataset_yaml"],
        generated_split_lists=runtime_inputs["generated_split_lists"],
        validation_strategy=runtime_inputs["validation_strategy"],
        backend=backend_config["backend"],
        backend_version=backend_config["backend_version"],
        artifacts={"dataset_scan": runtime_inputs["dataset_scan_path"], "generated_dataset_yaml": runtime_inputs["generated_dataset_yaml"]},
    )

    threading.Thread(
        target=_launch_training_job_async,
        args=(job.job_id, run_dir, config_payload),
        daemon=True,
    ).start()
    return {"job_id": job.job_id, "run_id": job.run_id, "status": "queued"}


@app.get("/api/train/jobs")
def list_training_jobs() -> list[dict[str, Any]]:
    manager, _registry = require_training_available()
    return manager.list()


@app.get("/api/train/jobs/{job_id}")
def get_training_job(job_id: str) -> dict[str, Any]:
    manager, _registry = require_training_available()
    job = manager.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Training job not found")
    return job.as_dict()


@app.post("/api/train/jobs/{job_id}/stop")
def stop_training_job(job_id: str) -> dict[str, Any]:
    manager, _registry = require_training_available()
    job = manager.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Training job not found")
    manager.request_stop(job_id)
    return {"ok": True}


@app.get("/api/train/runs/recent")
def recent_training_runs(limit: int = 20) -> list[dict[str, Any]]:
    manager, _registry = require_training_available()
    bounded_limit = min(max(int(limit), 1), 200)
    return manager.list_recent_runs(limit=bounded_limit)


@app.get("/api/train/runs/{run_id}")
def get_training_run(run_id: str) -> dict[str, Any]:
    manager, _registry = require_training_available()
    job = manager.get_by_run_id(run_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Training run not found")
    return job.as_dict()


@app.post("/api/train/runs/{run_id}/activate")
def activate_training_run(run_id: str) -> dict[str, Any]:
    manager, registry = require_training_available()
    job = manager.get_by_run_id(run_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Training run not found")
    if job.status != "completed":
        raise HTTPException(status_code=409, detail="Only completed training runs can be activated")
    if not job.best_checkpoint:
        raise HTTPException(status_code=409, detail="Completed training run has no best checkpoint")
    checkpoint_path = Path(job.best_checkpoint).expanduser().resolve()
    if not checkpoint_path.exists() or not checkpoint_path.is_file():
        raise HTTPException(status_code=409, detail=f"Completed training run checkpoint is missing: {checkpoint_path}")

    try:
        entry = registry.activate_detector(
            run_id=run_id,
            checkpoint_path=str(checkpoint_path),
            run_name=str(job.config.get("run_name") or run_id),
            base_weights=str(job.config.get("base_weights") or "soccana"),
            metrics=job.metrics,
            created_at=job.created_at,
            resolved_device=job.resolved_device,
            backend=job.backend,
            backend_version=job.backend_version,
            summary_path=job.summary_path,
            artifacts=job.artifacts,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    manager.append_log(job.job_id, f"Activated detector {entry['id']}.")
    return {"success": True, "active_detector": entry["id"]}


@app.get("/api/train/registry")
def get_training_registry() -> dict[str, Any]:
    return build_training_registry_snapshot()


@app.post("/api/train/registry/activate")
def activate_registered_detector(request: TrainingRegistryActivateRequest) -> dict[str, Any]:
    _manager, registry = require_training_available()
    normalized_detector_id = request.detector_id.strip()
    registry_snapshot = build_training_registry_snapshot()
    known_detector_ids = {str(item.get("id") or "") for item in registry_snapshot.get("detectors") or []}
    if normalized_detector_id not in known_detector_ids:
        raise HTTPException(status_code=404, detail=f"Detector {normalized_detector_id or request.detector_id} is not registered")
    try:
        entry = registry.activate_detector_id(normalized_detector_id)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"success": True, "active_detector": entry["id"]}


@app.get("/api/jobs")
def list_jobs() -> list[JobStateResponse]:
    return [JobStateResponse.model_validate(item) for item in job_manager.list()]


@app.get("/api/experiments/active")
def active_experiment() -> ActiveExperimentResponse | None:
    experiment = load_active_batch_experiment()
    return ActiveExperimentResponse.model_validate(experiment) if experiment else None


@app.get("/api/soccernet/config")
def soccernet_config() -> SoccerNetConfigResponse:
    split_names = ["train", "valid", "test", "challenge"]
    split_counts = {split: len(getListGames(split, task="spotting", dataset="SoccerNet")) for split in split_names}
    return SoccerNetConfigResponse.model_validate({
        "dataset_dir": str(SOCCERNET_DIR),
        "splits": split_names,
        "split_counts": split_counts,
        "video_files": ["1_720p.mkv", "2_720p.mkv", "1_224p.mkv", "2_224p.mkv"],
        "label_files": ["Labels-v2.json", "Labels.json"],
        "notes": [
            "SoccerNet original videos are downloaded into backend/datasets/soccernet.",
            "720p halves are best for actual model runs. 224p halves are smaller for quick inspection.",
            "For experimental goal-timestamp work, prefer Labels-v2.json.",
        ],
    })


@app.get("/api/soccernet/games")
def soccernet_games(split: str = Query(default="train"), query: str = Query(default=""), limit: int = Query(default=200)) -> SoccerNetGamesResponse:
    normalized_split = split.strip().lower()
    if normalized_split not in {"train", "valid", "test", "challenge"}:
        raise HTTPException(status_code=400, detail="Split must be train, valid, test, or challenge")

    games = getListGames(normalized_split, task="spotting", dataset="SoccerNet")
    query_text = query.strip().lower()
    if query_text:
        games = [game for game in games if query_text in game.lower()]

    bounded_limit = min(max(limit, 1), 500)
    return SoccerNetGamesResponse.model_validate({
        "split": normalized_split,
        "count": len(games),
        "games": games[:bounded_limit],
    })


@app.get("/api/soccernet/downloads")
def list_soccernet_downloads() -> list[dict[str, Any]]:
    return download_job_manager.list()


@app.get("/api/soccernet/downloads/{job_id}")
def get_soccernet_download(job_id: str) -> dict[str, Any]:
    job = download_job_manager.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Download job not found")
    return job.as_dict()


@app.post("/api/soccernet/download")
def start_soccernet_download(request: SoccerNetDownloadRequest) -> JSONResponse:
    split = request.split.strip().lower()
    if split not in {"train", "valid", "test", "challenge"}:
        raise HTTPException(status_code=400, detail="Split must be train, valid, test, or challenge")
    if not request.password.strip():
        raise HTTPException(status_code=400, detail="SoccerNet password is required")
    if not request.game.strip():
        raise HTTPException(status_code=400, detail="Select a SoccerNet game")
    if not request.files:
        raise HTTPException(status_code=400, detail="Select at least one file to download")

    output_dir = SOCCERNET_DIR / request.game
    job = download_job_manager.create(split=split, game=request.game, files=request.files, output_dir=output_dir)
    download_job_manager.log(job.job_id, f"Queued SoccerNet download for {request.game}")

    thread = threading.Thread(
        target=_run_soccernet_download_job,
        args=(job.job_id, split, request.game, list(request.files), request.password),
        daemon=True,
    )
    thread.start()
    return JSONResponse(job.as_dict())


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> JobStateResponse:
    job = job_manager.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobStateResponse.model_validate(job.as_dict())


def require_job_control(job_id: str) -> tuple[JobState, JobControl]:
    job = job_manager.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    control = job_control_manager.get(job_id)
    if control is None:
        raise HTTPException(status_code=409, detail="Job is not controllable anymore")
    return job, control


@app.post("/api/jobs/{job_id}/pause")
def pause_job(job_id: str) -> JobStateResponse:
    job, control = require_job_control(job_id)
    if job.status not in {"queued", "running", "paused"}:
        raise HTTPException(status_code=409, detail="Only queued or running jobs can be paused")
    if job.status != "paused":
        control.request_pause()
        job_manager.update(job_id, status="paused")
        job_manager.log(job_id, "Pause requested")
    return JobStateResponse.model_validate(job_manager.get(job_id).as_dict())


@app.post("/api/jobs/{job_id}/resume")
def resume_job(job_id: str) -> JobStateResponse:
    job, control = require_job_control(job_id)
    if job.status != "paused":
        raise HTTPException(status_code=409, detail="Only paused jobs can be resumed")
    control.request_resume()
    job_manager.update(job_id, status="running")
    job_manager.log(job_id, "Resume requested")
    return JobStateResponse.model_validate(job_manager.get(job_id).as_dict())


@app.post("/api/jobs/{job_id}/stop")
def stop_job(job_id: str) -> JobStateResponse:
    job, control = require_job_control(job_id)
    if job.status not in {"queued", "running", "paused", "stopping"}:
        raise HTTPException(status_code=409, detail="Only active jobs can be stopped")
    control.request_stop()
    if job.status != "stopping":
        job_manager.update(job_id, status="stopping")
        job_manager.log(job_id, "Stop requested")
    return JobStateResponse.model_validate(job_manager.get(job_id).as_dict())


def load_persisted_run(run_dir: Path) -> dict[str, Any]:
    summary_path = run_dir / "outputs" / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary = normalize_persisted_summary(summary)
    created_at = str(summary.get("created_at") or datetime.utcfromtimestamp(summary_path.stat().st_mtime).isoformat() + "Z")
    overlay_path = run_dir / "outputs" / "overlay.mp4"
    logs = [f"Loaded completed run {run_dir.name} from disk."]
    if overlay_path.exists():
        logs.append(f"Overlay video available: {overlay_path.name}")

    return {
        "job_state_version": JOB_STATE_SCHEMA_VERSION,
        "job_id": f"persisted-{run_dir.name}",
        "run_id": run_dir.name,
        "status": "completed",
        "created_at": created_at,
        "progress": 100.0,
        "logs": logs,
        "run_dir": str(run_dir),
        "summary": summary,
        "error": None,
        "persisted": True,
    }


def make_persisted_run_id(run_dir: Path) -> str:
    experiment_runs_root = EXPERIMENTS_DIR.resolve()
    resolved = run_dir.resolve()
    try:
        relative = resolved.relative_to(experiment_runs_root)
    except ValueError:
        return run_dir.name

    parts = list(relative.parts)
    if len(parts) >= 3 and parts[1] == "runs":
        experiment_name = parts[0]
        run_name = parts[2]
        return f"exp--{experiment_name}--{run_name}"
    return run_dir.name


def parse_persisted_run_id(run_id: str) -> tuple[str | None, str]:
    if run_id.startswith("exp--"):
        _, experiment_name, run_name = run_id.split("--", 2)
        return experiment_name, run_name
    return None, run_id


def hydrate_persisted_run(run_dir: Path) -> dict[str, Any]:
    payload = load_persisted_run(run_dir)
    run_id = make_persisted_run_id(run_dir)
    payload["run_id"] = run_id
    summary = dict(payload.get("summary") or {})

    def rewrite_output_url(current_value: Any) -> Any:
        if not current_value:
            return current_value
        name = Path(str(current_value)).name
        if not name:
            return current_value
        return f"/api/runs/{run_id}/outputs/{name}"

    for key in (
        "overlay_video",
        "detections_csv",
        "track_summary_csv",
        "projection_csv",
        "entropy_timeseries_csv",
        "goal_events_csv",
        "summary_json",
        "all_outputs_zip",
        "diagnostics_json",
    ):
        if key in summary:
            summary[key] = rewrite_output_url(summary.get(key))
    payload["summary"] = summary

    try:
        relative = run_dir.resolve().relative_to(EXPERIMENTS_DIR.resolve())
    except ValueError:
        payload["experiment_batch"] = None
    else:
        parts = list(relative.parts)
        payload["experiment_batch"] = parts[0] if parts else None
    return payload


def iter_persisted_run_dirs() -> list[Path]:
    run_dirs: list[Path] = []
    if RUNS_DIR.exists():
        run_dirs.extend([path for path in RUNS_DIR.iterdir() if path.is_dir()])
    if EXPERIMENTS_DIR.exists():
        for experiment_dir in EXPERIMENTS_DIR.iterdir():
            if not experiment_dir.is_dir():
                continue
            batch_runs_dir = experiment_dir / "runs"
            if not batch_runs_dir.exists() or not batch_runs_dir.is_dir():
                continue
            run_dirs.extend([path for path in batch_runs_dir.iterdir() if path.is_dir()])
    return run_dirs


def resolve_persisted_run_dir(run_id: str) -> Path:
    experiment_name, run_name = parse_persisted_run_id(run_id)
    if experiment_name is None:
        run_dir = (RUNS_DIR / run_name).resolve()
        try:
            run_dir.relative_to(RUNS_DIR.resolve())
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="Invalid run id") from exc
        return run_dir

    experiment_dir = (EXPERIMENTS_DIR / experiment_name / "runs" / run_name).resolve()
    try:
        experiment_dir.relative_to(EXPERIMENTS_DIR.resolve())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid run id") from exc
    return experiment_dir


def normalize_persisted_summary(summary: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(summary)
    normalized["summary_version"] = normalized.get("summary_version", SUMMARY_SCHEMA_VERSION)
    normalized["learn_cards"] = TACTICAL_LEARN_CARDS
    normalized["experiments"] = list(normalized.get("experiments") or [])
    normalized["entropy_timeseries_csv"] = normalized.get("entropy_timeseries_csv")
    normalized["diagnostics_source"] = normalized.get("diagnostics_source", "heuristic")
    normalized["diagnostics_provider"] = normalized.get("diagnostics_provider")
    normalized["diagnostics_model"] = normalized.get("diagnostics_model")
    normalized["diagnostics_status"] = normalized.get("diagnostics_status", "unknown")
    normalized["diagnostics_summary_line"] = normalized.get("diagnostics_summary_line", "")
    normalized["diagnostics_error"] = normalized.get("diagnostics_error", "")
    normalized["diagnostics_json"] = normalized.get("diagnostics_json")
    normalized["diagnostics_prompt_context"] = normalized.get("diagnostics_prompt_context")
    diagnostics_prompt_version = normalized.get("diagnostics_prompt_version")
    if not diagnostics_prompt_version:
        run_dir_value = normalized.get("run_dir")
        if run_dir_value:
            artifact_path = Path(str(run_dir_value)) / "outputs" / "diagnostics_ai.json"
            if artifact_path.exists():
                try:
                    artifact_payload = json.loads(artifact_path.read_text(encoding="utf-8"))
                    diagnostics_prompt_version = artifact_payload.get("prompt_version")
                except Exception:
                    diagnostics_prompt_version = None
    normalized["diagnostics_prompt_version"] = diagnostics_prompt_version
    normalized["diagnostics_current_prompt_version"] = CURRENT_DIAGNOSTICS_PROMPT_VERSION
    normalized["diagnostics_stale"] = bool(
        normalized.get("diagnostics_source") == "ai"
        and diagnostics_prompt_version != CURRENT_DIAGNOSTICS_PROMPT_VERSION
    )
    if normalized["diagnostics_stale"]:
        normalized["diagnostics_stale_reason"] = (
            "Stored AI diagnostics are from an older analysis build and may no longer match the current runtime behavior."
        )
    else:
        normalized["diagnostics_stale_reason"] = ""
    has_identity_metrics = any(
        key in normalized
        for key in (
            "player_tracker_mode",
            "raw_unique_player_track_ids",
            "tracklet_merges_applied",
            "stitched_track_id_reduction",
            "identity_embedding_updates",
        )
    )
    if has_identity_metrics:
        normalized["player_tracker_mode"] = normalized.get("player_tracker_mode", DEFAULT_PLAYER_TRACKER_MODE)
        normalized["resolved_player_tracker_mode"] = normalized.get("resolved_player_tracker_mode", normalized["player_tracker_mode"])
        normalized["requested_player_tracker_mode"] = normalized.get("requested_player_tracker_mode")
        normalized["player_tracker_runtime"] = normalized.get("player_tracker_runtime")
        normalized["raw_unique_player_track_ids"] = normalized.get("raw_unique_player_track_ids", normalized.get("unique_player_track_ids", 0))
        normalized["tracklet_merges_applied"] = normalized.get("tracklet_merges_applied", 0)
    else:
        normalized["player_tracker_mode"] = "historical/unknown"
        normalized["resolved_player_tracker_mode"] = "historical/unknown"
        normalized["requested_player_tracker_mode"] = None
        normalized["player_tracker_runtime"] = None
        normalized["raw_unique_player_track_ids"] = None
        normalized["tracklet_merges_applied"] = None
        normalized["stitched_track_id_reduction"] = None
        normalized["identity_embedding_updates"] = None
        normalized["identity_embedding_interval_frames"] = None
    normalized["player_tracker_backend"] = normalized.get("player_tracker_backend")
    normalized["player_tracker_stitching_enabled"] = normalized.get("player_tracker_stitching_enabled")
    normalized["field_calibration_success_rate"] = normalized.get("field_calibration_success_rate", 0.0)
    normalized["field_calibration_refresh_rejections"] = normalized.get("field_calibration_refresh_rejections", 0)
    normalized["field_keypoint_confidence_threshold"] = normalized.get("field_keypoint_confidence_threshold", 0.25)
    normalized["field_calibration_min_visible_keypoints"] = normalized.get("field_calibration_min_visible_keypoints", 4)
    normalized["field_calibration_stale_recovery_min_visible_keypoints"] = normalized.get("field_calibration_stale_recovery_min_visible_keypoints", 5)
    normalized["field_calibration_rejections_no_candidate"] = normalized.get("field_calibration_rejections_no_candidate", 0)
    low_visible_rejections = normalized.get(
        "field_calibration_rejections_low_visible_count",
        normalized.get("field_calibration_rejections_low_visible_keypoints", 0),
    )
    normalized["field_calibration_rejections_low_visible_count"] = low_visible_rejections
    normalized["field_calibration_rejections_low_visible_keypoints"] = normalized.get(
        "field_calibration_rejections_low_visible_keypoints",
        low_visible_rejections,
    )
    normalized["field_calibration_rejections_low_inliers"] = normalized.get("field_calibration_rejections_low_inliers", 0)
    normalized["field_calibration_rejections_high_reprojection_error"] = normalized.get("field_calibration_rejections_high_reprojection_error", 0)
    normalized["field_calibration_rejections_high_temporal_drift"] = normalized.get("field_calibration_rejections_high_temporal_drift", 0)
    normalized["field_calibration_rejections_invalid_candidate"] = normalized.get("field_calibration_rejections_invalid_candidate", 0)
    normalized["field_calibration_primary_rejections_no_candidate"] = normalized.get("field_calibration_primary_rejections_no_candidate", 0)
    low_visible_primary_rejections = normalized.get(
        "field_calibration_primary_rejections_low_visible_count",
        normalized.get("field_calibration_primary_rejections_low_visible_keypoints", 0),
    )
    normalized["field_calibration_primary_rejections_low_visible_count"] = low_visible_primary_rejections
    normalized["field_calibration_primary_rejections_low_visible_keypoints"] = normalized.get(
        "field_calibration_primary_rejections_low_visible_keypoints",
        low_visible_primary_rejections,
    )
    normalized["field_calibration_primary_rejections_low_inliers"] = normalized.get("field_calibration_primary_rejections_low_inliers", 0)
    normalized["field_calibration_primary_rejections_high_reprojection_error"] = normalized.get("field_calibration_primary_rejections_high_reprojection_error", 0)
    normalized["field_calibration_primary_rejections_high_temporal_drift"] = normalized.get("field_calibration_primary_rejections_high_temporal_drift", 0)
    normalized["field_calibration_primary_rejections_invalid_candidate"] = normalized.get("field_calibration_primary_rejections_invalid_candidate", 0)
    normalized["detector_class_names_source"] = normalized.get("detector_class_names_source", "unknown")
    normalized["player_detector_class_ids"] = normalized.get("player_detector_class_ids", [])
    normalized["ball_detector_class_ids"] = normalized.get("ball_detector_class_ids", [])
    normalized["referee_detector_class_ids"] = normalized.get("referee_detector_class_ids", [])
    normalized["calibration_debug_csv"] = normalized.get("calibration_debug_csv")
    normalized["frames_with_field_homography"] = normalized.get("frames_with_field_homography", 0)
    normalized["frames_with_usable_homography"] = normalized.get("frames_with_usable_homography", 0)
    normalized["frames_with_nonstale_homography"] = normalized.get("frames_with_nonstale_homography", 0)
    normalized["frames_with_stale_homography"] = normalized.get("frames_with_stale_homography", 0)
    normalized["frames_projection_blocked_by_stale"] = normalized.get("frames_projection_blocked_by_stale", 0)
    normalized["frames_projected_with_last_known_homography"] = normalized.get("frames_projected_with_last_known_homography", 0)
    normalized["frames_with_player_anchors"] = normalized.get("frames_with_player_anchors", 0)
    normalized["frames_with_projected_points"] = normalized.get("frames_with_projected_points", 0)
    normalized["frames_with_homography_but_no_player_anchors"] = normalized.get("frames_with_homography_but_no_player_anchors", 0)
    normalized["projected_player_points_fresh"] = normalized.get("projected_player_points_fresh", 0)
    normalized["projected_player_points_stale"] = normalized.get("projected_player_points_stale", 0)
    normalized["projected_ball_points_fresh"] = normalized.get("projected_ball_points_fresh", 0)
    normalized["projected_ball_points_stale"] = normalized.get("projected_ball_points_stale", 0)
    normalized["player_rows_while_calibration_fresh"] = normalized.get("player_rows_while_calibration_fresh", 0)
    normalized["player_rows_while_calibration_stale"] = normalized.get("player_rows_while_calibration_stale", 0)
    normalized["field_calibration_stale_recovery_attempts"] = normalized.get("field_calibration_stale_recovery_attempts", 0)
    normalized["field_calibration_stale_recovery_successes"] = normalized.get("field_calibration_stale_recovery_successes", 0)
    normalized["field_calibration_stale_recovery_rejections"] = normalized.get(
        "field_calibration_stale_recovery_rejections",
        max(
            int(normalized["field_calibration_stale_recovery_attempts"]) - int(normalized["field_calibration_stale_recovery_successes"]),
            0,
        ),
    )
    normalized["detector_debug_sample_frames"] = normalized.get("detector_debug_sample_frames", 0)
    normalized["raw_detector_boxes_sampled"] = normalized.get("raw_detector_boxes_sampled", 0)
    normalized["raw_detector_class_histogram_sample"] = normalized.get("raw_detector_class_histogram_sample", {})

    is_historical_summary = "field_calibration_refresh_frames" not in normalized
    if is_historical_summary:
        normalized["diagnostics"] = [
            {
                "level": "warn",
                "title": "Historical run format",
                "message": "This run predates the automatic field-calibration pipeline and is being shown for reference only.",
                "next_step": "Rerun the clip to see current calibration metrics, live-preview-aligned diagnostics, and the Soccana model stack.",
            }
        ]
        normalized["homography_enabled"] = False
        normalized["field_registered_frames"] = 0
        normalized["field_registered_ratio"] = 0.0
        normalized["field_calibration_refresh_frames"] = CALIBRATION_REFRESH_FRAMES
        normalized["field_calibration_refresh_attempts"] = 0
        normalized["field_calibration_refresh_successes"] = 0
        normalized["field_calibration_success_rate"] = 0.0
        normalized["field_calibration_refresh_rejections"] = 0
        normalized["field_keypoint_confidence_threshold"] = 0.25
        normalized["field_calibration_min_visible_keypoints"] = 4
        normalized["field_calibration_stale_recovery_min_visible_keypoints"] = 5
        normalized["field_calibration_rejections_no_candidate"] = 0
        normalized["field_calibration_rejections_low_visible_count"] = 0
        normalized["field_calibration_rejections_low_visible_keypoints"] = 0
        normalized["field_calibration_rejections_low_inliers"] = 0
        normalized["field_calibration_rejections_high_reprojection_error"] = 0
        normalized["field_calibration_rejections_high_temporal_drift"] = 0
        normalized["field_calibration_rejections_invalid_candidate"] = 0
        normalized["field_calibration_primary_rejections_no_candidate"] = 0
        normalized["field_calibration_primary_rejections_low_visible_count"] = 0
        normalized["field_calibration_primary_rejections_low_visible_keypoints"] = 0
        normalized["field_calibration_primary_rejections_low_inliers"] = 0
        normalized["field_calibration_primary_rejections_high_reprojection_error"] = 0
        normalized["field_calibration_primary_rejections_high_temporal_drift"] = 0
        normalized["field_calibration_primary_rejections_invalid_candidate"] = 0
        normalized["detector_class_names_source"] = "unknown"
        normalized["player_detector_class_ids"] = []
        normalized["ball_detector_class_ids"] = []
        normalized["referee_detector_class_ids"] = []
        normalized["calibration_debug_csv"] = None
        normalized["frames_with_field_homography"] = 0
        normalized["frames_with_usable_homography"] = 0
        normalized["frames_with_nonstale_homography"] = 0
        normalized["frames_with_stale_homography"] = 0
        normalized["frames_projection_blocked_by_stale"] = 0
        normalized["frames_projected_with_last_known_homography"] = 0
        normalized["frames_with_player_anchors"] = 0
        normalized["frames_with_projected_points"] = 0
        normalized["frames_with_homography_but_no_player_anchors"] = 0
        normalized["projected_player_points_fresh"] = 0
        normalized["projected_player_points_stale"] = 0
        normalized["projected_ball_points_fresh"] = 0
        normalized["projected_ball_points_stale"] = 0
        normalized["player_rows_while_calibration_fresh"] = 0
        normalized["player_rows_while_calibration_stale"] = 0
        normalized["field_calibration_stale_recovery_attempts"] = 0
        normalized["field_calibration_stale_recovery_successes"] = 0
        normalized["field_calibration_stale_recovery_rejections"] = 0
        normalized["detector_debug_sample_frames"] = 0
        normalized["raw_detector_boxes_sampled"] = 0
        normalized["raw_detector_class_histogram_sample"] = {}
        normalized["average_visible_pitch_keypoints"] = 0.0
        normalized["last_good_calibration_frame"] = -1
        normalized["entropy_timeseries_csv"] = None
        normalized["experiments"] = []
        normalized["player_tracker_mode"] = "historical"
        normalized["resolved_player_tracker_mode"] = "historical"
        normalized["requested_player_tracker_mode"] = None
        normalized["player_tracker_runtime"] = None
        normalized["raw_unique_player_track_ids"] = normalized.get("unique_player_track_ids", 0)
        normalized["tracklet_merges_applied"] = 0
        normalized["historical_summary"] = True
    else:
        normalized["historical_summary"] = False

    if normalized.get("diagnostics_stale") and not normalized.get("historical_summary"):
        stale_title = "Stored AI diagnostics are outdated"
        existing = list(normalized.get("diagnostics") or [])
        if not any(str(item.get("title")) == stale_title for item in existing if isinstance(item, dict)):
            existing.insert(
                0,
                {
                    "level": "warn",
                    "title": stale_title,
                    "message": normalized["diagnostics_stale_reason"],
                    "next_step": "Click Regenerate in Run Review to rebuild this run's AI diagnostics for the current analysis build.",
                },
            )
            normalized["diagnostics"] = existing

    return normalized


def refresh_run_diagnostics(run_dir: Path) -> dict[str, Any]:
    outputs_dir = run_dir / "outputs"
    summary_path = outputs_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    heuristic_diagnostics = list(summary.get("heuristic_diagnostics") or summary.get("diagnostics") or [])
    diagnostics, artifact = generate_run_diagnostics(
        summary=summary,
        heuristic_diagnostics=heuristic_diagnostics,
        outputs_dir=outputs_dir,
        job_id=f"refresh-{run_dir.name}",
        job_manager=None,
    )
    summary["diagnostics"] = diagnostics
    summary["heuristic_diagnostics"] = heuristic_diagnostics
    summary["diagnostics_source"] = "ai" if artifact.get("status") == "completed" else "heuristic"
    summary["diagnostics_provider"] = artifact.get("provider")
    summary["diagnostics_model"] = artifact.get("model")
    summary["diagnostics_status"] = artifact.get("status")
    summary["diagnostics_summary_line"] = artifact.get("summary_line", "")
    summary["diagnostics_error"] = artifact.get("error", "")
    summary["diagnostics_json"] = f"/runs/{run_dir.name}/outputs/diagnostics_ai.json"
    summary["diagnostics_prompt_context"] = artifact.get("prompt_context")
    summary["diagnostics_prompt_version"] = artifact.get("prompt_version")
    summary["diagnostics_current_prompt_version"] = CURRENT_DIAGNOSTICS_PROMPT_VERSION
    summary["diagnostics_stale"] = False
    summary["diagnostics_stale_reason"] = ""
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def inspect_video(path: Path) -> dict[str, Any]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = safe_int(cap.get(cv2.CAP_PROP_FRAME_WIDTH), 0)
    height = safe_int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT), 0)
    frame_count = safe_int(cap.get(cv2.CAP_PROP_FRAME_COUNT), 0)
    cap.release()
    duration_seconds = float(frame_count / fps) if fps and fps > 0 and frame_count > 0 else 0.0
    return {
        "fps": round(float(fps), 4) if fps and fps > 0 else 0.0,
        "width": width,
        "height": height,
        "frame_count": frame_count,
        "duration_seconds": round(duration_seconds, 4),
        "size_mb": round(path.stat().st_size / 1024 / 1024, 2),
    }


def resolve_source_path(source_id: str = "", local_video_path: str = "") -> Path:
    if source_id.strip():
        source = source_manager.get(source_id.strip())
        if source is None:
            raise HTTPException(status_code=404, detail="Source not found")
        path = Path(source.path).expanduser().resolve()
        if not path.exists() or not path.is_file():
            raise HTTPException(status_code=404, detail="Source file no longer exists")
        return path

    if local_video_path.strip():
        path = Path(local_video_path).expanduser().resolve()
        if not path.exists() or not path.is_file():
            raise HTTPException(status_code=400, detail="Local video path does not exist")
        return path

    raise HTTPException(status_code=400, detail="Provide either a source id or a local video path")


def _run_soccernet_download_job(job_id: str, split: str, game: str, files: list[str], password: str) -> None:
    original_https_context_factory = ssl._create_default_https_context
    try:
        download_job_manager.update(job_id, status="running", progress=1.0)
        downloader = SoccerNetDownloader(str(SOCCERNET_DIR))
        downloader.password = password
        ssl._create_default_https_context = ssl._create_unverified_context
        download_job_manager.log(job_id, "SoccerNet OwnCloud certificate verification disabled for this download session")

        total_files = max(len(files), 1)
        for index, file_name in enumerate(files, start=1):
            download_job_manager.log(job_id, f"Downloading {file_name}")
            downloader.downloadGame(game=game, files=[file_name], spl=split, verbose=False)
            local_file = SOCCERNET_DIR / game / file_name
            if local_file.exists():
                size_mb = round(local_file.stat().st_size / 1024 / 1024, 2)
                download_job_manager.log(job_id, f"Saved {file_name} ({size_mb} MB)")
            else:
                download_job_manager.log(job_id, f"Requested {file_name}, but it was not found locally after download")
            progress = min(99.0, (index / total_files) * 100.0)
            download_job_manager.update(job_id, progress=progress)

        download_job_manager.update(job_id, status="completed", progress=100.0)
        download_job_manager.log(job_id, f"SoccerNet download complete in {SOCCERNET_DIR / game}")
    except Exception as exc:
        download_job_manager.update(job_id, status="failed", error=str(exc))
        download_job_manager.log(job_id, f"SoccerNet download failed: {exc}")
    finally:
        ssl._create_default_https_context = original_https_context_factory


def list_persisted_runs(limit: int = 8) -> list[dict[str, Any]]:
    persisted_runs: list[dict[str, Any]] = []
    for run_dir in iter_persisted_run_dirs():
        try:
            persisted_runs.append(hydrate_persisted_run(run_dir))
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            continue
    persisted_runs.sort(key=lambda item: str(item.get("created_at", "")), reverse=True)
    return persisted_runs[: max(limit, 1)]


def _read_tail_lines(path: Path, limit: int = 250) -> list[str]:
    if not path.exists():
        return []
    try:
        return path.read_text(encoding="utf-8", errors="replace").splitlines()[-limit:]
    except Exception:
        return []


def load_active_batch_experiment() -> dict[str, Any] | None:
    candidates: list[tuple[float, Path, dict[str, Any], list[str]]] = []
    freshness_cutoff = time.time() - ACTIVE_EXPERIMENT_FRESHNESS_SECONDS
    for experiment_dir in EXPERIMENTS_DIR.iterdir():
        if not experiment_dir.is_dir():
            continue
        manifest_path = experiment_dir / "manifest.json"
        batch_log_path = experiment_dir / "batch.log"
        tmux_log_path = experiment_dir / "tmux_stdout.log"
        if not manifest_path.exists() or not batch_log_path.exists():
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        lines = _read_tail_lines(batch_log_path, limit=400)
        if any("completed batch" in line for line in lines[-25:]):
            continue
        latest_mtime = max(
            batch_log_path.stat().st_mtime,
            tmux_log_path.stat().st_mtime if tmux_log_path.exists() else 0.0,
        )
        if latest_mtime < freshness_cutoff:
            continue
        candidates.append((latest_mtime, experiment_dir, manifest, lines))

    if not candidates:
        return None

    _mtime, experiment_dir, manifest, lines = max(candidates, key=lambda item: item[0])
    games = list(manifest.get("games") or [])
    files = list(manifest.get("files") or [])
    half_files = [file_name for file_name in files if str(file_name).startswith(("1_", "2_"))]
    total_halves = max(len(games) * max(len(half_files), 1), 1)

    runs_csv_path = experiment_dir / "runs.csv"
    halves_processed = 0
    if runs_csv_path.exists():
        try:
            halves_processed = max(sum(1 for _ in runs_csv_path.open("r", encoding="utf-8")) - 1, 0)
        except Exception:
            halves_processed = 0

    current_game = ""
    current_game_index = 0
    current_half_tag = ""
    current_half_progress = 0.0
    progress_pattern = re.compile(r"\[(game-(\d{3})-([12]))\] progress=(\d+(?:\.\d+)?)")
    game_pattern = re.compile(r"\[(game-(\d{3}))\] processing (.+)")

    for line in lines:
        game_match = game_pattern.search(line)
        if game_match:
            current_game_index = int(game_match.group(2))
            current_game = game_match.group(3).strip()
        progress_match = progress_pattern.search(line)
        if progress_match:
            current_game_index = int(progress_match.group(2))
            current_half_tag = progress_match.group(3)
            current_half_progress = float(progress_match.group(4))

    if not current_game and current_game_index > 0 and current_game_index <= len(games):
        current_game = str(games[current_game_index - 1])

    current_half_file = ""
    if current_half_tag:
        current_half_file = next((file_name for file_name in half_files if str(file_name).startswith(f"{current_half_tag}_")), "")
    current_source_video_path = ""
    current_label_path = ""
    if current_game and current_half_file:
        current_source_video_path = str((SOCCERNET_DIR / current_game / current_half_file).resolve())
    label_file = next((file_name for file_name in files if str(file_name).lower().endswith(".json")), "")
    if current_game and label_file:
        current_label_path = str((SOCCERNET_DIR / current_game / label_file).resolve())

    progress = (halves_processed / total_halves) * 100.0
    if current_half_progress > 0.0 and halves_processed < total_halves:
        progress = ((halves_processed + (current_half_progress / 100.0)) / total_halves) * 100.0

    logs = lines[-250:] if lines else [f"Batch {experiment_dir.name} is active."]
    return {
        "job_state_version": JOB_STATE_SCHEMA_VERSION,
        "job_id": f"batch-{experiment_dir.name}",
        "status": "running",
        "created_at": str(manifest.get("created_at") or datetime.utcfromtimestamp((experiment_dir / "manifest.json").stat().st_mtime).isoformat() + "Z"),
        "progress": round(progress, 2),
        "logs": logs,
        "run_dir": str(experiment_dir),
        "summary": {
            "batch_name": experiment_dir.name,
            "games_requested": len(games),
            "halves_total": total_halves,
            "halves_processed": halves_processed,
            "current_game": current_game,
            "current_game_index": current_game_index,
            "current_half_tag": current_half_tag,
            "current_half_file": current_half_file,
            "current_source_video_path": current_source_video_path,
            "current_label_path": current_label_path,
            "files": files,
        },
        "error": None,
    }


def resolve_analysis_detector_model(detector_model: str, player_model: str = "") -> str:
    requested_detector = detector_model.strip()
    if requested_detector:
        if requested_detector == "soccana":
            return resolve_model_path("soccana", "detector")
        return requested_detector

    requested_player = player_model.strip()
    if requested_player:
        if requested_player == "soccana":
            return resolve_model_path("soccana", "detector")
        return requested_player
    if training_available():
        active_entry = training_registry.get_active_entry()
        active_detector_id = str(active_entry.get("id") or "soccana")
        try:
            active_path = Path(training_registry.get_active_path()).expanduser().resolve()
        except Exception as exc:
            raise HTTPException(
                status_code=409,
                detail=f"Active detector {active_detector_id} could not be resolved for analysis: {exc}",
            ) from exc
        if active_detector_id != "soccana" and (not active_path.exists() or not active_path.is_file()):
            raise HTTPException(
                status_code=409,
                detail=f"Active detector {active_detector_id} checkpoint is missing: {active_path}",
            )
        return str(active_path)
    return requested_detector or requested_player or "soccana"


def build_training_registry_snapshot() -> dict[str, Any]:
    manager, registry = require_training_available()
    for job in manager.list_states():
        if job.status != "completed" or not job.best_checkpoint:
            continue
        checkpoint_path = Path(job.best_checkpoint).expanduser()
        if not checkpoint_path.exists():
            continue
        registry.register_detector(
            run_id=job.run_id,
            checkpoint_path=str(checkpoint_path),
            run_name=str(job.config.get("run_name") or job.run_id),
            base_weights=str(job.config.get("base_weights") or "soccana"),
            metrics=job.metrics,
            created_at=job.created_at,
            resolved_device=job.resolved_device,
            backend=job.backend,
            backend_version=job.backend_version,
            summary_path=job.summary_path,
            artifacts=job.artifacts,
            activate=registry.get_active_detector_id() == f"custom_{job.run_id}",
        )
    return registry.snapshot()


@app.get("/api/runs/recent")
def recent_runs(limit: int = 8) -> list[JobStateResponse]:
    bounded_limit = min(max(limit, 1), 1000)
    return [JobStateResponse.model_validate(item) for item in list_persisted_runs(limit=bounded_limit)]


@app.post("/api/source")
async def prepare_source(
    video_file: UploadFile | None = File(default=None),
    local_video_path: str = Form(default=""),
) -> SourceResponse:
    if video_file is None and not local_video_path.strip():
        raise HTTPException(status_code=400, detail="Provide either a file upload or a local video path")

    if video_file is not None:
        suffix = Path(video_file.filename or "uploaded_video.mp4").suffix or ".mp4"
        source_path = UPLOADS_DIR / f"source_{uuid.uuid4().hex[:12]}{suffix}"
        with source_path.open("wb") as out_file:
            shutil.copyfileobj(video_file.file, out_file)
        source = source_manager.register(source_path, video_file.filename or source_path.name, uploaded=True)
    else:
        source_path = resolve_source_path(local_video_path=local_video_path)
        source = source_manager.register(source_path, source_path.name, uploaded=False)

    metadata = inspect_video(Path(source.path))
    return SourceResponse.model_validate(source.as_dict(metadata))


@app.get("/api/source/{source_id}/video")
def get_source_video(source_id: str) -> FileResponse:
    source = source_manager.get(source_id)
    if source is None:
        raise HTTPException(status_code=404, detail="Source not found")

    source_path = Path(source.path).expanduser().resolve()
    if not source_path.exists() or not source_path.is_file():
        raise HTTPException(status_code=404, detail="Source file no longer exists")

    media_type = mimetypes.guess_type(str(source_path))[0] or "application/octet-stream"
    return FileResponse(path=source_path, media_type=media_type, filename=source.display_name)


@app.get("/api/runs/{run_id}")
def get_persisted_run(run_id: str) -> JobStateResponse:
    run_dir = resolve_persisted_run_dir(run_id)
    if not run_dir.exists() or not run_dir.is_dir():
        raise HTTPException(status_code=404, detail="Run not found")

    try:
        return JobStateResponse.model_validate(hydrate_persisted_run(run_dir))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Run summary not found") from exc
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=500, detail="Run summary is not valid JSON") from exc


@app.get("/api/runs/{run_id}/outputs/{filename}")
def get_persisted_run_output(run_id: str, filename: str) -> FileResponse:
    if Path(filename).name != filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    run_dir = resolve_persisted_run_dir(run_id)
    if not run_dir.exists() or not run_dir.is_dir():
        raise HTTPException(status_code=404, detail="Run not found")

    output_path = (run_dir / "outputs" / filename).resolve()
    try:
        output_path.relative_to((run_dir / "outputs").resolve())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid output path") from exc

    if not output_path.exists() or not output_path.is_file():
        raise HTTPException(status_code=404, detail="Output file not found")

    return FileResponse(path=output_path, filename=output_path.name)


@app.post("/api/runs/{run_id}/refresh-diagnostics")
def refresh_persisted_run_diagnostics(run_id: str) -> JobStateResponse:
    run_dir = resolve_persisted_run_dir(run_id)
    if not run_dir.exists() or not run_dir.is_dir():
        raise HTTPException(status_code=404, detail="Run not found")

    try:
        refresh_run_diagnostics(run_dir)
        return JobStateResponse.model_validate(hydrate_persisted_run(run_dir))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Run summary not found") from exc
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=500, detail="Run summary is not valid JSON") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not refresh diagnostics: {exc}") from exc


@app.get("/api/live-preview")
def live_preview(
    source_id: str = Query(default=""),
    local_video_path: str = Query(default=""),
    detector_model: str = Query(default="soccana"),
    player_model: str = Query(default=""),
    tracker_mode: str = Query(default=DEFAULT_PLAYER_TRACKER_MODE),
    include_ball: bool = Query(default=True),
    player_conf: float = Query(default=0.25),
    ball_conf: float = Query(default=0.20),
    iou: float = Query(default=0.50),
) -> StreamingResponse:
    source_video_path = resolve_source_path(source_id=source_id, local_video_path=local_video_path)
    resolved_detector_model = resolve_analysis_detector_model(detector_model, player_model)

    stream = generate_live_preview_stream(
        source_video_path=source_video_path,
        config_payload={
            "player_model": resolved_detector_model,
            "ball_model": resolved_detector_model,
            "tracker_mode": tracker_mode,
            "include_ball": include_ball,
            "player_conf": float(player_conf),
            "ball_conf": float(ball_conf),
            "iou": float(iou),
        },
    )
    return StreamingResponse(stream, media_type="multipart/x-mixed-replace; boundary=frame")


@app.post("/api/scan-folder")
def scan_folder(request: FolderScanRequest) -> FolderScanResponse:
    folder = Path(request.folder_path).expanduser().resolve()
    if not folder.exists() or not folder.is_dir():
        raise HTTPException(status_code=400, detail="Folder does not exist")

    video_suffixes = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}
    annotation_suffixes = {".json", ".csv", ".txt"}

    videos: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    for item in sorted(folder.rglob("*")):
        if item.is_dir():
            continue
        suffix = item.suffix.lower()
        if suffix in video_suffixes:
            videos.append({"name": item.name, "path": str(item), "size_mb": round(item.stat().st_size / 1024 / 1024, 2)})
        elif suffix in annotation_suffixes:
            annotations.append({"name": item.name, "path": str(item)})

    return FolderScanResponse.model_validate({
        "folder": str(folder),
        "videos": videos[:200],
        "annotations": annotations[:200],
        "notes": [
            "Useful for extracted Kaggle clips or a local SoccerNet checkout.",
            "Click a video path in the UI to load it into the analyze form.",
        ],
    })


@app.post("/api/analyze")
async def analyze(
    video_file: UploadFile | None = File(default=None),
    local_video_path: str = Form(default=""),
    source_id: str = Form(default=""),
    label_path: str = Form(default=""),
    detector_model: str = Form(default="soccana"),
    player_model: str = Form(default=""),
    tracker_mode: str = Form(default=DEFAULT_PLAYER_TRACKER_MODE),
    include_ball: bool = Form(default=True),
    player_conf: float = Form(default=0.25),
    ball_conf: float = Form(default=0.20),
    iou: float = Form(default=0.50),
) -> AnalyzeAcceptedResponse:
    if video_file is None and not local_video_path.strip() and not source_id.strip():
        raise HTTPException(status_code=400, detail="Provide a source id, file upload, or local video path")

    run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:6]
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    inputs_dir = run_dir / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)

    source_video_path: Path
    if video_file is not None:
        suffix = Path(video_file.filename or "uploaded_video.mp4").suffix or ".mp4"
        source_video_path = inputs_dir / f"uploaded{suffix}"
        with source_video_path.open("wb") as out_file:
            shutil.copyfileobj(video_file.file, out_file)
    elif source_id.strip():
        source_video_path = resolve_source_path(source_id=source_id)
    else:
        source_video_path = resolve_source_path(local_video_path=local_video_path)

    resolved_detector_model = resolve_analysis_detector_model(detector_model, player_model)

    config_payload = {
        "source_video_path": str(source_video_path),
        "label_path": label_path.strip(),
        "player_model": resolved_detector_model,
        "ball_model": resolved_detector_model,
        "tracker_mode": tracker_mode,
        "include_ball": include_ball,
        "player_conf": float(player_conf),
        "ball_conf": float(ball_conf),
        "iou": float(iou),
    }

    job = job_manager.create(run_dir, restart_config=config_payload)
    job_control_manager.create(job.job_id)
    job_manager.log(job.job_id, f"Queued run in {run_dir.name}")
    job_manager.update(job.job_id, status="running")

    thread = threading.Thread(
        target=_run_analysis_job,
        args=(job.job_id, run_dir, config_payload),
        daemon=True,
    )
    thread.start()

    return AnalyzeAcceptedResponse(job_id=job.job_id, run_id=run_id, run_dir=str(run_dir))


def _run_analysis_job(job_id: str, run_dir: Path, config_payload: dict[str, Any]) -> None:
    try:
        control = job_control_manager.get(job_id)
        summary = analyze_wide_angle_video(
            job_id=job_id,
            run_dir=run_dir,
            config_payload=config_payload,
            job_manager=job_manager,
            job_control=control,
        )
        job_manager.update(job_id, status="completed", progress=100.0, summary=summary)
        job_manager.log(job_id, "Run completed")
    except AnalysisStoppedError:
        job_manager.update(job_id, status="stopped", error=None)
        job_manager.log(job_id, "Run stopped")
    except Exception as exc:
        job_manager.update(job_id, status="failed", error=str(exc))
        job_manager.log(job_id, f"Run failed: {exc}")
    finally:
        job_control_manager.clear(job_id)


def _launch_training_job_async(job_id: str, run_dir: Path, config_payload: dict[str, Any]) -> None:
    if training_manager is None:
        return
    try:
        training_manager.launch(job_id, run_dir, config_payload)
    except Exception as exc:
        training_manager.update(job_id, status="failed", error=str(exc))
        training_manager.append_log(job_id, f"Training failed to launch: {exc}")

def choose_device() -> str:
    try:
        import torch

        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    except Exception:
        return "cpu"


def safe_int(value: object, default: int = -1) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except Exception:
        return default


def draw_label(frame: np.ndarray, text: str, x1: int, y1: int, color: tuple[int, int, int]) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thickness = 2
    (w, h), baseline = cv2.getTextSize(text, font, scale, thickness)
    top = max(0, y1 - h - baseline - 8)
    right = min(frame.shape[1] - 1, x1 + w + 8)
    bottom = max(0, y1)
    cv2.rectangle(frame, (x1, top), (right, bottom), (20, 20, 20), -1)
    cv2.putText(frame, text, (x1 + 4, bottom - 4), font, scale, color, thickness, cv2.LINE_AA)


def draw_pose(frame: np.ndarray, keypoints_xyc: np.ndarray, color: tuple[int, int, int]) -> int:
    visible = 0
    for edge_start, edge_end in SKELETON_EDGES:
        x1, y1, c1 = keypoints_xyc[edge_start]
        x2, y2, c2 = keypoints_xyc[edge_end]
        if c1 >= KEYPOINT_CONF_THRESHOLD and c2 >= KEYPOINT_CONF_THRESHOLD:
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2, cv2.LINE_AA)

    for x, y, conf in keypoints_xyc:
        if conf >= KEYPOINT_CONF_THRESHOLD:
            visible += 1
            cv2.circle(frame, (int(x), int(y)), 3, color, -1)
    return visible


def build_empty_keypoints(num_keypoints: int = 17) -> np.ndarray:
    return np.zeros((num_keypoints, 3), dtype=np.float32)


def normalize_keypoints_to_bbox(keypoints_xyc: np.ndarray, bbox_xyxy: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    x1, y1, x2, y2 = bbox_xyxy.astype(np.float32)
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    scale = max(float(x2 - x1), float(y2 - y1), 1.0)
    normalized = keypoints_xyc.copy().astype(np.float32)
    normalized[:, 0] = (normalized[:, 0] - cx) / scale
    normalized[:, 1] = (normalized[:, 1] - cy) / scale
    return normalized, np.array([cx, cy], dtype=np.float32), float(scale)


def zip_paths(output_zip_path: Path, paths: list[Path]) -> None:
    with zipfile.ZipFile(output_zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            if path.is_dir():
                for child in sorted(path.rglob("*")):
                    if child.is_file():
                        archive.write(child, arcname=str(child.relative_to(output_zip_path.parent)))
            elif path.is_file():
                archive.write(path, arcname=path.name)


def finalize_overlay_video(raw_video_path: Path, final_video_path: Path, job_id: str) -> None:
    if not raw_video_path.exists():
        raise RuntimeError(f"Overlay video was not written: {raw_video_path}")

    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        job_manager.log(job_id, "ffmpeg not found; keeping raw mp4v overlay video, which may not play in browsers.")
        raw_video_path.replace(final_video_path)
        return

    transcoded_path = final_video_path.with_name(f"{final_video_path.stem}.browser.mp4")
    command = [
        ffmpeg_path,
        "-y",
        "-i",
        str(raw_video_path),
        "-map",
        "0:v:0",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-an",
        str(transcoded_path),
    ]

    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        if transcoded_path.exists():
            transcoded_path.unlink()
        stderr_lines = [line.strip() for line in exc.stderr.splitlines() if line.strip()]
        detail = stderr_lines[-1] if stderr_lines else str(exc)
        job_manager.log(job_id, f"ffmpeg H.264 transcode failed; keeping raw mp4v overlay video. {detail}")
        raw_video_path.replace(final_video_path)
        return

    transcoded_path.replace(final_video_path)
    raw_video_path.unlink(missing_ok=True)
    job_manager.log(job_id, "Overlay video transcoded to H.264 for browser playback")


def parse_homography_points(raw_value: object) -> tuple[np.ndarray | None, np.ndarray | None]:
    if raw_value is None:
        return None, None
    raw_text = str(raw_value).strip()
    if not raw_text:
        return None, None

    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ValueError("Homography points must be valid JSON") from exc

    def _parse_points(points: object, label: str) -> np.ndarray:
        if not isinstance(points, list) or len(points) != 4:
            raise ValueError(f"{label} must be a JSON array with four [x, y] pairs")
        parsed: list[list[float]] = []
        for point in points:
            if not isinstance(point, (list, tuple)) or len(point) != 2:
                raise ValueError(f"Each item in {label} must be a two-item array")
            parsed.append([float(point[0]), float(point[1])])
        return np.array(parsed, dtype=np.float32)

    if isinstance(payload, dict):
        source_points = payload.get("source") or payload.get("source_points")
        target_points = payload.get("target") or payload.get("target_points")
        if source_points is None or target_points is None:
            raise ValueError("Homography JSON objects must include both source and target point arrays")
        return _parse_points(source_points, "source points"), _parse_points(target_points, "target points")

    return _parse_points(payload, "homography points"), pitch_destination_points()


def extract_jersey_color_feature(frame: np.ndarray, bbox_xyxy: np.ndarray) -> np.ndarray | None:
    frame_height, frame_width = frame.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox_xyxy]
    x1 = max(0, min(x1, frame_width - 1))
    x2 = max(0, min(x2, frame_width))
    y1 = max(0, min(y1, frame_height - 1))
    y2 = max(0, min(y2, frame_height))

    box_width = x2 - x1
    box_height = y2 - y1
    if box_width < 14 or box_height < 28:
        return None

    jersey_top = y1 + int(box_height * 0.18)
    jersey_bottom = y1 + int(box_height * 0.58)
    jersey_left = x1 + int(box_width * 0.2)
    jersey_right = x1 + int(box_width * 0.8)
    if jersey_right - jersey_left < 8 or jersey_bottom - jersey_top < 8:
        return None

    crop = frame[jersey_top:jersey_bottom, jersey_left:jersey_right]
    if crop.size == 0:
        return None

    hsv_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    lab_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)

    saturation = hsv_crop[:, :, 1]
    value = hsv_crop[:, :, 2]
    hue = hsv_crop[:, :, 0]
    colorful_mask = (saturation >= 40) & (value >= 45)
    grass_mask = (hue >= 28) & (hue <= 95) & (saturation >= 55)
    mask = colorful_mask & ~grass_mask

    if int(mask.sum()) < 24:
        pixels = lab_crop.reshape(-1, 3).astype(np.float32)
    else:
        pixels = lab_crop[mask].reshape(-1, 3).astype(np.float32)

    if len(pixels) == 0:
        return None
    return pixels.mean(axis=0)


def fit_two_cluster_kmeans(features: np.ndarray, max_iterations: int = 20) -> tuple[np.ndarray, np.ndarray]:
    if len(features) < 2:
        raise ValueError("Need at least two feature vectors for K-means")

    centroid_guess = features.mean(axis=0)
    first_index = int(np.argmin(np.sum((features - centroid_guess) ** 2, axis=1)))
    first_center = features[first_index]
    second_index = int(np.argmax(np.sum((features - first_center) ** 2, axis=1)))
    second_center = features[second_index]

    if np.allclose(first_center, second_center):
        second_center = first_center + np.array([1.0, -1.0, 1.0], dtype=np.float32)

    centers = np.stack([first_center, second_center]).astype(np.float32)
    labels = np.zeros(len(features), dtype=np.int32)

    for _ in range(max_iterations):
        distances = np.sum((features[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        labels = np.argmin(distances, axis=1).astype(np.int32)
        if np.unique(labels).size < 2:
            split_index = int(np.argmax(distances[:, 0]))
            labels[split_index] = 1

        next_centers = centers.copy()
        for cluster_index in range(2):
            cluster_features = features[labels == cluster_index]
            if len(cluster_features) > 0:
                next_centers[cluster_index] = cluster_features.mean(axis=0)

        if np.allclose(next_centers, centers):
            centers = next_centers
            break
        centers = next_centers

    return labels, centers


def cluster_team_name_map(centers: np.ndarray) -> dict[int, str]:
    order = np.lexsort((centers[:, 2], centers[:, 1], centers[:, 0]))
    return {
        int(order[0]): "home",
        int(order[1]): "away",
    }


def assign_team_labels(
    player_records: list[dict[str, Any]],
    feature_entries: list[tuple[int, int, np.ndarray]],
) -> tuple[dict[int, dict[str, float | int | str]], float, int, list[list[float]]]:
    for record in player_records:
        record["team_label"] = "unassigned"
        record["team_vote_ratio"] = 0.0

    if len(feature_entries) < 6:
        return {}, 0.0, len(feature_entries), []

    features = np.stack([entry[2] for entry in feature_entries], axis=0).astype(np.float32)
    labels, centers = fit_two_cluster_kmeans(features)
    team_map = cluster_team_name_map(centers)

    track_votes: defaultdict[int, dict[str, int]] = defaultdict(lambda: {"home": 0, "away": 0})
    for (record_index, track_id, _feature), cluster_label in zip(feature_entries, labels):
        team_label = team_map[int(cluster_label)]
        player_records[record_index]["team_feature_label"] = team_label
        if track_id >= 0:
            track_votes[track_id][team_label] += 1

    track_team_info: dict[int, dict[str, float | int | str]] = {}
    for track_id, votes in track_votes.items():
        total_votes = votes["home"] + votes["away"]
        if total_votes <= 0:
            continue
        team_label = "home" if votes["home"] >= votes["away"] else "away"
        vote_ratio = max(votes["home"], votes["away"]) / total_votes
        track_team_info[track_id] = {
            "team_label": team_label,
            "team_vote_ratio": round(float(vote_ratio), 4),
            "feature_count": total_votes,
        }

    for record in player_records:
        track_id = int(record["track_id"])
        if track_id in track_team_info:
            team_info = track_team_info[track_id]
            record["team_label"] = str(team_info["team_label"])
            record["team_vote_ratio"] = float(team_info["team_vote_ratio"])

    cluster_distance = float(np.linalg.norm(centers[0] - centers[1]))
    center_list = [[round(float(value), 4) for value in center] for center in centers]
    return track_team_info, cluster_distance, len(feature_entries), center_list


def pitch_destination_points() -> np.ndarray:
    width, height = PITCH_MAP_SIZE
    pad = 18
    return np.array(
        [
            [pad, pad],
            [width - pad, pad],
            [width - pad, height - pad],
            [pad, height - pad],
        ],
        dtype=np.float32,
    )


def project_point(point_xy: tuple[float, float], homography_matrix: np.ndarray) -> tuple[int, int] | None:
    point = np.array([[[float(point_xy[0]), float(point_xy[1])]]], dtype=np.float32)
    projected = cv2.perspectiveTransform(point, homography_matrix)[0][0]
    width, height = PITCH_MAP_SIZE
    x = int(round(float(projected[0])))
    y = int(round(float(projected[1])))
    if x < 0 or x >= width or y < 0 or y >= height:
        return None
    return x, y


def build_pitch_map(player_records: list[dict[str, Any]], ball_record: dict[str, Any] | None) -> np.ndarray:
    width, height = PITCH_MAP_SIZE
    pitch = np.full((height, width, 3), (30, 92, 54), dtype=np.uint8)
    pad = 18
    cv2.rectangle(pitch, (pad, pad), (width - pad, height - pad), (238, 242, 240), 2, cv2.LINE_AA)
    cv2.line(pitch, (width // 2, pad), (width // 2, height - pad), (238, 242, 240), 2, cv2.LINE_AA)
    cv2.circle(pitch, (width // 2, height // 2), 24, (238, 242, 240), 2, cv2.LINE_AA)

    box_width = 54
    box_height = 92
    cv2.rectangle(pitch, (pad, (height - box_height) // 2), (pad + box_width, (height + box_height) // 2), (238, 242, 240), 2, cv2.LINE_AA)
    cv2.rectangle(pitch, (width - pad - box_width, (height - box_height) // 2), (width - pad, (height + box_height) // 2), (238, 242, 240), 2, cv2.LINE_AA)

    for record in player_records:
        pitch_point = record.get("pitch_point")
        if pitch_point is None:
            continue
        team_label = str(record.get("team_label", "unassigned"))
        color = TEAM_DRAW_COLORS.get(team_label, TEAM_DRAW_COLORS["unassigned"])
        cv2.circle(pitch, pitch_point, 6, color, -1, cv2.LINE_AA)
        track_id = int(record.get("track_id", -1))
        if track_id >= 0:
            cv2.putText(
                pitch,
                str(track_id),
                (pitch_point[0] + 8, pitch_point[1] - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

    if ball_record is not None and ball_record.get("pitch_point") is not None:
        cv2.circle(pitch, ball_record["pitch_point"], 5, TEAM_DRAW_COLORS["ball"], -1, cv2.LINE_AA)

    return pitch


def overlay_pitch_map(frame: np.ndarray, pitch_map: np.ndarray) -> None:
    frame_height, frame_width = frame.shape[:2]
    map_height, map_width = pitch_map.shape[:2]
    max_width = int(frame_width * 0.3)
    if map_width > max_width > 0:
        scale = max_width / map_width
        pitch_map = cv2.resize(pitch_map, (int(map_width * scale), int(map_height * scale)), interpolation=cv2.INTER_AREA)
        map_height, map_width = pitch_map.shape[:2]

    margin = 18
    x1 = max(margin, frame_width - map_width - margin)
    y1 = max(margin, frame_height - map_height - margin)
    x2 = min(frame_width, x1 + map_width)
    y2 = min(frame_height, y1 + map_height)

    panel = frame[y1 - 8:y2 + 8, x1 - 8:x2 + 8]
    if panel.size > 0:
        panel[:] = cv2.addWeighted(panel, 0.25, np.full_like(panel, (8, 16, 14)), 0.75, 0.0)
    frame[y1:y2, x1:x2] = pitch_map[: y2 - y1, : x2 - x1]
    cv2.putText(frame, "Projected minimap", (x1, max(18, y1 - 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)


def analyze_video(job_id: str, run_dir: Path, config_payload: dict[str, Any]) -> dict[str, Any]:
    source_video_path = Path(config_payload["source_video_path"])
    player_model_name = str(config_payload.get("player_model") or config_payload.get("pose_model") or "yolo11n.pt")
    ball_model_name = str(config_payload["ball_model"])
    include_ball = bool(config_payload["include_ball"])
    player_conf = float(config_payload.get("player_conf") or config_payload.get("pose_conf") or 0.25)
    ball_conf = float(config_payload["ball_conf"])
    iou = float(config_payload["iou"])
    homography_points, homography_target_points = parse_homography_points(config_payload.get("homography_points", ""))
    homography_matrix = cv2.getPerspectiveTransform(homography_points, homography_target_points) if homography_points is not None and homography_target_points is not None else None

    outputs_dir = run_dir / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    overlay_video_path = outputs_dir / "overlay.mp4"
    raw_overlay_video_path = outputs_dir / "overlay_raw.mp4"
    detections_csv_path = outputs_dir / "detections.csv"
    track_summary_csv_path = outputs_dir / "track_summary.csv"
    projection_csv_path = outputs_dir / "projection.csv"
    summary_json_path = outputs_dir / "summary.json"
    full_outputs_zip_path = outputs_dir / "all_outputs.zip"

    job_manager.log(job_id, f"Opening video: {source_video_path}")
    device = choose_device()
    job_manager.log(job_id, f"Device chosen: {device}")
    job_manager.log(job_id, f"Loading player checkpoint: {player_model_name}")
    player_model = YOLO(player_model_name)

    ball_model: YOLO | None = None
    if include_ball:
        job_manager.log(job_id, f"Loading ball checkpoint: {ball_model_name}")
        ball_model = YOLO(ball_model_name)
    else:
        job_manager.log(job_id, "Ball stage disabled")

    cap = cv2.VideoCapture(str(source_video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {source_video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    width = safe_int(cap.get(cv2.CAP_PROP_FRAME_WIDTH), 1280)
    height = safe_int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT), 720)
    total_frames = safe_int(cap.get(cv2.CAP_PROP_FRAME_COUNT), 0)

    player_records: list[dict[str, Any]] = []
    ball_records: list[dict[str, Any]] = []
    player_records_by_frame: list[list[dict[str, Any]]] = []
    ball_record_by_frame: list[dict[str, Any] | None] = []
    team_feature_entries: list[tuple[int, int, np.ndarray]] = []
    person_track_ids_seen: set[int] = set()
    ball_track_ids_seen: set[int] = set()
    player_detections_per_frame: list[int] = []
    ball_detections_by_frame: list[int] = []
    player_row_count = 0
    ball_row_count = 0
    frame_index = 0

    csv_headers = [
        "frame_index",
        "row_type",
        "track_id",
        "class_name",
        "confidence",
        "x1",
        "y1",
        "x2",
        "y2",
        "anchor_x",
        "anchor_y",
        "team_label",
        "team_vote_ratio",
        "pitch_x",
        "pitch_y",
    ]

    start_time = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_player_records: list[dict[str, Any]] = []
        frame_ball_record: dict[str, Any] | None = None

        player_results = player_model.track(
            source=frame,
            persist=True,
            tracker=DEFAULT_TRACKER,
            conf=player_conf,
            iou=iou,
            device=device,
            classes=[PERSON_CLASS_ID],
            verbose=False,
        )
        player_result = player_results[0]
        player_boxes = player_result.boxes

        if player_boxes is not None and len(player_boxes) > 0:
            xyxy = player_boxes.xyxy.cpu().numpy()
            confs = player_boxes.conf.cpu().numpy() if player_boxes.conf is not None else np.zeros(len(xyxy), dtype=np.float32)
            ids = player_boxes.id.cpu().numpy().astype(int) if player_boxes.id is not None else np.full(len(xyxy), -1, dtype=int)

            for idx, box in enumerate(xyxy):
                x1, y1, x2, y2 = [int(v) for v in box]
                det_conf = float(confs[idx])
                track_id = int(ids[idx])
                anchor_x = float((x1 + x2) / 2.0)
                anchor_y = float(y2)
                bbox_array = np.array([x1, y1, x2, y2], dtype=np.float32)
                feature = extract_jersey_color_feature(frame, bbox_array)

                record = {
                    "frame_index": frame_index,
                    "track_id": track_id,
                    "class_name": "player",
                    "confidence": det_conf,
                    "bbox": bbox_array,
                    "anchor_point": (anchor_x, anchor_y),
                    "team_label": "unassigned",
                    "team_vote_ratio": 0.0,
                    "pitch_point": None,
                }
                record_index = len(player_records)
                player_records.append(record)
                frame_player_records.append(record)
                player_row_count += 1

                if track_id >= 0:
                    person_track_ids_seen.add(track_id)
                    if feature is not None:
                        team_feature_entries.append((record_index, track_id, feature.astype(np.float32)))

        if include_ball and ball_model is not None:
            ball_results = ball_model.track(
                source=frame,
                persist=True,
                tracker=DEFAULT_TRACKER,
                conf=ball_conf,
                iou=iou,
                device=device,
                classes=[BALL_CLASS_ID],
                verbose=False,
            )
            ball_result = ball_results[0]
            ball_boxes = ball_result.boxes
            if ball_boxes is not None and len(ball_boxes) > 0:
                b_xyxy = ball_boxes.xyxy.cpu().numpy()
                b_confs = ball_boxes.conf.cpu().numpy() if ball_boxes.conf is not None else np.zeros(len(b_xyxy), dtype=np.float32)
                b_ids = ball_boxes.id.cpu().numpy().astype(int) if ball_boxes.id is not None else np.full(len(b_xyxy), -1, dtype=int)
                best_index = int(np.argmax(b_confs)) if len(b_confs) > 0 else 0
                x1, y1, x2, y2 = [int(v) for v in b_xyxy[best_index]]
                det_conf = float(b_confs[best_index]) if len(b_confs) > 0 else 0.0
                track_id = int(b_ids[best_index]) if len(b_ids) > 0 else -1
                if track_id >= 0:
                    ball_track_ids_seen.add(track_id)

                frame_ball_record = {
                    "frame_index": frame_index,
                    "track_id": track_id,
                    "class_name": "sports ball",
                    "confidence": det_conf,
                    "bbox": np.array([x1, y1, x2, y2], dtype=np.float32),
                    "anchor_point": (float((x1 + x2) / 2.0), float((y1 + y2) / 2.0)),
                    "pitch_point": None,
                }
                ball_records.append(frame_ball_record)
                ball_row_count += 1

        player_records_by_frame.append(frame_player_records)
        ball_record_by_frame.append(frame_ball_record)
        player_detections_per_frame.append(len(frame_player_records))
        ball_detections_by_frame.append(1 if frame_ball_record is not None else 0)
        frame_index += 1

        if total_frames > 0:
            progress = max(1.0, min(55.0, (frame_index / total_frames) * 55.0))
            job_manager.update(job_id, progress=progress)
        if frame_index % 25 == 0:
            elapsed = time.time() - start_time
            fps_effective = frame_index / elapsed if elapsed > 0 else 0.0
            job_manager.log(job_id, f"Detected frame {frame_index} at {fps_effective:.2f} fps")

    cap.release()

    track_team_info, cluster_distance, feature_count, cluster_centers = assign_team_labels(player_records, team_feature_entries)
    average_vote_ratio = float(
        np.mean([float(info["team_vote_ratio"]) for info in track_team_info.values()])
    ) if track_team_info else 0.0
    job_manager.log(job_id, f"Collected {feature_count} jersey-color samples for team clustering")

    projected_player_points = 0
    projected_ball_points = 0
    if homography_matrix is not None:
        job_manager.log(job_id, "Applying manual 4-point homography for minimap projection")
        for record in player_records:
            pitch_point = project_point(record["anchor_point"], homography_matrix)
            record["pitch_point"] = pitch_point
            if pitch_point is not None:
                projected_player_points += 1
        for record in ball_records:
            pitch_point = project_point(record["anchor_point"], homography_matrix)
            record["pitch_point"] = pitch_point
            if pitch_point is not None:
                projected_ball_points += 1
    else:
        job_manager.log(job_id, "No homography provided; skipping minimap projection")

    with detections_csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(csv_headers)
        for record in player_records:
            x1, y1, x2, y2 = [int(v) for v in record["bbox"]]
            pitch_x = record["pitch_point"][0] if record["pitch_point"] is not None else ""
            pitch_y = record["pitch_point"][1] if record["pitch_point"] is not None else ""
            csv_writer.writerow([
                int(record["frame_index"]),
                "player",
                int(record["track_id"]),
                str(record["class_name"]),
                round(float(record["confidence"]), 5),
                x1,
                y1,
                x2,
                y2,
                round(float(record["anchor_point"][0]), 3),
                round(float(record["anchor_point"][1]), 3),
                str(record["team_label"]),
                round(float(record["team_vote_ratio"]), 4),
                pitch_x,
                pitch_y,
            ])
        for record in ball_records:
            x1, y1, x2, y2 = [int(v) for v in record["bbox"]]
            pitch_x = record["pitch_point"][0] if record["pitch_point"] is not None else ""
            pitch_y = record["pitch_point"][1] if record["pitch_point"] is not None else ""
            csv_writer.writerow([
                int(record["frame_index"]),
                "ball",
                int(record["track_id"]),
                str(record["class_name"]),
                round(float(record["confidence"]), 5),
                x1,
                y1,
                x2,
                y2,
                round(float(record["anchor_point"][0]), 3),
                round(float(record["anchor_point"][1]), 3),
                "",
                "",
                pitch_x,
                pitch_y,
            ])

    if homography_matrix is not None:
        with projection_csv_path.open("w", newline="", encoding="utf-8") as projection_file:
            projection_writer = csv.writer(projection_file)
            projection_writer.writerow(["frame_index", "row_type", "track_id", "team_label", "pitch_x", "pitch_y"])
            for record in player_records:
                if record["pitch_point"] is None:
                    continue
                projection_writer.writerow([
                    int(record["frame_index"]),
                    "player",
                    int(record["track_id"]),
                    str(record["team_label"]),
                    int(record["pitch_point"][0]),
                    int(record["pitch_point"][1]),
                ])
            for record in ball_records:
                if record["pitch_point"] is None:
                    continue
                projection_writer.writerow([
                    int(record["frame_index"]),
                    "ball",
                    int(record["track_id"]),
                    "",
                    int(record["pitch_point"][0]),
                    int(record["pitch_point"][1]),
                ])

    job_manager.log(job_id, "Writing track summary files")
    player_rows_by_track: defaultdict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in player_records:
        track_id = int(record["track_id"])
        if track_id >= 0:
            player_rows_by_track[track_id].append(record)

    longest_track_length = 0
    average_track_length = 0.0
    track_rows: list[dict[str, Any]] = []
    track_lengths: list[int] = []
    with track_summary_csv_path.open("w", newline="", encoding="utf-8") as track_csv_file:
        writer_csv = csv.writer(track_csv_file)
        writer_csv.writerow([
            "track_id",
            "frames",
            "first_frame",
            "last_frame",
            "team_label",
            "team_vote_ratio",
            "average_confidence",
            "average_bbox_area",
            "projected_points",
        ])

        for track_id, rows in sorted(player_rows_by_track.items(), key=lambda item: len(item[1]), reverse=True):
            rows = sorted(rows, key=lambda row: int(row["frame_index"]))
            track_length = len(rows)
            track_lengths.append(track_length)
            longest_track_length = max(longest_track_length, track_length)

            confidences = np.array([float(row["confidence"]) for row in rows], dtype=np.float32)
            bboxes = np.stack([row["bbox"] for row in rows], axis=0).astype(np.float32)
            bbox_areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
            projected_points = sum(1 for row in rows if row["pitch_point"] is not None)
            team_info = track_team_info.get(track_id, {"team_label": "unassigned", "team_vote_ratio": 0.0})

            row_summary = {
                "track_id": track_id,
                "frames": track_length,
                "first_frame": int(rows[0]["frame_index"]),
                "last_frame": int(rows[-1]["frame_index"]),
                "team_label": str(team_info["team_label"]),
                "team_vote_ratio": round(float(team_info["team_vote_ratio"]), 4),
                "average_confidence": round(float(np.mean(confidences)), 4),
                "average_bbox_area": round(float(np.mean(bbox_areas)), 4),
                "projected_points": projected_points,
            }
            track_rows.append(row_summary)
            writer_csv.writerow([
                row_summary["track_id"],
                row_summary["frames"],
                row_summary["first_frame"],
                row_summary["last_frame"],
                row_summary["team_label"],
                row_summary["team_vote_ratio"],
                row_summary["average_confidence"],
                row_summary["average_bbox_area"],
                row_summary["projected_points"],
            ])

    if track_lengths:
        average_track_length = float(np.mean(track_lengths))

    job_manager.log(job_id, "Rendering tactical overlay video")
    cap = cv2.VideoCapture(str(source_video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not reopen video for rendering: {source_video_path}")

    writer = cv2.VideoWriter(
        str(raw_overlay_video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create overlay writer: {raw_overlay_video_path}")

    render_index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        annotated = frame.copy()
        current_players = player_records_by_frame[render_index] if render_index < len(player_records_by_frame) else []
        current_ball = ball_record_by_frame[render_index] if render_index < len(ball_record_by_frame) else None

        for record in current_players:
            x1, y1, x2, y2 = [int(v) for v in record["bbox"]]
            team_label = str(record.get("team_label", "unassigned"))
            color = TEAM_DRAW_COLORS.get(team_label, TEAM_DRAW_COLORS["unassigned"])
            track_id = int(record["track_id"])

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.circle(annotated, (int(record["anchor_point"][0]), int(record["anchor_point"][1])), 4, color, -1, cv2.LINE_AA)
            label_text = f"{team_label} #{track_id}" if track_id >= 0 else f"{team_label} player"
            draw_label(annotated, label_text, x1, y1, (255, 255, 255))

        if current_ball is not None:
            x1, y1, x2, y2 = [int(v) for v in current_ball["bbox"]]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), TEAM_DRAW_COLORS["ball"], 2)
            cv2.circle(
                annotated,
                (int(current_ball["anchor_point"][0]), int(current_ball["anchor_point"][1])),
                4,
                TEAM_DRAW_COLORS["ball"],
                -1,
                cv2.LINE_AA,
            )
            draw_label(annotated, f"ball #{int(current_ball['track_id'])}", x1, y1, (255, 245, 180))

        if homography_points is not None:
            cv2.polylines(annotated, [homography_points.astype(np.int32)], True, (190, 235, 190), 2, cv2.LINE_AA)
            pitch_map = build_pitch_map(current_players, current_ball)
            overlay_pitch_map(annotated, pitch_map)

        status = f"frame {render_index + 1}"
        if total_frames > 0:
            status += f"/{total_frames}"
        cv2.putText(annotated, status, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(annotated, f"player model: {player_model_name}", (20, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 228, 236), 2, cv2.LINE_AA)
        writer.write(annotated)
        render_index += 1

        if total_frames > 0:
            progress = 55.0 + min(43.0, (render_index / total_frames) * 43.0)
            job_manager.update(job_id, progress=progress)
        if render_index % 50 == 0:
            job_manager.log(job_id, f"Rendered {render_index} overlay frames")

    cap.release()
    writer.release()
    finalize_overlay_video(raw_overlay_video_path, overlay_video_path, job_id)

    home_tracks = sum(1 for row in track_rows if row["team_label"] == "home")
    away_tracks = sum(1 for row in track_rows if row["team_label"] == "away")
    unassigned_tracks = sum(1 for row in track_rows if row["team_label"] == "unassigned")

    diagnostics: list[dict[str, str]] = []
    churn_ratio = len(person_track_ids_seen) / max(frame_index, 1)
    avg_player = float(np.mean(player_detections_per_frame)) if player_detections_per_frame else 0.0
    avg_ball = float(np.mean(ball_detections_by_frame)) if ball_detections_by_frame else 0.0

    if churn_ratio > 0.10:
        diagnostics.append({
            "level": "warn",
            "title": "Tracker churn looks high",
            "message": f"{len(person_track_ids_seen)} player IDs across {frame_index} frames suggests fragmented wide-angle tracking.",
            "next_step": "Use shorter clips, lift confidence slightly, or pick steadier camera phases before trusting team or projection outputs.",
        })
    else:
        diagnostics.append({
            "level": "good",
            "title": "Tracker churn is workable",
            "message": f"{len(person_track_ids_seen)} tracked player IDs across {frame_index} frames is reasonable for a first wide-angle demo.",
            "next_step": "Inspect the longest tracks and confirm they stay glued to one player through pans and overlaps.",
        })

    if feature_count < 30:
        diagnostics.append({
            "level": "warn",
            "title": "Team color evidence is thin",
            "message": f"Only {feature_count} jersey-color samples survived filtering, so team labels may be unstable.",
            "next_step": "Use clips with stronger shirt contrast or larger on-screen players so the upper-body crop contains real jersey pixels.",
        })
    elif cluster_distance < 18.0 or average_vote_ratio < 0.65:
        diagnostics.append({
            "level": "warn",
            "title": "Team separation is noisy",
            "message": f"Cluster distance is {cluster_distance:.2f} with average track vote ratio {average_vote_ratio:.2f}.",
            "next_step": "Expect referee and goalkeeper leakage; inspect a few tracks manually before claiming home versus away is solid.",
        })
    else:
        diagnostics.append({
            "level": "good",
            "title": "Team colors separate cleanly enough",
            "message": f"Used {feature_count} jersey samples and reached an average per-track team vote ratio of {average_vote_ratio:.2f}.",
            "next_step": "Remember the labels are unsupervised buckets; they are useful for separation even though home and away are assigned heuristically.",
        })

    if include_ball:
        if avg_ball < 0.2:
            diagnostics.append({
                "level": "warn",
                "title": "Ball tracking is sparse",
                "message": f"Average tracked ball detections per frame is {avg_ball:.2f}.",
                "next_step": "Keep the ball as a bonus layer unless you move to a tighter clip or a more specialized ball detector.",
            })
        else:
            diagnostics.append({
                "level": "good",
                "title": "Ball stage adds usable context",
                "message": f"Average tracked ball detections per frame is {avg_ball:.2f}.",
                "next_step": "Check the overlay for false positives on socks, signs, and crowd highlights before trusting every ball ID.",
            })
    else:
        diagnostics.append({
            "level": "warn",
            "title": "Ball stage is disabled",
            "message": "This run focused only on players and team separation.",
            "next_step": "Turn the ball stage back on when you want the tactical demo to show all four layers together.",
        })

    if homography_matrix is None:
        diagnostics.append({
            "level": "warn",
            "title": "No minimap projection yet",
            "message": "The run completed without manual pitch points, so the overlay stays in image space only.",
            "next_step": "Add four pitch corners in top-left, top-right, bottom-right, bottom-left order to project player foot-points onto a minimap.",
        })
    elif projected_player_points == 0:
        diagnostics.append({
            "level": "warn",
            "title": "Homography did not land on the pitch map",
            "message": "Projection was enabled, but no player anchors ended up inside the minimap bounds.",
            "next_step": "Re-enter the four pitch points carefully. A bad point order breaks the projection immediately.",
        })
    else:
        diagnostics.append({
            "level": "good",
            "title": "Minimap projection is active",
            "message": f"Projected {projected_player_points} player anchors and {projected_ball_points} ball anchors onto the minimap.",
            "next_step": "Treat this as a rough tactical view, not calibrated tracking. The goal here is a believable end-to-end demo.",
        })

    summary = {
        "job_id": job_id,
        "run_dir": str(run_dir),
        "input_video": str(source_video_path),
        "overlay_video": f"/runs/{run_dir.name}/outputs/overlay.mp4",
        "detections_csv": f"/runs/{run_dir.name}/outputs/detections.csv",
        "track_summary_csv": f"/runs/{run_dir.name}/outputs/track_summary.csv",
        "projection_csv": f"/runs/{run_dir.name}/outputs/projection.csv" if homography_matrix is not None else None,
        "summary_json": f"/runs/{run_dir.name}/outputs/summary.json",
        "all_outputs_zip": f"/runs/{run_dir.name}/outputs/all_outputs.zip",
        "device": device,
        "player_model": player_model_name,
        "pose_model": player_model_name,
        "ball_model": ball_model_name if include_ball else "off",
        "include_ball": include_ball,
        "homography_enabled": homography_matrix is not None,
        "homography_points": homography_points.tolist() if homography_points is not None else None,
        "homography_target_points": homography_target_points.tolist() if homography_target_points is not None else None,
        "player_conf": player_conf,
        "pose_conf": player_conf,
        "ball_conf": ball_conf,
        "iou": iou,
        "frames_processed": frame_index,
        "fps": round(float(fps), 4),
        "player_rows": player_row_count,
        "person_rows": player_row_count,
        "ball_rows": ball_row_count,
        "unique_player_track_ids": len(person_track_ids_seen),
        "unique_ball_track_ids": len(ball_track_ids_seen),
        "tracks_collected": len(player_rows_by_track),
        "home_tracks": home_tracks,
        "away_tracks": away_tracks,
        "unassigned_tracks": unassigned_tracks,
        "longest_track_length": longest_track_length,
        "average_track_length": round(float(average_track_length), 4),
        "average_player_detections_per_frame": round(float(avg_player), 4),
        "average_person_detections_per_frame": round(float(avg_player), 4),
        "average_ball_detections_per_frame": round(float(avg_ball), 4),
        "team_color_features_used": feature_count,
        "team_cluster_distance": round(float(cluster_distance), 4),
        "team_cluster_centers_lab": cluster_centers,
        "average_team_vote_ratio": round(float(average_vote_ratio), 4),
        "projected_player_points": projected_player_points,
        "projected_ball_points": projected_ball_points,
        "top_tracks": track_rows[:20],
        "diagnostics": diagnostics,
        "learn_cards": LEARN_CARDS,
        "created_at": datetime.utcnow().isoformat() + "Z",
    }

    summary_json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    zip_targets = [overlay_video_path, detections_csv_path, track_summary_csv_path, summary_json_path]
    if homography_matrix is not None and projection_csv_path.exists():
        zip_targets.append(projection_csv_path)
    zip_paths(full_outputs_zip_path, zip_targets)
    job_manager.log(job_id, f"Summary written to {summary_json_path.name}")
    return summary
