from __future__ import annotations

import csv
import json
import shutil
import subprocess
import time
import zipfile
from collections import Counter, defaultdict, deque
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

from app.ai_diagnostics import generate_run_diagnostics
from app.reid_tracker import (
    BALL_TRACKER_NAME,
    DEFAULT_PLAYER_TRACKER_MODE,
    HybridReIDTracker,
    LEGACY_PLAYER_TRACKER_MODE,
    PLAYER_TRACKER_MODE_OPTIONS,
    build_stitched_track_map,
    normalize_player_tracker_mode,
    tracker_mode_label,
)

DEFAULT_TRACKER = BALL_TRACKER_NAME
MODEL_CACHE_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
BROWSER_VIDEO_FOURCCS = ("avc1", "H264", "h264", "X264")

DETECTOR_MODEL_DEFAULT = "soccana"
KEYPOINT_MODEL_DEFAULT = "soccana_keypoint"
CALIBRATION_REFRESH_FRAMES = 10
CALIBRATION_SMOOTHING_WINDOW = 5
CALIBRATION_MAX_AGE_FRAMES = 30
FIELD_KEYPOINT_CONFIDENCE = 0.25
MIN_CALIBRATION_VISIBLE_KEYPOINTS = 4
MIN_CALIBRATION_INLIERS = 4
MAX_CALIBRATION_REPROJECTION_ERROR_CM = 250.0
MAX_CALIBRATION_TEMPORAL_DRIFT_CM = 1800.0
STALE_RECOVERY_MIN_CALIBRATION_VISIBLE_KEYPOINTS = 5
STALE_RECOVERY_MIN_CALIBRATION_INLIERS = 5
STALE_RECOVERY_MAX_CALIBRATION_REPROJECTION_ERROR_CM = 350.0
STALE_RECOVERY_TEMPORAL_DRIFT_CM = 5000.0
EXPERIMENT_WINDOW_SECONDS = 10.0
EXPERIMENT_GRID_COLS = 6
EXPERIMENT_GRID_ROWS = 4
GOAL_LOOKAHEAD_SECONDS = (30, 60)
PROGRESS_TRACKING_MAX = 68.0
PROGRESS_RENDER_END = 90.0
PROGRESS_VIDEO_FINALIZE_END = 94.0
PROGRESS_DIAGNOSTICS_END = 98.0
PROGRESS_PACKAGING_END = 99.0
DETECTOR_DEBUG_SAMPLE_FRAMES = 5

DETECTOR_MODEL_SOURCES = {
    "soccana": {
        "repo_id": "Adit-jain/soccana",
        "filename": "Model/weights/best.pt",
        "player_class_id": 0,
        "ball_class_id": 1,
        "referee_class_id": 2,
    },
}

KEYPOINT_MODEL_SOURCES = {
    "soccana_keypoint": {
        "repo_id": "Adit-jain/Soccana_Keypoint",
        "filename": "Model/weights/best.pt",
    },
}

PLAYER_MODEL_OPTIONS = [
    "soccana",
]

BALL_MODEL_OPTIONS = [
    "shared-detector",
]

PITCH_LENGTH_CM = 12000.0
PITCH_WIDTH_CM = 7000.0
PITCH_RENDER_MARGIN = 14
PITCH_REFERENCE_POINTS = np.array(
    [
        (0.0, 0.0),
        (0.0, 1450.0),
        (2015.0, 1450.0),
        (0.0, 5550.0),
        (2015.0, 5550.0),
        (0.0, 2584.0),
        (550.0, 2584.0),
        (0.0, 4416.0),
        (550.0, 4416.0),
        (0.0, 7000.0),
        (2932.0, 3500.0),
        (6000.0, 0.0),
        (6000.0, 7000.0),
        (6000.0, 2585.0),
        (6000.0, 4415.0),
        (6000.0, 3500.0),
        (12000.0, 0.0),
        (12000.0, 1450.0),
        (9985.0, 1450.0),
        (12000.0, 5550.0),
        (9985.0, 5550.0),
        (12000.0, 2584.0),
        (11450.0, 2584.0),
        (12000.0, 4416.0),
        (11450.0, 4416.0),
        (12000.0, 7000.0),
        (9069.0, 3500.0),
        (5085.0, 3500.0),
        (6915.0, 3500.0),
    ],
    dtype=np.float32,
)

TACTICAL_LEARN_CARDS = [
    {
        "title": "1. Player detection + hybrid ReID tracker",
        "what_it_does": "Uses football-specific detection, sparse appearance embeddings, field-aware association, and a stitch pass so player IDs can survive short occlusions and re-entries.",
        "what_breaks": "Crowded same-kit players, close-up broadcast cuts, and tiny distant crops can still create duplicate IDs.",
        "what_to_try_next": "Inspect raw-vs-stitched ID counts and longest tracks before trusting any per-player sequence.",
    },
    {
        "title": "2. Team ID from jersey colors",
        "what_it_does": "Samples upper-body jersey colors, clusters them into two groups, and tags tracks as home or away.",
        "what_breaks": "Goalkeepers, referees, shadows, and tiny crops can poison the color signal.",
        "what_to_try_next": "Use clips with clear kit contrast first and treat the labels as approximate, not gospel.",
    },
    {
        "title": "3. Ball tracking",
        "what_it_does": "Runs a second detector for the ball so the demo is about play context, not just player boxes.",
        "what_breaks": "The ball is small, fast, and happy to hide behind players or advertising boards.",
        "what_to_try_next": "Keep the detector lightweight for now and judge it by overlay sanity, not by perfection.",
    },
    {
        "title": "4. Automatic field registration",
        "what_it_does": "Refreshes pitch calibration from field keypoints every frame and smooths the last few accepted homographies so the minimap can follow a moving broadcast camera without jitter.",
        "what_breaks": "If the keypoint model loses field structure during a fast cut or tight zoom, the projection has to coast on the last good calibration.",
        "what_to_try_next": "Watch the live preview for calibration dropouts instead of trusting the minimap blindly.",
    },
    {
        "title": "5. Demo first, fancy later",
        "what_it_does": "Packages detection, appearance-aware tracking, team ID, and rough spatial context into one football-first pipeline.",
        "what_breaks": "Long match-wide identity is still harder than short tactical windows, especially across cutaways and resets.",
        "what_to_try_next": "Treat stitched IDs as tactical tracklets first, then add jersey numbers or shot segmentation if match-long identity still matters.",
    },
]

HELP_CATALOG_PATH = Path(__file__).with_name("help_catalog.json")
TACTICAL_HELP_CATALOG = json.loads(HELP_CATALOG_PATH.read_text(encoding="utf-8"))

TEAM_BOX_COLORS = {
    "home": (255, 110, 80),
    "away": (80, 110, 255),
    "unassigned": (170, 170, 170),
    "ball": (0, 215, 255),
}
CALIBRATION_REJECTION_REASON_KEYS = (
    "no_candidate",
    "low_visible_count",
    "low_inliers",
    "high_reprojection_error",
    "high_temporal_drift",
)
CALIBRATION_PRIMARY_REJECTION_REASON_KEYS = CALIBRATION_REJECTION_REASON_KEYS + ("invalid_candidate",)
PLAYER_CLASS_LABEL_HINTS = ("player", "players", "person", "footballer", "athlete", "goalkeeper", "goalie", "keeper")
BALL_CLASS_LABEL_HINTS = ("ball", "soccer ball", "sports ball", "football")
REFEREE_CLASS_LABEL_HINTS = ("referee", "ref", "official", "linesman")


def _normalize_four_points(points: Any, label: str) -> list[list[float]]:
    if not isinstance(points, list) or len(points) != 4:
        raise ValueError(f"{label} must be a JSON array of four [x, y] pairs.")

    normalized: list[list[float]] = []
    for item in points:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError(f"Each point in {label} must be a two-item array like [123, 456].")
        normalized.append([float(item[0]), float(item[1])])
    return normalized


def parse_homography_points(raw_value: str) -> dict[str, list[list[float]]] | None:
    if not raw_value or not raw_value.strip():
        return None

    data = json.loads(raw_value)
    if isinstance(data, dict):
        source_points = _normalize_four_points(data.get("source"), "source points")
        target_points = _normalize_four_points(data.get("target"), "target points")
        return {
            "source": source_points,
            "target": target_points,
        }

    return {
        "source": _normalize_four_points(data, "homography points"),
        "target": [],
    }


def expected_model_cache_path(model_name: str, model_kind: str) -> Path | None:
    if model_kind == "detector" and model_name in DETECTOR_MODEL_SOURCES:
        model_info = DETECTOR_MODEL_SOURCES[model_name]
        return (MODEL_CACHE_DIR / model_name / model_info["filename"]).resolve()
    if model_kind == "keypoint" and model_name in KEYPOINT_MODEL_SOURCES:
        model_info = KEYPOINT_MODEL_SOURCES[model_name]
        return (MODEL_CACHE_DIR / model_name / model_info["filename"]).resolve()
    return None


@lru_cache(maxsize=8)
def resolve_model_path(model_name: str, model_kind: str) -> str:
    model_key = model_name.strip()
    if not model_key:
        model_key = DETECTOR_MODEL_DEFAULT if model_kind == "detector" else KEYPOINT_MODEL_DEFAULT

    cached_path = expected_model_cache_path(model_key, model_kind)
    if cached_path is not None and cached_path.exists():
        return str(cached_path)

    if model_kind == "detector" and model_key in DETECTOR_MODEL_SOURCES:
        model_info = DETECTOR_MODEL_SOURCES[model_key]
        path = hf_hub_download(
            repo_id=model_info["repo_id"],
            filename=model_info["filename"],
            local_dir=str(MODEL_CACHE_DIR / model_key),
        )
        return str(Path(path).resolve())

    if model_kind == "keypoint" and model_key in KEYPOINT_MODEL_SOURCES:
        model_info = KEYPOINT_MODEL_SOURCES[model_key]
        path = hf_hub_download(
            repo_id=model_info["repo_id"],
            filename=model_info["filename"],
            local_dir=str(MODEL_CACHE_DIR / model_key),
        )
        return str(Path(path).resolve())

    candidate = Path(model_key).expanduser()
    if candidate.exists():
        return str(candidate.resolve())
    return model_key


def prewarm_default_models() -> dict[str, str]:
    return {
        "detector": resolve_model_path(DETECTOR_MODEL_DEFAULT, "detector"),
        "field_calibration": resolve_model_path(KEYPOINT_MODEL_DEFAULT, "keypoint"),
    }


def _normalize_class_name(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().replace("_", " ").replace("-", " ").split())


def _coerce_class_name_map(raw_names: Any) -> dict[int, str]:
    if isinstance(raw_names, dict):
        names: dict[int, str] = {}
        for key, value in raw_names.items():
            try:
                names[int(key)] = str(value)
            except Exception:
                continue
        return dict(sorted(names.items()))
    if isinstance(raw_names, (list, tuple)):
        return {index: str(value) for index, value in enumerate(raw_names)}
    return {}


def _resolve_names_from_dataset_yaml(dataset_yaml_path: Path) -> dict[int, str]:
    if not dataset_yaml_path.exists():
        return {}
    with dataset_yaml_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return _coerce_class_name_map(data.get("names"))


def _candidate_dataset_yaml_paths(weights_path: Path, model: YOLO) -> list[Path]:
    candidates: list[Path] = []
    override_data = model.overrides.get("data")
    if override_data:
        candidates.append(Path(str(override_data)).expanduser())

    run_dir = weights_path.parent.parent
    args_yaml_path = run_dir / "yolo_output" / "train" / "args.yaml"
    if args_yaml_path.exists():
        try:
            with args_yaml_path.open("r", encoding="utf-8") as handle:
                args_yaml = yaml.safe_load(handle) or {}
            if args_yaml.get("data"):
                candidates.append(Path(str(args_yaml["data"])).expanduser())
        except Exception:
            pass

    config_path = run_dir / "config.json"
    if config_path.exists():
        try:
            config_payload = json.loads(config_path.read_text(encoding="utf-8"))
            dataset_path = config_payload.get("dataset_path")
            if dataset_path:
                candidates.append(Path(str(dataset_path)).expanduser() / "dataset.yaml")
        except Exception:
            pass

    candidates.extend(
        [
            weights_path.with_suffix(".yaml"),
            weights_path.parent / "dataset.yaml",
            weights_path.parent.parent / "dataset.yaml",
            weights_path.parent.parent / "data.yaml",
        ]
    )
    deduped: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        resolved = str(candidate)
        if resolved in seen:
            continue
        seen.add(resolved)
        deduped.append(candidate)
    return deduped


@lru_cache(maxsize=8)
def resolve_detector_class_names(weights_path: str) -> tuple[dict[int, str], str]:
    resolved_path = Path(weights_path).expanduser().resolve()
    model = YOLO(str(resolved_path))
    names = _coerce_class_name_map(getattr(model, "names", None))
    if names:
        return names, "checkpoint metadata"

    for dataset_yaml_path in _candidate_dataset_yaml_paths(resolved_path, model):
        names = _resolve_names_from_dataset_yaml(dataset_yaml_path)
        if names:
            return names, str(dataset_yaml_path)

    return {}, "unresolved"


def _matching_class_ids(names: dict[int, str], hints: tuple[str, ...]) -> list[int]:
    hint_set = {_normalize_class_name(hint) for hint in hints}
    matched_ids: list[int] = []
    for class_id, label in names.items():
        normalized = _normalize_class_name(label)
        if not normalized:
            continue
        tokens = set(normalized.split())
        if normalized in hint_set or tokens.intersection(hint_set):
            matched_ids.append(int(class_id))
    return sorted(set(matched_ids))


def format_class_id_list(class_ids: list[int], class_names: dict[int, str]) -> str:
    if not class_ids:
        return "none"
    return ", ".join(f"{class_id}:{class_names.get(class_id, '?')}" for class_id in class_ids)


def format_class_histogram(class_histogram: dict[int, int], class_names: dict[int, str]) -> str:
    if not class_histogram:
        return "none"
    return ", ".join(
        f"{class_id}:{class_names.get(class_id, '?')}={count}"
        for class_id, count in sorted(class_histogram.items())
    )


def sample_detector_class_histogram(
    frame: np.ndarray,
    detector_model: YOLO,
    detector_device: str,
    confidence: float,
    iou: float,
) -> tuple[int, dict[int, int]]:
    results = detector_model(
        source=frame,
        conf=confidence,
        iou=iou,
        device=detector_device,
        verbose=False,
    )
    if not results:
        return 0, {}
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0 or boxes.cls is None:
        return 0, {}
    class_ids = boxes.cls.cpu().numpy().astype(int).tolist()
    return len(class_ids), {int(class_id): int(count) for class_id, count in Counter(class_ids).items()}


def resolve_detector_spec(model_name: str) -> dict[str, Any]:
    model_key = model_name.strip() or DETECTOR_MODEL_DEFAULT
    if model_key in DETECTOR_MODEL_SOURCES:
        model_info = DETECTOR_MODEL_SOURCES[model_key]
        class_names = {
            int(model_info["player_class_id"]): "player",
            int(model_info["ball_class_id"]): "ball",
            int(model_info["referee_class_id"]): "referee",
        }
        return {
            "name": model_key,
            "weights_path": resolve_model_path(model_key, "detector"),
            "class_names": class_names,
            "class_names_source": "built-in detector spec",
            "player_class_ids": [int(model_info["player_class_id"])],
            "ball_class_ids": [int(model_info["ball_class_id"])],
            "referee_class_ids": [int(model_info["referee_class_id"])],
            "player_class_id": int(model_info["player_class_id"]),
            "ball_class_id": int(model_info["ball_class_id"]),
            "referee_class_id": int(model_info["referee_class_id"]),
        }

    candidate = resolve_model_path(model_key, "detector")
    class_names, class_names_source = resolve_detector_class_names(candidate)
    player_class_ids = _matching_class_ids(class_names, PLAYER_CLASS_LABEL_HINTS)
    ball_class_ids = _matching_class_ids(class_names, BALL_CLASS_LABEL_HINTS)
    referee_class_ids = _matching_class_ids(class_names, REFEREE_CLASS_LABEL_HINTS)
    if not class_names or not player_class_ids or not ball_class_ids:
        discovered = format_class_histogram({class_id: 1 for class_id in class_names}, class_names) if class_names else "none"
        raise RuntimeError(
            "Could not resolve detector class ids for custom checkpoint "
            f"{candidate}. Names source: {class_names_source}. Discovered classes: {discovered}. "
            "Expected to find player/goalkeeper and ball labels in checkpoint metadata or adjacent dataset YAML."
        )
    return {
        "name": model_key,
        "weights_path": candidate,
        "class_names": class_names,
        "class_names_source": class_names_source,
        "player_class_ids": player_class_ids,
        "ball_class_ids": ball_class_ids,
        "referee_class_ids": referee_class_ids,
        "player_class_id": int(player_class_ids[0]),
        "ball_class_id": int(ball_class_ids[0]),
        "referee_class_id": int(referee_class_ids[0]) if referee_class_ids else -1,
    }


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


def choose_keypoint_device(detector_device: str) -> str:
    if detector_device == "mps":
        return "cpu"
    return detector_device


def safe_int(value: object, default: int = -1) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except Exception:
        return default


def resolve_player_tracker_mode(config_payload: dict[str, Any]) -> str:
    return normalize_player_tracker_mode(
        config_payload.get("tracker_mode") or config_payload.get("player_tracker_mode")
    )


def requested_player_tracker_mode(config_payload: dict[str, Any]) -> str | None:
    raw_value = str(config_payload.get("tracker_mode") or config_payload.get("player_tracker_mode") or "").strip()
    return raw_value or None


def resolved_tracker_runtime_label(tracker_mode: str) -> str:
    return "legacy_bytetrack" if tracker_mode == LEGACY_PLAYER_TRACKER_MODE else "hybrid_reid_stitch"


def default_tracker_backend() -> dict[str, Any]:
    return {
        "tracker_mode": DEFAULT_PLAYER_TRACKER_MODE,
        "embedding_source": "none",
        "embedding_error": None,
        "deep_feature_updates": 0,
        "deep_feature_interval": 0,
        "assignment_count": 0,
        "new_track_count": 0,
        "max_missing_frames": 0,
        "stitch_gap_frames": 0,
    }


def new_calibration_rejection_counts() -> dict[str, int]:
    return {key: 0 for key in CALIBRATION_REJECTION_REASON_KEYS}


def calibration_visible_keypoint_minimum(stale_recovery_mode: bool) -> int:
    return STALE_RECOVERY_MIN_CALIBRATION_VISIBLE_KEYPOINTS if stale_recovery_mode else MIN_CALIBRATION_VISIBLE_KEYPOINTS


def calibration_inlier_minimum(stale_recovery_mode: bool) -> int:
    return STALE_RECOVERY_MIN_CALIBRATION_INLIERS if stale_recovery_mode else MIN_CALIBRATION_INLIERS


def calibration_reprojection_limit_cm(stale_recovery_mode: bool) -> float:
    return STALE_RECOVERY_MAX_CALIBRATION_REPROJECTION_ERROR_CM if stale_recovery_mode else MAX_CALIBRATION_REPROJECTION_ERROR_CM


def calibration_temporal_drift_limit_cm(stale_recovery_mode: bool) -> float:
    return STALE_RECOVERY_TEMPORAL_DRIFT_CM if stale_recovery_mode else MAX_CALIBRATION_TEMPORAL_DRIFT_CM


def calibration_rejection_flags(
    homography_candidate: np.ndarray | None,
    visible_count: int,
    inlier_count: int,
    reprojection_error: float,
    temporal_drift: float,
    min_visible_count: int,
    min_inlier_count: int,
    reprojection_limit_cm: float,
    temporal_drift_limit_cm: float,
) -> list[str]:
    flags: list[str] = []
    if homography_candidate is None:
        flags.append("no_candidate")
    if visible_count < min_visible_count:
        flags.append("low_visible_count")
    if inlier_count < min_inlier_count:
        flags.append("low_inliers")
    if homography_candidate is not None and (
        not np.isfinite(reprojection_error) or reprojection_error > reprojection_limit_cm
    ):
        flags.append("high_reprojection_error")
    if homography_candidate is not None and (
        not np.isfinite(temporal_drift) or temporal_drift > temporal_drift_limit_cm
    ):
        flags.append("high_temporal_drift")
    return flags


def primary_calibration_rejection_reason(rejection_flags: list[str]) -> str:
    for reason in CALIBRATION_REJECTION_REASON_KEYS:
        if reason in rejection_flags:
            return reason
    return "invalid_candidate"


def calibration_success_rate(successes: int, attempts: int) -> float:
    if attempts <= 0:
        return 0.0
    return float(successes / attempts)


def calibration_rejection_summary(rejections: dict[str, int], invalid_candidate_rejections: int = 0) -> str:
    parts = [
        f"no cand {int(rejections.get('no_candidate', 0))}",
        f"low vis {int(rejections.get('low_visible_count', 0))}",
        f"low inliers {int(rejections.get('low_inliers', 0))}",
        f"reproj {int(rejections.get('high_reprojection_error', 0))}",
        f"drift {int(rejections.get('high_temporal_drift', 0))}",
    ]
    if invalid_candidate_rejections > 0:
        parts.append(f"invalid {int(invalid_candidate_rejections)}")
    return ", ".join(parts)


def compute_track_length_stats(rows_by_track: dict[int, list[dict[str, Any]]]) -> tuple[int, float]:
    if not rows_by_track:
        return 0, 0.0
    lengths = [len(rows) for rows in rows_by_track.values()]
    return max(lengths), float(np.mean(lengths))


def rebuild_player_rows_by_track(frame_records: list[dict[str, Any]]) -> tuple[dict[int, list[dict[str, Any]]], set[int]]:
    rows_by_track: dict[int, list[dict[str, Any]]] = defaultdict(list)
    track_ids_seen: set[int] = set()
    for record in frame_records:
        for player_row in record["players"]:
            track_id = int(player_row["track_id"])
            if track_id < 0:
                continue
            rows_by_track[track_id].append(player_row)
            track_ids_seen.add(track_id)
    return rows_by_track, track_ids_seen


def apply_player_track_id_map(
    frame_records: list[dict[str, Any]],
    jersey_sample_track_ids: list[int],
    stitched_track_map: dict[int, int],
) -> None:
    if not stitched_track_map:
        return
    for record in frame_records:
        for player_row in record["players"]:
            track_id = int(player_row["track_id"])
            if track_id >= 0:
                player_row["track_id"] = int(stitched_track_map.get(track_id, track_id))
    for index, track_id in enumerate(jersey_sample_track_ids):
        jersey_sample_track_ids[index] = int(stitched_track_map.get(track_id, track_id))


def export_player_tracklets_from_rows(rows_by_track: dict[int, list[dict[str, Any]]]) -> dict[int, dict[str, Any]]:
    tracklets: dict[int, dict[str, Any]] = {}
    for track_id, rows in rows_by_track.items():
        if not rows:
            continue
        sorted_rows = sorted(rows, key=lambda row: int(row["frame_index"]))
        confidences = [float(row["confidence"]) for row in sorted_rows]
        bbox_areas = [
            float((row["bbox"][2] - row["bbox"][0]) * (row["bbox"][3] - row["bbox"][1]))
            for row in sorted_rows
        ]
        identity_features = [
            np.asarray(row["identity_feature"], dtype=np.float32)
            for row in sorted_rows
            if row.get("identity_feature") is not None
        ]
        mean_feature = None
        if identity_features:
            mean_feature = np.mean(np.stack(identity_features, axis=0), axis=0).astype(np.float32)
            norm = float(np.linalg.norm(mean_feature))
            if norm > 1e-8:
                mean_feature = mean_feature / norm
            else:
                mean_feature = None
        tracklets[int(track_id)] = {
            "track_id": int(track_id),
            "first_frame": int(sorted_rows[0]["frame_index"]),
            "last_frame": int(sorted_rows[-1]["frame_index"]),
            "first_anchor": tuple(sorted_rows[0]["anchor"]),
            "last_anchor": tuple(sorted_rows[-1]["anchor"]),
            "first_field_point": sorted_rows[0].get("field_point"),
            "last_field_point": sorted_rows[-1].get("field_point"),
            "average_confidence": float(np.mean(confidences)) if confidences else 0.0,
            "average_bbox_area": float(np.mean(bbox_areas)) if bbox_areas else 0.0,
            "observation_count": len(sorted_rows),
            "mean_feature": mean_feature,
        }
    return tracklets


def group_rows_by_canonical_track(
    rows_by_track: dict[int, list[dict[str, Any]]],
    stitched_track_map: dict[int, int],
) -> dict[int, list[dict[str, Any]]]:
    if not stitched_track_map:
        return rows_by_track

    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for raw_track_id, rows in rows_by_track.items():
        canonical_track_id = int(stitched_track_map.get(raw_track_id, raw_track_id))
        grouped[canonical_track_id].extend(rows)

    for rows in grouped.values():
        rows.sort(key=lambda row: int(row["frame_index"]))
    return grouped


def clamp_box(box: np.ndarray, frame_width: int, frame_height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = [int(v) for v in box]
    x1 = max(0, min(frame_width - 1, x1))
    y1 = max(0, min(frame_height - 1, y1))
    x2 = max(x1 + 1, min(frame_width, x2))
    y2 = max(y1 + 1, min(frame_height, y2))
    return x1, y1, x2, y2


def build_overlay_style(frame_width: int, frame_height: int) -> dict[str, Any]:
    short_side = max(1, min(frame_width, frame_height))
    margin = max(8, int(round(short_side * 0.035)))
    label_scale = max(0.34, min(0.58, short_side / 620.0))
    status_scale = max(0.34, min(0.82, short_side / 500.0))
    small_status_scale = max(0.3, min(0.66, short_side / 620.0))
    thickness = 1 if short_side < 540 else 2
    line_gap = max(20, int(round(short_side * 0.12)))
    minimap_width = int(min(frame_width * 0.34, frame_height * 0.42 * 105.0 / 68.0, 360.0))
    minimap_width = max(120, minimap_width)
    minimap_height = int(round(minimap_width * 68.0 / 105.0))
    return {
        "margin": margin,
        "label_scale": label_scale,
        "status_scale": status_scale,
        "small_status_scale": small_status_scale,
        "text_thickness": thickness,
        "line_gap": line_gap,
        "label_padding_x": max(4, int(round(short_side * 0.012))),
        "label_padding_y": max(3, int(round(short_side * 0.01))),
        "box_thickness": max(1, int(round(short_side / 320.0))),
        "keypoint_radius": max(2, int(round(short_side * 0.01))),
        "minimap_width": minimap_width,
        "minimap_height": minimap_height,
        "minimap_title_scale": max(0.34, min(0.58, short_side / 720.0)),
        "minimap_point_radius": max(2, int(round(short_side * 0.008))),
        "minimap_ball_radius": max(2, int(round(short_side * 0.006))),
        "panel_padding": max(6, int(round(short_side * 0.02))),
        "title_band_height": max(18, int(round(short_side * 0.08))),
    }


def pitch_render_margin(map_width: int, map_height: int) -> int:
    return max(6, min(PITCH_RENDER_MARGIN, int(round(min(map_width, map_height) * 0.06))))


def draw_label(
    frame: np.ndarray,
    text: str,
    x1: int,
    y1: int,
    color: tuple[int, int, int],
    scale: float = 0.55,
    thickness: int = 2,
    padding_x: int = 4,
    padding_y: int = 4,
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    (width, height), baseline = cv2.getTextSize(text, font, scale, thickness)
    top = max(0, y1 - height - baseline - (padding_y * 2))
    right = min(frame.shape[1] - 1, x1 + width + (padding_x * 2))
    bottom = max(0, y1)
    cv2.rectangle(frame, (x1, top), (right, bottom), (20, 20, 20), -1)
    cv2.putText(frame, text, (x1 + padding_x, bottom - padding_y), font, scale, color, thickness, cv2.LINE_AA)


def draw_status_banner(
    frame: np.ndarray,
    text: str,
    style: dict[str, Any],
    background_color: tuple[int, int, int] = (28, 28, 148),
    text_color: tuple[int, int, int] = (255, 255, 255),
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = style["small_status_scale"]
    thickness = style["text_thickness"]
    padding_x = style["label_padding_x"] + 2
    padding_y = style["label_padding_y"] + 1
    (width, height), baseline = cv2.getTextSize(text, font, scale, thickness)
    left = style["margin"]
    top = style["margin"]
    right = min(frame.shape[1] - 1, left + width + padding_x * 2)
    bottom = min(frame.shape[0] - 1, top + height + baseline + padding_y * 2)
    cv2.rectangle(frame, (left, top), (right, bottom), background_color, -1)
    cv2.putText(
        frame,
        text,
        (left + padding_x, bottom - padding_y - baseline),
        font,
        scale,
        text_color,
        thickness,
        cv2.LINE_AA,
    )


def zip_paths(output_zip_path: Path, paths: list[Path]) -> None:
    with zipfile.ZipFile(output_zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            if path.is_dir():
                for child in sorted(path.rglob("*")):
                    if child.is_file():
                        archive.write(child, arcname=str(child.relative_to(output_zip_path.parent)))
            elif path.is_file():
                archive.write(path, arcname=path.name)


def try_open_video_writer(
    output_path: Path,
    fps: float,
    frame_size: tuple[int, int],
    codec_candidates: tuple[str, ...],
) -> tuple[cv2.VideoWriter | None, str | None]:
    for codec in codec_candidates:
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*codec),
            fps,
            frame_size,
        )
        if writer.isOpened():
            return writer, codec
        writer.release()
    return None, None


def create_overlay_writer(
    overlay_video_path: Path,
    raw_overlay_video_path: Path,
    fps: float,
    frame_size: tuple[int, int],
    job_id: str,
    job_manager: Any,
) -> tuple[cv2.VideoWriter, Path, bool]:
    ffmpeg_available = shutil.which("ffmpeg") is not None
    if not ffmpeg_available:
        writer, codec = try_open_video_writer(overlay_video_path, fps, frame_size, BROWSER_VIDEO_FOURCCS)
        if writer is not None and codec is not None:
            job_manager.log(job_id, f"Using direct {codec} overlay encoding for browser playback")
            return writer, overlay_video_path, False

    writer, codec = try_open_video_writer(raw_overlay_video_path, fps, frame_size, ("mp4v",))
    if writer is None or codec is None:
        writer, codec = try_open_video_writer(overlay_video_path, fps, frame_size, BROWSER_VIDEO_FOURCCS)
        if writer is not None and codec is not None:
            job_manager.log(job_id, f"Using direct {codec} overlay encoding for browser playback")
            return writer, overlay_video_path, False
        raise RuntimeError(f"Could not create overlay writer: {raw_overlay_video_path}")

    if ffmpeg_available:
        job_manager.log(job_id, "Writing overlay with mp4v then transcoding to H.264")
    else:
        job_manager.log(job_id, "ffmpeg not found; direct browser codec unavailable, falling back to mp4v overlay output")
    return writer, raw_overlay_video_path, True


def update_job_progress(job_manager: Any, job_id: str, value: float) -> None:
    job_manager.update(job_id, progress=max(0.0, min(100.0, float(value))))


def finalize_overlay_video(raw_video_path: Path, final_video_path: Path, job_id: str, job_manager: Any) -> None:
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


def extract_jersey_feature(frame: np.ndarray, bbox_xyxy: tuple[int, int, int, int]) -> np.ndarray | None:
    x1, y1, x2, y2 = bbox_xyxy
    width = x2 - x1
    height = y2 - y1
    if width < 12 or height < 24:
        return None

    crop_x1 = x1 + int(width * 0.2)
    crop_x2 = x2 - int(width * 0.2)
    crop_y1 = y1 + int(height * 0.15)
    crop_y2 = y1 + int(height * 0.55)
    if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
        return None

    crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
    if crop.size == 0:
        return None

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hsv_pixels = hsv.reshape(-1, 3)
    bgr_pixels = crop.reshape(-1, 3)
    hue = hsv_pixels[:, 0]
    sat = hsv_pixels[:, 1]
    val = hsv_pixels[:, 2]

    colorful = (sat > 35) & (val > 40)
    non_grass = ~((hue >= 35) & (hue <= 95) & (sat > 45))
    mask = colorful & non_grass
    if int(mask.sum()) < max(25, hsv_pixels.shape[0] // 20):
        mask = colorful
    if int(mask.sum()) < 25:
        return None

    rgb_pixels = bgr_pixels[mask][:, ::-1].astype(np.float32) / 255.0
    return rgb_pixels.mean(axis=0).astype(np.float32)


def kmeans_two_clusters(samples: np.ndarray, max_iters: int = 25) -> tuple[np.ndarray, np.ndarray]:
    if samples.ndim != 2 or samples.shape[0] < 2:
        raise ValueError("Need at least two samples for K-means.")

    low_index = int(np.argmin(samples[:, 0]))
    high_index = int(np.argmax(samples[:, 0]))
    centroids = np.stack([samples[low_index], samples[high_index]], axis=0).astype(np.float32)
    if np.allclose(centroids[0], centroids[1]):
        centroids[1] = samples[len(samples) // 2]

    labels = np.zeros(samples.shape[0], dtype=np.int32)
    for _ in range(max_iters):
        distances = np.linalg.norm(samples[:, None, :] - centroids[None, :, :], axis=2)
        new_labels = np.argmin(distances, axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels

        new_centroids = centroids.copy()
        for cluster_index in range(2):
            cluster_samples = samples[labels == cluster_index]
            if len(cluster_samples) == 0:
                farthest_index = int(np.argmax(np.min(distances, axis=1)))
                new_centroids[cluster_index] = samples[farthest_index]
            else:
                new_centroids[cluster_index] = cluster_samples.mean(axis=0)

        if np.allclose(new_centroids, centroids):
            centroids = new_centroids
            break
        centroids = new_centroids

    return centroids, labels


def create_pitch_map(map_width: int, map_height: int) -> np.ndarray:
    image = np.full((map_height, map_width, 3), (38, 96, 58), dtype=np.uint8)
    line_color = (238, 238, 238)
    margin = pitch_render_margin(map_width, map_height)
    line_thickness = 1 if min(map_width, map_height) < 160 else 2
    cv2.rectangle(image, (margin, margin), (map_width - margin, map_height - margin), line_color, line_thickness)

    center_x = map_width // 2
    center_y = map_height // 2
    cv2.line(image, (center_x, margin), (center_x, map_height - margin), line_color, line_thickness)
    cv2.circle(image, (center_x, center_y), max(8, map_height // 7), line_color, line_thickness)

    penalty_box_width = int(round((16.5 / 105.0) * (map_width - 2 * margin)))
    penalty_box_height = int(round((40.3 / 68.0) * (map_height - 2 * margin)))
    top_box_y = center_y - penalty_box_height // 2
    bottom_box_y = center_y + penalty_box_height // 2
    cv2.rectangle(image, (margin, top_box_y), (margin + penalty_box_width, bottom_box_y), line_color, line_thickness)
    cv2.rectangle(image, (map_width - margin - penalty_box_width, top_box_y), (map_width - margin, bottom_box_y), line_color, line_thickness)
    return image


def compute_homography_matrix(homography_payload: dict[str, list[list[float]]] | None, map_width: int, map_height: int) -> tuple[np.ndarray | None, np.ndarray | None]:
    if not homography_payload or not homography_payload.get("source"):
        return None, None

    source = np.array(homography_payload["source"], dtype=np.float32)
    if homography_payload.get("target"):
        destination = np.array(homography_payload["target"], dtype=np.float32)
    else:
        margin = float(pitch_render_margin(map_width, map_height))
        destination = np.array(
            [
                [margin, margin],
                [map_width - margin, margin],
                [map_width - margin, map_height - margin],
                [margin, map_height - margin],
            ],
            dtype=np.float32,
        )
    matrix = cv2.getPerspectiveTransform(source, destination)
    return matrix, source


def project_point(point_xy: tuple[float, float], homography_matrix: np.ndarray | None) -> tuple[float, float] | None:
    if homography_matrix is None:
        return None

    source = np.array([[[float(point_xy[0]), float(point_xy[1])]]], dtype=np.float32)
    projected = cv2.perspectiveTransform(source, homography_matrix)[0, 0]
    if not np.isfinite(projected).all():
        return None

    x_value = float(projected[0])
    y_value = float(projected[1])
    if x_value < -500.0 or x_value > PITCH_LENGTH_CM + 500.0:
        return None
    if y_value < -500.0 or y_value > PITCH_WIDTH_CM + 500.0:
        return None
    return x_value, y_value


def field_point_to_minimap(field_point: tuple[float, float] | None, map_width: int, map_height: int) -> tuple[float, float] | None:
    if field_point is None:
        return None

    margin = pitch_render_margin(map_width, map_height)
    usable_width = map_width - 2 * margin
    usable_height = map_height - 2 * margin
    x_value = margin + (float(np.clip(field_point[0], 0.0, PITCH_LENGTH_CM)) / PITCH_LENGTH_CM) * usable_width
    y_value = margin + (float(np.clip(field_point[1], 0.0, PITCH_WIDTH_CM)) / PITCH_WIDTH_CM) * usable_height
    return float(x_value), float(y_value)


def calibration_stale_for_frame(
    homography_matrix: np.ndarray | None,
    last_good_calibration_frame: int,
    frame_index: int,
) -> bool:
    if homography_matrix is None or last_good_calibration_frame < 0:
        return True
    return (frame_index - last_good_calibration_frame) > CALIBRATION_MAX_AGE_FRAMES


def homography_reprojection_error_cm(
    homography_matrix: np.ndarray,
    image_points: np.ndarray,
    field_points: np.ndarray,
) -> float:
    if image_points.size == 0 or field_points.size == 0:
        return float("inf")
    projected = cv2.perspectiveTransform(image_points.reshape(1, -1, 2), homography_matrix)[0]
    if projected.shape != field_points.shape or not np.isfinite(projected).all():
        return float("inf")
    errors = np.linalg.norm(projected - field_points, axis=1)
    if errors.size == 0 or not np.isfinite(errors).all():
        return float("inf")
    return float(np.mean(errors))


def homography_temporal_drift_cm(
    previous_homography: np.ndarray | None,
    candidate_homography: np.ndarray,
    frame_width: int,
    frame_height: int,
) -> float:
    if previous_homography is None:
        return 0.0

    sample_points = np.array(
        [
            [frame_width * 0.18, frame_height * 0.82],
            [frame_width * 0.35, frame_height * 0.86],
            [frame_width * 0.50, frame_height * 0.90],
            [frame_width * 0.65, frame_height * 0.86],
            [frame_width * 0.82, frame_height * 0.82],
        ],
        dtype=np.float32,
    ).reshape(1, -1, 2)
    previous_projected = cv2.perspectiveTransform(sample_points, previous_homography)[0]
    candidate_projected = cv2.perspectiveTransform(sample_points, candidate_homography)[0]
    if not np.isfinite(previous_projected).all() or not np.isfinite(candidate_projected).all():
        return float("inf")

    drifts = np.linalg.norm(candidate_projected - previous_projected, axis=1)
    if drifts.size == 0 or not np.isfinite(drifts).all():
        return float("inf")
    return float(np.median(drifts))


def normalize_homography_matrix(homography_matrix: np.ndarray | None) -> np.ndarray | None:
    if homography_matrix is None:
        return None
    normalized = homography_matrix.astype(np.float32).copy()
    scale = float(normalized[2, 2]) if normalized.shape == (3, 3) else 0.0
    if np.isfinite(scale) and abs(scale) > 1e-6:
        normalized /= scale
    if not np.isfinite(normalized).all():
        return None
    return normalized


def smooth_homography_history(history: deque[np.ndarray]) -> np.ndarray | None:
    if not history:
        return None
    stack = np.stack([matrix for matrix in history], axis=0).astype(np.float32)
    averaged = np.mean(stack, axis=0)
    if not np.isfinite(averaged).all():
        return history[-1].copy()
    scale = float(averaged[2, 2]) if averaged.shape == (3, 3) else 0.0
    if np.isfinite(scale) and abs(scale) > 1e-6:
        averaged /= scale
    else:
        averaged[2, 2] = 1.0
    return averaged.astype(np.float32)


class AnalysisStoppedError(RuntimeError):
    pass


def checkpoint_job_control(job_control: Any, job_id: str, job_manager: Any) -> None:
    if job_control is None:
        return

    while True:
        if job_control.is_stop_requested():
            raise AnalysisStoppedError("Run stopped by user.")
        if not job_control.is_pause_requested():
            return
        time.sleep(0.25)


def detect_pitch_homography(
    frame: np.ndarray,
    keypoint_model: YOLO,
    device: str,
    confidence_threshold: float = FIELD_KEYPOINT_CONFIDENCE,
) -> tuple[np.ndarray | None, np.ndarray | None, int, int, float]:
    results = keypoint_model(source=frame, conf=0.05, device=device, verbose=False)
    if not results:
        return None, None, 0, 0, float("inf")

    result = results[0]
    if not hasattr(result, "keypoints") or result.keypoints is None or result.keypoints.data is None:
        return None, None, 0, 0, float("inf")

    keypoints_data = result.keypoints.data.cpu().numpy().astype(np.float32)
    if keypoints_data.size == 0:
        return None, None, 0, 0, float("inf")

    best_index = int(np.argmax((keypoints_data[:, :, 2] >= confidence_threshold).sum(axis=1)))
    keypoints = keypoints_data[best_index]
    valid_mask = keypoints[:, 2] >= confidence_threshold
    visible_count = int(valid_mask.sum())
    if visible_count < 4:
        return None, keypoints, visible_count, 0, float("inf")

    image_points = keypoints[valid_mask, :2].astype(np.float32)
    field_points = PITCH_REFERENCE_POINTS[valid_mask].astype(np.float32)
    homography_matrix, inlier_mask = cv2.findHomography(image_points, field_points, cv2.RANSAC, 35.0)
    inlier_count = int(inlier_mask.sum()) if inlier_mask is not None else visible_count
    if homography_matrix is None or inlier_count < 4:
        return None, keypoints, visible_count, inlier_count, float("inf")
    reprojection_error = homography_reprojection_error_cm(homography_matrix.astype(np.float32), image_points, field_points)
    return homography_matrix.astype(np.float32), keypoints, visible_count, inlier_count, reprojection_error


def overlay_minimap(frame: np.ndarray, minimap: np.ndarray, style: dict[str, Any]) -> None:
    inset_height, inset_width = minimap.shape[:2]
    margin = style["margin"]
    panel_padding = style["panel_padding"]
    title_band_height = style["title_band_height"]
    panel_top = max(0, frame.shape[0] - inset_height - margin - title_band_height - panel_padding * 2)
    panel_left = max(0, frame.shape[1] - inset_width - margin - panel_padding * 2)
    panel_bottom = min(frame.shape[0], panel_top + inset_height + title_band_height + panel_padding * 2)
    panel_right = min(frame.shape[1], panel_left + inset_width + panel_padding * 2)
    cv2.rectangle(frame, (panel_left, panel_top), (panel_right, panel_bottom), (18, 18, 18), -1)

    map_top = panel_top + title_band_height + panel_padding
    map_left = panel_left + panel_padding
    frame[map_top:map_top + inset_height, map_left:map_left + inset_width] = minimap
    cv2.putText(
        frame,
        "Minimap",
        (panel_left + panel_padding, panel_top + title_band_height - max(4, panel_padding // 2)),
        cv2.FONT_HERSHEY_SIMPLEX,
        style["minimap_title_scale"],
        (245, 245, 245),
        style["text_thickness"],
        cv2.LINE_AA,
    )


def draw_detected_field_keypoints(
    frame: np.ndarray,
    keypoints: np.ndarray | None,
    style: dict[str, Any],
    confidence_threshold: float = FIELD_KEYPOINT_CONFIDENCE,
) -> None:
    if keypoints is None:
        return
    for x_value, y_value, confidence in keypoints:
        if confidence < confidence_threshold:
            continue
        cv2.circle(frame, (int(round(x_value)), int(round(y_value))), style["keypoint_radius"], (120, 220, 255), -1, cv2.LINE_AA)


def default_team_info() -> dict[str, Any]:
    return {
        "team_label": "unassigned",
        "team_vote_ratio": 0.0,
        "sample_count": 0,
        "average_color_rgb": [0.0, 0.0, 0.0],
    }


def compute_convex_hull_area(points: np.ndarray) -> float:
    if points.shape[0] < 3:
        return 0.0
    hull = cv2.convexHull(points.astype(np.float32))
    return float(cv2.contourArea(hull))


def compute_spatial_entropy(points: np.ndarray, cols: int = EXPERIMENT_GRID_COLS, rows: int = EXPERIMENT_GRID_ROWS) -> float:
    if points.shape[0] == 0:
        return float("nan")

    x_values = np.clip(points[:, 0], 0.0, PITCH_LENGTH_CM - 1e-6)
    y_values = np.clip(points[:, 1], 0.0, PITCH_WIDTH_CM - 1e-6)
    col_indices = np.minimum((x_values / PITCH_LENGTH_CM * cols).astype(int), cols - 1)
    row_indices = np.minimum((y_values / PITCH_WIDTH_CM * rows).astype(int), rows - 1)

    counts = np.zeros((rows, cols), dtype=np.float32)
    for row_index, col_index in zip(row_indices, col_indices):
        counts[row_index, col_index] += 1.0

    total = float(counts.sum())
    if total <= 0:
        return float("nan")

    probabilities = counts[counts > 0] / total
    entropy = float(-(probabilities * np.log(probabilities)).sum())
    normalization = float(np.log(rows * cols))
    return entropy / normalization if normalization > 0 else entropy


def compute_team_shape_metrics(points: np.ndarray) -> dict[str, float]:
    if points.shape[0] == 0:
        return {
            "player_count": 0.0,
            "centroid_x_cm": float("nan"),
            "centroid_y_cm": float("nan"),
            "spread_rms_cm": float("nan"),
            "length_axis_cm": float("nan"),
            "width_axis_cm": float("nan"),
            "hull_area_cm2": float("nan"),
        }

    centroid = points.mean(axis=0)
    centered = points - centroid
    distances_sq = np.sum(centered ** 2, axis=1)
    spread_rms = float(np.sqrt(np.mean(distances_sq)))
    length_axis = float(np.percentile(points[:, 0], 90) - np.percentile(points[:, 0], 10)) if points.shape[0] >= 2 else 0.0
    width_axis = float(np.percentile(points[:, 1], 90) - np.percentile(points[:, 1], 10)) if points.shape[0] >= 2 else 0.0

    return {
        "player_count": float(points.shape[0]),
        "centroid_x_cm": float(centroid[0]),
        "centroid_y_cm": float(centroid[1]),
        "spread_rms_cm": spread_rms,
        "length_axis_cm": length_axis,
        "width_axis_cm": width_axis,
        "hull_area_cm2": compute_convex_hull_area(points),
    }


def rolling_nanmean(values: list[float], window: int) -> list[float]:
    result: list[float] = []
    for index in range(len(values)):
        start_index = max(0, index - window + 1)
        window_values = [value for value in values[start_index:index + 1] if np.isfinite(value)]
        result.append(float(np.mean(window_values)) if window_values else float("nan"))
    return result


def zscore_series(values: list[float]) -> list[float]:
    finite_values = np.array([value for value in values if np.isfinite(value)], dtype=np.float32)
    if finite_values.size == 0:
        return [float("nan")] * len(values)
    mean_value = float(finite_values.mean())
    std_value = float(finite_values.std())
    if std_value < 1e-8:
        return [0.0 if np.isfinite(value) else float("nan") for value in values]
    return [((float(value) - mean_value) / std_value) if np.isfinite(value) else float("nan") for value in values]


def build_geometric_volatility_experiment(frame_records: list[dict[str, Any]], fps: float) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    frame_rows: list[dict[str, float]] = []
    seconds_per_frame = 1.0 / fps if fps > 0 else 1.0 / 25.0

    for frame_index, record in enumerate(frame_records):
        home_points = np.array(
            [row["field_point"] for row in record["players"] if row["team_label"] == "home" and row["field_point"] is not None],
            dtype=np.float32,
        )
        away_points = np.array(
            [row["field_point"] for row in record["players"] if row["team_label"] == "away" and row["field_point"] is not None],
            dtype=np.float32,
        )
        all_points = np.concatenate([home_points, away_points], axis=0) if home_points.size and away_points.size else (home_points if home_points.size else away_points)

        home_metrics = compute_team_shape_metrics(home_points)
        away_metrics = compute_team_shape_metrics(away_points)
        centroid_distance = (
            float(np.linalg.norm(np.array([home_metrics["centroid_x_cm"], home_metrics["centroid_y_cm"]]) - np.array([away_metrics["centroid_x_cm"], away_metrics["centroid_y_cm"]])))
            if np.isfinite(home_metrics["centroid_x_cm"]) and np.isfinite(away_metrics["centroid_x_cm"])
            else float("nan")
        )
        entropy_grid = compute_spatial_entropy(all_points) if all_points.size else float("nan")

        frame_rows.append(
            {
                "frame_index": float(frame_index),
                "seconds": frame_index * seconds_per_frame,
                "home_player_count": home_metrics["player_count"],
                "away_player_count": away_metrics["player_count"],
                "home_centroid_x_cm": home_metrics["centroid_x_cm"],
                "home_centroid_y_cm": home_metrics["centroid_y_cm"],
                "away_centroid_x_cm": away_metrics["centroid_x_cm"],
                "away_centroid_y_cm": away_metrics["centroid_y_cm"],
                "home_spread_rms_cm": home_metrics["spread_rms_cm"],
                "away_spread_rms_cm": away_metrics["spread_rms_cm"],
                "home_length_axis_cm": home_metrics["length_axis_cm"],
                "away_length_axis_cm": away_metrics["length_axis_cm"],
                "home_width_axis_cm": home_metrics["width_axis_cm"],
                "away_width_axis_cm": away_metrics["width_axis_cm"],
                "home_hull_area_cm2": home_metrics["hull_area_cm2"],
                "away_hull_area_cm2": away_metrics["hull_area_cm2"],
                "centroid_distance_cm": centroid_distance,
                "entropy_grid": entropy_grid,
            }
        )

    numeric_keys = [
        "home_player_count",
        "away_player_count",
        "home_centroid_x_cm",
        "home_centroid_y_cm",
        "away_centroid_x_cm",
        "away_centroid_y_cm",
        "home_spread_rms_cm",
        "away_spread_rms_cm",
        "home_length_axis_cm",
        "away_length_axis_cm",
        "home_width_axis_cm",
        "away_width_axis_cm",
        "home_hull_area_cm2",
        "away_hull_area_cm2",
        "centroid_distance_cm",
        "entropy_grid",
    ]

    rows_by_second: dict[int, list[dict[str, float]]] = defaultdict(list)
    for row in frame_rows:
        rows_by_second[int(row["seconds"])].append(row)

    second_rows: list[dict[str, Any]] = []
    for second_bucket in sorted(rows_by_second):
        bucket_rows = rows_by_second[second_bucket]
        aggregated: dict[str, Any] = {"second": second_bucket, "seconds": float(second_bucket)}
        for key in numeric_keys:
            finite_values = [float(row[key]) for row in bucket_rows if np.isfinite(row[key])]
            aggregated[key] = float(np.mean(finite_values)) if finite_values else float("nan")
        second_rows.append(aggregated)

    window_seconds = max(3, int(round(EXPERIMENT_WINDOW_SECONDS)))
    volatility_features = [
        "home_spread_rms_cm",
        "away_spread_rms_cm",
        "home_length_axis_cm",
        "away_length_axis_cm",
        "centroid_distance_cm",
        "entropy_grid",
    ]

    for feature in volatility_features:
        delta_key = f"{feature}_delta"
        volatility_key = f"{feature}_volatility"
        zscore_key = f"{feature}_z"
        deltas: list[float] = []
        values = [float(row[feature]) for row in second_rows]
        for index, value in enumerate(values):
            if index == 0 or not np.isfinite(value) or not np.isfinite(values[index - 1]):
                deltas.append(float("nan"))
            else:
                deltas.append(abs(value - values[index - 1]))
        rolling_values = rolling_nanmean(deltas, window_seconds)
        z_values = zscore_series(rolling_values)
        for row, delta_value, rolling_value, z_value in zip(second_rows, deltas, rolling_values, z_values):
            row[delta_key] = delta_value
            row[volatility_key] = rolling_value
            row[zscore_key] = z_value

    combined_values: list[float] = []
    for row in second_rows:
        z_values = [row[f"{feature}_z"] for feature in volatility_features if np.isfinite(row[f"{feature}_z"])]
        combined = float(np.mean(z_values)) if z_values else float("nan")
        row["vol_index"] = combined
        combined_values.append(combined)

    finite_combined = [value for value in combined_values if np.isfinite(value)]
    experiment_card = {
        "id": "geometric_volatility_index",
        "status": "experimental",
        "title": "Geometric Volatility Index",
        "summary": "1 Hz team-shape volatility built from spread, team length, inter-team centroid distance, and spatial entropy.",
        "interpretation": "Higher values indicate rapid spatial reorganization and a more unstable game state.",
        "metrics": [
            {
                "label": "Peak vol index",
                "value": round(float(max(finite_combined)), 4) if finite_combined else 0.0,
                "hint": "Highest normalized combined volatility over the half.",
            },
            {
                "label": "Average vol index",
                "value": round(float(np.mean(finite_combined)), 4) if finite_combined else 0.0,
                "hint": "Baseline volatility across the half.",
            },
            {
                "label": "Latest vol index",
                "value": round(float(finite_combined[-1]), 4) if finite_combined else 0.0,
                "hint": "Most recent combined volatility reading.",
            },
            {
                "label": "Sampling",
                "value": "1 Hz / 10s",
                "hint": "Per-second series with 10-second rolling mean absolute deltas.",
            },
        ],
    }
    return second_rows, experiment_card


def resolve_goal_label_path(source_video_path: Path, explicit_label_path: str = "") -> Path | None:
    candidate_paths: list[Path] = []

    def add_candidate(path: Path) -> None:
        try:
            normalized = path.expanduser().resolve()
        except Exception:
            normalized = path.expanduser()
        if normalized not in candidate_paths:
            candidate_paths.append(normalized)

    if explicit_label_path.strip():
        add_candidate(Path(explicit_label_path.strip()))

    search_directories: list[Path] = [source_video_path.parent]
    search_directories.extend(list(source_video_path.parents[:4]))
    for directory in search_directories:
        add_candidate(directory / "Labels-v2.json")
        add_candidate(directory / "Labels.json")

    return next((path for path in candidate_paths if path.exists() and path.is_file()), None)


def load_soccernet_goal_events(source_video_path: Path, explicit_label_path: str = "") -> tuple[list[dict[str, Any]], str | None]:
    label_path = resolve_goal_label_path(source_video_path, explicit_label_path)
    if label_path is None:
        return [], None

    half_number = 1 if source_video_path.name.startswith("1") else 2 if source_video_path.name.startswith("2") else None
    try:
        payload = json.loads(label_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return [], str(label_path)

    annotations = payload.get("annotations") if isinstance(payload, dict) else None
    if not isinstance(annotations, list):
        return [], str(label_path)

    goal_events: list[dict[str, Any]] = []
    for annotation in annotations:
        if not isinstance(annotation, dict):
            continue
        label = str(annotation.get("label", ""))
        if "goal" not in label.lower():
            continue

        game_time = str(annotation.get("gameTime", "")).strip()
        try:
            half_text, clock_text = [part.strip() for part in game_time.split("-", 1)]
            annotation_half = int(half_text)
            minute_text, second_text = [part.strip() for part in clock_text.split(":", 1)]
            game_clock_seconds = int(minute_text) * 60 + int(second_text)
        except Exception:
            annotation_half = None
            game_clock_seconds = None

        try:
            position_ms = int(str(annotation.get("position", "0")))
        except ValueError:
            position_ms = 0

        if half_number is not None and annotation_half is not None and annotation_half != half_number:
            continue

        event_seconds = position_ms / 1000.0
        goal_events.append(
            {
                "half": annotation_half,
                "seconds": round(event_seconds, 4),
                "position_ms": position_ms,
                "game_clock_seconds": game_clock_seconds,
                "team": annotation.get("team"),
                "visibility": annotation.get("visibility"),
                "label": label,
            }
        )

    goal_events.sort(key=lambda item: float(item["seconds"]))
    return goal_events, str(label_path)


def attach_goal_targets(
    timeseries_rows: list[dict[str, Any]],
    goal_events: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not goal_events:
        for row in timeseries_rows:
            row["seconds_to_next_goal"] = float("nan")
            row["next_goal_team"] = ""
            for lookahead in GOAL_LOOKAHEAD_SECONDS:
                row[f"goal_in_next_{lookahead}s"] = 0
        return timeseries_rows, {
            "goals_in_clip": 0,
            "avg_pre_goal_vol_index_30s": 0.0,
            "avg_baseline_vol_index_30s": 0.0,
            "pre_goal_uplift_30s": 0.0,
        }

    pre_goal_values_30: list[float] = []
    baseline_values_30: list[float] = []

    for row in timeseries_rows:
        seconds = float(row["seconds"])
        future_goals = [goal for goal in goal_events if float(goal["seconds"]) >= seconds]
        if future_goals:
            next_goal = future_goals[0]
            seconds_to_next_goal = float(next_goal["seconds"]) - seconds
            next_goal_team = next_goal.get("team")
        else:
            seconds_to_next_goal = float("nan")
            next_goal_team = None

        row["seconds_to_next_goal"] = seconds_to_next_goal
        row["next_goal_team"] = next_goal_team or ""
        for lookahead in GOAL_LOOKAHEAD_SECONDS:
            row[f"goal_in_next_{lookahead}s"] = int(np.isfinite(seconds_to_next_goal) and 0.0 <= seconds_to_next_goal <= lookahead)

        vol_index = float(row["vol_index"]) if np.isfinite(row["vol_index"]) else float("nan")
        if np.isfinite(vol_index):
            if row["goal_in_next_30s"]:
                pre_goal_values_30.append(vol_index)
            else:
                baseline_values_30.append(vol_index)

    avg_pre_goal_volatility = float(np.mean(pre_goal_values_30)) if pre_goal_values_30 else 0.0
    avg_baseline_volatility = float(np.mean(baseline_values_30)) if baseline_values_30 else 0.0
    uplift = ((avg_pre_goal_volatility - avg_baseline_volatility) / avg_baseline_volatility) if avg_baseline_volatility > 0 else 0.0

    return timeseries_rows, {
        "goals_in_clip": len(goal_events),
        "avg_pre_goal_vol_index_30s": round(avg_pre_goal_volatility, 4),
        "avg_baseline_vol_index_30s": round(avg_baseline_volatility, 4),
        "pre_goal_uplift_30s": round(float(uplift), 4),
    }


def _compute_online_track_team_info(
    jersey_samples: list[np.ndarray],
    jersey_sample_track_ids: list[int],
) -> tuple[dict[int, dict[str, Any]], float]:
    if len(jersey_samples) < 8 or len(set(jersey_sample_track_ids)) < 2:
        return {}, 0.0

    feature_matrix = np.stack(jersey_samples, axis=0).astype(np.float32)
    centers, labels = kmeans_two_clusters(feature_matrix)
    cluster_distance = float(np.linalg.norm(centers[0] - centers[1]))
    cluster_order = np.lexsort((centers[:, 2], centers[:, 1], centers[:, 0]))
    cluster_to_team = {
        int(cluster_order[0]): "home",
        int(cluster_order[1]): "away",
    }

    votes_by_track: dict[int, list[str]] = defaultdict(list)
    features_by_track: dict[int, list[np.ndarray]] = defaultdict(list)
    for index, track_id in enumerate(jersey_sample_track_ids):
        team_label = cluster_to_team[int(labels[index])]
        votes_by_track[track_id].append(team_label)
        features_by_track[track_id].append(feature_matrix[index])

    team_info_by_track: dict[int, dict[str, Any]] = {}
    for track_id, votes in votes_by_track.items():
        vote_counter = Counter(votes)
        team_label, vote_count = vote_counter.most_common(1)[0]
        feature_stack = np.stack(features_by_track[track_id], axis=0).astype(np.float32)
        team_info_by_track[track_id] = {
            "team_label": team_label,
            "team_vote_ratio": vote_count / len(votes),
            "sample_count": len(votes),
            "average_color_rgb": [round(float(value), 4) for value in feature_stack.mean(axis=0)],
        }
    return team_info_by_track, cluster_distance


def detect_players_for_frame(
    frame: np.ndarray,
    player_model: YOLO,
    detector_spec: dict[str, Any],
    detector_device: str,
    player_conf: float,
    iou: float,
    frame_width: int,
    frame_height: int,
    field_homography: np.ndarray | None,
    frame_index: int,
    tracker_mode: str,
    player_tracker: HybridReIDTracker | None,
) -> list[dict[str, Any]]:
    detections: list[dict[str, Any]] = []
    if tracker_mode == LEGACY_PLAYER_TRACKER_MODE:
        player_results = player_model.track(
            source=frame,
            persist=True,
            tracker=DEFAULT_TRACKER,
            conf=player_conf,
            iou=iou,
            device=detector_device,
            classes=detector_spec["player_class_ids"],
            verbose=False,
        )
    else:
        player_results = player_model(
            source=frame,
            conf=player_conf,
            iou=iou,
            device=detector_device,
            classes=detector_spec["player_class_ids"],
            verbose=False,
        )
    player_boxes = player_results[0].boxes
    if player_boxes is None or len(player_boxes) == 0:
        return detections

    xyxy = player_boxes.xyxy.cpu().numpy()
    confidences = player_boxes.conf.cpu().numpy() if player_boxes.conf is not None else np.zeros(len(xyxy), dtype=np.float32)
    track_ids = (
        player_boxes.id.cpu().numpy().astype(int)
        if tracker_mode == LEGACY_PLAYER_TRACKER_MODE and player_boxes.id is not None
        else np.full(len(xyxy), -1, dtype=int)
    )
    bboxes: list[tuple[int, int, int, int]] = []
    anchors: list[tuple[float, float]] = []
    for index, box in enumerate(xyxy):
        x1, y1, x2, y2 = clamp_box(box, frame_width, frame_height)
        anchor = (float((x1 + x2) / 2.0), float(y2))
        bboxes.append((x1, y1, x2, y2))
        anchors.append(anchor)

    for index, bbox in enumerate(bboxes):
        detections.append(
            {
                "track_id": int(track_ids[index]),
                "confidence": float(confidences[index]),
                "bbox": bbox,
                "anchor": anchors[index],
                "field_point": project_point(anchors[index], field_homography),
                "identity_feature": None,
            }
        )
    if tracker_mode != LEGACY_PLAYER_TRACKER_MODE and player_tracker is not None:
        assigned_track_ids = player_tracker.update(frame, detections, frame_index)
        for detection, track_id in zip(detections, assigned_track_ids):
            detection["track_id"] = int(track_id)
    return detections


def generate_live_preview_stream(source_video_path: Path, config_payload: dict[str, Any]):
    detector_spec = resolve_detector_spec(str(config_payload.get("player_model") or DETECTOR_MODEL_DEFAULT))
    include_ball = bool(config_payload["include_ball"])
    player_conf = float(config_payload["player_conf"])
    ball_conf = float(config_payload["ball_conf"])
    iou = float(config_payload["iou"])
    tracker_mode = resolve_player_tracker_mode(config_payload)

    detector_device = choose_device()
    keypoint_device = choose_keypoint_device(detector_device)
    player_model = YOLO(detector_spec["weights_path"])
    ball_model: YOLO | None = YOLO(detector_spec["weights_path"]) if include_ball else None
    keypoint_model = YOLO(resolve_model_path(KEYPOINT_MODEL_DEFAULT, "keypoint"))

    cap = cv2.VideoCapture(str(source_video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {source_video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0
    frame_width = safe_int(cap.get(cv2.CAP_PROP_FRAME_WIDTH), 1280)
    frame_height = safe_int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT), 720)
    style = build_overlay_style(frame_width, frame_height)
    minimap_width = style["minimap_width"]
    minimap_height = style["minimap_height"]
    player_tracker = (
        HybridReIDTracker(
            fps=fps,
            frame_size=(frame_width, frame_height),
            detection_confidence_floor=player_conf,
            device=detector_device,
        )
        if tracker_mode != LEGACY_PLAYER_TRACKER_MODE
        else None
    )
    tracker_runtime = player_tracker.describe_backend() if player_tracker is not None else default_tracker_backend()
    tracker_status_label = tracker_mode_label(tracker_mode)
    field_homography: np.ndarray | None = None
    accepted_homographies: deque[np.ndarray] = deque(maxlen=CALIBRATION_SMOOTHING_WINDOW)
    latest_field_keypoints: np.ndarray | None = None
    latest_visible_keypoints = 0
    latest_inlier_count = 0
    last_calibration_frame = -1
    latest_reprojection_error_cm = float("inf")

    jersey_samples: list[np.ndarray] = []
    jersey_sample_track_ids: list[int] = []
    team_info_by_track: dict[int, dict[str, Any]] = {}
    frame_index = 0

    try:
        while True:
            frame_started = time.perf_counter()
            ok, frame = cap.read()
            if not ok:
                break

            if frame_index % CALIBRATION_REFRESH_FRAMES == 0 or field_homography is None:
                homography_candidate, detected_keypoints, visible_count, inlier_count, reprojection_error = detect_pitch_homography(frame, keypoint_model, keypoint_device)
                calibration_was_stale = calibration_stale_for_frame(field_homography, last_calibration_frame, frame_index)
                stale_recovery_mode = calibration_was_stale and field_homography is not None
                min_visible_count = calibration_visible_keypoint_minimum(stale_recovery_mode)
                min_inlier_count = calibration_inlier_minimum(stale_recovery_mode)
                reprojection_limit_cm = calibration_reprojection_limit_cm(stale_recovery_mode)
                temporal_drift_limit_cm = calibration_temporal_drift_limit_cm(stale_recovery_mode)
                temporal_drift = homography_temporal_drift_cm(field_homography, homography_candidate, frame_width, frame_height) if homography_candidate is not None else float("inf")
                candidate_is_usable = (
                    homography_candidate is not None
                    and visible_count >= min_visible_count
                    and inlier_count >= min_inlier_count
                    and reprojection_error <= reprojection_limit_cm
                    and temporal_drift <= temporal_drift_limit_cm
                )
                if candidate_is_usable:
                    normalized_candidate = normalize_homography_matrix(homography_candidate)
                    if normalized_candidate is not None:
                        if stale_recovery_mode:
                            accepted_homographies.clear()
                            accepted_homographies.append(normalized_candidate)
                            field_homography = normalized_candidate
                        else:
                            accepted_homographies.append(normalized_candidate)
                            smoothed_homography = smooth_homography_history(accepted_homographies)
                            field_homography = smoothed_homography if smoothed_homography is not None else normalized_candidate
                    last_calibration_frame = frame_index
                    latest_reprojection_error_cm = reprojection_error
                latest_field_keypoints = detected_keypoints
                latest_visible_keypoints = visible_count
                latest_inlier_count = inlier_count
            calibration_is_stale = calibration_stale_for_frame(field_homography, last_calibration_frame, frame_index)
            projection_homography = None if calibration_is_stale else field_homography

            current_players: list[dict[str, Any]] = []
            current_ball: dict[str, Any] | None = None

            player_detections = detect_players_for_frame(
                frame=frame,
                player_model=player_model,
                detector_spec=detector_spec,
                detector_device=detector_device,
                player_conf=player_conf,
                iou=iou,
                frame_width=frame_width,
                frame_height=frame_height,
                field_homography=projection_homography,
                frame_index=frame_index,
                tracker_mode=tracker_mode,
                player_tracker=player_tracker,
            )
            for detection in player_detections:
                x1, y1, x2, y2 = detection["bbox"]
                track_id = int(detection["track_id"])
                feature = extract_jersey_feature(frame, (x1, y1, x2, y2))
                if feature is not None and track_id >= 0:
                    jersey_samples.append(feature)
                    jersey_sample_track_ids.append(track_id)

                current_players.append(
                    {
                        "track_id": track_id,
                        "confidence": float(detection["confidence"]),
                        "bbox": (x1, y1, x2, y2),
                        "anchor": detection["anchor"],
                        "team_label": "unassigned",
                        "team_vote_ratio": 0.0,
                        "pitch_point": None,
                    }
                )

            if include_ball and ball_model is not None:
                ball_results = ball_model.track(
                    source=frame,
                    persist=True,
                    tracker=DEFAULT_TRACKER,
                    conf=ball_conf,
                    iou=iou,
                    device=detector_device,
                    classes=detector_spec["ball_class_ids"],
                    verbose=False,
                )
                ball_boxes = ball_results[0].boxes
                if ball_boxes is not None and len(ball_boxes) > 0:
                    b_xyxy = ball_boxes.xyxy.cpu().numpy()
                    b_confidences = ball_boxes.conf.cpu().numpy() if ball_boxes.conf is not None else np.zeros(len(b_xyxy), dtype=np.float32)
                    b_track_ids = ball_boxes.id.cpu().numpy().astype(int) if ball_boxes.id is not None else np.full(len(b_xyxy), -1, dtype=int)
                    best_index = int(np.argmax(b_confidences))
                    x1, y1, x2, y2 = clamp_box(b_xyxy[best_index], frame_width, frame_height)
                    current_ball = {
                        "track_id": int(b_track_ids[best_index]),
                        "confidence": float(b_confidences[best_index]),
                        "bbox": (x1, y1, x2, y2),
                        "anchor": (float((x1 + x2) / 2.0), float((y1 + y2) / 2.0)),
                        "pitch_point": None,
                    }

            if frame_index % 5 == 0:
                team_info_by_track, _ = _compute_online_track_team_info(jersey_samples, jersey_sample_track_ids)

            annotated = frame.copy()
            draw_detected_field_keypoints(annotated, latest_field_keypoints, style)

            for player_row in current_players:
                team_info = team_info_by_track.get(player_row["track_id"], default_team_info())
                player_row["team_label"] = team_info["team_label"]
                player_row["team_vote_ratio"] = float(team_info["team_vote_ratio"])
                player_row["field_point"] = project_point(player_row["anchor"], projection_homography)
                player_row["pitch_point"] = field_point_to_minimap(player_row["field_point"], minimap_width, minimap_height)
                x1, y1, x2, y2 = player_row["bbox"]
                color = TEAM_BOX_COLORS.get(player_row["team_label"], TEAM_BOX_COLORS["unassigned"])
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, style["box_thickness"])
                label = f"{player_row['team_label']} #{player_row['track_id']}" if player_row["track_id"] >= 0 else player_row["team_label"]
                draw_label(
                    annotated,
                    label,
                    x1,
                    y1,
                    (255, 255, 255),
                    scale=style["label_scale"],
                    thickness=style["text_thickness"],
                    padding_x=style["label_padding_x"],
                    padding_y=style["label_padding_y"],
                )

            if current_ball is not None:
                current_ball["field_point"] = project_point(current_ball["anchor"], projection_homography)
                current_ball["pitch_point"] = field_point_to_minimap(current_ball["field_point"], minimap_width, minimap_height)
                x1, y1, x2, y2 = current_ball["bbox"]
                cv2.rectangle(annotated, (x1, y1), (x2, y2), TEAM_BOX_COLORS["ball"], style["box_thickness"])
                ball_label = f"ball #{current_ball['track_id']}" if current_ball["track_id"] >= 0 else "ball"
                draw_label(
                    annotated,
                    ball_label,
                    x1,
                    y1,
                    (255, 245, 200),
                    scale=style["label_scale"],
                    thickness=style["text_thickness"],
                    padding_x=style["label_padding_x"],
                    padding_y=style["label_padding_y"],
                )

            text_x = style["margin"]
            first_line_y = style["margin"] + style["line_gap"]
            cv2.putText(
                annotated,
                f"live preview frame {frame_index + 1}",
                (text_x, first_line_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                style["status_scale"],
                (255, 255, 255),
                style["text_thickness"],
                cv2.LINE_AA,
            )
            status_text = (
                f"field calib {latest_visible_keypoints} kp / {latest_inlier_count} inliers"
                if field_homography is not None and not calibration_is_stale
                else (
                    f"field calib stale ({latest_visible_keypoints} kp / {latest_inlier_count} inliers) · minimap paused"
                    if field_homography is not None
                    else f"field calib waiting ({latest_visible_keypoints} kp)"
                )
            )
            cv2.putText(
                annotated,
                status_text,
                (text_x, first_line_y + style["line_gap"]),
                cv2.FONT_HERSHEY_SIMPLEX,
                style["small_status_scale"],
                (220, 238, 248),
                style["text_thickness"],
                cv2.LINE_AA,
            )
            cv2.putText(
                annotated,
                (
                    f"tracker {tracker_status_label} · refresh every {CALIBRATION_REFRESH_FRAMES} · stale since {last_calibration_frame if last_calibration_frame >= 0 else 'none'}"
                    if calibration_is_stale
                    else (
                        f"tracker {tracker_status_label} · refresh every {CALIBRATION_REFRESH_FRAMES} · last good {last_calibration_frame if last_calibration_frame >= 0 else 'none'} · reproj {latest_reprojection_error_cm:.0f}cm"
                        if np.isfinite(latest_reprojection_error_cm)
                        else f"tracker {tracker_status_label} · refresh every {CALIBRATION_REFRESH_FRAMES} · last good {last_calibration_frame if last_calibration_frame >= 0 else 'none'}"
                    )
                ),
                (text_x, first_line_y + style["line_gap"] * 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                style["small_status_scale"],
                (220, 238, 248),
                style["text_thickness"],
                cv2.LINE_AA,
            )
            if player_tracker is not None:
                cv2.putText(
                    annotated,
                    f"id source {tracker_runtime.get('embedding_source', 'hsv_hist_only')}",
                    (text_x, first_line_y + style["line_gap"] * 3),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    style["small_status_scale"],
                    (220, 238, 248),
                    style["text_thickness"],
                    cv2.LINE_AA,
                )
            if calibration_is_stale and field_homography is not None:
                draw_status_banner(annotated, "STALE CALIBRATION - MINIMAP PAUSED", style)

            if projection_homography is not None:
                minimap = create_pitch_map(minimap_width, minimap_height)
                for player_row in current_players:
                    if player_row["pitch_point"] is None:
                        continue
                    point = (int(round(player_row["pitch_point"][0])), int(round(player_row["pitch_point"][1])))
                    color = TEAM_BOX_COLORS.get(player_row["team_label"], TEAM_BOX_COLORS["unassigned"])
                    cv2.circle(minimap, point, style["minimap_point_radius"], color, -1)
                if current_ball is not None and current_ball["pitch_point"] is not None:
                    point = (int(round(current_ball["pitch_point"][0])), int(round(current_ball["pitch_point"][1])))
                    cv2.circle(minimap, point, style["minimap_ball_radius"], TEAM_BOX_COLORS["ball"], -1)
                overlay_minimap(annotated, minimap, style)

            ok, encoded = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
            if not ok:
                continue
            payload = encoded.tobytes()
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"Cache-Control: no-cache\r\n\r\n" + payload + b"\r\n"
            )

            frame_index += 1
            elapsed = time.perf_counter() - frame_started
            frame_delay = max(0.0, (1.0 / fps) - elapsed)
            if frame_delay > 0:
                time.sleep(min(frame_delay, 0.25))
    finally:
        cap.release()


def analyze_video(job_id: str, run_dir: Path, config_payload: dict[str, Any], job_manager: Any, job_control: Any | None = None) -> dict[str, Any]:
    source_video_path = Path(config_payload["source_video_path"])
    label_path = str(config_payload.get("label_path") or "").strip()
    player_model_name = str(config_payload["player_model"])
    requested_tracker_mode = requested_player_tracker_mode(config_payload)
    tracker_mode = resolve_player_tracker_mode(config_payload)
    tracker_mode_name = tracker_mode_label(tracker_mode)
    tracker_runtime_label = resolved_tracker_runtime_label(tracker_mode)
    include_ball = bool(config_payload["include_ball"])
    player_conf = float(config_payload["player_conf"])
    ball_conf = float(config_payload["ball_conf"])
    iou = float(config_payload["iou"])
    homography_points = config_payload.get("homography_points")

    outputs_dir = run_dir / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    overlay_video_path = outputs_dir / "overlay.mp4"
    raw_overlay_video_path = outputs_dir / "overlay_raw.mp4"
    detections_csv_path = outputs_dir / "detections.csv"
    track_summary_csv_path = outputs_dir / "track_summary.csv"
    projection_csv_path = outputs_dir / "projections.csv"
    calibration_debug_csv_path = outputs_dir / "calibration_debug.csv"
    entropy_timeseries_csv_path = outputs_dir / "entropy_timeseries.csv"
    goal_events_csv_path = outputs_dir / "goal_events.csv"
    summary_json_path = outputs_dir / "summary.json"
    full_outputs_zip_path = outputs_dir / "all_outputs.zip"

    job_manager.log(job_id, f"Opening video: {source_video_path}")
    detector_device = choose_device()
    keypoint_device = choose_keypoint_device(detector_device)
    job_manager.log(job_id, f"Detector device chosen: {detector_device}")
    job_manager.log(job_id, f"Field calibration device chosen: {keypoint_device}")

    cap = cv2.VideoCapture(str(source_video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {source_video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0
    frame_width = safe_int(cap.get(cv2.CAP_PROP_FRAME_WIDTH), 1280)
    frame_height = safe_int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT), 720)
    total_frames = safe_int(cap.get(cv2.CAP_PROP_FRAME_COUNT), 0)

    style = build_overlay_style(frame_width, frame_height)
    minimap_width = style["minimap_width"]
    minimap_height = style["minimap_height"]
    manual_homography_matrix, _ = compute_homography_matrix(homography_points, minimap_width, minimap_height)
    detector_spec = resolve_detector_spec(player_model_name)
    detector_path = detector_spec["weights_path"]
    keypoint_model_path = resolve_model_path(KEYPOINT_MODEL_DEFAULT, "keypoint")
    job_manager.log(job_id, f"Loading detector weights: {detector_spec['name']}")
    job_manager.log(job_id, f"Loading field calibration weights: {KEYPOINT_MODEL_DEFAULT}")
    keypoint_model = YOLO(keypoint_model_path)
    player_model = YOLO(detector_path)
    player_tracker = (
        HybridReIDTracker(
            fps=fps,
            frame_size=(frame_width, frame_height),
            detection_confidence_floor=player_conf,
            device=detector_device,
        )
        if tracker_mode != LEGACY_PLAYER_TRACKER_MODE
        else None
    )
    tracker_backend = player_tracker.describe_backend() if player_tracker is not None else default_tracker_backend()
    job_manager.log(
        job_id,
        f"Player tracker mode resolved to {tracker_mode_name} ({tracker_runtime_label})"
        + (f" from requested '{requested_tracker_mode}'" if requested_tracker_mode else " from default"),
    )
    job_manager.log(
        job_id,
        "Detector class ids: "
        f"players [{format_class_id_list(detector_spec['player_class_ids'], detector_spec['class_names'])}] · "
        f"ball [{format_class_id_list(detector_spec['ball_class_ids'], detector_spec['class_names'])}] · "
        f"referee [{format_class_id_list(detector_spec['referee_class_ids'], detector_spec['class_names'])}] "
        f"from {detector_spec['class_names_source']}",
    )
    if player_tracker is not None:
        job_manager.log(job_id, f"Identity embedding source: {tracker_backend['embedding_source']}")
        if tracker_backend.get("embedding_error"):
            job_manager.log(job_id, f"Identity embedding fallback active: {tracker_backend['embedding_error']}")
    ball_model: YOLO | None = None
    if include_ball:
        job_manager.log(job_id, "Ball tracking enabled through the shared football detector")
        ball_model = YOLO(detector_path)
    else:
        job_manager.log(job_id, "Ball stage disabled")

    frame_records: list[dict[str, Any]] = []
    player_rows_by_track: dict[int, list[dict[str, Any]]] = defaultdict(list)
    raw_player_track_ids_seen: set[int] = set()
    ball_track_ids_seen: set[int] = set()
    player_detections_per_frame: list[int] = []
    ball_detections_per_frame: list[int] = []
    jersey_samples: list[np.ndarray] = []
    jersey_sample_track_ids: list[int] = []
    calibration_visible_counts: list[int] = []
    calibration_inlier_counts: list[int] = []
    calibration_refresh_attempts = 0
    calibration_refresh_successes = 0
    calibration_refresh_rejections = 0
    calibration_rejections_by_reason = new_calibration_rejection_counts()
    calibration_primary_rejections_by_reason = {key: 0 for key in CALIBRATION_PRIMARY_REJECTION_REASON_KEYS}
    calibration_invalid_candidate_rejections = 0
    calibration_stale_recovery_attempts = 0
    calibration_stale_recovery_successes = 0
    calibration_stale_recovery_rejections = 0
    calibration_debug_rows: list[dict[str, Any]] = []
    frames_with_field_homography = 0
    frames_with_usable_homography = 0
    frames_with_nonstale_homography = 0
    frames_with_stale_homography = 0
    frames_projection_blocked_by_stale = 0
    frames_projected_with_last_known_homography = 0
    frames_with_player_anchors = 0
    frames_with_projected_points = 0
    frames_with_homography_but_no_player_anchors = 0
    player_rows_while_calibration_fresh = 0
    player_rows_while_calibration_stale = 0
    projected_player_points_fresh = 0
    projected_player_points_stale = 0
    projected_ball_points_fresh = 0
    projected_ball_points_stale = 0
    raw_detector_debug_sample_frames = 0
    raw_detector_boxes_sampled = 0
    raw_detector_class_histogram_sample: Counter[int] = Counter()
    field_registered_frames = 0
    last_good_calibration_frame = -1
    active_field_homography = manual_homography_matrix
    accepted_homographies: deque[np.ndarray] = deque(maxlen=CALIBRATION_SMOOTHING_WINDOW)
    latest_calibration_reprojection_error_cm = float("inf")
    frame_index = 0
    player_row_count = 0
    ball_row_count = 0

    start_time = datetime.utcnow().timestamp()
    while True:
        checkpoint_job_control(job_control, job_id, job_manager)
        ok, frame = cap.read()
        if not ok:
            break

        if raw_detector_debug_sample_frames < DETECTOR_DEBUG_SAMPLE_FRAMES:
            raw_box_count, raw_class_histogram = sample_detector_class_histogram(
                frame=frame,
                detector_model=player_model,
                detector_device=detector_device,
                confidence=player_conf,
                iou=iou,
            )
            raw_detector_debug_sample_frames += 1
            raw_detector_boxes_sampled += raw_box_count
            raw_detector_class_histogram_sample.update(raw_class_histogram)
            job_manager.log(
                job_id,
                f"Detector raw frame {frame_index}: boxes {raw_box_count} · "
                f"{format_class_histogram(raw_class_histogram, detector_spec['class_names'])}",
            )

        calibration_refresh_debug: dict[str, Any] | None = None
        if frame_index % CALIBRATION_REFRESH_FRAMES == 0 or active_field_homography is None:
            calibration_refresh_attempts += 1
            homography_candidate, _detected_keypoints, visible_count, inlier_count, reprojection_error = detect_pitch_homography(frame, keypoint_model, keypoint_device)
            calibration_visible_counts.append(visible_count)
            calibration_inlier_counts.append(inlier_count)
            calibration_was_stale = calibration_stale_for_frame(active_field_homography, last_good_calibration_frame, frame_index)
            stale_recovery_mode = calibration_was_stale and active_field_homography is not None
            if stale_recovery_mode:
                calibration_stale_recovery_attempts += 1
            min_visible_count = calibration_visible_keypoint_minimum(stale_recovery_mode)
            min_inlier_count = calibration_inlier_minimum(stale_recovery_mode)
            reprojection_limit_cm = calibration_reprojection_limit_cm(stale_recovery_mode)
            temporal_drift_limit_cm = calibration_temporal_drift_limit_cm(stale_recovery_mode)
            temporal_drift = homography_temporal_drift_cm(active_field_homography, homography_candidate, frame_width, frame_height) if homography_candidate is not None else float("inf")
            rejection_flags = calibration_rejection_flags(
                homography_candidate=homography_candidate,
                visible_count=visible_count,
                inlier_count=inlier_count,
                reprojection_error=reprojection_error,
                temporal_drift=temporal_drift,
                min_visible_count=min_visible_count,
                min_inlier_count=min_inlier_count,
                reprojection_limit_cm=reprojection_limit_cm,
                temporal_drift_limit_cm=temporal_drift_limit_cm,
            )
            candidate_is_usable = (
                homography_candidate is not None
                and visible_count >= min_visible_count
                and inlier_count >= min_inlier_count
                and reprojection_error <= reprojection_limit_cm
                and temporal_drift <= temporal_drift_limit_cm
            )
            candidate_accepted = False
            if candidate_is_usable:
                normalized_candidate = normalize_homography_matrix(homography_candidate)
                if normalized_candidate is not None:
                    if stale_recovery_mode:
                        accepted_homographies.clear()
                        accepted_homographies.append(normalized_candidate)
                        active_field_homography = normalized_candidate
                    else:
                        accepted_homographies.append(normalized_candidate)
                        smoothed_homography = smooth_homography_history(accepted_homographies)
                        active_field_homography = smoothed_homography if smoothed_homography is not None else normalized_candidate
                    calibration_refresh_successes += 1
                    if stale_recovery_mode:
                        calibration_stale_recovery_successes += 1
                    last_good_calibration_frame = frame_index
                    latest_calibration_reprojection_error_cm = reprojection_error
                    candidate_accepted = True
                else:
                    calibration_refresh_rejections += 1
                    calibration_primary_rejections_by_reason["invalid_candidate"] += 1
                    calibration_invalid_candidate_rejections += 1
                    if stale_recovery_mode:
                        calibration_stale_recovery_rejections += 1
            else:
                calibration_refresh_rejections += 1
                for rejection_flag in rejection_flags:
                    calibration_rejections_by_reason[rejection_flag] += 1
                calibration_primary_rejections_by_reason[primary_calibration_rejection_reason(rejection_flags)] += 1
                if stale_recovery_mode:
                    calibration_stale_recovery_rejections += 1
            calibration_refresh_debug = {
                "frame_index": frame_index,
                "candidate_exists": homography_candidate is not None,
                "visible_count": visible_count,
                "inlier_count": inlier_count,
                "reprojection_error": reprojection_error,
                "temporal_drift": temporal_drift,
                "min_visible_count": min_visible_count,
                "min_inlier_count": min_inlier_count,
                "reprojection_limit_cm": reprojection_limit_cm,
                "temporal_drift_limit_cm": temporal_drift_limit_cm,
                "candidate_is_usable": candidate_is_usable,
                "candidate_accepted": candidate_accepted,
                "stale_recovery_mode": stale_recovery_mode,
                "rejection_no_candidate": int("no_candidate" in rejection_flags),
                "rejection_low_visible_count": int("low_visible_count" in rejection_flags),
                "rejection_low_inliers": int("low_inliers" in rejection_flags),
                "rejection_high_reprojection_error": int("high_reprojection_error" in rejection_flags),
                "rejection_high_temporal_drift": int("high_temporal_drift" in rejection_flags),
                "rejection_invalid_candidate": int(candidate_is_usable and not candidate_accepted),
            }

        calibration_is_stale = calibration_stale_for_frame(active_field_homography, last_good_calibration_frame, frame_index)
        projection_homography = None if calibration_is_stale else active_field_homography
        if active_field_homography is not None:
            frames_with_field_homography += 1
            if calibration_is_stale:
                frames_with_stale_homography += 1
                frames_projection_blocked_by_stale += 1
            else:
                frames_with_usable_homography += 1
                frames_with_nonstale_homography += 1
        if calibration_refresh_debug is not None:
            primary_rejection_reason = (
                "accepted"
                if calibration_refresh_debug["candidate_accepted"]
                else primary_calibration_rejection_reason(rejection_flags)
            )
            reprojection_text = (
                f"{calibration_refresh_debug['reprojection_error']:.0f}cm"
                if np.isfinite(calibration_refresh_debug["reprojection_error"])
                else "inf"
            )
            drift_text = (
                f"{calibration_refresh_debug['temporal_drift']:.0f}cm"
                if np.isfinite(calibration_refresh_debug["temporal_drift"])
                else "inf"
            )
            job_manager.log(
                job_id,
                f"Calib refresh frame {calibration_refresh_debug['frame_index']}: "
                f"kp {calibration_refresh_debug['visible_count']} · "
                f"inliers {calibration_refresh_debug['inlier_count']} · "
                f"reproj {reprojection_text}/{calibration_refresh_debug['reprojection_limit_cm']:.0f}cm · "
                f"drift {drift_text}/{calibration_refresh_debug['temporal_drift_limit_cm']:.0f}cm · "
                f"recovery {int(calibration_refresh_debug['stale_recovery_mode'])} · "
                f"reason {primary_rejection_reason} · "
                f"usable {int(calibration_refresh_debug['candidate_is_usable'])} · "
                f"stale {int(calibration_is_stale)}",
            )
            calibration_debug_rows.append(
                {
                    "frame_index": int(calibration_refresh_debug["frame_index"]),
                    "candidate_exists": int(calibration_refresh_debug["candidate_exists"]),
                    "visible_count": int(calibration_refresh_debug["visible_count"]),
                    "visible_threshold": int(calibration_refresh_debug["min_visible_count"]),
                    "inlier_count": int(calibration_refresh_debug["inlier_count"]),
                    "inlier_threshold": int(calibration_refresh_debug["min_inlier_count"]),
                    "reprojection_error_cm": float(calibration_refresh_debug["reprojection_error"]),
                    "reprojection_threshold_cm": float(calibration_refresh_debug["reprojection_limit_cm"]),
                    "temporal_drift_cm": float(calibration_refresh_debug["temporal_drift"]),
                    "temporal_drift_threshold_cm": float(calibration_refresh_debug["temporal_drift_limit_cm"]),
                    "stale_recovery_mode": int(calibration_refresh_debug["stale_recovery_mode"]),
                    "candidate_usable": int(calibration_refresh_debug["candidate_is_usable"]),
                    "candidate_accepted": int(calibration_refresh_debug["candidate_accepted"]),
                    "rejection_reason_primary": primary_rejection_reason,
                    "rejection_no_candidate": int(calibration_refresh_debug["rejection_no_candidate"]),
                    "rejection_low_visible_count": int(calibration_refresh_debug["rejection_low_visible_count"]),
                    "rejection_low_inliers": int(calibration_refresh_debug["rejection_low_inliers"]),
                    "rejection_high_reprojection_error": int(calibration_refresh_debug["rejection_high_reprojection_error"]),
                    "rejection_high_temporal_drift": int(calibration_refresh_debug["rejection_high_temporal_drift"]),
                    "rejection_invalid_candidate": int(calibration_refresh_debug["rejection_invalid_candidate"]),
                    "calibration_stale_after_refresh": int(calibration_is_stale),
                    "last_good_calibration_frame_after_refresh": int(last_good_calibration_frame),
                }
            )

        player_detection_count = 0
        ball_detection_count = 0
        frame_players: list[dict[str, Any]] = []
        frame_ball: dict[str, Any] | None = None
        current_visible_keypoints = calibration_visible_counts[-1] if calibration_visible_counts else 0
        current_inlier_count = calibration_inlier_counts[-1] if calibration_inlier_counts else 0
        projected_player_in_frame = False

        player_detections = detect_players_for_frame(
            frame=frame,
            player_model=player_model,
            detector_spec=detector_spec,
            detector_device=detector_device,
            player_conf=player_conf,
            iou=iou,
            frame_width=frame_width,
            frame_height=frame_height,
            field_homography=projection_homography,
            frame_index=frame_index,
            tracker_mode=tracker_mode,
            player_tracker=player_tracker,
        )
        if player_detections:
            frames_with_player_anchors += 1
        elif projection_homography is not None:
            frames_with_homography_but_no_player_anchors += 1
        for detection in player_detections:
            x1, y1, x2, y2 = detection["bbox"]
            confidence = float(detection["confidence"])
            track_id = int(detection["track_id"])
            if calibration_is_stale:
                player_rows_while_calibration_stale += 1
            else:
                player_rows_while_calibration_fresh += 1
            color_feature = extract_jersey_feature(frame, (x1, y1, x2, y2))
            if color_feature is not None and track_id >= 0:
                jersey_samples.append(color_feature)
                jersey_sample_track_ids.append(track_id)

            pitch_point = field_point_to_minimap(detection["field_point"], minimap_width, minimap_height)
            row = {
                "frame_index": frame_index,
                "row_type": "player",
                "track_id": track_id,
                "class_name": "player",
                "confidence": confidence,
                "bbox": (x1, y1, x2, y2),
                "anchor": detection["anchor"],
                "feature": color_feature,
                "identity_feature": detection.get("identity_feature"),
                "team_label": "unassigned",
                "team_vote_ratio": 0.0,
                "field_point": detection["field_point"],
                "pitch_point": pitch_point,
            }
            frame_players.append(row)
            if track_id >= 0:
                player_rows_by_track[track_id].append(row)
                raw_player_track_ids_seen.add(track_id)
            if detection["field_point"] is not None:
                field_registered_frames += 1
                projected_player_in_frame = True
                if calibration_is_stale:
                    projected_player_points_stale += 1
                else:
                    projected_player_points_fresh += 1
            player_detection_count += 1
            player_row_count += 1
        if projected_player_in_frame:
            frames_with_projected_points += 1

        if include_ball and ball_model is not None:
            ball_results = ball_model.track(
                source=frame,
                persist=True,
                tracker=DEFAULT_TRACKER,
                conf=ball_conf,
                iou=iou,
                device=detector_device,
                classes=detector_spec["ball_class_ids"],
                verbose=False,
            )
            ball_boxes = ball_results[0].boxes
            if ball_boxes is not None and len(ball_boxes) > 0:
                b_xyxy = ball_boxes.xyxy.cpu().numpy()
                b_confidences = ball_boxes.conf.cpu().numpy() if ball_boxes.conf is not None else np.zeros(len(b_xyxy), dtype=np.float32)
                b_track_ids = ball_boxes.id.cpu().numpy().astype(int) if ball_boxes.id is not None else np.full(len(b_xyxy), -1, dtype=int)
                best_index = int(np.argmax(b_confidences))
                x1, y1, x2, y2 = clamp_box(b_xyxy[best_index], frame_width, frame_height)
                track_id = int(b_track_ids[best_index])
                confidence = float(b_confidences[best_index])
                anchor_x = float((x1 + x2) / 2.0)
                anchor_y = float((y1 + y2) / 2.0)
                field_point = project_point((anchor_x, anchor_y), projection_homography)
                pitch_point = field_point_to_minimap(field_point, minimap_width, minimap_height)
                frame_ball = {
                    "frame_index": frame_index,
                    "row_type": "ball",
                    "track_id": track_id,
                    "class_name": "ball",
                    "confidence": confidence,
                    "bbox": (x1, y1, x2, y2),
                    "anchor": (anchor_x, anchor_y),
                    "team_label": "",
                    "team_vote_ratio": 0.0,
                    "field_point": field_point,
                    "pitch_point": pitch_point,
                }
                if track_id >= 0:
                    ball_track_ids_seen.add(track_id)
                if field_point is not None:
                    if calibration_is_stale:
                        projected_ball_points_stale += 1
                    else:
                        projected_ball_points_fresh += 1
                ball_detection_count = 1
                ball_row_count += 1

        frame_records.append(
            {
                "players": frame_players,
                "ball": frame_ball,
                "field_calibration_active": active_field_homography is not None,
                "field_calibration_stale": calibration_is_stale,
                "field_projection_active": projection_homography is not None,
                "calibration_state": (
                    "waiting"
                    if active_field_homography is None
                    else ("stale" if calibration_is_stale else "fresh")
                ),
                "projection_state": (
                    "active"
                    if projection_homography is not None
                    else ("blocked_by_stale" if calibration_is_stale and active_field_homography is not None else "unavailable")
                ),
                "visible_pitch_keypoints": current_visible_keypoints,
                "homography_inliers": current_inlier_count,
                "last_good_calibration_frame": last_good_calibration_frame,
                "calibration_reprojection_error_cm": latest_calibration_reprojection_error_cm,
            }
        )
        player_detections_per_frame.append(player_detection_count)
        ball_detections_per_frame.append(ball_detection_count)
        frame_index += 1

        if total_frames > 0:
            progress = max(1.0, min(PROGRESS_TRACKING_MAX, (frame_index / total_frames) * PROGRESS_TRACKING_MAX))
            update_job_progress(job_manager, job_id, progress)
        if frame_index % 25 == 0:
            elapsed = datetime.utcnow().timestamp() - start_time
            fps_effective = frame_index / elapsed if elapsed > 0 else 0.0
            if active_field_homography is not None and calibration_is_stale:
                job_manager.log(
                    job_id,
                    f"Tracked {frame_index} frames at {fps_effective:.2f} fps · calib stale since {last_good_calibration_frame}",
                )
            elif np.isfinite(latest_calibration_reprojection_error_cm):
                job_manager.log(job_id, f"Tracked {frame_index} frames at {fps_effective:.2f} fps · calib reproj {latest_calibration_reprojection_error_cm:.0f}cm")
            else:
                job_manager.log(job_id, f"Tracked {frame_index} frames at {fps_effective:.2f} fps")

    cap.release()
    if frame_index == 0:
        raise RuntimeError("No frames were decoded from the source video.")

    raw_player_rows_by_track = player_rows_by_track
    raw_unique_player_track_ids = len(raw_player_track_ids_seen)
    raw_longest_track_length, raw_average_track_length = compute_track_length_stats(raw_player_rows_by_track)
    stitched_track_map: dict[int, int] = {}
    stitch_stats = {
        "merge_count": 0,
        "raw_track_count": raw_unique_player_track_ids,
        "stitched_track_count": raw_unique_player_track_ids,
        "max_gap_frames": 0,
    }
    if player_tracker is not None:
        raw_tracklets = player_tracker.export_tracklets()
        if not raw_tracklets:
            raw_tracklets = export_player_tracklets_from_rows(raw_player_rows_by_track)
        stitched_track_map, stitch_stats = build_stitched_track_map(raw_tracklets, fps=fps)
        if stitch_stats["merge_count"] > 0:
            job_manager.log(
                job_id,
                f"Stitched {stitch_stats['merge_count']} fragmented player tracklets into {stitch_stats['stitched_track_count']} canonical IDs",
            )
        else:
            job_manager.log(job_id, "No player tracklet merges passed the stitcher gates")
        tracker_backend = player_tracker.describe_backend()
    apply_player_track_id_map(frame_records, jersey_sample_track_ids, stitched_track_map)
    canonical_player_rows_by_track, player_track_ids_seen = rebuild_player_rows_by_track(frame_records)
    longest_track_length, average_track_length = compute_track_length_stats(canonical_player_rows_by_track)

    track_team_info, team_cluster_distance = _compute_online_track_team_info(jersey_samples, jersey_sample_track_ids)
    if track_team_info:
        job_manager.log(job_id, f"Built two team color clusters from {len(jersey_samples)} jersey crops")
    else:
        for track_id in canonical_player_rows_by_track:
            track_team_info[track_id] = default_team_info()
        job_manager.log(job_id, "Not enough jersey crops for reliable team clustering; leaving tracks unassigned")

    home_tracks = 0
    away_tracks = 0
    unassigned_tracks = 0
    projected_player_points = 0
    projected_ball_points = 0
    projection_rows: list[dict[str, Any]] = []

    for record in frame_records:
        checkpoint_job_control(job_control, job_id, job_manager)
        for player_row in record["players"]:
            team_info = track_team_info.get(player_row["track_id"], default_team_info())
            player_row["team_label"] = team_info["team_label"]
            player_row["team_vote_ratio"] = float(team_info["team_vote_ratio"])
            if player_row["field_point"] is not None and player_row["pitch_point"] is not None:
                projected_player_points += 1
                projection_rows.append(
                    {
                        "frame_index": player_row["frame_index"],
                        "row_type": "player",
                        "track_id": player_row["track_id"],
                        "team_label": player_row["team_label"],
                        "field_x_cm": round(float(player_row["field_point"][0]), 4),
                        "field_y_cm": round(float(player_row["field_point"][1]), 4),
                        "map_x": round(float(player_row["pitch_point"][0]), 4),
                        "map_y": round(float(player_row["pitch_point"][1]), 4),
                        "source_x": round(float(player_row["anchor"][0]), 4),
                        "source_y": round(float(player_row["anchor"][1]), 4),
                    }
                )

        ball_row = record["ball"]
        if ball_row is not None and ball_row["field_point"] is not None and ball_row["pitch_point"] is not None:
            projected_ball_points += 1
            projection_rows.append(
                {
                    "frame_index": ball_row["frame_index"],
                    "row_type": "ball",
                    "track_id": ball_row["track_id"],
                    "team_label": "",
                    "field_x_cm": round(float(ball_row["field_point"][0]), 4),
                    "field_y_cm": round(float(ball_row["field_point"][1]), 4),
                    "map_x": round(float(ball_row["pitch_point"][0]), 4),
                    "map_y": round(float(ball_row["pitch_point"][1]), 4),
                    "source_x": round(float(ball_row["anchor"][0]), 4),
                    "source_y": round(float(ball_row["anchor"][1]), 4),
                }
            )

    for info in track_team_info.values():
        if info["team_label"] == "home":
            home_tracks += 1
        elif info["team_label"] == "away":
            away_tracks += 1
        else:
            unassigned_tracks += 1

    detection_headers = [
        "frame_index",
        "row_type",
        "track_id",
        "class_name",
        "team_label",
        "team_vote_ratio",
        "confidence",
        "x1",
        "y1",
        "x2",
        "y2",
        "anchor_x",
        "anchor_y",
        "field_x_cm",
        "field_y_cm",
        "map_x",
        "map_y",
        "calibration_state",
        "projection_state",
        "calibration_visible_keypoints",
        "calibration_inliers",
        "color_r",
        "color_g",
        "color_b",
    ]
    with detections_csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(detection_headers)
        for record in frame_records:
            for player_row in record["players"]:
                x1, y1, x2, y2 = player_row["bbox"]
                feature = player_row["feature"]
                field_point = player_row["field_point"]
                pitch_point = player_row["pitch_point"]
                csv_writer.writerow(
                    [
                        player_row["frame_index"],
                        player_row["row_type"],
                        player_row["track_id"],
                        player_row["class_name"],
                        player_row["team_label"],
                        round(float(player_row["team_vote_ratio"]), 4),
                        round(float(player_row["confidence"]), 5),
                        x1,
                        y1,
                        x2,
                        y2,
                        round(float(player_row["anchor"][0]), 4),
                        round(float(player_row["anchor"][1]), 4),
                        round(float(field_point[0]), 4) if field_point is not None else "",
                        round(float(field_point[1]), 4) if field_point is not None else "",
                        round(float(pitch_point[0]), 4) if pitch_point is not None else "",
                        round(float(pitch_point[1]), 4) if pitch_point is not None else "",
                        record["calibration_state"],
                        record["projection_state"],
                        record["visible_pitch_keypoints"],
                        record["homography_inliers"],
                        round(float(feature[0]), 5) if feature is not None else "",
                        round(float(feature[1]), 5) if feature is not None else "",
                        round(float(feature[2]), 5) if feature is not None else "",
                    ]
                )

            ball_row = record["ball"]
            if ball_row is not None:
                x1, y1, x2, y2 = ball_row["bbox"]
                field_point = ball_row["field_point"]
                pitch_point = ball_row["pitch_point"]
                csv_writer.writerow(
                    [
                        ball_row["frame_index"],
                        ball_row["row_type"],
                        ball_row["track_id"],
                        ball_row["class_name"],
                        "",
                        "",
                        round(float(ball_row["confidence"]), 5),
                        x1,
                        y1,
                        x2,
                        y2,
                        round(float(ball_row["anchor"][0]), 4),
                        round(float(ball_row["anchor"][1]), 4),
                        round(float(field_point[0]), 4) if field_point is not None else "",
                        round(float(field_point[1]), 4) if field_point is not None else "",
                        round(float(pitch_point[0]), 4) if pitch_point is not None else "",
                        round(float(pitch_point[1]), 4) if pitch_point is not None else "",
                        record["calibration_state"],
                        record["projection_state"],
                        record["visible_pitch_keypoints"],
                        record["homography_inliers"],
                        "",
                        "",
                        "",
                    ]
                )

    projection_csv_key: str | None = None
    if projection_rows:
        with projection_csv_path.open("w", newline="", encoding="utf-8") as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(["frame_index", "row_type", "track_id", "team_label", "field_x_cm", "field_y_cm", "map_x", "map_y", "source_x", "source_y"])
            for row in projection_rows:
                checkpoint_job_control(job_control, job_id, job_manager)
                csv_writer.writerow(
                    [
                        row["frame_index"],
                        row["row_type"],
                        row["track_id"],
                        row["team_label"],
                        row["field_x_cm"],
                        row["field_y_cm"],
                        row["map_x"],
                        row["map_y"],
                        row["source_x"],
                        row["source_y"],
                    ]
                )
        projection_csv_key = f"/runs/{run_dir.name}/outputs/projections.csv"

    calibration_debug_csv_key: str | None = None
    if calibration_debug_rows:
        with calibration_debug_csv_path.open("w", newline="", encoding="utf-8") as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(
                [
                    "frame_index",
                    "candidate_exists",
                    "visible_count",
                    "visible_threshold",
                    "inlier_count",
                    "inlier_threshold",
                    "reprojection_error_cm",
                    "reprojection_threshold_cm",
                    "temporal_drift_cm",
                    "temporal_drift_threshold_cm",
                    "stale_recovery_mode",
                    "candidate_usable",
                    "candidate_accepted",
                    "rejection_reason_primary",
                    "rejection_no_candidate",
                    "rejection_low_visible_count",
                    "rejection_low_inliers",
                    "rejection_high_reprojection_error",
                    "rejection_high_temporal_drift",
                    "rejection_invalid_candidate",
                    "calibration_stale_after_refresh",
                    "last_good_calibration_frame_after_refresh",
                ]
            )
            for row in calibration_debug_rows:
                checkpoint_job_control(job_control, job_id, job_manager)
                csv_writer.writerow(
                    [
                        row["frame_index"],
                        row["candidate_exists"],
                        row["visible_count"],
                        row["visible_threshold"],
                        row["inlier_count"],
                        row["inlier_threshold"],
                        round(float(row["reprojection_error_cm"]), 4) if np.isfinite(row["reprojection_error_cm"]) else "",
                        round(float(row["reprojection_threshold_cm"]), 4),
                        round(float(row["temporal_drift_cm"]), 4) if np.isfinite(row["temporal_drift_cm"]) else "",
                        round(float(row["temporal_drift_threshold_cm"]), 4),
                        row["stale_recovery_mode"],
                        row["candidate_usable"],
                        row["candidate_accepted"],
                        row["rejection_reason_primary"],
                        row["rejection_no_candidate"],
                        row["rejection_low_visible_count"],
                        row["rejection_low_inliers"],
                        row["rejection_high_reprojection_error"],
                        row["rejection_high_temporal_drift"],
                        row["rejection_invalid_candidate"],
                        row["calibration_stale_after_refresh"],
                        row["last_good_calibration_frame_after_refresh"],
                    ]
                )
        calibration_debug_csv_key = f"/runs/{run_dir.name}/outputs/calibration_debug.csv"

    entropy_timeseries_rows, experiment_card = build_geometric_volatility_experiment(frame_records, fps if fps > 0 else 25.0)
    goal_events, goal_label_source = load_soccernet_goal_events(source_video_path, label_path)
    entropy_timeseries_rows, goal_metrics = attach_goal_targets(entropy_timeseries_rows, goal_events)
    experiment_card["metrics"].extend(
        [
            {
                "label": "Goals in clip",
                "value": goal_metrics["goals_in_clip"],
                "hint": "Count of SoccerNet goal annotations aligned to this half.",
            },
            {
                "label": "30s pre-goal index",
                "value": goal_metrics["avg_pre_goal_vol_index_30s"],
                "hint": "Average combined volatility index in rows that precede a goal within 30 seconds.",
            },
            {
                "label": "30s baseline index",
                "value": goal_metrics["avg_baseline_vol_index_30s"],
                "hint": "Average combined volatility index outside those 30-second pre-goal windows.",
            },
            {
                "label": "30s uplift",
                "value": goal_metrics["pre_goal_uplift_30s"],
                "hint": "Relative lift of the volatility index in pre-goal windows versus baseline.",
            },
        ]
    )
    with entropy_timeseries_csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([
            "second",
            "seconds",
            "home_player_count",
            "away_player_count",
            "home_centroid_x_cm",
            "home_centroid_y_cm",
            "away_centroid_x_cm",
            "away_centroid_y_cm",
            "home_spread_rms_cm",
            "away_spread_rms_cm",
            "home_length_axis_cm",
            "away_length_axis_cm",
            "home_width_axis_cm",
            "away_width_axis_cm",
            "home_hull_area_cm2",
            "away_hull_area_cm2",
            "centroid_distance_cm",
            "entropy_grid",
            "home_spread_rms_cm_volatility",
            "away_spread_rms_cm_volatility",
            "home_length_axis_cm_volatility",
            "away_length_axis_cm_volatility",
            "centroid_distance_cm_volatility",
            "entropy_grid_volatility",
            "vol_index",
            "seconds_to_next_goal",
            "next_goal_team",
            "goal_in_next_30s",
            "goal_in_next_60s",
        ])
        for row in entropy_timeseries_rows:
            checkpoint_job_control(job_control, job_id, job_manager)
            csv_writer.writerow([
                row["second"],
                row["seconds"],
                row["home_player_count"],
                row["away_player_count"],
                round(float(row["home_centroid_x_cm"]), 4) if np.isfinite(row["home_centroid_x_cm"]) else "",
                round(float(row["home_centroid_y_cm"]), 4) if np.isfinite(row["home_centroid_y_cm"]) else "",
                round(float(row["away_centroid_x_cm"]), 4) if np.isfinite(row["away_centroid_x_cm"]) else "",
                round(float(row["away_centroid_y_cm"]), 4) if np.isfinite(row["away_centroid_y_cm"]) else "",
                round(float(row["home_spread_rms_cm"]), 4) if np.isfinite(row["home_spread_rms_cm"]) else "",
                round(float(row["away_spread_rms_cm"]), 4) if np.isfinite(row["away_spread_rms_cm"]) else "",
                round(float(row["home_length_axis_cm"]), 4) if np.isfinite(row["home_length_axis_cm"]) else "",
                round(float(row["away_length_axis_cm"]), 4) if np.isfinite(row["away_length_axis_cm"]) else "",
                round(float(row["home_width_axis_cm"]), 4) if np.isfinite(row["home_width_axis_cm"]) else "",
                round(float(row["away_width_axis_cm"]), 4) if np.isfinite(row["away_width_axis_cm"]) else "",
                round(float(row["home_hull_area_cm2"]), 4) if np.isfinite(row["home_hull_area_cm2"]) else "",
                round(float(row["away_hull_area_cm2"]), 4) if np.isfinite(row["away_hull_area_cm2"]) else "",
                round(float(row["centroid_distance_cm"]), 4) if np.isfinite(row["centroid_distance_cm"]) else "",
                round(float(row["entropy_grid"]), 6) if np.isfinite(row["entropy_grid"]) else "",
                round(float(row["home_spread_rms_cm_volatility"]), 6) if np.isfinite(row["home_spread_rms_cm_volatility"]) else "",
                round(float(row["away_spread_rms_cm_volatility"]), 6) if np.isfinite(row["away_spread_rms_cm_volatility"]) else "",
                round(float(row["home_length_axis_cm_volatility"]), 6) if np.isfinite(row["home_length_axis_cm_volatility"]) else "",
                round(float(row["away_length_axis_cm_volatility"]), 6) if np.isfinite(row["away_length_axis_cm_volatility"]) else "",
                round(float(row["centroid_distance_cm_volatility"]), 6) if np.isfinite(row["centroid_distance_cm_volatility"]) else "",
                round(float(row["entropy_grid_volatility"]), 6) if np.isfinite(row["entropy_grid_volatility"]) else "",
                round(float(row["vol_index"]), 6) if np.isfinite(row["vol_index"]) else "",
                round(float(row["seconds_to_next_goal"]), 4) if np.isfinite(row["seconds_to_next_goal"]) else "",
                row["next_goal_team"],
                row["goal_in_next_30s"],
                row["goal_in_next_60s"],
            ])
    entropy_timeseries_csv_key = f"/runs/{run_dir.name}/outputs/entropy_timeseries.csv"

    goal_events_csv_key: str | None = None
    if goal_events:
        with goal_events_csv_path.open("w", newline="", encoding="utf-8") as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(["half", "seconds", "position_ms", "game_clock_seconds", "team", "visibility", "label"])
            for event in goal_events:
                checkpoint_job_control(job_control, job_id, job_manager)
                csv_writer.writerow([
                    event["half"],
                    event["seconds"],
                    event["position_ms"],
                    event["game_clock_seconds"],
                    event["team"],
                    event["visibility"],
                    event["label"],
                ])
        goal_events_csv_key = f"/runs/{run_dir.name}/outputs/goal_events.csv"

    longest_track_length = 0
    average_track_length = 0.0
    track_rows: list[dict[str, Any]] = []
    track_lengths: list[int] = []
    with track_summary_csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(
            [
                "track_id",
                "team_label",
                "team_vote_ratio",
                "frames",
                "first_frame",
                "last_frame",
                "average_confidence",
                "average_bbox_area",
                "projected_points",
                "sampled_color_rgb",
            ]
        )

        for track_id, rows in sorted(canonical_player_rows_by_track.items(), key=lambda item: len(item[1]), reverse=True):
            checkpoint_job_control(job_control, job_id, job_manager)
            track_length = len(rows)
            track_lengths.append(track_length)
            longest_track_length = max(longest_track_length, track_length)
            frame_indices = [int(row["frame_index"]) for row in rows]
            confidences = [float(row["confidence"]) for row in rows]
            bbox_areas = [
                float((row["bbox"][2] - row["bbox"][0]) * (row["bbox"][3] - row["bbox"][1]))
                for row in rows
            ]
            projected_points = sum(1 for row in rows if row["field_point"] is not None)
            team_info = track_team_info.get(track_id, default_team_info())
            color_text = ",".join(f"{value:.4f}" for value in team_info["average_color_rgb"])
            csv_writer.writerow(
                [
                    track_id,
                    team_info["team_label"],
                    round(float(team_info["team_vote_ratio"]), 4),
                    track_length,
                    min(frame_indices),
                    max(frame_indices),
                    round(float(np.mean(confidences)), 4),
                    round(float(np.mean(bbox_areas)), 4),
                    projected_points,
                    color_text,
                ]
            )
            track_rows.append(
                {
                    "track_id": track_id,
                    "team_label": team_info["team_label"],
                    "team_vote_ratio": round(float(team_info["team_vote_ratio"]), 4),
                    "frames": track_length,
                    "first_frame": min(frame_indices),
                    "last_frame": max(frame_indices),
                    "average_confidence": round(float(np.mean(confidences)), 4),
                    "average_bbox_area": round(float(np.mean(bbox_areas)), 4),
                    "projected_points": projected_points,
                }
            )

    if track_lengths:
        average_track_length = float(np.mean(track_lengths))

    job_manager.log(job_id, "Rendering tactical overlay")
    update_job_progress(job_manager, job_id, PROGRESS_TRACKING_MAX + 1.0)
    render_cap = cv2.VideoCapture(str(source_video_path))
    if not render_cap.isOpened():
        raise RuntimeError(f"Could not reopen video for rendering: {source_video_path}")

    writer, overlay_write_path, finalize_overlay = create_overlay_writer(
        overlay_video_path=overlay_video_path,
        raw_overlay_video_path=raw_overlay_video_path,
        fps=fps,
        frame_size=(frame_width, frame_height),
        job_id=job_id,
        job_manager=job_manager,
    )

    for render_index, record in enumerate(frame_records):
        checkpoint_job_control(job_control, job_id, job_manager)
        ok, frame = render_cap.read()
        if not ok:
            break

        annotated = frame.copy()

        for player_row in record["players"]:
            x1, y1, x2, y2 = player_row["bbox"]
            team_label = str(player_row["team_label"])
            color = TEAM_BOX_COLORS.get(team_label, TEAM_BOX_COLORS["unassigned"])
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, style["box_thickness"])
            label = f"{team_label} #{player_row['track_id']}" if player_row["track_id"] >= 0 else team_label
            draw_label(
                annotated,
                label,
                x1,
                y1,
                (255, 255, 255),
                scale=style["label_scale"],
                thickness=style["text_thickness"],
                padding_x=style["label_padding_x"],
                padding_y=style["label_padding_y"],
            )

        ball_row = record["ball"]
        if ball_row is not None:
            x1, y1, x2, y2 = ball_row["bbox"]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), TEAM_BOX_COLORS["ball"], style["box_thickness"])
            ball_label = f"ball #{ball_row['track_id']}" if ball_row["track_id"] >= 0 else "ball"
            draw_label(
                annotated,
                ball_label,
                x1,
                y1,
                (255, 245, 200),
                scale=style["label_scale"],
                thickness=style["text_thickness"],
                padding_x=style["label_padding_x"],
                padding_y=style["label_padding_y"],
            )

        text_x = style["margin"]
        first_line_y = style["margin"] + style["line_gap"]
        status = f"frame {render_index + 1}"
        if total_frames > 0:
            status += f"/{total_frames}"
        cv2.putText(
            annotated,
            status,
            (text_x, first_line_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            style["status_scale"],
            (255, 255, 255),
            style["text_thickness"],
            cv2.LINE_AA,
        )
        cv2.putText(
            annotated,
            f"tracks raw {raw_unique_player_track_ids}  stitched {len(player_track_ids_seen)}  home {home_tracks}  away {away_tracks}",
            (text_x, first_line_y + style["line_gap"]),
            cv2.FONT_HERSHEY_SIMPLEX,
            style["small_status_scale"],
            (220, 238, 248),
            style["text_thickness"],
            cv2.LINE_AA,
        )
        field_status = (
            f"field calib {record['visible_pitch_keypoints']} kp / {record['homography_inliers']} inliers"
            if record.get("field_projection_active")
            else (
                f"field calib stale ({record['visible_pitch_keypoints']} kp / {record['homography_inliers']} inliers) · minimap paused"
                if record["field_calibration_active"]
                else f"field calib waiting ({record['visible_pitch_keypoints']} kp)"
            )
        )
        cv2.putText(
            annotated,
            field_status,
            (text_x, first_line_y + style["line_gap"] * 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            style["small_status_scale"],
            (220, 238, 248),
            style["text_thickness"],
            cv2.LINE_AA,
        )
        cv2.putText(
            annotated,
            (
                f"tracker {tracker_mode_label(tracker_mode)} · refresh every {CALIBRATION_REFRESH_FRAMES} · stale since {record['last_good_calibration_frame'] if record['last_good_calibration_frame'] >= 0 else 'none'}"
                if record.get("field_calibration_stale")
                else (
                    f"tracker {tracker_mode_label(tracker_mode)} · refresh every {CALIBRATION_REFRESH_FRAMES} · last good {record['last_good_calibration_frame'] if record['last_good_calibration_frame'] >= 0 else 'none'} · reproj {record['calibration_reprojection_error_cm']:.0f}cm"
                    if np.isfinite(record['calibration_reprojection_error_cm'])
                    else f"tracker {tracker_mode_label(tracker_mode)} · refresh every {CALIBRATION_REFRESH_FRAMES} · last good {record['last_good_calibration_frame'] if record['last_good_calibration_frame'] >= 0 else 'none'}"
                )
            ),
            (text_x, first_line_y + style["line_gap"] * 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            style["small_status_scale"],
            (220, 238, 248),
            style["text_thickness"],
            cv2.LINE_AA,
        )
        if player_tracker is not None:
            cv2.putText(
                annotated,
                f"id source {tracker_backend.get('embedding_source', 'hsv_hist_only')} · merges {stitch_stats['merge_count']}",
                (text_x, first_line_y + style["line_gap"] * 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                style["small_status_scale"],
                (220, 238, 248),
                style["text_thickness"],
                cv2.LINE_AA,
            )
        if record.get("field_calibration_stale") and record["field_calibration_active"]:
            draw_status_banner(annotated, "STALE CALIBRATION - MINIMAP PAUSED", style)

        if record.get("field_projection_active"):
            minimap = create_pitch_map(minimap_width, minimap_height)
            for player_row in record["players"]:
                if player_row["pitch_point"] is None:
                    continue
                team_label = str(player_row["team_label"])
                color = TEAM_BOX_COLORS.get(team_label, TEAM_BOX_COLORS["unassigned"])
                point = (int(round(player_row["pitch_point"][0])), int(round(player_row["pitch_point"][1])))
                cv2.circle(minimap, point, style["minimap_point_radius"], color, -1)

            if ball_row is not None and ball_row["pitch_point"] is not None:
                point = (int(round(ball_row["pitch_point"][0])), int(round(ball_row["pitch_point"][1])))
                cv2.circle(minimap, point, style["minimap_ball_radius"], TEAM_BOX_COLORS["ball"], -1)

            overlay_minimap(annotated, minimap, style)

        writer.write(annotated)

        if total_frames > 0:
            render_span = PROGRESS_RENDER_END - PROGRESS_TRACKING_MAX
            progress = PROGRESS_TRACKING_MAX + min(render_span, (render_index + 1) / total_frames * render_span)
            update_job_progress(job_manager, job_id, progress)

    render_cap.release()
    writer.release()
    update_job_progress(job_manager, job_id, PROGRESS_RENDER_END)
    if finalize_overlay:
        checkpoint_job_control(job_control, job_id, job_manager)
        job_manager.log(job_id, "Finalizing browser-ready overlay video")
        finalize_overlay_video(overlay_write_path, overlay_video_path, job_id, job_manager)
    else:
        job_manager.log(job_id, f"Overlay video written directly to {overlay_write_path.name}")
    update_job_progress(job_manager, job_id, PROGRESS_VIDEO_FINALIZE_END)

    diagnostics: list[dict[str, str]] = []
    churn_ratio = len(player_track_ids_seen) / max(frame_index, 1)
    raw_churn_ratio = raw_unique_player_track_ids / max(frame_index, 1)
    avg_players = float(np.mean(player_detections_per_frame)) if player_detections_per_frame else 0.0
    avg_ball = float(np.mean(ball_detections_per_frame)) if ball_detections_per_frame else 0.0
    avg_visible_pitch_keypoints = float(np.mean(calibration_visible_counts)) if calibration_visible_counts else 0.0
    registered_frame_ratio = field_registered_frames / max(player_row_count, 1)
    calibration_gate_summary = calibration_rejection_summary(
        calibration_rejections_by_reason,
        calibration_invalid_candidate_rejections,
    )
    field_calibration_success = calibration_success_rate(
        calibration_refresh_successes,
        calibration_refresh_attempts,
    )
    sampled_detector_histogram = format_class_histogram(
        {int(class_id): int(count) for class_id, count in raw_detector_class_histogram_sample.items()},
        detector_spec["class_names"],
    )
    assigned_vote_ratios = [
        float(team_info["team_vote_ratio"])
        for team_info in track_team_info.values()
        if str(team_info["team_label"]) in {"home", "away"}
    ]
    average_team_vote_ratio = float(np.mean(assigned_vote_ratios)) if assigned_vote_ratios else 0.0

    if player_row_count == 0:
        diagnostics.append(
            {
                "level": "warn",
                "title": "Detector produced no players",
                "message": (
                    f"Across {frame_index} frames, the run emitted 0 player rows and average player detections per frame stayed {avg_players:.2f}. "
                    f"Raw detector sampling saw {raw_detector_boxes_sampled} boxes across the first {raw_detector_debug_sample_frames} frames "
                    f"({sampled_detector_histogram}). That means the pipeline never reached usable tracking, team assignment, or player-space projection."
                ),
                "next_step": (
                    "Inspect the detector checkpoint, class mapping, and confidence thresholds first. "
                    "If this is a custom detector, verify the player class id matches the active model output before touching tracker settings."
                ),
            }
        )
    elif churn_ratio > 0.1:
        diagnostics.append(
            {
                "level": "warn",
                "title": "Tracking stability: fragmented",
                "message": (
                    f"{len(player_track_ids_seen)} stitched player track IDs across {frame_index} frames "
                    f"(raw {raw_unique_player_track_ids}, longest {longest_track_length}, mean {average_track_length:.1f})."
                ),
                "next_step": "Use a steadier wide-angle phase or inspect cutaways before trusting downstream team or pitch metrics.",
            }
        )
    else:
        diagnostics.append(
            {
                "level": "good",
                "title": "Tracking stability: acceptable",
                "message": (
                    f"{len(player_track_ids_seen)} stitched player track IDs across {frame_index} frames "
                    f"(raw {raw_unique_player_track_ids}, longest {longest_track_length}, mean {average_track_length:.1f})."
                ),
                "next_step": "Review the longest tracks through pans and occlusions to confirm IDs stay attached to the same players.",
            }
        )

    if player_tracker is not None:
        diagnostics.append(
            {
                "level": "good" if stitch_stats["merge_count"] > 0 else "warn",
                "title": "Identity stitching"
                if stitch_stats["merge_count"] > 0
                else "Identity stitching: no merges",
                "message": (
                    f"Player tracker ran in {tracker_mode_label(tracker_mode)} mode with "
                    f"{tracker_backend.get('embedding_source', 'hsv_hist_only')}. "
                    f"Raw track IDs dropped from {raw_unique_player_track_ids} to {len(player_track_ids_seen)} after "
                    f"{stitch_stats['merge_count']} accepted merges."
                ),
                "next_step": (
                    "Inspect long same-kit occlusions and restart phases to verify the stitched IDs stay on the same players."
                    if stitch_stats["merge_count"] > 0
                    else "If IDs still fragment, validate broadcast-cut segments and consider adding jersey-number evidence before using match-long identity."
                ),
            }
        )

    if home_tracks > 0 and away_tracks > 0 and team_cluster_distance > 0.08:
        diagnostics.append(
            {
                "level": "good",
                "title": "Team clustering: separated",
                "message": (
                    f"{home_tracks} home tracks and {away_tracks} away tracks from {len(jersey_samples)} jersey crops "
                    f"(cluster distance {team_cluster_distance:.3f}, mean vote ratio {average_team_vote_ratio:.2f})."
                ),
                "next_step": "Cross-check goalkeepers, referees, and edge cases before treating the home and away labels as clean.",
            }
        )
    else:
        diagnostics.append(
            {
                "level": "warn",
                "title": "Team clustering: weak separation",
                "message": (
                    f"{home_tracks} home tracks and {away_tracks} away tracks from {len(jersey_samples)} jersey crops "
                    f"(cluster distance {team_cluster_distance:.3f}, mean vote ratio {average_team_vote_ratio:.2f})."
                ),
                "next_step": "Use clips with better kit contrast or larger player crops before relying on team labels.",
            }
        )

    if include_ball:
        if ball_row_count == 0 or avg_ball < 0.15:
            diagnostics.append(
                {
                    "level": "warn",
                    "title": "Ball detections: sparse",
                    "message": f"Average ball detections per frame {avg_ball:.2f}; projected ball anchors {projected_ball_points}.",
                    "next_step": "Treat ball output as context only and inspect the overlay for missed frames before using it analytically.",
                }
            )
        else:
            diagnostics.append(
                {
                    "level": "good",
                    "title": "Ball detections: usable",
                    "message": f"Average ball detections per frame {avg_ball:.2f}; projected ball anchors {projected_ball_points}.",
                    "next_step": "Verify that the highlighted ball follows play movement rather than bright crowd or signage artifacts.",
                }
            )

    if player_row_count == 0 or projected_player_points == 0:
        diagnostics.append(
            {
                "level": "warn",
                "title": "Field calibration has no player projection",
                "message": (
                    f"{calibration_refresh_successes}/{calibration_refresh_attempts} refreshes succeeded with mean visible pitch keypoints {avg_visible_pitch_keypoints:.1f}, "
                    f"but projected player points stayed {projected_player_points} and registered ratio is {registered_frame_ratio * 100:.1f}%. "
                    f"Frames with fresh homography {frames_with_usable_homography}, stale homography {frames_with_stale_homography}, anchors {frames_with_player_anchors}, projected {frames_with_projected_points}. "
                    f"Stale recovery {calibration_stale_recovery_successes}/{calibration_stale_recovery_attempts}. "
                    f"Gate rejects: {calibration_gate_summary}."
                ),
                "next_step": "Do not treat calibration as healthy until players are actually being projected onto the pitch. Fix detector output or anchor projection before trusting the minimap.",
            }
        )
    elif calibration_refresh_successes > 0 and registered_frame_ratio >= 0.7:
        diagnostics.append(
            {
                "level": "good",
                "title": "Field calibration: active",
                "message": (
                    f"{calibration_refresh_successes}/{calibration_refresh_attempts} refreshes succeeded; "
                    f"mean visible pitch keypoints {avg_visible_pitch_keypoints:.1f}; "
                    f"{registered_frame_ratio * 100:.1f}% of player detections projected. "
                    f"Frames with fresh homography {frames_with_usable_homography}, stale homography {frames_with_stale_homography}, anchors {frames_with_player_anchors}, projected {frames_with_projected_points}. "
                    f"Stale recovery {calibration_stale_recovery_successes}/{calibration_stale_recovery_attempts}. "
                    f"Gate rejects: {calibration_gate_summary}."
                ),
                "next_step": "If the minimap drifts, inspect live preview frames for missing pitch keypoints before changing tracking settings.",
            }
        )
    else:
        diagnostics.append(
            {
                "level": "warn",
                "title": "Field calibration: unstable",
                "message": (
                    f"{calibration_refresh_successes}/{calibration_refresh_attempts} refreshes succeeded; "
                    f"mean visible pitch keypoints {avg_visible_pitch_keypoints:.1f}; "
                    f"{registered_frame_ratio * 100:.1f}% of player detections projected. "
                    f"Frames with fresh homography {frames_with_usable_homography}, stale homography {frames_with_stale_homography}, anchors {frames_with_player_anchors}, projected {frames_with_projected_points}. "
                    f"Stale recovery {calibration_stale_recovery_successes}/{calibration_stale_recovery_attempts}. "
                    f"Gate rejects: {calibration_gate_summary}."
                ),
                "next_step": "Use a wider camera phase with clearer pitch markings if you need reliable minimap projection.",
            }
        )
    if goal_events:
        diagnostics.append(
            {
                "level": "good",
                "title": "Goal labels: loaded",
                "message": f"{len(goal_events)} aligned goal events loaded from {Path(goal_label_source).name if goal_label_source else 'labels'}.",
                "next_step": "Use the experiment outputs to compare pre-goal windows against baseline match state.",
            }
        )
    else:
        diagnostics.append(
            {
                "level": "warn",
                "title": "Goal labels: missing",
                "message": "No aligned SoccerNet goal events were found for this clip.",
                "next_step": "Use a clip with Labels-v2.json available if you want to evaluate the example experiment against goals.",
            }
        )

    heuristic_diagnostics = diagnostics

    summary = {
        "job_id": job_id,
        "run_dir": str(run_dir),
        "input_video": str(source_video_path),
        "overlay_video": f"/runs/{run_dir.name}/outputs/overlay.mp4",
        "detections_csv": f"/runs/{run_dir.name}/outputs/detections.csv",
        "track_summary_csv": f"/runs/{run_dir.name}/outputs/track_summary.csv",
        "projection_csv": projection_csv_key,
        "calibration_debug_csv": calibration_debug_csv_key,
        "entropy_timeseries_csv": entropy_timeseries_csv_key,
        "goal_events_csv": goal_events_csv_key,
        "summary_json": f"/runs/{run_dir.name}/outputs/summary.json",
        "all_outputs_zip": f"/runs/{run_dir.name}/outputs/all_outputs.zip",
        "device": detector_device,
        "field_calibration_device": keypoint_device,
        "player_model": detector_spec["name"],
        "requested_player_tracker_mode": requested_tracker_mode,
        "player_tracker_mode": tracker_mode_label(tracker_mode),
        "resolved_player_tracker_mode": tracker_mode_name,
        "player_tracker_runtime": tracker_runtime_label,
        "player_tracker_backend": tracker_backend.get("embedding_source"),
        "player_tracker_embedding_error": tracker_backend.get("embedding_error"),
        "player_tracker_stitching_enabled": bool(player_tracker is not None),
        "detector_class_names_source": detector_spec["class_names_source"],
        "player_detector_class_ids": list(detector_spec["player_class_ids"]),
        "ball_detector_class_ids": list(detector_spec["ball_class_ids"]),
        "referee_detector_class_ids": list(detector_spec["referee_class_ids"]),
        "ball_model": detector_spec["name"] if include_ball else "off",
        "ball_tracker_mode": DEFAULT_TRACKER,
        "field_calibration_model": KEYPOINT_MODEL_DEFAULT,
        "include_ball": include_ball,
        "player_conf": player_conf,
        "ball_conf": ball_conf,
        "iou": iou,
        "frames_processed": frame_index,
        "fps": round(float(fps), 4),
        "player_rows": player_row_count,
        "ball_rows": ball_row_count,
        "unique_player_track_ids": len(player_track_ids_seen),
        "raw_unique_player_track_ids": raw_unique_player_track_ids,
        "unique_ball_track_ids": len(ball_track_ids_seen),
        "home_tracks": home_tracks,
        "away_tracks": away_tracks,
        "unassigned_tracks": unassigned_tracks,
        "average_player_detections_per_frame": round(float(avg_players), 4),
        "average_ball_detections_per_frame": round(float(avg_ball), 4),
        "longest_track_length": longest_track_length,
        "average_track_length": round(float(average_track_length), 4),
        "raw_longest_track_length": raw_longest_track_length,
        "raw_average_track_length": round(float(raw_average_track_length), 4),
        "player_track_churn_ratio": round(float(churn_ratio), 6),
        "raw_player_track_churn_ratio": round(float(raw_churn_ratio), 6),
        "tracklet_merges_applied": stitch_stats["merge_count"],
        "stitched_track_id_reduction": round(
            float((raw_unique_player_track_ids - len(player_track_ids_seen)) / raw_unique_player_track_ids),
            6,
        )
        if raw_unique_player_track_ids > 0
        else 0.0,
        "identity_embedding_updates": int(tracker_backend.get("deep_feature_updates") or 0),
        "identity_embedding_interval_frames": int(tracker_backend.get("deep_feature_interval") or 0),
        "projected_player_points": projected_player_points,
        "projected_ball_points": projected_ball_points,
        "projected_player_points_fresh": projected_player_points_fresh,
        "projected_player_points_stale": projected_player_points_stale,
        "projected_ball_points_fresh": projected_ball_points_fresh,
        "projected_ball_points_stale": projected_ball_points_stale,
        "player_rows_while_calibration_fresh": player_rows_while_calibration_fresh,
        "player_rows_while_calibration_stale": player_rows_while_calibration_stale,
        "field_registered_frames": field_registered_frames,
        "field_registered_ratio": round(float(registered_frame_ratio), 4),
        "frames_with_field_homography": frames_with_field_homography,
        "frames_with_usable_homography": frames_with_usable_homography,
        "frames_with_nonstale_homography": frames_with_nonstale_homography,
        "frames_with_stale_homography": frames_with_stale_homography,
        "frames_projection_blocked_by_stale": frames_projection_blocked_by_stale,
        "frames_projected_with_last_known_homography": frames_projected_with_last_known_homography,
        "frames_with_player_anchors": frames_with_player_anchors,
        "frames_with_projected_points": frames_with_projected_points,
        "frames_with_homography_but_no_player_anchors": frames_with_homography_but_no_player_anchors,
        "homography_enabled": calibration_refresh_successes > 0,
        "field_calibration_refresh_frames": CALIBRATION_REFRESH_FRAMES,
        "field_calibration_refresh_attempts": calibration_refresh_attempts,
        "field_calibration_refresh_successes": calibration_refresh_successes,
        "field_calibration_success_rate": round(field_calibration_success, 6),
        "field_calibration_refresh_rejections": calibration_refresh_rejections,
        "field_keypoint_confidence_threshold": FIELD_KEYPOINT_CONFIDENCE,
        "field_calibration_min_visible_keypoints": MIN_CALIBRATION_VISIBLE_KEYPOINTS,
        "field_calibration_stale_recovery_min_visible_keypoints": STALE_RECOVERY_MIN_CALIBRATION_VISIBLE_KEYPOINTS,
        "field_calibration_stale_recovery_attempts": calibration_stale_recovery_attempts,
        "field_calibration_stale_recovery_successes": calibration_stale_recovery_successes,
        "field_calibration_stale_recovery_rejections": calibration_stale_recovery_rejections,
        "field_calibration_rejections_no_candidate": int(calibration_rejections_by_reason["no_candidate"]),
        "field_calibration_rejections_low_visible_count": int(calibration_rejections_by_reason["low_visible_count"]),
        "field_calibration_rejections_low_visible_keypoints": int(calibration_rejections_by_reason["low_visible_count"]),
        "field_calibration_rejections_low_inliers": int(calibration_rejections_by_reason["low_inliers"]),
        "field_calibration_rejections_high_reprojection_error": int(calibration_rejections_by_reason["high_reprojection_error"]),
        "field_calibration_rejections_high_temporal_drift": int(calibration_rejections_by_reason["high_temporal_drift"]),
        "field_calibration_rejections_invalid_candidate": int(calibration_invalid_candidate_rejections),
        "field_calibration_primary_rejections_no_candidate": int(calibration_primary_rejections_by_reason["no_candidate"]),
        "field_calibration_primary_rejections_low_visible_count": int(calibration_primary_rejections_by_reason["low_visible_count"]),
        "field_calibration_primary_rejections_low_visible_keypoints": int(calibration_primary_rejections_by_reason["low_visible_count"]),
        "field_calibration_primary_rejections_low_inliers": int(calibration_primary_rejections_by_reason["low_inliers"]),
        "field_calibration_primary_rejections_high_reprojection_error": int(calibration_primary_rejections_by_reason["high_reprojection_error"]),
        "field_calibration_primary_rejections_high_temporal_drift": int(calibration_primary_rejections_by_reason["high_temporal_drift"]),
        "field_calibration_primary_rejections_invalid_candidate": int(calibration_primary_rejections_by_reason["invalid_candidate"]),
        "average_visible_pitch_keypoints": round(float(avg_visible_pitch_keypoints), 4),
        "last_good_calibration_frame": last_good_calibration_frame,
        "detector_debug_sample_frames": raw_detector_debug_sample_frames,
        "raw_detector_boxes_sampled": raw_detector_boxes_sampled,
        "raw_detector_class_histogram_sample": {
            str(class_id): int(count)
            for class_id, count in sorted(raw_detector_class_histogram_sample.items())
        },
        "goal_events_count": len(goal_events),
        "goal_label_source": goal_label_source,
        "team_cluster_distance": round(float(team_cluster_distance), 4),
        "jersey_crops_used": len(jersey_samples),
        "experiments": [experiment_card],
        "top_tracks": track_rows[:20],
        "diagnostics": heuristic_diagnostics,
        "learn_cards": TACTICAL_LEARN_CARDS,
        "created_at": datetime.utcnow().isoformat() + "Z",
    }

    update_job_progress(job_manager, job_id, PROGRESS_VIDEO_FINALIZE_END + 1.0)
    checkpoint_job_control(job_control, job_id, job_manager)
    ai_diagnostics, diagnostics_artifact = generate_run_diagnostics(
        summary=summary,
        heuristic_diagnostics=heuristic_diagnostics,
        outputs_dir=outputs_dir,
        job_id=job_id,
        job_manager=job_manager,
    )
    update_job_progress(job_manager, job_id, PROGRESS_DIAGNOSTICS_END)
    summary["diagnostics"] = ai_diagnostics
    summary["heuristic_diagnostics"] = heuristic_diagnostics
    summary["diagnostics_source"] = "ai" if diagnostics_artifact.get("status") == "completed" else "heuristic"
    summary["diagnostics_provider"] = diagnostics_artifact.get("provider")
    summary["diagnostics_model"] = diagnostics_artifact.get("model")
    summary["diagnostics_status"] = diagnostics_artifact.get("status")
    summary["diagnostics_summary_line"] = diagnostics_artifact.get("summary_line", "")
    summary["diagnostics_error"] = diagnostics_artifact.get("error", "")
    summary["diagnostics_json"] = f"/runs/{run_dir.name}/outputs/diagnostics_ai.json"
    summary["diagnostics_prompt_context"] = diagnostics_artifact.get("prompt_context")

    checkpoint_job_control(job_control, job_id, job_manager)
    summary_json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    zip_inputs = [overlay_video_path, detections_csv_path, track_summary_csv_path, summary_json_path]
    if projection_csv_key is not None and projection_csv_path.exists():
        zip_inputs.append(projection_csv_path)
    if calibration_debug_csv_key is not None and calibration_debug_csv_path.exists():
        zip_inputs.append(calibration_debug_csv_path)
    if entropy_timeseries_csv_path.exists():
        zip_inputs.append(entropy_timeseries_csv_path)
    if goal_events_csv_path.exists():
        zip_inputs.append(goal_events_csv_path)
    diagnostics_json_path = outputs_dir / "diagnostics_ai.json"
    if diagnostics_json_path.exists():
        zip_inputs.append(diagnostics_json_path)
    update_job_progress(job_manager, job_id, PROGRESS_PACKAGING_END)
    checkpoint_job_control(job_control, job_id, job_manager)
    zip_paths(full_outputs_zip_path, zip_inputs)
    job_manager.log(job_id, f"Summary written to {summary_json_path.name}")
    return summary
