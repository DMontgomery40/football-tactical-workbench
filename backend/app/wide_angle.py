from __future__ import annotations

import csv
import json
import shutil
import subprocess
import time
import zipfile
from collections import Counter, defaultdict
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

DEFAULT_TRACKER = "bytetrack.yaml"
MODEL_CACHE_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

DETECTOR_MODEL_DEFAULT = "soccana"
KEYPOINT_MODEL_DEFAULT = "soccana_keypoint"
CALIBRATION_REFRESH_FRAMES = 10
FIELD_KEYPOINT_CONFIDENCE = 0.35
EXPERIMENT_WINDOW_SECONDS = 10.0
EXPERIMENT_GRID_COLS = 6
EXPERIMENT_GRID_ROWS = 4
GOAL_LOOKAHEAD_SECONDS = (30, 60)

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
        "title": "1. Player detection + ByteTrack",
        "what_it_does": "Uses soccer-specific detector weights so player, ball, and referee detection are trained for broadcast football rather than generic COCO classes.",
        "what_breaks": "Crowded boxes, camera pans, and tiny distant players still fragment track IDs.",
        "what_to_try_next": "Start with shorter clips, then raise model size or lower confidence only after you inspect churn.",
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
        "what_it_does": "Refreshes pitch calibration from field keypoints every 10 frames so the minimap can follow a moving broadcast camera.",
        "what_breaks": "If the keypoint model loses field structure during a fast cut or tight zoom, the projection has to coast on the last good calibration.",
        "what_to_try_next": "Watch the live preview for calibration dropouts instead of trusting the minimap blindly.",
    },
    {
        "title": "5. Demo first, fancy later",
        "what_it_does": "Packages detection, tracking, team ID, and rough spatial context into one believable tactical pipeline.",
        "what_breaks": "It is easy to jump to pose or fancy models before the wide-angle basics are stable.",
        "what_to_try_next": "Get the overlay and minimap looking trustworthy before you add anything more ambitious.",
    },
]

TEAM_BOX_COLORS = {
    "home": (255, 110, 80),
    "away": (80, 110, 255),
    "unassigned": (170, 170, 170),
    "ball": (0, 215, 255),
}


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


@lru_cache(maxsize=8)
def resolve_model_path(model_name: str, model_kind: str) -> str:
    model_key = model_name.strip()
    if not model_key:
        model_key = DETECTOR_MODEL_DEFAULT if model_kind == "detector" else KEYPOINT_MODEL_DEFAULT

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


def resolve_detector_spec(model_name: str) -> dict[str, Any]:
    model_key = model_name.strip() or DETECTOR_MODEL_DEFAULT
    if model_key in DETECTOR_MODEL_SOURCES:
        model_info = DETECTOR_MODEL_SOURCES[model_key]
        return {
            "name": model_key,
            "weights_path": resolve_model_path(model_key, "detector"),
            "player_class_id": int(model_info["player_class_id"]),
            "ball_class_id": int(model_info["ball_class_id"]),
            "referee_class_id": int(model_info["referee_class_id"]),
        }

    candidate = resolve_model_path(model_key, "detector")
    return {
        "name": model_key,
        "weights_path": candidate,
        "player_class_id": 0,
        "ball_class_id": 32,
        "referee_class_id": -1,
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


def clamp_box(box: np.ndarray, frame_width: int, frame_height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = [int(v) for v in box]
    x1 = max(0, min(frame_width - 1, x1))
    y1 = max(0, min(frame_height - 1, y1))
    x2 = max(x1 + 1, min(frame_width, x2))
    y2 = max(y1 + 1, min(frame_height, y2))
    return x1, y1, x2, y2


def draw_label(frame: np.ndarray, text: str, x1: int, y1: int, color: tuple[int, int, int]) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thickness = 2
    (width, height), baseline = cv2.getTextSize(text, font, scale, thickness)
    top = max(0, y1 - height - baseline - 8)
    right = min(frame.shape[1] - 1, x1 + width + 8)
    bottom = max(0, y1)
    cv2.rectangle(frame, (x1, top), (right, bottom), (20, 20, 20), -1)
    cv2.putText(frame, text, (x1 + 4, bottom - 4), font, scale, color, thickness, cv2.LINE_AA)


def zip_paths(output_zip_path: Path, paths: list[Path]) -> None:
    with zipfile.ZipFile(output_zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            if path.is_dir():
                for child in sorted(path.rglob("*")):
                    if child.is_file():
                        archive.write(child, arcname=str(child.relative_to(output_zip_path.parent)))
            elif path.is_file():
                archive.write(path, arcname=path.name)


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
    margin = PITCH_RENDER_MARGIN
    cv2.rectangle(image, (margin, margin), (map_width - margin, map_height - margin), line_color, 2)

    center_x = map_width // 2
    center_y = map_height // 2
    cv2.line(image, (center_x, margin), (center_x, map_height - margin), line_color, 2)
    cv2.circle(image, (center_x, center_y), max(12, map_height // 7), line_color, 2)

    penalty_box_width = int(round((16.5 / 105.0) * (map_width - 2 * margin)))
    penalty_box_height = int(round((40.3 / 68.0) * (map_height - 2 * margin)))
    top_box_y = center_y - penalty_box_height // 2
    bottom_box_y = center_y + penalty_box_height // 2
    cv2.rectangle(image, (margin, top_box_y), (margin + penalty_box_width, bottom_box_y), line_color, 2)
    cv2.rectangle(image, (map_width - margin - penalty_box_width, top_box_y), (map_width - margin, bottom_box_y), line_color, 2)
    return image


def compute_homography_matrix(homography_payload: dict[str, list[list[float]]] | None, map_width: int, map_height: int) -> tuple[np.ndarray | None, np.ndarray | None]:
    if not homography_payload or not homography_payload.get("source"):
        return None, None

    source = np.array(homography_payload["source"], dtype=np.float32)
    if homography_payload.get("target"):
        destination = np.array(homography_payload["target"], dtype=np.float32)
    else:
        margin = float(PITCH_RENDER_MARGIN)
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

    usable_width = map_width - 2 * PITCH_RENDER_MARGIN
    usable_height = map_height - 2 * PITCH_RENDER_MARGIN
    x_value = PITCH_RENDER_MARGIN + (float(np.clip(field_point[0], 0.0, PITCH_LENGTH_CM)) / PITCH_LENGTH_CM) * usable_width
    y_value = PITCH_RENDER_MARGIN + (float(np.clip(field_point[1], 0.0, PITCH_WIDTH_CM)) / PITCH_WIDTH_CM) * usable_height
    return float(x_value), float(y_value)


def detect_pitch_homography(
    frame: np.ndarray,
    keypoint_model: YOLO,
    device: str,
    confidence_threshold: float = FIELD_KEYPOINT_CONFIDENCE,
) -> tuple[np.ndarray | None, np.ndarray | None, int, int]:
    results = keypoint_model(source=frame, conf=0.05, device=device, verbose=False)
    if not results:
        return None, None, 0, 0

    result = results[0]
    if not hasattr(result, "keypoints") or result.keypoints is None or result.keypoints.data is None:
        return None, None, 0, 0

    keypoints_data = result.keypoints.data.cpu().numpy().astype(np.float32)
    if keypoints_data.size == 0:
        return None, None, 0, 0

    best_index = int(np.argmax((keypoints_data[:, :, 2] >= confidence_threshold).sum(axis=1)))
    keypoints = keypoints_data[best_index]
    valid_mask = keypoints[:, 2] >= confidence_threshold
    visible_count = int(valid_mask.sum())
    if visible_count < 4:
        return None, keypoints, visible_count, 0

    image_points = keypoints[valid_mask, :2].astype(np.float32)
    field_points = PITCH_REFERENCE_POINTS[valid_mask].astype(np.float32)
    homography_matrix, inlier_mask = cv2.findHomography(image_points, field_points, cv2.RANSAC, 35.0)
    inlier_count = int(inlier_mask.sum()) if inlier_mask is not None else visible_count
    if homography_matrix is None or inlier_count < 4:
        return None, keypoints, visible_count, inlier_count
    return homography_matrix.astype(np.float32), keypoints, visible_count, inlier_count


def overlay_minimap(frame: np.ndarray, minimap: np.ndarray) -> None:
    inset_height, inset_width = minimap.shape[:2]
    margin = 18
    panel_top = max(0, frame.shape[0] - inset_height - margin - 26)
    panel_left = max(0, frame.shape[1] - inset_width - margin - 10)
    panel_bottom = min(frame.shape[0], panel_top + inset_height + 34)
    panel_right = min(frame.shape[1], panel_left + inset_width + 20)
    cv2.rectangle(frame, (panel_left, panel_top), (panel_right, panel_bottom), (18, 18, 18), -1)

    map_top = panel_top + 26
    map_left = panel_left + 10
    frame[map_top:map_top + inset_height, map_left:map_left + inset_width] = minimap
    cv2.putText(frame, "Minimap", (panel_left + 10, panel_top + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (245, 245, 245), 2, cv2.LINE_AA)


def draw_detected_field_keypoints(frame: np.ndarray, keypoints: np.ndarray | None, confidence_threshold: float = FIELD_KEYPOINT_CONFIDENCE) -> None:
    if keypoints is None:
        return
    for x_value, y_value, confidence in keypoints:
        if confidence < confidence_threshold:
            continue
        cv2.circle(frame, (int(round(x_value)), int(round(y_value))), 4, (120, 220, 255), -1, cv2.LINE_AA)


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


def load_soccernet_goal_events(source_video_path: Path) -> tuple[list[dict[str, Any]], str | None]:
    parent = source_video_path.parent
    label_candidates = [parent / "Labels-v2.json", parent / "Labels.json"]
    label_path = next((path for path in label_candidates if path.exists()), None)
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


def generate_live_preview_stream(source_video_path: Path, config_payload: dict[str, Any]):
    detector_spec = resolve_detector_spec(str(config_payload.get("player_model") or DETECTOR_MODEL_DEFAULT))
    include_ball = bool(config_payload["include_ball"])
    player_conf = float(config_payload["player_conf"])
    ball_conf = float(config_payload["ball_conf"])
    iou = float(config_payload["iou"])

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
    minimap_width = max(240, min(360, frame_width // 4))
    minimap_height = int(round(minimap_width * 68 / 105))
    field_homography: np.ndarray | None = None
    latest_field_keypoints: np.ndarray | None = None
    latest_visible_keypoints = 0
    latest_inlier_count = 0
    last_calibration_frame = -1

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
                homography_candidate, detected_keypoints, visible_count, inlier_count = detect_pitch_homography(frame, keypoint_model, keypoint_device)
                if homography_candidate is not None:
                    field_homography = homography_candidate
                    last_calibration_frame = frame_index
                latest_field_keypoints = detected_keypoints
                latest_visible_keypoints = visible_count
                latest_inlier_count = inlier_count

            current_players: list[dict[str, Any]] = []
            current_ball: dict[str, Any] | None = None

            player_results = player_model.track(
                source=frame,
                persist=True,
                tracker=DEFAULT_TRACKER,
                conf=player_conf,
                iou=iou,
                device=detector_device,
                classes=[detector_spec["player_class_id"]],
                verbose=False,
            )
            player_boxes = player_results[0].boxes
            if player_boxes is not None and len(player_boxes) > 0:
                xyxy = player_boxes.xyxy.cpu().numpy()
                confidences = player_boxes.conf.cpu().numpy() if player_boxes.conf is not None else np.zeros(len(xyxy), dtype=np.float32)
                track_ids = player_boxes.id.cpu().numpy().astype(int) if player_boxes.id is not None else np.full(len(xyxy), -1, dtype=int)

                for index, box in enumerate(xyxy):
                    x1, y1, x2, y2 = clamp_box(box, frame_width, frame_height)
                    track_id = int(track_ids[index])
                    confidence = float(confidences[index])
                    anchor = (float((x1 + x2) / 2.0), float(y2))
                    feature = extract_jersey_feature(frame, (x1, y1, x2, y2))
                    if feature is not None and track_id >= 0:
                        jersey_samples.append(feature)
                        jersey_sample_track_ids.append(track_id)

                    current_players.append(
                        {
                            "track_id": track_id,
                            "confidence": confidence,
                            "bbox": (x1, y1, x2, y2),
                            "anchor": anchor,
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
                    classes=[detector_spec["ball_class_id"]],
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
            draw_detected_field_keypoints(annotated, latest_field_keypoints)

            for player_row in current_players:
                team_info = team_info_by_track.get(player_row["track_id"], default_team_info())
                player_row["team_label"] = team_info["team_label"]
                player_row["team_vote_ratio"] = float(team_info["team_vote_ratio"])
                player_row["field_point"] = project_point(player_row["anchor"], field_homography)
                player_row["pitch_point"] = field_point_to_minimap(player_row["field_point"], minimap_width, minimap_height)
                x1, y1, x2, y2 = player_row["bbox"]
                color = TEAM_BOX_COLORS.get(player_row["team_label"], TEAM_BOX_COLORS["unassigned"])
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                label = f"{player_row['team_label']} #{player_row['track_id']}" if player_row["track_id"] >= 0 else player_row["team_label"]
                draw_label(annotated, label, x1, y1, (255, 255, 255))

            if current_ball is not None:
                current_ball["field_point"] = project_point(current_ball["anchor"], field_homography)
                current_ball["pitch_point"] = field_point_to_minimap(current_ball["field_point"], minimap_width, minimap_height)
                x1, y1, x2, y2 = current_ball["bbox"]
                cv2.rectangle(annotated, (x1, y1), (x2, y2), TEAM_BOX_COLORS["ball"], 2)
                ball_label = f"ball #{current_ball['track_id']}" if current_ball["track_id"] >= 0 else "ball"
                draw_label(annotated, ball_label, x1, y1, (255, 245, 200))

            cv2.putText(
                annotated,
                f"live preview frame {frame_index + 1}",
                (20, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.82,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            status_text = (
                f"field calib {latest_visible_keypoints} kp / {latest_inlier_count} inliers"
                if field_homography is not None
                else f"field calib waiting ({latest_visible_keypoints} kp)"
            )
            cv2.putText(
                annotated,
                status_text,
                (20, 64),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (220, 238, 248),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                annotated,
                f"refresh every {CALIBRATION_REFRESH_FRAMES} frames · last good {last_calibration_frame if last_calibration_frame >= 0 else 'none'}",
                (20, 94),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (220, 238, 248),
                2,
                cv2.LINE_AA,
            )

            if field_homography is not None:
                minimap = create_pitch_map(minimap_width, minimap_height)
                for player_row in current_players:
                    if player_row["pitch_point"] is None:
                        continue
                    point = (int(round(player_row["pitch_point"][0])), int(round(player_row["pitch_point"][1])))
                    color = TEAM_BOX_COLORS.get(player_row["team_label"], TEAM_BOX_COLORS["unassigned"])
                    cv2.circle(minimap, point, 4, color, -1)
                if current_ball is not None and current_ball["pitch_point"] is not None:
                    point = (int(round(current_ball["pitch_point"][0])), int(round(current_ball["pitch_point"][1])))
                    cv2.circle(minimap, point, 3, TEAM_BOX_COLORS["ball"], -1)
                overlay_minimap(annotated, minimap)

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


def analyze_video(job_id: str, run_dir: Path, config_payload: dict[str, Any], job_manager: Any) -> dict[str, Any]:
    source_video_path = Path(config_payload["source_video_path"])
    player_model_name = str(config_payload["player_model"])
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

    minimap_width = max(240, min(360, frame_width // 4))
    minimap_height = int(round(minimap_width * 68 / 105))
    manual_homography_matrix, _ = compute_homography_matrix(homography_points, minimap_width, minimap_height)
    detector_spec = resolve_detector_spec(player_model_name)
    detector_path = detector_spec["weights_path"]
    keypoint_model_path = resolve_model_path(KEYPOINT_MODEL_DEFAULT, "keypoint")
    job_manager.log(job_id, f"Loading detector weights: {detector_spec['name']}")
    job_manager.log(job_id, f"Loading field calibration weights: {KEYPOINT_MODEL_DEFAULT}")
    keypoint_model = YOLO(keypoint_model_path)
    player_model = YOLO(detector_path)
    ball_model: YOLO | None = None
    if include_ball:
        job_manager.log(job_id, "Ball tracking enabled through the shared soccer detector")
        ball_model = YOLO(detector_path)
    else:
        job_manager.log(job_id, "Ball stage disabled")

    frame_records: list[dict[str, Any]] = []
    player_rows_by_track: dict[int, list[dict[str, Any]]] = defaultdict(list)
    player_track_ids_seen: set[int] = set()
    ball_track_ids_seen: set[int] = set()
    player_detections_per_frame: list[int] = []
    ball_detections_per_frame: list[int] = []
    jersey_samples: list[np.ndarray] = []
    jersey_sample_track_ids: list[int] = []
    calibration_visible_counts: list[int] = []
    calibration_inlier_counts: list[int] = []
    calibration_refresh_attempts = 0
    calibration_refresh_successes = 0
    field_registered_frames = 0
    last_good_calibration_frame = -1
    active_field_homography = manual_homography_matrix
    frame_index = 0
    player_row_count = 0
    ball_row_count = 0

    start_time = datetime.utcnow().timestamp()
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_index % CALIBRATION_REFRESH_FRAMES == 0 or active_field_homography is None:
            calibration_refresh_attempts += 1
            homography_candidate, _detected_keypoints, visible_count, inlier_count = detect_pitch_homography(frame, keypoint_model, keypoint_device)
            calibration_visible_counts.append(visible_count)
            calibration_inlier_counts.append(inlier_count)
            if homography_candidate is not None:
                active_field_homography = homography_candidate
                calibration_refresh_successes += 1
                last_good_calibration_frame = frame_index

        player_detection_count = 0
        ball_detection_count = 0
        frame_players: list[dict[str, Any]] = []
        frame_ball: dict[str, Any] | None = None
        current_visible_keypoints = calibration_visible_counts[-1] if calibration_visible_counts else 0
        current_inlier_count = calibration_inlier_counts[-1] if calibration_inlier_counts else 0

        player_results = player_model.track(
            source=frame,
            persist=True,
            tracker=DEFAULT_TRACKER,
            conf=player_conf,
            iou=iou,
            device=detector_device,
            classes=[detector_spec["player_class_id"]],
            verbose=False,
        )
        player_boxes = player_results[0].boxes
        if player_boxes is not None and len(player_boxes) > 0:
            xyxy = player_boxes.xyxy.cpu().numpy()
            confidences = player_boxes.conf.cpu().numpy() if player_boxes.conf is not None else np.zeros(len(xyxy), dtype=np.float32)
            track_ids = player_boxes.id.cpu().numpy().astype(int) if player_boxes.id is not None else np.full(len(xyxy), -1, dtype=int)

            for index, box in enumerate(xyxy):
                x1, y1, x2, y2 = clamp_box(box, frame_width, frame_height)
                confidence = float(confidences[index])
                track_id = int(track_ids[index])
                anchor_x = float((x1 + x2) / 2.0)
                anchor_y = float(y2)
                color_feature = extract_jersey_feature(frame, (x1, y1, x2, y2))
                if color_feature is not None and track_id >= 0:
                    jersey_samples.append(color_feature)
                    jersey_sample_track_ids.append(track_id)

                field_point = project_point((anchor_x, anchor_y), active_field_homography)
                pitch_point = field_point_to_minimap(field_point, minimap_width, minimap_height)
                row = {
                    "frame_index": frame_index,
                    "row_type": "player",
                    "track_id": track_id,
                    "class_name": "player",
                    "confidence": confidence,
                    "bbox": (x1, y1, x2, y2),
                    "anchor": (anchor_x, anchor_y),
                    "feature": color_feature,
                    "team_label": "unassigned",
                    "team_vote_ratio": 0.0,
                    "field_point": field_point,
                    "pitch_point": pitch_point,
                }
                frame_players.append(row)
                if track_id >= 0:
                    player_rows_by_track[track_id].append(row)
                    player_track_ids_seen.add(track_id)
                if field_point is not None:
                    field_registered_frames += 1
                player_detection_count += 1
                player_row_count += 1

        if include_ball and ball_model is not None:
            ball_results = ball_model.track(
                source=frame,
                persist=True,
                tracker=DEFAULT_TRACKER,
                conf=ball_conf,
                iou=iou,
                device=detector_device,
                classes=[detector_spec["ball_class_id"]],
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
                field_point = project_point((anchor_x, anchor_y), active_field_homography)
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
                ball_detection_count = 1
                ball_row_count += 1

        frame_records.append(
            {
                "players": frame_players,
                "ball": frame_ball,
                "field_calibration_active": active_field_homography is not None,
                "visible_pitch_keypoints": current_visible_keypoints,
                "homography_inliers": current_inlier_count,
                "last_good_calibration_frame": last_good_calibration_frame,
            }
        )
        player_detections_per_frame.append(player_detection_count)
        ball_detections_per_frame.append(ball_detection_count)
        frame_index += 1

        if total_frames > 0:
            progress = max(1.0, min(60.0, (frame_index / total_frames) * 60.0))
            job_manager.update(job_id, progress=progress)
        if frame_index % 25 == 0:
            elapsed = datetime.utcnow().timestamp() - start_time
            fps_effective = frame_index / elapsed if elapsed > 0 else 0.0
            job_manager.log(job_id, f"Tracked {frame_index} frames at {fps_effective:.2f} fps")

    cap.release()
    if frame_index == 0:
        raise RuntimeError("No frames were decoded from the source video.")

    track_team_info, team_cluster_distance = _compute_online_track_team_info(jersey_samples, jersey_sample_track_ids)
    if track_team_info:
        job_manager.log(job_id, f"Built two team color clusters from {len(jersey_samples)} jersey crops")
    else:
        for track_id in player_rows_by_track:
            track_team_info[track_id] = default_team_info()
        job_manager.log(job_id, "Not enough jersey crops for reliable team clustering; leaving tracks unassigned")

    home_tracks = 0
    away_tracks = 0
    unassigned_tracks = 0
    projected_player_points = 0
    projected_ball_points = 0
    projection_rows: list[dict[str, Any]] = []

    for record in frame_records:
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

    entropy_timeseries_rows, experiment_card = build_geometric_volatility_experiment(frame_records, fps if fps > 0 else 25.0)
    goal_events, goal_label_source = load_soccernet_goal_events(source_video_path)
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
            "frame_index",
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

        for track_id, rows in sorted(player_rows_by_track.items(), key=lambda item: len(item[1]), reverse=True):
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
    render_cap = cv2.VideoCapture(str(source_video_path))
    if not render_cap.isOpened():
        raise RuntimeError(f"Could not reopen video for rendering: {source_video_path}")

    writer = cv2.VideoWriter(
        str(raw_overlay_video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (frame_width, frame_height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create overlay writer: {raw_overlay_video_path}")

    for render_index, record in enumerate(frame_records):
        ok, frame = render_cap.read()
        if not ok:
            break

        annotated = frame.copy()

        for player_row in record["players"]:
            x1, y1, x2, y2 = player_row["bbox"]
            team_label = str(player_row["team_label"])
            color = TEAM_BOX_COLORS.get(team_label, TEAM_BOX_COLORS["unassigned"])
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            label = f"{team_label} #{player_row['track_id']}" if player_row["track_id"] >= 0 else team_label
            draw_label(annotated, label, x1, y1, (255, 255, 255))

        ball_row = record["ball"]
        if ball_row is not None:
            x1, y1, x2, y2 = ball_row["bbox"]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), TEAM_BOX_COLORS["ball"], 2)
            ball_label = f"ball #{ball_row['track_id']}" if ball_row["track_id"] >= 0 else "ball"
            draw_label(annotated, ball_label, x1, y1, (255, 245, 200))

        status = f"frame {render_index + 1}"
        if total_frames > 0:
            status += f"/{total_frames}"
        cv2.putText(annotated, status, (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(
            annotated,
            f"tracks {len(player_track_ids_seen)}  home {home_tracks}  away {away_tracks}",
            (20, 64),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (220, 238, 248),
            2,
            cv2.LINE_AA,
        )
        field_status = (
            f"field calib {record['visible_pitch_keypoints']} kp / {record['homography_inliers']} inliers"
            if record["field_calibration_active"]
            else f"field calib waiting ({record['visible_pitch_keypoints']} kp)"
        )
        cv2.putText(annotated, field_status, (20, 94), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (220, 238, 248), 2, cv2.LINE_AA)
        cv2.putText(
            annotated,
            f"refresh every {CALIBRATION_REFRESH_FRAMES} frames · last good {record['last_good_calibration_frame'] if record['last_good_calibration_frame'] >= 0 else 'none'}",
            (20, 122),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (220, 238, 248),
            2,
            cv2.LINE_AA,
        )

        if record["field_calibration_active"]:
            minimap = create_pitch_map(minimap_width, minimap_height)
            for player_row in record["players"]:
                if player_row["pitch_point"] is None:
                    continue
                team_label = str(player_row["team_label"])
                color = TEAM_BOX_COLORS.get(team_label, TEAM_BOX_COLORS["unassigned"])
                point = (int(round(player_row["pitch_point"][0])), int(round(player_row["pitch_point"][1])))
                cv2.circle(minimap, point, 4, color, -1)

            if ball_row is not None and ball_row["pitch_point"] is not None:
                point = (int(round(ball_row["pitch_point"][0])), int(round(ball_row["pitch_point"][1])))
                cv2.circle(minimap, point, 3, TEAM_BOX_COLORS["ball"], -1)

            overlay_minimap(annotated, minimap)

        writer.write(annotated)

        if total_frames > 0:
            progress = 60.0 + min(39.0, (render_index + 1) / total_frames * 39.0)
            job_manager.update(job_id, progress=progress)

    render_cap.release()
    writer.release()
    finalize_overlay_video(raw_overlay_video_path, overlay_video_path, job_id, job_manager)

    diagnostics: list[dict[str, str]] = []
    churn_ratio = len(player_track_ids_seen) / max(frame_index, 1)
    avg_players = float(np.mean(player_detections_per_frame)) if player_detections_per_frame else 0.0
    avg_ball = float(np.mean(ball_detections_per_frame)) if ball_detections_per_frame else 0.0
    avg_visible_pitch_keypoints = float(np.mean(calibration_visible_counts)) if calibration_visible_counts else 0.0
    registered_frame_ratio = field_registered_frames / max(player_row_count, 1)

    if churn_ratio > 0.1:
        diagnostics.append(
            {
                "level": "warn",
                "title": "Tracker churn is still high",
                "message": f"{len(player_track_ids_seen)} tracked player IDs across {frame_index} frames is a noisy tactical demo.",
                "next_step": "Trim to a steadier wide-angle phase before touching anything more ambitious.",
            }
        )
    else:
        diagnostics.append(
            {
                "level": "good",
                "title": "Tracker churn is workable",
                "message": f"{len(player_track_ids_seen)} tracked player IDs across {frame_index} frames is believable for a demo overlay.",
                "next_step": "Inspect the longest tracks and decide whether the story on the overlay matches the play.",
            }
        )

    if home_tracks > 0 and away_tracks > 0 and team_cluster_distance > 0.08:
        diagnostics.append(
            {
                "level": "good",
                "title": "Team color split produced two groups",
                "message": f"Assigned {home_tracks} home tracks and {away_tracks} away tracks from {len(jersey_samples)} jersey crops.",
                "next_step": "Sanity-check goalkeepers and referees, because unsupervised color clustering will still lie sometimes.",
            }
        )
    else:
        diagnostics.append(
            {
                "level": "warn",
                "title": "Team color clustering is weak",
                "message": "The jersey crops did not cleanly separate into two teams.",
                "next_step": "Use clips with better kit contrast or larger player crops before trusting home versus away labels.",
            }
        )

    if include_ball:
        if avg_ball < 0.15:
            diagnostics.append(
                {
                    "level": "warn",
                    "title": "Ball tracking is sparse",
                    "message": f"Average ball detections per frame is {avg_ball:.2f}.",
                    "next_step": "Treat the ball overlay as context only until it stops blinking in and out.",
                }
            )
        else:
            diagnostics.append(
                {
                    "level": "good",
                    "title": "Ball stage adds context",
                    "message": f"Average ball detections per frame is {avg_ball:.2f}.",
                    "next_step": "Check whether the highlighted ball follows the play instead of bright noise in the stands.",
                }
            )

    if calibration_refresh_successes > 0:
        diagnostics.append(
            {
                "level": "good",
                "title": "Automatic field calibration is active",
                "message": f"Refreshed calibration {calibration_refresh_successes} times over {calibration_refresh_attempts} attempts using {avg_visible_pitch_keypoints:.1f} visible pitch keypoints on average.",
                "next_step": "If the minimap drifts, watch the live preview for calibration dropouts before touching player tracking.",
            }
        )
    else:
        diagnostics.append(
            {
                "level": "warn",
                "title": "Field registration never locked on",
                "message": "The pitch keypoint model did not produce a stable calibration during this clip.",
                "next_step": "Try a wider camera phase with clearer field markings. The minimap depends on visible pitch structure.",
            }
        )

    diagnostics.append(
        {
            "level": "warn",
            "title": "Experimental signal is exploratory",
            "message": "Spatial entropy volatility is an experiment layered on top of the core tracking pipeline, not a production betting model.",
            "next_step": "Use the experimental panel to compare spikes against the overlay before treating the metric as actionable.",
        }
    )
    if goal_events:
        diagnostics.append(
            {
                "level": "good",
                "title": "Goal labels attached to the experiment",
                "message": f"Loaded {len(goal_events)} SoccerNet goal events from {Path(goal_label_source).name if goal_label_source else 'labels'}.",
                "next_step": "Use the entropy timeseries CSV to compare volatility in pre-goal windows against the rest of the half.",
            }
        )
    else:
        diagnostics.append(
            {
                "level": "warn",
                "title": "No goal labels attached",
                "message": "This run has no aligned SoccerNet goal events, so the volatility experiment cannot be evaluated against scoring windows yet.",
                "next_step": "Run the pipeline on SoccerNet halves with Labels-v2.json available if you want meaningful experimental evaluation.",
            }
        )

    summary = {
        "job_id": job_id,
        "run_dir": str(run_dir),
        "input_video": str(source_video_path),
        "overlay_video": f"/runs/{run_dir.name}/outputs/overlay.mp4",
        "detections_csv": f"/runs/{run_dir.name}/outputs/detections.csv",
        "track_summary_csv": f"/runs/{run_dir.name}/outputs/track_summary.csv",
        "projection_csv": projection_csv_key,
        "entropy_timeseries_csv": entropy_timeseries_csv_key,
        "goal_events_csv": goal_events_csv_key,
        "summary_json": f"/runs/{run_dir.name}/outputs/summary.json",
        "all_outputs_zip": f"/runs/{run_dir.name}/outputs/all_outputs.zip",
        "device": detector_device,
        "field_calibration_device": keypoint_device,
        "player_model": detector_spec["name"],
        "ball_model": detector_spec["name"] if include_ball else "off",
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
        "unique_ball_track_ids": len(ball_track_ids_seen),
        "home_tracks": home_tracks,
        "away_tracks": away_tracks,
        "unassigned_tracks": unassigned_tracks,
        "average_player_detections_per_frame": round(float(avg_players), 4),
        "average_ball_detections_per_frame": round(float(avg_ball), 4),
        "longest_track_length": longest_track_length,
        "average_track_length": round(float(average_track_length), 4),
        "projected_player_points": projected_player_points,
        "projected_ball_points": projected_ball_points,
        "field_registered_frames": field_registered_frames,
        "field_registered_ratio": round(float(registered_frame_ratio), 4),
        "homography_enabled": calibration_refresh_successes > 0,
        "field_calibration_refresh_frames": CALIBRATION_REFRESH_FRAMES,
        "field_calibration_refresh_attempts": calibration_refresh_attempts,
        "field_calibration_refresh_successes": calibration_refresh_successes,
        "average_visible_pitch_keypoints": round(float(avg_visible_pitch_keypoints), 4),
        "last_good_calibration_frame": last_good_calibration_frame,
        "goal_events_count": len(goal_events),
        "goal_label_source": goal_label_source,
        "team_cluster_distance": round(float(team_cluster_distance), 4),
        "jersey_crops_used": len(jersey_samples),
        "experiments": [experiment_card],
        "top_tracks": track_rows[:20],
        "diagnostics": diagnostics,
        "learn_cards": TACTICAL_LEARN_CARDS,
        "created_at": datetime.utcnow().isoformat() + "Z",
    }

    summary_json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    zip_inputs = [overlay_video_path, detections_csv_path, track_summary_csv_path, summary_json_path]
    if projection_csv_key is not None and projection_csv_path.exists():
        zip_inputs.append(projection_csv_path)
    if entropy_timeseries_csv_path.exists():
        zip_inputs.append(entropy_timeseries_csv_path)
    if goal_events_csv_path.exists():
        zip_inputs.append(goal_events_csv_path)
    zip_paths(full_outputs_zip_path, zip_inputs)
    job_manager.log(job_id, f"Summary written to {summary_json_path.name}")
    return summary
