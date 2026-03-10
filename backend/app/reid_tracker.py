from __future__ import annotations

import math
import ssl
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib import request

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

DEFAULT_PLAYER_TRACKER_MODE = "hybrid_reid"
BYTETRACK_PLAYER_TRACKER_MODE = "bytetrack"
BALL_TRACKER_NAME = "bytetrack.yaml"
PLAYER_TRACKER_MODE_OPTIONS = [DEFAULT_PLAYER_TRACKER_MODE, BYTETRACK_PLAYER_TRACKER_MODE]
PLAYER_TRACKER_MODE_LABELS = {
    DEFAULT_PLAYER_TRACKER_MODE: "hybrid_reid",
    BYTETRACK_PLAYER_TRACKER_MODE: "bytetrack",
}
REID_MODEL_CACHE_DIR = Path(__file__).resolve().parent.parent / "models" / "appearance"
REID_MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def normalize_player_tracker_mode(value: Any) -> str:
    raw = str(value or "").strip().lower()
    if raw in {"", "auto", "default", "hybrid", "hybrid_reid", "reid", "appearance"}:
        return DEFAULT_PLAYER_TRACKER_MODE
    if raw in {"bytetrack", "byte"}:
        return BYTETRACK_PLAYER_TRACKER_MODE
    return DEFAULT_PLAYER_TRACKER_MODE


def tracker_mode_label(value: Any) -> str:
    return PLAYER_TRACKER_MODE_LABELS.get(normalize_player_tracker_mode(value), DEFAULT_PLAYER_TRACKER_MODE)


def _l2_normalize(vector: np.ndarray | None) -> np.ndarray | None:
    if vector is None:
        return None
    array = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(array))
    if norm <= 1e-8:
        return None
    return array / norm


def _bbox_area(bbox: tuple[int, int, int, int]) -> float:
    x1, y1, x2, y2 = bbox
    return float(max(0, x2 - x1) * max(0, y2 - y1))


def _bbox_center(bbox: tuple[int, int, int, int]) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (float((x1 + x2) / 2.0), float((y1 + y2) / 2.0))


def _shift_bbox(bbox: tuple[int, int, int, int], dx: float, dy: float, frame_size: tuple[int, int]) -> tuple[int, int, int, int]:
    width, height = frame_size
    x1, y1, x2, y2 = bbox
    shifted = (
        int(round(x1 + dx)),
        int(round(y1 + dy)),
        int(round(x2 + dx)),
        int(round(y2 + dy)),
    )
    sx1 = max(0, min(width - 1, shifted[0]))
    sy1 = max(0, min(height - 1, shifted[1]))
    sx2 = max(sx1 + 1, min(width, shifted[2]))
    sy2 = max(sy1 + 1, min(height, shifted[3]))
    return (sx1, sy1, sx2, sy2)


def _bbox_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    intersection = float(iw * ih)
    if intersection <= 0.0:
        return 0.0
    area_a = _bbox_area(a)
    area_b = _bbox_area(b)
    denom = area_a + area_b - intersection
    if denom <= 1e-8:
        return 0.0
    return intersection / denom


def _cosine_distance(a: np.ndarray | None, b: np.ndarray | None) -> float | None:
    na = _l2_normalize(a)
    nb = _l2_normalize(b)
    if na is None or nb is None:
        return None
    similarity = float(np.clip(np.dot(na, nb), -1.0, 1.0))
    return 1.0 - similarity


def _sample_identity_crop(frame: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray | None:
    frame_height, frame_width = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    box_width = x2 - x1
    box_height = y2 - y1
    if box_width < 10 or box_height < 16:
        return None

    pad_x = int(round(box_width * 0.08))
    pad_y_top = int(round(box_height * 0.08))
    pad_y_bottom = int(round(box_height * 0.04))
    left = max(0, x1 - pad_x)
    top = max(0, y1 - pad_y_top)
    right = min(frame_width, x2 + pad_x)
    bottom = min(frame_height, y2 + pad_y_bottom)
    if right - left < 10 or bottom - top < 16:
        return None
    return frame[top:bottom, left:right]


def _identity_color_feature(crop: np.ndarray | None) -> np.ndarray | None:
    if crop is None or crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    histograms: list[np.ndarray] = []
    channel_bins = ((0, 180, 16), (0, 256, 8), (0, 256, 8))
    for channel_index, (start, end, bins) in enumerate(channel_bins):
        hist = cv2.calcHist([hsv], [channel_index], None, [bins], [start, end]).reshape(-1)
        hist = hist.astype(np.float32)
        if hist.sum() > 0:
            hist /= hist.sum()
        histograms.append(hist)
    feature = np.concatenate(histograms, axis=0).astype(np.float32)
    return _l2_normalize(feature)


class SparseAppearanceEmbedder:
    def __init__(self, device: str = "cpu", deep_feature_interval: int = 12, min_deep_feature_area: float = 625.0) -> None:
        self.device_preference = device
        self.deep_feature_interval = max(1, int(deep_feature_interval))
        self.min_deep_feature_area = float(min_deep_feature_area)
        self.deep_feature_dim = 512
        self.color_feature_dim = 32
        self.deep_feature_source = "hsv_hist_only"
        self.deep_feature_error: str | None = None
        self.deep_feature_updates = 0
        self._torch = None
        self._model = None
        self._device = "cpu"
        self._deep_weight = 0.85
        self._color_weight = 0.35
        self._init_model()

    def _init_model(self) -> None:
        try:
            import torch
            from torchvision.models import ResNet18_Weights, resnet18
        except Exception as exc:
            self.deep_feature_error = f"torchvision unavailable: {exc}"
            return

        try:
            weights = ResNet18_Weights.DEFAULT
            model = resnet18(weights=None)
            state_dict_path = self._ensure_torchvision_weights(weights.url)
            state_dict = torch.load(state_dict_path, map_location="cpu", weights_only=True)
            model.load_state_dict(state_dict)
            model.fc = torch.nn.Identity()
            preferred_device = self.device_preference
            if preferred_device == "mps" and not torch.backends.mps.is_available():
                preferred_device = "cpu"
            if preferred_device == "cuda" and not torch.cuda.is_available():
                preferred_device = "cpu"
            model = model.to(preferred_device)
            model.eval()
            self._torch = torch
            self._device = preferred_device
            self._model = model
            self.deep_feature_source = f"torchvision_resnet18_imagenet@{preferred_device}"
        except Exception as exc:
            self.deep_feature_error = str(exc)
            self._model = None
            self._torch = None
            self._device = "cpu"
            self.deep_feature_source = "hsv_hist_only"

    def _ensure_torchvision_weights(self, url: str) -> Path:
        destination = REID_MODEL_CACHE_DIR / Path(url).name
        if destination.exists() and destination.stat().st_size > 0:
            return destination

        ssl_context = None
        try:
            import certifi

            ssl_context = ssl.create_default_context(cafile=certifi.where())
        except Exception:
            ssl_context = ssl.create_default_context()

        req = request.Request(url, headers={"User-Agent": "football-tactical-workbench/1.0"})
        with request.urlopen(req, timeout=120, context=ssl_context) as response:
            destination.write_bytes(response.read())
        return destination

    def describe(self) -> dict[str, Any]:
        return {
            "embedding_source": self.deep_feature_source,
            "embedding_error": self.deep_feature_error,
            "deep_feature_updates": self.deep_feature_updates,
            "deep_feature_interval": self.deep_feature_interval,
        }

    def _preprocess_for_deep_model(self, crop: np.ndarray) -> Any:
        torch = self._torch
        if torch is None:
            return None
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (128, 256), interpolation=cv2.INTER_LINEAR)
        array = resized.astype(np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=tensor.dtype).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=tensor.dtype).view(3, 1, 1)
        tensor = (tensor - mean) / std
        return tensor

    def encode(self, frame: np.ndarray, boxes: list[tuple[int, int, int, int]], frame_index: int) -> list[np.ndarray | None]:
        features: list[np.ndarray | None] = [None] * len(boxes)
        identity_crops = [_sample_identity_crop(frame, box) for box in boxes]
        color_features = [_identity_color_feature(crop) for crop in identity_crops]

        deep_features: dict[int, np.ndarray] = {}
        should_run_deep = self._model is not None and (frame_index % self.deep_feature_interval == 0)
        if should_run_deep and self._torch is not None:
            prepared_batches: list[Any] = []
            prepared_indices: list[int] = []
            for index, crop in enumerate(identity_crops):
                if crop is None or _bbox_area(boxes[index]) < self.min_deep_feature_area:
                    continue
                prepared = self._preprocess_for_deep_model(crop)
                if prepared is None:
                    continue
                prepared_batches.append(prepared)
                prepared_indices.append(index)

            if prepared_batches:
                torch = self._torch
                batch = torch.stack(prepared_batches, dim=0).to(self._device)
                with torch.inference_mode():
                    outputs = self._model(batch)
                vectors = outputs.detach().cpu().numpy().astype(np.float32)
                for local_index, detection_index in enumerate(prepared_indices):
                    vector = _l2_normalize(vectors[local_index])
                    if vector is not None:
                        deep_features[detection_index] = vector
                        self.deep_feature_updates += 1

        for index, color_feature in enumerate(color_features):
            deep_feature = deep_features.get(index)
            deep_part = np.zeros(self.deep_feature_dim, dtype=np.float32)
            color_part = np.zeros(self.color_feature_dim, dtype=np.float32)
            has_signal = False
            if deep_feature is not None:
                deep_part[: min(len(deep_feature), self.deep_feature_dim)] = deep_feature[: self.deep_feature_dim] * self._deep_weight
                has_signal = True
            if color_feature is not None:
                color_part[: min(len(color_feature), self.color_feature_dim)] = color_feature[: self.color_feature_dim] * self._color_weight
                has_signal = True
            if not has_signal:
                features[index] = None
                continue
            combined = np.concatenate([deep_part, color_part], axis=0).astype(np.float32)
            features[index] = _l2_normalize(combined)
        return features


@dataclass
class HybridTrack:
    track_id: int
    bbox: tuple[int, int, int, int]
    confidence: float
    anchor: tuple[float, float]
    field_point: tuple[float, float] | None
    frame_index: int
    feature: np.ndarray | None
    frame_size: tuple[int, int]
    first_frame: int = 0
    last_frame: int = 0
    first_anchor: tuple[float, float] = (0.0, 0.0)
    last_anchor: tuple[float, float] = (0.0, 0.0)
    first_field_point: tuple[float, float] | None = None
    last_field_point: tuple[float, float] | None = None
    velocity: tuple[float, float] = (0.0, 0.0)
    feature_sum: np.ndarray | None = None
    feature_count: int = 0
    smoothed_feature: np.ndarray | None = None
    confidence_sum: float = 0.0
    bbox_area_sum: float = 0.0
    observation_count: int = 0

    def __post_init__(self) -> None:
        self.first_frame = self.frame_index
        self.last_frame = self.frame_index
        self.first_anchor = self.anchor
        self.last_anchor = self.anchor
        self.first_field_point = self.field_point
        self.last_field_point = self.field_point
        self.confidence_sum = float(self.confidence)
        self.bbox_area_sum = _bbox_area(self.bbox)
        self.observation_count = 1
        if self.feature is not None:
            self.feature_sum = self.feature.copy()
            self.smoothed_feature = self.feature.copy()
            self.feature_count = 1

    def predicted_bbox(self, frame_index: int) -> tuple[int, int, int, int]:
        gap = max(0, int(frame_index) - int(self.last_frame))
        dx = self.velocity[0] * gap
        dy = self.velocity[1] * gap
        return _shift_bbox(self.bbox, dx, dy, self.frame_size)

    def predicted_anchor(self, frame_index: int) -> tuple[float, float]:
        gap = max(0, int(frame_index) - int(self.last_frame))
        return (
            float(self.last_anchor[0] + self.velocity[0] * gap),
            float(self.last_anchor[1] + self.velocity[1] * gap),
        )

    def update(self, bbox: tuple[int, int, int, int], confidence: float, anchor: tuple[float, float], field_point: tuple[float, float] | None, frame_index: int, feature: np.ndarray | None) -> None:
        gap = max(1, int(frame_index) - int(self.last_frame))
        self.velocity = (
            float((anchor[0] - self.last_anchor[0]) / gap),
            float((anchor[1] - self.last_anchor[1]) / gap),
        )
        self.bbox = bbox
        self.confidence = float(confidence)
        self.anchor = anchor
        self.field_point = field_point
        self.frame_index = int(frame_index)
        self.last_frame = int(frame_index)
        self.last_anchor = anchor
        if field_point is not None:
            if self.first_field_point is None:
                self.first_field_point = field_point
            self.last_field_point = field_point
        self.confidence_sum += float(confidence)
        self.bbox_area_sum += _bbox_area(bbox)
        self.observation_count += 1
        if feature is not None:
            if self.feature_sum is None:
                self.feature_sum = feature.copy()
                self.smoothed_feature = feature.copy()
                self.feature_count = 1
            else:
                self.feature_sum += feature
                self.feature_count += 1
                if self.smoothed_feature is None:
                    self.smoothed_feature = feature.copy()
                else:
                    self.smoothed_feature = _l2_normalize(self.smoothed_feature * 0.88 + feature * 0.12)
        self.feature = feature

    def export(self) -> dict[str, Any]:
        mean_feature = None
        if self.feature_sum is not None and self.feature_count > 0:
            mean_feature = _l2_normalize(self.feature_sum / float(self.feature_count))
        return {
            "track_id": self.track_id,
            "first_frame": self.first_frame,
            "last_frame": self.last_frame,
            "first_anchor": self.first_anchor,
            "last_anchor": self.last_anchor,
            "first_field_point": self.first_field_point,
            "last_field_point": self.last_field_point,
            "average_confidence": self.confidence_sum / max(self.observation_count, 1),
            "average_bbox_area": self.bbox_area_sum / max(self.observation_count, 1),
            "observation_count": self.observation_count,
            "mean_feature": mean_feature,
        }


class HybridReIDTracker:
    def __init__(self, fps: float, frame_size: tuple[int, int], detection_confidence_floor: float, device: str = "cpu") -> None:
        self.frame_size = frame_size
        self.frame_diagonal = math.hypot(frame_size[0], frame_size[1])
        self.max_missing_frames = max(45, int(round(fps * 3.0)))
        self.assignment_cost_threshold = 0.78
        self.stitch_gap_frames = max(75, int(round(fps * 6.0)))
        self.detection_confidence_floor = float(detection_confidence_floor)
        self.embedder = SparseAppearanceEmbedder(device=device)
        self._tracks: dict[int, HybridTrack] = {}
        self._next_track_id = 1
        self.assignment_count = 0
        self.new_track_count = 0

    def describe_backend(self) -> dict[str, Any]:
        return {
            "tracker_mode": DEFAULT_PLAYER_TRACKER_MODE,
            **self.embedder.describe(),
            "max_missing_frames": self.max_missing_frames,
            "stitch_gap_frames": self.stitch_gap_frames,
            "assignment_count": self.assignment_count,
            "new_track_count": self.new_track_count,
        }

    def export_tracklets(self) -> dict[int, dict[str, Any]]:
        return {track_id: track.export() for track_id, track in self._tracks.items()}

    def _association_cost(self, track: HybridTrack, detection: dict[str, Any], frame_index: int) -> float:
        gap = max(1, int(frame_index) - int(track.last_frame))
        if gap > self.max_missing_frames:
            return float("inf")

        predicted_bbox = track.predicted_bbox(frame_index)
        predicted_anchor = track.predicted_anchor(frame_index)
        iou = _bbox_iou(predicted_bbox, detection["bbox"])
        detection_anchor = detection["anchor"]
        center_distance_px = math.dist(predicted_anchor, detection_anchor)
        image_gate_px = max(
            (max(detection["bbox"][2] - detection["bbox"][0], detection["bbox"][3] - detection["bbox"][1]) * 2.25) + gap * 12.0,
            self.frame_diagonal * (0.06 + gap * 0.0025),
        )
        if iou < 0.01 and center_distance_px > image_gate_px:
            return float("inf")

        field_cost = min(1.0, center_distance_px / max(self.frame_diagonal * 0.35, 1.0))
        if track.last_field_point is not None and detection["field_point"] is not None:
            field_distance = float(math.dist(track.last_field_point, detection["field_point"]))
            field_gate = 260.0 + gap * 80.0
            if field_distance > field_gate:
                return float("inf")
            field_cost = min(1.0, field_distance / max(field_gate, 1.0))

        appearance_distance = _cosine_distance(track.smoothed_feature, detection["identity_feature"])
        if appearance_distance is None:
            appearance_cost = 0.32
        else:
            appearance_cost = appearance_distance
            if gap > 2 and iou < 0.08 and appearance_distance > 0.52:
                return float("inf")

        track_area = max(track.bbox_area_sum / max(track.observation_count, 1), 1.0)
        detection_area = max(_bbox_area(detection["bbox"]), 1.0)
        size_ratio = max(track_area, detection_area) / max(min(track_area, detection_area), 1.0)
        if size_ratio > 3.2:
            return float("inf")
        size_cost = min(1.0, abs(math.log(size_ratio)) / 1.2)

        gap_cost = min(1.0, gap / max(self.max_missing_frames, 1))
        iou_cost = 1.0 - iou
        return (0.34 * iou_cost) + (0.32 * appearance_cost) + (0.18 * field_cost) + (0.08 * size_cost) + (0.08 * gap_cost)

    def update(self, frame: np.ndarray, detections: list[dict[str, Any]], frame_index: int) -> list[int]:
        if not detections:
            return []

        features = self.embedder.encode(frame, [detection["bbox"] for detection in detections], frame_index)
        for detection, feature in zip(detections, features):
            detection["identity_feature"] = feature

        candidate_tracks = [
            track
            for track in self._tracks.values()
            if (frame_index - track.last_frame) <= self.max_missing_frames
        ]
        assigned_track_ids = [-1] * len(detections)

        if candidate_tracks:
            cost_matrix = np.full((len(candidate_tracks), len(detections)), 1e6, dtype=np.float32)
            for track_index, track in enumerate(candidate_tracks):
                for detection_index, detection in enumerate(detections):
                    cost = self._association_cost(track, detection, frame_index)
                    if math.isfinite(cost):
                        cost_matrix[track_index, detection_index] = cost

            if np.isfinite(cost_matrix).any():
                row_indices, col_indices = linear_sum_assignment(cost_matrix)
                for track_index, detection_index in zip(row_indices.tolist(), col_indices.tolist()):
                    cost = float(cost_matrix[track_index, detection_index])
                    if not math.isfinite(cost) or cost > self.assignment_cost_threshold:
                        continue
                    track = candidate_tracks[track_index]
                    detection = detections[detection_index]
                    track.update(
                        bbox=detection["bbox"],
                        confidence=float(detection["confidence"]),
                        anchor=detection["anchor"],
                        field_point=detection["field_point"],
                        frame_index=frame_index,
                        feature=detection.get("identity_feature"),
                    )
                    assigned_track_ids[detection_index] = int(track.track_id)
                    self.assignment_count += 1

        for detection_index, detection in enumerate(detections):
            if assigned_track_ids[detection_index] >= 0:
                continue
            if float(detection["confidence"]) < self.detection_confidence_floor:
                continue
            track_id = self._next_track_id
            self._next_track_id += 1
            track = HybridTrack(
                track_id=track_id,
                bbox=detection["bbox"],
                confidence=float(detection["confidence"]),
                anchor=detection["anchor"],
                field_point=detection["field_point"],
                frame_index=frame_index,
                feature=detection.get("identity_feature"),
                frame_size=self.frame_size,
            )
            self._tracks[track_id] = track
            assigned_track_ids[detection_index] = track_id
            self.new_track_count += 1

        return assigned_track_ids


def build_stitched_track_map(tracklets: dict[int, dict[str, Any]], fps: float) -> tuple[dict[int, int], dict[str, Any]]:
    if not tracklets:
        return {}, {"merge_count": 0, "raw_track_count": 0, "stitched_track_count": 0}

    max_gap_frames = max(75, int(round(float(fps) * 6.0)))
    sorted_track_ids = sorted(tracklets, key=lambda track_id: (tracklets[track_id]["first_frame"], track_id))
    candidate_links: list[tuple[float, int, int]] = []

    for previous_track_id in sorted_track_ids:
        previous_track = tracklets[previous_track_id]
        for next_track_id in sorted_track_ids:
            if next_track_id == previous_track_id:
                continue
            next_track = tracklets[next_track_id]
            gap = int(next_track["first_frame"]) - int(previous_track["last_frame"])
            if gap < 1 or gap > max_gap_frames:
                continue

            appearance_distance = _cosine_distance(previous_track.get("mean_feature"), next_track.get("mean_feature"))
            if appearance_distance is not None and appearance_distance > 0.34:
                continue

            previous_field = previous_track.get("last_field_point")
            next_field = next_track.get("first_field_point")
            continuity_score = 0.0
            if previous_field is not None and next_field is not None:
                field_distance = float(math.dist(previous_field, next_field))
                field_gate = 280.0 + gap * 85.0
                if field_distance > field_gate:
                    continue
                continuity_score = 1.0 - min(1.0, field_distance / max(field_gate, 1.0))
            else:
                previous_anchor = previous_track.get("last_anchor")
                next_anchor = next_track.get("first_anchor")
                if previous_anchor is None or next_anchor is None:
                    continue
                pixel_distance = float(math.dist(previous_anchor, next_anchor))
                pixel_gate = 240.0 + gap * 14.0
                if pixel_distance > pixel_gate:
                    continue
                continuity_score = 1.0 - min(1.0, pixel_distance / max(pixel_gate, 1.0))

            previous_area = max(float(previous_track.get("average_bbox_area") or 1.0), 1.0)
            next_area = max(float(next_track.get("average_bbox_area") or 1.0), 1.0)
            size_ratio = max(previous_area, next_area) / max(min(previous_area, next_area), 1.0)
            if size_ratio > 2.8:
                continue
            size_score = 1.0 - min(1.0, abs(math.log(size_ratio)) / 1.1)

            if appearance_distance is None:
                if gap > max(25, int(round(float(fps) * 1.5))) or continuity_score < 0.82:
                    continue
                appearance_score = 0.0
                score = (0.8 * continuity_score) + (0.2 * size_score) - 0.06
            else:
                appearance_score = 1.0 - appearance_distance
                if appearance_score < 0.66:
                    continue
                score = (0.58 * appearance_score) + (0.27 * continuity_score) + (0.15 * size_score)

            if score >= 0.72:
                candidate_links.append((float(score), previous_track_id, next_track_id))

    next_link: dict[int, int] = {}
    previous_link: dict[int, int] = {}
    for _score, previous_track_id, next_track_id in sorted(candidate_links, key=lambda item: item[0], reverse=True):
        if previous_track_id in next_link or next_track_id in previous_link:
            continue
        if int(tracklets[next_track_id]["first_frame"]) <= int(tracklets[previous_track_id]["last_frame"]):
            continue
        next_link[previous_track_id] = next_track_id
        previous_link[next_track_id] = previous_track_id

    stitched_track_map: dict[int, int] = {}
    merge_count = 0
    for track_id in sorted_track_ids:
        root = track_id
        while root in previous_link:
            root = previous_link[root]
        stitched_track_map[track_id] = root
        if root != track_id:
            merge_count += 1

    stitched_track_count = len(set(stitched_track_map.values()))
    return stitched_track_map, {
        "merge_count": merge_count,
        "raw_track_count": len(sorted_track_ids),
        "stitched_track_count": stitched_track_count,
        "max_gap_frames": max_gap_frames,
    }
