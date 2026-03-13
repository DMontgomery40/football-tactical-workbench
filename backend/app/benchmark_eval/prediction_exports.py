from __future__ import annotations

import importlib
import json
import shutil
import zipfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .common import BenchmarkPredictionExportUnavailable, ensure_dir


CALIBRATION_SPLIT = "valid"
CALIBRATION_RESOLUTION = (960, 540)
CALIBRATION_METADATA_FILENAMES = {
    "match_info.json",
    "per_match_info.json",
}
SYNLOC_SOURCE_FILENAME = "recipe_localization_predictions.json"
TEAM_SPOTTING_SOURCE_FILENAME = "recipe_event_spotting_predictions.json"
FOOTPASS_SOURCE_FILENAME = "recipe_playbyplay_predictions.json"
TRACKING_SOURCE_FILENAME = "recipe_tracking_predictions.json"
FOOTPASS_CLASS_IDS = {
    "drive": 1,
    "pass": 2,
    "cross": 3,
    "throw in": 4,
    "shot": 5,
    "header": 6,
    "tackle": 7,
    "block": 8,
}


def prepare_prediction_exports(
    *,
    suite: dict[str, Any],
    recipe: dict[str, Any],
    dataset_root: str,
    artifacts_dir: str | Path,
    benchmark_id: str,
) -> dict[str, Any]:
    protocol = str(suite.get("protocol") or "")
    if protocol == "synloc":
        return _prepare_synloc_exports(
            suite=suite,
            recipe=recipe,
            dataset_root=dataset_root,
            artifacts_dir=artifacts_dir,
            benchmark_id=benchmark_id,
        )
    if protocol == "calibration":
        return _prepare_calibration_exports(
            suite=suite,
            recipe=recipe,
            dataset_root=dataset_root,
            artifacts_dir=artifacts_dir,
            benchmark_id=benchmark_id,
        )
    if protocol == "team_spotting":
        return _prepare_team_spotting_exports(
            suite=suite,
            recipe=recipe,
            dataset_root=dataset_root,
            artifacts_dir=artifacts_dir,
            benchmark_id=benchmark_id,
        )
    if protocol == "pcbas":
        return _prepare_footpass_exports(
            suite=suite,
            recipe=recipe,
            dataset_root=dataset_root,
            artifacts_dir=artifacts_dir,
            benchmark_id=benchmark_id,
        )
    if protocol == "tracking":
        return _prepare_tracking_exports(
            suite=suite,
            recipe=recipe,
            dataset_root=dataset_root,
            artifacts_dir=artifacts_dir,
            benchmark_id=benchmark_id,
        )
    return {}


def ensure_calibration_prediction_export(
    *,
    recipe: dict[str, Any],
    dataset_root: str,
    artifacts_dir: str | Path,
) -> dict[str, Any]:
    prepared = _prepare_calibration_exports(
        suite={},
        recipe=recipe,
        dataset_root=dataset_root,
        artifacts_dir=artifacts_dir,
        benchmark_id="manual_calibration_export",
    )
    payload = dict((prepared.get("raw_result") or {}).get("prediction_export") or {})
    counts = dict(payload.get("counts") or {})
    artifacts = dict(prepared.get("artifacts") or {})
    return {
        "prediction_root": str(artifacts.get("predictions_root") or ""),
        "export_summary_json": str(artifacts.get("prediction_export_json") or ""),
        "exported_predictions": int(counts.get("generated_camera_files") or 0),
    }


def ensure_synloc_prediction_export(
    *,
    recipe: dict[str, Any] | None = None,
    dataset_root: str,
    artifacts_dir: str | Path,
) -> dict[str, Any]:
    prepared = _prepare_synloc_exports(
        suite={},
        recipe=dict(recipe or {}),
        dataset_root=dataset_root,
        artifacts_dir=artifacts_dir,
        benchmark_id="manual_synloc_export",
    )
    payload = dict((prepared.get("raw_result") or {}).get("prediction_export") or {})
    counts = dict(payload.get("counts") or {})
    artifacts = dict(prepared.get("artifacts") or {})
    return {
        "predictions_json": str(artifacts.get("predictions_json") or ""),
        "metadata_json": str(artifacts.get("metadata_json") or ""),
        "ground_truth_json": str(artifacts.get("ground_truth_json") or ""),
        "export_summary_json": str(artifacts.get("prediction_export_json") or ""),
        "images": int(counts.get("dataset_images") or 0),
        "predictions": int(counts.get("generated_predictions") or 0),
        "source_prediction_artifact": str(artifacts.get("prediction_source_json") or ""),
    }


def ensure_team_spotting_prediction_export(
    *,
    recipe: dict[str, Any] | None = None,
    dataset_root: str,
    artifacts_dir: str | Path,
) -> dict[str, Any]:
    prepared = _prepare_team_spotting_exports(
        suite={},
        recipe=dict(recipe or {}),
        dataset_root=dataset_root,
        artifacts_dir=artifacts_dir,
        benchmark_id="manual_team_spotting_export",
    )
    payload = dict((prepared.get("raw_result") or {}).get("prediction_export") or {})
    counts = dict(payload.get("counts") or {})
    artifacts = dict(prepared.get("artifacts") or {})
    return {
        "prediction_root": str(artifacts.get("predictions_root") or ""),
        "export_summary_json": str(artifacts.get("prediction_export_json") or ""),
        "games": int(counts.get("dataset_games") or 0),
        "events": int(counts.get("source_events") or 0),
    }


def ensure_footpass_prediction_export(
    *,
    recipe: dict[str, Any] | None = None,
    artifacts_dir: str | Path,
    dataset_root: str = "",
) -> dict[str, Any]:
    prepared = _prepare_footpass_exports(
        suite={},
        recipe=dict(recipe or {}),
        dataset_root=dataset_root,
        artifacts_dir=artifacts_dir,
        benchmark_id="manual_footpass_export",
    )
    payload = dict((prepared.get("raw_result") or {}).get("prediction_export") or {})
    counts = dict(payload.get("counts") or {})
    artifacts = dict(prepared.get("artifacts") or {})
    return {
        "predictions_json": str(artifacts.get("predictions_json") or ""),
        "export_summary_json": str(artifacts.get("prediction_export_json") or ""),
        "games": int(counts.get("source_keys") or 0),
        "events": int(counts.get("source_events") or 0),
        "source_prediction_artifact": str(artifacts.get("prediction_source_json") or ""),
    }


def ensure_tracking_prediction_export(
    *,
    recipe: dict[str, Any] | None = None,
    dataset_root: str,
    artifacts_dir: str | Path,
) -> dict[str, Any]:
    prepared = _prepare_tracking_exports(
        suite={},
        recipe=dict(recipe or {}),
        dataset_root=dataset_root,
        artifacts_dir=artifacts_dir,
        benchmark_id="manual_tracking_export",
    )
    payload = dict((prepared.get("raw_result") or {}).get("prediction_export") or {})
    counts = dict(payload.get("counts") or {})
    artifacts = dict(prepared.get("artifacts") or {})
    return {
        "tracker_submission_zip": str(artifacts.get("tracker_submission_zip") or ""),
        "tracker_submission_dir": str(artifacts.get("tracker_submission_dir") or ""),
        "export_summary_json": str(artifacts.get("prediction_export_json") or ""),
        "sequences": int(counts.get("emitted_sequence_files") or 0),
        "detections": int(counts.get("source_detections") or 0),
        "source_prediction_artifact": str(artifacts.get("prediction_source_json") or ""),
    }


def _base_manifest(
    *,
    protocol: str,
    suite: dict[str, Any],
    recipe: dict[str, Any],
    dataset_root: str,
    artifacts_dir: Path,
    benchmark_id: str,
) -> dict[str, Any]:
    return {
        "benchmark_id": benchmark_id,
        "suite_id": str(suite.get("id") or ""),
        "recipe_id": str(recipe.get("id") or ""),
        "protocol": protocol,
        "dataset_root": str(Path(dataset_root).expanduser().resolve()) if str(dataset_root).strip() else "",
        "artifacts_dir": str(artifacts_dir),
        "status": "ready",
        "blockers": [],
        "counts": {},
        "inputs": {},
        "outputs": {},
        "notes": [],
    }


def _write_manifest(artifacts_dir: Path, payload: dict[str, Any]) -> Path:
    manifest_path = artifacts_dir / "prediction_export.json"
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return manifest_path


def _raise_export_blocker(
    *,
    artifacts_dir: Path,
    payload: dict[str, Any],
    message: str,
    extra_artifacts: dict[str, Any] | None = None,
) -> None:
    payload["status"] = "blocked"
    payload["blockers"] = [str(message)]
    manifest_path = _write_manifest(artifacts_dir, payload)
    raise BenchmarkPredictionExportUnavailable(
        message,
        artifacts={
            "prediction_export_json": str(manifest_path),
            **dict(extra_artifacts or {}),
        },
        raw_result={"prediction_export": payload},
    )


def _resolve_source_json(
    *,
    recipe: dict[str, Any],
    artifacts_path: Path,
    filename: str,
    candidate_keys: tuple[str, ...],
) -> Path:
    candidates: list[Path] = []
    for key in candidate_keys:
        raw_value = str(recipe.get(key) or "").strip()
        if raw_value:
            candidates.append(Path(raw_value).expanduser().resolve())

    artifact_path_raw = str(recipe.get("artifact_path") or "").strip()
    if artifact_path_raw:
        artifact_path = Path(artifact_path_raw).expanduser().resolve()
        if artifact_path.is_file() and artifact_path.suffix.lower() == ".json":
            candidates.append(artifact_path)
        elif artifact_path.is_dir():
            candidates.append(artifact_path / filename)
            candidates.append(artifact_path / "predictions" / filename)

    candidates.append(artifacts_path / filename)

    seen: set[str] = set()
    normalized: list[Path] = []
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        normalized.append(candidate)

    for candidate in normalized:
        if candidate.exists():
            return candidate
    return normalized[-1]


def _prepare_calibration_exports(
    *,
    suite: dict[str, Any],
    recipe: dict[str, Any],
    dataset_root: str,
    artifacts_dir: str | Path,
    benchmark_id: str,
) -> dict[str, Any]:
    artifacts_path = ensure_dir(artifacts_dir)
    dataset_path = Path(dataset_root).expanduser().resolve()
    split_dir = dataset_path / CALIBRATION_SPLIT
    predictions_root = ensure_dir(artifacts_path / "predictions")
    prediction_split_dir = ensure_dir(predictions_root / CALIBRATION_SPLIT)

    payload = _base_manifest(
        protocol="calibration",
        suite=suite,
        recipe=recipe,
        dataset_root=str(dataset_path),
        artifacts_dir=artifacts_path,
        benchmark_id=benchmark_id,
    )
    payload["inputs"]["split_dir"] = str(split_dir)
    payload["outputs"]["predictions_root"] = str(predictions_root)
    payload["outputs"]["prediction_split_dir"] = str(prediction_split_dir)

    if not split_dir.exists():
        _raise_export_blocker(
            artifacts_dir=artifacts_path,
            payload=payload,
            message=(
                "Calibration export requires a split directory at "
                f"{split_dir} so camera_<frame_id>.json files can be generated."
            ),
            extra_artifacts={"predictions_root": str(predictions_root)},
        )

    annotation_files = sorted(
        path
        for path in split_dir.glob("*.json")
        if path.is_file() and path.name not in CALIBRATION_METADATA_FILENAMES
    )
    payload["counts"]["annotation_files"] = len(annotation_files)
    if not annotation_files:
        _raise_export_blocker(
            artifacts_dir=artifacts_path,
            payload=payload,
            message=(
                "Calibration export found no annotation JSON files under "
                f"{split_dir}; expected files like <frame_id>.json beside the source images."
            ),
            extra_artifacts={"predictions_root": str(predictions_root)},
        )

    frame_inputs: list[tuple[str, Path]] = []
    missing_images: list[str] = []
    for annotation_file in annotation_files:
        frame_id = annotation_file.stem
        image_path = _find_image_file(split_dir, frame_id)
        if image_path is None:
            missing_images.append(frame_id)
            continue
        frame_inputs.append((frame_id, image_path))

    payload["counts"]["frames_with_images"] = len(frame_inputs)
    if missing_images:
        payload["inputs"]["missing_image_frame_ids"] = missing_images[:10]
        _raise_export_blocker(
            artifacts_dir=artifacts_path,
            payload=payload,
            message=(
                "Calibration export needs the source frame images beside the annotation JSONs. "
                f"Missing {len(missing_images)} frame image(s) under {split_dir}; "
                f"examples: {', '.join(missing_images[:5])}."
            ),
            extra_artifacts={"predictions_root": str(predictions_root)},
        )

    predictor = _build_calibration_predictor(recipe)
    generated_files: list[str] = []
    skipped_frames: list[dict[str, Any]] = []
    for frame_id, image_path in frame_inputs:
        output_path = prediction_split_dir / f"camera_{frame_id}.json"
        if output_path.exists():
            generated_files.append(str(output_path))
            continue

        camera_payload = _camera_payload_from_frame(image_path=image_path, predictor=predictor)
        if camera_payload is None:
            skipped_frames.append({
                "frame_id": frame_id,
                "reason": "no_usable_homography",
            })
            continue

        output_path.write_text(json.dumps(camera_payload, indent=2), encoding="utf-8")
        generated_files.append(str(output_path))

    payload["counts"]["generated_camera_files"] = len(generated_files)
    payload["counts"]["skipped_frames"] = len(skipped_frames)
    payload["outputs"]["generated_camera_files"] = generated_files[:20]
    if skipped_frames:
        payload["status"] = "partial"
        payload["notes"].append(
            "Some frames did not yield a usable camera calibration and were left as missed predictions for the evaluator."
        )
        payload["outputs"]["skipped_frames"] = skipped_frames[:20]

    if not generated_files:
        _raise_export_blocker(
            artifacts_dir=artifacts_path,
            payload=payload,
            message=(
                "Calibration export produced 0 camera prediction files. "
                f"Expected at least one file under {prediction_split_dir} named camera_<frame_id>.json."
            ),
            extra_artifacts={"predictions_root": str(predictions_root)},
        )

    manifest_path = _write_manifest(artifacts_path, payload)
    return {
        "artifacts": {
            "prediction_export_json": str(manifest_path),
            "predictions_root": str(predictions_root),
        },
        "raw_result": {"prediction_export": payload},
    }


def _find_image_file(split_dir: Path, frame_id: str) -> Path | None:
    for suffix in (".jpg", ".jpeg", ".png", ".bmp"):
        candidate = split_dir / f"{frame_id}{suffix}"
        if candidate.exists():
            return candidate
    return None


def _build_calibration_predictor(recipe: dict[str, Any]) -> dict[str, Any]:
    from ultralytics import YOLO

    from app.wide_angle import (
        choose_device,
        choose_keypoint_device,
        resolve_model_path,
    )

    detector_device = choose_device()
    keypoint_device = choose_keypoint_device(detector_device)
    pipeline = str(recipe.get("pipeline") or "classic")
    keypoint_model_name = "soccermaster" if pipeline == "soccermaster" else str(recipe.get("keypoint_model") or "soccana_keypoint")
    if keypoint_model_name == "soccermaster":
        keypoint_model: Any = "soccermaster"
    else:
        keypoint_model = YOLO(resolve_model_path(keypoint_model_name, "keypoint"))
    return {
        "keypoint_model": keypoint_model,
        "keypoint_device": keypoint_device,
    }


def _camera_payload_from_frame(*, image_path: Path, predictor: dict[str, Any]) -> dict[str, Any] | None:
    from app.wide_angle import detect_pitch_homography

    frame = cv2.imread(str(image_path))
    if frame is None:
        raise ValueError(f"cv2.imread failed for {image_path}")

    resized = cv2.resize(frame, CALIBRATION_RESOLUTION, interpolation=cv2.INTER_LINEAR)
    homography_image_to_pipeline, _, _, _, _ = detect_pitch_homography(
        resized,
        predictor["keypoint_model"],
        str(predictor["keypoint_device"]),
    )
    if homography_image_to_pipeline is None:
        return None

    return _camera_payload_from_pipeline_homography(
        homography_image_to_pipeline=np.asarray(homography_image_to_pipeline, dtype=np.float64),
        frame_width=CALIBRATION_RESOLUTION[0],
        frame_height=CALIBRATION_RESOLUTION[1],
    )


def _camera_payload_from_pipeline_homography(
    *,
    homography_image_to_pipeline: np.ndarray,
    frame_width: int,
    frame_height: int,
) -> dict[str, Any] | None:
    import sys

    repo_dir = Path(__file__).resolve().parents[2]
    calibration_repo = repo_dir / "third_party" / "soccernet" / "sn-calibration"
    if str(calibration_repo) not in sys.path:
        sys.path.insert(0, str(calibration_repo))
    Camera = getattr(importlib.import_module("src.camera"), "Camera")

    if homography_image_to_pipeline.shape != (3, 3):
        return None

    world_to_pipeline = np.array(
        [
            [12000.0 / 105.0, 0.0, 6000.0],
            [0.0, 7000.0 / 68.0, 3500.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    try:
        homography_world_to_image = np.linalg.inv(homography_image_to_pipeline) @ world_to_pipeline
    except np.linalg.LinAlgError:
        return None
    if abs(float(homography_world_to_image[2, 2])) > 1e-9:
        homography_world_to_image = homography_world_to_image / float(homography_world_to_image[2, 2])

    camera = Camera(frame_width, frame_height)
    if not camera.from_homography(homography_world_to_image):
        return None
    return camera.to_json_parameters()


def _prepare_synloc_exports(
    *,
    suite: dict[str, Any],
    recipe: dict[str, Any],
    dataset_root: str,
    artifacts_dir: str | Path,
    benchmark_id: str,
) -> dict[str, Any]:
    artifacts_path = ensure_dir(artifacts_dir)
    dataset_path = Path(dataset_root).expanduser().resolve()
    predictions_path = artifacts_path / "results.json"
    metadata_path = artifacts_path / "metadata.json"
    ground_truth_path = _resolve_synloc_ground_truth(dataset_path)
    source_path = _resolve_source_json(
        recipe=recipe,
        artifacts_path=artifacts_path,
        filename=SYNLOC_SOURCE_FILENAME,
        candidate_keys=(
            "synloc_prediction_json",
            "localization_prediction_json",
            "prediction_json",
            "source_prediction_json",
        ),
    )

    payload = _base_manifest(
        protocol="synloc",
        suite=suite,
        recipe=recipe,
        dataset_root=str(dataset_path),
        artifacts_dir=artifacts_path,
        benchmark_id=benchmark_id,
    )
    payload["inputs"]["ground_truth_json"] = str(ground_truth_path)
    payload["inputs"]["source_prediction_json"] = str(source_path)
    payload["outputs"]["predictions_json"] = str(predictions_path)
    payload["outputs"]["metadata_json"] = str(metadata_path)

    if not ground_truth_path.exists():
        _raise_export_blocker(
            artifacts_dir=artifacts_path,
            payload=payload,
            message=(
                "SynLoc export requires a COCO-style ground-truth annotations file at "
                f"{ground_truth_path} with `images`, `annotations`, and per-image camera fields."
            ),
            extra_artifacts={
                "ground_truth_json": str(ground_truth_path),
                "prediction_source_json": str(source_path),
            },
        )

    try:
        ground_truth_payload = _load_json_file(ground_truth_path)
    except ValueError as exc:
        _raise_export_blocker(
            artifacts_dir=artifacts_path,
            payload=payload,
            message=str(exc),
            extra_artifacts={
                "ground_truth_json": str(ground_truth_path),
                "prediction_source_json": str(source_path),
            },
        )
    if not isinstance(ground_truth_payload, dict):
        _raise_export_blocker(
            artifacts_dir=artifacts_path,
            payload=payload,
            message=(
                "SynLoc export expects a COCO-style JSON object at "
                f"{ground_truth_path}, but found a JSON array."
            ),
            extra_artifacts={
                "ground_truth_json": str(ground_truth_path),
                "prediction_source_json": str(source_path),
            },
        )

    images = ground_truth_payload.get("images")
    if not isinstance(images, list) or not images:
        _raise_export_blocker(
            artifacts_dir=artifacts_path,
            payload=payload,
            message=(
                "SynLoc export requires at least one image entry in the COCO-style annotations file at "
                f"{ground_truth_path}."
            ),
            extra_artifacts={
                "ground_truth_json": str(ground_truth_path),
                "prediction_source_json": str(source_path),
            },
        )
    payload["counts"]["dataset_images"] = len(images)

    if source_path.exists():
        try:
            source_payload = _load_json_file(source_path)
            result_records, metadata_payload = _normalize_synloc_source(
                payload=source_payload,
                dataset_images=images,
            )
        except ValueError as exc:
            _raise_export_blocker(
                artifacts_dir=artifacts_path,
                payload=payload,
                message=str(exc),
                extra_artifacts={
                    "ground_truth_json": str(ground_truth_path),
                    "prediction_source_json": str(source_path),
                },
            )
    else:
        try:
            generated = _generate_synloc_predictions_from_recipe(
                recipe=recipe,
                dataset_path=dataset_path,
                ground_truth_payload=ground_truth_payload,
            )
        except ValueError as exc:
            _raise_export_blocker(
                artifacts_dir=artifacts_path,
                payload=payload,
                message=str(exc),
                extra_artifacts={
                    "ground_truth_json": str(ground_truth_path),
                    "prediction_source_json": str(source_path),
                },
            )
        result_records = list(generated.get("results") or [])
        metadata_payload = {
            "score_threshold": None,
            "position_from_keypoint_index": None,
        }
        payload["counts"]["source_images_with_detections"] = int(generated.get("images_with_detections") or 0)
        payload["counts"]["source_raw_detections"] = int(generated.get("raw_detections") or 0)
        payload["notes"].append(
            "SynLoc predictions were generated directly from the recipe over the dataset images because no staged localization JSON was supplied."
        )

    predictions_path.write_text(json.dumps(result_records, indent=2), encoding="utf-8")
    metadata_path.write_text(json.dumps(metadata_payload, indent=2), encoding="utf-8")
    payload["counts"]["generated_predictions"] = len(result_records)
    payload["outputs"]["predictions_json"] = str(predictions_path)
    payload["outputs"]["metadata_json"] = str(metadata_path)
    payload["outputs"]["ground_truth_json"] = str(ground_truth_path)
    if not result_records:
        payload["status"] = "partial"
        payload["notes"].append(
            "The export produced an empty results.json file because no valid localized detections were generated."
        )

    manifest_path = _write_manifest(artifacts_path, payload)
    return {
        "artifacts": {
            "prediction_export_json": str(manifest_path),
            "predictions_json": str(predictions_path),
            "metadata_json": str(metadata_path),
            "ground_truth_json": str(ground_truth_path),
            "prediction_source_json": str(source_path),
        },
        "raw_result": {"prediction_export": payload},
    }


def _resolve_synloc_ground_truth(dataset_path: Path) -> Path:
    candidates = [
        dataset_path / "annotations" / "val.json",
        dataset_path / "annotations" / "validation.json",
        dataset_path / "val.json",
        dataset_path / "validation.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _normalize_synloc_source(
    *,
    payload: dict[str, Any] | list[Any],
    dataset_images: list[Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if isinstance(payload, list):
        records_payload = payload
        metadata_payload: dict[str, Any] = {
            "score_threshold": None,
            "position_from_keypoint_index": None,
        }
    else:
        records_payload = payload.get("results")
        if not isinstance(records_payload, list):
            records_payload = payload.get("predictions")
        if not isinstance(records_payload, list):
            records_payload = payload.get("detections")
        if not isinstance(records_payload, list):
            raise ValueError(
                "SynLoc export source must contain a `results`, `predictions`, or `detections` array."
            )
        metadata_raw = payload.get("metadata")
        metadata_payload = dict(metadata_raw) if isinstance(metadata_raw, dict) else {}
        if "score_threshold" not in metadata_payload:
            metadata_payload["score_threshold"] = payload.get("score_threshold")
        if "position_from_keypoint_index" not in metadata_payload:
            metadata_payload["position_from_keypoint_index"] = payload.get("position_from_keypoint_index")

    image_ids = {
        int(image.get("id"))
        for image in dataset_images
        if isinstance(image, dict) and image.get("id") is not None
    }
    normalized_records: list[dict[str, Any]] = []
    for index, record in enumerate(records_payload, start=1):
        normalized = _synloc_result_record(record, result_id=index)
        if normalized is None:
            continue
        if int(normalized["image_id"]) not in image_ids:
            raise ValueError(
                "SynLoc export source references an image_id that is not present in the dataset annotations: "
                f"{normalized['image_id']}."
            )
        normalized_records.append(normalized)

    normalized_metadata = {
        "score_threshold": _coerce_optional_float(metadata_payload.get("score_threshold")),
        "position_from_keypoint_index": _coerce_optional_int(metadata_payload.get("position_from_keypoint_index")),
    }
    return normalized_records, normalized_metadata


def _synloc_result_record(record: Any, *, result_id: int) -> dict[str, Any] | None:
    if not isinstance(record, dict):
        return None
    try:
        image_id = int(record.get("image_id"))
        score = float(record.get("score", record.get("confidence")))
    except (TypeError, ValueError):
        return None
    bbox = _synloc_bbox(record.get("bbox"))
    position = _synloc_position(record.get("position_on_pitch"))
    if bbox is None or position is None:
        return None
    area = record.get("area")
    try:
        area_value = float(area) if area is not None else float(bbox[2] * bbox[3])
    except (TypeError, ValueError):
        area_value = float(bbox[2] * bbox[3])
    return {
        "id": int(result_id),
        "image_id": image_id,
        "category_id": int(record.get("category_id") or 1),
        "bbox": bbox,
        "area": area_value,
        "score": score,
        "position_on_pitch": position,
    }


def _synloc_bbox(value: Any) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        x, y, width, height = [float(item) for item in value]
    except (TypeError, ValueError):
        return None
    if width <= 0.0 or height <= 0.0:
        return None
    return [x, y, width, height]


def _synloc_position(value: Any) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    try:
        x, y = float(value[0]), float(value[1])
    except (TypeError, ValueError):
        return None
    return [x, y]


def _coerce_optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _generate_synloc_predictions_from_recipe(
    *,
    recipe: dict[str, Any],
    dataset_path: Path,
    ground_truth_payload: dict[str, Any],
) -> dict[str, Any]:
    images = ground_truth_payload.get("images")
    if not isinstance(images, list):
        raise ValueError("SynLoc export expected a COCO-style `images` array in the ground-truth annotations.")

    predictor = _build_synloc_predictor(recipe)
    generated: list[dict[str, Any]] = []
    missing_image_files: list[str] = []
    missing_projection_images: list[str] = []
    raw_detections = 0
    images_with_detections = 0

    for image in images:
        if not isinstance(image, dict):
            continue
        image_path = _resolve_synloc_image_path(dataset_path=dataset_path, image=image)
        if image_path is None or not image_path.exists():
            missing_image_files.append(str(image.get("file_name") or image.get("id") or "unknown"))
            continue

        frame = cv2.imread(str(image_path))
        if frame is None:
            raise ValueError(f"cv2.imread failed for {image_path}")

        detections = _run_synloc_detector_on_image(frame=frame, predictor=predictor)
        if detections:
            images_with_detections += 1
        raw_detections += len(detections)

        for detection in detections:
            position_on_pitch = _project_synloc_anchor_to_pitch(
                image=image,
                anchor=detection["anchor"],
            )
            if position_on_pitch is None:
                missing_projection_images.append(str(image.get("file_name") or image.get("id") or "unknown"))
                continue
            generated.append(
                {
                    "id": len(generated) + 1,
                    "image_id": int(image.get("id")),
                    "category_id": 1,
                    "bbox": [float(value) for value in detection["bbox"]],
                    "area": float(detection["bbox"][2] * detection["bbox"][3]),
                    "score": float(detection["score"]),
                    "position_on_pitch": position_on_pitch,
                }
            )

    if missing_image_files:
        examples = ", ".join(missing_image_files[:5])
        raise ValueError(
            "SynLoc export could not locate the validation image files referenced by the dataset annotations. "
            f"Missing {len(missing_image_files)} file(s) under {dataset_path}; examples: {examples}."
        )
    if missing_projection_images:
        examples = ", ".join(sorted(dict.fromkeys(missing_projection_images))[:5])
        raise ValueError(
            "SynLoc export could not project detections onto the pitch because the dataset annotations are missing "
            "camera fields (`camera_matrix`, `undist_poly`, `width`, `height`) for some images. "
            f"Examples: {examples}."
        )

    return {
        "results": generated,
        "raw_detections": raw_detections,
        "images_with_detections": images_with_detections,
    }


def _build_synloc_predictor(recipe: dict[str, Any]) -> dict[str, Any]:
    from ultralytics import YOLO

    from app.wide_angle import choose_device, resolve_detector_spec

    detector_path = str(recipe.get("artifact_path") or "").strip()
    if not detector_path:
        raise ValueError(
            "SynLoc export needs a detector-backed recipe with an `artifact_path` that points to a YOLO checkpoint."
        )
    detector_spec = resolve_detector_spec(detector_path)
    return {
        "detector_model": YOLO(detector_path),
        "detector_device": choose_device(),
        "detector_spec": detector_spec,
    }


def _run_synloc_detector_on_image(*, frame: np.ndarray, predictor: dict[str, Any]) -> list[dict[str, Any]]:
    results = predictor["detector_model"](
        source=frame,
        conf=0.25,
        iou=0.50,
        device=str(predictor["detector_device"]),
        classes=list((predictor.get("detector_spec") or {}).get("player_class_ids") or []),
        verbose=False,
    )
    boxes = results[0].boxes if results else None
    if boxes is None or len(boxes) == 0:
        return []
    xyxy = boxes.xyxy.cpu().numpy()
    confidences = boxes.conf.cpu().numpy() if boxes.conf is not None else np.ones(len(xyxy), dtype=np.float32)
    detections: list[dict[str, Any]] = []
    for index, box in enumerate(xyxy):
        x1, y1, x2, y2 = [float(value) for value in box.tolist()]
        width = max(x2 - x1, 0.0)
        height = max(y2 - y1, 0.0)
        if width <= 0.0 or height <= 0.0:
            continue
        detections.append(
            {
                "bbox": [x1, y1, width, height],
                "score": float(confidences[index]),
                "anchor": [float((x1 + x2) / 2.0), float(y2)],
            }
        )
    return detections


def _resolve_synloc_image_path(*, dataset_path: Path, image: dict[str, Any]) -> Path | None:
    file_name = str(image.get("file_name") or "").strip()
    if not file_name:
        return None
    candidates = [
        dataset_path / file_name,
        dataset_path / "val" / file_name,
        dataset_path / "validation" / file_name,
        dataset_path / "images" / file_name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _project_synloc_anchor_to_pitch(*, image: dict[str, Any], anchor: list[float]) -> list[float] | None:
    try:
        from sskit import image_to_ground
    except Exception as exc:  # pragma: no cover - runtime-specific dependency
        raise ValueError(f"SynLoc export could not import sskit.image_to_ground: {exc}") from exc

    camera_matrix = image.get("camera_matrix")
    undist_poly = image.get("undist_poly")
    width = image.get("width")
    height = image.get("height")
    if camera_matrix is None or undist_poly is None or width is None or height is None:
        return None
    try:
        width_value = float(width)
        height_value = float(height)
        camera_matrix_np = np.asarray(camera_matrix, dtype=np.float32)
        undist_poly_np = np.asarray(undist_poly, dtype=np.float32)
        anchor_np = np.asarray([[float(anchor[0]), float(anchor[1])]], dtype=np.float32)
    except (TypeError, ValueError):
        return None

    normalized_anchor = (
        anchor_np - np.asarray([[(width_value - 1.0) / 2.0, (height_value - 1.0) / 2.0]], dtype=np.float32)
    ) / width_value
    projected = image_to_ground(camera_matrix_np, undist_poly_np, normalized_anchor)
    if projected is None or len(projected) == 0:
        return None
    return [float(projected[0][0]), float(projected[0][1])]


def _prepare_team_spotting_exports(
    *,
    suite: dict[str, Any],
    recipe: dict[str, Any],
    dataset_root: str,
    artifacts_dir: str | Path,
    benchmark_id: str,
) -> dict[str, Any]:
    artifacts_path = ensure_dir(artifacts_dir)
    dataset_path = Path(dataset_root).expanduser().resolve()
    predictions_root = ensure_dir(artifacts_path / "predictions")
    source_path = _resolve_source_json(
        recipe=recipe,
        artifacts_path=artifacts_path,
        filename=TEAM_SPOTTING_SOURCE_FILENAME,
        candidate_keys=(
            "team_spotting_prediction_json",
            "prediction_json",
            "source_prediction_json",
        ),
    )

    payload = _base_manifest(
        protocol="team_spotting",
        suite=suite,
        recipe=recipe,
        dataset_root=str(dataset_path),
        artifacts_dir=artifacts_path,
        benchmark_id=benchmark_id,
    )
    payload["inputs"]["source_prediction_json"] = str(source_path)
    payload["outputs"]["predictions_root"] = str(predictions_root)

    if not source_path.exists():
        _raise_export_blocker(
            artifacts_dir=artifacts_path,
            payload=payload,
            message=(
                "Team spotting export source is missing. Expected a repo-owned event JSON at "
                f"{source_path} with game-keyed predictions that can be converted into "
                f"{predictions_root}/<game>/results_spotting.json. "
                "Each event must include at least `position_ms`, `label`, `team`, and `confidence`."
            ),
            extra_artifacts={
                "predictions_root": str(predictions_root),
                "prediction_source_json": str(source_path),
            },
        )

    game_paths = sorted(
        str(path.parent.relative_to(dataset_path))
        for path in dataset_path.rglob("Labels-ball.json")
        if path.is_file()
    )
    payload["counts"]["dataset_games"] = len(game_paths)
    if not game_paths:
        _raise_export_blocker(
            artifacts_dir=artifacts_path,
            payload=payload,
            message=(
                "Team spotting export requires a dataset root containing per-game Labels-ball.json files. "
                f"No Labels-ball.json files were found under {dataset_path}."
            ),
            extra_artifacts={"predictions_root": str(predictions_root)},
        )

    try:
        source_payload = _load_json_file(source_path)
    except ValueError as exc:
        _raise_export_blocker(
            artifacts_dir=artifacts_path,
            payload=payload,
            message=str(exc),
            extra_artifacts={
                "predictions_root": str(predictions_root),
                "prediction_source_json": str(source_path),
            },
        )
    games_payload = _normalize_team_spotting_source(source_payload)
    if not games_payload:
        _raise_export_blocker(
            artifacts_dir=artifacts_path,
            payload=payload,
            message=(
                "Team spotting export source is empty. Expected at least one game entry in "
                f"{source_path}."
            ),
            extra_artifacts={
                "predictions_root": str(predictions_root),
                "prediction_source_json": str(source_path),
            },
        )

    generated_files: list[str] = []
    source_events = 0
    for game_path in game_paths:
        output_dir = ensure_dir(predictions_root / game_path)
        normalized_predictions = [
            _team_spotting_prediction_record(event)
            for event in games_payload.get(game_path, [])
        ]
        prediction_rows = [item for item in normalized_predictions if item is not None]
        source_events += len(prediction_rows)
        output_payload = {
            "UrlLocal": game_path,
            "predictions": prediction_rows,
        }
        output_path = output_dir / "results_spotting.json"
        output_path.write_text(json.dumps(output_payload, indent=2), encoding="utf-8")
        generated_files.append(str(output_path))

    payload["counts"]["generated_prediction_files"] = len(generated_files)
    payload["counts"]["games_with_source_events"] = sum(1 for game_path in game_paths if games_payload.get(game_path))
    payload["counts"]["source_events"] = source_events
    payload["outputs"]["generated_prediction_files"] = generated_files[:20]
    if payload["counts"]["games_with_source_events"] == 0:
        payload["status"] = "partial"
        payload["notes"].append(
            "The export produced empty results_spotting.json files for all dataset games because the source file did not contain any events for them."
        )

    manifest_path = _write_manifest(artifacts_path, payload)
    return {
        "artifacts": {
            "prediction_export_json": str(manifest_path),
            "predictions_root": str(predictions_root),
            "prediction_source_json": str(source_path),
        },
        "raw_result": {"prediction_export": payload},
    }


def _load_json_file(path: Path) -> dict[str, Any] | list[Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Prediction export source is not valid JSON: {path} ({exc})") from exc
    if not isinstance(payload, (dict, list)):
        raise ValueError(f"Prediction export source must be a JSON object or array: {path}")
    return payload


def _normalize_team_spotting_source(payload: dict[str, Any] | list[Any]) -> dict[str, list[dict[str, Any]]]:
    if isinstance(payload, list):
        normalized: dict[str, list[dict[str, Any]]] = {}
        for game in payload:
            if not isinstance(game, dict):
                continue
            game_path = str(game.get("video") or game.get("game") or "").strip()
            if not game_path:
                continue
            events = game.get("events")
            normalized[game_path] = list(events) if isinstance(events, list) else []
        return normalized

    games = payload.get("games")
    if isinstance(games, dict):
        return {
            str(game_path): list(events) if isinstance(events, list) else []
            for game_path, events in games.items()
        }
    if isinstance(games, list):
        normalized = {}
        for game in games:
            if not isinstance(game, dict):
                continue
            game_path = str(game.get("video") or game.get("game") or game.get("key") or "").strip()
            if not game_path:
                continue
            events = game.get("events")
            normalized[game_path] = list(events) if isinstance(events, list) else []
        return normalized
    return {}


def _normalize_team_side(value: Any) -> str:
    normalized = str(value).strip().lower()
    if normalized in {"0", "home", "left"}:
        return "left"
    if normalized in {"1", "away", "right"}:
        return "right"
    return normalized


def _normalize_team_spotting_label(value: Any) -> str:
    return " ".join(str(value).strip().replace("_", " ").replace("-", " ").split()).upper()


def _team_spotting_prediction_record(event: Any) -> dict[str, Any] | None:
    if not isinstance(event, dict):
        return None
    try:
        confidence = float(event.get("confidence", event.get("score")))
    except (TypeError, ValueError):
        return None

    if event.get("position_ms") is not None:
        try:
            position_ms = int(event.get("position_ms"))
        except (TypeError, ValueError):
            return None
    elif event.get("frame") is not None:
        try:
            position_ms = int(int(event.get("frame")) / 25.0 * 1000.0)
        except (TypeError, ValueError):
            return None
    else:
        return None

    team = _normalize_team_side(event.get("team"))
    label = _normalize_team_spotting_label(event.get("label"))
    half = int(event.get("half") or 1)
    minutes = max(position_ms, 0) // 60000
    seconds = (max(position_ms, 0) % 60000) // 1000
    return {
        "gameTime": f"{half} - {minutes}:{seconds:02d}",
        "label": label,
        "position": position_ms,
        "half": half,
        "confidence": confidence,
        "team": team,
    }


def _prepare_footpass_exports(
    *,
    suite: dict[str, Any],
    recipe: dict[str, Any],
    dataset_root: str,
    artifacts_dir: str | Path,
    benchmark_id: str,
) -> dict[str, Any]:
    artifacts_path = ensure_dir(artifacts_dir)
    dataset_path = Path(dataset_root).expanduser().resolve() if str(dataset_root).strip() else Path(dataset_root)
    predictions_path = artifacts_path / "predictions.json"
    source_path = _resolve_source_json(
        recipe=recipe,
        artifacts_path=artifacts_path,
        filename=FOOTPASS_SOURCE_FILENAME,
        candidate_keys=(
            "pcbas_prediction_json",
            "playbyplay_prediction_json",
            "prediction_json",
            "source_prediction_json",
        ),
    )

    payload = _base_manifest(
        protocol="pcbas",
        suite=suite,
        recipe=recipe,
        dataset_root=str(dataset_path) if str(dataset_root).strip() else "",
        artifacts_dir=artifacts_path,
        benchmark_id=benchmark_id,
    )
    payload["inputs"]["source_prediction_json"] = str(source_path)
    payload["outputs"]["predictions_json"] = str(predictions_path)

    ground_truth_path = _resolve_footpass_ground_truth(dataset_path)
    payload["inputs"]["ground_truth_json"] = str(ground_truth_path)
    required_keys: list[str] = []
    if ground_truth_path.exists():
        gt_payload = json.loads(ground_truth_path.read_text(encoding="utf-8"))
        if isinstance(gt_payload, dict):
            required_keys = [str(key) for key in (gt_payload.get("keys") or []) if str(key)]
    payload["counts"]["ground_truth_keys"] = len(required_keys)

    if not source_path.exists():
        _raise_export_blocker(
            artifacts_dir=artifacts_path,
            payload=payload,
            message=(
                "FOOTPASS export source is missing. Expected a repo-owned play-by-play JSON at "
                f"{source_path} that can be converted into {predictions_path}. "
                "Each predicted event must resolve to `(frame, team, shirt, class, confidence)`. "
                "The vendored evaluator runtime is available, but the upstream FOOTPASS baseline helper stack is still "
                "not turnkey on this macOS arm64 host because decord==0.6.0 has no matching wheel here."
            ),
            extra_artifacts={
                "predictions_json": str(predictions_path),
                "prediction_source_json": str(source_path),
            },
        )

    try:
        source_payload = _load_json_file(source_path)
        normalized_payload = _normalize_footpass_source(source_payload)
    except ValueError as exc:
        _raise_export_blocker(
            artifacts_dir=artifacts_path,
            payload=payload,
            message=str(exc),
            extra_artifacts={
                "predictions_json": str(predictions_path),
                "prediction_source_json": str(source_path),
            },
        )
    payload["counts"]["source_keys"] = len(normalized_payload["keys"])
    payload["counts"]["source_events"] = int(
        sum(len(events) for events in (normalized_payload.get("events") or {}).values())
    )
    if not normalized_payload["keys"]:
        _raise_export_blocker(
            artifacts_dir=artifacts_path,
            payload=payload,
            message=(
                "FOOTPASS export source is empty. Expected at least one game-half key in "
                f"{source_path}."
            ),
            extra_artifacts={
                "predictions_json": str(predictions_path),
                "prediction_source_json": str(source_path),
            },
        )

    predictions_path.write_text(json.dumps(normalized_payload, indent=2), encoding="utf-8")
    payload["outputs"]["prediction_keys"] = normalized_payload["keys"][:20]
    if required_keys:
        missing_keys = [key for key in required_keys if key not in set(normalized_payload["keys"])]
        payload["counts"]["missing_ground_truth_keys"] = len(missing_keys)
        if missing_keys:
            payload["status"] = "partial"
            payload["notes"].append(
                "The play-by-play export does not include every ground-truth game-half key; the evaluator will count the missing keys as false negatives."
            )
            payload["outputs"]["missing_ground_truth_keys"] = missing_keys[:20]

    manifest_path = _write_manifest(artifacts_path, payload)
    return {
        "artifacts": {
            "prediction_export_json": str(manifest_path),
            "predictions_json": str(predictions_path),
            "prediction_source_json": str(source_path),
        },
        "raw_result": {"prediction_export": payload},
    }


def _resolve_footpass_ground_truth(dataset_path: Path) -> Path:
    if dataset_path.is_file():
        return dataset_path
    candidates = [
        dataset_path / "playbyplay_GT" / "playbyplay_val.json",
        dataset_path / "playbyplay_val.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _normalize_footpass_source(payload: dict[str, Any] | list[Any]) -> dict[str, Any]:
    if isinstance(payload, dict) and "keys" in payload and "events" in payload and isinstance(payload.get("events"), dict):
        keys = [str(key) for key in (payload.get("keys") or []) if str(key)]
        events_payload = payload.get("events") or {}
        normalized_events: dict[str, list[list[Any]]] = {}
        for key in keys:
            normalized_events[key] = [
                normalized
                for event in list(events_payload.get(key) or [])
                if (normalized := _footpass_event_row(event)) is not None
            ]
        return {
            "keys": keys,
            "events": normalized_events,
        }

    if isinstance(payload, dict):
        events = payload.get("events")
        if isinstance(events, dict):
            keys = [str(key) for key in events.keys()]
            normalized_events = {
                str(key): [
                    normalized
                    for event in list(value or [])
                    if (normalized := _footpass_event_row(event)) is not None
                ]
                for key, value in events.items()
            }
            return {
                "keys": keys,
                "events": normalized_events,
            }

        games = payload.get("games")
        if isinstance(games, list):
            keys: list[str] = []
            normalized_events: dict[str, list[list[Any]]] = {}
            for game in games:
                if not isinstance(game, dict):
                    continue
                key = str(game.get("key") or game.get("video") or "").strip()
                if not key:
                    continue
                keys.append(key)
                normalized_events[key] = [
                    normalized
                    for event in list(game.get("events") or [])
                    if (normalized := _footpass_event_row(event)) is not None
                ]
            return {
                "keys": keys,
                "events": normalized_events,
            }

    return {"keys": [], "events": {}}


def _footpass_event_row(event: Any) -> list[Any] | None:
    if isinstance(event, list):
        if len(event) < 4:
            return None
        try:
            frame = int(event[0])
            team = int(event[1])
            shirt = int(event[2])
            class_id = _normalize_footpass_class(event[3])
            confidence = float(event[4]) if len(event) > 4 else 1.0
        except (TypeError, ValueError):
            return None
        return [frame, team, shirt, class_id, confidence]
    if not isinstance(event, dict):
        return None
    try:
        frame = int(event.get("frame"))
        team = int(event.get("team", event.get("team_left_right")))
        shirt = int(event.get("shirt", event.get("shirt_number")))
        class_id = _normalize_footpass_class(event.get("class", event.get("class_id")))
        confidence = float(event.get("confidence", event.get("score", 1.0)))
    except (TypeError, ValueError):
        return None
    return [frame, team, shirt, class_id, confidence]


def _normalize_footpass_class(value: Any) -> int:
    if isinstance(value, str):
        normalized = " ".join(value.strip().replace("_", " ").replace("-", " ").split()).lower()
        if normalized not in FOOTPASS_CLASS_IDS:
            raise ValueError(f"Unsupported FOOTPASS action label: {value}")
        return FOOTPASS_CLASS_IDS[normalized]
    class_id = int(value)
    if class_id < 1 or class_id > 8:
        raise ValueError(f"Unsupported FOOTPASS class id: {class_id}")
    return class_id


def _prepare_tracking_exports(
    *,
    suite: dict[str, Any],
    recipe: dict[str, Any],
    dataset_root: str,
    artifacts_dir: str | Path,
    benchmark_id: str,
) -> dict[str, Any]:
    artifacts_path = ensure_dir(artifacts_dir)
    dataset_path = Path(dataset_root).expanduser().resolve() if str(dataset_root).strip() else Path(dataset_root)
    tracker_submission_dir = artifacts_path / "tracker_submission"
    tracker_submission_zip = artifacts_path / "tracker_submission.zip"
    source_path = _resolve_source_json(
        recipe=recipe,
        artifacts_path=artifacts_path,
        filename=TRACKING_SOURCE_FILENAME,
        candidate_keys=(
            "tracking_prediction_json",
            "tracking_predictions_json",
            "prediction_json",
            "source_prediction_json",
        ),
    )
    sample_submission_zip = _resolve_tracking_sample_submission(dataset_path)

    payload = _base_manifest(
        protocol="tracking",
        suite=suite,
        recipe=recipe,
        dataset_root=str(dataset_path) if str(dataset_root).strip() else "",
        artifacts_dir=artifacts_path,
        benchmark_id=benchmark_id,
    )
    payload["inputs"]["source_prediction_json"] = str(source_path)
    payload["inputs"]["sample_submission_zip"] = str(sample_submission_zip)
    payload["outputs"]["tracker_submission_dir"] = str(tracker_submission_dir)
    payload["outputs"]["tracker_submission_zip"] = str(tracker_submission_zip)

    template_sequences = _tracking_template_sequences(sample_submission_zip)
    payload["counts"]["template_sequences"] = len(template_sequences)
    if sample_submission_zip.exists() and not template_sequences:
        payload["notes"].append(
            "The staged sample_submission.zip did not expose any top-level SNMOT-*.txt members, so sequence validation falls back to the source payload."
        )

    if source_path.exists():
        try:
            source_payload = _load_json_file(source_path)
            normalized_payload = _normalize_tracking_source(source_payload)
            payload["notes"].append(
                "Tracking export used the staged repo-owned prediction JSON instead of direct recipe inference."
            )
        except ValueError as exc:
            _raise_export_blocker(
                artifacts_dir=artifacts_path,
                payload=payload,
                message=str(exc),
                extra_artifacts={
                    "tracker_submission_zip": str(tracker_submission_zip),
                    "prediction_source_json": str(source_path),
                },
            )
    else:
        try:
            normalized_payload = _generate_tracking_predictions_from_recipe(
                recipe=recipe,
                dataset_path=dataset_path,
                template_sequences=template_sequences,
            )
            payload["notes"].append(
                "Tracking export generated predictions directly from the recipe because no staged prediction JSON was present."
            )
        except ValueError as exc:
            _raise_export_blocker(
                artifacts_dir=artifacts_path,
                payload=payload,
                message=str(exc),
                extra_artifacts={
                    "tracker_submission_zip": str(tracker_submission_zip),
                    "prediction_source_json": str(source_path),
                },
            )

    source_sequences = sorted(normalized_payload.keys())
    payload["counts"]["source_sequences"] = len(source_sequences)
    payload["counts"]["source_detections"] = int(
        sum(len(rows) for rows in normalized_payload.values())
    )
    if not source_sequences:
        _raise_export_blocker(
            artifacts_dir=artifacts_path,
            payload=payload,
            message=(
                "Tracking export source is empty. Expected at least one sequence entry in "
                f"{source_path}."
            ),
            extra_artifacts={
                "tracker_submission_zip": str(tracker_submission_zip),
                "prediction_source_json": str(source_path),
            },
        )

    target_sequences = template_sequences or source_sequences
    if tracker_submission_dir.exists():
        shutil.rmtree(tracker_submission_dir)
    tracker_submission_dir = ensure_dir(tracker_submission_dir)
    if tracker_submission_zip.exists():
        tracker_submission_zip.unlink()

    emitted_files: list[str] = []
    missing_template_sequences = [sequence for sequence in target_sequences if sequence not in normalized_payload]
    extra_source_sequences = [sequence for sequence in source_sequences if sequence not in set(target_sequences)]
    for sequence_name in target_sequences:
        sequence_rows = normalized_payload.get(sequence_name, [])
        output_path = tracker_submission_dir / f"{sequence_name}.txt"
        _write_tracking_sequence_file(output_path, sequence_rows)
        emitted_files.append(str(output_path))

    with zipfile.ZipFile(tracker_submission_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file_path in sorted(tracker_submission_dir.glob("*.txt")):
            archive.write(file_path, arcname=file_path.name)

    payload["counts"]["emitted_sequence_files"] = len(emitted_files)
    payload["counts"]["missing_template_sequences"] = len(missing_template_sequences)
    payload["counts"]["extra_source_sequences"] = len(extra_source_sequences)
    payload["outputs"]["emitted_sequence_files"] = emitted_files[:20]
    if missing_template_sequences:
        payload["status"] = "partial"
        payload["notes"].append(
            "The tracking export filled some template sequences with empty files because the source payload did not contain detections for them."
        )
        payload["outputs"]["missing_template_sequences"] = missing_template_sequences[:20]
    if extra_source_sequences:
        payload["notes"].append(
            "The tracking export ignored source sequences that were not present in the staged sample_submission.zip template."
        )
        payload["outputs"]["extra_source_sequences"] = extra_source_sequences[:20]

    manifest_path = _write_manifest(artifacts_path, payload)
    return {
        "artifacts": {
            "prediction_export_json": str(manifest_path),
            "tracker_submission_zip": str(tracker_submission_zip),
            "tracker_submission_dir": str(tracker_submission_dir),
            "prediction_source_json": str(source_path),
        },
        "raw_result": {"prediction_export": payload},
    }


def _resolve_tracking_sample_submission(dataset_path: Path) -> Path:
    if dataset_path.is_file():
        return dataset_path
    candidate = dataset_path / "sample_submission.zip"
    return candidate


def _tracking_template_sequences(sample_submission_zip: Path) -> list[str]:
    if not sample_submission_zip.exists():
        return []
    try:
        with zipfile.ZipFile(sample_submission_zip, "r") as archive:
            sequence_names = sorted(
                {
                    Path(member).stem
                    for member in archive.namelist()
                    if member.endswith(".txt") and not member.endswith("/")
                }
            )
    except (OSError, zipfile.BadZipFile):
        return []
    return [name for name in sequence_names if name]


def _normalize_tracking_source(payload: dict[str, Any] | list[Any]) -> dict[str, list[list[Any]]]:
    normalized: dict[str, list[list[Any]]] = {}

    def ingest_sequence(sequence_name: Any, rows: Any) -> None:
        key = str(sequence_name or "").strip()
        if not key:
            return
        row_list = rows if isinstance(rows, list) else []
        normalized[key] = [
            converted
            for row in row_list
            if (converted := _tracking_detection_row(row)) is not None
        ]

    if isinstance(payload, dict):
        for collection_key in ("sequences", "videos", "games", "predictions", "tracks"):
            collection = payload.get(collection_key)
            if isinstance(collection, dict):
                for sequence_name, rows in collection.items():
                    ingest_sequence(sequence_name, rows)
                if normalized:
                    return normalized
            if isinstance(collection, list):
                for item in collection:
                    if not isinstance(item, dict):
                        continue
                    ingest_sequence(
                        item.get("sequence") or item.get("video") or item.get("game") or item.get("key") or item.get("name"),
                        item.get("detections") or item.get("tracks") or item.get("events") or item.get("rows") or item.get("predictions"),
                    )
                if normalized:
                    return normalized

        if payload and all(isinstance(value, list) for value in payload.values()):
            for sequence_name, rows in payload.items():
                ingest_sequence(sequence_name, rows)
            return normalized

    if isinstance(payload, list):
        for item in payload:
            if not isinstance(item, dict):
                continue
            ingest_sequence(
                item.get("sequence") or item.get("video") or item.get("game") or item.get("key") or item.get("name"),
                item.get("detections") or item.get("tracks") or item.get("events") or item.get("rows") or item.get("predictions"),
            )
    return normalized


def _tracking_detection_row(event: Any) -> list[Any] | None:
    if isinstance(event, list):
        if len(event) < 6:
            return None
        try:
            frame = int(event[0])
            track_id = int(event[1])
            left = float(event[2])
            top = float(event[3])
            width = float(event[4])
            height = float(event[5])
            confidence = float(event[6]) if len(event) > 6 else 1.0
        except (TypeError, ValueError):
            return None
        if width <= 0 or height <= 0:
            return None
        return [frame, track_id, left, top, width, height, confidence, -1, -1, -1]

    if not isinstance(event, dict):
        return None

    bbox = event.get("bbox_ltwh") or event.get("bbox") or event.get("tlwh")
    try:
        frame = int(event.get("frame", event.get("image_id")))
        track_id = int(event.get("track_id"))
        confidence = float(event.get("confidence", event.get("score", event.get("bbox_conf", 1.0))))
        if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
            left = float(bbox[0])
            top = float(bbox[1])
            width = float(bbox[2])
            height = float(bbox[3])
        else:
            left = float(event.get("left", event.get("x")))
            top = float(event.get("top", event.get("y")))
            width = float(event.get("width", event.get("w")))
            height = float(event.get("height", event.get("h")))
    except (TypeError, ValueError):
        return None
    if width <= 0 or height <= 0:
        return None
    return [frame, track_id, left, top, width, height, confidence, -1, -1, -1]


def _generate_tracking_predictions_from_recipe(
    *,
    recipe: dict[str, Any],
    dataset_path: Path,
    template_sequences: list[str],
) -> dict[str, list[list[Any]]]:
    from app.reid_tracker import HybridReIDTracker
    from app.wide_angle import (
        BYTETRACK_PLAYER_TRACKER_MODE,
        DEFAULT_TRACKER,
        choose_device,
        resolve_detector_spec,
        resolve_player_tracker_mode,
    )
    from ultralytics import YOLO

    detector_spec = resolve_detector_spec(str(recipe.get("artifact_path") or "soccana"))
    detector_path = str(detector_spec.get("weights_path") or recipe.get("artifact_path") or "")
    if not detector_path:
        raise ValueError(
            "Tracking export could not resolve detector weights for direct recipe inference. "
            "Provide a runnable tracking recipe or stage recipe_tracking_predictions.json explicitly."
        )

    split_root = dataset_path / "test"
    sequence_names = template_sequences or sorted(
        path.name
        for path in split_root.iterdir()
        if path.is_dir()
    ) if split_root.exists() else template_sequences
    if not sequence_names:
        raise ValueError(
            "Tracking export could not infer any dataset sequences for direct recipe inference. "
            f"Expected sequence folders under {split_root} or members in {dataset_path / 'sample_submission.zip'}."
        )

    tracker_mode = resolve_player_tracker_mode({"tracker_mode": recipe.get("requested_tracker_mode")})
    detector_device = choose_device()
    person_like_class_ids = sorted(
        {
            *[int(class_id) for class_id in (detector_spec.get("player_class_ids") or [])],
            *[int(class_id) for class_id in (detector_spec.get("referee_class_ids") or [])],
        }
    )
    ball_class_ids = [int(class_id) for class_id in (detector_spec.get("ball_class_ids") or [])]
    if not person_like_class_ids and not ball_class_ids:
        raise ValueError(
            "Tracking export could not resolve any detector classes for direct recipe inference."
        )

    generated: dict[str, list[list[Any]]] = {}
    for sequence_name in sequence_names:
        sequence_dir = split_root / sequence_name
        image_dir = sequence_dir / "img1"
        if not image_dir.exists():
            raise ValueError(
                "Tracking export could not run direct recipe inference because the expected image directory is missing: "
                f"{image_dir}. Materialize the full SoccerNetMOT sequence tree or stage recipe_tracking_predictions.json explicitly."
            )
        frame_paths = sorted(
            path
            for path in image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
        )
        if not frame_paths:
            raise ValueError(
                "Tracking export could not run direct recipe inference because the sequence image directory is empty: "
                f"{image_dir}."
            )

        person_model = YOLO(detector_path) if person_like_class_ids else None
        ball_model = YOLO(detector_path) if ball_class_ids else None
        tracker_rows: list[list[Any]] = []
        first_frame = cv2.imread(str(frame_paths[0]))
        if first_frame is None:
            raise ValueError(f"cv2.imread failed for {frame_paths[0]}")
        frame_height, frame_width = first_frame.shape[:2]
        hybrid_tracker = (
            None
            if tracker_mode == BYTETRACK_PLAYER_TRACKER_MODE or person_model is None
            else HybridReIDTracker(
                fps=_tracking_sequence_fps(sequence_dir),
                frame_size=(frame_width, frame_height),
                detection_confidence_floor=0.25,
                device=detector_device,
            )
        )

        for frame_number, frame_path in enumerate(frame_paths, start=1):
            frame = cv2.imread(str(frame_path))
            if frame is None:
                raise ValueError(f"cv2.imread failed for {frame_path}")

            if person_model is not None:
                if tracker_mode == BYTETRACK_PLAYER_TRACKER_MODE:
                    person_results = person_model.track(
                        source=frame,
                        persist=True,
                        tracker=DEFAULT_TRACKER,
                        conf=0.25,
                        iou=0.50,
                        device=detector_device,
                        classes=person_like_class_ids,
                        verbose=False,
                    )
                    tracker_rows.extend(
                        _tracking_rows_from_ultralytics_boxes(
                            frame_number=frame_number,
                            boxes=person_results[0].boxes,
                            track_id_offset=0,
                        )
                    )
                else:
                    person_results = person_model(
                        source=frame,
                        conf=0.25,
                        iou=0.50,
                        device=detector_device,
                        classes=person_like_class_ids,
                        verbose=False,
                    )
                    tracker_rows.extend(
                        _tracking_rows_from_hybrid_tracker(
                            frame_number=frame_number,
                            frame=frame,
                            boxes=person_results[0].boxes,
                            hybrid_tracker=hybrid_tracker,
                        )
                    )

            if ball_model is not None:
                ball_results = ball_model.track(
                    source=frame,
                    persist=True,
                    tracker=DEFAULT_TRACKER,
                    conf=0.20,
                    iou=0.50,
                    device=detector_device,
                    classes=ball_class_ids,
                    verbose=False,
                )
                tracker_rows.extend(
                    _tracking_rows_from_ultralytics_boxes(
                        frame_number=frame_number,
                        boxes=ball_results[0].boxes,
                        track_id_offset=1_000_000,
                    )
                )

        generated[sequence_name] = tracker_rows
    return generated


def _tracking_sequence_fps(sequence_dir: Path) -> float:
    seqinfo_path = sequence_dir / "seqinfo.ini"
    if not seqinfo_path.exists():
        return 25.0
    for line in seqinfo_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("frameRate="):
            try:
                return float(line.split("=", 1)[1].strip())
            except ValueError:
                return 25.0
    return 25.0


def _tracking_rows_from_ultralytics_boxes(
    *,
    frame_number: int,
    boxes: Any,
    track_id_offset: int,
) -> list[list[Any]]:
    if boxes is None or len(boxes) == 0:
        return []
    xyxy = boxes.xyxy.cpu().numpy()
    confidences = boxes.conf.cpu().numpy() if boxes.conf is not None else np.ones(len(xyxy), dtype=np.float32)
    track_ids = (
        boxes.id.cpu().numpy().astype(int)
        if boxes.id is not None
        else np.full(len(xyxy), -1, dtype=int)
    )
    rows: list[list[Any]] = []
    for index, box in enumerate(xyxy):
        track_id = int(track_ids[index])
        if track_id < 0:
            continue
        x1, y1, x2, y2 = [float(value) for value in box.tolist()]
        width = max(x2 - x1, 0.0)
        height = max(y2 - y1, 0.0)
        if width <= 0.0 or height <= 0.0:
            continue
        rows.append(
            [
                int(frame_number),
                int(track_id_offset + track_id),
                x1,
                y1,
                width,
                height,
                float(confidences[index]),
                -1,
                -1,
                -1,
            ]
        )
    return rows


def _tracking_rows_from_hybrid_tracker(
    *,
    frame_number: int,
    frame: np.ndarray,
    boxes: Any,
    hybrid_tracker: Any,
) -> list[list[Any]]:
    if boxes is None or len(boxes) == 0 or hybrid_tracker is None:
        return []
    xyxy = boxes.xyxy.cpu().numpy()
    confidences = boxes.conf.cpu().numpy() if boxes.conf is not None else np.ones(len(xyxy), dtype=np.float32)
    detections: list[dict[str, Any]] = []
    for index, box in enumerate(xyxy):
        x1, y1, x2, y2 = [int(round(float(value))) for value in box.tolist()]
        width = max(x2 - x1, 0)
        height = max(y2 - y1, 0)
        if width <= 0 or height <= 0:
            continue
        detections.append(
            {
                "track_id": -1,
                "confidence": float(confidences[index]),
                "bbox": (x1, y1, x2, y2),
                "anchor": (float((x1 + x2) / 2.0), float(y2)),
                "field_point": None,
                "identity_feature": None,
            }
        )
    assigned_track_ids = hybrid_tracker.update(frame, detections, frame_number)
    rows: list[list[Any]] = []
    for detection, track_id in zip(detections, assigned_track_ids):
        if int(track_id) < 0:
            continue
        x1, y1, x2, y2 = detection["bbox"]
        rows.append(
            [
                int(frame_number),
                int(track_id),
                float(x1),
                float(y1),
                float(max(x2 - x1, 0)),
                float(max(y2 - y1, 0)),
                float(detection["confidence"]),
                -1,
                -1,
                -1,
            ]
        )
    return rows


def _write_tracking_sequence_file(path: Path, rows: list[list[Any]]) -> None:
    serialized_rows = [
        ",".join(_format_tracking_value(value) for value in row)
        for row in rows
    ]
    path.write_text(
        ("\n".join(serialized_rows) + ("\n" if serialized_rows else "")),
        encoding="utf-8",
    )


def _format_tracking_value(value: Any) -> str:
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return f"{value:.6f}".rstrip("0").rstrip(".")
    return str(value)
