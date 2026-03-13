from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .common import BenchmarkPredictionExportUnavailable, ensure_dir


CALIBRATION_SPLIT = "valid"
CALIBRATION_RESOLUTION = (960, 540)
TEAM_SPOTTING_SOURCE_FILENAME = "recipe_event_spotting_predictions.json"
FOOTPASS_SOURCE_FILENAME = "recipe_playbyplay_predictions.json"
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
        if artifact_path.is_file():
            candidates.append(artifact_path)
        else:
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
        if path.is_file() and path.name != "per_match_info.json"
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
