from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[2]
BACKEND_DIR = ROOT_DIR / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.benchmark import benchmark_orchestrator
from app.benchmark_eval import prediction_exports
from app.benchmark_eval.prediction_exports import prepare_prediction_exports


def test_prepare_calibration_exports_writes_camera_json_and_manifest(tmp_path: Path) -> None:
    dataset_root = tmp_path / "calibration_dataset"
    split_dir = dataset_root / "valid"
    split_dir.mkdir(parents=True)
    (split_dir / "0001.json").write_text("{}", encoding="utf-8")
    (split_dir / "0001.jpg").write_bytes(b"not-a-real-image")
    artifacts_dir = tmp_path / "artifacts"

    fake_camera = {
        "pan_degrees": 0.0,
        "tilt_degrees": 0.0,
        "roll_degrees": 0.0,
        "position_meters": [0.0, 0.0, 20.0],
        "x_focal_length": 1000.0,
        "y_focal_length": 1000.0,
        "principal_point": [480.0, 270.0],
        "radial_distortion": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "tangential_distortion": [0.0, 0.0],
        "thin_prism_distortion": [0.0, 0.0, 0.0, 0.0],
    }

    with patch("app.wide_angle.choose_device", return_value="cpu"), patch(
        "app.wide_angle.choose_keypoint_device",
        return_value="cpu",
    ), patch(
        "app.wide_angle.detect_pitch_homography",
        return_value=(np.eye(3, dtype=np.float64), None, 8, 8, 0.0),
    ), patch(
        "app.benchmark_eval.prediction_exports.cv2.imread",
        return_value=np.zeros((540, 960, 3), dtype=np.uint8),
    ), patch(
        "app.benchmark_eval.prediction_exports._camera_payload_from_pipeline_homography",
        return_value=fake_camera,
    ):
        prepared = prepare_prediction_exports(
            suite={"id": "calib.sn_calib_medium_v1", "protocol": "calibration"},
            recipe={"id": "pipeline:soccermaster", "pipeline": "soccermaster"},
            dataset_root=str(dataset_root),
            artifacts_dir=artifacts_dir,
            benchmark_id="bench_calibration",
        )

    prediction_path = artifacts_dir / "predictions" / "valid" / "camera_0001.json"
    manifest_path = artifacts_dir / "prediction_export.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert prediction_path.exists()
    assert json.loads(prediction_path.read_text(encoding="utf-8")) == fake_camera
    assert prepared["artifacts"]["prediction_export_json"] == str(manifest_path)
    assert manifest["status"] == "ready"
    assert manifest["counts"]["generated_camera_files"] == 1


def test_prepare_calibration_exports_ignores_match_metadata_json(tmp_path: Path) -> None:
    dataset_root = tmp_path / "calibration_dataset"
    split_dir = dataset_root / "valid"
    split_dir.mkdir(parents=True)
    (split_dir / "0001.json").write_text("{}", encoding="utf-8")
    (split_dir / "0001.jpg").write_bytes(b"not-a-real-image")
    (split_dir / "match_info.json").write_text("{}", encoding="utf-8")
    (split_dir / "per_match_info.json").write_text("{}", encoding="utf-8")
    artifacts_dir = tmp_path / "artifacts"

    fake_camera = {
        "pan_degrees": 0.0,
        "tilt_degrees": 0.0,
        "roll_degrees": 0.0,
        "position_meters": [0.0, 0.0, 20.0],
        "x_focal_length": 1000.0,
        "y_focal_length": 1000.0,
        "principal_point": [480.0, 270.0],
        "radial_distortion": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "tangential_distortion": [0.0, 0.0],
        "thin_prism_distortion": [0.0, 0.0, 0.0, 0.0],
    }

    with patch("app.wide_angle.choose_device", return_value="cpu"), patch(
        "app.wide_angle.choose_keypoint_device",
        return_value="cpu",
    ), patch(
        "app.wide_angle.detect_pitch_homography",
        return_value=(np.eye(3, dtype=np.float64), None, 8, 8, 0.0),
    ), patch(
        "app.benchmark_eval.prediction_exports.cv2.imread",
        return_value=np.zeros((540, 960, 3), dtype=np.uint8),
    ), patch(
        "app.benchmark_eval.prediction_exports._camera_payload_from_pipeline_homography",
        return_value=fake_camera,
    ):
        prepare_prediction_exports(
            suite={"id": "calib.sn_calib_medium_v1", "protocol": "calibration"},
            recipe={"id": "pipeline:soccermaster", "pipeline": "soccermaster"},
            dataset_root=str(dataset_root),
            artifacts_dir=artifacts_dir,
            benchmark_id="bench_calibration_metadata",
        )

    manifest = json.loads((artifacts_dir / "prediction_export.json").read_text(encoding="utf-8"))

    assert manifest["status"] == "ready"
    assert manifest["counts"]["annotation_files"] == 1
    assert "missing_image_frame_ids" not in manifest["inputs"]


def test_prepare_synloc_exports_falls_back_to_recipe_inference_and_writes_results(tmp_path: Path) -> None:
    dataset_root = tmp_path / "synloc_dataset"
    annotations_dir = dataset_root / "annotations"
    image_dir = dataset_root / "val"
    annotations_dir.mkdir(parents=True)
    image_dir.mkdir(parents=True)
    (image_dir / "000001.jpg").write_bytes(b"not-a-real-image")
    (annotations_dir / "val.json").write_text(
        json.dumps(
            {
                "images": [
                    {
                        "id": 1,
                        "file_name": "000001.jpg",
                        "width": 960,
                        "height": 540,
                        "camera_matrix": np.eye(3, dtype=np.float32).tolist(),
                        "undist_poly": [0.0, 0.0, 0.0],
                    }
                ],
                "annotations": [
                    {
                        "id": 1,
                        "image_id": 1,
                        "category_id": 1,
                        "bbox": [10.0, 20.0, 30.0, 40.0],
                        "area": 1200.0,
                        "position_on_pitch": [1.0, 2.0],
                    }
                ],
                "categories": [{"id": 1, "name": "person"}],
            }
        ),
        encoding="utf-8",
    )
    artifacts_dir = tmp_path / "artifacts"

    with patch(
        "app.benchmark_eval.prediction_exports._build_synloc_predictor",
        return_value={"detector_model": object(), "detector_device": "cpu", "detector_spec": {"player_class_ids": [0]}},
    ), patch(
        "app.benchmark_eval.prediction_exports.cv2.imread",
        return_value=np.zeros((540, 960, 3), dtype=np.uint8),
    ), patch(
        "app.benchmark_eval.prediction_exports._run_synloc_detector_on_image",
        return_value=[
            {
                "bbox": [10.0, 20.0, 30.0, 40.0],
                "score": 0.9,
                "anchor": [25.0, 60.0],
            }
        ],
    ), patch(
        "app.benchmark_eval.prediction_exports._project_synloc_anchor_to_pitch",
        return_value=[1.0, 2.0],
    ):
        prepared = prepare_prediction_exports(
            suite={"id": "loc.synloc_quick_v1", "protocol": "synloc"},
            recipe={"id": "tracker:soccana+hybrid_reid+soccana_keypoint", "artifact_path": "/tmp/model.pt"},
            dataset_root=str(dataset_root),
            artifacts_dir=artifacts_dir,
            benchmark_id="bench_synloc",
        )

    predictions_path = artifacts_dir / "results.json"
    metadata_path = artifacts_dir / "metadata.json"
    manifest_path = artifacts_dir / "prediction_export.json"
    predictions = json.loads(predictions_path.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert prepared["artifacts"]["predictions_json"] == str(predictions_path)
    assert predictions == [
        {
            "id": 1,
            "image_id": 1,
            "category_id": 1,
            "bbox": [10.0, 20.0, 30.0, 40.0],
            "area": 1200.0,
            "score": 0.9,
            "position_on_pitch": [1.0, 2.0],
        }
    ]
    assert metadata == {
        "score_threshold": None,
        "position_from_keypoint_index": None,
    }
    assert manifest["status"] == "ready"
    assert manifest["counts"]["generated_predictions"] == 1
    assert manifest["counts"]["source_raw_detections"] == 1


def test_prepare_team_spotting_exports_converts_repo_owned_source_json(tmp_path: Path) -> None:
    dataset_root = tmp_path / "team_dataset"
    game_path = "england_efl/2019-2020/2019-10-01 - Stoke City - Huddersfield Town"
    labels_path = dataset_root / game_path / "Labels-ball.json"
    labels_path.parent.mkdir(parents=True)
    labels_path.write_text("{}", encoding="utf-8")

    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir(parents=True)
    source_path = artifacts_dir / prediction_exports.TEAM_SPOTTING_SOURCE_FILENAME
    source_path.write_text(
        json.dumps(
            [
                {
                    "video": game_path,
                    "events": [
                        {
                            "position_ms": 1500,
                            "label": "pass",
                            "team": "home",
                            "confidence": 0.75,
                        }
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )

    prepare_prediction_exports(
        suite={"id": "spot.team_bas_quick_v1", "protocol": "team_spotting"},
        recipe={"id": "synthetic:event-spotter"},
        dataset_root=str(dataset_root),
        artifacts_dir=artifacts_dir,
        benchmark_id="bench_team",
    )

    output_path = artifacts_dir / "predictions" / game_path / "results_spotting.json"
    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert payload["UrlLocal"] == game_path
    assert payload["predictions"] == [
        {
            "gameTime": "1 - 0:01",
            "label": "PASS",
            "position": 1500,
            "half": 1,
            "confidence": 0.75,
            "team": "left",
        }
    ]


def test_prepare_footpass_exports_converts_repo_owned_source_json(tmp_path: Path) -> None:
    dataset_root = tmp_path / "pcbas_dataset"
    ground_truth_dir = dataset_root / "playbyplay_GT"
    ground_truth_dir.mkdir(parents=True)
    (ground_truth_dir / "playbyplay_val.json").write_text(
        json.dumps({"keys": ["game_18_H1"], "events": {"game_18_H1": []}}),
        encoding="utf-8",
    )

    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir(parents=True)
    source_path = artifacts_dir / prediction_exports.FOOTPASS_SOURCE_FILENAME
    source_path.write_text(
        json.dumps(
            {
                "games": [
                    {
                        "key": "game_18_H1",
                        "events": [
                            {
                                "frame": 39,
                                "team_left_right": 0,
                                "shirt_number": 81,
                                "class": "pass",
                                "score": 0.8043,
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    prepare_prediction_exports(
        suite={"id": "spot.pcbas_medium_v1", "protocol": "pcbas"},
        recipe={"id": "synthetic:pcbas"},
        dataset_root=str(dataset_root),
        artifacts_dir=artifacts_dir,
        benchmark_id="bench_pcbas",
    )

    predictions_path = artifacts_dir / "predictions.json"
    payload = json.loads(predictions_path.read_text(encoding="utf-8"))

    assert payload == {
        "keys": ["game_18_H1"],
        "events": {
            "game_18_H1": [
                [39, 0, 81, 2, 0.8043],
            ]
        },
    }


def test_prepare_tracking_exports_converts_repo_owned_source_json(tmp_path: Path) -> None:
    dataset_root = tmp_path / "tracking_dataset"
    dataset_root.mkdir(parents=True)
    with zipfile.ZipFile(dataset_root / "sample_submission.zip", "w") as archive:
        archive.writestr("SNMOT-001.txt", "")
        archive.writestr("SNMOT-002.txt", "")

    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir(parents=True)
    source_path = artifacts_dir / prediction_exports.TRACKING_SOURCE_FILENAME
    source_path.write_text(
        json.dumps(
            {
                "sequences": [
                    {
                        "sequence": "SNMOT-001",
                        "detections": [
                            {
                                "frame": 1,
                                "track_id": 7,
                                "bbox_ltwh": [10, 20, 30, 40],
                                "confidence": 0.95,
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    prepare_prediction_exports(
        suite={"id": "track.sn_tracking_medium_v1", "protocol": "tracking"},
        recipe={"id": "synthetic:tracking"},
        dataset_root=str(dataset_root),
        artifacts_dir=artifacts_dir,
        benchmark_id="bench_tracking",
    )

    tracker_submission_zip = artifacts_dir / "tracker_submission.zip"
    manifest = json.loads((artifacts_dir / "prediction_export.json").read_text(encoding="utf-8"))
    with zipfile.ZipFile(tracker_submission_zip, "r") as archive:
        members = sorted(archive.namelist())
        payload = archive.read("SNMOT-001.txt").decode("utf-8")
        empty_payload = archive.read("SNMOT-002.txt").decode("utf-8")

    assert members == ["SNMOT-001.txt", "SNMOT-002.txt"]
    assert payload.strip() == "1,7,10,20,30,40,0.95,-1,-1,-1"
    assert empty_payload == ""
    assert manifest["status"] == "partial"
    assert manifest["counts"]["missing_template_sequences"] == 1


def test_prepare_tracking_exports_falls_back_to_recipe_inference_when_source_json_missing(tmp_path: Path) -> None:
    dataset_root = tmp_path / "tracking_dataset"
    dataset_root.mkdir(parents=True)
    with zipfile.ZipFile(dataset_root / "sample_submission.zip", "w") as archive:
        archive.writestr("SNMOT-001.txt", "")

    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir(parents=True)

    with patch(
        "app.benchmark_eval.prediction_exports._generate_tracking_predictions_from_recipe",
        return_value={
            "SNMOT-001": [
                [1, 5, 10, 20, 30, 40, 0.9, -1, -1, -1],
            ]
        },
    ):
        prepare_prediction_exports(
            suite={"id": "track.sn_tracking_medium_v1", "protocol": "tracking"},
            recipe={"id": "synthetic:tracking"},
            dataset_root=str(dataset_root),
            artifacts_dir=artifacts_dir,
            benchmark_id="bench_tracking_fallback",
        )

    tracker_submission_zip = artifacts_dir / "tracker_submission.zip"
    manifest = json.loads((artifacts_dir / "prediction_export.json").read_text(encoding="utf-8"))
    with zipfile.ZipFile(tracker_submission_zip, "r") as archive:
        payload = archive.read("SNMOT-001.txt").decode("utf-8")

    assert payload.strip() == "1,5,10,20,30,40,0.9,-1,-1,-1"
    assert any("directly from the recipe" in note for note in manifest["notes"])


def test_suite_runner_surfaces_team_spotting_export_blocker_with_artifact_reference(tmp_path: Path) -> None:
    dataset_root = tmp_path / "team_dataset"
    game_path = "england_efl/2019-2020/2019-10-01 - Stoke City - Huddersfield Town"
    labels_path = dataset_root / game_path / "Labels-ball.json"
    labels_path.parent.mkdir(parents=True)
    labels_path.write_text("{}", encoding="utf-8")

    suite = {
        "id": "spot.team_bas_quick_v1",
        "protocol": "team_spotting",
        "metric_columns": ["team_map_at_1", "map_at_1", "clips_per_second"],
        "primary_metric": "team_map_at_1",
        "required_capabilities": ["event_spotting", "team_id"],
    }
    recipe = {
        "id": "synthetic:event-spotter",
        "available": True,
        "capabilities": {"event_spotting": True, "team_id": True},
    }

    result = benchmark_orchestrator._run_suite_recipe(
        benchmark_id="bench_team_blocked",
        benchmark_dir=tmp_path,
        suite=suite,
        suite_dataset_state={"ready": True},
        recipe=recipe,
        dataset_root=str(dataset_root),
    )

    manifest_path = Path(str(result["artifacts"]["prediction_export_json"]))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert result["status"] == "blocked"
    assert prediction_exports.TEAM_SPOTTING_SOURCE_FILENAME in str(result["error"])
    assert manifest["status"] == "blocked"
    assert result["raw_result"]["prediction_export"]["status"] == "blocked"


def test_suite_runner_surfaces_footpass_export_blocker_with_artifact_reference(tmp_path: Path) -> None:
    dataset_root = tmp_path / "pcbas_dataset"
    ground_truth_dir = dataset_root / "playbyplay_GT"
    ground_truth_dir.mkdir(parents=True)
    (ground_truth_dir / "playbyplay_val.json").write_text(
        json.dumps({"keys": ["game_18_H1"], "events": {"game_18_H1": []}}),
        encoding="utf-8",
    )

    suite = {
        "id": "spot.pcbas_medium_v1",
        "protocol": "pcbas",
        "metric_columns": ["f1_at_15", "precision_at_15", "recall_at_15", "clips_per_second"],
        "primary_metric": "f1_at_15",
        "required_capabilities": ["event_spotting", "team_id", "role_id", "jersey_ocr"],
    }
    recipe = {
        "id": "synthetic:pcbas",
        "available": True,
        "capabilities": {
            "event_spotting": True,
            "team_id": True,
            "role_id": True,
            "jersey_ocr": True,
        },
    }

    result = benchmark_orchestrator._run_suite_recipe(
        benchmark_id="bench_pcbas_blocked",
        benchmark_dir=tmp_path,
        suite=suite,
        suite_dataset_state={"ready": True},
        recipe=recipe,
        dataset_root=str(dataset_root),
    )

    manifest_path = Path(str(result["artifacts"]["prediction_export_json"]))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert result["status"] == "blocked"
    assert prediction_exports.FOOTPASS_SOURCE_FILENAME in str(result["error"])
    assert "decord==0.6.0" in str(result["error"])
    assert manifest["status"] == "blocked"
    assert result["raw_result"]["prediction_export"]["status"] == "blocked"


def test_suite_runner_surfaces_tracking_export_blocker_with_artifact_reference(tmp_path: Path) -> None:
    dataset_root = tmp_path / "tracking_dataset"
    dataset_root.mkdir(parents=True)
    with zipfile.ZipFile(dataset_root / "sample_submission.zip", "w") as archive:
        archive.writestr("SNMOT-001.txt", "")
    with zipfile.ZipFile(dataset_root / "gt.zip", "w") as archive:
        archive.writestr("test/SNMOT-001/seqinfo.ini", "[Sequence]\nseqLength=1\n")
        archive.writestr("test/SNMOT-001/gt/gt.txt", "1,1,10,20,30,40,1,-1,-1,-1\n")
    (dataset_root / "seqmap.txt").write_text("name\nSNMOT-001\n", encoding="utf-8")

    suite = {
        "id": "track.sn_tracking_medium_v1",
        "protocol": "tracking",
        "metric_columns": ["hota", "deta", "assa", "frames_per_second"],
        "primary_metric": "hota",
        "required_capabilities": ["tracking"],
    }
    recipe = {
        "id": "synthetic:tracking",
        "available": True,
        "capabilities": {"tracking": True},
    }

    result = benchmark_orchestrator._run_suite_recipe(
        benchmark_id="bench_tracking_blocked",
        benchmark_dir=tmp_path,
        suite=suite,
        suite_dataset_state={"ready": True},
        recipe=recipe,
        dataset_root=str(dataset_root),
    )

    manifest_path = Path(str(result["artifacts"]["prediction_export_json"]))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert result["status"] == "blocked"
    assert prediction_exports.TRACKING_SOURCE_FILENAME in str(result["error"])
    assert manifest["status"] == "blocked"
    assert result["raw_result"]["prediction_export"]["status"] == "blocked"
