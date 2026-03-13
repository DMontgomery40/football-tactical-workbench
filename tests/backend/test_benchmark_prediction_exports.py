from __future__ import annotations

import json
import sys
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
