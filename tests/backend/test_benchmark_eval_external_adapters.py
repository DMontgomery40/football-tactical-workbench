from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[2]
BACKEND_DIR = ROOT_DIR / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.benchmark_eval import calibration, pcbas, prediction_exports, team_spotting


def test_external_adapters_use_repo_owned_wrapper_scripts(tmp_path: Path) -> None:
    calibration_dataset = tmp_path / "calibration_dataset" / "valid"
    calibration_dataset.mkdir(parents=True)
    image = np.zeros((540, 960, 3), dtype=np.uint8)
    cv2.imwrite(str(calibration_dataset / "frame001.jpg"), image)
    (calibration_dataset / "frame001.json").write_text("{}", encoding="utf-8")
    calibration_artifacts = tmp_path / "calibration"
    calibration_artifacts.mkdir(parents=True)
    (calibration_artifacts / "predictions" / "valid").mkdir(parents=True)
    (calibration_artifacts / "predictions" / "valid" / "camera_frame001.json").write_text("{}", encoding="utf-8")

    team_dataset_root = tmp_path / "team_dataset"
    team_game_path = "england_efl/2019-2020/2019-10-01 - Stoke City - Huddersfield Town"
    (team_dataset_root / team_game_path).mkdir(parents=True)
    (team_dataset_root / team_game_path / "Labels-ball.json").write_text("{}", encoding="utf-8")
    team_artifacts = tmp_path / "team"
    team_artifacts.mkdir(parents=True)
    (team_artifacts / prediction_exports.TEAM_SPOTTING_SOURCE_FILENAME).write_text(
        json.dumps(
            [
                {
                    "video": team_game_path,
                    "events": [{"frame": 25, "label": "pass", "team": "left", "score": 0.8}],
                }
            ]
        ),
        encoding="utf-8",
    )

    pcbas_dataset_root = tmp_path / "pcbas_dataset"
    (pcbas_dataset_root / "playbyplay_GT").mkdir(parents=True)
    (pcbas_dataset_root / "playbyplay_GT" / "playbyplay_val.json").write_text("{}", encoding="utf-8")
    pcbas_artifacts = tmp_path / "pcbas"
    pcbas_artifacts.mkdir(parents=True)
    (pcbas_artifacts / prediction_exports.FOOTPASS_SOURCE_FILENAME).write_text(
        json.dumps(
            {
                "games": [
                    {
                        "key": "game_18_H1",
                        "events": [{"frame": 10, "team_left_right": 1, "shirt_number": 23, "class_id": 2, "score": 0.9}],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    captured: list[dict[str, object]] = []

    def fake_run_external_json_command(**kwargs: object) -> dict[str, object]:
        captured.append(kwargs)
        return {
            "_external_result_path": str(tmp_path / "external_result.json"),
            "completeness_x_jac_5": 0.5,
            "completeness": 0.6,
            "jac_5": 0.7,
            "team_map_at_1": 0.8,
            "map_at_1": 0.75,
            "f1_at_15": 0.9,
            "precision_at_15": 0.85,
            "recall_at_15": 0.8,
        }

    prediction_exports.ensure_team_spotting_prediction_export(
        dataset_root=str(team_dataset_root),
        artifacts_dir=team_artifacts,
    )
    prediction_exports.ensure_footpass_prediction_export(
        artifacts_dir=pcbas_artifacts,
    )

    with patch("app.benchmark_eval.calibration.run_external_json_command", side_effect=fake_run_external_json_command), patch(
        "app.benchmark_eval.team_spotting.run_external_json_command",
        side_effect=fake_run_external_json_command,
    ), patch("app.benchmark_eval.pcbas.run_external_json_command", side_effect=fake_run_external_json_command):
        calibration.evaluate_calibration(
            suite={"id": "calib.sn_calib_medium_v1"},
            recipe={"id": "detector:soccana"},
            dataset_root=str(tmp_path / "calibration_dataset"),
            artifacts_dir=calibration_artifacts,
            benchmark_id="bench_calibration",
        )
        team_spotting.evaluate_team_spotting(
            suite={"id": "spot.team_bas_quick_v1"},
            recipe={"id": "detector:soccana"},
            dataset_root=str(team_dataset_root),
            artifacts_dir=team_artifacts,
            benchmark_id="bench_team",
        )
        pcbas.evaluate_pcbas(
            suite={"id": "spot.pcbas_medium_v1"},
            recipe={"id": "detector:soccana"},
            dataset_root=str(pcbas_dataset_root),
            artifacts_dir=pcbas_artifacts,
            benchmark_id="bench_pcbas",
        )

    assert len(captured) == 3
    command_labels = [str(call["command"][1]) for call in captured]
    assert any(label.endswith("run_calibration_eval.py") for label in command_labels)
    assert any(label.endswith("run_team_spotting_eval.py") for label in command_labels)
    assert any(label.endswith("run_footpass_eval.py") for label in command_labels)
    assert {str(call["runtime_key"]) for call in captured} == {
        "sn_calibration_legacy",
        "modern_action_spotting",
        "footpass_eval",
    }


def test_calibration_export_helper_writes_camera_jsons(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset" / "valid"
    dataset_root.mkdir(parents=True)
    image = np.zeros((540, 960, 3), dtype=np.uint8)
    cv2.imwrite(str(dataset_root / "frame001.jpg"), image)
    (dataset_root / "frame001.json").write_text("{}", encoding="utf-8")

    fake_camera_payload = {
        "pan_degrees": 0.0,
        "tilt_degrees": 0.0,
        "roll_degrees": 0.0,
        "position_meters": [0.0, 0.0, 0.0],
        "x_focal_length": 1000.0,
        "y_focal_length": 1000.0,
        "principal_point": [480.0, 270.0],
        "radial_distortion": [0.0] * 6,
        "tangential_distortion": [0.0, 0.0],
        "thin_prism_distortion": [0.0] * 4,
    }

    with patch(
        "app.benchmark_eval.prediction_exports._build_calibration_predictor",
        return_value="stub_keypoint_model",
    ), patch(
        "app.benchmark_eval.prediction_exports._camera_payload_from_frame",
        return_value=fake_camera_payload,
    ):
        export_info = prediction_exports.ensure_calibration_prediction_export(
            recipe={"id": "tracker:soccana+hybrid_reid+soccana_keypoint"},
            dataset_root=str(tmp_path / "dataset"),
            artifacts_dir=tmp_path / "artifacts",
        )

    prediction_path = tmp_path / "artifacts" / "predictions" / "valid" / "camera_frame001.json"
    assert prediction_path.exists()
    payload = json.loads(prediction_path.read_text(encoding="utf-8"))
    assert payload["principal_point"] == [480.0, 270.0]
    assert export_info["exported_predictions"] == 1
    assert Path(str(export_info["export_summary_json"])).exists()


def test_calibration_camera_payload_helper_imports_vendored_camera_module() -> None:
    captured: dict[str, object] = {}

    class FakeCamera:
        def __init__(self, width: int, height: int) -> None:
            captured["size"] = (width, height)

        def from_homography(self, homography: object) -> bool:
            captured["homography"] = homography
            return True

        def to_json_parameters(self) -> dict[str, object]:
            return {"ok": True}

    class FakeModule:
        Camera = FakeCamera

    with patch(
        "app.benchmark_eval.prediction_exports.importlib.import_module",
        return_value=FakeModule(),
    ):
        payload = prediction_exports._camera_payload_from_pipeline_homography(
            homography_image_to_pipeline=np.eye(3, dtype=np.float64),
            frame_width=960,
            frame_height=540,
        )

    assert payload == {"ok": True}
    assert captured["size"] == (960, 540)
    assert "homography" in captured


def test_team_spotting_export_helper_writes_results_tree(tmp_path: Path) -> None:
    dataset_root = tmp_path / "team_dataset"
    game_path = "england_efl/2019-2020/2019-10-01 - Stoke City - Huddersfield Town"
    (dataset_root / game_path).mkdir(parents=True)
    (dataset_root / game_path / "Labels-ball.json").write_text("{}", encoding="utf-8")

    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir(parents=True)
    (artifacts_dir / prediction_exports.TEAM_SPOTTING_SOURCE_FILENAME).write_text(
        json.dumps(
            [
                {
                    "video": game_path,
                    "events": [
                        {"frame": 25, "label": "pass", "team": "left", "score": 0.85},
                        {"position_ms": 2000, "label": "throw in", "team": "right", "confidence": 0.4},
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )

    export_info = prediction_exports.ensure_team_spotting_prediction_export(
        dataset_root=str(dataset_root),
        artifacts_dir=artifacts_dir,
    )

    results_path = artifacts_dir / "predictions" / game_path / "results_spotting.json"
    payload = json.loads(results_path.read_text(encoding="utf-8"))

    assert payload["UrlLocal"] == game_path
    assert payload["predictions"][0]["label"] == "PASS"
    assert payload["predictions"][0]["position"] == 1000
    assert payload["predictions"][1]["label"] == "THROW IN"
    assert payload["predictions"][1]["team"] == "right"
    assert export_info["games"] == 1
    assert export_info["events"] == 2


def test_footpass_export_helper_writes_predictions_json(tmp_path: Path) -> None:
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir(parents=True)
    (artifacts_dir / prediction_exports.FOOTPASS_SOURCE_FILENAME).write_text(
        json.dumps(
            {
                "games": [
                    {
                        "key": "game_18_H1",
                        "events": [
                            {"frame": 64, "team_left_right": 0, "shirt_number": 3, "class": "drive", "score": 0.9},
                            [72, 1, 23, 2, 0.7],
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    export_info = prediction_exports.ensure_footpass_prediction_export(
        artifacts_dir=artifacts_dir,
    )

    payload = json.loads((artifacts_dir / "predictions.json").read_text(encoding="utf-8"))

    assert payload["keys"] == ["game_18_H1"]
    assert payload["events"]["game_18_H1"][0] == [64, 0, 3, 1, 0.9]
    assert payload["events"]["game_18_H1"][1] == [72, 1, 23, 2, 0.7]
    assert export_info["games"] == 1
    assert export_info["events"] == 2
