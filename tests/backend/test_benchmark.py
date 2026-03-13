from __future__ import annotations

import json
import sys
from unittest.mock import patch
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
BACKEND_DIR = ROOT_DIR / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app import main
from app.benchmark import benchmark_config_snapshot, hydrate_legacy_benchmark
from app.benchmark import benchmark_orchestrator
from app.benchmark_eval.external_cli import run_external_json_command
from app.benchmark_eval.gamestate import probe_gamestate_blockers
from app.benchmark_eval.runtime_profiles import probe_runtime_profile
from app.benchmark_catalog import list_assets, list_recipes
from app.benchmark_suites import build_suite_dataset_state, get_suite_definition


def test_benchmark_config_snapshot_exposes_suite_asset_and_recipe_catalogs() -> None:
    snapshot = benchmark_config_snapshot()

    assert snapshot["schema_version"] == 2
    assert len(snapshot["suites"]) >= 10
    assert any(suite["id"] == "det.roles_quick_v1" for suite in snapshot["suites"])
    assert any(asset["asset_id"] == "detector.soccana" for asset in snapshot["assets"])
    assert any(recipe["id"] == "detector:soccana" for recipe in snapshot["recipes"])
    assert any(state["suite_id"] == "ops.clip_review_v1" for state in snapshot["dataset_states"])


def test_recipe_catalog_contains_tracker_variants_for_soccana() -> None:
    recipes = list_recipes()
    recipe_ids = {recipe["id"] for recipe in recipes}

    assert "detector:soccana" in recipe_ids
    assert "pipeline:sn-gamestate-tracklab" in recipe_ids
    assert "tracker:soccana+bytetrack+soccana_keypoint" in recipe_ids
    assert "tracker:soccana+hybrid_reid+soccana_keypoint" in recipe_ids


def test_asset_catalog_contains_expected_builtin_assets() -> None:
    assets = list_assets()
    asset_ids = {asset["asset_id"] for asset in assets}

    assert "detector.soccana" in asset_ids
    assert "tracker.bytetrack" in asset_ids
    assert "tracker.hybrid_reid" in asset_ids
    assert "pipeline.soccermaster" in asset_ids
    assert "pipeline.tracklab_sn_gamestate" in asset_ids


def test_tracking_dataset_state_reports_concrete_expected_root_and_adapter_blocker() -> None:
    suite = get_suite_definition("track.sn_tracking_medium_v1")
    state = build_suite_dataset_state(suite)

    assert state["ready"] is False
    assert state["readiness_status"] == "blocked"
    assert str(state["dataset_root"]).endswith("backend/benchmarks/_datasets/track.sn_tracking_medium_v1/SoccerNetMOT")
    assert any("SoccerNetMOT" in blocker for blocker in state["blockers"])
    assert any("evaluate_soccernet_v3_tracking.py" in blocker for blocker in state["blockers"])
    assert any("recipe-to-submission bridge is still missing" in blocker for blocker in state["blockers"])
    assert not any(blocker.startswith("Dataset root is missing.") for blocker in state["blockers"])


def test_gsr_dataset_state_reports_concrete_expected_root_and_adapter_blocker() -> None:
    suite = get_suite_definition("gsr.medium_v1")
    state = build_suite_dataset_state(suite)

    assert state["ready"] is False
    assert state["readiness_status"] == "blocked"
    assert str(state["dataset_root"]).endswith("backend/benchmarks/_datasets/gsr.medium_v1/SoccerNetGS")
    assert any("SoccerNetGS" in blocker for blocker in state["blockers"])
    assert any("TrackLab/sn-gamestate" in blocker for blocker in state["blockers"])
    assert not any(blocker.startswith("Dataset root is missing.") for blocker in state["blockers"])


def test_quick_stage2_target_suites_use_exact_manifest_blockers_not_generic_missing_root() -> None:
    for suite_id in ("det.ball_quick_v1", "loc.synloc_quick_v1"):
        suite = get_suite_definition(suite_id)
        state = build_suite_dataset_state(suite)

        assert state["ready"] is False
        assert state["manifest_exists"] is True
        assert state["blockers"]
        assert not any(blocker.startswith("Dataset root is missing.") for blocker in state["blockers"])


def test_backend_default_runtime_profile_matches_current_process() -> None:
    probe = probe_runtime_profile("backend_default")

    assert probe["available"] is True
    assert probe["python_executable"] == sys.executable
    assert str(probe["python_version"]).startswith(str(sys.version_info.major))


def test_external_cli_records_runtime_metadata_for_backend_default(tmp_path: Path) -> None:
    payload = run_external_json_command(
        command=["python", "-c", "import json; print(json.dumps({'ok': True}))"],
        cwd=tmp_path,
        artifacts_dir=tmp_path,
        runtime_key="backend_default",
    )

    external_result_path = Path(str(payload["_external_result_path"]))
    persisted = json.loads(external_result_path.read_text(encoding="utf-8"))

    assert payload["ok"] is True
    assert payload["_runner"]["profile_id"] == "backend_default"
    assert persisted["runtime_profile"]["profile_id"] == "backend_default"
    assert persisted["command"][0] == sys.executable


def test_gamestate_probe_surfaces_target_runtime_unavailability() -> None:
    suite = get_suite_definition("gsr.medium_v1")

    with patch(
        "app.benchmark_eval.gamestate.probe_runtime_profile",
        return_value={
            "profile_id": "tracklab_gamestate_py39_np1",
            "label": "TrackLab + sn-gamestate (Python 3.9 / NumPy <2)",
            "available": False,
            "missing_reasons": ["requires Python 3.9.x but the target runtime reports Python 3.12.7."],
        },
    ), patch(
        "app.benchmark_eval.gamestate.runtime_unavailable_message",
        return_value=(
            "Runtime profile 'tracklab_gamestate_py39_np1' (TrackLab + sn-gamestate (Python 3.9 / NumPy <2)) "
            "is unavailable. requires Python 3.9.x but the target runtime reports Python 3.12.7."
        ),
    ):
        blockers = probe_gamestate_blockers(
            suite=suite,
            dataset_root="",
            manifest_payload={},
        )

    assert any("tracklab_gamestate_py39_np1" in blocker for blocker in blockers)
    assert any("Python 3.9" in blocker for blocker in blockers)


def test_blocked_tracking_cells_surface_truthful_reason_in_run_results(tmp_path: Path) -> None:
    suite = get_suite_definition("track.sn_tracking_medium_v1")
    suite_state = build_suite_dataset_state(suite)
    recipe = next(recipe for recipe in list_recipes() if recipe["id"] == "tracker:soccana+bytetrack+soccana_keypoint")

    result = benchmark_orchestrator._run_suite_recipe(
        benchmark_id="test_tracking_blocked",
        benchmark_dir=tmp_path,
        suite=suite,
        suite_dataset_state=suite_state,
        recipe=recipe,
        dataset_root=str(suite_state.get("dataset_root") or ""),
    )

    assert result["status"] == "blocked"
    assert "SoccerNetMOT" in str(result["error"])
    assert "evaluate_soccernet_v3_tracking.py" in str(result["error"])
    assert result["blockers"]
    assert result["runtime_context"] == {}


def test_blocked_gsr_cells_surface_truthful_reason_in_run_results(tmp_path: Path) -> None:
    suite = get_suite_definition("gsr.medium_v1")
    suite_state = build_suite_dataset_state(suite)
    recipe = next(recipe for recipe in list_recipes() if recipe["id"] == "pipeline:sn-gamestate-tracklab")

    result = benchmark_orchestrator._run_suite_recipe(
        benchmark_id="test_gsr_blocked",
        benchmark_dir=tmp_path,
        suite=suite,
        suite_dataset_state=suite_state,
        recipe=recipe,
        dataset_root=str(suite_state.get("dataset_root") or ""),
    )

    assert result["status"] == "blocked"
    assert "SoccerNetGS" in str(result["error"])
    assert "TrackLab/sn-gamestate" in str(result["error"])
    assert result["blockers"]
    assert result["runtime_context"] == {}


def test_team_spotting_cells_surface_precise_missing_raw_prediction_artifact(tmp_path: Path) -> None:
    suite = get_suite_definition("spot.team_bas_quick_v1")
    dataset_root = tmp_path / "team_dataset"
    game_path = dataset_root / "england_efl" / "2019-2020" / "2019-10-01 - Stoke City - Huddersfield Town"
    game_path.mkdir(parents=True)
    (game_path / "Labels-ball.json").write_text("{}", encoding="utf-8")
    recipe = {
        "id": "pipeline:event-capable",
        "label": "Event-capable recipe",
        "available": True,
        "capabilities": {
            "event_spotting": True,
            "team_id": True,
        },
        "compatible_suite_ids": [suite["id"]],
    }

    result = benchmark_orchestrator._run_suite_recipe(
        benchmark_id="test_team_spotting_export_blocker",
        benchmark_dir=tmp_path,
        suite=suite,
        suite_dataset_state={"ready": True, "blockers": []},
        recipe=recipe,
        dataset_root=str(dataset_root),
    )

    assert result["status"] == "blocked"
    assert "recipe_event_spotting_predictions.json" in str(result["error"])
    assert result["blockers"]


def test_pcbas_cells_surface_precise_missing_raw_prediction_artifact(tmp_path: Path) -> None:
    suite = get_suite_definition("spot.pcbas_medium_v1")
    dataset_root = tmp_path / "pcbas_dataset"
    (dataset_root / "playbyplay_GT").mkdir(parents=True)
    (dataset_root / "playbyplay_GT" / "playbyplay_val.json").write_text("{}", encoding="utf-8")
    recipe = {
        "id": "pipeline:pcbas-capable",
        "label": "PCBAS-capable recipe",
        "available": True,
        "capabilities": {
            "event_spotting": True,
            "team_id": True,
            "role_id": True,
            "jersey_ocr": True,
        },
        "compatible_suite_ids": [suite["id"]],
    }

    result = benchmark_orchestrator._run_suite_recipe(
        benchmark_id="test_pcbas_export_blocker",
        benchmark_dir=tmp_path,
        suite=suite,
        suite_dataset_state={"ready": True, "blockers": []},
        recipe=recipe,
        dataset_root=str(dataset_root),
    )

    assert result["status"] == "blocked"
    assert "recipe_playbyplay_predictions.json" in str(result["error"])
    assert result["blockers"]


def test_team_spotting_dataset_state_requires_labels_and_source_video(tmp_path: Path) -> None:
    suite = dict(get_suite_definition("spot.team_bas_quick_v1"))
    dataset_root = tmp_path / "team_dataset"
    game_path = dataset_root / "england_efl" / "2019-2020" / "2019-10-01 - Middlesbrough - Preston North End"
    game_path.mkdir(parents=True)
    (game_path / "Labels-ball.json").write_text("{}", encoding="utf-8")
    manifest_path = tmp_path / "spot.team_bas_quick_v1.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "suite_id": "spot.team_bas_quick_v1",
                "kind": "dataset_manifest",
                "materialization": {"status": "blocked"},
            }
        ),
        encoding="utf-8",
    )
    suite["dataset_root"] = str(dataset_root)
    suite["manifest_path"] = str(manifest_path)

    state = build_suite_dataset_state(suite)

    assert state["ready"] is False
    assert any("224p.mp4" in blocker or "720p.mp4" in blocker for blocker in state["blockers"])


def test_team_spotting_dataset_state_is_ready_with_labels_and_video(tmp_path: Path) -> None:
    suite = dict(get_suite_definition("spot.team_bas_quick_v1"))
    dataset_root = tmp_path / "team_dataset"
    game_path = dataset_root / "england_efl" / "2019-2020" / "2019-10-01 - Middlesbrough - Preston North End"
    game_path.mkdir(parents=True)
    (game_path / "Labels-ball.json").write_text("{}", encoding="utf-8")
    (game_path / "224p.mp4").write_bytes(b"not-a-real-video-but-present")
    manifest_path = tmp_path / "spot.team_bas_quick_v1.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "suite_id": "spot.team_bas_quick_v1",
                "kind": "dataset_manifest",
                "items": ["england_efl/2019-2020/2019-10-01 - Middlesbrough - Preston North End"],
                "task_coverage": ["team_spotting"],
            }
        ),
        encoding="utf-8",
    )
    suite["dataset_root"] = str(dataset_root)
    suite["manifest_path"] = str(manifest_path)

    state = build_suite_dataset_state(suite)

    assert state["ready"] is True
    assert state["manifest_summary"]["materialization_status"] is None


def test_manifest_materialization_blocker_keeps_partial_pcbas_dataset_blocked(tmp_path: Path) -> None:
    suite = dict(get_suite_definition("spot.pcbas_medium_v1"))
    dataset_root = tmp_path / "pcbas_dataset"
    (dataset_root / "playbyplay_GT").mkdir(parents=True)
    (dataset_root / "playbyplay_GT" / "playbyplay_val.json").write_text("{}", encoding="utf-8")
    manifest_path = tmp_path / "spot.pcbas_medium_v1.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "suite_id": "spot.pcbas_medium_v1",
                "kind": "dataset_manifest",
                "materialization": {
                    "status": "partial",
                    "blockers": [
                        "Validation ground-truth is materialized locally, but tactical_data_VAL.zip is still gated."
                    ],
                },
            }
        ),
        encoding="utf-8",
    )
    suite["dataset_root"] = str(dataset_root)
    suite["manifest_path"] = str(manifest_path)

    state = build_suite_dataset_state(suite)

    assert state["dataset_exists"] is True
    assert state["manifest_exists"] is True
    assert state["ready"] is False
    assert any("tactical_data_VAL.zip" in blocker for blocker in state["blockers"])
    assert state["manifest_summary"]["materialization_status"] == "partial"


def test_backend_executor_respects_recipe_compatible_suite_ids(tmp_path: Path) -> None:
    suite = get_suite_definition("det.roles_quick_v1")
    suite_state = build_suite_dataset_state(suite)
    recipe = next(recipe for recipe in list_recipes() if recipe["id"] == "pipeline:soccermaster")

    result = benchmark_orchestrator._run_suite_recipe(
        benchmark_id="test_recipe_compatibility",
        benchmark_dir=tmp_path,
        suite=suite,
        suite_dataset_state=suite_state,
        recipe=recipe,
        dataset_root=str(suite_state.get("dataset_root") or ""),
    )

    assert result["status"] == "not_supported"
    assert "compatibility" in str(result["error"]).lower()


def test_hydrate_legacy_benchmark_maps_old_summary_to_operational_suite(tmp_path: Path) -> None:
    benchmark_dir = tmp_path / "legacy_benchmark"
    benchmark_dir.mkdir(parents=True)
    legacy_payload = {
        "benchmark_id": "legacy_123",
        "status": "completed",
        "created_at": "2026-03-12T00:00:00Z",
        "candidates": [
            {"id": "soccana", "label": "soccana (pretrained)", "source": "pretrained"},
        ],
        "leaderboard": [
            {
                "candidate_id": "soccana",
                "status": "completed",
                "throughput": 25.0,
                "track_stability": 82.5,
                "calibration": 91.0,
                "coverage": 88.0,
                "run_id": "bench_legacy",
            }
        ],
        "logs": [],
    }

    hydrated = hydrate_legacy_benchmark(legacy_payload, benchmark_dir)

    assert hydrated["legacy_record"] is True
    assert hydrated["primary_suite_id"] == "ops.clip_review_v1"
    assert "ops.clip_review_v1" in hydrated["suite_results"]
    assert hydrated["suite_results"]["ops.clip_review_v1"]["detector:soccana"]["status"] == "completed"


def test_benchmark_config_endpoint_returns_v2_payload() -> None:
    result = main.benchmark_config()
    payload = result.model_dump(mode="json")

    assert "suites" in payload
    assert "assets" in payload
    assert "recipes" in payload
    assert "dataset_states" in payload
    assert payload["schema_version"] == 2
