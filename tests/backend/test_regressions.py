from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi import HTTPException
from pydantic_ai.models.test import TestModel

ROOT_DIR = Path(__file__).resolve().parents[2]
BACKEND_DIR = ROOT_DIR / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app import main  # noqa: E402
import app.ai_diagnostics as ai_diagnostics  # noqa: E402
from app.ai_diagnostics import DiagnosticsAgentItem, DiagnosticsAgentOutput  # noqa: E402
import app.training_ai_analysis as training_ai_analysis  # noqa: E402
from app.schemas import JobStateResponse, RunSummary  # noqa: E402
from app.training import inspect_training_dataset  # noqa: E402
from app.training_provenance import resolve_dvc_tracking  # noqa: E402
import app.training_registry as training_registry_module  # noqa: E402
from app.training_registry import DEFAULT_CLASS_IDS, TrainingRegistry  # noqa: E402


def test_resolve_analysis_detector_model_explicit_soccana_bypasses_active_registry(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    resolved_pretrained = tmp_path / "soccana.pt"
    fake_registry = Mock()

    monkeypatch.setattr(main, "training_available", lambda: True)
    monkeypatch.setattr(main, "training_registry", fake_registry)
    monkeypatch.setattr(main, "resolve_model_path", lambda model_name, model_kind: str(resolved_pretrained))

    assert main.resolve_analysis_detector_model("soccana") == str(resolved_pretrained)
    fake_registry.get_active_entry.assert_not_called()
    fake_registry.get_active_path.assert_not_called()


def test_resolve_analysis_detector_model_surfaces_active_detector_resolution_failures_when_falling_back(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_registry = Mock()
    fake_registry.get_active_entry.return_value = {"id": "custom_dead"}
    fake_registry.get_active_path.side_effect = RuntimeError("registry path lookup failed")

    monkeypatch.setattr(main, "training_available", lambda: True)
    monkeypatch.setattr(main, "training_registry", fake_registry)

    with pytest.raises(HTTPException) as exc_info:
        main.resolve_analysis_detector_model("")

    assert exc_info.value.status_code == 409
    assert "custom_dead" in str(exc_info.value.detail)
    assert "could not be resolved" in str(exc_info.value.detail)


def test_activate_training_run_rejects_missing_checkpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    missing_checkpoint = ROOT_DIR / "tests" / ".artifacts" / "missing-best.pt"
    fake_job = SimpleNamespace(
        job_id="job-123",
        status="completed",
        best_checkpoint=str(missing_checkpoint),
        config={"run_name": "demo"},
        metrics={},
        created_at="2026-03-10T00:00:00Z",
        resolved_device="cpu",
        backend="ultralytics",
        backend_version="1.0",
        summary_path="/tmp/summary.json",
        artifacts={},
    )
    fake_manager = Mock()
    fake_manager.get_by_run_id.return_value = fake_job

    monkeypatch.setattr(main, "require_training_available", lambda: (fake_manager, Mock()))

    with pytest.raises(HTTPException) as exc_info:
        main.activate_training_run("run-123")

    assert exc_info.value.status_code == 409
    assert "checkpoint is missing" in str(exc_info.value.detail)


def test_activate_training_run_promotes_checkpoint_and_writes_provenance(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "training_runs" / "run-123"
    weights_dir = run_dir / "weights"
    weights_dir.mkdir(parents=True)
    source_checkpoint = weights_dir / "best.pt"
    source_checkpoint.write_bytes(b"weights")

    promoted_dir = tmp_path / "models" / "promoted" / "custom_run-123"
    promoted_checkpoint = promoted_dir / "best.pt"
    promoted_provenance = promoted_dir / "training_provenance.json"

    fake_job = SimpleNamespace(
        job_id="job-123",
        run_id="run-123",
        run_dir=str(run_dir),
        status="completed",
        best_checkpoint=str(source_checkpoint),
        config={"run_name": "demo", "base_weights": "soccana", "dataset_path": str(tmp_path / "dataset")},
        metrics={"mAP50": 0.73},
        created_at="2026-03-10T00:00:00Z",
        resolved_device="cpu",
        backend="ultralytics",
        backend_version="8.3.81",
        summary_path=str(run_dir / "summary.json"),
        artifacts={"dataset_scan": str(run_dir / "dataset_scan.json")},
        generated_dataset_yaml=str(run_dir / "dataset_runtime.yaml"),
        generated_split_lists={"train": str(run_dir / "splits" / "train.txt"), "val": str(run_dir / "splits" / "val.txt")},
        training_provenance_path=str(run_dir / "training_provenance.json"),
    )
    fake_manager = Mock()
    fake_manager.get_by_run_id.return_value = fake_job
    fake_registry = Mock()
    fake_registry.activate_detector.return_value = {"id": "custom_run-123"}

    def promote_checkpoint(run_id: str, checkpoint_path: str) -> str:
        promoted_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        Path(checkpoint_path).replace(promoted_checkpoint)
        source_checkpoint.write_bytes(b"weights")
        return str(promoted_checkpoint)

    monkeypatch.setattr(main, "require_training_available", lambda: (fake_manager, fake_registry))
    monkeypatch.setattr(main, "stage_promoted_detector_checkpoint", promote_checkpoint)
    monkeypatch.setattr(main, "resolve_promoted_provenance_path", lambda run_id: promoted_provenance)

    response = main.activate_training_run("run-123")

    assert response == {"success": True, "active_detector": "custom_run-123"}
    assert promoted_checkpoint.exists()
    assert promoted_provenance.exists()
    fake_manager.refresh_training_provenance.assert_called_once()
    fake_manager.append_log.assert_any_call("job-123", f"Promoted detector checkpoint to {promoted_checkpoint}.")
    fake_registry.activate_detector.assert_called_once()
    assert fake_registry.activate_detector.call_args.kwargs["checkpoint_path"] == str(promoted_checkpoint)
    assert fake_registry.activate_detector.call_args.kwargs["training_provenance_path"] == str(promoted_provenance)
    assert fake_registry.activate_detector.call_args.kwargs["training_provenance"]["activation"]["detector_id"] == "custom_run-123"


def test_resolve_dvc_tracking_detects_ancestor_pointer(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    tracked_dir = repo_root / "backend" / "models" / "promoted" / "custom_run-123"
    tracked_dir.mkdir(parents=True)
    (repo_root / ".dvc").mkdir()
    (repo_root / ".dvc" / "config").write_text("[core]\n    no_scm = false\n", encoding="utf-8")
    tracked_file = tracked_dir / "best.pt"
    tracked_file.write_bytes(b"weights")
    pointer_file = tracked_dir.parent / "custom_run-123.dvc"
    pointer_file.write_text("outs:\n- path: custom_run-123\n", encoding="utf-8")

    tracking = resolve_dvc_tracking(tracked_file, repo_root)

    assert tracking is not None
    assert tracking["tracked"] is True
    assert tracking["tracking_scope"] == "ancestor"
    assert tracking["pointer_relative_path"] == "backend/models/promoted/custom_run-123.dvc"


def test_activate_registered_detector_accepts_whitespace_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_registry = Mock()
    fake_registry.activate_detector_id.return_value = {"id": "custom_live"}
    request = main.TrainingRegistryActivateRequest.model_validate({"detector_id": " custom_live "})

    monkeypatch.setattr(main, "require_training_available", lambda: (Mock(), fake_registry))
    monkeypatch.setattr(
        main,
        "build_training_registry_snapshot",
        lambda: {"detectors": [{"id": "custom_live"}], "active_detector": "soccana"},
    )

    response = main.activate_registered_detector(request)

    assert response == {"success": True, "active_detector": "custom_live"}
    fake_registry.activate_detector_id.assert_called_once_with("custom_live")


def test_inspect_training_dataset_finds_nested_dataset_yaml(tmp_path: Path) -> None:
    dataset_root = tmp_path / "detector-dataset"
    yaml_path = dataset_root / "nested" / "meta" / "dataset.yaml"
    train_image = dataset_root / "images" / "train" / "frame-1.png"
    val_image = dataset_root / "images" / "val" / "frame-2.png"
    train_label = dataset_root / "labels" / "train" / "frame-1.txt"
    val_label = dataset_root / "labels" / "val" / "frame-2.txt"

    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    train_image.parent.mkdir(parents=True, exist_ok=True)
    val_image.parent.mkdir(parents=True, exist_ok=True)
    train_label.parent.mkdir(parents=True, exist_ok=True)
    val_label.parent.mkdir(parents=True, exist_ok=True)

    yaml_path.write_text(
        "\n".join(
            [
                "path: ../..",
                "train: images/train",
                "val: images/val",
                "names:",
                "  0: player",
                "  1: ball",
                "  2: referee",
                "",
            ]
        ),
        encoding="utf-8",
    )
    train_image.write_bytes(b"fake")
    val_image.write_bytes(b"fake")
    train_label.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    val_label.write_text("1 0.5 0.5 0.1 0.1\n", encoding="utf-8")

    inspection = inspect_training_dataset(dataset_root)

    assert inspection.yaml_path == yaml_path.resolve()
    assert inspection.can_start is True
    assert inspection.errors == []
    assert inspection.splits["train"].images == 1
    assert inspection.splits["train"].label_files == 1
    assert inspection.splits["val"].images == 1


def test_inspect_training_dataset_flags_fiftyone_json_export_as_non_yolo(tmp_path: Path) -> None:
    dataset_root = tmp_path / "fiftyone-export"
    image_path = dataset_root / "data" / "data_0" / "frame-1.png"
    metadata_path = dataset_root / "metadata.json"
    samples_path = dataset_root / "samples.json"
    fiftyone_path = dataset_root / "fiftyone.yml"

    image_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.write_bytes(b"fake")
    fiftyone_path.write_text("dataset: demo\n", encoding="utf-8")
    metadata_path.write_text(
        json.dumps(
            {
                "sample_fields": [
                    {
                        "name": "ground_truth",
                        "embedded_doc_type": "fiftyone.core.labels.Detections",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    samples_path.write_text(json.dumps({"samples": [{"filepath": "data/data_0/frame-1.png"}]}), encoding="utf-8")

    inspection = inspect_training_dataset(dataset_root)
    fiftyone_errors = [error for error in inspection.errors if "FiftyOne-style dataset export" in error]

    assert inspection.can_start is False
    assert inspection.splits["train"].images == 1
    assert len(fiftyone_errors) == 1
    assert "convert or export this dataset to YOLO" in fiftyone_errors[0]


def test_training_registry_activate_detector_id_rejects_missing_custom_checkpoint(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry.json"
    missing_checkpoint = tmp_path / "missing-custom.pt"
    registry_path.write_text(
        json.dumps(
            {
                "active_detector": "soccana",
                "detectors": [
                    {
                        "id": "custom_dead",
                        "label": "Dead detector",
                        "path": str(missing_checkpoint),
                        "is_pretrained": False,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    registry = TrainingRegistry(registry_path)

    with pytest.raises(RuntimeError, match="checkpoint is missing"):
        registry.activate_detector_id("custom_dead")


def test_training_registry_activate_detector_id_rejects_missing_pretrained_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    registry_path = tmp_path / "registry.json"
    missing_pretrained = tmp_path / "missing-soccana.pt"

    monkeypatch.setattr(training_registry_module, "resolve_model_path", lambda model_name, model_kind: str(missing_pretrained))
    monkeypatch.setattr(training_registry_module, "resolve_registered_class_ids", lambda checkpoint_path: dict(DEFAULT_CLASS_IDS))

    registry = TrainingRegistry(registry_path)

    with pytest.raises(RuntimeError, match="checkpoint is missing"):
        registry.activate_detector_id("soccana")


def test_serialize_run_summary_stays_pydantic_v2_compatible_for_nested_review_data() -> None:
    serialized = main.serialize_run_summary(
        {
            "job_id": "job-456",
            "run_dir": "/tmp/fake-run",
            "field_calibration_refresh_frames": main.CALIBRATION_REFRESH_FRAMES,
            "field_calibration_refresh_attempts": 3,
            "field_calibration_refresh_successes": 2,
            "player_tracker_mode": "hybrid_reid",
            "resolved_player_tracker_mode": "hybrid_reid",
            "raw_unique_player_track_ids": 9,
            "tracklet_merges_applied": 2,
            "diagnostics": [
                {
                    "level": "warn",
                    "title": "Detector drift",
                    "message": "Custom detector could not be resolved.",
                    "next_step": "Fix the detector path.",
                    "implementation_diagnosis": "The active detector path failed before analysis started.",
                    "suggested_fix": "Repair the registry entry.",
                    "code_refs": ["backend/app/main.py::resolve_analysis_detector_model"],
                }
            ],
            "heuristic_diagnostics": [
                {
                    "level": "good",
                    "title": "Fallback remains available",
                    "message": "Heuristic diagnostics can still render.",
                    "next_step": "Use them only as backup.",
                }
            ],
            "top_tracks": [
                {
                    "track_id": 7,
                    "team_label": "home",
                    "frames": 14,
                }
            ],
        }
    )

    validated = RunSummary.model_validate(serialized)
    job_state = JobStateResponse.model_validate(
        {
            "job_id": "job-456",
            "status": "completed",
            "created_at": "2026-03-10T00:00:00Z",
            "progress": 100.0,
            "summary": serialized,
        }
    )

    assert validated.summary_version == main.SUMMARY_SCHEMA_VERSION
    assert validated.diagnostics[0].title == "Detector drift"
    assert validated.top_tracks[0].track_id == 7
    assert isinstance(job_state.summary, RunSummary)
    assert job_state.summary.heuristic_diagnostics[0].title == "Fallback remains available"


def test_pydantic_ai_output_validator_requires_warn_items_to_include_code_level_fields() -> None:
    with pytest.raises(ValueError, match="implementation_diagnosis"):
        ai_diagnostics._normalize_diagnostics_agent_output(
            DiagnosticsAgentOutput(
                summary_line="Calibration and tracking both look weak.",
                diagnostics=[
                    DiagnosticsAgentItem(
                        level="warn",
                        title="Missing code diagnosis",
                        message="The calibration gate is unstable.",
                        next_step="Inspect the exact rejection gate and patch it.",
                        suggested_fix="Log the rejection reason per frame.",
                        code_refs=["backend/app/wide_angle.py::analyze_video"],
                    ),
                    DiagnosticsAgentItem(
                        level="good",
                        title="Context",
                        message="Overlay still rendered.",
                        next_step="Keep the rest of the stack unchanged.",
                    ),
                    DiagnosticsAgentItem(
                        level="good",
                        title="Context 2",
                        message="Heuristic fallback exists.",
                        next_step="Use it only as backup.",
                    ),
                ],
            )
        )


def test_build_diagnostics_agent_accepts_structured_testmodel_output() -> None:
    agent = ai_diagnostics.build_diagnostics_agent(
        TestModel(
            custom_output_args={
                "summary_line": "Calibration and tracking both need attention.",
                "diagnostics": [
                    {
                        "level": "warn",
                        "title": "Detector class mapping drift",
                        "message": "Player classes do not line up with the emitted labels.",
                        "next_step": "Inspect the resolved class ids before touching tracker settings.",
                        "implementation_diagnosis": "The runtime filter is pointing at the wrong classes.",
                        "suggested_fix": "Resolve ids from checkpoint metadata or dataset YAML.",
                        "code_refs": ["backend/app/wide_angle.py::resolve_detector_spec"],
                        "evidence_keys": ["player_rows"],
                    },
                    {
                        "level": "good",
                        "title": "Tracker fallback",
                        "message": "Tracker wiring still initialized.",
                        "next_step": "Leave it unchanged until detections recover.",
                    },
                    {
                        "level": "good",
                        "title": "Projection context",
                        "message": "Projection stayed downstream-empty.",
                        "next_step": "Revisit projection only after detections recover.",
                    },
                ],
            }
        ),
        "prompt",
        max_output_tokens=3000,
        timeout_seconds=75.0,
    )

    result = agent.run_sync("context")

    assert isinstance(result.output, DiagnosticsAgentOutput)
    assert result.output.summary_line == "Calibration and tracking both need attention."
    assert result.output.diagnostics[0].code_refs == ["backend/app/wide_angle.py::resolve_detector_spec"]


def test_call_provider_uses_pydantic_ai_path(monkeypatch: pytest.MonkeyPatch) -> None:
    config = ai_diagnostics.ProviderConfig(
        provider="openai",
        model="gpt-5.4",
        base_url="https://api.openai.com/v1",
        api_key="test-key",
        timeout_seconds=30.0,
        extra_headers={},
        max_output_tokens=1200,
    )
    payload = json.dumps(
        {
            "summary_line": "PydanticAI path returned a typed diagnostics payload.",
            "diagnostics": [
                {
                    "level": "warn",
                    "title": "Detector mismatch",
                    "message": "Class ids do not line up.",
                    "next_step": "Fix the mapping.",
                    "implementation_diagnosis": "The runtime filter is pointing at the wrong classes.",
                    "suggested_fix": "Resolve ids from metadata.",
                    "code_refs": ["backend/app/wide_angle.py::resolve_detector_spec"],
                    "evidence_keys": ["player_rows"],
                },
                {
                    "level": "good",
                    "title": "Tracking context",
                    "message": "Tracker wiring still initialized.",
                    "next_step": "Leave it unchanged until detections recover.",
                },
                {
                    "level": "good",
                    "title": "Projection context",
                    "message": "Projection stayed downstream-empty.",
                    "next_step": "Revisit only after detections recover.",
                },
            ],
        }
    )

    monkeypatch.setattr(ai_diagnostics, "call_provider_via_pydantic_ai", lambda *_args, **_kwargs: payload)

    raw_text = ai_diagnostics.call_provider(
        config,
        system_prompt="prompt",
        context={"run_metrics": {}},
    )

    assert json.loads(raw_text)["summary_line"].startswith("PydanticAI path")


def test_generate_run_diagnostics_uses_heuristic_fallback_when_pydantic_ai_path_breaks(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = ai_diagnostics.ProviderConfig(
        provider="openai",
        model="gpt-5.4",
        base_url="https://api.openai.com/v1",
        api_key="test-key",
        timeout_seconds=30.0,
        extra_headers={},
        max_output_tokens=1200,
    )
    monkeypatch.setattr(ai_diagnostics, "resolve_provider_config", lambda: config)
    monkeypatch.setattr(
        ai_diagnostics,
        "call_provider_via_pydantic_ai",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("adapter import broke")),
    )
    monkeypatch.setattr(ai_diagnostics, "load_recent_logs", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(ai_diagnostics, "build_code_context", lambda *_args, **_kwargs: [])

    diagnostics, artifact = ai_diagnostics.generate_run_diagnostics(
        summary={"frames_processed": 12, "player_rows": 0, "ball_rows": 0},
        heuristic_diagnostics=[],
        outputs_dir=tmp_path,
        job_id="job-123",
        job_manager=None,
    )

    assert artifact["status"] == "failed"
    assert artifact["provider"] == "openai"
    assert "adapter import broke" in artifact["error"]
    assert diagnostics == artifact["diagnostics"]


def test_build_training_analysis_agent_accepts_structured_testmodel_output() -> None:
    agent = training_ai_analysis.build_training_analysis_agent(
        TestModel(
            custom_output_args={
                "summary_line": "Completed cleanly with a real validation split and a checkpoint worth holding for overlay review.",
                "overall_status": "mixed",
                "activation_recommendation": "hold",
                "sections": [
                    {
                        "id": "run_outcome",
                        "title": "Run outcome",
                        "status": "good",
                        "summary": "The worker completed.",
                        "details": "The run reached its planned terminal state with a checkpoint.",
                        "evidence": ["Status is completed."],
                        "actions": ["Use the rest of the sections before activating."],
                    },
                    {
                        "id": "dataset_contract",
                        "title": "Dataset contract",
                        "status": "neutral",
                        "summary": "The dataset contract is usable.",
                        "details": "Warnings exist, but the football mapping is still present.",
                        "evidence": ["Dataset tier is usable_with_warnings."],
                        "actions": ["Keep the scan snapshot attached to the run."],
                    },
                    {
                        "id": "training_dynamics",
                        "title": "Training dynamics",
                        "status": "good",
                        "summary": "Curves are present.",
                        "details": "Loss and optimizer samples were written over the run.",
                        "evidence": ["Loss points exist."],
                        "actions": ["Compare curve shape only against runs with the same validation contract."],
                    },
                    {
                        "id": "validation_signal",
                        "title": "Validation signal",
                        "status": "warn",
                        "summary": "Metrics are usable but not yet promotion-grade.",
                        "details": "The score is real, but the checkpoint should still wait for overlay validation.",
                        "evidence": ["mAP50 is present."],
                        "actions": ["Run overlay review before activation."],
                        "implementation_diagnosis": "Metric quality alone does not prove downstream football behavior.",
                        "suggested_fix": "Record overlay A/B review notes next to the checkpoint before promotion.",
                        "code_refs": ["backend/app/training_manager.py:697::_extract_final_metrics"],
                        "evidence_keys": ["metrics"],
                        "artifact_refs": ["results_csv"],
                    },
                    {
                        "id": "artifact_readiness",
                        "title": "Artifact readiness",
                        "status": "good",
                        "summary": "Checkpoint and provenance exist.",
                        "details": "The run wrote the expected checkpoint and lineage files.",
                        "evidence": ["best.pt exists."],
                        "actions": ["Keep provenance with the promoted checkpoint."],
                    },
                ],
            }
        ),
        "prompt",
        max_output_tokens=3000,
        timeout_seconds=75.0,
    )

    result = agent.run_sync("context")

    assert isinstance(result.output, training_ai_analysis.TrainingAnalysisOutput)
    assert result.output.sections[3].id == "validation_signal"
    assert result.output.sections[3].code_refs == ["backend/app/training_manager.py:697::_extract_final_metrics"]


def test_generate_training_run_analysis_uses_heuristic_fallback_when_pydantic_ai_path_breaks(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = ai_diagnostics.ProviderConfig(
        provider="openai",
        model="gpt-5.4",
        base_url="https://api.openai.com/v1",
        api_key="test-key",
        timeout_seconds=30.0,
        extra_headers={},
        max_output_tokens=1200,
    )
    monkeypatch.setattr(training_ai_analysis, "resolve_provider_config", lambda: config)
    monkeypatch.setattr(
        training_ai_analysis,
        "call_provider_via_pydantic_ai",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("training adapter import broke")),
    )

    run_dir = tmp_path / "training-run"
    run_dir.mkdir()
    analysis, artifact = training_ai_analysis.generate_training_run_analysis(
        {
          "run_id": "run-123",
          "run_dir": str(run_dir),
          "status": "failed",
          "progress": 1.0,
          "current_epoch": 0,
          "total_epochs": 50,
          "logs": ["[17:03:17] RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn"],
          "error": "Training process exited with code 1.",
          "config": {"base_weights": "soccana", "device": "auto"},
          "dataset_scan": {"tier": "valid", "class_mapping": {"player_class_ids": [0], "ball_class_ids": [1], "referee_class_ids": [2]}},
          "generated_split_lists": {},
          "metrics": {},
          "training_curves": {},
          "artifacts": {"train_log": str(run_dir / "train.log")},
        }
    )

    assert artifact["status"] == "failed"
    assert artifact["provider"] == "openai"
    assert "training adapter import broke" in artifact["error"]
    assert analysis["sections"][0]["id"] == "run_outcome"
    assert (run_dir / training_ai_analysis.ARTIFACT_FILENAME).exists()
    assert analysis["sections"][0]["code_refs"][0].startswith("backend/app/train_worker.py:")
