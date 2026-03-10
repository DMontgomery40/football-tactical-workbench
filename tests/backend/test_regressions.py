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
from app.schemas import JobStateResponse, RunSummary  # noqa: E402
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
