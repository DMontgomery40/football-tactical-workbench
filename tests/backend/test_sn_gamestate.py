from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

ROOT_DIR = Path(__file__).resolve().parents[2]
BACKEND_DIR = ROOT_DIR / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app import main  # noqa: E402


def test_config_includes_sn_gamestate_status(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(main, "resolve_provider_config", lambda: None)
    monkeypatch.setattr(main, "training_available", lambda: False)
    monkeypatch.setattr(main, "build_runtime_profile", lambda: {})
    monkeypatch.setattr(
        main,
        "sn_gamestate_status",
        lambda: {"available": True, "repo_path": "/tmp/sn-gamestate", "evaluation": {"metric": "GS-HOTA"}},
    )

    payload = main.config().model_dump()

    assert payload["sn_gamestate"]["available"] is True
    assert payload["sn_gamestate"]["evaluation"]["metric"] == "GS-HOTA"


def test_live_preview_rejects_sn_gamestate_pipeline() -> None:
    with pytest.raises(HTTPException) as exc_info:
        main.live_preview(source_id="source-123", pipeline="sn_gamestate")

    assert exc_info.value.status_code == 409
    assert "Live preview is not available" in str(exc_info.value.detail)


def test_analyze_skips_detector_resolution_for_sn_gamestate(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source_video = tmp_path / "clip.mp4"
    source_video.write_bytes(b"not-a-real-video")
    fake_job = SimpleNamespace(job_id="job-123")
    fake_runs_dir = tmp_path / "runs"

    monkeypatch.setattr(main, "RUNS_DIR", fake_runs_dir)
    monkeypatch.setattr(main, "resolve_source_path", lambda **_: source_video)
    monkeypatch.setattr(
        main,
        "resolve_analysis_detector_model",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("detector resolution should be skipped")),
    )
    monkeypatch.setattr(
        main,
        "job_manager",
        SimpleNamespace(
            create=lambda run_dir, restart_config: fake_job,
            log=lambda *_args, **_kwargs: None,
            update=lambda *_args, **_kwargs: None,
        ),
    )
    monkeypatch.setattr(main, "job_control_manager", SimpleNamespace(create=lambda job_id: None))

    class DummyThread:
        def __init__(self, target=None, args=(), daemon=None):
            self.target = target
            self.args = args
            self.daemon = daemon

        def start(self) -> None:
            return None

    monkeypatch.setattr(main.threading, "Thread", DummyThread)

    response = asyncio.run(
        main.analyze(
            video_file=None,
            local_video_path=str(source_video),
            source_id="",
            label_path="",
            pipeline="sn_gamestate",
            detector_model="soccana",
            player_model="",
            keypoint_model="soccana_keypoint",
            tracker_mode="hybrid_reid",
            include_ball=True,
            player_conf=0.25,
            ball_conf=0.20,
            iou=0.50,
        )
    )

    assert response.job_id == "job-123"
    assert response.run_id
    assert response.run_dir
