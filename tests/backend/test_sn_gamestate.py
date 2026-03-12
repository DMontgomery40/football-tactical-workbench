from __future__ import annotations

import asyncio
import json
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
from app import sn_gamestate  # noqa: E402


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


def test_run_sn_gamestate_analysis_uses_easyocr_fallback_when_mmcv_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repo_dir = tmp_path / "sn-gamestate"
    repo_dir.mkdir()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    source_video = tmp_path / "clip.mp4"
    source_video.write_bytes(b"fake")
    overlay_source = tmp_path / "overlay_source.mp4"
    overlay_source.write_bytes(b"fake")
    logged: list[str] = []
    captured_command: dict[str, list[str]] = {}

    monkeypatch.setattr(
        sn_gamestate,
        "sn_gamestate_status",
        lambda: {
            "available": True,
            "repo_path": str(repo_dir),
            "uv_path": "/opt/homebrew/bin/uv",
            "note": "ready",
            "evaluation": {"metric": "GS-HOTA"},
            "weights": {"mode": "automatic_first_run"},
        },
    )
    monkeypatch.setattr(
        sn_gamestate,
        "_sn_gamestate_python_module_available",
        lambda repo_path, module_name, uv_path=None: module_name == "easyocr",
    )
    monkeypatch.setattr(
        sn_gamestate,
        "_inspect_video",
        lambda path: {
            "fps": 25.0,
            "width": 1280,
            "height": 720,
            "frame_count": 100,
            "duration_seconds": 4.0,
        },
    )
    monkeypatch.setattr(sn_gamestate, "_find_visualization_video", lambda external_run_dir: overlay_source)
    monkeypatch.setattr(sn_gamestate.shutil, "copy2", lambda src, dst: None)
    monkeypatch.setattr(sn_gamestate.platform, "system", lambda: "Darwin")

    class DummyStdout:
        def __iter__(self):
            return iter(["tracklab line\n"])

        def close(self) -> None:
            return None

    class DummyProcess:
        def __init__(self) -> None:
            self.stdout = DummyStdout()

        def wait(self) -> int:
            return 0

    def fake_popen(command, cwd, stdout, stderr, text, bufsize, env):
        captured_command["command"] = list(command)
        captured_command["env"] = dict(env)
        return DummyProcess()

    monkeypatch.setattr(sn_gamestate.subprocess, "Popen", fake_popen)

    job_manager = SimpleNamespace(log=lambda _job_id, message: logged.append(message))
    summary = sn_gamestate.run_sn_gamestate_analysis(
        job_id="job-1",
        run_dir=run_dir,
        source_video_path=source_video,
        job_manager=job_manager,
    )

    assert captured_command["command"][:4] == ["/opt/homebrew/bin/uv", "run", "python", "-c"]
    assert "dataset.eval_set=val" in captured_command["command"]
    assert "test_tracking=True" in captured_command["command"]
    assert "num_cores=0" in captured_command["command"]
    assert "modules/jersey_number_detect=easyocr" in captured_command["command"]
    assert captured_command["env"]["PYTORCH_ENABLE_MPS_FALLBACK"] == "1"
    assert any("macOS stability mode" in message for message in logged)
    assert any("EasyOCR" in message for message in logged)
    assert summary["external_pipeline"]["jersey_number_backend"] == "easyocr_fallback"
    assert summary["external_pipeline"]["execution_mode"] == "cpu_safe_macos"
    summary_path = run_dir / "outputs" / "summary.json"
    assert summary_path.exists()
    saved = json.loads(summary_path.read_text(encoding="utf-8"))
    assert saved["external_pipeline"]["jersey_number_backend"] == "easyocr_fallback"
    assert saved["external_pipeline"]["execution_mode"] == "cpu_safe_macos"
