from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[2]
BACKEND_DIR = ROOT_DIR / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.benchmark_eval.common import BenchmarkEvaluationUnavailable
from app.benchmark_eval import operational


def test_soccermaster_operational_import_mismatch_becomes_blocked(monkeypatch, tmp_path: Path) -> None:
    def fail_analyze(**kwargs):  # noqa: ANN003
        raise ImportError("cannot import name 'is_offline_mode'")

    monkeypatch.setattr(operational, "analyze_wide_angle_video", fail_analyze)
    clip_path = tmp_path / "clip.mp4"
    clip_path.write_bytes(b"fake clip")

    with pytest.raises(BenchmarkEvaluationUnavailable) as exc_info:
        operational.evaluate_operational(
            suite={"id": "ops.clip_review_v1"},
            recipe={"id": "pipeline:soccermaster", "pipeline": "soccermaster"},
            dataset_root=str(clip_path),
            artifacts_dir=tmp_path / "artifacts",
            benchmark_id="test_operational",
        )

    assert "SoccerMaster operational review is blocked" in str(exc_info.value)
