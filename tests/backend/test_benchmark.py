from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT_DIR = Path(__file__).resolve().parents[2]
BACKEND_DIR = ROOT_DIR / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.benchmark import (
    SCORE_WEIGHT_CALIBRATION,
    SCORE_WEIGHT_COVERAGE,
    SCORE_WEIGHT_THROUGHPUT,
    SCORE_WEIGHT_TRACK_STABILITY,
    compute_composite_score,
    list_candidates,
)


# ---------------------------------------------------------------------------
# Composite scoring
# ---------------------------------------------------------------------------

def test_composite_score_weights_sum_to_one() -> None:
    total = (
        SCORE_WEIGHT_TRACK_STABILITY
        + SCORE_WEIGHT_CALIBRATION
        + SCORE_WEIGHT_COVERAGE
        + SCORE_WEIGHT_THROUGHPUT
    )
    assert abs(total - 1.0) < 1e-9


def test_composite_score_from_perfect_summary() -> None:
    summary = {
        "frames_processed": 100,
        "fps": 30,
        "average_track_length": 100,
        # NOTE: churn uses `or 1.0` default, so 0.0 is falsy and becomes 1.0.
        # Must pass a small nonzero value, or accept that 0.0 maps to "max churn".
        # The implementation treats missing/zero churn as worst-case (1.0).
        "player_track_churn_ratio": 0.01,
        "field_registered_ratio": 1.0,
        "average_player_detections_per_frame": 22,
    }
    result = compute_composite_score(summary)

    assert result["track_stability"] == 99.0  # (100/100) * (1 - 0.01) = 0.99 => 99.0
    assert result["calibration"] == 100.0
    assert result["coverage"] == 100.0
    assert result["throughput"] == 100.0
    # composite should be very close to 100
    assert result["composite"] > 99.0


def test_composite_score_from_empty_summary() -> None:
    result = compute_composite_score({})

    assert result["track_stability"] == 0.0
    assert result["calibration"] == 0.0
    assert result["coverage"] == 0.0
    assert result["throughput"] == 0.0
    assert result["composite"] == 0.0


def test_composite_score_partial_summary() -> None:
    summary = {
        "frames_processed": 200,
        "fps": 15,
        "average_track_length": 100,
        "player_track_churn_ratio": 0.5,
        "field_registered_ratio": 0.8,
        "average_player_detections_per_frame": 11,
    }
    result = compute_composite_score(summary)

    # track_stability: (100/200)=0.5 * (1-0.5)=0.5 => 0.25 => 25.0
    assert result["track_stability"] == 25.0
    # calibration: 0.8 => 80.0
    assert result["calibration"] == 80.0
    # coverage: 11/22 => 50.0
    assert result["coverage"] == 50.0
    # throughput: 15/30 => 50.0
    assert result["throughput"] == 50.0

    expected_composite = round(
        25.0 * SCORE_WEIGHT_TRACK_STABILITY
        + 80.0 * SCORE_WEIGHT_CALIBRATION
        + 50.0 * SCORE_WEIGHT_COVERAGE
        + 50.0 * SCORE_WEIGHT_THROUGHPUT,
        2,
    )
    assert result["composite"] == expected_composite


def test_composite_score_returns_expected_weights() -> None:
    result = compute_composite_score({"frames_processed": 1})
    weights = result["weights"]
    assert weights["track_stability"] == SCORE_WEIGHT_TRACK_STABILITY
    assert weights["calibration"] == SCORE_WEIGHT_CALIBRATION
    assert weights["coverage"] == SCORE_WEIGHT_COVERAGE
    assert weights["throughput"] == SCORE_WEIGHT_THROUGHPUT


def test_composite_score_caps_throughput_and_coverage_at_100() -> None:
    summary = {
        "frames_processed": 50,
        "fps": 120,
        "average_track_length": 500,
        "player_track_churn_ratio": 0.1,
        "field_registered_ratio": 1.0,
        "average_player_detections_per_frame": 44,
    }
    result = compute_composite_score(summary)

    assert result["throughput"] == 100.0
    assert result["coverage"] == 100.0
    # calibration is ratio * 100 without extra clamping; 1.0 => 100
    assert result["calibration"] == 100.0


# ---------------------------------------------------------------------------
# Candidate validation
# ---------------------------------------------------------------------------

def test_list_candidates_always_includes_pretrained(tmp_path: Path) -> None:
    fake_soccana = tmp_path / "soccana.pt"
    fake_soccana.write_bytes(b"fake")

    with patch("app.benchmark.resolve_model_path", return_value=str(fake_soccana)):
        with patch("app.benchmark._soccermaster_candidate", return_value=None):
            with patch("app.benchmark._registry_candidates", return_value=[]):
                with patch("app.benchmark._import_candidates", return_value=[]):
                    candidates = list_candidates()

    assert len(candidates) == 1
    assert candidates[0]["id"] == "soccana"
    assert candidates[0]["is_pretrained"] is True


def test_list_candidates_deduplicates_by_id(tmp_path: Path) -> None:
    fake_soccana = tmp_path / "soccana.pt"
    fake_soccana.write_bytes(b"fake")
    fake_custom = tmp_path / "custom.pt"
    fake_custom.write_bytes(b"fake")

    registry_dup = {
        "id": "soccana",
        "label": "soccana (dup)",
        "source": "registry",
        "path": str(fake_soccana),
        "is_pretrained": False,
    }
    registry_custom = {
        "id": "custom_run",
        "label": "Custom run",
        "source": "registry",
        "path": str(fake_custom),
        "is_pretrained": False,
    }

    with patch("app.benchmark.resolve_model_path", return_value=str(fake_soccana)):
        with patch("app.benchmark._soccermaster_candidate", return_value=None):
            with patch("app.benchmark._registry_candidates", return_value=[registry_dup, registry_custom]):
                with patch("app.benchmark._import_candidates", return_value=[]):
                    candidates = list_candidates()

    ids = [c["id"] for c in candidates]
    assert ids == ["soccana", "custom_run"]
    # soccana should be the pretrained version, not the registry duplicate
    assert candidates[0]["is_pretrained"] is True


# ---------------------------------------------------------------------------
# Endpoint contract checks
# ---------------------------------------------------------------------------

def test_benchmark_config_endpoint_returns_expected_keys() -> None:
    from app import main

    fake_clip = {"ready": False, "path": None}
    fake_candidates: list = []

    with patch("app.main.benchmark_clip_status", return_value=fake_clip):
        with patch("app.main.benchmark_list_candidates", return_value=fake_candidates):
            result = main.benchmark_config()

    assert "clip_status" in result
    assert "candidates" in result
    assert "runtime_profile" in result
    assert "benchmarks_dir" in result
