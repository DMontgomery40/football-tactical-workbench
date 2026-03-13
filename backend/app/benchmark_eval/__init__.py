from __future__ import annotations

from pathlib import Path
from typing import Any

from .calibration import evaluate_calibration
from .coco_detection import evaluate_coco_detection
from .common import BenchmarkEvaluationUnavailable
from .gamestate import evaluate_gamestate, probe_gamestate_blockers
from .operational import evaluate_operational
from .pcbas import evaluate_pcbas
from .runtime_profiles import runtime_profile
from .synloc import evaluate_synloc
from .team_spotting import evaluate_team_spotting
from .tracking import evaluate_tracking, probe_tracking_blockers

PROTOCOL_RUNNERS = {
    "coco_detection": evaluate_coco_detection,
    "synloc": evaluate_synloc,
    "team_spotting": evaluate_team_spotting,
    "calibration": evaluate_calibration,
    "tracking": evaluate_tracking,
    "pcbas": evaluate_pcbas,
    "gamestate": evaluate_gamestate,
    "operational": evaluate_operational,
}

PROTOCOL_BLOCKER_PROBES = {
    "tracking": probe_tracking_blockers,
    "gamestate": probe_gamestate_blockers,
}

PROTOCOL_RUNTIME_KEYS = {
    "coco_detection": "backend_default",
    "synloc": "backend_default",
    "team_spotting": "modern_action_spotting",
    "calibration": "sn_calibration_legacy",
    "tracking": "backend_default",
    "pcbas": "footpass_eval",
    "gamestate": "tracklab_gamestate_py39_np1",
    "operational": "backend_default",
}


def run_suite_evaluation(
    *,
    suite: dict[str, Any],
    recipe: dict[str, Any],
    dataset_root: str,
    artifacts_dir: str | Path,
    benchmark_id: str,
) -> dict[str, Any]:
    protocol = str(suite.get("protocol") or "")
    runner = PROTOCOL_RUNNERS.get(protocol)
    if runner is None:
        raise BenchmarkEvaluationUnavailable(f"Unsupported benchmark protocol: {protocol}")
    return runner(
        suite=suite,
        recipe=recipe,
        dataset_root=dataset_root,
        artifacts_dir=artifacts_dir,
        benchmark_id=benchmark_id,
    )


def probe_suite_blockers(
    *,
    suite: dict[str, Any],
    dataset_root: str,
    manifest_payload: dict[str, Any] | None = None,
) -> list[str]:
    protocol = str(suite.get("protocol") or "")
    probe = PROTOCOL_BLOCKER_PROBES.get(protocol)
    if probe is None:
        return []
    blockers = probe(
        suite=suite,
        dataset_root=dataset_root,
        manifest_payload=manifest_payload or {},
    )
    return [str(blocker).strip() for blocker in blockers if str(blocker).strip()]


def protocol_runtime_profile(protocol: str) -> dict[str, Any]:
    runtime_key = PROTOCOL_RUNTIME_KEYS.get(str(protocol or ""))
    if runtime_key is None:
        raise BenchmarkEvaluationUnavailable(f"Unsupported benchmark protocol: {protocol}")
    return runtime_profile(runtime_key)
