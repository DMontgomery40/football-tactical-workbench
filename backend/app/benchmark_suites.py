from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from app.benchmark_eval import probe_suite_blockers
from app.training_provenance import probe_dvc_runtime, resolve_dvc_tracking

BASE_DIR = Path(__file__).resolve().parent.parent
BENCHMARKS_DIR = BASE_DIR / "benchmarks"
DATASETS_DIR = BENCHMARKS_DIR / "_datasets"
MANIFESTS_DIR = BENCHMARKS_DIR / "_manifests"
CONVERSIONS_DIR = BENCHMARKS_DIR / "_conversions"
SUITES_JSON_PATH = Path(__file__).resolve().with_name("benchmark_suites.json")
SOCCERNET_VENDOR_DIR = BASE_DIR / "third_party" / "soccernet"
TRACKLAB_DIR = SOCCERNET_VENDOR_DIR / "tracklab"
SN_TRACKING_DIR = SOCCERNET_VENDOR_DIR / "sn-tracking"
SN_GAMESTATE_DIR = SOCCERNET_VENDOR_DIR / "sn-gamestate"

DATASETS_DIR.mkdir(parents=True, exist_ok=True)
MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)
CONVERSIONS_DIR.mkdir(parents=True, exist_ok=True)


def _resolve_repo_path(raw_path: str | None) -> str:
    if not raw_path:
        return ""
    return str((BASE_DIR.parent / raw_path).resolve())


def _dedupe_paths(paths: list[Path]) -> list[Path]:
    seen: set[str] = set()
    result: list[Path] = []
    for path in paths:
        resolved = path.expanduser().resolve()
        key = str(resolved)
        if key in seen:
            continue
        seen.add(key)
        result.append(resolved)
    return result


def _suite_dataset_candidates(suite: dict[str, Any]) -> list[Path]:
    suite_id = str(suite.get("id") or "")
    dataset_root = Path(str(suite.get("dataset_root") or "")).expanduser().resolve() if str(suite.get("dataset_root") or "") else None
    fallback_roots = [
        Path(str(path)).expanduser().resolve()
        for path in (suite.get("fallback_dataset_roots") or [])
        if str(path).strip()
    ]

    if suite_id == "track.sn_tracking_medium_v1":
        candidates = [
            *(([dataset_root / "SoccerNetMOT", dataset_root] if dataset_root is not None else [])),
            TRACKLAB_DIR / "data" / "SoccerNetMOT",
        ]
        return _dedupe_paths(candidates)
    if suite_id.startswith("gsr."):
        candidates = [
            *(([dataset_root / "SoccerNetGS", dataset_root] if dataset_root is not None else [])),
            SN_GAMESTATE_DIR / "data" / "SoccerNetGS",
            TRACKLAB_DIR / "data" / "SoccerNetGS",
        ]
        return _dedupe_paths(candidates)
    return _dedupe_paths([*(fallback_roots or []), *([dataset_root] if dataset_root is not None else [])])


def _looks_like_json_file(path: Path) -> bool:
    try:
        prefix = path.read_bytes()[:128].lstrip()
    except OSError:
        return False
    return prefix.startswith((b"{", b"["))


def _dataset_structure_present(suite_id: str, dataset_root: Path) -> bool:
    if suite_id == "track.sn_tracking_medium_v1":
        return dataset_root.exists() and (dataset_root / "train").exists() and (dataset_root / "test").exists()
    if suite_id == "gsr.medium_v1":
        valid_root = dataset_root / "valid"
        return valid_root.exists() and any(
            path.is_file() and path.name == "Labels-GameState.json" and _looks_like_json_file(path)
            for path in valid_root.rglob("Labels-GameState.json")
        )
    if suite_id == "gsr.long_v1":
        return (
            dataset_root.exists()
            and all((dataset_root / split).exists() for split in ("train", "valid", "test"))
            and any(
                path.is_file() and path.name == "Labels-GameState.json" and _looks_like_json_file(path)
                for path in (dataset_root / "valid").rglob("Labels-GameState.json")
            )
        )
    if suite_id == "calib.sn_calib_medium_v1":
        split_dir = dataset_root / "valid"
        return split_dir.exists() and any(
            path.is_file()
            and path.suffix == ".json"
            and path.name != "per_match_info.json"
            and _looks_like_json_file(path)
            for path in split_dir.iterdir()
        )
    if suite_id == "spot.team_bas_quick_v1":
        return dataset_root.exists() and any(
            path.is_file()
            and path.name == "Labels-ball.json"
            and _looks_like_json_file(path)
            and any(
                (path.parent / candidate).exists()
                for candidate in ("224p.mp4", "720p.mp4", "1_224p.mp4", "1_720p.mp4")
            )
            for path in dataset_root.rglob("Labels-ball.json")
        )
    if suite_id == "spot.pcbas_medium_v1":
        return dataset_root.exists() and any(
            candidate.exists() and _looks_like_json_file(candidate)
            for candidate in (
                dataset_root / "playbyplay_GT" / "playbyplay_val.json",
                dataset_root / "playbyplay_val.json",
            )
        )
    return dataset_root.exists()


def _dataset_blocker_message(suite_id: str, dataset_root: Path | None, suite: dict[str, Any]) -> str:
    root_text = str(dataset_root or str(suite.get("dataset_root") or ""))
    if suite_id == "calib.sn_calib_medium_v1":
        return (
            "Calibration dataset materialization is missing. Expected a split directory at "
            f"{Path(root_text) / 'valid'} with per-frame annotation JSON files and matching source images."
        )
    if suite_id == "spot.team_bas_quick_v1":
        return (
            "Team spotting dataset materialization is missing. Expected official SoccerNet-Ball game paths under "
            f"{root_text} with per-game Labels-ball.json files and at least one source video (224p.mp4 or 720p.mp4)."
        )
    if suite_id == "spot.pcbas_medium_v1":
        return (
            "FOOTPASS dataset materialization is missing. Expected a play-by-play ground-truth JSON at "
            f"{Path(root_text) / 'playbyplay_GT' / 'playbyplay_val.json'} or {Path(root_text) / 'playbyplay_val.json'}."
        )
    return (
        "Dataset root is missing. Materialize the benchmark dataset under "
        f"{root_text}."
    )


def _preferred_dataset_root(suite: dict[str, Any]) -> Path | None:
    candidates = _suite_dataset_candidates(suite)
    return candidates[0] if candidates else None


def _existing_dataset_root(suite: dict[str, Any]) -> Path | None:
    suite_id = str(suite.get("id") or "")
    for candidate in _suite_dataset_candidates(suite):
        if _dataset_structure_present(suite_id, candidate):
            return candidate
    return None


def _suite_execution_blocker(suite: dict[str, Any]) -> str | None:
    suite_id = str(suite.get("id") or "")
    if suite_id == "track.sn_tracking_medium_v1":
        evaluator_path = SN_TRACKING_DIR / "tools" / "evaluate_soccernet_v3_tracking.py"
        dataset_wrapper = TRACKLAB_DIR / "tracklab" / "wrappers" / "dataset" / "soccernet" / "soccernet_mot.py"
        if not evaluator_path.exists():
            return (
                "Vendored tracking evaluator is missing: "
                f"expected {evaluator_path} from sn-tracking."
            )
        if not dataset_wrapper.exists():
            return (
                "Vendored TrackLab SoccerNetMOT dataset wrapper is missing: "
                f"expected {dataset_wrapper}."
            )
        return (
            "Tracking cells are still blocked because Benchmark Lab does not yet export per-recipe "
            f"SoccerNetMOT prediction archives for the vendored evaluator at {evaluator_path}; "
            f"the TrackLab dataset wrapper lives at {dataset_wrapper}, but the current adapter still "
            "assumes a generic evaluate.py entrypoint instead of the real sn-tracking zip-based flow."
        )
    if suite_id.startswith("gsr."):
        baseline_config = SN_GAMESTATE_DIR / "sn_gamestate" / "configs" / "soccernet.yaml"
        dataset_wrapper = TRACKLAB_DIR / "tracklab" / "wrappers" / "dataset" / "soccernet" / "soccernet_game_state.py"
        if not baseline_config.exists():
            return (
                "Vendored sn-gamestate baseline config is missing: "
                f"expected {baseline_config}."
            )
        if not dataset_wrapper.exists():
            return (
                "Vendored TrackLab SoccerNetGS dataset wrapper is missing: "
                f"expected {dataset_wrapper}."
            )
        return (
            "Game-state cells are still blocked because Benchmark Lab does not yet run the vendored "
            f"TrackLab/sn-gamestate pipeline per recipe to emit GS-HOTA inputs; the baseline config "
            f"is {baseline_config} and the TrackLab dataset wrapper is {dataset_wrapper}, but the "
            "current adapter still assumes a generic evaluate.py entrypoint instead of the TrackLab config flow."
        )
    return None


def _suite_note(
    *,
    suite: dict[str, Any],
    dataset_root: Path | None,
    dataset_exists: bool,
    manifest_path: Path | None,
    manifest_exists: bool,
) -> str | None:
    suite_id = str(suite.get("id") or "")
    if bool(suite.get("requires_clip")):
        return "Prepare the benchmark clip before running operational review."
    if suite_id == "track.sn_tracking_medium_v1":
        expected_root = dataset_root or _preferred_dataset_root(suite)
        blocker = _suite_execution_blocker(suite)
        root_text = str(expected_root) if expected_root is not None else "backend/benchmarks/_datasets/track.sn_tracking_medium_v1/SoccerNetMOT"
        dataset_clause = (
            f"Expected a SoccerNetMOT tree at {root_text} with train/ and test/ splits."
            if not dataset_exists
            else f"SoccerNetMOT tree is present at {root_text}."
        )
        manifest_clause = (
            f" Manifest path: {manifest_path}."
            if manifest_exists and manifest_path is not None
            else f" Missing manifest at {manifest_path}."
            if manifest_path is not None
            else ""
        )
        blocker_clause = f" {blocker}" if blocker else ""
        return f"{dataset_clause}{manifest_clause}{blocker_clause}".strip()
    if suite_id == "gsr.medium_v1":
        expected_root = dataset_root or _preferred_dataset_root(suite)
        blocker = _suite_execution_blocker(suite)
        root_text = str(expected_root) if expected_root is not None else "backend/benchmarks/_datasets/gsr.medium_v1/SoccerNetGS"
        dataset_clause = (
            f"Expected a SoccerNetGS tree at {root_text} with a valid/ split and Labels-GameState.json files for the fixed 12-clip subset."
            if not dataset_exists
            else f"SoccerNetGS tree is present at {root_text}."
        )
        manifest_clause = (
            f" Manifest path: {manifest_path}."
            if manifest_exists and manifest_path is not None
            else f" Missing manifest at {manifest_path}."
            if manifest_path is not None
            else ""
        )
        blocker_clause = f" {blocker}" if blocker else ""
        return f"{dataset_clause}{manifest_clause}{blocker_clause}".strip()
    if suite_id == "gsr.long_v1":
        expected_root = dataset_root or _preferred_dataset_root(suite)
        blocker = _suite_execution_blocker(suite)
        root_text = str(expected_root) if expected_root is not None else "backend/benchmarks/_datasets/gsr.long_v1/SoccerNetGS"
        dataset_clause = (
            f"Expected a full SoccerNetGS tree at {root_text} with train/, valid/, and test/ splits."
            if not dataset_exists
            else f"SoccerNetGS tree is present at {root_text}."
        )
        manifest_clause = (
            f" Manifest path: {manifest_path}."
            if manifest_exists and manifest_path is not None
            else f" Missing manifest at {manifest_path}."
            if manifest_path is not None
            else ""
        )
        blocker_clause = f" {blocker}" if blocker else ""
        return f"{dataset_clause}{manifest_clause}{blocker_clause}".strip()
    if dataset_exists and manifest_exists:
        return None
    return "Materialize the suite dataset and manifest before running this benchmark."


@lru_cache(maxsize=1)
def _load_suites_payload() -> dict[str, Any]:
    payload = json.loads(SUITES_JSON_PATH.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Benchmark suites config is not a JSON object: {SUITES_JSON_PATH}")
    return payload


def list_suite_definitions() -> list[dict[str, Any]]:
    payload = _load_suites_payload()
    suites = payload.get("suites") or []
    result: list[dict[str, Any]] = []
    for suite in suites:
        if not isinstance(suite, dict):
            continue
        normalized = dict(suite)
        normalized["dataset_root"] = _resolve_repo_path(str(suite.get("dataset_root") or ""))
        normalized["fallback_dataset_roots"] = [
            _resolve_repo_path(str(path))
            for path in (suite.get("fallback_dataset_roots") or [])
            if str(path).strip()
        ]
        normalized["manifest_path"] = _resolve_repo_path(str(suite.get("manifest_path") or ""))
        result.append(normalized)
    return result


def get_suite_definition(suite_id: str) -> dict[str, Any]:
    normalized_id = str(suite_id or "").strip()
    for suite in list_suite_definitions():
        if str(suite.get("id") or "") == normalized_id:
            return suite
    raise KeyError(f"Unknown benchmark suite: {suite_id}")


def suite_manifest_payload(suite: dict[str, Any]) -> dict[str, Any]:
    manifest_path = Path(str(suite.get("manifest_path") or ""))
    if not manifest_path.exists():
        return {}
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _manifest_summary(payload: dict[str, Any]) -> dict[str, Any]:
    items = payload.get("items")
    classes = payload.get("classes")
    materialization = payload.get("materialization")
    materialization_status = materialization.get("status") if isinstance(materialization, dict) else None
    return {
        "kind": payload.get("kind"),
        "split": payload.get("split"),
        "selection": payload.get("selection"),
        "item_count": len(items) if isinstance(items, list) else None,
        "class_count": len(classes) if isinstance(classes, list) else None,
        "task_coverage": list(payload.get("task_coverage") or []),
        "materialization_status": materialization_status,
    }


def _manifest_materialization_blockers(payload: dict[str, Any]) -> list[str]:
    materialization = payload.get("materialization")
    if not isinstance(materialization, dict):
        return []
    blockers = materialization.get("blockers")
    if not isinstance(blockers, list):
        return []
    return [str(blocker).strip() for blocker in blockers if str(blocker).strip()]


def resolve_suite_dataset_root(suite: dict[str, Any]) -> str:
    existing_root = _existing_dataset_root(suite)
    if existing_root is not None:
        return str(existing_root)
    preferred_root = _preferred_dataset_root(suite)
    return str(preferred_root) if preferred_root is not None else str(suite.get("dataset_root") or "")


def build_suite_dataset_state(suite: dict[str, Any]) -> dict[str, Any]:
    resolved_dataset_root = resolve_suite_dataset_root(suite)
    dataset_root = Path(resolved_dataset_root).expanduser().resolve() if resolved_dataset_root else None
    manifest_path = Path(str(suite.get("manifest_path") or "")).expanduser().resolve() if str(suite.get("manifest_path") or "") else None
    suite_id = str(suite.get("id") or "")
    manifest_payload = suite_manifest_payload(suite)
    manifest_materialization_blockers = _manifest_materialization_blockers(manifest_payload)
    dataset_exists = bool(dataset_root) and _dataset_structure_present(suite_id, dataset_root)
    manifest_exists = bool(manifest_path) and manifest_path.exists()
    blockers: list[str] = []
    if not bool(suite.get("requires_clip")):
        if not dataset_exists and not manifest_materialization_blockers:
            blockers.append(_dataset_blocker_message(suite_id, dataset_root, suite))
        if not manifest_exists:
            blockers.append(
                "Manifest is missing. Expected "
                f"{manifest_path or str(suite.get('manifest_path') or '')}."
            )
    blockers.extend(
        probe_suite_blockers(
            suite=suite,
            dataset_root=str(dataset_root) if dataset_root is not None else "",
            manifest_payload=manifest_payload,
        )
    )
    blockers.extend(manifest_materialization_blockers)
    blockers = [str(blocker).strip() for blocker in blockers if str(blocker).strip()]
    ready = len(blockers) == 0 and (bool(suite.get("requires_clip")) or (dataset_exists and manifest_exists))
    note = None if ready else " ".join(dict.fromkeys(blockers))
    return {
        "suite_id": suite.get("id"),
        "dataset_root": str(dataset_root) if dataset_root is not None else None,
        "dataset_exists": dataset_exists,
        "dataset_dvc": resolve_dvc_tracking(str(dataset_root)) if dataset_exists and dataset_root is not None else None,
        "manifest_path": str(manifest_path) if manifest_path is not None else None,
        "manifest_exists": manifest_exists,
        "manifest_dvc": resolve_dvc_tracking(str(manifest_path)) if manifest_exists and manifest_path is not None else None,
        "conversion_root": str((CONVERSIONS_DIR / str(suite.get("id") or "")).resolve()),
        "ready": ready,
        "readiness_status": "ready" if ready else "blocked",
        "requires_clip": bool(suite.get("requires_clip")),
        "dvc_required": bool(suite.get("dvc_required", False)),
        "dvc_runtime": probe_dvc_runtime(),
        "note": note,
        "blockers": blockers,
        "manifest_summary": _manifest_summary(manifest_payload),
    }


def list_suite_dataset_states() -> list[dict[str, Any]]:
    return [build_suite_dataset_state(suite) for suite in list_suite_definitions()]
