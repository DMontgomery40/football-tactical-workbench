from __future__ import annotations

from pathlib import Path
from typing import Any

from .common import BenchmarkEvaluationUnavailable, metric_value
from .external_cli import run_external_json_command
from .prediction_exports import ensure_tracking_prediction_export

TRACKING_REPO_DIR = Path(__file__).resolve().parents[2] / "third_party" / "soccernet" / "sn-tracking"
TRACKING_EVALUATOR_PATH = TRACKING_REPO_DIR / "tools" / "evaluate_soccernet_v3_tracking.py"
TRACKING_SEQMAP_PATH = TRACKING_REPO_DIR / "tools" / "SNMOT-test.txt"
TRACKING_WRAPPER_PATH = Path(__file__).resolve().with_name("run_tracking_eval.py")
TRACKING_RUNTIME_KEY = "backend_default"


def probe_tracking_blockers(
    *,
    suite: dict[str, Any],
    dataset_root: str,
    manifest_payload: dict[str, Any] | None = None,
) -> list[str]:
    blockers: list[str] = []
    dataset_path = Path(dataset_root).expanduser().resolve() if dataset_root else None
    manifest_payload = manifest_payload or {}
    materialization = manifest_payload.get("materialization")
    manifest_materialization_blockers = materialization.get("blockers") if isinstance(materialization, dict) else None
    has_manifest_materialization_blockers = isinstance(manifest_materialization_blockers, list) and bool(manifest_materialization_blockers)

    if not TRACKING_REPO_DIR.exists():
        blockers.append(
            "Vendored sn-tracking sources are missing under backend/third_party/soccernet/sn-tracking."
        )
        return blockers

    if not TRACKING_EVALUATOR_PATH.exists():
        blockers.append(
            "The vendored sn-tracking checkout does not expose the expected evaluator entrypoint "
            f"{TRACKING_EVALUATOR_PATH}."
        )

    if not TRACKING_SEQMAP_PATH.exists():
        blockers.append(
            "The vendored sn-tracking checkout is missing tools/SNMOT-test.txt, so the benchmark "
            "cannot point the evaluator at the fixed SoccerNet tracking split."
        )

    if (dataset_path is None or not dataset_path.exists()) and not has_manifest_materialization_blockers:
        blockers.append(
            "Tracking dataset materialization is missing. Expected a SoccerNetMOT tree under "
            f"{dataset_root or 'backend/benchmarks/_datasets/track.sn_tracking_medium_v1/SoccerNetMOT'} "
            "with train/ and test/ splits."
        )

    if dataset_path is not None and dataset_path.exists():
        gt_zip_path = dataset_path / "gt.zip"
        if not gt_zip_path.exists():
            blockers.append(
                "Tracking dataset materialization is incomplete: expected the SoccerNet ground-truth ZIP at "
                f"{gt_zip_path}."
            )

        sample_submission_path = dataset_path / "sample_submission.zip"
        if not sample_submission_path.exists():
            blockers.append(
                "Tracking suite materialization is incomplete: expected a sample or template tracker submission ZIP at "
                f"{sample_submission_path} so the conversion path can target the actual sn-tracking input contract."
            )

    return blockers


def _resolve_tracking_seqmap_path(*, dataset_root: str, recipe: dict[str, Any]) -> Path:
    candidate_keys = ("tracking_seqmap_file", "seqmap_file")
    for key in candidate_keys:
        raw_value = str(recipe.get(key) or "").strip()
        if raw_value:
            candidate = Path(raw_value).expanduser().resolve()
            if candidate.exists():
                return candidate
    dataset_path = Path(dataset_root).expanduser().resolve() if dataset_root else None
    if dataset_path is not None:
        for relative_name in ("seqmap.txt", "SNMOT-test.txt"):
            candidate = dataset_path / relative_name
            if candidate.exists():
                return candidate
    return TRACKING_SEQMAP_PATH


def evaluate_tracking(
    *,
    suite: dict[str, Any],
    recipe: dict[str, Any],
    dataset_root: str,
    artifacts_dir: str | Path,
    benchmark_id: str,
) -> dict[str, Any]:
    export_info = ensure_tracking_prediction_export(
        recipe=recipe,
        dataset_root=dataset_root,
        artifacts_dir=artifacts_dir,
    )
    tracker_submission_zip = Path(str(export_info["tracker_submission_zip"])).expanduser().resolve()
    gt_zip_path = Path(dataset_root).expanduser().resolve() / "gt.zip"
    seqmap_path = _resolve_tracking_seqmap_path(dataset_root=dataset_root, recipe=recipe)
    blockers: list[str] = []
    if not gt_zip_path.exists():
        blockers.append(
            "Tracking ground-truth ZIP is missing. Expected "
            f"{gt_zip_path}."
        )
    if not tracker_submission_zip.exists():
        blockers.append(
            "Tracking prediction export did not emit the TRACKERS_FOLDER_ZIP bundle expected by the evaluator. "
            f"Expected {tracker_submission_zip}."
        )
    if not seqmap_path.exists():
        blockers.append(
            "Tracking evaluation requires a seqmap file. Expected either a dataset-local seqmap under "
            f"{Path(dataset_root).expanduser().resolve() / 'seqmap.txt'} or the vendored default at {TRACKING_SEQMAP_PATH}."
        )
    if blockers:
        raise BenchmarkEvaluationUnavailable(" ".join(dict.fromkeys(blockers)))

    payload = run_external_json_command(
        command=[
            "python",
            str(TRACKING_WRAPPER_PATH),
            "--tracker-submission-zip",
            str(tracker_submission_zip),
            "--ground-truth-zip",
            str(gt_zip_path),
            "--seqmap-file",
            str(seqmap_path),
            "--artifacts-dir",
            str(Path(artifacts_dir).expanduser().resolve()),
            "--tracker-name",
            "benchmark",
        ],
        cwd=TRACKING_REPO_DIR,
        artifacts_dir=artifacts_dir,
        runtime_key=TRACKING_RUNTIME_KEY,
    )
    external_result_path = payload.pop("_external_result_path", None)
    return {
        "metrics": {
            "hota": metric_value(payload.get("hota"), label="HOTA"),
            "deta": metric_value(payload.get("deta"), label="DetA"),
            "assa": metric_value(payload.get("assa"), label="AssA"),
            "frames_per_second": metric_value(payload.get("frames_per_second"), label="Frames/s", precision=2),
        },
        "artifacts": {
            "tracker_submission_zip": str(tracker_submission_zip),
            "prediction_export_summary_json": export_info.get("export_summary_json"),
            "source_prediction_artifact": export_info.get("source_prediction_artifact"),
            "seqmap_file": str(seqmap_path),
            **({"external_result_json": external_result_path} if external_result_path else {}),
        },
        "raw_result": payload,
    }
