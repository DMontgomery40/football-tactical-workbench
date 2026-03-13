from __future__ import annotations

from pathlib import Path
from typing import Any

from .common import BenchmarkEvaluationUnavailable, metric_value
from .external_cli import run_external_json_command

TRACKING_REPO_DIR = Path(__file__).resolve().parents[2] / "third_party" / "soccernet" / "sn-tracking"
TRACKING_EVALUATOR_PATH = TRACKING_REPO_DIR / "tools" / "evaluate_soccernet_v3_tracking.py"
TRACKING_SEQMAP_PATH = TRACKING_REPO_DIR / "tools" / "SNMOT-test.txt"
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

    blockers.append(
        "Benchmark Lab does not yet convert benchmark recipe outputs into the TRACKERS_FOLDER_ZIP bundle that "
        "tools/evaluate_soccernet_v3_tracking.py expects. The vendored evaluator is present, but the recipe-to-"
        "submission bridge is still missing."
    )
    return blockers


def evaluate_tracking(
    *,
    suite: dict[str, Any],
    recipe: dict[str, Any],
    dataset_root: str,
    artifacts_dir: str | Path,
    benchmark_id: str,
) -> dict[str, Any]:
    blockers = probe_tracking_blockers(
        suite=suite,
        dataset_root=dataset_root,
        manifest_payload={},
    )
    tracker_submission_zip = Path(artifacts_dir).expanduser().resolve() / "tracker_submission.zip"
    gt_zip_path = Path(dataset_root).expanduser().resolve() / "gt.zip"
    if not tracker_submission_zip.exists():
        blockers.append(
            "No tracker submission ZIP was emitted for this benchmark cell. Expected "
            f"{tracker_submission_zip}."
        )
    if blockers:
        raise BenchmarkEvaluationUnavailable(" ".join(dict.fromkeys(blockers)))

    payload = run_external_json_command(
        command=[
            "python",
            str(TRACKING_EVALUATOR_PATH.relative_to(TRACKING_REPO_DIR)),
            "--BENCHMARK",
            "SNMOT",
            "--DO_PREPROC",
            "False",
            "--SEQMAP_FILE",
            str(TRACKING_SEQMAP_PATH.relative_to(TRACKING_REPO_DIR)),
            "--TRACKERS_TO_EVAL",
            recipe.get("id") or "recipe",
            "--SPLIT_TO_EVAL",
            "test",
            "--OUTPUT_SUB_FOLDER",
            "eval_results",
            "--TRACKERS_FOLDER_ZIP",
            str(tracker_submission_zip),
            "--GT_FOLDER_ZIP",
            str(gt_zip_path),
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
            **({"external_result_json": external_result_path} if external_result_path else {}),
        },
        "raw_result": payload,
    }
