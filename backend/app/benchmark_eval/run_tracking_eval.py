from __future__ import annotations

import argparse
import contextlib
import io
import json
import shutil
import tempfile
import time
import zipfile
from pathlib import Path

import trackeval


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracker-submission-zip", required=True, type=Path)
    parser.add_argument("--ground-truth-zip", required=True, type=Path)
    parser.add_argument("--seqmap-file", required=True, type=Path)
    parser.add_argument("--artifacts-dir", required=True, type=Path)
    parser.add_argument("--tracker-name", default="benchmark", type=str)
    return parser.parse_args()


def _extract_tracker_zip(*, tracker_submission_zip: Path, scratch_dir: Path, tracker_name: str) -> Path:
    trackers_root = scratch_dir / "SNMOT-test" / tracker_name / "data"
    trackers_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(tracker_submission_zip, "r") as archive:
        archive.extractall(trackers_root)
    return trackers_root


def _find_split_root(root: Path) -> Path | None:
    direct_candidates = [
        root / "SNMOT-test",
        root / "test",
        root / "test-evalAI",
        root / "SNMOT-test_0" / "SNMOT-test",
        root / "SNMOT-test_0" / "test",
        root / "SNMOT-test_0" / "test-evalAI",
    ]
    for candidate in direct_candidates:
        if candidate.exists():
            return candidate
    for candidate in root.rglob("*"):
        if not candidate.is_dir():
            continue
        if candidate.name in {"SNMOT-test", "test", "test-evalAI"}:
            return candidate
    return None


def _extract_gt_zip(*, ground_truth_zip: Path, scratch_dir: Path) -> Path:
    extract_root = scratch_dir / "gt_extract"
    extract_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ground_truth_zip, "r") as archive:
        archive.extractall(extract_root)

    split_root = _find_split_root(extract_root)
    if split_root is None:
        raise RuntimeError(
            "Tracking ground-truth ZIP did not expose a usable SNMOT test split. "
            "Expected a tree containing test/ or SNMOT-test/ with seqinfo.ini and gt/gt.txt."
        )

    final_gt_root = scratch_dir / "gt" / "SNMOT-test"
    final_gt_root.parent.mkdir(parents=True, exist_ok=True)
    if final_gt_root.exists():
        shutil.rmtree(final_gt_root)
    if split_root.name == "SNMOT-test":
        shutil.copytree(split_root, final_gt_root)
    else:
        shutil.copytree(split_root, final_gt_root)
    return final_gt_root


def _seqmap_sequences(seqmap_file: Path) -> list[str]:
    lines = [
        line.strip()
        for line in seqmap_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not lines:
        return []
    if lines[0].lower() == "name":
        return lines[1:]
    return lines


def _seq_lengths_from_gt(gt_root: Path, sequence_names: list[str]) -> tuple[dict[str, int], int]:
    seq_info: dict[str, int] = {}
    total_frames = 0
    for sequence_name in sequence_names:
        seqinfo_path = gt_root / sequence_name / "seqinfo.ini"
        seq_length = None
        if seqinfo_path.exists():
            for line in seqinfo_path.read_text(encoding="utf-8").splitlines():
                if line.startswith("seqLength="):
                    try:
                        seq_length = int(line.split("=", 1)[1].strip())
                    except ValueError:
                        seq_length = None
                    break
        if seq_length is None:
            gt_path = gt_root / sequence_name / "gt" / "gt.txt"
            if gt_path.exists():
                try:
                    frames = [
                        int(row.split(",", 1)[0])
                        for row in gt_path.read_text(encoding="utf-8").splitlines()
                        if row.strip()
                    ]
                except ValueError:
                    frames = []
                seq_length = max(frames) if frames else 0
            else:
                seq_length = 0
        seq_info[sequence_name] = seq_length
        total_frames += seq_length
    return seq_info, total_frames


def _float_ratio(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(str(value)) / 100.0
    except (TypeError, ValueError):
        return None


def _extract_metrics(output_res: dict[str, object], *, tracker_name: str) -> dict[str, object]:
    dataset_name, dataset_payload = next(iter(output_res.items()))
    tracker_payload = dict(dataset_payload or {}).get(tracker_name)
    if not isinstance(tracker_payload, dict):
        raise RuntimeError(f"TrackEval did not return results for tracker '{tracker_name}'.")
    summaries = tracker_payload.get("SUMMARIES")
    if not isinstance(summaries, dict):
        raise RuntimeError("TrackEval did not emit summary metrics.")

    summary_key = None
    hota_summary = None
    for candidate_key in ("cls_comb_det_av", "pedestrian", "all", "cls_comb_cls_av"):
        candidate_summary = summaries.get(candidate_key)
        if isinstance(candidate_summary, dict) and isinstance(candidate_summary.get("HOTA"), dict):
            summary_key = candidate_key
            hota_summary = candidate_summary["HOTA"]
            break
    if hota_summary is None:
        raise RuntimeError(
            f"TrackEval summaries did not expose HOTA metrics under a recognized class key. "
            f"Available keys: {sorted(summaries.keys())}"
        )

    return {
        "dataset_name": dataset_name,
        "summary_class": summary_key,
        "summary_metrics": hota_summary,
        "hota": _float_ratio(hota_summary.get("HOTA")),
        "deta": _float_ratio(hota_summary.get("DetA")),
        "assa": _float_ratio(hota_summary.get("AssA")),
    }


def main() -> None:
    args = _parse_args()
    artifacts_dir = args.artifacts_dir.expanduser().resolve()
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    tracker_submission_zip = args.tracker_submission_zip.expanduser().resolve()
    ground_truth_zip = args.ground_truth_zip.expanduser().resolve()
    seqmap_file = args.seqmap_file.expanduser().resolve()
    if not tracker_submission_zip.exists():
        raise FileNotFoundError(f"Tracking submission ZIP is missing: {tracker_submission_zip}")
    if not ground_truth_zip.exists():
        raise FileNotFoundError(f"Tracking ground-truth ZIP is missing: {ground_truth_zip}")
    if not seqmap_file.exists():
        raise FileNotFoundError(f"Tracking seqmap file is missing: {seqmap_file}")

    tracker_name = str(args.tracker_name or "benchmark").strip() or "benchmark"
    eval_output_dir = artifacts_dir / "eval_results"
    eval_output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="benchmark_tracking_eval_", dir=str(artifacts_dir)) as scratch:
        scratch_dir = Path(scratch)
        _extract_tracker_zip(
            tracker_submission_zip=tracker_submission_zip,
            scratch_dir=scratch_dir,
            tracker_name=tracker_name,
        )
        gt_root = _extract_gt_zip(
            ground_truth_zip=ground_truth_zip,
            scratch_dir=scratch_dir,
        )
        sequence_names = _seqmap_sequences(seqmap_file)
        seq_info, total_frames = _seq_lengths_from_gt(gt_root, sequence_names)

        default_eval_config = trackeval.Evaluator.get_default_eval_config()
        default_eval_config.update(
            {
                "USE_PARALLEL": False,
                "BREAK_ON_ERROR": True,
                "PRINT_RESULTS": False,
                "PRINT_ONLY_COMBINED": True,
                "PRINT_CONFIG": False,
                "TIME_PROGRESS": False,
                "DISPLAY_LESS_PROGRESS": False,
                "OUTPUT_SUMMARY": True,
                "OUTPUT_DETAILED": True,
                "PLOT_CURVES": False,
            }
        )
        dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
        dataset_config.update(
            {
                "GT_FOLDER": str(scratch_dir / "gt"),
                "TRACKERS_FOLDER": str(scratch_dir),
                "OUTPUT_FOLDER": str(eval_output_dir),
                "TRACKERS_TO_EVAL": [tracker_name],
                "BENCHMARK": "SNMOT",
                "SPLIT_TO_EVAL": "test",
                "DO_PREPROC": False,
                "PRINT_CONFIG": False,
                "TRACKER_SUB_FOLDER": "data",
                "OUTPUT_SUB_FOLDER": "eval_results",
                "SEQMAP_FILE": str(seqmap_file),
                "SEQ_INFO": seq_info,
            }
        )
        metrics_config = {
            "METRICS": ["HOTA", "CLEAR", "Identity"],
            "THRESHOLD": 0.5,
            "PRINT_CONFIG": False,
        }
        started = time.perf_counter()
        with contextlib.redirect_stdout(io.StringIO()) as captured_stdout:
            metrics_list = [
                metric(metrics_config)
                for metric in (
                    trackeval.metrics.HOTA,
                    trackeval.metrics.CLEAR,
                    trackeval.metrics.Identity,
                )
            ]
            evaluator = trackeval.Evaluator(default_eval_config)
            dataset = trackeval.datasets.MotChallenge2DBox(dataset_config)
            output_res, _ = evaluator.evaluate([dataset], metrics_list)
        elapsed_seconds = max(time.perf_counter() - started, 1e-9)
        metrics_payload = _extract_metrics(output_res, tracker_name=tracker_name)

    frames_per_second = (total_frames / elapsed_seconds) if total_frames > 0 else None
    print(
        json.dumps(
            {
                "hota": metrics_payload["hota"],
                "deta": metrics_payload["deta"],
                "assa": metrics_payload["assa"],
                "frames_per_second": frames_per_second,
                "summary_class": metrics_payload["summary_class"],
                "summary_metrics": metrics_payload["summary_metrics"],
                "total_frames": total_frames,
                "seq_info": seq_info,
                "tracker_submission_zip": str(tracker_submission_zip),
                "ground_truth_zip": str(ground_truth_zip),
                "seqmap_file": str(seqmap_file),
                "eval_output_dir": str(eval_output_dir),
                "captured_stdout": captured_stdout.getvalue(),
            }
        )
    )


if __name__ == "__main__":
    main()
