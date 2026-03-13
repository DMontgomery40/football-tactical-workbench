from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from pathlib import Path
from time import perf_counter
from typing import Any


CLASS_NAMES = ["background", "drive", "pass", "cross", "throw-in", "shot", "header", "tackle", "block"]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FOOTPASS evaluation and emit JSON.")
    parser.add_argument("--predictions-file", required=True, type=Path)
    parser.add_argument("--ground-truth-file", required=True, type=Path)
    parser.add_argument("--delta", default=12, type=int)
    parser.add_argument("--confidence-threshold", default=0.15, type=float)
    parser.add_argument("--print-per-game", action="store_true")
    return parser.parse_args()


def _json_ready(value: Any) -> Any:
    try:
        import numpy as np
    except Exception:  # pragma: no cover - runtime-specific dependency
        np = None

    if np is not None:
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _evaluated_game_count(ground_truth_file: Path, predictions_file: Path) -> int:
    try:
        gt_payload = json.loads(ground_truth_file.read_text(encoding="utf-8"))
        pred_payload = json.loads(predictions_file.read_text(encoding="utf-8"))
    except Exception:  # pragma: no cover - defensive fallback for runtime-only smoke paths
        return 0
    gt_keys = {str(key) for key in (gt_payload.get("keys") or [])}
    pred_keys = {str(key) for key in (pred_payload.get("keys") or [])}
    return len(gt_keys.intersection(pred_keys))


def main() -> int:
    args = _parse_args()
    repo_dir = Path(__file__).resolve().parents[2] / "third_party" / "soccernet" / "FOOTPASS"
    sys.path.insert(0, str(repo_dir))
    from utils.metric_utils import evaluate_events_from_json

    start = perf_counter()
    captured_stdout = io.StringIO()
    with contextlib.redirect_stdout(captured_stdout):
        results = evaluate_events_from_json(
            gt_json_path=str(args.ground_truth_file.expanduser().resolve()),
            pred_json_path=str(args.predictions_file.expanduser().resolve()),
            class_names=CLASS_NAMES,
            delta=int(args.delta),
            conf_thresh=float(args.confidence_threshold),
            print_per_game=bool(args.print_per_game),
        )
    elapsed_seconds = perf_counter() - start
    overall = results.get("overall") or {}
    games_evaluated = _evaluated_game_count(
        args.ground_truth_file.expanduser().resolve(),
        args.predictions_file.expanduser().resolve(),
    )
    payload = {
        "f1_at_15": overall.get("f1"),
        "precision_at_15": overall.get("precision"),
        "recall_at_15": overall.get("recall"),
        "clips_per_second": (games_evaluated / elapsed_seconds) if elapsed_seconds > 0 and games_evaluated > 0 else None,
        "games_evaluated": games_evaluated,
        "elapsed_seconds": elapsed_seconds,
        "vendor_stdout": captured_stdout.getvalue(),
        "raw_results": _json_ready(results),
    }
    print(json.dumps(_json_ready(payload)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
