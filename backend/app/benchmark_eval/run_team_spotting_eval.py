from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from pathlib import Path
from time import perf_counter
from typing import Any


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run sn-teamspotting evaluation and emit JSON.")
    parser.add_argument("--labels-root", required=True, type=Path)
    parser.add_argument("--predictions-root", required=True, type=Path)
    parser.add_argument("--prediction-file", default="results_spotting.json")
    parser.add_argument("--split", default="test")
    parser.add_argument("--metric", default="at1")
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


def main() -> int:
    args = _parse_args()
    repo_dir = Path(__file__).resolve().parents[2] / "third_party" / "soccernet" / "sn-teamspotting"
    sys.path.insert(0, str(repo_dir))
    from util.eval import mAPevaluateCodabench
    from util.utils import getListGames

    start = perf_counter()
    captured_stdout = io.StringIO()
    with contextlib.redirect_stdout(captured_stdout):
        results = mAPevaluateCodabench(
            SoccerNet_path=str(args.labels_root.expanduser().resolve()),
            Predictions_path=str(args.predictions_root.expanduser().resolve()),
            prediction_file=str(args.prediction_file),
            split=str(args.split),
            printed=False,
            event_team=True,
            metric=str(args.metric),
        )
    elapsed_seconds = perf_counter() - start
    games_evaluated = len(getListGames(split=str(args.split)))
    payload = {
        "team_map_at_1": results.get("mAP"),
        "map_at_1": results.get("mAP_no_team"),
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
