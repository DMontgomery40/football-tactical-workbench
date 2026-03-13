from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from time import perf_counter


FINAL_SCORE_RE = re.compile(r"final score of\s*:\s*([0-9eE+\-.]+)")
COMPLETENESS_RE = re.compile(r"completeness rate of\s*:\s*([0-9eE+\-.]+)")
ACCURACY_RE = re.compile(r"accuracy mean value\s*:\s*([0-9eE+\-.]+)%")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run sn-calibration evaluation and emit JSON.")
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--prediction-root", required=True, type=Path)
    parser.add_argument("--split", default="valid")
    parser.add_argument("--threshold", default=5, type=int)
    parser.add_argument("--resolution-width", default=960, type=int)
    parser.add_argument("--resolution-height", default=540, type=int)
    parser.add_argument("--artifacts-dir", required=True, type=Path)
    return parser.parse_args()


def _match_float(pattern: re.Pattern[str], text: str) -> float | None:
    match = pattern.search(text)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def main() -> int:
    args = _parse_args()
    repo_dir = Path(__file__).resolve().parents[2] / "third_party" / "soccernet" / "sn-calibration"
    artifacts_dir = args.artifacts_dir.expanduser().resolve()
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)
    env["MPLBACKEND"] = "Agg"
    existing_pythonpath = env.get("PYTHONPATH", "").strip()
    pythonpath_parts = [str(repo_dir)]
    if existing_pythonpath:
        pythonpath_parts.append(existing_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)

    split_dir = args.dataset_root.expanduser().resolve() / str(args.split)
    evaluated_frames = len(
        [
            path for path in split_dir.glob("*.json")
            if path.is_file() and path.name != "per_match_info.json"
        ]
    )

    command = [
        sys.executable,
        "-m",
        "src.evaluate_camera",
        "-s",
        str(args.dataset_root.expanduser().resolve()),
        "-p",
        str(args.prediction_root.expanduser().resolve()),
        "--split",
        str(args.split),
        "-t",
        str(args.threshold),
        "--resolution_width",
        str(args.resolution_width),
        "--resolution_height",
        str(args.resolution_height),
    ]
    start = perf_counter()
    completed = subprocess.run(
        command,
        cwd=str(artifacts_dir),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    elapsed_seconds = perf_counter() - start
    if completed.returncode != 0:
        sys.stderr.write(completed.stderr or completed.stdout or "Calibration evaluator failed.\n")
        return completed.returncode

    stdout = completed.stdout
    jac_5_percent = _match_float(ACCURACY_RE, stdout)
    payload = {
        "completeness_x_jac_5": _match_float(FINAL_SCORE_RE, stdout),
        "completeness": _match_float(COMPLETENESS_RE, stdout),
        "jac_5": None if jac_5_percent is None else jac_5_percent / 100.0,
        "frames_per_second": (evaluated_frames / elapsed_seconds) if elapsed_seconds > 0 and evaluated_frames > 0 else None,
        "evaluated_frames": evaluated_frames,
        "elapsed_seconds": elapsed_seconds,
        "results_dir": str(artifacts_dir / "results"),
        "stdout": stdout,
    }
    print(json.dumps(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
