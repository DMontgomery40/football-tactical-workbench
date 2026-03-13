from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from time import perf_counter


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run sn-gamestate evaluation and emit JSON.")
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--artifacts-dir", required=True, type=Path)
    parser.add_argument("--state-load-file", required=True, type=Path)
    parser.add_argument("--eval-set", default="valid")
    return parser.parse_args()


def _count_frames(dataset_root: Path, eval_set: str) -> int:
    split_root = dataset_root / eval_set
    return sum(1 for _ in split_root.rglob("*.jpg"))


def _parse_summary(path: Path) -> dict[str, float]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) < 2:
        raise RuntimeError(f"Game-state summary file is incomplete: {path}")
    headers = lines[0].split()
    values = lines[1].split()
    if len(headers) != len(values):
        raise RuntimeError(f"Game-state summary file columns do not align: {path}")
    parsed: dict[str, float] = {}
    for key, value in zip(headers, values):
        try:
            parsed[key] = float(value)
        except ValueError:
            continue
    return parsed


def main() -> int:
    args = _parse_args()
    repo_dir = Path(__file__).resolve().parents[2] / "third_party" / "soccernet" / "sn-gamestate"
    config_dir = repo_dir / "sn_gamestate" / "configs"
    artifacts_dir = args.artifacts_dir.expanduser().resolve()
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    dataset_root = args.dataset_root.expanduser().resolve()
    state_load_file = args.state_load_file.expanduser().resolve()
    if not dataset_root.exists():
        raise FileNotFoundError(f"Game-state dataset root is missing: {dataset_root}")
    if not state_load_file.exists():
        raise FileNotFoundError(f"Game-state validation tracker state is missing: {state_load_file}")

    run_dir = artifacts_dir / "tracklab_run"
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        "-m",
        "tracklab.main",
        "--config-dir",
        str(config_dir),
        "-cn",
        "soccernet",
        f"dataset.dataset_path={dataset_root}",
        f"dataset.eval_set={args.eval_set}",
        "dataset.nvid=-1",
        "pipeline=[]",
        "test_tracking=True",
        "eval_tracking=True",
        "use_rich=False",
        "state.save_file=null",
        f"state.load_file={state_load_file}",
        "~engine.callbacks.vis",
        f"hydra.run.dir={run_dir}",
    ]
    env = dict(os.environ)
    env["MPLBACKEND"] = "Agg"
    start = perf_counter()
    completed = subprocess.run(
        command,
        cwd=str(repo_dir),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    elapsed_seconds = perf_counter() - start
    if completed.returncode != 0:
        sys.stderr.write(completed.stderr or completed.stdout or "Game-state evaluator failed.\n")
        return completed.returncode

    summary_path = run_dir / "eval" / "results" / "tracklab" / "cls_comb_det_av_summary.txt"
    if not summary_path.exists():
        raise FileNotFoundError(f"Game-state summary file is missing: {summary_path}")
    summary = _parse_summary(summary_path)
    frames = _count_frames(dataset_root, str(args.eval_set))
    predictions_zip = run_dir / "eval" / "pred" / "SoccerNetGS-valid.zip"
    main_log = run_dir / "main.log"
    payload = {
        "gs_hota": summary.get("HOTA", 0.0) / 100.0,
        "hota": summary.get("HOTA", 0.0) / 100.0,
        "deta": summary.get("DetA", 0.0) / 100.0,
        "assa": summary.get("AssA", 0.0) / 100.0,
        "frames_per_second": (frames / elapsed_seconds) if elapsed_seconds > 0 and frames > 0 else None,
        "avg_frame_latency_ms": ((elapsed_seconds / frames) * 1000.0) if elapsed_seconds > 0 and frames > 0 else None,
        "frames_evaluated": frames,
        "elapsed_seconds": elapsed_seconds,
        "summary_path": str(summary_path),
        "predictions_zip": str(predictions_zip) if predictions_zip.exists() else None,
        "main_log": str(main_log) if main_log.exists() else None,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "summary_metrics": summary,
    }
    print(json.dumps(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
