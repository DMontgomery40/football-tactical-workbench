#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import ssl
import sys
import time
import uuid
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from SoccerNet.Downloader import SoccerNetDownloader
from SoccerNet.utils import getListGames


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.wide_angle import analyze_video  # noqa: E402


SOCCERNET_DATASET_DIR = ROOT_DIR / "datasets" / "soccernet"
EXPERIMENTS_DIR = ROOT_DIR / "experiments"
DEFAULT_FILES = ["1_224p.mkv", "2_224p.mkv", "Labels-v2.json"]


class BatchJobManager:
    def __init__(self, log_file: Path) -> None:
        self.log_file = log_file
        self._last_progress_bucket: dict[str, int] = {}

    def log(self, job_id: str, message: str) -> None:
        stamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{stamp}] [{job_id}] {message}"
        print(line, flush=True)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        with self.log_file.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    def update(self, job_id: str, **kwargs: Any) -> None:
        progress = kwargs.get("progress")
        if progress is not None:
            bucket = int(float(progress) // 5)
            if self._last_progress_bucket.get(job_id) != bucket:
                self._last_progress_bucket[job_id] = bucket
                self.log(job_id, f"progress={float(progress):.2f}")


def select_games(split: str, limit: int, offset: int) -> list[str]:
    games = getListGames(split, task="spotting", dataset="SoccerNet")
    return games[offset:offset + limit]


def ensure_game_downloaded(game: str, split: str, files: list[str], password: str, logger: BatchJobManager, job_id: str) -> Path:
    output_dir = SOCCERNET_DATASET_DIR / game
    output_dir.mkdir(parents=True, exist_ok=True)

    original_https_context_factory = ssl._create_default_https_context
    try:
        ssl._create_default_https_context = ssl._create_unverified_context
        downloader = SoccerNetDownloader(str(SOCCERNET_DATASET_DIR))
        downloader.password = password

        for file_name in files:
            local_path = output_dir / file_name
            if local_path.exists() and local_path.stat().st_size > 0:
                logger.log(job_id, f"already have {local_path.name}")
                continue
            logger.log(job_id, f"downloading {file_name}")
            downloader.downloadGame(game=game, files=[file_name], spl=split, verbose=False)
            if not local_path.exists() or local_path.stat().st_size == 0:
                raise RuntimeError(f"download failed for {game}/{file_name}")
            logger.log(job_id, f"saved {file_name} ({local_path.stat().st_size / 1024 / 1024:.2f} MB)")
    finally:
        ssl._create_default_https_context = original_https_context_factory

    return output_dir


def aggregate_experiment_rows(experiment_csv_path: Path, game: str, split: str, half_file: str, run_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with experiment_csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            def parse_numeric(key: str) -> float | None:
                value = row.get(key, "")
                return float(value) if value not in ("", None) else None

            rows.append(
                {
                    "split": split,
                    "game": game,
                    "half_file": half_file,
                    "run_dir": str(run_dir),
                    "second": int(float(row["seconds"])),
                    "home_spread_rms_cm": parse_numeric("home_spread_rms_cm"),
                    "away_spread_rms_cm": parse_numeric("away_spread_rms_cm"),
                    "home_length_axis_cm": parse_numeric("home_length_axis_cm"),
                    "away_length_axis_cm": parse_numeric("away_length_axis_cm"),
                    "centroid_distance_cm": parse_numeric("centroid_distance_cm"),
                    "entropy_grid": parse_numeric("entropy_grid"),
                    "home_spread_rms_cm_volatility": parse_numeric("home_spread_rms_cm_volatility"),
                    "away_spread_rms_cm_volatility": parse_numeric("away_spread_rms_cm_volatility"),
                    "home_length_axis_cm_volatility": parse_numeric("home_length_axis_cm_volatility"),
                    "away_length_axis_cm_volatility": parse_numeric("away_length_axis_cm_volatility"),
                    "centroid_distance_cm_volatility": parse_numeric("centroid_distance_cm_volatility"),
                    "entropy_grid_volatility": parse_numeric("entropy_grid_volatility"),
                    "vol_index": parse_numeric("vol_index"),
                    "goal_in_next_30s": int(row["goal_in_next_30s"] or 0),
                    "goal_in_next_60s": int(row["goal_in_next_60s"] or 0),
                    "seconds_to_next_goal": parse_numeric("seconds_to_next_goal"),
                    "next_goal_team": row.get("next_goal_team", ""),
                }
            )

    return rows


def write_runs_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_windows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_windows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    positive = [row["vol_index"] for row in rows if row["vol_index"] is not None and row["goal_in_next_30s"] == 1]
    negative = [row["vol_index"] for row in rows if row["vol_index"] is not None and row["goal_in_next_30s"] == 0]

    positive_mean = float(sum(positive) / len(positive)) if positive else 0.0
    negative_mean = float(sum(negative) / len(negative)) if negative else 0.0
    uplift = ((positive_mean - negative_mean) / negative_mean) if negative_mean > 0 else 0.0

    return {
        "positive_window_count": len(positive),
        "baseline_window_count": len(negative),
        "mean_pre_goal_30s_vol_index": round(positive_mean, 6),
        "mean_baseline_vol_index": round(negative_mean, 6),
        "pre_goal_uplift_30s": round(float(uplift), 6),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Download SoccerNet games and run the spatial-entropy volatility experiment.")
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--password", default=os.environ.get("SOCCERNET_PASSWORD", ""))
    parser.add_argument("--batch-name", default="")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--files", nargs="+", default=DEFAULT_FILES)
    args = parser.parse_args()

    if not args.password:
        raise SystemExit("SoccerNet password is required. Pass --password or set SOCCERNET_PASSWORD.")

    selected_games = select_games(args.split, args.limit, args.offset)
    if not selected_games:
        raise SystemExit("No SoccerNet games matched the requested split/limit/offset.")

    batch_name = args.batch_name or f"soccernet_{args.split}_{args.offset}_{args.limit}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    batch_dir = EXPERIMENTS_DIR / batch_name
    runs_root = batch_dir / "runs"
    manifest_path = batch_dir / "manifest.json"
    runs_csv_path = batch_dir / "runs.csv"
    windows_csv_path = batch_dir / "entropy_windows_1hz.csv"
    summary_path = batch_dir / "summary.json"
    log_file = batch_dir / "batch.log"
    batch_dir.mkdir(parents=True, exist_ok=True)

    logger = BatchJobManager(log_file)
    logger.log("batch", f"starting batch {batch_name}")
    logger.log("batch", f"split={args.split} offset={args.offset} limit={args.limit} files={args.files}")
    logger.log("batch", f"dataset_dir={SOCCERNET_DATASET_DIR}")
    logger.log("batch", f"runs_root={runs_root}")

    manifest = {
        "batch_name": batch_name,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "split": args.split,
        "offset": args.offset,
        "limit": args.limit,
        "files": args.files,
        "games": selected_games,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    run_rows: list[dict[str, Any]] = []
    window_rows: list[dict[str, Any]] = []

    for game_index, game in enumerate(selected_games, start=1):
        game_job_id = f"game-{game_index:03d}"
        logger.log(game_job_id, f"processing {game}")
        game_dir = ensure_game_downloaded(game=game, split=args.split, files=args.files, password=args.password, logger=logger, job_id=game_job_id)

        for half_file in [file_name for file_name in args.files if file_name.startswith(("1_", "2_"))]:
            half_path = game_dir / half_file
            if not half_path.exists():
                logger.log(game_job_id, f"skipping missing half {half_file}")
                continue

            half_tag = half_file.split("_", 1)[0]
            safe_game_slug = f"{game_index:03d}_{uuid.uuid5(uuid.NAMESPACE_URL, game).hex[:10]}_{half_tag}"
            run_dir = runs_root / safe_game_slug
            summary_file = run_dir / "outputs" / "summary.json"

            if summary_file.exists() and not args.force:
                logger.log(game_job_id, f"reusing existing run for {half_file}")
                summary = json.loads(summary_file.read_text(encoding="utf-8"))
            else:
                run_dir.mkdir(parents=True, exist_ok=True)
                summary = analyze_video(
                    job_id=f"{game_job_id}-{half_tag}",
                    run_dir=run_dir,
                    config_payload={
                        "source_video_path": str(half_path),
                        "player_model": "soccana",
                        "ball_model": "soccana",
                        "include_ball": True,
                        "player_conf": 0.25,
                        "ball_conf": 0.20,
                        "iou": 0.50,
                    },
                    job_manager=logger,
                )

            summary["split"] = args.split
            summary["game"] = game
            summary["half_file"] = half_file
            summary_path_for_run = run_dir / "outputs" / "summary.json"
            summary_path_for_run.write_text(json.dumps(summary, indent=2), encoding="utf-8")

            run_rows.append(
                {
                    "split": args.split,
                    "game": game,
                    "half_file": half_file,
                    "run_dir": str(run_dir),
                    "frames_processed": summary.get("frames_processed", 0),
                    "goal_events_count": summary.get("goal_events_count", 0),
                    "field_calibration_refresh_successes": summary.get("field_calibration_refresh_successes", 0),
                    "field_registered_ratio": summary.get("field_registered_ratio", 0.0),
                    "home_tracks": summary.get("home_tracks", 0),
                    "away_tracks": summary.get("away_tracks", 0),
                    "team_cluster_distance": summary.get("team_cluster_distance", 0.0),
                    "overlay_video": summary.get("overlay_video"),
                    "entropy_timeseries_csv": summary.get("entropy_timeseries_csv"),
                    "goal_events_csv": summary.get("goal_events_csv"),
                }
            )

            entropy_csv_path = run_dir / "outputs" / "entropy_timeseries.csv"
            if entropy_csv_path.exists():
                window_rows.extend(aggregate_experiment_rows(entropy_csv_path, game=game, split=args.split, half_file=half_file, run_dir=run_dir))

            write_runs_csv(runs_csv_path, run_rows)
            write_windows_csv(windows_csv_path, window_rows)
            summary_path.write_text(
                json.dumps(
                    {
                        "batch_name": batch_name,
                        "created_at": manifest["created_at"],
                        "games_requested": len(selected_games),
                        "halves_processed": len(run_rows),
                        **summarize_windows(window_rows),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

    logger.log("batch", f"completed batch {batch_name}")
    logger.log("batch", f"runs_csv={runs_csv_path}")
    logger.log("batch", f"windows_csv={windows_csv_path}")
    logger.log("batch", f"summary_json={summary_path}")


if __name__ == "__main__":
    main()
