#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import ssl
import sys
import time
import uuid
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from SoccerNet.Downloader import SoccerNetDownloader
from SoccerNet.utils import getListGames


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.reid_tracker import (  # noqa: E402
    DEFAULT_PLAYER_TRACKER_MODE,
    LEGACY_PLAYER_TRACKER_MODE,
    normalize_player_tracker_mode,
)
from app.wide_angle import analyze_video  # noqa: E402


SOCCERNET_DATASET_DIR = ROOT_DIR / "datasets" / "soccernet"
EXPERIMENTS_DIR = ROOT_DIR / "experiments"
DEFAULT_FILES = ["1_224p.mkv", "2_224p.mkv", "Labels-v2.json"]
DEFAULT_COMPARE_TRACKER_MODES = [LEGACY_PLAYER_TRACKER_MODE, DEFAULT_PLAYER_TRACKER_MODE]


class BatchJobManager:
    def __init__(self, log_file: Path) -> None:
        self.log_file = log_file
        self._last_progress_bucket: dict[str, int] = {}

    def log(self, job_id: str, message: str) -> None:
        stamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")
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


def dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def resolve_tracker_modes(raw_modes: list[str] | None, compare_tracker_modes: bool) -> list[str]:
    if compare_tracker_modes and not raw_modes:
        return DEFAULT_COMPARE_TRACKER_MODES
    normalized = [normalize_player_tracker_mode(value) for value in (raw_modes or [DEFAULT_PLAYER_TRACKER_MODE])]
    return dedupe_preserve_order(normalized)


def find_local_label_path(video_path: Path) -> str:
    for file_name in ("Labels-v2.json", "Labels.json"):
        candidate = video_path.parent / file_name
        if candidate.exists():
            return str(candidate)
    return ""


def collect_local_video_paths(local_videos: list[str], local_video_glob: str) -> list[Path]:
    candidates: list[Path] = []
    for raw_path in local_videos:
        candidate = Path(raw_path).expanduser().resolve()
        if not candidate.exists() or not candidate.is_file():
            raise SystemExit(f"Local video does not exist: {candidate}")
        candidates.append(candidate)

    if local_video_glob.strip():
        matches = sorted(glob.glob(local_video_glob.strip(), recursive=True))
        for match in matches:
            candidate = Path(match).expanduser().resolve()
            if candidate.is_file():
                candidates.append(candidate)

    unique_paths = dedupe_preserve_order([str(path) for path in candidates])
    return [Path(path) for path in unique_paths]


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


def aggregate_experiment_rows(experiment_csv_path: Path, game: str, split: str, half_file: str, run_dir: Path, tracker_mode: str) -> list[dict[str, Any]]:
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
                    "tracker_mode": tracker_mode,
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


def summarize_windows(rows: list[dict[str, Any]], tracker_modes: list[str]) -> dict[str, Any]:
    positive = [row["vol_index"] for row in rows if row["vol_index"] is not None and row["goal_in_next_30s"] == 1]
    negative = [row["vol_index"] for row in rows if row["vol_index"] is not None and row["goal_in_next_30s"] == 0]

    positive_mean = float(sum(positive) / len(positive)) if positive else 0.0
    negative_mean = float(sum(negative) / len(negative)) if negative else 0.0
    uplift = ((positive_mean - negative_mean) / negative_mean) if negative_mean > 0 else 0.0

    summary: dict[str, Any] = {
        "positive_window_count": len(positive),
        "baseline_window_count": len(negative),
        "mean_pre_goal_30s_vol_index": round(positive_mean, 6),
        "mean_baseline_vol_index": round(negative_mean, 6),
        "pre_goal_uplift_30s": round(float(uplift), 6),
    }
    by_tracker: dict[str, Any] = {}
    for tracker_mode in tracker_modes:
        tracker_rows = [row for row in rows if row.get("tracker_mode") == tracker_mode]
        if not tracker_rows:
            continue
        tracker_positive = [row["vol_index"] for row in tracker_rows if row["vol_index"] is not None and row["goal_in_next_30s"] == 1]
        tracker_negative = [row["vol_index"] for row in tracker_rows if row["vol_index"] is not None and row["goal_in_next_30s"] == 0]
        tracker_positive_mean = float(sum(tracker_positive) / len(tracker_positive)) if tracker_positive else 0.0
        tracker_negative_mean = float(sum(tracker_negative) / len(tracker_negative)) if tracker_negative else 0.0
        tracker_uplift = ((tracker_positive_mean - tracker_negative_mean) / tracker_negative_mean) if tracker_negative_mean > 0 else 0.0
        by_tracker[tracker_mode] = {
            "positive_window_count": len(tracker_positive),
            "baseline_window_count": len(tracker_negative),
            "mean_pre_goal_30s_vol_index": round(tracker_positive_mean, 6),
            "mean_baseline_vol_index": round(tracker_negative_mean, 6),
            "pre_goal_uplift_30s": round(float(tracker_uplift), 6),
        }
    if by_tracker:
        summary["by_tracker_mode"] = by_tracker
    return summary


def build_run_row(summary: dict[str, Any], split: str, game: str, half_file: str, run_dir: Path, tracker_mode: str) -> dict[str, Any]:
    return {
        "split": split,
        "game": game,
        "half_file": half_file,
        "tracker_mode": tracker_mode,
        "run_dir": str(run_dir),
        "frames_processed": summary.get("frames_processed", 0),
        "goal_events_count": summary.get("goal_events_count", 0),
        "field_calibration_refresh_successes": summary.get("field_calibration_refresh_successes", 0),
        "field_registered_ratio": summary.get("field_registered_ratio", 0.0),
        "home_tracks": summary.get("home_tracks", 0),
        "away_tracks": summary.get("away_tracks", 0),
        "team_cluster_distance": summary.get("team_cluster_distance", 0.0),
        "player_tracker_backend": summary.get("player_tracker_backend"),
        "raw_unique_player_track_ids": summary.get("raw_unique_player_track_ids", summary.get("unique_player_track_ids", 0)),
        "unique_player_track_ids": summary.get("unique_player_track_ids", 0),
        "tracklet_merges_applied": summary.get("tracklet_merges_applied", 0),
        "stitched_track_id_reduction": summary.get("stitched_track_id_reduction", 0.0),
        "average_track_length": summary.get("average_track_length", 0.0),
        "raw_average_track_length": summary.get("raw_average_track_length", summary.get("average_track_length", 0.0)),
        "longest_track_length": summary.get("longest_track_length", 0),
        "player_track_churn_ratio": summary.get("player_track_churn_ratio"),
        "raw_player_track_churn_ratio": summary.get("raw_player_track_churn_ratio"),
        "average_player_detections_per_frame": summary.get("average_player_detections_per_frame", 0.0),
        "ball_rows": summary.get("ball_rows", 0),
        "unique_ball_track_ids": summary.get("unique_ball_track_ids", 0),
        "average_ball_detections_per_frame": summary.get("average_ball_detections_per_frame", 0.0),
        "overlay_video": summary.get("overlay_video"),
        "entropy_timeseries_csv": summary.get("entropy_timeseries_csv"),
        "goal_events_csv": summary.get("goal_events_csv"),
    }


def build_tracker_comparison_rows(run_rows: list[dict[str, Any]], baseline_mode: str) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in run_rows:
        grouped[(str(row["split"]), str(row["game"]), str(row["half_file"]))][str(row["tracker_mode"])] = row

    comparison_rows: list[dict[str, Any]] = []
    for (split, game, half_file), rows_by_mode in sorted(grouped.items()):
        baseline_row = rows_by_mode.get(baseline_mode)
        if baseline_row is None:
            continue
        for tracker_mode, candidate_row in sorted(rows_by_mode.items()):
            if tracker_mode == baseline_mode:
                continue
            baseline_ids = int(baseline_row.get("unique_player_track_ids") or 0)
            candidate_ids = int(candidate_row.get("unique_player_track_ids") or 0)
            baseline_avg_length = float(baseline_row.get("average_track_length") or 0.0)
            candidate_avg_length = float(candidate_row.get("average_track_length") or 0.0)
            comparison_rows.append(
                {
                    "split": split,
                    "game": game,
                    "half_file": half_file,
                    "baseline_tracker_mode": baseline_mode,
                    "candidate_tracker_mode": tracker_mode,
                    "baseline_unique_player_track_ids": baseline_ids,
                    "candidate_unique_player_track_ids": candidate_ids,
                    "unique_player_track_ids_delta": candidate_ids - baseline_ids,
                    "baseline_average_track_length": round(baseline_avg_length, 4),
                    "candidate_average_track_length": round(candidate_avg_length, 4),
                    "average_track_length_delta": round(candidate_avg_length - baseline_avg_length, 4),
                    "baseline_tracklet_merges_applied": int(baseline_row.get("tracklet_merges_applied") or 0),
                    "candidate_tracklet_merges_applied": int(candidate_row.get("tracklet_merges_applied") or 0),
                    "tracklet_merges_delta": int(candidate_row.get("tracklet_merges_applied") or 0) - int(baseline_row.get("tracklet_merges_applied") or 0),
                    "baseline_stitched_track_id_reduction": round(float(baseline_row.get("stitched_track_id_reduction") or 0.0), 6),
                    "candidate_stitched_track_id_reduction": round(float(candidate_row.get("stitched_track_id_reduction") or 0.0), 6),
                    "stitched_track_id_reduction_delta": round(
                        float(candidate_row.get("stitched_track_id_reduction") or 0.0) - float(baseline_row.get("stitched_track_id_reduction") or 0.0),
                        6,
                    ),
                    "candidate_wins_fewer_ids": int(candidate_ids < baseline_ids),
                    "candidate_wins_longer_tracks": int(candidate_avg_length > baseline_avg_length),
                }
            )
    return comparison_rows


def summarize_tracker_runs(run_rows: list[dict[str, Any]], tracker_modes: list[str], comparison_rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_tracker_mode: dict[str, Any] = {}
    for tracker_mode in tracker_modes:
        tracker_rows = [row for row in run_rows if row.get("tracker_mode") == tracker_mode]
        if not tracker_rows:
            continue
        by_tracker_mode[tracker_mode] = {
            "runs": len(tracker_rows),
            "mean_unique_player_track_ids": round(sum(float(row.get("unique_player_track_ids") or 0.0) for row in tracker_rows) / len(tracker_rows), 4),
            "mean_raw_unique_player_track_ids": round(sum(float(row.get("raw_unique_player_track_ids") or 0.0) for row in tracker_rows) / len(tracker_rows), 4),
            "mean_average_track_length": round(sum(float(row.get("average_track_length") or 0.0) for row in tracker_rows) / len(tracker_rows), 4),
            "mean_tracklet_merges_applied": round(sum(float(row.get("tracklet_merges_applied") or 0.0) for row in tracker_rows) / len(tracker_rows), 4),
            "mean_stitched_track_id_reduction": round(sum(float(row.get("stitched_track_id_reduction") or 0.0) for row in tracker_rows) / len(tracker_rows), 6),
        }

    tracker_comparison_summary: dict[str, Any] = {}
    if comparison_rows:
        tracker_comparison_summary = {
            "comparisons": len(comparison_rows),
            "candidate_wins_fewer_ids": sum(int(row["candidate_wins_fewer_ids"]) for row in comparison_rows),
            "candidate_wins_longer_tracks": sum(int(row["candidate_wins_longer_tracks"]) for row in comparison_rows),
            "mean_unique_player_track_ids_delta": round(
                sum(float(row["unique_player_track_ids_delta"]) for row in comparison_rows) / len(comparison_rows),
                4,
            ),
            "mean_average_track_length_delta": round(
                sum(float(row["average_track_length_delta"]) for row in comparison_rows) / len(comparison_rows),
                4,
            ),
            "mean_stitched_track_id_reduction_delta": round(
                sum(float(row["stitched_track_id_reduction_delta"]) for row in comparison_rows) / len(comparison_rows),
                6,
            ),
        }

    return {
        "by_tracker_mode": by_tracker_mode,
        "tracker_comparison_summary": tracker_comparison_summary,
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
    parser.add_argument("--tracker-modes", nargs="+", default=None, help="Player tracker modes to run. First mode is treated as baseline for comparisons.")
    parser.add_argument("--compare-tracker-modes", action="store_true", help="Shortcut for --tracker-modes bytetrack hybrid_reid")
    parser.add_argument("--local-video", action="append", default=[], help="Local video file to analyze. Can be passed multiple times.")
    parser.add_argument("--local-video-glob", default="", help="Glob for local videos to analyze instead of SoccerNet downloads.")
    args = parser.parse_args()

    tracker_modes = resolve_tracker_modes(args.tracker_modes, args.compare_tracker_modes)
    local_video_paths = collect_local_video_paths(args.local_video, args.local_video_glob)
    use_local_inputs = len(local_video_paths) > 0

    if not use_local_inputs and not args.password:
        raise SystemExit("SoccerNet password is required. Pass --password or set SOCCERNET_PASSWORD, or use --local-video.")

    selected_games: list[str] = []
    if not use_local_inputs:
        selected_games = select_games(args.split, args.limit, args.offset)
        if not selected_games:
            raise SystemExit("No SoccerNet games matched the requested split/limit/offset.")

    batch_name = args.batch_name or (
        f"tracker_ab_local_{len(local_video_paths)}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        if use_local_inputs
        else f"soccernet_{args.split}_{args.offset}_{args.limit}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    )
    batch_dir = EXPERIMENTS_DIR / batch_name
    runs_root = batch_dir / "runs"
    manifest_path = batch_dir / "manifest.json"
    runs_csv_path = batch_dir / "runs.csv"
    windows_csv_path = batch_dir / "entropy_windows_1hz.csv"
    comparison_csv_path = batch_dir / "tracker_comparison.csv"
    summary_path = batch_dir / "summary.json"
    log_file = batch_dir / "batch.log"
    batch_dir.mkdir(parents=True, exist_ok=True)

    logger = BatchJobManager(log_file)
    logger.log("batch", f"starting batch {batch_name}")
    logger.log("batch", f"tracker_modes={tracker_modes}")
    if use_local_inputs:
        logger.log("batch", f"local_videos={len(local_video_paths)}")
    else:
        logger.log("batch", f"split={args.split} offset={args.offset} limit={args.limit} files={args.files}")
    logger.log("batch", f"dataset_dir={SOCCERNET_DATASET_DIR}")
    logger.log("batch", f"runs_root={runs_root}")

    manifest = {
        "batch_name": batch_name,
        "created_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "tracker_modes": tracker_modes,
        "input_mode": "local" if use_local_inputs else "soccernet",
        "split": args.split if not use_local_inputs else "local",
        "offset": args.offset,
        "limit": args.limit,
        "files": args.files,
        "games": selected_games,
        "local_videos": [str(path) for path in local_video_paths],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    run_rows: list[dict[str, Any]] = []
    window_rows: list[dict[str, Any]] = []
    work_items: list[dict[str, Any]] = []

    if use_local_inputs:
        for video_index, video_path in enumerate(local_video_paths, start=1):
            safe_slug = f"local_{video_index:03d}_{uuid.uuid5(uuid.NAMESPACE_URL, str(video_path)).hex[:10]}"
            work_items.append(
                {
                    "split": "local",
                    "game": str(video_path.parent),
                    "half_file": video_path.name,
                    "video_path": video_path,
                    "safe_slug": safe_slug,
                    "label_path": find_local_label_path(video_path),
                    "job_id": f"local-{video_index:03d}",
                }
            )
    else:
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
                work_items.append(
                    {
                        "split": args.split,
                        "game": game,
                        "half_file": half_file,
                        "video_path": half_path,
                        "safe_slug": safe_game_slug,
                        "label_path": "",
                        "job_id": game_job_id,
                    }
                )

    for work_item in work_items:
        split = str(work_item["split"])
        game = str(work_item["game"])
        half_file = str(work_item["half_file"])
        video_path = Path(work_item["video_path"])
        safe_slug = str(work_item["safe_slug"])
        label_path = str(work_item.get("label_path") or "")
        work_job_id = str(work_item["job_id"])
        logger.log(work_job_id, f"running {half_file} from {video_path}")

        for tracker_mode in tracker_modes:
            run_slug = f"{safe_slug}_{tracker_mode}"
            run_dir = runs_root / run_slug
            summary_file = run_dir / "outputs" / "summary.json"
            logger.log(work_job_id, f"tracker_mode={tracker_mode}")

            if summary_file.exists() and not args.force:
                logger.log(work_job_id, f"reusing existing run for {half_file} [{tracker_mode}]")
                summary = json.loads(summary_file.read_text(encoding="utf-8"))
            else:
                run_dir.mkdir(parents=True, exist_ok=True)
                summary = analyze_video(
                    job_id=f"{work_job_id}-{tracker_mode}",
                    run_dir=run_dir,
                    config_payload={
                        "source_video_path": str(video_path),
                        "label_path": label_path,
                        "player_model": "soccana",
                        "ball_model": "soccana",
                        "tracker_mode": tracker_mode,
                        "include_ball": True,
                        "player_conf": 0.25,
                        "ball_conf": 0.20,
                        "iou": 0.50,
                    },
                    job_manager=logger,
                )

            summary["split"] = split
            summary["game"] = game
            summary["half_file"] = half_file
            summary["tracker_mode"] = tracker_mode
            summary_path_for_run = run_dir / "outputs" / "summary.json"
            summary_path_for_run.write_text(json.dumps(summary, indent=2), encoding="utf-8")

            run_rows.append(build_run_row(summary, split=split, game=game, half_file=half_file, run_dir=run_dir, tracker_mode=tracker_mode))

            entropy_csv_path = run_dir / "outputs" / "entropy_timeseries.csv"
            if entropy_csv_path.exists():
                window_rows.extend(
                    aggregate_experiment_rows(
                        entropy_csv_path,
                        game=game,
                        split=split,
                        half_file=half_file,
                        run_dir=run_dir,
                        tracker_mode=tracker_mode,
                    )
                )

            comparison_rows = build_tracker_comparison_rows(run_rows, baseline_mode=tracker_modes[0])
            write_runs_csv(runs_csv_path, run_rows)
            write_windows_csv(windows_csv_path, window_rows)
            write_runs_csv(comparison_csv_path, comparison_rows)
            summary_path.write_text(
                json.dumps(
                    {
                        "batch_name": batch_name,
                        "created_at": manifest["created_at"],
                        "input_mode": manifest["input_mode"],
                        "tracker_modes": tracker_modes,
                        "baseline_tracker_mode": tracker_modes[0],
                        "sources_requested": len(work_items),
                        "runs_completed": len(run_rows),
                        **summarize_windows(window_rows, tracker_modes=tracker_modes),
                        **summarize_tracker_runs(run_rows, tracker_modes=tracker_modes, comparison_rows=comparison_rows),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

    logger.log("batch", f"completed batch {batch_name}")
    logger.log("batch", f"runs_csv={runs_csv_path}")
    logger.log("batch", f"windows_csv={windows_csv_path}")
    logger.log("batch", f"comparison_csv={comparison_csv_path}")
    logger.log("batch", f"summary_json={summary_path}")


if __name__ == "__main__":
    main()
