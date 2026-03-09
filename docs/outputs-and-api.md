# Outputs, API, And Batch Experiments

## Run Directory Layout

Each analysis run writes to:

```text
backend/runs/<run_id>/
  inputs/
  outputs/
  job_state.json
```

The backend only treats a run as reviewable if `outputs/summary.json` exists.

## Output Files

The active pipeline writes the following artifacts.

| File | When it exists | Purpose |
| --- | --- | --- |
| `overlay.mp4` | always on successful run | Annotated review video |
| `detections.csv` | always on successful run | Per-frame player and ball records |
| `track_summary.csv` | always on successful run | Track-level rollup |
| `projections.csv` | when projected rows exist | Field and minimap coordinates |
| `entropy_timeseries.csv` | always on successful run | 1 Hz experiment series |
| `goal_events.csv` | only when goal labels exist | Parsed goal events beside the source clip |
| `diagnostics_ai.json` | always after diagnostics generation step | Per-run diagnostics artifact, with AI-curated output when a provider is configured |
| `summary.json` | always on successful run | Main run contract used by the UI |
| `all_outputs.zip` | always on successful run | Bundle of output artifacts |

## `summary.json` Highlights

The saved run summary is the main contract between backend and frontend.

### Identity and paths

- `job_id`
- `run_dir`
- `input_video`
- `overlay_video`
- `detections_csv`
- `track_summary_csv`
- `projection_csv`
- `entropy_timeseries_csv`
- `goal_events_csv`
- `summary_json`
- `all_outputs_zip`
- `diagnostics_json`

### Model and runtime configuration

- `device`
- `field_calibration_device`
- `player_model`
- `player_tracker_mode`
- `player_tracker_backend`
- `ball_model`
- `field_calibration_model`
- `include_ball`
- `player_conf`
- `ball_conf`
- `iou`

The app-level `/api/config` surface now also reports:

- `runtime_profile.backend`
- `runtime_profile.backend_label`
- `runtime_profile.host_platform`
- `runtime_profile.host_arch`
- `runtime_profile.preferred_device`
- `runtime_profile.available_devices`
- `runtime_profile.field_calibration_device_policy`
- `runtime_profile.detector_export_formats`
- `runtime_profile.planned_backends`
- `runtime_profile.runtime_notes`
- `runtime_profile.license_notes`

### Core run metrics

- `frames_processed`
- `fps`
- `player_rows`
- `ball_rows`
- `unique_player_track_ids`
- `raw_unique_player_track_ids`
- `unique_ball_track_ids`
- `home_tracks`
- `away_tracks`
- `unassigned_tracks`
- `average_player_detections_per_frame`
- `average_ball_detections_per_frame`
- `longest_track_length`
- `average_track_length`
- `raw_average_track_length`
- `tracklet_merges_applied`
- `stitched_track_id_reduction`
- `identity_embedding_updates`

### Field calibration metrics

- `projected_player_points`
- `projected_ball_points`
- `field_registered_frames`
- `field_registered_ratio`
- `homography_enabled`
- `field_calibration_refresh_frames`
- `field_calibration_refresh_attempts`
- `field_calibration_refresh_successes`
- `average_visible_pitch_keypoints`
- `last_good_calibration_frame`

### Team clustering and experiment fields

- `team_cluster_distance`
- `jersey_crops_used`
- `goal_events_count`
- `goal_label_source`
- `experiments`
- `top_tracks`

### Diagnostics fields

- `diagnostics`
- `heuristic_diagnostics`
- `diagnostics_source`
- `diagnostics_provider`
- `diagnostics_model`
- `diagnostics_status`
- `diagnostics_summary_line`
- `diagnostics_error`

## `detections.csv`

The current code writes these detection columns:

- `frame_index`
- `row_type`
- `track_id`
- `class_name`
- `team_label`
- `team_vote_ratio`
- `confidence`
- `x1`
- `y1`
- `x2`
- `y2`
- `anchor_x`
- `anchor_y`
- `field_x_cm`
- `field_y_cm`
- `map_x`
- `map_y`
- `calibration_visible_keypoints`
- `calibration_inliers`
- `color_r`
- `color_g`
- `color_b`

## `track_summary.csv`

The current track summary columns are:

- `track_id`
- `team_label`
- `team_vote_ratio`
- `frames`
- `first_frame`
- `last_frame`
- `average_confidence`
- `average_bbox_area`
- `projected_points`
- `sampled_color_rgb`

The frontend trajectory review surface combines `track_summary.csv` with `projections.csv` so it can rank projected tracks and draw the recent window of player and ball movement on the pitch map.

## `entropy_timeseries.csv`

The current experiment export is a 1 Hz rollup with:

- team player counts
- centroids
- spread RMS
- team length and width axes
- hull areas
- inter-team centroid distance
- spatial entropy
- rolling volatility measures
- combined `vol_index`
- goal lookahead columns

The current header includes:

- `second`
- `seconds`
- `home_player_count`
- `away_player_count`
- `home_centroid_x_cm`
- `home_centroid_y_cm`
- `away_centroid_x_cm`
- `away_centroid_y_cm`
- `home_spread_rms_cm`
- `away_spread_rms_cm`
- `home_length_axis_cm`
- `away_length_axis_cm`
- `home_width_axis_cm`
- `away_width_axis_cm`
- `home_hull_area_cm2`
- `away_hull_area_cm2`
- `centroid_distance_cm`
- `entropy_grid`
- `home_spread_rms_cm_volatility`
- `away_spread_rms_cm_volatility`
- `home_length_axis_cm_volatility`
- `away_length_axis_cm_volatility`
- `centroid_distance_cm_volatility`
- `entropy_grid_volatility`
- `vol_index`
- `seconds_to_next_goal`
- `next_goal_team`
- `goal_in_next_30s`
- `goal_in_next_60s`

## `diagnostics_ai.json`

The diagnostics artifact currently contains:

- `prompt_version`
- `generated_at`
- `status`
- `provider`
- `model`
- `summary_line`
- `error`
- `raw_text`
- `diagnostics`

This file is the stored per-run diagnostics artifact used by review flows and refresh actions.

With a configured provider, it contains an AI-curated summary line and diagnostics for the completed run.

If no provider resolves:

- the file still exists
- `status` is `disabled`
- the diagnostics list falls back to heuristic diagnostics

## Backend API Surface

The current FastAPI application exposes these routes:

### Configuration and health

- `GET /api/health`
- `GET /api/config`

### Analysis jobs

- `GET /api/jobs`
- `GET /api/jobs/{job_id}`
- `POST /api/analyze`

### Sources and live preview

- `POST /api/source`
- `GET /api/source/{source_id}/video`
- `GET /api/live-preview`

### Persisted runs

- `GET /api/runs/recent`
- `GET /api/runs/{run_id}`
- `POST /api/runs/{run_id}/refresh-diagnostics`

### Local folder scan

- `POST /api/scan-folder`

### SoccerNet

- `GET /api/soccernet/config`
- `GET /api/soccernet/games`
- `GET /api/soccernet/downloads`
- `GET /api/soccernet/downloads/{job_id}`
- `POST /api/soccernet/download`

## Important Request Parameters

### `POST /api/source`

Multipart form fields:

- `video_file`
- `local_video_path`

### `GET /api/live-preview`

Query parameters:

- `source_id`
- `local_video_path`
- `detector_model`
- `player_model`
- `tracker_mode`
- `include_ball`
- `player_conf`
- `ball_conf`
- `iou`

### `POST /api/analyze`

Multipart form fields:

- `video_file`
- `local_video_path`
- `source_id`
- `label_path`
- `detector_model`
- `player_model`
- `tracker_mode`
- `include_ball`
- `player_conf`
- `ball_conf`
- `iou`

### `POST /api/scan-folder`

JSON body:

- `folder_path`

### `POST /api/soccernet/download`

JSON body:

- `split`
- `game`
- `password`
- `files`

## Batch Experiment Tooling

The repository includes an offline batch runner:

```text
backend/scripts/soccernet_batch_experiment.py
```

It can:

- select SoccerNet games by split, limit, and offset
- accept local video files for quick tracker A/B checks
- ensure requested halves and labels are downloaded
- run the active analysis pipeline repeatedly
- aggregate run-level and window-level experiment outputs
- compare multiple player tracker modes on the same source clips

### Direct batch command

```bash
cd backend
source .venv/bin/activate
python scripts/soccernet_batch_experiment.py \
  --split train \
  --limit 20 \
  --offset 0 \
  --password "$SOCCERNET_PASSWORD" \
  --tracker-modes hybrid_reid
```

### Local tracker A/B command

```bash
cd backend
source .venv/bin/activate
python scripts/soccernet_batch_experiment.py \
  --local-video datasets/bundesliga_sample/reid_smoke_clip.mp4 \
  --compare-tracker-modes \
  --batch-name tracker_ab_smoke
```

### `tmux` launcher

```bash
cd backend
./scripts/start_soccernet_batch_tmux.sh train 20 0 "1_224p.mkv 2_224p.mkv Labels-v2.json" "bytetrack hybrid_reid"
```

That launcher:

- loads `SOCCERNET_PASSWORD` from the repository root `.env`
- creates a timestamped experiment directory in `backend/experiments/`
- runs the batch script inside a detached `tmux` session

## Batch Outputs

A batch run writes to a timestamped directory under `backend/experiments/` and produces:

- `manifest.json`
- `batch.log`
- `runs.csv`
- `tracker_comparison.csv`
- `entropy_windows_1hz.csv`
- `summary.json`
- `runs/` with one analysis run directory per processed half
