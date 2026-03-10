# Outputs, API, And Batch Experiments

This document covers both persisted analysis outputs and persisted detector-training outputs, plus the current FastAPI surface.

## Analysis Run Directory Layout

Each analysis run writes to:

```text
backend/runs/<run_id>/
  inputs/
  outputs/
  job_state.json
```

The backend only treats an analysis run as reviewable if `outputs/summary.json` exists.

## Detector Training Run Directory Layout

Each detector training run writes to:

```text
backend/training_runs/<run_id>/
  config.json
  dataset_scan.json
  dataset_runtime.yaml
  progress.json
  summary.json
  train.log
  job_state.json
  splits/
  weights/
  yolo_output/
```

The Training Studio reads job state from the persisted run folder and surfaces the current summary/artifact paths through `/api/train/jobs/*` and `/api/train/runs/*`.

## Analysis Output Files

| File | When it exists | Purpose |
| --- | --- | --- |
| `overlay.mp4` | always on successful run | Annotated review video |
| `detections.csv` | always on successful run | Per-frame player and ball records |
| `track_summary.csv` | always on successful run | Track-level rollup |
| `projections.csv` | when projected rows exist | Field and minimap coordinates |
| `entropy_timeseries.csv` | always on successful run | 1 Hz experiment series |
| `goal_events.csv` | only when goal labels exist | Parsed goal events beside the source clip |
| `diagnostics_ai.json` | always after diagnostics generation | Per-run diagnostics artifact |
| `summary.json` | always on successful run | Main analysis contract used by the UI |
| `all_outputs.zip` | always on successful run | Bundle of output artifacts |

## Detector Training Output Files

| File | When it exists | Purpose |
| --- | --- | --- |
| `config.json` | always | Requested training config |
| `dataset_scan.json` | always for a started run | Persisted scan snapshot used by the worker |
| `dataset_runtime.yaml` | always for a started run | Run-local training manifest handed to Ultralytics |
| `progress.json` | during and after worker execution | Structured progress handoff from worker to manager |
| `summary.json` | always after run creation | Main training contract used by Training Studio |
| `train.log` | always after worker launch | Full training subprocess log |
| `weights/best.pt` | on successful run | Best detector checkpoint |
| `yolo_output/train/results.csv` | on successful run | Ultralytics metrics export |
| `yolo_output/train/args.yaml` | on successful run | Effective Ultralytics args |
| `yolo_output/train/*.png`, `*.jpg` | on successful run | Curves, confusion matrix, label plots, batches |

## Detector Registry

The local detector registry lives at:

```text
backend/models/registry.json
```

It stores:

- the active detector ID
- pretrained and custom detector entries
- checkpoint path
- metrics summary
- class ID mapping
- backend/runtime metadata
- summary/artifact paths for custom runs

When a custom detector is active in the registry, analysis uses it whenever the analysis selector remains on `soccana`.

## Analysis `summary.json` Highlights

The saved analysis summary is the main contract between backend and frontend.

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

The app-level `/api/config` surface also reports:

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

## Detector Training `summary.json` Highlights

The saved training summary is the main contract between backend and Training Studio.

### Identity and status

- `job_id`
- `run_id`
- `run_dir`
- `status`
- `progress`
- `created_at`
- `started_at`
- `finished_at`

### Training configuration and scan state

- `config`
- `dataset_scan`
- `generated_dataset_yaml`
- `generated_split_lists`
- `validation_strategy`

### Runtime and metrics

- `resolved_device`
- `backend`
- `backend_version`
- `metrics`
- `best_checkpoint`

### Artifact paths

- `summary_path`
- `artifacts.config`
- `artifacts.dataset_scan`
- `artifacts.generated_dataset_yaml`
- `artifacts.train_log`
- `artifacts.progress`
- `artifacts.weights_dir`
- `artifacts.best_checkpoint`
- `artifacts.results_csv`
- `artifacts.args_yaml`
- `artifacts.plots`

## `detections.csv`

The current analysis detection export writes:

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

The frontend trajectory review surface combines `track_summary.csv` with `projections.csv` so it can rank projected tracks and draw recent player and ball movement on the pitch map.

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

If no provider resolves:

- the file still exists
- `status` is `disabled`
- the diagnostics list falls back to heuristic diagnostics

## Backend API Surface

The current FastAPI application exposes these routes.

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

### Persisted analysis runs

- `GET /api/runs/recent`
- `GET /api/runs/{run_id}`
- `POST /api/runs/{run_id}/refresh-diagnostics`

### Detector training

- `GET /api/train/config`
- `POST /api/train/datasets/scan`
- `GET /api/train/jobs`
- `GET /api/train/jobs/{job_id}`
- `POST /api/train/jobs/{job_id}/stop`
- `POST /api/train/jobs/detect`
- `GET /api/train/runs/recent`
- `GET /api/train/runs/{run_id}`
- `POST /api/train/runs/{run_id}/activate`
- `GET /api/train/registry`

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

### `POST /api/train/datasets/scan`

JSON body:

- `path`

### `POST /api/train/jobs/detect`

JSON body:

- `base_weights`
- `dataset_path`
- `run_name`
- `epochs`
- `imgsz`
- `batch`
- `device`
- `workers`
- `patience`
- `freeze`
- `cache`

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

The repository still includes an offline batch runner:

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

## Batch Outputs

A batch run writes to a timestamped directory under `backend/experiments/` and produces:

- `manifest.json`
- `batch.log`
- `runs.csv`
- `tracker_comparison.csv`
- `entropy_windows_1hz.csv`
- `summary.json`
- `runs/` with one analysis run directory per processed half
