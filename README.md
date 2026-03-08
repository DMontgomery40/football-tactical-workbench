# Football Tactical Workbench

A local Mac-friendly workbench for wide-angle football analysis with a real UI, live model preview, and automatic pitch calibration.

## What it does

- React UI on **http://127.0.0.1:4317**
- FastAPI backend on **http://127.0.0.1:8431**
- Soccer-specific detection with **Soccana** weights
- Player, ball, and referee detection
- ByteTrack-based multi-object tracking
- Unsupervised home/away separation from jersey colors
- Automatic field-keypoint calibration with **Soccana_Keypoint**
- Pitch projection refreshed every **10 frames**
- Live model preview in the browser while the clip plays
- Saved overlay video plus CSV diagnostics and summaries

## Current model stack

- Detector: `backend/models/soccana/Model/weights/best.pt`
- Field calibration: `backend/models/soccana_keypoint/Model/weights/best.pt`

These are local `.pt` files. The backend can also resolve them from Hugging Face if missing. Local model weights and cached downloads are not meant to be committed to git.

## What the UI is for

The UI is the primary debugging surface.

You should be able to tell from the browser:

- whether player tracking is stable
- whether the ball detector is firing or hallucinating
- whether team clustering looks plausible
- whether field calibration is locking and staying locked
- whether the projected minimap matches the play

This repo is not meant to be “terminal only”.

## Fastest way to run

### One command

```bash
echo "starting workbench" && cd football-tactical-workbench && bash run_all.sh
```

### Two terminal method

Terminal 1:

```bash
echo "starting backend" && cd football-tactical-workbench/backend && bash run_backend.sh
```

Terminal 2:

```bash
echo "starting frontend" && cd football-tactical-workbench/frontend && bash run_frontend.sh
```

## Typical workflow

1. Load a clip from disk or upload one in the UI.
2. Click `Load clip`.
3. Start `Live model preview`.
4. Watch the browser overlay for:
   - player and ball boxes
   - team labels
   - calibration status
   - minimap behavior
5. If the preview looks sane, click `Run tactical demo`.
6. Review the saved overlay and diagnostics after completion.

## Good first use

Start with a short clip first.

- 10 to 30 seconds
- one broadcast camera phase
- wide-angle view with visible field structure
- minimal replay cuts

Full matches are supported, but they are long-running jobs and should be treated like batch work.

## Where to point it

You have two choices in the UI:

1. upload a video file
2. paste a local path to a video on your Mac

You can also scan a dataset folder and click one of the discovered videos to auto-fill the path field.

## Included local data

- Bundesliga sample clips: `backend/datasets/bundesliga_sample`
- YouTube match downloads: `backend/datasets/youtube_clips`
- SoccerNet downloads: `backend/datasets/soccernet`

These are local working datasets and are not meant to be committed to git.

## SoccerNet access

SoccerNet video access is not public in the normal sense.

- It is password protected.
- It is only available to people who have personally signed the SoccerNet NDA.
- Access is tied to the SoccerNet terms you agreed to, including non-commercial usage restrictions.

Do not share downloaded SoccerNet videos or credentials through this repo.

## SoccerNet labels for the experiment

For the spatial-entropy volatility experiment, the important file is:

- `Labels-v2.json`

That file contains event timestamps, including goals, at 1-second resolution. The backend now prefers `Labels-v2.json` when it is present and aligns goal events to the current half so the experiment can compare volatility in pre-goal windows against baseline match state.

## What files you get

Inside each run folder under `backend/runs/<run_id>/outputs/`:

- `overlay.mp4`
- `detections.csv`
- `track_summary.csv`
- `projections.csv` when field registration is active
- `entropy_timeseries.csv` for the experimental volatility signal
- `goal_events.csv` when SoccerNet goal labels are available
- `summary.json`
- `all_outputs.zip`

## What to look at first

### If `unique_player_track_ids` is huge

The tracker is fragmenting players.

### If `average_ball_detections_per_frame` is near zero

The ball stage is too weak to trust as a tactical signal.

### If `field_calibration_refresh_successes` is low

The pitch-keypoint model is not seeing enough field structure to keep projection stable.

### If `field_registered_ratio` is low

Most player anchors are not landing on the pitch model, so spatial metrics downstream are weak.

### If `team_cluster_distance` is low

The jersey-color split is noisy and home/away labels should be treated carefully.

### If `goal_events_count` is zero

The experiment is not attached to scoring events for that run. Without goal labels, the volatility signal is exploratory only and not yet meaningful for outcome modeling.

## Validation commands

Backend syntax check:

```bash
echo "checking backend" && cd football-tactical-workbench/backend && python -m py_compile app/main.py app/wide_angle.py
```

Frontend production build:

```bash
echo "building frontend" && cd football-tactical-workbench/frontend && npm run build
```

## Stop the background backend

```bash
echo "stopping backend" && cd football-tactical-workbench && bash stop_backend.sh
```
