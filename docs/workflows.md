# Workflows

## Top-Level App Layout

The frontend is one SPA with two top-level product surfaces.

### `Analysis Workspace`

- left sidebar:
  - `Prepare an input clip`
  - `SoccerNet`
  - `Scan a local dataset folder`
- right workspace tabs:
  - `Input`
  - `Live Preview`
  - `Active Job`
  - `Run Review`
- review tabs inside `Run Review`:
  - `Overview`
  - `Tracks`
  - `Files`

### `Training Studio`

- top-level sub-navigation:
  - `Datasets`
  - `Train`
  - `Jobs`
  - `Registry`

The browser stores several values in local storage:

- theme mode
- analysis form fields
- analysis sidebar width
- SoccerNet split, query, and selected files
- training dataset path

The header `Reset` control clears the saved browser state.

## Workflow 1: Analyse A Local Clip

1. Open the app at `http://127.0.0.1:4317`.
2. Stay in `Analysis Workspace`.
3. In `Prepare an input clip`, either upload a video or paste a local path.
4. Optionally adjust:
   - detector weights
   - label path for goal-aligned experiments
   - player tracker mode
   - include ball tracking
   - player confidence
   - ball confidence
   - IOU
5. Use the inline help popovers when you need current pipeline, runtime, or benchmark context.
6. Click `Load input clip`.
7. Click `Analyze loaded clip`.
8. Move to `Active Job` to watch logs and progress.
9. When the job completes, move into `Run Review`.

## Workflow 2: Use Live Preview

Live preview is a transient stream for fast inspection. Completed analysis runs create the saved review artifacts.

1. Load a source clip first.
2. Open the `Live Preview` workspace tab.
3. Start live preview.
4. The backend streams annotated JPEG frames from `/api/live-preview`.

The current live preview draws:

- player boxes and IDs
- ball box and ID when enabled
- detected field keypoints
- minimap when calibration is active
- calibration status text

## Workflow 3: Review A Saved Run

Saved runs come from `backend/runs/<run_id>/outputs/summary.json`.

### Entry

1. Open `Run Review`.
2. Select one saved run from the list.

The review surface reads persisted run data, not a live in-memory job directly. That distinction matters:

- `Input` and `Live Preview` reflect the currently loaded clip
- `Run Review` reflects the selected saved run and its persisted diagnostics artifacts

### Overview

The `Overview` review tab shows:

- overlay playback
- trajectory window for projected player and ball paths
- run brief
- headline diagnostics
- run metrics
- experimental signals
- detailed diagnostics

### Tracks

The `Tracks` tab shows the top rows from `summary.top_tracks`.

### Files

The `Files` tab links to exported run artifacts through `/runs/...`.

## Workflow 4: Use Training Studio

Training Studio is a separate top-level surface. Do not treat it as another analysis tab.

### `Datasets`

Use this tab to validate a local YOLO detector dataset before training:

- dataset path
- dataset YAML detection
- split paths
- image and label counts
- labeled-image counts
- class mapping back to analysis
- warnings and blocking errors

### `Train`

Use this tab to launch a detector fine-tune:

- training family is detector-only in V1
- base weights default to `soccana`
- device can stay on `auto` or be pinned to `mps`, `cpu`, or `cuda` when available
- the worker writes a run-local `dataset_runtime.yaml` and does not mutate the source dataset
- the tab also shows whether DVC is ready for durable dataset or promoted-checkpoint tracking on this machine

### `Jobs`

Use this tab to inspect detector fine-tuning jobs:

- queued / running / completed / failed status
- progress
- logs
- metrics
- generated dataset manifest path
- artifact paths
- best checkpoint path
- training provenance and DVC tracked/untracked state

### `Registry`

Use this tab to inspect detector checkpoints that are available to analysis:

- active detector
- created time
- base weights
- metrics summary
- class ID mapping
- checkpoint path
- training provenance and DVC tracked/untracked state
- activation control

When a custom detector is activated here, analysis uses it whenever the analysis selector remains on `soccana`. Activation also copies the checkpoint into `backend/models/promoted/custom_<run_id>/` and writes `training_provenance.json` beside it.

## Workflow 5: Use SoccerNet In The UI

The SoccerNet panel lets you:

- choose split: `train`, `valid`, `test`, `challenge`
- search official game paths
- select video halves and label files
- start a download job
- scan the local SoccerNet dataset directory into the folder scanner

The UI defaults the selected file list to:

- `1_720p.mkv`
- `2_720p.mkv`
- `Labels-v2.json`

The frontend refreshes matching games reactively when you change split or search text.

## Workflow 6: Scan A Local Folder

The folder scanner recursively searches a local directory for:

- videos with suffixes `.mp4`, `.mov`, `.avi`, `.mkv`, `.m4v`
- annotation-like files with suffixes `.json`, `.csv`, `.txt`

You can click a discovered video chip to:

- populate the local video path
- immediately load that clip into the source preview

## What Persists And What Does Not

### Persisted on disk

- completed analysis runs under `backend/runs/`
- completed detector training runs under `backend/training_runs/`
- downloaded model weights under `backend/models/`
- activated detector promotions under `backend/models/promoted/`
- detector registry under `backend/models/registry.json`
- downloaded SoccerNet files under `backend/datasets/soccernet/`
- source registry under `backend/uploads/sources.json`
- analysis job snapshots under each analysis run directory as `job_state.json`
- training job snapshots under each training run directory as `job_state.json`

### In memory only

- current live preview stream state
- current browser polling state
- current SoccerNet download polling timer

If the backend restarts:

- loaded sources remain available
- interrupted analysis jobs reappear as persisted snapshots
- interrupted training jobs reappear as persisted snapshots
- completed analysis and training runs remain available

## Current Runtime Defaults

- detector model field starts at `soccana`
- ball tracking is enabled by default
- player confidence defaults to `0.25`
- ball confidence defaults to `0.20`
- IOU defaults to `0.50`
- field calibration is automatic and has no manual UI control
- detector training base weight defaults to `soccana`
- detector training remains local-only and detector-only in V1

## Goal-Aligned Experiment Caveat

The current experiment output is always written, but goal-aligned interpretation only becomes meaningful when the backend finds a label file beside the source clip.

The goal loader currently checks:

- an explicit label path from the UI when you provide one
- the source clip directory
- nearby parent directories for `Labels-v2.json` or `Labels.json`

If no label file resolves:

- `goal_events_count` is `0`
- `goal_events_csv` is `null`
- the experiment still exists, but pre-goal metrics remain zeroed
