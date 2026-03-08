# Football Tactical Workbench Instructions

This repository is a browser-first football analysis tool. Keep the UI and the backend aligned.

## Current product direction

- This is no longer a pose-first project.
- The active pipeline is:
  - `soccana` object detection
  - ByteTrack tracking
  - jersey-color clustering for home/away separation
  - `soccana_keypoint` field registration
  - automatic pitch calibration refresh every **10 frames**
  - live browser preview plus saved overlay output

Do not reintroduce the earlier manual homography-point workflow unless the user explicitly asks for it.

## Repository map

- `backend/app/main.py`
  - FastAPI entrypoint
  - source loading endpoints
  - live preview endpoint
  - analyze job creation and persisted run loading
- `backend/app/wide_angle.py`
  - core video analysis pipeline
  - Soccana detector and field-keypoint model resolution
  - live preview generation
  - overlay rendering and CSV/summary export
- `backend/models/`
  - local model cache
  - active detector: `backend/models/soccana/Model/weights/best.pt`
  - active field model: `backend/models/soccana_keypoint/Model/weights/best.pt`
- `backend/runs/<run_id>/outputs/`
  - generated overlays, CSVs, and summaries
- `frontend/src/App.jsx`
  - primary app UI
  - if the backend contract changes, this file usually needs to change too
- `frontend/src/styles.css`
  - styling for the single-page UI

## Backend rules

- Default detector choice is `soccana`.
- Default field calibration model is `soccana_keypoint`.
- Pitch calibration is automatic and refreshed every 10 frames.
- Ball detection uses the same soccer-specific detector unless the user explicitly asks to change that.
- Keep all user-visible status in job logs and `summary.json`; the frontend depends on that.
- If you change the exported summary shape, update the frontend in the same turn.

## Frontend rules

- The UI is the main debugging surface. Prioritize clarity over cleverness.
- The user must be able to see:
  - source clip playback
  - live preview state
  - calibration health
  - tracking quality
  - saved overlay outputs
- Do not add controls that the backend no longer uses.
- Avoid stale or duplicate controls. If a workflow is removed backend-side, remove its UI.

## Validation expectations

After backend changes:

```bash
cd backend && python -m py_compile app/main.py app/wide_angle.py
```

After frontend changes:

```bash
cd frontend && npm run build
```

When changing the pipeline:

- Run at least one real clip through `/api/live-preview`
- Run at least one real clip through `/api/analyze`
- Prefer clips in `backend/datasets/bundesliga_sample/` for quick validation

## Performance expectations

- Full matches are long-running jobs. Do not assume they will finish quickly.
- Use short clips for iteration, then full matches for end-to-end validation.
- If you add a heavier model step, surface that cost in logs or UI text.

## Common pitfalls

- Do not silently fall back to generic `yolo11n.pt` / `yolo11n-pose.pt` without telling the user.
- Do not mix outdated README/UI copy from the old pose/windows prototype into the current app.
- Do not change the browser workflow in a way that makes it harder to tell whether the tracker, ball detector, or field calibration is actually working.
