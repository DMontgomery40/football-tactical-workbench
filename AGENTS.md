# Football Tactical Workbench Instructions

This repository is a browser-first football analysis tool. Keep the UI and the backend aligned.

## Current product direction

- This is no longer a pose-first project.
- The active pipeline is:
  - `soccana` object detection
  - hybrid appearance-aware player tracking plus tracklet stitching
  - ByteTrack ball tracking
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
- `backend/app/ai_diagnostics.py`
  - provider-backed run diagnostics
  - supports OpenAI, OpenRouter, Anthropic, and local OpenAI-compatible endpoints
  - current OpenAI default path is `gpt-5.4`
- `backend/app/wide_angle.py`
  - core video analysis pipeline
  - Soccana detector and field-keypoint model resolution
  - player ReID tracking and tracklet stitching
  - live preview generation
  - overlay rendering and CSV/summary export
- `backend/app/reid_tracker.py`
  - sparse appearance embedding extraction
  - field-aware player association
  - post-pass tracklet stitching
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
- Default player tracker mode is `hybrid_reid`.
- Keep the legacy `bytetrack` player mode available as an explicit comparison fallback, not as the silent default.
- Pitch calibration is automatic and refreshed every 10 frames.
- Ball detection uses the same soccer-specific detector unless the user explicitly asks to change that.
- AI diagnostics are a real post-run path now. Do not revert to heuristic-only diagnostics unless explicitly requested.
- Keep diagnostics provider/model/source fields aligned between backend summary output and frontend display.
- Keep all user-visible status in job logs and `summary.json`; the frontend depends on that.
- Warn-level AI diagnostics should identify the exact function / condition / fallback most likely responsible and propose a concrete code change, not just tell the user to inspect an area.
- When tracking changes, keep `raw_*` and stitched identity metrics both available in `summary.json` so regression review can compare them.
- If you change the exported summary shape, update the frontend in the same turn.

## Frontend rules

- The UI is the main debugging surface. Prioritize clarity over cleverness.
- The user must be able to see:
  - source clip playback
  - live preview state
  - calibration health
  - tracking quality
  - saved overlay outputs
- The app must feel like a stateful tool, not a stack of unrelated cards.
- Persist obvious repeated values locally:
  - last-used local video path
  - last-used dataset folder
  - theme mode
  - useful SoccerNet browse state
- Do not make the user re-enter paths that are predictably the same every session.
- Overlay playback is the primary artifact. In run review, the video must be visually dominant.
- Diagnostics and overlay are compatible views. Do not force them into mutually exclusive tabs unless explicitly asked.
- Keep the main diagnostic cards readable, but put nitty-gritty code diagnosis and suggested fixes behind collapsible drilldowns instead of hiding them entirely.
- Passive state cards and interactive controls must look unmistakably different.
- Do not default the page into stale saved-run review when there is no current input clip loaded.
- SoccerNet browsing should be reactive and scalable. Avoid giant scroll dumps and avoid unnecessary “fetch” clicks for simple filtering.
- Do not add controls that the backend no longer uses.
- Avoid stale or duplicate controls. If a workflow is removed backend-side, remove its UI.
- If the UI starts getting worse from incremental patches, prefer a coherent refactor over more local fixes.

## Error handling rules

- Silent error swallowing is banned by default. Do not introduce empty `catch {}`, `catch { return fallback; }`, `except Exception: pass`, or broad fallback returns that erase an operator-visible failure.
- If an error affects a user-triggered action, state-changing request, dataset scan, training job, registry activation, artifact load, diagnostics generation, or API response parsing, surface it explicitly. Valid sinks are the visible UI state, job logs, `summary.json`, or structured server logging. Pick the sink the operator will actually see.
- If a mutation succeeds but a follow-up refresh fails, never report the mutation itself as failed and never hide the refresh failure. Surface a partial-success warning that says what succeeded and what did not refresh.
- The only allowed silent-swallow cases are truly best-effort cleanup, telemetry, or local persistence where ignoring the failure cannot change runtime correctness or operator decisions.
- Every intentional swallow must include an inline comment starting with `INTENTIONAL_SWALLOW:` that names the exact failure being ignored, why it is safe here, and what fallback behavior is preserved.
- Intentional swallows must catch as narrowly as practical. If a broad catch is unavoidable, explain why in the `INTENTIONAL_SWALLOW:` comment.
- Never silently swallow API parse failures, model-loading failures, file reads for user-visible artifacts, registry mutations, training control requests, or anything that can make the UI claim a stale or guessed state is current.

## Tooltip rules

- The central source of truth for educational tooltip content is [backend/app/help_catalog.json](backend/app/help_catalog.json).
- If you add, remove, rename, or substantially change a technical control, metric, or review concept, update that JSON in the same turn.
- For anything materially technical, tooltips are required going forward. That includes model choices, tracker modes, calibration concepts, projection concepts, and non-obvious review metrics.
- Do not rely on inline prose alone for technical explanation. Use an explicit info-button tooltip/popover driven from the central catalog.
- Not all on-screen hints need to go away. Keep inline text for live state, warnings, errors, confirmations, progress, and operator-critical status that must remain visible without hover/click.
- Tooltip quality bar:
  - The title must be specific, not generic.
  - The summary must explain why the concept matters in this product.
  - The body must be verbose enough to teach the operator what the stage does, how it fails, and how to interpret it in review.
  - Use links only when they add real value. Many entries should have none.
  - When links are appropriate, use current primary sources and allow up to roughly 4 strong links.
- Research-link recency rule:
  - Prefer March 2026 sources whenever realistically available.
  - 2025 is acceptable fallback.
  - Pre-2025 is banned for models or fast-moving technical topics unless the user explicitly approves an exception in private chat.
  - Never surface exception/override language in the UI.
- Do not use native browser `title` tooltips as the primary help mechanism for technical guidance.

## Validation expectations

Automated test placement:

- Put all new automated tests under the top-level `tests/` directory.
- Keep language-specific groupings inside `tests/` (for example `tests/backend/` and `tests/frontend/`) instead of scattering tests across the repo.
- Use `pytest` for Python-side automated tests under `tests/backend/`.
- Keep generated test artifacts out of git by using ignore rules under `tests/` rather than ignoring the tests themselves.

Full suite requirement:

- After every change, run the full suite with:

```bash
bash tests/testsuite_full.sh
```

After backend changes:

```bash
cd backend && python -m py_compile app/main.py app/wide_angle.py
```

After frontend changes:

```bash
cd frontend && npm run build
```

For substantial frontend changes:

- Use a real browser check, not just a build.
- Verify layout and interaction in-browser with Playwright or equivalent.
- Check at minimum:
  - default boot state
  - input clip loading
  - live preview state
  - saved run review
  - overlay plus diagnostics visibility
  - SoccerNet browse/filter behavior
  - obvious path persistence across refresh

When changing the pipeline:

- Run at least one real clip through `/api/live-preview`
- Run at least one real clip through `/api/analyze`
- If the change affects player identity, run at least one tracker A/B comparison between `hybrid_reid` and `bytetrack`
- Prefer clips in `backend/datasets/bundesliga_sample/` for quick validation

## Performance expectations

- Full matches are long-running jobs. Do not assume they will finish quickly.
- Use short clips for iteration, then full matches for end-to-end validation.
- If you add a heavier model step, surface that cost in logs or UI text.
- Sparse appearance embeddings are acceptable; per-detection deep ReID on every frame is not.

## Common pitfalls

- Do not silently fall back to generic `yolo11n.pt` / `yolo11n-pose.pt` without telling the user.
- Do not mix outdated README/UI copy from the old pose/windows prototype into the current app.
- Do not describe stitched player IDs as full match-long identity unless the metrics actually support it.
- Do not change the browser workflow in a way that makes it harder to tell whether the tracker, ball detector, or field calibration is actually working.
- Do not treat visual polish as success if the interaction model is still confusing.
- Do not hide the most important evidence, especially the overlay video, behind weak layout ratios.
- Do not let AI-generated diagnostics sound like canned heuristics or generic chatbot prose.
