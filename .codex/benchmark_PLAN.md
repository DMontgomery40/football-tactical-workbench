# Benchmark Lab V2: Real Multi-Suite Benchmark Matrix

## Related Memory

- Repo-local memory hub: [`.codex/MEMORY.md`](./MEMORY.md)
- Repo-local Stage 2 handoff: [`.codex/STAGE2_HANDOFF_PROMPT.md`](./STAGE2_HANDOFF_PROMPT.md)
- Canonical project memory index: [/Users/davidmontgomery/.codex/projects/-Users-davidmontgomery-football_pose_workbench/MEMORY.md](/Users/davidmontgomery/.codex/projects/-Users-davidmontgomery-football_pose_workbench/MEMORY.md)
- Benchmark V2 source-of-truth memory: [/Users/davidmontgomery/.codex/projects/-Users-davidmontgomery-football_pose_workbench/memory/benchmark-lab-v2-source-of-truth.md](/Users/davidmontgomery/.codex/projects/-Users-davidmontgomery-football_pose_workbench/memory/benchmark-lab-v2-source-of-truth.md)
- Live port ownership memory: [/Users/davidmontgomery/.codex/projects/-Users-davidmontgomery-football_pose_workbench/memory/benchmark-live-process-paths.md](/Users/davidmontgomery/.codex/projects/-Users-davidmontgomery-football_pose_workbench/memory/benchmark-live-process-paths.md)

## Execution Ledger

- Snapshot date: 2026-03-12.
- Plan handling rule: preserve the original bullets below and annotate them in place. Do not compress this file into a shorter rewrite when status changes.
- Working definition of “finished” for this plan:
  - real benchmark suites are runnable instead of merely scaffolded
  - docs/tooltips/UI language describe the new suite/recipe matrix honestly
  - DVC durability is real for benchmark materials
  - required real runs and validation artifacts exist
  - no dead transition code remains in the Benchmark Lab surface

## Summary

- Rebuild Benchmark Lab around named benchmark suites, recipe-level rows, official or standard evaluators, and DVC-tracked suite manifests.
  - Status on 2026-03-12: substantially implemented. Named suites, recipe rows, backend contracts, TS/Tailwind frontend shell, matrix/detail flow, benchmark DVC tracking, vendored evaluator trees, real v2 smoke artifacts, matrix-synced benchmark charts, and suite-evaluation charts now exist. Remaining finish work is honest executable readiness for every non-detection suite, fuller docs/help cleanup, and deeper suite-specific validation.
- Ship these suite IDs in v1:
  - `det.roles_quick_v1`: [martinjolif/football-player-detection](https://huggingface.co/datasets/martinjolif/football-player-detection), primary `AP@[.50:.95]`
    - Status on 2026-03-12: suite definition and manifest exist; detection evaluator is real `pycocotools`-based code. This is the most complete labeled suite path in the repo right now.
  - `det.ball_quick_v1`: [martinjolif/football-ball-detection](https://huggingface.co/datasets/martinjolif/football-ball-detection), primary `AP_ball@[.50:.95]`
    - Status on 2026-03-12: suite definition and manifest exist; it will use the same real COCO evaluator path once dataset materialization is present.
  - `loc.synloc_quick_v1`: [Spiideo SynLoc / `sskit`](https://github.com/Spiideo/sskit), primary `mAP-LocSim`
    - Status on 2026-03-12: suite definition and adapter scaffold exist; vendored tool checkout and executable validation path are still missing.
  - `spot.team_bas_quick_v1`: [SoccerNet/SN-BAS-2025](https://huggingface.co/datasets/SoccerNet/SN-BAS-2025/tree/main) via [sn-teamspotting](https://github.com/SoccerNet/sn-teamspotting) and [sn-spotting](https://github.com/SoccerNet/sn-spotting), primary `Team-mAP@1`
    - Status on 2026-03-13: suite definition, vendored evaluator sources, isolated action-spotting runtime, repo-owned JSON wrapper, and a repo-owned prediction-tree bridge now exist. `backend/app/benchmark_eval/prediction_exports.py` can convert raw per-game event predictions into `artifacts/predictions/<game>/results_spotting.json`, and compatible cells now fail with a precise blocker naming the missing raw source artifact (`artifacts/recipe_event_spotting_predictions.json`) or missing `Labels-ball.json` game path instead of the old placeholder “missing export” state. A real adapter smoke was run end-to-end against a minimal evaluator-valid two-game test split and repo-owned raw predictions; the wrapper/export path completed successfully and returned `map_at_1 = 0.08333333333333333` with `team_map_at_1 = NaN` on that tiny synthetic split. Stage 2 now has a real manifest plus a downloaded official `SN-BAS-2025` `valid.zip` under `backend/datasets/huggingface/soccernet_sn_bas_2025/valid.zip`, but the extracted `Labels-ball.json` under `backend/benchmarks/_datasets/spot.team_bas_quick_v1/` is still zero-filled/unreadable on this machine, so the suite remains honestly blocked on password-protected validation members and on the still-missing `event_spotting` recipe family.
  - `calib.sn_calib_medium_v1`: [sn-calibration](https://github.com/SoccerNet/sn-calibration), primary `Completeness x JaC@5`
    - Status on 2026-03-13: suite definition, vendored checkout, legacy runtime, repo-owned JSON wrapper, and the prediction-export bridge now exist. `backend/app/benchmark_eval/prediction_exports.py` runs repo-owned calibration inference over `dataset_root/valid/*.jpg`, converts image-to-pipeline homographies into SoccerNet camera JSONs, and writes `artifacts/predictions/valid/camera_<frame_id>.json` plus an export summary before the wrapper runs. A real adapter smoke was run end-to-end by extracting a frame from the local benchmark clip, deriving evaluator-valid ground-truth line annotations from the predicted camera, and then running the salvage calibration adapter; metrics came back `completeness_x_jac_5 = 1.0`, `completeness = 1.0`, `jac_5 = 1.0`. A real bug in the camera-module import path surfaced during that smoke and was fixed in the same turn. Stage 2 now also has a real suite manifest that points at the expected `calibration-2023/valid` downloader output instead of the old generic “dataset missing” prose, but the official validation tree is still absent locally.
  - `track.sn_tracking_medium_v1`: [sn-tracking](https://github.com/SoccerNet/sn-tracking) plus [sn-trackeval](https://github.com/SoccerNet/sn-trackeval), primary `HOTA`
    - Status on 2026-03-13: suite definition, vendored sources, and blocker reporting exist, and Stage 2 now has a real suite manifest that names the exact missing `SoccerNetMOT` tree plus the required `gt.zip` and `sample_submission.zip`. The suite is still honestly blocked because `backend/benchmarks/_datasets/track.sn_tracking_medium_v1/SoccerNetMOT` is not materialized and the current adapter still lacks the per-recipe `TRACKERS_FOLDER_ZIP` export bridge for `backend/third_party/soccernet/sn-tracking/tools/evaluate_soccernet_v3_tracking.py`.
  - `spot.pcbas_medium_v1`: [SoccerNet/SN-PCBAS-2026](https://huggingface.co/datasets/SoccerNet/SN-PCBAS-2026/tree/main) via [FOOTPASS](https://github.com/JeremieOchin/FOOTPASS), primary `F1@15%`
    - Status on 2026-03-13: suite definition, FOOTPASS checkout, evaluation runtime, repo-owned JSON wrapper, and the play-by-play export bridge now exist. `backend/app/benchmark_eval/prediction_exports.py` converts repo-owned raw play-by-play predictions into `artifacts/predictions.json`, and `evaluate_pcbas(...)` real-smoked successfully on this machine against vendored FOOTPASS sample validation files (`playbyplay_PRED/playbyplay_TAAD_val.json` against `playbyplay_GT/playbyplay_val.json`) with F1@15% `0.40995387750724016`. Stage 2 now has a real suite manifest plus a locally staged benchmark ground-truth file at `backend/benchmarks/_datasets/spot.pcbas_medium_v1/playbyplay_GT/playbyplay_val.json`, so the suite no longer reports a generic missing root. It remains honestly blocked because the official validation tactical/video archives are gated on Hugging Face (`401 GatedRepoError` for `tactical_data_VAL.zip`) and the shipped catalog still has no compatible event-capable recipe.
  - `gsr.medium_v1`: [SoccerNet/SN-GSR-2025](https://huggingface.co/datasets/SoccerNet/SN-GSR-2025), DVC-pinned 12-clip validation subset, primary `GS-HOTA`
    - Status on 2026-03-13: suite definition, vendored TrackLab/sn-gamestate sources, blocker reporting, and the dedicated `py39 + numpy<2` runtime now exist. Stage 2 now has a real suite manifest that names the expected `valid.zip` source archive and the remaining placeholder 12-clip ids. The suite is still honestly blocked by dataset materialization and the missing TrackLab/Hydra recipe bridge, and the local archive probe now has a concrete reason: `SoccerNet/SN-GSR-2025` only exposes `valid.zip` (~11.17 GB) and this workspace does not currently have enough free disk to stage that archive.
  - `gsr.long_v1`: full `SN-GSR-2025` validation via [TrackLab](https://github.com/TrackingLaboratory/tracklab), [sn-gamestate](https://github.com/SoccerNet/sn-gamestate), and [sn-trackeval](https://github.com/SoccerNet/sn-trackeval), primary `GS-HOTA`
    - Status on 2026-03-13: suite definition, long-run manifest, and the dedicated TrackLab/sn-gamestate runtime now exist. The suite is still unfinished for the same explicit dataset-materialization and TrackLab adapter reasons as `gsr.medium_v1`, and the Stage 2 blocker is now concrete: the full validation `valid.zip` is the only upstream archive and the current machine does not have enough free disk to stage and unpack it.
  - `ops.clip_review_v1`: preserved clip pipeline, primary `FPS`, secondary `track_stability`, `calibration`, `coverage`, explicitly labeled non-ground-truth
    - Status on 2026-03-12: implemented and carried forward. Legacy benchmark records hydrate into this suite, and the operational evaluator wraps the real analysis pipeline.
- No global composite for real suites. The matrix is sort/filter only. Unsupported cells render `N/A`.
  - Status on 2026-03-12: implemented in the active frontend shell. `N/A`/unsupported status is wired backend-side and rendered, and the matrix now carries preset, filter, and sort controls.
- Keep DVC as the benchmark durability layer for suite datasets, manifests, conversions, and provenance.
  - Status on 2026-03-13: the product contract still treats DVC as the intended durability layer, but this salvage worktree does not currently have the benchmark-root `.dvc` pointer files present. Dataset/manifests/conversions therefore surface honestly as local present/untracked state here instead of pretending DVC tracking is active.
- All new frontend benchmark work starts in TypeScript and uses Tailwind utilities as the primary styling layer. Do not add new benchmark UI in plain JS or new legacy `styles.css`-first patterns.
  - Status on 2026-03-12: implemented. The new benchmark surface is in TS/TSX and Tailwind-first. Remaining cleanup is removing dead TS transition leftovers and stale old-model docs/help copy.

## Execution Teams

### Team 1: Suite Data And Provenance
- Write scope: [pull_hf_football_datasets.py](/Users/davidmontgomery/football_pose_workbench/backend/scripts/pull_hf_football_datasets.py), new [benchmark_suites.py](/Users/davidmontgomery/football_pose_workbench/backend/app/benchmark_suites.py), new [benchmark_provenance.py](/Users/davidmontgomery/football_pose_workbench/backend/app/benchmark_provenance.py), `backend/benchmarks/_datasets/`, `backend/benchmarks/_manifests/`, `backend/benchmarks/_conversions/`.
  - Status on 2026-03-13: moved beyond scaffolding. The salvage tree now has real suite manifest files under `backend/benchmarks/_manifests/`, a downloaded `SN-BAS-2025` validation archive under `backend/datasets/huggingface/soccernet_sn_bas_2025/`, and a locally staged `spot.pcbas_medium_v1` validation GT JSON under `backend/benchmarks/_datasets/spot.pcbas_medium_v1/playbyplay_GT/playbyplay_val.json`.
- Add `backend/app/benchmark_suites.json` with the 10 suite definitions above, including `tier`, `source_url`, `license`, `protocol`, `primary_metric`, `metric_columns`, `required_capabilities`, `dataset_root`, `manifest_path`, and `dvc_required=true`.
  - Status on 2026-03-12: implemented. The suite registry JSON exists and is loaded by `benchmark_suites.py`.
- DVC-track `backend/benchmarks/_datasets/<suite_id>/`, `backend/benchmarks/_manifests/<suite_id>.json`, and `backend/benchmarks/_conversions/<suite_id>/`.
  - Status on 2026-03-13: not yet true in this salvage worktree. The benchmark directories exist and Benchmark Lab now reports their DVC status honestly as present/untracked because the root `.dvc` pointer files are not present here.
- For `gsr.medium_v1`, create a fixed `gsr_medium_12clip_manifest.json`; for `gsr.long_v1`, use the full validation manifest.
  - Status on 2026-03-13: still partially implemented. The suite manifests now carry concrete archive/blocker details, but the `gsr.medium_v1` item list is still placeholder data because the full `valid.zip` archive has not been staged locally and the 12-clip subset cannot be locked honestly yet.

### Team 2: Evaluator Stack
- Write scope: `backend/requirements.txt`, new `backend/app/benchmark_eval/`, new `backend/third_party/soccernet/`, new `backend/third_party/LOCKS.md`.
  - Status on 2026-03-12: implemented for repository materialization. Requirements and benchmark-eval package updates landed; `LOCKS.md` exists; vendored third-party repo payloads are now present under `backend/third_party/soccernet/`.
- Add `pycocotools`, `sskit`, and `sn-trackeval` to `backend/requirements.txt`.
  - Status on 2026-03-12: implemented.
- Vendor pinned snapshots of `sn-calibration`, `sn-tracking`, `sn-spotting`, `sn-teamspotting`, `sn-gamestate`, `FOOTPASS`, and `TrackLab` under `backend/third_party/soccernet/`; record upstream URL + commit SHA in `backend/third_party/LOCKS.md`.
  - Status on 2026-03-12: implemented. The pinned snapshots were cloned at the locked commits and vendored into `backend/third_party/soccernet/`.
- Implement adapter modules:
  - `coco_detection.py`: convert YOLO labels/preds to COCO JSON once, score with `pycocotools.COCOeval`
    - Status on 2026-03-12: implemented as the real detection evaluator.
  - `synloc.py`: call `sskit` and persist `mAP-LocSim`
    - Status on 2026-03-12: scaffolded only; still depends on missing vendored/runtime pieces.
  - `team_spotting.py`: call `sn-teamspotting` and `sn-spotting`
    - Status on 2026-03-12: scaffolded only; still depends on missing vendored/runtime pieces.
  - `calibration.py`: call `sn-calibration`
    - Status on 2026-03-12: scaffolded only; still depends on missing vendored/runtime pieces.
  - `tracking.py`: call `sn-tracking` and `sn-trackeval`
    - Status on 2026-03-12: no longer a generic scaffold. The adapter now points at the real vendored `sn-tracking` evaluator path and emits explicit blocker reasons, but honest execution is still blocked by missing `SoccerNetMOT` materialization and the absent recipe-to-`TRACKERS_FOLDER_ZIP` export bridge.
  - `pcbas.py`: call `FOOTPASS`
    - Status on 2026-03-12: scaffolded only; still depends on missing vendored/runtime pieces.
  - `gamestate.py`: call `TrackLab` + `sn-gamestate` + `sn-trackeval`
    - Status on 2026-03-12: no longer a generic scaffold. The adapter now emits explicit blocker reasons tied to the vendored TrackLab/sn-gamestate stack, but honest execution is still blocked by missing `SoccerNetGS` materialization, placeholder medium-manifest clip ids, Python/NumPy incompatibility (`sn-gamestate` wants `<3.10` / `numpy<2`), missing installed `tracklab`/`sn_gamestate` packages, and the absent recipe-to-Hydra bridge.
  - `operational.py`: wrap the current clip pipeline unchanged except for honest naming
    - Status on 2026-03-12: implemented.
- Do not hand-roll any metric family above.
  - Status on 2026-03-12: implemented in spirit. Real metric families are used for detection and the remaining adapters are intentionally external-evaluator wrappers rather than custom metric math.

### Team 3: Catalog, Recipes, Orchestration
- Write scope: [benchmark.py](/Users/davidmontgomery/football_pose_workbench/backend/app/benchmark.py), new [benchmark_catalog.py](/Users/davidmontgomery/football_pose_workbench/backend/app/benchmark_catalog.py), new `backend/app/benchmark_catalog.json`, [main.py](/Users/davidmontgomery/football_pose_workbench/backend/app/main.py), [schemas.py](/Users/davidmontgomery/football_pose_workbench/backend/app/schemas.py).
  - Status on 2026-03-12: implemented.
- Add canonical asset catalog fields: `asset_id`, `kind`, `provider`, `label`, `version`, `architecture`, `artifact_path`, `capabilities`, `bundle_mode`, `runtime_binding`, `class_mapping`, `compatible_suites`.
  - Status on 2026-03-12: implemented.
- Generate recipe rows, not raw checkpoint rows:
  - `detector:soccana`
    - Status on 2026-03-12: implemented.
  - `detector:<custom>`
    - Status on 2026-03-12: implemented via registry/import catalog expansion.
  - `pipeline:soccermaster`
    - Status on 2026-03-12: implemented, with compatibility narrowed so it does not pretend to be a detector-only recipe.
  - `pipeline:sn-gamestate-tracklab`
    - Status on 2026-03-12: implemented as the explicit vendored TrackLab/sn-gamestate baseline row. It is first-class in the recipe catalog instead of being treated as hidden background tooling.
  - `tracker:soccana+bytetrack+soccana_keypoint`
    - Status on 2026-03-12: implemented.
  - `tracker:soccana+hybrid_reid+soccana_keypoint`
    - Status on 2026-03-12: implemented.
- Compatibility rule is fixed: a recipe is runnable for a suite only if `recipe.capabilities` satisfy `suite.required_capabilities`; otherwise every suite metric cell is `N/A`.
  - Status on 2026-03-12: implemented.
- New persisted run shape:
  - `backend/benchmarks/<benchmark_id>/benchmark.json`
    - Status on 2026-03-12: implemented.
  - `backend/benchmarks/<benchmark_id>/environment.json`
    - Status on 2026-03-12: implemented.
  - `backend/benchmarks/<benchmark_id>/suite_results/<suite_id>/<recipe_id>/result.json`
    - Status on 2026-03-12: implemented and exercised by real persisted v2 runs, including `20260312_175409_2a48ad`, `20260312_175156_134e9a`, `20260312_184212_11ffc2`, `20260312_185244_273baa`, and `20260312_185955_b062b7`.
  - `backend/benchmarks/<benchmark_id>/suite_results/<suite_id>/<recipe_id>/artifacts/*`
    - Status on 2026-03-12: implemented in the orchestrator contract, with operational artifacts working when that suite runs.
- Persist `runtime_env` with OS, CPU/GPU, torch, ultralytics, evaluator versions, and device selection.
  - Status on 2026-03-12: implemented.
- Legacy benchmark records with only `benchmark_summary.json` load as `ops.clip_review_v1`; do not rewrite old files.
  - Status on 2026-03-12: implemented.

### Team 4: API Contracts
- Write scope: [main.py](/Users/davidmontgomery/football_pose_workbench/backend/app/main.py), [schemas.py](/Users/davidmontgomery/football_pose_workbench/backend/app/schemas.py), [contracts.ts](/Users/davidmontgomery/football_pose_workbench/frontend/src/lib/api/contracts.ts), `packages/contracts/generated/openapi.json`, `packages/contracts/generated/schema.ts`.
  - Status on 2026-03-12: implemented.
- Keep endpoint names but change payloads:
  - `GET /api/benchmark/config`: `suites`, `assets`, `recipes`, `dataset_states`, `dvc_runtime`, `legacy_clip_status`
    - Status on 2026-03-12: implemented.
  - `POST /api/benchmark/run`: `{ suite_ids: string[], recipe_ids: string[], label?: string }`
    - Status on 2026-03-12: implemented.
  - `GET /api/benchmark/history`: normalized v2 runs plus hydrated legacy operational runs
    - Status on 2026-03-12: implemented.
  - `GET /api/benchmark/jobs/{benchmark_id}`: full suite-by-recipe results
    - Status on 2026-03-12: implemented.
  - `POST /api/benchmark/ensure-clip`: preserved for `ops.clip_review_v1` only
    - Status on 2026-03-12: implemented.
- Move Benchmark Lab off ad hoc `fetch()` calls and onto the generated OpenAPI client.
  - Status on 2026-03-12: implemented through `frontend/src/lib/api/contracts.ts`.

### Team 5: Frontend Matrix
- Write scope: replace JS benchmark UI files with TS/TSX under [frontend/src/benchmarkLab](/Users/davidmontgomery/football_pose_workbench/frontend/src/benchmarkLab).
  - Status on 2026-03-12: implemented.
- Convert these benchmark UI files to TypeScript during the rebuild instead of patching them in JS:
  - [BenchmarkLabShell.jsx](/Users/davidmontgomery/football_pose_workbench/frontend/src/benchmarkLab/BenchmarkLabShell.jsx) -> `BenchmarkLabShell.tsx`
    - Status on 2026-03-12: implemented.
  - [state.js](/Users/davidmontgomery/football_pose_workbench/frontend/src/benchmarkLab/state.js) -> `state.ts`
    - Status on 2026-03-12: implemented.
  - [Leaderboard.jsx](/Users/davidmontgomery/football_pose_workbench/frontend/src/benchmarkLab/Leaderboard.jsx) -> `ResultsMatrix.tsx`
    - Status on 2026-03-12: implemented.
  - [CandidateDetail.jsx](/Users/davidmontgomery/football_pose_workbench/frontend/src/benchmarkLab/CandidateDetail.jsx) -> `RecipeDetailPanel.tsx`
    - Status on 2026-03-12: implemented by replacement. The active detail surface is `DetailPanel.tsx`, and the dead transition file has been removed.
  - [CandidateLibrary.jsx](/Users/davidmontgomery/football_pose_workbench/frontend/src/benchmarkLab/CandidateLibrary.jsx) -> `AssetBrowser.tsx`
    - Status on 2026-03-12: implemented.
  - [BenchmarkControls.jsx](/Users/davidmontgomery/football_pose_workbench/frontend/src/benchmarkLab/BenchmarkControls.jsx) -> `RunControls.tsx`
    - Status on 2026-03-12: implemented.
  - [ClipCard.jsx](/Users/davidmontgomery/football_pose_workbench/frontend/src/benchmarkLab/ClipCard.jsx) -> `OperationalReviewCard.tsx`
    - Status on 2026-03-12: implemented.
- Add typed view models in `frontend/src/benchmarkLab/types.ts` derived from the generated API schema where practical; only add local UI types for view-state-specific shapes.
  - Status on 2026-03-12: implemented.
- Use Tailwind utility classes for all new benchmark layout and styling. Keep existing `styles.css` only for old untouched surfaces or shared theme tokens, not as the primary styling mechanism for new benchmark UI.
  - Status on 2026-03-12: implemented.
- Replace the current leaderboard with a dense matrix:
  - rows: recipes
    - Status on 2026-03-12: implemented.
  - columns: task-appropriate metrics for the active suite or preset
    - Status on 2026-03-12: implemented. The active matrix now focuses one suite at a time and renders that suite's own metric columns.
  - presets: `Detection`, `Spotting`, `Localization`, `Calibration`, `Tracking`, `Game State`, `Operational`
    - Status on 2026-03-12: implemented in the active matrix controls.
- Sorting is mandatory on every metric column.
  - Status on 2026-03-12: implemented in the active matrix controls.
- Filters are mandatory for `suite`, `tier`, `capability`, `provider`, `architecture`, `bundle_mode`, `status`, `supports_active_suite`, and `has_na`.
  - Status on 2026-03-12: implemented in the active matrix controls.
- Default sort is `suite.primary_metric`, descending, except latency columns sort ascending when selected.
  - Status on 2026-03-12: implemented in the active matrix controls.
- The right-hand detail panel shows `metrics`, `runtime`, `artifacts`, `DVC state`, `license`, `source`, and compatibility notes.
  - Status on 2026-03-12: substantially implemented. Metrics, artifacts, composition, compatibility, suite source/license, and benchmark/runtime context are now surfaced; keep enriching this only if new suite-specific context becomes necessary.
- Add a first-class `Matrix Filters` section instead of hiding those controls inside the table.
  - Status on 2026-03-12: implemented via `MatrixToolbar.tsx`, with shared preset/filter/sort state feeding the matrix and charts together.
- Add `Benchmark Charts` with active-suite primary-metric comparison, runtime comparison, metric selector, line/bar toggle, legend spotlighting, and recipe selection.
  - Status on 2026-03-12: implemented in the active shell. Recharts drives the benchmark comparison surface, the line/bar toggle is live, and recipe visibility/selection stay synchronized with the matrix/detail flow. A late same-day regression where a stale saved active suite could keep the matrix on `ops.clip_review_v1` while the preset/selector showed `Tracking` was fixed by preferring the visible-suite pool and resetting detail selection against the visible suite/recipe pool instead of the whole run.
- Add `Suite Evaluation Charts` so the product can compare benchmark suites themselves across the shared recipe pool.
  - Status on 2026-03-12: implemented in the active shell. Comparable recipe count, average primary metric, spread/dispersion, runtime burden, blocked/unavailable rate, materialization coverage, and rank correlation are now charted from persisted suite results plus dataset-state readiness.
- Add help entries for every shipped suite and for `AP@[.50:.95]`, `mAP@1`, `Team-mAP@1`, `mAP-LocSim`, `Completeness x JaC@5`, `HOTA`, `GS-HOTA`, `F1@15%`, and `N/A`.
  - Status on 2026-03-12: substantially implemented. The missing suite entries plus matrix/detail/browser/run-control help entries have been added. Continue checking for any stale old-model benchmark copy outside `help_catalog.json`.
- Add help entries for the chart surface and suite-evaluation concepts: chart mode, recipe spotlighting, primary-metric comparison, runtime burden, metric explorer, metric profile, discriminative power, blocked rate, and rank correlation.
  - Status on 2026-03-12: implemented in `backend/app/help_catalog.json`, and the new chart components now reference those entries directly.

## Validation

- Backend tests: `test_benchmark_suites.py`, `test_benchmark_catalog.py`, `test_benchmark_eval_detection.py`, `test_benchmark_eval_tracking.py`, `test_benchmark_eval_gamestate.py`, `test_benchmark_migration.py`.
  - Status on 2026-03-12: not finished in this exact planned shape. Coverage currently lives in `tests/backend/test_benchmark.py`; more granular suite/evaluator tests still need to be split out or added.
- Frontend tests: write new benchmark UI tests in TS-friendly form or keep them in `.mjs` while targeting the new TS components.
  - `benchmarkLabCharts.test.mjs`
    - Status on 2026-03-12: implemented as `tests/frontend/benchmarkLabCharts.test.mjs`, covering chart metric options, recipe spotlighting, suite-evaluation aggregation, blocked-rate math, and shared-pool rank correlation.
  - `benchmarkLabMatrix.test.mjs`
    - Status on 2026-03-12: implemented as `tests/frontend/benchmarkLabMatrix.test.mjs`.
  - `benchmarkLabFilters.test.mjs`
    - Status on 2026-03-12: implemented as `tests/frontend/benchmarkLabFilters.test.mjs`.
  - `benchmarkLabPresets.test.mjs`
    - Status on 2026-03-12: implemented as `tests/frontend/benchmarkLabPresets.test.mjs`.
  - `benchmarkLabMigration.test.mjs`
    - Status on 2026-03-12: implemented as `tests/frontend/benchmarkLabMigration.test.mjs`.
- Required real runs:
  - `det.roles_quick_v1` on `soccana` and one promoted custom detector
    - Status on 2026-03-12: completed via benchmark `20260312_175409_2a48ad`.
  - `track.sn_tracking_medium_v1` on `soccana+bytetrack+soccana_keypoint` and `soccana+hybrid_reid+soccana_keypoint`
    - Status on 2026-03-12: not finished. Exact blocker: `backend/benchmarks/_datasets/track.sn_tracking_medium_v1/SoccerNetMOT` is not materialized, and the backend still lacks the per-recipe export bridge required by `backend/third_party/soccernet/sn-tracking/tools/evaluate_soccernet_v3_tracking.py`.
  - `gsr.medium_v1` on `pipeline:soccermaster` and `tracker:soccana+hybrid_reid+soccana_keypoint`
    - Status on 2026-03-12: not finished. Exact blocker: `backend/benchmarks/_datasets/gsr.medium_v1/SoccerNetGS` is not materialized for the fixed 12-clip subset, and the backend still lacks the TrackLab/sn-gamestate per-recipe execution bridge required by `backend/third_party/soccernet/sn-gamestate/sn_gamestate/configs/soccernet.yaml`.
  - one `ops.clip_review_v1` run to confirm overlay/diagnostics survive intact
    - Status on 2026-03-12: completed via persisted v2 operational results under `backend/benchmarks/20260312_175156_134e9a/` and the completed multi-suite validation benchmark `backend/benchmarks/20260312_185244_273baa/`. A dedicated Soccermaster blocker verification artifact also exists at `backend/benchmarks/20260312_185955_b062b7/`.
- Required checks:
  - `bash tests/testsuite_full.sh`
    - Status on 2026-03-12: completed after the current rewrite.
  - `cd backend && python -m py_compile app/main.py app/wide_angle.py`
    - Status on 2026-03-12: completed after the current rewrite.
  - `cd frontend && npm run build`
    - Status on 2026-03-12: completed after the current rewrite.
- Browser validation of suite switching, sort/filter, chart mode, and legacy operational record loading
  - Status on 2026-03-12: completed from this workspace via the real Google Chrome app driven with AppleScript after the Playwright MCP Chrome-session conflict blocked the normal MCP path, then re-checked with headless Chrome via `npx -p playwright` for exact control interactions. Verified on `http://127.0.0.1:4317` against backend `:8431`: all nine Benchmark Lab sections render, the line/bar toggle is live, switching the view preset to `Tracking` updates the matrix columns and chart labels to HOTA/DetA/AssA/Frames-s, shared search filters narrow both matrix rows and chart legend series together, blocked medium-suite cells stay visible, and completed detection plus operational benchmark records are visible in the run history.

## Hard Decisions

- Detection scoring uses `pycocotools`, not a workbench-defined score and not Ultralytics output as the scoring authority.
  - Status on 2026-03-12: implemented.
- Official SoccerNet metrics stay separate. `AP`, `mAP@1`, `F1@15%`, `mAP-LocSim`, `Completeness x JaC@5`, `HOTA`, and `GS-HOTA` never collapse into one number.
  - Status on 2026-03-12: implemented in backend model/metric structure; stale docs/help still need cleanup so the operator-facing language matches this decision everywhere.
- Kaggle-only suites such as `SoccerTrack` and `TeamTrack` are not shipped defaults in v1; they become importable external suite definitions after the core suite system lands.
  - Status on 2026-03-12: implemented by omission; these suites are not shipped defaults.
- `spot.team_bas_quick_v1` and `spot.pcbas_medium_v1` ship even if today’s asset catalog has zero compatible recipes; they must show `N/A` honestly until event-capable models are added.
  - Status on 2026-03-12: implemented at the suite/catalog contract level.
- Current clip-derived benchmark records remain in place and reopen as `ops.clip_review_v1`, never as “real benchmark” rows.
  - Status on 2026-03-12: implemented.
- New benchmark frontend code starts in TypeScript now; no new benchmark surface should be added in JS and later migrated.
  - Status on 2026-03-12: implemented.
- New benchmark frontend styling uses Tailwind first; do not add new large benchmark-specific CSS blocks to `styles.css` unless they are unavoidable shared theme primitives.
  - Status on 2026-03-12: implemented.

## Immediate Finish Queue (2026-03-12)

- Finish the frontend sort/filter/preset surface without adding a second benchmark mode.
  - Concrete close-out criteria:
    - preset picker rendered in the Benchmark Lab shell
    - filter controls rendered and persisted
    - metric sorting active in the matrix
    - `supports_active_suite` and `has_na` filters behave honestly
  - Status on 2026-03-12: finished for the current shell.
- Finish the chart-rich benchmark surface without detaching it from the matrix.
  - Concrete close-out criteria:
    - `Benchmark Charts` rendered in the main Benchmark Lab flow
    - active-suite metric selector plus line/bar toggle live
    - chart legend spotlight synchronized with the matrix recipe pool
    - `Suite Evaluation Charts` rendered with suite-level comparison metrics
  - Status on 2026-03-12: finished for the current shell.
- Remove dead transition files such as `frontend/src/benchmarkLab/RecipeDetailPanel.tsx`.
  - Concrete close-out criteria:
    - file removed or explicitly wired as the only detail panel
    - no dead benchmark transition file remains in `frontend/src/benchmarkLab/`
  - Status on 2026-03-12: finished for the currently known transition files.
- Rewrite stale Benchmark Lab help entries and component help references so nothing in the UI still explains the old composite leaderboard.
  - Concrete close-out criteria:
    - `benchmark.leaderboard` and `benchmark.composite_score` are no longer the active explanatory target for the new matrix UI
    - every shipped suite has a help entry
    - active metric/tooltips align with the suite/recipe matrix model
  - Status on 2026-03-12: finished for the active benchmark surface.
- Rewrite stale README and docs Benchmark Lab sections so the product description matches the implemented suite/recipe matrix.
  - Concrete close-out criteria:
    - README Benchmark Lab summary and quick-pass steps no longer describe a locked-clip detector leaderboard
    - `docs/workflows.md` and `docs/outputs-and-api.md` match the v2 API and artifact layout
  - Status on 2026-03-12: finished for the checked-in docs.
- Materialize actual DVC tracking artifacts for benchmark manifests/datasets/conversions, or document the exact local blocker if the DVC CLI remains unavailable.
  - Concrete close-out criteria:
    - benchmark manifests, dataset roots, and conversion roots are represented by real `.dvc` tracking artifacts or explicit repo-checked instructions for non-git-safe large materials
    - benchmark config and detail views surface the tracked/untracked state honestly
  - Status on 2026-03-12: finished for the current benchmark roots.
- Vendor or otherwise materialize the pinned SoccerNet/FOOTPASS/TrackLab evaluator sources under `backend/third_party/soccernet/` so non-detection suites are not just wrappers around missing directories.
  - Concrete close-out criteria:
    - vendored source trees exist locally under `backend/third_party/soccernet/`
    - adapter entrypoints are verified against those sources instead of guessed
    - missing-suite failures, if any remain, are dataset/runtime blockers rather than absent code blockers
  - Status on 2026-03-12: source materialization is finished, and the adapter-entrypoint truth is now pinned for calibration, team-spotting, and FOOTPASS via repo-owned JSON wrappers. Deeper medium-suite execution still depends on dataset and prediction-export completion.
- Execute at least the required real benchmark smoke runs and persist v2 `suite_results` artifacts.
  - Concrete close-out criteria:
    - at least one real quick detection run persisted in the new v2 folder shape
    - at least one new-format operational run persisted
    - any heavier suite not yet run has an explicit blocker recorded here instead of being silently skipped
  - Status on 2026-03-12: partially finished. The quick detection and operational smoke runs now exist in the v2 folder shape; heavier tracking/GSR suite runs remain blocked for explicitly recorded reasons instead of being silently skipped.

## Completion Plan To Finish The Original Plan (2026-03-12 reset)

- This section is the endgame plan for finishing the original Benchmark Lab V2 promise instead of stopping at shell completeness.
- Rule:
  - `mechanical pass` means scripted interactions and layout invariants worked.
  - `human-usable` means the page is understandable and readable to a human operator.
  - Do not call the original plan finished on mechanical checks alone.

### Stage 0: Lock the benchmark product bar

- Goal: stop drifting between “UI shell exists” and “original plan is finished.”
- Exit criteria:
  - Benchmark Lab UX is human-usable at normal desktop width, not just mechanically interactive.
  - left rail is compact and navigable
  - raw paths are secondary disclosures, not primary content
  - plan, docs, and memory all use the same completion language
- Current status on 2026-03-12:
  - substantial UX cleanup has landed in the salvage worktree
  - remaining bar is human signoff, not more blind automation

### Stage 1: Finish the evaluator runtime substrate

- Goal: make the remaining non-detection suites executable in principle before tackling suite-specific adapters one by one.
- Tasks:
  - add isolated runner strategy for external evaluator families that do not match the main backend env
    - `sn-calibration`
    - `sn-teamspotting` / `sn-spotting`
    - `FOOTPASS`
    - `TrackLab` / `sn-gamestate`
  - pin and document each evaluator runtime in a reproducible way
  - ensure every adapter writes the same v2 result/artifact/provenance shape as detection and operational runs
- Critical blocker this stage must resolve:
  - `sn-gamestate` currently wants Python `<3.10` and `numpy<2`, while the repo runtime is Python `3.12` with NumPy `2.1.1`
- Status on 2026-03-12:
  - the initial substrate slice has landed in the salvage worktree:
    - `backend/app/benchmark_eval/runtime_profiles.py`
    - env-aware `external_cli.py`
    - protocol runtime mapping in `benchmark_eval/__init__.py`
    - external adapters now declare runtime keys
    - `gamestate.py` now probes the target runtime profile instead of blaming the current backend process
  - the isolated evaluator envs are now materialized on disk:
    - `backend/.venv-benchmark-gamestate-py39`
    - `backend/.venv-benchmark-calibration-legacy`
    - `backend/.venv-benchmark-action-spotting`
    - `backend/.venv-benchmark-footpass`
  - runtime probes are now green for the intended evaluator surfaces:
    - `tracklab_gamestate_py39_np1` -> Python `3.9.20`, NumPy `1.26.4`, `tracklab` + `sn_gamestate`
    - `sn_calibration_legacy` -> Python `3.9.20`, NumPy `1.21.6`
    - `modern_action_spotting` -> Python `3.11.7`, NumPy `2.1.2`
    - `footpass_eval` -> Python `3.11.7`, NumPy `1.26.4`, `h5py`
  - repo-owned JSON wrappers now replace the fake entrypoint guesses:
    - `run_calibration_eval.py`
    - `run_team_spotting_eval.py`
    - `run_footpass_eval.py`
  - Status update on 2026-03-13:
    - `backend/app/benchmark_eval/prediction_exports.py` now lands the Stage 1 export-bridge slice:
      - calibration: generate `artifacts/predictions/valid/camera_<frame_id>.json`
      - team spotting: generate `artifacts/predictions/<game>/results_spotting.json` from repo-owned raw event JSON
      - FOOTPASS: generate `artifacts/predictions.json` from repo-owned raw play-by-play JSON
    - the benchmark runner now prepares those bridge artifacts before evaluator dispatch, and the protocol adapters also use the same bridge helpers when called directly
    - the broader FOOTPASS baseline-side `decord` limitation is now kept explicitly separate from the evaluator path; it still blocks baseline/helper provisioning, but not `evaluation.py`
  - remaining work in this stage:
    - keep Stage 1 focused on honest runtime + export-contract truth; do not let Stage 3 recipe or dataset gaps get relabeled as runtime gaps
- Exit criteria:
  - every remaining adapter can be invoked against its real evaluator entrypoint in some honest runtime, even if the suite dataset is still missing
  - Status on 2026-03-13: met for calibration, team-spotting, and FOOTPASS. Tracking and gamestate still need their separate Stage 3 recipe bridges.

### Stage 2: Materialize every suite dataset and manifest honestly

- Goal: remove “missing dataset root” as the generic blocker and replace it with real runnable materials.
- Stage 2 counts as successful for a suite only when one of these is true:
  - the suite has a real benchmark dataset root with evaluator-relevant files that are actually readable on this machine
  - or the suite has a manifest-backed blocker that names the exact expected path, exact expected files, exact source artifact attempted, and exact failure reason
- Stage 2 status vocabulary is closed:
  - `materialized`: readable evaluator-relevant dataset files are present on disk
  - `partially_materialized`: some evaluator-relevant artifacts are present, but the suite is still blocked by named missing artifacts
  - `blocked_with_exact_evidence`: no usable dataset root yet, but the manifest names exact path/files/source/failure
  - `not_started`: forbidden once a suite is in the Stage 2 target list
- Stage 2 does **not** include:
  - adding event-capable recipes for `spot.team_bas_quick_v1` or `spot.pcbas_medium_v1`
  - implementing `TRACKERS_FOLDER_ZIP` for `track.sn_tracking_medium_v1`
  - implementing the TrackLab/Hydra recipe bridge for `gsr.medium_v1` or `gsr.long_v1`
  - relabeling runtime or recipe gaps as dataset work
- Tasks:
  - materialize and DVC-pin:
    - `det.ball_quick_v1`
    - `loc.synloc_quick_v1`
    - `spot.team_bas_quick_v1`
    - `calib.sn_calib_medium_v1`
    - `track.sn_tracking_medium_v1`
    - `spot.pcbas_medium_v1`
    - `gsr.medium_v1`
    - `gsr.long_v1`
  - replace placeholder `gsr.medium_v1` clip ids with the real locked 12-clip validation subset
  - ensure every dataset-state note in the API names the concrete root, split, and missing material when blocked
- Current status on 2026-03-13:
  - `spot.pcbas_medium_v1` now has a real local GT file in the benchmark root and a manifest-backed blocker naming the gated missing `tactical_data_VAL.zip` / video archives instead of a generic missing root.
  - `spot.team_bas_quick_v1` now has a real local archive download plus a manifest-backed blocker naming the unusable extracted validation members instead of a generic missing root.
  - `calib.sn_calib_medium_v1`, `track.sn_tracking_medium_v1`, `gsr.medium_v1`, and `gsr.long_v1` now have manifest-backed dataset blockers that name the exact expected roots/materials instead of vague “not ready” prose.
  - `gsr.medium_v1` is still not lockable because the manifest item ids remain placeholders and the only exposed validation archive is too large for the current free disk.
- Per-suite Stage 2 status on 2026-03-13:
  - `det.ball_quick_v1`: `blocked_with_exact_evidence`. Manifest now names the exact expected YOLO dataset contract and the absence of any locally staged `football-ball-detection` source tree.
  - `loc.synloc_quick_v1`: `blocked_with_exact_evidence`. Manifest now names the exact expected suite root plus the missing vendored `sskit` checkout.
  - `spot.team_bas_quick_v1`: `partially_materialized`. Official `valid.zip` is downloaded, but extracted validation members are unusable here, so the suite remains dataset-blocked with an exact source/failure record.
  - `calib.sn_calib_medium_v1`: `blocked_with_exact_evidence`. Manifest now names the exact expected `calibration-2023/valid` layout and local absence.
  - `track.sn_tracking_medium_v1`: `blocked_with_exact_evidence`. Manifest now names the exact missing `SoccerNetMOT` tree plus `gt.zip` and `sample_submission.zip`.
  - `spot.pcbas_medium_v1`: `partially_materialized`. Local evaluator GT is staged; official validation tactical/video archives remain gated and missing.
  - `gsr.medium_v1`: `blocked_with_exact_evidence`. Manifest now names the exact `valid.zip` source and the disk-space blocker; 12-clip ids are still placeholders.
  - `gsr.long_v1`: `blocked_with_exact_evidence`. Manifest now names the exact `valid.zip` source and the disk-space blocker.
- Exit criteria:
  - benchmark config can truthfully distinguish runnable suites from adapter/runtime blockers instead of lumping them into “dataset missing”

### Stage 3: Finish suite-specific execution bridges in order of original risk

- Goal: make every shipped suite in the original v1 list honestly runnable.
- Priority order:
  - `det.ball_quick_v1`
    - lowest-risk completion because it reuses the real COCO evaluator path
  - `calib.sn_calib_medium_v1`
  - `loc.synloc_quick_v1`
  - `spot.team_bas_quick_v1`
  - `spot.pcbas_medium_v1`
  - `track.sn_tracking_medium_v1`
    - implement the per-recipe `TRACKERS_FOLDER_ZIP` export bridge expected by `sn-tracking`
  - `gsr.medium_v1`
    - implement the recipe-to-TrackLab / Hydra bridge for the fixed 12-clip subset
  - `gsr.long_v1`
    - same bridge, full validation scale
- Exit criteria:
  - each suite produces a real `suite_results/<suite_id>/<recipe_id>/result.json` backed by the intended evaluator family, not a scaffold or guessed metric path

### Stage 4: Satisfy the original required real-run matrix

- Goal: close the exact validation promises already written into this plan.
- Required runs still missing from the original plan:
  - `track.sn_tracking_medium_v1` on:
    - `tracker:soccana+bytetrack+soccana_keypoint`
    - `tracker:soccana+hybrid_reid+soccana_keypoint`
  - `gsr.medium_v1` on:
    - `pipeline:soccermaster`
    - `tracker:soccana+hybrid_reid+soccana_keypoint`
- Additional completion runs needed so the shipped suite list is honest:
  - one real `det.ball_quick_v1` run
  - one real `calib.sn_calib_medium_v1` run
  - one real `loc.synloc_quick_v1` run
  - one real `spot.team_bas_quick_v1` or explicit evaluator blocker artifact if that baseline still cannot execute
  - one real `spot.pcbas_medium_v1` or explicit evaluator blocker artifact if that baseline still cannot execute
  - Status update on 2026-03-13:
    - `spot.pcbas_medium_v1`: evaluator path now has a real smoke artifact from the vendored sample validation files; keep the suite-level benchmark validation requirement open until a benchmark cell runs from a real compatible recipe
    - `calib.sn_calib_medium_v1`: real adapter smoke completed using a one-frame dataset built from the local benchmark clip plus evaluator-valid line annotations derived from the predicted camera
    - `spot.team_bas_quick_v1`: real adapter smoke completed using a minimal evaluator-valid two-game test split plus repo-owned raw event predictions
    - suite-level blocker verification now exists through the real benchmark runner for:
      - `spot.team_bas_quick_v1`: blocked on the downloaded-but-unusable validation archive members instead of a generic missing root
      - `spot.pcbas_medium_v1`: blocked on the local GT + gated upstream tactical/video archives instead of a generic missing root
- Exit criteria:
  - every suite in the original shipped list has either:
    - a real persisted benchmark artifact, or
    - an explicit artifact-backed blocker that names the exact dataset/runtime/evaluator reason it still cannot run

### Stage 5: Finish docs/help and human review as the last gate

- Goal: align operator language with the actual shipped system and stop equating automation with approval.
- Tasks:
  - re-review README, workflow docs, and help text once all suites above have truthful status
  - add explicit operator-facing wording for “mechanical pass” versus “human-usable”
  - perform a final visible browser review with a human bar:
    - left rail understandable
    - no path spam
    - suite intent obvious
    - matrix/detail/review flow understandable without explanation
- Final finish condition for the original plan:
  - all original shipped suites are either runnable or explicitly artifact-blocked
  - required real runs exist
  - docs/help match reality
  - the user can look at Benchmark Lab without apologizing for it
