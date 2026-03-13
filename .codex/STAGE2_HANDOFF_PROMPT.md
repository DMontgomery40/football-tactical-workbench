# Benchmark Lab V2 Stage 2 Handoff Prompt

Use this as the exact prompt/context handoff for the next agent.

## Mission

You are continuing Benchmark Lab V2 in:

- repo: `/Users/davidmontgomery/football_pose_workbench_salvage`
- branch: `feat/benchmark-v2-salvage`

You are **not** starting from scratch.

Your job is to continue from the current real state and push **Stage 2: dataset/materialization** as far as you honestly can in one uninterrupted implementation pass. Do not stop at analysis. Do not stop at partial notes. Do not stop after adding scaffolding. Carry work through code changes, real dataset/materialization progress, validation, plan/memory updates, and an honest closeout.

## Read First, In This Order

1. `/Users/davidmontgomery/football_pose_workbench_salvage/AGENTS.md`
2. `/Users/davidmontgomery/football_pose_workbench_salvage/.codex/MEMORY.md`
3. `/Users/davidmontgomery/.codex/projects/-Users-davidmontgomery-football_pose_workbench/memory/benchmark-lab-v2-source-of-truth.md`
4. `/Users/davidmontgomery/.codex/projects/-Users-davidmontgomery-football_pose_workbench/memory/benchmark-lab-v2-evaluator-runtime-strategy.md`
5. `/Users/davidmontgomery/football_pose_workbench_salvage/.codex/benchmark_PLAN.md`

## Non-Negotiable Truth

The original plan is **not done**.

The current real state is:

- Stage 1 runtime substrate exists.
- Stage 1 export bridges for calibration, team-spotting, and FOOTPASS now exist.
- Real adapter smokes were completed for:
  - `calib.sn_calib_medium_v1`
  - `spot.team_bas_quick_v1`
  - `spot.pcbas_medium_v1`
- This does **not** mean the full Benchmark Lab plan is complete.
- The next major unfinished area is **Stage 2: suite dataset/materialization**.

Do not relabel Stage 2/3/4 work as “done” because Stage 1 is now real.

## What Landed In Stage 1

### Export/runtime bridge files

- `/Users/davidmontgomery/football_pose_workbench_salvage/backend/app/benchmark_eval/prediction_exports.py`
- `/Users/davidmontgomery/football_pose_workbench_salvage/backend/app/benchmark_eval/calibration.py`
- `/Users/davidmontgomery/football_pose_workbench_salvage/backend/app/benchmark_eval/team_spotting.py`
- `/Users/davidmontgomery/football_pose_workbench_salvage/backend/app/benchmark_eval/pcbas.py`
- `/Users/davidmontgomery/football_pose_workbench_salvage/backend/app/benchmark_eval/run_calibration_eval.py`
- `/Users/davidmontgomery/football_pose_workbench_salvage/backend/app/benchmark_eval/run_team_spotting_eval.py`
- `/Users/davidmontgomery/football_pose_workbench_salvage/backend/app/benchmark_eval/run_footpass_eval.py`
- `/Users/davidmontgomery/football_pose_workbench_salvage/backend/app/benchmark.py`

### Relevant tests

- `/Users/davidmontgomery/football_pose_workbench_salvage/tests/backend/test_benchmark.py`
- `/Users/davidmontgomery/football_pose_workbench_salvage/tests/backend/test_benchmark_eval_external_adapters.py`
- `/Users/davidmontgomery/football_pose_workbench_salvage/tests/backend/test_benchmark_prediction_exports.py`

### Real Stage 1 smoke results already observed

- Calibration smoke:
  - source clip frame extracted from:
    - `/Users/davidmontgomery/football_pose_workbench/backend/benchmarks/_clip_cache/benchmark_clip.mp4`
  - adapter path:
    - `/Users/davidmontgomery/football_pose_workbench_salvage/backend/app/benchmark_eval/calibration.py`
  - observed metrics:
    - `completeness_x_jac_5 = 1.0`
    - `completeness = 1.0`
    - `jac_5 = 1.0`
    - `frames_per_second = 0.05703489036590283`

- Team spotting smoke:
  - adapter path:
    - `/Users/davidmontgomery/football_pose_workbench_salvage/backend/app/benchmark_eval/team_spotting.py`
  - observed metrics on a tiny evaluator-valid synthetic two-game split:
    - `team_map_at_1 = null`
    - `map_at_1 = 0.08333333333333333`
    - `clips_per_second = 4.062287385843565`

- FOOTPASS smoke:
  - adapter path:
    - `/Users/davidmontgomery/football_pose_workbench_salvage/backend/app/benchmark_eval/pcbas.py`
  - observed metrics:
    - `f1_at_15 = 0.40995387750724016`
    - `precision_at_15 = 0.303912213740458`
    - `recall_at_15 = 0.6296540362438221`
    - `clips_per_second = 218.84930905256635`

## Important Real Bug Already Found And Fixed

There was a real calibration bug in:

- `/Users/davidmontgomery/football_pose_workbench_salvage/backend/app/benchmark_eval/prediction_exports.py`

The vendored `sn-calibration` camera module was initially imported from the wrong path during real smoke work. The fix was to import `src.camera` from the vendored repo root, not bare `camera`.

Do not regress that.

## What Stage 2 Means Here

Stage 2 is **dataset/materialization**, not more runtime work.

Your job is to remove “missing dataset root/materialization” as the generic blocker for as many shipped suites as you can, with emphasis on the unfinished medium suites.

For the avoidance of doubt, a suite counts as “advanced in Stage 2” only if one of these is true:

1. The suite has a real benchmark dataset root with evaluator-relevant files that are actually readable on this machine.
2. The suite is still blocked, but the blocker is manifest-backed and names:
   - exact expected path
   - exact expected files
   - exact source artifact attempted
   - exact reason it failed here

Stage 2 does **not** include:

- adding event-capable recipe rows
- adding `TRACKERS_FOLDER_ZIP`
- adding the TrackLab/Hydra recipe bridge
- reworking runtime wrappers again
- calling a suite “Stage 2 done” just because the evaluator runtime exists

Allowed Stage 2 suite states are only:

- `materialized`
- `partially_materialized`
- `blocked_with_exact_evidence`

`not_started`, `kind of ready`, `almost there`, `missing dataset`, or any other softer label is not acceptable for a Stage 2 target suite.

From the plan, the Stage 2 target list is:

- `det.ball_quick_v1`
- `loc.synloc_quick_v1`
- `spot.team_bas_quick_v1`
- `calib.sn_calib_medium_v1`
- `track.sn_tracking_medium_v1`
- `spot.pcbas_medium_v1`
- `gsr.medium_v1`
- `gsr.long_v1`

In practice, the highest-value Stage 2 work now is:

1. Materialize real dataset roots and manifests for:
   - `calib.sn_calib_medium_v1`
   - `spot.team_bas_quick_v1`
   - `spot.pcbas_medium_v1`
   - `track.sn_tracking_medium_v1`
   - `gsr.medium_v1`
2. Make dataset-state API notes/blockers point at exact concrete missing material.
3. Keep DVC truth honest.
4. Do **not** fake dataset readiness.

## Current Per-Suite Stage 2 Truth

- `det.ball_quick_v1`
  - `blocked_with_exact_evidence`
  - exact expected YOLO dataset contract is now named in the suite manifest
  - no locally staged `football-ball-detection` source tree exists yet
- `loc.synloc_quick_v1`
  - `blocked_with_exact_evidence`
  - exact expected suite root is named in the suite manifest
  - no local SynLoc archive/conversion root exists and no vendored `sskit` checkout exists
- `spot.team_bas_quick_v1`
  - `partially_materialized`
  - official `SN-BAS-2025` `valid.zip` is downloaded locally
  - extracted validation members are unusable on this machine, so the suite stays dataset-blocked
  - this is not Stage 2 completion
- `calib.sn_calib_medium_v1`
  - `blocked_with_exact_evidence`
  - manifest now names the exact expected `calibration-2023/valid` layout
  - no official validation tree is materialized locally yet
- `track.sn_tracking_medium_v1`
  - `blocked_with_exact_evidence`
  - manifest now names the exact missing `SoccerNetMOT` tree plus `gt.zip` and `sample_submission.zip`
  - Stage 3 still owns `TRACKERS_FOLDER_ZIP`
- `spot.pcbas_medium_v1`
  - `partially_materialized`
  - local evaluator GT is staged under the benchmark root
  - official tactical/video archives are still gated and missing
  - this is not recipe readiness
- `gsr.medium_v1`
  - `blocked_with_exact_evidence`
  - manifest now names the exact `valid.zip` source archive and the current disk-space blocker
  - the 12-clip ids are still placeholders, so the subset is not durably locked yet
- `gsr.long_v1`
  - `blocked_with_exact_evidence`
  - manifest now names the exact `valid.zip` source archive and the current disk-space blocker
  - Stage 3 still owns the TrackLab/Hydra recipe bridge

## What Not To Waste Time On First

Do **not** spend the next turn primarily on:

- frontend polish
- new UI affordances
- help catalog cleanup
- more wrapper/runtime refactors
- “future-proof” abstractions
- generic README cleanup
- inventing event-capable recipe rows without real backing assets/pipeline outputs

The current bottleneck is dataset/materialization and manifest truth.

## Concrete Stage 2 Starting Points

Inspect these first:

- `/Users/davidmontgomery/football_pose_workbench_salvage/backend/app/benchmark_suites.py`
- `/Users/davidmontgomery/football_pose_workbench_salvage/backend/app/benchmark_suites.json`
- `/Users/davidmontgomery/football_pose_workbench_salvage/backend/scripts/pull_hf_football_datasets.py`
- `/Users/davidmontgomery/football_pose_workbench_salvage/backend/benchmarks/_datasets`
- `/Users/davidmontgomery/football_pose_workbench_salvage/backend/benchmarks/_manifests`
- `/Users/davidmontgomery/football_pose_workbench_salvage/backend/benchmarks/_conversions`
- `/Users/davidmontgomery/football_pose_workbench_salvage/backend/third_party/soccernet/`

Especially inspect current dataset-state logic for:

- exact expected roots
- fallback roots
- manifest summaries
- blocker text

## Specific Things That Are Still True

- `spot.team_bas_quick_v1` still has no shipped `event_spotting` recipe in the catalog. That is a Stage 3/recipe problem. Stage 2 should still materialize its dataset honestly.
- `spot.pcbas_medium_v1` still has no shipped compatible recipe row. Same note.
- `track.sn_tracking_medium_v1` still lacks the `TRACKERS_FOLDER_ZIP` export bridge. That is a Stage 3 bridge problem, not a Stage 2 runtime problem.
- `gsr.medium_v1` and `gsr.long_v1` still lack the TrackLab/Hydra recipe bridge. Again Stage 3.
- The broader FOOTPASS helper stack still hits `decord==0.6.0` macOS arm64 wheel limits. Do not confuse that with evaluator runtime readiness.

## How To Behave

- Do the work, don’t narrate completion early.
- If you can materialize real suite datasets/manifests, do it.
- If a dataset cannot be materialized locally, produce an exact artifact-backed blocker naming:
  - expected path
  - expected files
  - what source was attempted
  - why it failed
- No fake fallback modes.
- No old-mode escape hatches.
- No vague “not ready” prose.

## Validation You Must Run Before Stopping

Minimum:

1. `cd backend && .venv/bin/python -m py_compile app/benchmark_eval/*.py`
2. `cd backend && .venv/bin/python -m pytest ../tests/backend/test_benchmark.py ../tests/backend/test_benchmark_eval_external_adapters.py -q`
3. `bash tests/testsuite_full.sh`

If Stage 2 changes dataset/materialization logic, also run the narrowest relevant dataset-state / benchmark tests you add.

If you materialize a real suite dataset, run the most relevant real evaluation or blocker verification you can from the full benchmark path, not just the direct adapter.

## Files You Must Update Before Stopping

- `/Users/davidmontgomery/football_pose_workbench_salvage/.codex/benchmark_PLAN.md`
- `/Users/davidmontgomery/football_pose_workbench_salvage/.codex/MEMORY.md`
- `/Users/davidmontgomery/.codex/projects/-Users-davidmontgomery-football_pose_workbench/memory/benchmark-lab-v2-evaluator-runtime-strategy.md`

If Stage 2 teaches a new durable lesson that does not fit that file, create a new canonical memory note under:

- `/Users/davidmontgomery/.codex/projects/-Users-davidmontgomery-football_pose_workbench/memory/`

and add it to:

- `/Users/davidmontgomery/.codex/projects/-Users-davidmontgomery-football_pose_workbench/MEMORY.md`

## Exact Current Validation State

At the time of this handoff, these passed:

- `cd backend && .venv/bin/python -m py_compile app/benchmark_eval/*.py`
- `cd backend && .venv/bin/python -m pytest ../tests/backend/test_benchmark.py ../tests/backend/test_benchmark_eval_external_adapters.py -q`
- `bash tests/testsuite_full.sh`

Do not assume they still pass after your changes. Re-run them.

## Final Reminder

The plan’s definition of done is still the one in:

- `/Users/davidmontgomery/football_pose_workbench_salvage/.codex/benchmark_PLAN.md`

Do not confuse “Stage 1 is now real” with “Benchmark Lab V2 is done.”

Your immediate goal is:

- push Stage 2 dataset/materialization truth as far as possible
- keep blockers concrete
- leave the repo in a state where the next missing work is unmistakably Stage 3+, not hidden under fake Stage 2 ambiguity
