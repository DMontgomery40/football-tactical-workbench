# Football Tactical Workbench

Browser-first football analysis and detector-fine-tuning workbench built with React and FastAPI.

The app now has two top-level product surfaces:

- `Analysis Workspace`
  - load a local clip or upload a source
  - stream `/api/live-preview`
  - run `/api/analyze`
  - review saved overlay, diagnostics, metrics, and exports
  - browse SoccerNet and scan local folders
- `Training Studio`
  - scan a YOLO detector dataset on local disk
  - fine-tune from football-pretrained `soccana`
  - monitor training jobs, logs, and artifacts
  - register a finished checkpoint and optionally promote it into analysis defaults

This README stays intentionally high-level. The detailed setup, workflow, and API contracts live in the linked docs below.

## Screenshots

![Workbench dashboard](docs/screenshots/workbench-dashboard.png)

![Workbench full page](docs/screenshots/workbench-fullpage.png)

![Workbench run review](docs/screenshots/workbench-run-review.png)

## Current Pipeline

The active analysis/runtime path is:

- `soccana` player / referee / ball detection
- hybrid appearance-aware player tracking plus tracklet stitching
- jersey-colour home/away separation
- `soccana_keypoint` field registration
- automatic pitch-calibration refresh every 10 frames
- live browser preview plus saved overlay output

The current training V1 is deliberately narrower:

- detector fine-tuning only
- base weights default to `soccana`
- local YOLO-style datasets only
- on-disk JSON manifests, local run folders, and optional DVC-backed durable pointers
- no cloud training
- no database
- keypoint / ReID / team-classifier training are future families, not part of V1

## Current Defaults

- player detector: `soccana`
- ball detector: shared `soccana` detector
- field calibration model: `soccana_keypoint`
- player tracker: `hybrid_reid`
- ball tracker: `bytetrack.yaml`
- calibration refresh cadence: every 10 frames
- detector training base weight: `soccana`
- training runtime today: Ultralytics + PyTorch, Mac-first on Apple Silicon with MPS preferred when available
- planned inference runway: ONNX Runtime with CoreML on Apple Silicon and ONNX Runtime with CUDA on GPU hosts
- frontend dev server: `0.0.0.0:4317`
- backend API server: `0.0.0.0:8431`
- analysis runs: `backend/runs/`
- training runs: `backend/training_runs/`
- detector registry: `backend/models/registry.json`
- model cache: `backend/models/`

## Quick Start

1. Install backend dependencies.

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

2. Install frontend dependencies.

```bash
cd frontend
npm install
```

3. Start both halves together.

```bash
./run_all.sh
```

Or start them separately:

```bash
cd backend
./run_backend.sh
```

```bash
cd frontend
./run_frontend.sh
```

For the full setup guide, environment variables, training notes, and first validation passes, use [Getting Started](docs/getting-started.md).

## First Run Paths

### Analysis quick pass

1. Open `http://127.0.0.1:4317`.
2. In `Analysis Workspace`, paste a local video path or upload a clip.
3. Click `Load input clip`.
4. Optionally open `Live Preview`.
5. Click `Analyze loaded clip`.
6. Inspect the overlay, diagnostics, and files in `Run Review`.

### Training quick pass

1. Switch to `Training Studio`.
2. In `Datasets`, scan a local YOLO detector dataset.
3. In `Train`, fine-tune from `soccana`.
4. In `Jobs`, watch progress and logs.
5. In `Registry`, inspect the completed checkpoint and activate it when you want analysis to use it while the analysis selector stays on `soccana`.
6. If you care about durable artifact lineage, use the DVC/provenance details surfaced in Training Studio and the deeper notes in [Workflows](docs/workflows.md) and [Outputs, API, and Batch Experiments](docs/outputs-and-api.md).

The full UI walk-through lives in [Workflows](docs/workflows.md).

## Runtime Notes

- Backend startup prewarms the default detector and field-calibration models from the local cache or model source.
- On Apple Silicon, detector inference prefers MPS while field calibration still falls back to CPU.
- Training jobs run in isolated Python subprocesses and persist their own `config.json`, `dataset_scan.json`, `dataset_runtime.yaml`, `train.log`, `summary.json`, and artifacts inside `backend/training_runs/<run_id>/`.
- Terminal training runs also persist an AI-or-heuristic `training_analysis_ai.json` artifact so Training Studio can reopen the multi-section run review without scraping raw logs.
- Completed detector runs are added to the local detector registry, and an activated checkpoint is copied into `backend/models/promoted/custom_<run_id>/` with `training_provenance.json` before it becomes the active detector for analysis whenever the analysis selector remains on `soccana`.
- The app stores analysis sources, analysis job snapshots, and training job snapshots on disk so restarts do not silently wipe state.
- AI diagnostics remain a real post-run path. If a provider is configured, completed analysis runs get AI-curated run briefs and diagnostics; otherwise they fall back to heuristic diagnostics.
- Warn-level diagnostics include a collapsed code drilldown in review with the likely failing function and a concrete code change to try next.

## Documentation

- [Getting Started](docs/getting-started.md)
  Setup, environment variables, launch commands, model/runtime notes, first validation, and training prerequisites.
- [Workflows](docs/workflows.md)
  Analysis Workspace and Training Studio UI flows, persistence rules, and operator-facing behavior.
- [Outputs, API, and Batch Experiments](docs/outputs-and-api.md)
  Analysis outputs, training outputs, FastAPI endpoints, and batch tooling.
- [AGENTS.md](AGENTS.md)
  Repo-local implementation guardrails for future coding agents and contributors working in this tree.

## Repository Map

- `backend/app/main.py`
  FastAPI entrypoint for analysis, training, source loading, SoccerNet, live preview, and persisted run loading.
- `backend/app/wide_angle.py`
  Active football analysis pipeline, detector resolution, field calibration, tracking, live preview generation, and overlay export.
- `backend/app/reid_tracker.py`
  Sparse appearance embedding extraction, field-aware player association, and post-pass tracklet stitching.
- `backend/app/ai_diagnostics.py`
  Provider selection, prompt construction, OpenAI-compatible / Anthropic calls, and diagnostics artifact writing.
- `backend/app/training.py`
  Training Studio dataset scan, runtime dataset-manifest generation, backend metadata, and artifact helpers.
- `backend/app/training_manager.py`
  Detector training job persistence, log streaming, progress ingestion, and training summary writing.
- `backend/app/training_registry.py`
  Local detector registry and active-detector resolution for analysis defaults.
- `backend/app/train_worker.py`
  Detector fine-tuning worker subprocess entrypoint.
- `frontend/src/App.jsx`
  Top-level app shell, Analysis Workspace, persisted state, SoccerNet UI, and review flows.
- `frontend/src/TrainingStudio.jsx`
  Dedicated training shell for dataset scan, training form, jobs, and registry.
- `frontend/src/styles.css`
  Full shared styling for analysis and training surfaces.

## SoccerNet

The app still has first-class SoccerNet support:

- browse `train`, `valid`, `test`, and `challenge`
- search official game paths reactively from the UI
- download halves and labels into `backend/datasets/soccernet/`
- scan that folder directly from the UI
- prefer `Labels-v2.json` for goal-aligned experiment work

## Acknowledgements

- [SoccerNet](https://www.soccer-net.org/) for match data structure, labels, and downloader tooling
- [Soccana](https://huggingface.co/Adit-jain/soccana) for football-specific detector weights
- [Soccana Keypoint](https://huggingface.co/Adit-jain/Soccana_Keypoint) for pitch keypoint weights
- [Ultralytics](https://www.ultralytics.com/) for the current YOLO runtime used by detector and training paths
- ByteTrack via Ultralytics tracker integration for the current ball-tracking path and the explicit player-tracking comparison path
- Torchvision ResNet-18 weights for sparse appearance embeddings in the player ReID path

## Runtime Note

This repository is MIT, but the current detector and detector-training runtime still depends on the local Ultralytics stack, which Ultralytics distributes under AGPL-3.0 by default unless you have a separate enterprise license. The longer-term inference direction remains ONNX Runtime so the app can stay Mac-first on Apple Silicon while keeping a clean path to CUDA hosts.

## License

MIT. See [LICENSE](LICENSE).
