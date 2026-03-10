# Getting Started

This guide covers local setup, launch commands, runtime notes, and the quickest first-pass validation for both `Analysis Workspace` and `Training Studio`.

## Prerequisites

- macOS on Apple Silicon is the primary local target today
- Python 3 with `venv`
- Node.js and `npm`
- `ffmpeg` on your `PATH` if you want browser-friendly H.264 overlay exports
- network access on first model run so the backend can download weights if they are not already cached

Other environments can still work, but the code and runtime profile are currently tuned around a Mac-first local workflow.

## Backend Setup

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The current backend requirements include:

- `fastapi`
- `uvicorn[standard]`
- `python-multipart`
- `numpy`
- `opencv-python`
- `ultralytics`
- `huggingface_hub`
- `SoccerNet`

## Frontend Setup

```bash
cd frontend
npm install
```

The frontend is a Vite + React SPA. It currently uses:

- `react`
- `react-dom`
- `@floating-ui/react`

## Environment Variables

The backend loads `.env` and `.env.local` from the repository root if they exist.

You do not need any environment variables for local analysis or local detector fine-tuning.

### AI diagnostics

The code supports four diagnostics providers:

- OpenAI
- OpenRouter
- Anthropic
- local OpenAI-compatible endpoint

Supported variables in code:

| Variable | Purpose |
| --- | --- |
| `AI_DIAGNOSTICS_PROVIDER` | `auto`, `openai`, `openrouter`, `anthropic`, `local`, or `off` |
| `AI_DIAGNOSTICS_ORCHESTRATOR` | `auto`, `pydantic_ai`, or `legacy` |
| `AI_DIAGNOSTICS_MODEL` | Shared override model name |
| `AI_DIAGNOSTICS_TIMEOUT_SECONDS` | Optional request timeout override |
| `AI_DIAGNOSTICS_BASE_URL` | Local OpenAI-compatible base URL |
| `AI_DIAGNOSTICS_API_KEY` | Local OpenAI-compatible API key |
| `OPENAI_API_KEY` | Enables OpenAI diagnostics |
| `OPENAI_MODEL` | OpenAI model override |
| `OPENROUTER_API_KEY` | Enables OpenRouter diagnostics |
| `OPENROUTER_MODEL` | OpenRouter model override |
| `OPENROUTER_HTTP_REFERER` | Optional OpenRouter header |
| `OPENROUTER_APP_TITLE` | Optional OpenRouter header |
| `ANTHROPIC_API_KEY` | Enables Anthropic diagnostics |
| `ANTHROPIC_MODEL` | Anthropic model override |
| `LOCAL_LLM_MODEL` | Local model name hint |

The code also accepts local OpenAI-compatible aliases through:

- `OPENAI_COMPAT_BASE_URL`
- `OPENAI_COMPAT_API_KEY`
- `LOCAL_LLM_BASE_URL`
- `LOCAL_LLM_API_KEY`

`AI_DIAGNOSTICS_PROVIDER=auto` checks providers in this order:

1. OpenAI
2. OpenRouter
3. Anthropic
4. local endpoint

`AI_DIAGNOSTICS_ORCHESTRATOR=auto` tries the typed PydanticAI diagnostics path first and falls back to the legacy per-provider JSON path if that adapter fails. Set it to `pydantic_ai` to force the new path or `legacy` to bypass it while debugging provider issues.

### Validation gates

- `bash tests/testsuite_full.sh` is the single validation wrapper used locally and in CI.
- `scripts/install_git_hooks.sh` configures the repo-managed pre-push hook so broken changes are blocked before they reach GitHub.
- `.github/workflows/validate.yml` runs the same wrapper on pushes and pull requests.

### SoccerNet batch launcher

`backend/scripts/start_soccernet_batch_tmux.sh` expects:

| Variable | Purpose |
| --- | --- |
| `SOCCERNET_PASSWORD` | Password for SoccerNet downloads |

## Start Commands

### Backend only

```bash
cd backend
./run_backend.sh
```

This script:

- creates `backend/.venv` if needed
- upgrades `pip`
- installs `backend/requirements.txt`
- starts FastAPI with reload on `127.0.0.1:8431`
- prewarms the default detector and field-calibration models during backend startup

### Frontend only

```bash
cd frontend
./run_frontend.sh
```

This script:

- runs `npm install`
- starts Vite on `127.0.0.1:4317`
- points the frontend at `http://127.0.0.1:8431`

### Combined launcher

```bash
./run_all.sh
```

This installs both halves, starts the backend in the background, then starts the frontend in the foreground.

### Stop helper

```bash
./stop_backend.sh
```

This only stops a backend that wrote `backend_server.pid`.

## Models, Runtime, And Cache Behaviour

The active model names in code are:

- detector: `soccana`
- field calibration: `soccana_keypoint`
- detector training base weight: `soccana`
- detector runtime today: Ultralytics + PyTorch
- planned inference runway: ONNX Runtime with CoreML on Apple Silicon and ONNX Runtime with CUDA on GPU hosts

The backend caches the default weights under:

- `backend/models/soccana/Model/weights/best.pt`
- `backend/models/soccana_keypoint/Model/weights/best.pt`

On Apple Silicon:

- detector inference prefers MPS
- detector fine-tuning can use `auto` or `mps`
- field calibration still falls back to CPU when detector inference uses MPS

Detector fine-tuning remains local-first and writes to:

- `backend/training_runs/<run_id>/`
- `backend/models/registry.json`

## Recommended First Validation

Backend syntax check:

```bash
cd backend
python -m py_compile app/main.py app/wide_angle.py app/training.py app/training_manager.py app/training_registry.py app/train_worker.py
```

Frontend build:

```bash
cd frontend
npm run build
```

The repository currently includes short football clips in:

```text
backend/datasets/bundesliga_sample/Bundesliga/Clips/
```

That remains the quickest place to test live preview and saved analysis runs.

## Analysis First Run

1. Open `http://127.0.0.1:4317`.
2. In `Analysis Workspace`, paste a local clip path or upload a file.
3. Click `Load input clip`.
4. Optionally open `Live Preview`.
5. Click `Analyze loaded clip`.
6. Watch `Active Job`.
7. Inspect the saved overlay, diagnostics, and files in `Run Review`.

## Training Studio First Run

1. Switch to `Training Studio`.
2. In `Datasets`, scan a local YOLO detector dataset root.
3. Make sure the scan reports usable class mapping and no blocking errors.
4. In `Train`, keep the base weights on `soccana` for V1.
5. Start a short smoke run.
6. Inspect logs, artifact paths, and the run summary in `Jobs`.
7. In `Registry`, inspect the checkpoint and activate it only when you want analysis to use it while the analysis selector remains on `soccana`.

The detector-training V1 expects:

- a YOLO detector dataset
- usable class names in `dataset.yaml` or `data.yaml`
- classes that let the app resolve player / goalkeeper and ball labels back into analysis

## Real-Run Validation Baseline

The current code has been validated against:

- `/api/live-preview` on a real local football clip
- `/api/analyze` on the same clip
- `/api/train/datasets/scan` on both valid and invalid detector datasets
- `/api/train/jobs/detect` on a real smoke training run
- `/api/train/runs/{run_id}/activate` on that completed smoke run
- a browser pass covering:
  - default boot state
  - input clip loading
  - live preview state
  - saved run review entry
  - SoccerNet browse/filter behavior
  - Training Studio dataset scan
  - Training Studio jobs / registry
  - path persistence across refresh

## Troubleshooting

### Browser overlay video does not play

Check whether `ffmpeg` is installed. The backend falls back to a raw `mp4v` file when `ffmpeg` is missing or transcode fails.

### Backend startup takes longer than expected

The backend prewarms the default detector and field-calibration models during startup. That moves model resolution earlier so the first real run is ready sooner.

### Source or job state after backend restart

Loaded analysis sources are restored from the source registry, interrupted analysis jobs reappear as snapshots, and completed runs still live under `backend/runs/`.

Training jobs also persist under `backend/training_runs/` with `summary.json`, logs, and artifact paths.

### Detector training scan says the dataset is invalid

The current V1 scan will reject detector datasets when:

- no usable training images are found
- the training split has no usable labeled detections
- label rows are malformed for YOLO detector training
- class names cannot be resolved well enough to map the detector back into analysis

### Goal metrics stay at zero

Use either an adjacent `Labels-v2.json` / `Labels.json` file or the explicit label-path field in the UI so the analysis run can attach goal events to the experiment outputs.

### Diagnostics do not use a language model

If no provider resolves, the backend still completes the run and writes heuristic diagnostics instead of an AI-curated run brief.

## Licensing Note

This repository is MIT, but the current detector and detector-training runtime still depends on the local Ultralytics stack, which Ultralytics distributes under AGPL-3.0 by default unless you have a separate enterprise license.
