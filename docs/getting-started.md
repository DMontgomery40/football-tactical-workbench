# Getting Started

## Prerequisites

- macOS or another environment with local filesystem access to your clips
- Python 3 with `venv`
- Node.js and `npm`
- `ffmpeg` on your `PATH` if you want browser-friendly H.264 overlays
- network access on first model run so the backend can download model weights if they are not already cached

## Backend Setup

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The backend dependencies currently declared in code-facing setup are:

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

The frontend is a Vite + React single-page app.

## Environment Variables

The backend loads `.env` and `.env.local` from the repository root if they exist.

### AI diagnostics

The code supports four diagnostics providers:

- OpenAI
- OpenRouter
- Anthropic
- local OpenAI-compatible endpoint

Relevant variables supported by the code:

| Variable | Purpose |
| --- | --- |
| `AI_DIAGNOSTICS_PROVIDER` | `auto`, `openai`, `openrouter`, `anthropic`, `local`, or `off` |
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

Every completed run still gets diagnostics. With a configured provider, the backend writes an AI-curated run brief plus per-run diagnostics; otherwise it writes heuristic diagnostics.

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

This script installs both halves, starts the backend in the background, then starts the frontend in the foreground.

### Stop helper

```bash
./stop_backend.sh
```

This only stops a backend that wrote `backend_server.pid`.

## Models And Cache Behaviour

The active model names in code are:

- detector: `soccana`
- field calibration: `soccana_keypoint`

The backend resolves them through Hugging Face and caches them under:

- `backend/models/soccana/Model/weights/best.pt`
- `backend/models/soccana_keypoint/Model/weights/best.pt`

The detector uses the football-specific class mapping embedded in `backend/app/wide_angle.py`:

- player class id: `0`
- ball class id: `1`
- referee class id: `2`

## Recommended First Validation

Backend syntax check:

```bash
cd backend
python -m py_compile app/main.py app/wide_angle.py
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

That directory is the quickest place to test live preview and analysis.

## Real-Run Validation Baseline

The current code was validated against:

- `/api/live-preview` on `backend/datasets/bundesliga_sample/Bundesliga/Clips/08fd33_1.mp4`
- `/api/analyze` on the same clip

Observed runtime behaviour from the current code:

- live preview returned multipart MJPEG frames
- analysis completed successfully
- the run wrote overlay video, detections CSV, track summary CSV, projections CSV, entropy timeseries CSV, summary JSON, per-run diagnostics JSON, and a zip bundle
- interrupted jobs survived restart as failed job snapshots instead of disappearing
- loaded sources survived restart through the persisted source registry

## Troubleshooting

### Browser overlay video does not play

Check whether `ffmpeg` is installed. The backend falls back to a raw `mp4v` file when `ffmpeg` is missing or transcode fails.

### Backend startup takes longer than expected

The backend now prewarms the default detector and field-calibration models during startup. That moves model resolution work earlier so the first real analysis run is ready sooner.

### Source or job state after backend restart

Loaded sources are restored from the source registry, and interrupted jobs are restored as failed snapshots. Completed runs still live under `backend/runs/`.

### Goal metrics stay at zero

Use either an adjacent `Labels-v2.json` / `Labels.json` file or the explicit label-path field in the UI so the analysis run can attach goal events to the experiment outputs.

### Diagnostics do not use a language model

If no provider resolves, the backend still completes the run and writes heuristic diagnostics instead of an AI-curated run brief.
