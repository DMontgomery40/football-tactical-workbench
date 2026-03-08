#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")"/../.. && pwd)"
BACKEND_DIR="$ROOT_DIR/backend"
ENV_FILE="$ROOT_DIR/.env"

if [ ! -f "$ENV_FILE" ]; then
  echo "[batch] missing $ENV_FILE" >&2
  exit 1
fi

set -a
source "$ENV_FILE"
set +a

if [ -z "${SOCCERNET_PASSWORD:-}" ]; then
  echo "[batch] SOCCERNET_PASSWORD is not set in $ENV_FILE" >&2
  exit 1
fi

SPLIT="${1:-train}"
LIMIT="${2:-20}"
OFFSET="${3:-0}"
FILES="${4:-1_224p.mkv 2_224p.mkv Labels-v2.json}"
TRACKER_MODES="${5:-hybrid_reid}"

STAMP="$(date +%Y%m%d_%H%M%S)"
SESSION_NAME="sn_${SPLIT}_${LIMIT}_${STAMP}"
BATCH_NAME="soccernet_${SPLIT}_${LIMIT}_goalaligned_v1_${STAMP}"
LOG_DIR="$BACKEND_DIR/experiments/$BATCH_NAME"
mkdir -p "$LOG_DIR"

echo "[batch] root: $ROOT_DIR"
echo "[batch] session: $SESSION_NAME"
echo "[batch] batch: $BATCH_NAME"
echo "[batch] log dir: $LOG_DIR"
echo "[batch] tracker modes: $TRACKER_MODES"

tmux new-session -d -s "$SESSION_NAME" \
  "cd \"$BACKEND_DIR\" && source .venv/bin/activate && SOCCERNET_PASSWORD=\"$SOCCERNET_PASSWORD\" python scripts/soccernet_batch_experiment.py --split \"$SPLIT\" --offset \"$OFFSET\" --limit \"$LIMIT\" --batch-name \"$BATCH_NAME\" --files $FILES --tracker-modes $TRACKER_MODES 2>&1 | tee \"$LOG_DIR/tmux_stdout.log\""

echo "[batch] started tmux session $SESSION_NAME"
