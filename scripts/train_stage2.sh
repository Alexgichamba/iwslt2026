#!/usr/bin/env bash
# Stage 2: Projector alignment (freeze encoder + LLM)
# Usage: bash scripts/train_stage2.sh [--resume_from PATH]

set -euo pipefail

CONFIG=${1:-configs/experiment/stage2.yaml}
RESUME=${2:-""}

RESUME_FLAG=""
if [ -n "$RESUME" ]; then
    RESUME_FLAG="--resume_from $RESUME"
fi

python -m st.training.train_st \
    --config "$CONFIG" \
    $RESUME_FLAG
