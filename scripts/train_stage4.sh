#!/usr/bin/env bash
# Stage 4: Full fine-tuning (encoder + projector + full LLM)
# Usage: bash scripts/train_stage4.sh [CONFIG] [RESUME_FROM]

set -euo pipefail

CONFIG=${1:-configs/experiment/stage4_full.yaml}
RESUME=${2:-""}

RESUME_FLAG=""
if [ -n "$RESUME" ]; then
    RESUME_FLAG="--resume_from $RESUME"
fi

python -m st.training.train_st \
    --config "$CONFIG" \
    $RESUME_FLAG
