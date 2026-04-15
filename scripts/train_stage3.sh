#!/usr/bin/env bash
# Stage 3: Projector + LoRA fine-tuning
# Usage: bash scripts/train_stage3.sh [CONFIG] [RESUME_FROM]

set -euo pipefail

CONFIG=${1:-configs/experiment/stage3.yaml}
RESUME=${2:-""}

RESUME_FLAG=""
if [ -n "$RESUME" ]; then
    RESUME_FLAG="--resume_from $RESUME"
fi

python -m st.training.train_st \
    --config "$CONFIG" \
    $RESUME_FLAG
