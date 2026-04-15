#!/usr/bin/env bash
# Stage 1: CTC encoder pretraining
# Usage: bash scripts/pretrain_encoder.sh [--resume_from PATH]

set -euo pipefail

CONFIG=${1:-configs/experiment/pretrain_ctc.yaml}
RESUME=${2:-""}

RESUME_FLAG=""
if [ -n "$RESUME" ]; then
    RESUME_FLAG="--resume_from $RESUME"
fi

python -m st.training.pretrain_ctc \
    --config "$CONFIG" \
    $RESUME_FLAG
