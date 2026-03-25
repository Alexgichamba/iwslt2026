#!/bin/bash
# Stage 2/3: Train speech translation (projector +/- LoRA)
#
# Usage:
#   ./scripts/train_st.sh configs/experiment/train_st_stage3.yaml
#   ./scripts/train_st.sh configs/experiment/train_st_stage3.yaml --resume_from checkpoints/st/st_step10000.pt

set -euo pipefail

CONFIG=${1:?Usage: $0 <config> [--resume_from <checkpoint>]}
shift

echo "=== Speech Translation Training ==="
echo "Config: $CONFIG"

python -m st.training.train_st \
    --config "$CONFIG" \
    "$@"