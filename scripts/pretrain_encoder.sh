#!/bin/bash
# Stage 1: Pretrain speech encoder with CTC
#
# Usage:
#   ./scripts/pretrain_encoder.sh configs/experiment/pretrain_ctc.yaml
#   ./scripts/pretrain_encoder.sh configs/experiment/pretrain_ctc.yaml --resume_from checkpoints/encoder/encoder_step50000.pt

set -euo pipefail

CONFIG=${1:?Usage: $0 <config> [--resume_from <checkpoint>]}
shift

echo "=== Stage 1: CTC Pretraining ==="
echo "Config: $CONFIG"

python -m st.training.pretrain_encoder \
    --config "$CONFIG" \
    "$@"