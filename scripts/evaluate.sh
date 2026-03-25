#!/bin/bash
# Translate audio files and optionally compute BLEU/chrF

set -euo pipefail

CONFIG=${1:-configs/experiment/train_st_stage3.yaml}
ENCODER_CKPT=${2:-checkpoints/encoder/encoder_final.pt}
PROJECTOR_CKPT=${3:-checkpoints/st/st_epoch20.pt}
LORA_PATH=${4:-checkpoints/st/lora_epoch20}

shift 4 || true
AUDIO_FILES=("$@")

if [ ${#AUDIO_FILES[@]} -eq 0 ]; then
    echo "Usage: $0 <config> <encoder_ckpt> <proj_ckpt> <lora_path> <audio1.wav> [audio2.wav ...]"
    exit 1
fi

echo "=== Inference ==="
python -m st.inference.generate \
    --config "$CONFIG" \
    --encoder_ckpt "$ENCODER_CKPT" \
    --projector_ckpt "$PROJECTOR_CKPT" \
    --lora_path "$LORA_PATH" \
    --audio_paths "${AUDIO_FILES[@]}" \
    --max_new_tokens 256 \
    --num_beams 4
