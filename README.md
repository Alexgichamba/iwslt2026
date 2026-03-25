# IWSLT 2026 — End-to-End Low-Resource Speech Translation

End-to-end speech translation system for the IWSLT 2026 low-resource track, targeting African languages (Bemba, Hausa, Igbo, Yoruba → English).

## Architecture

```
Audio → Log-Mel → ConvSubsampler(4x) → Conformer Encoder → Projector → LLM → Translation
```

**Training stages:**

1. **Stage 1 — CTC Pretraining**: Train the conformer speech encoder on ASR data with CTC loss.
2. **Stage 2 — Projector Training**: Freeze encoder + LLM, train only the modality projector on ST data.
3. **Stage 3 — LoRA Fine-tuning**: Freeze encoder, train projector + LoRA adapters on the LLM.

## Setup

```bash
pip install -e ".[dev]"
```

## Data Format

Manifest files (CSV/TSV) with columns:

| Column | Description |
|---|---|
| `audio_path` | Relative path to audio file |
| `transcript` | Source language transcript (for CTC) |
| `translation` | English translation (for ST) |
| `language` | Source language code |
| `duration` | Duration in seconds |

## Usage

### Stage 1: Pretrain encoder
```bash
bash scripts/pretrain_encoder.sh configs/experiment/pretrain_ctc.yaml
```

### Stage 2: Train projector
```bash
bash scripts/train_st.sh configs/experiment/train_st_stage2.yaml
```

### Stage 3: Train projector + LoRA
```bash
bash scripts/train_st.sh configs/experiment/train_st_stage3.yaml
```

### Inference
```bash
python -m st.inference.generate \
    --config configs/experiment/train_st_stage3.yaml \
    --encoder_ckpt checkpoints/encoder/encoder_final.pt \
    --projector_ckpt checkpoints/st/st_epoch20.pt \
    --lora_path checkpoints/st/lora_epoch20 \
    --audio_paths test1.wav test2.wav
```

### Tests
```bash
pytest tests/ -v
```

## Config System

Configs are plain YAML, composed into full experiment configs under `configs/experiment/`. Swap components by changing the experiment file:

- **Encoder**: `configs/encoder/` — conformer_base, conformer_small
- **Projector**: `configs/projector/` — linear, mlp, conv, stack
- **LLM**: `configs/llm/` — afriquellm (with/without LoRA)
- **Training**: `configs/training/` — pretrain_ctc, train_st

## Project Structure

```
iwslt2026/
├── configs/
│   ├── encoder/          # Conformer architecture configs
│   ├── projector/        # Projector type configs
│   ├── llm/              # LLM + LoRA configs
│   ├── training/         # Hyperparameter configs
│   └── experiment/       # Composed full-experiment configs
├── src/st/
│   ├── models/
│   │   ├── speech_encoder.py   # Conformer + CTC head
│   │   ├── projector.py        # Linear / MLP / Conv / Stack projectors
│   │   ├── llm_wrapper.py      # HuggingFace LM wrapper + LoRA
│   │   └── speech_llm.py       # Full encoder→projector→LLM model
│   ├── data/                   # Dataset, collators
│   ├── training/               # Training scripts (CTC, ST)
│   ├── inference/              # Generation / translation
│   └── utils/                  # Audio features, metrics, config loading
├── scripts/                    # Shell launch scripts
├── tests/                      # Smoke tests
└── pyproject.toml
```
