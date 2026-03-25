#!/usr/bin/env bash
source /ocean/projects/cis250145p/gichamba/miniconda3/etc/profile.d/conda.sh
conda activate wakanda_stt
echo "Active environment: $CONDA_DEFAULT_ENV"
export HF_HOME=/ocean/projects/cis250145p/gichamba/huggingface
./scripts/train_st.sh configs/ablations/p1d_transformer.yaml
./scripts/train_st.sh configs/ablations/p1e_qformer.yaml