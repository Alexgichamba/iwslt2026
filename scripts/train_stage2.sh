#!/bin/bash
#SBATCH -p GPU-shared
#SBATCH --gres=gpu:h100-80:1
#SBATCH -t 48:00:00
#SBATCH -A cis250145p
#SBATCH -J Aura_Stage2_1GPU
#SBATCH -o /ocean/projects/cis250145p/tanghang/iwslt2026/runs/stage2/checkpoint_step2000/logs/slurm-%j.out
#SBATCH -e /ocean/projects/cis250145p/tanghang/iwslt2026/runs/stage2/checkpoint_step2000/logs/slurm-%j.err

module load cuda/12.4.0
PYTHON_ENV="/ocean/projects/cis250145p/tanghang/Aura_base/env/bin/python"
cd /ocean/projects/cis250145p/tanghang/iwslt2026/src

PYTHONPATH=$(pwd) $PYTHON_ENV -m st.training.train_st \
    --config /ocean/projects/cis250145p/tanghang/iwslt2026/configs/experiment/stage2.yaml
    # --resume_from /ocean/projects/cis250145p/tanghang/iwslt2026/runs/stage2/checkpoint_step2000
