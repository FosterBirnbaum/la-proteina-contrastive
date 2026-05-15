#!/bin/bash
#SBATCH --mem=150G
#SBATCH --mincpu=12
#SBATCH --gres=gpu:1
#SBATCH --time=96:00:00
#SBATCH -p pi_keating
#SBATCH -o /orcd/pool/005/keating_shared/fosterb/laproteina_self_consistency/run.out
#SBATCH -e /orcd/pool/005/keating_shared/fosterb/laproteina_self_consistency/run.err

mkdir -p /orcd/pool/005/keating_shared/fosterb/laproteina_self_consistency

CONDA_ROOT=/home/software/anaconda3/2021.11
source ${CONDA_ROOT}/etc/profile.d/conda.sh
# Use the same env that runs Struct2Struct's full pipeline (PottsMPNN here);
# Rosetta is shelled out to its own env via rosetta_python in the YAML.
conda activate PottsMPNN

REPO_ROOT="/orcd/pool/005/keating_shared/fosterb/la-proteina-contrastive"
CONFIG="/orcd/pool/005/keating_shared/fosterb/la-proteina-contrastive/script_utils/self_consistency_config_example.yaml"

python "${REPO_ROOT}/script_utils/self_consistency_pipeline.py" --config "${CONFIG}"
