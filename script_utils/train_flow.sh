#!/bin/bash
#SBATCH --job-name=flow_contrastive_small
#SBATCH --mem=120G
#SBATCH --mincpu=12
#SBATCH --gres=gpu:1
#SBATCH -x node1927
#SBATCH --time=96:00:00
#SBATCH -p pi_keating
#SBATCH -o /orcd/pool/005/keating_shared/fosterb/la-proteina-contrastive/experiments/%x/train-output.out
#SBATCH -e /orcd/pool/005/keating_shared/fosterb/la-proteina-contrastive/experiments/%x/train-error.out

# To run the BASELINE flow model instead, change THREE lines:
#   --job-name=flow_baseline_small      (in #SBATCH header)
#   EXPERIMENT_NAME=flow_baseline_small
#   CONFIG_NAME=training_local_latents_baseline_small
#
# The flow model only trains the score network; the VAE is fully frozen.
# The autoencoder_ckpt_path is set inside the YAML config — no override needed.
EXPERIMENT_NAME=flow_contrastive_small
CONFIG_NAME=training_local_latents_contrastive_small

PROJECT_DIR=/orcd/pool/005/keating_shared/fosterb/la-proteina-contrastive
OUTPUT_DIR=${PROJECT_DIR}/experiments/${EXPERIMENT_NAME}
mkdir -p "${OUTPUT_DIR}"

CONDA_ROOT=/home/software/anaconda3/2023.07
source ${CONDA_ROOT}/etc/profile.d/conda.sh
conda activate laproteina_env

export DATA_PATH=${PROJECT_DIR}/datasets
export PYTHONUNBUFFERED=1   # flush logger output immediately so .out updates live

mkdir -p "${DATA_PATH}/pdb_train"
touch "${DATA_PATH}/pdb_train/cc-to-pdb.tdd"

cd "${PROJECT_DIR}"

# +cluster=true: use config batch_size (not overridden to 2) and bf16-mixed precision.
# +single=true: single-GPU mode (ngpus_per_node_ = 1).
# +nolog=true: skip W&B logging.
TMPDIR=/dev/shm python proteinfoundation/train.py \
    --config-name="${CONFIG_NAME}" \
    +cluster=true \
    +single=true \
    +nolog=true
