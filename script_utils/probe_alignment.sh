#!/bin/bash
#SBATCH --job-name=probe_alignment_small
#SBATCH --mem=32G
#SBATCH --mincpu=4
#SBATCH --gres=gpu:1
#SBATCH -x node1927
#SBATCH --time=01:00:00
#SBATCH -p pi_keating
#SBATCH -o /orcd/pool/005/keating_shared/fosterb/la-proteina-contrastive/experiments/%x/probe-output.out
#SBATCH -e /orcd/pool/005/keating_shared/fosterb/la-proteina-contrastive/experiments/%x/probe-error.out

# Usage: sbatch script_utils/probe_alignment.sh
#
# Edit the CKPT_CONTRASTIVE and CKPT_BASELINE paths below to point at the
# best-step checkpoints from each run (not the -EMA.ckpt files — the script
# reads EMA params from the regular checkpoint automatically).
#
# Results are written to experiments/probe_alignment/ as CSV files and the
# printed summary goes to the SLURM .out file.

PROJECT_DIR=/orcd/pool/005/keating_shared/fosterb/la-proteina-contrastive
DATASET=pdb/pdb_train_ucond_small
N_BATCHES=40      # 40 batches
BATCH_SIZE=8
CKPT_CONTRASTIVE=${PROJECT_DIR}/store/laproteina_ae_contrastive/lightning_logs/version_1/checkpoints/epoch=237-step=385000.ckpt
CKPT_BASELINE=${PROJECT_DIR}/store/laproteina_ae_baseline/lightning_logs/version_3/checkpoints/epoch=203-step=330000.ckpt

OUTPUT_DIR=${PROJECT_DIR}/experiments/probe_alignment_small
mkdir -p "${OUTPUT_DIR}"

CONDA_ROOT=/home/software/anaconda3/2023.07
source ${CONDA_ROOT}/etc/profile.d/conda.sh
conda activate laproteina_env

export DATA_PATH=${PROJECT_DIR}/datasets
mkdir -p "${DATA_PATH}/pdb_train"
touch "${DATA_PATH}/pdb_train/cc-to-pdb.tdd"

cd "${PROJECT_DIR}"

echo "===== Contrastive model ====="
TMPDIR=/dev/shm python script_utils/probe_modality_alignment.py \
    --ckpt   "${CKPT_CONTRASTIVE}" \
    --dataset "${DATASET}" \
    --n_batches ${N_BATCHES} \
    --batch_size ${BATCH_SIZE} \
    --out    "${OUTPUT_DIR}/probe_contrastive.csv"

echo ""
echo "===== Baseline model ====="
TMPDIR=/dev/shm python script_utils/probe_modality_alignment.py \
    --ckpt   "${CKPT_BASELINE}" \
    --dataset "${DATASET}" \
    --n_batches ${N_BATCHES} \
    --batch_size ${BATCH_SIZE} \
    --out    "${OUTPUT_DIR}/probe_baseline.csv"

echo ""
echo "Done. CSVs written to ${OUTPUT_DIR}/"
