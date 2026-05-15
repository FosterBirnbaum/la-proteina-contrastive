#!/bin/bash
#SBATCH --job-name=contrastive_full
#SBATCH --mem=150G
#SBATCH --mincpu=24
#SBATCH --gres=gpu:h100:1
#SBATCH --time=48:00:00
#SBATCH -p mit_preemptable
#SBATCH -x node1927
#SBATCH -x node1709
#SBATCH -o /orcd/pool/005/keating_shared/fosterb/la-proteina-contrastive/experiments/%x/train-output.out
#SBATCH -e /orcd/pool/005/keating_shared/fosterb/la-proteina-contrastive/experiments/%x/train-error.out

# IMPORTANT: keep EXPERIMENT_NAME in sync with --job-name above; SLURM expands
# %x to --job-name at job-launch time so shell variables cannot be used there.
EXPERIMENT_NAME=contrastive_full
CONFIG_NAME=training_ae_contrastive
DATASET=pdb/pdb_train_ucond_full

PROJECT_DIR=/orcd/pool/005/keating_shared/fosterb/la-proteina-contrastive
OUTPUT_DIR=${PROJECT_DIR}/experiments/${EXPERIMENT_NAME}

# The output dir must exist before SLURM opens the -o/-e files (which happens
# at job launch, before this script runs). Create it here for any Python
# outputs, and run  mkdir -p <OUTPUT_DIR>  once before the first sbatch call
# for a new experiment name.
mkdir -p "${OUTPUT_DIR}"

# Environment
CONDA_ROOT=/home/software/anaconda3/2023.07
source ${CONDA_ROOT}/etc/profile.d/conda.sh
conda activate laproteina_env

export DATA_PATH=${PROJECT_DIR}/datasets

# Route Python multiprocessing temp files to /dev/shm (RAM-backed tmpfs,
# always local to the node) to avoid NFS silly-rename (.nfsXXXX) OSErrors
# during worker cleanup. Scoped inline to the python command so your broader
# TMPDIR setting (for other tools) is left untouched.

# Graphein's PDBManager unconditionally tries to download a ligand map from
# ligand-expo.rcsb.org, which RCSB decommissioned. Since we never filter by
# ligands (has_ligands/remove_ligands are empty), a placeholder file is enough
# to satisfy the "if not os.path.exists(...)" check and skip the dead download.
mkdir -p "${DATA_PATH}/pdb_train"
touch "${DATA_PATH}/pdb_train/cc-to-pdb.tdd"

# train.py inserts os.path.abspath(".") into sys.path, so we must cd to the
# project root for imports (proteinfoundation.*) to resolve correctly.
cd "${PROJECT_DIR}"

# Training
TMPDIR=/dev/shm python proteinfoundation/partial_autoencoder/train.py \
    --config-name="${CONFIG_NAME}" \
    dataset="${DATASET}" \
    +single=true \
    +nolog=true \
    dataset.datamodule.max_tokens_per_batch=1500 \
    loss.contrastive.minibatch_size=512
