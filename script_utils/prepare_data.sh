#!/bin/bash
#SBATCH --job-name=prep_pdb_data
#SBATCH --mem=256G
#SBATCH --mincpu=32
#SBATCH -x node330,node329,node1927
#SBATCH --time=96:00:00
#SBATCH -p pi_keating
#SBATCH -o /orcd/pool/005/keating_shared/fosterb/la-proteina-contrastive/prep-data-output.out
#SBATCH -e /orcd/pool/005/keating_shared/fosterb/la-proteina-contrastive/prep-data-error.out

# This is a CPU-only job (no --gres=gpu); data preprocessing doesn't need a GPU.
# node330 has an older GLIBC (< 2.28) that's incompatible with the conda env's
# PyTorch build, so it's excluded along with other known-bad nodes. Training
# jobs land on H100 nodes which all have current GLIBC, hence why train_ae.sh
# works without this exclusion.

# Run this ONCE via sbatch to download and preprocess the PDB dataset subsets.
#
# This may take 30-90 minutes for the tiny subset and several hours for small/full.
# Once it finishes, all sbatch training jobs will find the data cached and skip
# straight to training.

DATASET=pdb/pdb_train_ucond_full

PROJECT_DIR=/orcd/pool/005/keating_shared/fosterb/la-proteina-contrastive

CONDA_ROOT=/home/software/anaconda3/2023.07
source ${CONDA_ROOT}/etc/profile.d/conda.sh
conda activate laproteina_env

export DATA_PATH=${PROJECT_DIR}/datasets

# Graphein's PDBManager unconditionally tries to download a ligand map from
# ligand-expo.rcsb.org, which RCSB decommissioned. Placeholder satisfies the
# existence check without requiring the dead endpoint.
mkdir -p "${DATA_PATH}/pdb_train"
touch "${DATA_PATH}/pdb_train/cc-to-pdb.tdd"

cd "${PROJECT_DIR}"

echo "Starting data preparation for dataset=${DATASET}"
echo "DATA_PATH=${DATA_PATH}"
echo "This will download PDB structures and preprocess them into .pt files."
echo ""

python - <<EOF
import sys
sys.path.insert(0, ".")

import hydra
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import os

config_dir = os.path.abspath("configs")
with initialize_config_dir(config_dir=config_dir, version_base=None):
    cfg = compose(
        config_name="training_ae_baseline",
        overrides=["dataset=${DATASET}"],
    )

import hydra
datamodule = hydra.utils.instantiate(cfg.dataset.datamodule)
print("Calling prepare_data() [this is where downloads happen...]")
datamodule.prepare_data()
print("Done. Data is cached; training jobs can now run on compute nodes.")
EOF