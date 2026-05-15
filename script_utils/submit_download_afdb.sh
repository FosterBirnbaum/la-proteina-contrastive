#!/bin/bash
#SBATCH --job-name=prep_pdb_data
#SBATCH --mem=256G
#SBATCH --mincpu=32
#SBATCH -x node1927,node1708,node1709
#SBATCH --time=76:00:00
#SBATCH -p pi_keating
#SBATCH -o /orcd/pool/005/keating_shared/fosterb/la-proteina-contrastive/download-data-output.out
#SBATCH -e /orcd/pool/005/keating_shared/fosterb/la-proteina-contrastive/download-data-error.out

# Environment
CONDA_ROOT=/home/software/anaconda3/2023.07
source ${CONDA_ROOT}/etc/profile.d/conda.sh
conda activate laproteina_env

python scripts/download_afdb.py \
  --ids /orcd/data/keating/001/fosterb/afdb/laproteina_afdb_ids/AFDB_IDs-512.txt \
  --out /orcd/data/keating/001/fosterb/afdb/laproteina_afdb_ids/afdb_344k/raw \
  --workers 32