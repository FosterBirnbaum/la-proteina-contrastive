"""
Modality-alignment probe for the contrastive VAE.

For each batch in the val set, the model is run twice:
  Pass A  — full-info batch (unchanged)
  Pass B  — same batch with residue_type zeroed (structure-only view)

The per-residue cosine similarity between the encoder *mean* vectors from
the two passes is then computed.  Expected readings:

  Contrastive model  →  mean cosine sim  ≥ 0.95   (high alignment)
  Baseline model     →  mean cosine sim  ~ 0.6–0.85 (no alignment signal)

The script also breaks down cosine similarity by amino-acid type so you can
see whether the alignment is uniform across residue types or concentrated in
a subset.

Usage (from project root on the cluster or locally):
    cd $PROJECT_DIR
    export DATA_PATH=$PROJECT_DIR/datasets

    python script_utils/probe_modality_alignment.py \\
        --ckpt  store/laproteina_ae_contrastive/checkpoints/chk_...ckpt \\
        --dataset  pdb/pdb_train_ucond_small \\
        --n_batches  40 \\
        --out  probe_results_contrastive.csv

To compare the baseline at the same time, run the script once per checkpoint
and compare the printed summaries / output CSVs.

EMA weights:
    Training uses EMA (decay=0.999) and validation metrics are computed with
    EMA weights.  This script automatically loads EMA weights from the regular
    checkpoint file (they are stored in optimizer_states[0]["ema"]).  Pass
    --no_ema to use the raw training weights instead.
"""

import argparse
import os
import sys

# Allow running from the project root without `pip install -e .`
sys.path.insert(0, os.path.abspath("."))

import copy
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from loguru import logger
from omegaconf import OmegaConf

import hydra
from hydra import compose, initialize_config_dir

from proteinfoundation.partial_autoencoder.autoencoder import AutoEncoder
from proteinfoundation.partial_autoencoder.contrastive import build_masked_batch

# ── Amino-acid index → 1-letter code (standard atom37 ordering) ──────────────
AA_NAMES = [
    "ALA", "ARG", "ASN", "ASP", "CYS",
    "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO",
    "SER", "THR", "TRP", "TYR", "VAL",
]
AA_1LETTER = list("ARNDCQEGHILKMFPSTWYV")


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Modality-alignment probe")
    p.add_argument(
        "--ckpt", required=True,
        help="Path to a .ckpt checkpoint file (regular, not -EMA).",
    )
    p.add_argument(
        "--dataset", default="pdb/pdb_train_ucond_small",
        help="Hydra dataset config name relative to configs/dataset/, "
             "e.g. pdb/pdb_train_ucond_small",
    )
    p.add_argument(
        "--n_batches", type=int, default=40,
        help="Number of val batches to process (default 40).",
    )
    p.add_argument(
        "--batch_size", type=int, default=8,
        help="Batch size override for the dataloader (default 8).",
    )
    p.add_argument(
        "--out", default="probe_alignment.csv",
        help="Output CSV path for per-residue results (default probe_alignment.csv).",
    )
    p.add_argument(
        "--no_ema", action="store_true",
        help="Use raw training weights instead of EMA weights.",
    )
    p.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device (default: cuda if available).",
    )
    return p.parse_args()


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model_with_ema(ckpt_path: str, device: str, use_ema: bool = True) -> AutoEncoder:
    """Load AutoEncoder from a Lightning checkpoint, optionally applying EMA weights.

    The checkpoint saves both the regular model state_dict and the EMA params
    inside optimizer_states[0]["ema"] (a tuple of tensors, one per model param).
    We load the model from state_dict first, then overwrite with EMA params if
    requested.
    """
    logger.info(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # Restore model from hparams + state_dict
    model = AutoEncoder.load_from_checkpoint(ckpt_path, map_location="cpu")
    model.eval()

    if use_ema:
        try:
            ema_params = ckpt["optimizer_states"][0]["ema"]
            logger.info(f"Applying EMA weights ({len(ema_params)} tensors).")
            with torch.no_grad():
                for param, ema_p in zip(model.parameters(), ema_params):
                    param.data.copy_(ema_p.to("cpu"))
        except (KeyError, IndexError, TypeError) as e:
            logger.warning(
                f"Could not load EMA weights ({e}); using regular weights instead."
            )

    model = model.to(device)
    logger.info(f"Model loaded. Contrastive enabled: {model._contrastive_enabled()}")
    return model


# ── Dataset loading ───────────────────────────────────────────────────────────

def load_val_dataloader(dataset_cfg_name: str, batch_size: int):
    """Instantiate the PDB val dataloader from a Hydra dataset config."""
    config_dir = os.path.abspath("configs")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        # We only need the dataset sub-config; compose with a minimal base.
        cfg = compose(
            config_name="training_ae_contrastive",
            overrides=[f"dataset={dataset_cfg_name}"],
        )

    cfg.dataset.datamodule.batch_size = batch_size
    cfg.dataset.datamodule.num_workers = min(
        cfg.dataset.datamodule.get("num_workers", 4), 8
    )

    datamodule = hydra.utils.instantiate(cfg.dataset.datamodule)
    datamodule.prepare_data()
    datamodule.setup("fit")
    val_loader = datamodule.val_dataloader()
    logger.info(
        f"Val dataloader: {len(val_loader)} batches "
        f"(batch_size={batch_size}, "
        f"~{len(val_loader) * batch_size} structures total)"
    )
    return val_loader


# ── Probe logic ───────────────────────────────────────────────────────────────

@torch.no_grad()
def probe_batch(
    model: AutoEncoder,
    batch: Dict[str, torch.Tensor],
    device: str,
) -> Dict[str, torch.Tensor]:
    """Run the modality-alignment probe on a single batch.

    Returns a dict with:
        cos_sim  [M]  — per-valid-residue cosine similarity (A vs B mean vectors)
        res_type [M]  — amino-acid index for each valid residue
    """
    # Move batch to device
    batch = {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }
    # Recursively move nested dicts (e.g. mask_dict)
    for k, v in batch.items():
        if isinstance(v, dict):
            batch[k] = {
                kk: vv.to(device) if isinstance(vv, torch.Tensor) else vv
                for kk, vv in v.items()
            }

    mask = batch["mask_dict"]["coords"][..., 0, 0]  # [b, n]
    batch["mask"] = mask

    # Pass A: full-info encoding
    out_a = model.encoder(batch)  # mean: [b, n, d]

    # Pass B: structure-only (residue_type zeroed)
    batch_b = build_masked_batch(batch, keys=["residue_type"])
    batch_b["mask"] = mask
    out_b = model.encoder(batch_b)  # mean: [b, n, d]

    mean_a = out_a["mean"]  # [b, n, d]
    mean_b = out_b["mean"]  # [b, n, d]

    # Extract valid residues only
    idx = mask.nonzero(as_tuple=False)  # [M, 2]
    za = mean_a[idx[:, 0], idx[:, 1]]   # [M, d]
    zb = mean_b[idx[:, 0], idx[:, 1]]   # [M, d]

    # Per-residue cosine similarity
    cos_sim = F.cosine_similarity(za, zb, dim=-1)  # [M]

    # Residue types for breakdown
    res_type = batch["residue_type"][idx[:, 0], idx[:, 1]]  # [M]

    return {"cos_sim": cos_sim.cpu(), "res_type": res_type.cpu()}


# ── Summary helpers ───────────────────────────────────────────────────────────

def print_summary(all_cos: np.ndarray, all_rt: np.ndarray, ckpt_path: str) -> None:
    print("\n" + "=" * 65)
    print(f"  Modality-alignment probe — {os.path.basename(ckpt_path)}")
    print("=" * 65)
    print(f"  Total valid residues evaluated : {len(all_cos):,}")
    print(f"  Mean cosine similarity         : {all_cos.mean():.4f}")
    print(f"  Median cosine similarity       : {np.median(all_cos):.4f}")
    print(f"  Std cosine similarity          : {all_cos.std():.4f}")
    print(f"  Fraction > 0.90                : {(all_cos > 0.90).mean():.3f}")
    print(f"  Fraction > 0.95                : {(all_cos > 0.95).mean():.3f}")
    print(f"  Fraction > 0.99                : {(all_cos > 0.99).mean():.3f}")
    print()
    print("  Per-amino-acid breakdown (mean cos sim):")
    print(f"  {'AA':<6}  {'Name':<5}  {'N':>7}  {'Mean':>7}  {'Std':>7}")
    print("  " + "-" * 40)
    for i, (aa1, aa3) in enumerate(zip(AA_1LETTER, AA_NAMES)):
        mask = all_rt == i
        if mask.sum() == 0:
            continue
        sims = all_cos[mask]
        print(
            f"  {aa1:<6}  {aa3:<5}  {mask.sum():>7,}  {sims.mean():>7.4f}  {sims.std():>7.4f}"
        )
    print("=" * 65 + "\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Graphein ligand-map placeholder (avoids dead rcsb.org download)
    data_path = os.environ.get("DATA_PATH", "./datasets")
    os.makedirs(os.path.join(data_path, "pdb_train"), exist_ok=True)
    placeholder = os.path.join(data_path, "pdb_train", "cc-to-pdb.tdd")
    if not os.path.exists(placeholder):
        open(placeholder, "w").close()

    model = load_model_with_ema(args.ckpt, args.device, use_ema=not args.no_ema)

    val_loader = load_val_dataloader(args.dataset, args.batch_size)

    all_cos: List[torch.Tensor] = []
    all_rt:  List[torch.Tensor] = []

    for i, batch in enumerate(val_loader):
        if i >= args.n_batches:
            break
        result = probe_batch(model, batch, args.device)
        all_cos.append(result["cos_sim"])
        all_rt.append(result["res_type"])
        if (i + 1) % 10 == 0:
            running_mean = torch.cat(all_cos).mean().item()
            logger.info(
                f"  Batch {i+1:3d}/{args.n_batches}  "
                f"running mean cos_sim = {running_mean:.4f}"
            )

    all_cos_np = torch.cat(all_cos).numpy()
    all_rt_np  = torch.cat(all_rt).numpy()

    print_summary(all_cos_np, all_rt_np, args.ckpt)

    # Save per-residue CSV
    df = pd.DataFrame({
        "cos_sim":  all_cos_np,
        "res_type": all_rt_np,
        "res_name": [AA_NAMES[int(r)] if 0 <= int(r) < 20 else "UNK" for r in all_rt_np],
    })
    df.to_csv(args.out, index=False)
    logger.info(f"Per-residue results saved to {args.out}")


if __name__ == "__main__":
    main()
