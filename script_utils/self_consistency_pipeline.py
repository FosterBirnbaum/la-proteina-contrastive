#!/usr/bin/env python3
"""
Self-consistency pipeline for La-Proteina generations.

Compares baseline vs contrastive flow generations on:
  (a) as-generated self-consistency: AF3-refold the generated sequence,
      score against the generated structure; Rosetta FastRelax + score the
      generated sequence threaded onto the generated backbone.
  (b) redesigned self-consistency: redesign the sequence with Struct2Struct,
      then repeat (a) on the redesigned sequence.

The hypothesis is that contrastive has higher (a) scores and a smaller
(b) - (a) gap. Both are reported per-PDB and aggregated per-run.

Both stages reuse Struct2Struct's run_full_pipeline.py:
  - as_generated:  run_inference=false, FASTA written from each generated
                   PDB's own aatype.
  - redesigned:    run_inference=true, S2S writes a new FASTA.
The AF3 + Rosetta stages then compare against pdb_dir, which we point at the
generated structures. This makes "self-consistency vs the generated structure"
fall out of the existing tooling.

Usage:
    python self_consistency_pipeline.py --config self_consistency_config.yaml
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


# ---------------------------------------------------------------------------
# Stage 0: gather generated PDBs
# ---------------------------------------------------------------------------

def collect_generated_pdbs(gen_root: str) -> list[Path]:
    """Find all job_*.pdb files in a la-proteina inference/<config> tree."""
    root = Path(gen_root)
    if not root.is_dir():
        raise FileNotFoundError(f"Generation root not found: {gen_root}")
    pdbs = sorted(root.rglob("job_*.pdb"))
    if not pdbs:
        raise RuntimeError(f"No job_*.pdb files under {gen_root}")
    return pdbs


def stage_pdb_dir(pdbs: list[Path], staging_dir: Path) -> list[str]:
    """Copy generated PDBs into a flat dir; return their stem names."""
    staging_dir.mkdir(parents=True, exist_ok=True)
    names = []
    for src in pdbs:
        # The PDB and its parent dir share a name (job_X_n_Y_id_Z), so the
        # stem is unique. Strip the .pdb when writing the list file later.
        dst = staging_dir / src.name
        if not dst.exists():
            shutil.copy2(src, dst)
        names.append(src.stem)
    return names


def write_pdb_list(names: list[str], path: Path) -> None:
    path.write_text("\n".join(names) + "\n")


# ---------------------------------------------------------------------------
# Stage A: extract the generated sequence and write FASTA
# ---------------------------------------------------------------------------

def seq_from_pdb(pdb_path: Path) -> str:
    """One-letter sequence from a PDB's CA atoms (chain A, in residue order)."""
    seq_chars = []
    seen = set()
    with open(pdb_path) as fh:
        for line in fh:
            if not line.startswith("ATOM"):
                continue
            atom = line[12:16].strip()
            if atom != "CA":
                continue
            chain = line[21]
            resnum = line[22:27]  # number + insertion code
            key = (chain, resnum)
            if key in seen:
                continue
            seen.add(key)
            three = line[17:20].strip()
            seq_chars.append(THREE_TO_ONE.get(three, "X"))
    seq = "".join(seq_chars)
    if not seq:
        raise RuntimeError(f"No CA residues parsed from {pdb_path}")
    return seq


def write_as_generated_fasta(pdbs: list[Path], fasta_path: Path) -> None:
    """Write FASTA whose entries are the generated sequence inside each PDB.

    Header == PDB stem so the AF3-stage helper can match it back to the same
    pdb_dir entry that the redesigned stage uses.
    """
    fasta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(fasta_path, "w") as fh:
        for pdb in pdbs:
            seq = seq_from_pdb(pdb)
            fh.write(f">{pdb.stem}\n{seq}\n")


# ---------------------------------------------------------------------------
# Stage B: invoke Struct2Struct's run_full_pipeline.py
# ---------------------------------------------------------------------------

def write_pipeline_yaml(template: dict, overrides: dict, out_path: Path) -> None:
    cfg = dict(template)
    cfg.update(overrides)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False)


def run_pipeline(s2s_repo: str, config_path: Path, python_exe: str | None = None) -> None:
    runner = Path(s2s_repo) / "inference" / "run_full_pipeline.py"
    if not runner.is_file():
        raise FileNotFoundError(runner)
    py = python_exe or sys.executable
    cmd = [py, str(runner), "--config", str(config_path)]
    print(f"\n>>> {' '.join(cmd)}\n", flush=True)
    subprocess.run(cmd, check=True)


# ---------------------------------------------------------------------------
# Result aggregation
# ---------------------------------------------------------------------------

# AF3 stats columns we care about. The PottsMPNN_dave helper writes these into
# stats.csv keyed by 'pdb' (lowercased basename). We rename on load.
_AF3_KEEP = ["scrmsd", "sctm", "plddt", "ptm"]


def _norm_pdb_id(s: str) -> str:
    return Path(str(s)).stem.lower()


def load_af3_stats(epoch_dir: Path) -> pd.DataFrame:
    """Load <epoch>/predicted_sequences_af3/stats.csv if it exists."""
    p = epoch_dir / "predicted_sequences_af3" / "stats.csv"
    if not p.is_file():
        print(f"  (missing AF3 stats: {p})")
        return pd.DataFrame(columns=["pdb"] + _AF3_KEEP)
    df = pd.read_csv(p)
    # Lowercase columns; the dave helper isn't 100% consistent in casing.
    df.columns = [c.lower() for c in df.columns]
    if "pdb" not in df.columns:
        # Try common alternatives
        for cand in ("name", "id", "target"):
            if cand in df.columns:
                df = df.rename(columns={cand: "pdb"})
                break
    df["pdb"] = df["pdb"].map(_norm_pdb_id)
    keep = ["pdb"] + [c for c in _AF3_KEEP if c in df.columns]
    return df[keep]


def load_rosetta_metrics(epoch_dir: Path) -> pd.DataFrame:
    """Load <epoch>/pyrosetta_results_metrics/predicted_sequences_rosetta_metrics.csv."""
    p = epoch_dir / "pyrosetta_results_metrics" / "predicted_sequences_rosetta_metrics.csv"
    if not p.is_file():
        print(f"  (missing Rosetta metrics: {p})")
        return pd.DataFrame(columns=["pdb", "total_score", "rmsd", "packstat"])
    df = pd.read_csv(p)
    df["pdb"] = df["pdb"].map(_norm_pdb_id)
    return df[["pdb", "total_score", "rmsd", "packstat"]]


def merge_stage(epoch_dir: Path, suffix: str) -> pd.DataFrame:
    """One row per PDB: AF3 + Rosetta columns, suffixed (e.g. _pre, _post)."""
    af3 = load_af3_stats(epoch_dir)
    ros = load_rosetta_metrics(epoch_dir).rename(columns={
        "total_score": "rosetta_score",
        "rmsd": "rosetta_relax_rmsd",
        "packstat": "rosetta_packstat",
    })
    df = af3.merge(ros, on="pdb", how="outer")
    rename = {c: f"{c}_{suffix}" for c in df.columns if c != "pdb"}
    return df.rename(columns=rename)


# ---------------------------------------------------------------------------
# Per-run driver
# ---------------------------------------------------------------------------

def run_one(run_name: str, gen_root: str, work_root: Path,
            template: dict, s2s_repo: str,
            skip_pre: bool = False, skip_post: bool = False) -> pd.DataFrame:
    print(f"\n{'='*70}\nRun: {run_name}\n  gen_root: {gen_root}\n{'='*70}")

    run_dir = work_root / run_name
    pdb_dir = run_dir / "pdbs"
    pre_dir = run_dir / "as_generated"
    post_dir = run_dir / "redesigned"
    list_file = run_dir / "pdb_list.txt"

    pdbs = collect_generated_pdbs(gen_root)
    print(f"Found {len(pdbs)} generated PDB(s)")
    names = stage_pdb_dir(pdbs, pdb_dir)
    write_pdb_list(names, list_file)
    pdbs_in_staging = [pdb_dir / f"{n}.pdb" for n in names]

    common = {
        "pdb_dir": str(pdb_dir),
        "pdb_list_file": str(list_file),
        "checkpoint_dir": None,
        "epoch_list": None,
        "run_best": False,
    }

    # ---- Stage A: as-generated ----
    if not skip_pre:
        pre_dir.mkdir(parents=True, exist_ok=True)
        write_as_generated_fasta(
            pdbs_in_staging, pre_dir / "predicted_sequences.fasta"
        )
        cfg_pre = pre_dir / "config.yaml"
        write_pipeline_yaml(template, {**common,
            "output_dir": str(pre_dir),
            "run_inference": False,   # use the FASTA we just wrote
            "run_af3": True,
            "run_rosetta": True,
        }, cfg_pre)
        run_pipeline(s2s_repo, cfg_pre)
    else:
        print("Skipping as-generated stage (skip_pre=true)")

    # ---- Stage B: redesigned ----
    if not skip_post:
        post_dir.mkdir(parents=True, exist_ok=True)
        cfg_post = post_dir / "config.yaml"
        write_pipeline_yaml(template, {**common,
            "output_dir": str(post_dir),
            "run_inference": True,
            "run_af3": True,
            "run_rosetta": True,
        }, cfg_post)
        run_pipeline(s2s_repo, cfg_post)
    else:
        print("Skipping redesigned stage (skip_post=true)")

    pre_df = merge_stage(pre_dir, "pre")
    post_df = merge_stage(post_dir, "post")
    df = pre_df.merge(post_df, on="pdb", how="outer")
    df.insert(0, "run", run_name)

    # Per-PDB gap = post - pre. For scRMSD lower is better, so a NEGATIVE gap
    # is improvement; for sctm/plddt/ptm a POSITIVE gap is improvement; for
    # rosetta_score (Rosetta REU) lower is better.
    for col in ("scrmsd", "sctm", "plddt", "ptm",
                "rosetta_score", "rosetta_packstat"):
        a, b = f"{col}_pre", f"{col}_post"
        if a in df.columns and b in df.columns:
            df[f"{col}_gap"] = df[b] - df[a]
    return df


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [c for c in df.columns
                   if c.endswith(("_pre", "_post", "_gap"))]
    rows = []
    for run, sub in df.groupby("run"):
        row = {"run": run, "n": len(sub)}
        for c in metric_cols:
            vals = sub[c].dropna()
            row[f"{c}_mean"] = float(vals.mean()) if len(vals) else np.nan
            row[f"{c}_median"] = float(vals.median()) if len(vals) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Self-consistency pipeline (baseline vs contrastive)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", required=True,
                        help="Path to YAML driver config (see "
                             "self_consistency_config_example.yaml).")
    args = parser.parse_args()

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

    s2s_repo = cfg["struct2struct_repo"]
    work_root = Path(cfg["work_dir"]).resolve()
    work_root.mkdir(parents=True, exist_ok=True)

    # Pipeline-level template: everything shared by every per-run YAML the
    # script writes. Required keys: checkpoint, af3_*, rosetta_python,
    # pottsmpnn_dave_dir. Optional keys (opt_type, num_samples, ...) are
    # passed through verbatim.
    template = dict(cfg["pipeline_template"])
    runs = cfg["runs"]  # list of {name, gen_root}

    skip_pre = bool(cfg.get("skip_pre", False))
    skip_post = bool(cfg.get("skip_post", False))

    all_dfs = []
    for r in runs:
        df = run_one(r["name"], r["gen_root"], work_root,
                     template, s2s_repo,
                     skip_pre=skip_pre, skip_post=skip_post)
        all_dfs.append(df)

    full = pd.concat(all_dfs, ignore_index=True)
    out_csv = work_root / "self_consistency_per_pdb.csv"
    full.to_csv(out_csv, index=False)
    print(f"\nPer-PDB results -> {out_csv}")

    summary = summarize(full)
    out_sum = work_root / "self_consistency_summary.csv"
    summary.to_csv(out_sum, index=False)
    print(f"Per-run summary -> {out_sum}\n")
    with pd.option_context("display.max_columns", None,
                           "display.width", 200):
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
