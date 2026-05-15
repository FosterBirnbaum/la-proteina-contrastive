"""
Plot flow-matching training curves: baseline VAE latents vs contrastive VAE latents.

Produces a 3-panel figure suitable for a slide:

    [ total flow loss ]   [ CA loss ]   [ local-latents loss ]

Each panel shows:
    - Smoothed train loss (rolling mean over the noisy per-step values)
    - Validation loss (markers at each val checkpoint)

Both runs are overlaid in each panel.

Usage (from project root):
    python script_utils/plot_flow_curves.py
    python script_utils/plot_flow_curves.py --out flow_curves.png --window 500

Then drop the PNG straight into the slide deck.
"""

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--contrastive_csv",
        default="store/flow_contrastive_small/metrics.csv",
        help="Path to contrastive run metrics.csv",
    )
    p.add_argument(
        "--baseline_csv",
        default="store/flow_baseline_small/metrics.csv",
        help="Path to baseline run metrics.csv",
    )
    p.add_argument("--out", default="flow_curves.png")
    p.add_argument(
        "--window", type=int, default=500,
        help="Rolling-mean window (steps) for smoothing train loss",
    )
    p.add_argument(
        "--max_step", type=int, default=None,
        help="Truncate plots at this step (default: min(max_step) across runs)",
    )
    p.add_argument(
        "--log_y", action="store_true",
        help="Use log-scale on y-axes (helpful if losses span orders of magnitude)",
    )
    return p.parse_args()


# ── Column groups ─────────────────────────────────────────────────────────────
PANELS = [
    {
        "title": "Total flow loss",
        "train_col": "train/loss_step",
        "val_col":   "validation_loss/loss_step",
    },
    {
        "title": "CA-coordinate loss",
        "train_col": "train/loss_bb_ca_step",
        "val_col":   "validation_loss/loss_bb_ca_step",
    },
    {
        "title": "Local-latents loss",
        "train_col": "train/loss_local_latents_step",
        "val_col":   "validation_loss/loss_local_latents_step",
    },
]

COLORS = {
    "baseline":    "#1f77b4",  # blue
    "contrastive": "#d62728",  # red
}


def load_run(csv_path: str, label: str):
    """Load a metrics CSV and return a dict with train/val series for each panel column."""
    df = pd.read_csv(csv_path)
    # Some runs have step as int, some as float — coerce
    df = df.dropna(subset=["step"]).copy()
    df["step"] = df["step"].astype(int)
    return df


def rolling_mean_by_step(df: pd.DataFrame, col: str, window: int):
    """Return (steps, smoothed_values) for the named column.

    Uses an integer-step rolling mean. NaN rows are dropped first so the window
    operates on the (sparser) actual logged values.
    """
    sub = df[["step", col]].dropna()
    if len(sub) == 0:
        return np.array([]), np.array([])
    sub = sub.sort_values("step")
    smooth = sub[col].rolling(window=window, min_periods=max(1, window // 5)).mean()
    return sub["step"].to_numpy(), smooth.to_numpy()


def val_points(df: pd.DataFrame, col: str):
    sub = df[["step", col]].dropna().sort_values("step")
    # Many "_step" val rows are duplicates within one val pass; collapse to
    # one point per unique step so the plot isn't cluttered.
    sub = sub.groupby("step", as_index=False)[col].mean()
    return sub["step"].to_numpy(), sub[col].to_numpy()


def main():
    args = parse_args()

    df_b = load_run(args.baseline_csv, "baseline")
    df_c = load_run(args.contrastive_csv, "contrastive")

    max_step = args.max_step or min(df_b["step"].max(), df_c["step"].max())
    df_b = df_b[df_b["step"] <= max_step]
    df_c = df_c[df_c["step"] <= max_step]

    print(f"Plotting up to step {max_step:,}")
    print(f"  Baseline rows:    {len(df_b):,}")
    print(f"  Contrastive rows: {len(df_c):,}")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharex=True)

    for ax, panel in zip(axes, PANELS):
        for label, df in [("baseline", df_b), ("contrastive", df_c)]:
            color = COLORS[label]

            # Smoothed training loss
            xs, ys = rolling_mean_by_step(df, panel["train_col"], args.window)
            if len(xs) > 0:
                ax.plot(
                    xs, ys, color=color, linewidth=1.6, alpha=0.95,
                    label=f"{label} (train)",
                )

            # Validation loss as markers + thin connecting line
            vx, vy = val_points(df, panel["val_col"])
            if len(vx) > 0:
                ax.plot(
                    vx, vy, color=color, linewidth=0.8, alpha=0.55,
                    linestyle="--",
                )
                # Subsample markers if there are too many
                stride = max(1, len(vx) // 30)
                ax.scatter(
                    vx[::stride], vy[::stride], color=color, s=22,
                    edgecolor="white", linewidth=0.6, zorder=5,
                    label=f"{label} (val)",
                )

        ax.set_title(panel["title"], fontsize=13)
        ax.set_xlabel("Training step")
        ax.grid(True, alpha=0.25)
        if args.log_y:
            ax.set_yscale("log")
        ax.set_xlim(0, max_step)

    axes[0].set_ylabel("Loss")
    # Single legend on the right-most panel to avoid clutter
    axes[-1].legend(loc="upper right", fontsize=9, framealpha=0.9)

    fig.suptitle(
        "Flow matching on baseline vs. contrastive VAE latents (small PDB subset)",
        fontsize=14, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    print(f"Saved {args.out}")

    # ── Quick text summary for the slide caption ──────────────────────────────
    print("\n=== Final-window summary (last ~200 steps of training) ===")
    for label, df in [("baseline", df_b), ("contrastive", df_c)]:
        print(f"\n{label}:")
        for panel in PANELS:
            sub = df[["step", panel["train_col"]]].dropna().sort_values("step")
            train_final = sub[panel["train_col"]].tail(200).mean()
            sub_v = df[["step", panel["val_col"]]].dropna().sort_values("step")
            val_final = sub_v[panel["val_col"]].tail(50).mean()
            print(
                f"  {panel['title']:<22}  "
                f"train={train_final:7.4f}   val={val_final:7.4f}"
            )


if __name__ == "__main__":
    main()
