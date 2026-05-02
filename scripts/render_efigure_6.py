#!/usr/bin/env python3
"""Render eFigure 6 — calibration assessment for Amendment A13.

5-panel figure (2x3 grid; bottom-right empty):
    [1] T1 reliability diagram (binary)
    [2] T5 reliability diagram (binary)
    [3] T7 reliability diagram (binary)
    [4] T3 IPCW Cox calibration at 24 mo (per-patient)
    [5] T4 IPCW Cox calibration at 36 mo (per-patient)

300 dpi PNG + vector PDF. Style matches F1–F4.

Reads:
    results/calibration/reliability_bins_t{1,5,7}.parquet
    results/calibration/cox_calibration_t{3,4}.parquet

Outputs:
    results/figures/efigure_6_calibration.png
    results/figures/efigure_6_calibration.pdf
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
INPUT_DIR = ROOT / "results" / "calibration"
OUTPUT_DIR = ROOT / "results" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Consistent FM colour mapping with F1-F4
FM_COLORS = {
    "fmcib": "#1f77b4",
    "ct_fm": "#2ca02c",
    "biomedclip": "#9467bd",
    "rad_dino": "#d62728",
    "dinov2": "#ff7f0e",
    "random_init": "#7f7f7f",
}
FM_DISPLAY = {
    "fmcib": "FMCIB",
    "ct_fm": "CT-FM",
    "biomedclip": "BiomedCLIP",
    "rad_dino": "RAD-DINO",
    "dinov2": "DINOv2",
    "random_init": "random_init",
}

PANEL_ORDER = ["t1", "t5", "t7", "t3", "t4"]
PANEL_TITLES = {
    "t1": "T1 — AutoPET-I FDG patches (binary)",
    "t5": "T5 — PSMA zero-shot (binary)",
    "t7": "T7 — ACRIN-NSCLC 2y OS (binary)",
    "t3": "T3 — HECKTOR RFS (Cox, 24 mo)",
    "t4": "T4 — NSCLC OS (Cox, 24 mo)",
}


def plot_reliability(ax, task: str):
    fp = INPUT_DIR / f"reliability_bins_{task}.parquet"
    if not fp.exists():
        ax.text(0.5, 0.5, f"{task}: no calibration data",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title(PANEL_TITLES[task]); return
    df = pd.read_parquet(fp)
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="_perfect")
    for fm in sorted(df["fm"].unique()):
        sub = df[df["fm"] == fm].sort_values("mean_p")
        c = FM_COLORS.get(fm, "#444")
        ax.plot(sub["mean_p"], sub["obs_p"], "-o", color=c, ms=3, lw=1,
                label=FM_DISPLAY.get(fm, fm))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed event rate")
    ax.set_title(PANEL_TITLES[task], fontsize=10)
    ax.grid(alpha=0.3)


def plot_cox_calibration(ax, task: str, horizon_months: int):
    fp = INPUT_DIR / f"cox_calibration_{task}.parquet"
    if not fp.exists():
        ax.text(0.5, 0.5, f"{task}: no calibration data",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title(PANEL_TITLES[task]); return
    df = pd.read_parquet(fp)
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="_perfect")
    for fm in sorted(df["fm"].unique()):
        sub = df[df["fm"] == fm].sort_values("mean_pred_surv")
        c = FM_COLORS.get(fm, "#444")
        ax.plot(sub["mean_pred_surv"], sub["obs_surv_km"], "-o", color=c, ms=3, lw=1,
                label=FM_DISPLAY.get(fm, fm))
    # Tight axis range matching cohort survival pattern (avoid empty 0-0.5 for high-survival cohorts)
    if len(df):
        all_pred = df["mean_pred_surv"].values
        all_obs = df["obs_surv_km"].values
        lo = max(0.0, min(all_pred.min(), all_obs.min()) - 0.05)
        hi = min(1.0, max(all_pred.max(), all_obs.max()) + 0.05)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    else:
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel(f"Mean predicted survival at {horizon_months} mo")
    ax.set_ylabel(f"KM observed survival")
    ax.set_title(PANEL_TITLES[task], fontsize=10)
    ax.grid(alpha=0.3)


def main():
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.5))
    plot_reliability(axes[0][0], "t1")
    plot_reliability(axes[0][1], "t5")
    plot_reliability(axes[0][2], "t7")
    plot_cox_calibration(axes[1][0], "t3", 24)
    plot_cox_calibration(axes[1][1], "t4", 24)
    axes[1][2].axis("off")  # legend only

    # Shared legend in the bottom-right
    handles, labels = axes[0][0].get_legend_handles_labels()
    if not handles:
        for ax in [axes[0][1], axes[0][2], axes[1][0], axes[1][1]]:
            h, l = ax.get_legend_handles_labels()
            if h:
                handles, labels = h, l
                break
    if handles:
        # Add the perfect-calibration dashed reference
        from matplotlib.lines import Line2D
        handles = [Line2D([0], [0], color="k", lw=1, ls="--", alpha=0.5,
                          label="Perfect calibration")] + handles
        labels = ["Perfect calibration"] + labels
        axes[1][2].legend(handles, labels, loc="center", fontsize=10, frameon=True)

    fig.suptitle("eFigure 6. Calibration assessment (Amendment A13)",
                  fontsize=12, fontweight="bold", y=1.00)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    png_path = OUTPUT_DIR / "efigure_6_calibration.png"
    pdf_path = OUTPUT_DIR / "efigure_6_calibration.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"✓ Wrote {png_path} and {pdf_path}")


if __name__ == "__main__":
    main()
