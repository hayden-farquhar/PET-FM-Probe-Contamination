# %% [markdown]
# # PET-FM-Bench manuscript figures (F1–F4) — publication-quality renders
#
# Single notebook that produces all four manuscript figures from the public
# Phase 5 freeze CSVs. Designed for Kaggle (CPU kernel; no GPU required).
#
# **Inputs.** Attach the Kaggle dataset `pet-fm-bench-formal-probe-results-v1`
# (or set `INPUT_DIR` to a local path containing the same CSVs):
#
# - `fm_task_matrix.csv` — primary-metric point estimate per (FM × task) cell
# - `all_probe_results.csv` — patient-clustered bootstrap CIs per cell
# - `t6_test_retest_results.csv`, `t9_test_retest_results.csv` — H6 specifics
# - `contamination_audit.csv` — Phase 2 v2 contamination tier matrix (54 cells)
#
# **Outputs (PNG @ 300 dpi + PDF vector for each figure):**
#
# - `figure_1_heatmap.{png,pdf}`
# - `figure_2_contamination.{png,pdf}`
# - `figure_3_h6_test_retest.{png,pdf}`
# - `figure_4_forest.{png,pdf}`
#
# **Reproducibility.** Style and palette are set once at the top; every render
# call goes through the same helpers. Bootstrap CIs read from the freeze CSVs
# where available; the fallback is documented but not exercised on the freeze.

# %% [markdown]
# ## 1. Imports and publication-style configuration

# %%
from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.lines import Line2D

warnings.filterwarnings("ignore", category=FutureWarning)


# Paths — Kaggle defaults; override locally by setting PET_FM_BENCH_INPUT_DIR.
# The Phase 5 freeze can be uploaded to Kaggle either as a flat directory of
# CSVs or with each CSV in its own subfolder (results/, phase_2_*/v2/, etc.).
# To stay robust to either layout we build a *file map* once at startup that
# remembers where each canonical CSV lives, and every loader queries that map
# rather than assuming a single shared directory.
SENTINEL_FILE = "fm_task_matrix.csv"
# Required files: missing any of these is fatal because F1, F3, or F4 cannot
# be rendered without them.
REQUIRED_FILES = {
    "fm_task_matrix.csv",
    "all_probe_results.csv",
    "t6_test_retest_results.csv",
    "t9_test_retest_results.csv",
}
# Optional files: missing them only affects F2 (contamination matrix). The
# Phase 2 v2 freeze lives in a different OSF folder than the Phase 5 results,
# so a Kaggle dataset that mirrors only Phase 5 will not include this CSV.
OPTIONAL_FILES = {
    "contamination_audit.csv",
}
EXPECTED_FILES = REQUIRED_FILES | OPTIONAL_FILES


def _resolve_freeze_files() -> Tuple[Path, Dict[str, Path]]:
    """Build a {filename → path} map for every freeze CSV the notebook needs.

    Resolution order:
      1. ``PET_FM_BENCH_INPUT_DIR`` if it exists and contains the sentinel
         file directly — flat layout, preferred when running locally.
      2. Recursive scan of ``/kaggle/input/`` for each expected CSV — handles
         both single-subfolder uploads and OSF-mirrored multi-subfolder
         uploads (e.g. ``results/fm_task_matrix.csv`` plus
         ``phase_2_freeze_contamination_audit/v2/contamination_audit.csv``).

    Returns the directory that *appears* to be the primary input root (used
    only for the diagnostic print) and the file map used by every loader.
    """
    env_path = os.environ.get("PET_FM_BENCH_INPUT_DIR")
    primary_root = Path(env_path) if env_path else Path("/kaggle/input")

    file_map: Dict[str, Path] = {}

    # Flat layout shortcut: every expected file directly under the env path.
    if env_path:
        env = Path(env_path)
        if env.is_dir():
            flat_hits = {f: env / f for f in EXPECTED_FILES if (env / f).is_file()}
            file_map.update(flat_hits)

    # Recursive search for anything not yet found.
    kaggle_input = Path("/kaggle/input")
    search_roots = [r for r in (primary_root, kaggle_input) if r.is_dir()]
    for missing in EXPECTED_FILES - file_map.keys():
        for root in search_roots:
            hits = list(root.rglob(missing))
            if hits:
                # Prefer the shallowest hit (closest to root) when ambiguous —
                # OSF deposits sometimes contain both a flat working copy and
                # a versioned freeze copy under a deeper path.
                hits.sort(key=lambda p: len(p.parts))
                file_map[missing] = hits[0]
                break

    if SENTINEL_FILE in file_map:
        primary_root = file_map[SENTINEL_FILE].parent

    missing_required = REQUIRED_FILES - file_map.keys()
    missing_optional = OPTIONAL_FILES - file_map.keys()

    if missing_required:
        print("ERROR: could not locate the following REQUIRED freeze CSVs:")
        for m in sorted(missing_required):
            print(f"  - {m}")
        if kaggle_input.is_dir():
            print("\nContents of /kaggle/input/ (top three levels):")
            for path in sorted(kaggle_input.rglob("*"))[:80]:
                rel = path.relative_to(kaggle_input)
                if len(rel.parts) <= 3:
                    print(f"  {rel}")
        raise FileNotFoundError(
            f"Missing {len(missing_required)} required freeze CSV(s): "
            f"{sorted(missing_required)}. Attach the Kaggle dataset "
            f"`pet-fm-bench-formal-probe-results-v1` (Add Input → Datasets "
            f"→ search) or set PET_FM_BENCH_INPUT_DIR to a directory "
            f"containing the freeze CSVs."
        )

    if missing_optional:
        print(
            f"\nWARNING: optional freeze CSV(s) not found: "
            f"{sorted(missing_optional)}.  "
            f"Figure 2 (contamination tier matrix) will be skipped.  "
            f"To render Figure 2, upload `contamination_audit.csv` from OSF "
            f"aqmkb / `phase_2_freeze_contamination_audit/v2/` to the same "
            f"Kaggle dataset and re-run this notebook."
        )

    return primary_root, file_map


INPUT_DIR, FREEZE_FILES = _resolve_freeze_files()
OUTPUT_DIR = Path(os.environ.get("PET_FM_BENCH_OUTPUT_DIR", "/kaggle/working"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Resolved freeze CSV locations:")
for name in sorted(FREEZE_FILES):
    print(f"  {name:<32s} → {FREEZE_FILES[name]}")


def apply_publication_style() -> None:
    """Set matplotlib rcParams for publication-quality figures.

    Embeds TrueType fonts (pdf.fonttype=42) so journals can extract text from
    the PDF; sets sans-serif stack with Arial/Helvetica preference; pins a
    consistent baseline of font sizes, line widths, and DPI.
    """
    mpl.rcParams.update({
        # Vector-friendly font handling for journal submission.
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        # Sans-serif stack (falls through to DejaVu Sans on Kaggle).
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        # Sizes calibrated for double-column figure at ~180 mm wide.
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.titlesize": 11,
        # Lines and grids.
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "grid.linewidth": 0.5,
        "grid.alpha": 0.4,
        "axes.grid": False,
        # DPI — render at 300 for PNG; PDF is vector.
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })


apply_publication_style()


# %% [markdown]
# ## 2. Canonical orderings, labels, and colour palette
#
# A single source of truth for FM names, task IDs, and colours. Every figure
# pulls from these; no figure-specific re-spellings.

# %%
# Foundation models: ordered so the H6 winners (3D medical CT) sit at the top
# of all matrix-style figures. random_init and IBSI sit at the bottom.
FM_ORDER: List[str] = [
    "fmcib",
    "ct_fm",
    "biomedclip",
    "rad_dino",
    "dinov2",
    "ibsi_radiomics_baseline",
    "random_init",
]

# Display labels with consistent capitalisation.
FM_LABELS: Dict[str, str] = {
    "fmcib": "FMCIB",
    "ct_fm": "CT-FM",
    "biomedclip": "BiomedCLIP",
    "rad_dino": "RAD-DINO",
    "dinov2": "DINOv2",
    "ibsi_radiomics_baseline": "IBSI radiomics",
    "random_init": "random_init (10-seed median)",
}

# Short labels for tight panels (forest plot rows).
FM_LABELS_SHORT: Dict[str, str] = {
    "fmcib": "FMCIB",
    "ct_fm": "CT-FM",
    "biomedclip": "BiomedCLIP",
    "rad_dino": "RAD-DINO",
    "dinov2": "DINOv2",
    "ibsi_radiomics_baseline": "IBSI",
    "random_init": "random_init",
}

# Colour palette: H6 winners (3D medical CT) in blues; H6 losers (2D non-medical
# / biomedical-caption) in warm tones; IBSI in dark grey; random_init in pale
# grey. Chosen from the colour-blind-safe Wong palette and adjusted for print
# legibility. The grouping is intentional: any reviewer who scans Figure 3 or
# Figure 4 should immediately see that the two blue bars cluster above the
# baseline while the warm bars cluster below.
FM_COLOURS: Dict[str, str] = {
    "fmcib": "#1f4e79",          # dark blue — H6 winner
    "ct_fm": "#5b9bd5",          # mid blue — H6 winner
    "biomedclip": "#ed7d31",     # orange — H6 loser
    "rad_dino": "#c00000",       # red — H6 loser, chest-X-ray pretraining
    "dinov2": "#bf9000",         # mustard — H6 loser, ImageNet pretraining
    "ibsi_radiomics_baseline": "#404040",  # dark grey — handcrafted baseline
    "random_init": "#a6a6a6",    # light grey — untrained control
}

# Tasks: ordered by metric family so within-family comparisons are visually
# adjacent. AUROC tasks first; survival c-index tasks; CCC tasks last.
TASK_ORDER: List[str] = [
    "t1", "t2", "t2_gtvp_only", "t5", "t7", "t8",  # AUROC
    "t3", "t4",                                     # c-index
    "t6", "t9",                                     # Lin's CCC
]

# Task display labels: two-line, dataset shorthand only. Cohort sizes go in
# the figure caption rather than the tick label so the x-axis stays legible.
TASK_LABELS: Dict[str, str] = {
    "t1": "T1\nAutoPET-I",
    "t2": "T2\nHECKTOR",
    "t2_gtvp_only": "T2-GTVp\n(HECKTOR)",
    "t5": "T5\nPSMA (0-shot)",
    "t7": "T7\nACRIN",
    "t8": "T8\nLung-PET-CT",
    "t3": "T3\nHECKTOR RFS",
    "t4": "T4\nNSCLC-RG",
    "t6": "T6\nRIDER-Lung",
    "t9": "T9\nVienna QUADRA",
}

TASK_LABELS_SHORT: Dict[str, str] = {
    "t1": "T1", "t2": "T2", "t2_gtvp_only": "T2-GTVp",
    "t3": "T3", "t4": "T4", "t5": "T5",
    "t6": "T6", "t7": "T7", "t8": "T8", "t9": "T9",
}

# Vertical separator positions in the matrix figures (between metric families).
TASK_FAMILY_BREAKS = [6, 8]  # after T8 (AUROC family), after T4 (c-index family)


# %% [markdown]
# ## 3. Data loaders
#
# Defensive loaders that map the freeze CSV column names to canonical names
# (the freeze uses lowercase task IDs; some scripts emit uppercase). Print a
# small head and shape so reviewers running this notebook can sanity-check.

# %%
def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase column names; map common variants to canonical task IDs."""
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    rename = {
        "t2_gtvp": "t2_gtvp_only",
        "t2-gtvp": "t2_gtvp_only",
        "t2_gtvponly": "t2_gtvp_only",
    }
    df = df.rename(columns=rename)
    return df


def load_fm_task_matrix() -> pd.DataFrame:
    """Load the (FM × task) primary-metric matrix indexed by FM."""
    df = _normalise_columns(pd.read_csv(FREEZE_FILES["fm_task_matrix.csv"]))
    fm_col = "fm" if "fm" in df.columns else df.columns[0]
    df = df.set_index(fm_col)
    df.index = df.index.str.lower().str.strip()
    return df


def _canonicalise_value_columns(df: pd.DataFrame, value_target: str) -> pd.DataFrame:
    """Map the freeze's (metric_name + value + ci_low + ci_high) schema onto
    this notebook's vocabulary (`value_target` for the point estimate; `lower`
    / `upper` for the CI).

    The Phase 5 freeze CSVs have two columns the name 'metric' could refer to:
      - `metric` — the *name* of the metric (e.g. "lin_ccc", "auroc", "c_index")
      - `value` — the numeric point estimate
    We drop the name-string column and rename `value` → `value_target`.
    """
    df = df.copy()
    if "metric" in df.columns and "value" in df.columns:
        # `metric` holds the metric-name string; drop it and use `value`.
        df = df.drop(columns=["metric"])
    if "value" in df.columns and value_target not in df.columns:
        df = df.rename(columns={"value": value_target})
    if "ci_low" in df.columns and "lower" not in df.columns:
        df = df.rename(columns={"ci_low": "lower"})
    if "ci_high" in df.columns and "upper" not in df.columns:
        df = df.rename(columns={"ci_high": "upper"})
    # Coerce the numeric columns just in case any reader interpreted them as
    # object dtype (would break `-x` arithmetic in F3's sort key).
    for col in (value_target, "lower", "upper"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_probe_results() -> pd.DataFrame:
    """Load per-cell point estimates and bootstrap CIs.

    Returns long-format with canonical columns: fm, task, metric, lower, upper.
    """
    df = _normalise_columns(pd.read_csv(FREEZE_FILES["all_probe_results.csv"]))
    df["fm"] = df["fm"].str.lower().str.strip()
    df["task"] = df["task"].str.lower().str.strip()
    df = _canonicalise_value_columns(df, value_target="metric")
    return df


def load_test_retest_csvs() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load the per-task T6 and T9 test-retest CSVs (CCC + bootstrap CI)."""
    t6 = _normalise_columns(pd.read_csv(FREEZE_FILES["t6_test_retest_results.csv"]))
    t9 = _normalise_columns(pd.read_csv(FREEZE_FILES["t9_test_retest_results.csv"]))
    out = []
    for d in (t6, t9):
        d["fm"] = d["fm"].str.lower().str.strip()
        if "task" in d.columns:
            d["task"] = d["task"].str.lower().str.strip()
        d = _canonicalise_value_columns(d, value_target="ccc")
        out.append(d)
    return out[0], out[1]


def load_contamination_audit() -> pd.DataFrame:
    """Load the 54-cell contamination audit (FM × task → tier + overlap %)."""
    df = _normalise_columns(pd.read_csv(FREEZE_FILES["contamination_audit.csv"]))
    df["fm"] = df["fm"].str.lower().str.strip()
    df["task"] = df["task"].str.lower().str.strip()
    return df


# %% [markdown]
# ## 4. Figure 1 — Cross-FM × cross-task primary-metric heatmap
#
# Per-task z-score colouring (so within-task differences show up despite metrics
# spanning AUROC, c-index, and CCC). Cell text shows the actual point estimate
# to three decimals so reviewers can read off numbers without consulting a
# table. Vertical separator lines mark the AUROC / c-index / CCC family
# boundaries.

# %%
def render_figure_1(matrix: pd.DataFrame, output_stem: str = "figure_1_heatmap") -> Path:
    # Reindex to canonical FM and task ordering.
    mat = matrix.reindex(FM_ORDER)[TASK_ORDER]

    # Per-task z-score (column-wise) for the colour scale only.
    col_mean = mat.mean(axis=0, skipna=True)
    col_std = mat.std(axis=0, ddof=0, skipna=True).replace(0, np.nan)
    z = (mat - col_mean) / col_std

    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    cmap = mpl.colormaps["viridis"]
    vmin, vmax = -2.0, 2.0
    im = ax.imshow(z.values, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)

    def _text_colour_for(z_value: float) -> str:
        """Pick black or white based on the actual luminance of the cell colour.

        The viridis colormap passes through teal/green at the midpoint, where
        neither pure white nor pure black is high-contrast. Computing the
        relative luminance of the cell's RGB and thresholding at 0.55 gives
        readable text on every cell, including the mid-range.
        """
        if pd.isna(z_value):
            return "#444444"
        norm_v = max(0.0, min(1.0, (z_value - vmin) / (vmax - vmin)))
        r, g, b, _ = cmap(norm_v)
        luminance = 0.299 * r + 0.587 * g + 0.114 * b
        return "black" if luminance > 0.55 else "white"

    # Tick labels.
    ax.set_xticks(range(len(TASK_ORDER)))
    ax.set_xticklabels([TASK_LABELS[t] for t in TASK_ORDER], fontsize=8)
    ax.tick_params(axis="x", pad=2)
    ax.set_yticks(range(len(FM_ORDER)))
    ax.set_yticklabels(
        [FM_LABELS[f] for f in FM_ORDER],
        fontsize=8.5,
        fontweight="bold",
    )

    # Cell annotations: point estimate, luminance-aware text colour.
    for i in range(len(FM_ORDER)):
        for j in range(len(TASK_ORDER)):
            v = mat.iloc[i, j]
            zv = z.iloc[i, j]
            if pd.isna(v):
                text = "—"
                colour = "#888888"
            else:
                text = f"{v:.3f}"
                colour = _text_colour_for(zv)
            ax.text(j, i, text, ha="center", va="center", fontsize=7.5,
                    color=colour, fontweight="medium")

    # Vertical separators between metric families.
    for x in TASK_FAMILY_BREAKS:
        ax.axvline(x - 0.5, color="white", linewidth=2.4)

    # Family labels above the panel.
    fam_centres = {
        "AUROC (classification)": (0 + 5) / 2,
        "c-index (survival)": (6 + 7) / 2,
        "Lin's CCC (test–retest)": (8 + 9) / 2,
    }
    for label, x in fam_centres.items():
        ax.text(x, -1.6, label, ha="center", va="bottom",
                fontsize=8.5, fontweight="bold")

    # Colour bar.
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.012)
    cbar.set_label("Per-task z-score (point estimate)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    # No box around the panel; ticks only on the bottom and left.
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    ax.set_title(
        "Figure 1.  Cross-FM × cross-task primary-metric matrix.  "
        "Cells annotated with point estimate; colour = per-task z-score.",
        loc="left", pad=18, fontsize=9.5,
    )

    fig.tight_layout()
    return _save(fig, output_stem)


# %% [markdown]
# ## 5. Figure 2 — Contamination tier matrix (Phase 2 v2 audit)
#
# Discrete categorical colour scheme. Tier 4 (caption-scan proxy) is unused in
# the v2 audit and therefore absent from the legend. Tier 1+2 cells are
# annotated with overlap fraction; Tier 3 and Tier 5 cells are unannotated to
# avoid clutter. random_init is included as a row to make the "declared clean
# by construction" rationale visible.

# %%
TIER_TO_INT = {1: 1, 2: 2, 3: 3, 5: 4}  # collapse Tier 5 to position 4 for cmap
TIER_COLOURS = ["#c00000", "#ed7d31", "#ffd966", "#9bc36b"]  # red, orange, yellow, green
TIER_NAMES = {
    1: "Tier 1 — study-UID match",
    2: "Tier 2 — patient-ID match (same collection)",
    3: "Tier 3 — institutional-context proxy",
    5: "Tier 5 — declared clean by construction",
}


def _contamination_pivot(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Reshape the long-format audit into tier and overlap-fraction matrices.

    The audit CSV enumerates the 9 base tasks (T1–T9) but does not separately
    list T2-GTVp because it shares the HECKTOR 2025 cohort with T2 and
    inherits T2's contamination profile (Methods §"Contamination audit"). We
    backfill the T2-GTVp column from T2 so the rendered matrix has the same
    column structure as Figure 1, and we ensure every task in TASK_ORDER has
    a column even if the audit omits it (defaulting to Tier 5 with zero
    overlap, which is the audit's by-construction rule for omitted cells).
    """
    fm_subset = [f for f in FM_ORDER if f in df["fm"].unique()]
    pivot_tier = (
        df.pivot_table(index="fm", columns="task", values="tier", aggfunc="first")
          .reindex(fm_subset)
    )
    overlap_col = next(
        (c for c in ("overlap_fraction", "overlap_pct", "overlap")
         if c in df.columns),
        None,
    )
    if overlap_col is None:
        pivot_overlap = pd.DataFrame(
            np.nan, index=pivot_tier.index, columns=pivot_tier.columns
        )
    else:
        pivot_overlap = (
            df.pivot_table(index="fm", columns="task", values=overlap_col,
                           aggfunc="first")
              .reindex(fm_subset)
        )
        # Normalise to a [0, 1] fraction if values are reported as percentages.
        if pivot_overlap.max(numeric_only=True).max() > 1.5:
            pivot_overlap = pivot_overlap / 100.0

    # Backfill T2-GTVp from T2 (shared cohort) when the audit omits it.
    if "t2_gtvp_only" not in pivot_tier.columns and "t2" in pivot_tier.columns:
        pivot_tier["t2_gtvp_only"] = pivot_tier["t2"]
        pivot_overlap["t2_gtvp_only"] = pivot_overlap["t2"]
    # Ensure every TASK_ORDER column exists; missing ones default to Tier 5.
    for task in TASK_ORDER:
        if task not in pivot_tier.columns:
            pivot_tier[task] = 5
            pivot_overlap[task] = 0.0
    pivot_tier = pivot_tier[TASK_ORDER]
    pivot_overlap = pivot_overlap[TASK_ORDER]
    return pivot_tier, pivot_overlap


def render_figure_2(audit: pd.DataFrame, output_stem: str = "figure_2_contamination") -> Path:
    pivot_tier, pivot_overlap = _contamination_pivot(audit)

    # For tasks where the audit was Tier 5 by construction across all FMs and
    # the freeze CSV omits those rows, fill missing cells with Tier 5.
    pivot_tier = pivot_tier.fillna(5).astype(int)

    # Map tier values to colour indices.
    coded = pivot_tier.replace(TIER_TO_INT).values
    cmap = ListedColormap(TIER_COLOURS)
    norm = BoundaryNorm([0.5, 1.5, 2.5, 3.5, 4.5], cmap.N)

    fig, ax = plt.subplots(figsize=(10.5, 3.6))
    ax.imshow(coded, aspect="auto", cmap=cmap, norm=norm)

    fm_rows = list(pivot_tier.index)
    ax.set_yticks(range(len(fm_rows)))
    ax.set_yticklabels(
        [FM_LABELS[f] for f in fm_rows],
        fontsize=8.5,
        fontweight="bold",
    )
    ax.set_xticks(range(len(TASK_ORDER)))
    ax.set_xticklabels([TASK_LABELS_SHORT[t] for t in TASK_ORDER], fontsize=8.5)
    ax.tick_params(top=False, bottom=True, left=True, right=False)

    # Annotate Tier 1+2 cells with overlap %.
    for i, fm in enumerate(fm_rows):
        for j, task in enumerate(TASK_ORDER):
            tier = int(pivot_tier.iloc[i, j])
            if tier in (1, 2):
                frac = pivot_overlap.iloc[i, j]
                annotation = "" if pd.isna(frac) else f"{frac * 100:.0f}%"
                ax.text(j, i, annotation, ha="center", va="center",
                        fontsize=7.5, color="white", fontweight="bold")
            elif tier == 3:
                ax.text(j, i, "·", ha="center", va="center",
                        fontsize=10, color="#404040")

    # Cell borders for visual separation.
    for x in range(coded.shape[1] + 1):
        ax.axvline(x - 0.5, color="white", linewidth=0.6)
    for y in range(coded.shape[0] + 1):
        ax.axhline(y - 0.5, color="white", linewidth=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    # Categorical legend on the right.
    legend_handles = [
        mpatches.Patch(color=TIER_COLOURS[0], label=TIER_NAMES[1]),
        mpatches.Patch(color=TIER_COLOURS[1], label=TIER_NAMES[2]),
        mpatches.Patch(color=TIER_COLOURS[2], label=TIER_NAMES[3]),
        mpatches.Patch(color=TIER_COLOURS[3], label=TIER_NAMES[5]),
    ]
    leg = ax.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        frameon=False,
        fontsize=8,
        title="Audit tier",
        title_fontsize=8.5,
    )
    leg.get_title().set_fontweight("bold")

    ax.set_title(
        "Figure 2.  Contamination audit matrix (Phase 2 v2; 54 cells, 83% Tier 5).  "
        "Tier 1+2 cells annotated with overlap %.",
        loc="left", pad=10, fontsize=9.5,
    )

    fig.tight_layout()
    return _save(fig, output_stem)


# %% [markdown]
# ## 6. Figure 3 — H6 headline: test–retest CCC vs IBSI radiomics
#
# Two side-by-side panels (T6, T9). FMs ordered by point estimate within each
# panel. IBSI baseline rendered as a horizontal **shaded band** (95% bootstrap
# CI) — not a dashed line, per the v2 evaluation. Bars are coloured by FM
# category (3D medical CT = blue; 2D non-medical = warm); random_init has
# diagonal hatching to distinguish the untrained control.
#
# Bar height = primary-metric point estimate; whiskers = patient-clustered
# bootstrap 95% CI from `all_probe_results.csv`. For the random_init row the
# whisker is the 10-seed range.

# %%
def _read_ccc_with_ci(df: pd.DataFrame) -> pd.DataFrame:
    """Pass-through. Test-retest CSVs are already canonicalised by the loader
    (`_canonicalise_value_columns(value_target='ccc')`); kept for backwards
    compatibility with older callers.
    """
    return df


def render_figure_3(t6: pd.DataFrame, t9: pd.DataFrame,
                    output_stem: str = "figure_3_h6_test_retest") -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8), sharey=True)

    panels = [
        ("T6 — RIDER-Lung\ncancer-patient test–retest (n = 16 retest pairs)", t6, axes[0]),
        ("T9 — Vienna QUADRA\nhealthy-control test–retest (n = 48 retest pairs)", t9, axes[1]),
    ]

    bar_fms = [f for f in FM_ORDER if f != "ibsi_radiomics_baseline"]

    for panel_title, df_panel, ax in panels:
        df_panel = _read_ccc_with_ci(df_panel)

        # Pull out IBSI band (point estimate + bootstrap CI).
        ibsi_row = df_panel[df_panel["fm"] == "ibsi_radiomics_baseline"]
        if len(ibsi_row) == 0:
            ibsi_point = float("nan")
            ibsi_lower = ibsi_upper = float("nan")
        else:
            ibsi_point = float(ibsi_row["ccc"].iloc[0])
            ibsi_lower = float(ibsi_row["lower"].iloc[0]) if "lower" in df_panel.columns else ibsi_point
            ibsi_upper = float(ibsi_row["upper"].iloc[0]) if "upper" in df_panel.columns else ibsi_point

        # Order bars by point estimate descending; random_init last.
        df_bars = df_panel[df_panel["fm"].isin(bar_fms)].copy()
        df_bars["sort_key"] = df_bars["fm"].apply(
            lambda f: (1 if f == "random_init" else 0, -df_bars.loc[df_bars["fm"] == f, "ccc"].values[0])
        )
        df_bars = df_bars.sort_values("sort_key").reset_index(drop=True)

        x = np.arange(len(df_bars))
        heights = df_bars["ccc"].values
        lowers = (heights - df_bars["lower"].values) if "lower" in df_bars.columns else np.zeros_like(heights)
        uppers = (df_bars["upper"].values - heights) if "upper" in df_bars.columns else np.zeros_like(heights)
        colours = [FM_COLOURS[f] for f in df_bars["fm"]]
        hatches = ["//" if f == "random_init" else "" for f in df_bars["fm"]]

        # IBSI shaded band first (so bars draw over it). Numerical IBSI value
        # goes in the figure caption rather than as an in-panel annotation;
        # the legend identifies the dashed line + shaded band.
        if not np.isnan(ibsi_point):
            ax.axhspan(ibsi_lower, ibsi_upper, color="#404040", alpha=0.13,
                       zorder=1, label="_nolegend_")
            ax.axhline(ibsi_point, color="#404040", linewidth=1.2,
                       linestyle="--", zorder=2)

        # Bars with whiskers.
        bars = ax.bar(
            x, heights, width=0.72,
            color=colours,
            edgecolor="white", linewidth=0.8,
            zorder=3,
        )
        for bar, hatch in zip(bars, hatches):
            if hatch:
                bar.set_hatch(hatch)
                bar.set_edgecolor("#666666")
        ax.errorbar(
            x, heights, yerr=[lowers, uppers],
            fmt="none", ecolor="#1a1a1a",
            capsize=3, linewidth=1.0, zorder=4,
        )

        # Per-bar value annotation, placed above the upper whisker tip so the
        # error bar never passes through the digits.
        for xi, h, up in zip(x, heights, uppers):
            top = h + up + 0.015
            ax.text(xi, top, f"{h:.3f}",
                    ha="center", va="bottom", fontsize=7.5, color="#1a1a1a")

        ax.set_xticks(x)
        ax.set_xticklabels(
            [FM_LABELS_SHORT[f] for f in df_bars["fm"]],
            rotation=22, ha="right", fontsize=8.5,
        )
        ax.set_ylim(-0.05, 1.0)
        ax.set_yticks(np.arange(0.0, 1.01, 0.1))
        ax.axhline(0, color="#888888", linewidth=0.4, zorder=0)
        ax.grid(axis="y", linestyle=":", alpha=0.45, zorder=0)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.set_title(panel_title, fontsize=9.5, pad=8)

    axes[0].set_ylabel("Lin's concordance correlation coefficient (CCC)", fontsize=9)

    # Legend below the panels — one row, with the colour-grouping spelled out.
    legend_handles = [
        mpatches.Patch(facecolor=FM_COLOURS["fmcib"], label="FMCIB (3D medical CT)"),
        mpatches.Patch(facecolor=FM_COLOURS["ct_fm"], label="CT-FM (3D medical CT)"),
        mpatches.Patch(facecolor=FM_COLOURS["biomedclip"], label="BiomedCLIP (2D biomedical VL)"),
        mpatches.Patch(facecolor=FM_COLOURS["dinov2"], label="DINOv2 (2D ImageNet)"),
        mpatches.Patch(facecolor=FM_COLOURS["rad_dino"], label="RAD-DINO (2D chest X-ray)"),
        mpatches.Patch(facecolor=FM_COLOURS["random_init"], hatch="//",
                       edgecolor="#666666", label="random_init (10-seed)"),
        Line2D([0], [0], color="#404040", linestyle="--", linewidth=1.2,
               label="IBSI radiomics ± 95% CI"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.05),
        ncol=4, frameon=False, fontsize=8,
    )

    fig.suptitle(
        "Figure 3.  H6 — test–retest reproducibility on cancer (T6) and healthy controls (T9).  "
        "FMCIB and CT-FM exceed IBSI; three FMs sit below.",
        fontsize=10, y=1.02, x=0.02, ha="left",
    )

    fig.tight_layout(rect=[0, 0.05, 1, 0.99])
    return _save(fig, output_stem)


# %% [markdown]
# ## 7. Figure 4 — Per-task FM forest plot (3 × 3 grid)
#
# Nine tasks (T2-GTVp moved to supplementary per v2 evaluation). Each panel:
# rows are FMs sorted by point estimate within that panel; horizontal whiskers
# are patient-clustered bootstrap 95% CI; a vertical dashed line marks the
# random_init 10-seed median; a vertical solid line marks the IBSI baseline on
# the two test-retest panels (T6, T9). x-axis ranges are shared within a metric
# family (AUROC 0.0–1.0; c-index 0.4–0.8; CCC −0.1–1.0) so panel-to-panel
# comparisons within a family are immediate.

# %%
FOREST_TASKS = ["t1", "t2", "t5", "t3", "t4", "t6", "t7", "t8", "t9"]
METRIC_FAMILY = {
    "t1": "AUROC", "t2": "AUROC", "t5": "AUROC", "t7": "AUROC", "t8": "AUROC",
    "t3": "c-index", "t4": "c-index",
    "t6": "CCC", "t9": "CCC",
}
METRIC_RANGE = {
    "AUROC": (0.30, 1.02),
    "c-index": (0.35, 0.80),
    "CCC": (-0.10, 1.00),
}


def render_figure_4(probe: pd.DataFrame, output_stem: str = "figure_4_forest") -> Path:
    fig, axes = plt.subplots(3, 3, figsize=(11, 9.5))
    axes = axes.flatten()

    fm_rows = [f for f in FM_ORDER if f != "ibsi_radiomics_baseline"]

    for idx, task in enumerate(FOREST_TASKS):
        ax = axes[idx]
        family = METRIC_FAMILY[task]
        xlim = METRIC_RANGE[family]

        df_t = probe[probe["task"] == task].copy()
        df_t["fm"] = df_t["fm"].str.lower().str.strip()

        # IBSI reference (vertical line) — only for T6 / T9.
        ibsi_val = float("nan")
        if task in ("t6", "t9"):
            ibsi = df_t[df_t["fm"] == "ibsi_radiomics_baseline"]
            if len(ibsi):
                ibsi_val = float(ibsi["metric"].iloc[0])

        # random_init reference (vertical line).
        ri = df_t[df_t["fm"] == "random_init"]
        ri_val = float(ri["metric"].iloc[0]) if len(ri) else float("nan")

        # FM rows for this panel; sort by point estimate descending.
        rows = df_t[df_t["fm"].isin(fm_rows)].copy()
        if "metric" in rows.columns:
            rows = rows.sort_values("metric", ascending=True).reset_index(drop=True)
        # Place random_init at the bottom always (visual anchor).
        rows["_order"] = rows["fm"].apply(lambda f: -1 if f == "random_init" else 0)
        rows = rows.sort_values(["_order", "metric"], ascending=[False, True]).reset_index(drop=True)

        if len(rows) == 0:
            ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                    ha="center", va="center", fontsize=9, color="#888888")
            ax.set_title(f"{TASK_LABELS_SHORT[task]} — {family}", fontsize=9.5)
            ax.set_xlim(*xlim)
            ax.set_yticks([])
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            continue

        y_pos = np.arange(len(rows))
        points = rows["metric"].values
        lowers = rows["lower"].values if "lower" in rows.columns else points
        uppers = rows["upper"].values if "upper" in rows.columns else points
        colours = [FM_COLOURS[f] for f in rows["fm"]]
        markers = ["s" if f == "random_init" else "o" for f in rows["fm"]]

        # Horizontal CI whiskers.
        for yi, (lo, hi, c) in enumerate(zip(lowers, uppers, colours)):
            ax.plot([lo, hi], [yi, yi], color=c, linewidth=1.6, alpha=0.85, zorder=2)
        # Point markers.
        for yi, (p, c, m) in enumerate(zip(points, colours, markers)):
            ax.scatter(p, yi, color=c, marker=m, s=42,
                       edgecolor="white", linewidth=0.6, zorder=3)

        # Reference lines.
        if not np.isnan(ri_val):
            ax.axvline(ri_val, color="#a6a6a6", linestyle=":", linewidth=1.0,
                       zorder=1, label="random_init median")
        if task in ("t6", "t9") and not np.isnan(ibsi_val):
            ax.axvline(ibsi_val, color="#404040", linestyle="--", linewidth=1.0,
                       zorder=1, label="IBSI radiomics")

        ax.set_yticks(y_pos)
        ax.set_yticklabels([FM_LABELS_SHORT[f] for f in rows["fm"]], fontsize=8)
        ax.set_xlim(*xlim)
        ax.tick_params(axis="x", labelsize=7.5)
        ax.grid(axis="x", linestyle=":", alpha=0.4)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

        # Panel title with task ID + cohort + metric.
        ax.set_title(
            f"{TASK_LABELS_SHORT[task]} — {family}",
            fontsize=9.5, fontweight="bold", loc="left", pad=4,
        )
        # Metric axis label only on the bottom row of the grid.
        if idx >= 6:
            ax.set_xlabel(family, fontsize=8.5)

    # Combined legend at top centre.
    legend_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=FM_COLOURS["fmcib"],
               markersize=7, label="FMCIB"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=FM_COLOURS["ct_fm"],
               markersize=7, label="CT-FM"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=FM_COLOURS["biomedclip"],
               markersize=7, label="BiomedCLIP"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=FM_COLOURS["dinov2"],
               markersize=7, label="DINOv2"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=FM_COLOURS["rad_dino"],
               markersize=7, label="RAD-DINO"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor=FM_COLOURS["random_init"],
               markersize=7, label="random_init (10-seed)"),
        Line2D([0], [0], color="#a6a6a6", linestyle=":", label="random_init median"),
        Line2D([0], [0], color="#404040", linestyle="--", label="IBSI radiomics"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=4, frameon=False, fontsize=8,
    )

    fig.suptitle(
        "Figure 4.  Per-task FM ranking with patient-clustered bootstrap 95% CI.  "
        "Reference lines: random_init 10-seed median (dotted); IBSI baseline (dashed, T6/T9 only).",
        fontsize=10, y=1.06, x=0.02, ha="left",
    )

    fig.tight_layout(rect=[0, 0, 1, 0.99])
    return _save(fig, output_stem)


# %% [markdown]
# ## 8. Save helper

# %%
def _save(fig: plt.Figure, stem: str) -> Path:
    """Write both PNG (300 dpi) and PDF (vector) for a single figure."""
    png = OUTPUT_DIR / f"{stem}.png"
    pdf = OUTPUT_DIR / f"{stem}.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    print(f"  wrote {png} ({png.stat().st_size / 1024:.1f} KiB)")
    print(f"  wrote {pdf} ({pdf.stat().st_size / 1024:.1f} KiB)")
    return png


# %% [markdown]
# ## 9. Render all four figures

# %%
print(f"Reading freeze CSVs from: {INPUT_DIR}")
print(f"Writing figures to:        {OUTPUT_DIR}")
print()

print("Figure 1: cross-FM × cross-task primary-metric heatmap")
matrix = load_fm_task_matrix()
render_figure_1(matrix)
print()

if "contamination_audit.csv" in FREEZE_FILES:
    print("Figure 2: contamination tier matrix (Phase 2 v2)")
    audit = load_contamination_audit()
    render_figure_2(audit)
    print()
else:
    print("Figure 2: SKIPPED (contamination_audit.csv not attached)")
    print()

print("Figure 3: H6 test–retest CCC vs IBSI (T6 + T9)")
t6, t9 = load_test_retest_csvs()
render_figure_3(t6, t9)
print()

print("Figure 4: per-task FM forest plot (3 × 3)")
probe = load_probe_results()
render_figure_4(probe)
print()

print("All figures rendered.")

# %% [markdown]
# ## 10. Done
#
# The four figures (PNG @ 300 dpi + PDF vector) are in `/kaggle/working/`.
# Use Kaggle's "Output" tab to download. No inline preview is rendered here
# — Kaggle auto-numbers each `plt.show()` as a separate output panel and an
# inline re-display would duplicate the figures under unrelated numbers.
