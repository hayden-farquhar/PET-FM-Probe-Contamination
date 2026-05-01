# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # PET-FM-Bench: Probe Analysis v6 (Phase 5 formal run)
#
# **DOI:** [10.17605/OSF.IO/DQ2JA](https://doi.org/10.17605/OSF.IO/DQ2JA)
#
# **Runtime:** CPU (no GPU needed) | **Internet:** Off OK
#
# **v6 changes vs v5:**
# - Loading loop extended to include T1 + T5 (latent v5 bug: dispatch blocks
#   were unreachable because the loop omitted them — only got merged into the
#   codebase after the loop was written, never patched) AND HECKTOR (T2 + T3).
# - `find_embed_dir()` now also accepts the HECKTOR-style "-2025-" slug variant
#   (`pet-fm-bench-hecktor-2025-embeddings-v3`), since the published HECKTOR
#   embeddings dataset deviated from the canonical `{task}-embeddings-v3` form.
# - **NEW T2 dispatch** (per A12a): per-patch lesion-classification AUROC on
#   HECKTOR cohort (`task1_patient == True`), GroupKFold patient-level CV.
#   Plus T2 GTVp-only sensitivity analysis (lesion_class==1 only, primary tumour).
# - **NEW T3 dispatch** (per A12c): patient-pooled CoxPH RFS prediction on HECKTOR
#   cohort (`task2_patient == True`). Mean-pools embeddings across patient's
#   lesion patches before CoxPH (background patches excluded from pooling).
# - Cross-FM summary in §6 extended to include T2 + T3 + T2-GTVp rows.
#
# Loads cached embeddings from completed tasks and runs:
# 1. **Linear classification probes** (T1 patch, T2 patch, T8 subtype)
# 2. **Survival probes** (T3 RFS, T4 NSCLC survival)
# 3. **Zero-shot transfer** (T5 PSMA from T1 FDG)
# 4. **Test-retest stability** (T6 cancer, T9 healthy controls)
# 5. **Response assessment** (T7 ACRIN — baseline embeddings)
#
# **Input datasets:** Attach all 8 **v3** embedding datasets:
# - `pet-fm-bench-t1-embeddings-v3`
# - `pet-fm-bench-t4-embeddings-v3`
# - `pet-fm-bench-t5-embeddings-v3`
# - `pet-fm-bench-t6-embeddings-v3`
# - `pet-fm-bench-t7-embeddings-v3`
# - `pet-fm-bench-t8-embeddings-v3`
# - `pet-fm-bench-t9-embeddings-v3`
# - `pet-fm-bench-hecktor-2025-embeddings-v3` (T2 + T3 unified)
#
# Plus all multi-seed datasets: `pet-fm-bench-{tX,hecktor}-randominit-multiseed-v3`
# (per amendment A3, N=10 seeds collapsed to median + IQR).
# Plus Phase 4 v4 task_splits: `pet-fm-bench-task-splits-v4`.
# Plus Phase 2 v2 contamination audit (for §6 contamination-tier merge).
#
# **Why v3, not v1:** v1 patches had a Bq/mL → float16 overflow producing
# 18–86% all-NaN embeddings on the 3D FMs (FMCIB, CT-FM) and 0/16 surviving
# T6 retest pairs. v3 patches use the (companion project) SUV pipeline (with the GML/BQML
# Units branch) and recover 16/20 retest pairs. The Phase 1 checkpoint
# manifest and Phase 4 task splits are both keyed to v3 patient IDs — v1
# embeddings would silently drop patients via the NaN-removal step and
# produce a cohort that does not match the pre-registered splits.
#
# `find_embed_dir()` below auto-prefers v3 via rglob; the v1 fallback exists
# only as a safety net and should not be relied on for the formal run.

# %% [markdown]
# ## 1. Setup

# %%
import os
import re
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler, LabelEncoder

# v2: silence the expected sksurv numerical warnings during CoxPH alpha sweeps
# (overflow in exp / divide-by-zero in log are recoverable; the loss is still
# evaluated stably at scoring time). ConvergenceWarning is also expected when
# the alpha grid is wide.
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning, module="sksurv")

!pip install -q scikit-survival

OUTPUT_DIR = Path("/kaggle/working/results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## 2. Locate and load embedding datasets

# %%
def find_embed_dir(task_pattern):
    """Find an embedding dataset matching a pattern in /kaggle/input/.

    Kaggle mounts datasets at varying depths depending on how they were added:
    /kaggle/input/<dataset>/ or /kaggle/input/datasets/<user>/<dataset>/

    Prefers v3 datasets when both v1 and v3 are attached (defends against
    accidental double-attachment producing non-deterministic dataset selection).
    """
    # Prefer v3 embeddings dataset if attached anywhere under /kaggle/input.
    # v6: also accept the HECKTOR-style "{task_pattern}-2025-embeddings-v3" slug
    # variant, since the published HECKTOR embeddings dataset uses that form.
    v3_patterns = [
        f"pet-fm-bench-{task_pattern}-embeddings-v3",
        f"pet-fm-bench-{task_pattern}-2025-embeddings-v3",
    ]
    for v3_pattern in v3_patterns:
        v3_candidates = list(Path("/kaggle/input").rglob(v3_pattern))
        if v3_candidates:
            embed_dir = v3_candidates[0]
            nested = embed_dir / "embeddings"
            if nested.exists():
                return nested
            for sub in embed_dir.iterdir():
                if sub.is_dir():
                    nested2 = sub / "embeddings"
                    if nested2.exists():
                        return nested2
            return embed_dir

    # Fallback: original loose-pattern discovery (handles v1 datasets / unusual mounts)
    search_roots = [
        Path("/kaggle/input"),
        Path("/kaggle/input/datasets"),
    ]
    for root in list(search_roots):
        if root.exists():
            for sub in root.iterdir():
                if sub.is_dir() and not sub.name.startswith("."):
                    search_roots.append(sub)

    for root in search_roots:
        candidates = list(root.glob(f"*{task_pattern}*"))
        if candidates:
            embed_dir = candidates[0]
            nested = embed_dir / "embeddings"
            if nested.exists():
                return nested
            for sub in embed_dir.iterdir():
                if sub.is_dir():
                    nested2 = sub / "embeddings"
                    if nested2.exists():
                        return nested2
            return embed_dir

    print(f"  WARNING: No dataset found matching '{task_pattern}'")
    return None


def load_fm_embeddings(embed_dir):
    """Load all FM parquets from an embedding directory.

    Returns dict: fm_name → DataFrame with patient_id, view, layer, d0000...dNNNN.

    v6: skip ALL files ending in '_labels' (not just t<digit>_labels). HECKTOR's
    label parquet is `hecktor_labels.parquet` which doesn't match the t<digit>_labels
    pattern; without this fix, it gets wrongly loaded as a 0-dim FM.
    """
    if embed_dir is None:
        return {}
    fm_data = {}
    for f in sorted(embed_dir.glob("*.parquet")):
        fm_name = f.stem
        if fm_name.endswith("_labels"):
            continue  # Skip any *_labels.parquet (T1, T5, HECKTOR all use this)
        df = pd.read_parquet(f)
        fm_data[fm_name] = df
        dim = sum(1 for c in df.columns if c.startswith("d"))
        print(f"  {fm_name}: {len(df)} rows, {dim}-dim")
    return fm_data


def load_labels(embed_dir, label_pattern="*labels*"):
    """Load labels file from an embedding directory — dispatches on extension.

    v6: T1/T5/HECKTOR ship per-patch labels as PARQUET (typed columns, lesion_voxels
    floats); T4/T6/T7/T8/T9 ship as CSV. Without this dispatch, rglob('*labels*')
    matched any extension and pd.read_csv() crashed on parquet binary.
    Prefer .csv first (matches v5 behaviour for the originally-supported tasks),
    fall back to .parquet (T1/T5/HECKTOR).
    """
    if embed_dir is None:
        return None
    candidates_csv = list(embed_dir.glob(label_pattern + ".csv"))
    if not candidates_csv:
        candidates_csv = list(embed_dir.glob(label_pattern))
        candidates_csv = [c for c in candidates_csv if c.suffix == ".csv"]
    if not candidates_csv:
        candidates_csv = list(embed_dir.parent.rglob(label_pattern + ".csv"))
    if candidates_csv:
        return pd.read_csv(candidates_csv[0])
    candidates_parquet = list(embed_dir.glob(label_pattern + ".parquet"))
    if not candidates_parquet:
        candidates_parquet = list(embed_dir.glob(label_pattern))
        candidates_parquet = [c for c in candidates_parquet if c.suffix == ".parquet"]
    if not candidates_parquet:
        candidates_parquet = list(embed_dir.parent.rglob(label_pattern + ".parquet"))
    if candidates_parquet:
        return pd.read_parquet(candidates_parquet[0])
    return None


def get_patient_embedding(df, patient_id, view="coronal", layer="cls"):
    """Extract a single embedding vector for a patient.

    For 2D FMs: uses specified view and layer.
    For 3D FMs: uses 'pool' layer (mean across patches).
    """
    # Try exact match
    mask = df["patient_id"] == patient_id
    if "view" in df.columns:
        if "volume" in df["view"].values:
            mask = mask & (df["view"] == "volume")
        else:
            mask = mask & (df["view"] == view)
    if "layer" in df.columns:
        if "pool" in df[df["patient_id"] == patient_id]["layer"].values:
            mask = mask & (df["layer"] == "pool")
        else:
            mask = mask & (df["layer"] == layer)

    rows = df[mask]
    if len(rows) == 0:
        return None

    dim_cols = [c for c in df.columns if c.startswith("d")]
    return rows[dim_cols].values[0]


# Also handle T9's subject_id column
def get_session_embedding(df, patient_id, session, view="coronal", layer="cls"):
    """Extract embedding for a specific session (test-retest tasks)."""
    id_col = "patient_id" if "patient_id" in df.columns else "subject_id"
    sess_col = "session" if "session" in df.columns else None

    mask = df[id_col] == patient_id
    if sess_col and sess_col in df.columns:
        mask = mask & (df[sess_col] == session)

    if "view" in df.columns:
        if "volume" in df["view"].values:
            mask = mask & (df["view"] == "volume")
        else:
            mask = mask & (df["view"] == view)
    if "layer" in df.columns:
        if "pool" in df[mask]["layer"].values if mask.any() else False:
            mask = mask & (df["layer"] == "pool")
        else:
            mask = mask & (df["layer"] == layer)

    rows = df[mask]
    if len(rows) == 0:
        return None

    dim_cols = [c for c in df.columns if c.startswith("d")]
    return rows[dim_cols].values[0]


print("Loading embedding datasets...\n")


# v3: multi-seed random_init aggregation. Single-seed random_init is not a
# defensible baseline — different seeds can swing AUROC ±0.10. The pre-registered
# baseline expects N seeds → median + IQR. This regex matches per-seed parquet
# stems. v6 post-bug-fix: allows optional separator BOTH before "seed" AND
# between "seed" and the digit, to catch both:
#   - older T4-T9: `random_init_seed0` (no underscore between "seed" and digit)
#   - newer T1/T5/HECKTOR: `random_init_seed_0` (with underscore)
# Defined here (before find_multiseed_dir / load_multiseed_random_init) so those
# helpers can reference it during the EMBEDDINGS loading loop. Original location
# at §5 (~line 539) is now a re-import for readability of the helper functions.
RANDOM_INIT_SEED_PATTERN = re.compile(
    r"^random_init[_-]?(?:seed)?[_-]?(\d+)$", re.IGNORECASE
)


def find_multiseed_dir(task):
    """Find the random_init multi-seed dataset for a task.

    v6 (post-bug-fix): the multi-seed parquets (random_init_seed_0.parquet
    ... random_init_seed_9.parquet) live in a SEPARATE Kaggle dataset
    (`pet-fm-bench-{task}-randominit-multiseed-v3` or `-multiseed`), NOT
    in the same dir as the FM embeddings dataset. find_embed_dir only
    resolves the embeddings dataset; this helper resolves the multi-seed
    dataset so that EMBEDDINGS[task] can be merged with the multi-seed
    parquets to satisfy A3 (N=10 seeds → median + IQR aggregation).
    """
    for slug in [
        f"pet-fm-bench-{task}-randominit-multiseed-v3",
        f"pet-fm-bench-{task}-randominit-multiseed",  # legacy v1 naming
    ]:
        candidates = list(Path("/kaggle/input").rglob(slug))
        if candidates:
            return candidates[0]
    return None


def load_multiseed_random_init(multiseed_dir):
    """Load all random_init_seed*.parquet files from a multi-seed dataset.

    v6 post-bug-fix: permissive glob `random_init*.parquet` then regex-filter via
    RANDOM_INIT_SEED_PATTERN. Catches both naming conventions:
    - older T4-T9 scripts: `random_init_seed0.parquet`
    - newer T1/T5/HECKTOR scripts: `random_init_seed_0.parquet`
    Skips the single-seed `random_init.parquet` if it happens to live in the
    multi-seed dir (the regex requires a trailing digit to match).
    """
    if multiseed_dir is None:
        return {}
    candidates = list(multiseed_dir.glob("random_init*.parquet"))
    if not candidates:
        nested = multiseed_dir / "embeddings"
        if nested.exists():
            candidates = list(nested.glob("random_init*.parquet"))
    if not candidates:
        candidates = list(multiseed_dir.rglob("random_init*.parquet"))

    seed_data = {}
    for f in sorted(candidates):
        fm_name = f.stem  # e.g. "random_init_seed_0" or "random_init_seed0"
        if not RANDOM_INIT_SEED_PATTERN.match(fm_name):
            continue  # Skips bare "random_init" (no trailing digit)
        df = pd.read_parquet(f)
        seed_data[fm_name] = df
        dim = sum(1 for c in df.columns if c.startswith("d"))
        print(f"    [multiseed] {fm_name}: {len(df)} rows, {dim}-dim")
    return seed_data


# Locate all datasets
# v6: extends v5's task list with t1, t5, and hecktor (the unified T2+T3 dataset).
# v5 had a latent bug where t1/t5 dispatch blocks were unreachable because the
# loading loop omitted them — they only got merged into the codebase after the
# loop was written and never patched. v6 closes that gap, plus adds hecktor.
TASK_DIRS = {}
for task in ["t1", "t4", "t5", "t6", "t7", "t8", "t9", "hecktor"]:
    d = find_embed_dir(task)
    if d:
        print(f"\n{task.upper()}:")
        TASK_DIRS[task] = d

# Load embeddings (FM dataset + multi-seed random_init dataset, merged into one dict
# per task). v6 post-bug-fix: previously only the single-seed random_init from the
# embeddings dataset was loaded; the 10 random_init_seed_* parquets in the multi-seed
# dataset were ignored, triggering the legacy single-seed warning path. Now the
# multi-seed parquets are merged into EMBEDDINGS[task] alongside the FM embeddings
# so aggregate_random_init_seeds() can produce the A3-compliant median + IQR.
EMBEDDINGS = {}
for task, d in TASK_DIRS.items():
    EMBEDDINGS[task] = load_fm_embeddings(d)
    multiseed_dir = find_multiseed_dir(task)
    if multiseed_dir is not None:
        seed_data = load_multiseed_random_init(multiseed_dir)
        EMBEDDINGS[task].update(seed_data)
    else:
        print(f"  ⚠ {task}: no multi-seed dataset found (looked for "
              f"pet-fm-bench-{task}-randominit-multiseed[-v3]). "
              f"A3 baseline will fall back to single-seed warning path.")

# Load labels
LABELS = {}
for task, d in TASK_DIRS.items():
    lbl = load_labels(d)
    if lbl is not None:
        LABELS[task] = lbl
        print(f"\n{task.upper()} labels: {list(lbl.columns)}")

# v4: load Phase 4 task splits if attached. The freeze artefact is
# `task_splits.parquet` produced by 06_task_splits.py and uploaded to OSF
# aqmkb/phase_4_freeze_task_splits/. Two attachment paths supported:
#   - Kaggle dataset `pet-fm-bench-task-splits` (preferred: standard format)
#   - Inside any embedding dataset's parent if user co-attaches
# If task_splits.parquet is not found, probe_analysis.py falls back to
# cross-task 5-fold CV (the v3 dry-run behaviour) and prints a warning.
def find_task_splits():
    candidates = list(Path("/kaggle/input").rglob("task_splits.parquet"))
    if not candidates:
        return None
    # v6: explicitly prefer v4 (9-task universe) > v3 > v2 > unknown.
    # If multiple task_splits versions are attached (e.g. user accidentally
    # leaves v2 attached alongside v4), this guarantees v4 wins.
    for version_suffix in ("-v4", "-v3", "-v2"):
        for c in candidates:
            if f"pet-fm-bench-task-splits{version_suffix}" in str(c):
                return c
    # Fallback: any pet-fm-bench-task-splits file
    for c in candidates:
        if "pet-fm-bench-task-splits" in str(c):
            return c
    return candidates[0]


SPLITS_PATH = find_task_splits()
if SPLITS_PATH is not None:
    SPLITS_DF = pd.read_parquet(SPLITS_PATH)
    print(f"\n=== Phase 4 task splits ({SPLITS_PATH}) ===")
    print(SPLITS_DF.groupby(["task", "split"]).size().to_string())
else:
    SPLITS_DF = None
    print("\n⚠ task_splits.parquet not attached — formal-run held-out eval "
          "will fall back to cross-task 5-fold CV. Attach the Phase 4 freeze "
          "dataset (`pet-fm-bench-task-splits`) for registration-compliant eval.")

# v3: report random_init seed coverage up-front so the user knows whether the
# multi-seed baseline path will engage or the legacy single-seed warning path.
print("\n=== random_init seed coverage ===")
for task in sorted(EMBEDDINGS):
    # v6 post-bug-fix (Step 65b): use the module-level RANDOM_INIT_SEED_PATTERN
    # which handles BOTH `random_init_seed0` (older T4-T9) AND
    # `random_init_seed_0` (newer T1/T5/HECKTOR) naming forms. Previously this
    # was a hardcoded inline regex matching only the older form, so the newer
    # task multi-seeds were silently flagged as legacy single-seed.
    n_seeds = sum(
        1 for k in EMBEDDINGS[task]
        if RANDOM_INIT_SEED_PATTERN.match(k)
    )
    has_legacy = "random_init" in EMBEDDINGS[task]
    if n_seeds >= 2:
        print(f"  {task}: {n_seeds} seeds ✓ (multi-seed aggregation will engage)")
    elif n_seeds == 1:
        print(f"  {task}: 1 seed ⚠ (will be treated as legacy single-seed)")
    elif has_legacy:
        print(f"  {task}: legacy single-seed `random_init` ⚠ (re-run "
              f"08_random_init_multiseed.py for formal baseline)")
    else:
        print(f"  {task}: no random_init found")

# %% [markdown]
# ## 3. Classification probe: T8 (lung cancer subtype)
#
# Multi-class classification: adenocarcinoma vs small_cell vs squamous_cell.
# Per registration: LogisticRegression with L2, C grid, 5-fold patient-level CV.

# %%
N_FOLDS = 5
C_GRID = [0.001, 0.01, 0.1, 1, 10, 100]
N_BOOTSTRAP = 1000


# v4: split-dispatch helpers. Decide per task whether to do 5-fold CV across
# all patients (cv_pool — T4 per registration §3.3) or held-out eval
# (train/cal/test — T7/T8). Falls back to CV when SPLITS_DF is absent.
def get_task_splits(task):
    """Return dict of {split_name: [patient_ids]} for a task, or None if unavailable."""
    if SPLITS_DF is None:
        return None
    sub = SPLITS_DF[SPLITS_DF["task"] == task]
    if sub.empty:
        return None
    return {
        sp: sub.loc[sub["split"] == sp, "patient_id"].astype(str).tolist()
        for sp in sub["split"].unique()
    }


def task_eval_mode(task):
    """Return 'heldout', 'cv', 'test_retest', 'zero_shot', or 'cv_fallback'.

    v5 (per A9b): T5 uses 'zero_shot' — single test_zero_shot split label, no
    train/cal partition; the T5 probe is fit on T1 train embeddings and
    evaluated zero-shot on the entire T5 cohort.
    """
    sp = get_task_splits(task)
    if sp is None:
        return "cv_fallback"
    if "test_retest" in sp:
        return "test_retest"
    if "test_zero_shot" in sp:
        return "zero_shot"
    if "cv_pool" in sp:
        return "cv"
    if "test" in sp and "train" in sp:
        return "heldout"
    return "cv_fallback"


def get_patch_embedding(df, patient_id, patch_id, view="axial", layer="cls"):
    """v5 per-patch embedding lookup for T1/T5 (vs get_patient_embedding for T4-T9).

    T1/T5 manifests have one row per (patient_id, patch_id, fm, view, layer);
    this helper returns the single embedding vector matching the requested
    (patient_id, patch_id, view, layer). 3D FMs default to view='volume',
    layer='pool'; caller must pass appropriate kwargs.
    """
    id_col = "patient_id" if "patient_id" in df.columns else "subject_id"
    sub = df[(df[id_col] == patient_id) & (df["patch_id"] == patch_id)]
    if "view" in df.columns and view in df["view"].values:
        sub = sub[sub["view"] == view]
    if "layer" in df.columns and layer in df["layer"].values:
        sub = sub[sub["layer"] == layer]
    if len(sub) == 0:
        return None
    dim_cols = [c for c in df.columns if c.startswith("d")]
    return sub[dim_cols].values[0]


# v4: Lin's Concordance Correlation Coefficient — registration-primary
# test-retest metric (§1.3 H6, §3.1). For embedding stability we compute
# per-dimension CCC across the test-retest cohort, then average across dims.
# This is the canonical radiomics test-retest convention (cf. Aerts 2014).
def lin_ccc(x, y):
    """Lin's CCC for paired scalar measurements (1-D arrays)."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < 2 or len(y) < 2:
        return float("nan")
    mx, my = x.mean(), y.mean()
    vx, vy = x.var(), y.var()
    cov = np.mean((x - mx) * (y - my))
    denom = vx + vy + (mx - my) ** 2
    if denom <= 0:
        return float("nan")
    return float(2 * cov / denom)


def embedding_ccc(test_embs, retest_embs):
    """Per-dimension Lin's CCC averaged across embedding dimensions.

    test_embs, retest_embs: (n_pairs, n_dim) arrays of paired embeddings.
    Returns mean CCC across dims (ignoring NaN dimensions where variance was 0).
    """
    test_embs = np.asarray(test_embs, dtype=np.float64)
    retest_embs = np.asarray(retest_embs, dtype=np.float64)
    assert test_embs.shape == retest_embs.shape, (
        f"shape mismatch {test_embs.shape} vs {retest_embs.shape}"
    )
    if test_embs.shape[0] < 2:
        return float("nan")
    n_dim = test_embs.shape[1]
    cccs = []
    for d in range(n_dim):
        c = lin_ccc(test_embs[:, d], retest_embs[:, d])
        if not np.isnan(c):
            cccs.append(c)
    return float(np.mean(cccs)) if cccs else float("nan")


# v6 post-bug-fix: RANDOM_INIT_SEED_PATTERN is defined earlier (before
# find_multiseed_dir / load_multiseed_random_init) so those helpers can use it
# during the EMBEDDINGS loading loop. See line ~285. The aggregation helpers
# below (count_random_init_seeds, aggregate_random_init_seeds) use the same
# module-level constant.


def count_random_init_seeds(embeddings_dict):
    """Return the number of distinct random_init_seed* keys."""
    return sum(
        1 for k in embeddings_dict
        if RANDOM_INIT_SEED_PATTERN.match(k)
    )


def aggregate_random_init_seeds(results_df):
    """Collapse `random_init_seed{N}` rows into a single `random_init` row.

    For multi-seed inputs: replaces N seed rows with 1 row whose `value` is
    the seed median, and whose `ci_low`/`ci_high` are the seed IQR (25th/75th
    percentile). The bootstrap CI is dropped for the aggregated row because
    seed variance dominates bootstrap variance for an N≥10 random-init
    baseline; reporting both would be misleading.

    For single-seed (legacy `random_init` only): passes through with a
    warning and `n_seeds=1` so the manuscript table can flag it.

    For no random_init: passes through unchanged.
    """
    if results_df.empty:
        return results_df

    is_seed = results_df["fm"].apply(lambda s: bool(RANDOM_INIT_SEED_PATTERN.match(s)))
    is_legacy = results_df["fm"] == "random_init"
    n_seed_rows = int(is_seed.sum())
    n_legacy_rows = int(is_legacy.sum())

    if n_seed_rows == 0 and n_legacy_rows == 0:
        return results_df  # no random_init in this task

    if n_seed_rows == 0:
        # Legacy single-seed: warn but pass through with n_seeds=1.
        print(f"  ⚠ random_init has only 1 seed (no random_init_seed* parquets "
              f"attached). NOT suitable as the formal pre-reg baseline. "
              f"Re-run after `08_random_init_multiseed.py` produces N≥10 seeds.")
        out = results_df.copy()
        out.loc[is_legacy, "n_seeds"] = 1
        return out

    # Multi-seed: aggregate.
    seed_rows = results_df[is_seed].copy()
    other_rows = results_df[~is_seed & ~is_legacy].copy()

    # v6 post-bug-fix (Step 65): median-aggregate ALL numeric columns, not just
    # `value`. T5's secondary metrics (auprc, brier, cross_tracer_transfer_penalty)
    # vary per seed and were previously taken from iloc[0] alone. Same for
    # T1/T2's auprc/brier/best_C and T3/T4's best_alpha — all now properly
    # medianed. Columns that don't vary across seeds (n_train_patches,
    # n_test_patches, etc.) median to themselves; safe no-op.
    _PROTECTED_COLS = {"value", "ci_low", "ci_high", "n_seeds", "seed_min", "seed_max"}

    agg_records = []
    for (task, metric), grp in seed_rows.groupby(["task", "metric"]):
        median_val = float(grp["value"].median())
        iqr_low = float(grp["value"].quantile(0.25))
        iqr_high = float(grp["value"].quantile(0.75))
        # Template from first row, then overwrite seed-aggregated fields.
        rec = grp.iloc[0].to_dict()
        # Median-aggregate any other numeric column (auprc, brier, best_C,
        # transfer_penalty, etc.). pandas .median() defaults to skipna=True so
        # NaN seeds (e.g. probe_failed) don't poison the aggregate.
        for col in grp.columns:
            if col in _PROTECTED_COLS or col == "fm":
                continue
            if pd.api.types.is_numeric_dtype(grp[col]):
                try:
                    rec[col] = float(grp[col].median())
                except (TypeError, ValueError):
                    pass  # Keep iloc[0] template value if median fails
        rec["fm"] = "random_init"
        rec["value"] = median_val
        rec["ci_low"] = iqr_low
        rec["ci_high"] = iqr_high
        rec["n_seeds"] = len(grp)
        rec["seed_min"] = float(grp["value"].min())
        rec["seed_max"] = float(grp["value"].max())
        agg_records.append(rec)

    print(f"  ✓ Aggregated {n_seed_rows} random_init seeds across "
          f"{len(agg_records)} (task, metric) pair(s) → median + IQR")

    return pd.concat([other_rows, pd.DataFrame(agg_records)], ignore_index=True)


def run_classification_probe(embeddings_dict, labels_df, label_col, task_name):
    """Run linear probe classification for all FMs on a task."""
    results = []

    for fm_name, fm_df in embeddings_dict.items():
        # Get patient-level embeddings
        id_col = "patient_id" if "patient_id" in fm_df.columns else "subject_id"
        patients = sorted(fm_df[id_col].unique())

        X_list, y_list, pid_list = [], [], []
        for pid in patients:
            emb = get_patient_embedding(fm_df, pid)
            if emb is None:
                continue
            label_row = labels_df[labels_df.iloc[:, 0] == pid]
            if len(label_row) == 0:
                continue
            label = label_row[label_col].values[0]
            if pd.isna(label):
                continue
            X_list.append(emb)
            y_list.append(label)
            pid_list.append(pid)

        if len(X_list) < 20:
            print(f"  {fm_name}: too few samples ({len(X_list)}), skipping")
            continue

        X = np.array(X_list)
        y = np.array(y_list)

        # Drop rows with NaN embeddings, replace remaining NaN with 0
        nan_mask = np.isnan(X).any(axis=1)
        if nan_mask.sum() > 0:
            print(f"  {fm_name}: dropping {nan_mask.sum()} NaN embeddings")
            X = X[~nan_mask]
            y = y[~nan_mask]
        np.nan_to_num(X, copy=False)

        if len(X) < 20:
            print(f"  {fm_name}: too few samples after NaN removal ({len(X)}), skipping")
            continue

        # Encode labels
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        classes = le.classes_

        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Select best C via CV
        best_c, best_score = 1.0, -1
        cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

        for c in C_GRID:
            model = LogisticRegression(C=c, penalty="l2", solver="lbfgs",
                                       max_iter=1000, random_state=42)
            try:
                preds = cross_val_predict(model, X_scaled, y_encoded, cv=cv,
                                          method="predict_proba")
                # Macro AUROC (one-vs-rest)
                auroc = roc_auc_score(y_encoded, preds, multi_class="ovr", average="macro")
                if auroc > best_score:
                    best_score = auroc
                    best_c = c
            except Exception:
                continue

        # Final predictions with best C
        model = LogisticRegression(C=best_c, penalty="l2", solver="lbfgs",
                                   max_iter=1000, random_state=42)
        y_pred = cross_val_predict(model, X_scaled, y_encoded, cv=cv,
                                    method="predict_proba")
        auroc = roc_auc_score(y_encoded, y_pred, multi_class="ovr", average="macro")

        # Bootstrap CI
        rng = np.random.RandomState(42)
        boot_aucs = []
        for _ in range(N_BOOTSTRAP):
            idx = rng.choice(len(y_encoded), size=len(y_encoded), replace=True)
            if len(np.unique(y_encoded[idx])) < len(classes):
                continue
            try:
                boot_aucs.append(roc_auc_score(y_encoded[idx], y_pred[idx],
                                                multi_class="ovr", average="macro"))
            except ValueError:
                continue

        ci_low = np.percentile(boot_aucs, 2.5) if boot_aucs else np.nan
        ci_high = np.percentile(boot_aucs, 97.5) if boot_aucs else np.nan

        results.append({
            "task": task_name, "fm": fm_name, "metric": "auroc_macro",
            "value": float(auroc), "ci_low": float(ci_low), "ci_high": float(ci_high),
            "best_c": best_c, "n_patients": len(X), "n_classes": len(classes),
            "classes": str(list(classes)),
        })
        print(f"  {fm_name}: AUROC={auroc:.3f} [{ci_low:.3f}, {ci_high:.3f}] (C={best_c}, n={len(X)})")

    return pd.DataFrame(results)


def run_classification_probe_heldout(embeddings_dict, labels_df, label_col, task_name):
    """v4: held-out test eval per registration §3.3 + Phase 4 v2 freeze.

    Filters patients to (train+cal) for hyperparameter tuning via 5-fold inner
    CV on (train+cal); refits on the full (train+cal) at the chosen C; reports
    test-set AUROC with patient-level bootstrap CI on the held-out test split.
    """
    results = []
    splits = get_task_splits(task_name)
    if splits is None:
        print(f"  {task_name}: task_splits.parquet not loaded — falling back to CV")
        return run_classification_probe(embeddings_dict, labels_df, label_col, task_name)

    train_cal_pids = set(splits.get("train", []) + splits.get("cal", []))
    test_pids = set(splits.get("test", []))
    print(f"  {task_name} splits: train+cal={len(train_cal_pids)}, test={len(test_pids)}")

    for fm_name, fm_df in embeddings_dict.items():
        id_col = "patient_id" if "patient_id" in fm_df.columns else "subject_id"

        def _collect(pid_subset):
            X_list, y_list = [], []
            for pid in fm_df[id_col].unique():
                if str(pid) not in pid_subset:
                    continue
                emb = get_patient_embedding(fm_df, pid)
                if emb is None:
                    continue
                label_row = labels_df[labels_df.iloc[:, 0] == pid]
                if len(label_row) == 0:
                    continue
                label = label_row[label_col].values[0]
                if pd.isna(label):
                    continue
                X_list.append(emb)
                y_list.append(label)
            return np.array(X_list) if X_list else None, np.array(y_list) if y_list else None

        X_tc, y_tc = _collect(train_cal_pids)
        X_te, y_te = _collect(test_pids)

        if X_tc is None or X_te is None or len(X_tc) < 20 or len(X_te) < 5:
            print(f"  {fm_name}: insufficient train+cal ({0 if X_tc is None else len(X_tc)}) "
                  f"or test ({0 if X_te is None else len(X_te)}), skipping")
            continue

        # NaN guard
        for X in (X_tc, X_te):
            np.nan_to_num(X, copy=False)

        # Encode labels using train+cal classes; require all test classes present
        le = LabelEncoder()
        y_tc_enc = le.fit_transform(y_tc)
        try:
            y_te_enc = le.transform(y_te)
        except ValueError:
            print(f"  {fm_name}: test set contains classes not in train+cal — skipping")
            continue
        n_classes = len(le.classes_)

        # Standardize on train+cal only (no leakage from test)
        scaler = StandardScaler()
        X_tc_s = scaler.fit_transform(X_tc)
        X_te_s = scaler.transform(X_te)

        # Inner 5-fold CV on train+cal for C selection
        cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
        best_c, best_score = 1.0, -1
        for c in C_GRID:
            try:
                model = LogisticRegression(C=c, penalty="l2", solver="lbfgs",
                                           max_iter=1000, random_state=42)
                preds = cross_val_predict(model, X_tc_s, y_tc_enc, cv=cv,
                                          method="predict_proba")
                if n_classes == 2:
                    s = roc_auc_score(y_tc_enc, preds[:, 1])
                else:
                    s = roc_auc_score(y_tc_enc, preds, multi_class="ovr",
                                      average="macro")
                if s > best_score:
                    best_score, best_c = s, c
            except Exception:
                continue

        # Refit on full train+cal with chosen C; predict on test
        model = LogisticRegression(C=best_c, penalty="l2", solver="lbfgs",
                                   max_iter=1000, random_state=42)
        model.fit(X_tc_s, y_tc_enc)
        y_te_proba = model.predict_proba(X_te_s)
        if n_classes == 2:
            auroc = roc_auc_score(y_te_enc, y_te_proba[:, 1])
        else:
            auroc = roc_auc_score(y_te_enc, y_te_proba, multi_class="ovr",
                                  average="macro")

        # Patient-level bootstrap on the held-out test set
        rng = np.random.RandomState(42)
        boot_aucs = []
        for _ in range(N_BOOTSTRAP):
            idx = rng.choice(len(y_te_enc), size=len(y_te_enc), replace=True)
            if len(np.unique(y_te_enc[idx])) < n_classes:
                continue
            try:
                if n_classes == 2:
                    boot_aucs.append(roc_auc_score(y_te_enc[idx], y_te_proba[idx, 1]))
                else:
                    boot_aucs.append(roc_auc_score(y_te_enc[idx], y_te_proba[idx],
                                                   multi_class="ovr", average="macro"))
            except ValueError:
                continue
        ci_low = float(np.percentile(boot_aucs, 2.5)) if boot_aucs else float("nan")
        ci_high = float(np.percentile(boot_aucs, 97.5)) if boot_aucs else float("nan")

        results.append({
            "task": task_name, "fm": fm_name, "metric": "auroc_macro",
            "value": float(auroc), "ci_low": ci_low, "ci_high": ci_high,
            "best_c": best_c,
            "n_train_cal": int(len(y_tc_enc)),
            "n_test": int(len(y_te_enc)),
            "n_classes": int(n_classes),
            "classes": str(list(le.classes_)),
            "eval_mode": "heldout",
        })
        print(f"  {fm_name}: AUROC={auroc:.3f} [{ci_low:.3f}, {ci_high:.3f}] "
              f"(C={best_c}, n_train_cal={len(y_tc_enc)}, n_test={len(y_te_enc)})")

    return pd.DataFrame(results)


if "t8" in EMBEDDINGS and "t8" in LABELS:
    print("\n" + "=" * 60)
    print("T8: Lung Cancer Subtype Classification")
    print("=" * 60)
    mode = task_eval_mode("t8")
    print(f"  eval mode: {mode}")
    if mode == "heldout":
        t8_results = run_classification_probe_heldout(
            EMBEDDINGS["t8"], LABELS["t8"], "subtype", "t8"
        )
    else:
        t8_results = run_classification_probe(
            EMBEDDINGS["t8"], LABELS["t8"], "subtype", "t8"
        )
    t8_results = aggregate_random_init_seeds(t8_results)
    t8_results.to_csv(OUTPUT_DIR / "t8_classification_results.csv", index=False)
else:
    print("T8 data not available")
    t8_results = pd.DataFrame()


# %% [markdown]
# ## 3b. T1: AutoPET-I FDG lesion-patch classification (per A9a)
#
# **Per-patch** binary classifier (lesion vs background). Each manifest row is a
# sample, not each patient. Uses **GroupKFold on patient_id** for inner CV so no
# patient straddles train/test. Held-out test eval on the T1 test split per
# Phase 4 v3 task_splits.
#
# Tests **registration H1**: at least one FM exceeds IBSI pyradiomics baseline by
# ≥3 pp AUROC. Reports per-FM patch-level AUROC + AUPRC + Brier + DeLong vs
# pyradiomics baseline.

# %%
def run_t1_lesion_patch_probe(embeddings_dict, labels_df, task_name="t1"):
    """Per-patch held-out classification probe with GroupKFold patient-level CV.

    embeddings_dict: {fm_name: parquet_df} where each parquet has one row per
        (patient_id, patch_id, view, layer, dXXXX...).
    labels_df: t1_labels.parquet with columns (patient_id, patch_id, label,
        cancer_type, lesion_index, iou, study_date).
    """
    from sklearn.model_selection import GroupKFold

    splits = get_task_splits(task_name)
    if splits is None:
        print(f"  {task_name}: task_splits.parquet not loaded — cannot run heldout")
        return pd.DataFrame()

    train_cal_pids = set(splits.get("train", []) + splits.get("cal", []))
    test_pids = set(splits.get("test", []))
    print(f"  {task_name} splits: train+cal={len(train_cal_pids)} patients, "
          f"test={len(test_pids)} patients")

    results = []
    for fm_name, fm_df in embeddings_dict.items():
        # Pick a default view+layer per FM family (consistent with other tasks).
        if {"volume", "pool"}.issubset(set(fm_df["view"].unique()) | set(fm_df["layer"].unique())):
            default_view, default_layer = "volume", "pool"
        elif "axial" in fm_df["view"].values:
            default_view, default_layer = "axial", "cls"
        else:
            default_view = sorted(fm_df["view"].unique())[0]
            default_layer = sorted(fm_df["layer"].unique())[0]

        sub = fm_df[(fm_df["view"] == default_view) & (fm_df["layer"] == default_layer)]
        if len(sub) == 0:
            print(f"  {fm_name}: no rows at view={default_view} layer={default_layer}, skipping")
            continue

        # Join embeddings with per-patch labels
        merged = sub.merge(
            labels_df[["patient_id", "patch_id", "label"]],
            on=["patient_id", "patch_id"],
            how="inner",
        )
        if len(merged) == 0:
            print(f"  {fm_name}: empty join with labels, skipping")
            continue

        # Partition merged into train+cal vs test by patient_id
        merged["_split"] = merged["patient_id"].astype(str).map(
            lambda p: "tc" if p in train_cal_pids else ("te" if p in test_pids else "drop")
        )
        merged = merged[merged["_split"] != "drop"].reset_index(drop=True)

        tc = merged[merged["_split"] == "tc"]
        te = merged[merged["_split"] == "te"]
        if len(tc) < 50 or len(te) < 20:
            print(f"  {fm_name}: insufficient train+cal ({len(tc)}) or test ({len(te)}), skipping")
            continue

        dim_cols = [c for c in fm_df.columns if c.startswith("d")]
        X_tc = tc[dim_cols].values.astype(np.float32)
        y_tc = tc["label"].values.astype(int)
        groups_tc = tc["patient_id"].values
        X_te = te[dim_cols].values.astype(np.float32)
        y_te = te["label"].values.astype(int)

        np.nan_to_num(X_tc, copy=False)
        np.nan_to_num(X_te, copy=False)

        scaler = StandardScaler()
        X_tc_s = scaler.fit_transform(X_tc)
        X_te_s = scaler.transform(X_te)

        # Inner GroupKFold CV for C selection — keeps patients intact across folds
        n_splits_inner = min(5, len(np.unique(groups_tc)))
        if n_splits_inner < 2:
            print(f"  {fm_name}: <2 unique groups in train+cal, skipping")
            continue
        cv = GroupKFold(n_splits=n_splits_inner)
        best_c, best_score = 1.0, -1
        for c in C_GRID:
            try:
                preds = np.zeros_like(y_tc, dtype=float)
                for tr_idx, va_idx in cv.split(X_tc_s, y_tc, groups=groups_tc):
                    m = LogisticRegression(C=c, penalty="l2", solver="lbfgs",
                                           max_iter=1000, random_state=42)
                    m.fit(X_tc_s[tr_idx], y_tc[tr_idx])
                    preds[va_idx] = m.predict_proba(X_tc_s[va_idx])[:, 1]
                s = roc_auc_score(y_tc, preds)
                if s > best_score:
                    best_score, best_c = s, c
            except Exception:
                continue

        # Refit on full train+cal, predict on test
        model = LogisticRegression(C=best_c, penalty="l2", solver="lbfgs",
                                   max_iter=1000, random_state=42)
        model.fit(X_tc_s, y_tc)
        y_te_proba = model.predict_proba(X_te_s)[:, 1]
        auroc = roc_auc_score(y_te, y_te_proba)

        # Patient-grouped bootstrap CI on test set
        rng = np.random.RandomState(42)
        boot = []
        unique_te_pids = te["patient_id"].unique()
        for _ in range(N_BOOTSTRAP):
            sampled = rng.choice(unique_te_pids, size=len(unique_te_pids), replace=True)
            mask = te["patient_id"].isin(sampled)
            if mask.sum() == 0 or len(np.unique(y_te[mask.values])) < 2:
                continue
            try:
                boot.append(roc_auc_score(y_te[mask.values], y_te_proba[mask.values]))
            except Exception:
                continue
        ci_low = float(np.percentile(boot, 2.5)) if boot else float("nan")
        ci_high = float(np.percentile(boot, 97.5)) if boot else float("nan")

        from sklearn.metrics import average_precision_score, brier_score_loss
        auprc = average_precision_score(y_te, y_te_proba)
        brier = brier_score_loss(y_te, y_te_proba)

        n_lesion = int((y_te == 1).sum())
        n_bg = int((y_te == 0).sum())

        results.append({
            "fm": fm_name,
            "task": task_name,
            "metric": "auroc",
            "value": float(auroc),
            "ci_low": ci_low,
            "ci_high": ci_high,
            "auprc": float(auprc),
            "brier": float(brier),
            "best_C": float(best_c),
            "n_train_patches": int(len(tc)),
            "n_test_patches": int(len(te)),
            "n_test_lesion": n_lesion,
            "n_test_background": n_bg,
            "view": default_view,
            "layer": default_layer,
        })
        print(f"  {fm_name}: AUROC={auroc:.3f} [{ci_low:.3f}, {ci_high:.3f}], "
              f"AUPRC={auprc:.3f}, Brier={brier:.3f}, "
              f"n_test={len(te)} patches ({n_lesion} lesion / {n_bg} bg)")

    return pd.DataFrame(results)


if "t1" in EMBEDDINGS and "t1" in LABELS:
    print("\n" + "=" * 60)
    print("T1: AutoPET-I FDG Lesion-Patch Classification (H1)")
    print("=" * 60)
    mode = task_eval_mode("t1")
    print(f"  eval mode: {mode}")
    if mode == "heldout":
        t1_results = run_t1_lesion_patch_probe(EMBEDDINGS["t1"], LABELS["t1"], "t1")
    else:
        print(f"  {mode!r} not implemented for T1 — needs Phase 4 v3 task_splits with train/test partition")
        t1_results = pd.DataFrame()
    t1_results = aggregate_random_init_seeds(t1_results)
    t1_results.to_csv(OUTPUT_DIR / "t1_lesion_patch_results.csv", index=False)
else:
    print("T1 data not available (gated on AutoPET-I FDAT preprocessing + embedding extraction)")
    t1_results = pd.DataFrame()


# %% [markdown]
# ## 3c. T5: AutoPET-III PSMA cross-tracer detection (per A9b, registration H5)
#
# **Zero-shot** evaluation: train probe on T1 (FDG) train embeddings, evaluate on
# entire T5 (PSMA) cohort. Per registration §3.1 + §5.6 zero-shot transfer line:
# "Train linear probe on T1 (AutoPET FDG) embeddings; evaluate zero-shot on T5
# (PSMA-PET-CT-Lesions)."
#
# Tests **registration H5**: FDG-trained linear probes show AUROC < 0.60 on PSMA
# for all FMs (cross-tracer failure hypothesis). Also reports cross-tracer
# transfer penalty (T1 within-tracer test AUROC − T5 zero-shot AUROC) per FM.

# %%
def run_t5_zero_shot_probe(t1_embeddings_dict, t1_labels_df, t5_embeddings_dict,
                            t5_labels_df, t1_test_aurocs):
    """Train per-FM linear probe on T1 train split, evaluate on T5 entire cohort.

    t1_test_aurocs: dict {fm_name: T1 within-tracer test AUROC} from
        run_t1_lesion_patch_probe; used to compute cross-tracer transfer penalty.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

    t1_splits = get_task_splits("t1")
    t5_splits = get_task_splits("t5")
    if t1_splits is None or t5_splits is None:
        print("  t1 or t5 task_splits missing — cannot run zero-shot")
        return pd.DataFrame()

    t1_train_pids = set(t1_splits.get("train", []) + t1_splits.get("cal", []))
    # v6 post-bug-fix (Step 67): T5's split label in task_splits.parquet is
    # "test_retest" (hardcoded by all_test() in 06_task_splits.py regardless of
    # the stratum_label argument), not the originally-anticipated
    # "test_zero_shot" or "test". Use the union of ALL T5 patients across any
    # split label — for zero-shot evaluation, the entire T5 cohort is the
    # held-out test set by registration §3.1 design.
    t5_test_pids = set()
    for split_label, pids in t5_splits.items():
        t5_test_pids.update(pids)

    print(f"  T1 train+cal: {len(t1_train_pids)} patients (probe fitting)")
    print(f"  T5 zero-shot: {len(t5_test_pids)} patients (held-out eval)")

    results = []
    common_fms = set(t1_embeddings_dict.keys()) & set(t5_embeddings_dict.keys())
    for fm_name in sorted(common_fms):
        t1_df, t5_df = t1_embeddings_dict[fm_name], t5_embeddings_dict[fm_name]

        # View/layer convention selection (same logic as T1)
        if "volume" in t1_df["view"].values:
            v, l = "volume", "pool"
        elif "axial" in t1_df["view"].values:
            v, l = "axial", "cls"
        else:
            v, l = sorted(t1_df["view"].unique())[0], sorted(t1_df["layer"].unique())[0]

        t1_sub = t1_df[(t1_df["view"] == v) & (t1_df["layer"] == l)]
        t5_sub = t5_df[(t5_df["view"] == v) & (t5_df["layer"] == l)]

        # T1 train: filter to T1 train+cal patients, join with t1 labels
        t1_train_merged = t1_sub.merge(
            t1_labels_df[["patient_id", "patch_id", "label"]],
            on=["patient_id", "patch_id"], how="inner",
        )
        t1_train_merged = t1_train_merged[
            t1_train_merged["patient_id"].astype(str).isin(t1_train_pids)
        ]

        # T5 eval: filter to T5 test patients, join with t5 labels
        t5_eval_merged = t5_sub.merge(
            t5_labels_df[["patient_id", "patch_id", "label"]],
            on=["patient_id", "patch_id"], how="inner",
        )
        t5_eval_merged = t5_eval_merged[
            t5_eval_merged["patient_id"].astype(str).isin(t5_test_pids)
        ]

        if len(t1_train_merged) < 50 or len(t5_eval_merged) < 20:
            print(f"  {fm_name}: insufficient train ({len(t1_train_merged)}) "
                  f"or eval ({len(t5_eval_merged)}), skipping")
            continue

        dim_cols = [c for c in t1_df.columns if c.startswith("d")]
        X_tr = t1_train_merged[dim_cols].values.astype(np.float32)
        y_tr = t1_train_merged["label"].values.astype(int)
        X_te = t5_eval_merged[dim_cols].values.astype(np.float32)
        y_te = t5_eval_merged["label"].values.astype(int)
        np.nan_to_num(X_tr, copy=False); np.nan_to_num(X_te, copy=False)

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        # Use a fixed C=1.0 for zero-shot (consistent with registration §5.4 default)
        model = LogisticRegression(C=1.0, penalty="l2", solver="lbfgs",
                                   max_iter=1000, random_state=42)
        model.fit(X_tr_s, y_tr)
        y_te_proba = model.predict_proba(X_te_s)[:, 1]

        if len(np.unique(y_te)) < 2:
            print(f"  {fm_name}: T5 eval set has only one class — AUROC undefined, skipping")
            continue
        auroc = roc_auc_score(y_te, y_te_proba)
        auprc = average_precision_score(y_te, y_te_proba)
        brier = brier_score_loss(y_te, y_te_proba)

        # Cross-tracer transfer penalty.
        # T1's per-FM AUROCs come from aggregated t1_results (random_init seeds
        # already collapsed to "random_init"). For ANY seed-prefixed T5 FM
        # (random_init_seed_*, random_init_seed*, etc.) we look up the aggregated
        # parent name to avoid NaN penalties on the multi-seed branch.
        # v6 post-bug-fix: use the regex match instead of literal startswith
        # to handle both `random_init_seed_0` and `random_init_seed0` naming.
        t1_lookup_key = fm_name
        if RANDOM_INIT_SEED_PATTERN.match(fm_name):
            t1_lookup_key = "random_init"
        t1_te_auroc = t1_test_aurocs.get(t1_lookup_key, float("nan"))
        transfer_penalty = float(t1_te_auroc - auroc) if not np.isnan(t1_te_auroc) else float("nan")

        results.append({
            "fm": fm_name,
            "task": "t5",
            "metric": "auroc_zero_shot",
            "value": float(auroc),
            "auprc": float(auprc),
            "brier": float(brier),
            "n_train_patches": int(len(t1_train_merged)),
            "n_eval_patches": int(len(t5_eval_merged)),
            "n_eval_lesion": int((y_te == 1).sum()),
            "n_eval_background": int((y_te == 0).sum()),
            "t1_within_tracer_auroc": float(t1_te_auroc),
            "cross_tracer_transfer_penalty": transfer_penalty,
            "view": v,
            "layer": l,
        })
        print(f"  {fm_name}: T5 zero-shot AUROC={auroc:.3f} "
              f"(T1 within-tracer={t1_te_auroc:.3f}, "
              f"transfer penalty={transfer_penalty:+.3f})")

    return pd.DataFrame(results)


if "t1" in EMBEDDINGS and "t5" in EMBEDDINGS and "t1" in LABELS and "t5" in LABELS:
    print("\n" + "=" * 60)
    print("T5: AutoPET-III PSMA Zero-Shot Detection (H5)")
    print("=" * 60)
    t1_test_aurocs = (
        dict(zip(t1_results["fm"], t1_results["value"]))
        if len(t1_results) and "value" in t1_results.columns
        else {}
    )
    t5_results = run_t5_zero_shot_probe(
        EMBEDDINGS["t1"], LABELS["t1"],
        EMBEDDINGS["t5"], LABELS["t5"],
        t1_test_aurocs,
    )
    # Aggregate multi-seed random_init seeds into median + IQR (per amendment A3).
    # T5 has 10 random_init_seed_N entries that need to collapse to one
    # "random_init" row before saving the formal results CSV.
    t5_results = aggregate_random_init_seeds(t5_results)
    t5_results.to_csv(OUTPUT_DIR / "t5_zero_shot_results.csv", index=False)
else:
    print("T5 data not available (needs T1 + T5 embeddings + labels)")
    t5_results = pd.DataFrame()


# %% [markdown]
# ## 3d. T2: HECKTOR 2025 HN tumour patch-classification (per A12a)
#
# **Per-patch** binary classifier (lesion vs background) for the HECKTOR T2 cohort
# (`task1_patient == True`, n=680 patients, 3,708 patches per the freeze metadata).
# Methodology IDENTICAL to T1: GroupKFold patient-level CV for inner C selection,
# patient-grouped bootstrap CI on test set, per-FM AUROC + AUPRC + Brier.
# Cohort filter applied via `hecktor_labels.parquet[task1_patient]`.
#
# Per A12a: Dice/HD95 evaluation reduced to patch-classification AUROC for
# (i) frozen-probe paradigm compatibility, (ii) cross-FM comparability, (iii)
# cross-task uniform AUROC space. Per A12a sensitivity: also report `lesion_class==1`
# (GTVp-only, primary-tumour-detection, n=629) alongside the binary union (which is
# cervical-node-detection-dominated since 1,186 of 1,854 lesion patches are GTVn).

# %%
def run_hecktor_lesion_patch_probe(embeddings_dict, labels_df, task_name="t2",
                                    cohort_filter="task1_patient",
                                    lesion_class_filter=None,
                                    splits_task=None):
    """Per-patch held-out classification probe — HECKTOR analogue of T1.

    Three extra filter arguments vs run_t1_lesion_patch_probe:
    - cohort_filter: column in labels_df to restrict to (T2 = task1_patient).
    - lesion_class_filter: optional set of lesion_class values for GTVp-only
      sensitivity (e.g. {1} for primary-tumour-only). Backgrounds (label=0,
      lesion_class=0) always retained.
    - splits_task: explicit override of which task_splits.parquet rows to use
      (defaults to task_name). Use "t2" when running the GTVp-only sensitivity
      to share the primary T2 patient train/test partition — the sensitivity
      differs by patch-class subset, NOT by patient-level split.
    """
    from sklearn.model_selection import GroupKFold

    splits_task = splits_task or task_name
    splits = get_task_splits(splits_task)
    if splits is None:
        print(f"  {task_name}: task_splits.parquet has no '{splits_task}' rows — "
              f"cannot run heldout")
        return pd.DataFrame()

    train_cal_pids = set(splits.get("train", []) + splits.get("cal", []))
    test_pids = set(splits.get("test", []))
    print(f"  {task_name} splits: train+cal={len(train_cal_pids)} patients, "
          f"test={len(test_pids)} patients")

    if cohort_filter not in labels_df.columns:
        print(f"  {task_name}: labels_df missing '{cohort_filter}' column, skipping")
        return pd.DataFrame()
    # CRITICAL: fillna(False) before astype(bool) — bool(NaN) is True in Python,
    # so any patient with a missing task1_patient/task2_patient flag would be
    # silently included in the cohort, inflating size + corrupting label dist.
    cohort_labels = labels_df[labels_df[cohort_filter].fillna(False).astype(bool)].copy()

    if lesion_class_filter is not None and "lesion_class" in cohort_labels.columns:
        keep_mask = (
            (cohort_labels["label"] == 0)
            | (cohort_labels["lesion_class"].isin(lesion_class_filter))
        )
        cohort_labels = cohort_labels[keep_mask].copy()
        print(f"  {task_name}: lesion_class filter={lesion_class_filter} → "
              f"{len(cohort_labels)} patches retained "
              f"({(cohort_labels['label']==1).sum()} lesion / "
              f"{(cohort_labels['label']==0).sum()} background)")

    if len(cohort_labels) == 0:
        print(f"  {task_name}: empty cohort after filter, skipping")
        return pd.DataFrame()

    results = []
    for fm_name, fm_df in embeddings_dict.items():
        if {"volume", "pool"}.issubset(set(fm_df["view"].unique()) | set(fm_df["layer"].unique())):
            default_view, default_layer = "volume", "pool"
        elif "axial" in fm_df["view"].values:
            default_view, default_layer = "axial", "cls"
        else:
            default_view = sorted(fm_df["view"].unique())[0]
            default_layer = sorted(fm_df["layer"].unique())[0]

        sub = fm_df[(fm_df["view"] == default_view) & (fm_df["layer"] == default_layer)]
        if len(sub) == 0:
            print(f"  {fm_name}: no rows at view={default_view} layer={default_layer}, skipping")
            continue

        merged = sub.merge(
            cohort_labels[["patient_id", "patch_id", "label"]],
            on=["patient_id", "patch_id"], how="inner",
        )
        if len(merged) == 0:
            print(f"  {fm_name}: empty join with labels, skipping")
            continue

        merged["_split"] = merged["patient_id"].astype(str).map(
            lambda p: "tc" if p in train_cal_pids else ("te" if p in test_pids else "drop")
        )
        merged = merged[merged["_split"] != "drop"].reset_index(drop=True)

        tc = merged[merged["_split"] == "tc"]
        te = merged[merged["_split"] == "te"]
        if len(tc) < 50 or len(te) < 20:
            print(f"  {fm_name}: insufficient train+cal ({len(tc)}) or test ({len(te)}), skipping")
            continue

        dim_cols = [c for c in fm_df.columns if c.startswith("d")]
        X_tc = tc[dim_cols].values.astype(np.float32)
        y_tc = tc["label"].values.astype(int)
        groups_tc = tc["patient_id"].values
        X_te = te[dim_cols].values.astype(np.float32)
        y_te = te["label"].values.astype(int)

        np.nan_to_num(X_tc, copy=False)
        np.nan_to_num(X_te, copy=False)

        scaler = StandardScaler()
        X_tc_s = scaler.fit_transform(X_tc)
        X_te_s = scaler.transform(X_te)

        n_splits_inner = min(5, len(np.unique(groups_tc)))
        if n_splits_inner < 2:
            print(f"  {fm_name}: <2 unique groups in train+cal, skipping")
            continue
        cv = GroupKFold(n_splits=n_splits_inner)
        best_c, best_score = 1.0, -1
        for c in C_GRID:
            try:
                preds = np.zeros_like(y_tc, dtype=float)
                for tr_idx, va_idx in cv.split(X_tc_s, y_tc, groups=groups_tc):
                    m = LogisticRegression(C=c, penalty="l2", solver="lbfgs",
                                           max_iter=1000, random_state=42)
                    m.fit(X_tc_s[tr_idx], y_tc[tr_idx])
                    preds[va_idx] = m.predict_proba(X_tc_s[va_idx])[:, 1]
                s = roc_auc_score(y_tc, preds)
                if s > best_score:
                    best_score, best_c = s, c
            except Exception:
                continue

        model = LogisticRegression(C=best_c, penalty="l2", solver="lbfgs",
                                   max_iter=1000, random_state=42)
        model.fit(X_tc_s, y_tc)
        y_te_proba = model.predict_proba(X_te_s)[:, 1]
        auroc = roc_auc_score(y_te, y_te_proba)

        rng = np.random.RandomState(42)
        boot = []
        unique_te_pids = te["patient_id"].unique()
        for _ in range(N_BOOTSTRAP):
            sampled = rng.choice(unique_te_pids, size=len(unique_te_pids), replace=True)
            mask = te["patient_id"].isin(sampled)
            if mask.sum() == 0 or len(np.unique(y_te[mask.values])) < 2:
                continue
            try:
                boot.append(roc_auc_score(y_te[mask.values], y_te_proba[mask.values]))
            except Exception:
                continue
        ci_low = float(np.percentile(boot, 2.5)) if boot else float("nan")
        ci_high = float(np.percentile(boot, 97.5)) if boot else float("nan")

        from sklearn.metrics import average_precision_score, brier_score_loss
        auprc = average_precision_score(y_te, y_te_proba)
        brier = brier_score_loss(y_te, y_te_proba)

        n_lesion = int((y_te == 1).sum())
        n_bg = int((y_te == 0).sum())

        results.append({
            "fm": fm_name, "task": task_name, "metric": "auroc",
            "value": float(auroc), "ci_low": ci_low, "ci_high": ci_high,
            "auprc": float(auprc), "brier": float(brier),
            "best_C": float(best_c),
            "n_train_patches": int(len(tc)), "n_test_patches": int(len(te)),
            "n_test_lesion": n_lesion, "n_test_background": n_bg,
            "view": default_view, "layer": default_layer,
        })
        print(f"  {fm_name}: AUROC={auroc:.3f} [{ci_low:.3f}, {ci_high:.3f}], "
              f"AUPRC={auprc:.3f}, Brier={brier:.3f}, "
              f"n_test={len(te)} patches ({n_lesion} lesion / {n_bg} bg)")

    return pd.DataFrame(results)


if "hecktor" in EMBEDDINGS and "hecktor" in LABELS:
    print("\n" + "=" * 60)
    print("T2: HECKTOR 2025 HN Tumour Patch-Classification (per A12a)")
    print("=" * 60)
    mode = task_eval_mode("t2")
    print(f"  eval mode: {mode}")
    if mode == "heldout":
        t2_results = run_hecktor_lesion_patch_probe(
            EMBEDDINGS["hecktor"], LABELS["hecktor"],
            task_name="t2", cohort_filter="task1_patient",
            lesion_class_filter=None,
        )
    else:
        print(f"  {mode!r} not implemented for T2 — needs Phase 4 v3 task_splits with train/test partition")
        t2_results = pd.DataFrame()
    t2_results = aggregate_random_init_seeds(t2_results)
    t2_results.to_csv(OUTPUT_DIR / "t2_lesion_patch_results.csv", index=False)

    # A12a sensitivity: GTVp-only (primary tumour detection only, excludes nodal)
    print("\n" + "=" * 60)
    print("T2 sensitivity: GTVp-only (primary tumour, per A12a)")
    print("=" * 60)
    if mode == "heldout":
        # Explicit splits_task="t2" — the GTVp-only sensitivity uses the SAME
        # patient train/test partition as the primary T2 analysis; only the
        # patch-class subset differs. Without this, get_task_splits would fail
        # to find "t2_gtvp_only" rows in task_splits.parquet and silently
        # produce an empty CSV.
        t2_gtvp_results = run_hecktor_lesion_patch_probe(
            EMBEDDINGS["hecktor"], LABELS["hecktor"],
            task_name="t2_gtvp_only", cohort_filter="task1_patient",
            lesion_class_filter={1}, splits_task="t2",
        )
    else:
        t2_gtvp_results = pd.DataFrame()
    t2_gtvp_results = aggregate_random_init_seeds(t2_gtvp_results)
    t2_gtvp_results.to_csv(OUTPUT_DIR / "t2_gtvp_only_results.csv", index=False)
else:
    print("HECKTOR data not available — T2 dispatch skipped")
    t2_results = pd.DataFrame()
    t2_gtvp_results = pd.DataFrame()

# %% [markdown]
# ## 4. Survival probe: T4 (NSCLC survival)
#
# Per registration: CoxPH with L2 penalty, 5-fold nested CV.
#
# **v2 changes** (after draft run produced near-chance c-indices and overflow
# warnings on FMCIB/CT-FM/DINOv2):
# 1. Alpha grid widened to `[0.01 … 1000]` — draft run repeatedly hit the
#    upper boundary (alpha=10), implying the optimum lay outside the search.
# 2. PCA(50) applied before Cox when `p > n` or `events < 100` — eliminates
#    the under-determined optimization that produces `exp` overflow on
#    high-dim FM embeddings (FMCIB 4096-d, BiomedCLIP 512-d) with only 63 events.
# 3. Scaler + PCA now fit **per fold** on training data only (no leakage).

# %%
def run_survival_probe(embeddings_dict, labels_df, task_name):
    """Run CoxPH survival probe for all FMs."""
    try:
        from sksurv.linear_model import CoxPHSurvivalAnalysis
        from sksurv.metrics import concordance_index_censored
    except ImportError:
        print("  scikit-survival not installed, skipping survival probes")
        return pd.DataFrame()

    results = []

    for fm_name, fm_df in embeddings_dict.items():
        id_col = "patient_id" if "patient_id" in fm_df.columns else "subject_id"
        patients = sorted(fm_df[id_col].unique())

        X_list, time_list, event_list, pid_list = [], [], [], []
        for pid in patients:
            emb = get_patient_embedding(fm_df, pid)
            if emb is None:
                continue
            label_row = labels_df[labels_df["patient_id"] == pid]
            if len(label_row) == 0:
                continue

            event = label_row["event"].values[0]
            time_val = label_row["time_to_death"].values[0]

            if pd.isna(event) or pd.isna(time_val) or time_val <= 0:
                continue

            X_list.append(emb)
            time_list.append(float(time_val))
            event_list.append(int(event))
            pid_list.append(pid)

        if len(X_list) < 20:
            print(f"  {fm_name}: too few samples ({len(X_list)}), skipping")
            continue

        X = np.array(X_list)
        time_arr = np.array(time_list)
        event_arr = np.array(event_list)

        # Drop NaN embeddings
        nan_mask = np.isnan(X).any(axis=1)
        if nan_mask.sum() > 0:
            print(f"  {fm_name}: dropping {nan_mask.sum()} NaN embeddings")
            X = X[~nan_mask]
            time_arr = time_arr[~nan_mask]
            event_arr = event_arr[~nan_mask]
        np.nan_to_num(X, copy=False)

        if len(X) < 20:
            print(f"  {fm_name}: too few samples after NaN removal ({len(X)}), skipping")
            continue

        # Create structured array
        y_struct = np.array(
            [(bool(e), float(t)) for e, t in zip(event_arr, time_arr)],
            dtype=[("event", bool), ("time", float)]
        )

        # v2: decide whether to PCA-reduce before Cox. Under p≫n or with
        # <100 events, raw FM embeddings produce a Cox optimization problem
        # that is severely under-determined — risk scores explode and
        # `np.exp(xw)` overflows. ~50 PCs preserve the signal without overfit
        # and match the convention used by Merlin/FMCIB downstream protocols.
        n_events = int(event_arr.sum())
        use_pca = (X.shape[1] > X.shape[0]) or (n_events < 100)
        pca_dim = min(50, X.shape[0] - 1, X.shape[1]) if use_pca else None

        # v3: alpha grid widened in both directions. v1 was censored upward
        # (CV picked alpha=10 = upper bound). v2 widened to 1000 upward but
        # CT-FM dry run picked alpha=0.01 = new lower bound, so we extend
        # downward too.
        alpha_grid = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
        cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

        # v2: scaler + PCA fit per-fold on training data only (no leakage).
        def _fit_transform_fold(X_train_raw, X_test_raw):
            scaler_f = StandardScaler()
            X_tr = scaler_f.fit_transform(X_train_raw)
            X_te = scaler_f.transform(X_test_raw)
            if use_pca:
                pca = PCA(n_components=pca_dim, random_state=42)
                X_tr = pca.fit_transform(X_tr)
                X_te = pca.transform(X_te)
            return X_tr, X_te

        best_alpha, best_score = 1.0, -1
        for alpha in alpha_grid:
            fold_scores = []
            for train_idx, test_idx in cv.split(X, event_arr):
                X_tr, X_te = _fit_transform_fold(X[train_idx], X[test_idx])
                model = CoxPHSurvivalAnalysis(alpha=alpha)
                try:
                    model.fit(X_tr, y_struct[train_idx])
                    preds = model.predict(X_te)
                    c, _, _, _, _ = concordance_index_censored(
                        event_arr[test_idx].astype(bool), time_arr[test_idx], preds
                    )
                    fold_scores.append(c)
                except Exception:
                    continue
            if fold_scores and np.mean(fold_scores) > best_score:
                best_score = np.mean(fold_scores)
                best_alpha = alpha

        # Out-of-fold c-index
        risk_scores = np.zeros(len(X))
        for train_idx, test_idx in cv.split(X, event_arr):
            X_tr, X_te = _fit_transform_fold(X[train_idx], X[test_idx])
            model = CoxPHSurvivalAnalysis(alpha=best_alpha)
            try:
                model.fit(X_tr, y_struct[train_idx])
                risk_scores[test_idx] = model.predict(X_te)
            except Exception:
                risk_scores[test_idx] = 0.0

        c_index, _, _, _, _ = concordance_index_censored(
            event_arr.astype(bool), time_arr, risk_scores
        )

        # Bootstrap CI
        rng = np.random.RandomState(42)
        boot_cs = []
        for _ in range(N_BOOTSTRAP):
            idx = rng.choice(len(event_arr), size=len(event_arr), replace=True)
            if event_arr[idx].sum() < 2:
                continue
            try:
                c, _, _, _, _ = concordance_index_censored(
                    event_arr[idx].astype(bool), time_arr[idx], risk_scores[idx]
                )
                boot_cs.append(c)
            except Exception:
                continue

        ci_low = np.percentile(boot_cs, 2.5) if boot_cs else np.nan
        ci_high = np.percentile(boot_cs, 97.5) if boot_cs else np.nan

        results.append({
            "task": task_name, "fm": fm_name, "metric": "c_index",
            "value": float(c_index), "ci_low": float(ci_low), "ci_high": float(ci_high),
            "best_alpha": best_alpha, "n_patients": len(X),
            "n_events": int(event_arr.sum()),
        })
        print(f"  {fm_name}: c-index={c_index:.3f} [{ci_low:.3f}, {ci_high:.3f}] (alpha={best_alpha}, n={len(X)}, events={event_arr.sum()})")

    return pd.DataFrame(results)


if "t4" in EMBEDDINGS and "t4" in LABELS:
    print("\n" + "=" * 60)
    print("T4: NSCLC Survival Prediction")
    print("=" * 60)
    t4_results = run_survival_probe(EMBEDDINGS["t4"], LABELS["t4"], "t4")
    t4_results = aggregate_random_init_seeds(t4_results)
    t4_results.to_csv(OUTPUT_DIR / "t4_survival_results.csv", index=False)
else:
    print("T4 data not available")
    t4_results = pd.DataFrame()


# %% [markdown]
# ## 4b. Survival probe: T3 (HECKTOR 2025 RFS prediction, per A12c)
#
# **Patient-pooled** CoxPH for the HECKTOR T3 cohort (`task2_patient == True`,
# n=651 patients with valid Relapse + RFS, 132 events / 20.3% event rate per the
# preprocessing QC). Labels parquet uses `relapse` (event indicator) and `rfs_days`
# (time-to-event) — distinct from T4's `event` / `time_to_death` schema.
#
# Critical difference from T4: HECKTOR embeddings have ONE row per (patient_id,
# patch_id, view, layer) — multi-patch per patient. T4 has one row per patient.
# For T3 we mean-pool across the patient's lesion patches (label==1) to produce
# a single patient-level embedding before CoxPH. Background patches excluded
# from pooling (the survival signal lives in the lesions, not the SUV-low background).
#
# Methodology otherwise IDENTICAL to T4: alpha grid {0.001-1000}, per-fold
# scaler+PCA fit on training data only, 5-fold StratifiedKFold on event indicator,
# patient-clustered bootstrap CI. Per A12c projection: c-index CI half-width ~±0.04
# vs registration's original ~±0.025 (132 events vs the projected ~250).
#
# **PCA condition tightened vs T4** to handle T3's intermediate event count (132).
# T4's `n_events < 100` condition does NOT fire on T3, but 512-d FMs (CT-FM,
# BiomedCLIP) at 132 events have an EPV (events-per-variable) of ~0.26 — well below
# the canonical 5-EPV rule (Vittinghoff/Peduzzi). Without PCA, Cox optimisation on
# 512-d/132-event would be severely under-determined. Solution: add an EPV-based
# clause `dim/events > 3` that fires on all current FM × T3 combinations
# (DINOv2/RAD-DINO 768-d → ratio 5.8, BiomedCLIP/CT-FM 512-d → ratio 3.9, FMCIB
# 4096-d → ratio 31.0). T4 behaviour unchanged because its existing `n_events<100`
# condition already fires for all FMs at 63 events.

# %%
def run_hecktor_survival_probe(embeddings_dict, labels_df, task_name="t3"):
    """Patient-pooled CoxPH probe for HECKTOR RFS (T3).

    Filters to task2_patient cohort with valid relapse + rfs_days. Mean-pools
    embeddings across the patient's lesion patches (label==1) for one row per patient.
    """
    try:
        from sksurv.linear_model import CoxPHSurvivalAnalysis
        from sksurv.metrics import concordance_index_censored
    except ImportError:
        print("  scikit-survival not installed, skipping survival probes")
        return pd.DataFrame()

    # Filter labels to T3 cohort with valid survival data
    if "task2_patient" not in labels_df.columns:
        print(f"  {task_name}: labels_df missing 'task2_patient' column, skipping")
        return pd.DataFrame()

    # CRITICAL: fillna(False) before astype(bool) — bool(NaN) is True in Python.
    # Without this, any patient with a missing task2_patient flag would be silently
    # included in the T3 cohort.
    t3_eligible = labels_df[
        labels_df["task2_patient"].fillna(False).astype(bool)
        & labels_df["relapse"].notna()
        & labels_df["rfs_days"].notna()
        & (labels_df["rfs_days"] > 0)
    ].copy()
    n_eligible_patients = t3_eligible["patient_id"].nunique()
    t3_labels = t3_eligible[t3_eligible["label"] == 1].copy()
    n_label_filtered_patients = t3_labels["patient_id"].nunique()
    # v6 post-final-review diagnostic (Step 65): surface any patients lost to the
    # `label==1` filter (i.e., patients whose patches in the parquet are all
    # background). Pre-filter shortfall = silent cohort deflation, would otherwise
    # only show up in the printed n_patients later.
    if n_label_filtered_patients < n_eligible_patients:
        print(f"  {task_name}: {n_eligible_patients - n_label_filtered_patients} "
              f"patients lost to label==1 filter "
              f"({n_eligible_patients} eligible → {n_label_filtered_patients} with lesion patches)")
    if len(t3_labels) == 0:
        print(f"  {task_name}: empty cohort after filter, skipping")
        return pd.DataFrame()

    # Per-patient survival metadata (deduplicate across patches; relapse + rfs_days
    # are patient-level, replicated across the patient's lesion-patch rows).
    patient_meta = (
        t3_labels.drop_duplicates("patient_id")
        [["patient_id", "relapse", "rfs_days"]]
        .reset_index(drop=True)
    )
    print(f"  {task_name}: {len(patient_meta)} patients in cohort, "
          f"{int(patient_meta['relapse'].sum())} events "
          f"({100 * patient_meta['relapse'].mean():.1f}%)")

    results = []
    for fm_name, fm_df in embeddings_dict.items():
        if {"volume", "pool"}.issubset(set(fm_df["view"].unique()) | set(fm_df["layer"].unique())):
            default_view, default_layer = "volume", "pool"
        elif "axial" in fm_df["view"].values:
            default_view, default_layer = "axial", "cls"
        else:
            default_view = sorted(fm_df["view"].unique())[0]
            default_layer = sorted(fm_df["layer"].unique())[0]

        sub = fm_df[(fm_df["view"] == default_view) & (fm_df["layer"] == default_layer)]
        if len(sub) == 0:
            print(f"  {fm_name}: no rows at view={default_view} layer={default_layer}, skipping")
            continue

        # Restrict embeddings to lesion patches of T3 patients (per t3_labels).
        # Vectorised merge instead of row-wise apply (~1.2M ops → single hash join).
        keep_df = t3_labels[["patient_id", "patch_id"]].copy()
        keep_df["patient_id"] = keep_df["patient_id"].astype(str)
        keep_df["patch_id"] = keep_df["patch_id"].astype(str)
        sub = sub.copy()
        sub["patient_id"] = sub["patient_id"].astype(str)
        sub["patch_id"] = sub["patch_id"].astype(str)
        sub = sub.merge(keep_df, on=["patient_id", "patch_id"], how="inner")
        if len(sub) == 0:
            print(f"  {fm_name}: no T3 lesion patches in this FM's embeddings, skipping")
            continue

        # Mean-pool across patient's lesion patches → one row per patient
        dim_cols = [c for c in fm_df.columns if c.startswith("d")]
        pooled = sub.groupby("patient_id")[dim_cols].mean().reset_index()

        # Join with patient survival metadata
        merged = pooled.merge(patient_meta, on="patient_id", how="inner")
        if len(merged) < 20:
            print(f"  {fm_name}: too few patients after pooling+merge ({len(merged)}), skipping")
            continue

        X = merged[dim_cols].values.astype(np.float32)
        time_arr = merged["rfs_days"].values.astype(float)
        event_arr = merged["relapse"].values.astype(int)

        nan_mask = np.isnan(X).any(axis=1)
        if nan_mask.sum() > 0:
            print(f"  {fm_name}: dropping {nan_mask.sum()} NaN embeddings")
            X = X[~nan_mask]
            time_arr = time_arr[~nan_mask]
            event_arr = event_arr[~nan_mask]
        np.nan_to_num(X, copy=False)
        if len(X) < 20:
            print(f"  {fm_name}: too few samples after NaN removal ({len(X)}), skipping")
            continue

        y_struct = np.array(
            [(bool(e), float(t)) for e, t in zip(event_arr, time_arr)],
            dtype=[("event", bool), ("time", float)]
        )

        n_events = int(event_arr.sum())
        # T3-tightened PCA condition: also fire when EPV (events per variable) is
        # below 3, catching 512-d FMs at 132 events that T4's `n_events<100` misses.
        # Ratio dim/events > 3 is equivalent to events/dim < 1/3, well below the
        # canonical 5-EPV rule. Conservative across all FM × T3 combinations.
        epv_ratio = X.shape[1] / max(1, n_events)
        use_pca = (
            (X.shape[1] > X.shape[0])
            or (n_events < 100)
            or (epv_ratio > 3)
        )
        pca_dim = min(50, X.shape[0] - 1, X.shape[1]) if use_pca else None

        alpha_grid = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
        cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

        def _fit_transform_fold(X_train_raw, X_test_raw):
            scaler_f = StandardScaler()
            X_tr = scaler_f.fit_transform(X_train_raw)
            X_te = scaler_f.transform(X_test_raw)
            if use_pca:
                pca = PCA(n_components=pca_dim, random_state=42)
                X_tr = pca.fit_transform(X_tr)
                X_te = pca.transform(X_te)
            return X_tr, X_te

        best_alpha, best_score = 1.0, -1
        for alpha in alpha_grid:
            fold_scores = []
            for train_idx, test_idx in cv.split(X, event_arr):
                X_tr, X_te = _fit_transform_fold(X[train_idx], X[test_idx])
                model = CoxPHSurvivalAnalysis(alpha=alpha)
                try:
                    model.fit(X_tr, y_struct[train_idx])
                    preds = model.predict(X_te)
                    c, _, _, _, _ = concordance_index_censored(
                        event_arr[test_idx].astype(bool), time_arr[test_idx], preds
                    )
                    fold_scores.append(c)
                except Exception:
                    continue
            if fold_scores and np.mean(fold_scores) > best_score:
                best_score = np.mean(fold_scores)
                best_alpha = alpha

        risk_scores = np.zeros(len(X))
        for train_idx, test_idx in cv.split(X, event_arr):
            X_tr, X_te = _fit_transform_fold(X[train_idx], X[test_idx])
            model = CoxPHSurvivalAnalysis(alpha=best_alpha)
            try:
                model.fit(X_tr, y_struct[train_idx])
                risk_scores[test_idx] = model.predict(X_te)
            except Exception:
                risk_scores[test_idx] = 0.0

        c_index, _, _, _, _ = concordance_index_censored(
            event_arr.astype(bool), time_arr, risk_scores
        )

        rng = np.random.RandomState(42)
        boot_cs = []
        for _ in range(N_BOOTSTRAP):
            idx = rng.choice(len(event_arr), size=len(event_arr), replace=True)
            if event_arr[idx].sum() < 2:
                continue
            try:
                c, _, _, _, _ = concordance_index_censored(
                    event_arr[idx].astype(bool), time_arr[idx], risk_scores[idx]
                )
                boot_cs.append(c)
            except Exception:
                continue

        ci_low = np.percentile(boot_cs, 2.5) if boot_cs else np.nan
        ci_high = np.percentile(boot_cs, 97.5) if boot_cs else np.nan

        results.append({
            "task": task_name, "fm": fm_name, "metric": "c_index",
            "value": float(c_index), "ci_low": float(ci_low), "ci_high": float(ci_high),
            "best_alpha": best_alpha, "n_patients": len(X),
            "n_events": int(event_arr.sum()),
            "use_pca": bool(use_pca), "epv_ratio": float(epv_ratio),
        })
        print(f"  {fm_name}: c-index={c_index:.3f} [{ci_low:.3f}, {ci_high:.3f}] "
              f"(alpha={best_alpha}, n={len(X)}, events={event_arr.sum()}, "
              f"PCA={'on' if use_pca else 'off'}, EPV={1/epv_ratio:.2f})")

    return pd.DataFrame(results)


if "hecktor" in EMBEDDINGS and "hecktor" in LABELS:
    print("\n" + "=" * 60)
    print("T3: HECKTOR 2025 RFS Survival Prediction (per A12c)")
    print("=" * 60)
    t3_results = run_hecktor_survival_probe(
        EMBEDDINGS["hecktor"], LABELS["hecktor"], "t3"
    )
    t3_results = aggregate_random_init_seeds(t3_results)
    t3_results.to_csv(OUTPUT_DIR / "t3_survival_results.csv", index=False)
else:
    print("HECKTOR data not available — T3 dispatch skipped")
    t3_results = pd.DataFrame()

# %% [markdown]
# ## 5. Test-retest stability: T9 (healthy controls) + T6 (cancer patients)
#
# **Headline metric — Lin's CCC** (registration §1.3 H6, §3.1 primary).
# Computed per-embedding-dimension across the test-retest cohort, then
# averaged across dimensions. This is the canonical radiomics test-retest
# convention (Aerts et al. 2014) and is the registration-primary measure
# of embedding stability.
#
# **Secondary discriminative augmentation — cosine gap (Sw − Sb).** The v3
# dry-run revealed `random_init` cosines of 0.97+, comparable to trained
# FMs, because pooled features from a 3D conv net are dominated by gross
# input statistics (volume size, intensity histogram, body position)
# regardless of weights. Reporting `Sw` alone cannot distinguish "model
# representation is patient-specific" from "model output is patient-agnostic
# but the inputs look similar." We therefore also report:
#
# - `Sw` = mean cosine of (patient_a session_1, patient_a session_2) — within-patient
# - `Sb` = mean cosine of (patient_a session_1, patient_b session_1) for b≠a — between-patient
# - Gap = Sw − Sb (1000-permutation p-value for Sw > Sb)
#
# Both metrics are saved; manuscript reporting follows the registration
# (CCC primary, cosine secondary, gap as the discriminative complement).

# %%
def run_test_retest(embeddings_dict, manifest_df, task_name):
    """Compute test-retest embedding stability for all FMs.

    Pairs are derived from the embedding parquets themselves — every parquet
    has (patient_id|subject_id, session) columns and only contains rows for
    sessions that actually got embedded. This is the only source of truth
    that's guaranteed to match the embeddings (manifest CSVs in the patches
    dataset can pre-date file-existence filtering and be misleading).
    """
    results = []

    if not embeddings_dict:
        print(f"  No embeddings available for {task_name}")
        return pd.DataFrame()

    # Discover (id, session) groups from the first FM parquet
    first_fm_df = next(iter(embeddings_dict.values()))
    id_col = "patient_id" if "patient_id" in first_fm_df.columns else "subject_id"
    if "session" not in first_fm_df.columns:
        print(f"  No 'session' column in {task_name} embeddings — cannot run test-retest")
        return pd.DataFrame()

    sessions_per_id = (
        first_fm_df[[id_col, "session"]]
        .drop_duplicates()
        .groupby(id_col)["session"]
        .apply(lambda s: sorted(s.tolist()))
    )

    # T6 also needs to restrict to test-retest-flagged patients (the rest are
    # single-study cancer patients, not test-retest cases).
    if task_name == "t6" and manifest_df is not None:
        retest_col = "is_retest_patient"
        if retest_col in manifest_df.columns:
            retest_ids = set(
                manifest_df.loc[manifest_df[retest_col] == True, id_col].tolist()
            )
            sessions_per_id = sessions_per_id[sessions_per_id.index.isin(retest_ids)]

    pairs = [(pid, s[0], s[1]) for pid, s in sessions_per_id.items() if len(s) >= 2]

    if not pairs:
        print(f"  No test-retest pairs found for {task_name}")
        return pd.DataFrame()

    print(f"  Found {len(pairs)} test-retest pairs")

    def _cosine(a, b):
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na == 0 or nb == 0:
            return None
        return float(np.dot(a, b) / (na * nb))

    for fm_name, fm_df in embeddings_dict.items():
        # === Within-patient cosines (Sw) + paired embedding arrays (for CCC) ===
        sw_cosines = []
        sw_pairs_used = []  # for between-patient sampling: track first-session embeddings
        ccc_test_embs = []  # session-1 embeddings (paired with retest)
        ccc_retest_embs = []  # session-2 embeddings (paired with test)
        for pid, sess1, sess2 in pairs:
            emb1 = get_session_embedding(fm_df, pid, sess1)
            emb2 = get_session_embedding(fm_df, pid, sess2)
            if emb1 is None or emb2 is None:
                continue
            cos = _cosine(emb1, emb2)
            if cos is not None:
                sw_cosines.append(cos)
                sw_pairs_used.append((pid, emb1))  # keep first-session embedding
                ccc_test_embs.append(emb1)
                ccc_retest_embs.append(emb2)

        if not sw_cosines:
            print(f"  {fm_name}: no valid pairs")
            continue

        sw_cosines = np.array(sw_cosines)

        # === Lin's CCC (registration-primary metric per H6) ===
        ccc_value = embedding_ccc(np.array(ccc_test_embs), np.array(ccc_retest_embs))
        # Bootstrap CI for CCC at the pair level (resample patients).
        rng_ccc = np.random.RandomState(42)
        boot_cccs = []
        n_pairs = len(ccc_test_embs)
        ccc_test_arr = np.array(ccc_test_embs)
        ccc_retest_arr = np.array(ccc_retest_embs)
        for _ in range(N_BOOTSTRAP):
            idx = rng_ccc.choice(n_pairs, size=n_pairs, replace=True)
            boot_cccs.append(embedding_ccc(ccc_test_arr[idx], ccc_retest_arr[idx]))
        boot_cccs = np.array([c for c in boot_cccs if not np.isnan(c)])
        if len(boot_cccs):
            ccc_ci_low = float(np.percentile(boot_cccs, 2.5))
            ccc_ci_high = float(np.percentile(boot_cccs, 97.5))
        else:
            ccc_ci_low = ccc_ci_high = float("nan")

        # === Between-patient cosines (Sb) ===
        # For each within-patient pair (pid_a, emb_a), sample n_b random other
        # patients b ≠ a and compute cosine(emb_a, emb_b_session1). Report
        # mean over all (a, b) tuples.
        rng = np.random.RandomState(42)
        sb_cosines = []
        n_other_per_anchor = 5  # 5 between-patient comparisons per anchor patient
        n_anchors = len(sw_pairs_used)
        if n_anchors >= 2:
            for i, (pid_a, emb_a) in enumerate(sw_pairs_used):
                # Sample up to n_other_per_anchor other anchors
                other_idx = [j for j in range(n_anchors) if j != i]
                k = min(n_other_per_anchor, len(other_idx))
                sampled = rng.choice(other_idx, size=k, replace=False)
                for j in sampled:
                    pid_b, emb_b = sw_pairs_used[j]
                    cos = _cosine(emb_a, emb_b)
                    if cos is not None:
                        sb_cosines.append(cos)
        sb_cosines = np.array(sb_cosines) if sb_cosines else np.array([np.nan])

        # === Bootstrap CI for Sw ===
        rng_b = np.random.RandomState(42)
        boot_sw_means = []
        for _ in range(N_BOOTSTRAP):
            idx = rng_b.choice(len(sw_cosines), size=len(sw_cosines), replace=True)
            boot_sw_means.append(sw_cosines[idx].mean())
        sw_ci_low = float(np.percentile(boot_sw_means, 2.5))
        sw_ci_high = float(np.percentile(boot_sw_means, 97.5))

        sw_mean = float(sw_cosines.mean())
        sb_mean = float(np.nanmean(sb_cosines))
        gap = sw_mean - sb_mean

        # === Permutation test for Sw > Sb ===
        # H0: within- and between-patient cosines are exchangeable (gap = 0).
        # Pool, shuffle labels, compute gap; p = fraction of permuted gaps ≥ observed.
        if len(sb_cosines) > 0 and not np.isnan(sb_cosines).all():
            pooled = np.concatenate([sw_cosines, sb_cosines])
            n_sw = len(sw_cosines)
            rng_p = np.random.RandomState(42)
            perm_gaps = np.empty(N_BOOTSTRAP)
            for k in range(N_BOOTSTRAP):
                shuf = rng_p.permutation(pooled)
                perm_gaps[k] = shuf[:n_sw].mean() - shuf[n_sw:].mean()
            p_value = float((perm_gaps >= gap).mean())
        else:
            p_value = float("nan")

        results.append({
            "task": task_name, "fm": fm_name, "metric": "lin_ccc",
            # v4: Lin's CCC is now the registration-primary headline metric
            # (§1.3 H6, §3.1). The cosine-gap (Sw − Sb) is reported
            # alongside as a discriminative augmentation.
            "value": ccc_value,
            "ci_low": ccc_ci_low,
            "ci_high": ccc_ci_high,
            "sw_mean": sw_mean,
            "sw_std": float(sw_cosines.std()),
            "sw_min": float(sw_cosines.min()),
            "sw_ci_low": sw_ci_low,
            "sw_ci_high": sw_ci_high,
            "sb_mean": sb_mean,
            "sb_std": float(np.nanstd(sb_cosines)),
            "cosine_gap": gap,
            "cosine_gap_ci_low": sw_ci_low - sb_mean,
            "cosine_gap_ci_high": sw_ci_high - sb_mean,
            "n_within_pairs": len(sw_cosines),
            "n_between_pairs": int(np.sum(~np.isnan(sb_cosines))),
            "perm_p_value_sw_gt_sb": p_value,
        })
        print(f"  {fm_name}: CCC={ccc_value:.4f} [{ccc_ci_low:.4f}, {ccc_ci_high:.4f}] "
              f"| Sw={sw_mean:.4f} Sb={sb_mean:.4f} gap={gap:+.4f} "
              f"(p={p_value:.3f}, n_w={len(sw_cosines)}, n_b={len(sb_cosines)})")

    return pd.DataFrame(results)


# T9 test-retest — pairs derived from embedding parquets directly (no manifest needed)
if "t9" in EMBEDDINGS:
    print("\n" + "=" * 60)
    print("T9: Healthy-Control Test-Retest Stability")
    print("=" * 60)
    t9_results = run_test_retest(EMBEDDINGS["t9"], None, "t9")
    t9_results = aggregate_random_init_seeds(t9_results)
    if len(t9_results) > 0:
        t9_results.to_csv(OUTPUT_DIR / "t9_test_retest_results.csv", index=False)
else:
    t9_results = pd.DataFrame()

# T6 test-retest — needs labels CSV to filter to retest-flagged patients only
if "t6" in EMBEDDINGS:
    print("\n" + "=" * 60)
    print("T6: Cancer-Patient Test-Retest Stability")
    print("=" * 60)
    t6_manifest = LABELS.get("t6")  # t6_labels.csv has is_retest_patient flag
    if t6_manifest is None:
        print("  WARNING: t6_labels.csv not found — running test-retest over ALL "
              "T6 patients with 2+ sessions (may include non-retest pairs)")
    t6_results = run_test_retest(EMBEDDINGS["t6"], t6_manifest, "t6")
    t6_results = aggregate_random_init_seeds(t6_results)
    if len(t6_results) > 0:
        t6_results.to_csv(OUTPUT_DIR / "t6_test_retest_results.csv", index=False)
else:
    t6_results = pd.DataFrame()


# v4: merge IBSI-radiomics CCC baseline (Step 14 / amendment A8) into t6/t9
# test-retest results so the H6 comparator (registration §1.3 H6) appears
# alongside the FM CCC values in the final cross-FM summary table.
def _load_pyradiomics_baseline():
    """Locate `pyradiomics_test_retest_ccc.csv` in attached datasets."""
    candidates = list(Path("/kaggle/input").rglob("pyradiomics_test_retest_ccc.csv"))
    if not candidates:
        return None
    try:
        df = pd.read_csv(candidates[0])
        print(f"  → IBSI radiomics baseline (H6) loaded: {len(df)} rows from "
              f"{candidates[0]}")
        return df
    except Exception as e:
        print(f"  ⚠ could not load pyradiomics baseline: {e}")
        return None


_pyradiomics_baseline = _load_pyradiomics_baseline()
if _pyradiomics_baseline is not None:
    # Append matching task rows to t6_results / t9_results for the cross-FM table
    for task_key, results_var_name in [("t6", "t6_results"), ("t9", "t9_results")]:
        sub = _pyradiomics_baseline[_pyradiomics_baseline["task"] == task_key]
        if len(sub) == 0:
            continue
        existing = locals().get(results_var_name, pd.DataFrame())
        merged = pd.concat([existing, sub], ignore_index=True)
        # Re-assign back to globals so cross-FM summary section sees the merged frame
        if task_key == "t6":
            t6_results = merged
        else:
            t9_results = merged
        print(f"  → merged {len(sub)} pyradiomics row(s) into {task_key}_results")

# %% [markdown]
# ## 5b. T7 Response prediction probe (ACRIN-NSCLC-FDG-PET)
#
# Pre-registered: response prediction from baseline PET scan. Outcome is from
# the ACRIN-6668 clinical CSVs (downloaded by t7_01_preprocess and packaged
# inside the patches dataset).
#
# **v4 outcome-resolution priority** (verified against ACRIN-6668 data
# dictionary, 2026-04-27):
#
# 1. **Derivation (default, registration-aligned).** `T7_OUTCOME_DERIVATION
#    = "2yr_os"` engages `derive_t7_2yr_os()` which combines F1.csv (vital
#    status + visit dates) and DS.csv (premature discontinuation + dates)
#    to produce a binary `event_2yr_os` per patient. This matches the
#    ACRIN-6668 trial's primary endpoint per Machtay et al. 2013. Local
#    smoke-test against the published cohort produced 57.2% 2y mortality
#    (Machtay reported 57.5%) — match within 0.3pp.
# 2. **Explicit override** via `T7_OUTCOME_OVERRIDE = (filename, column)`.
#    Used only if `T7_OUTCOME_DERIVATION` is set to `None`. For sensitivity
#    analyses pinning a specific column.
# 3. **Auto-discovery** (last resort): ranks columns by name pattern +
#    cardinality. **WARNING:** the dry-run scoring would have ranked
#    `(DS.csv, rec)` highly, but the data dictionary reveals `rec` in DS.csv
#    is "Data receipt (from base date)" — a metadata field about when the
#    record was received by the data center, NOT recurrence. Auto-discovery
#    is therefore **not recommended** for the registered analysis; use the
#    derivation path instead.
#
# Column mapping confirmed against `QIN_6668 Data Dictionary.xls`:
#   F1.f1e2  = PATIENT'S VITAL STATUS (1=Alive, 2=Dead, 3=LTFU, 9=Dead-DOD-unk)
#   F1.F1e1d = days from baseline to clinical assessment (every visit)
#   F1.F1e3d = days from baseline to last contact OR death (final visit only)
#   DS.dse3  = primary reason for premature discontinuation (1=Withdraw,
#              2=Death, 3=LTFU, 88=Other)
#   DS.DSe2d = days from baseline to date of premature discontinuation

# %%
# v3: ACRIN-6668 ships clinical data in a CDISC-style multi-CSV schema, where
# each case-report form has its own CSV (DS = Disease Status, A0 = enrolment,
# LE = Lesions, F1/F2 = Follow-up, etc.) and column names follow a
# `<form><field><N>` pattern (e.g., `dse10`, `lee14`, `o1e3`). The dry-run
# discovery code searched for clinical-trial-summary names like "BestResponse"
# and never matched. v3 takes two complementary approaches:
#
# 1. T7_OUTCOME_OVERRIDE — explicit (filename, column) pin. Use this once the
#    ACRIN-6668 data dictionary has been consulted. RECOMMENDED for the formal
#    run.
# 2. Auto-discovery — scans every CSV for binary/categorical columns whose
#    names match outcome-like patterns (rec, dse10, ose, prog, surv, status,
#    response, vital). Used as a fallback when T7_OUTCOME_OVERRIDE is None,
#    and prints the candidate ranking so the user can pick the right one.
#
# The DS.csv `rec` column is the leading candidate based on column-name
# inspection (DS = Disease Status, rec = recurrence flag) — a sensible default
# but should be VERIFIED against the ACRIN-6668 data dictionary.

T7_OUTCOME_OVERRIDE = None  # e.g., ("DS.csv", "rec") to pin explicitly

# v4: registration-aligned T7 outcome is 2-year overall survival
# (Machtay et al. 2013, J Clin Oncol 31:3823 — primary endpoint).
# This must be DERIVED from F1.csv (vital status + days to last contact/death)
# and DS.csv (premature discontinuation reason + date), not read from a
# single column. Setting T7_OUTCOME_DERIVATION engages the derivation path
# and supersedes T7_OUTCOME_OVERRIDE if both are set.
#
# Verified locally 2026-04-27 against ACRIN-6668 File Set 1: derivation
# produces 57.2% 2-year mortality (Machtay 2013 published value: 57.5%),
# 180/230 patients labelled, 50 censored < 2y dropped.
T7_OUTCOME_DERIVATION = "2yr_os"  # None | "2yr_os"
T7_OS_THRESHOLD_DAYS = 730        # 2 years

T7_OUTCOME_PATTERNS = [
    # CDISC-style ACRIN-6668 candidates (case-insensitive substring match).
    # These are the columns most likely to encode response / progression /
    # survival across the 20 ACRIN forms.
    "rec",            # recurrence flag — appears in DS, A1, SF, PR forms
    "dse10",          # disease status form, 2-year follow-up question
    "dse12",          # disease status, late follow-up
    "ose",            # outcome status (O1.csv form)
    "vital",          # vital status
    "deat",           # death indicator
    "death",
    "surv",           # survival
    "prog",           # progression
    "response",
    "best_response",
    "bestresponse",
    "status",
]


def find_t7_clinical_csvs():
    """Locate ACRIN clinical CSVs from the t7-patches dataset.

    v5: dedupe by absolute path only — NOT by filename. ACRIN-6668 ships
    clinical data as two TCIA file sets ("ACRIN 6668…" = 75% sample, 184
    patients; "ACRIN 6668HB…" = 25% sample, 59 patients) in adjacent
    subdirectories of the patches dataset. Both sets contain a full
    20-form CSV bundle with non-overlapping `cn` ranges; together they
    are the complete ACRIN-6668 cohort (~243 patients).

    The v4 dedupe-by-filename was a bug — it silently kept only one file
    set, dropping ~75% of the trial cohort and producing N=57 derivable
    2y-OS labels instead of the expected N≈230.

    `derive_t7_2yr_os()` and any override-path readers must concatenate
    files of the same name (e.g., both F1.csv copies) and dedupe by `cn`.
    """
    seen_paths = set()
    csvs_out = []
    for dataset_pattern in ["pet-fm-bench-t7-patches-v3", "pet-fm-bench-t7"]:
        for dataset_root in Path("/kaggle/input").rglob(dataset_pattern):
            for cdir in dataset_root.rglob("clinical"):
                for csv_path in cdir.rglob("*.csv"):
                    abs_path = csv_path.resolve()
                    if abs_path in seen_paths:
                        continue
                    seen_paths.add(abs_path)
                    csvs_out.append(csv_path)
    return csvs_out


def _read_concat_clinical(csv_files, target_name):
    """Read every csv_files entry whose .name == target_name and concatenate.

    Handles ACRIN's File Set 1 / File Set 2 split where the same form
    (e.g., F1.csv) appears in two subdirectories. The two file sets are
    **non-overlapping random partitions** of the trial cohort (each `cn`
    is in exactly one set), so concat alone is correct.

    **Do NOT dedupe by cn here.** Many ACRIN forms are intrinsically
    multi-row-per-patient (F1 = one row per q3-month follow-up visit;
    AE = one row per adverse event; LE = one row per lesion). Deduping
    by cn would silently collapse longitudinal data and corrupt
    downstream derivations. Callers are responsible for any per-patient
    aggregation (e.g., `derive_t7_2yr_os` does `groupby("cn").tail(1)`
    after sort by visit date).

    If the same `cn` ever DOES appear in both file sets — a TCIA data
    integrity issue, not expected — emit a warning so the user knows.
    """
    parts = []
    for p in csv_files:
        if p.name == target_name:
            try:
                parts.append(pd.read_csv(p, low_memory=False))
            except Exception as e:
                print(f"  ⚠ failed to read {p}: {e}")
    if not parts:
        return None
    df = pd.concat(parts, ignore_index=True)

    # Cross-file-set cn collision check (warn-only; do not silently dedupe)
    if len(parts) > 1 and "cn" in df.columns:
        per_set_cns = [set(p["cn"].dropna().unique()) for p in parts if "cn" in p.columns]
        cross_set_collisions = set()
        for i in range(len(per_set_cns)):
            for j in range(i + 1, len(per_set_cns)):
                cross_set_collisions |= per_set_cns[i] & per_set_cns[j]
        if cross_set_collisions:
            print(f"  ⚠ {target_name}: {len(cross_set_collisions)} cn values appear "
                  f"in MULTIPLE file sets — TCIA data integrity issue. "
                  f"Concat keeps all rows; downstream derivation may double-count.")

    if len(parts) > 1:
        n_unique = df["cn"].nunique() if "cn" in df.columns else len(df)
        print(f"  {target_name}: concatenated {len(parts)} file set(s) → "
              f"{len(df):,} rows, {n_unique} unique patients")
    return df


def _column_outcome_score(col, series):
    """Score a column for outcome-likeness. Returns a float; higher = more likely.

    Heuristics:
    - Name matches one of T7_OUTCOME_PATTERNS (case-insensitive substring) → +5
    - Name contains 'd' suffix (date column like 'dse2d') → -3 (these are dates)
    - 2 unique non-null values (binary outcome) → +3
    - 3-5 unique values (categorical response: CR/PR/SD/PD) → +2
    - Mostly populated (>50% non-null) → +1
    - Numeric-looking → +1
    """
    score = 0.0
    name_lc = col.lower().strip()
    for pat in T7_OUTCOME_PATTERNS:
        if pat in name_lc:
            score += 5.0
            break
    if name_lc.endswith("d") and any(p in name_lc[:-1] for p in T7_OUTCOME_PATTERNS):
        score -= 3.0  # date column
    n = series.notna().sum()
    if n < 20:
        return -10.0  # too few labels to probe
    n_unique = series.dropna().nunique()
    if n_unique == 2:
        score += 3.0
    elif 3 <= n_unique <= 5:
        score += 2.0
    elif n_unique == 1 or n_unique > 50:
        score -= 5.0  # constant or quasi-continuous
    if n / len(series) > 0.5:
        score += 1.0
    return score


def discover_outcome_column(csv_files):
    """v3: rank every (csv, column) pair by outcome-likeness score; return best.

    Prints the top-10 candidates so the user can verify the auto-pick or set
    T7_OUTCOME_OVERRIDE.
    """
    candidates = []
    csv_dfs = {}
    for csv_path in csv_files:
        try:
            df = pd.read_csv(csv_path, low_memory=False)
        except Exception as e:
            print(f"    skip {csv_path.name}: {e}")
            continue
        csv_dfs[csv_path] = df
        for col in df.columns:
            score = _column_outcome_score(col, df[col])
            if score > 0:
                candidates.append({
                    "csv": csv_path.name,
                    "column": col,
                    "score": score,
                    "n_nonnull": int(df[col].notna().sum()),
                    "n_unique": int(df[col].dropna().nunique()),
                    "values_preview": str(
                        df[col].dropna().value_counts().head(5).to_dict()
                    )[:80],
                    "_csv_path": csv_path,
                    "_df": df,
                })

    if not candidates:
        return None, None, None

    candidates.sort(key=lambda c: c["score"], reverse=True)

    print(f"\n  Top outcome-column candidates (auto-discovery):")
    print(f"  {'csv':<14} {'column':<14} {'score':<6} {'n':<5} {'uniq':<5} values")
    for c in candidates[:10]:
        print(f"    {c['csv']:<12} {c['column']:<14} {c['score']:<6.1f} "
              f"{c['n_nonnull']:<5} {c['n_unique']:<5} {c['values_preview']}")

    best = candidates[0]
    return best["_csv_path"], best["column"], best["_df"]


def derive_t7_2yr_os(csv_files):
    """v4: derive 2-year overall survival from F1.csv + DS.csv.

    Registration-aligned T7 endpoint (Machtay et al. 2013 J Clin Oncol 31:3823).
    Returns (synthetic_csv_path, "event_2yr_os", DataFrame) — same signature
    as discover_outcome_column / override path so the call site is uniform.

    Derivation logic:
      Per patient (cn):
        Take last F1 row by F1e1d (days from baseline to clinical assessment).
        - f1e2 == 2 (Dead): event=1 if F1e3d (or F1e1d) <= 730d, else event=0
        - f1e2 == 9 (Dead, DOD unknown): event=1 if last visit <= 730d
        - f1e2 == 1 (Alive): event=0 if last visit >= 730d, else CENSORED (drop)
        - f1e2 == 3 (LTFU): drop
      For patients absent from F1 but in DS with dse3 == 2 (Death) and
      DSe2d <= 730d: event=1.

    Local sanity check: produces 57.2% 2y mortality on ACRIN-6668 File Set 1,
    matching Machtay 2013's published 57.5% within 0.3pp (n=180/230 labelled,
    50 censored < 2y dropped).
    """
    # v5: concat across ACRIN File Set 1 + File Set 2 (each ships its own
    # F1.csv / DS.csv for non-overlapping patient subsets).
    f1 = _read_concat_clinical(csv_files, "F1.csv")
    ds = _read_concat_clinical(csv_files, "DS.csv")

    if f1 is None:
        print("  ✗ T7_OUTCOME_DERIVATION='2yr_os' set but F1.csv not found.")
        return None, None, None

    f1c = f1.dropna(subset=["f1e2"]).copy()
    # Use F1e1d (visit date) as canonical follow-up time; F1e3d falls back
    # for the rare case where only the final-contact row is filled.
    f1c["fu_days"] = f1c["F1e1d"].fillna(f1c.get("F1e3d"))
    f1c = f1c.dropna(subset=["fu_days"])

    last_f1 = (f1c.sort_values(["cn", "fu_days"])
                  .groupby("cn", as_index=False).tail(1)
                  .set_index("cn"))

    out = {}
    for cn, row in last_f1.iterrows():
        vs = row["f1e2"]
        last_visit = row["fu_days"]
        death_days = row["F1e3d"] if pd.notna(row.get("F1e3d")) else last_visit
        if vs == 2:
            out[cn] = 1 if death_days <= T7_OS_THRESHOLD_DAYS else 0
        elif vs == 9 and last_visit <= T7_OS_THRESHOLD_DAYS:
            out[cn] = 1
        elif vs == 1 and last_visit >= T7_OS_THRESHOLD_DAYS:
            out[cn] = 0
        # Otherwise censored < 2y, drop

    if ds is not None:
        ds_idx = ds.set_index("cn")
        for cn in set(ds["cn"]) - set(out):
            if cn not in ds_idx.index:
                continue
            r = ds_idx.loc[cn]
            if r.get("dse3") == 2 and pd.notna(r.get("DSe2d")) \
               and r["DSe2d"] <= T7_OS_THRESHOLD_DAYS:
                out[cn] = 1

    if not out:
        print("  ✗ T7 2y-OS derivation produced 0 labelled patients.")
        return None, None, None

    # ACRIN clinical CSVs use integer `cn` as the case number; embedding
    # parquets typically carry TCIA-style strings like
    # `ACRIN-NSCLC-FDG-PET-001`. Emit several ID-format variants per patient
    # — the probe's `labels_df.iloc[:, 0] == pid` join naturally filters
    # to whichever variant matches the embedding's actual format.
    records = []
    for cn, event in out.items():
        if isinstance(cn, (int, np.integer)) or (isinstance(cn, float) and cn.is_integer()):
            n = int(cn)
            for pid_form in (
                n,
                str(n),
                f"ACRIN-NSCLC-FDG-PET-{n:03d}",
                f"ACRIN-NSCLC-FDG-PET-{n:04d}",
                f"ACRIN-NSCLC-FDG-PET-{n}",
            ):
                records.append({"patient_id": pid_form, "event_2yr_os": int(event)})
        else:
            records.append({"patient_id": str(cn), "event_2yr_os": int(event)})

    df = pd.DataFrame(records)
    n_e1 = int(((df["event_2yr_os"] == 1) & ~df["patient_id"].duplicated()).sum())
    n_unique = df.drop_duplicates("event_2yr_os" if False else None)
    # Per-patient counts (any ID format counted once)
    n_patients = len(out)
    n_e1_unique = sum(1 for v in out.values() if v == 1)
    n_e0_unique = sum(1 for v in out.values() if v == 0)
    print(f"  → T7 2y-OS derived: {n_patients} patients labelled "
          f"({n_e1_unique} died ≤2y, {n_e0_unique} alive ≥2y, "
          f"{n_e1_unique/n_patients:.1%} mortality; Machtay 2013 expected ~57.5%). "
          f"{len(df)} ID-format rows emitted to match either integer or TCIA-style "
          f"embedding patient_ids.")
    return None, "event_2yr_os", df


def resolve_t7_outcome(csv_files):
    """v4: derivation path > override > auto-discovery."""
    # 1. Derivation path (registration-aligned 2-year OS)
    if T7_OUTCOME_DERIVATION == "2yr_os":
        return derive_t7_2yr_os(csv_files)

    # 2. Explicit override (v5: concats File Set 1 + 2 if both attached)
    if T7_OUTCOME_OVERRIDE is not None:
        target_name, target_col = T7_OUTCOME_OVERRIDE
        df = _read_concat_clinical(csv_files, target_name)
        if df is None:
            print(f"  ✗ T7_OUTCOME_OVERRIDE pinned {target_name} but file not found")
            return None, None, None
        if target_col not in df.columns:
            print(f"  ✗ T7_OUTCOME_OVERRIDE pinned column '{target_col}' "
                  f"but {target_name} columns are: {list(df.columns)[:20]}")
            return None, None, None
        print(f"  → T7_OUTCOME_OVERRIDE pinned: {target_name}.{target_col} "
              f"(n={df[target_col].notna().sum()})")
        # Find a representative path for chosen_csv (first match)
        first_match = next((p for p in csv_files if p.name == target_name), None)
        return first_match, target_col, df

    return discover_outcome_column(csv_files)


def run_t7_response_probe_heldout(embeddings_dict, labels_df, outcome_col):
    """v4: held-out test eval per Phase 4 v2 freeze.

    Mirrors run_classification_probe_heldout but filters to baseline scans
    (scan_id='study_0') first, then dispatches train/cal/test using
    task_splits.parquet.
    """
    results = []
    splits = get_task_splits("t7")
    if splits is None:
        return run_t7_response_probe(embeddings_dict, labels_df, outcome_col)

    train_cal_pids = set(splits.get("train", []) + splits.get("cal", []))
    test_pids = set(splits.get("test", []))
    print(f"  t7 splits: train+cal={len(train_cal_pids)}, test={len(test_pids)}")

    for fm_name, fm_df in embeddings_dict.items():
        # v6 post-bug-fix (Step 67): handle both `scan_id="study_0"` (string,
        # used by tX_02_embeddings.py) AND `study_index=0` (integer, used by
        # 08_random_init_multiseed.py for the older T4-T9 multi-seed
        # parquets). Both encode "baseline scan".
        if "scan_id" in fm_df.columns:
            baseline = fm_df[fm_df["scan_id"] == "study_0"].copy()
        elif "study_index" in fm_df.columns:
            # study_index dtype may be int (typical) or object/string (if manifest
            # serialised integers as strings). Coerce defensively — int vs str==0
            # comparison would silently produce an empty baseline.
            baseline = fm_df[
                pd.to_numeric(fm_df["study_index"], errors="coerce") == 0
            ].copy()
        else:
            print(f"  {fm_name}: no scan_id nor study_index column — skipping")
            continue

        def _collect(pid_subset):
            X_list, y_list = [], []
            for pid in baseline["patient_id"].unique():
                if str(pid) not in pid_subset:
                    continue
                emb = get_patient_embedding(baseline, pid)
                if emb is None:
                    continue
                label_row = labels_df[labels_df.iloc[:, 0] == pid]
                if len(label_row) == 0:
                    continue
                label = label_row[outcome_col].values[0]
                if pd.isna(label):
                    continue
                X_list.append(emb)
                y_list.append(label)
            return np.array(X_list) if X_list else None, np.array(y_list) if y_list else None

        X_tc, y_tc = _collect(train_cal_pids)
        X_te, y_te = _collect(test_pids)
        if X_tc is None or X_te is None or len(X_tc) < 20 or len(X_te) < 5:
            print(f"  {fm_name}: insufficient train+cal/test, skipping")
            continue

        for X in (X_tc, X_te):
            np.nan_to_num(X, copy=False)

        le = LabelEncoder()
        y_tc_enc = le.fit_transform(y_tc)
        try:
            y_te_enc = le.transform(y_te)
        except ValueError:
            print(f"  {fm_name}: test set contains classes not in train+cal — skipping")
            continue
        n_classes = len(le.classes_)

        scaler = StandardScaler()
        X_tc_s = scaler.fit_transform(X_tc)
        X_te_s = scaler.transform(X_te)

        cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
        best_c, best_score = 1.0, -1
        for c in C_GRID:
            try:
                model = LogisticRegression(C=c, penalty="l2", solver="lbfgs",
                                           max_iter=1000, random_state=42)
                preds = cross_val_predict(model, X_tc_s, y_tc_enc, cv=cv,
                                          method="predict_proba")
                if n_classes == 2:
                    s = roc_auc_score(y_tc_enc, preds[:, 1])
                else:
                    s = roc_auc_score(y_tc_enc, preds, multi_class="ovr",
                                      average="macro")
                if s > best_score:
                    best_score, best_c = s, c
            except Exception:
                continue

        model = LogisticRegression(C=best_c, penalty="l2", solver="lbfgs",
                                   max_iter=1000, random_state=42)
        model.fit(X_tc_s, y_tc_enc)
        y_te_proba = model.predict_proba(X_te_s)
        if n_classes == 2:
            auroc = roc_auc_score(y_te_enc, y_te_proba[:, 1])
        else:
            auroc = roc_auc_score(y_te_enc, y_te_proba, multi_class="ovr",
                                  average="macro")

        rng = np.random.RandomState(42)
        boot = []
        for _ in range(N_BOOTSTRAP):
            idx = rng.choice(len(y_te_enc), size=len(y_te_enc), replace=True)
            if len(np.unique(y_te_enc[idx])) < n_classes:
                continue
            try:
                if n_classes == 2:
                    boot.append(roc_auc_score(y_te_enc[idx], y_te_proba[idx, 1]))
                else:
                    boot.append(roc_auc_score(y_te_enc[idx], y_te_proba[idx],
                                              multi_class="ovr", average="macro"))
            except ValueError:
                continue
        ci_low = float(np.percentile(boot, 2.5)) if boot else float("nan")
        ci_high = float(np.percentile(boot, 97.5)) if boot else float("nan")

        results.append({
            "task": "t7", "fm": fm_name, "metric": "auroc_response",
            "value": float(auroc), "ci_low": ci_low, "ci_high": ci_high,
            "best_c": best_c,
            "n_train_cal": int(len(y_tc_enc)),
            "n_test": int(len(y_te_enc)),
            "n_classes": int(n_classes),
            "outcome_col": outcome_col,
            "classes": str(list(le.classes_)),
            "eval_mode": "heldout",
        })
        print(f"  {fm_name}: AUROC={auroc:.3f} [{ci_low:.3f}, {ci_high:.3f}] "
              f"(C={best_c}, n_train_cal={len(y_tc_enc)}, n_test={len(y_te_enc)})")

    return pd.DataFrame(results)


def run_t7_response_probe(embeddings_dict, labels_df, outcome_col):
    """Classification probe: baseline embeddings → response label."""
    results = []
    for fm_name, fm_df in embeddings_dict.items():
        # v6 post-bug-fix (Step 67): handle both `scan_id="study_0"` (string,
        # used by tX_02_embeddings.py) AND `study_index=0` (integer, used by
        # 08_random_init_multiseed.py for the older T4-T9 multi-seed parquets).
        if "scan_id" in fm_df.columns:
            baseline = fm_df[fm_df["scan_id"] == "study_0"].copy()
        elif "study_index" in fm_df.columns:
            # study_index dtype may be int (typical) or object/string (if manifest
            # serialised integers as strings). Coerce defensively — int vs str==0
            # comparison would silently produce an empty baseline.
            baseline = fm_df[
                pd.to_numeric(fm_df["study_index"], errors="coerce") == 0
            ].copy()
        else:
            print(f"  {fm_name}: no scan_id nor study_index column — skipping")
            continue

        X_list, y_list, pid_list = [], [], []
        for pid in baseline["patient_id"].unique():
            emb = get_patient_embedding(baseline, pid)
            if emb is None:
                continue
            label_row = labels_df[labels_df.iloc[:, 0] == pid]
            if len(label_row) == 0:
                continue
            label = label_row[outcome_col].values[0]
            if pd.isna(label):
                continue
            X_list.append(emb)
            y_list.append(label)
            pid_list.append(pid)

        if len(X_list) < 20:
            print(f"  {fm_name}: too few labelled samples ({len(X_list)}) — skipping")
            continue

        X = np.array(X_list)
        y = np.array(y_list)

        nan_mask = np.isnan(X).any(axis=1)
        X = X[~nan_mask]
        y = y[~nan_mask]
        np.nan_to_num(X, copy=False)

        if len(X) < 20:
            continue

        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        n_classes = len(le.classes_)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
        # Single C for smoke-test; expand to grid for formal run
        best_c, best_score = 1.0, -1
        for c in C_GRID:
            model = LogisticRegression(C=c, penalty="l2", solver="lbfgs",
                                       max_iter=1000, random_state=42)
            try:
                method = "predict_proba" if n_classes > 1 else "predict"
                preds = cross_val_predict(model, X_scaled, y_encoded, cv=cv,
                                          method="predict_proba")
                if n_classes == 2:
                    score = roc_auc_score(y_encoded, preds[:, 1])
                else:
                    score = roc_auc_score(y_encoded, preds, multi_class="ovr",
                                          average="macro")
                if score > best_score:
                    best_score, best_c = score, c
            except Exception:
                continue

        # Final OOF predictions
        model = LogisticRegression(C=best_c, penalty="l2", solver="lbfgs",
                                   max_iter=1000, random_state=42)
        y_pred_proba = cross_val_predict(model, X_scaled, y_encoded, cv=cv,
                                         method="predict_proba")
        if n_classes == 2:
            auroc = roc_auc_score(y_encoded, y_pred_proba[:, 1])
        else:
            auroc = roc_auc_score(y_encoded, y_pred_proba, multi_class="ovr",
                                  average="macro")

        # Bootstrap CI
        rng = np.random.RandomState(42)
        boot = []
        for _ in range(N_BOOTSTRAP):
            idx = rng.choice(len(y_encoded), size=len(y_encoded), replace=True)
            if len(np.unique(y_encoded[idx])) < n_classes:
                continue
            try:
                if n_classes == 2:
                    boot.append(roc_auc_score(y_encoded[idx], y_pred_proba[idx, 1]))
                else:
                    boot.append(roc_auc_score(y_encoded[idx], y_pred_proba[idx],
                                              multi_class="ovr", average="macro"))
            except ValueError:
                continue
        ci_low = np.percentile(boot, 2.5) if boot else np.nan
        ci_high = np.percentile(boot, 97.5) if boot else np.nan

        results.append({
            "task": "t7", "fm": fm_name,
            "metric": "auroc_response",
            "value": float(auroc), "ci_low": float(ci_low), "ci_high": float(ci_high),
            "best_c": best_c, "n_patients": len(X), "n_classes": n_classes,
            "outcome_col": outcome_col,
            "classes": str(list(le.classes_)),
        })
        print(f"  {fm_name}: AUROC={auroc:.3f} [{ci_low:.3f}, {ci_high:.3f}] "
              f"(n={len(X)}, n_classes={n_classes})")

    return pd.DataFrame(results)


# Run T7 probe if embeddings + clinical data both available
if "t7" in EMBEDDINGS:
    print("\n" + "=" * 60)
    print("T7: Response Prediction (baseline PET → outcome)")
    print("=" * 60)

    csvs = find_t7_clinical_csvs()
    if not csvs:
        print("  ✗ No ACRIN clinical CSVs found in attached datasets.")
        print("    Attach pet-fm-bench-t7-patches-v3 (which contains clinical/) "
              "to enable T7 probe.")
        t7_results = pd.DataFrame()
    else:
        print(f"  Found {len(csvs)} clinical CSV(s):")
        for c in csvs:
            print(f"    {c.name}")
        chosen_csv, outcome_col, clinical_df = resolve_t7_outcome(csvs)
        if outcome_col is None:
            print(f"  ✗ No outcome column resolved (override unset and "
                  f"auto-discovery found no candidates with score > 0). "
                  f"Set T7_OUTCOME_OVERRIDE = ('<file.csv>', '<column>') "
                  f"after inspecting the ACRIN-6668 data dictionary.")
            t7_results = pd.DataFrame()
        else:
            src_label = chosen_csv.name if chosen_csv is not None else "(derived)"
            print(f"  Using outcome column: {outcome_col} from {src_label}")
            print(f"  Distribution: {clinical_df[outcome_col].value_counts().to_dict()}")
            # Standardize the patient-ID column to first column
            id_cols = ["PatientID", "Case ID", "Patient ID", "case_id", "patient_id"]
            for ic in id_cols:
                if ic in clinical_df.columns:
                    clinical_df = clinical_df.rename(columns={ic: "patient_id"})
                    break
            # Move patient_id to first column for the probe's iloc[:, 0] lookup
            if "patient_id" in clinical_df.columns:
                cols = ["patient_id"] + [c for c in clinical_df.columns if c != "patient_id"]
                clinical_df = clinical_df[cols]
            mode = task_eval_mode("t7")
            print(f"  eval mode: {mode}")
            if mode == "heldout":
                t7_results = run_t7_response_probe_heldout(
                    EMBEDDINGS["t7"], clinical_df, outcome_col
                )
            else:
                t7_results = run_t7_response_probe(
                    EMBEDDINGS["t7"], clinical_df, outcome_col
                )
            t7_results = aggregate_random_init_seeds(t7_results)
            if len(t7_results) > 0:
                t7_results.to_csv(OUTPUT_DIR / "t7_response_results.csv", index=False)
else:
    t7_results = pd.DataFrame()


# %% [markdown]
# ## 6. Summary: Cross-FM Performance Table

# %%
print("\n" + "=" * 60)
print("CROSS-FM PERFORMANCE SUMMARY")
print("=" * 60)

all_results = []
for name, df in [("t1_lesion_patch", t1_results if 't1_results' in dir() else pd.DataFrame()),
                  ("t2_lesion_patch", t2_results if 't2_results' in dir() else pd.DataFrame()),
                  ("t2_gtvp_only", t2_gtvp_results if 't2_gtvp_results' in dir() else pd.DataFrame()),
                  ("t3_survival", t3_results if 't3_results' in dir() else pd.DataFrame()),
                  ("t5_zero_shot", t5_results if 't5_results' in dir() else pd.DataFrame()),
                  ("t8_classification", t8_results),
                  ("t4_survival", t4_results),
                  ("t7_response", t7_results if 't7_results' in dir() else pd.DataFrame()),
                  ("t9_test_retest", t9_results if 't9_results' in dir() else pd.DataFrame()),
                  ("t6_test_retest", t6_results if 't6_results' in dir() else pd.DataFrame())]:
    if len(df) > 0:
        all_results.append(df)

if all_results:
    combined = pd.concat(all_results, ignore_index=True)

    # v6: merge Phase 2 contamination tiers into the per-row results so the
    # manuscript table can show "(FM × task) probe metric AND contamination
    # tier" side-by-side. Prefer the Phase 2 freeze artefact
    # `contamination_audit.csv` (registration-grade) over the Stage 2
    # intermediate `contamination_summary.csv`. Both have the same join columns
    # (fm, task, tier, overlap_fraction, n_contaminated, n_clean); audit is a
    # strict superset (also includes Stage 3 within-patient stats).
    _audit_paths = list(Path("/kaggle/input").rglob("contamination_audit.csv"))
    # Prefer files inside a freeze-v2 dataset path if multiple are attached
    _contam_path = None
    for p in _audit_paths:
        if "contamination-freeze-v2" in str(p):
            _contam_path = p
            break
    if _contam_path is None and _audit_paths:
        _contam_path = _audit_paths[0]
    # Fallback: legacy Stage 2 intermediate (pre-v2 freeze runs)
    if _contam_path is None:
        _contam_path = next(
            Path("/kaggle/input").rglob("contamination_summary.csv"), None
        )
    if _contam_path is not None:
        try:
            _contam = pd.read_csv(_contam_path)
            _join_cols = ["fm", "task"]
            _keep_cols = ["tier", "overlap_fraction", "n_contaminated", "n_clean"]
            _merge = _contam[_join_cols + _keep_cols].rename(
                columns={
                    "tier": "contamination_tier",
                    "overlap_fraction": "contamination_overlap_fraction",
                    "n_contaminated": "contamination_n_dirty",
                    "n_clean": "contamination_n_clean",
                }
            )
            combined = combined.merge(_merge, on=_join_cols, how="left")
            print(f"  → Phase 2 contamination tiers merged from {_contam_path.name}: "
                  f"{_contam['fm'].nunique()} FMs × "
                  f"{_contam['task'].nunique()} tasks")
        except Exception as e:
            print(f"  ⚠ failed to merge contamination tiers: {e}")
    else:
        print("  ⚠ contamination_audit.csv / contamination_summary.csv not "
              "attached — no tier column in all_probe_results.csv. Attach "
              "pet-fm-bench-contamination-freeze-v2 for the formal run.")

    combined.to_csv(OUTPUT_DIR / "all_probe_results.csv", index=False)

    # Pivot: FM × Task
    pivot = combined.pivot_table(
        index="fm", columns="task", values="value", aggfunc="first"
    )

    # v3: explain what each column's `value` means, since metrics differ
    # across tasks AND test-retest now reports the discriminative gap
    # instead of raw within-patient cosine.
    metric_legend = (
        combined[["task", "metric"]]
        .drop_duplicates()
        .set_index("task")["metric"]
        .to_dict()
    )
    print("\nMetric per task:")
    for task, metric in sorted(metric_legend.items()):
        label = {
            "auroc_macro":     "macro AUROC (higher better)",
            "auroc_response":  "binary AUROC (higher better)",
            "c_index":         "concordance index (higher better)",
            "cosine_gap":      "Sw − Sb (positive = patient-discriminative)",
            "cosine_mean":     "raw within-patient cosine (legacy)",
        }.get(metric, metric)
        print(f"  {task}: {label}")

    print("\n" + pivot.round(3).to_string())

    # Save pivot
    pivot.to_csv(OUTPUT_DIR / "fm_task_matrix.csv")
else:
    print("No results to summarize")

# %% [markdown]
# ## 7. Output files

# %%
print("\nOutput files:")
total_kb = 0
for f in sorted(OUTPUT_DIR.rglob("*")):
    if f.is_file():
        kb = f.stat().st_size / 1e3
        total_kb += kb
        print(f"  {f.name} ({kb:.0f} KB)")
print(f"\nTotal: {total_kb:.0f} KB")

# %% [markdown]
# ## 8. Done
#
# Commit with **"Save & Run All"** on CPU.
# Output → **"New Dataset"** → `pet-fm-bench-probe-results`
