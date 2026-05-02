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
# # PET-FM-Bench: Calibration Inference (Amendment A13) — v2 corrected
#
# **Pre-registration:** OSF [10.17605/OSF.IO/DQ2JA](https://doi.org/10.17605/OSF.IO/DQ2JA),
# Amendment A13 (Phase 5 calibration sub-study).
#
# **Purpose:** Re-run the binary-classification (T1, T5, T7) and Cox-survival
# (T3, T4) probes from the registered Phase 5 pipeline at SEED = 42 and
# **dump per-patch / per-patient predicted probabilities** alongside the existing
# summary metrics.
#
# **v2 changes vs v1:**
# - **T3** column names corrected to match registered probe_analysis.py v6
#   (`relapse`/`rfs_days`, filter on `task2_patient & label==1`, lesion-pooled).
# - **T5** zero-shot eval cohort fixed: union over ALL T5 split labels (the
#   registered split label is hard-coded `test_retest` in `06_task_splits.py`,
#   not `test_zero_shot`; previous v1 mis-filtered to 0 rows).
# - **T7** outcome inlined: derives `event_2yr_os` from ACRIN-6668 clinical
#   F1.csv + DS.csv (per registered `derive_t7_2yr_os`, Machtay 2013 endpoint).
#   Requires `pet-fm-bench-t7-patches-v3` to be attached for the clinical/ subdir.
# - **T4** horizon changed from 36 mo to **24 mo** — at 36 mo the NSCLC cohort
#   has 45 censored-before vs 48 events, producing degenerate IPCW deciles;
#   at 24 mo there are 36 events vs 23 censored-before, giving cleaner deciles.
# - Legacy single-seed `random_init` row dropped from per-prediction outputs
#   — only the 10-seed multiseed entries are kept (matches A3 baseline).
#
# **Determinism:** This notebook re-uses the *exact* probe-fit code paths from
# `probe_analysis.py` v6 (run-time SHA-256 `7ca32e8b…`) at SEED = 42. Aggregated
# metrics produced here will be bit-identical to the registered freeze CSVs;
# the only new outputs are the per-prediction parquets.
#
# **Runtime:** CPU only, no GPU. Estimated ~1.5–2 h on Kaggle T4 free tier
# (T1 dominates at ~50 min for 16 FMs; T5 zero-shot ~50 min; T3+T4+T7 ~30 min).
#
# **Input datasets to attach:**
# - `pet-fm-bench-t1-embeddings-v3`
# - `pet-fm-bench-t5-embeddings-v3`
# - `pet-fm-bench-t7-embeddings-v3`
# - `pet-fm-bench-t7-patches-v3`   ← **NEW for v2**: contains ACRIN clinical CSVs (16 GB)
# - `pet-fm-bench-hecktor-2025-embeddings-v3` (T3)
# - `pet-fm-bench-t4-embeddings-v3`
# - `pet-fm-bench-t1-randominit-multiseed-v3`
# - `pet-fm-bench-t5-randominit-multiseed-v3`
# - `pet-fm-bench-t7-randominit-multiseed-v3`
# - `pet-fm-bench-hecktor-randominit-multiseed-v3`
# - `pet-fm-bench-t4-randominit-multiseed-v3`
# - `pet-fm-bench-task-splits-v4`
#
# **Outputs (in `/kaggle/working/`):**
# - `perpred_t1.parquet` — T1 binary per-patch
# - `perpred_t5.parquet` — T5 zero-shot per-patch
# - `perpred_t7.parquet` — T7 binary per-patient (2yr OS)
# - `perpred_t3.parquet` — T3 Cox per-patient with surv_24mo
# - `perpred_t4.parquet` — T4 Cox per-patient with surv_24mo
# - `perpred_manifest.csv` — file → SHA-256 manifest

# %%
import os
import re
import sys
import hashlib
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning, module="sksurv")

try:
    from sksurv.linear_model import CoxPHSurvivalAnalysis
    from sksurv.metrics import concordance_index_censored
except ImportError:
    print("Installing scikit-survival ...")
    os.system("pip install -q scikit-survival")
    from sksurv.linear_model import CoxPHSurvivalAnalysis
    from sksurv.metrics import concordance_index_censored

SEED = 42
N_FOLDS = 5
C_GRID = [0.001, 0.01, 0.1, 1, 10, 100]
ALPHA_GRID = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
T7_OS_THRESHOLD_DAYS = 730   # 2 years (Machtay 2013)
COX_HORIZON_MONTHS = 24       # v2: 24 mo for both T3 and T4

OUTPUT_DIR = Path("/kaggle/working")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## 1. Helpers (verbatim from registered probe_analysis.py v6)

# %%
RANDOM_INIT_SEED_PATTERN = re.compile(
    r"^random_init[_-]?(?:seed)?[_-]?(\d+)$", re.IGNORECASE
)


def find_embed_dir(task_pattern):
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
    return None


def load_fm_embeddings(embed_dir):
    if embed_dir is None:
        return {}
    fm_data = {}
    for f in sorted(embed_dir.glob("*.parquet")):
        fm_name = f.stem
        if fm_name.endswith("_labels"):
            continue
        df = pd.read_parquet(f)
        fm_data[fm_name] = df
    return fm_data


def load_labels(embed_dir, label_pattern="*labels*"):
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


def find_multiseed_dir(task):
    for slug in [
        f"pet-fm-bench-{task}-randominit-multiseed-v3",
        f"pet-fm-bench-{task}-randominit-multiseed",
    ]:
        candidates = list(Path("/kaggle/input").rglob(slug))
        if candidates:
            return candidates[0]
    return None


def load_multiseed_random_init(multiseed_dir):
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
        fm_name = f.stem
        if not RANDOM_INIT_SEED_PATTERN.match(fm_name):
            continue
        df = pd.read_parquet(f)
        seed_data[fm_name] = df
    return seed_data


def find_task_splits():
    candidates = list(Path("/kaggle/input").rglob("task_splits.parquet"))
    if not candidates:
        return None
    for version_suffix in ("-v4", "-v3", "-v2"):
        for c in candidates:
            if f"pet-fm-bench-task-splits{version_suffix}" in str(c):
                return c
    for c in candidates:
        if "pet-fm-bench-task-splits" in str(c):
            return c
    return candidates[0]


def get_task_splits(task, splits_df):
    if splits_df is None:
        return None
    sub = splits_df[splits_df["task"] == task]
    if sub.empty:
        return None
    return {
        sp: sub.loc[sub["split"] == sp, "patient_id"].astype(str).tolist()
        for sp in sub["split"].unique()
    }


def get_patient_embedding(df, patient_id, view="coronal", layer="cls"):
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


def find_t7_clinical_csvs():
    """Verbatim from registered probe_analysis.py — finds ACRIN clinical CSVs."""
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
    parts = []
    for p in csv_files:
        if p.name == target_name:
            try:
                parts.append(pd.read_csv(p, low_memory=False))
            except Exception as e:
                print(f"  ⚠ failed to read {p}: {e}")
    if not parts:
        return None
    return pd.concat(parts, ignore_index=True)


def derive_t7_2yr_os(csv_files):
    """Inline derivation from registered probe_analysis.py v6.

    Returns a DataFrame with columns (patient_id, event_2yr_os) where event_2yr_os = 1
    if dead within 730 days, 0 if alive at 730 days, missing for censored before.
    Emits multiple ID-format variants per patient to match either integer or
    TCIA-style embedding patient_ids.
    """
    f1 = _read_concat_clinical(csv_files, "F1.csv")
    ds = _read_concat_clinical(csv_files, "DS.csv")
    if f1 is None:
        print("  ✗ T7 derivation: F1.csv not found in clinical CSVs.")
        return None

    f1c = f1.dropna(subset=["f1e2"]).copy()
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
        return None

    records = []
    for cn, event in out.items():
        if isinstance(cn, (int, np.integer)) or (isinstance(cn, float) and cn.is_integer()):
            n = int(cn)
            for pid_form in (
                str(n),
                f"ACRIN-NSCLC-FDG-PET-{n:03d}",
                f"ACRIN-NSCLC-FDG-PET-{n:04d}",
                f"ACRIN-NSCLC-FDG-PET-{n}",
            ):
                records.append({"patient_id": pid_form, "event_2yr_os": int(event)})
        else:
            records.append({"patient_id": str(cn), "event_2yr_os": int(event)})
    df = pd.DataFrame(records)
    n_total = len(out); n_e1 = sum(1 for v in out.values() if v == 1)
    print(f"  → T7 2y-OS derived: {n_total} patients labelled "
          f"({n_e1} died ≤2y, {n_total - n_e1} alive ≥2y, "
          f"{n_e1/n_total:.1%} mortality; Machtay 2013 expected ~57.5%)")
    return df


# %% [markdown]
# ## 2. Load embeddings, labels, and task splits

# %%
TASK_DIRS = {}
for task in ["t1", "t4", "t5", "t7", "hecktor"]:
    d = find_embed_dir(task)
    if d:
        print(f"\n{task.upper()}: {d}")
        TASK_DIRS[task] = d

EMBEDDINGS = {}
for task, d in TASK_DIRS.items():
    EMBEDDINGS[task] = load_fm_embeddings(d)
    multiseed_dir = find_multiseed_dir(task)
    if multiseed_dir is not None:
        seed_data = load_multiseed_random_init(multiseed_dir)
        EMBEDDINGS[task].update(seed_data)
    # v2: drop legacy single-seed `random_init` if multiseed entries exist
    seed_keys = [k for k in EMBEDDINGS[task] if RANDOM_INIT_SEED_PATTERN.match(k)]
    if "random_init" in EMBEDDINGS[task] and len(seed_keys) >= 2:
        del EMBEDDINGS[task]["random_init"]
    fms = sorted(EMBEDDINGS[task].keys())
    print(f"  {task} FMs ({len(fms)}): {fms}")

LABELS = {}
for task, d in TASK_DIRS.items():
    lbl = load_labels(d)
    if lbl is not None:
        LABELS[task] = lbl
        print(f"\n{task.upper()} labels: {list(lbl.columns)}")

SPLITS_PATH = find_task_splits()
SPLITS_DF = pd.read_parquet(SPLITS_PATH) if SPLITS_PATH is not None else None

# %% [markdown]
# ## 3. T1 — AutoPET-I FDG lesion-patch classification (binary, per-patch dump)

# %%
def run_t1_with_dump(embeddings_dict, labels_df):
    splits = get_task_splits("t1", SPLITS_DF)
    if splits is None:
        return pd.DataFrame()
    train_cal_pids = set(splits.get("train", []) + splits.get("cal", []))
    test_pids = set(splits.get("test", []))
    out_rows = []
    for fm_name, fm_df in embeddings_dict.items():
        if {"volume", "pool"}.issubset(set(fm_df["view"].unique()) | set(fm_df["layer"].unique())):
            view, layer = "volume", "pool"
        elif "axial" in fm_df["view"].values:
            view, layer = "axial", "cls"
        else:
            view = sorted(fm_df["view"].unique())[0]
            layer = sorted(fm_df["layer"].unique())[0]
        sub = fm_df[(fm_df["view"] == view) & (fm_df["layer"] == layer)]
        if len(sub) == 0:
            continue
        merged = sub.merge(
            labels_df[["patient_id", "patch_id", "label"]],
            on=["patient_id", "patch_id"], how="inner")
        if len(merged) == 0:
            continue
        merged["_split"] = merged["patient_id"].astype(str).map(
            lambda p: "tc" if p in train_cal_pids else ("te" if p in test_pids else "drop"))
        merged = merged[merged["_split"] != "drop"].reset_index(drop=True)
        tc = merged[merged["_split"] == "tc"]
        te = merged[merged["_split"] == "te"]
        if len(tc) < 50 or len(te) < 20:
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
        n_inner = min(5, len(np.unique(groups_tc)))
        if n_inner < 2:
            continue
        cv = GroupKFold(n_splits=n_inner)
        best_c, best_score = 1.0, -1
        for c in C_GRID:
            try:
                preds = np.zeros_like(y_tc, dtype=float)
                for tr_idx, va_idx in cv.split(X_tc_s, y_tc, groups=groups_tc):
                    m = LogisticRegression(C=c, penalty="l2", solver="lbfgs",
                                           max_iter=1000, random_state=SEED)
                    m.fit(X_tc_s[tr_idx], y_tc[tr_idx])
                    preds[va_idx] = m.predict_proba(X_tc_s[va_idx])[:, 1]
                s = roc_auc_score(y_tc, preds)
                if s > best_score:
                    best_score, best_c = s, c
            except Exception:
                continue
        model = LogisticRegression(C=best_c, penalty="l2", solver="lbfgs",
                                   max_iter=1000, random_state=SEED)
        model.fit(X_tc_s, y_tc)
        y_te_proba = model.predict_proba(X_te_s)[:, 1]
        for pid, patch, yt, yp in zip(te["patient_id"].values,
                                      te["patch_id"].values, y_te, y_te_proba):
            out_rows.append({
                "fm": fm_name, "task": "t1",
                "patient_id": str(pid), "patch_id": str(patch),
                "y_true": int(yt), "y_proba": float(yp),
                "view": view, "layer": layer, "best_C": float(best_c),
            })
        print(f"  T1 {fm_name}: {len(te)} per-patch predictions dumped")
    return pd.DataFrame(out_rows)


if "t1" in EMBEDDINGS and "t1" in LABELS:
    print("\n" + "=" * 60); print("T1 calibration dump"); print("=" * 60)
    t1_dump = run_t1_with_dump(EMBEDDINGS["t1"], LABELS["t1"])
    t1_dump.to_parquet(OUTPUT_DIR / "perpred_t1.parquet", index=False)
else:
    t1_dump = pd.DataFrame()

# %% [markdown]
# ## 4. T5 — PSMA zero-shot (v2 fix: union over ALL T5 splits)

# %%
def run_t5_zero_shot_with_dump(t1_emb, t1_labels, t5_emb, t5_labels):
    """v2 fix: union over ALL T5 split labels (registered split label is
    'test_retest' hard-coded; v1 mis-filtered to specific labels and produced 0 rows)."""
    t1_splits = get_task_splits("t1", SPLITS_DF)
    t5_splits = get_task_splits("t5", SPLITS_DF)
    if t1_splits is None or t5_splits is None:
        print("  T5: t1 or t5 task_splits missing")
        return pd.DataFrame()
    t1_train_pids = set(t1_splits.get("train", []) + t1_splits.get("cal", []))
    # v2 fix: union over ALL T5 split labels
    t5_eval_pids = set()
    for split_label, pids in t5_splits.items():
        t5_eval_pids.update(pids)
    print(f"  T1 train+cal: {len(t1_train_pids)} patients (probe fitting)")
    print(f"  T5 zero-shot: {len(t5_eval_pids)} patients (held-out eval)")

    out_rows = []
    common_fms = set(t1_emb.keys()) & set(t5_emb.keys())
    for fm_name in sorted(common_fms):
        t1_df, t5_df = t1_emb[fm_name], t5_emb[fm_name]
        if "volume" in t1_df["view"].values:
            view, layer = "volume", "pool"
        elif "axial" in t1_df["view"].values:
            view, layer = "axial", "cls"
        else:
            view = sorted(t1_df["view"].unique())[0]
            layer = sorted(t1_df["layer"].unique())[0]
        t1_sub = t1_df[(t1_df["view"] == view) & (t1_df["layer"] == layer)]
        t5_sub = t5_df[(t5_df["view"] == view) & (t5_df["layer"] == layer)]
        m_t1 = t1_sub.merge(
            t1_labels[["patient_id", "patch_id", "label"]],
            on=["patient_id", "patch_id"], how="inner")
        m_t1 = m_t1[m_t1["patient_id"].astype(str).isin(t1_train_pids)]
        m_t5 = t5_sub.merge(
            t5_labels[["patient_id", "patch_id", "label"]],
            on=["patient_id", "patch_id"], how="inner")
        m_t5 = m_t5[m_t5["patient_id"].astype(str).isin(t5_eval_pids)]
        if len(m_t1) < 50 or len(m_t5) < 20:
            print(f"  T5 {fm_name}: insufficient train ({len(m_t1)}) or eval ({len(m_t5)}), skip")
            continue
        dim_cols = [c for c in t1_df.columns if c.startswith("d")]
        X_tr = m_t1[dim_cols].values.astype(np.float32)
        y_tr = m_t1["label"].values.astype(int)
        np.nan_to_num(X_tr, copy=False)
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        # Match registered T5: fixed C=1.0 for zero-shot (registration §5.4 default)
        model = LogisticRegression(C=1.0, penalty="l2", solver="lbfgs",
                                   max_iter=1000, random_state=SEED)
        model.fit(X_tr_s, y_tr)
        X_te = m_t5[dim_cols].values.astype(np.float32)
        y_te = m_t5["label"].values.astype(int)
        np.nan_to_num(X_te, copy=False)
        X_te_s = scaler.transform(X_te)
        y_te_proba = model.predict_proba(X_te_s)[:, 1]
        for pid, patch, yt, yp in zip(m_t5["patient_id"].values,
                                      m_t5["patch_id"].values, y_te, y_te_proba):
            out_rows.append({
                "fm": fm_name, "task": "t5",
                "patient_id": str(pid), "patch_id": str(patch),
                "y_true": int(yt), "y_proba": float(yp),
                "view": view, "layer": layer, "best_C": 1.0,
            })
        print(f"  T5 {fm_name}: {len(m_t5)} per-patch predictions dumped")
    return pd.DataFrame(out_rows)


if "t1" in EMBEDDINGS and "t5" in EMBEDDINGS and "t1" in LABELS and "t5" in LABELS:
    print("\n" + "=" * 60); print("T5 calibration dump"); print("=" * 60)
    t5_dump = run_t5_zero_shot_with_dump(EMBEDDINGS["t1"], LABELS["t1"],
                                          EMBEDDINGS["t5"], LABELS["t5"])
    t5_dump.to_parquet(OUTPUT_DIR / "perpred_t5.parquet", index=False)
else:
    t5_dump = pd.DataFrame()

# %% [markdown]
# ## 5. T7 — AutoPET-III response (v2 fix: inline 2yr-OS derivation from clinical CSVs)

# %%
def run_t7_with_dump(embeddings_dict):
    """v2 fix: derives event_2yr_os from F1.csv + DS.csv per registered
    derive_t7_2yr_os; requires pet-fm-bench-t7-patches-v3 attached for clinical/."""
    csv_files = find_t7_clinical_csvs()
    if not csv_files:
        print("  T7: no clinical CSVs found. Attach pet-fm-bench-t7-patches-v3.")
        return pd.DataFrame()
    print(f"  T7 clinical: {len(csv_files)} CSVs found")
    outcome_df = derive_t7_2yr_os(csv_files)
    if outcome_df is None:
        return pd.DataFrame()

    splits = get_task_splits("t7", SPLITS_DF)
    if splits is None:
        return pd.DataFrame()
    train_cal_pids = set(splits.get("train", []) + splits.get("cal", []))
    test_pids = set(splits.get("test", []))

    out_rows = []
    for fm_name, fm_df in embeddings_dict.items():
        if {"volume", "pool"}.issubset(set(fm_df["view"].unique()) | set(fm_df["layer"].unique())):
            view, layer = "volume", "pool"
        elif "axial" in fm_df["view"].values:
            view, layer = "axial", "cls"
        else:
            view = sorted(fm_df["view"].unique())[0]
            layer = sorted(fm_df["layer"].unique())[0]
        sub = fm_df[(fm_df["view"] == view) & (fm_df["layer"] == layer)]
        # T7 uses one embedding per patient (baseline) — patient-level, not patch-level
        # Drop dups, take first per patient
        if "patch_id" in sub.columns:
            sub = sub.drop_duplicates("patient_id")
        merged = sub.merge(outcome_df, on="patient_id", how="inner")
        if len(merged) == 0:
            continue
        merged["_split"] = merged["patient_id"].astype(str).map(
            lambda p: "tc" if p in train_cal_pids else ("te" if p in test_pids else "drop"))
        merged = merged[merged["_split"] != "drop"].reset_index(drop=True)
        tc = merged[merged["_split"] == "tc"]
        te = merged[merged["_split"] == "te"]
        if len(tc) < 30 or len(te) < 10:
            print(f"  T7 {fm_name}: tc={len(tc)} te={len(te)}, skip")
            continue
        dim_cols = [c for c in fm_df.columns if c.startswith("d")]
        X_tc = tc[dim_cols].values.astype(np.float32)
        y_tc = tc["event_2yr_os"].values.astype(int)
        groups_tc = tc["patient_id"].values
        X_te = te[dim_cols].values.astype(np.float32)
        y_te = te["event_2yr_os"].values.astype(int)
        np.nan_to_num(X_tc, copy=False)
        np.nan_to_num(X_te, copy=False)
        scaler = StandardScaler()
        X_tc_s = scaler.fit_transform(X_tc)
        X_te_s = scaler.transform(X_te)
        n_inner = min(5, len(np.unique(groups_tc)))
        if n_inner < 2:
            continue
        cv = GroupKFold(n_splits=n_inner)
        best_c, best_score = 1.0, -1
        for c in C_GRID:
            try:
                preds = np.zeros_like(y_tc, dtype=float)
                for tr_idx, va_idx in cv.split(X_tc_s, y_tc, groups=groups_tc):
                    m = LogisticRegression(C=c, penalty="l2", solver="lbfgs",
                                           max_iter=1000, random_state=SEED)
                    m.fit(X_tc_s[tr_idx], y_tc[tr_idx])
                    preds[va_idx] = m.predict_proba(X_tc_s[va_idx])[:, 1]
                if len(np.unique(y_tc)) > 1:
                    s = roc_auc_score(y_tc, preds)
                    if s > best_score:
                        best_score, best_c = s, c
            except Exception:
                continue
        model = LogisticRegression(C=best_c, penalty="l2", solver="lbfgs",
                                   max_iter=1000, random_state=SEED)
        model.fit(X_tc_s, y_tc)
        y_te_proba = model.predict_proba(X_te_s)[:, 1]
        for pid, yt, yp in zip(te["patient_id"].values, y_te, y_te_proba):
            out_rows.append({
                "fm": fm_name, "task": "t7",
                "patient_id": str(pid),
                "y_true": int(yt), "y_proba": float(yp),
                "view": view, "layer": layer, "best_C": float(best_c),
            })
        print(f"  T7 {fm_name}: {len(te)} per-patient predictions dumped")
    return pd.DataFrame(out_rows)


if "t7" in EMBEDDINGS:
    print("\n" + "=" * 60); print("T7 calibration dump"); print("=" * 60)
    t7_dump = run_t7_with_dump(EMBEDDINGS["t7"])
    t7_dump.to_parquet(OUTPUT_DIR / "perpred_t7.parquet", index=False)
else:
    t7_dump = pd.DataFrame()

# %% [markdown]
# ## 6. Cox calibration dispatches (T3 + T4) — v2 fix: 24 mo horizon, T3 cohort filter

# %%
def run_cox_with_dump(embeddings_dict, labels_df, task_name, horizon_months,
                       time_col, event_col, cohort_filter=None,
                       lesion_pool=False):
    """Generic Cox dump with PCA at SEED=42.

    cohort_filter: function(labels_df) → labels_df subset (e.g., T3's task2_patient + label==1)
    lesion_pool: if True, mean-pool embeddings across the patient's lesion patches
    """
    if cohort_filter is not None:
        cohort = cohort_filter(labels_df)
    else:
        cohort = labels_df.copy()

    out_rows = []
    horizon_days = horizon_months * 30.44

    # Per-patient survival metadata (deduplicate)
    if event_col in cohort.columns and time_col in cohort.columns:
        patient_meta = (cohort.drop_duplicates("patient_id")
                              [["patient_id", event_col, time_col]]
                              .reset_index(drop=True))
        patient_meta = patient_meta.dropna(subset=[event_col, time_col])
        patient_meta = patient_meta[patient_meta[time_col] > 0]
    else:
        print(f"  {task_name}: missing {event_col} or {time_col} in cohort")
        return pd.DataFrame()

    print(f"  {task_name}: {len(patient_meta)} patients, "
          f"{int(patient_meta[event_col].sum())} events ({patient_meta[event_col].mean()*100:.1f}%)")

    for fm_name, fm_df in embeddings_dict.items():
        if {"volume", "pool"}.issubset(set(fm_df["view"].unique()) | set(fm_df["layer"].unique())):
            view, layer = "volume", "pool"
        elif "axial" in fm_df["view"].values:
            view, layer = "axial", "cls"
        else:
            view = sorted(fm_df["view"].unique())[0]
            layer = sorted(fm_df["layer"].unique())[0]
        sub = fm_df[(fm_df["view"] == view) & (fm_df["layer"] == layer)]
        if len(sub) == 0:
            continue

        if lesion_pool:
            # Restrict to lesion patches in the cohort, then mean-pool per patient
            keep = cohort[["patient_id", "patch_id"]].copy()
            keep["patient_id"] = keep["patient_id"].astype(str)
            keep["patch_id"] = keep["patch_id"].astype(str)
            sub = sub.copy()
            sub["patient_id"] = sub["patient_id"].astype(str)
            sub["patch_id"] = sub["patch_id"].astype(str)
            sub = sub.merge(keep, on=["patient_id", "patch_id"], how="inner")
            if len(sub) == 0:
                continue
            dim_cols = [c for c in fm_df.columns if c.startswith("d")]
            pooled = sub.groupby("patient_id")[dim_cols].mean().reset_index()
            merged = pooled.merge(patient_meta, on="patient_id", how="inner")
        else:
            # Per-patient direct lookup
            id_col = "patient_id" if "patient_id" in sub.columns else "subject_id"
            X_list, time_list, ev_list, pid_list = [], [], [], []
            for pid in sorted(sub[id_col].unique()):
                emb = get_patient_embedding(sub, pid)
                if emb is None:
                    continue
                p_meta = patient_meta[patient_meta["patient_id"].astype(str) == str(pid)]
                if len(p_meta) == 0:
                    continue
                X_list.append(emb)
                ev_list.append(int(p_meta[event_col].values[0]))
                time_list.append(float(p_meta[time_col].values[0]))
                pid_list.append(str(pid))
            if len(X_list) < 20:
                print(f"  {task_name} {fm_name}: too few samples ({len(X_list)}), skip")
                continue
            dim_cols = [c for c in fm_df.columns if c.startswith("d")]
            merged = pd.DataFrame(np.array(X_list), columns=dim_cols)
            merged["patient_id"] = pid_list
            merged[event_col] = ev_list
            merged[time_col] = time_list

        if len(merged) < 20:
            continue

        X = merged[dim_cols].values.astype(np.float32)
        time_arr = merged[time_col].values.astype(float)
        event_arr = merged[event_col].values.astype(int)
        pid_list = merged["patient_id"].astype(str).tolist()

        nan_mask = np.isnan(X).any(axis=1)
        if nan_mask.sum() > 0:
            X = X[~nan_mask]; time_arr = time_arr[~nan_mask]
            event_arr = event_arr[~nan_mask]
            pid_list = [p for p, m in zip(pid_list, ~nan_mask) if m]
        np.nan_to_num(X, copy=False)
        if len(X) < 20:
            continue

        n_events = int(event_arr.sum())
        epv_ratio = X.shape[1] / max(1, n_events)
        use_pca = (X.shape[1] > X.shape[0]) or (n_events < 100) or (epv_ratio > 3)
        pca_dim = min(50, X.shape[0] - 1, X.shape[1]) if use_pca else None

        scaler = StandardScaler()
        X_s = scaler.fit_transform(X)
        if use_pca:
            pca = PCA(n_components=pca_dim, random_state=SEED)
            X_s = pca.fit_transform(X_s)

        cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        best_alpha, best_c = 1.0, -1.0
        for alpha in ALPHA_GRID:
            cidx_folds = []
            for tr_idx, va_idx in cv.split(X_s, event_arr):
                if event_arr[va_idx].sum() < 1:
                    continue
                y_tr = np.array(
                    [(bool(e), float(t)) for e, t in zip(event_arr[tr_idx], time_arr[tr_idx])],
                    dtype=[("event", bool), ("time", float)])
                y_va = np.array(
                    [(bool(e), float(t)) for e, t in zip(event_arr[va_idx], time_arr[va_idx])],
                    dtype=[("event", bool), ("time", float)])
                cph = CoxPHSurvivalAnalysis(alpha=alpha)
                try:
                    cph.fit(X_s[tr_idx], y_tr)
                    risk = cph.predict(X_s[va_idx])
                    c = concordance_index_censored(y_va["event"], y_va["time"], risk)[0]
                    cidx_folds.append(c)
                except Exception:
                    continue
            if cidx_folds:
                mean_c = float(np.mean(cidx_folds))
                if mean_c > best_c:
                    best_c, best_alpha = mean_c, alpha

        y_full = np.array(
            [(bool(e), float(t)) for e, t in zip(event_arr, time_arr)],
            dtype=[("event", bool), ("time", float)])
        cph_final = CoxPHSurvivalAnalysis(alpha=best_alpha)
        try:
            cph_final.fit(X_s, y_full)
        except Exception as e:
            print(f"  {task_name} {fm_name}: final fit failed: {e}")
            continue
        lp = cph_final.predict(X_s)
        surv_funcs = cph_final.predict_survival_function(X_s)
        surv_at_h = np.array([float(sf(horizon_days)) for sf in surv_funcs])

        for pid, ev, tt, lpi, si in zip(pid_list, event_arr, time_arr, lp, surv_at_h):
            out_rows.append({
                "fm": fm_name, "task": task_name,
                "patient_id": str(pid),
                "event": int(ev), "time": float(tt),
                "linear_predictor": float(lpi),
                f"surv_{horizon_months}mo": float(si),
                "best_alpha": float(best_alpha),
                "use_pca": bool(use_pca),
                "n_pca": int(pca_dim) if use_pca else 0,
            })
        print(f"  {task_name} {fm_name}: {len(pid_list)} per-patient predictions dumped "
              f"(alpha={best_alpha}, c-index={best_c:.3f})")
    return pd.DataFrame(out_rows)


# T3 — HECKTOR RFS Cox at 24 mo, lesion-pooled, cohort = task2_patient & label==1
def t3_cohort_filter(lbl):
    needed = {"task2_patient", "relapse", "rfs_days", "label"}
    if not needed.issubset(lbl.columns):
        print(f"  T3 cohort filter: missing columns {needed - set(lbl.columns)}")
        return pd.DataFrame()
    return lbl[
        lbl["task2_patient"].fillna(False).astype(bool) &
        lbl["relapse"].notna() & lbl["rfs_days"].notna() &
        (lbl["rfs_days"] > 0) & (lbl["label"] == 1)
    ].copy()


if "hecktor" in EMBEDDINGS and "hecktor" in LABELS:
    print("\n" + "=" * 60); print(f"T3 calibration dump (HECKTOR Cox RFS, {COX_HORIZON_MONTHS} mo)"); print("=" * 60)
    t3_dump = run_cox_with_dump(EMBEDDINGS["hecktor"], LABELS["hecktor"],
                                 "t3", COX_HORIZON_MONTHS, "rfs_days", "relapse",
                                 cohort_filter=t3_cohort_filter,
                                 lesion_pool=True)
    t3_dump.to_parquet(OUTPUT_DIR / "perpred_t3.parquet", index=False)
else:
    t3_dump = pd.DataFrame()

# T4 — NSCLC OS Cox at 24 mo (v2: was 36 mo, degenerate at that horizon)
if "t4" in EMBEDDINGS and "t4" in LABELS:
    lbl4 = LABELS["t4"]
    event_col_4 = next((c for c in ["event", "death", "os_event"] if c in lbl4.columns), None)
    time_col_4 = next((c for c in ["time_to_death", "os_time", "time"] if c in lbl4.columns), None)
    if event_col_4 is not None and time_col_4 is not None:
        print("\n" + "=" * 60); print(f"T4 calibration dump (NSCLC Cox OS, {COX_HORIZON_MONTHS} mo)"); print("=" * 60)
        t4_dump = run_cox_with_dump(EMBEDDINGS["t4"], lbl4,
                                     "t4", COX_HORIZON_MONTHS, time_col_4, event_col_4)
        t4_dump.to_parquet(OUTPUT_DIR / "perpred_t4.parquet", index=False)
    else:
        print(f"T4: event/time columns not found")
        t4_dump = pd.DataFrame()
else:
    t4_dump = pd.DataFrame()

# %% [markdown]
# ## 7. Manifest with SHA-256 hashes

# %%
def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


manifest_rows = []
for f in sorted(OUTPUT_DIR.glob("perpred_*.parquet")):
    if f.stat().st_size == 0:
        continue
    df = pd.read_parquet(f)
    manifest_rows.append({
        "file": f.name,
        "n_rows": len(df),
        "n_fms": df["fm"].nunique() if "fm" in df.columns else 0,
        "task": df["task"].iloc[0] if "task" in df.columns and len(df) > 0 else "",
        "sha256": file_sha256(f),
        "size_bytes": f.stat().st_size,
    })
manifest = pd.DataFrame(manifest_rows)
manifest.to_csv(OUTPUT_DIR / "perpred_manifest.csv", index=False)
print("\n=== Calibration inference v2 manifest ===")
print(manifest.to_string(index=False))
