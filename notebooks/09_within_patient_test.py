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
# # PET-FM-Bench: Phase 2 Stage 3 — Within-Patient Dirty-vs-Clean Test
#
# **Runtime:** CPU | **Internet:** Off OK | **Time:** ~10-30 min | **GPU:** Not needed
#
# Pre-registration §4.6 / §5.6.6 Stage 3: for each (FM × task) pair where the
# Stage 2 contamination intersection identified BOTH ≥1 contaminated and ≥1
# clean evaluation patient, run the registered probe separately on the dirty
# and clean subsets, and test whether the FM performs better on patients it
# previously saw in pretraining (= memorisation signal).
#
# **Permutation null:** the `in_fm_training_data` flag is shuffled across
# patients within the task; the dirty-minus-clean delta is recomputed under
# the shuffled labels 1000 times. The p-value is the fraction of permuted
# deltas ≥ the observed delta.
#
# **Datasets to attach:**
# - `pet-fm-bench-contamination-stage2` (08 output:
#   `contamination_per_patient.parquet`)
# - `pet-fm-bench-t4-embeddings-v3`
# - `pet-fm-bench-t6-embeddings-v3`
# - `pet-fm-bench-t7-embeddings-v3`
# - `pet-fm-bench-t8-embeddings-v3`
# - `pet-fm-bench-t9-embeddings-v3`
# - `pet-fm-bench-t7-patches-v3` (clinical CSVs for T7 outcome derivation)
#
# **Output:**
# - `dirty_vs_clean_results.csv` — schema `(fm, task, metric, dirty_value,
#   clean_value, delta, perm_p_value, n_dirty, n_clean, status)`

# %% [markdown]
# ## 1. Setup

# %%
import json
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler, LabelEncoder

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning, module="sksurv")

!pip install -q scikit-survival

OUT_DIR = Path("/kaggle/working")
OUT_DIR.mkdir(parents=True, exist_ok=True)

freeze_timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
print(f"Freeze timestamp (UTC): {freeze_timestamp}")

N_FOLDS = 5
# v2 (2026-04-27): N_PERMUTATIONS reduced from 1000 → 100 after compute-cost
# audit. With 1000 perms × 2 probe runs × ~30 sec/Cox survival fit, a single
# (FM × task) cell would take ~17 hours, exceeding Kaggle's 12-hour CPU
# session limit before three Tier 1 cells could complete. 100 perms gives
# minimum reportable p = 0.01 (adequate for the registered α=0.05 threshold)
# at ~3 hours total runtime across the three runnable cells.
N_PERMUTATIONS = 100
ALPHA_GRID_COX = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
C_GRID = [0.001, 0.01, 0.1, 1, 10, 100]


def find_input(name):
    matches = list(Path("/kaggle/input").rglob(name))
    return matches[0] if matches else None


# %% [markdown]
# ## 2. Load Stage 2 contamination flags + embeddings

# %%
contam_path = find_input("contamination_per_patient.parquet")
if contam_path is None:
    raise FileNotFoundError(
        "contamination_per_patient.parquet not attached. Add the "
        "pet-fm-bench-contamination-stage2 Kaggle dataset (08 output)."
    )
contam_df = pd.read_parquet(contam_path)
print(f"Stage 2 per-patient flags: {len(contam_df)} rows, "
      f"{contam_df['fm'].nunique()} FMs × {contam_df['task'].nunique()} tasks")


def find_embed_dir(task):
    v3 = list(Path("/kaggle/input").rglob(f"pet-fm-bench-{task}-embeddings-v3"))
    if not v3:
        return None
    embed_dir = v3[0]
    nested = embed_dir / "embeddings"
    return nested if nested.exists() else embed_dir


def load_fm_embeddings(embed_dir):
    if embed_dir is None:
        return {}
    fm_data = {}
    for f in sorted(embed_dir.glob("*.parquet")):
        if f.stem.startswith("t") and f.stem[1:].split("_")[0].isdigit():
            continue
        fm_data[f.stem] = pd.read_parquet(f)
    return fm_data


def load_labels(embed_dir):
    if embed_dir is None:
        return None
    cands = list(embed_dir.glob("*labels*"))
    if not cands:
        cands = list(embed_dir.parent.rglob("*labels*"))
    return pd.read_csv(cands[0]) if cands else None


TASK_DIRS = {t: find_embed_dir(t) for t in ["t1", "t4", "t5", "t6", "t7", "t8", "t9"]}
EMBEDDINGS = {t: load_fm_embeddings(d) for t, d in TASK_DIRS.items() if d}
LABELS = {t: load_labels(d) for t, d in TASK_DIRS.items() if d}
LABELS = {t: l for t, l in LABELS.items() if l is not None}

for t, fms in EMBEDDINGS.items():
    print(f"  {t}: {len(fms)} FMs loaded")


def get_patient_embedding(df, patient_id, view="coronal", layer="cls"):
    id_col = "patient_id" if "patient_id" in df.columns else "subject_id"
    mask = df[id_col].astype(str) == str(patient_id)
    if "view" in df.columns:
        if "volume" in df["view"].values:
            mask = mask & (df["view"] == "volume")
        else:
            mask = mask & (df["view"] == view)
    if "layer" in df.columns:
        sub = df[df[id_col].astype(str) == str(patient_id)]
        if "pool" in sub["layer"].values:
            mask = mask & (df["layer"] == "pool")
        else:
            mask = mask & (df["layer"] == layer)
    rows = df[mask]
    if len(rows) == 0:
        return None
    dim_cols = [c for c in df.columns if c.startswith("d")]
    return rows[dim_cols].values[0]


# %% [markdown]
# ## 3. Probe runners on a fixed patient subset
#
# Each runner takes a list of patient IDs (the "subset") and returns the
# probe metric on that subset. Implementation matches `probe_analysis.py`
# v4 (per-fold scaler + PCA, widened alpha/C grids, NaN-handling).

# %%
def run_classification_subset(fm_df, labels_df, label_col, patient_ids):
    """Cross-validated macro AUROC on a subset of patients."""
    X_list, y_list = [], []
    for pid in patient_ids:
        emb = get_patient_embedding(fm_df, pid)
        if emb is None:
            continue
        row = labels_df[labels_df.iloc[:, 0].astype(str) == str(pid)]
        if len(row) == 0:
            continue
        lab = row[label_col].values[0]
        if pd.isna(lab):
            continue
        X_list.append(emb)
        y_list.append(lab)
    if len(X_list) < 10:
        return None, len(X_list)
    X = np.array(X_list); y = np.array(y_list)
    np.nan_to_num(X, copy=False)
    le = LabelEncoder(); y_enc = le.fit_transform(y)
    if len(le.classes_) < 2:
        return None, len(X_list)
    # v2: check n_splits BEFORE constructing StratifiedKFold — the constructor
    # raises ValueError if n_splits < 2, so the post-hoc guard never runs.
    # This trips on permuted T8 (multi-class) when a shuffle leaves only 1
    # patient in the rarest class.
    n_splits_target = min(N_FOLDS, int(np.bincount(y_enc).min()))
    if n_splits_target < 2:
        return None, len(X_list)
    cv = StratifiedKFold(n_splits=n_splits_target, shuffle=True, random_state=42)
    best_score = -1
    for c in C_GRID:
        try:
            scaler = StandardScaler()
            Xs = scaler.fit_transform(X)
            preds = cross_val_predict(
                LogisticRegression(C=c, penalty="l2", solver="lbfgs",
                                   max_iter=1000, random_state=42),
                Xs, y_enc, cv=cv, method="predict_proba"
            )
            if len(le.classes_) == 2:
                s = roc_auc_score(y_enc, preds[:, 1])
            else:
                s = roc_auc_score(y_enc, preds, multi_class="ovr", average="macro")
            best_score = max(best_score, s)
        except Exception:
            continue
    return (best_score if best_score > -1 else None), len(X_list)


def run_survival_subset(fm_df, labels_df, patient_ids):
    """5-fold CV concordance index on a subset (T4 protocol)."""
    from sksurv.linear_model import CoxPHSurvivalAnalysis
    from sksurv.metrics import concordance_index_censored

    X_list, time_list, event_list = [], [], []
    for pid in patient_ids:
        emb = get_patient_embedding(fm_df, pid)
        if emb is None:
            continue
        row = labels_df[labels_df["patient_id"].astype(str) == str(pid)]
        if len(row) == 0:
            continue
        ev = row["event"].values[0]; t = row["time_to_death"].values[0]
        if pd.isna(ev) or pd.isna(t) or t <= 0:
            continue
        X_list.append(emb); time_list.append(float(t)); event_list.append(int(ev))

    if len(X_list) < 20 or sum(event_list) < 5:
        return None, len(X_list)
    X = np.array(X_list); time_arr = np.array(time_list); event_arr = np.array(event_list)
    np.nan_to_num(X, copy=False)
    y_struct = np.array([(bool(e), float(t)) for e, t in zip(event_arr, time_arr)],
                        dtype=[("event", bool), ("time", float)])

    use_pca = (X.shape[1] > X.shape[0]) or (event_arr.sum() < 100)
    pca_dim = min(50, X.shape[0] - 1, X.shape[1])

    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    best_c = None
    for alpha in ALPHA_GRID_COX:
        scores = []
        for tr, te in cv.split(X, event_arr):
            try:
                sc = StandardScaler(); Xtr = sc.fit_transform(X[tr]); Xte = sc.transform(X[te])
                if use_pca:
                    pca = PCA(n_components=pca_dim, random_state=42)
                    Xtr = pca.fit_transform(Xtr); Xte = pca.transform(Xte)
                m = CoxPHSurvivalAnalysis(alpha=alpha)
                m.fit(Xtr, y_struct[tr])
                pr = m.predict(Xte)
                c, *_ = concordance_index_censored(
                    event_arr[te].astype(bool), time_arr[te], pr)
                scores.append(c)
            except Exception:
                continue
        if scores and (best_c is None or np.mean(scores) > best_c[1]):
            best_c = (alpha, float(np.mean(scores)))
    return (best_c[1] if best_c else None), len(X_list)


# T7 2y-OS derivation re-used from probe_analysis.py
# v2: ACRIN-6668 ships clinical data as TWO file sets (75% / 25% random
# partitions of the trial cohort). Earlier `next(rglob)` picked only the
# first F1.csv found, missing ~25% of the cohort. v2 concatenates ALL
# F1.csv and DS.csv files under the patches dir.
def derive_t7_2yr_os_inline(t7_patches_dir):
    if t7_patches_dir is None:
        return None
    f1_paths = list(t7_patches_dir.rglob("F1.csv"))
    ds_paths = list(t7_patches_dir.rglob("DS.csv"))
    if not f1_paths:
        return None
    f1 = pd.concat([pd.read_csv(p, low_memory=False) for p in f1_paths],
                   ignore_index=True)
    ds = (pd.concat([pd.read_csv(p, low_memory=False) for p in ds_paths],
                    ignore_index=True)
          if ds_paths else None)
    print(f"  T7 clinical concat: {len(f1_paths)} F1.csv files → "
          f"{len(f1)} rows, {f1['cn'].nunique()} unique patients; "
          f"{len(ds_paths)} DS.csv files → "
          f"{len(ds) if ds is not None else 0} rows")
    f1c = f1.dropna(subset=["f1e2"]).copy()
    f1c["fu_days"] = f1c["F1e1d"].fillna(f1c.get("F1e3d"))
    f1c = f1c.dropna(subset=["fu_days"])
    last = (f1c.sort_values(["cn", "fu_days"]).groupby("cn").tail(1).set_index("cn"))
    out = {}
    for cn, r in last.iterrows():
        vs, lv = r["f1e2"], r["fu_days"]
        dd = r.get("F1e3d") if pd.notna(r.get("F1e3d")) else lv
        if vs == 2: out[cn] = 1 if dd <= 730 else 0
        elif vs == 9 and lv <= 730: out[cn] = 1
        elif vs == 1 and lv >= 730: out[cn] = 0
    if ds is not None:
        ds_idx = ds.set_index("cn")
        for cn in set(ds["cn"]) - set(out):
            if cn in ds_idx.index:
                rr = ds_idx.loc[cn]
                if rr.get("dse3") == 2 and pd.notna(rr.get("DSe2d")) and rr["DSe2d"] <= 730:
                    out[cn] = 1
    rows = []
    for cn, ev in out.items():
        n = int(cn) if isinstance(cn, (int, np.integer, float)) else cn
        for f in (n, str(n), f"ACRIN-NSCLC-FDG-PET-{n:03d}",
                  f"ACRIN-NSCLC-FDG-PET-{n:04d}", f"ACRIN-NSCLC-FDG-PET-{n}"):
            rows.append({"patient_id": f, "event_2yr_os": int(ev)})
    return pd.DataFrame(rows)


# %% [markdown]
# ## 4. Run dirty-vs-clean test per (FM × task)

# %%
TASK_PROBE_FN = {
    "t4": "survival",
    "t8": "classification:subtype",
    "t7": "classification:event_2yr_os",
    # T1 (lesion-patch classification) — per-patch labels at patient-level;
    # within-patient probe technically possible but expected zero contamination
    # (FM training cohorts disjoint from AutoPET-I FDAT) → skipped via
    # insufficient_subsets path.
    # T1 entry omitted intentionally: contamination is dirty=0 across all FMs
    # by construction, so the loop will skip with "insufficient_subsets" reason
    # without needing a custom probe runner.
}
# T6/T9 are test-retest tasks; T5 is zero-shot detection (per registration §3.1
# T5 has no within-task train labels — fits via T1 → T5 transfer at Phase 5).
# T2 and T3 are HECKTOR 2025 (per A12) — contamination is dirty=0 by construction
# since no audited FM (FMCIB/CT-FM/BiomedCLIP/DINOv2/RAD-DINO/random_init) has
# HECKTOR in its training manifest. The dirty-vs-clean within-patient test is
# undefined when there are no dirty patients to compare against.
# Dirty-vs-clean test isn't naturally defined for any of these.
SKIP_AS_NON_CLASSIFICATION = {"t2", "t3", "t5", "t6", "t9"}

# T7 needs derived labels
t7_patches = next(Path("/kaggle/input").rglob("pet-fm-bench-t7-patches-v3"), None)
t7_labels = derive_t7_2yr_os_inline(t7_patches) if t7_patches else None
if t7_labels is not None:
    print(f"T7 derived 2y-OS labels: {t7_labels['patient_id'].nunique()} unique patient IDs")

results = []

for (fm, task), grp in contam_df.groupby(["fm", "task"]):
    if task in SKIP_AS_NON_CLASSIFICATION:
        # T5 = zero-shot (no within-task train labels), T6/T9 = test-retest
        # (no per-patient classification target), T2/T3 = HECKTOR with dirty=0
        # by construction (per A12). Dirty-vs-clean test undefined.
        if task == "t5":
            skip_label = "skipped:zero_shot_task"
        elif task in {"t6", "t9"}:
            skip_label = "skipped:test_retest_task"
        elif task in {"t2", "t3"}:
            skip_label = "skipped:no_contamination_by_construction"
        else:
            skip_label = "skipped:non_classification"
        results.append({"fm": fm, "task": task, "status": skip_label,
                        "n_dirty": int(grp["in_fm_training_data"].sum()),
                        "n_clean": int((~grp["in_fm_training_data"]).sum())})
        continue
    if task not in TASK_PROBE_FN:
        # T1 falls here — no custom probe runner (deferred); contamination
        # expected dirty=0 across all FMs by construction (AutoPET-I disjoint
        # from all FM training cohorts). The insufficient_subsets path below
        # will handle this gracefully.
        results.append({"fm": fm, "task": task, "status": "skipped:no_probe_runner",
                        "n_dirty": int(grp["in_fm_training_data"].sum()),
                        "n_clean": int((~grp["in_fm_training_data"]).sum())})
        continue

    fm_data = EMBEDDINGS.get(task, {}).get(fm)
    if fm_data is None:
        results.append({"fm": fm, "task": task, "status": "skipped:no_embeddings"})
        continue

    dirty_pids = set(grp.loc[grp["in_fm_training_data"], "patient_id"].astype(str))
    clean_pids = set(grp.loc[~grp["in_fm_training_data"], "patient_id"].astype(str))

    if len(dirty_pids) < 10 or len(clean_pids) < 10:
        results.append({
            "fm": fm, "task": task,
            "status": f"skipped:insufficient_subsets (dirty={len(dirty_pids)}, clean={len(clean_pids)})",
            "n_dirty": len(dirty_pids), "n_clean": len(clean_pids),
        })
        continue

    probe_kind = TASK_PROBE_FN[task]
    if probe_kind == "survival":
        labels_df = LABELS["t4"]
        runner = lambda pids: run_survival_subset(fm_data, labels_df, pids)
        metric_name = "c_index"
    elif probe_kind.startswith("classification:"):
        col = probe_kind.split(":", 1)[1]
        labels_df = t7_labels if task == "t7" else LABELS[task]
        if labels_df is None or col not in labels_df.columns:
            results.append({"fm": fm, "task": task,
                            "status": f"skipped:no_label_column ({col})"})
            continue
        runner = lambda pids: run_classification_subset(fm_data, labels_df, col, pids)
        metric_name = "auroc_macro" if task == "t8" else "auroc_2yr_os"

    # Observed dirty/clean metrics
    dirty_metric, n_dirty_used = runner(list(dirty_pids))
    clean_metric, n_clean_used = runner(list(clean_pids))
    if dirty_metric is None or clean_metric is None:
        results.append({
            "fm": fm, "task": task, "metric": metric_name,
            "n_dirty": n_dirty_used, "n_clean": n_clean_used,
            "status": "skipped:probe_failed_on_subset",
        })
        continue

    delta = dirty_metric - clean_metric

    # Permutation null: shuffle the in_fm_training_data flag across patients.
    # v2: progress prints every 10 perms so the run isn't silent for hours.
    rng = np.random.RandomState(42)
    all_pids = list(dirty_pids | clean_pids)
    n_dirty_orig = len(dirty_pids)
    perm_deltas = []
    print(f"  {fm} × {task}: starting {N_PERMUTATIONS} permutations "
          f"(observed Δ={delta:+.4f}, n_dirty={n_dirty_used}, "
          f"n_clean={n_clean_used})")
    import time
    t0 = time.time()
    for k in range(N_PERMUTATIONS):
        shuf = rng.permutation(all_pids)
        sd = set(shuf[:n_dirty_orig])
        sc = set(shuf[n_dirty_orig:])
        d_m, _ = runner(list(sd))
        c_m, _ = runner(list(sc))
        if d_m is not None and c_m is not None:
            perm_deltas.append(d_m - c_m)
        if (k + 1) % 10 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (k + 1) * (N_PERMUTATIONS - k - 1)
            print(f"    perm {k+1}/{N_PERMUTATIONS} ({elapsed:.0f}s elapsed, "
                  f"~{eta:.0f}s remaining); valid_perms={len(perm_deltas)}")
    perm_deltas = np.array(perm_deltas)
    p_value = float((perm_deltas >= delta).mean()) if len(perm_deltas) else float("nan")

    results.append({
        "fm": fm, "task": task, "metric": metric_name,
        "dirty_value": float(dirty_metric), "clean_value": float(clean_metric),
        "delta": float(delta), "perm_p_value": p_value,
        "n_dirty": int(n_dirty_used), "n_clean": int(n_clean_used),
        "n_perm": len(perm_deltas), "status": "ok",
    })
    print(f"  {fm:11s} × {task}: dirty={dirty_metric:.3f} clean={clean_metric:.3f} "
          f"Δ={delta:+.3f} (p={p_value:.3f}, n_d={n_dirty_used}, n_c={n_clean_used})")

# %% [markdown]
# ## 5. Save outputs

# %%
results_df = pd.DataFrame(results)
out_path = OUT_DIR / "dirty_vs_clean_results.csv"
metadata_path = OUT_DIR / "within_patient_test_metadata.json"

results_df.to_csv(out_path, index=False)
print(f"\nWrote: {out_path}")

with open(metadata_path, "w") as f:
    json.dump({
        "stage": "Phase 2 Stage 3 — Within-Patient Dirty-vs-Clean Test",
        "freeze_timestamp_utc": freeze_timestamp,
        "registration_section": "§4.6 / §5.6.6",
        "n_permutations": N_PERMUTATIONS,
        "n_pairs_evaluated": len(results),
        "n_pairs_ok": int((results_df["status"] == "ok").sum()) if "status" in results_df.columns else 0,
        "skip_reasons": (
            results_df["status"].value_counts().to_dict()
            if "status" in results_df.columns else {}
        ),
    }, f, indent=2)
print(f"Wrote: {metadata_path}")

# %% [markdown]
# ## 6. Done
#
# Save & Run All → Output → New Dataset → `pet-fm-bench-contamination-stage3`.
# Combined with Stage 2 output, this becomes the Phase 2 freeze artefact in
# Stage 4 (`10_contamination_freeze.py`).
