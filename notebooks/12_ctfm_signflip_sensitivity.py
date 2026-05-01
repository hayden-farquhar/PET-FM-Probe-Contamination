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
# # PET-FM-Bench: CT-FM × T4 Cox Sign-Flip Sensitivity (Step 15)
#
# **Runtime:** CPU | **Internet:** ON for pip install | **Time:** ~2-5 min |
# **GPU:** Not needed
#
# Sensitivity analysis for the dry-run finding that **CT-FM × T4 c-index =
# 0.468** (sub-chance, CI [0.397, 0.541] including 0.5). Two interpretations
# for sub-chance c-index in registration §5.6.8 negative-transfer logic:
#
# 1. **Noise interpretation:** the c-index is just below 0.5 due to small-N
#    sampling variation. Sign-flipping the risk score should produce
#    `1 - c_index ≈ 0.532` — if so, the original sub-chance value carries
#    no anti-prediction signal.
# 2. **Genuine negative transfer:** CT-FM features are anti-correlated
#    with NSCLC survival in this cohort. Sign-flipped c-index would NOT
#    equal 1 - original; it'd be substantially different (e.g., 0.55+ or
#    <0.45) revealing structural anti-correlation.
#
# This notebook runs both `risk` and `-risk` Cox PH on the T4 v3 cohort
# (with the v2/v3 alpha grid + PCA(50) per amendment A2) and reports both
# c-indices with bootstrap CIs.
#
# **Datasets to attach:**
# - `pet-fm-bench-t4-embeddings-v3`
#
# **Output:**
# - `ctfm_signflip_results.csv` — schema
#   `(fm, task, sign, c_index, ci_low, ci_high, alpha, n, n_events)`

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
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning, module="sksurv")

!pip install -q scikit-survival

from sksurv.linear_model import CoxPHSurvivalAnalysis  # noqa: E402
from sksurv.metrics import concordance_index_censored  # noqa: E402

OUT_DIR = Path("/kaggle/working")
OUT_DIR.mkdir(parents=True, exist_ok=True)

freeze_timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
print(f"Freeze timestamp (UTC): {freeze_timestamp}")

N_FOLDS = 5
N_BOOTSTRAP = 1000
ALPHA_GRID = [0.001, 0.01, 0.1, 1, 10, 100, 1000]


# %% [markdown]
# ## 2. Load CT-FM T4 embeddings + labels

# %%
def find_embed_dir(task):
    v3 = list(Path("/kaggle/input").rglob(f"pet-fm-bench-{task}-embeddings-v3"))
    if not v3:
        return None
    embed_dir = v3[0]
    nested = embed_dir / "embeddings"
    return nested if nested.exists() else embed_dir


t4_dir = find_embed_dir("t4")
if t4_dir is None:
    raise FileNotFoundError("pet-fm-bench-t4-embeddings-v3 not attached")

ctfm_path = t4_dir / "ct_fm.parquet"
labels_path = next(t4_dir.glob("*labels*"), None) or next(
    t4_dir.parent.rglob("t4_labels.csv"), None
)
if not ctfm_path.exists() or labels_path is None:
    raise FileNotFoundError(f"ct_fm.parquet or t4_labels.csv missing in {t4_dir}")

ct_fm = pd.read_parquet(ctfm_path)
labels = pd.read_csv(labels_path)
print(f"CT-FM T4 embeddings: {len(ct_fm)} rows, "
      f"{sum(1 for c in ct_fm.columns if c.startswith('d'))}-dim")
print(f"T4 labels: {len(labels)} patients, "
      f"events: {int(labels['event'].sum())} / {labels['event'].notna().sum()}")


# %% [markdown]
# ## 3. Build (X, time, event) arrays per probe_analysis.py v4 conventions

# %%
def get_patient_emb(df, pid):
    sub = df[df["patient_id"] == pid]
    if "view" in df.columns and "volume" in df["view"].values:
        sub = sub[sub["view"] == "volume"]
    if "layer" in df.columns and "pool" in sub["layer"].values:
        sub = sub[sub["layer"] == "pool"]
    if len(sub) == 0:
        return None
    dim_cols = [c for c in df.columns if c.startswith("d")]
    return sub[dim_cols].values[0]


X_list, time_list, event_list = [], [], []
for pid in sorted(ct_fm["patient_id"].unique()):
    emb = get_patient_emb(ct_fm, pid)
    if emb is None:
        continue
    row = labels[labels["patient_id"] == pid]
    if len(row) == 0:
        continue
    ev = row["event"].values[0]
    t = row["time_to_death"].values[0]
    if pd.isna(ev) or pd.isna(t) or t <= 0:
        continue
    X_list.append(emb)
    time_list.append(float(t))
    event_list.append(int(ev))

X = np.array(X_list)
time_arr = np.array(time_list)
event_arr = np.array(event_list)
np.nan_to_num(X, copy=False)
print(f"Cohort: n={len(X)}, events={int(event_arr.sum())}, "
      f"embed_dim={X.shape[1]}")

y_struct = np.array(
    [(bool(e), float(t)) for e, t in zip(event_arr, time_arr)],
    dtype=[("event", bool), ("time", float)]
)


# %% [markdown]
# ## 4. CV training (best alpha) → OOF risk scores → flip test

# %%
def fit_oof_risk(X, y_struct, time_arr, event_arr):
    """Match probe_analysis.py v4 survival probe: per-fold scaler+PCA+Cox,
    OOF risk scores at the CV-best alpha."""
    use_pca = (X.shape[1] > X.shape[0]) or (event_arr.sum() < 100)
    pca_dim = min(50, X.shape[0] - 1, X.shape[1])
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

    def _fit_fold(Xtr, Xte, alpha):
        sc = StandardScaler()
        Xtr_s = sc.fit_transform(Xtr); Xte_s = sc.transform(Xte)
        if use_pca:
            pca = PCA(n_components=pca_dim, random_state=42)
            Xtr_s = pca.fit_transform(Xtr_s); Xte_s = pca.transform(Xte_s)
        m = CoxPHSurvivalAnalysis(alpha=alpha)
        return m, Xtr_s, Xte_s

    # Alpha selection
    best_alpha, best_score = 1.0, -1
    for alpha in ALPHA_GRID:
        scores = []
        for tr, te in cv.split(X, event_arr):
            try:
                m, Xtr_s, Xte_s = _fit_fold(X[tr], X[te], alpha)
                m.fit(Xtr_s, y_struct[tr])
                pr = m.predict(Xte_s)
                c, *_ = concordance_index_censored(
                    event_arr[te].astype(bool), time_arr[te], pr)
                scores.append(c)
            except Exception:
                continue
        if scores and np.mean(scores) > best_score:
            best_score = np.mean(scores)
            best_alpha = alpha

    # OOF risk at best alpha
    risk = np.zeros(len(X))
    for tr, te in cv.split(X, event_arr):
        m, Xtr_s, Xte_s = _fit_fold(X[tr], X[te], best_alpha)
        try:
            m.fit(Xtr_s, y_struct[tr])
            risk[te] = m.predict(Xte_s)
        except Exception:
            risk[te] = 0.0
    return risk, best_alpha


risk, alpha_chosen = fit_oof_risk(X, y_struct, time_arr, event_arr)
print(f"Best alpha (CV): {alpha_chosen}")


def cindex_with_ci(risk_arr):
    c, *_ = concordance_index_censored(
        event_arr.astype(bool), time_arr, risk_arr)
    rng = np.random.RandomState(42)
    boot = []
    for _ in range(N_BOOTSTRAP):
        idx = rng.choice(len(event_arr), size=len(event_arr), replace=True)
        if event_arr[idx].sum() < 2:
            continue
        try:
            cb, *_ = concordance_index_censored(
                event_arr[idx].astype(bool), time_arr[idx], risk_arr[idx])
            boot.append(cb)
        except Exception:
            continue
    return float(c), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


c_pos, lo_pos, hi_pos = cindex_with_ci(risk)
c_neg, lo_neg, hi_neg = cindex_with_ci(-risk)

print(f"\n=== Sign-flip sensitivity ===")
print(f"  +risk: c = {c_pos:.4f} [{lo_pos:.4f}, {hi_pos:.4f}]")
print(f"  -risk: c = {c_neg:.4f} [{lo_neg:.4f}, {hi_neg:.4f}]")
print(f"  Sum c(+) + c(-) = {c_pos + c_neg:.4f}  "
      f"(expected ≈ 1.0 if rank-based)")
print(f"  Symmetry deviation from 0.500: |c(+) - 0.5| = {abs(c_pos - 0.5):.4f}; "
      f"|c(-) - 0.5| = {abs(c_neg - 0.5):.4f}")

if abs(c_pos + c_neg - 1.0) < 0.01:
    interpretation = ("Sum is very close to 1.0 — the sub-chance c-index is "
                      "consistent with rank-symmetric noise, NOT genuine "
                      "anti-prediction. Negative-transfer interpretation per "
                      "registration §5.6.8 is therefore weak.")
else:
    interpretation = (f"Sum deviates from 1.0 by {abs(c_pos+c_neg-1.0):.4f} — "
                      "this is unusual for rank-based scoring and warrants "
                      "investigation (could indicate genuine asymmetric anti-"
                      "correlation, or a bootstrap-CI artefact at small N).")
print(f"\nInterpretation: {interpretation}")


# %% [markdown]
# ## 5. Save

# %%
results = pd.DataFrame([
    {"fm": "ct_fm", "task": "t4", "sign": "+risk",
     "c_index": c_pos, "ci_low": lo_pos, "ci_high": hi_pos,
     "alpha": alpha_chosen, "n": int(len(X)), "n_events": int(event_arr.sum())},
    {"fm": "ct_fm", "task": "t4", "sign": "-risk",
     "c_index": c_neg, "ci_low": lo_neg, "ci_high": hi_neg,
     "alpha": alpha_chosen, "n": int(len(X)), "n_events": int(event_arr.sum())},
])

out_path = OUT_DIR / "ctfm_signflip_results.csv"
meta_path = OUT_DIR / "ctfm_signflip_metadata.json"

results.to_csv(out_path, index=False)
with open(meta_path, "w") as f:
    json.dump({
        "stage": "Step 15 — CT-FM × T4 Cox sign-flip sensitivity",
        "freeze_timestamp_utc": freeze_timestamp,
        "registration_section": "§5.6.8 (negative-transfer test)",
        "context": (
            "Dry-run T4 c-index for CT-FM = 0.468 (sub-chance, CI includes 0.5). "
            "This sensitivity test distinguishes noise (c+ + c- ≈ 1.0) from "
            "genuine negative transfer (asymmetric c values)."
        ),
        "alpha_chosen": alpha_chosen,
        "results": results.to_dict(orient="records"),
        "interpretation": interpretation,
    }, f, indent=2)
print(f"\nWrote: {out_path}")
print(f"Wrote: {meta_path}")
print(results.to_string(index=False))
