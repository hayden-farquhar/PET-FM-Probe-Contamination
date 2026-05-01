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
# # PET-FM-Bench: FMCIB PET-Saturation Diagnostic (Step 16)
#
# **Runtime:** CPU | **Internet:** Off OK | **Time:** ~2-5 min |
# **GPU:** Not needed
#
# Quantifies the manuscript-grade observation D4 in `discussion_notes.md`:
# "FMCIB embeddings are near-degenerate on PET data" (Sw ≈ Sb ≈ 0.999 on T9
# test-retest, suggesting most embedding dimensions carry vanishing variance
# across PET inputs).
#
# **Method:** for each task's FMCIB v3 embeddings, compute per-dimension
# variance across the patient cohort. Report:
# - Histogram of log-variance per dimension
# - Fraction of dimensions with variance below thresholds (1e-6, 1e-3, 1e-1)
# - Top-10 most-active and least-active dimensions
# - Cross-task comparison (T4/T6/T7/T8/T9)
# - Same diagnostic for CT-FM (also fully auditable, different architecture)
#   as a within-FM contrast — does saturation affect FMCIB more than CT-FM?
#
# **Datasets to attach:** all five `pet-fm-bench-tX-embeddings-v3` (T4, T6,
# T7, T8, T9 — note T9 patches are v1 by design but embeddings are v3-named
# per D6b).
#
# **Output:**
# - `fm_saturation_summary.csv` — per (fm, task): n_dims, n_vanishing,
#   mean/median variance, max variance, fraction of dims with σ² < 1e-3
# - `fm_saturation_per_dim.parquet` — per (fm, task, dim): variance, rank
# - `fm_saturation_metadata.json` — config + headline numbers

# %% [markdown]
# ## 1. Setup

# %%
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

OUT_DIR = Path("/kaggle/working")
OUT_DIR.mkdir(parents=True, exist_ok=True)

freeze_timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
print(f"Freeze timestamp (UTC): {freeze_timestamp}")

VANISHING_THRESHOLDS = [1e-6, 1e-3, 1e-1]


# %% [markdown]
# ## 2. Locate v3 embedding datasets per task

# %%
def find_embed_dir(task):
    v3 = list(Path("/kaggle/input").rglob(f"pet-fm-bench-{task}-embeddings-v3"))
    if not v3:
        return None
    embed_dir = v3[0]
    nested = embed_dir / "embeddings"
    return nested if nested.exists() else embed_dir


def get_patient_emb(df, pid):
    """Pool-layer / volume-view extraction matching probe_analysis.py v4."""
    id_col = "patient_id" if "patient_id" in df.columns else "subject_id"
    sub = df[df[id_col] == pid]
    if "view" in df.columns and "volume" in df["view"].values:
        sub = sub[sub["view"] == "volume"]
    if "layer" in df.columns and "pool" in sub["layer"].values:
        sub = sub[sub["layer"] == "pool"]
    if len(sub) == 0:
        return None
    dim_cols = [c for c in df.columns if c.startswith("d")]
    return sub[dim_cols].values[0]


# %% [markdown]
# ## 3. Compute per-dimension variance for FMCIB + CT-FM × all 5 tasks

# %%
summary_rows = []
per_dim_rows = []

for task in ["t4", "t6", "t7", "t8", "t9"]:
    embed_dir = find_embed_dir(task)
    if embed_dir is None:
        print(f"  ✗ {task}: no v3 embeddings dataset attached, skipping")
        continue
    print(f"\n=== {task.upper()} ===")

    for fm in ["fmcib", "ct_fm"]:
        parq = embed_dir / f"{fm}.parquet"
        if not parq.exists():
            print(f"  {fm}: parquet not found, skipping")
            continue
        df = pd.read_parquet(parq)
        id_col = "patient_id" if "patient_id" in df.columns else "subject_id"

        # Build (n_patients, n_dim) array
        X_list = []
        for pid in sorted(df[id_col].unique()):
            emb = get_patient_emb(df, pid)
            if emb is not None and not np.isnan(emb).any():
                X_list.append(emb)
        if len(X_list) < 5:
            print(f"  {fm}: insufficient patient embeddings ({len(X_list)}), skipping")
            continue
        X = np.array(X_list)
        n_patients, n_dim = X.shape

        # Per-dimension variance
        dim_var = X.var(axis=0)
        dim_mean_abs = np.abs(X).mean(axis=0)

        # Vanishing-fraction at thresholds
        frac_vanishing = {f"frac_var_lt_{t:.0e}": float((dim_var < t).mean())
                          for t in VANISHING_THRESHOLDS}

        # Top-10 most-active and least-active dimensions
        order = np.argsort(dim_var)
        bottom10 = order[:10].tolist()
        top10 = order[-10:][::-1].tolist()

        summary_rows.append({
            "fm": fm,
            "task": task,
            "n_patients": int(n_patients),
            "n_dim": int(n_dim),
            "var_mean": float(dim_var.mean()),
            "var_median": float(np.median(dim_var)),
            "var_min": float(dim_var.min()),
            "var_max": float(dim_var.max()),
            "var_p99": float(np.percentile(dim_var, 99)),
            **frac_vanishing,
            "top10_dim_indices": str(top10),
            "bottom10_dim_indices": str(bottom10),
        })

        # Per-dim records (sample only top-100 + bottom-100 to keep the parquet small)
        keep_idx = np.concatenate([order[:100], order[-100:]])
        for d in keep_idx:
            per_dim_rows.append({
                "fm": fm,
                "task": task,
                "dim": int(d),
                "variance": float(dim_var[d]),
                "mean_abs": float(dim_mean_abs[d]),
                "rank_low": int(np.where(order == d)[0][0]) + 1,  # 1 = lowest variance
            })

        print(f"  {fm}: n_patients={n_patients}, n_dim={n_dim}, "
              f"var(mean)={dim_var.mean():.4g}, var(p99)={np.percentile(dim_var,99):.4g}, "
              f"frac<1e-3={frac_vanishing['frac_var_lt_1e-03']:.1%}")


# %% [markdown]
# ## 4. Save

# %%
summary_df = pd.DataFrame(summary_rows)
per_dim_df = pd.DataFrame(per_dim_rows)

summary_path = OUT_DIR / "fm_saturation_summary.csv"
per_dim_path = OUT_DIR / "fm_saturation_per_dim.parquet"
meta_path = OUT_DIR / "fm_saturation_metadata.json"

summary_df.to_csv(summary_path, index=False)
per_dim_df.to_parquet(per_dim_path, index=False)

# Headline interpretation
print("\n" + "=" * 60)
print("HEADLINE — fraction of dimensions with variance < 1e-3")
print("=" * 60)
piv = summary_df.pivot(index="fm", columns="task",
                       values="frac_var_lt_1e-03")
print(piv.round(3).to_string())

print("\n" + "=" * 60)
print("HEADLINE — variance ratio max/median (high = healthy spread)")
print("=" * 60)
summary_df["var_ratio_max_median"] = (summary_df["var_max"]
                                       / summary_df["var_median"])
piv2 = summary_df.pivot(index="fm", columns="task",
                        values="var_ratio_max_median")
print(piv2.round(1).to_string())

with open(meta_path, "w") as f:
    json.dump({
        "stage": "Step 16 — FMCIB / CT-FM PET saturation diagnostic",
        "freeze_timestamp_utc": freeze_timestamp,
        "context": (
            "D4 in manuscript/discussion_notes.md observed Sw ≈ Sb ≈ 0.999 "
            "for FMCIB on T9, suggesting most of FMCIB's 4096 embedding "
            "dimensions carry vanishing variance on PET inputs. This "
            "diagnostic quantifies the fraction of effectively-saturated "
            "dimensions and contrasts FMCIB (4096-d cancer-lesion CT pretrained) "
            "with CT-FM (512-d general CT pretrained) on the same cohorts."
        ),
        "vanishing_thresholds": VANISHING_THRESHOLDS,
        "summary": summary_df.to_dict(orient="records"),
    }, f, indent=2)

print(f"\nWrote: {summary_path}")
print(f"Wrote: {per_dim_path}")
print(f"Wrote: {meta_path}")
