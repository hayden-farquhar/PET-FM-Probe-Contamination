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
# # PET-FM-Bench: Pre-correction Diagnostic
#
# **Runtime:** CPU | **Internet:** Off OK | **GPU:** Not needed
#
# Single-shot diagnostic to run before deciding what to re-preprocess. Scans:
#
# 1. **Patches inf/NaN/saturation** per task — identifies SUV/float16 overflow
# 2. **Value-range distribution** — distinguishes Bq/mL (huge) from SUV (~0–50)
# 3. **T4 survival-label integrity** — confirms v2 fix populated time_to_death
# 4. **Embedding NaN cross-reference** — does FM dropout match patches diagnosis?
# 5. **Patch shape consistency** — sanity check
# 6. **Test-retest pair completeness** — T6 and T9
# 7. **Master summary** — SUV-fix scope decision + per-patient fix list
#
# **Datasets to attach (Add Data → search by name):**
# - `pet-fm-bench-t4-patches`, `t6-patches`, `t7-patches`, `t8-patches`, `t9-patches`
# - `pet-fm-bench-t4-embeddings`, `t6-embeddings`, `t7-embeddings`, `t8-embeddings`, `t9-embeddings`
#
# Outputs land in `/kaggle/working/diagnostic/` — save as Kaggle Dataset
# `pet-fm-bench-diagnostic` if you want them archived.

# %% [markdown]
# ## 1. Setup

# %%
import json
from pathlib import Path

import numpy as np
import pandas as pd

OUTPUT_DIR = Path("/kaggle/working/diagnostic")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TASKS = ["t4", "t6", "t7", "t8", "t9"]
FLOAT16_MAX = 65504.0
SATURATION_THRESHOLD = 60000.0  # near float16 max — likely overflow casualty


def find_dataset(name_pattern):
    """Locate a Kaggle dataset by partial name across known mount layouts."""
    seen = set()
    candidates = []
    for root in [Path("/kaggle/input"), Path("/kaggle/input/datasets")]:
        if not root.exists():
            continue
        for sub in root.rglob(f"*{name_pattern}*"):
            if sub.is_dir() and sub not in seen:
                seen.add(sub)
                candidates.append(sub)
    # Prefer ones containing patches_3d or parquet files
    for c in candidates:
        if list(c.rglob("patches_3d")) or list(c.glob("*.parquet")):
            return c
    return candidates[0] if candidates else None


print("Attached datasets:")
for task in TASKS:
    p = find_dataset(f"{task}-patches")
    e = find_dataset(f"{task}-embeddings")
    print(f"  [{task}] patches={'OK' if p else 'MISSING'}  embeddings={'OK' if e else 'MISSING'}")

# %% [markdown]
# ## 2. Per-task patches scan
#
# For each saved `patches.npz`, compute:
# - count of `inf`, `NaN`, and near-float16-max ("saturated") voxels
# - finite-value max, P99, P50 — distinguishes SUV (~0–50) vs Bq/mL (~10⁵)
# - patch tensor shape
#
# This is the load-bearing diagnostic: it tells us which tasks need the SUV fix.

# %%
per_patient_records = []

for task in TASKS:
    patches_root = find_dataset(f"{task}-patches")
    if patches_root is None:
        print(f"[{task}] patches dataset not attached — skipping")
        continue

    p3d_candidates = list(patches_root.rglob("patches_3d"))
    if not p3d_candidates:
        print(f"[{task}] no patches_3d/ found under {patches_root}")
        continue
    p3d = p3d_candidates[0]

    npz_files = sorted(p3d.rglob("patches.npz"))
    print(f"[{task}] scanning {len(npz_files)} patches.npz files in {p3d}")

    for npz_path in npz_files:
        rel = npz_path.relative_to(p3d)
        parts = rel.parts
        # Layouts:
        #   patches_3d/<pid>/patches.npz                (T4, T8 — one session per pid)
        #   patches_3d/<pid>/<session>/patches.npz       (T6, T7, T9)
        pid = parts[0]
        session = parts[1] if len(parts) == 3 else None

        try:
            data = np.load(npz_path)
            arr = data["patches"].astype(np.float32)  # promote from f16
        except Exception as e:
            per_patient_records.append({
                "task": task, "patient_id": pid, "session": session,
                "load_error": str(e),
            })
            continue

        n_total = arr.size
        n_inf = int(np.isinf(arr).sum())
        n_nan = int(np.isnan(arr).sum())
        n_sat = int((np.abs(arr) >= SATURATION_THRESHOLD).sum())
        frac_bad = (n_inf + n_nan) / n_total if n_total else 0.0

        finite = arr[np.isfinite(arr)]
        if finite.size:
            vmin = float(finite.min())
            vmax = float(finite.max())
            p50 = float(np.percentile(finite, 50))
            p99 = float(np.percentile(finite, 99))
        else:
            vmin = vmax = p50 = p99 = float("nan")

        per_patient_records.append({
            "task": task,
            "patient_id": pid,
            "session": session,
            "shape": str(arr.shape),
            "n_patches": int(arr.shape[0]) if arr.ndim >= 1 else 0,
            "n_total_voxels": n_total,
            "n_inf": n_inf,
            "n_nan": n_nan,
            "n_saturated": n_sat,
            "frac_inf_nan": round(frac_bad, 4),
            "vmin": vmin,
            "vmax": vmax,
            "p50": p50,
            "p99": p99,
        })

patient_df = pd.DataFrame(per_patient_records)
patient_df.to_csv(OUTPUT_DIR / "per_patient_diagnosis.csv", index=False)
print(f"\nScanned {len(patient_df)} patient-sessions across {patient_df['task'].nunique()} tasks")

# %% [markdown]
# ## 3. Per-task summary + SUV-fix diagnosis

# %%
def diagnose(task_df):
    """Decide whether a task needs SUV conversion fix."""
    if "vmax" not in task_df.columns or task_df["vmax"].dropna().empty:
        return "NO DATA"
    median_vmax = task_df["vmax"].median()
    n_affected = (task_df["frac_inf_nan"] > 0).sum()
    pct_affected = n_affected / len(task_df) * 100

    # Heuristics:
    # - SUV typically peaks around 5-30 (rarely >50). Vmax > 1000 → almost certainly Bq/mL.
    # - Any inf/NaN → float16 overflow happened during preprocess.
    if pct_affected > 5:
        return "SUV FIX NEEDED (overflow detected)"
    if median_vmax > 1000:
        return "SUV FIX NEEDED (Bq/mL values, no overflow yet but unsafe)"
    if median_vmax > 100:
        return "SUSPICIOUS — likely needs fix, manual inspection advised"
    return "Likely SUV-converted (clean)"


task_summaries = []
for task in TASKS:
    task_df = patient_df[patient_df["task"] == task]
    if task_df.empty:
        continue
    n_affected = int((task_df["frac_inf_nan"] > 0).sum())
    task_summaries.append({
        "task": task,
        "n_sessions": len(task_df),
        "n_patients": task_df["patient_id"].nunique(),
        "n_sessions_with_inf_nan": n_affected,
        "pct_affected": round(n_affected / len(task_df) * 100, 1),
        "median_vmax": round(task_df["vmax"].median(), 2),
        "max_vmax": round(task_df["vmax"].max(), 2),
        "median_p99": round(task_df["p99"].median(), 2),
        "diagnosis": diagnose(task_df),
    })

summary_df = pd.DataFrame(task_summaries)
summary_df.to_csv(OUTPUT_DIR / "task_diagnosis.csv", index=False)

print("\n" + "=" * 80)
print("SUV-FIX DIAGNOSIS BY TASK")
print("=" * 80)
print(summary_df.to_string(index=False))

# Per-patient list of who needs SUV fix
fix_list = patient_df[patient_df["frac_inf_nan"] > 0][
    ["task", "patient_id", "session", "frac_inf_nan", "vmax", "n_saturated"]
].sort_values(["task", "frac_inf_nan"], ascending=[True, False])
fix_list.to_csv(OUTPUT_DIR / "patients_needing_suv_fix.csv", index=False)
print(f"\n{len(fix_list)} patient-sessions flagged for SUV fix (full list saved)")

# %% [markdown]
# ## 4. T4 survival-label integrity
#
# v2 preprocessing was supposed to populate `time_to_death` for ALL patients
# (alive: censoring time = Last Known Alive − CT Date; dead: time-to-death).
# Pre-v2 only had time for dead patients, leading to n=4 valid in the previous
# probe run. This confirms the v2 fix actually landed.

# %%
t4_emb = find_dataset("t4-embeddings")
t4_label_check = {"status": "labels not found"}

if t4_emb:
    label_files = list(t4_emb.rglob("t4_labels.csv"))
    if label_files:
        labels = pd.read_csv(label_files[0])
        n_total = len(labels)
        n_alive = int((labels["event"] == 0).sum()) if "event" in labels.columns else None
        n_dead = int((labels["event"] == 1).sum()) if "event" in labels.columns else None
        n_with_time = int(labels["time_to_death"].notna().sum()) \
            if "time_to_death" in labels.columns else None
        n_alive_with_time = int(
            labels[labels["event"] == 0]["time_to_death"].notna().sum()
        ) if "event" in labels.columns and "time_to_death" in labels.columns else None
        n_dead_with_time = int(
            labels[labels["event"] == 1]["time_to_death"].notna().sum()
        ) if "event" in labels.columns and "time_to_death" in labels.columns else None

        # v2 fix is confirmed if alive patients have populated times
        v2_ok = (n_alive_with_time == n_alive) if n_alive is not None else False

        t4_label_check = {
            "status": "v2 OK" if v2_ok else "v2 NOT APPLIED — alive patients missing time",
            "n_total": n_total,
            "n_alive": n_alive,
            "n_dead": n_dead,
            "n_with_time_to_death": n_with_time,
            "n_alive_with_time": n_alive_with_time,
            "n_dead_with_time": n_dead_with_time,
            "columns": list(labels.columns),
        }

print("\n" + "=" * 80)
print("T4 SURVIVAL LABEL INTEGRITY")
print("=" * 80)
for k, v in t4_label_check.items():
    print(f"  {k}: {v}")

# %% [markdown]
# ## 5. Embedding-side NaN cross-reference
#
# If patches contain inf/NaN for K patients, FMCIB and CT-FM should produce NaN
# embeddings for ~K patients (the 3D path has no sanitisation). 2D FMs should
# show ~0% NaN regardless because `mip_to_rgb_tensor` sanitises before model.
#
# Mismatch between expected and actual would suggest a *separate* bug.

# %%
crossref_records = []

for task in TASKS:
    emb_root = find_dataset(f"{task}-embeddings")
    if emb_root is None:
        continue
    emb_dirs = [emb_root] + list(emb_root.rglob("embeddings"))
    emb_dir = next((d for d in emb_dirs if list(d.glob("*.parquet"))), None)
    if emb_dir is None:
        continue

    for parquet in sorted(emb_dir.glob("*.parquet")):
        fm_name = parquet.stem
        if fm_name.endswith("labels"):
            continue
        df_emb = pd.read_parquet(parquet)
        dim_cols = [c for c in df_emb.columns if c.startswith("d")]
        if not dim_cols:
            continue
        emb_arr = df_emb[dim_cols].values
        nan_row_mask = np.isnan(emb_arr).any(axis=1)
        id_col = "patient_id" if "patient_id" in df_emb.columns else "subject_id"
        n_unique_ids_with_nan = df_emb[nan_row_mask][id_col].nunique()
        crossref_records.append({
            "task": task,
            "fm": fm_name,
            "n_rows": len(df_emb),
            "n_nan_rows": int(nan_row_mask.sum()),
            "pct_nan_rows": round(nan_row_mask.sum() / len(df_emb) * 100, 1),
            "n_unique_ids_with_any_nan": int(n_unique_ids_with_nan),
            "embedding_dim": len(dim_cols),
        })

crossref_df = pd.DataFrame(crossref_records)
if not crossref_df.empty:
    crossref_df.to_csv(OUTPUT_DIR / "embeddings_nan_crossref.csv", index=False)
    print("\n" + "=" * 80)
    print("EMBEDDING NaN CROSS-REFERENCE")
    print("=" * 80)
    print(crossref_df.to_string(index=False))
else:
    print("\nNo embedding parquets found — skipping cross-reference")

# %% [markdown]
# ## 6. Patch shape consistency

# %%
shape_check = (
    patient_df.dropna(subset=["shape"])
    .groupby("task")["shape"]
    .agg(lambda s: dict(s.value_counts()))
    .to_dict()
)
print("\n" + "=" * 80)
print("PATCH SHAPE CONSISTENCY (counts by shape per task)")
print("=" * 80)
for task, shapes in shape_check.items():
    if len(shapes) == 1:
        shape, n = next(iter(shapes.items()))
        print(f"  [{task}] {n} sessions, all shape {shape}")
    else:
        print(f"  [{task}] HETEROGENEOUS — {len(shapes)} distinct shapes:")
        for s, n in shapes.items():
            print(f"      {s}: {n}")

# %% [markdown]
# ## 7. Test-retest pair completeness (T6 + T9)
#
# T6 expects ~16–20 retest patients with 2+ valid sessions; T9 expects 48
# subjects × 2 sessions. Confirms the test-retest probe will see the full
# pair count after embedding extraction.

# %%
def check_test_retest(task, expected_pairs=None):
    emb_root = find_dataset(f"{task}-embeddings")
    if emb_root is None:
        return f"[{task}] embeddings not attached"
    emb_dirs = [emb_root] + list(emb_root.rglob("embeddings"))
    emb_dir = next((d for d in emb_dirs if list(d.glob("*.parquet"))), None)
    if emb_dir is None:
        return f"[{task}] no parquets found"
    parquet = next(emb_dir.glob("*.parquet"), None)
    if parquet is None:
        return f"[{task}] no parquets"
    df_emb = pd.read_parquet(parquet)
    id_col = "patient_id" if "patient_id" in df_emb.columns else "subject_id"
    if "session" not in df_emb.columns:
        return f"[{task}] no session column"
    sessions_per_id = (
        df_emb[[id_col, "session"]]
        .drop_duplicates()
        .groupby(id_col)
        .size()
    )
    n_pairs = int((sessions_per_id >= 2).sum())
    n_solo = int((sessions_per_id == 1).sum())
    note = f" (expected ~{expected_pairs})" if expected_pairs else ""
    return f"[{task}] {n_pairs} test-retest pairs{note}, {n_solo} single-session"


print("\n" + "=" * 80)
print("TEST-RETEST PAIR COMPLETENESS")
print("=" * 80)
print(" ", check_test_retest("t6", expected_pairs=16))
print(" ", check_test_retest("t9", expected_pairs=48))

# %% [markdown]
# ## 8. Master summary — what to do next

# %%
print("\n" + "=" * 80)
print("MASTER SUMMARY")
print("=" * 80)
print(f"\nTotal patient-sessions scanned: {len(patient_df)}")
print(f"Total flagged for SUV fix: {len(fix_list)}")
print()
print("Per-task verdict:")
for _, row in summary_df.iterrows():
    marker = "✗" if "FIX NEEDED" in row["diagnosis"] or "SUSPICIOUS" in row["diagnosis"] else "✓"
    print(f"  {marker} {row['task']}: {row['diagnosis']} "
          f"(median vmax={row['median_vmax']}, {row['pct_affected']}% affected)")

# Overall scope decision
needs_fix = summary_df[summary_df["diagnosis"].str.contains("FIX|SUSPICIOUS")]
print(f"\nSUV fix scope: {sorted(needs_fix['task'].tolist())}")

# Save summary as JSON for downstream notebooks
final_summary = {
    "tasks_needing_suv_fix": sorted(needs_fix["task"].tolist()),
    "task_diagnoses": summary_df.to_dict(orient="records"),
    "t4_label_check": t4_label_check,
    "n_patient_sessions_scanned": len(patient_df),
    "n_patient_sessions_to_fix": len(fix_list),
}
with open(OUTPUT_DIR / "diagnostic_summary.json", "w") as f:
    json.dump(final_summary, f, indent=2, default=str)

# %% [markdown]
# ## 9. Output files

# %%
print("\nOutput files in /kaggle/working/diagnostic/:")
for f in sorted(OUTPUT_DIR.glob("*")):
    if f.is_file():
        kb = f.stat().st_size / 1e3
        print(f"  {f.name} ({kb:.1f} KB)")

# %% [markdown]
# ## 10. Done
#
# Save with **"Save & Run All"** if you want the artefacts archived. Otherwise
# the printed output above is enough to decide SUV-fix scope and you can close
# without committing.
#
# **Key files for the next step (SUV-fix notebook):**
# - `task_diagnosis.csv` — which tasks need the fix
# - `patients_needing_suv_fix.csv` — exact (task, patient, session) tuples to target
# - `diagnostic_summary.json` — machine-readable scope decision
