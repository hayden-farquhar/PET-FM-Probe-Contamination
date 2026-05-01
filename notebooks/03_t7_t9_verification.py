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
# # PET-FM-Bench: T7 + T9 Embedding Verification
#
# **Runtime:** CPU | **Internet:** Off OK | **Time:** <1 min | **GPU:** Not needed
#
# Verifies that `pet-fm-bench-t7-embeddings-v3` and `pet-fm-bench-t9-embeddings-v3`
# were extracted correctly:
#
# 1. Schema check — required columns (`patient_id`/`subject_id`, scan-level key, view, layer) present
# 2. Row count check — actual rows match `n_scans × n_views × n_layers`
# 3. Unique (id, scan/session) group count — matches expected manifest size
# 4. Per-FM presence — flags any missing models
#
# **Datasets to attach (Add Data → search by name):**
# - `pet-fm-bench-t7-embeddings-v3`
# - `pet-fm-bench-t9-embeddings-v3`
#
# **Decision:** if both `PASS`, the GPU phase is complete and probe analysis
# can run. If any `FAIL`, the message tells you which task / FM / metric is off.

# %% [markdown]
# ## 1. Setup

# %%
from pathlib import Path

import pandas as pd

EXPECTED = {
    "t7": {
        "id_col": "patient_id",
        "session_col": "scan_id",
        "n_groups": 430,                  # 230 patients × ~1.87 scans/patient
        "n_subjects": 230,
        "row_counts": {                   # rows per parquet (None = skip strict count)
            "dinov2": 5160,               # 4 layers × 3 views × 430 scans
            "rad_dino": 5160,
            "biomedclip": 1290,           # 1 layer × 3 views × 430 scans
            "random_init": 1290,
            "fmcib": 430,                 # T7 3D FMs pooled to one row per scan
            "ct_fm": 430,
        },
    },
    "t9": {
        "id_col": "subject_id",
        "session_col": "session",
        "n_groups": 96,                   # 48 subjects × 2 sessions
        "n_subjects": 48,
        "row_counts": {
            "dinov2": 1152,               # 4 × 3 × 96
            "rad_dino": 1152,
            "biomedclip": 288,            # 1 × 3 × 96
            "random_init": 288,
            # T9 3D FMs save dual schema (1 pool + N per-patch rows per scan).
            # Vienna QUADRA has uniform 45 patches/session → 96 × (1+45) = 4416.
            "fmcib": 4416,
            "ct_fm": 4416,
        },
    },
}


# %% [markdown]
# ## 2. Per-task verification

# %%
def check_task(task, cfg):
    print(f"\n{'='*60}")
    print(f"  {task.upper()}")
    print(f"{'='*60}")

    # Find the v3 dataset
    candidates = list(Path("/kaggle/input").rglob(f"pet-fm-bench-{task}-embeddings-v3"))
    if not candidates:
        print(f"  ✗ FAIL: pet-fm-bench-{task}-embeddings-v3 not attached")
        return False
    emb_dir = candidates[0]
    print(f"  Dataset: {emb_dir}")

    # Find parquets — may be inside an /embeddings/ subdirectory
    parquets = [p for p in emb_dir.rglob("*.parquet")
                if not p.stem.endswith("labels")]
    print(f"  Parquets found: {sorted(p.stem for p in parquets)}")

    expected_fms = set(cfg["row_counts"].keys())
    found_fms = {p.stem for p in parquets}
    missing = expected_fms - found_fms
    if missing:
        print(f"  ⚠ Missing FMs: {sorted(missing)} (Merlin known to fail to install)")

    all_pass = True
    for parquet in sorted(parquets):
        fm = parquet.stem
        df = pd.read_parquet(parquet)
        cols = list(df.columns[:6])
        n_rows = len(df)
        n_subj = df[cfg["id_col"]].nunique()
        has_session = cfg["session_col"] in df.columns

        if has_session:
            n_groups = df.groupby([cfg["id_col"], cfg["session_col"]]).ngroups
        else:
            n_groups = None

        expected_rows = cfg["row_counts"].get(fm)
        rows_ok = expected_rows is None or n_rows == expected_rows
        groups_ok = has_session and n_groups == cfg["n_groups"]
        subj_ok = n_subj == cfg["n_subjects"]

        per_fm_pass = has_session and groups_ok and rows_ok and subj_ok
        if not per_fm_pass:
            all_pass = False
        marker = "✓" if per_fm_pass else "✗"

        print(f"\n  [{fm}] {marker}")
        print(f"    Rows: {n_rows} (expected {expected_rows})")
        print(f"    Subjects: {n_subj} (expected {cfg['n_subjects']})")
        if has_session:
            print(f"    Unique ({cfg['id_col']}, {cfg['session_col']}) groups: "
                  f"{n_groups} (expected {cfg['n_groups']})")
        else:
            print(f"    ✗ Missing '{cfg['session_col']}' column — schema bug")
        print(f"    Sample columns: {cols}")

    return all_pass


# %% [markdown]
# ## 3. Run all checks

# %%
results = {task: check_task(task, cfg) for task, cfg in EXPECTED.items()}

# %% [markdown]
# ## 4. Final summary

# %%
print(f"\n{'='*60}")
print(f"FINAL VERDICT")
print(f"{'='*60}")
print(f"{sum(results.values())}/{len(results)} tasks passed\n")
for task, ok in results.items():
    label = "PASS" if ok else "FAIL"
    print(f"  {task.upper()}: {label}")

if all(results.values()):
    print("\n✓ All embedding datasets verified — probe analysis can proceed.")
    print("  Next step: run probe_analysis.py with all 5 v3 embedding datasets attached.")
else:
    failed = [t for t, ok in results.items() if not ok]
    print(f"\n✗ {len(failed)} task(s) failed verification: {failed}")
    print("  Inspect the per-FM blocks above to identify the issue.")
    print("  Common causes:")
    print("    - Schema bug (missing scan_id/session column) → re-run embeddings with patched notebook")
    print("    - Row count mismatch → some scans silently failed; check log of last embedding run")
    print("    - Subject count mismatch → file-existence filter dropped patients; expected for T6, surprising for T7/T9")

# %% [markdown]
# ## 5. Done
#
# This notebook produces no saved outputs — it's a one-shot verification.
# No need to "Save & Run All" or publish as a Kaggle Dataset; just read the
# verdict above.
#
# **If both T7 and T9 PASS:** all 5 v3 embedding datasets exist and validate.
# Move to probe analysis with `probe_analysis.py`.
#
# **If either FAILs:** address the specific issue (per the per-FM block diagnostics)
# and re-run that task's embedding notebook before proceeding.
