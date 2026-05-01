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
# # T7 scan_id reconstruction (no GPU re-run needed)
#
# **Runtime:** CPU | **Internet:** Off OK | **Time:** ~5 min | **GPU:** Not needed
#
# `pet-fm-bench-t7-embeddings-v3` was extracted with an older version of the
# embedding notebook that didn't include `scan_id` in the saved rows. The row
# counts are correct (1290 / 5160 / 430 rows per parquet — exactly matching
# manifest × view × layer math), proving every scan produced every expected
# row in deterministic order.
#
# This notebook **reconstructs `scan_id`** by matching row indices back to the
# filtered manifest, with no GPU re-run required. Output: replacement parquets
# with the `scan_id` column added.
#
# **Datasets to attach (Add Data → search by name):**
# - `pet-fm-bench-t7-patches-v3` (for the manifest)
# - `pet-fm-bench-t7-embeddings-v3` (the parquets that need fixing)
#
# **Output:** new parquets in `/kaggle/working/embeddings/` with scan_id added.
# Save as **`pet-fm-bench-t7-embeddings-v3`** (overwrite/new version of the
# existing dataset).

# %% [markdown]
# ## 1. Setup

# %%
from pathlib import Path

import numpy as np
import pandas as pd

OUT_DIR = Path("/kaggle/working/embeddings")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# How many rows per scan each FM produces. Must match t7_02_embeddings.py logic.
ROWS_PER_SCAN = {
    "dinov2": 4 * 3,        # 4 layers × 3 views
    "rad_dino": 4 * 3,
    "biomedclip": 1 * 3,    # 1 layer × 3 views
    "random_init": 1 * 3,
    "fmcib": 1,             # pooled to one row per scan
    "ct_fm": 1,
}

# %% [markdown]
# ## 2. Load and filter the manifest
#
# Apply the same file-existence filter as `t7_02_embeddings.py` to recover the
# exact subset of scans that produced embeddings (430 of 464 manifest entries).

# %%
patches_root = next(Path("/kaggle/input").rglob("pet-fm-bench-t7-patches-v3"))
manifest_path = next(patches_root.rglob("manifest.csv"))
PATCH_DIR = manifest_path.parent
print(f"Patches root: {PATCH_DIR}")

manifest = pd.read_csv(manifest_path)
print(f"Manifest (raw): {len(manifest)} scans")

def has_files(r):
    scan_id = f"study_{r['study_index']}"
    mips = PATCH_DIR / "mips_2d" / r["patient_id"] / scan_id / "mips.npz"
    pat3 = PATCH_DIR / "patches_3d" / r["patient_id"] / scan_id / "patches.npz"
    return mips.exists() and pat3.exists()

mask = manifest.apply(has_files, axis=1)
manifest_f = manifest[mask].reset_index(drop=True)
print(f"Manifest (filtered): {len(manifest_f)} scans from {manifest_f['patient_id'].nunique()} patients")
assert len(manifest_f) == 430, (
    f"Expected 430 filtered scans, got {len(manifest_f)}. "
    "Reconstruction will be unreliable — re-run T7 embeddings with the patched "
    "notebook instead."
)

# Ordered list of scan_ids matching what t7_02 would have iterated through
ordered_scan_ids = [f"study_{si}" for si in manifest_f["study_index"]]
ordered_patient_ids = manifest_f["patient_id"].tolist()
print(f"First 5 scans in iteration order: {list(zip(ordered_patient_ids[:5], ordered_scan_ids[:5]))}")

# %% [markdown]
# ## 3. Load embeddings and reconstruct scan_id per row

# %%
emb_root = next(Path("/kaggle/input").rglob("pet-fm-bench-t7-embeddings-v3"))
print(f"Embeddings root: {emb_root}")
parquets = [p for p in emb_root.rglob("*.parquet") if not p.stem.endswith("labels")]
print(f"Parquets: {sorted(p.stem for p in parquets)}")

for parquet in sorted(parquets):
    fm = parquet.stem
    rps = ROWS_PER_SCAN.get(fm)
    if rps is None:
        print(f"\n[{fm}] Unknown FM — skipping")
        continue

    df = pd.read_parquet(parquet)
    n_rows = len(df)
    expected_rows = rps * len(manifest_f)

    print(f"\n[{fm}]")
    print(f"  Rows: {n_rows} (expected {expected_rows})")
    if n_rows != expected_rows:
        print(f"  ✗ MISMATCH — reconstruction unreliable for this FM")
        print(f"    Either some rows were dropped or rows-per-scan differs from expected.")
        print(f"    Skip and re-run this FM via the patched notebook.")
        continue

    # Assign each row a scan_index = row_position // rows_per_scan
    scan_indices = np.arange(n_rows) // rps
    df["scan_id"] = [ordered_scan_ids[i] for i in scan_indices]

    # Sanity: verify patient_id in df matches manifest at the inferred index
    df_patient_ids = df["patient_id"].tolist()
    expected_patient_ids = [ordered_patient_ids[i] for i in scan_indices]
    mismatches = sum(a != b for a, b in zip(df_patient_ids, expected_patient_ids))
    if mismatches > 0:
        print(f"  ✗ {mismatches} rows have patient_id mismatching the inferred scan position")
        print(f"    Order assumption violated — reconstruction unreliable for this FM")
        continue

    # Reorder columns: put scan_id right after patient_id for readability
    cols = list(df.columns)
    cols.remove("scan_id")
    pid_idx = cols.index("patient_id")
    cols.insert(pid_idx + 1, "scan_id")
    df = df[cols]

    # Save
    out_path = OUT_DIR / f"{fm}.parquet"
    df.to_parquet(out_path, index=False)
    n_groups = df.groupby(["patient_id", "scan_id"]).ngroups
    print(f"  ✓ Reconstructed: {n_groups} unique (patient, scan) groups")
    print(f"    Saved: {out_path.name} ({out_path.stat().st_size/1e6:.1f} MB)")

# %% [markdown]
# ## 4. Copy labels CSV through (no changes needed)

# %%
import shutil
for label_file in emb_root.rglob("*labels*"):
    dest = OUT_DIR / label_file.name
    shutil.copy(label_file, dest)
    print(f"Copied: {label_file.name} → {dest}")

# %% [markdown]
# ## 5. Final summary

# %%
print(f"\n{'='*60}")
print(f"RECONSTRUCTION COMPLETE")
print(f"{'='*60}")
total_mb = 0
for f in sorted(OUT_DIR.glob("*")):
    if f.is_file():
        mb = f.stat().st_size / 1e6
        total_mb += mb
        print(f"  {f.name} ({mb:.1f} MB)")
print(f"\nTotal: {total_mb:.1f} MB")

# Final verification: load each parquet and confirm scan_id is present + 430 groups
print(f"\nVerification:")
for parquet in sorted(OUT_DIR.glob("*.parquet")):
    df = pd.read_parquet(parquet)
    n_groups = df.groupby(["patient_id", "scan_id"]).ngroups if "scan_id" in df.columns else None
    has_scan = "scan_id" in df.columns
    status = "✓" if has_scan and n_groups == 430 else "✗"
    print(f"  {status} {parquet.stem}: {len(df)} rows, "
          f"{'scan_id ✓' if has_scan else 'scan_id MISSING'}, "
          f"{n_groups} unique (patient, scan) groups")

# %% [markdown]
# ## 6. Save & publish
#
# Commit with **"Save & Run All"** to persist these reconstructed parquets.
# Then go to the Output tab → publish as a new version of
# `pet-fm-bench-t7-embeddings-v3` (overwrites the broken version).
#
# The probe analysis can then run against this corrected dataset without any
# GPU re-extraction.
