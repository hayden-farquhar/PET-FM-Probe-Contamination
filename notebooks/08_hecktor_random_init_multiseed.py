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
# # HECKTOR Multi-Seed Random-Init Baseline (Amendment A3 — T2 + T3 per-patch)
#
# **PET-FM-Bench** | DOI: [10.17605/OSF.IO/DQ2JA](https://doi.org/10.17605/OSF.IO/DQ2JA)
#
# **Runtime:** GPU T4 | **Internet:** On | **Time:** ~10-20 min
#
# **Input:** `pet-fm-bench-hecktor-patches-v3` (Kaggle dataset, HECKTOR 2025 HN patches).
# **Output:** 10 per-seed parquets `random_init_seed_{0..9}.parquet`, to be uploaded
# as Kaggle dataset `pet-fm-bench-hecktor-randominit-multiseed-v3`.
#
# This is the HECKTOR analogue of `08_t1_random_init_multiseed.py` and
# `08_t5_random_init_multiseed.py` — same architecture (DINOv2-base config, ViT-B/14,
# 768-d, cls-token, 3 MIP views per patch), different cohort (HECKTOR head-and-neck
# cancer instead of AutoPET FDG/PSMA whole-body).
#
# **Why HECKTOR runs faster than T1 + T5**: 3,708 patches vs T1's 10,092 / T5's 9,864.
# 10 seeds × ~11k forwards (3,708 × 3 MIPs) takes ~10-20 min on T4 vs T1's 127 min /
# T5's projected 80-100 min. HN-localised disease + sparser per-patient lesion count
# (median 2 vs T1's 4-5).
#
# **Closes A3** for T2 + T3 once published. After this run, all 9 active tasks
# (T1/T2/T3/T4/T5/T6/T7/T8/T9) will have N=10 random_init seeds available for
# `aggregate_random_init_seeds()` in `probe_analysis.py` v6.

# %% [markdown]
# ## 1. Install dependencies

# %%
# !pip install -q --upgrade transformers

import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# %% [markdown]
# ## 2. Locate input data
#
# Same 3-layout candidate detection as `hecktor_02_embeddings.py` and the analogous
# T1/T5 multi-seed scripts (handles Kaggle CLI auto-extraction of .tar.gz uploads).

# %%
V3_CANDIDATES = list(Path("/kaggle/input").rglob("pet-fm-bench-hecktor-patches-v3"))
INPUT_CANDIDATES = V3_CANDIDATES or list(Path("/kaggle/input").glob("pet-fm-bench-hecktor*"))
if not INPUT_CANDIDATES:
    INPUT_CANDIDATES = list(Path("/kaggle/input").iterdir())

print("Available inputs:")
for p in INPUT_CANDIDATES:
    print(f"  {p.name}")

INPUT_DIR = INPUT_CANDIDATES[0] if INPUT_CANDIDATES else Path("/kaggle/input/pet-fm-bench-hecktor-patches-v3")

_PATCHES_CANDIDATES = [
    INPUT_DIR / "hecktor_v3_patches" / "hecktor_v3_patches" / "patches",  # double nest (Kaggle auto-extract)
    INPUT_DIR / "hecktor_v3_patches" / "patches",                          # single nest
    INPUT_DIR / "patches",                                                  # direct directory upload
]
_PATCHES_DIR = next((c for c in _PATCHES_CANDIDATES if c.is_dir()), None)
PRE_EXTRACTED = _PATCHES_DIR is not None
TARBALL = next(INPUT_DIR.rglob("hecktor_v3_patches.tar.gz"), None)

if PRE_EXTRACTED:
    PATCH_DIR = _PATCHES_DIR.parent
    manifest_path = (PATCH_DIR / "manifest.parquet"
                     if (PATCH_DIR / "manifest.parquet").exists()
                     else INPUT_DIR / "manifest.parquet")
    print(f"\nKaggle pre-extracted layout. PATCH_DIR = {PATCH_DIR}")
elif TARBALL is not None:
    EXTRACT_DIR = Path("/tmp/hecktor_v3_extracted")
    if not (EXTRACT_DIR / "hecktor_v3_patches" / "patches").exists():
        print(f"\nExtracting {TARBALL.name} ({TARBALL.stat().st_size/1e9:.2f} GB) → {EXTRACT_DIR}/")
        EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
        import tarfile
        with tarfile.open(TARBALL, "r:gz") as tar:
            tar.extractall(EXTRACT_DIR)
        print("Extraction complete.")
    PATCH_DIR = EXTRACT_DIR / "hecktor_v3_patches"
    manifest_path = PATCH_DIR / "manifest.parquet"
else:
    manifest_path = list(INPUT_DIR.rglob("manifest.parquet"))[0]
    PATCH_DIR = manifest_path.parent

manifest = pd.read_parquet(manifest_path)

def _has_patch_file(r):
    p = PATCH_DIR / "patches" / r["patient_id"] / f"{r['patch_id']}.npz"
    return p.exists()

mask = manifest.apply(_has_patch_file, axis=1)
manifest = manifest[mask].reset_index(drop=True)
print(f"\nManifest (filtered): {len(manifest)} patches across {manifest['patient_id'].nunique()} patients")
print(f"  T2 cohort (task1_patient): "
      f"{manifest[manifest['task1_patient']]['patient_id'].nunique()} patients, "
      f"{(manifest['task1_patient']).sum()} patches")
print(f"  T3 cohort (task2_patient): "
      f"{manifest[manifest['task2_patient']]['patient_id'].nunique()} patients, "
      f"{(manifest['task2_patient']).sum()} patches")

EMBED_DIR = Path("/kaggle/working/embeddings")
EMBED_DIR.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## 3. Helpers (identical to `hecktor_02_embeddings.py` §4)

# %%
def load_patch(patient_id, patch_id):
    path = PATCH_DIR / "patches" / patient_id / f"{patch_id}.npz"
    data = np.load(path)
    return {
        "mip_coronal":  data["mip_coronal"].astype(np.float32),
        "mip_axial":    data["mip_axial"].astype(np.float32),
        "mip_sagittal": data["mip_sagittal"].astype(np.float32),
    }


def mip_to_rgb_tensor(mip_2d):
    mip_2d = np.nan_to_num(mip_2d, nan=0.0, posinf=0.0, neginf=0.0)
    vmin, vmax = float(mip_2d.min()), float(mip_2d.max())
    if vmax > vmin:
        normed = (mip_2d - vmin) / (vmax - vmin)
    else:
        normed = np.zeros_like(mip_2d)
    stacked = np.stack([normed, normed, normed])
    return torch.tensor(stacked, dtype=torch.float32).unsqueeze(0)


# %% [markdown]
# ## 4. Multi-seed runner

# %%
def run_random_init_seed(seed, manifest):
    """Initialise a fresh ViT-B/14 from `torch.manual_seed(seed)` and produce
    per-patch × per-view cls embeddings. Returns a list of dict rows."""
    from transformers import AutoConfig, AutoModel

    torch.manual_seed(seed)
    np.random.seed(seed)
    config = AutoConfig.from_pretrained("facebook/dinov2-base")
    model = AutoModel.from_config(config).to(DEVICE).eval()

    rows = []
    for _, r in tqdm(manifest.iterrows(), total=len(manifest), desc=f"seed={seed}"):
        npz = load_patch(r["patient_id"], r["patch_id"])
        for view_name, view_arr in [("coronal", npz["mip_coronal"]),
                                     ("axial",   npz["mip_axial"]),
                                     ("sagittal", npz["mip_sagittal"])]:
            inp = mip_to_rgb_tensor(view_arr).to(DEVICE)
            with torch.no_grad():
                out = model(inp)
            emb = out.last_hidden_state[0, 0].cpu().numpy()
            rows.append({"patient_id": r["patient_id"], "patch_id": r["patch_id"],
                         "view": view_name, "layer": "cls",
                         "seed": int(seed), "embedding": emb})
    del model
    torch.cuda.empty_cache()
    return rows


# %% [markdown]
# ## 5. Run 10 seeds (idempotent — skips existing parquets)

# %%
N_SEEDS = 10
SEEDS = list(range(N_SEEDS))

for seed in SEEDS:
    out_path = EMBED_DIR / f"random_init_seed_{seed}.parquet"
    if out_path.exists():
        print(f"\nSkipping seed {seed}: {out_path.name} already exists "
              f"({out_path.stat().st_size/1e6:.1f} MB)")
        continue

    print(f"\n{'='*60}")
    print(f"  seed {seed}/{N_SEEDS-1}")
    print(f"{'='*60}")

    rows = run_random_init_seed(seed, manifest)
    if not rows:
        print(f"  No embeddings — skipping seed {seed}")
        continue

    embed_dim = len(rows[0]["embedding"])
    records = []
    for row in rows:
        rec = {"patient_id": row["patient_id"],
               "patch_id":   row["patch_id"],
               "view":       row["view"],
               "layer":      row["layer"],
               "seed":       row["seed"]}
        for j in range(len(row["embedding"])):
            rec[f"d{j:04d}"] = float(row["embedding"][j])
        records.append(rec)

    df = pd.DataFrame(records)
    df.to_parquet(out_path, index=False)
    size_mb = out_path.stat().st_size / 1e6
    print(f"  Saved: {out_path.name} ({size_mb:.1f} MB, {len(df)} rows, {embed_dim}-dim)")
    torch.cuda.empty_cache()


# %% [markdown]
# ## 6. Summary

# %%
print("\n" + "=" * 60)
print("HECKTOR MULTI-SEED RANDOM-INIT EXTRACTION COMPLETE")
print("=" * 60)

total_mb = 0
for f in sorted(EMBED_DIR.glob("random_init_seed_*.parquet")):
    df = pd.read_parquet(f)
    mb = f.stat().st_size / 1e6
    total_mb += mb
    dim = sum(1 for c in df.columns if c.startswith("d"))
    print(f"\n{f.stem}:")
    print(f"  patients: {df['patient_id'].nunique()}")
    print(f"  patches:  {df['patch_id'].nunique()}")
    print(f"  views:    {sorted(df['view'].unique().tolist())}")
    print(f"  seed:     {df['seed'].unique()[0]}")
    print(f"  dim:      {dim}")
    print(f"  size:     {mb:.1f} MB")

print(f"\nTotal output: {total_mb:.1f} MB across {len(SEEDS)} seeds")

# %% [markdown]
# ## 7. Save
#
# **"Save & Run All"** on GPU T4. Save output as Kaggle Dataset:
# `pet-fm-bench-hecktor-randominit-multiseed-v3` (mirrors T1 / T4-T9 / T5 multi-seed naming).
#
# After this publishes, A3 (multi-seed random_init for ALL active tasks) is satisfied
# for the entire 9-task universe (T1/T2/T3/T4/T5/T6/T7/T8/T9). The formal Phase 5
# probe run via `probe_analysis.py` v6 will use `aggregate_random_init_seeds()` to
# produce median + IQR across the 10 seeds for each task.
