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
# # T5 Multi-Seed Random-Init Baseline (Amendment A3 — T5 per-patch)
#
# **PET-FM-Bench** | DOI: [10.17605/OSF.IO/DQ2JA](https://doi.org/10.17605/OSF.IO/DQ2JA)
#
# **Runtime:** GPU T4 | **Internet:** On | **Time:** ~25-40 min
#
# **Input:** `pet-fm-bench-t5-patches-v3` (Kaggle dataset, AutoPET-III PSMA patches).
# **Output:** 10 per-seed parquets `random_init_seed_{0..9}.parquet`, to be uploaded
# as Kaggle dataset `pet-fm-bench-t5-randominit-multiseed-v3`.
#
# This is the T5 analogue of `08_t1_random_init_multiseed.py` — same architecture
# (DINOv2-base config, ViT-B/14, 768-d, cls-token, 3 MIP views per patch),
# different cohort (T5 PSMA cancer instead of T1 FDG cancer/negative).
#
# **Why T5 needs more time than T1**: with multi-series patients (135/333 with
# ≥2 series), T5 has more patches per patient (~15k total expected vs T1's
# 10,092), so 10 seeds × ~45k forwards (15k × 3 MIPs) takes proportionally longer.

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
# Same 3-layout candidate detection as `t5_02_embeddings.py` and `t1_02_embeddings.py`
# (handles Kaggle CLI auto-extraction of .tar.gz uploads, which produces a doubly-
# nested directory structure).

# %%
V3_CANDIDATES = list(Path("/kaggle/input").rglob("pet-fm-bench-t5-patches-v3"))
INPUT_CANDIDATES = V3_CANDIDATES or list(Path("/kaggle/input").glob("pet-fm-bench-t5*"))
if not INPUT_CANDIDATES:
    INPUT_CANDIDATES = list(Path("/kaggle/input").iterdir())

print("Available inputs:")
for p in INPUT_CANDIDATES:
    print(f"  {p.name}")

INPUT_DIR = INPUT_CANDIDATES[0] if INPUT_CANDIDATES else Path("/kaggle/input/pet-fm-bench-t5-patches-v3")

_PATCHES_CANDIDATES = [
    INPUT_DIR / "t5_v3_patches" / "t5_v3_patches" / "patches",  # double nest (Kaggle auto-extract)
    INPUT_DIR / "t5_v3_patches" / "patches",                    # single nest
    INPUT_DIR / "patches",                                       # direct directory upload
]
_PATCHES_DIR = next((c for c in _PATCHES_CANDIDATES if c.is_dir()), None)
PRE_EXTRACTED = _PATCHES_DIR is not None
TARBALL = next(INPUT_DIR.rglob("t5_v3_patches.tar.gz"), None)

if PRE_EXTRACTED:
    PATCH_DIR = _PATCHES_DIR.parent
    manifest_path = (PATCH_DIR / "manifest.parquet"
                     if (PATCH_DIR / "manifest.parquet").exists()
                     else INPUT_DIR / "manifest.parquet")
    print(f"\nKaggle pre-extracted layout. PATCH_DIR = {PATCH_DIR}")
elif TARBALL is not None:
    EXTRACT_DIR = Path("/tmp/t5_v3_extracted")
    if not (EXTRACT_DIR / "t5_v3_patches" / "patches").exists():
        print(f"\nExtracting {TARBALL.name} ({TARBALL.stat().st_size/1e9:.2f} GB) → {EXTRACT_DIR}/")
        EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
        import tarfile
        with tarfile.open(TARBALL, "r:gz") as tar:
            tar.extractall(EXTRACT_DIR)
        print("Extraction complete.")
    PATCH_DIR = EXTRACT_DIR / "t5_v3_patches"
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
print(f"\nManifest (filtered): {len(manifest)} patches across {manifest['patient_id'].nunique()} patients, {manifest['series_uid'].nunique()} series")

EMBED_DIR = Path("/kaggle/working/embeddings")
EMBED_DIR.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## 3. Helpers (identical to `t5_02_embeddings.py` §4)

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
                         "series_uid": r.get("series_uid", ""),
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
               "series_uid": row["series_uid"],
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
print("T5 MULTI-SEED RANDOM-INIT EXTRACTION COMPLETE")
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
    print(f"  series:   {df['series_uid'].nunique()}")
    print(f"  views:    {sorted(df['view'].unique().tolist())}")
    print(f"  seed:     {df['seed'].unique()[0]}")
    print(f"  dim:      {dim}")
    print(f"  size:     {mb:.1f} MB")

print(f"\nTotal output: {total_mb:.1f} MB across {len(SEEDS)} seeds")

# %% [markdown]
# ## 7. Save
#
# **"Save & Run All"** on GPU T4. Save output as Kaggle Dataset:
# `pet-fm-bench-t5-randominit-multiseed-v3` (mirrors T1 + T4-T9 multi-seed naming).
#
# A3 will then be satisfied for T5 (T1 already done; the 5 originally-active
# tasks already had multi-seed). All 7 active tasks of this project will have multi-seed
# random_init available for `aggregate_random_init_seeds()` in `probe_analysis.py` v5.
