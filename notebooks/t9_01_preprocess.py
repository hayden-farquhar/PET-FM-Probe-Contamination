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
# # T9 Notebook 1/2: Download & Preprocess Vienna QUADRA
#
# **PET-FM-Bench** | DOI: [10.17605/OSF.IO/DQ2JA](https://doi.org/10.17605/OSF.IO/DQ2JA)
#
# **Runtime:** CPU (no GPU needed) | **Internet:** On | **Time:** ~45 min
#
# Downloads the Vienna QUADRA healthy-control test-retest dataset from Zenodo,
# preprocesses into patches and MIPs, and saves to notebook output (~7 GB).
# After committing, save the output as a Kaggle Dataset for use by the
# embedding extraction notebook.

# %% [markdown]
# ## 1. Setup

# %%
import os
import sys
import hashlib
import zipfile
import subprocess
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

!pip install -q SimpleITK>=2.3 tqdm requests

import SimpleITK as sitk
from tqdm import tqdm

# %% [markdown]
# ## 2. Configuration

# %%
# Raw data goes to /tmp/ (~70 GB available, not counted against 20 GB output limit)
# Preprocessed patches go to /kaggle/working/ (persisted on commit)
TMP_DIR = Path("/tmp/pet_fm_bench")
RAW_DIR = TMP_DIR / "raw"
PATCH_DIR = Path("/kaggle/working/t9_vienna_quadra")
OUTPUT_DIR = Path("/kaggle/working")

ZENODO_URL = "https://zenodo.org/records/16686025/files/QUADRA_HC.zip"

# Registration Section 5.1 parameters
PATCH_SIZE = (96, 96, 96)   # 3D FM input
SPACING = (2.0, 2.0, 2.0)  # Isotropic resampling

for d in [RAW_DIR, PATCH_DIR]:
    d.mkdir(parents=True, exist_ok=True)

print(f"Disk available: {os.statvfs('/kaggle/working').f_bavail * os.statvfs('/kaggle/working').f_frsize / 1e9:.1f} GB (output)")
print(f"Disk available: {os.statvfs('/tmp').f_bavail * os.statvfs('/tmp').f_frsize / 1e9:.1f} GB (tmp)")

# %% [markdown]
# ## 3. Download from Zenodo

# %%
zip_path = RAW_DIR / "QUADRA_HC.zip"

if not zip_path.exists():
    import requests

    print("Downloading QUADRA_HC.zip from Zenodo (~20 GB)...")
    response = requests.get(ZENODO_URL, stream=True)
    response.raise_for_status()
    total = int(response.headers.get("content-length", 0))

    with open(zip_path, "wb") as f, tqdm(total=total, unit="B", unit_scale=True) as pbar:
        for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
            f.write(chunk)
            pbar.update(len(chunk))

print(f"Downloaded: {zip_path.stat().st_size / 1e9:.1f} GB")

# SHA256 for reproducibility
sha256 = hashlib.sha256()
with open(zip_path, "rb") as f:
    for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
        sha256.update(chunk)
print(f"SHA256: {sha256.hexdigest()}")

# %% [markdown]
# ## 4. Extract

# %%
imaging_dir = RAW_DIR / "QUADRA_HC" / "Imaging Data"

if not imaging_dir.exists():
    print("Extracting...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(RAW_DIR)
    print("Done")

# Delete zip to free /tmp/ space
if zip_path.exists():
    zip_path.unlink()
    print("Zip removed")

subjects = sorted([d for d in imaging_dir.iterdir()
                   if d.is_dir() and d.name.startswith("QUADRA_HC_")])
print(f"Found {len(subjects)} subjects")

# %% [markdown]
# ## 5. Build manifest

# %%
rows = []
for subj_dir in subjects:
    for session in ["Test", "Retest"]:
        sess_dir = subj_dir / session
        if not sess_dir.exists():
            print(f"WARNING: missing {subj_dir.name}/{session}")
            continue
        rows.append({
            "subject_id": subj_dir.name,
            "session": session,
            "pet_path": str(next(sess_dir.glob("*_PT-SUV.nii.gz"), None)),
            "ct_path": str(next(sess_dir.glob("*_CT-AC.nii.gz"), None)),
        })

manifest = pd.DataFrame(rows)
print(f"Manifest: {len(manifest)} rows, {manifest['subject_id'].nunique()} subjects")
print(f"PET found: {manifest['pet_path'].notna().sum()}/{len(manifest)}")
manifest.to_csv(PATCH_DIR / "manifest.csv", index=False)

# %% [markdown]
# ## 6. Preprocessing functions

# %%
def resample_isotropic(img_sitk, spacing=(2.0, 2.0, 2.0)):
    orig_spacing = img_sitk.GetSpacing()
    orig_size = img_sitk.GetSize()
    new_size = [int(round(s * sp / t)) for s, sp, t in zip(orig_size, orig_spacing, spacing)]

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(img_sitk.GetDirection())
    resampler.SetOutputOrigin(img_sitk.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(0)
    resampler.SetInterpolator(sitk.sitkLinear)
    return resampler.Execute(img_sitk)


def extract_patches(vol, patch_size=(96, 96, 96)):
    """Non-overlapping 3D patches."""
    pz, py, px = patch_size
    patches, positions = [], []
    for z in range(0, max(1, vol.shape[0] - pz + 1), pz):
        for y in range(0, max(1, vol.shape[1] - py + 1), py):
            for x in range(0, max(1, vol.shape[2] - px + 1), px):
                p = vol[z:z+pz, y:y+py, x:x+px]
                if p.shape != (pz, py, px):
                    padded = np.zeros((pz, py, px), dtype=p.dtype)
                    padded[:p.shape[0], :p.shape[1], :p.shape[2]] = p
                    p = padded
                patches.append(p)
                positions.append((z, y, x))
    return np.stack(patches), positions


def resize_mip(arr_2d, size=224):
    """Resize a 2D array to (size, size)."""
    img = sitk.GetImageFromArray(arr_2d.astype(np.float32))
    resampler = sitk.ResampleImageFilter()
    resampler.SetSize((size, size))
    resampler.SetOutputSpacing((arr_2d.shape[1] / size, arr_2d.shape[0] / size))
    resampler.SetInterpolator(sitk.sitkLinear)
    return sitk.GetArrayFromImage(resampler.Execute(img))


# %% [markdown]
# ## 7. Process all sessions

# %%
log = []

for _, row in tqdm(manifest.iterrows(), total=len(manifest), desc="Processing"):
    subj = row["subject_id"]
    sess = row["session"]

    if row["pet_path"] == "None" or row["pet_path"] is None:
        log.append({"subject_id": subj, "session": sess, "status": "no_pet"})
        continue

    try:
        # Load and resample PET
        pet_sitk = sitk.ReadImage(row["pet_path"])
        pet_iso = resample_isotropic(pet_sitk, SPACING)
        pet = sitk.GetArrayFromImage(pet_iso).astype(np.float32)

        # 3D patches (for 3D FMs) — saved as float16
        patches, positions = extract_patches(pet, PATCH_SIZE)
        out_dir = PATCH_DIR / "patches_3d" / subj / sess
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_dir / "patches.npz",
                            patches=patches.astype(np.float16),
                            positions=np.array(positions))

        # 2D MIPs (for 2D FMs) — coronal, axial, sagittal at 224x224
        mip_dir = PATCH_DIR / "mips_2d" / subj / sess
        mip_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(mip_dir / "mips.npz",
                            coronal=resize_mip(pet.max(axis=1)).astype(np.float16),
                            axial=resize_mip(pet.max(axis=0)).astype(np.float16),
                            sagittal=resize_mip(pet.max(axis=2)).astype(np.float16))

        log.append({
            "subject_id": subj, "session": sess, "status": "ok",
            "shape_resampled": str(pet.shape),
            "n_patches": len(patches),
            "suv_mean": float(pet.mean()),
            "suv_max": float(pet.max()),
        })

    except Exception as e:
        print(f"ERROR {subj}/{sess}: {e}")
        log.append({"subject_id": subj, "session": sess, "status": f"error: {e}"})

log_df = pd.DataFrame(log)
log_df.to_csv(PATCH_DIR / "preprocessing_log.csv", index=False)

# %% [markdown]
# ## 8. QC

# %%
ok = log_df[log_df["status"] == "ok"]
print("=== T9 Vienna QUADRA Preprocessing QC ===")
print(f"Subjects: {ok['subject_id'].nunique()}")
print(f"Sessions: {len(ok)} (expect 96)")
print(f"Complete pairs: {(ok.groupby('subject_id')['session'].count() == 2).sum()}/48")
print(f"Patches per session: {ok['n_patches'].astype(int).mean():.0f}")
print(f"SUV mean: {ok['suv_mean'].astype(float).mean():.3f}")
print(f"Errors: {(log_df['status'] != 'ok').sum()}")

size_gb = sum(f.stat().st_size for f in PATCH_DIR.rglob("*") if f.is_file()) / 1e9
print(f"\nOutput size: {size_gb:.2f} GB / 19.5 GB limit")

# %% [markdown]
# ## 9. Clean up raw data

# %%
if TMP_DIR.exists():
    raw_gb = sum(f.stat().st_size for f in TMP_DIR.rglob("*") if f.is_file()) / 1e9
    shutil.rmtree(TMP_DIR)
    print(f"Cleaned up {raw_gb:.1f} GB from /tmp/")

# %% [markdown]
# ## 10. Done
#
# This notebook should now be committed with **"Save & Run All"** on **CPU**.
# After the commit completes:
# 1. Go to the notebook viewer page (Your Work → Code → click notebook name)
# 2. Click the **Output** tab
# 3. Click **"New Dataset"** to save as `haydenfarquhar/pet-fm-bench-t9-patches`
# 4. Then run the T9 embedding extraction notebook (GPU) with this dataset attached
