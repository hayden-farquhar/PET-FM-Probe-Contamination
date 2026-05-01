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
# # T9: Vienna QUADRA Healthy-Control Test-Retest — Download & Preprocess
#
# **PET-FM-Bench** | Pre-registration DOI: [10.17605/OSF.IO/DQ2JA](https://doi.org/10.17605/OSF.IO/DQ2JA)
#
# This notebook downloads the Vienna QUADRA healthy-control test-retest dataset
# from Zenodo, extracts it, and prepares preprocessed patches for FM embedding
# extraction.
#
# **Dataset:** Galler et al. (2025) "Whole-Body [18F]FDG-PET/CT Imaging of
# Healthy Controls: Test/Retest Data for Systemic, Multi-Organ Analysis"
# *Scientific Data*. Zenodo DOI: 10.5281/zenodo.16364694
#
# **Task:** T9 — Healthy-control test-retest embedding stability
# - 48 subjects, each scanned twice (Test + Retest), ~38 days apart
# - Already in NIfTI format, SUV-converted
# - Organ segmentations provided (MOOSE-generated)
# - No probe training — all data used for test-retest CCC evaluation
#
# **Output:** Preprocessed patches saved as Kaggle Dataset for downstream
# embedding extraction notebooks.

# %% [markdown]
# ## 0. Environment setup

# %%
import os
import sys
import hashlib
import zipfile
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

# Install dependencies not in default Kaggle environment
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                       "SimpleITK>=2.3", "monai>=1.3", "tqdm", "requests"])

import SimpleITK as sitk
from tqdm import tqdm

# %% [markdown]
# ## 1. Configuration

# %%
# Paths
# Raw data goes to /tmp/ (ephemeral, ~70 GB) to avoid filling the 20 GB output quota.
# Only preprocessed patches are saved to /kaggle/working/ (persists as notebook output).
WORK_DIR = Path("/kaggle/working")
TMP_DIR = Path("/tmp/pet_fm_bench")
RAW_DIR = TMP_DIR / "raw" / "vienna_quadra"
PATCH_DIR = WORK_DIR / "patches" / "t9_vienna_quadra"
OUTPUT_DIR = WORK_DIR / "output"

# Zenodo download URL
ZENODO_RECORD_ID = "16686025"
ZENODO_FILENAME = "QUADRA_HC.zip"
ZENODO_URL = f"https://zenodo.org/records/{ZENODO_RECORD_ID}/files/{ZENODO_FILENAME}"

# Preprocessing parameters (from registration Section 5.1)
PATCH_SIZE_3D = (96, 96, 96)  # For 3D FMs
SPACING_ISO = (2.0, 2.0, 2.0)  # Isotropic 2mm
MIP_SIZE_2D = (224, 224)  # For 2D FMs (BiomedCLIP, RAD-DINO, DINOv2)

# Storage optimisation
SAVE_CT_PATCHES = False  # CT-input sensitivity analysis extracted in separate notebook
USE_FLOAT16 = True       # SUV values don't need float32 precision

# Create directories
RAW_DIR.mkdir(parents=True, exist_ok=True)
PATCH_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Work directory: {WORK_DIR}")
print(f"Available disk: {os.statvfs(WORK_DIR).f_bavail * os.statvfs(WORK_DIR).f_frsize / 1e9:.1f} GB")

# %% [markdown]
# ## 2. Download from Zenodo

# %%
zip_path = RAW_DIR / ZENODO_FILENAME

def download_with_progress(url, dest_path):
    """Download a file with a single tqdm progress bar (no per-chunk log spam)."""
    import requests
    response = requests.get(url, stream=True)
    response.raise_for_status()
    total = int(response.headers.get("content-length", 0))
    with open(dest_path, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True, desc=dest_path.name
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
            f.write(chunk)
            pbar.update(len(chunk))


if not zip_path.exists():
    print(f"Downloading {ZENODO_FILENAME} from Zenodo (~20 GB) to /tmp/...")
    download_with_progress(ZENODO_URL, zip_path)
    print(f"Download complete: {zip_path.stat().st_size / 1e9:.1f} GB")
else:
    print(f"Already downloaded: {zip_path.stat().st_size / 1e9:.1f} GB")

# Compute SHA256 for reproducibility log
print("Computing SHA256...")
sha256 = hashlib.sha256()
with open(zip_path, "rb") as f:
    for chunk in iter(lambda: f.read(8192 * 1024), b""):
        sha256.update(chunk)
print(f"SHA256: {sha256.hexdigest()}")

# %% [markdown]
# ## 3. Extract

# %%
extract_dir = RAW_DIR / "QUADRA_HC"

if not extract_dir.exists():
    print("Extracting...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(RAW_DIR)
    print("Extraction complete")

    # Free disk space — remove zip after extraction
    zip_path.unlink()
    print("Removed zip to free disk space")
else:
    print(f"Already extracted: {extract_dir}")

# Subjects are inside QUADRA_HC/Imaging Data/
imaging_dir = extract_dir / "Imaging Data"
subjects = sorted([d for d in imaging_dir.iterdir() if d.is_dir() and d.name.startswith("QUADRA_HC_")])
print(f"Found {len(subjects)} subjects")
if subjects:
    first = subjects[0]
    print(f"\nStructure of {first.name}:")
    for p in sorted(first.rglob("*")):
        if p.is_file():
            rel = p.relative_to(first)
            size_mb = p.stat().st_size / 1e6
            print(f"  {rel} ({size_mb:.1f} MB)")

# %% [markdown]
# ## 4. Build subject manifest

# %%
manifest_rows = []
for subj_dir in subjects:
    subj_id = subj_dir.name  # e.g., QUADRA_HC_001

    for session_name in ["Test", "Retest"]:
        session_dir = subj_dir / session_name
        if not session_dir.exists():
            print(f"WARNING: Missing session {session_name} for {subj_id}")
            continue

        # File naming: *_PT-SUV.nii.gz, *_CT-AC.nii.gz, Segmentations/*_Organs.nii.gz
        pet_file = next(session_dir.glob("*_PT-SUV.nii.gz"), None)
        ct_file = next(session_dir.glob("*_CT-AC.nii.gz"), None)
        seg_dir = session_dir / "Segmentations"
        organs_file = next(seg_dir.glob("*_Organs.nii.gz"), None) if seg_dir.exists() else None

        manifest_rows.append({
            "subject_id": subj_id,
            "session": session_name,
            "pet_path": str(pet_file) if pet_file else None,
            "ct_path": str(ct_file) if ct_file else None,
            "seg_path": str(organs_file) if organs_file else None,
        })

manifest = pd.DataFrame(manifest_rows)
print(f"\nManifest: {len(manifest)} rows ({manifest['subject_id'].nunique()} subjects)")
print(f"Sessions per subject: {manifest.groupby('subject_id')['session'].count().value_counts().to_dict()}")
print(f"PET files found: {manifest['pet_path'].notna().sum()}/{len(manifest)}")
print(f"CT files found: {manifest['ct_path'].notna().sum()}/{len(manifest)}")
print(f"Seg files found: {manifest['seg_path'].notna().sum()}/{len(manifest)}")

# Show any missing files
missing = manifest[manifest[["pet_path", "ct_path"]].isna().any(axis=1)]
if len(missing) > 0:
    print(f"\nWARNING: {len(missing)} rows with missing PET or CT:")
    print(missing[["subject_id", "session", "pet_path", "ct_path"]])

manifest.to_csv(OUTPUT_DIR / "t9_manifest.csv", index=False)

# %% [markdown]
# ## 5. Preprocessing functions

# %%
def load_nifti(path):
    """Load a NIfTI file and return the image array and metadata."""
    img = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(img)  # (Z, Y, X)
    spacing = img.GetSpacing()  # (X, Y, Z)
    origin = img.GetOrigin()
    direction = img.GetDirection()
    return arr, {"spacing": spacing, "origin": origin, "direction": direction, "size": img.GetSize()}


def resample_to_isotropic(img_sitk, target_spacing=(2.0, 2.0, 2.0)):
    """Resample a SimpleITK image to isotropic spacing."""
    original_spacing = img_sitk.GetSpacing()
    original_size = img_sitk.GetSize()

    new_size = [
        int(round(osz * ospc / tspc))
        for osz, ospc, tspc in zip(original_size, original_spacing, target_spacing)
    ]

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(img_sitk.GetDirection())
    resampler.SetOutputOrigin(img_sitk.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(0)
    resampler.SetInterpolator(sitk.sitkLinear)

    return resampler.Execute(img_sitk)


def extract_whole_body_patches(pet_arr, patch_size=(96, 96, 96), stride_fraction=0.5):
    """Extract overlapping patches from a whole-body PET volume.

    For test-retest, we extract patches at fixed grid positions so the same
    anatomical regions are compared between Test and Retest scans.
    """
    pz, py, px = patch_size
    sz, sy, sx = pet_arr.shape
    stride_z = max(1, int(pz * stride_fraction))
    stride_y = max(1, int(py * stride_fraction))
    stride_x = max(1, int(px * stride_fraction))

    patches = []
    positions = []

    for z in range(0, max(1, sz - pz + 1), stride_z):
        for y in range(0, max(1, sy - py + 1), stride_y):
            for x in range(0, max(1, sx - px + 1), stride_x):
                patch = pet_arr[z:z+pz, y:y+py, x:x+px]
                # Pad if at boundary
                if patch.shape != (pz, py, px):
                    padded = np.zeros((pz, py, px), dtype=patch.dtype)
                    padded[:patch.shape[0], :patch.shape[1], :patch.shape[2]] = patch
                    patch = padded
                patches.append(patch)
                positions.append((z, y, x))

    return np.stack(patches), positions


def compute_mip(pet_arr):
    """Compute maximum intensity projection along coronal axis (axis=1).

    Returns a 2D MIP image suitable for 2D FMs.
    """
    return pet_arr.max(axis=1)  # (Z, X) coronal MIP


def resize_2d(arr_2d, target_size=(224, 224)):
    """Resize a 2D array to target size using SimpleITK."""
    img = sitk.GetImageFromArray(arr_2d.astype(np.float32))
    resampler = sitk.ResampleImageFilter()
    resampler.SetSize((target_size[1], target_size[0]))  # (X, Y)
    scale_x = arr_2d.shape[1] / target_size[1]
    scale_y = arr_2d.shape[0] / target_size[0]
    resampler.SetOutputSpacing((scale_x, scale_y))
    resampler.SetInterpolator(sitk.sitkLinear)
    resampled = resampler.Execute(img)
    return sitk.GetArrayFromImage(resampled)


# %% [markdown]
# ## 6. Process all subjects
#
# For test-retest (T9), we need paired representations from the same subject.
# We extract:
# 1. **3D patches** at fixed grid positions for 3D FMs (Merlin, CT-FM, FMCIB, Pillar-0)
# 2. **2D MIP** for 2D FMs (BiomedCLIP, RAD-DINO, DINOv2)
# 3. **Whole-volume resampled** for FMs that accept full volumes

# %%
results = []
save_dtype = np.float16 if USE_FLOAT16 else np.float32

for _, row in tqdm(manifest.iterrows(), total=len(manifest), desc="Processing"):
    subj_id = row["subject_id"]
    session = row["session"]

    if row["pet_path"] is None:
        print(f"Skipping {subj_id}/{session}: no PET file")
        continue

    try:
        # Load PET (already in SUV)
        pet_sitk = sitk.ReadImage(row["pet_path"])

        # Resample to isotropic 2mm
        pet_iso = resample_to_isotropic(pet_sitk, target_spacing=SPACING_ISO)
        pet_arr = sitk.GetArrayFromImage(pet_iso).astype(np.float32)

        # --- 3D patches (for 3D FMs) ---
        patches_3d, positions = extract_whole_body_patches(
            pet_arr, patch_size=PATCH_SIZE_3D, stride_fraction=1.0
        )

        patch_subdir = PATCH_DIR / "patches_3d" / subj_id / session
        patch_subdir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            patch_subdir / "patches.npz",
            patches=patches_3d.astype(save_dtype),
            positions=np.array(positions),
            spacing=np.array(SPACING_ISO),
        )

        # --- 2D MIPs (for 2D FMs) ---
        mip_coronal = compute_mip(pet_arr)
        mip_resized = resize_2d(mip_coronal, target_size=MIP_SIZE_2D)
        mip_axial = pet_arr.max(axis=0)
        mip_sagittal = pet_arr.max(axis=2)
        mip_axial_resized = resize_2d(mip_axial, target_size=MIP_SIZE_2D)
        mip_sagittal_resized = resize_2d(mip_sagittal, target_size=MIP_SIZE_2D)

        mip_subdir = PATCH_DIR / "mips_2d" / subj_id / session
        mip_subdir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            mip_subdir / "mips.npz",
            mip_coronal=mip_resized.astype(save_dtype),
            mip_axial=mip_axial_resized.astype(save_dtype),
            mip_sagittal=mip_sagittal_resized.astype(save_dtype),
        )

        # CT patches skipped — extracted in separate notebook for sensitivity analysis

        results.append({
            "subject_id": subj_id,
            "session": session,
            "pet_shape_original": pet_sitk.GetSize(),
            "pet_shape_resampled": pet_arr.shape,
            "n_patches_3d": len(patches_3d),
            "mip_shape": mip_resized.shape,
            "suv_mean": float(pet_arr.mean()),
            "suv_max": float(pet_arr.max()),
            "has_ct": row["ct_path"] is not None,
            "status": "ok",
        })

    except Exception as e:
        print(f"ERROR processing {subj_id}/{session}: {e}")
        results.append({
            "subject_id": subj_id,
            "session": session,
            "status": f"error: {e}",
        })

results_df = pd.DataFrame(results)
results_df.to_csv(OUTPUT_DIR / "t9_preprocessing_log.csv", index=False)

print(f"\nProcessed: {(results_df['status'] == 'ok').sum()}/{len(results_df)}")
print(f"Errors: {(results_df['status'] != 'ok').sum()}")

# %% [markdown]
# ## 7. QC summary

# %%
ok = results_df[results_df["status"] == "ok"]
print("=== T9 Vienna QUADRA Preprocessing QC ===")
print(f"Subjects: {ok['subject_id'].nunique()}")
print(f"Sessions: {len(ok)} (expect 96 = 48 x 2)")
print(f"Test sessions: {(ok['session'] == 'Test').sum()}")
print(f"Retest sessions: {(ok['session'] == 'Retest').sum()}")
print(f"\nPatches per session: {ok['n_patches_3d'].describe()}")
print(f"\nSUV stats:")
print(f"  Mean SUV (across sessions): {ok['suv_mean'].mean():.3f} +/- {ok['suv_mean'].std():.3f}")
print(f"  Max SUV range: {ok['suv_max'].min():.1f} - {ok['suv_max'].max():.1f}")

# Check test-retest completeness
subjects_with_both = ok.groupby("subject_id")["session"].nunique()
complete_pairs = (subjects_with_both == 2).sum()
print(f"\nComplete test-retest pairs: {complete_pairs}/{ok['subject_id'].nunique()}")

# Disk usage
patch_size_gb = sum(f.stat().st_size for f in PATCH_DIR.rglob("*") if f.is_file()) / 1e9
print(f"\nTotal patch storage: {patch_size_gb:.2f} GB")

# %% [markdown]
# ## 8. Clean up raw data and prepare output

# %%
import shutil

# Remove raw NIfTI files from /tmp/ to free disk space
if RAW_DIR.exists():
    raw_size = sum(f.stat().st_size for f in RAW_DIR.rglob("*") if f.is_file()) / 1e9
    print(f"Removing raw data from /tmp/ ({raw_size:.1f} GB)...")
    shutil.rmtree(TMP_DIR)
    print("Raw data removed")
else:
    print("Raw data already cleaned up")

# Report final disk usage
final_size = sum(f.stat().st_size for f in PATCH_DIR.rglob("*") if f.is_file()) / 1e9
print(f"Final patch dataset size: {final_size:.2f} GB")

# %% [markdown]
# ## 9. Create Kaggle Dataset metadata
#
# After this notebook runs, save the output as a new Kaggle Dataset
# (`haydenfarquhar/pet-fm-bench-t9-vienna-quadra`) for use by embedding
# extraction notebooks.

# %%
import json

dataset_metadata = {
    "title": "PET-FM-Bench T9: Vienna QUADRA Test-Retest Patches",
    "id": "haydenfarquhar/pet-fm-bench-t9-vienna-quadra",
    "licenses": [{"name": "CC-BY-4.0"}],
    "keywords": ["PET", "PET/CT", "test-retest", "foundation model", "nuclear medicine"],
}

metadata_path = PATCH_DIR / "dataset-metadata.json"
with open(metadata_path, "w") as f:
    json.dump(dataset_metadata, f, indent=2)

print(f"Dataset metadata written to {metadata_path}")
print(f"\nTo upload as Kaggle Dataset, run in a terminal:")
print(f"  kaggle datasets create -p {PATCH_DIR}")
print(f"\nOr save this notebook's output and convert to dataset via Kaggle UI.")

# %% [markdown]
# ## 10. Verification checklist
#
# Before proceeding to embedding extraction, verify:
# - [ ] All 48 subjects processed
# - [ ] All 96 sessions (48 Test + 48 Retest) have patches
# - [ ] No processing errors
# - [ ] 3D patches saved in `patches_3d/{subject}/{session}/patches.npz`
# - [ ] 2D MIPs saved in `mips_2d/{subject}/{session}/mips.npz`
# - [ ] CT patches saved in `patches_3d_ct/{subject}/{session}/patches.npz`
# - [ ] Manifest CSV saved
# - [ ] Raw data deleted to free disk space
# - [ ] Total patch size fits within Kaggle Dataset limits (~20 GB)
