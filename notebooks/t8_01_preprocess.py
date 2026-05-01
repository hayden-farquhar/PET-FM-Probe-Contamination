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
# # T8 Notebook 1/2: Download & Preprocess Lung-PET-CT-Dx
#
# **PET-FM-Bench** | DOI: [10.17605/OSF.IO/DQ2JA](https://doi.org/10.17605/OSF.IO/DQ2JA)
#
# **Runtime:** CPU | **Internet:** On | **Time:** ~30 min
#
# Downloads PET series from TCIA (selective, 3.8 GB — not the full 60 GB
# dataset), converts DICOM → NIfTI, extracts patches and MIPs.
#
# **Task:** T8 — Lung cancer subtype classification (multi-class)
# - 133 patients with PET scans
# - Classes: A=Adenocarcinoma (95), B=Small Cell (9), G=Squamous Cell (29)
# - Note: E=Large Cell has 0 PET patients — excluded
# - Patient ID prefix encodes the diagnosis

# %% [markdown]
# ## 1. Setup

# %%
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

!pip install -q tcia-utils SimpleITK pydicom dcm2niix tqdm

import SimpleITK as sitk
from tqdm import tqdm

# %% [markdown]
# ## 2. Configuration

# %%
TMP_DIR = Path("/tmp/pet_fm_bench")
DICOM_DIR = TMP_DIR / "dicom" / "lung_pet_ct_dx"
NIFTI_DIR = TMP_DIR / "nifti" / "lung_pet_ct_dx"
PATCH_DIR = Path("/kaggle/working/t8_lung_pet_ct_dx")

PATCH_SIZE = (96, 96, 96)
SPACING = (2.0, 2.0, 2.0)

# Cancer subtype mapping from patient ID prefix
SUBTYPE_MAP = {
    "A": "adenocarcinoma",
    "B": "small_cell",
    "E": "large_cell",
    "G": "squamous_cell",
}

for d in [DICOM_DIR, NIFTI_DIR, PATCH_DIR]:
    d.mkdir(parents=True, exist_ok=True)

print(f"Tmp disk: {os.statvfs('/tmp').f_bavail * os.statvfs('/tmp').f_frsize / 1e9:.1f} GB")
print(f"Output disk: {os.statvfs('/kaggle/working').f_bavail * os.statvfs('/kaggle/working').f_frsize / 1e9:.1f} GB")

# %% [markdown]
# ## 3. Query TCIA and select PET series

# %%
from tcia_utils import nbia

print("Querying TCIA...")
all_series = nbia.getSeries(collection="Lung-PET-CT-Dx", format="df")
print(f"Total series: {len(all_series)}")

# Select PET only — all are "PET WB Corrected", no filtering needed
pet_series = all_series[all_series["Modality"] == "PT"].copy()
print(f"PET series: {len(pet_series)} patients, {pet_series['FileSize'].sum()/1e9:.1f} GB")

# Extract cancer subtype from patient ID
pet_series["subtype"] = pet_series["PatientID"].apply(
    lambda x: SUBTYPE_MAP.get(x.split("-")[1][0], "unknown")
)
print(f"\nSubtypes in PET subset:")
print(pet_series.groupby("subtype")["PatientID"].nunique().to_dict())

# %% [markdown]
# ## 4. Download PET DICOM from TCIA

# %%
series_uids = pet_series["SeriesInstanceUID"].tolist()
print(f"Downloading {len(series_uids)} PET series ({pet_series['FileSize'].sum()/1e9:.1f} GB)...")

nbia.downloadSeries(
    series_uids,
    path=str(DICOM_DIR),
    input_type="list"
)

print("Download complete")
print(f"DICOM files: {len(list(DICOM_DIR.rglob('*.dcm')))}")

# %% [markdown]
# ## 5. Convert DICOM → NIfTI

# %%
uid_to_patient = dict(zip(pet_series["SeriesInstanceUID"], pet_series["PatientID"]))

converted = []
for sdir in tqdm(list(DICOM_DIR.iterdir()), desc="DICOM→NIfTI"):
    if not sdir.is_dir():
        continue
    uid = sdir.name
    pid = uid_to_patient.get(uid, "unknown")

    out_dir = NIFTI_DIR / pid
    out_dir.mkdir(parents=True, exist_ok=True)

    result = subprocess.run(
        ["dcm2niix", "-z", "y", "-f", f"{pid}_PT_%d", "-o", str(out_dir), str(sdir)],
        capture_output=True, text=True
    )
    niftis = list(out_dir.glob("*.nii.gz"))
    converted.append({"uid": uid, "patient_id": pid, "n_nifti": len(niftis),
                       "success": result.returncode == 0})

conv_df = pd.DataFrame(converted)
print(f"Converted: {conv_df['success'].sum()}/{len(conv_df)}")

# %% [markdown]
# ## 6. Build manifest with labels

# %%
manifest_rows = []
for pid in sorted(pet_series["PatientID"].unique()):
    nifti_dir = NIFTI_DIR / pid
    nifti_files = sorted(nifti_dir.glob("*.nii.gz")) if nifti_dir.exists() else []
    pet_file = max(nifti_files, key=lambda f: f.stat().st_size) if nifti_files else None

    prefix = pid.split("-")[1][0]
    subtype = SUBTYPE_MAP.get(prefix, "unknown")

    manifest_rows.append({
        "patient_id": pid,
        "pet_path": str(pet_file) if pet_file else None,
        "subtype": subtype,
    })

manifest = pd.DataFrame(manifest_rows)
print(f"Manifest: {len(manifest)} patients")
print(f"PET found: {manifest['pet_path'].notna().sum()}")
print(f"Subtypes: {manifest['subtype'].value_counts().to_dict()}")
manifest.to_csv(PATCH_DIR / "manifest.csv", index=False)

# %% [markdown]
# ## 7. Preprocessing functions

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
    img = sitk.GetImageFromArray(arr_2d.astype(np.float32))
    resampler = sitk.ResampleImageFilter()
    resampler.SetSize((size, size))
    resampler.SetOutputSpacing((arr_2d.shape[1] / size, arr_2d.shape[0] / size))
    resampler.SetInterpolator(sitk.sitkLinear)
    return sitk.GetArrayFromImage(resampler.Execute(img))

# %% [markdown]
# ## 8. Process all patients

# %%
log = []

for _, row in tqdm(manifest.iterrows(), total=len(manifest), desc="Processing"):
    pid = row["patient_id"]

    if row["pet_path"] is None or row["pet_path"] == "None":
        log.append({"patient_id": pid, "status": "no_pet"})
        continue

    try:
        pet_sitk = sitk.ReadImage(row["pet_path"])
        pet_iso = resample_isotropic(pet_sitk, SPACING)
        pet = sitk.GetArrayFromImage(pet_iso).astype(np.float32)

        # 3D patches
        patches, positions = extract_patches(pet, PATCH_SIZE)
        out_dir = PATCH_DIR / "patches_3d" / pid
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_dir / "patches.npz",
                            patches=patches.astype(np.float16),
                            positions=np.array(positions))

        # 2D MIPs
        mip_dir = PATCH_DIR / "mips_2d" / pid
        mip_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(mip_dir / "mips.npz",
                            coronal=resize_mip(pet.max(axis=1)).astype(np.float16),
                            axial=resize_mip(pet.max(axis=0)).astype(np.float16),
                            sagittal=resize_mip(pet.max(axis=2)).astype(np.float16))

        log.append({
            "patient_id": pid, "status": "ok",
            "shape": str(pet.shape), "n_patches": len(patches),
            "suv_mean": float(pet.mean()), "suv_max": float(pet.max()),
        })

    except Exception as e:
        print(f"ERROR {pid}: {e}")
        log.append({"patient_id": pid, "status": f"error: {e}"})

log_df = pd.DataFrame(log)
log_df.to_csv(PATCH_DIR / "preprocessing_log.csv", index=False)

# %% [markdown]
# ## 9. QC

# %%
ok = log_df[log_df["status"] == "ok"]
print("=== T8 Lung-PET-CT-Dx Preprocessing QC ===")
print(f"Patients processed: {len(ok)}/{len(manifest)}")
print(f"Errors: {(log_df['status'].str.startswith('error')).sum()}")
print(f"Patches per patient: {ok['n_patches'].astype(int).mean():.1f} +/- {ok['n_patches'].astype(int).std():.1f}")

merged = ok.merge(manifest[["patient_id", "subtype"]], on="patient_id")
print(f"\nBy subtype:")
print(merged["subtype"].value_counts().to_dict())

size_gb = sum(f.stat().st_size for f in PATCH_DIR.rglob("*") if f.is_file()) / 1e9
print(f"\nOutput size: {size_gb:.2f} GB / 19.5 GB limit")

# %% [markdown]
# ## 10. Clean up

# %%
import shutil
if TMP_DIR.exists():
    raw_gb = sum(f.stat().st_size for f in TMP_DIR.rglob("*") if f.is_file()) / 1e9
    shutil.rmtree(TMP_DIR)
    print(f"Cleaned up {raw_gb:.1f} GB from /tmp/")

# %% [markdown]
# ## 11. Done
#
# Commit with **"Save & Run All"** on CPU.
# Output tab → **"New Dataset"** → `pet-fm-bench-t8-patches`
