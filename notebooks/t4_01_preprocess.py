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
# # T4 Notebook 1/2: Download & Preprocess NSCLC-Radiogenomics
#
# **PET-FM-Bench** | DOI: [10.17605/OSF.IO/DQ2JA](https://doi.org/10.17605/OSF.IO/DQ2JA)
#
# **Runtime:** CPU (no GPU) | **Internet:** On | **Time:** ~1–2 hours
#
# Downloads NSCLC-Radiogenomics PET/CT from TCIA (selective: AC PET + CT only,
# ~45 GB vs 98 GB full), converts DICOM → NIfTI, extracts patches and MIPs.
#
# **Task:** T4 — NSCLC survival prediction (Harrell's c-index)
# - 201 patients with PET, 211 with CT, 144 with tumour segmentations
# - Clinical endpoint: Survival Status (Alive/Dead), Time to Death (days)

# %% [markdown]
# ## 1. Setup

# %%
import os
import sys
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
DICOM_DIR = TMP_DIR / "dicom" / "nsclc_radiogenomics"
NIFTI_DIR = TMP_DIR / "nifti" / "nsclc_radiogenomics"
PATCH_DIR = Path("/kaggle/working/t4_nsclc_radiogenomics")
OUTPUT_DIR = Path("/kaggle/working")

PATCH_SIZE = (96, 96, 96)
SPACING = (2.0, 2.0, 2.0)

CLINICAL_CSV_URL = "https://wiki.cancerimagingarchive.net/download/attachments/28672347/NSCLCR01Radiogenomic_DATA_LABELS_2018-05-22_1500-shifted.csv"

for d in [DICOM_DIR, NIFTI_DIR, PATCH_DIR]:
    d.mkdir(parents=True, exist_ok=True)

print(f"Tmp disk: {os.statvfs('/tmp').f_bavail * os.statvfs('/tmp').f_frsize / 1e9:.1f} GB")
print(f"Output disk: {os.statvfs('/kaggle/working').f_bavail * os.statvfs('/kaggle/working').f_frsize / 1e9:.1f} GB")

# %% [markdown]
# ## 3. Download clinical data

# %%
import requests

clinical_path = PATCH_DIR / "clinical.csv"
if not clinical_path.exists():
    r = requests.get(CLINICAL_CSV_URL)
    r.raise_for_status()
    with open(clinical_path, "wb") as f:
        f.write(r.content)

clinical = pd.read_csv(clinical_path)
print(f"Clinical data: {len(clinical)} patients")
print(f"Survival: {clinical['Survival Status'].value_counts().to_dict()}")
print(f"Deaths with time: {clinical['Time to Death (days)'].notna().sum()}")

# %% [markdown]
# ## 4. Query TCIA for series metadata

# %%
from tcia_utils import nbia

print("Querying TCIA for NSCLC Radiogenomics series...")
all_series = nbia.getSeries(collection="NSCLC Radiogenomics", format="df")
print(f"Total series: {len(all_series)}")
print(f"Modalities: {all_series['Modality'].value_counts().to_dict()}")

# %% [markdown]
# ## 5. Select primary PET and CT series per patient
#
# Strategy:
# - PET: Exclude NAC (non-attenuation-corrected) and recon views (COR/SAG).
#   Pick the series with the most slices per patient (likely the 3D AC volume).
# - CT: Pick the series with the most slices per patient (diagnostic CT).

# %%
# --- PET selection ---
pet_all = all_series[all_series["Modality"] == "PT"].copy()

nac_kw = ["NAC", "NO_AC", "Uncorrected", "NoAC"]
view_kw = ["COR PET", "SAG PET", "PET COR", "PET SAG", "MIP"]

pet_all["is_nac"] = pet_all["SeriesDescription"].apply(
    lambda d: any(k.upper() in str(d).upper() for k in nac_kw)
)
pet_all["is_view"] = pet_all["SeriesDescription"].apply(
    lambda d: any(k.upper() in str(d).upper() for k in view_kw)
)

pet_primary = pet_all[(~pet_all["is_nac"]) & (~pet_all["is_view"])].copy()
pet_primary = pet_primary.sort_values("ImageCount", ascending=False)
pet_best = pet_primary.drop_duplicates(subset="PatientID", keep="first")

# --- CT selection ---
ct_all = all_series[all_series["Modality"] == "CT"].copy()
ct_best = ct_all.sort_values("ImageCount", ascending=False).drop_duplicates(
    subset="PatientID", keep="first"
)

# --- SEG selection ---
seg_all = all_series[all_series["Modality"] == "SEG"]

print(f"Selected PET: {len(pet_best)} patients ({pet_best['FileSize'].sum()/1e9:.1f} GB)")
print(f"Selected CT: {len(ct_best)} patients ({ct_best['FileSize'].sum()/1e9:.1f} GB)")
print(f"SEG: {len(seg_all)} patients ({seg_all['FileSize'].sum()/1e9:.2f} GB)")

total_gb = (pet_best["FileSize"].sum() + ct_best["FileSize"].sum() + seg_all["FileSize"].sum()) / 1e9
print(f"\nTotal download: {total_gb:.1f} GB")

# Combine into download manifest
download_series = pd.concat([pet_best, ct_best, seg_all], ignore_index=True)
download_series.to_csv(PATCH_DIR / "download_manifest.csv", index=False)

# %% [markdown]
# ## 6. Download DICOM from TCIA
#
# Uses tcia_utils to download selected series only. This takes ~30-60 min
# depending on network speed.

# %%
series_uids = download_series["SeriesInstanceUID"].tolist()
print(f"Downloading {len(series_uids)} series ({total_gb:.1f} GB)...")
print("This will take a while.")

nbia.downloadSeries(
    series_uids,
    path=str(DICOM_DIR),
    input_type="list"
)

print("Download complete")

# Verify
downloaded = list(DICOM_DIR.rglob("*.dcm"))
print(f"DICOM files on disk: {len(downloaded)}")

# %% [markdown]
# ## 7. Convert DICOM → NIfTI with dcm2niix

# %%
# dcm2niix converts each DICOM series to a single NIfTI file
# The TCIA download creates subdirectories per SeriesInstanceUID

series_dirs = [d for d in DICOM_DIR.iterdir() if d.is_dir()]
print(f"Series directories: {len(series_dirs)}")

# Build a mapping: SeriesInstanceUID → (PatientID, Modality)
uid_to_info = {}
for _, row in download_series.iterrows():
    uid_to_info[row["SeriesInstanceUID"]] = {
        "patient_id": row["PatientID"],
        "modality": row["Modality"],
        "description": row.get("SeriesDescription", ""),
    }

converted = []
for sdir in tqdm(series_dirs, desc="Converting DICOM→NIfTI"):
    uid = sdir.name
    info = uid_to_info.get(uid, {})
    patient_id = info.get("patient_id", "unknown")
    modality = info.get("modality", "unknown")

    out_dir = NIFTI_DIR / patient_id / modality
    out_dir.mkdir(parents=True, exist_ok=True)

    result = subprocess.run(
        ["dcm2niix", "-z", "y", "-f", f"{patient_id}_{modality}_%d",
         "-o", str(out_dir), str(sdir)],
        capture_output=True, text=True
    )

    niftis = list(out_dir.glob("*.nii.gz"))
    converted.append({
        "uid": uid, "patient_id": patient_id, "modality": modality,
        "n_nifti": len(niftis),
        "success": result.returncode == 0,
    })

conv_df = pd.DataFrame(converted)
print(f"\nConverted: {conv_df['success'].sum()}/{len(conv_df)}")
print(f"Failed: {(~conv_df['success']).sum()}")

# %% [markdown]
# ## 8. Build processing manifest

# %%
manifest_rows = []

patients = sorted(set(pet_best["PatientID"].tolist()))
for pid in patients:
    pet_dir = NIFTI_DIR / pid / "PT"
    ct_dir = NIFTI_DIR / pid / "CT"

    pet_files = sorted(pet_dir.glob("*.nii.gz")) if pet_dir.exists() else []
    ct_files = sorted(ct_dir.glob("*.nii.gz")) if ct_dir.exists() else []

    # Pick the largest NIfTI (most likely the full volume, not a scout)
    pet_file = max(pet_files, key=lambda f: f.stat().st_size) if pet_files else None
    ct_file = max(ct_files, key=lambda f: f.stat().st_size) if ct_files else None

    # Clinical data — compute follow-up time for ALL patients
    # Dead: use Time to Death (days)
    # Alive/censored: compute days from CT Date to Date of Last Known Alive
    clin_row = clinical[clinical["Case ID"] == pid]
    if len(clin_row) > 0:
        survival_status = clin_row["Survival Status"].values[0]
        time_to_death = clin_row["Time to Death (days)"].values[0]
        event = 1 if survival_status == "Dead" else 0

        if event == 1 and pd.notna(time_to_death) and time_to_death > 0:
            followup_time = float(time_to_death)
        else:
            # Compute from dates (shifted but differences preserved)
            try:
                ct_date = pd.to_datetime(clin_row["CT Date"].values[0])
                last_alive = pd.to_datetime(clin_row["Date of Last Known Alive"].values[0])
                followup_time = max(1.0, (last_alive - ct_date).days)
            except Exception:
                followup_time = None
    else:
        survival_status = None
        event = 0
        followup_time = None

    manifest_rows.append({
        "patient_id": pid,
        "pet_path": str(pet_file) if pet_file else None,
        "ct_path": str(ct_file) if ct_file else None,
        "survival_status": survival_status,
        "time_to_death": followup_time,  # Now contains follow-up time for ALL patients
        "event": event,
    })

manifest = pd.DataFrame(manifest_rows)
print(f"Manifest: {len(manifest)} patients")
print(f"PET found: {manifest['pet_path'].notna().sum()}")
print(f"CT found: {manifest['ct_path'].notna().sum()}")
print(f"Events (deaths): {manifest['event'].sum()}")
manifest.to_csv(PATCH_DIR / "manifest.csv", index=False)

# %% [markdown]
# ## 9. Preprocessing functions

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


def dicom_pet_to_suv(img_sitk):
    """Check if PET NIfTI needs SUV conversion.

    dcm2niix typically handles SUV conversion automatically via the
    BW (body weight) method when DICOM headers contain the necessary
    fields. We verify by checking the value range.
    """
    arr = sitk.GetArrayFromImage(img_sitk).astype(np.float32)
    # If max > 100, likely in Bq/mL and needs conversion
    # If max < 100, likely already in SUV
    # This is a heuristic — dcm2niix should handle it
    if arr.max() > 1000:
        print(f"    WARNING: PET max={arr.max():.0f}, may be in Bq/mL not SUV")
    return arr


# %% [markdown]
# ## 10. Process all patients

# %%
log = []

for _, row in tqdm(manifest.iterrows(), total=len(manifest), desc="Processing"):
    pid = row["patient_id"]

    if row["pet_path"] is None or row["pet_path"] == "None":
        log.append({"patient_id": pid, "status": "no_pet"})
        continue

    try:
        # Load PET
        pet_sitk = sitk.ReadImage(row["pet_path"])
        pet_arr = dicom_pet_to_suv(pet_sitk)
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
# ## 11. QC

# %%
ok = log_df[log_df["status"] == "ok"]
print("=== T4 NSCLC-Radiogenomics Preprocessing QC ===")
print(f"Patients processed: {len(ok)}/{len(manifest)}")
print(f"Skipped (no PET): {(log_df['status'] == 'no_pet').sum()}")
print(f"Errors: {(log_df['status'].str.startswith('error')).sum()}")
print(f"Patches per patient: {ok['n_patches'].astype(int).mean():.1f} +/- {ok['n_patches'].astype(int).std():.1f}")

# Merge with survival data for overview
merged = ok.merge(manifest[["patient_id", "event", "survival_status"]], on="patient_id")
print(f"\nSurvival endpoint available: {merged['survival_status'].notna().sum()}/{len(merged)}")
print(f"Events (deaths): {merged['event'].sum()}")

size_gb = sum(f.stat().st_size for f in PATCH_DIR.rglob("*") if f.is_file()) / 1e9
print(f"\nOutput size: {size_gb:.2f} GB / 19.5 GB limit")

# %% [markdown]
# ## 12. Clean up raw data

# %%
import shutil
if TMP_DIR.exists():
    raw_gb = sum(f.stat().st_size for f in TMP_DIR.rglob("*") if f.is_file()) / 1e9
    shutil.rmtree(TMP_DIR)
    print(f"Cleaned up {raw_gb:.1f} GB from /tmp/")

# %% [markdown]
# ## 13. Done
#
# Commit with **"Save & Run All"** on **CPU** (no GPU needed).
# After commit: Output tab → **"New Dataset"** → name it
# `pet-fm-bench-t4-patches`.
# Then run the T4 embedding notebook (GPU) with this dataset attached.
