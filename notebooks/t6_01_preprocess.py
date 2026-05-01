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
# # T6 Notebook 1/2: Download & Preprocess RIDER-Lung-PET-CT
#
# **PET-FM-Bench** | DOI: [10.17605/OSF.IO/DQ2JA](https://doi.org/10.17605/OSF.IO/DQ2JA)
#
# **Runtime:** CPU | **Internet:** On | **Time:** ~30 min
#
# Downloads AC PET series from TCIA (~3 GB selective), converts DICOM → NIfTI,
# extracts patches and MIPs.
#
# **Task:** T6 — Cancer-patient test-retest embedding stability
# - 20 patients with 2+ PET studies (test-retest pairs)
# - ~96 patients with single PET study (used for general embedding analysis)
# - No probe training — all data used for test-retest CCC evaluation

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
DICOM_DIR = TMP_DIR / "dicom" / "rider_lung_pet_ct"
NIFTI_DIR = TMP_DIR / "nifti" / "rider_lung_pet_ct"
PATCH_DIR = Path("/kaggle/working/t6_rider_lung_pet_ct")

PATCH_SIZE = (96, 96, 96)
SPACING = (2.0, 2.0, 2.0)

for d in [DICOM_DIR, NIFTI_DIR, PATCH_DIR]:
    d.mkdir(parents=True, exist_ok=True)

print(f"Tmp disk: {os.statvfs('/tmp').f_bavail * os.statvfs('/tmp').f_frsize / 1e9:.1f} GB")
print(f"Output disk: {os.statvfs('/kaggle/working').f_bavail * os.statvfs('/kaggle/working').f_frsize / 1e9:.1f} GB")

# %% [markdown]
# ## 3. Query TCIA and select AC PET series

# %%
from tcia_utils import nbia

print("Querying TCIA...")
all_series = nbia.getSeries(collection="RIDER Lung PET-CT", format="df")
print(f"Total series: {len(all_series)}")
print(f"Modalities: {all_series['Modality'].value_counts().to_dict()}")

# Select PET, exclude NAC/attenuation series
pet_all = all_series[all_series["Modality"] == "PT"].copy()
nac_kw = ["NAC", "NO_AC", "Uncorrected", "NoAC", "Atten"]
pet_all.loc[:, "is_nac"] = pet_all["SeriesDescription"].apply(
    lambda d: any(k.upper() in str(d).upper() for k in nac_kw)
)
ac_pet = pet_all[~pet_all["is_nac"]].copy()

# Pick the largest AC series per (patient, study) pair
ac_pet = ac_pet.sort_values("ImageCount", ascending=False)
best_per_study = ac_pet.drop_duplicates(subset=["PatientID", "StudyInstanceUID"], keep="first")

# Identify test-retest patients (2+ studies)
studies_per = best_per_study.groupby("PatientID")["StudyInstanceUID"].nunique()
retest_pids = studies_per[studies_per >= 2].index.tolist()

print(f"\nAC PET selected: {len(best_per_study)} series from {best_per_study['PatientID'].nunique()} patients")
print(f"Test-retest patients (2+ studies): {len(retest_pids)}")
print(f"Single-study patients: {(studies_per == 1).sum()}")
print(f"Download size: {best_per_study['FileSize'].sum()/1e9:.1f} GB")

# %% [markdown]
# ## 4. Download PET DICOM from TCIA

# %%
series_uids = best_per_study["SeriesInstanceUID"].tolist()
print(f"Downloading {len(series_uids)} PET series ({best_per_study['FileSize'].sum()/1e9:.1f} GB)...")

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
uid_to_info = {}
for _, row in best_per_study.iterrows():
    uid_to_info[row["SeriesInstanceUID"]] = {
        "patient_id": row["PatientID"],
        "study_uid": row["StudyInstanceUID"],
    }

converted = []
for sdir in tqdm(list(DICOM_DIR.iterdir()), desc="DICOM→NIfTI"):
    if not sdir.is_dir():
        continue
    uid = sdir.name
    info = uid_to_info.get(uid, {})
    pid = info.get("patient_id", "unknown")
    study_uid = info.get("study_uid", "unknown")

    # Use a short study index to distinguish test vs retest
    out_dir = NIFTI_DIR / pid / study_uid[:12]
    out_dir.mkdir(parents=True, exist_ok=True)

    result = subprocess.run(
        ["dcm2niix", "-z", "y", "-f", f"{pid}_PT_%d", "-o", str(out_dir), str(sdir)],
        capture_output=True, text=True
    )
    niftis = list(out_dir.glob("*.nii.gz"))
    converted.append({"uid": uid, "patient_id": pid, "study_uid": study_uid,
                       "n_nifti": len(niftis), "success": result.returncode == 0})

conv_df = pd.DataFrame(converted)
print(f"Converted: {conv_df['success'].sum()}/{len(conv_df)}")

# %% [markdown]
# ## 6. Build manifest with test-retest labels

# %%
manifest_rows = []

for pid in sorted(best_per_study["PatientID"].unique()):
    pid_dir = NIFTI_DIR / pid
    if not pid_dir.exists():
        continue

    # Get study subdirectories, sorted to give consistent test/retest ordering
    study_dirs = sorted([d for d in pid_dir.iterdir() if d.is_dir()])
    is_retest = pid in retest_pids

    for idx, sdir in enumerate(study_dirs):
        nifti_files = sorted(sdir.glob("*.nii.gz"))
        pet_file = max(nifti_files, key=lambda f: f.stat().st_size) if nifti_files else None

        # Find the StudyInstanceUID from the conversion log
        matching = conv_df[(conv_df["patient_id"] == pid) &
                           (conv_df["study_uid"].str[:12] == sdir.name)]
        study_uid = matching["study_uid"].values[0] if len(matching) > 0 else sdir.name

        manifest_rows.append({
            "patient_id": pid,
            "study_index": idx,
            "session": f"scan_{idx}",
            "study_uid": study_uid,
            "pet_path": str(pet_file) if pet_file else None,
            "is_retest_patient": is_retest,
        })

manifest = pd.DataFrame(manifest_rows)
print(f"Manifest: {len(manifest)} sessions from {manifest['patient_id'].nunique()} patients")
print(f"Test-retest patients: {manifest[manifest['is_retest_patient']]['patient_id'].nunique()}")
print(f"Total test-retest sessions: {manifest[manifest['is_retest_patient']].shape[0]}")
print(f"PET found: {manifest['pet_path'].notna().sum()}/{len(manifest)}")
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
# ## 8. Process all sessions

# %%
log = []

for _, row in tqdm(manifest.iterrows(), total=len(manifest), desc="Processing"):
    pid = row["patient_id"]
    session = row["session"]

    if row["pet_path"] is None or row["pet_path"] == "None":
        log.append({"patient_id": pid, "session": session, "status": "no_pet"})
        continue

    try:
        pet_sitk = sitk.ReadImage(row["pet_path"])
        pet_iso = resample_isotropic(pet_sitk, SPACING)
        pet = sitk.GetArrayFromImage(pet_iso).astype(np.float32)

        # 3D patches
        patches, positions = extract_patches(pet, PATCH_SIZE)
        out_dir = PATCH_DIR / "patches_3d" / pid / session
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_dir / "patches.npz",
                            patches=patches.astype(np.float16),
                            positions=np.array(positions))

        # 2D MIPs
        mip_dir = PATCH_DIR / "mips_2d" / pid / session
        mip_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(mip_dir / "mips.npz",
                            coronal=resize_mip(pet.max(axis=1)).astype(np.float16),
                            axial=resize_mip(pet.max(axis=0)).astype(np.float16),
                            sagittal=resize_mip(pet.max(axis=2)).astype(np.float16))

        log.append({
            "patient_id": pid, "session": session, "status": "ok",
            "shape": str(pet.shape), "n_patches": len(patches),
            "suv_mean": float(pet.mean()), "suv_max": float(pet.max()),
        })

    except Exception as e:
        print(f"ERROR {pid}/{session}: {e}")
        log.append({"patient_id": pid, "session": session, "status": f"error: {e}"})

log_df = pd.DataFrame(log)
log_df.to_csv(PATCH_DIR / "preprocessing_log.csv", index=False)

# %% [markdown]
# ## 9. QC

# %%
ok = log_df[log_df["status"] == "ok"]
print("=== T6 RIDER-Lung-PET-CT Preprocessing QC ===")
print(f"Sessions processed: {len(ok)}/{len(manifest)}")
print(f"Patients: {ok['patient_id'].nunique()}")
print(f"Errors: {(log_df['status'].str.startswith('error')).sum()}")

# Test-retest completeness
retest_manifest = manifest[manifest["is_retest_patient"]]
retest_ok = ok[ok["patient_id"].isin(retest_manifest["patient_id"].unique())]
pairs = retest_ok.groupby("patient_id").size()
complete_pairs = (pairs >= 2).sum()
print(f"\nTest-retest patients with 2+ sessions processed: {complete_pairs}/{len(retest_manifest['patient_id'].unique())}")

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
# Output tab → **"New Dataset"** → `pet-fm-bench-t6-patches`
