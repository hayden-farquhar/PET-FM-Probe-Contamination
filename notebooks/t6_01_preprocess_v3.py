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
# # T6 Notebook 1/2 v3: Download & Preprocess RIDER-Lung-PET-CT (SUV-converted)
#
# **PET-FM-Bench** | DOI: [10.17605/OSF.IO/DQ2JA](https://doi.org/10.17605/OSF.IO/DQ2JA)
#
# **Runtime:** CPU | **Internet:** On | **Time:** ~30–45 min
#
# **What's new in v3 vs v1:**
# - Replaces dcm2niix with (companion project)'s validated `dicom_series_to_suv_sitk` pipeline
# - All saved patches are in **SUV_bw** units (not Bq/mL)
# - Size + slice + dose guards (per `02_t7_dose_investigation.py` findings)
# - **Bonus possibility**: pydicom + sitk handle eccentric DICOMs better than
#   dcm2niix. v1 lost 21/114 sessions at the dcm2niix conversion stage,
#   destroying every test-retest pair. v3 may recover some of those.
#
# **Why:** v1 had **34.4%** of T6 sessions with inf/NaN patches, plus
# **0 surviving test-retest pairs** (all 16 retest patients lost one session
# to dcm2niix failure, leaving each with 1 valid session — no pairs possible).
#
# **Output:** `pet-fm-bench-t6-patches-v3` (do NOT overwrite v1).

# %% [markdown]
# ## 1. Setup

# %%
import datetime
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

!pip install -q tcia-utils SimpleITK pydicom

import SimpleITK as sitk
import pydicom
from tcia_utils import nbia
from tqdm import tqdm

# %% [markdown]
# ## 2. Configuration

# %%
TMP_DIR = Path("/tmp/pet_fm_bench_v3")
DICOM_DIR = TMP_DIR / "dicom" / "rider_lung_pet_ct"
PATCH_DIR = Path("/kaggle/working/t6_rider_lung_pet_ct")

PATCH_SIZE = (96, 96, 96)
SPACING = (2.0, 2.0, 2.0)

MIN_SERIES_SIZE_MB = 1.0
MIN_SLICE_COUNT = 50
MIN_DOSE_MBQ = 10.0

for d in [DICOM_DIR, PATCH_DIR]:
    d.mkdir(parents=True, exist_ok=True)

print(f"Tmp disk: {os.statvfs('/tmp').f_bavail * os.statvfs('/tmp').f_frsize / 1e9:.1f} GB")
print(f"Output disk: {os.statvfs('/kaggle/working').f_bavail * os.statvfs('/kaggle/working').f_frsize / 1e9:.1f} GB")

# %% [markdown]
# ## 3. (companion project) SUV pipeline (inlined, validated 9/9 across Siemens/GE/Philips)

# %%
HALF_LIVES = {"F-18": 6586.2, "Ga-68": 4062.0, "C-11": 1223.4}


@dataclass
class PETMetadata:
    patient_id: str
    patient_weight_kg: float
    injected_dose_bq: float
    injection_time: datetime.datetime
    scan_time: datetime.datetime
    half_life_sec: float
    radionuclide: str
    manufacturer: str
    manufacturer_model: str
    series_uid: str
    study_uid: str
    study_date: str
    uptake_time_sec: float
    decay_factor: float


def _get_float(ds, tag):
    val = getattr(ds, tag, None)
    if val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _parse_dicom_datetime(date_str, time_str):
    date_str = date_str.strip()
    time_str = time_str.strip()
    if "." in time_str:
        parts = time_str.split(".")
        main = parts[0].ljust(6, "0")
        frac = parts[1][:6].ljust(6, "0")
        time_str = f"{main}.{frac}"
        fmt = "%Y%m%d%H%M%S.%f"
    else:
        time_str = time_str.ljust(6, "0")
        fmt = "%Y%m%d%H%M%S"
    return datetime.datetime.strptime(date_str + time_str, fmt)


def _infer_radionuclide(half_life_sec):
    for name, hl in HALF_LIVES.items():
        if abs(half_life_sec - hl) / hl < 0.01:
            return name
    return "Unknown"


def extract_pet_metadata(dcm_path):
    ds = pydicom.dcmread(dcm_path, stop_before_pixels=True)
    patient_id = str(getattr(ds, "PatientID", "UNKNOWN"))
    patient_weight_kg = _get_float(ds, "PatientWeight")
    if patient_weight_kg is None:
        raise ValueError(f"PatientWeight missing for {patient_id}")

    radio_seq = getattr(ds, "RadiopharmaceuticalInformationSequence", None)
    if not radio_seq:
        raise ValueError(f"RadiopharmaceuticalInformationSequence missing for {patient_id}")
    radio = radio_seq[0]

    injected_dose_bq = _get_float(radio, "RadionuclideTotalDose")
    if injected_dose_bq is None:
        raise ValueError(f"RadionuclideTotalDose missing for {patient_id}")
    half_life_sec = _get_float(radio, "RadionuclideHalfLife")
    if half_life_sec is None:
        raise ValueError(f"RadionuclideHalfLife missing for {patient_id}")

    radio_code_seq = getattr(radio, "RadionuclideCodeSequence", None)
    if radio_code_seq and len(radio_code_seq) > 0:
        radionuclide = str(getattr(radio_code_seq[0], "CodeMeaning", "Unknown"))
    else:
        radionuclide = _infer_radionuclide(half_life_sec)

    injection_time_str = getattr(radio, "RadiopharmaceuticalStartTime", None)
    if injection_time_str is None:
        raise ValueError(f"RadiopharmaceuticalStartTime missing for {patient_id}")
    scan_time_str = getattr(ds, "AcquisitionTime", None) or getattr(ds, "SeriesTime", None)
    if scan_time_str is None:
        raise ValueError(f"AcquisitionTime missing for {patient_id}")
    series_date = getattr(ds, "SeriesDate", None) or getattr(ds, "StudyDate", "19000101")
    injection_time = _parse_dicom_datetime(series_date, str(injection_time_str))
    scan_time = _parse_dicom_datetime(series_date, str(scan_time_str))
    if scan_time < injection_time:
        scan_time += datetime.timedelta(days=1)
    uptake_time_sec = (scan_time - injection_time).total_seconds()
    decay_factor = 2 ** (-uptake_time_sec / half_life_sec)

    return PETMetadata(
        patient_id=patient_id,
        patient_weight_kg=patient_weight_kg,
        injected_dose_bq=injected_dose_bq,
        injection_time=injection_time,
        scan_time=scan_time,
        half_life_sec=half_life_sec,
        radionuclide=radionuclide,
        manufacturer=str(getattr(ds, "Manufacturer", "Unknown")),
        manufacturer_model=str(getattr(ds, "ManufacturerModelName", "Unknown")),
        series_uid=str(getattr(ds, "SeriesInstanceUID", "")),
        study_uid=str(getattr(ds, "StudyInstanceUID", "")),
        study_date=str(getattr(ds, "StudyDate", "")),
        uptake_time_sec=uptake_time_sec,
        decay_factor=decay_factor,
    )


def dicom_series_to_suv_sitk(dicom_dir, meta):
    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(dicom_dir)
    if not series_ids:
        raise ValueError(f"No DICOM series in {dicom_dir}")
    file_names = reader.GetGDCMSeriesFileNames(dicom_dir, series_ids[0])

    # Check Units tag: BQML = raw Bq/mL (apply SUV formula); GML = already SUV.
    # Older GEMS scanners write GML (data pre-converted at the scanner). Without
    # this branch, applying the SUV formula a second time scales values by
    # ~0.0006, producing pseudo-SUV outputs around 0.02 instead of the real ~30.
    first_ds = pydicom.dcmread(file_names[0], stop_before_pixels=True)
    units = str(getattr(first_ds, "Units", "BQML")).strip().upper()

    slices = []
    for f in file_names:
        d = pydicom.dcmread(f)
        try:
            raw = d.pixel_array
        except (AttributeError, NotImplementedError) as e:
            raise ValueError(f"Cannot read pixel data from {f}: {e}")
        rs = float(getattr(d, "RescaleSlope", 1.0))
        ri = float(getattr(d, "RescaleIntercept", 0.0))
        slices.append(raw.astype(np.float64) * rs + ri)
    if not slices:
        raise ValueError(f"No slices with pixel data in {dicom_dir}")

    rescaled_3d = np.stack(slices, axis=0)
    if units == "GML":
        # Already SUV (older GEMS / some MIMvista-processed scans) — skip formula
        suv_array = rescaled_3d
    else:
        # Default: treat as raw activity (BQML, empty, or non-standard tag).
        # Permissive on purpose — some Philips/CPS scans use non-standard Units
        # strings that aren't GML. Better to apply formula and let SUVmax QC
        # catch any genuinely wrong cases than to silently reject patients.
        decay_corrected_dose = meta.injected_dose_bq * meta.decay_factor
        weight_g = meta.patient_weight_kg * 1000.0
        suv_array = rescaled_3d * weight_g / decay_corrected_dose

    reader.SetFileNames(file_names)
    reader.MetaDataDictionaryArrayUpdateOn()
    reader.LoadPrivateTagsOn()
    image = reader.Execute()

    suv_image = sitk.GetImageFromArray(suv_array)
    suv_image.CopyInformation(image)
    return suv_image


print("(companion project) SUV pipeline loaded")

# %% [markdown]
# ## 4. Query TCIA and select AC PET series

# %%
print("Querying TCIA...")
all_series = nbia.getSeries(collection="RIDER Lung PET-CT", format="df")
print(f"Total series: {len(all_series)}")
print(f"Modalities: {all_series['Modality'].value_counts().to_dict()}")

pet_all = all_series[all_series["Modality"] == "PT"].copy()
nac_kw = ["NAC", "NO_AC", "Uncorrected", "NoAC", "Atten"]
pet_all.loc[:, "is_nac"] = pet_all["SeriesDescription"].apply(
    lambda d: any(k.upper() in str(d).upper() for k in nac_kw)
)
ac_pet = pet_all[~pet_all["is_nac"]].copy()

# Size filter
ac_pet["FileSize_MB"] = ac_pet["FileSize"] / 1e6
n_before_size = len(ac_pet)
ac_pet = ac_pet[ac_pet["FileSize_MB"] >= MIN_SERIES_SIZE_MB].copy()
print(f"After size filter (>= {MIN_SERIES_SIZE_MB} MB): "
      f"{len(ac_pet)} series ({n_before_size - len(ac_pet)} dropped as placeholders)")

# Pick the largest AC series per (patient, study) pair
ac_pet = ac_pet.sort_values("ImageCount", ascending=False)
best_per_study = ac_pet.drop_duplicates(subset=["PatientID", "StudyInstanceUID"], keep="first")

# Identify test-retest patients (2+ studies)
studies_per = best_per_study.groupby("PatientID")["StudyInstanceUID"].nunique()
retest_pids = set(studies_per[studies_per >= 2].index.tolist())

print(f"\nAC PET selected: {len(best_per_study)} series from "
      f"{best_per_study['PatientID'].nunique()} patients")
print(f"Test-retest patients (2+ studies in TCIA): {len(retest_pids)}")
print(f"Single-study patients: {(studies_per == 1).sum()}")
print(f"Download size: {best_per_study['FileSize'].sum()/1e9:.1f} GB")

# %% [markdown]
# ## 5. Download PET DICOM (~3 GB, ~10-15 min)

# %%
series_uids = best_per_study["SeriesInstanceUID"].tolist()
print(f"Downloading {len(series_uids)} series ({best_per_study['FileSize'].sum()/1e9:.1f} GB)...")

nbia.downloadSeries(
    series_uids,
    path=str(DICOM_DIR),
    input_type="list",
)

dcm_count = len(list(DICOM_DIR.rglob("*.dcm")))
print(f"Download complete: {dcm_count} DICOM files in {DICOM_DIR}")

# %% [markdown]
# ## 6. Build manifest with multi-session structure
#
# Each patient has 1 or more PET studies. Test-retest patients have 2+.
# Sort sessions chronologically by StudyDate (consistent test/retest ordering).

# %%
uid_to_info = {}
for _, row in best_per_study.iterrows():
    uid_to_info[row["SeriesInstanceUID"]] = {
        "patient_id": row["PatientID"],
        "study_uid": row["StudyInstanceUID"],
    }

# First pass: gather per-series metadata so we can sort chronologically
series_meta_rows = []
for sdir in DICOM_DIR.iterdir():
    if not sdir.is_dir():
        continue
    uid = sdir.name
    info = uid_to_info.get(uid)
    if info is None:
        continue
    n_slices = len(list(sdir.glob("*.dcm")))
    skip_reason = None
    if n_slices < MIN_SLICE_COUNT:
        skip_reason = f"too few slices ({n_slices} < {MIN_SLICE_COUNT})"

    study_date = ""
    if n_slices > 0:
        try:
            sample = next(sdir.glob("*.dcm"))
            ds = pydicom.dcmread(str(sample), stop_before_pixels=True)
            study_date = str(getattr(ds, "StudyDate", "")) or ""
        except Exception:
            pass

    series_meta_rows.append({
        "patient_id": info["patient_id"],
        "series_uid": uid,
        "study_uid": info["study_uid"],
        "study_date": study_date,
        "dicom_dir": str(sdir),
        "n_slices": n_slices,
        "skip_reason": skip_reason,
        "is_retest_patient": info["patient_id"] in retest_pids,
    })

series_meta = pd.DataFrame(series_meta_rows)

# Assign session index per patient: chronological by study_date, fallback to study_uid
manifest_rows = []
for pid, group in series_meta.groupby("patient_id"):
    sort_keys = group.sort_values(
        ["study_date", "study_uid"],
        kind="mergesort",
    ).reset_index(drop=True)
    for idx, srow in sort_keys.iterrows():
        manifest_rows.append({
            **srow.to_dict(),
            "study_index": idx,
            "session": f"scan_{idx}",
        })

manifest = pd.DataFrame(manifest_rows)
print(f"Manifest: {len(manifest)} sessions from {manifest['patient_id'].nunique()} patients")
print(f"  with usable series: {manifest['skip_reason'].isna().sum()}")
print(f"  pre-flagged for skip: {manifest['skip_reason'].notna().sum()}")
print(f"  Test-retest patients (manifest): "
      f"{manifest[manifest['is_retest_patient']]['patient_id'].nunique()}")
print(f"  Total test-retest sessions: "
      f"{manifest[manifest['is_retest_patient']].shape[0]}")
manifest.to_csv(PATCH_DIR / "manifest.csv", index=False)

# %% [markdown]
# ## 7. Preprocessing functions (identical to T8/T4/T7 v3)

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
# ## 8. Process all sessions (DICOM → SUV → patches)

# %%
log = []

for _, row in tqdm(manifest.iterrows(), total=len(manifest), desc="Processing"):
    pid = row["patient_id"]
    session = row["session"]
    dcm_dir = row["dicom_dir"]

    if pd.notna(row["skip_reason"]):
        log.append({"patient_id": pid, "session": session,
                    "status": f"skipped: {row['skip_reason']}"})
        continue

    try:
        dcm_files = list(Path(dcm_dir).glob("*.dcm"))
        if not dcm_files:
            log.append({"patient_id": pid, "session": session, "status": "no_dcm_files"})
            continue

        meta = extract_pet_metadata(str(dcm_files[0]))
        dose_mbq = meta.injected_dose_bq / 1e6
        if dose_mbq < MIN_DOSE_MBQ:
            log.append({
                "patient_id": pid, "session": session,
                "status": f"skipped: dose {dose_mbq:.4f} MBq < {MIN_DOSE_MBQ}",
                "manufacturer": meta.manufacturer,
            })
            continue

        pet_sitk = dicom_series_to_suv_sitk(dcm_dir, meta)
        pet_iso = resample_isotropic(pet_sitk, SPACING)
        pet = sitk.GetArrayFromImage(pet_iso).astype(np.float32)

        suv_max = float(pet.max())
        suv_mean = float(pet.mean())
        if not np.isfinite(pet).all():
            log.append({"patient_id": pid, "session": session,
                        "status": "non_finite_suv", "manufacturer": meta.manufacturer})
            continue
        if suv_max > 200:
            print(f"  [QC FLAG] {pid}/{session}: SUVmax={suv_max:.1f}")

        patches, positions = extract_patches(pet, PATCH_SIZE)
        out_dir = PATCH_DIR / "patches_3d" / pid / session
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_dir / "patches.npz",
                            patches=patches.astype(np.float16),
                            positions=np.array(positions))

        mip_dir = PATCH_DIR / "mips_2d" / pid / session
        mip_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(mip_dir / "mips.npz",
                            coronal=resize_mip(pet.max(axis=1)).astype(np.float16),
                            axial=resize_mip(pet.max(axis=0)).astype(np.float16),
                            sagittal=resize_mip(pet.max(axis=2)).astype(np.float16))

        log.append({
            "patient_id": pid, "session": session,
            "status": "ok",
            "shape": str(pet.shape),
            "n_patches": len(patches),
            "suv_mean": suv_mean,
            "suv_max": suv_max,
            "manufacturer": meta.manufacturer,
            "model": meta.manufacturer_model,
            "weight_kg": meta.patient_weight_kg,
            "dose_mbq": dose_mbq,
            "uptake_min": meta.uptake_time_sec / 60,
            "study_date": meta.study_date,
        })

    except Exception as e:
        print(f"ERROR {pid}/{session}: {type(e).__name__}: {e}")
        log.append({"patient_id": pid, "session": session,
                    "status": f"error: {type(e).__name__}: {e}"})

log_df = pd.DataFrame(log)
log_df.to_csv(PATCH_DIR / "preprocessing_log.csv", index=False)

# %% [markdown]
# ## 9. QC — including the test-retest pair recovery test

# %%
ok = log_df[log_df["status"] == "ok"]
print("=== T6 v3 SUV-converted Preprocessing QC ===")
print(f"Sessions processed: {len(ok)}/{len(manifest)}")
print(f"Patients: {ok['patient_id'].nunique()}")
print(f"Errors: {(log_df['status'].str.startswith('error')).sum()}")
print(f"Skipped: {(log_df['status'].str.startswith('skipped')).sum()}")

if len(ok) > 0:
    print(f"\nSUV ranges:")
    print(f"  SUVmax median: {ok['suv_max'].median():.2f}, max: {ok['suv_max'].max():.2f}")
    print(f"  SUVmean median: {ok['suv_mean'].median():.4f}")
    n_extreme = (ok["suv_max"] > 1000).sum()
    print(f"  SUVmax > 1000: {n_extreme} (should be 0)")

    if "manufacturer" in ok.columns:
        print(f"\nBy manufacturer:")
        print(ok.groupby("manufacturer").agg(
            n=("patient_id", "count"),
            suvmax_median=("suv_max", "median"),
        ).to_string())

# Test-retest pair recovery — the headline question
retest_manifest = manifest[manifest["is_retest_patient"]]
ok_with_retest = ok.merge(
    manifest[["patient_id", "session", "is_retest_patient"]],
    on=["patient_id", "session"],
)
retest_ok = ok_with_retest[ok_with_retest["is_retest_patient"]]
pairs = retest_ok.groupby("patient_id").size()
n_complete_pairs = (pairs >= 2).sum()
n_retest_total = retest_manifest["patient_id"].nunique()

print(f"\n=== TEST-RETEST PAIR RECOVERY ===")
print(f"Test-retest patients (TCIA query): {n_retest_total}")
print(f"  with 2+ sessions successfully processed: {n_complete_pairs}")
print(f"  with only 1 session processed: {(pairs == 1).sum()}")
print(f"  with 0 sessions processed: "
      f"{n_retest_total - retest_ok['patient_id'].nunique()}")
print(f"\nv1 result: 0 complete pairs (every retest patient lost a session at "
      f"dcm2niix). v3 target: ≥10 complete pairs to make the test-retest probe "
      f"viable for H1/H2.")

size_gb = sum(f.stat().st_size for f in PATCH_DIR.rglob("*") if f.is_file()) / 1e9
print(f"\nOutput size: {size_gb:.2f} GB / 19.5 GB limit")

# %% [markdown]
# ## 10. Clean up

# %%
if TMP_DIR.exists():
    raw_gb = sum(f.stat().st_size for f in TMP_DIR.rglob("*") if f.is_file()) / 1e9
    shutil.rmtree(TMP_DIR)
    print(f"Cleaned up {raw_gb:.1f} GB from /tmp/")

# %% [markdown]
# ## 11. Done
#
# Commit with **"Save & Run All"** on CPU.
# Output → **"New Dataset"** → `pet-fm-bench-t6-patches-v3`.
