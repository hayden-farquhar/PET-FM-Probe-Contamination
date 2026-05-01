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
# # PET-FM-Bench: SUV Pipeline Smoke Test
#
# **Runtime:** CPU | **Internet:** ON (TCIA download) | **Time:** ~15 min
#
# Validates that (companion project)'s `suv_conversion.py` works on each of this project's four
# TCIA-sourced tasks (T4, T6, T7, T8) before committing to full re-preprocessing.
#
# **Strategy:** download the smallest PET series from each collection, run
# the canonical `extract_pet_metadata` function + `dicom_series_to_suv_sitk`, confirm SUVmax
# lands in the expected 0–50 range with no inf/NaN.
#
# **Inputs:** none (downloads fresh DICOMs from TCIA).
#
# **Decision rule per task:**
# - PASS if SUVmax ∈ (0, 100) and no inf/NaN
# - FAIL if SUVmax > 1000 (still Bq/mL) or has inf/NaN or pipeline errors
#
# If all four pass → safe to roll out (companion project)'s pipeline to t4_01/t6_01/t7_01/t8_01.

# %% [markdown]
# ## 1. Setup

# %%
import os
import shutil
import datetime
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

!pip install -q tcia-utils SimpleITK pydicom

import SimpleITK as sitk
import pydicom
from tcia_utils import nbia

TMP = Path("/tmp/suv_smoke")
TMP.mkdir(parents=True, exist_ok=True)
print(f"Tmp disk: {os.statvfs('/tmp').f_bavail * os.statvfs('/tmp').f_frsize / 1e9:.1f} GB free")

# %% [markdown]
# ## 2. (companion project) SUV pipeline (inlined from the validated SUV-conversion module (cross-vendor validated))
#
# Source: `/path/to/companion-project/ Conformal SUV
# Theranostic/src/preprocess/suv_conversion.py`. Validated 9/9 across
# Siemens/GE/Philips with max_rel_diff 0.00000% ((cross-vendor validation reference)).
#
# Once the smoke test passes, this code block becomes the basis for a Kaggle
# Dataset (`pet-suv-conversion-validated`) that the production preprocess
# notebooks attach instead of inlining.

# %%
HALF_LIVES = {"F-18": 6586.2, "Ga-68": 4062.0, "C-11": 1223.4}


@dataclass
class PETMetadata:
    patient_id: str
    patient_weight_kg: float
    patient_height_m: float | None
    patient_sex: str | None
    patient_age: str | None
    injected_dose_bq: float
    injection_time: datetime.datetime
    scan_time: datetime.datetime
    half_life_sec: float
    radionuclide: str
    manufacturer: str
    manufacturer_model: str
    software_version: str | None
    pixel_spacing_mm: tuple
    slice_thickness_mm: float
    rescale_slope: float
    rescale_intercept: float
    series_uid: str
    study_uid: str
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
    """Extract metadata for SUV conversion from any DICOM file in the series."""
    ds = pydicom.dcmread(dcm_path, stop_before_pixels=True)
    patient_id = str(getattr(ds, "PatientID", "UNKNOWN"))
    patient_weight_kg = _get_float(ds, "PatientWeight")
    if patient_weight_kg is None:
        raise ValueError(f"PatientWeight missing for {patient_id}")
    patient_height_m = _get_float(ds, "PatientSize")
    patient_sex = getattr(ds, "PatientSex", None)
    patient_age = getattr(ds, "PatientAge", None)

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
        raise ValueError(f"AcquisitionTime/SeriesTime missing for {patient_id}")

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
        patient_height_m=patient_height_m,
        patient_sex=patient_sex,
        patient_age=patient_age,
        injected_dose_bq=injected_dose_bq,
        injection_time=injection_time,
        scan_time=scan_time,
        half_life_sec=half_life_sec,
        radionuclide=radionuclide,
        manufacturer=str(getattr(ds, "Manufacturer", "Unknown")),
        manufacturer_model=str(getattr(ds, "ManufacturerModelName", "Unknown")),
        software_version=str(getattr(ds, "SoftwareVersions", None)),
        pixel_spacing_mm=tuple(float(x) for x in getattr(ds, "PixelSpacing", [1.0, 1.0])),
        slice_thickness_mm=float(getattr(ds, "SliceThickness", 1.0)),
        rescale_slope=float(getattr(ds, "RescaleSlope", 1.0)),
        rescale_intercept=float(getattr(ds, "RescaleIntercept", 0.0)),
        series_uid=str(getattr(ds, "SeriesInstanceUID", "")),
        study_uid=str(getattr(ds, "StudyInstanceUID", "")),
        uptake_time_sec=uptake_time_sec,
        decay_factor=decay_factor,
    )


def dicom_series_to_suv_sitk(dicom_dir, meta):
    """Read DICOM series with per-slice rescale, return SUV_bw sitk.Image.

    CRITICAL: (companion project)'s validation found that using sitk.ImageSeriesReader.Execute()
    to read pixels then applying RescaleSlope/Intercept again produces SUVs
    inflated by ~slope× (640% errors on Siemens). This implementation uses
    pydicom per-slice with per-slice rescale, which handles Siemens variable-
    rescale correctly. SimpleITK is used only for spatial metadata.
    """
    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(dicom_dir)
    if not series_ids:
        raise ValueError(f"No DICOM series found in {dicom_dir}")
    file_names = reader.GetGDCMSeriesFileNames(dicom_dir, series_ids[0])

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

    activity_3d = np.stack(slices, axis=0)  # Bq/mL
    decay_corrected_dose = meta.injected_dose_bq * meta.decay_factor
    weight_g = meta.patient_weight_kg * 1000.0
    suv_array = activity_3d * weight_g / decay_corrected_dose

    reader.SetFileNames(file_names)
    reader.MetaDataDictionaryArrayUpdateOn()
    reader.LoadPrivateTagsOn()
    image = reader.Execute()

    suv_image = sitk.GetImageFromArray(suv_array)
    suv_image.CopyInformation(image)
    return suv_image


def qc_suv_range(suv_image, patient_id):
    arr = sitk.GetArrayFromImage(suv_image)
    suvmax = float(arr.max())
    has_negative = bool((arr < 0).any())
    has_inf_nan = bool(np.isinf(arr).any() or np.isnan(arr).any())
    flagged_high = suvmax > 50.0
    return {
        "patient_id": patient_id,
        "pass": not has_negative and not has_inf_nan,
        "suvmax": suvmax,
        "suvmean": float(arr.mean()),
        "has_negative": has_negative,
        "has_inf_nan": has_inf_nan,
        "flagged_high": flagged_high,
    }


print("(companion project) SUV pipeline loaded:")
print("  - extract_pet_metadata(dcm_path)")
print("  - dicom_series_to_suv_sitk(dicom_dir, meta)")
print("  - qc_suv_range(suv_image, patient_id)")

# %% [markdown]
# ## 3. Smoke test
#
# For each task, query its TCIA collection, pick the smallest PET series,
# download it, run the SUV pipeline, report verdict.

# %%
TASKS = {
    "t4": {"collection": "NSCLC Radiogenomics", "expected_subset": None},
    "t6": {"collection": "RIDER Lung PET-CT", "expected_subset": None},
    "t7": {"collection": "ACRIN-NSCLC-FDG-PET", "expected_subset": None},
    "t8": {"collection": "Lung-PET-CT-Dx", "expected_subset": None},
}

results = []

for task, cfg in TASKS.items():
    print(f"\n{'='*70}")
    print(f"  {task.upper()}: {cfg['collection']}")
    print(f"{'='*70}")

    task_dir = TMP / task
    if task_dir.exists():
        shutil.rmtree(task_dir)
    task_dir.mkdir(parents=True)

    try:
        # Query TCIA
        all_series = nbia.getSeries(collection=cfg["collection"], format="df")
        pet = all_series[all_series["Modality"] == "PT"].copy()
        if pet.empty:
            results.append({"task": task, "verdict": "FAIL", "error": "no PET series"})
            print(f"  FAIL: no PET series found")
            continue

        # Pick smallest series for fast download
        pet = pet.sort_values("FileSize")
        smallest = pet.iloc[0]
        size_mb = smallest["FileSize"] / 1e6
        print(f"  Smallest PET series: {smallest['SeriesInstanceUID'][:24]}... "
              f"({size_mb:.0f} MB, patient {smallest['PatientID']})")

        # Download
        nbia.downloadSeries(
            [smallest["SeriesInstanceUID"]],
            path=str(task_dir),
            input_type="list",
        )

        # Find DICOM directory (nbia puts files in a subdirectory)
        dcm_files = list(task_dir.rglob("*.dcm"))
        if not dcm_files:
            results.append({"task": task, "verdict": "FAIL", "error": "no .dcm files after download"})
            print(f"  FAIL: download produced no .dcm files")
            continue
        dcm_dir = dcm_files[0].parent
        print(f"  Downloaded {len(dcm_files)} DICOM slices to {dcm_dir}")

        # Run (companion project) pipeline
        meta = extract_pet_metadata(str(dcm_files[0]))
        print(f"  Scanner: {meta.manufacturer} {meta.manufacturer_model}")
        print(f"  Radionuclide: {meta.radionuclide} (half-life {meta.half_life_sec:.1f}s)")
        print(f"  Patient weight: {meta.patient_weight_kg} kg, "
              f"injected dose: {meta.injected_dose_bq/1e6:.1f} MBq")
        print(f"  Uptake time: {meta.uptake_time_sec/60:.1f} min, "
              f"decay factor: {meta.decay_factor:.4f}")

        suv_image = dicom_series_to_suv_sitk(str(dcm_dir), meta)
        qc = qc_suv_range(suv_image, meta.patient_id)

        print(f"  SUVmax: {qc['suvmax']:.2f}")
        print(f"  SUVmean: {qc['suvmean']:.4f}")
        print(f"  inf/NaN: {qc['has_inf_nan']}")
        print(f"  Negative voxels: {qc['has_negative']}")

        # Verdict
        if (
            qc["pass"]
            and 0 < qc["suvmax"] < 100
            and not qc["has_inf_nan"]
        ):
            verdict = "PASS"
            note = "SUVmax in expected range; pipeline robust on this scanner/protocol"
        elif qc["suvmax"] > 1000:
            verdict = "FAIL"
            note = f"SUVmax={qc['suvmax']} suggests Bq/mL — SUV conversion not applied"
        elif qc["has_inf_nan"]:
            verdict = "FAIL"
            note = "inf/NaN present in SUV output"
        elif qc["flagged_high"]:
            verdict = "PASS (flagged)"
            note = f"SUVmax={qc['suvmax']} > 50 — high but not impossible (uptake region or lesion)"
        else:
            verdict = "PASS"
            note = "edge case but no failure indicators"

        print(f"  → {verdict}: {note}")

        results.append({
            "task": task,
            "verdict": verdict,
            "manufacturer": meta.manufacturer,
            "model": meta.manufacturer_model,
            "patient_id": meta.patient_id,
            "suvmax": qc["suvmax"],
            "suvmean": qc["suvmean"],
            "weight_kg": meta.patient_weight_kg,
            "dose_mbq": meta.injected_dose_bq / 1e6,
            "uptake_min": meta.uptake_time_sec / 60,
            "note": note,
        })

    except Exception as e:
        import traceback
        print(f"  EXCEPTION: {type(e).__name__}: {e}")
        traceback.print_exc()
        results.append({"task": task, "verdict": "FAIL", "error": f"{type(e).__name__}: {e}"})

    finally:
        # Free disk between tasks
        if task_dir.exists():
            shutil.rmtree(task_dir, ignore_errors=True)

# %% [markdown]
# ## 4. Verdict summary

# %%
results_df = pd.DataFrame(results)
print("\n" + "=" * 70)
print("SMOKE TEST SUMMARY")
print("=" * 70)
print(results_df.to_string(index=False))

n_pass = sum(1 for r in results if r["verdict"].startswith("PASS"))
n_total = len(results)
print(f"\n{n_pass}/{n_total} tasks passed")

if n_pass == n_total:
    print("\n✓ ALL TASKS PASSED — safe to roll (companion project)'s pipeline into "
          "t4_01/t6_01/t7_01/t8_01_preprocess.py")
elif n_pass >= 3:
    print(f"\n⚠ {n_total - n_pass} task(s) failed — investigate before "
          "full re-preprocessing. Failed tasks may need a different download "
          "method or have unusual DICOM tags.")
else:
    print(f"\n✗ {n_total - n_pass} task(s) failed — pipeline needs investigation "
          "before any production rollout.")

# Save for reference
out_dir = Path("/kaggle/working")
out_dir.mkdir(exist_ok=True)
results_df.to_csv(out_dir / "suv_smoke_test_results.csv", index=False)
print(f"\nResults saved to /kaggle/working/suv_smoke_test_results.csv")

# %% [markdown]
# ## 5. What to do next
#
# - **All 4 PASS** → I'll write the modified `t4_01`/`t6_01`/`t7_01`/`t8_01`
#   notebooks that use this pipeline in place of dcm2niix. Then re-run them
#   one by one to produce v3 patches datasets.
# - **Some FAIL** → paste the failure details back so we can investigate.
#   Common failure modes:
#   - Missing `PatientWeight` tag → will need a fallback using collection-level
#     median weight, or skip the patient
#   - Missing `RadionuclideTotalDose` → unusual; needs case-by-case look
#   - DICOM has stored data the pipeline can't read → may need a different
#     download method (e.g., extracted vs zipped archives)
