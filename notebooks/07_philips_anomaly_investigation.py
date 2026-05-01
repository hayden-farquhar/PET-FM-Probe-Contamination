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
# # Philips Medical Systems SUVmax 6.7 Anomaly Investigation
#
# **Runtime:** CPU | **Internet:** ON | **Time:** ~30 min | **GPU:** Not needed
#
# Across T4 (9 patients) and T7 (55 patients), Philips Medical Systems
# (no MIMvista suffix) scans consistently show SUVmax median ≈ 6.7 — too low
# to fit BQML→SUV (would predict ~30) and too high to fit GML double-conversion
# (would predict ~0.006). This notebook downloads a representative sample of
# affected Philips patients, inspects their DICOM tags, and identifies what
# DICOM convention differs from BQML.
#
# **Possible explanations** to test:
# 1. Non-standard `Units` value (e.g., `BQM`, `MBQ`, `KBQ`) treated as default-BQML
# 2. RescaleSlope reported in unusual units (e.g., 1e-3× expected)
# 3. PatientWeight in pounds (factor of 2.2× — wrong direction)
# 4. RadionuclideTotalDose stored in mCi (factor of 37 — wrong direction)
# 5. Per-slice rescale variation (Siemens-style) breaking the per-slice handling
#
# **Decision tree:**
# - If `Units = "BQML"` and rescale looks normal → biology, not pipeline
# - If `Units` is something else → add a branch to `dicom_series_to_suv_sitk`
# - If rescale slope is 1e-3 of expected → Philips uses `mBq/mL` convention

# %% [markdown]
# ## 1. Setup

# %%
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

!pip install -q tcia-utils SimpleITK pydicom

import SimpleITK as sitk
import pydicom
from tcia_utils import nbia

TMP = Path("/tmp/philips_investigation")
TMP.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## 2. Identify Philips patients across T4 and T7
#
# These were tagged "Philips Medical Systems" (without "/MIMvista" suffix) in
# the v3 preprocessing logs. The preprocessing_log.csv files in each patches
# dataset have manufacturer info per patient.

# %%
philips_targets = []  # list of dicts: {task, collection, patient_id}

for task, collection in [("t4", "NSCLC Radiogenomics"), ("t7", "ACRIN-NSCLC-FDG-PET")]:
    log_path = next(Path("/kaggle/input").rglob(f"pet-fm-bench-{task}-patches-v3/{task}_*/preprocessing_log.csv"), None)
    if log_path is None:
        print(f"[{task}] preprocessing_log.csv not found; will fall back to TCIA query")
        continue
    log = pd.read_csv(log_path)
    if "manufacturer" not in log.columns:
        print(f"[{task}] log lacks manufacturer column")
        continue
    affected = log[
        (log["manufacturer"] == "Philips Medical Systems")
        & (log["status"] == "ok")
    ].copy()
    print(f"[{task}] Philips Medical Systems (no MIMvista): {len(affected)} patients")
    # Sample 3 patients per task for the deep DICOM dive
    sample = affected.sample(n=min(3, len(affected)), random_state=42)
    for _, row in sample.iterrows():
        philips_targets.append({
            "task": task,
            "collection": collection,
            "patient_id": row["patient_id"],
            "suv_max": row.get("suv_max"),
        })

# Also pick a known-clean Siemens patient as a baseline comparison (T8 is all Siemens)
t8_log = next(Path("/kaggle/input").rglob("pet-fm-bench-t8-patches-v3/t8_*/preprocessing_log.csv"), None)
if t8_log is not None:
    t8 = pd.read_csv(t8_log)
    siemens_clean = t8[t8["status"] == "ok"].sample(n=1, random_state=42).iloc[0]
    philips_targets.append({
        "task": "t8",
        "collection": "Lung-PET-CT-Dx",
        "patient_id": siemens_clean["patient_id"],
        "suv_max": siemens_clean.get("suv_max"),
        "_baseline": True,
    })

print(f"\n{len(philips_targets)} target patients to inspect:")
for t in philips_targets:
    print(f"  {t['task']} / {t['patient_id']}: prior SUVmax={t.get('suv_max'):.2f}"
          if t.get('suv_max') else f"  {t['task']} / {t['patient_id']}")

# %% [markdown]
# ## 3. Download DICOMs and inspect tags

# %%
DICOM_TAGS_TO_REPORT = [
    "Manufacturer", "ManufacturerModelName", "Modality",
    "Units",                                # BQML vs GML vs other
    "DecayCorrection",                      # START vs ADMIN
    "RescaleSlope", "RescaleIntercept",
    "PatientWeight", "PatientSize", "PatientSex",
    "AcquisitionTime", "SeriesTime",
    "SoftwareVersions",
]


def query_largest_pet_series(collection, patient_id):
    """Find the largest AC PET series for a specific patient."""
    series = nbia.getSeries(collection=collection, patientId=patient_id, format="df")
    if len(series) == 0:
        return None
    pet = series[series["Modality"] == "PT"].copy()
    # Exclude NAC
    nac_kw = ["NAC", "NO_AC", "Uncorrected", "NoAC"]
    pet = pet[~pet["SeriesDescription"].fillna("").apply(
        lambda d: any(k.upper() in str(d).upper() for k in nac_kw)
    )]
    if len(pet) == 0:
        return None
    return pet.sort_values("ImageCount", ascending=False).iloc[0]


def inspect_dicom_tags(dcm_path):
    """Read one DICOM file and extract relevant tags."""
    ds = pydicom.dcmread(dcm_path, stop_before_pixels=True)
    info = {}
    for tag in DICOM_TAGS_TO_REPORT:
        v = getattr(ds, tag, None)
        info[tag] = str(v) if v is not None else "MISSING"

    # Radiopharmaceutical
    radio_seq = getattr(ds, "RadiopharmaceuticalInformationSequence", None)
    if radio_seq and len(radio_seq) > 0:
        radio = radio_seq[0]
        info["RadionuclideTotalDose_Bq"] = str(getattr(radio, "RadionuclideTotalDose", "MISSING"))
        info["RadionuclideTotalDoseUnits"] = str(getattr(radio, "RadionuclideTotalDoseUnits", "MISSING"))
        info["RadionuclideHalfLife"] = str(getattr(radio, "RadionuclideHalfLife", "MISSING"))
        info["RadiopharmaceuticalStartTime"] = str(getattr(radio, "RadiopharmaceuticalStartTime", "MISSING"))
    else:
        info["RadionuclideTotalDose_Bq"] = "RPI_SEQ_MISSING"

    # Inspect a few slices' RescaleSlope to detect Siemens-style per-slice variation
    info["raw_pixel_min"] = float(ds.pixel_array.min()) if "PixelData" in ds else None
    info["raw_pixel_max"] = float(ds.pixel_array.max()) if "PixelData" in ds else None

    return info


inspections = []
for t in philips_targets:
    pid = t["patient_id"]
    task = t["task"]
    coll = t["collection"]
    is_baseline = t.get("_baseline", False)

    label = f"BASELINE-{task}" if is_baseline else f"{task}"
    print(f"\n--- {label} / {pid} ---")

    series_meta = query_largest_pet_series(coll, pid)
    if series_meta is None:
        print(f"  No PET series found for {pid}")
        continue

    series_dir = TMP / f"{task}_{pid}"
    if series_dir.exists():
        shutil.rmtree(series_dir)
    series_dir.mkdir(parents=True)

    nbia.downloadSeries(
        [series_meta["SeriesInstanceUID"]],
        path=str(series_dir),
        input_type="list",
    )
    dcm_files = list(series_dir.rglob("*.dcm"))
    if not dcm_files:
        print(f"  Download failed")
        continue

    info = inspect_dicom_tags(str(dcm_files[0]))
    info["task"] = task
    info["patient_id"] = pid
    info["is_baseline"] = is_baseline
    info["n_slices"] = len(dcm_files)

    # Also check rescale slope variation across slices
    slopes = []
    for f in dcm_files[:10]:  # first 10 slices
        ds_s = pydicom.dcmread(f, stop_before_pixels=True)
        slopes.append(float(getattr(ds_s, "RescaleSlope", 1.0)))
    info["rescale_slope_first_10"] = ", ".join(f"{s:.4g}" for s in slopes)
    info["rescale_slope_unique_first_10"] = len(set(slopes))

    print(f"  Manufacturer: {info['Manufacturer']}")
    print(f"  Model: {info['ManufacturerModelName']}")
    print(f"  Units: {info['Units']}")
    print(f"  RescaleSlope (first slice): {info['RescaleSlope']}")
    print(f"  RescaleSlope unique across first 10 slices: {info['rescale_slope_unique_first_10']}")
    print(f"  PatientWeight: {info['PatientWeight']} (kg expected)")
    print(f"  RadionuclideTotalDose: {info['RadionuclideTotalDose_Bq']}")
    print(f"  RadionuclideTotalDoseUnits: {info['RadionuclideTotalDoseUnits']}")
    print(f"  Raw pixel range: [{info['raw_pixel_min']}, {info['raw_pixel_max']}]")

    inspections.append(info)

    shutil.rmtree(series_dir, ignore_errors=True)

# %% [markdown]
# ## 4. Cross-vendor comparison table

# %%
df = pd.DataFrame(inspections)
out_path = Path("/kaggle/working/philips_dicom_audit.csv")
df.to_csv(out_path, index=False)

print(f"\n{'='*70}")
print(f"DICOM TAG COMPARISON")
print(f"{'='*70}\n")

display_cols = [
    "task", "patient_id", "Manufacturer", "Units",
    "RescaleSlope", "rescale_slope_unique_first_10",
    "PatientWeight", "RadionuclideTotalDose_Bq",
    "raw_pixel_max",
]
have = [c for c in display_cols if c in df.columns]
with pd.option_context("display.width", 200, "display.max_colwidth", 30):
    print(df[have].to_string(index=False))

# %% [markdown]
# ## 5. Diagnosis hypotheses

# %%
print(f"\n{'='*70}")
print(f"DIAGNOSIS")
print(f"{'='*70}\n")

if df.empty:
    print("No data captured.")
else:
    # Look for the smoking gun
    philips_rows = df[df["Manufacturer"].str.contains("Philips", na=False)
                      & ~df["Manufacturer"].str.contains("MIMvista", na=False)]
    siemens_rows = df[df["Manufacturer"].str.contains("SIEMENS", na=False, case=False)]

    if not philips_rows.empty:
        print(f"\nPhilips (no MIMvista) Units values:")
        print(philips_rows["Units"].value_counts().to_dict())
        print(f"\nPhilips RescaleSlope values:")
        print(philips_rows["RescaleSlope"].value_counts().to_dict())
        print(f"\nPhilips RadionuclideTotalDose_Bq:")
        print(philips_rows["RadionuclideTotalDose_Bq"].tolist())
        print(f"\nPhilips raw_pixel_max:")
        print(philips_rows["raw_pixel_max"].tolist())

    if not siemens_rows.empty:
        print(f"\nSiemens (baseline) Units: {siemens_rows['Units'].tolist()}")
        print(f"Siemens RescaleSlope: {siemens_rows['RescaleSlope'].tolist()}")
        print(f"Siemens raw_pixel_max: {siemens_rows['raw_pixel_max'].tolist()}")

    # Quantitative check: if Philips raw_pixel_max × slope is 5x lower than
    # Siemens-equivalent, the pipeline is correct and biology is the answer.
    # If 5x higher, the pipeline is missing a unit conversion.
    print(f"\nLikely diagnosis (manual interpretation needed):")
    print(f"  - If Philips Units == 'BQML' and RescaleSlope is unremarkable (similar order to Siemens):")
    print(f"      → Real biology. SUVmax 6.7 reflects low-uptake patients in this Philips cohort.")
    print(f"  - If Philips Units != 'BQML' (e.g., 'BQM', 'CNTS', or empty):")
    print(f"      → Add a Units branch to dicom_series_to_suv_sitk.")
    print(f"  - If Philips RescaleSlope is ~1000x lower than Siemens:")
    print(f"      → Philips uses mBq/mL convention; multiply by 1000 in pipeline.")

# %% [markdown]
# ## 6. Done
#
# Investigation outputs to `/kaggle/working/philips_dicom_audit.csv`. Inspect
# the table, identify the diagnosis, and either:
# - Document Philips no-MIMvista as a "real biology" subgroup in the manuscript
# - Patch `dicom_series_to_suv_sitk` to handle the Philips convention and re-run
#   T4/T7 v3 preprocessing
#
# This investigation is non-blocking for the main analysis pipeline.
