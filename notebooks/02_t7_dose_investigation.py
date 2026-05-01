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
# # T7 (ACRIN-NSCLC-FDG-PET) Dose Metadata Investigation
#
# **Runtime:** CPU | **Internet:** ON | **Time:** ~15 min
#
# The smoke test (01_suv_smoke_test.py) found that T7's smallest PET series
# has a **`RadionuclideTotalDose` of 407 Bq** — six orders of magnitude too
# low for a real FDG injection. This produced an inflated SUVmax of 473,510.
#
# **Question:** is this a systematic ACRIN issue, or just a quirk of the
# smallest series?
#
# **Approach:** download 6 T7 series spanning the size distribution (smallest,
# 25th, 50th, 75th, largest percentile, plus one more), inspect the dose
# metadata for each, and characterise.
#
# **Decision tree:**
# - If all 6 series have dose < 10 MBq → systematic issue, ACRIN stores doses
#   in non-standard units. Need to add unit-correction logic before rollout.
# - If 5/6 have normal doses (185-555 MBq) → smallest series is just unusual.
#   Add a dose-sanity-check guard to the pipeline (skip series with dose <
#   10 MBq), proceed with rollout.
# - If results are scattered → need per-scanner or per-vendor handling.

# %% [markdown]
# ## 1. Setup

# %%
import os
import shutil
import datetime
from pathlib import Path

import numpy as np
import pandas as pd

!pip install -q tcia-utils SimpleITK pydicom

import SimpleITK as sitk
import pydicom
from tcia_utils import nbia

TMP = Path("/tmp/t7_investigation")
TMP.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## 2. Query T7 collection and pick 6 representative series

# %%
all_series = nbia.getSeries(collection="ACRIN-NSCLC-FDG-PET", format="df")
pet = all_series[all_series["Modality"] == "PT"].copy()
pet = pet.sort_values("FileSize").reset_index(drop=True)
n = len(pet)
print(f"T7 has {n} PET series. Size range: "
      f"{pet['FileSize'].min()/1e6:.0f} MB to {pet['FileSize'].max()/1e6:.0f} MB")

# Pick 6 series at different size percentiles
indices = [
    0,                 # smallest (the one that failed in smoke test)
    int(n * 0.20),
    int(n * 0.40),
    int(n * 0.60),
    int(n * 0.80),
    n - 1,             # largest
]
sample = pet.iloc[indices].copy()
print(f"\nSampled {len(sample)} series:")
for _, row in sample.iterrows():
    print(f"  {row['SeriesInstanceUID'][:24]}...  "
          f"{row['FileSize']/1e6:6.0f} MB  patient {row['PatientID']}")

# %% [markdown]
# ## 3. Inspect dose metadata WITHOUT computing SUV
#
# Just download each series, read the dose tag, report. This is faster than
# running the full SUV pipeline and gives us the diagnostic info we need.

# %%
def inspect_dose(dcm_path):
    """Read just the metadata fields relevant to SUV scaling."""
    ds = pydicom.dcmread(dcm_path, stop_before_pixels=True)
    radio_seq = getattr(ds, "RadiopharmaceuticalInformationSequence", None)
    if not radio_seq:
        return {"error": "no RadiopharmaceuticalInformationSequence"}
    radio = radio_seq[0]
    return {
        "patient_id": str(getattr(ds, "PatientID", "UNKNOWN")),
        "manufacturer": str(getattr(ds, "Manufacturer", "Unknown")),
        "model": str(getattr(ds, "ManufacturerModelName", "Unknown")),
        "weight_kg": float(getattr(ds, "PatientWeight", float("nan"))),
        "dose_raw": float(getattr(radio, "RadionuclideTotalDose", float("nan"))),
        "dose_units_field": str(getattr(radio, "RadionuclideTotalDoseUnits", "missing")),
        "half_life_sec": float(getattr(radio, "RadionuclideHalfLife", float("nan"))),
        "rescale_slope": float(getattr(ds, "RescaleSlope", 1.0)),
        "rescale_intercept": float(getattr(ds, "RescaleIntercept", 0.0)),
        "rescale_units": str(getattr(ds, "Units", "missing")),  # often 'BQML' or 'CNTS'
        "decay_correction": str(getattr(ds, "DecayCorrection", "missing")),
        "raw_pixel_dtype": str(ds.get("BitsStored", "?")),
    }


inspections = []
for idx, row in sample.iterrows():
    series_dir = TMP / f"series_{idx:03d}"
    if series_dir.exists():
        shutil.rmtree(series_dir)
    series_dir.mkdir(parents=True)

    print(f"\n--- Series {idx} ({row['FileSize']/1e6:.0f} MB) ---")
    try:
        nbia.downloadSeries(
            [row["SeriesInstanceUID"]],
            path=str(series_dir),
            input_type="list",
        )
        dcm_files = list(series_dir.rglob("*.dcm"))
        if not dcm_files:
            print(f"  Download produced no DICOM files")
            inspections.append({"index": idx, "size_mb": row["FileSize"]/1e6,
                                "error": "no DICOM after download"})
            continue

        info = inspect_dose(str(dcm_files[0]))
        info["index"] = idx
        info["size_mb"] = row["FileSize"] / 1e6
        info["n_slices"] = len(dcm_files)

        print(f"  Patient: {info.get('patient_id')}")
        print(f"  Scanner: {info.get('manufacturer')} {info.get('model')}")
        print(f"  Weight: {info.get('weight_kg')} kg")
        print(f"  Dose RAW: {info.get('dose_raw'):.3e}  ({info.get('dose_raw')/1e6:.3f} MBq)")
        print(f"  Dose units field: {info.get('dose_units_field')}")
        print(f"  Half-life: {info.get('half_life_sec'):.1f} sec")
        print(f"  Rescale: slope={info.get('rescale_slope')}  "
              f"intercept={info.get('rescale_intercept')}  units={info.get('rescale_units')}")

        # Verdict on the dose
        dose_mbq = info.get("dose_raw", 0) / 1e6
        if 50 < dose_mbq < 1000:
            verdict = "✓ NORMAL"
        elif dose_mbq < 0.001:
            verdict = "✗ IMPOSSIBLY LOW (likely unit issue)"
        elif 50 < info.get("dose_raw", 0) < 1000:
            verdict = "⚠ MAYBE STORED IN MBq (raw value 50-1000)"
        else:
            verdict = "⚠ UNUSUAL"
        print(f"  Verdict: {verdict}")
        info["verdict"] = verdict

        inspections.append(info)

    except Exception as e:
        print(f"  EXCEPTION: {type(e).__name__}: {e}")
        inspections.append({"index": idx, "size_mb": row["FileSize"]/1e6,
                            "error": f"{type(e).__name__}: {e}"})

    finally:
        if series_dir.exists():
            shutil.rmtree(series_dir, ignore_errors=True)

# %% [markdown]
# ## 4. Summary

# %%
df = pd.DataFrame(inspections)
print("\n" + "=" * 80)
print("T7 DOSE METADATA SUMMARY")
print("=" * 80)
display_cols = ["index", "size_mb", "patient_id", "manufacturer",
                "weight_kg", "dose_raw", "verdict"]
have_cols = [c for c in display_cols if c in df.columns]
print(df[have_cols].to_string(index=False))

if "verdict" in df.columns:
    n_normal = (df["verdict"].str.startswith("✓", na=False)).sum()
    n_low = (df["verdict"].str.contains("IMPOSSIBLY LOW", na=False)).sum()
    n_total = len(df)
    print(f"\n{n_normal}/{n_total} series have normal doses, "
          f"{n_low}/{n_total} have impossibly-low doses")

    if n_low == n_total:
        print("\n✗ SYSTEMATIC ISSUE: all sampled T7 series have malformed doses.")
        print("  ACRIN-NSCLC-FDG-PET likely stores doses in MBq not Bq, "
              "or uses a non-standard unit field.")
        print("  Action: pipeline needs unit-correction logic OR T7 needs "
              "alternative SUV computation path.")
    elif n_low == 0:
        print("\n✓ NO SYSTEMATIC ISSUE: all sampled doses look normal.")
        print("  The smoke-test failure on smallest series was an unlucky pick.")
        print("  Action: add dose-sanity-check guard to pipeline (skip series "
              "with dose < 10 MBq), proceed with rollout.")
    else:
        print(f"\n⚠ MIXED RESULT: {n_low}/{n_total} series broken. Need to "
              "characterise which scanner/protocol/year is affected.")

# Save
out_dir = Path("/kaggle/working")
df.to_csv(out_dir / "t7_dose_investigation.csv", index=False)
print(f"\nFull metadata saved to /kaggle/working/t7_dose_investigation.csv")
