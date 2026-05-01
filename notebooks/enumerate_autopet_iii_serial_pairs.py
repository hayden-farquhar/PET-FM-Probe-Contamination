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
# # AutoPET-III Multi-Series Interval Enumerator (A11 Decision Gate)
#
# **PET-FM-Bench** | DOI: [10.17605/OSF.IO/DQ2JA](https://doi.org/10.17605/OSF.IO/DQ2JA)
#
# **PLATFORM:** Google Colab (needs Drive mount for AutoPET-III DICOM headers).
# **Runtime:** CPU | **Time:** ~5-8 min for 497 series header reads.
#
# **Purpose:** Implements the decision gate for amendment A11 (T5 within-cohort
# PSMA test-retest as exploratory secondary analysis). Per A11:
#
# > Pre-specified inclusion threshold: **≥50 patients** with ≥1 same-tracer
# > (18F+18F or 68Ga+68Ga) pair within an **8-week interval window**.
# >
# > If threshold met → proceed with T5 within-cohort test-retest.
# > If threshold not met → report interval-distribution table as descriptive
# > supplementary; do NOT run test-retest.
#
# **Output:** `autopet_iii_pair_intervals.csv` — per-pair table with case_id,
# series_uid_a, series_uid_b, study_date_a, study_date_b, interval_days,
# tracer_a, tracer_b, same_tracer flag. Plus a summary table at the bottom.
#
# **Methodology:** mirrors (companion project)'s `enumerate_autopet_i_serial_pairs.py` for
# cross-project consistency. Uses parsed DICOM `StudyDate` per series.

# %% [markdown]
# ## 1. Setup

# %%
# !pip install -q pandas pydicom

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import pydicom

try:
    from google.colab import drive
    drive.mount("/content/drive", force_remount=False)
    DRIVE_ROOT = Path("/content/drive/My Drive/petfm_data/autopet_iii")
except ImportError:
    DRIVE_ROOT = Path("/path/to/manually/mounted/autopet_iii")
    print("⚠ Not on Colab — set DRIVE_ROOT manually.")

# Output dir for the enumeration result
OUT_DIR = Path("/content")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## 2. Load the reviewed lesion parquet (external companion-project artefact)
#
# Provides the (case_id, series_uid) pairs we need to enumerate. From
# `pet-fm-bench-t1-patches-v3` Kaggle dataset OR direct from Drive
# ((companion project) has it locally).

# %%
# Try multiple candidate locations
PARQUET_CANDIDATES = [
    DRIVE_ROOT / "lesion_tables" / "autopet_iii_lesions_reviewed.parquet",
    Path("/kaggle/input/pet-fm-bench-autopet-iii-lesions/autopet_iii_lesions_reviewed.parquet"),
    DRIVE_ROOT.parent / "p79_data_interim" / "autopet_iii_lesions_reviewed.parquet",
]
PARQUET = next((p for p in PARQUET_CANDIDATES if p.exists()), None)

if PARQUET is None:
    raise FileNotFoundError(
        "Could not locate autopet_iii_lesions_reviewed.parquet. "
        "Either copy from (companion project) local at "
        "`79 Conformal SUV Theranostic/data/interim/lesion_tables/` "
        "to one of the candidate paths, OR pull from OSF j5ry4."
    )

lesions = pd.read_parquet(PARQUET)
# Filter to reviewed cohort (matches A10 specification)
lesions = lesions[lesions["section_3_9_excluded"] == False].copy()
print(f"Loaded reviewed lesions: {len(lesions)} rows")
print(f"Unique cases: {lesions['case_id'].nunique()}")
print(f"Unique series: {lesions['series_uid'].nunique()}")

# %% [markdown]
# ## 3. Enumerate (case_id, series_uid) pairs

# %%
case_series = (
    lesions[["case_id", "series_uid", "radionuclide"]]
    .drop_duplicates(subset=["case_id", "series_uid"])
    .reset_index(drop=True)
)
print(f"\nUnique (case_id, series_uid) tuples: {len(case_series)}")

# Distribution: series count per case
series_per_case = case_series.groupby("case_id").size()
print(f"\nSeries-per-case distribution:")
print(series_per_case.value_counts().sort_index())
print(f"\nCases with ≥2 series: {(series_per_case >= 2).sum()}")

# %% [markdown]
# ## 4. Read StudyDate from DICOM headers per series

# %%
def find_first_dicom_for_series(series_uid, root=DRIVE_ROOT):
    """Locate any DICOM file under DRIVE_ROOT for the given series_uid.
    Heterogeneous storage: 1,107 series as `{uid}.zip`, 684 as `{uid}/`."""
    # Direct directory
    direct = root / series_uid
    if direct.is_dir():
        files = list(direct.glob("*.dcm"))
        if files:
            return files[0]
    # Zip-extracted location (if (companion project) extracted lazily)
    extracted = root / f"{series_uid}_extracted"
    if extracted.is_dir():
        files = list(extracted.glob("*.dcm"))
        if files:
            return files[0]
    # Direct zip — extract one slice header
    zip_path = root / f"{series_uid}.zip"
    if zip_path.is_file():
        import zipfile
        with zipfile.ZipFile(zip_path) as zf:
            members = [m for m in zf.namelist() if m.endswith(".dcm")]
            if members:
                # Read header from first slice without extracting whole zip
                with zf.open(members[0]) as fp:
                    return ("zip-stream", zip_path, members[0])
    return None


def parse_study_date(loc):
    if loc is None:
        return None
    if isinstance(loc, tuple):
        # ("zip-stream", zip_path, member_name)
        import zipfile
        with zipfile.ZipFile(loc[1]) as zf, zf.open(loc[2]) as fp:
            ds = pydicom.dcmread(fp, stop_before_pixels=True)
    else:
        ds = pydicom.dcmread(loc, stop_before_pixels=True)
    sd = getattr(ds, "StudyDate", None)
    if not sd:
        return None
    try:
        return datetime.strptime(str(sd).strip(), "%Y%m%d")
    except ValueError:
        return None


# %% [markdown]
# ## 5. Build the per-series StudyDate table

# %%
from tqdm.auto import tqdm

records = []
for _, row in tqdm(case_series.iterrows(), total=len(case_series),
                   desc="Reading DICOM StudyDate"):
    suid = row["series_uid"]
    loc = find_first_dicom_for_series(suid)
    sd = parse_study_date(loc) if loc is not None else None
    records.append({
        "case_id": row["case_id"],
        "series_uid": suid,
        "radionuclide": row["radionuclide"],
        "study_date": sd,
        "study_date_str": sd.strftime("%Y-%m-%d") if sd else None,
        "loc_found": loc is not None,
    })

series_dates = pd.DataFrame(records)
n_with_date = series_dates["study_date"].notna().sum()
print(f"\nSeries with parseable StudyDate: {n_with_date}/{len(series_dates)}")
if n_with_date < len(series_dates):
    missing = series_dates[series_dates["study_date"].isna()]
    print(f"\nSeries missing StudyDate (first 5):")
    print(missing.head().to_string())

# %% [markdown]
# ## 6. Canonicalise radionuclide for tracer-match comparison
#
# (companion project) documented two encodings for 68Ga: `Ga-68` and `^68^Gallium`. Same
# nuclide; treat as equal for tracer-match purposes.

# %%
RADIONUCLIDE_CANON = {
    "18F": "18F",
    "^18^Fluorine": "18F",
    "F-18": "18F",
    "Fluorine-18": "18F",
    "68Ga": "68Ga",
    "^68^Gallium": "68Ga",
    "Ga-68": "68Ga",
    "Gallium-68": "68Ga",
}
series_dates["nuclide_canon"] = series_dates["radionuclide"].map(
    lambda r: RADIONUCLIDE_CANON.get(str(r), str(r))
)
print(f"\nCanonicalised radionuclide distribution:")
print(series_dates["nuclide_canon"].value_counts().to_string())

# %% [markdown]
# ## 7. Enumerate same-patient pairs (chronologically ordered, no double-count)

# %%
pair_records = []
for case_id, grp in series_dates[series_dates["study_date"].notna()].groupby("case_id"):
    if len(grp) < 2:
        continue
    grp_sorted = grp.sort_values("study_date").reset_index(drop=True)
    for i in range(len(grp_sorted)):
        for j in range(i + 1, len(grp_sorted)):
            row_a = grp_sorted.iloc[i]
            row_b = grp_sorted.iloc[j]
            interval = (row_b["study_date"] - row_a["study_date"]).days
            pair_records.append({
                "case_id": case_id,
                "series_uid_a": row_a["series_uid"],
                "series_uid_b": row_b["series_uid"],
                "study_date_a": row_a["study_date_str"],
                "study_date_b": row_b["study_date_str"],
                "interval_days": interval,
                "nuclide_a": row_a["nuclide_canon"],
                "nuclide_b": row_b["nuclide_canon"],
                "same_tracer": row_a["nuclide_canon"] == row_b["nuclide_canon"],
            })

pairs = pd.DataFrame(pair_records)
print(f"\nTotal pairs across all multi-series patients: {len(pairs)}")
print(f"Same-tracer pairs: {pairs['same_tracer'].sum()}")
print(f"Cross-tracer pairs: {(~pairs['same_tracer']).sum()}")

# %% [markdown]
# ## 8. Window-sweep diagnostic (cumulative pair counts at each interval threshold)

# %%
WINDOWS_WEEKS = [4, 6, 8, 10, 12, 16, 26, 52]
diag = []
for wk in WINDOWS_WEEKS:
    cap = wk * 7
    in_window = pairs[pairs["interval_days"] <= cap]
    same = in_window[in_window["same_tracer"]]
    diag.append({
        "window_weeks": wk,
        "window_days": cap,
        "n_pairs_total": len(in_window),
        "n_pairs_same_tracer": len(same),
        "n_unique_patients_same_tracer": same["case_id"].nunique(),
    })
diag_df = pd.DataFrame(diag)
print("\nWindow-sweep diagnostic:")
print(diag_df.to_string(index=False))

# %% [markdown]
# ## 9. A11 decision gate

# %%
A11_GATE_WINDOW_WEEKS = 8
A11_GATE_THRESHOLD_PATIENTS = 50

n_qualifying = diag_df.loc[
    diag_df["window_weeks"] == A11_GATE_WINDOW_WEEKS,
    "n_unique_patients_same_tracer",
].iloc[0]

print(f"\n{'=' * 60}")
print(f"A11 DECISION GATE")
print(f"{'=' * 60}")
print(f"Window: {A11_GATE_WINDOW_WEEKS} weeks (matches the companion-project §3.5 stable-disease window)")
print(f"Threshold: ≥{A11_GATE_THRESHOLD_PATIENTS} patients with same-tracer pair")
print(f"Observed: {n_qualifying} patients with same-tracer pair within window")

if n_qualifying >= A11_GATE_THRESHOLD_PATIENTS:
    print(f"\n✓ GATE MET — proceed with T5 within-cohort test-retest analysis")
    print(f"  per A11. Qualifying-pair table written to "
          f"autopet_iii_qualifying_pairs.csv")
    qualifying = pairs[
        (pairs["interval_days"] <= A11_GATE_WINDOW_WEEKS * 7)
        & (pairs["same_tracer"])
    ].copy()
    qualifying.to_csv(OUT_DIR / "autopet_iii_qualifying_pairs.csv", index=False)
else:
    print(f"\n✗ GATE NOT MET — interval-distribution table reported as descriptive "
          f"supplementary only. T5 within-cohort test-retest NOT run "
          f"(per A11 pre-specified non-proceed branch).")

# %% [markdown]
# ## 10. Save outputs

# %%
pairs.to_csv(OUT_DIR / "autopet_iii_pair_intervals.csv", index=False)
diag_df.to_csv(OUT_DIR / "autopet_iii_window_sweep.csv", index=False)
series_dates.to_csv(OUT_DIR / "autopet_iii_series_dates.csv", index=False)

decision = {
    "freeze_timestamp_utc": datetime.utcnow().isoformat(timespec="seconds"),
    "amendment_ref": "A11",
    "gate_window_weeks": A11_GATE_WINDOW_WEEKS,
    "gate_threshold_patients": A11_GATE_THRESHOLD_PATIENTS,
    "n_unique_patients_same_tracer_in_window": int(n_qualifying),
    "gate_met": bool(n_qualifying >= A11_GATE_THRESHOLD_PATIENTS),
    "n_pairs_total": int(len(pairs)),
    "n_same_tracer_pairs_total": int(pairs["same_tracer"].sum()),
    "n_series_with_studydate": int(n_with_date),
    "n_series_total": int(len(series_dates)),
}
with open(OUT_DIR / "autopet_iii_a11_decision.json", "w") as f:
    json.dump(decision, f, indent=2)

print(f"\nWrote:")
for f in ["autopet_iii_pair_intervals.csv", "autopet_iii_window_sweep.csv",
          "autopet_iii_series_dates.csv", "autopet_iii_a11_decision.json"]:
    p = OUT_DIR / f
    if p.exists():
        print(f"  {p} ({p.stat().st_size/1024:.1f} KB)")

# %% [markdown]
# ## 11. Done
#
# Bank decision JSON in PROGRESS.md and amendment_log.md (as a follow-on entry
# under A11). If gate met: proceed with T5 test-retest analysis (Phase 5
# secondary). If not met: document descriptively in supplementary; T5 stays
# zero-shot only.
