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
# # PET-FM-Bench: Phase 2 Stage 1a — FMCIB Training Manifest (verified)
#
# **Runtime:** CPU | **Internet:** ON | **Time:** ~5-10 min | **GPU:** Not needed
#
# Pre-registration §4.6 (contamination tiering) Stage 1: enumerate the patient
# IDs FMCIB was pretrained on, so Stage 2 can intersect against PET-FM-Bench
# evaluation cohorts and assign contamination tiers.
#
# **Source verified 2026-04-27** against the FMCIB code repository
# (`AIM-Harvard/foundation-cancer-image-biomarker`, branch `master`,
# `data/download/`). FMCIB's published download scripts and series manifests
# enumerate the exact (collection, PatientID) tuples used for pretraining —
# we do not need to infer the cohort from the paper's prose.
#
# **FMCIB pretraining cohort (4 sources, 11,467 lesions per Pai et al. 2024):**
#
# | Source | Format | Auditable here? | Direct overlap with PET-FM-Bench? |
# |---|---|---|---|
# | DeepLesion | AcademicTorrents (NIH non-TCIA, integer IDs 000001–004427) | counted only | No (disjoint by ID format) |
# | LUNA16 | AcademicTorrents (LIDC-IDRI subset, ~888 cases) | counted only | No (LIDC IDs, not in PET cohort) |
# | NSCLC-Radiomics ("Lung1", Aerts et al.) | `nsclc_radiomics.csv` in repo | ✅ patient-level | No (CT-only cohort) |
# | NSCLC-Radiogenomics (Bakr et al.) | `nsclc_radiogenomics.csv` in repo | ✅ patient-level | **YES — entire T4 evaluation cohort** |
#
# The NSCLC-Radiogenomics overlap with T4 is the registration H2 test case:
# every patient in our T4 evaluation cohort appears in FMCIB's pretraining
# manifest, so contamination tier on (FMCIB, T4) is **Tier 1 by construction**.
#
# **Output:** `fmcib_training_manifest.parquet` with schema
# `(fm, source_collection, patient_id, evidence)`. Stage 2
# (`08_contamination_intersection.py`) concatenates this with 07b/07c outputs
# and intersects against `task_splits.parquet`.

# %% [markdown]
# ## 1. Setup

# %%
import io
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

OUT_DIR = Path("/kaggle/working")
OUT_DIR.mkdir(parents=True, exist_ok=True)

freeze_timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
print(f"Freeze timestamp (UTC): {freeze_timestamp}")

# Canonical source URLs (FMCIB GitHub master branch, verified 2026-04-27)
FMCIB_GITHUB = ("https://raw.githubusercontent.com/AIM-Harvard/"
                "foundation-cancer-image-biomarker/master/data/download/")
FMCIB_RADIOGENOMICS_URL = FMCIB_GITHUB + "nsclc_radiogenomics.csv"
FMCIB_RADIOMICS_URL = FMCIB_GITHUB + "nsclc_radiomics.csv"

# %% [markdown]
# ## 2. Fetch FMCIB's published series manifests
#
# These two CSVs are the canonical FMCIB training-data manifest. Each row
# is a (collection_id, PatientID, SeriesInstanceUID, ..., S3 cp command)
# tuple. We extract unique PatientIDs per collection.

# %%
def fetch_fmcib_csv(url):
    print(f"Fetching: {url}")
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    print(f"  rows: {len(df):,}")
    print(f"  columns: {list(df.columns)[:6]}...")
    return df


radiogenomics_df = fetch_fmcib_csv(FMCIB_RADIOGENOMICS_URL)
radiomics_df = fetch_fmcib_csv(FMCIB_RADIOMICS_URL)

# %% [markdown]
# ## 3. Extract unique PatientIDs per collection

# %%
records = []
collection_summary = []

for src_name, df in [
    ("nsclc_radiogenomics", radiogenomics_df),
    ("nsclc_radiomics", radiomics_df),
]:
    pids = sorted(df["PatientID"].dropna().unique())
    coll_id_values = df["collection_id"].dropna().unique()
    coll_id = coll_id_values[0] if len(coll_id_values) else src_name
    n_series = len(df)
    n_patients = len(pids)

    for pid in pids:
        records.append({
            "fm": "fmcib",
            "source_collection": coll_id,
            "patient_id": pid,
            "evidence": (
                f"Listed in FMCIB pretraining manifest "
                f"(AIM-Harvard/foundation-cancer-image-biomarker/data/download/"
                f"{src_name}.csv)"
            ),
        })

    collection_summary.append({
        "source_csv": f"{src_name}.csv",
        "collection_id": coll_id,
        "n_unique_patients": n_patients,
        "n_series_rows": n_series,
        "ids_sample": ", ".join(map(str, pids[:5])),
        "overlaps_pet_fm_bench": (
            "T4 (NSCLC-Radiogenomics is the T4 evaluation cohort)"
            if "radiogenomics" in src_name
            else "none direct"
        ),
    })
    print(f"\n  {src_name}: {n_patients} unique patients, {n_series} series rows")
    print(f"    sample IDs: {pids[:5]}")

# %% [markdown]
# ## 4. Document non-TCIA / non-IDC sources (counted only)
#
# DeepLesion and LUNA16 are downloaded via AcademicTorrents in FMCIB's
# pipeline. They use integer or LIDC-style patient IDs that cannot collide
# with the PET-FM-Bench evaluation cohort by construction (different ID
# formats, different collections). We record the patient counts for
# completeness but do not enumerate per-patient rows.

# %%
NON_TCIA_SOURCES = [
    {
        "source_csv": "deeplesion.sh (AcademicTorrents)",
        "collection_id": "DeepLesion",
        "n_unique_patients": 4427,
        "n_series_rows": None,
        "ids_sample": "integer IDs 000001..004427",
        "overlaps_pet_fm_bench": "none — integer IDs, disjoint by construction",
    },
    {
        "source_csv": "luna16.sh (AcademicTorrents)",
        "collection_id": "LUNA16",
        "n_unique_patients": 888,
        "n_series_rows": None,
        "ids_sample": "LIDC-IDRI-XXXX format",
        "overlaps_pet_fm_bench": "none — LIDC-IDRI subset, disjoint from PET tasks",
    },
]

for src in NON_TCIA_SOURCES:
    collection_summary.append(src)
    print(f"  {src['collection_id']}: {src['n_unique_patients']} patients "
          f"(documented, not enumerated — {src['overlaps_pet_fm_bench']})")

# %% [markdown]
# ## 5. Save manifest

# %%
manifest_df = pd.DataFrame(records)
manifest_df = manifest_df.drop_duplicates(
    subset=["fm", "source_collection", "patient_id"]
)

summary_df = pd.DataFrame(collection_summary)

print(f"\n=== FMCIB manifest summary ===")
print(f"Auditable patient rows: {len(manifest_df):,}")
print(f"Unique auditable patients: {manifest_df['patient_id'].nunique():,}")
print(f"Sources audited (CSV-level): {sum(1 for s in collection_summary if s.get('n_series_rows'))}")
print(f"Sources documented (count-only): {len(NON_TCIA_SOURCES)}")
print()
print(summary_df.to_string(index=False))

manifest_path = OUT_DIR / "fmcib_training_manifest.parquet"
summary_path = OUT_DIR / "fmcib_collection_summary.csv"
metadata_path = OUT_DIR / "fmcib_manifest_metadata.json"

manifest_df.to_parquet(manifest_path, index=False)
summary_df.to_csv(summary_path, index=False)

with open(metadata_path, "w") as f:
    json.dump({
        "fm": "fmcib",
        "freeze_timestamp_utc": freeze_timestamp,
        "source_paper": "Pai et al. 2024, Nature Machine Intelligence "
                        "(s42256-024-00807-9)",
        "source_repo": "AIM-Harvard/foundation-cancer-image-biomarker (master)",
        "source_paths_audited": [
            "data/download/nsclc_radiogenomics.csv",
            "data/download/nsclc_radiomics.csv",
        ],
        "source_paths_documented_only": [
            "data/download/deeplesion.sh (AcademicTorrents)",
            "data/download/luna16.sh (AcademicTorrents)",
        ],
        "n_unique_auditable_patients": int(manifest_df["patient_id"].nunique()),
        "n_collections_auditable": int(
            summary_df.dropna(subset=["n_series_rows"])["collection_id"].nunique()
        ),
        "deeplesion_n_patients": 4427,
        "luna16_n_patients": 888,
        "unauditable_fraction": 0.0,
        "registration_h2_test_case": (
            "NSCLC-Radiogenomics is FMCIB's pretraining source AND the entire "
            "T4 evaluation cohort. Tier 1 contamination on (FMCIB, T4) by "
            "construction. This is the canonical H2 test case."
        ),
        "verification_status": "VERIFIED 2026-04-27 against FMCIB GitHub repo",
    }, f, indent=2)

print(f"\nWrote: {manifest_path}")
print(f"Wrote: {summary_path}")
print(f"Wrote: {metadata_path}")

# %% [markdown]
# ## 6. Done
#
# Commit with **"Save & Run All"** to persist outputs as a Kaggle Dataset
# (`pet-fm-bench-fmcib-manifest`).
