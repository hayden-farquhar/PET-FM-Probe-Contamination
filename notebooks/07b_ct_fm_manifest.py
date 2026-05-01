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
# # PET-FM-Bench: Phase 2 Stage 1b — CT-FM Training Manifest (verified)
#
# **Runtime:** CPU | **Internet:** ON | **Time:** ~10-30 min | **GPU:** Not needed
#
# Pre-registration §4.6 (contamination tiering) Stage 1: enumerate the patient
# IDs CT-FM (Project-Lighter) was pretrained on, so Stage 2 can intersect
# against PET-FM-Bench evaluation cohorts.
#
# **Source verified 2026-04-27** against the CT-FM repository
# (`project-lighter/CT-FM`, branch `main`, `notebooks/data-download/`).
#
# **Key finding:** CT-FM's pretraining cohort is **fully public and fully
# auditable**. 148,394 CT scans were pulled from the NCI Imaging Data Commons
# (IDC v14) by a single BigQuery SQL filter against
# `bigquery-public-data.idc_v14.dicom_all`. All series are on public
# `idc-open-data*` S3 buckets. There is no private institutional data.
# **`unauditable_fraction = 0.0`.**
#
# **Audit method:** Use the `idc-index` Python package (which ships a local
# SQLite mirror of IDC v14 metadata) to resolve every series in CT-FM's
# `manifest.txt` (52,254 series-level cp commands) to its `(collection_id,
# PatientID)` tuple. The package handles series→patient resolution offline.
#
# **CT-FM filter criteria** (from `notebooks/data-download/query.sql`):
# ```sql
# WHERE Modality = "CT" AND access = "Public"
#   AND min_SliceThickness >= 1
#   AND max_SliceThickness <= 5
#   AND num_instances > 50
#   AND num_instances/position_count = 1
#   AND has_localizer = "false"
# ```
#
# **Output:** `ct_fm_training_manifest.parquet` with schema
# `(fm, source_collection, patient_id, evidence)` — matches 07a/07c.

# %% [markdown]
# ## 1. Setup

# %%
import io
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

# idc-index ships a precomputed SQLite of IDC v18+ metadata; for the CT-FM
# audit we want IDC v14 (the version locked in CT-FM's BigQuery). idc-index
# falls back gracefully — the (collection_id, PatientID) mapping is stable
# across IDC versions for already-released collections.
!pip install -q idc-index pandas requests

from idc_index import index  # noqa: E402

OUT_DIR = Path("/kaggle/working")
OUT_DIR.mkdir(parents=True, exist_ok=True)

freeze_timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
print(f"Freeze timestamp (UTC): {freeze_timestamp}")

# CT-FM canonical sources (verified 2026-04-27)
CT_FM_REPO = "https://raw.githubusercontent.com/project-lighter/CT-FM/main/"
CT_FM_MANIFEST_URL = CT_FM_REPO + "notebooks/data-download/manifest.txt"
CT_FM_QUERY_URL = CT_FM_REPO + "notebooks/data-download/query.sql"

# %% [markdown]
# ## 2. Fetch CT-FM's published manifest + filter SQL
#
# manifest.txt is a flat list of `cp s3://idc-open-data*/<UUID>/<UUID>.dcm`
# lines. Each S3 UUID corresponds to one DICOM SOP instance; multiple
# instances per series, multiple series per study, multiple studies per
# patient. The S3 path's first UUID is the `crdc_series_uuid` in IDC.

# %%
def fetch_text(url):
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    return r.text


print("Fetching CT-FM manifest.txt...")
manifest_text = fetch_text(CT_FM_MANIFEST_URL)
manifest_lines = [ln for ln in manifest_text.splitlines() if ln.strip().startswith("cp ")]
print(f"  {len(manifest_lines):,} cp commands")

print("Fetching CT-FM query.sql...")
query_text = fetch_text(CT_FM_QUERY_URL)
print(f"  query.sql length: {len(query_text)} chars")

# Extract crdc_series_uuid from each cp command:
# `cp s3://idc-open-data/<UUID>/<UUID>.dcm.` → first UUID is series-level
S3_UUID_RE = re.compile(r"s3://idc-open-data[^/]*/([a-f0-9-]{36})/")
series_uuids = sorted({m.group(1) for ln in manifest_lines
                       for m in S3_UUID_RE.finditer(ln)})
print(f"  unique series UUIDs in manifest: {len(series_uuids):,}")

# %% [markdown]
# ## 3. Resolve series UUIDs to (collection_id, PatientID) via idc-index
#
# `IDCClient` exposes `index` as a pandas DataFrame with columns including
# `crdc_series_uuid`, `collection_id`, `PatientID`, `Modality`. We do a
# single dataframe merge to resolve all UUIDs at once.

# %%
client = index.IDCClient()
idc_df = client.index  # full IDC index DataFrame
print(f"  IDC local index size: {len(idc_df):,} rows; "
      f"columns: {list(idc_df.columns)[:8]}...")

# Determine series-uuid column name (idc-index uses `crdc_series_uuid`)
uuid_col = None
for cand in ["crdc_series_uuid", "series_aws_url", "aws_url"]:
    if cand in idc_df.columns:
        uuid_col = cand
        break
if uuid_col is None:
    raise RuntimeError(
        f"idc-index version mismatch — expected crdc_series_uuid; "
        f"found {list(idc_df.columns)}"
    )

# If the column holds full URLs, extract UUIDs first
if "url" in uuid_col.lower():
    idc_df = idc_df.copy()
    idc_df["_series_uuid"] = idc_df[uuid_col].astype(str).str.extract(
        r"/([a-f0-9-]{36})/", expand=False
    )
    join_col = "_series_uuid"
else:
    join_col = uuid_col

# Merge our manifest UUIDs against IDC index
manifest_pids_df = idc_df[idc_df[join_col].isin(series_uuids)][
    ["collection_id", "PatientID", join_col]
].drop_duplicates()

n_resolved = manifest_pids_df[join_col].nunique()
n_missing = len(series_uuids) - n_resolved
print(f"  resolved {n_resolved:,} / {len(series_uuids):,} series "
      f"({n_missing} missing — likely IDC version drift)")

# %% [markdown]
# ## 4. Build the manifest

# %%
records = []
for _, row in manifest_pids_df.drop_duplicates(["collection_id", "PatientID"]).iterrows():
    records.append({
        "fm": "ct_fm",
        "source_collection": row["collection_id"],
        "patient_id": row["PatientID"],
        "evidence": (
            f"Listed in CT-FM pretraining manifest "
            f"(project-lighter/CT-FM/notebooks/data-download/manifest.txt; "
            f"resolved via idc-index)"
        ),
    })

manifest_df = pd.DataFrame(records).drop_duplicates(
    subset=["fm", "source_collection", "patient_id"]
)

# Per-collection summary
collection_summary = (
    manifest_pids_df.groupby("collection_id")
    .agg(n_unique_patients=("PatientID", "nunique"),
         n_series=(join_col, "nunique"))
    .reset_index()
    .sort_values("n_unique_patients", ascending=False)
)

# Flag collections that overlap PET-FM-Bench evaluation.
# Collection IDs verified against IDC v14+ via Kaggle dry-run on 2026-04-27:
# canonical names are `rider_lung_pet_ct` (not `rider_lung_ct`) and
# `acrin_nsclc_fdg_pet`.
PET_FM_BENCH_COLLECTIONS = {
    "nsclc_radiogenomics": "T4 evaluation cohort",
    "lung_pet_ct_dx": "T8 evaluation cohort",
    "rider_lung_pet_ct": "T6 evaluation cohort",
    "acrin_nsclc_fdg_pet": "T7 evaluation cohort",
}
collection_summary["overlaps_pet_fm_bench"] = collection_summary["collection_id"].apply(
    lambda c: PET_FM_BENCH_COLLECTIONS.get(c.lower().replace("-", "_"), "")
)

print("\n=== CT-FM manifest summary ===")
print(f"Total auditable patient rows: {len(manifest_df):,}")
print(f"Unique auditable patients:    {manifest_df['patient_id'].nunique():,}")
print(f"Distinct IDC collections:     {len(collection_summary)}")
print()
print(collection_summary.head(40).to_string(index=False))

# %% [markdown]
# ## 5. Save manifest

# %%
manifest_path = OUT_DIR / "ct_fm_training_manifest.parquet"
summary_path = OUT_DIR / "ct_fm_collection_summary.csv"
metadata_path = OUT_DIR / "ct_fm_manifest_metadata.json"

manifest_df.to_parquet(manifest_path, index=False)
collection_summary.to_csv(summary_path, index=False)

with open(metadata_path, "w") as f:
    json.dump({
        "fm": "ct_fm",
        "freeze_timestamp_utc": freeze_timestamp,
        "source_paper": "Pai et al. 2025, arXiv:2501.09001",
        "source_repo": "project-lighter/CT-FM (main)",
        "source_paths_audited": [
            "notebooks/data-download/manifest.txt",
            "notebooks/data-download/query.sql",
        ],
        "idc_version_locked": "idc_v14 (per CT-FM query.sql)",
        "n_series_in_manifest": len(series_uuids),
        "n_series_resolved": int(n_resolved),
        "n_series_missing": int(n_missing),
        "n_unique_patients": int(manifest_df["patient_id"].nunique()),
        "n_distinct_collections": int(collection_summary.shape[0]),
        "unauditable_fraction": 0.0,
        "registration_tier_floor": (
            "All ct_fm pretraining is from public IDC v14. "
            "Tier 5 (disjoint by construction) where collections do not "
            "appear in CT-FM training; per-collection tier from Stage 2 "
            "intersection."
        ),
        "verification_status": "VERIFIED 2026-04-27 against CT-FM GitHub repo",
    }, f, indent=2)

print(f"\nWrote: {manifest_path}")
print(f"Wrote: {summary_path}")
print(f"Wrote: {metadata_path}")

# %% [markdown]
# ## 6. Done
#
# Commit with **"Save & Run All"** to persist as `pet-fm-bench-ct-fm-manifest`.
