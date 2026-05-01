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
# # PET-FM-Bench: Phase 2 Stage 4 — Contamination Audit Freeze
#
# **Runtime:** CPU | **Internet:** Off OK | **Time:** ~2 min | **GPU:** Not needed
#
# Pre-registration §4.6 Stage 4: compile Stage 2 (intersection) and Stage 3
# (within-patient test) outputs into the Phase 2 freeze artefacts, hash
# them, write a metadata sidecar with full provenance, and produce upload
# instructions for OSF aqmkb / `phase_2_freeze_contamination_audit/`.
#
# **Datasets to attach:**
# - `pet-fm-bench-contamination-stage2` (08 output:
#   `contamination_per_patient.parquet`,
#   `contamination_summary.csv`,
#   `contamination_intersection_metadata.json`)
# - `pet-fm-bench-contamination-stage3` (09 output:
#   `dirty_vs_clean_results.csv`,
#   `within_patient_test_metadata.json`)
# - Optionally also the three Stage 1 manifests so their hashes can be
#   recorded as provenance (FMCIB / CT-FM / BiomedCLIP manifests).
#
# **Phase 2 freeze outputs (the artefacts cited by Phase 5 / manuscript):**
# - `contamination_manifest.parquet` — the per-patient flag table from Stage 2.
# - `contamination_audit.csv` — combined per (FM × task) summary: tier from
#   Stage 2 + dirty-vs-clean delta + perm-p from Stage 3.
# - `contamination_freeze_metadata.json` — provenance + SHA-256 + tier
#   distribution.

# %% [markdown]
# ## 1. Setup

# %%
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

OUT_DIR = Path("/kaggle/working")
OUT_DIR.mkdir(parents=True, exist_ok=True)

freeze_timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
print(f"Freeze timestamp (UTC): {freeze_timestamp}")


def find_input(name):
    matches = list(Path("/kaggle/input").rglob(name))
    return matches[0] if matches else None


def sha256_file(path, chunk=2**20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


# %% [markdown]
# ## 2. Locate Stage 2 + Stage 3 outputs

# %%
inputs = {
    "stage2_per_patient": find_input("contamination_per_patient.parquet"),
    "stage2_summary": find_input("contamination_summary.csv"),
    "stage2_meta": find_input("contamination_intersection_metadata.json"),
    "stage3_results": find_input("dirty_vs_clean_results.csv"),
    "stage3_meta": find_input("within_patient_test_metadata.json"),
    # Optional Stage 1 provenance
    "stage1_fmcib": find_input("fmcib_training_manifest.parquet"),
    "stage1_ct_fm": find_input("ct_fm_training_manifest.parquet"),
    "stage1_biomedclip": find_input("biomedclip_caption_matches.parquet"),
}

print("Inputs located:")
for k, v in inputs.items():
    print(f"  {k}: {v}")

required = ["stage2_per_patient", "stage2_summary", "stage3_results"]
missing = [k for k in required if inputs[k] is None]
if missing:
    raise FileNotFoundError(
        f"Missing required inputs: {missing}. Attach Stage 2 + Stage 3 "
        "Kaggle datasets before running Stage 4."
    )

# %% [markdown]
# ## 3. Load + combine

# %%
per_patient = pd.read_parquet(inputs["stage2_per_patient"])
summary = pd.read_csv(inputs["stage2_summary"])
dirty_clean = pd.read_csv(inputs["stage3_results"])

print(f"\nStage 2 per-patient: {len(per_patient):,} rows")
print(f"Stage 2 summary:     {len(summary)} (FM × task) pairs")
print(f"Stage 3 results:     {len(dirty_clean)} (FM × task) pairs")

# Merge Stage 2 summary + Stage 3 deltas into the audit table
audit = summary.merge(
    dirty_clean[[c for c in dirty_clean.columns
                 if c not in ("fm", "task", "n_dirty", "n_clean")
                 or c in ("fm", "task")]],
    on=["fm", "task"], how="left", suffixes=("", "_s3"),
)

print("\n=== Phase 2 audit table ===")
print(audit.to_string(index=False, max_colwidth=60))

# %% [markdown]
# ## 4. Write freeze artefacts

# %%
manifest_path = OUT_DIR / "contamination_manifest.parquet"
audit_path = OUT_DIR / "contamination_audit.csv"
meta_path = OUT_DIR / "contamination_freeze_metadata.json"

per_patient.to_parquet(manifest_path, index=False)
audit.to_csv(audit_path, index=False)

# Hash everything for provenance + auditability
manifest_sha = sha256_file(manifest_path)
audit_sha = sha256_file(audit_path)

# Hash the inputs we relied on (for the metadata sidecar)
input_hashes = {}
for k, p in inputs.items():
    if p is not None and Path(p).is_file():
        input_hashes[k] = {"path": str(p), "sha256": sha256_file(p)}

with open(meta_path, "w") as f:
    json.dump({
        "stage": "Phase 2 Stage 4 — Contamination Audit Freeze",
        "freeze_timestamp_utc": freeze_timestamp,
        "registration_section": "§4.6 (contamination tiering)",
        "osf_doi": "10.17605/OSF.IO/DQ2JA",
        "freeze_artefacts": {
            "contamination_manifest.parquet": {
                "n_rows": int(len(per_patient)),
                "n_fms": int(per_patient["fm"].nunique()),
                "n_tasks": int(per_patient["task"].nunique()),
                "sha256": manifest_sha,
            },
            "contamination_audit.csv": {
                "n_rows": int(len(audit)),
                "tier_distribution": (
                    audit["tier"].value_counts().sort_index().to_dict()
                ),
                "sha256": audit_sha,
            },
        },
        "input_provenance": input_hashes,
        "phase_4_v4_split_reference": (
            "OSF aqmkb / phase_4_freeze_task_splits/v4/task_splits.parquet "
            "SHA-256 3855e48362e646f4980df6b43f7bb881be3cd9ee0d98fb0799408d4fc21e0e63"
        ),
    }, f, indent=2)

print(f"\nWrote: {manifest_path}")
print(f"  SHA-256: {manifest_sha}")
print(f"\nWrote: {audit_path}")
print(f"  SHA-256: {audit_sha}")
print(f"\nWrote: {meta_path}")

# %% [markdown]
# ## 5. OSF upload instructions
#
# These artefacts close the Phase 2 freeze v2 (9-task universe per A12 — adds
# T2/T3 to v1's 7-task contamination audit). v1 closure on OSF was
# `phase_2_freeze_contamination_audit/` root; v2 closure target is the new
# `phase_2_freeze_contamination_audit/v2/` subfolder (matches the Phase 4 v4
# subfolder pattern). v1 manifest+audit+metadata stay archived in OSF history
# alongside the v2 subfolder.
#
# ```bash
# OSF_TOKEN="<your-token>"
# # Find existing phase_2_freeze_contamination_audit folder ID first via:
# # curl -H "Authorization: Bearer ${OSF_TOKEN}" \
# #   "https://api.osf.io/v2/nodes/aqmkb/files/osfstorage/?filter[name]=phase_2_freeze_contamination_audit"
# # Then create v2 subfolder inside it:
# PARENT_FOLDER_ID="<parent_folder_id>"
# curl -X PUT \
#   -H "Authorization: Bearer ${OSF_TOKEN}" \
#   "https://files.osf.io/v1/resources/aqmkb/providers/osfstorage/${PARENT_FOLDER_ID}/?kind=folder&name=v2"
#
# # Upload each artefact (use the folder ID returned by the previous call)
# FOLDER_ID="<folder_id>"
# for f in contamination_manifest.parquet contamination_audit.csv contamination_freeze_metadata.json; do
#   curl -X PUT \
#     -H "Authorization: Bearer ${OSF_TOKEN}" \
#     --data-binary "@$f" \
#     "https://files.osf.io/v1/resources/aqmkb/providers/osfstorage/${FOLDER_ID}/?kind=file&name=$f"
# done
# ```
#
# After upload, append a closure entry to `osf/amendment_log.md` (Phase 2
# freeze closed, with the OSF-side SHA-256 hashes) and re-upload the
# amendment log.

# %% [markdown]
# ## 6. Done
#
# Save & Run All → Output → New Dataset → `pet-fm-bench-contamination-freeze`
# (optional — for backup; the canonical artefact is OSF-side).
