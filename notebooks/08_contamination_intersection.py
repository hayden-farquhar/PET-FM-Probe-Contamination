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
# # PET-FM-Bench: Phase 2 Stage 2 — Contamination Intersection
#
# **Runtime:** CPU | **Internet:** Off OK | **Time:** ~2-5 min | **GPU:** Not needed
#
# Pre-registration §4.6 Stage 2: take the FM training manifests from Stage 1
# (`07a/07b/07c`) and the Phase 4 v2 task split freeze, compute patient-level
# set intersections per (FM × task), and assign contamination tiers per the
# registered §4.6 rubric.
#
# **Datasets to attach:**
# - `pet-fm-bench-fmcib-manifest` (07a output)
# - `pet-fm-bench-ct-fm-manifest` (07b output)
# - `pet-fm-bench-biomedclip-manifest` (07c output)
# - `pet-fm-bench-task-splits` (Phase 4 v2 freeze; OSF download or republish as Kaggle dataset)
#
# **Outputs:**
# - `contamination_per_patient.parquet` — schema `(fm, task, patient_id, in_fm_training_data, evidence)`
# - `contamination_summary.csv` — schema `(fm, task, n_eval, n_contaminated, overlap_fraction, tier, tier_rationale)`
#
# **Tiering rubric (registration §4.6, verbatim):**
#
# | Tier | Criterion |
# |---|---|
# | 1 | Documented direct overlap ≥10% |
# | 2 | Documented indirect overlap 1–10% |
# | 3 | Possible overlap (different collection, same institution) |
# | 4 | Documented disjoint by institution |
# | 5 | Disjoint by construction |

# %% [markdown]
# ## 1. Setup

# %%
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

OUT_DIR = Path("/kaggle/working")
OUT_DIR.mkdir(parents=True, exist_ok=True)

freeze_timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
print(f"Freeze timestamp (UTC): {freeze_timestamp}")


def find_input(name):
    """Locate a file by name across Kaggle's variable-depth mount layout."""
    matches = list(Path("/kaggle/input").rglob(name))
    return matches[0] if matches else None


# %% [markdown]
# ## 2. Load FM training manifests + task splits

# %%
manifest_paths = {
    "fmcib": find_input("fmcib_training_manifest.parquet"),
    "ct_fm": find_input("ct_fm_training_manifest.parquet"),
    "biomedclip": find_input("biomedclip_caption_matches.parquet"),
}
task_splits_path = find_input("task_splits.parquet")

print("Inputs located:")
for fm, p in manifest_paths.items():
    print(f"  {fm}: {p}")
print(f"  task_splits: {task_splits_path}")

if task_splits_path is None:
    raise FileNotFoundError(
        "task_splits.parquet not attached. Add the Phase 4 v2 freeze dataset "
        "(OSF aqmkb / phase_4_freeze_task_splits/v2/task_splits.parquet, "
        "republished as a Kaggle dataset)."
    )

task_splits = pd.read_parquet(task_splits_path)
print(f"\ntask_splits: {len(task_splits)} rows, "
      f"{task_splits['task'].nunique()} tasks, "
      f"{task_splits['split'].unique()} splits")

# Per-task evaluation cohort (the patients to audit)
def task_eval_cohort(task):
    """Return the evaluation patient_ids for a task per registration §3.3.

    For T4 (cv_pool) and T6/T9 (test_retest), the entire cohort is the
    audit target. For T7/T8 (held-out test), only the test-set patients
    are the registered eval cohort, but we audit all (train+cal+test) so
    that contamination can be characterised across the full split.
    """
    sub = task_splits[task_splits["task"] == task]
    return set(sub["patient_id"].astype(str).tolist())


tasks = sorted(task_splits["task"].unique())
task_cohorts = {t: task_eval_cohort(t) for t in tasks}
for t in tasks:
    print(f"  {t}: {len(task_cohorts[t])} patients")

# %% [markdown]
# ## 3. Tier-3 institutional knowledge table
#
# Tier 3 ("Possible overlap — different collection, same institution") cannot
# be derived from patient-ID intersection alone; it requires knowing whether
# the FM trained on imaging from the same *institution* as the task's source
# collection. The pre-registration permits this inference based on the FM's
# documented training corpus.
#
# Hardcoded from Stage 1 verification (2026-04-27):

# %%
# Map: FM → set of institutional/collection contexts the FM trained on.
# Used to assign Tier 3 when no patient-level overlap is detected but
# institutional overlap is plausible.
FM_INSTITUTIONAL_CONTEXTS = {
    "fmcib": {
        "lung_cancer_imaging",          # FMCIB trained on lung CT collections
        "tcia_lung_collections",         # NSCLC-Radiogenomics + Lung1 + LIDC
    },
    "ct_fm": {
        "idc_public_ct",                 # CT-FM trained on public IDC v14 CT
        "tcia_lung_collections",
        "tcia_general_ct",
    },
    "biomedclip": {
        "pmc_oa_captions",               # BiomedCLIP scanned PMC OA
    },
    "dinov2": set(),                     # ImageNet only
    "rad_dino": set(),                   # MIMIC-CXR only
    "random_init": set(),                # no training data
}

# Map: task → (canonical_collection_id, set of institutional contexts)
# v3 (post amendment A9 / A10): T1 + T5 added per A9a/A9b expansion.
# T1 = AutoPET-I FDAT FDG-PET-CT-Lesions; T5 = AutoPET-III TCIA PSMA-PET-CT-Lesions.
# Neither cohort shares context with any of the 6 this project FMs' training data
# (none trained on FDG-PET-CT-Lesions or PSMA-PET-CT-Lesions; FMCIB+CT-FM
# are CT-only trained, BiomedCLIP/DINOv2/RAD-DINO are non-PET pretrained).
# Default tier-5 outcome expected for both T1 and T5 unless direct patient-ID
# overlap with FMCIB's NSCLC-Radiogenomics list (T4 cohort) — vanishingly
# unlikely since AutoPET-I/III are different patient cohorts than NSCLC-R.
TASK_CONTEXTS = {
    "t1": ("autopet_i_fdat", {"fdg_pet_lesions", "fdat_release"}),
    "t2": ("hecktor_2025_hn", {"hn_pet_lesions", "hecktor_release"}),
    "t3": ("hecktor_2025_hn_rfs", {"hn_pet_rfs", "hecktor_release"}),
    "t4": ("nsclc_radiogenomics", {"lung_cancer_imaging", "tcia_lung_collections"}),
    "t5": ("autopet_iii_tcia_psma", {"psma_pet_lesions", "tcia_pet_collections"}),
    "t6": ("rider_lung_pet_ct", {"lung_cancer_imaging", "tcia_lung_collections"}),
    "t7": ("acrin_nsclc_fdg_pet", {"lung_cancer_imaging", "tcia_lung_collections"}),
    "t8": ("lung_pet_ct_dx", {"lung_cancer_imaging", "tcia_lung_collections"}),
    "t9": ("vienna_quadra", {"healthy_controls"}),
}
# v4 (2026-04-30, per amendment A12): T2 and T3 added. Both are HECKTOR 2025
# (head-and-neck cancer cohort acquired 2026-04-29). No FM in the audited set
# (FMCIB / CT-FM / BiomedCLIP / DINOv2 / RAD-DINO / random_init) lists HECKTOR
# in its training manifest, so the institutional context is unique to T2 + T3
# and Tier 5 (no contamination) is the expected outcome by construction.

print("Institutional context table:")
for fm, ctx in FM_INSTITUTIONAL_CONTEXTS.items():
    print(f"  {fm}: {sorted(ctx) if ctx else '(none)'}")
print()
for task, (coll, ctx) in TASK_CONTEXTS.items():
    print(f"  {task} ({coll}): {sorted(ctx)}")

# %% [markdown]
# ## 4. Per-FM patient ID sets

# %%
def normalise_patient_id(pid):
    """Generate prefix-preserving zero-padding variants.

    v2 (after Stage 2 false-positive bug, 2026-04-27): the original v1
    expanded `R01-005` to include bare `5` and `005` — but bare numbers
    collide across collections (R01-5 ↔ ACRIN-NSCLC-FDG-PET-5 ↔ LUNG1-5 →
    spurious cross-collection matches). v2 keeps only **prefix-preserving**
    variants: `R01-005` ↔ `R01-5` ↔ `R01-0005` (all the same patient under
    the same prefix), but never `R01-5` ↔ `LUNG1-5`.

    Also generates separator variants (`-`/`_`/none) to handle TCIA/IDC
    naming inconsistencies (e.g., `RIDER-1129159` ↔ `RIDER1129159`).

    Returns a set of all plausible normalisations of one patient ID.
    """
    s = str(pid).strip()
    forms = {s, s.lower(), s.upper()}

    # Match alphabetic prefix + optional separator + numeric suffix
    m = re.match(r"^([A-Za-z][A-Za-z0-9]*)([-_]?)(\d+)$", s)
    if m:
        prefix, sep, num = m.group(1), m.group(2), m.group(3)
        n = int(num)
        # Prefix-preserving zero-padding variants
        for pad_width in (0, 3, 4, 5, 6, 7):
            padded = f"{n:0{pad_width}d}" if pad_width else str(n)
            # Try multiple separator conventions
            for sep_try in ("-", "_", ""):
                forms.add(f"{prefix}{sep_try}{padded}")
                forms.add(f"{prefix}{sep_try}{padded}".lower())
                forms.add(f"{prefix}{sep_try}{padded}".upper())

    # Also handle multi-segment prefixes like ACRIN-NSCLC-FDG-PET-001
    m2 = re.match(r"^([A-Za-z][A-Za-z0-9_-]*?-)(\d+)$", s)
    if m2:
        prefix2, num2 = m2.group(1), m2.group(2)
        n2 = int(num2)
        for pad_width in (0, 3, 4, 5, 6, 7):
            padded2 = f"{n2:0{pad_width}d}" if pad_width else str(n2)
            forms.add(f"{prefix2}{padded2}")
            forms.add(f"{prefix2}{padded2}".lower())
            forms.add(f"{prefix2}{padded2}".upper())

    return forms


def build_fm_id_set(manifest_df, fm_name):
    """Expand manifest patient IDs into all normalised forms."""
    if manifest_df is None or len(manifest_df) == 0:
        return set()
    raw_ids = manifest_df["patient_id"].dropna().astype(str)
    expanded = set()
    for pid in raw_ids:
        expanded |= normalise_patient_id(pid)
    return expanded


fm_manifests = {}
fm_id_sets = {}
fm_audit_incomplete = {}  # FM → True if its manifest is from a partial/sample scan
for fm, p in manifest_paths.items():
    if p is None:
        print(f"\n  ✗ {fm}: manifest not attached, treating as empty set")
        fm_manifests[fm] = pd.DataFrame()
        fm_id_sets[fm] = set()
        fm_audit_incomplete[fm] = False
    else:
        df = pd.read_parquet(p)
        fm_manifests[fm] = df
        fm_id_sets[fm] = build_fm_id_set(df, fm)

        # Detect audit-incomplete state by checking the manifest's metadata
        # sidecar (07c writes `audit_mode = "sample"` when in sample mode).
        meta_path = Path(str(p).replace(".parquet", "_metadata.json")
                                 .replace("_matches", "_manifest"))
        # Try multiple metadata-naming conventions
        for cand in [
            Path(str(p).replace(".parquet", "_metadata.json")),
            p.parent / f"{fm}_manifest_metadata.json",
            p.parent / "biomedclip_manifest_metadata.json",
        ]:
            if cand.exists():
                try:
                    with open(cand) as mf:
                        md = json.load(mf)
                    if md.get("audit_mode") == "sample" or md.get("audit_type") == "upper_bound":
                        if md.get("audit_mode") == "sample":
                            fm_audit_incomplete[fm] = True
                except Exception:
                    pass
                break
        fm_audit_incomplete.setdefault(fm, False)

        flag = " [AUDIT INCOMPLETE — sample mode]" if fm_audit_incomplete[fm] else ""
        print(f"\n  {fm}: {len(df)} manifest rows, "
              f"{df['patient_id'].nunique() if 'patient_id' in df.columns else 0} unique IDs, "
              f"{len(fm_id_sets[fm])} normalised ID forms{flag}")

# Tier-5 FMs (no training data overlap by construction)
TIER5_FMS = {"dinov2": "ImageNet only — no medical training data",
             "rad_dino": "CXR-only (MIMIC-CXR) — no PET overlap by construction",
             "random_init": "Random weights — no training data"}

# %% [markdown]
# ## 5. Compute intersections + assign tiers

# %%
def assign_tier(fm, task, overlap_fraction, has_overlap, manifest_present,
                audit_incomplete=False):
    """Apply registration §4.6 rubric.

    - Manifest-empty FMs (DINOv2/RAD-DINO/random_init) → Tier 5 always.
    - **Audit-incomplete FMs (e.g., BiomedCLIP sample-mode) → Tier "INCOMPLETE"**
      (NOT Tier 5). The pre-reg requires a full audit before tier assignment;
      sample-mode outputs are a dry-run sanity check only.
    - Direct overlap ≥10% → Tier 1.
    - Direct overlap 1–10% → Tier 2.
    - Direct overlap 0–1% (sparse) → Tier 3 if institutional context overlaps,
      else Tier 5.
    - Zero overlap, institutional context overlaps → Tier 3.
    - Zero overlap, no institutional context → Tier 5.
    """
    if fm in TIER5_FMS:
        return 5, TIER5_FMS[fm]
    if audit_incomplete:
        return "INCOMPLETE", (
            f"Audit not yet complete for {fm} — current manifest is from a "
            "partial scan (e.g., 07c sample mode). Re-run audit in full mode "
            "before assigning a registration tier."
        )
    if not manifest_present:
        return 5, "FM has no training-data manifest — disjoint by absence of audit object"

    fm_ctx = FM_INSTITUTIONAL_CONTEXTS.get(fm, set())
    _, task_ctx = TASK_CONTEXTS.get(task, ("", set()))
    institutional_overlap = bool(fm_ctx & task_ctx)

    if has_overlap:
        if overlap_fraction >= 0.10:
            return 1, f"Direct overlap {overlap_fraction:.1%} ≥ 10%"
        elif overlap_fraction >= 0.01:
            return 2, f"Direct overlap {overlap_fraction:.1%} (1-10%)"
        else:
            return 3, (f"Sparse direct overlap {overlap_fraction:.1%} (<1%); "
                       f"institutional context {'matches' if institutional_overlap else 'no match'}")
    else:
        if institutional_overlap:
            return 3, ("Zero patient-level overlap, but institutional context matches "
                       f"({sorted(fm_ctx & task_ctx)})")
        return 5, "Zero patient-level overlap and no institutional context match"


per_patient_records = []
summary_records = []

for fm in ("fmcib", "ct_fm", "biomedclip", "dinov2", "rad_dino", "random_init"):
    fm_ids = fm_id_sets.get(fm, set())
    manifest_present = bool(fm_ids)
    fm_manifest = fm_manifests.get(fm, pd.DataFrame())

    for task in tasks:
        eval_pids = task_cohorts[task]
        eval_pids_normed = {n for pid in eval_pids for n in normalise_patient_id(pid)}

        # Patient-level intersection: a task-side ID is contaminated if any of
        # its normalised forms appears in the FM's expanded ID set.
        contaminated = set()
        for pid in eval_pids:
            if normalise_patient_id(pid) & fm_ids:
                contaminated.add(pid)

        n_eval = len(eval_pids)
        n_contam = len(contaminated)
        overlap_frac = n_contam / max(n_eval, 1)

        tier, rationale = assign_tier(
            fm, task, overlap_frac, has_overlap=(n_contam > 0),
            manifest_present=manifest_present,
            audit_incomplete=fm_audit_incomplete.get(fm, False),
        )

        # Evidence string for per-patient rows: pull from manifest if present
        evidence_lookup = {}
        if "patient_id" in fm_manifest.columns and "evidence" in fm_manifest.columns:
            for _, r in fm_manifest.iterrows():
                for nf in normalise_patient_id(r["patient_id"]):
                    evidence_lookup[nf] = r["evidence"]

        for pid in eval_pids:
            is_contaminated = pid in contaminated
            evidence = ""
            if is_contaminated:
                for nf in normalise_patient_id(pid):
                    if nf in evidence_lookup:
                        evidence = evidence_lookup[nf]
                        break
            per_patient_records.append({
                "fm": fm,
                "task": task,
                "patient_id": pid,
                "in_fm_training_data": is_contaminated,
                "evidence": evidence,
            })

        summary_records.append({
            "fm": fm,
            "task": task,
            "n_eval": n_eval,
            "n_contaminated": n_contam,
            "n_clean": n_eval - n_contam,
            "overlap_fraction": round(overlap_frac, 4),
            "tier": tier,
            "tier_rationale": rationale,
        })
        print(f"  {fm:11s} × {task}: n_eval={n_eval:3d}, "
              f"contaminated={n_contam:3d} ({overlap_frac:.1%}), "
              f"tier={tier} — {rationale}")

# %% [markdown]
# ## 6. Save outputs

# %%
per_patient_df = pd.DataFrame(per_patient_records)
summary_df = pd.DataFrame(summary_records)

per_patient_path = OUT_DIR / "contamination_per_patient.parquet"
summary_path = OUT_DIR / "contamination_summary.csv"
metadata_path = OUT_DIR / "contamination_intersection_metadata.json"

per_patient_df.to_parquet(per_patient_path, index=False)
summary_df.to_csv(summary_path, index=False)

print(f"\n=== Stage 2 summary ===")
print(summary_df.pivot_table(
    index="fm", columns="task", values="tier", aggfunc="first"
).to_string())

with open(metadata_path, "w") as f:
    json.dump({
        "stage": "Phase 2 Stage 2 — Contamination Intersection",
        "freeze_timestamp_utc": freeze_timestamp,
        "registration_section": "§4.6 (contamination tiering)",
        "input_manifests": {fm: str(p) if p else None for fm, p in manifest_paths.items()},
        "task_splits_source": str(task_splits_path),
        "n_per_patient_rows": len(per_patient_df),
        "n_summary_rows": len(summary_df),
        "tier_distribution": summary_df["tier"].value_counts().sort_index().to_dict(),
    }, f, indent=2)

print(f"\nWrote: {per_patient_path}")
print(f"Wrote: {summary_path}")
print(f"Wrote: {metadata_path}")

# %% [markdown]
# ## 7. Done
#
# Save & Run All → Output → New Dataset → `pet-fm-bench-contamination-stage2`.
# This becomes the input to Stage 3 (`09_within_patient_test.py`).
