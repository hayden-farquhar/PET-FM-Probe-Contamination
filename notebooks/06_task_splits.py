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
# # PET-FM-Bench: Phase 4 Freeze v4 — Task Splits Manifest (9-task universe)
#
# **Runtime:** CPU | **Internet:** Off OK | **Time:** ~5 min | **GPU:** Not needed
#
# Pre-registration §3.3 freeze: locks patient-level train/cal/test splits per
# task with a fixed random seed. Required BEFORE any probe training.
#
# **v4 (2026-04-30 amendment, per `osf/amendment_log.md` A12):** Adds T2 + T3
# (HECKTOR 2025) to v3's 7-task universe. Both consume the unified HECKTOR labels
# parquet (filtered by `task1_patient` for T2 and `task2_patient` for T3).
#
# **v3 → v4 changes:**
# - **T2 (HECKTOR HN tumour patch-classification)**: patient-level 70/15/15
#   random split per A12b (Mondrian per-centre downgraded to sensitivity due
#   to MDA dominance 442/680=58%). Cohort = patients with `task1_patient == True`
#   in HECKTOR labels parquet (n=680).
# - **T3 (HECKTOR RFS prediction)**: 5-fold nested cross-validation across all
#   patients (`split=cv_pool`), stratified by relapse event indicator per
#   registration §3.3 + A12c. Cohort = patients with `task2_patient == True`
#   AND non-NaN relapse + rfs_days (n=651, 132 events).
#
# **Coverage (v4 — 9 tasks):**
# - **T1 (AutoPET-I FDG)**: patient-level 70/15/15 stratified by `cancer_type` (per A9a)
# - **T2 (HECKTOR HN)**: patient-level 70/15/15 random (per A12b)  [NEW v4]
# - **T3 (HECKTOR RFS)**: cv_pool (5-fold nested CV stratified by relapse) [NEW v4]
# - **T4 (NSCLC survival)**: cv_pool (5-fold nested CV stratified by event)
# - **T5 (AutoPET-III PSMA)**: test_zero_shot (entire cohort, probe fit on T1)
# - **T6 (RIDER test-retest)**: test_retest (no split)
# - **T7 (ACRIN response)**: patient-level 70/15/15 random
# - **T8 (Lung-PET-CT-Dx subtype)**: patient-level 70/15/15 stratified by subtype
# - **T9 (QUADRA test-retest)**: test_retest (no split)
#
# **Supersession:** Phase 4 v3 (added T1+T5) and v2 (5-task) remain in OSF
# history but are superseded by v4. Any analysis citing Phase 4 must reference
# the v4 freeze.
#
# **Output:** `task_splits.parquet` for upload to OSF as Phase 4 freeze artefact.
# Long format: one row per (task, patient_id, split, stratum) tuple.
#
# **Datasets to attach (v4 — 8 embeddings datasets):**
# - `pet-fm-bench-t1-embeddings-v3` (for t1_labels.parquet with cancer_type)
# - `pet-fm-bench-t4-embeddings-v3` (for t4_labels.csv with event/survival)
# - `pet-fm-bench-t5-embeddings-v3` (for t5_labels.parquet, T5 cohort)
# - `pet-fm-bench-t6-embeddings-v3` (for t6_labels.csv with is_retest_patient)
# - `pet-fm-bench-t7-embeddings-v3` (patient list)
# - `pet-fm-bench-t8-embeddings-v3` (for t8_labels.csv with subtype)
# - `pet-fm-bench-t9-embeddings-v3` (subject list)
# - `pet-fm-bench-hecktor-2025-embeddings-v3` (for hecktor_labels.parquet —
#   note slug deviation: "-2025-" in middle, NOT just "-hecktor-")

# %% [markdown]
# ## 1. Setup

# %%
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42                       # Pre-registration §3.3 seed
# v2: 70/15/15 for T7/T8 per registration §3.3 (was 60/20/20 in v1).
TRAIN_FRAC, CAL_FRAC, TEST_FRAC = 0.70, 0.15, 0.15
OUT_DIR = Path("/kaggle/working")
OUT_DIR.mkdir(parents=True, exist_ok=True)

freeze_timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
print(f"Freeze timestamp (UTC): {freeze_timestamp}")
print(f"Random seed: {SEED}")
print(f"Split fractions: train={TRAIN_FRAC}, cal={CAL_FRAC}, test={TEST_FRAC}")


def find_dataset(name):
    """Locate a Kaggle dataset by name across mount layouts."""
    candidates = list(Path("/kaggle/input").rglob(name))
    return candidates[0] if candidates else None


def load_labels(task):
    """Load tX_labels.csv from the v3 embeddings dataset (T4 / T6 / T7 / T8 / T9 — CSV)."""
    emb_root = find_dataset(f"pet-fm-bench-{task}-embeddings-v3")
    if emb_root is None:
        return None
    candidates = list(emb_root.rglob(f"{task}_labels.csv"))
    if not candidates:
        return None
    return pd.read_csv(candidates[0])


def load_labels_parquet(task):
    """Load tX_labels.parquet from the v3 embeddings dataset (T1 / T5 — per-patch).

    T1 and T5 emit a parquet (not CSV) because per-patch rows benefit from typed
    columns and the lesion_voxels / iou floats. Per-patient stratifiers are
    obtained by deduplicating on patient_id and taking the first cancer_type.
    """
    emb_root = find_dataset(f"pet-fm-bench-{task}-embeddings-v3")
    if emb_root is None:
        return None
    candidates = list(emb_root.rglob(f"{task}_labels.parquet"))
    if not candidates:
        return None
    return pd.read_parquet(candidates[0])


def load_labels_hecktor():
    """Load hecktor_labels.parquet — handles HECKTOR's slug deviation.

    The published HECKTOR embeddings dataset slug is
    `pet-fm-bench-hecktor-2025-embeddings-v3` (with "2025"), NOT the
    convention-following `pet-fm-bench-hecktor-embeddings-v3`. Try both forms.
    """
    for name in [
        "pet-fm-bench-hecktor-embeddings-v3",
        "pet-fm-bench-hecktor-2025-embeddings-v3",
    ]:
        emb_root = find_dataset(name)
        if emb_root is None:
            continue
        candidates = list(emb_root.rglob("hecktor_labels.parquet"))
        if candidates:
            return pd.read_parquet(candidates[0])
    return None


def patient_ids_from_embeddings(task):
    """Recover the canonical patient/subject ID list from any FM parquet."""
    emb_root = find_dataset(f"pet-fm-bench-{task}-embeddings-v3")
    if emb_root is None:
        return []
    parquets = [p for p in emb_root.rglob("*.parquet") if not p.stem.endswith("labels")]
    if not parquets:
        return []
    df = pd.read_parquet(parquets[0])
    id_col = "subject_id" if "subject_id" in df.columns else "patient_id"
    return sorted(df[id_col].unique().tolist())


# %% [markdown]
# ## 2. Per-task split assignment

# %%
records = []
task_summaries = []


def add_split(task, patient_id, split, stratum=None):
    records.append({
        "task": task,
        "patient_id": patient_id,
        "split": split,
        "stratum": stratum,
    })


def cv_pool(patient_ids, stratum_values, task_name):
    """v2: T4-style — no held-out test split, all patients in CV pool.

    Registration §3.3 specifies 5-fold nested cross-validation for T4.
    Encode every patient with split='cv_pool' so probe_analysis.py v4
    dispatches on this label and runs CV across all of them.
    """
    pids = np.asarray(patient_ids)
    strata = np.asarray(stratum_values)
    for pid, stratum in zip(pids, strata):
        add_split(task_name, pid, "cv_pool", str(stratum))
    summary = {
        "task": task_name,
        "n_patients": len(pids),
        "n_train": 0, "n_cal": 0, "n_test": 0,
        "n_cv_pool": len(pids),
        "stratification": "outcome (used in CV stratification, not split)",
        "n_strata": int(len(np.unique(strata))),
    }
    print(f"  {task_name}: {summary['n_patients']} patients → cv_pool "
          f"(5-fold nested CV; stratified by {summary['n_strata']}-class outcome)")
    return summary


def stratified_split(patient_ids, stratum_values, task_name):
    """Patient-level stratified split using TRAIN_FRAC/CAL_FRAC/TEST_FRAC.

    v2: ratios are 70/15/15 per registration §3.3 (was 60/20/20 in v1).
    """
    pids = np.asarray(patient_ids)
    strata = np.asarray(stratum_values)
    assert len(pids) == len(strata)

    # First split off test (20%); then split remainder into train+cal (75%/25% of remainder = 60%/20% of total)
    pids_trainCal, pids_test, strata_trainCal, _ = train_test_split(
        pids, strata, test_size=TEST_FRAC, random_state=SEED, stratify=strata,
    )
    pids_train, pids_cal, _, _ = train_test_split(
        pids_trainCal, strata_trainCal,
        test_size=CAL_FRAC / (TRAIN_FRAC + CAL_FRAC),
        random_state=SEED, stratify=strata_trainCal,
    )

    pid_to_split = {pid: "train" for pid in pids_train}
    pid_to_split.update({pid: "cal" for pid in pids_cal})
    pid_to_split.update({pid: "test" for pid in pids_test})

    pid_to_stratum = dict(zip(pids, strata))

    for pid in pids:
        add_split(task_name, pid, pid_to_split[pid], str(pid_to_stratum[pid]))

    summary = {
        "task": task_name,
        "n_patients": len(pids),
        "n_train": int((pd.Series([pid_to_split[p] for p in pids]) == "train").sum()),
        "n_cal": int((pd.Series([pid_to_split[p] for p in pids]) == "cal").sum()),
        "n_test": int((pd.Series([pid_to_split[p] for p in pids]) == "test").sum()),
        "stratification": "outcome",
        "n_strata": len(np.unique(strata)),
    }
    print(f"  {task_name}: {summary['n_patients']} patients → "
          f"train={summary['n_train']}, cal={summary['n_cal']}, test={summary['n_test']} "
          f"(stratified by {summary['n_strata']}-class outcome)")
    return summary


def all_test(patient_ids, task_name, stratum_label):
    """Test-retest tasks: no train/cal/test split — every patient is in 'test_retest'."""
    for pid in patient_ids:
        add_split(task_name, pid, "test_retest", stratum_label)
    summary = {
        "task": task_name,
        "n_patients": len(patient_ids),
        "n_train": 0, "n_cal": 0, "n_test": 0,
        "stratification": "none (test-retest task)",
        "n_strata": 1,
    }
    print(f"  {task_name}: {summary['n_patients']} patients → all → test_retest")
    return summary


def random_split(patient_ids, task_name, stratum_label):
    """Patient-level random split using TRAIN_FRAC/CAL_FRAC/TEST_FRAC.

    v2: ratios are 70/15/15 per registration §3.3 (was 60/20/20 in v1).
    """
    pids = np.asarray(patient_ids)
    pids_trainCal, pids_test = train_test_split(
        pids, test_size=TEST_FRAC, random_state=SEED,
    )
    pids_train, pids_cal = train_test_split(
        pids_trainCal,
        test_size=CAL_FRAC / (TRAIN_FRAC + CAL_FRAC),
        random_state=SEED,
    )
    pid_to_split = {pid: "train" for pid in pids_train}
    pid_to_split.update({pid: "cal" for pid in pids_cal})
    pid_to_split.update({pid: "test" for pid in pids_test})
    for pid in pids:
        add_split(task_name, pid, pid_to_split[pid], stratum_label)
    summary = {
        "task": task_name,
        "n_patients": len(pids),
        "n_train": int(sum(1 for p in pids if pid_to_split[p] == "train")),
        "n_cal": int(sum(1 for p in pids if pid_to_split[p] == "cal")),
        "n_test": int(sum(1 for p in pids if pid_to_split[p] == "test")),
        "stratification": "none (deferred — see PROGRESS notes)",
        "n_strata": 1,
    }
    print(f"  {task_name}: {summary['n_patients']} patients → "
          f"train={summary['n_train']}, cal={summary['n_cal']}, test={summary['n_test']} "
          f"(random, no outcome stratification)")
    return summary


print("\nProcessing tasks...")

# --- T1: AutoPET-I FDG lesion-patch classification, stratified by cancer_type (per A9a) ---
# Patient-level 70/15/15 split. Each patient contributes their patches to a single
# split partition (registration §3.3 "maintaining patient integrity") so probe
# evaluation never sees the same patient in train and test.
print("\n[T1 — AutoPET-I FDG lesion-patch (per A9a)]")
labels_t1 = load_labels_parquet("t1")
if labels_t1 is not None:
    patient_cancer = (
        labels_t1.dropna(subset=["cancer_type"])
        .groupby("patient_id")["cancer_type"].first()
        .reset_index()
    )
    summary = stratified_split(
        patient_cancer["patient_id"].values,
        patient_cancer["cancer_type"].values,
        task_name="t1",
    )
    task_summaries.append(summary)
else:
    print("  ✗ t1_labels.parquet not found — skipping (T1 embeddings dataset not attached)")

# --- T2: HECKTOR HN tumour patch-classification, random 70/15/15 (per A12b) ---
# Per A12b: patient-level 70/15/15 random split GLOBALLY (Mondrian per-centre
# downgraded from primary to sensitivity due to MDA dominance 442/680 = 65% in
# the Task1 cohort and HMR n=11 too small for stratified split). Per-centre
# AUROC computed post-hoc in supplementary. Cohort = patients with task1_patient
# flag in HECKTOR labels parquet (n=680 per the freeze metadata QC).
print("\n[T2 — HECKTOR HN tumour patch-classification (per A12a/b)]")
labels_hecktor = load_labels_hecktor()
if labels_hecktor is not None:
    # CRITICAL: fillna(False) before astype(bool) — bool(NaN) is True in Python.
    # Defensive guard; should never fire but matches probe_analysis.py v6 fix #2.
    t2_mask = labels_hecktor["task1_patient"].fillna(False).astype(bool)
    t2_patients = sorted(labels_hecktor.loc[t2_mask, "patient_id"].unique())
    if t2_patients:
        summary = random_split(
            t2_patients, task_name="t2", stratum_label="hecktor_2025_hn",
        )
        task_summaries.append(summary)
    else:
        print("  ✗ HECKTOR labels has no task1_patient=True rows")
else:
    print("  ✗ hecktor_labels.parquet not found — skipping (HECKTOR embeddings dataset not attached)")

# --- T4: NSCLC survival, 5-fold nested CV (no held-out test) per reg §3.3 ---
print("\n[T4 — NSCLC survival]")
labels_t4 = load_labels("t4")
if labels_t4 is not None:
    # Drop patients without an event flag (shouldn't happen but be safe)
    labels_t4 = labels_t4.dropna(subset=["event"]).copy()
    labels_t4["event"] = labels_t4["event"].astype(int)
    summary = cv_pool(
        labels_t4["patient_id"].values,
        labels_t4["event"].values,
        task_name="t4",
    )
    task_summaries.append(summary)
else:
    print("  ✗ t4_labels.csv not found — skipping")

# --- T3: HECKTOR RFS prediction, 5-fold nested CV (no held-out test) per reg §3.3 + A12c ---
# Cohort = patients with task2_patient AND non-NaN relapse + rfs_days (n=651 per
# preprocessing QC, 132 events / 20.3% event rate). cv_pool stratified by relapse
# event for proper StratifiedKFold behaviour (matches T4's cv_pool pattern).
# HPV-stratified sub-analysis is a registration §5.6 secondary analysis (n=549 with
# resolved HPV status); not implemented in the splits manifest — handled by
# probe_analysis.py v6 if/when called.
print("\n[T3 — HECKTOR RFS prediction (per A12c)]")
if labels_hecktor is not None:
    # CRITICAL: fillna(False) before astype(bool) (matches probe_analysis.py v6 fix #2).
    t3_mask = (
        labels_hecktor["task2_patient"].fillna(False).astype(bool)
        & labels_hecktor["relapse"].notna()
        & labels_hecktor["rfs_days"].notna()
        & (labels_hecktor["rfs_days"] > 0)
    )
    t3_subset = labels_hecktor.loc[t3_mask].drop_duplicates("patient_id")
    if len(t3_subset) > 0:
        summary = cv_pool(
            t3_subset["patient_id"].values,
            t3_subset["relapse"].astype(int).values,
            task_name="t3",
        )
        task_summaries.append(summary)
    else:
        print("  ✗ HECKTOR labels has no T3-eligible patients")
else:
    print("  ✗ hecktor_labels.parquet not found — T3 skipped (already warned for T2)")

# --- T6: RIDER test-retest, no split ---
print("\n[T6 — RIDER test-retest]")
labels_t6 = load_labels("t6")
if labels_t6 is not None:
    pids = sorted(labels_t6["patient_id"].unique())
    task_summaries.append(all_test(pids, task_name="t6", stratum_label="test_retest"))
else:
    pids = patient_ids_from_embeddings("t6")
    if pids:
        task_summaries.append(all_test(pids, task_name="t6", stratum_label="test_retest"))
    else:
        print("  ✗ T6 patient list not found — skipping")

# --- T7: ACRIN response, random split (outcome stratification deferred) ---
print("\n[T7 — ACRIN response]")
pids_t7 = patient_ids_from_embeddings("t7")
if pids_t7:
    task_summaries.append(random_split(
        pids_t7, task_name="t7", stratum_label="response_label_pending",
    ))
else:
    print("  ✗ T7 patient list not found — skipping")

# --- T8: Lung-PET-CT-Dx subtype, stratified by subtype ---
print("\n[T8 — Lung-PET-CT-Dx subtype]")
labels_t8 = load_labels("t8")
if labels_t8 is not None:
    labels_t8 = labels_t8.dropna(subset=["subtype"]).copy()
    summary = stratified_split(
        labels_t8["patient_id"].values,
        labels_t8["subtype"].values,
        task_name="t8",
    )
    task_summaries.append(summary)
else:
    print("  ✗ t8_labels.csv not found — skipping")

# --- T5: AutoPET-III PSMA cross-tracer detection, ALL test (zero-shot per A9b / §3.1) ---
# Per registration §3.1 T5 is zero-shot — no train/cal partition; entire cohort is
# the held-out evaluation set. Train embeddings come from T1 train split (probe
# is fit on T1, evaluated on T5). Stratification not applicable.
print("\n[T5 — AutoPET-III PSMA zero-shot (per A9b)]")
labels_t5 = load_labels_parquet("t5")
if labels_t5 is not None:
    pids_t5 = sorted(labels_t5["patient_id"].unique())
    task_summaries.append(all_test(pids_t5, task_name="t5", stratum_label="test_zero_shot"))
else:
    pids_t5 = patient_ids_from_embeddings("t5")
    if pids_t5:
        task_summaries.append(all_test(pids_t5, task_name="t5", stratum_label="test_zero_shot"))
    else:
        print("  ✗ T5 patient list not found — skipping (T5 embeddings dataset not attached)")

# --- T9: QUADRA test-retest, no split ---
print("\n[T9 — QUADRA test-retest]")
pids_t9 = patient_ids_from_embeddings("t9")
if pids_t9:
    task_summaries.append(all_test(pids_t9, task_name="t9", stratum_label="test_retest"))
else:
    print("  ✗ T9 subject list not found — skipping")

# %% [markdown]
# ## 3. Save splits parquet + metadata sidecar

# %%
splits_df = pd.DataFrame(records)
splits_path = OUT_DIR / "task_splits.parquet"
splits_df.to_parquet(splits_path, index=False)

# Hash the parquet for reproducibility verification
def sha256_file(path, chunk=2**20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


splits_sha = sha256_file(splits_path)

metadata = {
    "freeze_timestamp_utc": freeze_timestamp,
    "freeze_purpose": "Pre-registration §3.3 — patient-level task splits before probe training (v4 freeze, supersedes v3; adds T2+T3 per amendment A12 in osf/amendment_log.md v11 SHA d68e3a9a...)",
    "freeze_version": "v4",
    "supersedes_v1_sha256": "d7cd877432fb7ee448caf21ede7123baaa0d221d5c14b82ca8d29d756d2e0ba9",
    "supersedes_v2_sha256": "064f5c7966924fe88e8b62962714d37ad0c7643d6a3405fd2a6dedd64210b6e6",
    "supersedes_v3_sha256": "<TBD: hash from v3 freeze when uploaded to OSF>",
    "amendment_ref": (
        "A9a (T1 source AutoPET-I FDAT) + A9b (T5 source AutoPET-III TCIA PSMA) + "
        "A9d (v3 freeze) + A10 (T5 cohort provenance update) + "
        "A11 (T5 within-cohort test-retest exploratory secondary, GATE NOT MET) + "
        "A12a (T2 evaluation reduction Dice/HD95 → patch-classification AUROC) + "
        "A12b (T2 cohort 1200→726, 11→7 centres, MDA dominance) + "
        "A12c (T3 RFS cohort confirmation 1200→678, 132 events, 20.3% event rate)"
    ),
    "osf_doi": "10.17605/OSF.IO/DQ2JA",
    "random_seed": SEED,
    "split_fractions": {"train": TRAIN_FRAC, "cal": CAL_FRAC, "test": TEST_FRAC},
    "n_total_records": len(splits_df),
    "n_tasks": int(splits_df["task"].nunique()),
    "task_summaries": task_summaries,
    "task_splits_sha256": splits_sha,
    "v3_embedding_datasets_locked": [
        "pet-fm-bench-t1-embeddings-v3",
        "pet-fm-bench-t4-embeddings-v3",
        "pet-fm-bench-t5-embeddings-v3",
        "pet-fm-bench-t6-embeddings-v3",
        "pet-fm-bench-t7-embeddings-v3",
        "pet-fm-bench-t8-embeddings-v3",
        "pet-fm-bench-t9-embeddings-v3",
        "pet-fm-bench-hecktor-2025-embeddings-v3",
    ],
    "deferred_items": [
        "T7 outcome stratification — random split used; re-stratify by response label "
        "after T7 probe section is implemented (see PROGRESS deferred items).",
        "T2 GTVp-only sensitivity (per A12a) — uses the same patient train/test partition "
        "as primary T2 in probe_analysis.py v6 via splits_task='t2' fallback; no separate "
        "task_splits row needed.",
        "T3 HPV-stratified sensitivity (per A12c) — handled by probe_analysis.py v6 "
        "post-CV stratification on the cv_pool patients with non-missing HPV status.",
    ],
}
metadata_path = OUT_DIR / "task_splits.metadata.json"
with open(metadata_path, "w") as f:
    json.dump(metadata, f, indent=2)

# %% [markdown]
# ## 4. Final summary

# %%
print(f"\n{'='*70}")
print(f"PHASE 4 FREEZE — TASK SPLITS")
print(f"{'='*70}\n")
print(f"Total records: {len(splits_df)}")
print(f"Tasks: {sorted(splits_df['task'].unique().tolist())}")
print()
display_df = pd.DataFrame(task_summaries)
print(display_df.to_string(index=False))
print(f"\nTask splits SHA-256: {splits_sha}")
print(f"\nFreeze timestamp (UTC): {freeze_timestamp}")
print(f"\nOutput files:")
for f in sorted(OUT_DIR.glob("task_splits*")):
    print(f"  {f.name} ({f.stat().st_size/1e3:.1f} KB)")

# %% [markdown]
# ## 5. Done — upload to OSF
#
# 1. Save & Run All to commit this notebook
# 2. Download the two output files (`task_splits.parquet` +
#    `task_splits.metadata.json`)
# 3. Upload as Kaggle Dataset: `pet-fm-bench-task-splits-v4` (private, CC-BY-NC-4.0)
# 4. Upload to OSF project [aqmkb](https://osf.io/aqmkb/) under
#    `phase_4_freeze_task_splits/v4/` (NEW folder; v3 archived in OSF history)
# 5. Phase 4 v4 freeze gate satisfied — combined with Phase 1 freeze
#    (FM checkpoints) and pending Phase 2 v2 freeze (contamination audit with
#    9-task universe), the formal Phase 5 probe run via `probe_analysis.py` v6
#    can proceed once all three are in place.
