# PET-FM-Bench: Working Notebooks (Public Release)

**Project:** PET-FM-Bench — Cross-foundation-model benchmark on nuclear-medicine
imaging tasks with systematic contamination audit.

**Pre-registration DOI:** [10.17605/OSF.IO/DQ2JA](https://doi.org/10.17605/OSF.IO/DQ2JA)

**OSF storage node:** [aqmkb](https://osf.io/aqmkb/)

**Reporting standards:** TRIPOD-AI + CLAIM

**Code license:** MIT (this directory). Note that bundled data references
inherit the licences of their respective source datasets (TCIA CC BY 4.0,
FDAT CC BY-NC 4.0, HECKTOR 2025 CC BY 4.0, Vienna QUADRA Zenodo terms).

---

## What this directory contains

This is a **sanitised public snapshot** of the working analysis notebooks for
the formal Phase 5 probe analysis run. It is intended as the registration-grade
audit trail companion to:

- Pre-registration (registration.md, OSF aqmkb root)
- Amendment log (amendment_log.md, OSF aqmkb root, currently v12 covering A1–A12 + Phase 5 Closure)
- Phase 1 freeze: FM checkpoint manifest (aqmkb / phase_1_freeze_checkpoint_manifest/)
- Phase 2 v2 freeze: 9-task contamination audit (aqmkb / phase_2_freeze_contamination_audit/v2/)
- Phase 4 v4 freeze: 9-task patient splits (aqmkb / phase_4_freeze_task_splits/v4/)
- Phase 5 freeze: formal probe results, 12 CSVs (aqmkb / phase_5_freeze_formal_results/)

A reviewer reading this directory should be able to reconstruct the analysis
end-to-end given the OSF freeze artefacts and the Kaggle datasets named below.

---

## Sanitisation note

This is a **sanitised public release**. Internal portfolio cross-references
(e.g., to companion projects with shared infrastructure) have been replaced
with `(companion project)` markers. Local researcher Mac paths have been
replaced with generic placeholders. OSF storage IDs (e.g., `aqmkb` for this
project, `j5ry4` for the AutoPET-III SUV-conversion + nnU-Net companion) are
preserved because they resolve to publicly readable OSF nodes that external
reviewers can follow.

The unsanitised internal version is maintained in the project's working
directory and is not part of the registration audit trail.

---

## Notebook organisation

### Phase 1 — FM checkpoint manifest (5 notebooks)
- `00_diagnostic.py` — environment + dataset reachability check
- `05_checkpoint_manifest.py` — Phase 1 freeze artefact production

### Phase 2 — Contamination audit (4 notebooks + 3 manifest scrapes)
- `07a_fmcib_manifest.py` — FMCIB training cohort enumeration (~633 unique patients)
- `07b_ct_fm_manifest.py` — CT-FM training cohort enumeration via IDC (~24,909 patients)
- `07c_biomedclip_caption_scan.py` — PMC-OA full-mode caption scan
- `08_contamination_intersection.py` — Stage 2 patient-ID intersection (Tier 1-5)
- `09_within_patient_test.py` — Stage 3 dirty-vs-clean permutation probe (H2 within-patient)
- `10_contamination_freeze.py` — Stage 4 freeze artefact production

### Per-task preprocessing (10 notebooks)
- `t1_01_preprocess_v3.py` — AutoPET-I FDG (FDAT, 900 patients, 10,092 patches)
- `t4_01_preprocess_v3.py` — NSCLC-Radiogenomics (201 patients)
- `t5_01_preprocess_v3.py` — AutoPET-III PSMA (333 patients, 9,864 patches)
- `t6_01_preprocess_v3.py` — RIDER cancer test-retest (67 patients, 16 retest pairs)
- `t7_01_preprocess_v3.py` — ACRIN-NSCLC-FDG-PET response (230 patients)
- `t8_01_preprocess_v3.py` — Lung-PET-CT-Dx subtype (133 patients)
- `t9_vienna_quadra_download_preprocess.py` — Vienna QUADRA healthy test-retest (48 subjects)
- `hecktor_01_preprocess_v3.py` — HECKTOR 2025 HN tumour (T2: 680 patients / 3,708 patches; T3: 651 with valid RFS)

### Per-task embedding extraction (8 notebooks)
- `tX_02_embeddings.py` for X ∈ {t1, t4, t5, t6, t7, t8, t9, hecktor}
  — extracts 6 FM embeddings (BiomedCLIP, CT-FM, DINOv2, FMCIB, RAD-DINO,
  random_init) per patch / per patient

### Multi-seed random_init baselines (4 notebooks, satisfying amendment A3)
- `08_random_init_multiseed.py` — original 5-task version (T4/T6/T7/T8/T9)
- `08_t1_random_init_multiseed.py` — T1 multi-seed
- `08_t5_random_init_multiseed.py` — T5 multi-seed
- `08_hecktor_random_init_multiseed.py` — HECKTOR (T2 + T3) multi-seed

### Phase 3 — IBSI baseline + sensitivity analyses (3 notebooks)
- `11_pyradiomics_baseline.py` — IBSI-validated MIRP test-retest CCC for T6 + T9 (per A8)
- `12_ctfm_signflip_sensitivity.py` — registration §5.6.8 negative-transfer check
- `13_fmcib_saturation_diagnostic.py` — FMCIB embedding-manifold variance audit

### Phase 4 — Patient-level task splits
- `06_task_splits.py` — produces `task_splits.parquet` per registration §3.3

### Phase 5 — Formal probe analysis
- `probe_analysis.py` — consumes all freeze artefacts; emits 11 result CSVs

### Auxiliary diagnostics
- `01_suv_smoke_test.py` — cross-vendor SUV-conversion validation
- `02_t7_dose_investigation.py`, `03_t7_t9_verification.py`,
  `04_t7_scan_id_reconstruction.py` — T7-specific data-quality investigations
- `07_philips_anomaly_investigation.py` — Philips DICOM Decay Correction audit
- `enumerate_autopet_iii_serial_pairs.py` — Amendment A11 decision-gate
  enumerator (gate NOT MET — AutoPET-III has 0 same-tracer ≤8wk pairs)

---

## Execution order (for reproduction)

The notebooks are designed to run in three layered phases on Kaggle:

```
[per-task preprocessing on Colab] → tarball → [Kaggle dataset]
                                                     ↓
[per-task embedding extraction on Kaggle GPU] → [Kaggle dataset]
                                                     ↓
[per-task multi-seed random_init on Kaggle GPU] → [Kaggle dataset]
                                                     ↓
[Phase 4 task_splits regenerate] → [Kaggle dataset + OSF freeze]
                                                     ↓
[Phase 2 Stage 1 manifests on Kaggle CPU] → [Kaggle datasets]
                                                     ↓
[Phase 2 Stage 2/3/4 (08/09/10)] → [Kaggle dataset + OSF freeze]
                                                     ↓
[Phase 5 formal probe_analysis] → [Kaggle dataset + OSF freeze]
```

Per-task work is independent and parallelisable across Kaggle accounts.
Phase 2 + Phase 4 + Phase 5 are sequentially dependent.

---

## Kaggle datasets attached for the formal Phase 5 probe run

20 datasets total (~38 GB):

**Embeddings (8):** `pet-fm-bench-{t1,t4,t5,t6,t7,t8,t9}-embeddings-v3` +
`pet-fm-bench-hecktor-2025-embeddings-v3`

**Multi-seed random_init (8):** `pet-fm-bench-{t1,t5,hecktor}-randominit-multiseed-v3`
+ `pet-fm-bench-{t4,t6,t7,t8,t9}-randominit-multiseed`

**Phase 4 v4 task splits:** `pet-fm-bench-task-splits-v4`

**Phase 2 v2 contamination freeze:** `pet-fm-bench-contamination-freeze-v2`

**IBSI baseline:** `pet-fm-bench-pyradiomics-baseline`

**T7 patches (for outcome derivation):** `pet-fm-bench-t7-patches-v3`

---

## Amendments referenced in code

This codebase reflects amendments **A1 through A12** logged in
`osf/amendment_log.md` v12 (OSF aqmkb root, SHA-256
`ad4a84a0a48754d5997ed0f3dbff8ef0b717f880b2d12bb1c34a42faff62ad3d`),
together with the appended "Phase 5 — Formal Probe Analysis Closure"
section that documents the registered analysis closure on 2026-04-30
(12 result CSVs frozen at `aqmkb/phase_5_freeze_formal_results/`).

Inline references in notebook markdown indicate which amendment each
methodological choice corresponds to:

- **A1**: Phase 4 v2 corrective freeze (60/20/20 → 70/15/15 splits)
- **A2**: CoxPH alpha grid widened to {0.001 … 1000}
- **A3**: Multi-seed random_init baseline (N=10)
- **A4**: Lin's CCC primary, cosine secondary on test-retest
- **A5**: T7 outcome resolution → 2-yr OS (Machtay 2013)
- **A6**: probe_analysis.py consumes task_splits.parquet
- **A7**: pyradiomics 3.1 → 3.0.1 (PyPI bug)
- **A8**: pyradiomics → MIRP (Python 3.12 compat)
- **A9**: T1 source = AutoPET-I FDAT NIfTI; T5 source = TCIA AutoPET-III PSMA
- **A10**: T5 cohort provenance (333 patients / 8,768 lesions)
- **A11**: T5 within-cohort PSMA test-retest (conditional pre-reg; **gate NOT MET**)
- **A12**: HECKTOR 2025 cohort confirmation + T2 evaluation reduction
  (Dice/HD95 → patch AUROC)

---

## Code provenance

Each notebook's runtime + dependencies are documented in its top-of-file
markdown block. Foundation-model weights are loaded from public Hugging Face
hubs (DINOv2, RAD-DINO, BiomedCLIP), Zenodo (FMCIB, record 10528450), and
Lighter Zoo (CT-FM, `project-lighter/ct_fm_feature_extractor`).

External dependency note: T5 (AutoPET-III PSMA) preprocessing depends on
nnU-Net SEG NIfTI + reviewed lesion parquet from a companion project's OSF
node `j5ry4` (Conformal SUV Theranostic). The dependency is documented inline
in `t5_01_preprocess_v3.py` §1; reviewers can follow the OSF link to access
the SEG masks. The preprocessing pipeline regenerates SUV NIfTIs on the fly
via the canonical `dicom_series_to_suv_sitk` function (cross-vendor validated
on Siemens/GE/Philips at 9/9 PASS).

---

## Files in this directory

`*.py` — Jupyter notebooks in jupytext "percent" format. Convert to `.ipynb`
via:

```
find . -name '*.py' -not -name 'sanitise_notebooks.py' \
  -exec jupytext --to notebook {} \;
```

`README.md` — this file.

`probe_analysis_v5_backup.py` and `06_task_splits_v3_backup.py` — explicitly
EXCLUDED from this public release. They exist only in the working directory
to preserve supersession provenance for the freeze chain.

---

## Citation

When citing this codebase, please cite the pre-registration:

> Farquhar H. PET-FM-Bench: Cross-foundation-model benchmark on nuclear-medicine
> imaging tasks with systematic contamination audit. OSF pre-registration.
> DOI: 10.17605/OSF.IO/DQ2JA. Registered 2026-04-25.

The associated manuscript (in preparation) will provide the formal results
synthesis and is targeted for Radiology: AI / EJNMMI Research / Journal of
Nuclear Medicine.
