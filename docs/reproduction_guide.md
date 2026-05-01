# Reproduction guide

Two reproduction paths are supported, in order of effort:

1. **Verification path** (~15 minutes, no GPU, no patient cohorts): regenerate Figures 1–4 from the public result CSVs and verify the manuscript's reported numerics.
2. **Full reproduction path** (multi-day; requires Kaggle T4 GPU access and acquisition of all eight cohorts): re-derive the 12 result CSVs and the contamination audit from raw imaging.

## Path 1 — Verification

```bash
# 1. Clone this repository
git clone https://github.com/hayden-farquhar/PET-FM-Probe-Contamination.git
cd PET-FM-Probe-Contamination

# 2. Install the few dependencies needed for figure rendering
pip install pandas numpy matplotlib

# 3. Pull the public result CSVs from Kaggle
#    (uses the kaggle CLI; see https://github.com/Kaggle/kaggle-api for setup)
kaggle datasets download -d haydenfarquhar/pet-fm-bench-formal-probe-results-v1 \
       --unzip -p data/freeze_csvs/

# 4. Render Figures 1–4
PET_FM_BENCH_INPUT_DIR=data/freeze_csvs \
PET_FM_BENCH_OUTPUT_DIR=outputs/figures \
python3 notebooks/manuscript_figures_all.py
```

Outputs: `outputs/figures/figure_{1..4}_*.{png,pdf}`. These should be bit-similar (within rounding) to the figures shipped in this repository.

To verify a specific manuscript number:

```python
import pandas as pd
df = pd.read_csv("data/freeze_csvs/t6_test_retest_results.csv")
print(df.query("fm == 'fmcib'")[["fm", "value", "ci_low", "ci_high"]])
# Expected: FMCIB Lin's CCC = 0.779 (95 % CI 0.494–0.845) — matches manuscript H6 finding.
```

## Path 2 — Full reproduction

The full pipeline regenerates the 12 result CSVs and the contamination audit from raw imaging. Read this section before attempting it; the raw cohorts are a non-trivial amount of data and the embedding extraction needs a GPU.

### Prerequisites

- Kaggle T4 GPU runtime (free tier) — used for FM embedding extraction (~150 T4-hours total)
- Local CPU — used for the IBSI baseline, the probe pipeline, the contamination audit, and figure rendering
- ~200 GB of local storage for the eight cohorts after preprocessing

### Step 1 — Acquire cohorts

Follow the per-cohort instructions in [`../data/README.md`](../data/README.md). All eight cohorts are publicly available; some require free Grand Challenge / TCIA / FDAT account registration. The Vienna QUADRA test-retest (T9) is the fastest to acquire (single Zenodo download).

### Step 2 — Per-task preprocessing and embedding extraction

For each task, run preprocessing first, then embedding extraction. These are Kaggle-runnable notebooks; on Kaggle, attach the relevant cohort dataset and the FM checkpoint dataset, then "Save & Run All".

```text
T1:   notebooks/t1_01_preprocess_v3.py            → notebooks/t1_02_embeddings.py
T2,T3: notebooks/hecktor_01_preprocess_v3.py      → notebooks/hecktor_02_embeddings.py
T4:   notebooks/t4_01_preprocess_v3.py            → notebooks/t4_02_embeddings.py
T5:   notebooks/t5_01_preprocess_v3.py            → notebooks/t5_02_embeddings.py
T6:   notebooks/t6_01_preprocess_v3.py            → notebooks/t6_02_embeddings.py
T7:   notebooks/t7_01_preprocess_v3.py            → notebooks/t7_02_embeddings.py
T8:   notebooks/t8_01_preprocess_v3.py            → notebooks/t8_02_embeddings.py
T9:   notebooks/t9_vienna_quadra_download_preprocess.py
                                                   → notebooks/t9_vienna_quadra_embed.py
```

Outputs: per-task parquet files of patch metadata + per-FM embedding parquets keyed by patch identifier and FM name.

### Step 3 — Phase 1 freeze: FM checkpoint manifest

Verify SHA-256 of every FM checkpoint against the registered Phase 1 manifest:

```bash
python notebooks/05_checkpoint_manifest.py
```

Should report 5 FM checkpoints + IBSI baseline = 6 entries verified bit-identical.

### Step 4 — Phase 4 freeze: patient-level splits

Generate or verify the registered 70/15/15 patient-level splits:

```bash
python notebooks/06_task_splits.py
```

Outputs: `task_splits.parquet` (Phase 4 v4 freeze; SHA-256 `3855e483…`).

### Step 5 — Phase 2 contamination audit

Run the three contamination notebooks in order:

```bash
python notebooks/08_contamination_intersection.py    # patient-ID intersection per FM × task
python notebooks/09_within_patient_test.py           # within-patient permutation on Tier 1+2 cells
python notebooks/10_contamination_freeze.py          # freeze the 54-cell matrix
```

Outputs: `contamination_audit.csv` (Phase 2 v2 freeze; SHA-256 `7e3b7b73…`).

### Step 6 — IBSI radiomics baseline (T6 + T9 only)

Per amendment A8, the IBSI baseline uses MIRP v2.5+ (pyradiomics 3.1 had a PyPI packaging defect under Python 3.12).

```bash
python notebooks/11_pyradiomics_baseline.py
```

Outputs: IBSI features for T6 (RIDER-Lung) and T9 (Vienna QUADRA) at the registered configuration (`binWidth` = 0.25 SUV; isotropic 2 mm resampling; resegmentation range [0.5, ∞]).

### Step 7 — Random-initialisation 10-seed control (amendment A3)

```bash
python notebooks/08_t1_random_init_multiseed.py
python notebooks/08_t5_random_init_multiseed.py
python notebooks/08_hecktor_random_init_multiseed.py
python notebooks/08_random_init_multiseed.py        # T4, T7, T8, T9 catch-all
```

Each script runs 10 independent random initialisations of the DINOv2 ViT-B/14 architecture and produces per-seed embeddings.

### Step 8 — Phase 5 main probe pipeline

The centrepiece. Deterministic at SEED = 42 across NumPy, scikit-learn, scikit-survival, and the patient-clustered bootstrap RNG.

```bash
python notebooks/probe_analysis.py
```

Pipeline run-time SHA-256 at the published run: `7ca32e8bb244845df6313de146d01ac8aaf7871cfcbccb2fdb61e62374227401` (2,960 lines). Outputs: 12 result CSVs deterministically reproducible — bit-identical between independent runs (verified at the Phase 5 freeze).

The 12 result CSVs are the same as those distributed in the public Kaggle dataset.

### Step 9 — Render figures

```bash
PET_FM_BENCH_INPUT_DIR=<path/to/result_csvs> \
PET_FM_BENCH_OUTPUT_DIR=outputs/figures \
python notebooks/manuscript_figures_all.py
```

Or upload `notebooks/manuscript_figures_all.ipynb` to Kaggle and Save & Run All with the public result-CSV dataset attached.

### Optional — sensitivity and diagnostic notebooks

These were used to investigate specific aspects but are not required to reproduce the headline findings:

| Notebook | Purpose |
|---|---|
| `notebooks/00_diagnostic.py` | Environment / dependency sanity check |
| `notebooks/01_suv_smoke_test.py` | Cross-vendor SUV conversion validation (9-of-9 PASS at registration) |
| `notebooks/02_t7_dose_investigation.py` | T7 dose-injection metadata sensitivity |
| `notebooks/03_t7_t9_verification.py` | T7 / T9 metadata cross-checks |
| `notebooks/04_t7_scan_id_reconstruction.py` | T7 scan-ID utility (used by bug #16 fix) |
| `notebooks/07_philips_anomaly_investigation.py` | Vendor-specific Philips DICOM check |
| `notebooks/12_ctfm_signflip_sensitivity.py` | CT-FM embedding sign-flip sensitivity |
| `notebooks/13_fmcib_saturation_diagnostic.py` | FMCIB saturation diagnostic on T1 |
| `notebooks/enumerate_autopet_iii_serial_pairs.py` | A11 closure (gate not met for T5 PSMA test–retest) |

## Common pitfalls

1. **DICOM → SUV conversion.** All preprocessing notebooks call a cross-vendor-validated SUV-by-body-weight conversion module. Vendor-specific quirks (Siemens, GE, Philips) are handled. The validation suite passes 9 of 9 reference cases at registration. The conversion module is in `src/preprocess/core.py`.
2. **Patch sizing.** 3D FMs use 96-voxel cubic patches at isotropic 2 mm spacing (192 mm physical extent per side). 2D FMs use maximum-intensity projections (MIP) and three orthogonal slices at the lesion centroid (primary analysis: MIP).
3. **Background patches.** Rejection-sampled where the segmentation mask is zero and SUV < 2.5 (T1 and T5) or where neither GTVp nor GTVn is set (T2, T2-GTVp). This SUV-threshold negative-sampling is the explanatory mechanism for the saturation observed at T1, T2, T2-GTVp, and T5 — see manuscript Discussion §"H1 and the saturation of SUV-threshold patch-classification benchmarks".
4. **HECKTOR 2025 amendment trail.** T2 evaluation reduced from Dice/HD95 to patch AUROC (A12a); cohort sizes corrected to 680 (T2) and 651 (T3, with valid RFS labels at probe time) per A12b/A12c. The HECKTOR 2025 dataset has 726 total patients; some are excluded from individual tasks per the registered eligibility rules.
5. **A11 closure.** The conditional pre-registration of T5 within-cohort PSMA test–retest was closed at "gate not met" because the released AutoPET-III subset contains zero patients with same-tracer ≤ 8-week serial pairs. This was logged on OSF prior to data inspection.

## Verifying determinism

Sample 14 datapoints across T1, T2, T2-GTVp, T3, T4, T6, T8, and T9 from your run and compare to the Phase 5 freeze CSVs (or to the public Kaggle dataset). All 14 should reproduce bit-identically. If any differ:

- Check that `SEED = 42` is propagated (the `probe_analysis.py` notebook prints it at startup).
- Verify the FM checkpoint SHA-256s with `notebooks/05_checkpoint_manifest.py`.
- Verify your patient-level splits parquet against the Phase 4 v4 freeze SHA-256 (`3855e483…`).
- Verify your contamination audit CSV against the Phase 2 v2 freeze SHA-256 (`7e3b7b73…`).
- Confirm software versions (Python 3.12.4; scikit-learn 1.5.0; scikit-survival 0.24.0; numpy 2.0.1; pandas 2.2.2; MIRP 2.5.1).

A determinism-failing run indicates a divergence somewhere in the input chain — typically an FM checkpoint version, a cohort filter, or a NumPy/scikit-learn version drift.
