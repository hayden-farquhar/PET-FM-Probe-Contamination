# PET-FM-Bench

Code repository for: **Pretraining Domain Predicts Test–Retest Reproducibility of Foundation Models on PET: A Pre-Registered Nine-Task Benchmark with Contamination Audit (PET-FM-Bench).**

Hayden Farquhar MBBS MPHTM. Independent Researcher, Finley NSW, Australia. ORCID [0009-0002-6226-440X](https://orcid.org/0009-0002-6226-440X).

| | Link |
|---|---|
| Pre-registration | OSF [10.17605/OSF.IO/DQ2JA](https://doi.org/10.17605/OSF.IO/DQ2JA) (registered 2026-04-25) |
| Result CSVs (Phase 5 freeze + Phase 2 v2 contamination audit) | Kaggle [`pet-fm-bench-formal-probe-results-v1`](https://www.kaggle.com/datasets/haydenfarquhar/pet-fm-bench-formal-probe-results-v1) (CC0-1.0) |
| Preprint | medRxiv (DOI to be added when posted) |
| Software DOI | Zenodo (concept DOI to be added on first release) |

## Overview

PET-FM-Bench is a pre-registered cross-foundation-model benchmark on nine downstream nuclear-medicine imaging tasks. Six foundation models — FMCIB, CT-FM, BiomedCLIP, RAD-DINO, DINOv2, and a 10-seed random-initialisation control — are evaluated against an IBSI-compliant pyradiomics baseline (MIRP v2.5+) using frozen-embedding linear or Cox probes at fixed patient-level splits with `SEED = 42`. A 54-cell (FM × task) contamination audit classifies patient-level overlap between each FM's training manifest and each evaluation cohort into a five-tier confidence schema. This repository contains every analysis script that produced the result CSVs, the rendered figures, and the manuscript's reported numerics.

The headline registration-grade finding is that **pretraining-domain match to PET, not foundation-model status per se, predicts test–retest reproducibility**: only the two FMs pretrained on three-dimensional medical CT (FMCIB and CT-FM) outperform IBSI radiomics on cancer-patient and healthy-control retest cohorts.

## Repository contents

| Folder | Purpose |
|---|---|
| `notebooks/` | All analysis Kaggle notebooks (preprocess + embed per task, contamination audit, IBSI baseline, probes, figure rendering). Plus `manuscript_figures_all.ipynb` for one-click figure regeneration. |
| `src/` | Python package modules used by multiple notebooks (probe classes, contamination utilities, preprocessing helpers). |
| `data/` | Cohort acquisition instructions. The patient-level imaging cohorts are not redistributed here — see `data/README.md` for download URLs and access requirements per cohort. |
| `outputs/figures/` | The four manuscript figures rendered at 300 dpi PNG and vector PDF, regeneratable from `notebooks/manuscript_figures_all.py`. |
| `docs/reproduction_guide.md` | Step-by-step end-to-end reproduction guide. |
| `data_dictionary.md` | Column definitions for every CSV in the public Kaggle result dataset. |
| `pyproject.toml` | Python dependency specification. |
| `.zenodo.json` | Software-deposit metadata (used by GitHub↔Zenodo sync to mint DOIs on release). |
| `CITATION.cff` | GitHub citation widget metadata. |

## Data sources

| Source | URL | Access | License |
|---|---|---|---|
| AutoPET-I FDAT (T1 — FDG lesion classification, 900 patients) | [doi.org/10.57754/FDAT.wf9fy-txq84](https://doi.org/10.57754/FDAT.wf9fy-txq84) | Free; FDAT registration | CC BY 4.0 |
| HECKTOR 2025 (T2 / T3 — head-and-neck) | [grand-challenge.org/competitions/hecktor25](https://hecktor25.grand-challenge.org/) | Free; competition registration + DUA | Challenge-specific |
| NSCLC-Radiogenomics (T4) | [TCIA collection NSCLC-Radiogenomics](https://www.cancerimagingarchive.net/collection/nsclc-radiogenomics/) | Free; TCIA terms | CC BY 3.0 |
| AutoPET-III (T5 — PSMA zero-shot; companion-project nnU-Net segmentations on OSF j5ry4) | [autopet-iii.grand-challenge.org](https://autopet-iii.grand-challenge.org/) | Free; competition registration | CC BY 4.0 |
| RIDER-Lung-PET-CT (T6 — cancer-patient test–retest) | [TCIA collection RIDER-Lung-PET-CT](https://www.cancerimagingarchive.net/collection/rider-lung-pet-ct/) | Free; TCIA terms | TCIA Limited |
| ACRIN-NSCLC-FDG-PET (T7) | [TCIA collection ACRIN-NSCLC-FDG-PET](https://www.cancerimagingarchive.net/collection/acrin-nsclc-fdg-pet/) | Free; TCIA terms | TCIA Limited |
| Lung-PET-CT-Dx (T8) | [TCIA collection Lung-PET-CT-Dx](https://www.cancerimagingarchive.net/collection/lung-pet-ct-dx/) | Free; TCIA terms | TCIA Limited |
| Vienna QUADRA Whole-Body FDG Test-Retest (T9 — healthy-control test–retest) | [Zenodo 16686025](https://zenodo.org/records/16686025); paper: [Sci Data 12:1855 (2025)](https://doi.org/10.1038/s41597-025-05997-4) | Free | CC BY 4.0 |

Per-cohort acquisition steps and expected directory structures are in [`data/README.md`](data/README.md).

## Foundation-model checkpoints

All FM checkpoints used are publicly hosted. The repository does not redistribute the weights — they are loaded at runtime from their canonical sources:

| FM | Source | License |
|---|---|---|
| FMCIB | [Zenodo 10528450](https://zenodo.org/records/10528450) | MSRLA |
| CT-FM | [HuggingFace `project-lighter/ct_fm_feature_extractor`](https://huggingface.co/project-lighter/ct_fm_feature_extractor) | Apache-2.0 |
| BiomedCLIP | [HuggingFace `microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224`](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224) | MIT |
| RAD-DINO | [HuggingFace `microsoft/rad-dino`](https://huggingface.co/microsoft/rad-dino) | MSRLA |
| DINOv2 | [HuggingFace `facebook/dinov2-base`](https://huggingface.co/facebook/dinov2-base) | Apache-2.0 |

Checkpoint SHA-256 fingerprints are pinned in the [Phase 1 freeze manifest](https://osf.io/aqmkb/) and verified at run time by `notebooks/05_checkpoint_manifest.py`.

## Requirements

- Python 3.12 (3.10+ supported; pinned at 3.12.4 for the published run)
- Kaggle T4 GPU runtime (free tier sufficient) for the embedding-extraction notebooks; CPU is sufficient for the probe pipeline and figure rendering.

```bash
pip install -e .[dev]
```

Pinned versions of the analysis-critical libraries (matching the published run): scikit-learn 1.5.0, scikit-survival 0.24.0, numpy 2.0.1, pandas 2.2.2, MIRP 2.5.1.

## Quick reproduction (lightweight path)

If you only want to verify the manuscript's reported numerics from the public result CSVs and regenerate Figures 1–4 (no GPU, no patient cohorts required):

```bash
# 1. Clone this repository
git clone https://github.com/hayden-farquhar/PET-FM-Probe-Contamination.git
cd PET-FM-Probe-Contamination

# 2. Install Python dependencies
pip install pandas numpy matplotlib

# 3. Download the public result CSVs from Kaggle
#    (or attach the Kaggle dataset to a Kaggle notebook and run from there)
kaggle datasets download -d haydenfarquhar/pet-fm-bench-formal-probe-results-v1 --unzip -p data/freeze_csvs/

# 4. Render Figures 1–4
PET_FM_BENCH_INPUT_DIR=data/freeze_csvs python3 notebooks/manuscript_figures_all.py
```

Outputs land in `/kaggle/working/` by default; on a local machine set `PET_FM_BENCH_OUTPUT_DIR=outputs/figures/` to redirect.

## Full reproduction (re-derive result CSVs from raw)

The full pipeline regenerates the 12 result CSVs and the contamination audit from raw imaging cohorts. See [`docs/reproduction_guide.md`](docs/reproduction_guide.md) for the step-by-step path. Order of operations:

1. Acquire all 8 cohorts per `data/README.md`.
2. Per-task preprocessing notebooks: `t{N}_01_preprocess*.py` (or `hecktor_01_preprocess_v3.py` for T2/T3).
3. Per-task embedding extraction: `t{N}_02_embeddings.py` (or `hecktor_02_embeddings.py`).
4. Phase 1 FM checkpoint manifest verification: `05_checkpoint_manifest.py`.
5. Phase 2 contamination audit: `08_contamination_intersection.py` → `09_within_patient_test.py` → `10_contamination_freeze.py`.
6. Phase 4 patient-level splits: `06_task_splits.py`.
7. IBSI radiomics baseline (T6, T9): `11_pyradiomics_baseline.py`.
8. Random-initialisation 10-seed controls: `08_t1_random_init_multiseed.py`, `08_t5_random_init_multiseed.py`, `08_hecktor_random_init_multiseed.py`, `08_random_init_multiseed.py`.
9. Phase 5 main probe analysis: `probe_analysis.py` (the centrepiece; deterministic at SEED = 42).
10. Figure rendering: `manuscript_figures_all.py` (or `manuscript_figures_all.ipynb`).

The Phase 5 freeze CSVs are bit-identical between independent runs separated by minor patches (verified via SHA-256). Run-time SHA-256 of `probe_analysis.py` v6 at the published run: `7ca32e8bb244845df6313de146d01ac8aaf7871cfcbccb2fdb61e62374227401`.

## Pre-registration and amendments

The benchmark was pre-registered on the Open Science Framework on 2026-04-25 prior to any probe analysis (DOI [10.17605/OSF.IO/DQ2JA](https://doi.org/10.17605/OSF.IO/DQ2JA)). Twelve methodological deviations (A1–A12) were logged in the OSF amendment log before each affected analysis was executed. The amendment log v12 is deposited at the OSF project root with SHA-256 `ad4a84a0a48754d5997ed0f3dbff8ef0b717f880b2d12bb1c34a42faff62ad3d`.

A summary of amendments and their rationales is in the manuscript Methods §"Registered deviations (A1–A12)".

## Outputs in this repository

| File | Manuscript reference |
|---|---|
| `outputs/figures/figure_1_heatmap.{png,pdf}` | Figure 1 (Cross-FM × cross-task primary-metric matrix) |
| `outputs/figures/figure_2_contamination.{png,pdf}` | Figure 2 (Phase 2 v2 contamination tier matrix) |
| `outputs/figures/figure_3_h6_test_retest.{png,pdf}` | Figure 3 (H6 — test–retest CCC vs IBSI) |
| `outputs/figures/figure_4_forest.{png,pdf}` | Figure 4 (per-task FM ranking forest plot) |

The 12 result CSVs that feed these figures are deposited as the public Kaggle dataset (link above) and on the OSF project (Phase 5 freeze) with verified SHA-256 hashes.

## Citation

If you use this code or its outputs, please cite:

```
Farquhar H. Pretraining Domain Predicts Test–Retest Reproducibility of
Foundation Models on PET: A Pre-Registered Nine-Task Benchmark with
Contamination Audit (PET-FM-Bench). 2026.
Pre-registration: OSF DOI 10.17605/OSF.IO/DQ2JA.
```

A `CITATION.cff` is included so GitHub auto-renders the citation widget.

## License

- **Code** (everything in `notebooks/`, `src/`, and the analysis scripts): MIT License.
- **Documentation, data dictionary, figures, and README**: CC-BY 4.0.

See [`LICENSE`](LICENSE) for full text.

## Contact

Hayden Farquhar — `hayden.farquhar@icloud.com` — ORCID [0009-0002-6226-440X](https://orcid.org/0009-0002-6226-440X).

Issues, bug reports, and pull requests welcome via [GitHub issues](https://github.com/hayden-farquhar/PET-FM-Probe-Contamination/issues).
