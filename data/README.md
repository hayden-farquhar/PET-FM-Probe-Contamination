# Data acquisition

This repository does not redistribute patient-level imaging data. All eight cohorts are publicly available with permissive licenses; this guide tells you exactly where to obtain each one and how the analysis notebooks expect them to be laid out on disk.

A separate **public result CSV dataset** (the Phase 5 freeze + Phase 2 v2 contamination audit) is deposited on Kaggle at [`pet-fm-bench-formal-probe-results-v1`](https://www.kaggle.com/datasets/haydenfarquhar/pet-fm-bench-formal-probe-results-v1) (CC0-1.0). That dataset is enough for figure regeneration and any verification of the manuscript's reported numerics — it does not require any patient cohort access.

## Cohorts (full reproduction path)

### T1 — AutoPET-I FDAT (FDG lesion classification, 900 patients)

| | |
|---|---|
| URL | https://doi.org/10.57754/FDAT.wf9fy-txq84 |
| Access | Free, FDAT account registration required |
| License | CC BY 4.0 |
| Notebooks | `notebooks/t1_01_preprocess_v3.py` (preprocess) → `notebooks/t1_02_embeddings.py` (embeddings) |
| Expected layout | DICOM series organised by `<patient_id>/<study_uid>/<series_uid>/` |

The amendment log (A9a) explains why T1 was switched from AutoPET-II/III to AutoPET-I FDAT — the canonical lesion-tumour annotations on FDAT are the registered source.

### T2 / T3 — HECKTOR 2025 (head-and-neck binary + RFS survival, 680 / 651 patients)

| | |
|---|---|
| URL | https://hecktor25.grand-challenge.org/ |
| Access | Free, Grand Challenge account + DUA agreement required |
| License | Challenge-specific (per HECKTOR DUA) |
| Notebooks | `notebooks/hecktor_01_preprocess_v3.py` → `notebooks/hecktor_02_embeddings.py` |
| Expected layout | DICOM-format CT and PT series with corresponding RTSTRUCT contours per patient |

T2 evaluation is patch-classification AUROC (per amendment A12a, reduced from Dice/HD95 segmentation). T3 is the recurrence-free-survival subset (651 patients with valid RFS labels, 132 events at probe time per amendment A12c).

### T4 — NSCLC-Radiogenomics (Bakr 2018; 201 patients, 63 events)

| | |
|---|---|
| URL | https://www.cancerimagingarchive.net/collection/nsclc-radiogenomics/ |
| Access | Free, TCIA account required |
| License | CC BY 3.0 |
| Notebooks | `notebooks/t4_01_preprocess_v3.py` → `notebooks/t4_02_embeddings.py` |
| Expected layout | TCIA standard layout |

### T5 — AutoPET-III PSMA (cross-tracer zero-shot; 333 patients, 9,864 patches)

| | |
|---|---|
| URL | https://autopet-iii.grand-challenge.org/ |
| Access | Free, Grand Challenge account + DUA agreement required |
| License | CC BY 4.0 (per amendment A9 correction; original release was CC BY-NC 4.0) |
| Notebooks | `notebooks/t5_01_preprocess_v3.py` → `notebooks/t5_02_embeddings.py` |

The companion-project nnU-Net LesionTracer segmentations on OSF [`j5ry4`](https://osf.io/j5ry4/) supply the lesion segmentations used at probe time (per amendment A10).

### T6 — RIDER-Lung-PET-CT (cancer-patient test–retest; 67 patients, 16 retest pairs)

| | |
|---|---|
| URL | https://www.cancerimagingarchive.net/collection/rider-lung-pet-ct/ |
| Access | Free, TCIA account required |
| License | TCIA Limited |
| Notebooks | `notebooks/t6_01_preprocess_v3.py` → `notebooks/t6_02_embeddings.py` |

Retest pairs are identified by the standard RIDER metadata; 16 valid pairs across 67 patients.

### T7 — ACRIN-NSCLC-FDG-PET (response prediction; 230 patients, 31 held-out, 11 events)

| | |
|---|---|
| URL | https://www.cancerimagingarchive.net/collection/acrin-nsclc-fdg-pet/ |
| Access | Free, TCIA account required |
| License | TCIA Limited |
| Notebooks | `notebooks/t7_01_preprocess_v3.py` → `notebooks/t7_02_embeddings.py` |

Outcome is operationalised as 2-year overall survival per amendment A5 (validated against published 57.5 % mortality rate; observed 57.2 %).

### T8 — Lung-PET-CT-Dx (lung subtype classification; 133 patients, 20 held-out)

| | |
|---|---|
| URL | https://www.cancerimagingarchive.net/collection/lung-pet-ct-dx/ |
| Access | Free, TCIA account required |
| License | TCIA Limited |
| Notebooks | `notebooks/t8_01_preprocess_v3.py` → `notebooks/t8_02_embeddings.py` |

### T9 — Vienna QUADRA Whole-Body FDG Test-Retest (healthy-control; 48 retest pairs)

| | |
|---|---|
| Dataset URL | https://zenodo.org/records/16686025 |
| Companion paper | https://doi.org/10.1038/s41597-025-05997-4 (Gutschmayer et al., *Sci Data* 2025;12:1855) |
| Access | Free download |
| License | CC BY 4.0 |
| Notebooks | `notebooks/t9_vienna_quadra_download_preprocess.py` → `notebooks/t9_vienna_quadra_embed.py` (alternative entry: `t9_01_preprocess.py` → `t9_02_embeddings.py`) |

48 healthy controls scanned in a fixed test–retest FDG PET/CT protocol on a Siemens Biograph Vision QUADRA total-body scanner.

## Foundation model checkpoints

| FM | Source | License |
|---|---|---|
| FMCIB | [Zenodo 10528450](https://zenodo.org/records/10528450) | MSRLA |
| CT-FM | [HuggingFace `project-lighter/ct_fm_feature_extractor`](https://huggingface.co/project-lighter/ct_fm_feature_extractor) | Apache-2.0 |
| BiomedCLIP | [HuggingFace `microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224`](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224) | MIT |
| RAD-DINO | [HuggingFace `microsoft/rad-dino`](https://huggingface.co/microsoft/rad-dino) | MSRLA |
| DINOv2 | [HuggingFace `facebook/dinov2-base`](https://huggingface.co/facebook/dinov2-base) | Apache-2.0 |

Pinned SHA-256 fingerprints are encoded in `notebooks/05_checkpoint_manifest.py` and verified at runtime.

Two FMs were pre-registered but dropped before probe analysis (amendment log v12): **Merlin** failed to install reproducibly on the Kaggle T4 environment; **Pillar-0** was gate-access and could not be obtained within the registered 18-week window.

## Public Kaggle result dataset (for verification, not full reproduction)

```bash
# Anyone can grab this without any cohort access:
kaggle datasets download -d haydenfarquhar/pet-fm-bench-formal-probe-results-v1 --unzip
```

Contains 13 CSVs (12 result CSVs + `contamination_audit.csv`). The contamination audit CSV is bit-identical (SHA-256 `7e3b7b73…`) to the OSF Phase 2 v2 freeze.

## OSF freeze artefacts

- Pre-registration: [OSF `aqmkb`](https://osf.io/aqmkb/) (DOI `10.17605/OSF.IO/DQ2JA`)
- Phase 1 FM checkpoint manifest
- Phase 2 v2 contamination audit
- Phase 4 v4 patient-level splits
- Phase 5 formal probe results (12 result CSVs + Phase 2 audit)
- A11 closure artefacts (gate not met)
- Sanitised public code snapshot tarball
- Amendment log v12 (SHA-256 `ad4a84a0…`)
- Ethics determination
