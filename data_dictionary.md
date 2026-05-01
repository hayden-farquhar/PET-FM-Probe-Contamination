# Data dictionary — PET-FM-Bench public result CSVs

Public Kaggle dataset: [`haydenfarquhar/pet-fm-bench-formal-probe-results-v1`](https://www.kaggle.com/datasets/haydenfarquhar/pet-fm-bench-formal-probe-results-v1) (CC0-1.0).

This dictionary defines every column in every CSV that feeds the manuscript's reported numerics, the figures, and the supplementary materials.

## Common column conventions

| Column | Type | Description |
|---|---|---|
| `fm` | str | Foundation model identifier — one of `fmcib`, `ct_fm`, `biomedclip`, `rad_dino`, `dinov2`, `ibsi_radiomics_baseline`, `random_init` |
| `task` | str | Task identifier — one of `t1`, `t2`, `t2_gtvp_only`, `t3`, `t4`, `t5`, `t6`, `t7`, `t8`, `t9` |
| `metric` | str | Metric name — one of `auroc`, `auprc`, `c_index`, `lin_ccc` |
| `value` | float | Point estimate of the metric |
| `ci_low` | float | Lower bound of 1,000-resample patient-clustered bootstrap 95 % CI (FMs); 10-seed minimum (random_init) |
| `ci_high` | float | Upper bound of CI (analogous) |

## `fm_task_matrix.csv`

| Column | Type | Description |
|---|---|---|
| `fm` | str | FM row label |
| `t1`, `t2`, `t2_gtvp_only`, `t3`, `t4`, `t5`, `t6`, `t7`, `t8`, `t9` | float | Per-task primary-metric point estimate. Empty for IBSI on classification/survival tasks (deferred analysis per Methods). |

## `all_probe_results.csv`

Long-format master table with one row per (FM, task) cell. Columns include the common conventions above plus task-specific extras:

| Column | Type | Description |
|---|---|---|
| `metric` | str | Metric name (`auroc`, `c_index`, `lin_ccc`, etc.) |
| `value` | float | Point estimate |
| `ci_low`, `ci_high` | float | Bootstrap CI bounds |
| `auprc`, `brier` | float | Secondary classification metrics (where applicable) |
| `best_C` | float | Selected logistic-regression regularisation strength (classification probes) |
| `best_alpha` | float | Selected Cox L2 regularisation strength (survival probes) |
| `n_train_patches`, `n_test_patches`, `n_test_lesion`, `n_test_background` | int | Patch counts at train/test partitions |
| `n_patients`, `n_events` | int | Patient and event counts (survival tasks) |
| `view` | str | 2D-FM view selector (`axial` / `volume` / `mip`) |
| `layer` | str | FM depth-stage selector (`cls` / `pool` / `early` / `middle`) |
| `n_seeds`, `seed_min`, `seed_max` | int / float | Random-init multi-seed metadata (only populated for `random_init` rows) |
| `contamination_tier` | int | Phase 2 v2 audit tier (1, 2, 3, 5) |
| `contamination_overlap_fraction` | float | Patient-level overlap fraction (0–1; only populated for Tier 1+2 cells) |

## Per-task CSVs

`t1_lesion_patch_results.csv`, `t2_lesion_patch_results.csv`, `t2_gtvp_only_results.csv`, `t3_survival_results.csv`, `t4_survival_results.csv`, `t5_zero_shot_results.csv`, `t7_response_results.csv`, `t8_classification_results.csv` share the same long-format schema with task-specific subset of columns.

`t6_test_retest_results.csv` and `t9_test_retest_results.csv` share a richer schema:

| Column | Type | Description |
|---|---|---|
| `task` | str | `t6` or `t9` |
| `fm` | str | FM identifier |
| `metric` | str | `lin_ccc` (Lin's CCC is the registered primary metric for test–retest) |
| `value` | float | Point estimate of Lin's CCC |
| `ci_low`, `ci_high` | float | Patient-clustered bootstrap 95 % CI bounds |
| `sw_mean`, `sw_std`, `sw_min`, `sw_ci_low`, `sw_ci_high` | float | Within-pair cosine similarity statistics (secondary diagnostic; cosine gap retained per amendment A4) |
| `sb_mean`, `sb_std` | float | Between-patient cosine similarity statistics (null distribution for the cosine-gap diagnostic) |
| `cosine_gap`, `cosine_gap_ci_low`, `cosine_gap_ci_high` | float | Cosine-gap (within − between) point estimate and bootstrap CI |
| `n_within_pairs` | int | Number of within-patient retest pairs (T6 = 16; T9 = 48) |
| `n_between_pairs` | int | Number of between-patient pairs in the null |
| `perm_p_value_sw_gt_sb` | float | Permutation p-value for "within > between" |
| `n_seeds`, `seed_min`, `seed_max` | int / float | Random-init multi-seed metadata (only populated for `random_init`) |

## `contamination_audit.csv`

Long-format Phase 2 v2 audit, 30 rows (the 24 Tier 5 cells are implicit; the audit only enumerates the 30 non-Tier-5 cells per the registered protocol).

| Column | Type | Description |
|---|---|---|
| `fm` | str | FM identifier |
| `task` | str | Task identifier (`t1`–`t9`) |
| `n_eval` | int | Number of patients in the held-out evaluation cohort |
| `n_contaminated` | int | Number of `n_eval` patients also in this FM's published training manifest |
| `n_clean` | int | `n_eval − n_contaminated` |
| `overlap_fraction` | float | `n_contaminated / n_eval` |
| `tier` | int | Five-tier confidence schema (1 = study-UID match; 2 = patient-ID match same collection; 3 = institutional-context proxy; 5 = declared clean by training-data construction) |
| `tier_rationale` | str | Free-text rationale for the assigned tier |
| `metric` | str | Primary metric used for within-patient permutation (`auroc` / `c_index`) |
| `status` | str | Permutation-test status (`ran`, `skipped:no_probe_runner`, `skipped:no_contamination_by_construction`, `skipped:zero_shot_task`, `skipped:test_retest_task`, `skipped:insufficient_subsets`) |
| `dirty_value` | float | Held-out metric on the dirty (overlapping) subset |
| `clean_value` | float | Held-out metric on the clean (non-overlapping) subset |
| `delta` | float | `dirty_value − clean_value` |
| `perm_p_value` | float | One-tailed permutation p-value (registered direction; 100 permutations per amendment) |
| `n_perm` | int | Permutations used (100 per the amendment-acknowledged reduction from the registered 10,000) |

## Five-tier contamination schema

| Tier | Definition | Example |
|---|---|---|
| 1 | Confirmed study-UID match between the FM's published training manifest and the evaluation cohort | CT-FM × T7 (ACRIN-NSCLC-FDG-PET in CT-FM's TCIA training corpus) |
| 2 | Patient-ID match within the same TCIA collection and modality | BiomedCLIP × T4 (6.5 % patient-ID overlap via PMC-15M caption-matched images) |
| 3 | Institutional-context proxy (no patient-level match documented but training and evaluation share institutional provenance) | BiomedCLIP × T6 (institutional context match without direct overlap) |
| 4 | Caption-scan or institutional-provenance proxy (registered Tier; not exercised in the v2 audit) | (none in v2) |
| 5 | Declared clean by training-data construction | DINOv2 × any task (ImageNet-only); RAD-DINO × any task (chest-X-ray-only); random_init (untrained) |

## Reproducibility hashes

| Artefact | SHA-256 |
|---|---|
| Phase 5 freeze CSVs (12 result CSVs) | per-file in OSF amendment log v12 |
| Phase 2 v2 contamination audit (`contamination_audit.csv`) | `7e3b7b7344177027b9d0db924466193bacdcef0ed02d01fe99b3a6183205eb44` |
| Phase 4 v4 patient-level splits (`task_splits.parquet`) | `3855e483…` |
| Run-time pipeline (`probe_analysis.py` v6, 2,960 lines) | `7ca32e8bb244845df6313de146d01ac8aaf7871cfcbccb2fdb61e62374227401` |
| Amendment log v12 | `ad4a84a0a48754d5997ed0f3dbff8ef0b717f880b2d12bb1c34a42faff62ad3d` |
