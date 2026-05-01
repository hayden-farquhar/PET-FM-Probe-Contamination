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
# # PET-FM-Bench: IBSI Radiomics Test-Retest Baseline (registration H6)
#
# **Runtime:** CPU | **Internet:** ON for pip install | **Time:** ~10-30 min |
# **GPU:** Not needed
#
# Pre-registration H6 ("at least one foundation model achieves higher embedding
# cosine concordance (Lin's CCC) than the corresponding IBSI pyradiomics
# feature concordance"): produces the **IBSI-compliant CCC comparator** for
# T6 (RIDER-Lung-PET-CT cancer test-retest) and T9 (Vienna QUADRA healthy
# test-retest).
#
# **Library: MIRP (Oncoray Medical Image Radiomics Processor) v2.5+** — see
# amendment A8. Registration §5.3 named pyradiomics v3.1, but neither
# pyradiomics 3.1.0 (PyPI metadata defect) nor 3.0.1 (no Python 3.12 wheel
# + setup.py uses removed `numpy.distutils`) installs cleanly on Kaggle's
# pinned Python 3.12 environment. MIRP is IBSI-validated by the IBSI
# consortium, computes the same feature families (statistical = first-order;
# morphological = shape; texture = GLCM/GLRLM/GLSZM/GLDM/NGTDM), and supports
# Python 3.12.
#
# **Datasets to attach:**
# - `pet-fm-bench-t6-patches-v3` (T6 cancer test-retest)
# - `pet-fm-bench-t9-patches` (T9 healthy test-retest — v1, by design per D6b)
#
# **Output:** `pyradiomics_test_retest_ccc.csv` — schema
# `(task, fm, metric, value, ci_low, ci_high, n_pairs, n_features)`
# matching probe_analysis.py v4's t6/t9 test-retest output. (Filename retains
# `pyradiomics_` prefix for backwards compatibility with downstream code; the
# actual library used is recorded in the metadata sidecar per amendment A8.)

# %% [markdown]
# ## 1. Setup

# %%
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# MIRP per amendment A8. Stable, Python 3.12+ compatible, IBSI-validated.
!pip install -q mirp

import SimpleITK as sitk  # noqa: E402
from mirp import extract_features  # noqa: E402

# MIRP's top-level module doesn't expose __version__; use importlib.metadata
# (Python 3.8+ stdlib) which queries the installed package's pip metadata.
from importlib.metadata import version as _pkg_version  # noqa: E402

try:
    MIRP_VERSION = _pkg_version("mirp")
except Exception:
    MIRP_VERSION = "unknown"
print(f"MIRP version installed: {MIRP_VERSION}")

OUT_DIR = Path("/kaggle/working")
OUT_DIR.mkdir(parents=True, exist_ok=True)

freeze_timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
print(f"Freeze timestamp (UTC): {freeze_timestamp}")

N_BOOTSTRAP = 1000
SUV_THRESHOLD = 0.5  # registration §5.3 resegmentRange [0.5, null]


# %% [markdown]
# ## 2. MIRP feature extraction (registered config, mapped to MIRP API)
#
# Mapping registration §5.3 → MIRP settings:
# - `binWidth: 0.25` → `discretisation_bin_width=0.25` with fixed-bin-width method
# - `resampledPixelSpacing: [2, 2, 2]` → `new_spacing=[2.0, 2.0, 2.0]`
# - `resegmentRange: [0.5, null]` → SUV > 0.5 mask + `resegmentation_intensity_range=[0.5, np.inf]`
# - feature classes shape/firstorder/GLCM/GLRLM/GLSZM/GLDM/NGTDM →
#   MIRP `base_feature_families = ["statistical", "morphological",
#                                  "texture"]` (texture includes all 5 GL-*
#   matrix-based families per IBSI standard).

# %%
def write_temp_nifti(arr3d, voxel_spacing=(2.0, 2.0, 2.0), suffix="_img.nii.gz"):
    """Write a 3D numpy array to a temp NIfTI file via SimpleITK; return path."""
    img = sitk.GetImageFromArray(arr3d.astype(np.float32))
    img.SetSpacing(voxel_spacing)
    img.SetOrigin((0.0, 0.0, 0.0))
    fp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    fp.close()
    sitk.WriteImage(img, fp.name)
    return fp.name


def extract_features_volume(volume_np):
    """Extract IBSI features from a 3D SUV volume; return dict feat→value.

    Uses SUV > SUV_THRESHOLD as the ROI mask (registration §5.3).
    """
    if volume_np.ndim != 3:
        return {}
    mask_np = (volume_np > SUV_THRESHOLD).astype(np.uint8)
    if mask_np.sum() < 32:
        return {}

    img_path = write_temp_nifti(volume_np, suffix="_img.nii.gz")
    mask_path = write_temp_nifti(mask_np, suffix="_msk.nii.gz")

    try:
        # MIRP uses IBSI-canonical feature-family codes, not pyradiomics
        # umbrella names. Mapping registration §5.3 → MIRP:
        #   shape       → morphological
        #   firstorder  → statistical
        #   GLCM        → cm
        #   GLRLM       → rlm
        #   GLSZM       → szm
        #   GLDM        → ngldm   (IBSI's neighbourhood GLDM)
        #   NGTDM       → ngtdm
        # Note inconsistent MIRP naming: discretisation METHOD is called
        # "fixed_bin_size" (not "fixed_bin_width") even though the parameter
        # for the bin width itself is `base_discretisation_bin_width`.
        # MIRP's documented convention for half-open resegmentation ranges:
        # use `np.nan` as the upper bound, not `np.inf`.
        results = extract_features(
            image=img_path,
            mask=mask_path,
            new_spacing=[2.0, 2.0, 2.0],
            base_discretisation_method="fixed_bin_size",
            base_discretisation_bin_width=0.25,
            resegmentation_intensity_range=[float(SUV_THRESHOLD), float("nan")],
            base_feature_families=[
                "morphological",
                "statistical",
                "cm",
                "rlm",
                "szm",
                "ngldm",
                "ngtdm",
            ],
        )
    except Exception as e:
        print(f"  MIRP extract_features failed: {type(e).__name__}: {e}")
        return {}
    finally:
        Path(img_path).unlink(missing_ok=True)
        Path(mask_path).unlink(missing_ok=True)

    # MIRP returns a list of pandas DataFrames (one per image/mask pair).
    if isinstance(results, list) and len(results) > 0:
        df = results[0]
    elif isinstance(results, pd.DataFrame):
        df = results
    else:
        return {}

    if df is None or df.empty:
        return {}

    feats = {}
    for col in df.columns:
        try:
            v = df[col].iloc[0]
        except Exception:
            continue
        if isinstance(v, (int, float, np.number)) and not pd.isna(v):
            feats[col] = float(v)
    return feats


# %% [markdown]
# ## 3. Test-retest pair discovery

# %%
def load_best_patch(patches_npz_path, threshold=SUV_THRESHOLD):
    """Return the most signal-rich patch from a session's patches.npz.

    Prior version (`load_first_patch`) used patch[0] which is often background
    air for whole-body PET (healthy T9 cohort: 46 patches/session, the lesion
    or brain-uptake patch is rarely #0). Replaced with a score-based picker:

      score(patch) = (n_voxels above threshold) × (max voxel value)

    This combines two signals:
    - High count of above-threshold voxels = lesion/organ presence
    - High peak SUV = focal uptake (lesion confidence)

    For cancer cohort (T6): consistently picks the lesion-containing patch.
    For healthy cohort (T9): picks the brain or heart depending on what's in the FOV.
    """
    try:
        data = np.load(patches_npz_path)
        patches = data["patches"]
        if patches.shape[0] == 0:
            return None
        scores = []
        for p in patches:
            n_above = int((p > threshold).sum())
            peak = float(p.max()) if p.size > 0 else 0.0
            scores.append(n_above * peak)
        best_idx = int(np.argmax(scores))
        if scores[best_idx] == 0:
            # No patch has any above-threshold voxels — entire session is background
            return None
        return patches[best_idx].astype(np.float32)
    except Exception as e:
        print(f"  load failed {patches_npz_path}: {e}")
        return None


# Keep the old function name as an alias for backwards compatibility within the notebook
load_first_patch = load_best_patch


def discover_test_retest_pairs(task):
    """Return list of (patient_or_subject_id, session_a_path, session_b_path)."""
    if task == "t6":
        root = next(Path("/kaggle/input").rglob("pet-fm-bench-t6-patches-v3"), None)
        labels_path = next(Path("/kaggle/input").rglob("t6_labels.csv"), None)
    elif task == "t9":
        root = next(Path("/kaggle/input").rglob("pet-fm-bench-t9-patches"), None)
        labels_path = None
    else:
        raise ValueError(f"unsupported task: {task}")

    if root is None:
        print(f"  ✗ {task}: patches dataset not attached")
        return []

    manifest_path = next(root.rglob("manifest.csv"), None)
    if manifest_path is None:
        print(f"  ✗ {task}: manifest.csv not found")
        return []

    manifest = pd.read_csv(manifest_path)
    patches_root = manifest_path.parent / "patches_3d"
    id_col = "patient_id" if "patient_id" in manifest.columns else "subject_id"

    if task == "t6" and labels_path is not None:
        labels = pd.read_csv(labels_path)
        if "is_retest_patient" in labels.columns:
            retest_ids = set(labels.loc[labels["is_retest_patient"] == True, id_col])
            manifest = manifest[manifest[id_col].isin(retest_ids)]
            print(f"  T6: filtered to {len(retest_ids)} retest patients "
                  f"({len(manifest)} sessions)")

    sess_col = "session" if "session" in manifest.columns else (
        "study_index" if "study_index" in manifest.columns else None)
    sessions_per = manifest.groupby(id_col)
    pairs = []
    for pid, grp in sessions_per:
        if sess_col:
            grp_sorted = grp.sort_values(sess_col)
        else:
            grp_sorted = grp
        if len(grp_sorted) < 2:
            continue

        def session_path(sess_row):
            sess_val = sess_row.get(sess_col, "") if sess_col else ""
            for cand in [
                patches_root / pid / str(sess_val) / "patches.npz",
                patches_root / pid / f"study_{sess_val}" / "patches.npz",
                patches_root / pid / f"session_{sess_val}" / "patches.npz",
            ]:
                if cand.exists():
                    return cand
            return None

        sess_a = grp_sorted.iloc[0]
        sess_b = grp_sorted.iloc[1]
        path_a = session_path(sess_a)
        path_b = session_path(sess_b)
        if path_a and path_b:
            pairs.append((pid, path_a, path_b))
    return pairs


# %% [markdown]
# ## 4. Lin's CCC (mirrors probe_analysis.py)

# %%
def lin_ccc(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < 2:
        return float("nan")
    mx, my = x.mean(), y.mean()
    vx, vy = x.var(), y.var()
    cov = np.mean((x - mx) * (y - my))
    denom = vx + vy + (mx - my) ** 2
    if denom <= 0:
        return float("nan")
    return float(2 * cov / denom)


def features_ccc(test_features_list, retest_features_list):
    if not test_features_list or not retest_features_list:
        return float("nan"), 0
    common = set(test_features_list[0].keys())
    for f in test_features_list + retest_features_list:
        common &= set(f.keys())
    if not common:
        return float("nan"), 0
    common = sorted(common)
    test_arr = np.array([[f[k] for k in common] for f in test_features_list])
    retest_arr = np.array([[f[k] for k in common] for f in retest_features_list])
    cccs = []
    for d in range(test_arr.shape[1]):
        c = lin_ccc(test_arr[:, d], retest_arr[:, d])
        if not np.isnan(c):
            cccs.append(c)
    return (float(np.mean(cccs)) if cccs else float("nan"), len(cccs))


# %% [markdown]
# ## 5. Run for T6 + T9

# %%
results = []

for task in ("t6", "t9"):
    print(f"\n{'=' * 60}\n{task.upper()}: MIRP IBSI feature CCC\n{'=' * 60}")
    pairs = discover_test_retest_pairs(task)
    print(f"  {len(pairs)} test-retest pairs found")
    if not pairs:
        continue

    test_feats, retest_feats = [], []
    for pid, path_a, path_b in pairs:
        vol_a = load_first_patch(path_a)
        vol_b = load_first_patch(path_b)
        if vol_a is None or vol_b is None:
            continue
        f_a = extract_features_volume(vol_a)
        f_b = extract_features_volume(vol_b)
        if not f_a or not f_b:
            continue
        test_feats.append(f_a)
        retest_feats.append(f_b)
        if len(test_feats) % 5 == 0:
            print(f"    ...{len(test_feats)} pairs extracted")

    print(f"  ✓ extracted {len(test_feats)} pair feature sets")
    if len(test_feats) < 2:
        print(f"  ✗ {task}: too few feature pairs to compute CCC")
        continue

    ccc_value, n_features = features_ccc(test_feats, retest_feats)
    rng = np.random.RandomState(42)
    boot_cccs = []
    n_pairs = len(test_feats)
    for _ in range(N_BOOTSTRAP):
        idx = rng.choice(n_pairs, size=n_pairs, replace=True)
        c, _ = features_ccc([test_feats[i] for i in idx],
                            [retest_feats[i] for i in idx])
        if not np.isnan(c):
            boot_cccs.append(c)
    if boot_cccs:
        ci_low = float(np.percentile(boot_cccs, 2.5))
        ci_high = float(np.percentile(boot_cccs, 97.5))
    else:
        ci_low = ci_high = float("nan")

    results.append({
        "task": task,
        "fm": "ibsi_radiomics_baseline",
        "metric": "lin_ccc",
        "value": ccc_value,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "n_pairs": n_pairs,
        "n_features": n_features,
    })
    print(f"  → {task}: IBSI-radiomics CCC = {ccc_value:.4f} "
          f"[{ci_low:.4f}, {ci_high:.4f}] (n_pairs={n_pairs}, "
          f"n_features={n_features})")

# %% [markdown]
# ## 6. Save outputs

# %%
results_df = pd.DataFrame(results)
out_path = OUT_DIR / "pyradiomics_test_retest_ccc.csv"
metadata_path = OUT_DIR / "pyradiomics_baseline_metadata.json"

results_df.to_csv(out_path, index=False)

with open(metadata_path, "w") as f:
    json.dump({
        "stage": "Registration H6 — IBSI radiomics CCC baseline",
        "freeze_timestamp_utc": freeze_timestamp,
        "registration_section": "§5.3 (IBSI radiomics) + H6 (test-retest stability)",
        "library_used": "MIRP (Oncoray Medical Image Radiomics Processor)",
        "library_version": MIRP_VERSION,
        "library_deviation_note": (
            "Registration §5.3 specified pyradiomics v3.1; switched to MIRP "
            "per amendment A8 (pyradiomics 3.0.1+ has no Python 3.12 support; "
            "MIRP is IBSI-validated by the IBSI consortium and computes the "
            "same feature families)."
        ),
        "config": {
            "base_discretisation_method": "fixed_bin_size",
            "base_discretisation_bin_width": 0.25,
            "new_spacing": [2.0, 2.0, 2.0],
            "resegmentation_intensity_range_lower": SUV_THRESHOLD,
            "resegmentation_intensity_range_upper": "nan (MIRP half-open convention)",
            "base_feature_families": [
                "morphological", "statistical",
                "cm", "rlm", "szm", "ngldm", "ngtdm",
            ],
            "feature_family_mapping_to_registration_5p3": {
                "shape": "morphological",
                "firstorder": "statistical",
                "GLCM": "cm",
                "GLRLM": "rlm",
                "GLSZM": "szm",
                "GLDM": "ngldm",
                "NGTDM": "ngtdm",
            },
            "ibsi_compliant": True,
        },
        "input_volumes": (
            "Best 3D patch (96³) per session from patches_3d, selected by "
            "score = (n_voxels above SUV threshold 0.5) × (max SUV). For "
            "cancer (T6) this consistently selects the lesion-containing "
            "patch; for healthy controls (T9) the brain/heart/liver region "
            "depending on FOV. Replaces an earlier first-patch heuristic "
            "that produced 0/48 successful T9 extractions."
        ),
        "results": results,
    }, f, indent=2)

print(f"\nWrote: {out_path}")
print(results_df.to_string(index=False))
print(f"\nWrote: {metadata_path}")

# %% [markdown]
# ## 7. Done
#
# Save & Run All → Output → New Dataset → `pet-fm-bench-pyradiomics-baseline`.
# (Dataset name retains the `pyradiomics-` prefix for downstream-tooling
# stability; the actual library is MIRP per amendment A8 — recorded in
# the metadata sidecar.)
