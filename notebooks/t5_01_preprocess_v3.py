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
# # T5 Notebook 1/2 v3: AutoPET-III PSMA — Preprocess for Cross-Tracer Detection
#
# **PET-FM-Bench** | DOI: [10.17605/OSF.IO/DQ2JA](https://doi.org/10.17605/OSF.IO/DQ2JA)
#
# **PLATFORM:** Google **Colab** (NOT Kaggle) — needs Drive mount for AutoPET-III
# DICOM tree (93 GB) and (companion project) segmentation outputs.
# **Runtime:** CPU | **Time:** ~2.5–3.5 hr (497 disease-positive PT series → ~15k patches) | **Disk:** ~13 GB output
#
# **Status:** Wired against A10 + A11 (amendment_log.md v10). Data is use-grade per
# (companion project)'s Phase 1 closure 2026-04-28; SEG NIfTI / softmax / reviewed lesion parquet
# all available on Drive + (companion project) OSF j5ry4. T5 patch extraction goes per-(case_id,
# series_uid) — multi-series patients (135/333) contribute multiple manifest rows;
# patient-level CV grouping in `probe_analysis.py` v5 prevents leakage during
# zero-shot evaluation.
#
# ## What this notebook does (per A9b + A10)
#
# - Reads the reviewed lesion parquet (external companion-project artefact) `autopet_iii_lesions_reviewed.parquet`
#   (filtered `section_3_9_excluded == False` → 8,768 lesions / 333 patients).
# - For each (case_id, series_uid) tuple in the parquet:
#   - Locate SEG NIfTI at `/My Drive/petfm_data/autopet_iii/segmentations/{series_uid}.nii.gz`.
#   - Locate or regenerate SUV NIfTI (paired_inputs/ if exists; else regenerate via
#     (companion project) `dicom_series_to_suv_sitk` from the case's DICOM series).
#   - Extract per-lesion 96³ patches centred on the parquet-supplied centroid
#     (voxel coords on PET grid; matches SEG/SUV grid 1:1 per A10).
#   - Sample matched-count background patches via rejection sampling (same
#     pattern as t1_01_preprocess_v3.py — cheap, OOM-resistant).
# - Applies the canonical SUV-conversion module (Amendment 3, 9/9 PASS) for any
#   case where SUV NIfTI doesn't already exist on Drive.
# - Lesion-aware patch labelling with (companion project)'s nnU-Net SEG masks. Per A9b.
# - **Test-only split** for T5 per registration §3.1 (zero-shot).

# %% [markdown]
# ## 1. Setup (Colab)

# %%
# !pip install -q SimpleITK pydicom scipy scikit-image pandas pyarrow tqdm

import datetime
import gc
import json
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
import SimpleITK as sitk
from scipy import ndimage
from tqdm.auto import tqdm

# Colab-Drive mount
try:
    from google.colab import drive
    drive.mount("/content/drive", force_remount=False)
    DRIVE_ROOT = Path("/content/drive/My Drive/petfm_data/autopet_iii")
except ImportError:
    DRIVE_ROOT = Path("/path/to/manually/mounted/autopet_iii")
    print("⚠ Not on Colab — set DRIVE_ROOT manually for local dry-run.")

DICOM_ROOT     = DRIVE_ROOT                          # zips + extracted dirs
SEG_ROOT       = DRIVE_ROOT / "segmentations"        # (companion project) nnU-Net output {series_uid}.nii.gz
PAIRED_INPUTS  = DRIVE_ROOT / "paired_inputs"        # (companion project) partial SUV NIfTI cache (268/597 cases)

# the reviewed lesion parquet (external companion-project artefact) — try multiple candidate locations
PARQUET_CANDIDATES = [
    DRIVE_ROOT / "lesion_tables" / "autopet_iii_lesions_reviewed.parquet",
    DRIVE_ROOT / "autopet_iii_lesions_reviewed.parquet",
    DRIVE_ROOT.parent / "p79_data_interim" / "autopet_iii_lesions_reviewed.parquet",
]
LESIONS_PARQUET = next((p for p in PARQUET_CANDIDATES if p.exists()), None)

if LESIONS_PARQUET is None:
    # Fallback: also try a glob search in case the parquet lives at an
    # unexpected depth in the Drive tree (one-time exploration).
    glob_hits = list(DRIVE_ROOT.parent.rglob("autopet_iii_lesions_reviewed.parquet"))
    if glob_hits:
        LESIONS_PARQUET = glob_hits[0]
        print(f"  Found via rglob: {LESIONS_PARQUET}")

if LESIONS_PARQUET is None:
    # Auto-upload fallback: prompt user via Colab file picker, then mirror
    # the file to the canonical Drive path so future runs find it directly.
    try:
        from google.colab import files as _colab_files
        print("⚠ autopet_iii_lesions_reviewed.parquet not found on Drive.")
        print("  Open the file picker and select it from your laptop at:")
        print("  /path/to/companion-project/ Conformal SUV Theranostic/data/interim/lesion_tables/autopet_iii_lesions_reviewed.parquet")
        uploaded = _colab_files.upload()
        if not uploaded:
            raise FileNotFoundError("No file uploaded.")
        src_name = list(uploaded.keys())[0]
        target_dir = DRIVE_ROOT / "lesion_tables"
        target_dir.mkdir(parents=True, exist_ok=True)
        LESIONS_PARQUET = target_dir / "autopet_iii_lesions_reviewed.parquet"
        shutil.move(f"/content/{src_name}", LESIONS_PARQUET)
        print(f"  Mirrored to Drive: {LESIONS_PARQUET}")
        # Verify against A10's pinned hash
        import hashlib
        h = hashlib.sha256()
        with open(LESIONS_PARQUET, "rb") as f:
            for blk in iter(lambda: f.read(2**20), b""):
                h.update(blk)
        h_observed = h.hexdigest()
        EXPECTED_SHA = "4ef08570271cd6155036925e115ab271b5b0e556ea1fbfd06ef4be8911ad6251"
        print(f"  SHA-256: {h_observed}")
        if h_observed == EXPECTED_SHA:
            print(f"  ✓ Matches A10-pinned hash")
        else:
            print(f"  ⚠ MISMATCH with A10-pinned hash {EXPECTED_SHA}")
            print(f"     Continuing anyway, but re-check (companion project) source.")
    except ImportError:
        raise FileNotFoundError(
            "Could not locate autopet_iii_lesions_reviewed.parquet (not on Colab; "
            "auto-upload unavailable). Either copy from (companion project) local at "
            "`79 Conformal SUV Theranostic/data/interim/lesion_tables/` "
            "into one of the candidate Drive paths, OR pull from (companion project) OSF j5ry4 "
            "(SHA `4ef08570…`)."
        )

# Local Colab disk for transient + final outputs
TMP_DIR   = Path("/content/t5_v3_tmp")
PATCH_DIR = Path("/content/t5_v3_patches")
TMP_DIR.mkdir(parents=True, exist_ok=True)
PATCH_DIR.mkdir(parents=True, exist_ok=True)

freeze_timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
print(f"Freeze timestamp (UTC): {freeze_timestamp}")
print(f"AutoPET-III DICOM root: {DICOM_ROOT}")
print(f"(companion project) SEG root: {SEG_ROOT}")
print(f"(companion project) paired_inputs (partial SUV cache): {PAIRED_INPUTS}")
print(f"Reviewed parquet: {LESIONS_PARQUET}")

# %% [markdown]
# ## 2. (companion project) SUV pipeline import (or inline copy)
#
# the canonical SUV-conversion module (post-Amendment-3, 9/9 PASS at
# max_rel_diff=0.00000% across Siemens/GE + empirical Ga-68 PSMA confirmation).
# Same module already inlined in `t6_01_preprocess_v3.py` and `t8_01_preprocess_v3.py`.
#
# Two import paths:
# 1. If (companion project) has shared a Python package or copied module to Drive — use that.
# 2. Otherwise: copy from (companion project)'s local repo into the running session.
#
# For the live run, ensure `/content/suv_conversion.py` exists with the post-bug-fix
# version (see (companion project) src/preprocess/suv_conversion.py). The validation note: must
# contain "per-slice basis with each slice's own" in the dicom_series_to_suv_sitk
# docstring.

# %%
# Bootstraps suv_conversion.py from (in order of preference):
#   1. Drive cache at /My Drive/petfm_data/code/suv_conversion.py (persistent across sessions).
#   2. /content/suv_conversion.py (already present in this session).
#   3. Colab file picker fallback (uploads from laptop, then mirrors to Drive cache).
#
# Always verifies the post-bug-fix marker ((companion project) Amendment 3 docstring snippet) before
# proceeding — using the pre-bug-fix version produces incorrect SUVs on Siemens
# variable-rescale scanners.

import sys
SUV_DRIVE_CACHE = DRIVE_ROOT.parent / "code" / "suv_conversion.py"
SUV_CONTENT = Path("/content/suv_conversion.py")

if SUV_DRIVE_CACHE.exists() and not SUV_CONTENT.exists():
    SUV_CONTENT.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(str(SUV_DRIVE_CACHE), str(SUV_CONTENT))
    print(f"  Copied suv_conversion.py from Drive cache → /content/")

if "/content" not in sys.path:
    sys.path.insert(0, "/content")
# Bust any stale failed-import cache from a prior cell
for _mod in list(sys.modules):
    if _mod.startswith("suv_conversion"):
        del sys.modules[_mod]

try:
    from suv_conversion import extract_pet_metadata, dicom_series_to_suv_sitk  # noqa: F401
    print("✓ (companion project) suv_conversion module imported")
except ImportError:
    # Auto-upload fallback
    try:
        from google.colab import files as _colab_files
        print("⚠ suv_conversion.py not found locally or on Drive cache.")
        print("  Open the file picker and select it from your laptop at:")
        print("  /path/to/companion-project/ Conformal SUV Theranostic/src/preprocess/suv_conversion.py")
        uploaded = _colab_files.upload()
        if uploaded:
            src_name = list(uploaded.keys())[0]
            shutil.move(f"/content/{src_name}", str(SUV_CONTENT))
            # Mirror to Drive cache for future sessions
            SUV_DRIVE_CACHE.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(str(SUV_CONTENT), str(SUV_DRIVE_CACHE))
            print(f"  Mirrored to Drive cache: {SUV_DRIVE_CACHE}")
            for _mod in list(sys.modules):
                if _mod.startswith("suv_conversion"):
                    del sys.modules[_mod]
            from suv_conversion import extract_pet_metadata, dicom_series_to_suv_sitk
            print("✓ (companion project) suv_conversion module imported")
    except ImportError:
        print("⚠ Not on Colab and suv_conversion.py not found — §6 main loop will fail.")
        print("  Manual fix: copy (companion project)'s src/preprocess/suv_conversion.py to /content/.")

# Verify post-bug-fix version ((companion project) Amendment 3 marker) — mandatory for SUV correctness
try:
    import inspect
    src_text = inspect.getsource(dicom_series_to_suv_sitk)
    if "per-slice basis with each slice's own" in src_text:
        print("✓ Post-bug-fix version verified (per-slice rescale handling)")
    else:
        print("⚠ WARNING: pre-bug-fix version detected (no per-slice rescale marker).")
        print("   This will produce incorrect SUVs on Siemens variable-rescale scanners.")
        print("   Replace /content/suv_conversion.py with (companion project)'s post-Amendment-3 version.")
except (NameError, ImportError):
    pass  # Module not yet imported; warning already printed above

# %% [markdown]
# ## 3. Configuration

# %%
PATCH_SIZE_3D = (96, 96, 96)
SPACING       = (2.0, 2.0, 2.0)
MIP_SIZE_2D   = 224

# Lesion-aware patch extraction parameters (registration §3.1 / A9b / A10)
IOU_THRESHOLD       = 0.5
MIN_LESION_VOXELS   = 27
BG_SUV_THRESHOLD    = 2.5
BG_PATCHES_PER_LESION = 1
MAX_PATCHES_PER_PATIENT = 50

# A10 specifies: filter section_3_9_excluded == False → 8,768 reviewed lesions
# A11 multi-series: include ALL series per patient (135/333 patients have ≥2)
INCLUDE_ALL_SERIES = True

# Sanity guards
MIN_DOSE_MBQ = 10.0  # malformed dose tags read as 0.0004 MBq sometimes

print(f"Patch config: {PATCH_SIZE_3D} @ {SPACING} spacing")
print(f"Multi-series mode: {INCLUDE_ALL_SERIES} (A11 supports multi-PT-series)")

# %% [markdown]
# ## 4. Load the reviewed lesion parquet (external companion-project artefact)
#
# Schema (per A10): `case_id, lesion_id, series_uid, suvmax, suvmean, suvpeak,
# tlg, volume_ml, n_voxels, centroid_0/1/2, voxel_spacing_0/1/2, dataset='autopet_iii',
# tracer='PSMA', radionuclide ∈ {18F, 68Ga, ^68^Gallium, Ga-68}, vendor,
# softmax_mean, softmax_entropy, section_3_9_excluded`.

# %%
lesions = pd.read_parquet(LESIONS_PARQUET)
print(f"Raw parquet rows: {len(lesions)}")

# A10 filter: drop the 3 §3.9-excluded lesions
lesions = lesions[lesions["section_3_9_excluded"] == False].copy()
print(f"After section_3_9_excluded filter: {len(lesions)} lesions")

# Confirm cohort-level numbers match A10 spec
n_cases = lesions["case_id"].nunique()
n_series = lesions["series_uid"].nunique()
print(f"Unique cases: {n_cases}")
print(f"Unique series_uid: {n_series}")
print(f"All tracer values: {lesions['tracer'].unique().tolist()}")

# Group lesions by (case_id, series_uid) for per-series processing
case_series = (
    lesions[["case_id", "series_uid", "radionuclide", "vendor"]]
    .drop_duplicates(subset=["case_id", "series_uid"])
    .reset_index(drop=True)
)
print(f"\n(case_id, series_uid) tuples to process: {len(case_series)}")
print(f"Series-per-case distribution:")
print(case_series.groupby("case_id").size().value_counts().sort_index().to_string())

# %% [markdown]
# ## 5. Helpers
#
# Patch geometry + 2D MIP rendering identical to `t1_01_preprocess_v3.py` §4.
# Differences for T5:
# - `find_dicom_dir_for_series()` to handle the heterogeneous Drive layout
#   (1,107 zips + 684 extracted dirs).
# - `load_or_generate_suv()` checks paired_inputs/ first, falls back to
#   `dicom_series_to_suv_sitk` from raw DICOM.

# %%
def find_dicom_dir_for_series(series_uid):
    """Return (DICOM dir Path) or None.
    Heterogeneous: prefer extracted dir, fall back to extracting the zip lazily."""
    extracted = DICOM_ROOT / series_uid
    if extracted.is_dir() and any(extracted.glob("*.dcm")):
        return extracted
    zip_path = DICOM_ROOT / f"{series_uid}.zip"
    if zip_path.is_file():
        # Extract once to TMP_DIR if not already done
        target = TMP_DIR / series_uid
        if not target.exists():
            target.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(target)
        return target
    return None


def load_or_generate_suv(series_uid):
    """Load SUV NIfTI from paired_inputs/ if available; otherwise regenerate
    from raw DICOM using the validated SUV-conversion module (cross-vendor validated) module."""
    cache = PAIRED_INPUTS / f"{series_uid}_0001.nii.gz"
    if cache.is_file():
        return sitk.ReadImage(str(cache)), "cache_hit"
    # Fallback: regenerate from DICOM
    dcm_dir = find_dicom_dir_for_series(series_uid)
    if dcm_dir is None:
        raise FileNotFoundError(f"No DICOM source for series {series_uid}")
    dcm_files = list(dcm_dir.glob("*.dcm"))
    if not dcm_files:
        raise FileNotFoundError(f"No .dcm files in {dcm_dir}")
    meta = extract_pet_metadata(str(dcm_files[0]))
    dose_mbq = meta.injected_dose_bq / 1e6
    if dose_mbq < MIN_DOSE_MBQ:
        raise ValueError(f"Implausible dose {dose_mbq:.4f} MBq for {series_uid}")
    suv_img = dicom_series_to_suv_sitk(str(dcm_dir), meta)
    return suv_img, "regenerated"


def load_seg(series_uid):
    """Load (companion project)'s nnU-Net binary segmentation NIfTI."""
    p = SEG_ROOT / f"{series_uid}.nii.gz"
    if not p.is_file():
        raise FileNotFoundError(f"No SEG NIfTI at {p}")
    return sitk.ReadImage(str(p))


def resample_isotropic(img_sitk, spacing=SPACING, interpolator=sitk.sitkLinear):
    orig_spacing = img_sitk.GetSpacing()
    orig_size = img_sitk.GetSize()
    new_size = [int(round(s * sp / t)) for s, sp, t in zip(orig_size, orig_spacing, spacing)]
    rs = sitk.ResampleImageFilter()
    rs.SetOutputSpacing(spacing)
    rs.SetSize(new_size)
    rs.SetOutputDirection(img_sitk.GetDirection())
    rs.SetOutputOrigin(img_sitk.GetOrigin())
    rs.SetTransform(sitk.Transform())
    rs.SetDefaultPixelValue(0)
    rs.SetInterpolator(interpolator)
    return rs.Execute(img_sitk)


def extract_centred_patch(volume, centre_zyx, patch_size=PATCH_SIZE_3D):
    pz, py, px = patch_size
    cz, cy, cx = centre_zyx
    z0 = int(cz - pz // 2); y0 = int(cy - py // 2); x0 = int(cx - px // 2)
    z1, y1, x1 = z0 + pz, y0 + py, x0 + px
    sz0, sy0, sx0 = max(0, z0), max(0, y0), max(0, x0)
    sz1 = min(volume.shape[0], z1)
    sy1 = min(volume.shape[1], y1)
    sx1 = min(volume.shape[2], x1)
    pz0, py0, px0 = sz0 - z0, sy0 - y0, sx0 - x0
    pz1, py1, px1 = pz0 + (sz1 - sz0), py0 + (sy1 - sy0), px0 + (sx1 - sx0)
    out = np.zeros(patch_size, dtype=volume.dtype)
    out[pz0:pz1, py0:py1, px0:px1] = volume[sz0:sz1, sy0:sy1, sx0:sx1]
    return out, (z0, y0, x0)


def compute_iou(seg_patch, full_seg, patch_origin):
    pz, py, px = seg_patch.shape
    z0, y0, x0 = patch_origin
    inter = int(seg_patch.sum())
    if inter == 0:
        return 0.0
    z1, y1, x1 = z0 + pz, y0 + py, x0 + px
    z0c, y0c, x0c = max(0, z0), max(0, y0), max(0, x0)
    z1c = min(full_seg.shape[0], z1)
    y1c = min(full_seg.shape[1], y1)
    x1c = min(full_seg.shape[2], x1)
    full_in_patch = full_seg[z0c:z1c, y0c:y1c, x0c:x1c]
    union = int(full_in_patch.sum() + seg_patch.sum() - inter)
    if union == 0:
        return 0.0
    return inter / union


def sample_background_patches(suv, seg, n_wanted, patch_size=PATCH_SIZE_3D,
                               suv_max=BG_SUV_THRESHOLD, rng=None,
                               max_tries_factor=200):
    """Memory-efficient rejection sampling — same pattern as t1_01_preprocess_v3.py
    post-OOM patch. Avoids np.argwhere O(N) allocation."""
    if rng is None:
        rng = np.random.default_rng(42)
    pz, py, px = patch_size
    hz, hy, hx = pz // 2, py // 2, px // 2
    sz, sy, sx = suv.shape
    if sz <= pz or sy <= py or sx <= px:
        return []
    centres = []
    max_tries = max(n_wanted * max_tries_factor, 100)
    for _ in range(max_tries):
        if len(centres) >= n_wanted:
            break
        z = int(rng.integers(hz, sz - hz))
        y = int(rng.integers(hy, sy - hy))
        x = int(rng.integers(hx, sx - hx))
        if seg[z, y, x] == 0 and suv[z, y, x] < suv_max:
            centres.append((z, y, x))
    return centres


def patch_to_mip_views(patch_3d, target_size=MIP_SIZE_2D):
    def _mip_resize(arr):
        img = sitk.GetImageFromArray(arr.astype(np.float32))
        rs = sitk.ResampleImageFilter()
        rs.SetSize((target_size, target_size))
        rs.SetOutputSpacing((arr.shape[1] / target_size, arr.shape[0] / target_size))
        rs.SetInterpolator(sitk.sitkLinear)
        return sitk.GetArrayFromImage(rs.Execute(img))
    return {
        "coronal":  _mip_resize(patch_3d.max(axis=1)),
        "axial":    _mip_resize(patch_3d.max(axis=0)),
        "sagittal": _mip_resize(patch_3d.max(axis=2)),
    }


# %% [markdown]
# ## 6. Per-(case_id, series_uid) processing loop
#
# **Resume support + checkpointing** — same pattern as t1_01_preprocess_v3.py
# post-OOM patch. Saves manifest_partial every CHECKPOINT_EVERY series.

# %%
CHECKPOINT_EVERY = 25  # T5 has 497 series; checkpoint every 25 = 20 checkpoints

PARTIAL_MANIFEST = PATCH_DIR / "manifest_partial.parquet"
PARTIAL_LOG      = PATCH_DIR / "preprocessing_log_partial.csv"

# Drive-mirror checkpoint location — survives Colab runtime restart (which
# wipes /content). Lesson learned 2026-04-29: T5 first run lost 200 series of
# work when "Disconnect and delete runtime" wiped /content. /content checkpoint
# is the FAST path; Drive checkpoint is the PERSISTENT path. We rsync /content
# → Drive at every CHECKPOINT_EVERY, and on resume, restore from Drive if
# /content is empty.
DRIVE_BACKUP_DIR = DRIVE_ROOT.parent / "t5_v3_patches_drive_mirror"


def mirror_patches_to_drive(verbose=False):
    """rsync-style incremental mirror PATCH_DIR → DRIVE_BACKUP_DIR.
    Only copies new/changed files; cheap at each checkpoint."""
    import subprocess as _sp
    DRIVE_BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    cmd = ["rsync", "-a", "--update", f"{PATCH_DIR}/", f"{DRIVE_BACKUP_DIR}/"]
    result = _sp.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return False, result.stderr[:300]
    return True, ""


def restore_patches_from_drive():
    """Restore PATCH_DIR ← DRIVE_BACKUP_DIR if /content is empty but Drive has work."""
    import subprocess as _sp
    drive_manifest = DRIVE_BACKUP_DIR / "manifest_partial.parquet"
    if not drive_manifest.exists():
        return False
    PATCH_DIR.mkdir(parents=True, exist_ok=True)
    cmd = ["rsync", "-a", f"{DRIVE_BACKUP_DIR}/", f"{PATCH_DIR}/"]
    result = _sp.run(cmd, capture_output=True, text=True)
    return result.returncode == 0


# If /content is empty (post-runtime-restart) but Drive has the backup, restore.
if not PARTIAL_MANIFEST.exists() and (DRIVE_BACKUP_DIR / "manifest_partial.parquet").exists():
    print("⚠ /content is empty (likely post-runtime-restart). Restoring from Drive mirror...")
    if restore_patches_from_drive():
        print(f"✓ Restored PATCH_DIR from {DRIVE_BACKUP_DIR}")
    else:
        print(f"⚠ Restore failed — proceeding with fresh start")

manifest_rows = []
log_rows = []
already_processed = set()  # set of (case_id, series_uid) tuples

if PARTIAL_MANIFEST.exists() and PARTIAL_LOG.exists():
    prev_manifest = pd.read_parquet(PARTIAL_MANIFEST)
    prev_log = pd.read_csv(PARTIAL_LOG)
    ok_log = prev_log[prev_log["status"] == "ok"].copy()
    err_log = prev_log[prev_log["status"] != "ok"].copy()
    manifest_rows = prev_manifest.to_dict("records")
    log_rows = ok_log.to_dict("records")
    already_processed = set(zip(ok_log["case_id"].astype(str),
                                 ok_log["series_uid"].astype(str)))
    print(f"RESUMING from partial output:")
    print(f"  {len(already_processed)} (case, series) tuples done")
    print(f"  {len(err_log)} errored previously (RETRY on resume)")
    print(f"  {len(manifest_rows)} patches in manifest (preserved)")
else:
    print("No partial output found — starting fresh")


def process_series(case_id, series_uid, radionuclide, vendor, lesions_df, rng):
    """Process one (case_id, series_uid). Returns (rows_for_manifest, log_record)."""
    # Load SUV (paired_inputs/ cache OR regenerate from DICOM)
    suv_img, suv_source = load_or_generate_suv(series_uid)
    seg_img = load_seg(series_uid)

    # Resample both to common 2mm isotropic geometry
    suv_iso = resample_isotropic(suv_img, SPACING, sitk.sitkLinear)
    seg_iso = resample_isotropic(seg_img, SPACING, sitk.sitkNearestNeighbor)
    suv = sitk.GetArrayFromImage(suv_iso).astype(np.float32)
    seg = (sitk.GetArrayFromImage(seg_iso) > 0).astype(np.uint8)
    suv_max_obs = float(suv.max())
    suv_mean_obs = float(suv.mean())
    shape_iso = str(suv.shape)
    del suv_img, seg_img, suv_iso, seg_iso

    # Series-specific lesions from parquet (already filtered to reviewed cohort)
    series_lesions = lesions_df[
        (lesions_df["case_id"] == case_id) & (lesions_df["series_uid"] == series_uid)
    ].copy()

    # Filter by minimum size (matches MIN_LESION_VOXELS used in T1)
    series_lesions = series_lesions[series_lesions["n_voxels"] >= MIN_LESION_VOXELS]

    n_lesions_kept = len(series_lesions)
    if n_lesions_kept > MAX_PATCHES_PER_PATIENT // 2:
        series_lesions = series_lesions.sample(
            n=MAX_PATCHES_PER_PATIENT // 2,
            random_state=int(abs(hash(series_uid)) % (2**31)),
        ).reset_index(drop=True)

    # Short series_uid suffix for patch_id (last 8 hex chars after final dot)
    sid_short = series_uid.split(".")[-1][:8]
    patient_dir = PATCH_DIR / "patches" / case_id
    patient_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    # Lesion patches (label = 1) — use parquet-supplied centroid (voxel grid per A10)
    # Note: parquet centroids are pre-resampling PET grid. After resampling to
    # SPACING, the indices change. We rescale the centroid by the spacing ratio.
    # (For SEG/SUV grid 1:1 per A10, scaling factor = original_spacing / SPACING)
    for _, lesion in series_lesions.iterrows():
        # Original PET-grid centroid → resampled grid centroid
        cz_orig = float(lesion["centroid_0"])
        cy_orig = float(lesion["centroid_1"])
        cx_orig = float(lesion["centroid_2"])
        sz_orig = float(lesion["voxel_spacing_0"])
        sy_orig = float(lesion["voxel_spacing_1"])
        sx_orig = float(lesion["voxel_spacing_2"])
        cz = cz_orig * sz_orig / SPACING[0]
        cy = cy_orig * sy_orig / SPACING[1]
        cx = cx_orig * sx_orig / SPACING[2]

        patch_3d, origin = extract_centred_patch(suv, (cz, cy, cx))
        seg_patch, _    = extract_centred_patch(seg, (cz, cy, cx))
        iou = compute_iou(seg_patch, seg, origin)
        mips = patch_to_mip_views(patch_3d)

        patch_id = f"{case_id}_{sid_short}_lesion_{int(lesion['lesion_id']):03d}"
        np.savez_compressed(
            patient_dir / f"{patch_id}.npz",
            patch_3d=patch_3d.astype(np.float16),
            mip_coronal=mips["coronal"].astype(np.float16),
            mip_axial=mips["axial"].astype(np.float16),
            mip_sagittal=mips["sagittal"].astype(np.float16),
        )
        rows.append({
            "patient_id":     case_id,                    # case_id == DICOM PatientID per A10
            "patch_id":       patch_id,
            "series_uid":     series_uid,
            "label":          1,
            "lesion_index":   int(lesion["lesion_id"]),
            "lesion_voxels":  int(lesion["n_voxels"]),
            "iou":            float(iou),
            "patch_origin_zyx": json.dumps(list(origin)),
            "patch_centre_zyx": json.dumps([float(cz), float(cy), float(cx)]),
            "study_date":     "",                          # filled later if needed
            "radionuclide":   str(radionuclide),
            "vendor":         str(vendor),
            "softmax_mean":   float(lesion.get("softmax_mean", float("nan"))),
            "softmax_entropy": float(lesion.get("softmax_entropy", float("nan"))),
        })

    # Background patches (label = 0): match lesion count
    n_bg = max(BG_PATCHES_PER_LESION * len(series_lesions), 1) if len(series_lesions) > 0 else 5
    bg_centres = sample_background_patches(suv, seg, n_bg, rng=rng)
    for bi, centre in enumerate(bg_centres):
        patch_3d, origin = extract_centred_patch(suv, centre)
        mips = patch_to_mip_views(patch_3d)
        patch_id = f"{case_id}_{sid_short}_bg_{bi:03d}"
        np.savez_compressed(
            patient_dir / f"{patch_id}.npz",
            patch_3d=patch_3d.astype(np.float16),
            mip_coronal=mips["coronal"].astype(np.float16),
            mip_axial=mips["axial"].astype(np.float16),
            mip_sagittal=mips["sagittal"].astype(np.float16),
        )
        rows.append({
            "patient_id":     case_id,
            "patch_id":       patch_id,
            "series_uid":     series_uid,
            "label":          0,
            "lesion_index":   -1,
            "lesion_voxels":  0,
            "iou":            0.0,
            "patch_origin_zyx": json.dumps(list(origin)),
            "patch_centre_zyx": json.dumps([float(c) for c in centre]),
            "study_date":     "",
            "radionuclide":   str(radionuclide),
            "vendor":         str(vendor),
            "softmax_mean":   float("nan"),
            "softmax_entropy": float("nan"),
        })

    log_record = {
        "case_id":          case_id,
        "series_uid":       series_uid,
        "patient_id":       case_id,
        "status":           "ok",
        "n_lesions_kept":   len(series_lesions),
        "n_lesions_raw":    n_lesions_kept,
        "n_bg_sampled":     len(bg_centres),
        "suv_max":          suv_max_obs,
        "suv_mean":         suv_mean_obs,
        "shape_iso":        shape_iso,
        "suv_source":       suv_source,
        "radionuclide":     str(radionuclide),
        "vendor":           str(vendor),
    }
    return rows, log_record


def save_partial(manifest_rows, log_rows):
    pd.DataFrame(manifest_rows).to_parquet(PARTIAL_MANIFEST, index=False)
    pd.DataFrame(log_rows).to_csv(PARTIAL_LOG, index=False)


# %% [markdown]
# ## 7. Main loop with tqdm + checkpoints

# %%
remaining = [
    (r["case_id"], r["series_uid"], r["radionuclide"], r["vendor"])
    for _, r in case_series.iterrows()
    if (r["case_id"], r["series_uid"]) not in already_processed
]
print(f"\nSeries to process this session: {len(remaining)}")
print(f"Already done (resumed): {len(already_processed)}")
print(f"Checkpoint every: {CHECKPOINT_EVERY} series\n")

pbar = tqdm(total=len(remaining), desc="T5 (case, series)", unit="series")

try:
    for i, (case_id, series_uid, radionuclide, vendor) in enumerate(remaining):
        rng = np.random.default_rng(int(abs(hash(series_uid)) % (2**31)))
        try:
            rows, log_record = process_series(
                case_id, series_uid, radionuclide, vendor, lesions, rng
            )
            manifest_rows.extend(rows)
            log_rows.append(log_record)
            pbar.set_postfix_str(
                f"last={case_id[-8:]} sid={series_uid.split('.')[-1][:8]} "
                f"les={log_record['n_lesions_kept']} pat+={len(rows)} tot={len(manifest_rows)}",
                refresh=False,
            )
        except Exception as e:
            log_rows.append({
                "case_id": case_id, "series_uid": series_uid, "patient_id": case_id,
                "status": f"error: {type(e).__name__}: {e}",
            })
            pbar.write(f"ERROR ({case_id}, …{series_uid[-12:]}): {type(e).__name__}: {e}")

        gc.collect()
        pbar.update(1)

        if (i + 1) % CHECKPOINT_EVERY == 0:
            save_partial(manifest_rows, log_rows)
            pbar.write(f"  ✓ checkpoint @ i={i+1}: "
                       f"{len(manifest_rows)} patches, {len(log_rows)} log rows")
            ok, err = mirror_patches_to_drive()
            if ok:
                pbar.write(f"  ✓ Drive mirror synced (rsync incremental)")
            else:
                pbar.write(f"  ⚠ Drive mirror FAILED: {err}")
finally:
    pbar.close()
    save_partial(manifest_rows, log_rows)
    # Final mirror — guarantees Drive has the latest state on graceful exit
    ok, err = mirror_patches_to_drive()
    if ok:
        print(f"✓ Final Drive mirror synced to {DRIVE_BACKUP_DIR}")
    else:
        print(f"⚠ Final Drive mirror FAILED: {err}")
    print(f"\nFinal: {len(manifest_rows)} patches across {len(log_rows)} (case, series) records")

# %% [markdown]
# ## 8. Finalisation — promote partial → manifest.parquet + QC

# %%
manifest = pd.DataFrame(manifest_rows)
log_df = pd.DataFrame(log_rows)

manifest.to_parquet(PATCH_DIR / "manifest.parquet", index=False)
log_df.to_csv(PATCH_DIR / "preprocessing_log.csv", index=False)

ok = log_df[log_df["status"] == "ok"]
print("=== T5 v3 AutoPET-III PSMA Preprocessing QC ===")
print(f"(case, series) records ok: {len(ok)}/{len(log_df)}")
print(f"Errors: {(log_df['status'].str.startswith('error')).sum()}")
print(f"Total patches: {len(manifest)}")
print(f"  lesion patches (label=1): {(manifest['label'] == 1).sum()}")
print(f"  background patches (label=0): {(manifest['label'] == 0).sum()}")

if len(ok):
    print(f"\nLesion counts per (case, series):")
    print(f"  median: {ok['n_lesions_kept'].median():.0f}, max: {ok['n_lesions_kept'].max()}")
    print(f"\nUnique patients in manifest: {manifest['patient_id'].nunique()}")
    print(f"Unique series in manifest: {manifest['series_uid'].nunique()}")
    print(f"\nRadionuclide distribution (patient-level):")
    print(manifest.drop_duplicates("patient_id")["radionuclide"]
          .value_counts(dropna=False).to_string())

size_gb = sum(f.stat().st_size for f in PATCH_DIR.rglob("*") if f.is_file()) / 1e9
print(f"\nOutput size: {size_gb:.2f} GB")

# %% [markdown]
# ## 9. Tarball + move to Drive

# %%
import tarfile
TARBALL = Path("/content/t5_v3_patches.tar.gz")
with tarfile.open(TARBALL, "w:gz") as tar:
    tar.add(PATCH_DIR, arcname="t5_v3_patches")
print(f"Tarball: {TARBALL} ({TARBALL.stat().st_size / 1e9:.2f} GB)")

TARBALL_DRIVE = DRIVE_ROOT.parent / "t5_v3_patches.tar.gz"
shutil.move(str(TARBALL), TARBALL_DRIVE)
print(f"Moved to Drive: {TARBALL_DRIVE}")

# %% [markdown]
# ## 10. SHA-256 + freeze metadata

# %%
import hashlib

def sha256_file(path, chunk=2**20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()

manifest_sha = sha256_file(PATCH_DIR / "manifest.parquet")
tarball_sha = sha256_file(TARBALL_DRIVE)

with open(PATCH_DIR / "t5_v3_freeze_metadata.json", "w") as f:
    json.dump({
        "task": "T5",
        "task_description": "AutoPET-III PSMA cross-tracer detection (per A9b + A10 + A11-multi-series-handling)",
        "freeze_timestamp_utc": freeze_timestamp,
        "amendment_log_ref": "A9b/A10/A11 (osf/amendment_log.md v10, SHA a3197e4f…)",
        "source": {
            "release": "AutoPET-III TCIA PSMA-PET-CT-Lesions",
            "tcia_collection": "PSMA-PET-CT-Lesions",
            "licence": "CC BY 4.0",
            "version": "Version 2 (released 2026-02-26)",
            "dicom_path": str(DICOM_ROOT),
        },
        "p79_provenance": {
            "reviewed_parquet_sha256": "4ef08570271cd6155036925e115ab271b5b0e556ea1fbfd06ef4be8911ad6251",
            "nnunet_checkpoint_sha256": "29a2b99097666f418b4fb7c50908eb2416158dcc54e7c8fb38d110f0135f49d4",
            "p79_osf_project": "https://osf.io/j5ry4/",
        },
        "patches": {
            "n_patients_ok": int(manifest["patient_id"].nunique()),
            "n_series_ok": int(manifest["series_uid"].nunique()),
            "n_patches_total": int(len(manifest)),
            "n_lesion": int((manifest["label"] == 1).sum()),
            "n_background": int((manifest["label"] == 0).sum()),
            "patch_size_3d": list(PATCH_SIZE_3D),
            "spacing": list(SPACING),
            "mip_size_2d": MIP_SIZE_2D,
        },
        "config": {
            "iou_threshold": IOU_THRESHOLD,
            "min_lesion_voxels": MIN_LESION_VOXELS,
            "bg_suv_threshold": BG_SUV_THRESHOLD,
            "bg_patches_per_lesion": BG_PATCHES_PER_LESION,
            "max_patches_per_patient": MAX_PATCHES_PER_PATIENT,
            "include_all_series": INCLUDE_ALL_SERIES,
        },
        "hashes": {
            "manifest.parquet": manifest_sha,
            "t5_v3_patches.tar.gz": tarball_sha,
        },
    }, f, indent=2)

print(f"manifest.parquet SHA-256: {manifest_sha}")
print(f"t5_v3_patches.tar.gz SHA-256: {tarball_sha}")

# %% [markdown]
# ## 11. Done
#
# **"Save & Run All"** on Colab CPU. Then upload `t5_v3_patches.tar.gz` to Kaggle
# as `pet-fm-bench-t5-patches-v3` (mirror the T1 upload flow). When Kaggle ingests
# the tarball, it will produce the same doubly-nested layout as T1; `t5_02_embeddings.py`'s
# patched §3 (3-layout candidate detection) handles it.
#
# ## 12. Output schema (for `t5_02_embeddings.py`)
#
# **manifest.parquet** — one row per patch:
#   - `patient_id, patch_id, series_uid, label, lesion_index, lesion_voxels, iou,
#      patch_origin_zyx, patch_centre_zyx, study_date, radionuclide, vendor,
#      softmax_mean, softmax_entropy`
#
# **patches/{case_id}/{patch_id}.npz** — per patch:
#   - `patch_3d, mip_coronal, mip_axial, mip_sagittal` (all float16)
#
# Compared to T1's manifest schema, T5 adds: `series_uid, radionuclide, vendor,
# softmax_mean, softmax_entropy`. Downstream `t5_02_embeddings.py` already
# preserves these columns when emitting `t5_labels.parquet`.
