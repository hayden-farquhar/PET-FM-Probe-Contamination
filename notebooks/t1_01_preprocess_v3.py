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
# # T1 Notebook 1/2 v3: AutoPET-I FDAT — Preprocess for FDG Lesion-Patch Classification
#
# **PET-FM-Bench** | DOI: [10.17605/OSF.IO/DQ2JA](https://doi.org/10.17605/OSF.IO/DQ2JA)
#
# **PLATFORM:** Google **Colab** (NOT Kaggle) — needs Drive mount for AutoPET-I FDAT zip.
# **Runtime:** CPU | **Time:** ~1.5–2.5 hr (461 patients, single-pass) | **Disk:** ~15 GB output
#
# **STATUS: SKELETON — code review only. Not yet executed. Awaiting:**
# 1. User sign-off on Option 1 architecture (Colab preprocess → Kaggle embeddings).
# 2. A9a logged on OSF (✅ done — amendment_log.md v8, SHA `dd0b1e99…`).
#
# ## What this notebook does (per A9a)
#
# - Reads `/My Drive/petfm_data/autopet_i/fdg-pet-ct-lesions.zip` (282.9 GB FDAT release;
#   pre-computed SUV + ground-truth SEG NIfTIs shipped together).
# - For each patient: load SUV.nii.gz + SEG.nii.gz, resample to 2mm isotropic, extract
#   **lesion-centred patches** + **matched-count background controls**.
# - Saves per-patch parquet (patch-level metadata + labels) and per-patch float16
#   3D + 2D-MIP arrays.
# - Output → upload to Kaggle as `pet-fm-bench-t1-patches-v3` for downstream embedding.
#
# ## Critical deviations from existing v3 task pattern
#
# 1. **No SUV computation.** FDAT pre-computed SUV. Applying `suv_conversion.py` would
#    re-scale already-SUV-converted values. Skip that step entirely.
# 2. **Patch unit = lesion-or-background, not whole patient.** Existing T4/T6/T7/T8/T9
#    use one row per (patient_id, fm, layer, view); T1 uses one row per (patient_id,
#    patch_id, fm, layer, view). Embedding extraction (`t1_02_embeddings.py`) iterates
#    patches, not patients. probe_analysis.py v5 will need a T1-specific eval path.
# 3. **Per-patch 2D MIPs.** 2D FMs (BiomedCLIP, DINOv2, RAD-DINO) consume a 224×224
#    MIP derived from each individual 96³ patch — not a whole-body MIP. Three views
#    (coronal/axial/sagittal) saved per patch.
# 4. **Output is patch-level parquet + per-patch npz**, not patient-level npz. Schema
#    documented in §10 below.

# %% [markdown]
# ## 1. Setup (Colab)

# %%
# !pip install -q SimpleITK nibabel scipy scikit-image pandas pyarrow

from pathlib import Path
import io
import json
import zipfile
import shutil
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy import ndimage
from skimage import measure

# Colab-Drive mount
try:
    from google.colab import drive
    drive.mount("/content/drive", force_remount=False)
    DRIVE_ROOT = Path("/content/drive/My Drive/petfm_data/autopet_i")
except ImportError:
    # Allow local-laptop dry-run with a manually-set path
    DRIVE_ROOT = Path("/path/to/manually/mounted/autopet_i")
    print("⚠ Not on Colab — set DRIVE_ROOT manually for local dry-run.")

ZIP_PATH = DRIVE_ROOT / "fdg-pet-ct-lesions.zip"
assert ZIP_PATH.is_file(), f"AutoPET-I zip missing at {ZIP_PATH}"

# Local Colab disk for patch output (move to Drive at end if persisting beyond session)
TMP_DIR = Path("/content/t1_v3_tmp")          # transient unzip workspace
PATCH_DIR = Path("/content/t1_v3_patches")    # final output, will be zipped to Drive
TMP_DIR.mkdir(parents=True, exist_ok=True)
PATCH_DIR.mkdir(parents=True, exist_ok=True)

freeze_timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
print(f"Freeze timestamp (UTC): {freeze_timestamp}")
print(f"AutoPET-I zip: {ZIP_PATH} ({ZIP_PATH.stat().st_size / 1e9:.1f} GB)")

# %% [markdown]
# ## 2. Configuration
#
# Patch geometry MUST match the existing v3 task convention so downstream FM
# embedding extraction notebooks operate without per-task changes.

# %%
PATCH_SIZE_3D = (96, 96, 96)         # matches T4/T6/T7/T8 v3
SPACING       = (2.0, 2.0, 2.0)      # 2mm isotropic, matches v3 convention
MIP_SIZE_2D   = 224                   # for 2D FM consumption

# Lesion-aware patch extraction parameters (registration §3.1 / A9a)
IOU_THRESHOLD       = 0.5             # patch labelled positive if seg-overlap IoU ≥ 0.5
MIN_LESION_VOXELS   = 27              # discard <3³-voxel speckle (likely seg artefacts)
BG_SUV_THRESHOLD    = 2.5             # background patches sampled where SUV < this
BG_PATCHES_PER_LESION = 1             # 1:1 lesion:background matched-count sampling
MAX_PATCHES_PER_PATIENT = 50          # safety cap (some patients have 30+ lesions)

# AutoPET-I cancer-type stratification (per registration §3.3 stratified 70/15/15)
# Diagnosis category extracted from FDAT metadata.json (per patient) — done in §6.
CANCER_TYPES_EXPECTED = {"melanoma", "lung_cancer", "lymphoma", "negative"}

print("Patch config:", PATCH_SIZE_3D, "@", SPACING, "spacing")
print(f"IoU threshold: {IOU_THRESHOLD}; bg SUV cap: {BG_SUV_THRESHOLD}")

# %% [markdown]
# ## 3. Patient enumeration from FDAT zip
#
# Verified zip structure (2026-04-28):
# ```
# fdg-pet-ct-lesions.zip
# ├── FDG-PET-CT-Lesions/
# │   ├── PETCT_<patient_hash>/
# │   │   └── <MM-DD-YYYY-NA-PET-CT…study_suffix>/
# │   │       ├── SUV.nii.gz
# │   │       ├── SEG.nii.gz
# │   │       └── CTres.nii.gz
# │   └── …
# └── home/rakuest1/fdg_metadata.csv
# ```
# Patient ID = level-1 folder name (e.g. `PETCT_4d7b745a7b`), matches the `Subject ID`
# column in metadata.csv. Study folder name has a leading `MM-DD-YYYY` we parse for
# chronological sort. Multi-study patients exist (longitudinal) — for T1 lesion-patch
# classification we use the EARLIEST study per patient (single-time-point classification
# semantics per registration H1).

# %%
COLLECTION_PREFIX = "FDG-PET-CT-Lesions"   # zip's top-level data folder

patient_studies = {}  # patient_id -> {study_dir: set(filenames)}
with zipfile.ZipFile(ZIP_PATH) as zf:
    names = zf.namelist()
    for n in names:
        parts = n.strip("/").split("/")
        if len(parts) < 4 or parts[0] != COLLECTION_PREFIX:
            continue
        pid, study_dir, fname = parts[1], parts[2], parts[-1]
        if fname not in ("SUV.nii.gz", "SEG.nii.gz"):
            continue
        patient_studies.setdefault(pid, {}).setdefault(study_dir, set()).add(fname)

def _parse_study_date(study_dir):
    """Extract MM-DD-YYYY prefix from FDAT study folder name."""
    try:
        return datetime.strptime(study_dir[:10], "%m-%d-%Y")
    except (ValueError, IndexError):
        return datetime.max  # parse failure → sort to end (use whatever else is available)

patient_first_study = {}
for pid, studies in patient_studies.items():
    complete = [sd for sd, files in studies.items()
                if {"SUV.nii.gz", "SEG.nii.gz"}.issubset(files)]
    if not complete:
        continue
    patient_first_study[pid] = min(complete, key=_parse_study_date)

print(f"Patients enumerated in zip: {len(patient_studies)}")
print(f"Patients with usable first study (SUV + SEG present): {len(patient_first_study)}")
# Expected: ~900 patients total (raw FDAT), of which we keep all with both files.
# (companion project)'s 461 gate is the lesion-positive subset; for T1 binary lesion-patch
# classification we want both lesion-positive AND lesion-negative cohorts.

# %% [markdown]
# ## 4. Helpers: zip-streaming NIfTI read, resample, lesion-aware patch sampler

# %%
def read_nifti_from_zip(zf, member_name):
    """Stream a NIfTI from zip into a SimpleITK image without disk extraction.

    Uses TMP_DIR (under /content, ~107 GB) instead of /tmp (Colab's /tmp often
    shares disk with Drive FUSE cache and can fill in long runs). Tempfile is
    deleted immediately after sitk.ReadImage returns; the SimpleITK image holds
    its own in-memory copy.
    """
    import uuid
    tmp_path = TMP_DIR / f"nifti_{uuid.uuid4().hex}.nii.gz"
    try:
        with zf.open(member_name) as fp, open(tmp_path, "wb") as tf:
            shutil.copyfileobj(fp, tf)
        img = sitk.ReadImage(str(tmp_path))
    finally:
        tmp_path.unlink(missing_ok=True)
    return img


def resample_isotropic(img_sitk, spacing=SPACING, interpolator=sitk.sitkLinear):
    orig_spacing = img_sitk.GetSpacing()
    orig_size = img_sitk.GetSize()
    new_size = [int(round(s * sp / t)) for s, sp, t in zip(orig_size, orig_spacing, spacing)]
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(img_sitk.GetDirection())
    resampler.SetOutputOrigin(img_sitk.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(0)
    resampler.SetInterpolator(interpolator)
    return resampler.Execute(img_sitk)


def extract_centred_patch(volume, centre_zyx, patch_size=PATCH_SIZE_3D):
    """Extract a 3D patch centred on (z, y, x); zero-pad if patch crosses volume edge."""
    pz, py, px = patch_size
    cz, cy, cx = centre_zyx
    z0 = int(cz - pz // 2)
    y0 = int(cy - py // 2)
    x0 = int(cx - px // 2)
    z1, y1, x1 = z0 + pz, y0 + py, x0 + px

    # Compute clipped indices into the volume + corresponding write slice in the padded patch
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
    """IoU of the patch's seg-positive voxels vs the lesion's total voxels in the
    surrounding region. We compute against the patch-extent of full_seg."""
    pz, py, px = seg_patch.shape
    z0, y0, x0 = patch_origin
    # Pure intra-patch IoU of seg vs full (since `seg_patch` is already a slice of `full_seg`,
    # this collapses to "fraction of full_seg lesion captured by the patch ROI").
    # For lesion-centred patches the IoU is dominated by whether the patch fully contains
    # the lesion. Computed as |patch ∩ lesion| / |patch ∪ lesion within patch bbox|.
    inter = int(seg_patch.sum())
    if inter == 0:
        return 0.0
    # Union over the lesion-component voxels touched by the patch
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
    """Memory-efficient rejection sampling for background patch centres.

    Avoids `np.argwhere((seg == 0) & (suv < suv_max))` which materialises
    a (~30M × 3 int64) ≈ 720 MB allocation on whole-body PET volumes — the
    primary cause of Colab kernel OOM in v1 of this skeleton.

    Strategy: sample random in-bounds voxels and accept those satisfying the
    seg==0 and SUV<suv_max predicate. Each rejection costs a single voxel
    lookup (O(1)). Volume background fraction on AutoPET-I FDAT is typically
    >95% (lesions are sparse + most-of-body is low-SUV), so acceptance rate
    is high; max_tries_factor caps total samples at n_wanted * factor to avoid
    pathological loops on corner cases (whole-volume tumour, etc.).
    """
    if rng is None:
        rng = np.random.default_rng(42)
    pz, py, px = patch_size
    hz, hy, hx = pz // 2, py // 2, px // 2
    sz, sy, sx = suv.shape
    if sz <= pz or sy <= py or sx <= px:
        return []  # volume smaller than a patch — skip

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
    """Compute coronal/axial/sagittal MIPs of a 3D patch and resize to target_size×target_size."""
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
# ## 5. Per-patient processing loop
#
# **Memory-efficient design** (post-OOM patch, 2026-04-28):
# 1. Per-patient logic wrapped in `process_patient()` so locals fall out of scope
#    at function return — Python releases SUV/SEG/resampled buffers reliably.
# 2. Explicit `gc.collect()` after each patient to force minor + major GC cycles.
# 3. Progressive checkpoint save every CHECKPOINT_EVERY patients —
#    `manifest_partial.parquet` + `preprocessing_log_partial.csv` written to disk
#    so a kernel OOM/restart doesn't lose work; rerun resumes from these.
# 4. Background patches via rejection sampling (see §4 `sample_background_patches`)
#    avoids the np.argwhere O(N) allocation that was the OOM root cause.

# %%
import gc

CHECKPOINT_EVERY = 50  # save manifest_partial every N patients

PARTIAL_MANIFEST = PATCH_DIR / "manifest_partial.parquet"
PARTIAL_LOG = PATCH_DIR / "preprocessing_log_partial.csv"

# Resume support: if partial outputs exist, skip patients already processed.
manifest_rows = []
log_rows = []
already_processed = set()
if PARTIAL_MANIFEST.exists() and PARTIAL_LOG.exists():
    prev_manifest = pd.read_parquet(PARTIAL_MANIFEST)
    prev_log = pd.read_csv(PARTIAL_LOG)
    # On resume, only treat status == "ok" as "done" — error rows should be retried
    # (e.g., transient disk-full failures from a previous run). Drop the error rows
    # from log_rows so they don't accumulate as duplicates.
    ok_log = prev_log[prev_log["status"] == "ok"].copy()
    err_log = prev_log[prev_log["status"] != "ok"].copy()
    manifest_rows = prev_manifest.to_dict("records")
    log_rows = ok_log.to_dict("records")
    already_processed = set(ok_log["patient_id"].astype(str))
    print(f"RESUMING from partial output:")
    print(f"  {len(already_processed)} patients ok (skip on resume)")
    print(f"  {len(err_log)} patients errored previously (RETRY on resume)")
    print(f"  {len(manifest_rows)} patches in manifest (preserved)")
else:
    print("No partial output found — starting fresh")


def process_patient(pid, study_dir, zf, rng):
    """Process one patient → returns (rows_for_manifest, log_record).

    All locals fall out of scope on return → Python reliably frees SUV/SEG/
    resampled buffers between iterations.
    """
    suv_member = f"{COLLECTION_PREFIX}/{pid}/{study_dir}/SUV.nii.gz"
    seg_member = f"{COLLECTION_PREFIX}/{pid}/{study_dir}/SEG.nii.gz"

    suv_img = read_nifti_from_zip(zf, suv_member)
    seg_img = read_nifti_from_zip(zf, seg_member)

    suv_iso = resample_isotropic(suv_img, SPACING, sitk.sitkLinear)
    seg_iso = resample_isotropic(seg_img, SPACING, sitk.sitkNearestNeighbor)

    suv = sitk.GetArrayFromImage(suv_iso).astype(np.float32)
    seg = (sitk.GetArrayFromImage(seg_iso) > 0).astype(np.uint8)
    suv_max = float(suv.max())
    suv_mean = float(suv.mean())
    shape_iso = str(suv.shape)

    # Drop SimpleITK objects ASAP — only the numpy views are needed downstream
    del suv_img, seg_img, suv_iso, seg_iso

    labels, n_lesions = ndimage.label(seg)
    lesion_props = measure.regionprops(labels)
    lesion_props = [p for p in lesion_props if p.area >= MIN_LESION_VOXELS]
    n_lesions_kept = len(lesion_props)

    if n_lesions_kept > MAX_PATCHES_PER_PATIENT // 2:
        keep_idx = rng.choice(n_lesions_kept,
                              size=MAX_PATCHES_PER_PATIENT // 2,
                              replace=False)
        lesion_props = [lesion_props[i] for i in sorted(keep_idx)]

    patient_dir = PATCH_DIR / "patches" / pid
    patient_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    # Lesion patches (label = 1)
    for li, lp in enumerate(lesion_props):
        cz, cy, cx = lp.centroid
        patch_3d, origin = extract_centred_patch(suv, (cz, cy, cx))
        seg_patch, _    = extract_centred_patch(seg, (cz, cy, cx))
        iou = compute_iou(seg_patch, seg, origin)
        mips = patch_to_mip_views(patch_3d)

        patch_id = f"{pid}_lesion_{li:03d}"
        np.savez_compressed(
            patient_dir / f"{patch_id}.npz",
            patch_3d=patch_3d.astype(np.float16),
            mip_coronal=mips["coronal"].astype(np.float16),
            mip_axial=mips["axial"].astype(np.float16),
            mip_sagittal=mips["sagittal"].astype(np.float16),
        )
        rows.append({
            "patient_id": pid, "patch_id": patch_id,
            "label": 1, "lesion_index": li,
            "lesion_voxels": int(lp.area), "iou": float(iou),
            "patch_origin_zyx": json.dumps(list(origin)),
            "patch_centre_zyx": json.dumps([float(cz), float(cy), float(cx)]),
            "study_date": study_dir,
        })

    # Background patches (label = 0)
    n_bg = max(BG_PATCHES_PER_LESION * len(lesion_props), 1) if len(lesion_props) > 0 else 5
    bg_centres = sample_background_patches(suv, seg, n_bg, rng=rng)
    for bi, centre in enumerate(bg_centres):
        patch_3d, origin = extract_centred_patch(suv, centre)
        mips = patch_to_mip_views(patch_3d)
        patch_id = f"{pid}_bg_{bi:03d}"
        np.savez_compressed(
            patient_dir / f"{patch_id}.npz",
            patch_3d=patch_3d.astype(np.float16),
            mip_coronal=mips["coronal"].astype(np.float16),
            mip_axial=mips["axial"].astype(np.float16),
            mip_sagittal=mips["sagittal"].astype(np.float16),
        )
        rows.append({
            "patient_id": pid, "patch_id": patch_id,
            "label": 0, "lesion_index": -1,
            "lesion_voxels": 0, "iou": 0.0,
            "patch_origin_zyx": json.dumps(list(origin)),
            "patch_centre_zyx": json.dumps([float(c) for c in centre]),
            "study_date": study_dir,
        })

    log_record = {
        "patient_id": pid,
        "status": "ok",
        "n_lesions_kept": n_lesions_kept,
        "n_lesions_raw": int(n_lesions),
        "n_bg_sampled": len(bg_centres),
        "suv_max": suv_max,
        "suv_mean": suv_mean,
        "shape_iso": shape_iso,
    }
    return rows, log_record


def save_partial(manifest_rows, log_rows):
    pd.DataFrame(manifest_rows).to_parquet(PARTIAL_MANIFEST, index=False)
    pd.DataFrame(log_rows).to_csv(PARTIAL_LOG, index=False)


try:
    from tqdm.auto import tqdm as tqdm_auto  # auto-detects notebook vs terminal
except ImportError:
    from tqdm import tqdm as tqdm_auto

zf = zipfile.ZipFile(ZIP_PATH)
remaining = [(pid, sd) for pid, sd in patient_first_study.items()
             if pid not in already_processed]
print(f"Patients to process this session: {len(remaining)}")
print(f"Already done (resumed): {len(already_processed)}")
print(f"Checkpoint every: {CHECKPOINT_EVERY} patients\n")

# Progress reporting layers (so the user sees both real-time and summary signals):
#   1. tqdm bar — live iteration count + ETA + iter/sec
#   2. Per-patient one-liner — pid + n_lesions + n_patches (folded into bar postfix)
#   3. Checkpoint markers — every CHECKPOINT_EVERY patients ("✓ checkpoint saved …")

pbar = tqdm_auto(total=len(remaining), desc="T1 patients", unit="patient")

try:
    for i, (pid, study_dir) in enumerate(remaining):
        rng = np.random.default_rng(int(abs(hash(pid)) % (2**31)))
        try:
            rows, log_record = process_patient(pid, study_dir, zf, rng)
            manifest_rows.extend(rows)
            log_rows.append(log_record)
            pbar.set_postfix_str(
                f"last={pid[-8:]} les={log_record['n_lesions_kept']} "
                f"patches+={len(rows)} total={len(manifest_rows)}",
                refresh=False,
            )
        except Exception as e:
            log_rows.append({"patient_id": pid,
                             "status": f"error: {type(e).__name__}: {e}"})
            pbar.write(f"ERROR {pid}: {type(e).__name__}: {e}")

        gc.collect()
        pbar.update(1)

        if (i + 1) % CHECKPOINT_EVERY == 0:
            save_partial(manifest_rows, log_rows)
            pbar.write(f"  ✓ checkpoint @ i={i+1}: "
                       f"{len(manifest_rows)} patches, {len(log_rows)} patient log rows")
finally:
    pbar.close()
    zf.close()
    save_partial(manifest_rows, log_rows)  # final flush regardless of exit path
    print(f"\nFinal: {len(manifest_rows)} patches across {len(log_rows)} patient log rows")

# %% [markdown]
# ## 6. Cancer-type metadata join (FDAT `home/rakuest1/fdg_metadata.csv`)
#
# Verified 2026-04-28: FDAT zip ships `home/rakuest1/fdg_metadata.csv` with patient-
# level `Subject ID` + `diagnosis` columns. 4 categories: NEGATIVE / MELANOMA /
# LUNG_CANCER / LYMPHOMA. We join diagnosis into `manifest.parquet` so that Phase 4 v3
# `06_task_splits.py` can do stratified 70/15/15 (registration §3.3) without
# re-reading the zip.

# %%
META_MEMBER = "home/rakuest1/fdg_metadata.csv"

with zipfile.ZipFile(ZIP_PATH) as zf:
    with zf.open(META_MEMBER) as fp:
        meta_df = pd.read_csv(fp)

# Each patient has multiple series rows; diagnosis is identical across rows for any
# given patient. Group by Subject ID and take first.
patient_diagnosis = meta_df.groupby("Subject ID")["diagnosis"].first().to_dict()

# Map FDAT label vocabulary → registered cancer-type vocabulary (lowercase + snake_case)
DIAG_MAP = {
    "MELANOMA":    "melanoma",
    "LUNG_CANCER": "lung_cancer",
    "LYMPHOMA":    "lymphoma",
    "NEGATIVE":    "negative",
}
patient_diagnosis = {pid: DIAG_MAP.get(d, str(d).lower())
                     for pid, d in patient_diagnosis.items()}

print(f"FDAT metadata patients: {len(patient_diagnosis)}")
print("Diagnosis distribution (patient-level):")
diag_series = pd.Series(list(patient_diagnosis.values()))
print(diag_series.value_counts().to_string())

manifest = pd.DataFrame(manifest_rows)
manifest["cancer_type"] = manifest["patient_id"].map(patient_diagnosis)
log_df = pd.DataFrame(log_rows)

# Coverage sanity check
processed_patients = set(manifest["patient_id"]) if len(manifest) else set()
unmapped = processed_patients - set(patient_diagnosis.keys())
if unmapped:
    print(f"\n⚠ {len(unmapped)} processed patients missing in metadata.csv: "
          f"{list(unmapped)[:5]}")
n_nan = int(manifest["cancer_type"].isna().sum())
if n_nan:
    print(f"⚠ {n_nan} manifest rows have NaN cancer_type")

print("\nFinal manifest cancer-type distribution (patient-level):")
patient_level = manifest.drop_duplicates("patient_id")
print(patient_level["cancer_type"].value_counts(dropna=False).to_string())

manifest.to_parquet(PATCH_DIR / "manifest.parquet", index=False)
log_df.to_csv(PATCH_DIR / "preprocessing_log.csv", index=False)

# %% [markdown]
# ## 7. QC + summary

# %%
ok = log_df[log_df["status"] == "ok"]
print("=== T1 v3 AutoPET-I FDG Preprocessing QC ===")
print(f"Patients ok: {len(ok)}/{len(log_df)}")
print(f"Errors: {(log_df['status'].str.startswith('error')).sum()}")
print(f"Total patches: {len(manifest)}")
print(f"  lesion patches: {(manifest['label'] == 1).sum()}")
print(f"  background patches: {(manifest['label'] == 0).sum()}")

if len(ok):
    print(f"\nLesion counts per patient:")
    print(f"  median: {ok['n_lesions_kept'].median():.0f}")
    print(f"  max: {ok['n_lesions_kept'].max()}")
    print(f"  patients with 0 lesions: {(ok['n_lesions_kept'] == 0).sum()}")

size_gb = sum(f.stat().st_size for f in PATCH_DIR.rglob("*") if f.is_file()) / 1e9
print(f"\nOutput size: {size_gb:.2f} GB")

# %% [markdown]
# ## 8. Stage output for Kaggle upload
#
# Compress the patch directory into a single tarball for upload as
# `pet-fm-bench-t1-patches-v3` on Kaggle. Kaggle dataset size limit per
# file is 20 GB; if the tarball exceeds that, switch to per-shard upload
# (split by patient_id mod N).

# %%
TARBALL = Path("/content/t1_v3_patches.tar.gz")
import tarfile
with tarfile.open(TARBALL, "w:gz") as tar:
    tar.add(PATCH_DIR, arcname="t1_v3_patches")
print(f"Tarball: {TARBALL} ({TARBALL.stat().st_size / 1e9:.2f} GB)")

# Move to Drive for persistent access
TARBALL_DRIVE = DRIVE_ROOT / "t1_v3_patches.tar.gz"
shutil.move(str(TARBALL), TARBALL_DRIVE)
print(f"Moved to Drive: {TARBALL_DRIVE}")

# %% [markdown]
# ## 9. SHA-256 + freeze metadata

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

with open(PATCH_DIR / "t1_v3_freeze_metadata.json", "w") as f:
    json.dump({
        "task": "T1",
        "task_description": "AutoPET-I FDG lesion-patch classification (per A9a)",
        "freeze_timestamp_utc": freeze_timestamp,
        "amendment_log_ref": "A9a (osf/amendment_log.md v8, SHA dd0b1e99…)",
        "source": {
            "release": "AutoPET-I FDAT",
            "doi": "10.57754/FDAT.wf9fy-txq84",
            "licence": "CC BY-NC 4.0",
            "zip_path": str(ZIP_PATH),
        },
        "patches": {
            "n_patients_ok": int((log_df["status"] == "ok").sum()),
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
        },
        "hashes": {
            "manifest.parquet": manifest_sha,
            "t1_v3_patches.tar.gz": tarball_sha,
        },
    }, f, indent=2)

print(f"manifest.parquet SHA-256: {manifest_sha}")
print(f"t1_v3_patches.tar.gz SHA-256: {tarball_sha}")

# %% [markdown]
# ## 10. Output schema (for downstream `t1_02_embeddings.py`)
#
# **manifest.parquet** — one row per patch:
#   - `patient_id`        str   — FDAT patient identifier
#   - `patch_id`          str   — `{patient_id}_lesion_{nnn}` or `{patient_id}_bg_{nnn}`
#   - `label`             int   — 1 = lesion patch (IoU ≥ 0.5), 0 = background
#   - `lesion_index`      int   — connected-component index, -1 for background
#   - `lesion_voxels`     int   — total voxels in the lesion component
#   - `iou`               float — IoU of patch ROI vs lesion (0 for background)
#   - `patch_origin_zyx`  str   — JSON list of (z, y, x) corner of patch in resampled volume
#   - `patch_centre_zyx`  str   — JSON list of (z, y, x) lesion centroid (or bg sample point)
#   - `study_date`        str   — FDAT study_date subfolder name
#   - `cancer_type`       str   — FDAT diagnosis category (for stratified split)
#
# **patches/{patient_id}/{patch_id}.npz** — per patch:
#   - `patch_3d`     float16, shape (96, 96, 96)
#   - `mip_coronal`  float16, shape (224, 224)
#   - `mip_axial`    float16, shape (224, 224)
#   - `mip_sagittal` float16, shape (224, 224)
#
# **t1_v3_freeze_metadata.json** — provenance + SHA-256
# **preprocessing_log.csv** — per-patient processing status
#
# Downstream `t1_02_embeddings.py` (to be written):
#   - 3D FMs (FMCIB, CT-FM, random_init): consume `patch_3d` directly.
#   - 2D FMs (BiomedCLIP, DINOv2, RAD-DINO): consume `mip_axial` (default) or all
#     three MIP views averaged. Convention to be set in `t1_02_embeddings.py`.
#   - Per-patch embeddings → patch-level parquet emitted as
#     `pet-fm-bench-t1-embeddings-v3`.
#
# probe_analysis.py v5 will need a T1-specific dispatch path: per-patch
# AUROC with patient-level CV (no patient straddles train/test splits).
