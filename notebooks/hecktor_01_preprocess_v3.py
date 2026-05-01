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
# # HECKTOR 2025 Notebook 1/2 v3: Preprocess for T2 (HN tumour patch-classification) + T3 (RFS prediction)
#
# **PET-FM-Bench** | DOI: [10.17605/OSF.IO/DQ2JA](https://doi.org/10.17605/OSF.IO/DQ2JA)
#
# **PLATFORM:** Google **Colab** (NOT Kaggle) — needs Drive mount for HECKTOR 2025 zip.
# **Runtime:** CPU | **Time:** ~3-5 hr (~707 patients, single-pass) | **Disk:** ~6-8 GB output
#
# **STATUS: DRAFT — code review only. Not yet executed. Awaiting:**
# 1. User code review.
# 2. A12 logged on OSF (✅ done — amendment_log.md v11, SHA `d68e3a9a…`).
#
# ## What this notebook does (per A12a + A12b + A12c)
#
# - Reads `/My Drive/HECKTOR 2025/hecktor_2025_training_defaced.zip` (48.33 GB; pre-SUV
#   PT, pre-HU CT, GTVp + GTVn segmentation NIfTIs shipped together; defaced training
#   release accessed 2026-04-29, SHA-256 `1abcf1d9…`).
# - Reads EHR clinical xlsx for Task1 (segmentation→T2), Task2 (RFS→T3), Task3,
#   Relapse, RFS, centre_id, HPV status, T/N/M-stage labels.
# - For each patient in union(Task1, Task2) = ~707 patients: load PT + SEG + CT,
#   resample SEG → PT grid (CRITICAL: HECKTOR ships them on different grids),
#   resample to 2mm isotropic SUV grid, extract lesion-centred patches + matched
#   background controls.
# - SEG label semantics: 0 = background, 1 = GTVp (primary tumour), 2 = GTVn
#   (involved nodal). Primary T2 evaluation = binary GTVp ∪ GTVn (analogous to T1).
#   GTVp-only is a sensitivity analysis (requires post-hoc filter on `lesion_class`).
# - Saves per-patch parquet (patch-level metadata + labels) and per-patch float16
#   3D + 2D-MIP arrays.
# - Output → upload to Kaggle as `pet-fm-bench-hecktor-patches-v3` for downstream
#   embedding extraction (single dataset serves both T2 and T3 cohorts via manifest
#   filter).
#
# ## Critical deviations from t1_01_preprocess_v3.py + t5_01_preprocess_v3.py
#
# 1. **SEG and PT on different spatial grids** (verified scratch.md D4). HECKTOR ships
#    SEG at e.g. (200, 200, 256) at 1.0 mm vs PT at (200, 200, ~410) variable spacing.
#    Resample SEG → PT GRID first via SimpleITK ResampleImageFilter with
#    `sitk.sitkNearestNeighbor` interpolation (preserves label values 0/1/2).
#    THEN resample both PT and SEG to 2mm isotropic SUV grid for the patch extractor.
# 2. **Multi-class SEG.** Persist `lesion_class` ∈ {1=GTVp, 2=GTVn} per patch for the
#    GTVp-only sensitivity analysis. Primary T2 binary classification uses (lesion_class > 0).
# 3. **EHR-driven cohort definition.** Patient inclusion requires Task1=1 OR Task2=1
#    in the EHR Data sheet. Both labels are persisted in the manifest so the downstream
#    probe_analysis.py can dispatch T2 (Task1==1) or T3 (Task2==1) via filter.
# 4. **Both T2 and T3 in one notebook.** T3 (RFS) doesn't need new patches — it reuses
#    the T2 patches with manifest filter `task2_patient == True`. probe_analysis.py
#    v6 dispatches T3 from the same patch dataset.
# 5. **Drive mirror rsync** on every checkpoint (T5 lesson banked). Restores from Drive
#    on runtime restart.

# %% [markdown]
# ## 1. Setup (Colab)

# %%
# !pip install -q SimpleITK nibabel scipy scikit-image pandas pyarrow openpyxl

from pathlib import Path
import io
import json
import zipfile
import shutil
import subprocess
import uuid
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
    DRIVE_ROOT = Path("/content/drive/My Drive/HECKTOR 2025")
except ImportError:
    DRIVE_ROOT = Path("/path/to/manually/mounted/hecktor_2025")
    print("⚠ Not on Colab — set DRIVE_ROOT manually for local dry-run.")

DRIVE_ZIP = DRIVE_ROOT / "hecktor_2025_training_defaced.zip"
assert DRIVE_ZIP.is_file(), f"HECKTOR 2025 zip missing at {DRIVE_ZIP}"

# Local Colab disk for patch output (move tarball to Drive at end)
TMP_DIR = Path("/content/hecktor_v3_tmp")          # transient unzip workspace
PATCH_DIR = Path("/content/hecktor_v3_patches")    # final output, will be tarballed
DRIVE_MIRROR = DRIVE_ROOT / "hecktor_v3_patches_drive_mirror"
TMP_DIR.mkdir(parents=True, exist_ok=True)
PATCH_DIR.mkdir(parents=True, exist_ok=True)
DRIVE_MIRROR.mkdir(parents=True, exist_ok=True)

freeze_timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
print(f"Freeze timestamp (UTC): {freeze_timestamp}")
print(f"Drive zip: {DRIVE_ZIP} ({DRIVE_ZIP.stat().st_size / 1e9:.1f} GB)")
print(f"Drive mirror: {DRIVE_MIRROR}")

# %% [markdown]
# ## 1.5. Local zip pre-stage (5-10x I/O speedup vs Drive FUSE)
#
# Reading the 48 GB zip via Google Drive FUSE is the dominant per-patient cost
# (T5 measured ~17.8 s/series in steady state, ~80% of which was zip-stream
# overhead). Local-SSD reads run 5-10x faster. One ~10-15 min upfront copy
# trades for a wall-time drop from ~3-5 hr → ~45-60 min on the main loop.
#
# Behaviour:
# - Disk-space gate: requires ~53 GB free on /content; falls back to Drive-direct
#   read if insufficient (preserves correctness, just slower).
# - Idempotent: re-runs verify SHA-256 and skip copy if local matches canonical.
# - On Colab disconnect, /content is wiped → re-stage automatically on resume.
# - Canonical SHA from user's 2026-04-29 Drive download verification.

# %%
import hashlib
import time

LOCAL_ZIP = Path("/content/hecktor_2025_training_defaced.zip")
DRIVE_ZIP_CANONICAL_SHA = "1abcf1d96d38bb3d7b1eaf1889fa8ddd688f14b70876a1c7cf0cd7482d076df2"


def _file_sha256(path, chunk=2**20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


def stage_zip_locally():
    """Copy the Drive zip to /content for fast local-SSD reads.

    Returns the path to the active zip (local if staged, Drive otherwise).
    Idempotent + SHA-verified + disk-space-gated + fallback-tolerant.
    """
    free_gb = shutil.disk_usage("/content").free / 1e9
    needed_gb = DRIVE_ZIP.stat().st_size / 1e9
    if free_gb < needed_gb + 5:  # 5 GB headroom for tmp + output
        print(f"⚠ /content has {free_gb:.1f} GB free; need ~{needed_gb + 5:.1f} GB.")
        print("  Falling back to Drive-direct read (slower, but correct).")
        return DRIVE_ZIP

    # Idempotency check
    if LOCAL_ZIP.exists():
        local_size = LOCAL_ZIP.stat().st_size
        drive_size = DRIVE_ZIP.stat().st_size
        if local_size == drive_size:
            print(f"Local zip exists ({local_size / 1e9:.1f} GB) — verifying SHA…")
            local_sha = _file_sha256(LOCAL_ZIP)
            if local_sha == DRIVE_ZIP_CANONICAL_SHA:
                print(f"✓ Local zip SHA matches canonical — reusing (no copy needed)")
                return LOCAL_ZIP
            print(f"⚠ Local SHA {local_sha[:16]}… ≠ canonical {DRIVE_ZIP_CANONICAL_SHA[:16]}…")
            print("  Re-copying.")
            LOCAL_ZIP.unlink()
        else:
            print(f"⚠ Local zip partial ({local_size / 1e9:.1f} / {drive_size / 1e9:.1f} GB) — re-copying.")
            LOCAL_ZIP.unlink()

    # Copy with timing
    print(f"Pre-staging zip: Drive → /content/ ({needed_gb:.1f} GB, ~10-15 min)…")
    t0 = time.time()
    shutil.copy(str(DRIVE_ZIP), str(LOCAL_ZIP))
    elapsed = time.time() - t0
    print(f"  copy complete in {elapsed / 60:.1f} min "
          f"({needed_gb / (elapsed / 60):.1f} GB/min)")

    # Verify post-copy integrity
    local_sha = _file_sha256(LOCAL_ZIP)
    if local_sha != DRIVE_ZIP_CANONICAL_SHA:
        print(f"⚠ Post-copy SHA mismatch! Local={local_sha[:16]}…, expected={DRIVE_ZIP_CANONICAL_SHA[:16]}…")
        print("  Removing corrupt local copy and falling back to Drive-direct read.")
        LOCAL_ZIP.unlink(missing_ok=True)
        return DRIVE_ZIP
    print(f"✓ SHA verified ({local_sha[:16]}…)")
    return LOCAL_ZIP


ZIP_PATH = stage_zip_locally()
print(f"\nActive ZIP_PATH: {ZIP_PATH}")
print(f"  Read-from-local: {ZIP_PATH == LOCAL_ZIP}")
print(f"  Estimated per-patient time: {'~3-5 sec (local SSD)' if ZIP_PATH == LOCAL_ZIP else '~15-20 sec (Drive FUSE)'}")

# %% [markdown]
# ## 2. Configuration
#
# Patch geometry MUST match T1/T5 v3 convention so downstream FM embedding
# extraction notebooks operate without per-task changes (cross-task FM ranking
# requires uniform patch geometry).

# %%
PATCH_SIZE_3D = (96, 96, 96)         # matches T1/T5 v3
SPACING       = (2.0, 2.0, 2.0)      # 2mm isotropic, matches v3 convention
MIP_SIZE_2D   = 224                   # for 2D FM consumption

# Lesion-aware patch extraction parameters (per A12a — patch-classification reduction)
IOU_THRESHOLD       = 0.5             # patch labelled positive if seg-overlap IoU ≥ 0.5
MIN_LESION_VOXELS   = 27              # discard <3³-voxel speckle (likely seg artefacts)
BG_SUV_THRESHOLD    = 2.5             # background patches sampled where SUV < this
BG_PATCHES_PER_LESION = 1             # 1:1 lesion:background matched-count sampling
MAX_PATCHES_PER_PATIENT = 50          # safety cap

# HECKTOR-specific: SEG label values
SEG_LABEL_BACKGROUND = 0
SEG_LABEL_GTVP       = 1              # primary tumour
SEG_LABEL_GTVN       = 2              # involved nodal
SEG_LABELS_LESION    = (1, 2)         # binary union for primary T2 evaluation

# Centre stratification (per A12b)
CENTRE_PREFIXES_EXPECTED = {"CHUM", "CHUP", "CHUS", "CHUV", "MDA", "USZ", "HMR"}

print("Patch config:", PATCH_SIZE_3D, "@", SPACING, "spacing")
print(f"IoU threshold: {IOU_THRESHOLD}; bg SUV cap: {BG_SUV_THRESHOLD}")
print(f"SEG label semantics: 0=bg, 1=GTVp, 2=GTVn; binary union for T2 primary")

# %% [markdown]
# ## 3. EHR cohort enumeration from xlsx
#
# Verified zip structure (scratch.md D2, 2026-04-29):
# ```
# hecktor_2025_training_defaced.zip
# ├── HECKTOR 2025 Training Data Defaced ALL/
# │   ├── HECKTOR_2025_Training_EHR_with_Data_Dictionary.xlsx
# │   ├── <PATIENT_ID>/
# │   │   ├── <PATIENT_ID>__PT.nii.gz   (~ 8-15 MB)
# │   │   ├── <PATIENT_ID>__CT.nii.gz   (~ 3-30 MB)
# │   │   └── <PATIENT_ID>.nii.gz       (~ 150 KB SEG mask)
# │   └── ... (726 patient folders)
# ```

# %%
DATA_ROOT = "HECKTOR 2025 Training Data Defaced ALL/"
EHR_MEMBER = f"{DATA_ROOT}HECKTOR_2025_Training_EHR_with_Data_Dictionary.xlsx"

# Read EHR xlsx
with zipfile.ZipFile(ZIP_PATH) as zf, zf.open(EHR_MEMBER) as fp:
    ehr = pd.read_excel(io.BytesIO(fp.read()), sheet_name="Data")

# Defensive: strip column-name whitespace (Excel auto-fill sometimes adds trailing spaces)
ehr.columns = ehr.columns.str.strip()

print(f"Total EHR rows: {len(ehr)}")
print(f"EHR columns: {ehr.columns.tolist()}")

# Resolve Task 1 / Task 2 column names defensively (D3 confirmed "Task 1" / "Task 2" with space)
TASK1_COL_CANDIDATES = ["Task 1", "Task1", "Task_1", "task_1", "task1"]
TASK2_COL_CANDIDATES = ["Task 2", "Task2", "Task_2", "task_2", "task2"]
task1_col = next((c for c in TASK1_COL_CANDIDATES if c in ehr.columns), None)
task2_col = next((c for c in TASK2_COL_CANDIDATES if c in ehr.columns), None)
assert task1_col is not None, f"Could not find Task1 column among {TASK1_COL_CANDIDATES}. Got {ehr.columns.tolist()}"
assert task2_col is not None, f"Could not find Task2 column among {TASK2_COL_CANDIDATES}. Got {ehr.columns.tolist()}"
print(f"Resolved task columns: Task1='{task1_col}', Task2='{task2_col}'")

# Coerce values to numeric to handle string-typed flags (e.g. '1' vs 1) — silent empty-cohort guard
ehr[task1_col] = pd.to_numeric(ehr[task1_col], errors="coerce")
ehr[task2_col] = pd.to_numeric(ehr[task2_col], errors="coerce")

# Cohort filter: union(Task1=1, Task2=1) — both candidate-task indicators retained.
# Task3 patients excluded for now (not in registered scope; would need a future amendment).
ehr["task1_patient"] = (ehr[task1_col] == 1)
ehr["task2_patient"] = (ehr[task2_col] == 1)
in_cohort = ehr["task1_patient"] | ehr["task2_patient"]
cohort_df = ehr[in_cohort].copy()
assert len(cohort_df) > 0, (
    f"Cohort is empty after Task1/Task2 filter. "
    f"Task1 sum={ehr['task1_patient'].sum()}, Task2 sum={ehr['task2_patient'].sum()}. "
    f"Check column dtypes/values: {ehr[[task1_col, task2_col]].dtypes.to_dict()}"
)
print(f"\nCohort: union(Task1, Task2) = {len(cohort_df)} patients")
print(f"  Task1=1 only: {(cohort_df['task1_patient'] & ~cohort_df['task2_patient']).sum()}")
print(f"  Task2=1 only: {(~cohort_df['task1_patient'] & cohort_df['task2_patient']).sum()}")
print(f"  Task1=1 AND Task2=1: {(cohort_df['task1_patient'] & cohort_df['task2_patient']).sum()}")

# Patient ID column inference (HECKTOR 2025 typically uses 'PatientID' or similar;
# detect the column that matches the patient subdirectory naming convention).
PATIENT_ID_COL_CANDIDATES = ["PatientID", "Patient ID", "Patient_ID", "patient_id", "ID", "Subject ID"]
patient_id_col = next((c for c in PATIENT_ID_COL_CANDIDATES if c in cohort_df.columns), None)
assert patient_id_col is not None, (
    f"Could not find patient ID column among {PATIENT_ID_COL_CANDIDATES}. "
    f"Available columns: {cohort_df.columns.tolist()}"
)
print(f"Using patient ID column: '{patient_id_col}'")

# Centre ID column inference (D3 confirmed 'CenterID'; allow alternates).
CENTRE_ID_COL_CANDIDATES = ["CenterID", "Center", "CentreID", "Centre", "center_id", "centre_id"]
centre_id_col = next((c for c in CENTRE_ID_COL_CANDIDATES if c in cohort_df.columns), None)
print(f"Using centre ID column: '{centre_id_col}'")

# Build per-patient lookup of all task labels + clinical metadata
patient_meta = cohort_df.set_index(patient_id_col).to_dict(orient="index")
print(f"\nPer-patient metadata indexed: {len(patient_meta)} patients")
print(f"Centre distribution in cohort:")
if centre_id_col:
    print(cohort_df[centre_id_col].value_counts().sort_index().to_string())

# RFS event rate sanity check (per A12c)
if "Relapse" in cohort_df.columns and "RFS" in cohort_df.columns:
    t3_cohort = cohort_df[cohort_df["task2_patient"]]
    n_event = (t3_cohort["Relapse"] == 1).sum()
    n_total = t3_cohort["Relapse"].notna().sum()
    print(f"\nT3 RFS sanity (per A12c): {n_event}/{n_total} events "
          f"({100*n_event/max(n_total,1):.1f}%)")

# %% [markdown]
# ## 4. Patient enumeration from zip (verifies all 3 NIfTI files present)

# %%
# Build patient_id -> {filenames present} lookup from zip
patient_files = {}
with zipfile.ZipFile(ZIP_PATH) as zf:
    names = zf.namelist()
    for n in names:
        if not n.startswith(DATA_ROOT) or n.startswith("__MACOSX"):
            continue
        rest = n[len(DATA_ROOT):]
        if "/" not in rest:
            continue
        pid, fname = rest.split("/", 1)
        if pid in {"", "."}:
            continue
        if fname.endswith("/"):
            continue
        # Skip nested files (we only care about top-level patient files: PT/CT/SEG)
        if "/" in fname:
            continue
        patient_files.setdefault(pid, set()).add(fname)

# Filter patient_files to those with all 3 expected NIfTIs
required_files = lambda pid: {f"{pid}__PT.nii.gz", f"{pid}__CT.nii.gz", f"{pid}.nii.gz"}
patients_complete = {pid for pid, files in patient_files.items()
                     if required_files(pid).issubset(files)}
print(f"Patients with all 3 NIfTIs in zip: {len(patients_complete)}")

# Final cohort: intersection(EHR-cohort, complete-files-on-disk)
cohort_patients = sorted(set(patient_meta.keys()) & patients_complete)
missing_files = sorted(set(patient_meta.keys()) - patients_complete)
extra_files = sorted(patients_complete - set(patient_meta.keys()))
print(f"\nFinal cohort (EHR ∩ files): {len(cohort_patients)}")
if missing_files:
    print(f"⚠ {len(missing_files)} EHR-cohort patients missing files in zip "
          f"(first 5: {missing_files[:5]})")
if extra_files:
    print(f"  ({len(extra_files)} patients in zip but not in cohort — non-T1/T2 patients, expected)")

# %% [markdown]
# ## 5. Helpers: zip-streaming NIfTI read, SEG→PT resampling, lesion-aware patch sampler

# %%
def read_nifti_from_zip(zf, member_name):
    """Stream a NIfTI from zip into a SimpleITK image without disk extraction."""
    tmp_path = TMP_DIR / f"nifti_{uuid.uuid4().hex}.nii.gz"
    try:
        with zf.open(member_name) as fp, open(tmp_path, "wb") as tf:
            shutil.copyfileobj(fp, tf)
        img = sitk.ReadImage(str(tmp_path))
    finally:
        tmp_path.unlink(missing_ok=True)
    return img


def resample_to_reference(moving_img, reference_img, interpolator=sitk.sitkLinear):
    """Resample `moving_img` onto the spatial grid of `reference_img`.

    Critical for HECKTOR: SEG ships at a different grid than PT (D4 finding).
    We resample SEG → PT grid via nearest-neighbour to preserve discrete label
    values (0, 1, 2). Then both volumes can be co-resampled to the 2mm SUV grid
    in a second pass.
    """
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference_img)
    resampler.SetInterpolator(interpolator)
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(0)
    return resampler.Execute(moving_img)


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

    sz0, sy0, sx0 = max(0, z0), max(0, y0), max(0, x0)
    sz1 = min(volume.shape[0], z1)
    sy1 = min(volume.shape[1], y1)
    sx1 = min(volume.shape[2], x1)

    pz0, py0, px0 = sz0 - z0, sy0 - y0, sx0 - x0
    pz1, py1, px1 = pz0 + (sz1 - sz0), py0 + (sy1 - sy0), px0 + (sx1 - sx0)

    out = np.zeros(patch_size, dtype=volume.dtype)
    out[pz0:pz1, py0:py1, px0:px1] = volume[sz0:sz1, sy0:sy1, sx0:sx1]
    return out, (z0, y0, x0)


def compute_iou(seg_patch_binary, total_lesion_voxels):
    """Lesion-capture ratio = (lesion voxels inside patch) / (total lesion voxels).

    Bug fix vs T1/T5 templates (which had a no-op formula that always returned 1.0
    because both terms of the union were sliced from the same patch window).
    Renamed semantically — this is technically a "containment ratio", not Jaccard
    IoU, but it matches the intent of the IOU_THRESHOLD≥0.5 design: "fraction of
    the lesion captured by the patch must be ≥50%".

    For HECKTOR HN tumours, large primaries (T4 stage) and bilateral nodal disease
    can exceed 96³ @ 2mm = 19.2cm patch box; this metric will correctly drop below
    1.0 for those cases. T1/T5 lesions are usually compact enough that the broken
    formula was visibly correct in practice; HECKTOR may not be.
    """
    inter = int(seg_patch_binary.sum())
    if inter == 0 or total_lesion_voxels <= 0:
        return 0.0
    return inter / float(total_lesion_voxels)


def sample_background_patches(suv, seg_binary, n_wanted, patch_size=PATCH_SIZE_3D,
                               suv_max=BG_SUV_THRESHOLD, rng=None,
                               max_tries_factor=200):
    """Memory-efficient rejection sampling for background patch centres.

    Avoids np.argwhere allocation (T1 OOM lesson). HECKTOR HN volumes are
    smaller than whole-body PET so the OOM risk is lower, but the pattern
    is preserved for consistency with T1/T5.
    """
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
        if seg_binary[z, y, x] == 0 and suv[z, y, x] < suv_max:
            centres.append((z, y, x))
    return centres


def patch_to_mip_views(patch_3d, target_size=MIP_SIZE_2D):
    """Compute coronal/axial/sagittal MIPs of a 3D patch and resize."""
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
# ## 6. Drive mirror rsync helpers (T5 lesson banked)
#
# Mirror PATCH_DIR → DRIVE_MIRROR via rsync incremental at every checkpoint.
# Restores from Drive on session restart so a runtime delete doesn't lose work.

# %%
def mirror_patches_to_drive():
    """Incremental rsync /content/hecktor_v3_patches/ → Drive mirror."""
    src = str(PATCH_DIR) + "/"
    dst = str(DRIVE_MIRROR) + "/"
    try:
        result = subprocess.run(
            ["rsync", "-a", "--inplace", src, dst],
            capture_output=True, text=True, timeout=600,
        )
        if result.returncode == 0:
            return True
        print(f"⚠ rsync failed (code {result.returncode}): {result.stderr[:200]}")
        return False
    except subprocess.TimeoutExpired:
        print("⚠ rsync timeout (10 min)")
        return False
    except FileNotFoundError:
        print("⚠ rsync not available")
        return False


def restore_patches_from_drive():
    """On session start, copy Drive mirror back to /content if mirror has a partial manifest."""
    drive_partial = DRIVE_MIRROR / "manifest_partial.parquet"
    if not drive_partial.exists():
        return False
    print(f"Drive mirror has partial manifest at {drive_partial} — restoring to {PATCH_DIR}…")
    src = str(DRIVE_MIRROR) + "/"
    dst = str(PATCH_DIR) + "/"
    result = subprocess.run(
        ["rsync", "-a", "--inplace", src, dst],
        capture_output=True, text=True, timeout=1800,
    )
    if result.returncode == 0:
        print("✓ restore complete")
        return True
    print(f"⚠ restore failed: {result.stderr[:200]}")
    return False


# Auto-restore on notebook start (idempotent — no-op if Drive mirror empty)
restore_patches_from_drive()

# %% [markdown]
# ## 7. Per-patient processing loop
#
# **Memory + crash-resilience design** (T1 + T5 lessons banked):
# 1. Per-patient logic in `process_patient()` so locals fall out of scope at return.
# 2. Explicit `gc.collect()` after each patient.
# 3. Progressive checkpoint save every CHECKPOINT_EVERY patients.
# 4. Drive rsync at every checkpoint (T5 lesson).
# 5. Resume support that re-tries error rows.

# %%
import gc

CHECKPOINT_EVERY = 25  # save manifest_partial every N patients

PARTIAL_MANIFEST = PATCH_DIR / "manifest_partial.parquet"
PARTIAL_LOG = PATCH_DIR / "preprocessing_log_partial.csv"

# Resume support: drop error rows so they are re-tried on this run
manifest_rows = []
log_rows = []
already_processed = set()
if PARTIAL_MANIFEST.exists() and PARTIAL_LOG.exists():
    prev_manifest = pd.read_parquet(PARTIAL_MANIFEST)
    prev_log = pd.read_csv(PARTIAL_LOG)
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


def derive_centre_id(pid, ehr_value=None):
    """Centre ID resolution: prefer EHR field (str or numeric), fall back to ID prefix split."""
    if ehr_value is not None and pd.notna(ehr_value):
        s = str(ehr_value).strip()
        if s:
            return s
    return pid.split("-")[0] if "-" in pid else pid[:4]


def process_patient(pid, zf, rng):
    """Process one HECKTOR patient → returns (rows_for_manifest, log_record).

    Steps:
      1. Read PT, SEG (small file = `<PID>.nii.gz`, NOT `<PID>__SEG.nii.gz`).
      2. Resample SEG → PT grid via nearest-neighbour (D4 multi-grid finding).
      3. Resample BOTH PT and SEG-on-PT-grid to 2mm isotropic.
      4. Extract per-lesion patches (GTVp + GTVn separated by lesion_class).
      5. Extract matched-count background patches.
    """
    pt_member  = f"{DATA_ROOT}{pid}/{pid}__PT.nii.gz"
    seg_member = f"{DATA_ROOT}{pid}/{pid}.nii.gz"

    pt_img  = read_nifti_from_zip(zf, pt_member)
    seg_img = read_nifti_from_zip(zf, seg_member)

    # Step 1: SEG → PT grid (preserves discrete labels via NN)
    seg_on_pt_img = resample_to_reference(seg_img, pt_img, sitk.sitkNearestNeighbor)

    # Step 2: PT → isotropic (linear), SEG → isotropic (NN)
    pt_iso  = resample_isotropic(pt_img, SPACING, sitk.sitkLinear)
    seg_iso = resample_isotropic(seg_on_pt_img, SPACING, sitk.sitkNearestNeighbor)

    suv = sitk.GetArrayFromImage(pt_iso).astype(np.float32)
    seg = sitk.GetArrayFromImage(seg_iso).astype(np.uint8)  # values: 0, 1, 2
    seg_binary = (seg > 0).astype(np.uint8)
    suv_max = float(suv.max())
    suv_mean = float(suv.mean())
    shape_iso = str(suv.shape)

    del pt_img, seg_img, seg_on_pt_img, pt_iso, seg_iso

    # Find connected components on BINARY mask (so adjacent GTVp+GTVn merge into one
    # lesion if touching). lesion_class for each component = mode of seg label values
    # within the component (1=GTVp, 2=GTVn, or mixed if a primary touches a node).
    labels, n_lesions_raw = ndimage.label(seg_binary)
    lesion_props = measure.regionprops(labels)
    lesion_props = [p for p in lesion_props if p.area >= MIN_LESION_VOXELS]
    n_lesions_kept = len(lesion_props)

    # Per-component lesion class via majority vote on multi-label seg
    def lesion_class_for(prop):
        coords = prop.coords  # (N, 3) array of voxel indices
        vals = seg[coords[:, 0], coords[:, 1], coords[:, 2]]
        # If purely GTVp, return 1; purely GTVn, return 2; mixed → 3
        u = np.unique(vals[vals > 0])
        if len(u) == 1:
            return int(u[0])
        return 3  # mixed = primary touching nodal

    if n_lesions_kept > MAX_PATCHES_PER_PATIENT // 2:
        keep_idx = rng.choice(n_lesions_kept,
                              size=MAX_PATCHES_PER_PATIENT // 2,
                              replace=False)
        lesion_props = [lesion_props[i] for i in sorted(keep_idx)]

    patient_dir = PATCH_DIR / "patches" / pid
    patient_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    pmeta = patient_meta.get(pid, {})
    centre_id = derive_centre_id(pid, pmeta.get(centre_id_col) if centre_id_col else None)

    # Lesion patches (label = 1)
    for li, lp in enumerate(lesion_props):
        cz, cy, cx = lp.centroid
        patch_3d, origin = extract_centred_patch(suv, (cz, cy, cx))
        seg_patch, _    = extract_centred_patch(seg_binary, (cz, cy, cx))
        iou = compute_iou(seg_patch, lp.area)
        mips = patch_to_mip_views(patch_3d)
        lc = lesion_class_for(lp)

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
            "lesion_class": lc,  # 1=GTVp, 2=GTVn, 3=mixed
            "lesion_voxels": int(lp.area), "iou": float(iou),
            "patch_origin_zyx": json.dumps(list(origin)),
            "patch_centre_zyx": json.dumps([float(cz), float(cy), float(cx)]),
            "centre_id": centre_id,
            "task1_patient": bool(pmeta.get("task1_patient", False)),
            "task2_patient": bool(pmeta.get("task2_patient", False)),
            "relapse": pmeta.get("Relapse"),
            "rfs_days": pmeta.get("RFS"),
            "t_stage": pmeta.get("T-stage"),
            "n_stage": pmeta.get("N-stage"),
            "m_stage": pmeta.get("M-stage"),
            "hpv_status": pmeta.get("HPV Status"),
        })

    # Background patches (label = 0)
    n_bg = max(BG_PATCHES_PER_LESION * len(lesion_props), 1) if len(lesion_props) > 0 else 5
    bg_centres = sample_background_patches(suv, seg_binary, n_bg, rng=rng)
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
            "lesion_class": 0,
            "lesion_voxels": 0, "iou": 0.0,
            "patch_origin_zyx": json.dumps(list(origin)),
            "patch_centre_zyx": json.dumps([float(c) for c in centre]),
            "centre_id": centre_id,
            "task1_patient": bool(pmeta.get("task1_patient", False)),
            "task2_patient": bool(pmeta.get("task2_patient", False)),
            "relapse": pmeta.get("Relapse"),
            "rfs_days": pmeta.get("RFS"),
            "t_stage": pmeta.get("T-stage"),
            "n_stage": pmeta.get("N-stage"),
            "m_stage": pmeta.get("M-stage"),
            "hpv_status": pmeta.get("HPV Status"),
        })

    log_record = {
        "patient_id": pid,
        "status": "ok",
        "centre_id": centre_id,
        "n_lesions_kept": n_lesions_kept,
        "n_lesions_raw": int(n_lesions_raw),
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
    from tqdm.auto import tqdm as tqdm_auto
except ImportError:
    from tqdm import tqdm as tqdm_auto

zf = zipfile.ZipFile(ZIP_PATH)
remaining = [pid for pid in cohort_patients if pid not in already_processed]
print(f"Patients to process this session: {len(remaining)}")
print(f"Already done (resumed): {len(already_processed)}")
print(f"Checkpoint every: {CHECKPOINT_EVERY} patients")
print(f"Drive mirror: {DRIVE_MIRROR}\n")

pbar = tqdm_auto(total=len(remaining), desc="HECKTOR patients", unit="patient")

try:
    for i, pid in enumerate(remaining):
        rng = np.random.default_rng(int(abs(hash(pid)) % (2**31)))
        try:
            rows, log_record = process_patient(pid, zf, rng)
            manifest_rows.extend(rows)
            log_rows.append(log_record)
            pbar.set_postfix_str(
                f"last={pid[-10:]} les={log_record['n_lesions_kept']} "
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
            ok = mirror_patches_to_drive()
            pbar.write(f"  ✓ checkpoint @ i={i+1}: "
                       f"{len(manifest_rows)} patches, "
                       f"{len(log_rows)} log rows; "
                       f"Drive mirror {'synced' if ok else 'FAILED'}")
finally:
    pbar.close()
    zf.close()
    save_partial(manifest_rows, log_rows)
    mirror_patches_to_drive()
    print(f"\nFinal: {len(manifest_rows)} patches across {len(log_rows)} patient log rows")

# %% [markdown]
# ## 8. Final manifest + EHR join (Task labels + clinical metadata already embedded per-row in §7)

# %%
manifest = pd.DataFrame(manifest_rows)
log_df = pd.DataFrame(log_rows)

manifest.to_parquet(PATCH_DIR / "manifest.parquet", index=False)
log_df.to_csv(PATCH_DIR / "preprocessing_log.csv", index=False)

print(f"manifest.parquet rows: {len(manifest)}")
print(f"preprocessing_log.csv rows: {len(log_df)}")

# %% [markdown]
# ## 9. QC + summary

# %%
ok = log_df[log_df["status"] == "ok"]
print("=== HECKTOR 2025 v3 Preprocessing QC ===")
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

# Per-cohort breakdowns
print(f"\nT2 cohort (Task1=1):")
t2_patients = manifest[manifest["task1_patient"]]["patient_id"].nunique()
t2_lesion = ((manifest["task1_patient"]) & (manifest["label"] == 1)).sum()
t2_bg = ((manifest["task1_patient"]) & (manifest["label"] == 0)).sum()
print(f"  patients: {t2_patients}")
print(f"  lesion patches: {t2_lesion}")
print(f"  background patches: {t2_bg}")

print(f"\nT3 cohort (Task2=1):")
t3_patients = manifest[manifest["task2_patient"]]["patient_id"].nunique()
print(f"  patients: {t3_patients}")
t3_unique = manifest[manifest["task2_patient"]].drop_duplicates("patient_id")
n_event = (t3_unique["relapse"] == 1).sum()
n_total = t3_unique["relapse"].notna().sum()
if n_total > 0:
    print(f"  RFS event rate: {n_event}/{n_total} = {100*n_event/n_total:.1f}%")

# Per-centre breakdown
if "centre_id" in manifest.columns:
    print(f"\nPer-centre patient counts:")
    centre_counts = manifest.drop_duplicates("patient_id")["centre_id"].value_counts().sort_index()
    print(centre_counts.to_string())

# Lesion class distribution (for GTVp-only sensitivity feasibility)
print(f"\nLesion class distribution (label=1 patches):")
les_class = manifest[manifest["label"] == 1]["lesion_class"].value_counts().sort_index()
print(f"  1 (GTVp pure):  {les_class.get(1, 0)}")
print(f"  2 (GTVn pure):  {les_class.get(2, 0)}")
print(f"  3 (GTVp+GTVn mixed): {les_class.get(3, 0)}")

size_gb = sum(f.stat().st_size for f in PATCH_DIR.rglob("*") if f.is_file()) / 1e9
print(f"\nOutput size: {size_gb:.2f} GB")

# %% [markdown]
# ## 10. Stage output for Kaggle upload

# %%
TARBALL = Path("/content/hecktor_v3_patches.tar.gz")
import tarfile
with tarfile.open(TARBALL, "w:gz") as tar:
    tar.add(PATCH_DIR, arcname="hecktor_v3_patches")
print(f"Tarball: {TARBALL} ({TARBALL.stat().st_size / 1e9:.2f} GB)")

# %% [markdown]
# ## 11. SHA-256 + freeze metadata
#
# Hash the tarball BEFORE the Drive move (FUSE write may not be flushed at the time
# of move-return; reading back over FUSE could hash an incompletely-mirrored file).

# %%
import hashlib

def sha256_file(path, chunk=2**20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()

manifest_sha = sha256_file(PATCH_DIR / "manifest.parquet")
tarball_sha = sha256_file(TARBALL)  # hash on /content (local, fast, deterministic)

TARBALL_DRIVE = DRIVE_ROOT / "hecktor_v3_patches.tar.gz"
shutil.move(str(TARBALL), TARBALL_DRIVE)
print(f"Moved to Drive: {TARBALL_DRIVE}")

# Sanity: hash the Drive copy and confirm match (after FUSE flushes the move)
tarball_drive_sha = sha256_file(TARBALL_DRIVE)
assert tarball_sha == tarball_drive_sha, (
    f"Tarball SHA mismatch after Drive move:\n  /content: {tarball_sha}\n  Drive:   {tarball_drive_sha}"
)

with open(PATCH_DIR / "hecktor_v3_freeze_metadata.json", "w") as f:
    json.dump({
        "task": "T2+T3",
        "task_description": "HECKTOR 2025 HN tumour patch-classification (T2) + RFS prediction (T3) per A12",
        "freeze_timestamp_utc": freeze_timestamp,
        "amendment_log_ref": "A12 (osf/amendment_log.md v11, SHA d68e3a9a…)",
        "source": {
            "release": "HECKTOR 2025 Training Data Defaced ALL",
            "zip_path": str(ZIP_PATH),
            "zip_sha256": "1abcf1d96d38bb3d7b1eaf1889fa8ddd688f14b70876a1c7cf0cd7482d076df2",
        },
        "patches": {
            "n_patients_ok": int((log_df["status"] == "ok").sum()),
            "n_patches_total": int(len(manifest)),
            "n_lesion": int((manifest["label"] == 1).sum()),
            "n_background": int((manifest["label"] == 0).sum()),
            "n_t2_patients": int(t2_patients),
            "n_t3_patients": int(t3_patients),
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
            "hecktor_v3_patches.tar.gz": tarball_sha,
        },
    }, f, indent=2)

print(f"manifest.parquet SHA-256: {manifest_sha}")
print(f"hecktor_v3_patches.tar.gz SHA-256: {tarball_sha}")

# %% [markdown]
# ## 12. Output schema (for downstream `hecktor_02_embeddings.py`)
#
# **manifest.parquet** — one row per patch:
#   - `patient_id`        str   — HECKTOR 2025 patient identifier
#   - `patch_id`          str   — `{patient_id}_lesion_{nnn}` or `{patient_id}_bg_{nnn}`
#   - `label`             int   — 1 = lesion patch, 0 = background
#   - `lesion_index`      int   — connected-component index, -1 for background
#   - `lesion_class`      int   — 0=bg, 1=GTVp, 2=GTVn, 3=GTVp+GTVn mixed
#   - `lesion_voxels`     int   — total voxels in the lesion component
#   - `iou`               float — IoU of patch ROI vs lesion (0 for background)
#   - `patch_origin_zyx`  str   — JSON list of (z, y, x) corner of patch in resampled volume
#   - `patch_centre_zyx`  str   — JSON list of (z, y, x) lesion centroid (or bg sample point)
#   - `centre_id`         str   — CHUM/CHUP/CHUS/CHUV/MDA/USZ/HMR (Mondrian stratifier)
#   - `task1_patient`     bool  — patient is in T2 (segmentation/patch-classification) cohort
#   - `task2_patient`     bool  — patient is in T3 (RFS prediction) cohort
#   - `relapse`           float — Relapse label (0/1, NaN if not in T2 cohort)
#   - `rfs_days`          float — RFS time in days (T3 only, NaN otherwise)
#   - `t_stage`, `n_stage`, `m_stage`, `hpv_status` — clinical covariates
#
# **patches/{patient_id}/{patch_id}.npz** — per patch:
#   - `patch_3d`     float16, shape (96, 96, 96)
#   - `mip_coronal`  float16, shape (224, 224)
#   - `mip_axial`    float16, shape (224, 224)
#   - `mip_sagittal` float16, shape (224, 224)
#
# **hecktor_v3_freeze_metadata.json** — provenance + SHA-256
# **preprocessing_log.csv** — per-patient processing status
#
# Downstream `hecktor_02_embeddings.py` (to be drafted, clones t5_02_embeddings.py
# with HECKTOR-specific paths):
#   - 3D FMs (FMCIB, CT-FM, random_init): consume `patch_3d` directly.
#   - 2D FMs (BiomedCLIP, DINOv2, RAD-DINO): consume axial/coronal/sagittal MIPs.
#   - Per-patch embeddings → patch-level parquet emitted as
#     `pet-fm-bench-hecktor-embeddings-v3`.
#
# probe_analysis.py v6:
#   - T2 dispatch (filter manifest task1_patient==True) → AUROC patch-classification
#     per A12a (analogous to T1).
#   - T3 dispatch (filter manifest task2_patient==True) → CoxPH on patient-level
#     pooled embeddings (mean-pool over patient's patches) per registration §5.4
#     5-fold nested CV.
