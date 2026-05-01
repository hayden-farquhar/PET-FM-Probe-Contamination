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
# # HECKTOR Notebook 2/2: Extract FM Embeddings (HECKTOR 2025 — T2 + T3)
#
# **PET-FM-Bench** | DOI: [10.17605/OSF.IO/DQ2JA](https://doi.org/10.17605/OSF.IO/DQ2JA)
#
# **Runtime:** GPU T4 | **Internet:** On | **Time:** ~1.5-2.5 hours
#
# **Input dataset:** `pet-fm-bench-hecktor-patches-v3` (output of
# `hecktor_01_preprocess_v3.py`).
#
# **Tasks served:** T2 (HN tumour patch-classification, AUROC per A12a) and T3 (RFS
# prediction, CoxPH per A12c). Single embedding extraction; downstream
# `probe_analysis.py` v6 dispatches T2 (filter `task1_patient == True`, AUROC) and
# T3 (filter `task2_patient == True`, CoxPH on patient-pooled embeddings) from the
# same dataset.
#
# **STATUS: DRAFT — code review only. Not yet executed. Awaits:**
# 1. `hecktor_01_preprocess_v3.py` Colab run completion + tarball Drive copy.
# 2. `pet-fm-bench-hecktor-patches-v3` Kaggle dataset upload (~6-8 GB tarball
#    + 3 sidecar files: manifest.parquet, preprocessing_log.csv,
#    hecktor_v3_freeze_metadata.json).
#
# ## Schema parity with T1/T5 (intentional)
#
# Output FM parquets use the same column convention as T1/T5: `patient_id, patch_id,
# view, layer, d0000, d0001, ..., d{N-1}`. View/layer conventions are identical to
# T1/T5. This means `probe_analysis.py` v6 can join HECKTOR embeddings with
# task_splits.parquet on `(patient_id, patch_id, view, layer)` without any per-task
# code-paths in the probe.
#
# **HECKTOR-specific columns in `hecktor_labels.parquet`** (vs T1/T5 per-patch labels):
#   - Standard: `patient_id, patch_id, label, lesion_index, iou`
#   - HECKTOR-only: `lesion_class` (0=bg, 1=GTVp pure, 2=GTVn pure, 3=mixed),
#     `task1_patient` (T2 cohort flag), `task2_patient` (T3 cohort flag),
#     `relapse` (binary, T3 endpoint), `rfs_days` (float, T3 time-to-event),
#     `centre_id` (Mondrian stratifier per A12b), `hpv_status, t_stage, n_stage,
#     m_stage` (clinical covariates).
#   - Note: `cancer_type` and `study_date` columns absent (HECKTOR is HN-only +
#     single-time-point per patient).

# %% [markdown]
# ## 1. Install dependencies

# %%
# !pip install -q open_clip_torch
# !pip install -q lighter-zoo
# # FMCIB loaded manually via Zenodo (pip package broken on Kaggle).

# %% [markdown]
# ## 2. Setup

# %%
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

try:
    from kaggle_secrets import UserSecretsClient
    HF_TOKEN = UserSecretsClient().get_secret("HF_TOKEN")
    print("HF token loaded")
except Exception:
    HF_TOKEN = None

# %% [markdown]
# ## 3. Locate input data
#
# Three layouts to handle (same logic as t1_02_embeddings.py + t5_02_embeddings.py
# — Kaggle CLI auto-extracts .tar.gz uploads despite docs saying otherwise, AND the
# tarball was created with arcname="hecktor_v3_patches", causing two levels of
# nesting). The candidate list checks all three.

# %%
V3_CANDIDATES = list(Path("/kaggle/input").rglob("pet-fm-bench-hecktor-patches-v3"))
INPUT_CANDIDATES = V3_CANDIDATES or list(Path("/kaggle/input").glob("pet-fm-bench-hecktor*"))
if not INPUT_CANDIDATES:
    INPUT_CANDIDATES = list(Path("/kaggle/input").iterdir())

print("Available inputs:")
for p in INPUT_CANDIDATES:
    print(f"  {p.name}")

INPUT_DIR = INPUT_CANDIDATES[0] if INPUT_CANDIDATES else Path("/kaggle/input/pet-fm-bench-hecktor-patches-v3")

_PATCHES_CANDIDATES = [
    INPUT_DIR / "hecktor_v3_patches" / "hecktor_v3_patches" / "patches",  # double nest (Kaggle auto-extract)
    INPUT_DIR / "hecktor_v3_patches" / "patches",                          # single nest
    INPUT_DIR / "patches",                                                  # direct directory upload
]
_PATCHES_DIR = next((c for c in _PATCHES_CANDIDATES if c.is_dir()), None)
PRE_EXTRACTED = _PATCHES_DIR is not None
TARBALL = next(INPUT_DIR.rglob("hecktor_v3_patches.tar.gz"), None)

if PRE_EXTRACTED:
    PATCH_DIR = _PATCHES_DIR.parent
    manifest_path = (PATCH_DIR / "manifest.parquet"
                     if (PATCH_DIR / "manifest.parquet").exists()
                     else INPUT_DIR / "manifest.parquet")
    print(f"\nKaggle pre-extracted layout detected. PATCH_DIR = {PATCH_DIR}")
elif TARBALL is not None:
    EXTRACT_DIR = Path("/tmp/hecktor_v3_extracted")
    if not (EXTRACT_DIR / "hecktor_v3_patches" / "patches").exists():
        print(f"\nExtracting {TARBALL.name} ({TARBALL.stat().st_size/1e9:.2f} GB) → {EXTRACT_DIR}/")
        EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
        import tarfile
        with tarfile.open(TARBALL, "r:gz") as tar:
            tar.extractall(EXTRACT_DIR)
        print("Extraction complete.")
    PATCH_DIR = EXTRACT_DIR / "hecktor_v3_patches"
    manifest_path = PATCH_DIR / "manifest.parquet"
else:
    manifest_path = list(INPUT_DIR.rglob("manifest.parquet"))[0]
    PATCH_DIR = manifest_path.parent

manifest = pd.read_parquet(manifest_path)
print(f"\nManifest (raw): {len(manifest)} patches across {manifest['patient_id'].nunique()} patients")
print(f"Label distribution: {manifest['label'].value_counts().to_dict()}")
print(f"\nT2 cohort (task1_patient): "
      f"{manifest[manifest['task1_patient']]['patient_id'].nunique()} patients, "
      f"{(manifest['task1_patient']).sum()} patches")
print(f"T3 cohort (task2_patient): "
      f"{manifest[manifest['task2_patient']]['patient_id'].nunique()} patients, "
      f"{(manifest['task2_patient']).sum()} patches")
if "centre_id" in manifest.columns:
    print(f"\nPer-centre distribution:")
    print(manifest.drop_duplicates("patient_id")["centre_id"].value_counts().sort_index().to_string())
if "lesion_class" in manifest.columns:
    print(f"\nLesion class distribution (label=1 patches):")
    lc = manifest[manifest["label"] == 1]["lesion_class"].value_counts().sort_index().to_dict()
    print(f"  bg={lc.get(0,0)} GTVp={lc.get(1,0)} GTVn={lc.get(2,0)} mixed={lc.get(3,0)}")

def _has_patch_file(r):
    p = PATCH_DIR / "patches" / r["patient_id"] / f"{r['patch_id']}.npz"
    return p.exists()

mask = manifest.apply(_has_patch_file, axis=1)
dropped_n = (~mask).sum()
manifest = manifest[mask].reset_index(drop=True)
print(f"\nManifest (filtered to on-disk patches): {len(manifest)} patches")
print(f"Dropped {dropped_n} manifest rows missing patch files")

EMBED_DIR = Path("/kaggle/working/embeddings")
EMBED_DIR.mkdir(parents=True, exist_ok=True)

# Save HECKTOR per-patch labels for downstream T2/T3 probes
label_cols = ["patient_id", "patch_id", "label", "lesion_index", "iou"]
for opt_col in ("lesion_class", "task1_patient", "task2_patient",
                "relapse", "rfs_days", "centre_id",
                "hpv_status", "t_stage", "n_stage", "m_stage"):
    if opt_col in manifest.columns:
        label_cols.append(opt_col)
manifest[label_cols].to_parquet(EMBED_DIR / "hecktor_labels.parquet", index=False)
print(f"Saved hecktor_labels.parquet with columns: {label_cols}")

# %% [markdown]
# ## 4. Per-patch data loading helpers (identical to T1/T5)

# %%
def load_patch(patient_id, patch_id):
    path = PATCH_DIR / "patches" / patient_id / f"{patch_id}.npz"
    data = np.load(path)
    return {
        "patch_3d":     data["patch_3d"].astype(np.float32),
        "mip_coronal":  data["mip_coronal"].astype(np.float32),
        "mip_axial":    data["mip_axial"].astype(np.float32),
        "mip_sagittal": data["mip_sagittal"].astype(np.float32),
    }


def mip_to_rgb_tensor(mip_2d):
    mip_2d = np.nan_to_num(mip_2d, nan=0.0, posinf=0.0, neginf=0.0)
    vmin, vmax = float(mip_2d.min()), float(mip_2d.max())
    if vmax > vmin:
        normed = (mip_2d - vmin) / (vmax - vmin)
    else:
        normed = np.zeros_like(mip_2d)
    stacked = np.stack([normed, normed, normed])
    return torch.tensor(stacked, dtype=torch.float32).unsqueeze(0)


# %% [markdown]
# ## 5. FM definitions (identical FM runners to t1/t5_02_embeddings.py)

# %%
# ═══════════════════════════════════════════════════════════════════════
# DINOv2 (2D, ViT-B/14, 768-dim)
# ═══════════════════════════════════════════════════════════════════════

def run_dinov2(manifest):
    from transformers import AutoModel
    model = AutoModel.from_pretrained("facebook/dinov2-base",
                                      revision="f9e44c8").to(DEVICE).eval()
    rows = []
    for _, r in tqdm(manifest.iterrows(), total=len(manifest), desc="DINOv2"):
        npz = load_patch(r["patient_id"], r["patch_id"])
        for view_name, view_arr in [("coronal", npz["mip_coronal"]),
                                     ("axial", npz["mip_axial"]),
                                     ("sagittal", npz["mip_sagittal"])]:
            inp = mip_to_rgb_tensor(view_arr).to(DEVICE)
            with torch.no_grad():
                out = model(inp, output_hidden_states=True)
            for layer_name, layer_idx in [("layer03", 3), ("layer06", 6),
                                           ("layer11", 11), ("cls", -1)]:
                if layer_name == "cls":
                    emb = out.last_hidden_state[0, 0].cpu().numpy()
                else:
                    emb = out.hidden_states[layer_idx][0, 0].cpu().numpy()
                rows.append({"patient_id": r["patient_id"], "patch_id": r["patch_id"],
                             "view": view_name, "layer": layer_name, "embedding": emb})
    del model; torch.cuda.empty_cache()
    return rows


# ═══════════════════════════════════════════════════════════════════════
# RAD-DINO (2D, ViT-B/14, 768-dim)
# ═══════════════════════════════════════════════════════════════════════

def run_rad_dino(manifest):
    from transformers import AutoModel
    model = AutoModel.from_pretrained("microsoft/rad-dino",
                                      revision="2ec9ca0").to(DEVICE).eval()
    rows = []
    for _, r in tqdm(manifest.iterrows(), total=len(manifest), desc="RAD-DINO"):
        npz = load_patch(r["patient_id"], r["patch_id"])
        for view_name, view_arr in [("coronal", npz["mip_coronal"]),
                                     ("axial", npz["mip_axial"]),
                                     ("sagittal", npz["mip_sagittal"])]:
            inp = mip_to_rgb_tensor(view_arr).to(DEVICE)
            with torch.no_grad():
                out = model(inp, output_hidden_states=True)
            for layer_name, layer_idx in [("layer03", 3), ("layer06", 6),
                                           ("layer11", 11), ("cls", -1)]:
                if layer_name == "cls":
                    emb = out.last_hidden_state[0, 0].cpu().numpy()
                else:
                    emb = out.hidden_states[layer_idx][0, 0].cpu().numpy()
                rows.append({"patient_id": r["patient_id"], "patch_id": r["patch_id"],
                             "view": view_name, "layer": layer_name, "embedding": emb})
    del model; torch.cuda.empty_cache()
    return rows


# ═══════════════════════════════════════════════════════════════════════
# BiomedCLIP (2D, ViT-B/16, 512-dim)
# ═══════════════════════════════════════════════════════════════════════

def run_biomedclip(manifest):
    import open_clip
    model, _, _ = open_clip.create_model_and_transforms(
        "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    )
    model = model.to(DEVICE).eval()
    rows = []
    for _, r in tqdm(manifest.iterrows(), total=len(manifest), desc="BiomedCLIP"):
        npz = load_patch(r["patient_id"], r["patch_id"])
        for view_name, view_arr in [("coronal", npz["mip_coronal"]),
                                     ("axial", npz["mip_axial"]),
                                     ("sagittal", npz["mip_sagittal"])]:
            inp = mip_to_rgb_tensor(view_arr).to(DEVICE)
            with torch.no_grad():
                emb = model.encode_image(inp)[0].cpu().numpy()
            rows.append({"patient_id": r["patient_id"], "patch_id": r["patch_id"],
                         "view": view_name, "layer": "cls", "embedding": emb})
    del model; torch.cuda.empty_cache()
    return rows


# ═══════════════════════════════════════════════════════════════════════
# Random Init Control (2D, ViT-B/14, 768-dim)
# ═══════════════════════════════════════════════════════════════════════

def run_random_init(manifest):
    from transformers import AutoConfig, AutoModel
    config = AutoConfig.from_pretrained("facebook/dinov2-base")
    model = AutoModel.from_config(config).to(DEVICE).eval()
    rows = []
    for _, r in tqdm(manifest.iterrows(), total=len(manifest), desc="RandomInit"):
        npz = load_patch(r["patient_id"], r["patch_id"])
        for view_name, view_arr in [("coronal", npz["mip_coronal"]),
                                     ("axial", npz["mip_axial"]),
                                     ("sagittal", npz["mip_sagittal"])]:
            inp = mip_to_rgb_tensor(view_arr).to(DEVICE)
            with torch.no_grad():
                out = model(inp)
            emb = out.last_hidden_state[0, 0].cpu().numpy()
            rows.append({"patient_id": r["patient_id"], "patch_id": r["patch_id"],
                         "view": view_name, "layer": "cls", "embedding": emb})
    del model; torch.cuda.empty_cache()
    return rows


# ═══════════════════════════════════════════════════════════════════════
# FMCIB (3D, ResNet-50-2x, 4096-dim) — manual loader (proven pattern)
# ═══════════════════════════════════════════════════════════════════════

class _Bottleneck3D(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, mid_ch, 1, bias=False)
        self.bn1 = nn.BatchNorm3d(mid_ch)
        self.conv2 = nn.Conv3d(mid_ch, mid_ch, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(mid_ch)
        self.conv3 = nn.Conv3d(mid_ch, out_ch, 1, bias=False)
        self.bn3 = nn.BatchNorm3d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        if in_ch != out_ch or stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, 1, stride=stride, bias=True),
                nn.BatchNorm3d(out_ch),
            )

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


class FMCIBEncoder(nn.Module):
    def __init__(self, state_dict):
        super().__init__()
        self.conv1 = nn.Conv3d(1, 128, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(128)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(128, 128, 512, 3, stride=1)
        self.layer2 = self._make_layer(512, 256, 1024, 4, stride=2)
        self.layer3 = self._make_layer(1024, 512, 2048, 6, stride=2)
        self.layer4 = self._make_layer(2048, 1024, 4096, 3, stride=2)
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.load_state_dict(state_dict, strict=True)

    def _make_layer(self, in_ch, mid_ch, out_ch, n, stride):
        layers = [_Bottleneck3D(in_ch, mid_ch, out_ch, stride=stride)]
        for _ in range(1, n):
            layers.append(_Bottleneck3D(out_ch, mid_ch, out_ch, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return x.flatten(1)


def run_fmcib(manifest, batch_size=4):
    import requests as req

    weight_path = "/tmp/fmcib_weights.torch"
    if not os.path.exists(weight_path) or os.path.getsize(weight_path) < 1e6:
        print("  Downloading FMCIB weights (738 MB)...")
        url = "https://zenodo.org/api/records/10528450/files/model_weights.torch/content"
        r = req.get(url, stream=True)
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(weight_path, "wb") as f, tqdm(total=total, unit="B", unit_scale=True) as pbar:
            for chunk in r.iter_content(8 * 1024 * 1024):
                f.write(chunk)
                pbar.update(len(chunk))

    trunk = torch.load(weight_path, map_location="cpu", weights_only=False)["trunk_state_dict"]
    model = FMCIBEncoder(trunk).to(DEVICE).eval()

    rows = []
    n = len(manifest)
    pbar = tqdm(total=n, desc="FMCIB")
    for batch_start in range(0, n, batch_size):
        batch_rows = manifest.iloc[batch_start:batch_start + batch_size]
        batch_patches = []
        batch_meta = []
        for _, r in batch_rows.iterrows():
            npz = load_patch(r["patient_id"], r["patch_id"])
            batch_patches.append(npz["patch_3d"])
            batch_meta.append((r["patient_id"], r["patch_id"]))
        tensor = torch.tensor(np.stack(batch_patches), dtype=torch.float32).unsqueeze(1).to(DEVICE)
        with torch.no_grad():
            out = model(tensor).cpu().numpy()
        for (pid, pat_id), emb in zip(batch_meta, out):
            rows.append({"patient_id": pid, "patch_id": pat_id,
                         "view": "volume", "layer": "pool", "embedding": emb})
        pbar.update(len(batch_rows))
    pbar.close()
    del model; torch.cuda.empty_cache()
    return rows


# ═══════════════════════════════════════════════════════════════════════
# CT-FM (3D, SegResEncoder, 512-dim)
# ═══════════════════════════════════════════════════════════════════════

def run_ct_fm(manifest, batch_size=4):
    from lighter_zoo import SegResEncoder
    model = SegResEncoder.from_pretrained(
        "project-lighter/ct_fm_feature_extractor"
    ).to(DEVICE).eval()

    rows = []
    n = len(manifest)
    pbar = tqdm(total=n, desc="CT-FM")
    for batch_start in range(0, n, batch_size):
        batch_rows = manifest.iloc[batch_start:batch_start + batch_size]
        batch_patches = []
        batch_meta = []
        for _, r in batch_rows.iterrows():
            npz = load_patch(r["patient_id"], r["patch_id"])
            batch_patches.append(npz["patch_3d"])
            batch_meta.append((r["patient_id"], r["patch_id"]))
        tensor = torch.tensor(np.stack(batch_patches), dtype=torch.float32).unsqueeze(1).to(DEVICE)
        with torch.no_grad():
            out = model(tensor)[-1]
            pooled = torch.nn.functional.adaptive_avg_pool3d(out, 1)
            pooled = pooled.squeeze(-1).squeeze(-1).squeeze(-1).cpu().numpy()
        for (pid, pat_id), emb in zip(batch_meta, pooled):
            rows.append({"patient_id": pid, "patch_id": pat_id,
                         "view": "volume", "layer": "pool", "embedding": emb})
        pbar.update(len(batch_rows))
    pbar.close()
    del model; torch.cuda.empty_cache()
    return rows


# %% [markdown]
# ## 6. Run all FMs

# %%
FM_RUNNERS = [
    ("dinov2",      run_dinov2),
    ("rad_dino",    run_rad_dino),
    ("biomedclip",  run_biomedclip),
    ("random_init", run_random_init),
    ("fmcib",       run_fmcib),
    ("ct_fm",       run_ct_fm),
]

for fm_name, runner_fn in FM_RUNNERS:
    print(f"\n{'='*60}")
    print(f"  {fm_name}")
    print(f"{'='*60}")

    try:
        rows = runner_fn(manifest)
        if not rows:
            print(f"  No embeddings — skipping")
            continue

        embed_dim = len(rows[0]["embedding"])
        records = []
        for row in rows:
            rec = {"patient_id": row["patient_id"],
                   "patch_id": row["patch_id"],
                   "view": row["view"],
                   "layer": row["layer"]}
            for j in range(len(row["embedding"])):
                rec[f"d{j:04d}"] = float(row["embedding"][j])
            records.append(rec)

        df = pd.DataFrame(records)
        out_path = EMBED_DIR / f"{fm_name}.parquet"
        df.to_parquet(out_path, index=False)
        size_mb = out_path.stat().st_size / 1e6
        print(f"  Saved: {out_path.name} ({size_mb:.1f} MB, {len(df)} rows, {embed_dim}-dim)")

    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback; traceback.print_exc()

    torch.cuda.empty_cache()

# %% [markdown]
# ## 7. Summary

# %%
print("\n" + "=" * 60)
print("HECKTOR EMBEDDING EXTRACTION COMPLETE")
print("=" * 60)

total_mb = 0
for f in sorted(EMBED_DIR.glob("*.parquet")):
    df = pd.read_parquet(f)
    mb = f.stat().st_size / 1e6
    total_mb += mb
    print(f"\n{f.stem}:")
    print(f"  patients: {df['patient_id'].nunique()}")
    print(f"  patches:  {df['patch_id'].nunique()}")
    if "view" not in df.columns:
        print(f"  size:     {mb:.1f} MB  (labels parquet, no embedding cols)")
        continue
    dim = sum(1 for c in df.columns if c.startswith("d"))
    print(f"  views:    {sorted(df['view'].unique().tolist())}")
    print(f"  layers:   {sorted(df['layer'].unique().tolist())}")
    print(f"  dim:      {dim}")
    print(f"  size:     {mb:.1f} MB")

print(f"\nTotal output: {total_mb:.1f} MB")

# %% [markdown]
# ## 8. Save
#
# **"Save & Run All"** on GPU T4. Then save output as Kaggle Dataset:
# `pet-fm-bench-hecktor-embeddings-v3` (NEW dataset).
#
# ## 9. Schema (for downstream `probe_analysis.py` v6 T2/T3 dispatch)
#
# Each FM produces one parquet `{fm_name}.parquet` with columns:
#   - `patient_id, patch_id, view, layer, d0000, d0001, ..., d{N-1}`
# Schema is bit-identical to T1/T5 embeddings parquets.
#
# Plus `hecktor_labels.parquet` carrying:
#   - Standard: `patient_id, patch_id, label, lesion_index, iou`
#   - HECKTOR-specific: `lesion_class, task1_patient, task2_patient, relapse, rfs_days,
#     centre_id, hpv_status, t_stage, n_stage, m_stage`
#
# `probe_analysis.py` v6 dispatch:
#   - **T2 dispatch** (filter `task1_patient == True`): per-patch lesion-classification
#     AUROC with patient-level GroupKFold k=5, patient-clustered bootstrap CI, IBSI
#     radiomics comparator (per A8 MIRP). Per-centre AUROC supplementary
#     (Mondrian per A12b).
#   - **T3 dispatch** (filter `task2_patient == True`): patient-level CoxPH on mean-pooled
#     patient embeddings (mean over the patient's lesion patches), 5-fold nested CV per
#     registration §5.4, c-index + bootstrap CI. HPV-stratified sensitivity per A12c.
#   - **GTVp-only sensitivity**: T2 secondary filter `lesion_class == 1` → AUROC on
#     primary tumours only (excludes nodal lesions); per A12a sensitivity.
