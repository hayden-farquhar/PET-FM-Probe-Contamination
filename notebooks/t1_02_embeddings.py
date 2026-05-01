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
# # T1 Notebook 2/2: Extract FM Embeddings (AutoPET-I FDG Lesion-Patch Classification)
#
# **PET-FM-Bench** | DOI: [10.17605/OSF.IO/DQ2JA](https://doi.org/10.17605/OSF.IO/DQ2JA)
#
# **Runtime:** GPU T4 | **Internet:** On | **Time:** ~2-3 hours
#
# **Input dataset:** `pet-fm-bench-t1-patches-v3` (output of `t1_01_preprocess_v3.py`).
#
# **Task:** T1 — FDG lesion-patch binary classification (registration H1).
# Per-patch lesion vs background. Patches are pre-extracted lesion-centred + matched
# background controls; this notebook produces per-patch FM embeddings ready for the
# probe stage.
#
# **STATUS: SKELETON — code review only. Not yet executed. Will run on Kaggle once
# `pet-fm-bench-t1-patches-v3` is uploaded.**
#
# ## Critical schema deviation from existing v3 task pattern
#
# Other v3 tasks (T4/T6/T7/T8/T9) emit embeddings at the **patient** level (3D FMs
# mean-pool across patches; 2D FMs use whole-body MIPs). T1 emits embeddings at the
# **patch** level — one row per (patient_id, patch_id, fm, view, layer). Each patch
# is an independent sample for the lesion-patch binary classifier per registration H1.
# probe_analysis.py v5 dispatches T1 through `run_t1_lesion_patch_probe()` which uses
# GroupKFold on patient_id to prevent split leakage.

# %% [markdown]
# ## 1. Install dependencies

# %%
# !pip install -q open_clip_torch
# !pip install -q lighter-zoo
# # FMCIB loaded manually via Zenodo (pip package broken on Kaggle, see notebook 9 in (companion project)).

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

# %%
V3_CANDIDATES = list(Path("/kaggle/input").rglob("pet-fm-bench-t1-patches-v3"))
INPUT_CANDIDATES = V3_CANDIDATES or list(Path("/kaggle/input").glob("pet-fm-bench-t1*"))
if not INPUT_CANDIDATES:
    INPUT_CANDIDATES = list(Path("/kaggle/input").iterdir())

print("Available inputs:")
for p in INPUT_CANDIDATES:
    print(f"  {p.name}")

INPUT_DIR = INPUT_CANDIDATES[0] if INPUT_CANDIDATES else Path("/kaggle/input/pet-fm-bench-t1-patches-v3")

# T1 patches were uploaded to Kaggle as `t1_v3_patches.tar.gz` alongside
# manifest.parquet + log + freeze metadata. Three layouts to handle on Kaggle:
#   (1) **Kaggle pre-extracted** the tarball into a `t1_v3_patches/` subdir
#       (observed 2026-04-28 — Kaggle CLI ingest now auto-unpacks .tar.gz
#       despite the docs saying otherwise). PATCH_DIR points one level deeper.
#       Note: Kaggle's auto-extraction wraps the tarball contents in a directory
#       named after the tarball stem AND the tarball itself was created with
#       `arcname="t1_v3_patches"`, so we get TWO `t1_v3_patches/` levels:
#       INPUT_DIR/t1_v3_patches/t1_v3_patches/patches/...
#       The patches-dir search below handles both single- and double-nesting.
#   (2) **Tarball still .tar.gz** (unextracted) — we extract to /tmp ourselves.
#   (3) **Directory upload** (patches/ directly under INPUT_DIR) — fallback.

# Try common layouts in order of likelihood
_PATCHES_CANDIDATES = [
    INPUT_DIR / "t1_v3_patches" / "t1_v3_patches" / "patches",  # double nest (Kaggle CLI auto-extract, observed)
    INPUT_DIR / "t1_v3_patches" / "patches",                    # single nest (arcname="")
    INPUT_DIR / "patches",                                       # direct directory upload
]
_PATCHES_DIR = next((c for c in _PATCHES_CANDIDATES if c.is_dir()), None)
PRE_EXTRACTED = _PATCHES_DIR is not None
TARBALL = next(INPUT_DIR.rglob("t1_v3_patches.tar.gz"), None)

if PRE_EXTRACTED:
    PATCH_DIR = _PATCHES_DIR.parent
    # Manifest may be alongside patches/ (from tarball arcname) OR at INPUT_DIR
    # root (side-by-side upload) — prefer the one alongside patches/ since it
    # travelled with the patches and is guaranteed to match.
    if (PATCH_DIR / "manifest.parquet").exists():
        manifest_path = PATCH_DIR / "manifest.parquet"
    else:
        manifest_path = INPUT_DIR / "manifest.parquet"
    print(f"\nKaggle pre-extracted layout detected. PATCH_DIR = {PATCH_DIR}")
elif TARBALL is not None:
    EXTRACT_DIR = Path("/tmp/t1_v3_extracted")
    if not (EXTRACT_DIR / "t1_v3_patches" / "patches").exists():
        print(f"\nExtracting {TARBALL.name} ({TARBALL.stat().st_size/1e9:.2f} GB) → {EXTRACT_DIR}/")
        EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
        import tarfile
        with tarfile.open(TARBALL, "r:gz") as tar:
            tar.extractall(EXTRACT_DIR)
        print("Extraction complete.")
    else:
        print(f"\nTarball already extracted at {EXTRACT_DIR}/t1_v3_patches/")
    PATCH_DIR = EXTRACT_DIR / "t1_v3_patches"
    manifest_path = PATCH_DIR / "manifest.parquet"
else:
    # Final fallback: directory-upload scenario (patches/ directly under INPUT_DIR)
    manifest_path = list(INPUT_DIR.rglob("manifest.parquet"))[0]
    PATCH_DIR = manifest_path.parent

manifest = pd.read_parquet(manifest_path)
print(f"\nManifest (raw): {len(manifest)} patches across {manifest['patient_id'].nunique()} patients")
print(f"Label distribution:")
print(manifest["label"].value_counts().to_dict())
print(f"Cancer-type distribution (patient-level):")
print(manifest.drop_duplicates("patient_id")["cancer_type"].value_counts(dropna=False).to_dict())

# Filter to patches with files actually on disk (defensive — patch parquet should match files 1:1)
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

# Save (patient_id, patch_id, label, cancer_type) mapping for downstream probe
manifest[["patient_id", "patch_id", "label", "cancer_type",
          "lesion_index", "iou", "study_date"]].to_parquet(
    EMBED_DIR / "t1_labels.parquet", index=False
)

# %% [markdown]
# ## 4. Per-patch data loading helpers

# %%
def load_patch(patient_id, patch_id):
    """Load a single patch's npz. Returns dict with patch_3d + 3 MIP views."""
    path = PATCH_DIR / "patches" / patient_id / f"{patch_id}.npz"
    data = np.load(path)
    return {
        "patch_3d":     data["patch_3d"].astype(np.float32),
        "mip_coronal":  data["mip_coronal"].astype(np.float32),
        "mip_axial":    data["mip_axial"].astype(np.float32),
        "mip_sagittal": data["mip_sagittal"].astype(np.float32),
    }


def mip_to_rgb_tensor(mip_2d):
    """2D MIP → 3-channel min-max-normalised tensor for 2D ViT FMs.
    Same convention as existing tX_02_embeddings.py."""
    mip_2d = np.nan_to_num(mip_2d, nan=0.0, posinf=0.0, neginf=0.0)
    vmin, vmax = float(mip_2d.min()), float(mip_2d.max())
    if vmax > vmin:
        normed = (mip_2d - vmin) / (vmax - vmin)
    else:
        normed = np.zeros_like(mip_2d)
    stacked = np.stack([normed, normed, normed])
    return torch.tensor(stacked, dtype=torch.float32).unsqueeze(0)


# %% [markdown]
# ## 5. FM definitions
#
# All loading code identical to existing `tX_02_embeddings.py` v3 notebooks (proven
# pattern across T4/T6/T7/T8/T9). The only difference is per-patch iteration: each
# manifest row produces one or more (patch_id, view, layer) embedding rows.
#
# **3D FMs (FMCIB, CT-FM, random_init-3D)** consume `patch_3d` directly — no patient-
# level mean-pool, since each patch is the sample.
#
# **2D FMs (DINOv2, RAD-DINO, BiomedCLIP)** consume per-patch MIP views (coronal /
# axial / sagittal of the 96³ patch volume) instead of whole-body MIPs.

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
# Random Init Control (2D, ViT-B/14, 768-dim) — single-seed, primary entry
# Multi-seed (N=10) re-extracted by 08_random_init_multiseed.py (amendment A3)
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
    # Batched forward across consecutive manifest rows for GPU efficiency
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
print("T1 EMBEDDING EXTRACTION COMPLETE")
print("=" * 60)

total_mb = 0
for f in sorted(EMBED_DIR.glob("*.parquet")):
    df = pd.read_parquet(f)
    mb = f.stat().st_size / 1e6
    total_mb += mb
    print(f"\n{f.stem}:")
    print(f"  patients: {df['patient_id'].nunique()}")
    print(f"  patches:  {df['patch_id'].nunique()}")
    # Skip view/layer/dim print for non-embedding parquets (e.g. t1_labels.parquet
    # which has only metadata columns).
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
# `pet-fm-bench-t1-embeddings-v3` (do NOT version the existing v1 if it ever
# existed — start a NEW dataset to preserve audit trail).
#
# Dataset URL: `https://www.kaggle.com/datasets/<your-user>/pet-fm-bench-t1-embeddings-v3`
#
# ## 9. Schema (for downstream `probe_analysis.py` v5)
#
# Each FM produces one parquet `{fm_name}.parquet` with columns:
#   - `patient_id, patch_id, view, layer, d0000, d0001, ..., d{N-1}`
# where N = embedding dimension (DINOv2/RAD-DINO 768, BiomedCLIP 512, CT-FM 512,
# FMCIB 4096, random_init 768).
#
# View/layer convention:
#   - 2D FMs: view ∈ {coronal, axial, sagittal}, layer ∈ {layer03, layer06, layer11, cls}
#     (BiomedCLIP only has cls; random_init only has cls).
#   - 3D FMs: view = "volume", layer = "pool".
#
# Plus `t1_labels.parquet` carrying (patient_id, patch_id, label, cancer_type,
# lesion_index, iou, study_date) — the per-patch ground-truth used by
# `probe_analysis.py` v5 `run_t1_lesion_patch_probe()`. The probe joins embeddings
# on (patient_id, patch_id), trains LogisticRegression with patient-level
# GroupKFold to prevent split leakage, reports patch-level AUROC + AUPRC + Brier
# + ECE + DeLong-test against IBSI radiomics baseline (registration H1: ≥3pp lift).
