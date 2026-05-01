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
# # T8 Notebook 2/2: Extract FM Embeddings (Lung-PET-CT-Dx)
#
# **PET-FM-Bench** | DOI: [10.17605/OSF.IO/DQ2JA](https://doi.org/10.17605/OSF.IO/DQ2JA)
#
# **Runtime:** GPU T4 | **Internet:** On | **Time:** ~1 hour
#
# **Input dataset:** `haydenfarquhar/pet-fm-bench-t8-patches`
#
# **Task:** T8 — Lung cancer subtype classification. Embeddings saved with
# subtype labels for downstream multi-class classification probes.

# %% [markdown]
# ## 1. Install dependencies

# %%
!pip install -q open_clip_torch
!pip install -q lighter-zoo
# FMCIB loaded manually (pip package broken on Kaggle)

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
# Prefer v3 dataset if attached; falls back to v1 or any pet-fm-bench-t8* dataset.
# rglob handles Kaggle's variable mount depth (/kaggle/input/<ds>/ vs /kaggle/input/datasets/<user>/<ds>/).
V3_CANDIDATES = list(Path("/kaggle/input").rglob("pet-fm-bench-t8-patches-v3"))
INPUT_CANDIDATES = V3_CANDIDATES or list(Path("/kaggle/input").glob("pet-fm-bench-t8*"))
if not INPUT_CANDIDATES:
    INPUT_CANDIDATES = list(Path("/kaggle/input").iterdir())

print("Available inputs:")
for p in INPUT_CANDIDATES:
    print(f"  {p.name}")

PATCH_DIR = INPUT_CANDIDATES[0] if INPUT_CANDIDATES else Path("/kaggle/input/pet-fm-bench-t8-patches")

# Find manifest
manifest_path = list(PATCH_DIR.rglob("manifest.csv"))[0]
PATCH_DIR = manifest_path.parent
manifest = pd.read_csv(manifest_path)
print(f"\nManifest (raw): {len(manifest)} patients")

# File-existence filter: t8_01 writes manifest before per-patient loop, so
# failed/skipped patients stay in manifest but have no patch files.
def _has_files(r):
    mips = PATCH_DIR / "mips_2d" / r["patient_id"] / "mips.npz"
    pat3 = PATCH_DIR / "patches_3d" / r["patient_id"] / "patches.npz"
    return mips.exists() and pat3.exists()

mask = manifest.apply(_has_files, axis=1)
dropped_n = (~mask).sum()
manifest = manifest[mask].reset_index(drop=True)
print(f"Manifest (filtered): {len(manifest)} patients")
print(f"Dropped {dropped_n} patients with missing patch files")
print(f"Subtypes: {manifest['subtype'].value_counts().to_dict()}")

EMBED_DIR = Path("/kaggle/working/embeddings")
EMBED_DIR.mkdir(parents=True, exist_ok=True)

# Save subtype labels alongside embeddings for downstream probes
manifest[["patient_id", "subtype"]].to_csv(
    EMBED_DIR / "t8_labels.csv", index=False
)

# %% [markdown]
# ## 4. Data loading helpers

# %%
def load_patches_3d(patient_id):
    path = PATCH_DIR / "patches_3d" / patient_id / "patches.npz"
    data = np.load(path)
    return data["patches"].astype(np.float32), data["positions"]


def load_mips(patient_id):
    path = PATCH_DIR / "mips_2d" / patient_id / "mips.npz"
    data = np.load(path)
    return {k: data[k].astype(np.float32) for k in ["coronal", "axial", "sagittal"]}


def mip_to_rgb_tensor(mip_2d):
    mip_2d = np.nan_to_num(mip_2d, nan=0.0, posinf=0.0, neginf=0.0)
    vmin, vmax = float(mip_2d.min()), float(mip_2d.max())
    if vmax > vmin:
        normed = (mip_2d - vmin) / (vmax - vmin)
    else:
        normed = np.zeros_like(mip_2d)
    stacked = np.stack([normed, normed, normed])
    return torch.tensor(stacked, dtype=torch.float32).unsqueeze(0)


# Filter manifest to only patients with preprocessed data
valid_patients = []
for _, row in manifest.iterrows():
    pid = row["patient_id"]
    has_patches = (PATCH_DIR / "patches_3d" / pid / "patches.npz").exists()
    has_mips = (PATCH_DIR / "mips_2d" / pid / "mips.npz").exists()
    if has_patches and has_mips:
        valid_patients.append(pid)

manifest = manifest[manifest["patient_id"].isin(valid_patients)].reset_index(drop=True)
print(f"Patients with preprocessed data: {len(manifest)}")

# %% [markdown]
# ## 5. FM definitions
#
# Identical to T9 notebook — proven loading code.

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
        mips = load_mips(r["patient_id"])
        for view, arr in mips.items():
            inp = mip_to_rgb_tensor(arr).to(DEVICE)
            with torch.no_grad():
                out = model(inp, output_hidden_states=True)
            for layer_name, layer_idx in [("layer03", 3), ("layer06", 6),
                                           ("layer11", 11), ("cls", -1)]:
                if layer_name == "cls":
                    emb = out.last_hidden_state[0, 0].cpu().numpy()
                else:
                    emb = out.hidden_states[layer_idx][0, 0].cpu().numpy()
                rows.append({"patient_id": r["patient_id"], "view": view,
                             "layer": layer_name, "embedding": emb})
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
        mips = load_mips(r["patient_id"])
        for view, arr in mips.items():
            inp = mip_to_rgb_tensor(arr).to(DEVICE)
            with torch.no_grad():
                out = model(inp, output_hidden_states=True)
            for layer_name, layer_idx in [("layer03", 3), ("layer06", 6),
                                           ("layer11", 11), ("cls", -1)]:
                if layer_name == "cls":
                    emb = out.last_hidden_state[0, 0].cpu().numpy()
                else:
                    emb = out.hidden_states[layer_idx][0, 0].cpu().numpy()
                rows.append({"patient_id": r["patient_id"], "view": view,
                             "layer": layer_name, "embedding": emb})
    del model; torch.cuda.empty_cache()
    return rows


# ═══════════════════════════════════════════════════════════════════════
# BiomedCLIP (2D, ViT-B/16, 512-dim)
# ═══════════════════════════════════════════════════════════════════════

def run_biomedclip(manifest):
    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms(
        "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    )
    model = model.to(DEVICE).eval()
    rows = []
    for _, r in tqdm(manifest.iterrows(), total=len(manifest), desc="BiomedCLIP"):
        mips = load_mips(r["patient_id"])
        for view, arr in mips.items():
            inp = mip_to_rgb_tensor(arr).to(DEVICE)
            with torch.no_grad():
                emb = model.encode_image(inp)[0].cpu().numpy()
            rows.append({"patient_id": r["patient_id"], "view": view,
                         "layer": "cls", "embedding": emb})
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
        mips = load_mips(r["patient_id"])
        for view, arr in mips.items():
            inp = mip_to_rgb_tensor(arr).to(DEVICE)
            with torch.no_grad():
                out = model(inp)
            emb = out.last_hidden_state[0, 0].cpu().numpy()
            rows.append({"patient_id": r["patient_id"], "view": view,
                         "layer": "cls", "embedding": emb})
    del model; torch.cuda.empty_cache()
    return rows


# ═══════════════════════════════════════════════════════════════════════
# FMCIB (3D, ResNet-50-2x, 4096-dim) — manual loader
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


def run_fmcib(manifest):
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
    for _, r in tqdm(manifest.iterrows(), total=len(manifest), desc="FMCIB"):
        patches, positions = load_patches_3d(r["patient_id"])
        tensor = torch.tensor(patches, dtype=torch.float32).unsqueeze(1)
        all_emb = []
        for i in range(0, len(tensor), 4):
            batch = tensor[i:i+4].to(DEVICE)
            with torch.no_grad():
                out = model(batch)
            all_emb.append(out.cpu().numpy())
        embeddings = np.concatenate(all_emb, axis=0)
        # Mean-pool across patches for patient-level embedding
        rows.append({"patient_id": r["patient_id"], "view": "volume",
                     "layer": "pool", "embedding": embeddings.mean(axis=0)})
    del model; torch.cuda.empty_cache()
    return rows


# ═══════════════════════════════════════════════════════════════════════
# CT-FM (3D, SegResEncoder, 512-dim)
# ═══════════════════════════════════════════════════════════════════════

def run_ct_fm(manifest):
    from lighter_zoo import SegResEncoder
    model = SegResEncoder.from_pretrained(
        "project-lighter/ct_fm_feature_extractor"
    ).to(DEVICE).eval()

    rows = []
    for _, r in tqdm(manifest.iterrows(), total=len(manifest), desc="CT-FM"):
        patches, positions = load_patches_3d(r["patient_id"])
        tensor = torch.tensor(patches, dtype=torch.float32).unsqueeze(1)
        all_emb = []
        for i in range(0, len(tensor), 4):
            batch = tensor[i:i+4].to(DEVICE)
            with torch.no_grad():
                out = model(batch)[-1]
                pooled = torch.nn.functional.adaptive_avg_pool3d(out, 1)
                pooled = pooled.squeeze(-1).squeeze(-1).squeeze(-1)
            all_emb.append(pooled.cpu().numpy())
        embeddings = np.concatenate(all_emb, axis=0)
        rows.append({"patient_id": r["patient_id"], "view": "volume",
                     "layer": "pool", "embedding": embeddings.mean(axis=0)})
    del model; torch.cuda.empty_cache()
    return rows


# ═══════════════════════════════════════════════════════════════════════
# Merlin (3D, I3D ResNet-152)
# ═══════════════════════════════════════════════════════════════════════

def run_merlin(manifest):
    try:
        from merlin import Merlin
        model = Merlin(ImageEmbedding=True).to(DEVICE).eval()
    except Exception as e:
        print(f"Could not load Merlin: {e}")
        return []

    rows = []
    for _, r in tqdm(manifest.iterrows(), total=len(manifest), desc="Merlin"):
        patches, positions = load_patches_3d(r["patient_id"])
        tensor = torch.tensor(patches, dtype=torch.float32).unsqueeze(1)
        all_emb = []
        for i in range(0, len(tensor), 2):
            batch = tensor[i:i+2].to(DEVICE)
            with torch.no_grad():
                out = model(batch)
            if isinstance(out, dict):
                emb = list(out.values())[0]
            elif isinstance(out, (tuple, list)):
                emb = out[0]
            else:
                emb = out
            all_emb.append(emb.cpu().numpy())
        embeddings = np.concatenate(all_emb, axis=0)
        rows.append({"patient_id": r["patient_id"], "view": "volume",
                     "layer": "pool", "embedding": embeddings.mean(axis=0)})
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
    ("merlin",      run_merlin),
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
            rec = {"patient_id": row["patient_id"], "view": row["view"],
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
print("T8 EMBEDDING EXTRACTION COMPLETE")
print("=" * 60)

total_mb = 0
for f in sorted(EMBED_DIR.glob("*.parquet")):
    df = pd.read_parquet(f)
    mb = f.stat().st_size / 1e6
    total_mb += mb
    dim = sum(1 for c in df.columns if c.startswith("d"))
    print(f"\n{f.stem}: {df['patient_id'].nunique()} patients, {dim}-dim, {mb:.1f} MB")

print(f"\nTotal output: {total_mb:.1f} MB")
print(f"Labels file: {(EMBED_DIR / 't8_labels.csv').stat().st_size / 1e3:.0f} KB")

# %% [markdown]
# ## 8. Save
#
# Commit with **"Save & Run All"**. Then save output as Kaggle Dataset:
# `haydenfarquhar/pet-fm-bench-t8-embeddings`
