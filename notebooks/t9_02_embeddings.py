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
# # T9 Notebook 2/2: Extract FM Embeddings
#
# **PET-FM-Bench** | DOI: [10.17605/OSF.IO/DQ2JA](https://doi.org/10.17605/OSF.IO/DQ2JA)
#
# **Runtime:** GPU T4 | **Internet:** On | **Time:** ~1–2 hours
#
# Loads preprocessed patches from the T9 Kaggle Dataset (attached as input),
# extracts frozen embeddings from each FM, saves only embedding parquets
# (~50 MB total) to output.
#
# **Input dataset:** `haydenfarquhar/pet-fm-bench-t9-patches` (attach via
# Add Data → Your Datasets)

# %% [markdown]
# ## 1. Install dependencies (one at a time to isolate failures)

# %%
!pip install -q open_clip_torch
!pip install -q lighter-zoo
# NOTE: foundation-cancer-image-biomarker has broken build deps on Kaggle.
# FMCIB model is loaded manually from Zenodo weights instead (see FM definitions below).
# transformers, timm, torch, monai, SimpleITK are pre-installed on Kaggle GPU

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

# HuggingFace token for gated models
try:
    from kaggle_secrets import UserSecretsClient
    HF_TOKEN = UserSecretsClient().get_secret("HF_TOKEN")
    print("HF token loaded")
except Exception:
    HF_TOKEN = None
    print("No HF token — gated models will be skipped")

# %% [markdown]
# ## 3. Locate input data

# %%
# Find the attached dataset — Kaggle mounts inputs under /kaggle/input/
INPUT_CANDIDATES = list(Path("/kaggle/input").glob("pet-fm-bench-t9*"))
if not INPUT_CANDIDATES:
    # Try broader search
    INPUT_CANDIDATES = list(Path("/kaggle/input").iterdir())

print("Available input datasets:")
for p in INPUT_CANDIDATES:
    print(f"  {p.name}")

# Set the input path — adjust if the dataset name differs
PATCH_DIR = INPUT_CANDIDATES[0] if INPUT_CANDIDATES else Path("/kaggle/input/pet-fm-bench-t9-patches")
print(f"\nUsing: {PATCH_DIR}")

# Verify structure
manifest_path = PATCH_DIR / "manifest.csv"
if not manifest_path.exists():
    # Might be nested one level deeper
    manifest_path = list(PATCH_DIR.rglob("manifest.csv"))[0]
    PATCH_DIR = manifest_path.parent

manifest = pd.read_csv(manifest_path)
print(f"Manifest (raw): {len(manifest)} sessions, {manifest['subject_id'].nunique()} subjects")

# File-existence filter: t9_01 writes manifest before per-session loop, so
# failed/skipped sessions stay in manifest but have no patch files.
def _has_files(r):
    mips = PATCH_DIR / "mips_2d" / r["subject_id"] / r["session"] / "mips.npz"
    pat3 = PATCH_DIR / "patches_3d" / r["subject_id"] / r["session"] / "patches.npz"
    return mips.exists() and pat3.exists()

mask = manifest.apply(_has_files, axis=1)
dropped_n = (~mask).sum()
manifest = manifest[mask].reset_index(drop=True)
print(f"Manifest (filtered): {len(manifest)} sessions, {manifest['subject_id'].nunique()} subjects")
print(f"Dropped {dropped_n} sessions with missing patch files")

# Output directory
EMBED_DIR = Path("/kaggle/working/embeddings")
EMBED_DIR.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## 4. Data loading helpers

# %%
def load_patches_3d(subject_id, session):
    """Load preprocessed 3D patches for a session."""
    path = PATCH_DIR / "patches_3d" / subject_id / session / "patches.npz"
    data = np.load(path)
    return data["patches"].astype(np.float32), data["positions"]


def load_mips(subject_id, session):
    """Load preprocessed 2D MIPs for a session."""
    path = PATCH_DIR / "mips_2d" / subject_id / session / "mips.npz"
    data = np.load(path)
    return {
        "coronal": data["coronal"].astype(np.float32),
        "axial": data["axial"].astype(np.float32),
        "sagittal": data["sagittal"].astype(np.float32),
    }


def mip_to_rgb_tensor(mip_2d):
    """Convert (224,224) MIP to (1,3,224,224) tensor normalised to [0,1]."""
    mip_2d = np.nan_to_num(mip_2d, nan=0.0, posinf=0.0, neginf=0.0)
    vmin, vmax = float(mip_2d.min()), float(mip_2d.max())
    if vmax > vmin:
        normed = (mip_2d - vmin) / (vmax - vmin)
    else:
        normed = np.zeros_like(mip_2d)
    stacked = np.stack([normed, normed, normed])  # (3, 224, 224)
    return torch.tensor(stacked, dtype=torch.float32).unsqueeze(0)  # (1, 3, 224, 224)


# %% [markdown]
# ## 5. FM definitions
#
# Each FM has a `load` and `extract` function. FMs are processed one at a time
# and unloaded after to free VRAM.

# %%
# ═══════════════════════════════════════════════════════════════════════
# DINOv2 (2D, ViT-B/14, 768-dim, ImageNet)
# ═══════════════════════════════════════════════════════════════════════

def run_dinov2(manifest):
    from transformers import AutoModel
    model = AutoModel.from_pretrained("facebook/dinov2-base",
                                      revision="f9e44c8").to(DEVICE).eval()
    rows = []
    for _, r in tqdm(manifest.iterrows(), total=len(manifest), desc="DINOv2"):
        mips = load_mips(r["subject_id"], r["session"])
        for view, arr in mips.items():
            inp = mip_to_rgb_tensor(arr).to(DEVICE)
            with torch.no_grad():
                out = model(inp, output_hidden_states=True)
            # 4 layers: early(3), mid(6), penultimate(11), final CLS
            for layer_name, layer_idx in [("layer03", 3), ("layer06", 6),
                                           ("layer11", 11), ("cls", -1)]:
                if layer_name == "cls":
                    emb = out.last_hidden_state[0, 0].cpu().numpy()
                else:
                    emb = out.hidden_states[layer_idx][0, 0].cpu().numpy()
                rows.append({
                    "subject_id": r["subject_id"], "session": r["session"],
                    "view": view, "layer": layer_name,
                    "embedding": emb,
                })
    del model; torch.cuda.empty_cache()
    return rows


# ═══════════════════════════════════════════════════════════════════════
# RAD-DINO (2D, ViT-B/14, 768-dim, CXR)
# ═══════════════════════════════════════════════════════════════════════

def run_rad_dino(manifest):
    from transformers import AutoModel
    model = AutoModel.from_pretrained("microsoft/rad-dino",
                                      revision="2ec9ca0").to(DEVICE).eval()
    rows = []
    for _, r in tqdm(manifest.iterrows(), total=len(manifest), desc="RAD-DINO"):
        mips = load_mips(r["subject_id"], r["session"])
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
                rows.append({
                    "subject_id": r["subject_id"], "session": r["session"],
                    "view": view, "layer": layer_name,
                    "embedding": emb,
                })
    del model; torch.cuda.empty_cache()
    return rows


# ═══════════════════════════════════════════════════════════════════════
# BiomedCLIP (2D, ViT-B/16, 512-dim, PMC-15M)
# ═══════════════════════════════════════════════════════════════════════

def run_biomedclip(manifest):
    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms(
        "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    )
    model = model.to(DEVICE).eval()
    rows = []
    for _, r in tqdm(manifest.iterrows(), total=len(manifest), desc="BiomedCLIP"):
        mips = load_mips(r["subject_id"], r["session"])
        for view, arr in mips.items():
            inp = mip_to_rgb_tensor(arr).to(DEVICE)
            with torch.no_grad():
                emb = model.encode_image(inp)[0].cpu().numpy()
            rows.append({
                "subject_id": r["subject_id"], "session": r["session"],
                "view": view, "layer": "cls",
                "embedding": emb,
            })
    del model; torch.cuda.empty_cache()
    return rows


# ═══════════════════════════════════════════════════════════════════════
# Random Init Control (2D, ViT-B/14 with random weights, 768-dim)
# ═══════════════════════════════════════════════════════════════════════

def run_random_init(manifest):
    from transformers import AutoConfig, AutoModel
    config = AutoConfig.from_pretrained("facebook/dinov2-base")
    model = AutoModel.from_config(config).to(DEVICE).eval()
    rows = []
    for _, r in tqdm(manifest.iterrows(), total=len(manifest), desc="RandomInit"):
        mips = load_mips(r["subject_id"], r["session"])
        for view, arr in mips.items():
            inp = mip_to_rgb_tensor(arr).to(DEVICE)
            with torch.no_grad():
                out = model(inp)
            emb = out.last_hidden_state[0, 0].cpu().numpy()
            rows.append({
                "subject_id": r["subject_id"], "session": r["session"],
                "view": view, "layer": "cls",
                "embedding": emb,
            })
    del model; torch.cuda.empty_cache()
    return rows


# ═══════════════════════════════════════════════════════════════════════
# FMCIB (3D, ResNet-50-2x, 4096-dim, cancer lesions)
# Loaded manually from Zenodo weights — pip package has broken build deps
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

    # Download weights from Zenodo
    weight_path = "/tmp/fmcib_weights.torch"
    if not os.path.exists(weight_path) or os.path.getsize(weight_path) < 1e6:
        print("  Downloading FMCIB weights from Zenodo (738 MB)...")
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
        patches, positions = load_patches_3d(r["subject_id"], r["session"])
        tensor = torch.tensor(patches, dtype=torch.float32).unsqueeze(1)
        all_emb = []
        for i in range(0, len(tensor), 4):
            batch = tensor[i:i+4].to(DEVICE)
            with torch.no_grad():
                out = model(batch)
            all_emb.append(out.cpu().numpy())
        embeddings = np.concatenate(all_emb, axis=0)
        rows.append({
            "subject_id": r["subject_id"], "session": r["session"],
            "view": "volume", "layer": "pool",
            "embedding": embeddings.mean(axis=0),
        })
        for idx in range(len(embeddings)):
            rows.append({
                "subject_id": r["subject_id"], "session": r["session"],
                "view": "volume", "layer": f"patch_{idx:03d}",
                "embedding": embeddings[idx],
            })
    del model; torch.cuda.empty_cache()
    return rows


# ═══════════════════════════════════════════════════════════════════════
# CT-FM (3D, SegResEncoder, 512-dim, CT)
# ═══════════════════════════════════════════════════════════════════════

def run_ct_fm(manifest):
    from lighter_zoo import SegResEncoder
    model = SegResEncoder.from_pretrained(
        "project-lighter/ct_fm_feature_extractor"
    ).to(DEVICE).eval()
    rows = []
    for _, r in tqdm(manifest.iterrows(), total=len(manifest), desc="CT-FM"):
        patches, positions = load_patches_3d(r["subject_id"], r["session"])
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
        rows.append({
            "subject_id": r["subject_id"], "session": r["session"],
            "view": "volume", "layer": "pool",
            "embedding": embeddings.mean(axis=0),
        })
        for idx in range(len(embeddings)):
            rows.append({
                "subject_id": r["subject_id"], "session": r["session"],
                "view": "volume", "layer": f"patch_{idx:03d}",
                "embedding": embeddings[idx],
            })
    del model; torch.cuda.empty_cache()
    return rows


# ═══════════════════════════════════════════════════════════════════════
# Merlin (3D, I3D ResNet-152, CT VL — try/catch, may need debugging)
# ═══════════════════════════════════════════════════════════════════════

def run_merlin(manifest):
    try:
        from merlin import Merlin
        model = Merlin(ImageEmbedding=True).to(DEVICE).eval()
    except Exception as e:
        print(f"Could not load Merlin: {e}")
        print("Install with: pip install merlin-vlm")
        return []

    rows = []
    for _, r in tqdm(manifest.iterrows(), total=len(manifest), desc="Merlin"):
        patches, positions = load_patches_3d(r["subject_id"], r["session"])
        tensor = torch.tensor(patches, dtype=torch.float32).unsqueeze(1)
        all_emb = []
        for i in range(0, len(tensor), 2):  # smaller batches — large model
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
        rows.append({
            "subject_id": r["subject_id"], "session": r["session"],
            "view": "volume", "layer": "pool",
            "embedding": embeddings.mean(axis=0),
        })
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
            print(f"  No embeddings produced — skipping")
            continue

        # Convert embedding arrays to columnar format for parquet
        embed_dim = len(rows[0]["embedding"])
        records = []
        for row in rows:
            rec = {
                "subject_id": row["subject_id"],
                "session": row["session"],
                "view": row["view"],
                "layer": row["layer"],
            }
            emb = row["embedding"]
            for j in range(len(emb)):
                rec[f"d{j:04d}"] = float(emb[j])
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
print("EMBEDDING EXTRACTION COMPLETE")
print("=" * 60)

total_mb = 0
for f in sorted(EMBED_DIR.glob("*.parquet")):
    df = pd.read_parquet(f)
    mb = f.stat().st_size / 1e6
    total_mb += mb
    dim = sum(1 for c in df.columns if c.startswith("d"))
    n_subj = df["subject_id"].nunique()
    layers = df["layer"].unique()
    print(f"\n{f.stem}:")
    print(f"  {n_subj} subjects, {dim}-dim, {len(df)} rows, {mb:.1f} MB")
    print(f"  Layers: {sorted(layers)[:5]}{'...' if len(layers) > 5 else ''}")

print(f"\nTotal output: {total_mb:.1f} MB")

# %% [markdown]
# ## 8. Save
#
# Commit with **"Save & Run All"** (or Quick Save if you ran as draft).
# Then save the output as Kaggle Dataset: `haydenfarquhar/pet-fm-bench-t9-embeddings`
