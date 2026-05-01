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
# # T9: Vienna QUADRA — Download, Preprocess & Extract Embeddings
#
# **PET-FM-Bench** | DOI: [10.17605/OSF.IO/DQ2JA](https://doi.org/10.17605/OSF.IO/DQ2JA)
#
# Combined pipeline: downloads dataset, preprocesses in `/tmp/`, extracts
# frozen embeddings from all FMs, and saves only the embeddings (~50 MB total)
# to `/kaggle/working/` for persistence.
#
# **Runtime:** GPU T4 | **Internet:** On | **Estimated time:** 2-3 hours

# %% [markdown]
# ## 0. Setup

# %%
import os
import sys
import hashlib
import zipfile
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

# Install dependencies — use --break-system-packages for Kaggle's managed Python
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                       "--break-system-packages",
                       "SimpleITK>=2.3", "tqdm", "requests",
                       "transformers>=4.40", "timm>=0.9",
                       "open_clip_torch",
                       "lighter-zoo",
                       "foundation-cancer-image-biomarker",
                       "monai>=1.3"])

import torch
import SimpleITK as sitk
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

# %% [markdown]
# ## 1. Configuration

# %%
# Paths — raw data in /tmp/, only embeddings saved to /kaggle/working/
TMP_DIR = Path("/tmp/pet_fm_bench")
RAW_DIR = TMP_DIR / "raw" / "vienna_quadra"
EMBED_DIR = Path("/kaggle/working/embeddings/t9_vienna_quadra")
OUTPUT_DIR = Path("/kaggle/working/output")

ZENODO_RECORD_ID = "16686025"
ZENODO_FILENAME = "QUADRA_HC.zip"
ZENODO_URL = f"https://zenodo.org/records/{ZENODO_RECORD_ID}/files/{ZENODO_FILENAME}"

# Preprocessing parameters (registration Section 5.1)
PATCH_SIZE_3D = (96, 96, 96)
SPACING_ISO = (2.0, 2.0, 2.0)
MIP_SIZE_2D = (224, 224)

# HuggingFace token for gated models (Pillar-0)
# Set in Kaggle: Add-ons → Secrets → add HF_TOKEN
try:
    from kaggle_secrets import UserSecretsClient
    HF_TOKEN = UserSecretsClient().get_secret("HF_TOKEN")
    print("HF_TOKEN loaded from Kaggle Secrets")
except Exception:
    HF_TOKEN = os.environ.get("HF_TOKEN", None)
    if HF_TOKEN:
        print("HF_TOKEN loaded from environment")
    else:
        print("WARNING: No HF_TOKEN found — gated models will be skipped")

for d in [RAW_DIR, EMBED_DIR, OUTPUT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## 2. Download and extract

# %%
def download_with_progress(url, dest_path):
    import requests
    response = requests.get(url, stream=True)
    response.raise_for_status()
    total = int(response.headers.get("content-length", 0))
    with open(dest_path, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True, desc=dest_path.name
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
            f.write(chunk)
            pbar.update(len(chunk))


zip_path = RAW_DIR / ZENODO_FILENAME
imaging_dir = RAW_DIR / "QUADRA_HC" / "Imaging Data"

if not imaging_dir.exists():
    if not zip_path.exists():
        print("Downloading from Zenodo (~20 GB)...")
        download_with_progress(ZENODO_URL, zip_path)

    # Compute SHA256
    sha256 = hashlib.sha256()
    with open(zip_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192 * 1024), b""):
            sha256.update(chunk)
    print(f"SHA256: {sha256.hexdigest()}")

    print("Extracting...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(RAW_DIR)
    zip_path.unlink()
    print("Extraction complete, zip removed")
else:
    print(f"Already extracted: {imaging_dir}")

subjects = sorted([d for d in imaging_dir.iterdir()
                   if d.is_dir() and d.name.startswith("QUADRA_HC_")])
print(f"Found {len(subjects)} subjects")

# %% [markdown]
# ## 3. Build manifest

# %%
manifest_rows = []
for subj_dir in subjects:
    subj_id = subj_dir.name
    for session_name in ["Test", "Retest"]:
        session_dir = subj_dir / session_name
        if not session_dir.exists():
            continue
        pet_file = next(session_dir.glob("*_PT-SUV.nii.gz"), None)
        ct_file = next(session_dir.glob("*_CT-AC.nii.gz"), None)
        manifest_rows.append({
            "subject_id": subj_id,
            "session": session_name,
            "pet_path": str(pet_file) if pet_file else None,
            "ct_path": str(ct_file) if ct_file else None,
        })

manifest = pd.DataFrame(manifest_rows)
print(f"Manifest: {len(manifest)} rows, {manifest['subject_id'].nunique()} subjects")
print(f"PET found: {manifest['pet_path'].notna().sum()}/{len(manifest)}")
manifest.to_csv(OUTPUT_DIR / "t9_manifest.csv", index=False)

# %% [markdown]
# ## 4. Preprocessing functions

# %%
def resample_to_isotropic(img_sitk, target_spacing=(2.0, 2.0, 2.0)):
    original_spacing = img_sitk.GetSpacing()
    original_size = img_sitk.GetSize()
    new_size = [int(round(osz * ospc / tspc))
                for osz, ospc, tspc in zip(original_size, original_spacing, target_spacing)]
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(img_sitk.GetDirection())
    resampler.SetOutputOrigin(img_sitk.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(0)
    resampler.SetInterpolator(sitk.sitkLinear)
    return resampler.Execute(img_sitk)


def extract_patches_grid(volume, patch_size=(96, 96, 96)):
    """Non-overlapping patches from a 3D volume."""
    pz, py, px = patch_size
    patches, positions = [], []
    for z in range(0, max(1, volume.shape[0] - pz + 1), pz):
        for y in range(0, max(1, volume.shape[1] - py + 1), py):
            for x in range(0, max(1, volume.shape[2] - px + 1), px):
                patch = volume[z:z+pz, y:y+py, x:x+px]
                if patch.shape != (pz, py, px):
                    padded = np.zeros((pz, py, px), dtype=patch.dtype)
                    padded[:patch.shape[0], :patch.shape[1], :patch.shape[2]] = patch
                    patch = padded
                patches.append(patch)
                positions.append((z, y, x))
    return np.stack(patches), positions


def compute_mips_224(volume):
    """Compute 3 MIPs resized to 224x224."""
    def _resize(arr_2d):
        img = sitk.GetImageFromArray(arr_2d.astype(np.float32))
        resampler = sitk.ResampleImageFilter()
        resampler.SetSize((224, 224))
        resampler.SetOutputSpacing((arr_2d.shape[1] / 224, arr_2d.shape[0] / 224))
        resampler.SetInterpolator(sitk.sitkLinear)
        return sitk.GetArrayFromImage(resampler.Execute(img))

    return {
        "coronal": _resize(volume.max(axis=1)),
        "axial": _resize(volume.max(axis=0)),
        "sagittal": _resize(volume.max(axis=2)),
    }


def preprocess_session(pet_path):
    """Load PET, resample, return patches and MIPs."""
    pet_sitk = sitk.ReadImage(pet_path)
    pet_iso = resample_to_isotropic(pet_sitk, target_spacing=SPACING_ISO)
    pet_arr = sitk.GetArrayFromImage(pet_iso).astype(np.float32)

    patches_3d, positions = extract_patches_grid(pet_arr, PATCH_SIZE_3D)
    mips = compute_mips_224(pet_arr)

    return patches_3d, positions, mips

# %% [markdown]
# ## 5. Foundation model loaders
#
# Each FM is loaded, used to extract embeddings for all sessions, then unloaded
# to free VRAM. Embeddings are saved incrementally.

# %%
# ── 2D FM helpers ────────────────────────────────────────────────────────

def mip_to_rgb_tensor(mip_2d):
    """Convert a single-channel MIP to a 3-channel tensor for 2D FMs.
    Input: (224, 224) float32 numpy array.
    Output: (3, 224, 224) float32 tensor, normalised to [0, 1].
    """
    # Normalise to [0, 1]
    vmin, vmax = mip_2d.min(), mip_2d.max()
    if vmax > vmin:
        normed = (mip_2d - vmin) / (vmax - vmin)
    else:
        normed = np.zeros_like(mip_2d)
    # Stack to 3 channels, add batch dim handled by caller
    return torch.tensor(np.stack([normed, normed, normed]), dtype=torch.float32)


# ── DINOv2 ───────────────────────────────────────────────────────────────

def load_dinov2():
    from transformers import AutoModel, AutoImageProcessor
    model = AutoModel.from_pretrained("facebook/dinov2-base",
                                      revision="f9e44c8").to(DEVICE).eval()
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base",
                                                    revision="f9e44c8")
    return model, processor


def extract_dinov2(model, processor, mips):
    """Extract DINOv2 CLS embeddings from 3 MIPs. Returns dict of arrays."""
    embeddings = {}
    for view_name, mip_arr in mips.items():
        img_tensor = mip_to_rgb_tensor(mip_arr).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            outputs = model(img_tensor, output_hidden_states=True)
        # Layer-wise: early(3), mid(6), penultimate(11), final CLS
        hidden = outputs.hidden_states
        embeddings[f"cls_{view_name}"] = outputs.last_hidden_state[:, 0].cpu().numpy()
        embeddings[f"layer3_{view_name}"] = hidden[3][:, 0].cpu().numpy()
        embeddings[f"layer6_{view_name}"] = hidden[6][:, 0].cpu().numpy()
        embeddings[f"layer11_{view_name}"] = hidden[11][:, 0].cpu().numpy()
    return embeddings


# ── RAD-DINO ─────────────────────────────────────────────────────────────

def load_rad_dino():
    from transformers import AutoModel, AutoImageProcessor
    model = AutoModel.from_pretrained("microsoft/rad-dino",
                                      revision="2ec9ca0").to(DEVICE).eval()
    processor = AutoImageProcessor.from_pretrained("microsoft/rad-dino",
                                                    revision="2ec9ca0")
    return model, processor


def extract_rad_dino(model, processor, mips):
    """Same architecture as DINOv2, same extraction logic."""
    return extract_dinov2(model, processor, mips)  # identical ViT-B/14


# ── BiomedCLIP ───────────────────────────────────────────────────────────

def load_biomedclip():
    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms(
        "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    )
    model = model.to(DEVICE).eval()
    return model, preprocess


def extract_biomedclip(model, preprocess, mips):
    """Extract BiomedCLIP vision embeddings from MIPs."""
    embeddings = {}
    for view_name, mip_arr in mips.items():
        img_tensor = mip_to_rgb_tensor(mip_arr).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            features = model.encode_image(img_tensor)
        embeddings[f"cls_{view_name}"] = features.cpu().numpy()
    return embeddings


# ── 3D FM helpers ────────────────────────────────────────────────────────

def patches_to_tensor_3d(patches_np, add_channel=True):
    """Convert (N, D, H, W) float32 patches to torch tensor.
    Optionally adds channel dim → (N, 1, D, H, W).
    """
    t = torch.tensor(patches_np, dtype=torch.float32)
    if add_channel:
        t = t.unsqueeze(1)
    return t


# ── FMCIB ────────────────────────────────────────────────────────────────

def load_fmcib():
    from fmcib.models import fmcib_model
    model = fmcib_model()
    model = model.to(DEVICE).eval()
    return model


def extract_fmcib(model, patches_3d):
    """Extract FMCIB features from 3D patches.
    FMCIB expects (B, 1, H, W, D) and 50x50x50 recommended, but accepts
    variable sizes. We use our 96^3 patches.
    Output: 4096-dim per patch.
    """
    all_feats = []
    tensor = patches_to_tensor_3d(patches_3d)  # (N, 1, 96, 96, 96)
    for i in range(0, len(tensor), 4):  # batch of 4 to manage VRAM
        batch = tensor[i:i+4].to(DEVICE)
        with torch.no_grad():
            feats = model(batch)
        all_feats.append(feats.cpu().numpy())
    return {"patch_embed": np.concatenate(all_feats, axis=0)}


# ── CT-FM ────────────────────────────────────────────────────────────────

def load_ct_fm():
    from lighter_zoo import SegResEncoder
    model = SegResEncoder.from_pretrained("project-lighter/ct_fm_feature_extractor")
    model = model.to(DEVICE).eval()
    return model


def extract_ct_fm(model, patches_3d):
    """Extract CT-FM features from 3D patches.
    Input: (B, 1, D, H, W). Output: 512-dim after adaptive avg pool.
    """
    all_feats = []
    tensor = patches_to_tensor_3d(patches_3d)
    for i in range(0, len(tensor), 4):
        batch = tensor[i:i+4].to(DEVICE)
        with torch.no_grad():
            out = model(batch)[-1]  # last encoder stage
            pooled = torch.nn.functional.adaptive_avg_pool3d(out, 1).squeeze(-1).squeeze(-1).squeeze(-1)
        all_feats.append(pooled.cpu().numpy())
    return {"patch_embed": np.concatenate(all_feats, axis=0)}


# ── Merlin ───────────────────────────────────────────────────────────────

def load_merlin():
    try:
        from merlin import Merlin
        model = Merlin(ImageEmbedding=True)
        model = model.to(DEVICE).eval()
        return model
    except Exception as e:
        print(f"WARNING: Could not load Merlin: {e}")
        return None


def extract_merlin(model, patches_3d):
    """Extract Merlin vision embeddings from 3D patches."""
    if model is None:
        return None
    all_feats = []
    tensor = patches_to_tensor_3d(patches_3d)
    for i in range(0, len(tensor), 2):  # smaller batches, Merlin is large
        batch = tensor[i:i+2].to(DEVICE)
        with torch.no_grad():
            out = model(batch)
            if isinstance(out, dict):
                feats = out.get("image_embedding", out.get("embedding", list(out.values())[0]))
            elif isinstance(out, (tuple, list)):
                feats = out[0]
            else:
                feats = out
        all_feats.append(feats.cpu().numpy())
    return {"patch_embed": np.concatenate(all_feats, axis=0)}


# ── Pillar-0 ─────────────────────────────────────────────────────────────

def load_pillar0(variant="AbdomenCT"):
    """Load Pillar-0 variant. Requires HF_TOKEN for gated access."""
    try:
        from huggingface_hub import hf_hub_download
        import safetensors.torch
        # Pillar-0 doesn't have a standard transformers API;
        # we load weights and wrap in the Atlas encoder
        # This is a placeholder — exact loading depends on the model's config
        print(f"WARNING: Pillar-0 {variant} loading requires custom Atlas encoder code.")
        print("Skipping Pillar-0 for now — will implement once API is documented.")
        return None
    except Exception as e:
        print(f"WARNING: Could not load Pillar-0 {variant}: {e}")
        return None


def extract_pillar0(model, patches_3d):
    if model is None:
        return None
    return None  # Placeholder


# ── Random initialisation control ────────────────────────────────────────

def load_random_init():
    """DINOv2-base architecture with random weights (no pretraining)."""
    from transformers import AutoConfig, AutoModel
    config = AutoConfig.from_pretrained("facebook/dinov2-base")
    model = AutoModel.from_config(config).to(DEVICE).eval()
    return model


def extract_random_init(model, mips):
    """Same extraction as DINOv2 but with random weights."""
    embeddings = {}
    for view_name, mip_arr in mips.items():
        img_tensor = mip_to_rgb_tensor(mip_arr).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            outputs = model(img_tensor)
        embeddings[f"cls_{view_name}"] = outputs.last_hidden_state[:, 0].cpu().numpy()
    return embeddings


# %% [markdown]
# ## 6. FM registry
#
# Enable/disable FMs here. Disabled FMs are skipped gracefully.

# %%
FM_REGISTRY = [
    # 2D FMs (process MIPs)
    {"name": "dinov2",       "type": "2d", "load": load_dinov2,      "extract": extract_dinov2,      "enabled": True},
    {"name": "rad_dino",     "type": "2d", "load": load_rad_dino,    "extract": extract_rad_dino,    "enabled": True},
    {"name": "biomedclip",   "type": "2d", "load": load_biomedclip,  "extract": extract_biomedclip,  "enabled": True},
    {"name": "random_init",  "type": "2d", "load": load_random_init, "extract": extract_random_init, "enabled": True},

    # 3D FMs (process patches)
    {"name": "fmcib",        "type": "3d", "load": load_fmcib,       "extract": extract_fmcib,       "enabled": True},
    {"name": "ct_fm",        "type": "3d", "load": load_ct_fm,       "extract": extract_ct_fm,       "enabled": True},
    {"name": "merlin",       "type": "3d", "load": load_merlin,      "extract": extract_merlin,      "enabled": True},
    {"name": "pillar0",      "type": "3d", "load": load_pillar0,     "extract": extract_pillar0,     "enabled": False},  # pending API docs
]

print("Enabled FMs:")
for fm in FM_REGISTRY:
    if fm["enabled"]:
        print(f"  {fm['name']} ({fm['type']})")

# %% [markdown]
# ## 7. Main extraction loop
#
# For each FM: load → process all sessions → save embeddings → unload.

# %%
for fm_info in FM_REGISTRY:
    if not fm_info["enabled"]:
        print(f"\n--- Skipping {fm_info['name']} (disabled) ---")
        continue

    fm_name = fm_info["name"]
    print(f"\n{'='*60}")
    print(f"FM: {fm_name} ({fm_info['type']})")
    print(f"{'='*60}")

    # Load model
    try:
        model_obj = fm_info["load"]()
        if model_obj is None:
            print(f"  Model returned None, skipping")
            continue
        # For 2D FMs, load returns (model, processor); for 3D, just model
        if isinstance(model_obj, tuple):
            model, processor = model_obj
        else:
            model, processor = model_obj, None
    except Exception as e:
        print(f"  ERROR loading {fm_name}: {e}")
        continue

    # Extract embeddings for all sessions
    all_rows = []
    for _, row in tqdm(manifest.iterrows(), total=len(manifest),
                       desc=f"  {fm_name}"):
        if row["pet_path"] is None:
            continue

        try:
            patches_3d, positions, mips = preprocess_session(row["pet_path"])

            if fm_info["type"] == "2d":
                embeds = fm_info["extract"](model, processor, mips)
            else:
                embeds = fm_info["extract"](model, patches_3d)

            if embeds is None:
                continue

            # Flatten embeddings into rows
            for key, arr in embeds.items():
                # arr shape: (1, dim) for MIP-level or (N, dim) for patch-level
                if arr.ndim == 1:
                    arr = arr.reshape(1, -1)
                for idx in range(arr.shape[0]):
                    all_rows.append({
                        "subject_id": row["subject_id"],
                        "session": row["session"],
                        "embed_key": key,
                        "embed_idx": idx,
                        **{f"d{j}": float(arr[idx, j]) for j in range(arr.shape[1])},
                    })

        except Exception as e:
            print(f"  ERROR {row['subject_id']}/{row['session']}: {e}")

    # Save embeddings as parquet
    if all_rows:
        embed_df = pd.DataFrame(all_rows)
        out_path = EMBED_DIR / f"{fm_name}.parquet"
        embed_df.to_parquet(out_path, index=False)
        size_mb = out_path.stat().st_size / 1e6
        print(f"  Saved: {out_path.name} ({size_mb:.1f} MB, {len(embed_df)} rows)")
    else:
        print(f"  No embeddings produced for {fm_name}")

    # Unload model to free VRAM
    del model
    if processor is not None:
        del processor
    torch.cuda.empty_cache()
    print(f"  VRAM freed")

# %% [markdown]
# ## 8. Summary and QC

# %%
print("=" * 60)
print("T9 EMBEDDING EXTRACTION SUMMARY")
print("=" * 60)

total_size = 0
for f in sorted(EMBED_DIR.glob("*.parquet")):
    df = pd.read_parquet(f)
    size_mb = f.stat().st_size / 1e6
    total_size += size_mb

    # Count unique subjects and sessions
    n_subj = df["subject_id"].nunique()
    n_sess = df.groupby(["subject_id", "session"]).ngroups
    embed_dim = sum(1 for c in df.columns if c.startswith("d"))
    embed_keys = df["embed_key"].unique()

    print(f"\n{f.stem}:")
    print(f"  Subjects: {n_subj}, Sessions: {n_sess}")
    print(f"  Embedding dim: {embed_dim}")
    print(f"  Keys: {list(embed_keys)}")
    print(f"  File size: {size_mb:.1f} MB")

print(f"\nTotal embedding storage: {total_size:.1f} MB")
print(f"Output dir size: {sum(f.stat().st_size for f in Path('/kaggle/working').rglob('*') if f.is_file()) / 1e6:.1f} MB")

# %% [markdown]
# ## 9. Save as Kaggle Dataset
#
# Use "Save & Run All" to commit this notebook. Then go to the Output tab
# and click "New Dataset" to create `haydenfarquhar/pet-fm-bench-t9-embeddings`.

# %%
import json

dataset_metadata = {
    "title": "PET-FM-Bench T9: Vienna QUADRA Embeddings",
    "id": "haydenfarquhar/pet-fm-bench-t9-embeddings",
    "licenses": [{"name": "CC-BY-4.0"}],
}

meta_path = EMBED_DIR / "dataset-metadata.json"
with open(meta_path, "w") as f:
    json.dump(dataset_metadata, f, indent=2)

print(f"Metadata written to {meta_path}")
print(f"After committing, save output as Kaggle Dataset from the Output tab.")
