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
# # PET-FM-Bench: Phase 1 Freeze — FM Checkpoint Manifest
#
# **Runtime:** CPU | **Internet:** ON | **Time:** ~15-25 min | **GPU:** Not needed
#
# Pre-registration §11 freeze: SHA-256 hashes of every FM weight file used to
# extract the v3 embeddings. Locks in exact model versions for reproducibility.
#
# **Coverage:**
# - DINOv2 (facebook/dinov2-base, revision f9e44c8)
# - RAD-DINO (microsoft/rad-dino, revision 2ec9ca0)
# - BiomedCLIP (microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224)
# - FMCIB (Zenodo record 10528450)
# - CT-FM (project-lighter/ct_fm_feature_extractor)
# - Random Init control — documented, not hashable (no explicit seed in
#   embedding notebooks, weights non-reproducible by design as a control)
# - Merlin — attempt download; documented as skipped if module unavailable
#
# **Output:** `checkpoint_manifest.csv` for upload to OSF as supplementary file.
# Per pre-reg §11, this freeze must precede any formal probe training.

# %% [markdown]
# ## 1. Setup

# %%
import hashlib
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

!pip install -q transformers torch open_clip_torch huggingface_hub requests

CACHE_DIR = Path("/tmp/fm_weights")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR = Path("/kaggle/working")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def sha256_file(path, chunk_size=2**20):
    """Compute SHA-256 of a file, streaming so large weights don't blow memory."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def add_record(records, **kwargs):
    """Append a record with consistent schema; missing fields default to None."""
    schema = {
        "fm_name": None,
        "source": None,
        "source_id": None,
        "revision_or_version": None,
        "weight_file_name": None,
        "sha256": None,
        "file_size_bytes": None,
        "embedding_dim": None,
        "license": None,
        "download_url": None,
        "notes": None,
    }
    schema.update(kwargs)
    records.append(schema)


records = []
freeze_timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
print(f"Freeze timestamp (UTC): {freeze_timestamp}")

# %% [markdown]
# ## 2. DINOv2 (HuggingFace, ImageNet pretraining)

# %%
print("\n=== DINOv2 ===")
try:
    from huggingface_hub import snapshot_download
    snap = Path(snapshot_download(
        "facebook/dinov2-base",
        revision="f9e44c8",
        cache_dir=str(CACHE_DIR / "dinov2"),
    ))
    print(f"Snapshot: {snap}")

    # Find main weight file (prefer safetensors)
    weight_files = sorted(list(snap.glob("model.safetensors")) +
                          list(snap.glob("pytorch_model.bin")))
    for wf in weight_files:
        sha = sha256_file(wf)
        size = wf.stat().st_size
        print(f"  {wf.name}: {sha[:16]}... ({size/1e6:.1f} MB)")
        add_record(records,
                   fm_name="dinov2",
                   source="huggingface",
                   source_id="facebook/dinov2-base",
                   revision_or_version="f9e44c8",
                   weight_file_name=wf.name,
                   sha256=sha,
                   file_size_bytes=size,
                   embedding_dim=768,
                   license="Apache-2.0",
                   download_url="https://huggingface.co/facebook/dinov2-base",
                   notes="ViT-B/14 patch, 4-layer extraction (3, 6, 11, cls)")
except Exception as e:
    print(f"  FAILED: {type(e).__name__}: {e}")
    add_record(records, fm_name="dinov2", source="huggingface",
               source_id="facebook/dinov2-base",
               revision_or_version="f9e44c8",
               notes=f"hash failed: {type(e).__name__}: {e}")

# %% [markdown]
# ## 3. RAD-DINO (HuggingFace, CXR pretraining)

# %%
print("\n=== RAD-DINO ===")
try:
    snap = Path(snapshot_download(
        "microsoft/rad-dino",
        revision="2ec9ca0",
        cache_dir=str(CACHE_DIR / "rad_dino"),
    ))
    print(f"Snapshot: {snap}")
    weight_files = sorted(list(snap.glob("model.safetensors")) +
                          list(snap.glob("pytorch_model.bin")))
    for wf in weight_files:
        sha = sha256_file(wf)
        size = wf.stat().st_size
        print(f"  {wf.name}: {sha[:16]}... ({size/1e6:.1f} MB)")
        add_record(records,
                   fm_name="rad_dino",
                   source="huggingface",
                   source_id="microsoft/rad-dino",
                   revision_or_version="2ec9ca0",
                   weight_file_name=wf.name,
                   sha256=sha,
                   file_size_bytes=size,
                   embedding_dim=768,
                   license="MSRLA (research)",
                   download_url="https://huggingface.co/microsoft/rad-dino",
                   notes="ViT-B/14 CXR pretrained, 4-layer extraction")
except Exception as e:
    print(f"  FAILED: {type(e).__name__}: {e}")
    add_record(records, fm_name="rad_dino", source="huggingface",
               source_id="microsoft/rad-dino", revision_or_version="2ec9ca0",
               notes=f"hash failed: {type(e).__name__}: {e}")

# %% [markdown]
# ## 4. BiomedCLIP (HuggingFace, PMC-15M biomedical VL)

# %%
print("\n=== BiomedCLIP ===")
BIOMEDCLIP_ID = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
try:
    # No explicit revision pin in the embedding notebooks — record current main
    snap = Path(snapshot_download(BIOMEDCLIP_ID, cache_dir=str(CACHE_DIR / "biomedclip")))
    print(f"Snapshot: {snap}")
    # BiomedCLIP packages weights as open_clip_pytorch_model.bin
    weight_files = sorted(list(snap.glob("*.bin")) + list(snap.glob("*.safetensors")))
    for wf in weight_files:
        sha = sha256_file(wf)
        size = wf.stat().st_size
        print(f"  {wf.name}: {sha[:16]}... ({size/1e6:.1f} MB)")
        add_record(records,
                   fm_name="biomedclip",
                   source="huggingface",
                   source_id=BIOMEDCLIP_ID,
                   revision_or_version="main (no explicit pin in embedding notebooks)",
                   weight_file_name=wf.name,
                   sha256=sha,
                   file_size_bytes=size,
                   embedding_dim=512,
                   license="MIT",
                   download_url=f"https://huggingface.co/{BIOMEDCLIP_ID}",
                   notes="ViT-B/16 + PubMedBERT, single-layer (cls) extraction")
except Exception as e:
    print(f"  FAILED: {type(e).__name__}: {e}")
    add_record(records, fm_name="biomedclip", source="huggingface",
               source_id=BIOMEDCLIP_ID,
               notes=f"hash failed: {type(e).__name__}: {e}")

# %% [markdown]
# ## 5. FMCIB (Zenodo, foundation cancer-image biomarker)

# %%
print("\n=== FMCIB ===")
FMCIB_URL = "https://zenodo.org/api/records/10528450/files/model_weights.torch/content"
fmcib_path = CACHE_DIR / "fmcib_weights.torch"
try:
    if not fmcib_path.exists() or fmcib_path.stat().st_size < 1e6:
        print(f"  Downloading from Zenodo (738 MB)...")
        import requests
        r = requests.get(FMCIB_URL, stream=True, timeout=600)
        r.raise_for_status()
        with open(fmcib_path, "wb") as f:
            for chunk in r.iter_content(8 * 1024 * 1024):
                f.write(chunk)
    sha = sha256_file(fmcib_path)
    size = fmcib_path.stat().st_size
    print(f"  fmcib_weights.torch: {sha[:16]}... ({size/1e6:.1f} MB)")
    add_record(records,
               fm_name="fmcib",
               source="zenodo",
               source_id="10528450",
               revision_or_version="v1 (Zenodo deposit 2024-01)",
               weight_file_name="model_weights.torch",
               sha256=sha,
               file_size_bytes=size,
               embedding_dim=4096,
               license="CC BY 4.0",
               download_url=FMCIB_URL,
               notes="ResNet-50-2x with bias=True on downsample convs; "
                     "loaded via state['trunk_state_dict']")
except Exception as e:
    print(f"  FAILED: {type(e).__name__}: {e}")
    add_record(records, fm_name="fmcib", source="zenodo", source_id="10528450",
               notes=f"hash failed: {type(e).__name__}: {e}")

# %% [markdown]
# ## 6. CT-FM (HuggingFace, project-lighter)

# %%
print("\n=== CT-FM ===")
CTFM_ID = "project-lighter/ct_fm_feature_extractor"
try:
    snap = Path(snapshot_download(CTFM_ID, cache_dir=str(CACHE_DIR / "ct_fm")))
    print(f"Snapshot: {snap}")
    weight_files = sorted(list(snap.glob("*.ckpt")) +
                          list(snap.glob("*.bin")) +
                          list(snap.glob("*.safetensors")) +
                          list(snap.glob("*.pt")) +
                          list(snap.glob("*.pth")))
    if not weight_files:
        # Look one level deeper in case weights are in a subdirectory
        for sub in snap.iterdir():
            if sub.is_dir():
                weight_files += list(sub.rglob("*.ckpt")) + list(sub.rglob("*.bin"))
    for wf in weight_files:
        sha = sha256_file(wf)
        size = wf.stat().st_size
        print(f"  {wf.relative_to(snap)}: {sha[:16]}... ({size/1e6:.1f} MB)")
        add_record(records,
                   fm_name="ct_fm",
                   source="huggingface",
                   source_id=CTFM_ID,
                   revision_or_version="main (no explicit pin)",
                   weight_file_name=str(wf.relative_to(snap)),
                   sha256=sha,
                   file_size_bytes=size,
                   embedding_dim=512,
                   license="Apache-2.0",
                   download_url=f"https://huggingface.co/{CTFM_ID}",
                   notes="SegResEncoder, last-stage adaptive_avg_pool3d to 512-dim")
except Exception as e:
    print(f"  FAILED: {type(e).__name__}: {e}")
    add_record(records, fm_name="ct_fm", source="huggingface", source_id=CTFM_ID,
               notes=f"hash failed: {type(e).__name__}: {e}")

# %% [markdown]
# ## 7. Random Init Control (no weights to hash)

# %%
print("\n=== Random Init Control ===")
print("  Procedure: AutoModel.from_config(AutoConfig.from_pretrained('facebook/dinov2-base'))")
print("  No explicit seed set in embedding notebooks → weights non-reproducible by design")
add_record(records,
           fm_name="random_init",
           source="dinov2-config-random-init",
           source_id="facebook/dinov2-base (config only)",
           revision_or_version="N/A — random initialisation",
           weight_file_name="(none — runtime random)",
           sha256=None,
           file_size_bytes=None,
           embedding_dim=768,
           license="N/A",
           download_url=None,
           notes="Architecture: dinov2-base config; weights initialised via "
                 "AutoModel.from_config; no torch.manual_seed call in embedding "
                 "notebooks → exact weights at extraction time not recoverable. "
                 "Acceptable for control purposes (the role is to provide a "
                 "non-pretrained baseline, not to be exactly reproducible).")

# %% [markdown]
# ## 8. Merlin (attempt — known to fail at install on Kaggle)

# %%
print("\n=== Merlin ===")
try:
    subprocess.check_call(["pip", "install", "-q", "merlin-vlm"])
    from merlin import Merlin  # noqa: F401
    # If we got here, attempt to find the cached weights
    print("  Merlin imported successfully — weight location unknown without inspection")
    add_record(records,
               fm_name="merlin",
               source="huggingface (via merlin-vlm)",
               source_id="merlin-vlm package",
               revision_or_version="latest (pip install merlin-vlm)",
               weight_file_name="(unknown — needs Merlin inspection)",
               notes="Module imported but weight file not located in this run. "
                     "Needs follow-up. NOT used in v3 embeddings (consistently "
                     "failed to install on Kaggle during embedding extraction).")
except Exception as e:
    print(f"  Could not load Merlin: {e}")
    add_record(records,
               fm_name="merlin",
               source="huggingface (via merlin-vlm)",
               source_id="merlin-vlm package",
               revision_or_version="N/A",
               notes=f"NOT used in v3 embeddings — install failed: "
                     f"{type(e).__name__}: {e}")

# %% [markdown]
# ## 9. Save manifest

# %%
df = pd.DataFrame(records)
manifest_path = OUT_DIR / "checkpoint_manifest.csv"
df.to_csv(manifest_path, index=False)

# Also save a metadata sidecar with provenance
metadata = {
    "freeze_timestamp_utc": freeze_timestamp,
    "freeze_purpose": "Pre-registration §11 — FM weight SHA-256 manifest before probe analysis",
    "osf_doi": "10.17605/OSF.IO/DQ2JA",
    "n_fms_recorded": len(df["fm_name"].unique()),
    "n_fms_with_sha256": int(df["sha256"].notna().sum()),
    "n_files_hashed": int((df["sha256"].notna()).sum()),
    "v3_embedding_datasets_locked": [
        "pet-fm-bench-t4-embeddings-v3",
        "pet-fm-bench-t6-embeddings-v3",
        "pet-fm-bench-t7-embeddings-v3",
        "pet-fm-bench-t8-embeddings-v3",
        "pet-fm-bench-t9-embeddings-v3",
    ],
}
import json
sidecar_path = OUT_DIR / "checkpoint_manifest.metadata.json"
with open(sidecar_path, "w") as f:
    json.dump(metadata, f, indent=2)

# %% [markdown]
# ## 10. Final summary

# %%
print(f"\n{'='*70}")
print(f"PHASE 1 FREEZE — CHECKPOINT MANIFEST")
print(f"{'='*70}\n")
display_cols = ["fm_name", "source_id", "revision_or_version",
                "weight_file_name", "sha256", "file_size_bytes", "embedding_dim"]
have = [c for c in display_cols if c in df.columns]
print(df[have].to_string(index=False))

print(f"\nFM coverage: {df['fm_name'].nunique()} FMs in manifest")
print(f"  with SHA-256 hashes: {df['sha256'].notna().sum()}")
print(f"  documented but unhashable: {df['sha256'].isna().sum()}")
print(f"\nFreeze timestamp (UTC): {freeze_timestamp}")
print(f"\nOutput files:")
for f in sorted(OUT_DIR.glob("checkpoint_manifest*")):
    print(f"  {f.name} ({f.stat().st_size/1e3:.1f} KB)")

# %% [markdown]
# ## 11. Done — upload to OSF
#
# 1. Save & Run All to commit this notebook
# 2. Download the two output files (`checkpoint_manifest.csv` +
#    `checkpoint_manifest.metadata.json`) from the Output tab
# 3. Upload both to OSF project [aqmkb](https://osf.io/aqmkb/) under a folder
#    named `phase_1_freeze_checkpoint_manifest`
# 4. Record the OSF upload date in PROGRESS.md
# 5. Phase 1 freeze gate is now satisfied — Phase 2 (contamination audit) is
#    the next research-phase block before the formal probe analysis run
