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
# # PET-FM-Bench: Multi-Seed Random-Init Re-Extraction
#
# **Runtime:** GPU (T4) | **Internet:** ON | **Time:** ~10-30 min per task |
# **GPU quota:** ~1-3 GPU-hours per task × 5 tasks = ~10-15 GPU-hours total
#
# Re-extracts random-initialised DINOv2 embeddings with **N independent seeds**
# (default N=10) so the formal probe baseline can report median + IQR over
# seeds instead of a single realisation. Without this, `random_init` is a
# single weight draw and not a defensible baseline — dry-run results showed
# it dominating most tasks at AUROC 0.72-0.97, which a multi-seed report
# would contextualise as one tail of a distribution.
#
# **What this notebook produces:**
# - `random_init_seed{0..N-1}.parquet` files in `/kaggle/working/embeddings/`
#   for the chosen task. Each parquet has the same schema as the existing
#   single-seed `random_init.parquet` already in the v3 embedding datasets.
#
# **How to use:**
# 1. Set `TASK` below to one of `t4`, `t6`, `t7`, `t8`, `t9`.
# 2. Attach the corresponding `pet-fm-bench-tX-patches-v3` dataset.
# 3. Save & Run All. Wait for completion.
# 4. Output → "New Dataset" → name it `pet-fm-bench-tX-randominit-multiseed`.
# 5. Repeat for each of the 5 tasks (one Kaggle notebook commit per task).
# 6. In `probe_analysis.py`, attach all 5 multi-seed datasets alongside the
#    existing v3 embedding datasets — the new aggregation logic will detect
#    the per-seed parquets and report median + IQR.

# %% [markdown]
# ## 1. Setup

# %%
TASK = "t8"        # one of: t4, t6, t7, t8, t9
N_SEEDS = 10       # number of independent random initialisations
SEED_OFFSET = 0    # set to N_SEEDS for a follow-up batch (seeds 10..19)

# %%
import os
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE} | Task: {TASK} | Seeds: "
      f"{SEED_OFFSET}..{SEED_OFFSET + N_SEEDS - 1}")

# %% [markdown]
# ## 2. Locate task patches dataset

# %%
# Locate patches dataset.
# - T4/T6/T7/T8 have v3 patches (post-SUV-pipeline-rebuild): `pet-fm-bench-{TASK}-patches-v3`
# - T9 has no v3 (Vienna QUADRA was SUV-converted at source on Zenodo, never had
#   the Bq/mL overflow bug that triggered the v3 rebuild for the others).
#   Use `pet-fm-bench-t9-patches` (v1 naming).
V3_CANDIDATES = list(Path("/kaggle/input").rglob(f"pet-fm-bench-{TASK}-patches-v3"))
V1_CANDIDATES = list(Path("/kaggle/input").rglob(f"pet-fm-bench-{TASK}-patches"))
candidates = V3_CANDIDATES or V1_CANDIDATES
if not candidates:
    raise FileNotFoundError(
        f"No patches dataset for {TASK} found. Tried "
        f"`pet-fm-bench-{TASK}-patches-v3` and `pet-fm-bench-{TASK}-patches`. "
        f"Attach via Add Data → search."
    )
PATCH_DIR = candidates[0]
print(f"Patch dataset: {PATCH_DIR.name} "
      f"({'v3' if 'v3' in PATCH_DIR.name else 'v1 — T9 by design, not a missing v3'})")

manifest_path = list(PATCH_DIR.rglob("manifest.csv"))[0]
PATCH_DIR = manifest_path.parent
manifest = pd.read_csv(manifest_path)
print(f"Patch dir: {PATCH_DIR}")
print(f"Manifest: {len(manifest)} rows")

# Patient-ID column name: T9 uses `subject_id` (healthy-control naming),
# all other tasks use `patient_id`. Standardise on `patient_id` internally.
ID_COL = None
for cand in ["patient_id", "subject_id"]:
    if cand in manifest.columns:
        ID_COL = cand
        break
if ID_COL is None:
    raise ValueError(
        f"manifest.csv has no patient_id or subject_id column. "
        f"Columns: {list(manifest.columns)}"
    )
if ID_COL != "patient_id":
    manifest = manifest.rename(columns={ID_COL: "patient_id"})
    print(f"Renamed column `{ID_COL}` → `patient_id` for downstream uniformity")

# Single-session vs multi-session detection
SESSION_COL = None
for cand in ["session", "study_index", "scan_id"]:
    if cand in manifest.columns:
        SESSION_COL = cand
        break
print(f"Session column: {SESSION_COL or '(single-session task)'}")

# %% [markdown]
# ## 3. Patch-existence filter
#
# Mirrors the file-existence filter from `tX_02_embeddings.py`. The MIP
# directory layout for multi-session tasks differs slightly (subdir per
# session); we handle both.

# %%
def _candidate_mips_paths(pid, sess):
    """Return all candidate mips.npz paths for a (patient_id, session) tuple.

    Per-task naming conventions observed:
    - T6, T9: directory named after raw session/subject_id value
              (e.g., `mips_2d/SUBJ_001/Test/mips.npz`)
    - T7:     directory named `study_{study_index}`
              (e.g., `mips_2d/PATIENT_001/study_0/mips.npz`)
    - other:  flattened or prefixed variants
    """
    return [
        PATCH_DIR / "mips_2d" / pid / str(sess) / "mips.npz",
        PATCH_DIR / "mips_2d" / pid / f"study_{sess}" / "mips.npz",   # T7
        PATCH_DIR / "mips_2d" / pid / f"session_{sess}" / "mips.npz",
        PATCH_DIR / "mips_2d" / pid / f"{sess}.npz",
        PATCH_DIR / "mips_2d" / f"{pid}_{sess}" / "mips.npz",
    ]


def _has_mips(row):
    pid = row["patient_id"]
    if SESSION_COL:
        candidates = _candidate_mips_paths(pid, row[SESSION_COL])
    else:
        candidates = [PATCH_DIR / "mips_2d" / pid / "mips.npz"]
    return any(c.exists() for c in candidates)


def _mips_path(row):
    pid = row["patient_id"]
    if SESSION_COL:
        for cand in _candidate_mips_paths(pid, row[SESSION_COL]):
            if cand.exists():
                return cand
        return None
    return PATCH_DIR / "mips_2d" / pid / "mips.npz"


mask = manifest.apply(_has_mips, axis=1)
dropped = int((~mask).sum())
manifest = manifest[mask].reset_index(drop=True)
print(f"After file-existence filter: {len(manifest)} rows ({dropped} dropped)")


def load_mips(row):
    path = _mips_path(row)
    if path is None:
        return None
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


# %% [markdown]
# ## 4. Multi-seed extraction loop
#
# For each seed: set `torch.manual_seed(s)`, instantiate a fresh DINOv2-config
# model with random weights, run inference over all (patient, session, view)
# tuples, save as `random_init_seed{s}.parquet`.
#
# The architecture matches the single-seed `run_random_init` in
# `tX_02_embeddings.py` exactly: `facebook/dinov2-base` config, 768-dim ViT-B/14,
# CLS-token output. This is critical for the seed-aggregated baseline to be a
# valid replacement for the single-seed parquet.

# %%
EMBED_DIR = Path("/kaggle/working/embeddings")
EMBED_DIR.mkdir(parents=True, exist_ok=True)

from transformers import AutoConfig, AutoModel  # noqa: E402

for seed in range(SEED_OFFSET, SEED_OFFSET + N_SEEDS):
    print(f"\n{'='*60}\nSeed {seed}\n{'='*60}")

    # Determinism: set the seed BEFORE instantiating the model, so each seed
    # produces a fully distinct weight draw. cuDNN nondeterminism is fine for
    # a baseline (it only adds within-seed noise, not between-seed bias).
    torch.manual_seed(seed)
    np.random.seed(seed)

    config = AutoConfig.from_pretrained("facebook/dinov2-base")
    model = AutoModel.from_config(config).to(DEVICE).eval()

    rows = []
    for _, r in tqdm(manifest.iterrows(), total=len(manifest),
                     desc=f"seed{seed}"):
        mips = load_mips(r)
        if mips is None:
            continue
        for view, arr in mips.items():
            inp = mip_to_rgb_tensor(arr).to(DEVICE)
            with torch.no_grad():
                out = model(inp)
            emb = out.last_hidden_state[0, 0].cpu().numpy()
            rec = {"patient_id": r["patient_id"], "view": view, "layer": "cls"}
            if SESSION_COL:
                rec[SESSION_COL] = r[SESSION_COL]
            for j, v in enumerate(emb):
                rec[f"d{j:04d}"] = float(v)
            rows.append(rec)

    df = pd.DataFrame(rows)
    out_path = EMBED_DIR / f"random_init_seed{seed}.parquet"
    df.to_parquet(out_path, index=False)
    print(f"  Saved: {out_path.name} "
          f"({out_path.stat().st_size/1e6:.1f} MB, {len(df)} rows)")

    # Free GPU memory before the next seed.
    del model
    torch.cuda.empty_cache()

# %% [markdown]
# ## 5. Summary

# %%
print(f"\n{'='*60}\nMulti-seed extraction complete\n{'='*60}")
total_kb = 0
for f in sorted(EMBED_DIR.glob("random_init_seed*.parquet")):
    kb = f.stat().st_size / 1e3
    total_kb += kb
    print(f"  {f.name}: {kb:.0f} KB")
print(f"\nTotal: {total_kb/1e3:.1f} MB across {N_SEEDS} seeds for task {TASK}")
print(f"\nNext step: Save & Run All → Output → 'New Dataset' → "
      f"name as 'pet-fm-bench-{TASK}-randominit-multiseed'.")
