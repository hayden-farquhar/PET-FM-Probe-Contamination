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
# # PET-FM-Bench: T4 calibration dump (fix-up; Amendment A13)
#
# **Purpose:** Re-run **only T4 (NSCLC OS Cox)** at SEED = 42 with the corrected
# `get_patient_embedding` dispatch. The full v2 calibration_inference notebook
# pre-filtered embeddings by `(view, layer)` and then called
# `get_patient_embedding(sub, pid)` with the helper's default args
# (`view="coronal" layer="cls"`). For 2D FMs the helper saw no `coronal` rows in
# the pre-filtered slice and returned `None` for every patient; only FMCIB and
# CT-FM (which use `view=volume`) survived.
#
# **Fix:** match the registered `probe_analysis.py` v6 code path — pass the
# unfiltered `fm_df` to `get_patient_embedding` and let the helper handle
# view/layer selection internally.
#
# **Runtime:** ~5 min. CPU only.
#
# **Datasets to attach (3 — all required for the full 15-FM dump):**
# - `pet-fm-bench-t4-embeddings-v3`           (5 FMs + legacy random_init)
# - `pet-fm-bench-t4-randominit-multiseed`    (10 random_init seeds; A3 baseline)
# - `pet-fm-bench-task-splits-v4`             (patient-level splits)
#
# **NOTE:** The previous T4-fix attempt missed the multiseed dataset; that
# produced only 6 FMs (5 + legacy random_init) instead of the registered 15
# (5 + 10 multiseed entries). Make sure all three are attached.
#
# **Output (in `/kaggle/working/`):** `perpred_t4.parquet` — full 15-FM dump.

# %%
import os
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning, module="sksurv")

try:
    from sksurv.linear_model import CoxPHSurvivalAnalysis
    from sksurv.metrics import concordance_index_censored
except ImportError:
    print("Installing scikit-survival ...")
    os.system("pip install -q scikit-survival")
    from sksurv.linear_model import CoxPHSurvivalAnalysis
    from sksurv.metrics import concordance_index_censored

SEED = 42
N_FOLDS = 5
ALPHA_GRID = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
HORIZON_MONTHS = 24

OUTPUT_DIR = Path("/kaggle/working")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Helpers (verbatim from registered probe_analysis.py v6)

# %%
RANDOM_INIT_SEED_PATTERN = re.compile(
    r"^random_init[_-]?(?:seed)?[_-]?(\d+)$", re.IGNORECASE
)


def find_embed_dir(task_pattern):
    v3_patterns = [
        f"pet-fm-bench-{task_pattern}-embeddings-v3",
        f"pet-fm-bench-{task_pattern}-2025-embeddings-v3",
    ]
    for v3_pattern in v3_patterns:
        v3_candidates = list(Path("/kaggle/input").rglob(v3_pattern))
        if v3_candidates:
            embed_dir = v3_candidates[0]
            nested = embed_dir / "embeddings"
            if nested.exists():
                return nested
            for sub in embed_dir.iterdir():
                if sub.is_dir():
                    nested2 = sub / "embeddings"
                    if nested2.exists():
                        return nested2
            return embed_dir
    return None


def load_fm_embeddings(embed_dir):
    if embed_dir is None:
        return {}
    fm_data = {}
    for f in sorted(embed_dir.glob("*.parquet")):
        fm_name = f.stem
        if fm_name.endswith("_labels"):
            continue
        df = pd.read_parquet(f)
        fm_data[fm_name] = df
    return fm_data


def load_labels(embed_dir, label_pattern="*labels*"):
    if embed_dir is None:
        return None
    candidates_csv = list(embed_dir.glob(label_pattern + ".csv"))
    if not candidates_csv:
        candidates_csv = list(embed_dir.glob(label_pattern))
        candidates_csv = [c for c in candidates_csv if c.suffix == ".csv"]
    if not candidates_csv:
        candidates_csv = list(embed_dir.parent.rglob(label_pattern + ".csv"))
    if candidates_csv:
        return pd.read_csv(candidates_csv[0])
    candidates_parquet = list(embed_dir.glob(label_pattern + ".parquet"))
    if not candidates_parquet:
        candidates_parquet = list(embed_dir.glob(label_pattern))
        candidates_parquet = [c for c in candidates_parquet if c.suffix == ".parquet"]
    if not candidates_parquet:
        candidates_parquet = list(embed_dir.parent.rglob(label_pattern + ".parquet"))
    if candidates_parquet:
        return pd.read_parquet(candidates_parquet[0])
    return None


def find_multiseed_dir(task):
    for slug in [
        f"pet-fm-bench-{task}-randominit-multiseed-v3",
        f"pet-fm-bench-{task}-randominit-multiseed",
    ]:
        candidates = list(Path("/kaggle/input").rglob(slug))
        if candidates:
            return candidates[0]
    return None


def load_multiseed_random_init(multiseed_dir):
    if multiseed_dir is None:
        return {}
    candidates = list(multiseed_dir.glob("random_init*.parquet"))
    if not candidates:
        nested = multiseed_dir / "embeddings"
        if nested.exists():
            candidates = list(nested.glob("random_init*.parquet"))
    if not candidates:
        candidates = list(multiseed_dir.rglob("random_init*.parquet"))
    seed_data = {}
    for f in sorted(candidates):
        fm_name = f.stem
        if not RANDOM_INIT_SEED_PATTERN.match(fm_name):
            continue
        df = pd.read_parquet(f)
        seed_data[fm_name] = df
    return seed_data


def get_patient_embedding(df, patient_id, view="coronal", layer="cls"):
    """Verbatim from registered probe_analysis.py v6."""
    mask = df["patient_id"] == patient_id
    if "view" in df.columns:
        if "volume" in df["view"].values:
            mask = mask & (df["view"] == "volume")
        else:
            mask = mask & (df["view"] == view)
    if "layer" in df.columns:
        if "pool" in df[df["patient_id"] == patient_id]["layer"].values:
            mask = mask & (df["layer"] == "pool")
        else:
            mask = mask & (df["layer"] == layer)
    rows = df[mask]
    if len(rows) == 0:
        return None
    dim_cols = [c for c in df.columns if c.startswith("d")]
    return rows[dim_cols].values[0]


# %% [markdown]
# ## Load T4 embeddings + labels

# %%
T4_EMBED_DIR = find_embed_dir("t4")
print(f"T4 embedding dir: {T4_EMBED_DIR}")

T4_EMBEDDINGS = load_fm_embeddings(T4_EMBED_DIR)
print(f"After loading t4-embeddings-v3: {len(T4_EMBEDDINGS)} FMs "
      f"({sorted(T4_EMBEDDINGS.keys())})")

# Load the t4-randominit-multiseed dataset and merge in the 10 seeds
T4_MULTISEED_DIR = find_multiseed_dir("t4")
print(f"T4 multiseed dir: {T4_MULTISEED_DIR}")
if T4_MULTISEED_DIR is None:
    print("\n⚠⚠⚠ MULTISEED DATASET NOT FOUND ⚠⚠⚠")
    print("  Attach `pet-fm-bench-t4-randominit-multiseed` (22 MB) before running.")
    print("  Without it, only the 5 main FMs + legacy random_init will be processed.")
else:
    seed_data = load_multiseed_random_init(T4_MULTISEED_DIR)
    T4_EMBEDDINGS.update(seed_data)
    print(f"After loading t4-randominit-multiseed: {len(T4_EMBEDDINGS)} FMs total "
          f"(+{len(seed_data)} multiseed entries)")

multiseed_keys = [k for k in T4_EMBEDDINGS if RANDOM_INIT_SEED_PATTERN.match(k)]
print(f"\nT4 random_init_seed* keys loaded: {len(multiseed_keys)} (expected 10)")
print(f"T4 random_init_seed* names: {sorted(multiseed_keys)}")

# Drop legacy single-seed `random_init` if multiseed entries exist (per A3)
if "random_init" in T4_EMBEDDINGS and len(multiseed_keys) >= 2:
    del T4_EMBEDDINGS["random_init"]
    print("Dropped legacy single-seed `random_init` (10 multiseed entries present; A3 baseline)")

print(f"\n=== Final T4 FMs ({len(T4_EMBEDDINGS)}, expected 15) ===")
for k in sorted(T4_EMBEDDINGS.keys()):
    print(f"  {k}: {len(T4_EMBEDDINGS[k])} rows")

T4_LABELS = load_labels(T4_EMBED_DIR)
print(f"\nT4 label columns: {list(T4_LABELS.columns)}")

# Resolve event/time columns
event_col = next((c for c in ["event", "death", "os_event"] if c in T4_LABELS.columns), None)
time_col = next((c for c in ["time_to_death", "os_time", "time"] if c in T4_LABELS.columns), None)
print(f"event_col = {event_col}, time_col = {time_col}")

# %% [markdown]
# ## T4 Cox dump (FIXED: uses fm_df not sub for get_patient_embedding)

# %%
def run_t4_cox_with_dump(embeddings_dict, labels_df, event_col, time_col,
                          horizon_months=HORIZON_MONTHS):
    """Per-patient Cox dump for T4. **Fix vs v2:** passes fm_df (unfiltered) to
    get_patient_embedding instead of pre-filtered sub — matches registered
    probe_analysis.py run_survival_probe code path."""

    # Per-patient survival metadata
    patient_meta = (labels_df.drop_duplicates("patient_id")
                            [["patient_id", event_col, time_col]]
                            .reset_index(drop=True))
    patient_meta = patient_meta.dropna(subset=[event_col, time_col])
    patient_meta = patient_meta[patient_meta[time_col] > 0]
    print(f"  T4: {len(patient_meta)} patients with valid survival, "
          f"{int(patient_meta[event_col].sum())} events "
          f"({patient_meta[event_col].mean()*100:.1f}%)")

    horizon_days = horizon_months * 30.44
    out_rows = []

    for fm_name, fm_df in embeddings_dict.items():
        # FIX: do NOT pre-filter fm_df. Pass directly to get_patient_embedding,
        # which handles view/layer selection internally via the registered convention.
        id_col = "patient_id" if "patient_id" in fm_df.columns else "subject_id"
        patients = sorted(fm_df[id_col].unique())

        X_list, time_list, ev_list, pid_list = [], [], [], []
        for pid in patients:
            emb = get_patient_embedding(fm_df, pid)
            if emb is None:
                continue
            p_meta = patient_meta[patient_meta["patient_id"].astype(str) == str(pid)]
            if len(p_meta) == 0:
                continue
            X_list.append(emb)
            ev_list.append(int(p_meta[event_col].values[0]))
            time_list.append(float(p_meta[time_col].values[0]))
            pid_list.append(str(pid))

        if len(X_list) < 20:
            print(f"  t4 {fm_name}: too few samples ({len(X_list)}), skip")
            continue

        X = np.array(X_list)
        time_arr = np.array(time_list)
        event_arr = np.array(ev_list)

        nan_mask = np.isnan(X).any(axis=1)
        if nan_mask.sum() > 0:
            X = X[~nan_mask]
            time_arr = time_arr[~nan_mask]
            event_arr = event_arr[~nan_mask]
            pid_list = [p for p, m in zip(pid_list, ~nan_mask) if m]
        np.nan_to_num(X, copy=False)
        if len(X) < 20:
            continue

        n_events = int(event_arr.sum())
        epv_ratio = X.shape[1] / max(1, n_events)
        use_pca = (X.shape[1] > X.shape[0]) or (n_events < 100) or (epv_ratio > 3)
        pca_dim = min(50, X.shape[0] - 1, X.shape[1]) if use_pca else None

        scaler = StandardScaler()
        X_s = scaler.fit_transform(X)
        if use_pca:
            pca = PCA(n_components=pca_dim, random_state=SEED)
            X_s = pca.fit_transform(X_s)

        cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        best_alpha, best_c = 1.0, -1.0
        for alpha in ALPHA_GRID:
            cidx_folds = []
            for tr_idx, va_idx in cv.split(X_s, event_arr):
                if event_arr[va_idx].sum() < 1:
                    continue
                y_tr = np.array(
                    [(bool(e), float(t)) for e, t in zip(event_arr[tr_idx], time_arr[tr_idx])],
                    dtype=[("event", bool), ("time", float)])
                y_va = np.array(
                    [(bool(e), float(t)) for e, t in zip(event_arr[va_idx], time_arr[va_idx])],
                    dtype=[("event", bool), ("time", float)])
                cph = CoxPHSurvivalAnalysis(alpha=alpha)
                try:
                    cph.fit(X_s[tr_idx], y_tr)
                    risk = cph.predict(X_s[va_idx])
                    c = concordance_index_censored(y_va["event"], y_va["time"], risk)[0]
                    cidx_folds.append(c)
                except Exception:
                    continue
            if cidx_folds:
                mean_c = float(np.mean(cidx_folds))
                if mean_c > best_c:
                    best_c, best_alpha = mean_c, alpha

        y_full = np.array(
            [(bool(e), float(t)) for e, t in zip(event_arr, time_arr)],
            dtype=[("event", bool), ("time", float)])
        cph_final = CoxPHSurvivalAnalysis(alpha=best_alpha)
        try:
            cph_final.fit(X_s, y_full)
        except Exception as e:
            print(f"  t4 {fm_name}: final fit failed: {e}")
            continue
        lp = cph_final.predict(X_s)
        surv_funcs = cph_final.predict_survival_function(X_s)
        surv_at_h = np.array([float(sf(horizon_days)) for sf in surv_funcs])

        for pid, ev, tt, lpi, si in zip(pid_list, event_arr, time_arr, lp, surv_at_h):
            out_rows.append({
                "fm": fm_name, "task": "t4",
                "patient_id": str(pid),
                "event": int(ev), "time": float(tt),
                "linear_predictor": float(lpi),
                f"surv_{horizon_months}mo": float(si),
                "best_alpha": float(best_alpha),
                "use_pca": bool(use_pca),
                "n_pca": int(pca_dim) if use_pca else 0,
            })
        print(f"  t4 {fm_name}: {len(pid_list)} per-patient predictions dumped "
              f"(alpha={best_alpha}, c-index={best_c:.3f})")
    return pd.DataFrame(out_rows)


print("\n" + "=" * 60)
print(f"T4 calibration dump — FIX-UP (NSCLC Cox OS, {HORIZON_MONTHS} mo)")
print("=" * 60)
t4_dump = run_t4_cox_with_dump(T4_EMBEDDINGS, T4_LABELS, event_col, time_col)
print(f"\nTotal rows: {len(t4_dump)}")
print(f"FMs successful: {sorted(t4_dump['fm'].unique()) if len(t4_dump) else []}")

t4_dump.to_parquet(OUTPUT_DIR / "perpred_t4.parquet", index=False)

# Manifest with hash
import hashlib
def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

f = OUTPUT_DIR / "perpred_t4.parquet"
print(f"\nperpred_t4.parquet: {len(t4_dump):,} rows, "
      f"{t4_dump['fm'].nunique() if len(t4_dump) else 0} FMs, "
      f"SHA-256 = {file_sha256(f)}, "
      f"{f.stat().st_size:,} bytes")
