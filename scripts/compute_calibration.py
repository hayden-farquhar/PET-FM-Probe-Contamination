#!/usr/bin/env python3
"""Compute calibration metrics for PET-FM-Bench Phase 5 calibration sub-study (A13).

Reads per-prediction parquets produced by `kaggle_notebooks/calibration_inference.py`
and computes:

  - Brier score (binary tasks only)
  - Calibration intercept (a0) and slope (b)
  - Reliability diagram bin counts + observed risk per quantile-decile
  - Expected Calibration Error (ECE, 10 equal-width bins)
  - For Cox tasks: IPCW-weighted observed survival in deciles of predicted survival

All metrics are reported as point estimates with 95 % CI from 1,000 patient-clustered
bootstrap iterations at SEED = 42 (same resampler convention as the registered
probe pipeline).

Output: results/calibration/calibration_results.csv
        results/calibration/reliability_bins_t{1,5,7}.parquet
        results/calibration/cox_calibration_t{3,4}.parquet

Inputs expected at:  data/calibration_perpred/perpred_t{1,3,4,5,7}.parquet

Usage:
    python3 scripts/compute_calibration.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

RANDOM_INIT_SEED_PATTERN = re.compile(
    r"^random_init[_-]?(?:seed)?[_-]?(\d+)$", re.IGNORECASE
)


def aggregate_random_init_seeds(rows: pd.DataFrame) -> pd.DataFrame:
    """Collapse random_init_seed{N} rows to a single random_init row using
    seed-median for the central estimate and seed-IQR for CI bounds.

    Matches the registered probe_analysis.py v6 aggregate_random_init_seeds
    convention so the calibration eTable's random_init line is comparable to
    the freeze CSVs' random_init line.
    """
    if rows.empty:
        return rows
    is_seed = rows["fm"].apply(lambda s: bool(RANDOM_INIT_SEED_PATTERN.match(str(s))))
    if is_seed.sum() == 0:
        return rows
    main = rows[~is_seed].copy()
    seeds = rows[is_seed].copy()
    # Group seeds per task
    agg_rows = []
    for task in seeds["task"].unique():
        sub = seeds[seeds["task"] == task]
        agg = {"fm": "random_init", "task": task, "n_seeds": len(sub)}
        for col in sub.columns:
            if col in ("fm", "task"):
                continue
            if pd.api.types.is_numeric_dtype(sub[col]):
                vals = sub[col].dropna().values
                if len(vals) == 0:
                    agg[col] = float("nan")
                elif col.endswith("_lo"):
                    agg[col] = float(np.percentile(vals, 25))
                elif col.endswith("_hi"):
                    agg[col] = float(np.percentile(vals, 75))
                else:
                    agg[col] = float(np.median(vals))
            else:
                agg[col] = sub[col].iloc[0]
        agg_rows.append(agg)
    return pd.concat([main, pd.DataFrame(agg_rows)], ignore_index=True, sort=False)

try:
    from lifelines.utils import concordance_index
except ImportError:
    concordance_index = None

SEED = 42
N_BOOTSTRAP = 1000
N_BINS = 10

ROOT = Path(__file__).resolve().parent.parent
INPUT_DIR = ROOT / "data" / "calibration_perpred"
OUTPUT_DIR = ROOT / "results" / "calibration"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ─── Binary calibration metrics ────────────────────────────────────────────────


def calibration_intercept_slope(y_true: np.ndarray, y_proba: np.ndarray):
    """Logistic regression of y on logit(p). Intercept/slope: 0/1 = perfect."""
    eps = 1e-9
    p = np.clip(y_proba, eps, 1 - eps)
    logit = np.log(p / (1 - p)).reshape(-1, 1)
    if len(np.unique(y_true)) < 2:
        return float("nan"), float("nan")
    lr = LogisticRegression(C=1e9, solver="lbfgs", max_iter=1000, random_state=SEED)
    lr.fit(logit, y_true)
    return float(lr.intercept_[0]), float(lr.coef_[0][0])


def expected_calibration_error(y_true: np.ndarray, y_proba: np.ndarray, n_bins: int = N_BINS) -> float:
    """ECE with equal-width bins on [0, 1]."""
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (y_proba >= lo) & (y_proba < hi if i < n_bins - 1 else y_proba <= hi)
        if mask.sum() == 0:
            continue
        avg_p = y_proba[mask].mean()
        avg_y = y_true[mask].mean()
        ece += (mask.sum() / n) * abs(avg_p - avg_y)
    return float(ece)


def reliability_bins(y_true: np.ndarray, y_proba: np.ndarray, n_bins: int = N_BINS) -> pd.DataFrame:
    """Quantile-binned reliability diagram input — n_bins of equal patient count."""
    n = len(y_true)
    if n < n_bins:
        return pd.DataFrame()
    order = np.argsort(y_proba)
    y_sorted = y_true[order]
    p_sorted = y_proba[order]
    rows = []
    for i in range(n_bins):
        lo = int(np.floor(i * n / n_bins))
        hi = int(np.floor((i + 1) * n / n_bins))
        if hi <= lo:
            continue
        rows.append({
            "bin": i + 1,
            "n": hi - lo,
            "mean_p": float(p_sorted[lo:hi].mean()),
            "obs_p": float(y_sorted[lo:hi].mean()),
            "p_low": float(p_sorted[lo]),
            "p_high": float(p_sorted[hi - 1]),
        })
    return pd.DataFrame(rows)


def patient_clustered_bootstrap_all(y_true: np.ndarray, y_proba: np.ndarray,
                                     pid_array: np.ndarray, n_iters: int = N_BOOTSTRAP,
                                     seed: int = SEED):
    """Single-pass bootstrap that computes all 4 binary calibration metrics
    (Brier, ECE, intercept, slope) per resample. Returns dict of (lo, hi) per
    metric.

    Optimisation: precompute patient_id -> index array once; per iteration just
    np.concatenate of the 1-D index arrays. ~1000x faster than np.where per pid.
    """
    pids, pid_inverse = np.unique(pid_array, return_inverse=True)
    pid_to_idx = {i: np.where(pid_inverse == i)[0] for i in range(len(pids))}
    rng = np.random.default_rng(seed)

    metrics = {"brier": [], "ece": [], "intercept": [], "slope": []}
    for _ in range(n_iters):
        sampled = rng.integers(0, len(pids), size=len(pids))
        idx = np.concatenate([pid_to_idx[i] for i in sampled])
        if len(idx) == 0:
            continue
        y = y_true[idx]; p = y_proba[idx]
        if len(np.unique(y)) < 2:
            continue
        try:
            metrics["brier"].append(brier_score_loss(y, p))
            metrics["ece"].append(expected_calibration_error(y, p))
            a0, b = calibration_intercept_slope(y, p)
            metrics["intercept"].append(a0)
            metrics["slope"].append(b)
        except Exception:
            continue

    out = {}
    for k, vals in metrics.items():
        if not vals:
            out[k] = (float("nan"), float("nan"))
        else:
            out[k] = (float(np.percentile(vals, 2.5)),
                      float(np.percentile(vals, 97.5)))
    return out


# ─── IPCW Cox calibration (for T3, T4) ─────────────────────────────────────────


def km_per_decile(time: np.ndarray, event: np.ndarray, surv_h: np.ndarray, horizon_days: float) -> pd.DataFrame:
    """Kaplan-Meier observed survival at horizon per decile of predicted survival.

    Standard Cox calibration plot input. Within each decile (binned by predicted
    survival), compute a Kaplan-Meier estimate of survival at `horizon_days` —
    handles right-censoring correctly without requiring IPCW reweighting.
    Returns a DataFrame with one row per decile.
    """
    n = len(time)
    if n < N_BINS:
        return pd.DataFrame()

    order_p = np.argsort(surv_h)
    time_s = time[order_p]
    event_s = event[order_p]
    surv_s = surv_h[order_p]

    rows = []
    for i in range(N_BINS):
        lo = int(np.floor(i * n / N_BINS))
        hi = int(np.floor((i + 1) * n / N_BINS))
        if hi <= lo:
            continue
        # KM survival at horizon for this decile
        t_d = time_s[lo:hi]
        e_d = event_s[lo:hi]
        # Sort decile by time
        order_t = np.argsort(t_d)
        t_dd = t_d[order_t]
        e_dd = e_d[order_t]
        n_d = len(t_dd)
        s = 1.0
        at_risk = n_d
        for k in range(n_d):
            if t_dd[k] > horizon_days:
                break
            if e_dd[k] == 1 and at_risk > 0:
                s *= (1 - 1.0 / at_risk)
            at_risk -= 1
        rows.append({
            "bin": i + 1,
            "n": hi - lo,
            "n_events": int((e_dd[t_dd <= horizon_days] == 1).sum()),
            "mean_pred_surv": float(surv_s[lo:hi].mean()),
            "obs_surv_km": float(s),
            "surv_low": float(surv_s[lo:hi].min()),
            "surv_high": float(surv_s[lo:hi].max()),
        })
    return pd.DataFrame(rows)


# ─── Per-task drivers ──────────────────────────────────────────────────────────


def process_binary_task(parquet_path: Path, task: str) -> pd.DataFrame:
    if not parquet_path.exists():
        print(f"  [skip] {parquet_path.name}: not found")
        return pd.DataFrame()
    df = pd.read_parquet(parquet_path)
    if df.empty:
        return pd.DataFrame()
    print(f"  {task}: {len(df)} predictions across {df['fm'].nunique()} FMs")

    rows = []
    bins_all = []
    for fm in sorted(df["fm"].unique()):
        sub = df[df["fm"] == fm].reset_index(drop=True)
        y = sub["y_true"].values.astype(int)
        p = sub["y_proba"].values.astype(float)
        if len(y) < N_BINS or len(np.unique(y)) < 2:
            continue
        brier = brier_score_loss(y, p)
        a0, b = calibration_intercept_slope(y, p)
        ece = expected_calibration_error(y, p)

        bin_df = reliability_bins(y, p)
        bin_df["fm"] = fm
        bin_df["task"] = task
        bins_all.append(bin_df)

        # Single-pass bootstrap for all 4 binary calibration metrics
        ci = patient_clustered_bootstrap_all(
            y, p, sub["patient_id"].values)
        brier_lo, brier_hi = ci["brier"]
        ece_lo, ece_hi = ci["ece"]
        a0_lo, a0_hi = ci["intercept"]
        b_lo, b_hi = ci["slope"]

        rows.append({
            "fm": fm, "task": task,
            "n": len(sub), "n_patients": sub["patient_id"].nunique(),
            "brier": brier, "brier_lo": brier_lo, "brier_hi": brier_hi,
            "ece": ece, "ece_lo": ece_lo, "ece_hi": ece_hi,
            "intercept": a0, "intercept_lo": a0_lo, "intercept_hi": a0_hi,
            "slope": b, "slope_lo": b_lo, "slope_hi": b_hi,
        })
        print(f"    {fm:>20s}: Brier={brier:.4f} [{brier_lo:.4f},{brier_hi:.4f}], "
              f"a0={a0:+.3f} [{a0_lo:+.3f},{a0_hi:+.3f}], "
              f"b={b:.3f} [{b_lo:.3f},{b_hi:.3f}], ECE={ece:.4f}")

    if bins_all:
        bins_df = pd.concat(bins_all, ignore_index=True)
        # Aggregate random_init_seed{N} bins to a single random_init line per bin
        seed_mask = bins_df["fm"].apply(lambda s: bool(RANDOM_INIT_SEED_PATTERN.match(str(s))))
        if seed_mask.sum() > 0:
            non_seed = bins_df[~seed_mask].copy()
            seed = bins_df[seed_mask].copy()
            agg = (seed.groupby(["task", "bin"], as_index=False)
                       .agg({"n": "median", "mean_p": "median", "obs_p": "median",
                             "p_low": "median", "p_high": "median"}))
            agg["fm"] = "random_init"
            bins_df = pd.concat([non_seed, agg], ignore_index=True, sort=False)
        bins_df.to_parquet(OUTPUT_DIR / f"reliability_bins_{task}.parquet", index=False)
    return pd.DataFrame(rows)


def process_cox_task(parquet_path: Path, task: str, horizon_months: int) -> pd.DataFrame:
    if not parquet_path.exists():
        print(f"  [skip] {parquet_path.name}: not found")
        return pd.DataFrame()
    df = pd.read_parquet(parquet_path)
    if df.empty:
        return pd.DataFrame()
    horizon_days = horizon_months * 30.44
    # Auto-detect available surv_*mo column (data may have been generated at a
    # different horizon than the one passed in); prefer exact match.
    surv_col = f"surv_{horizon_months}mo"
    if surv_col not in df.columns:
        candidates = [c for c in df.columns if c.startswith("surv_") and c.endswith("mo")]
        if not candidates:
            print(f"  {task}: no surv_*mo column in data; skip")
            return pd.DataFrame()
        surv_col = candidates[0]
        actual_horizon = int(surv_col[5:-2])
        print(f"  {task}: {surv_col} found (data is at {actual_horizon}mo, requested {horizon_months}mo)")
        horizon_months = actual_horizon
        horizon_days = horizon_months * 30.44
    print(f"  {task}: {len(df)} per-patient predictions across {df['fm'].nunique()} FMs at {horizon_months} mo")

    rows = []
    bins_all = []
    for fm in sorted(df["fm"].unique()):
        sub = df[df["fm"] == fm].reset_index(drop=True)
        if surv_col not in sub.columns or len(sub) < N_BINS:
            continue
        time_arr = sub["time"].values.astype(float)
        ev_arr = sub["event"].values.astype(int)
        surv_arr = sub[surv_col].values.astype(float)

        bin_df = km_per_decile(time_arr, ev_arr, surv_arr, horizon_days)
        if bin_df.empty:
            continue
        bin_df["fm"] = fm
        bin_df["task"] = task
        bin_df["horizon_months"] = horizon_months
        bins_all.append(bin_df)

        mean_pred = float(np.mean(surv_arr))
        weighted_obs = float((bin_df["obs_surv_km"] * bin_df["n"]).sum() / bin_df["n"].sum())
        x = bin_df["mean_pred_surv"].values
        y = bin_df["obs_surv_km"].values
        if len(x) >= 2 and np.std(x) > 0:
            slope = float(np.polyfit(x, y, 1)[0])
            intercept = float(np.polyfit(x, y, 1)[1])
        else:
            slope = float("nan"); intercept = float("nan")

        rows.append({
            "fm": fm, "task": task, "horizon_months": horizon_months,
            "n": len(sub), "n_events": int(ev_arr.sum()),
            "mean_pred_surv": mean_pred,
            "km_obs_surv": weighted_obs,
            "calibration_slope_decile": slope,
            "calibration_intercept_decile": intercept,
        })
        print(f"    {fm:>20s}: mean_pred={mean_pred:.3f}, KM_obs={weighted_obs:.3f}, "
              f"slope={slope:.3f}, intercept={intercept:+.3f}")

    if bins_all:
        bins_df = pd.concat(bins_all, ignore_index=True)
        # Aggregate random_init_seed{N} bins to a single random_init line per bin
        seed_mask = bins_df["fm"].apply(lambda s: bool(RANDOM_INIT_SEED_PATTERN.match(str(s))))
        if seed_mask.sum() > 0:
            non_seed = bins_df[~seed_mask].copy()
            seed = bins_df[seed_mask].copy()
            agg_cols = {"n": "median", "mean_pred_surv": "median",
                        "obs_surv_km": "median", "surv_low": "median",
                        "surv_high": "median"}
            if "n_events" in seed.columns:
                agg_cols["n_events"] = "median"
            agg = (seed.groupby(["task", "bin", "horizon_months"], as_index=False)
                       .agg(agg_cols))
            agg["fm"] = "random_init"
            bins_df = pd.concat([non_seed, agg], ignore_index=True, sort=False)
        bins_df.to_parquet(OUTPUT_DIR / f"cox_calibration_{task}.parquet", index=False)
    return pd.DataFrame(rows)


# ─── Main ──────────────────────────────────────────────────────────────────────


def main():
    print(f"Reading per-prediction parquets from {INPUT_DIR}/")
    if not INPUT_DIR.exists():
        print(f"  ERROR: {INPUT_DIR} does not exist. Download `pet-fm-bench-calibration-perpred-v1` "
              f"from Kaggle into {INPUT_DIR} first.")
        sys.exit(1)

    all_rows = []
    for task in ("t1", "t5", "t7"):
        print(f"\n=== {task.upper()} (binary calibration) ===")
        rows = process_binary_task(INPUT_DIR / f"perpred_{task}.parquet", task)
        if not rows.empty:
            all_rows.append(rows)

    for task, horizon in (("t3", 24), ("t4", 36)):
        print(f"\n=== {task.upper()} (Cox calibration at {horizon} mo) ===")
        rows = process_cox_task(INPUT_DIR / f"perpred_{task}.parquet", task, horizon)
        if not rows.empty:
            all_rows.append(rows)

    if all_rows:
        combined = pd.concat(all_rows, ignore_index=True, sort=False)
        # Collapse random_init_seed{N} → single random_init row (seed-median + IQR)
        combined = aggregate_random_init_seeds(combined)
        out_path = OUTPUT_DIR / "calibration_results.csv"
        combined.to_csv(out_path, index=False)
        n_per_task = combined.groupby("task")["fm"].nunique().to_dict()
        print(f"\n✓ {len(combined)} (FM × task) calibration cells written to {out_path}")
        print(f"   FMs per task (after seed aggregation): {n_per_task}")
    else:
        print("\n⚠ No calibration results produced — check that perpred parquets exist.")


if __name__ == "__main__":
    main()
