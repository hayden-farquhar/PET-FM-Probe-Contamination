"""Survival probe for PET-FM-Bench.

Pre-registration Section 5.4: CoxPH with L2 penalty,
alpha in {0.001, 0.01, 0.1, 1, 10}, 5-fold nested CV.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold


ALPHA_GRID = [0.001, 0.01, 0.1, 1, 10]
N_FOLDS = 5
N_BOOTSTRAP = 1000


def train_cox_probe(
    X: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    patient_ids: np.ndarray,
    n_folds: int = N_FOLDS,
    alpha_grid: list = ALPHA_GRID,
    random_state: int = 42,
) -> dict:
    """Train CoxPH probe with nested CV.

    Args:
        X: Embedding matrix (n_patients, embed_dim).
        time: Survival/follow-up time.
        event: Event indicator (1=dead, 0=censored).
        patient_ids: Patient IDs for splitting.

    Returns:
        dict with c-index, CI, best_alpha.
    """
    try:
        from sksurv.linear_model import CoxPHSurvivalAnalysis
        from sksurv.metrics import concordance_index_censored
    except ImportError:
        return {"error": "scikit-survival not installed"}

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Create structured array for sksurv
    y_struct = np.array(
        [(bool(e), float(t)) for e, t in zip(event, time)],
        dtype=[("event", bool), ("time", float)]
    )

    # Nested CV
    cv = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    best_alpha = _select_alpha(X_scaled, y_struct, cv, alpha_grid)

    # Out-of-fold risk scores
    risk_scores = np.zeros(len(X))
    for train_idx, test_idx in cv.split(X_scaled):
        model = CoxPHSurvivalAnalysis(alpha=best_alpha)
        try:
            model.fit(X_scaled[train_idx], y_struct[train_idx])
            risk_scores[test_idx] = model.predict(X_scaled[test_idx])
        except Exception:
            risk_scores[test_idx] = 0.0

    # Harrell's c-index
    c_index, _, _, _, _ = concordance_index_censored(
        event.astype(bool), time, risk_scores
    )

    # Bootstrap CI
    ci_low, ci_high = _bootstrap_cindex_ci(
        event, time, risk_scores, N_BOOTSTRAP, random_state
    )

    return {
        "c_index": float(c_index),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "best_alpha": float(best_alpha),
        "n_patients": len(X),
        "n_events": int(event.sum()),
        "risk_scores": risk_scores,
    }


def _select_alpha(X, y_struct, cv, alpha_grid):
    """Inner CV for alpha selection."""
    from sksurv.linear_model import CoxPHSurvivalAnalysis
    from sksurv.metrics import concordance_index_censored

    best_alpha = 1.0
    best_score = -1

    for alpha in alpha_grid:
        scores = []
        for train_idx, test_idx in cv.split(X):
            model = CoxPHSurvivalAnalysis(alpha=alpha)
            try:
                model.fit(X[train_idx], y_struct[train_idx])
                preds = model.predict(X[test_idx])
                events = y_struct["event"][test_idx]
                times = y_struct["time"][test_idx]
                c, _, _, _, _ = concordance_index_censored(events, times, preds)
                scores.append(c)
            except Exception:
                continue

        if scores and np.mean(scores) > best_score:
            best_score = np.mean(scores)
            best_alpha = alpha

    return best_alpha


def _bootstrap_cindex_ci(event, time, risk_scores, n_bootstrap, random_state, alpha=0.05):
    """Bootstrap CI for concordance index."""
    from sksurv.metrics import concordance_index_censored

    rng = np.random.RandomState(random_state)
    cindices = []
    n = len(event)

    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        e = event[idx].astype(bool)
        t = time[idx]
        r = risk_scores[idx]
        if e.sum() < 2:
            continue
        try:
            c, _, _, _, _ = concordance_index_censored(e, t, r)
            cindices.append(c)
        except Exception:
            continue

    cindices = np.array(cindices)
    return float(np.percentile(cindices, 100 * alpha / 2)), float(np.percentile(cindices, 100 * (1 - alpha / 2)))
