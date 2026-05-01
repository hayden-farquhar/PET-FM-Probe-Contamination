"""Linear probe classification for PET-FM-Bench.

Pre-registration Section 5.4: LogisticRegression with L2 penalty,
C in {0.001, 0.01, 0.1, 1, 10, 100}, 5-fold patient-level CV.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import bootstrap


C_GRID = [0.001, 0.01, 0.1, 1, 10, 100]
N_FOLDS = 5
N_BOOTSTRAP = 1000


def train_linear_probe(
    X: np.ndarray,
    y: np.ndarray,
    patient_ids: np.ndarray,
    n_folds: int = N_FOLDS,
    c_grid: list = C_GRID,
    random_state: int = 42,
) -> dict:
    """Train a linear probe with nested CV for hyperparameter selection.

    Per registration: patient-level splits, no within-patient leakage.

    Args:
        X: Embedding matrix (n_patients, embed_dim).
        y: Binary labels (n_patients,).
        patient_ids: Patient IDs for patient-level splitting.
        n_folds: Number of CV folds.
        c_grid: L2 regularization grid.
        random_state: Random seed.

    Returns:
        dict with AUROC, CI, best_C, per-fold predictions.
    """
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Outer CV for evaluation
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    best_c = _select_c_inner_cv(X_scaled, y, cv, c_grid, random_state)

    # Get out-of-fold predictions with best C
    model = LogisticRegression(C=best_c, penalty="l2", solver="lbfgs",
                               max_iter=1000, random_state=random_state)
    y_pred = cross_val_predict(model, X_scaled, y, cv=cv, method="predict_proba")[:, 1]

    # Compute AUROC
    auroc = roc_auc_score(y, y_pred)

    # Bootstrap CI (patient-level resampling per registration)
    ci_low, ci_high = _bootstrap_auroc_ci(y, y_pred, n_bootstrap=N_BOOTSTRAP,
                                           random_state=random_state)

    return {
        "auroc": float(auroc),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "best_c": float(best_c),
        "n_patients": len(y),
        "n_events": int(y.sum()),
        "y_pred": y_pred,
        "y_true": y,
        "patient_ids": patient_ids,
    }


def _select_c_inner_cv(X, y, outer_cv, c_grid, random_state):
    """Inner CV to select best C across the grid."""
    best_c = 1.0
    best_score = -1

    for c in c_grid:
        model = LogisticRegression(C=c, penalty="l2", solver="lbfgs",
                                   max_iter=1000, random_state=random_state)
        try:
            preds = cross_val_predict(model, X, y, cv=outer_cv, method="predict_proba")[:, 1]
            score = roc_auc_score(y, preds)
            if score > best_score:
                best_score = score
                best_c = c
        except Exception:
            continue

    return best_c


def _bootstrap_auroc_ci(y_true, y_score, n_bootstrap=1000, alpha=0.05, random_state=42):
    """Patient-level bootstrap CI for AUROC."""
    rng = np.random.RandomState(random_state)
    aucs = []
    n = len(y_true)

    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        if len(np.unique(y_true[idx])) < 2:
            continue
        try:
            aucs.append(roc_auc_score(y_true[idx], y_score[idx]))
        except ValueError:
            continue

    aucs = np.array(aucs)
    ci_low = np.percentile(aucs, 100 * alpha / 2)
    ci_high = np.percentile(aucs, 100 * (1 - alpha / 2))
    return ci_low, ci_high


def delong_test(y_true, y_score_a, y_score_b):
    """DeLong test for comparing two AUROCs on the same data.

    Returns p-value for the two-sided test H0: AUROC_A == AUROC_B.
    Implementation follows Sun & Xu (2014).
    """
    from scipy.stats import norm

    n1 = np.sum(y_true == 1)
    n0 = np.sum(y_true == 0)

    # Placement values
    def _placement(scores, y):
        pos = scores[y == 1]
        neg = scores[y == 0]
        # For each positive, fraction of negatives scored below
        v10 = np.array([np.mean(neg < p) + 0.5 * np.mean(neg == p) for p in pos])
        # For each negative, fraction of positives scored above
        v01 = np.array([np.mean(pos > n) + 0.5 * np.mean(pos == n) for n in neg])
        return v10, v01

    v10_a, v01_a = _placement(y_score_a, y_true)
    v10_b, v01_b = _placement(y_score_b, y_true)

    auc_a = np.mean(v10_a)
    auc_b = np.mean(v10_b)

    # Covariance matrix
    s10 = np.cov(v10_a, v10_b)
    s01 = np.cov(v01_a, v01_b)

    s = s10 / n1 + s01 / n0

    # Variance of the difference
    var_diff = s[0, 0] + s[1, 1] - 2 * s[0, 1]

    if var_diff <= 0:
        return 1.0  # Cannot distinguish

    z = (auc_a - auc_b) / np.sqrt(var_diff)
    p_value = 2 * norm.sf(abs(z))

    return float(p_value)
