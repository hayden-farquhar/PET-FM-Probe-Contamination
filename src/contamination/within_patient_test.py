"""Within-patient contamination test: compare probe accuracy on dirty vs clean patients.

Pre-registration Section 4.4: For FMs where patient-level contamination status
is known, partition the test set into dirty (patient data appeared in FM training)
and clean (did not appear) subsets. Test AUROC_dirty > AUROC_clean via
permutation test (10,000 permutations).
"""

import numpy as np
from sklearn.metrics import roc_auc_score


def within_patient_contamination_test(
    y_true: np.ndarray,
    y_score: np.ndarray,
    is_contaminated: np.ndarray,
    n_permutations: int = 10_000,
    min_per_group: int = 15,
    random_state: int = 42,
) -> dict:
    """Test whether FM probe performs better on contaminated vs clean patients.

    Args:
        y_true: Ground truth labels (binary).
        y_score: Predicted probabilities from FM probe.
        is_contaminated: Boolean array — True if patient was in FM training set.
        n_permutations: Number of permutations for the test.
        min_per_group: Minimum patients per group for formal testing.
        random_state: Random seed.

    Returns:
        dict with AUROC_dirty, AUROC_clean, delta, p_value, and test validity.
    """
    rng = np.random.RandomState(random_state)

    dirty_mask = is_contaminated.astype(bool)
    clean_mask = ~dirty_mask

    n_dirty = dirty_mask.sum()
    n_clean = clean_mask.sum()

    # Check minimum sample sizes
    if n_dirty < min_per_group or n_clean < min_per_group:
        return {
            "auroc_dirty": _safe_auroc(y_true[dirty_mask], y_score[dirty_mask]),
            "auroc_clean": _safe_auroc(y_true[clean_mask], y_score[clean_mask]),
            "delta": None,
            "p_value": None,
            "n_dirty": int(n_dirty),
            "n_clean": int(n_clean),
            "valid": False,
            "reason": f"Below minimum sample size ({min_per_group}): dirty={n_dirty}, clean={n_clean}",
        }

    # Observed AUROCs
    auroc_dirty = _safe_auroc(y_true[dirty_mask], y_score[dirty_mask])
    auroc_clean = _safe_auroc(y_true[clean_mask], y_score[clean_mask])

    if auroc_dirty is None or auroc_clean is None:
        return {
            "auroc_dirty": auroc_dirty,
            "auroc_clean": auroc_clean,
            "delta": None,
            "p_value": None,
            "n_dirty": int(n_dirty),
            "n_clean": int(n_clean),
            "valid": False,
            "reason": "AUROC undefined (single-class group)",
        }

    observed_delta = auroc_dirty - auroc_clean

    # Permutation test: shuffle contamination labels
    null_deltas = []
    for _ in range(n_permutations):
        perm = rng.permutation(len(is_contaminated))
        perm_dirty = dirty_mask[perm]
        perm_clean = ~perm_dirty

        auc_d = _safe_auroc(y_true[perm_dirty], y_score[perm_dirty])
        auc_c = _safe_auroc(y_true[perm_clean], y_score[perm_clean])

        if auc_d is not None and auc_c is not None:
            null_deltas.append(auc_d - auc_c)

    null_deltas = np.array(null_deltas)
    # One-tailed p-value: proportion of permuted deltas >= observed
    p_value = (null_deltas >= observed_delta).mean()

    return {
        "auroc_dirty": float(auroc_dirty),
        "auroc_clean": float(auroc_clean),
        "delta": float(observed_delta),
        "p_value": float(p_value),
        "n_dirty": int(n_dirty),
        "n_clean": int(n_clean),
        "n_permutations": len(null_deltas),
        "valid": True,
    }


def _safe_auroc(y_true, y_score):
    """Compute AUROC, returning None if undefined (single class)."""
    if len(y_true) < 2 or len(np.unique(y_true)) < 2:
        return None
    try:
        return roc_auc_score(y_true, y_score)
    except ValueError:
        return None
