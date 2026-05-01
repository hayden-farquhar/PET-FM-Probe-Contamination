"""Test-retest embedding stability analysis for PET-FM-Bench.

Pre-registration Section 5.4 (T6, T9): Compute cosine similarity between
embeddings of serial PET scans for the same subject. Report Lin's CCC,
ICC(3,1), and mean absolute cosine distance.
"""

import numpy as np
from scipy.stats import pearsonr


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def lins_ccc(x: np.ndarray, y: np.ndarray) -> float:
    """Lin's concordance correlation coefficient.

    Measures agreement between paired measurements.
    CCC = 1 means perfect agreement; CCC = 0 means no agreement.
    """
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    var_x = np.var(x, ddof=1)
    var_y = np.var(y, ddof=1)
    cov_xy = np.cov(x, y, ddof=1)[0, 1]

    numerator = 2 * cov_xy
    denominator = var_x + var_y + (mean_x - mean_y) ** 2

    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


def icc_3_1(x: np.ndarray, y: np.ndarray) -> float:
    """Intraclass correlation coefficient ICC(3,1) — two-way mixed, consistency.

    For test-retest with exactly 2 measurements per subject.
    """
    n = len(x)
    if n < 2:
        return 0.0

    # Stack as (n, 2) matrix
    data = np.column_stack([x, y])
    grand_mean = data.mean()
    row_means = data.mean(axis=1)
    col_means = data.mean(axis=0)

    # Sum of squares
    ss_total = np.sum((data - grand_mean) ** 2)
    ss_rows = 2 * np.sum((row_means - grand_mean) ** 2)  # k=2 measurements
    ss_cols = n * np.sum((col_means - grand_mean) ** 2)
    ss_error = ss_total - ss_rows - ss_cols

    # Mean squares
    k = 2  # number of raters/measurements
    ms_rows = ss_rows / (n - 1)
    ms_error = ss_error / ((n - 1) * (k - 1))

    # ICC(3,1)
    if (ms_rows + ms_error) == 0:
        return 0.0
    return float((ms_rows - ms_error) / (ms_rows + ms_error))


def compute_test_retest_stability(
    embeddings_test: np.ndarray,
    embeddings_retest: np.ndarray,
    patient_ids: list[str],
) -> dict:
    """Compute test-retest stability metrics for paired embeddings.

    Args:
        embeddings_test: (n_patients, embed_dim) for test session.
        embeddings_retest: (n_patients, embed_dim) for retest session.
        patient_ids: Patient IDs for the pairs.

    Returns:
        dict with cosine similarities, CCC, ICC, summary stats.
    """
    n = len(patient_ids)
    assert embeddings_test.shape[0] == n
    assert embeddings_retest.shape[0] == n

    # Per-patient cosine similarity
    cosines = np.array([
        cosine_similarity(embeddings_test[i], embeddings_retest[i])
        for i in range(n)
    ])

    # Per-dimension CCC and ICC (computed across patients, per embedding dimension)
    embed_dim = embeddings_test.shape[1]
    dim_cccs = []
    for d in range(embed_dim):
        ccc = lins_ccc(embeddings_test[:, d], embeddings_retest[:, d])
        dim_cccs.append(ccc)
    dim_cccs = np.array(dim_cccs)

    # Also compute CCC on the flattened embedding norms
    norms_test = np.linalg.norm(embeddings_test, axis=1)
    norms_retest = np.linalg.norm(embeddings_retest, axis=1)
    norm_ccc = lins_ccc(norms_test, norms_retest)
    norm_icc = icc_3_1(norms_test, norms_retest)

    return {
        "n_pairs": n,
        "cosine_mean": float(cosines.mean()),
        "cosine_std": float(cosines.std()),
        "cosine_min": float(cosines.min()),
        "cosine_max": float(cosines.max()),
        "cosine_per_patient": cosines,
        "dim_ccc_mean": float(dim_cccs.mean()),
        "dim_ccc_median": float(np.median(dim_cccs)),
        "dim_ccc_min": float(dim_cccs.min()),
        "norm_ccc": float(norm_ccc),
        "norm_icc": float(norm_icc),
        "patient_ids": patient_ids,
    }


def bootstrap_ccc_ci(
    embeddings_test: np.ndarray,
    embeddings_retest: np.ndarray,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    random_state: int = 42,
) -> tuple[float, float]:
    """Bootstrap CI for mean cosine similarity."""
    rng = np.random.RandomState(random_state)
    n = len(embeddings_test)
    cosines_boot = []

    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        cos_vals = [
            cosine_similarity(embeddings_test[idx[i]], embeddings_retest[idx[i]])
            for i in range(n)
        ]
        cosines_boot.append(np.mean(cos_vals))

    cosines_boot = np.array(cosines_boot)
    return (
        float(np.percentile(cosines_boot, 100 * alpha / 2)),
        float(np.percentile(cosines_boot, 100 * (1 - alpha / 2))),
    )
