"""
Statistical tests for uniformity.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def chisq_uniform_1d(v: np.ndarray, b: int = 20, eps: float = 1e-12) -> tuple[float, float, np.ndarray]:
    """
    Chi-square test for Unif(0,1).

    v: (N,) values in (0,1)
    """
    v = np.clip(v, 0.0 + eps, 1.0 - eps)
    counts, _ = np.histogram(v, bins=b, range=(0.0, 1.0))
    expected = np.ones(b) * (len(v) / b)
    chi2 = ((counts - expected) ** 2 / expected).sum()
    p_val = 1.0 - stats.chi2.cdf(chi2, df=b - 1)
    return chi2, p_val, counts
