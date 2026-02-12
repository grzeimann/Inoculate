"""Priors and component selection for PCA-based sky models.

Google-style docstrings are used throughout.
"""
from __future__ import annotations

from typing import Any, Dict, List


def small_coeff_prior(var_explained: List[float], max_components: int | None = None) -> Dict[str, Any]:
    """Return a simple prior for selecting PCA components.

    The placeholder prior prefers fewer components and down-weights
    high-order components with small explained variance.

    Args:
        var_explained: Cumulative or per-component explained variance ratios.
        max_components: Optional hard cap on the number of components.

    Returns:
        A mapping describing the selected number of components and weights.
    """
    n = len(var_explained)
    if max_components is not None:
        n = min(n, max_components)
    weights = [1.0 / (i + 1) for i in range(n)]
    return {"n_components": n, "weights": weights}
