"""Base data structures for model specification.

Defines simple dataclasses for the first SingleShot vertical slice.

Google-style docstrings are used throughout.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional


@dataclass
class ModelSpec:
    """Specification for the shot-level sky model.

    Attributes:
        poly_order: Order of the shared additive polynomial across exposures.
        n_pca: Number of PCA components per exposure.
        mult_bounds: Tuple of (low, high) bounds for multiplicative scaling.
        wave_mask_frac: Fraction of central wavelengths to use for robust steps (0..1).
        robust_loss: Robust loss for PCA/coeff fits: 'huber' or 'tukey'.
        huber_delta: Huber tuning constant.
        tukey_c: Tukey biweight tuning constant.
        metadata: Arbitrary metadata mapping captured at creation.
    """

    poly_order: int = 11
    n_pca: int = 6
    mult_bounds: tuple[float, float] = (0.5, 2.0)
    wave_mask_frac: float = 0.8
    robust_loss: str = "huber"
    huber_delta: float = 1.0
    tukey_c: float = 4.685
    metadata: Dict[str, Any] | None = None

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable dictionary representation.

        Returns:
            A plain mapping suitable for JSON encoding.
        """
        d = asdict(self)
        return d


@dataclass
class Priors:
    """Container for model priors (placeholder)."""

    params: Dict[str, Any]


@dataclass
class Constraints:
    """Container for model parameter constraints (placeholder)."""

    bounds: Dict[str, Any]
