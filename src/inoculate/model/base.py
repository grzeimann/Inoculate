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
        poly_degree: Degree of 2D polynomial used in mult field model (Poly2D).
        metadata: Arbitrary metadata mapping captured at creation.
    """

    poly_order: int = 11
    n_pca: int = 6
    mult_bounds: tuple[float, float] = (0.5, 2.0)
    wave_mask_frac: float = 0.96
    robust_loss: str = "huber"
    huber_delta: float = 1.0
    tukey_c: float = 4.685
    poly_degree: int = 3
    metadata: Dict[str, Any] | None = None

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable dictionary representation.

        Returns:
            A plain mapping suitable for JSON encoding.
        """
        d = asdict(self)
        return d

    def to_json(self) -> str:
        """Serialize to a JSON string."""
        import json
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ModelSpec":
        """Construct from a dictionary (extra keys ignored)."""
        allowed = {f for f in (
            "poly_order","n_pca","mult_bounds","wave_mask_frac","robust_loss",
            "huber_delta","tukey_c","poly_degree","metadata"
        )}
        data = {k: d[k] for k in d.keys() if k in allowed}
        ms = cls(**data)  # type: ignore[arg-type]
        ms.validate()
        return ms

    @classmethod
    def from_json(cls, s: str) -> "ModelSpec":
        import json
        return cls.from_dict(json.loads(s))

    def validate(self) -> None:
        """Basic validation of parameter ranges."""
        lo, hi = self.mult_bounds
        if not (0.0 < lo < hi):
            raise ValueError("mult_bounds must be (low, high) with 0 < low < high")
        if not (0.0 < self.wave_mask_frac <= 1.0):
            raise ValueError("wave_mask_frac must be in (0, 1]")
        if self.robust_loss not in ("huber", "tukey"):
            raise ValueError("robust_loss must be 'huber' or 'tukey'")
        if not (isinstance(self.poly_degree, int) and self.poly_degree >= 0):
            raise ValueError("poly_degree must be a non-negative integer")


@dataclass
class Priors:
    """Container for model priors (placeholder)."""

    params: Dict[str, Any]


@dataclass
class Constraints:
    """Container for model parameter constraints (placeholder)."""

    bounds: Dict[str, Any]
