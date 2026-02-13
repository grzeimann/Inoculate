"""Multiplicative throughput model components.

Placeholders for building, applying, and setting priors on multiplicative
factors. Google-style docstrings are used throughout.
"""
from __future__ import annotations

from .build import build_mult_scale  # re-export
from .poly2d import build_mult_poly2d, Poly2DResult  # new export

__all__ = [
    "build_mult_scale",
    "build_mult_poly2d",
    "Poly2DResult",
]
