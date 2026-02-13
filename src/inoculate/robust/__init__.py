"""Robust statistics utilities.

This module provides simple, dependency-light robust estimators needed by the
MVP: biweight location/scale or fallbacks using nanmedian and MAD, as well as
basic robust loss functions (Huber/Tukey) and an IRLS solver.
"""
from __future__ import annotations

from .biweight import biweight_location, biweight_scale, mad
from .loss import huber_weights, tukey_weights, robust_linear_least_squares

__all__ = [
    "biweight_location",
    "biweight_scale",
    "mad",
    "huber_weights",
    "tukey_weights",
    "robust_linear_least_squares",
]
