"""Robust statistics utilities.

This module provides simple, dependency-light robust estimators needed by the
MVP: biweight location/scale or fallbacks using nanmedian and MAD.
"""
from __future__ import annotations

from .biweight import biweight_location, biweight_scale, mad

__all__ = [
    "biweight_location",
    "biweight_scale",
    "mad",
]
