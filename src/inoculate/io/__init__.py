"""I/O utilities for Inoculate.

This subpackage provides lightweight readers for HDF5 inputs and helpers for
indexing fibers by amplifier and exposure without loading full arrays.
"""
from __future__ import annotations

__all__ = [
    "amp_exposure_slice",
    "open_h5",
]

from .h5 import amp_exposure_slice, open_h5  # re-export