"""Aggregation utilities for BW_amp and BW_full computation.

This module computes robust per-amplifier spectra (BW_amp) and aggregates them
across amplifiers to form BW_full, following the SingleShot workflow guide.

All functions use Google-style docstrings and full type hints.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

from ..io.h5 import H5VIRUS
from ..robust import biweight_location


def compute_bw_amp(h5: H5VIRUS, cols: List[str] | None = None) -> np.ndarray:
    """Compute BW_amp[amp, exp, wave] for one shot file.

    BW_amp is the robust (biweight) mean across the 112 fiber spectra within an
    amplifier block for each exposure.

    Args:
        h5: Open H5VIRUS reader.
        cols: Spectral column names to read. If None, uses ["spectrum", "skyspectrum"].

    Returns:
        BW_amp array with shape (n_amp, n_exp, n_wave) as float64.
    """
    if cols is None:
        cols = ["spectrum", "skyspectrum"]

    info = h5.read_info()
    n_amp = int(info["n_amp"])  # type: ignore[arg-type]
    n_exp = int(info["exposures"])  # type: ignore[arg-type]
    n_wave = int(info["n_wave"])  # type: ignore[arg-type]

    bw_amp = np.zeros((n_amp, n_exp, n_wave), dtype=float)

    for a, e, _s, arrays in h5.iter_amp_blocks(cols):
        # total = spectrum + skyspectrum as per guide
        total = arrays["spectrum"] + arrays["skyspectrum"]
        # Shape: (112, n_wave). Compute robust location over fiber axis=0.
        bw = biweight_location(total, axis=0)
        bw_amp[a, e, :] = np.asarray(bw, dtype=float)

    return bw_amp


def compute_bw_full(bw_amp: np.ndarray) -> np.ndarray:
    """Compute BW_full[exp, wave] from BW_amp by robustly aggregating amps.

    Args:
        bw_amp: Amplifier spectra with shape (n_amp, n_exp, n_wave).

    Returns:
        BW_full with shape (n_exp, n_wave).
    """
    if bw_amp.ndim != 3:
        raise ValueError("bw_amp must be (n_amp, n_exp, n_wave)")
    # robust location across amplifier axis=0
    return biweight_location(bw_amp, axis=0)  # (n_exp, n_wave)
