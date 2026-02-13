"""Build multiplicative throughput scaling from BW products.

Implements a simple bounded multiplicative scale per amplifier/exposure by
comparing BW_amp to BW_full over a wavelength mask.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np


def build_mult_scale(
    bw_amp: np.ndarray,
    bw_full: np.ndarray,
    good_mask: np.ndarray,
    wave_mask: np.ndarray,
    bounds: Tuple[float, float] = (0.5, 2.0),
) -> np.ndarray:
    """Build bounded multiplicative scale mult_scale[amp, exp].

    Args:
        bw_amp: Array (n_amp, n_exp, n_wave).
        bw_full: Array (n_exp, n_wave).
        good_mask: Boolean mask (n_amp,) selecting good amplifiers.
        wave_mask: Boolean mask (n_wave,) selecting stable wavelengths.
        bounds: Tuple of (lower, upper) bounds to clip scales.

    Returns:
        Array (n_amp, n_exp) with bounded scaling factors.
    """
    n_amp, n_exp, n_wave = bw_amp.shape
    if bw_full.shape != (n_exp, n_wave):
        raise ValueError("bw_full shape mismatch")
    if good_mask.shape[0] != n_amp:
        raise ValueError("good_mask length mismatch")
    if wave_mask.shape[0] != n_wave:
        raise ValueError("wave_mask length mismatch")

    lo, hi = bounds
    mult = np.ones((n_amp, n_exp), dtype=float)

    wm = wave_mask.astype(bool)
    for a in range(n_amp):
        for e in range(n_exp):
            num = bw_amp[a, e, wm]
            den = bw_full[e, wm]
            # Avoid division by zero with small epsilon; silence warnings
            with np.errstate(all="ignore"):
                ratio = num / np.where(np.abs(den) > 0, den, np.nan)
                scale = np.nanmedian(ratio)
            if not np.isfinite(scale):
                scale = 1.0
            mult[a, e] = float(np.clip(scale, lo, hi))

    # Optionally set mult=1.0 for bad amps
    bad = ~good_mask
    mult[bad, :] = 1.0
    return mult
