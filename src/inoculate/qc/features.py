"""Feature extraction for amplifier quality control.

Implements light-weight, dependency-minimal features sufficient for labeling
amplifiers for the SingleShot workflow.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd


def compute_amp_features(
    bw_amp: np.ndarray,
    bw_full: np.ndarray,
    wave_mask: np.ndarray,
) -> pd.DataFrame:
    """Compute simple per-amp/per-exp features for labeling.

    Features include median ratio to BW_full, MAD of residual, and zero fraction.

    Args:
        bw_amp: Array (n_amp, n_exp, n_wave).
        bw_full: Array (n_exp, n_wave).
        wave_mask: Boolean mask (n_wave,) selecting robust wavelengths.

    Returns:
        DataFrame with columns: [amp, exp, med_ratio, mad_resid, zero_frac].
    """
    n_amp, n_exp, n_wave = bw_amp.shape
    rows = []
    wm = wave_mask.astype(bool)
    for a in range(n_amp):
        for e in range(n_exp):
            x = bw_amp[a, e, wm]
            ref = bw_full[e, wm]
            with np.errstate(all="ignore"):
                ratio = x / np.where(np.abs(ref) > 0, ref, np.nan)
                med_ratio_val = np.nanmedian(ratio)
                x_med = np.nanmedian(x)
                resid = x - x_med
                mad_val = np.nanmedian(np.abs(resid))
                zero_frac_val = np.mean(~np.isfinite(x) | (x == 0))
            med_ratio = float(med_ratio_val) if np.isfinite(med_ratio_val) else 1.0
            mad = float(mad_val) if np.isfinite(mad_val) else float("nan")
            zero_frac = float(zero_frac_val) if np.isfinite(zero_frac_val) else 1.0
            rows.append({
                "amp": a,
                "exp": e,
                "med_ratio": med_ratio,
                "mad_resid": mad,
                "zero_frac": zero_frac,
            })
    return pd.DataFrame(rows)
