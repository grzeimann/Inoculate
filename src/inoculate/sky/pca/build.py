"""Build PCA bases for sky residuals per exposure.

Implements a simple SVD-based PCA on good amplifier spectra after scaling by the
multiplicative model, per exposure.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from ...robust import biweight_location, robust_linear_least_squares, huber_weights, tukey_weights


def build_shot_pca(
    bw_amp: np.ndarray,
    bw_full: np.ndarray,
    mult_scale: np.ndarray,
    good_mask: np.ndarray,
    wave_mask: np.ndarray,
    poly_beta: np.ndarray,
    n_components: int = 6,
    *,
    loss: str = "huber",
    huber_delta: float = 1.0,
    tukey_c: float = 4.685,
) -> Dict[str, np.ndarray]:
    """Build PCA basis per exposure for one shot using SVD with robust weighting.

    Constructs PCA on residual_2 defined as:
      residual_2 = (bw_amp / mult_scale) - bw_full - poly(amp)
    over the modeling wavelength mask. A per-amp polynomial (shared across
    exposures) is provided via ``poly_beta`` and is evaluated on the modeling
    grid.

    Args:
        bw_amp: Array (n_amp, n_exp, n_wave).
        bw_full: Array (n_exp, n_wave).
        mult_scale: Array (n_amp, n_exp) multiplicative factors.
        good_mask: Boolean mask (n_amp,) selecting good amplifiers.
        wave_mask: Boolean mask (n_wave,) selecting wavelengths for PCA.
        poly_beta: Array (n_amp, n_poly) of per-amp polynomial coefficients.
        n_components: Number of principal components to keep.

    Returns:
        Mapping with arrays: pca_mean (n_exp, n_wave), pca_evecs (n_exp, n_comp, n_wave),
        explained_variance_ratio (n_exp, n_comp).
    """
    n_amp, n_exp, n_wave = bw_amp.shape
    wm = wave_mask.astype(bool)

    # Build polynomial design matrix on the masked grid using normalized x in [-1,1]
    x = np.linspace(-1.0, 1.0, n_wave)[wm]
    n_poly = poly_beta.shape[1] if poly_beta.ndim == 2 else 0
    P = np.vstack([x ** i for i in range(n_poly)]).T if n_poly > 0 else np.zeros((wm.sum(), 0))

    pca_mean = np.zeros((n_exp, n_wave), dtype=float)
    pca_evecs = np.zeros((n_exp, n_components, n_wave), dtype=float)
    evr = np.zeros((n_exp, n_components), dtype=float)

    for e in range(n_exp):
        # Select good amps and scale
        X = bw_amp[good_mask, e, :]
        if X.size == 0:
            continue
        X = X / np.clip(mult_scale[good_mask, e][:, None], 1e-6, np.inf)
        # Mask wavelengths
        Xw = X[:, wm]
        # Subtract BW_full on mask
        Xw = Xw - bw_full[e, wm][None, :]
        # Subtract per-amp polynomial on mask
        if n_poly > 0:
            poly_curves = (P @ poly_beta[good_mask].T).T  # (n_good, nw_mask)
            Xw = Xw - poly_curves
        # Center with robust location (biweight/nanmedian fallback)
        with np.errstate(all="ignore"):
            mu = biweight_location(Xw, axis=0)
        Xc = Xw - mu
        # Compute per-row robust weights to downweight outlier amps for this exposure
        # Use L2 norm across wavelengths to form residual magnitude per row.
        with np.errstate(all="ignore"):
            row_scale = np.sqrt(np.nanmean(Xc * Xc, axis=1))  # (n_good,)
            med = np.nanmedian(row_scale)
            mad = np.nanmedian(np.abs(row_scale - med))
            s = max(1e-12, 1.4826 * mad)
            z = (row_scale - med) / s if s > 0 else row_scale * 0.0
        if loss == "tukey":
            w_rows = tukey_weights(z, c=tukey_c)
        else:
            w_rows = huber_weights(z, delta=huber_delta)
        # Guard against all-zero weights
        if not np.any(w_rows > 0):
            w_rows = np.ones_like(w_rows)
        # Apply sqrt row-weights before SVD (equivalent to WLS)
        sw = np.sqrt(w_rows)[:, None]
        Xcw = Xc * sw
        # SVD on weighted centered matrix
        U, S, VT = np.linalg.svd(np.nan_to_num(Xcw, nan=0.0), full_matrices=False)
        k = min(n_components, VT.shape[0])
        V = VT[:k, :]  # (k, nw_mask)
        # Explained variance ratio approximation using weighted variance proxy
        var = (S ** 2) / max(Xcw.shape[0] - 1, 1)
        var_ratio = var / var.sum() if var.sum() > 0 else np.zeros_like(var)
        pca_mean[e, wm] = mu
        # Assign eigenvectors without triggering NumPy advanced-indexing axis reordering
        vec_view = pca_evecs[e, :k, :]
        vec_view[:, wm] = V
        evr[e, :k] = var_ratio[:k]

    return {
        "pca_mean": pca_mean,
        "pca_evecs": pca_evecs,
        "explained_variance_ratio": evr,
    }
