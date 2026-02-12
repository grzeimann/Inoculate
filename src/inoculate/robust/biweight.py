"""
Minimal robust estimators: biweight location/scale with safe fallbacks.

If Astropy is available, we defer to astropy.stats for biweight. Otherwise we
use simple, well-behaved approximations based on the median and MAD that are
sufficient for early prototyping and unit-less feature extraction.
"""
from __future__ import annotations

import numpy as np

try:
    from astropy.stats import biweight_location as _astro_biweight_location  # type: ignore
    from astropy.stats import biweight_scale as _astro_biweight_scale  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    _astro_biweight_location = None  # type: ignore
    _astro_biweight_scale = None  # type: ignore


def mad(x: np.ndarray, axis: int | None = None, scale: float = 1.4826) -> np.ndarray:
    """Median absolute deviation with Gaussian scaling.

    Parameters
    ----------
    x : array-like
        Input data.
    axis : int or None
        Axis along which to compute. None flattens the array.
    scale : float
        Factor to scale the raw MAD to be comparable to standard deviation for
        Gaussian data (default 1.4826).
    """
    x = np.asanyarray(x)
    med = np.nanmedian(x, axis=axis, keepdims=True)
    mad_raw = np.nanmedian(np.abs(x - med), axis=axis)
    return scale * mad_raw


def biweight_location(x: np.ndarray, axis: int | None = None) -> np.ndarray:
    """Robust central location.

    Uses astropy.stats.biweight_location if available, else nanmedian.
    """
    if _astro_biweight_location is not None:  # pragma: no cover - depends on env
        return _astro_biweight_location(x, axis=axis)
    return np.nanmedian(np.asanyarray(x), axis=axis)


def biweight_scale(x: np.ndarray, axis: int | None = None) -> np.ndarray:
    """Robust scale estimator.

    Uses astropy.stats.biweight_scale if available, else MAD.
    """
    if _astro_biweight_scale is not None:  # pragma: no cover - depends on env
        return _astro_biweight_scale(x, axis=axis)
    return mad(np.asanyarray(x), axis=axis)
