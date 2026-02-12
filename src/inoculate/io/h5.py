"""
Lightweight HDF5 accessors tailored to VIRUS HETDEX shot files.

The functions here avoid loading full fiber×wavelength arrays by providing
slice helpers that operate amplifier-by-amplifier and exposure-by-exposure.

This module depends optionally on PyTables. If it's not installed, an
ImportError with a helpful message is raised upon use.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator, Optional

from ..constants import EXPOSURES_PER_SHOT, NUM_FIBERS_PER_AMP

try:  # Optional dependency
    import tables  # type: ignore
except Exception:  # pragma: no cover - handled at call time
    tables = None  # type: ignore


@contextmanager
def open_h5(path: str):
    """Context manager that opens an HDF5 file with PyTables if available.

    Parameters
    ----------
    path:
        File path to the VIRUS spectral HDF5 file.

    Yields
    ------
    tables.File
        An open PyTables handle. Closes automatically on exit.
    """
    if tables is None:  # pragma: no cover - environment dependent
        raise ImportError(
            "PyTables is required to read HDF5 inputs. Install with `pip install tables`"
        )
    h5 = tables.open_file(path)
    try:
        yield h5
    finally:
        h5.close()


def amp_exposure_slice(amp_index: int, exposure_index: int, fibers_per_amp: int = NUM_FIBERS_PER_AMP,
                       exposures_per_shot: int = EXPOSURES_PER_SHOT) -> slice:
    """Return a Python slice for rows corresponding to a given amp and exposure.

    The Goals document specifies 112 fibers per amplifier and fibers ordered as
    [exp1(112), exp2(112), exp3(112)] within each amplifier block.

    Parameters
    ----------
    amp_index:
        Zero-based amplifier index within the file.
    exposure_index:
        Zero-based exposure index within the shot (0..2).
    fibers_per_amp:
        Number of fibers per amplifier (default 112).
    exposures_per_shot:
        Number of exposures per shot (default 3).

    Returns
    -------
    slice
        A slice(start, stop) selecting the fibers for the requested amp and exposure.
    """
    if not (0 <= exposure_index < exposures_per_shot):
        raise ValueError("exposure_index out of range")
    if amp_index < 0:
        raise ValueError("amp_index must be non-negative")

    block = exposures_per_shot * fibers_per_amp
    start = amp_index * block + exposure_index * fibers_per_amp
    stop = start + fibers_per_amp
    return slice(start, stop)
