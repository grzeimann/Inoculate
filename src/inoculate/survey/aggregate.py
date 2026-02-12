"""Aggregation over many HDF5 files to produce survey tables.

Google-style docstrings are used throughout.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List


def find_h5_files(root: str | Path) -> List[Path]:
    """Return a list of HDF5 files under the given root (placeholder).

    Args:
        root: Root directory to search.

    Returns:
        List of file paths ending with .h5 or .hdf5.
    """
    root = Path(root)
    return sorted([p for p in root.rglob("*.h5")] + [p for p in root.rglob("*.hdf5")])
