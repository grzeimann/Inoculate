"""Writers for Inoculate products (parquet/tables and model outputs).

These are placeholders for future implementations that will write per-file
statistics, features, labels, and model products to portable formats.

All functions use Google-style docstrings.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any


def write_parquet(data: Any, path: str | Path) -> None:
    """Write tabular data to Parquet.

    This is a placeholder. In future iterations we will prefer pyarrow.

    Args:
        data: Tabular-like object (e.g., pandas DataFrame).
        path: Destination file path.

    Raises:
        NotImplementedError: Always, until implemented.
    """
    raise NotImplementedError("Parquet writing is not yet implemented.")
