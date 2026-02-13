"""Writers for Inoculate products (parquet/tables and model outputs).

All functions use Google-style docstrings.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any


def write_parquet(data: Any, path: str | Path) -> None:
    """Write tabular data to Parquet using pandas.

    Args:
        data: Tabular-like object (e.g., pandas DataFrame) with a ``to_parquet`` method.
        path: Destination file path.

    Raises:
        ImportError: If pandas/pyarrow are not available.
        ValueError: If ``data`` does not provide ``to_parquet``.
    """
    try:
        import pandas as pd  # type: ignore
    except Exception as exc:  # pragma: no cover - env dependent
        raise ImportError("pandas is required to write parquet outputs") from exc

    if not hasattr(data, "to_parquet"):
        raise ValueError("data must be a pandas DataFrame or implement to_parquet")

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    data.to_parquet(p, index=False)  # relies on pyarrow or fastparquet
