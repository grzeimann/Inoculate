"""Optional caching layer for large survey runs.

Google-style docstrings are used throughout.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional


def cache_path(root: str | Path, key: str) -> Path:
    """Return a filesystem path for a cache entry under the given root.

    Args:
        root: Base directory of the cache.
        key: Cache key (filename-safe preferred).

    Returns:
        Path to the cache entry (no I/O is performed).
    """
    return Path(root) / f"{key}.npz"


def load_cache(path: str | Path) -> Optional[Dict[str, Any]]:
    """Load a cache entry if it exists (placeholder: JSON in NPZ path).

    Args:
        path: Cache file path.

    Returns:
        Decoded object if present, otherwise None.
    """
    import json

    p = Path(path)
    if not p.exists():
        return None
    # Placeholder: store JSON even though extension is .npz for future use
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_cache(path: str | Path, obj: Dict[str, Any]) -> None:
    """Save a cache entry (placeholder JSON writer).

    Args:
        path: Destination file path.
        obj: Serializable mapping.
    """
    import json

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f)
