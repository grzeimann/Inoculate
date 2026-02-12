"""
Configuration helpers for Inoculate.

This module will grow to include structured configuration (pydantic or
OmegaConf). For now it provides a simple dictionary-based config with a few
well-named keys used by early prototypes.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class InoculateConfig:
    """Minimal configuration container.

    Parameters
    ----------
    data_root:
        Optional root directory that contains HDF5 inputs. CLI tools may use
        this as a default when a file list is not provided.
    work_dir:
        Location to write intermediate products (e.g., parquet summaries) in
        future iterations.
    log_level:
        Default log level for CLI commands.
    """

    data_root: Path | None = None
    work_dir: Path | None = None
    log_level: str = "INFO"


DEFAULT_CONFIG = InoculateConfig()
