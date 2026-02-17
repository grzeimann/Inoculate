"""Survey plan and standardized path resolution for multi-shot orchestration.

This module centralizes directory layout and artifact paths for survey-level
processing (aggregating across many shots). It mirrors ShotPlan/IFUPlan in
spirit and keeps file naming in one place.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict


@dataclass(frozen=True)
class SurveyPlan:
    """Resolver for survey-level artifact paths.

    Attributes:
        survey_root: Base directory for survey outputs (registry, manifests, cache).
    """

    survey_root: Path

    @property
    def registry_dir(self) -> Path:
        return Path(self.survey_root) / "registry"

    @property
    def cache_dir(self) -> Path:
        return Path(self.survey_root) / "cache"

    @property
    def logs_dir(self) -> Path:
        return Path(self.survey_root) / "logs"

    @property
    def plots_dir(self) -> Path:
        """Directory for survey-level plots (e.g., IFU profile overlays)."""
        return Path(self.survey_root) / "plots"

    def ensure_dirs(self) -> None:
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)

    def ifu_profiles_plot_path(self, ifu: int) -> Path:
        """Return path for per-IFU overlay plot of individual profiles + mean."""
        return self.plots_dir / f"ifu{int(ifu):03d}_profiles.png"

    def paths(self) -> Dict[str, Path]:
        out = Path(self.survey_root)
        return {
            "Survey_Manifest": out / "survey_manifest.json",
            "Survey_Stats": out / "survey_stats.json",
            "Shot_Index": out / "shots_index.json",
            # Directories
            "Registry_Dir": self.registry_dir,
            "Cache_Dir": self.cache_dir,
            "Logs_Dir": self.logs_dir,
            "Shots_Dir": out / "shots",
        }

    def ifu_registry_path(self, ifu: int) -> Path:
        return self.registry_dir / f"ifu{int(ifu):03d}_stats.json"

    @property
    def shots_dir(self) -> Path:
        return Path(self.survey_root) / "shots"

    def ensure_dirs(self) -> None:
        # Reorder to ensure shots_dir exists too
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.shots_dir.mkdir(parents=True, exist_ok=True)
