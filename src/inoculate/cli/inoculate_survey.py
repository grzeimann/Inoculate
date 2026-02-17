"""CLI entry point for running the survey-level orchestrator.

Usage examples
--------------
- Aggregate all shots under a root directory and write registry/prior stats:

    inoculate-survey /path/to/shots_root --outdir survey_out

- Limit to first N shots for a quick check:

    inoculate-survey /path/to/shots_root --outdir survey_out --max-shots 5
"""
from __future__ import annotations

import argparse
from pathlib import Path
import logging

from ..survey.pipeline import run_survey, SurveyOptions
from ..utils.logging import setup_logging

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="inoculate-survey", description="Inoculate survey orchestrator (build per-shot outputs and/or aggregate across shots)")
    p.add_argument("shots_root", type=str, help="Root directory containing either H5 files (with --build-from-h5) or existing shot outdirs (manifests)")
    p.add_argument("--outdir", dest="survey_root", type=str, default="survey_out", help="Survey output directory (registry, manifests)")
    p.add_argument("--resume", action="store_true", help="Resume aggregation (reserved)")
    p.add_argument("--max-shots", type=int, default=None, help="Optional cap on number of shots or H5 files (dev/testing)")
    p.add_argument("--build-from-h5", action="store_true", help="Discover .h5 under shots_root and run per-shot pipelines into survey_root/shots/<name>/ before aggregation")
    p.add_argument("--shot-resume", action="store_true", help="Pass resume to run_shot when building from H5")
    p.add_argument("--ifu-resume", action="store_true", help="Pass resume to run_ifu when building from H5")
    p.add_argument("--suppress-warnings", action="store_true", help="Suppress runtime warnings in per-shot run_shot")
    p.add_argument("--log-level", default="INFO", help="Logging level (default: INFO)")
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    setup_logging(args.log_level)

    shots_root = Path(args.shots_root)
    if not shots_root.exists():
        parser.error(f"shots_root not found: {shots_root}")

    opts = SurveyOptions(
        max_shots=args.max_shots,
        build_from_h5=args.build_from_h5,
        shot_resume=args.shot_resume,
        ifu_resume=args.ifu_resume,
        suppress_warnings=args.suppress_warnings,
    )
    result = run_survey(shots_root=str(shots_root), survey_root=args.survey_root, resume=args.resume, options=opts)
    logger.info("Survey completed. Manifest: %s", result)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
