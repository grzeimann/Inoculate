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
    p = argparse.ArgumentParser(prog="inoculate-survey", description="Inoculate survey orchestrator (aggregate across shots)")
    p.add_argument("shots_root", type=str, help="Root directory containing many shot outdirs (each with stage_07 manifest)")
    p.add_argument("--outdir", dest="survey_root", type=str, default="survey_out", help="Survey output directory (registry, manifests)")
    p.add_argument("--resume", action="store_true", help="Resume mode (reserved for future stages)")
    p.add_argument("--max-shots", type=int, default=None, help="Optional cap on number of shots to process (dev/testing)")
    p.add_argument("--log-level", default="INFO", help="Logging level (default: INFO)")
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    setup_logging(args.log_level)

    shots_root = Path(args.shots_root)
    if not shots_root.exists():
        parser.error(f"shots_root not found: {shots_root}")

    opts = SurveyOptions(max_shots=args.max_shots)
    result = run_survey(shots_root=str(shots_root), survey_root=args.survey_root, resume=args.resume, options=opts)
    logger.info("Survey completed. Manifest: %s", result)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
