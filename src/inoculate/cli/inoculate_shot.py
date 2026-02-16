"""Run basic per-shot checks and print amplifier/exposure slices.

This is an initial placeholder CLI that validates the environment and can show
how amplifier/exposure indexing works for a given amp and exposure.

Examples
--------
$ inoculate-shot --show-slice --amp 0 --exp 1
"""
from __future__ import annotations

import argparse
from pathlib import Path
import logging

from ..io import amp_exposure_slice
from ..shot.pipeline import run_shot
from ..utils.logging import setup_logging

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="inoculate-shot", description="Inoculate single-shot pipeline runner")
    p.add_argument("h5file", nargs="?", help="Path to VIRUS spectral HDF5 file (optional for --show-slice)")
    p.add_argument("--outdir", type=str, default="inoculate_out", help="Output directory for stage artifacts")
    p.add_argument("--resume", action="store_true", help="Resume and skip completed stages")
    p.add_argument("--make-plots", action="store_true", help="Generate diagnostic plots into OUTDIR/plots after processing")
    p.add_argument("--max-plots", type=int, default=12, help="Maximum number of plots to generate when --make-plots is set")
    p.add_argument("--suppress-warnings", action="store_true", help="Suppress runtime warnings (e.g., All-NaN slice) during processing")
    p.add_argument("--amp", type=int, default=0, help="Amplifier index (0-based)")
    p.add_argument("--exp", type=int, default=0, help="Exposure index (0..2)")
    p.add_argument("--show-slice", action="store_true", help="Show the fiber slice for amp/exp and exit")
    p.add_argument("--log-level", default="INFO", help="Logging level (default: INFO)")
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    setup_logging(args.log_level)

    if args.show_slice:
        s = amp_exposure_slice(args.amp, args.exp)
        logger.info("slice amp=%d exp=%d -> slice(%s,%s)", args.amp, args.exp, s.start, s.stop)
        return 0

    if not args.h5file:
        parser.error("h5file is required unless --show-slice is used")

    path = Path(args.h5file)
    if not path.exists():
        parser.error(f"HDF5 file not found: {path}")

    result = run_shot(str(path), args.outdir, modelspec=None, resume=args.resume, make_plots=args.make_plots, max_plots=args.max_plots, suppress_warnings=args.suppress_warnings)
    logger.info("Completed. Manifest: %s", result.get("manifest"))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
