"""Run IFU (fiber-level) refinement after inoculate-shot.

This CLI consumes the shot-level artifacts in --outdir and computes per-IFU
(per-amp/per-exp/per-fiber) delta multiplicative tweaks and source masks using
inoculate.ifu.run_ifu.

Examples
--------
$ inoculate-ifu /path/to/shot.h5 --outdir inoculate_out
$ inoculate-ifu /path/to/shot.h5 --outdir inoculate_out --ifus 0,1,2 --resume
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable, List, Optional

from ..utils.logging import setup_logging
from ..ifu.pipeline import run_ifu, IFUOptions

logger = logging.getLogger(__name__)


def _parse_ifus(val: Optional[str]) -> Optional[List[int]]:
    if val is None:
        return None
    s = val.strip()
    if not s:
        return None
    # Support comma and/or whitespace separated lists
    tokens = [t for chunk in s.split(",") for t in chunk.split()] if "," in s or " " in s else [s]
    ints: List[int] = []
    for t in tokens:
        t = t.strip()
        if not t:
            continue
        try:
            ints.append(int(t))
        except ValueError:
            raise argparse.ArgumentTypeError(f"Invalid IFU index: {t}")
    return ints if ints else None


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="inoculate-ifu", description="Inoculate IFU (fiber-level) refinement runner")
    p.add_argument("h5file", help="Path to VIRUS spectral HDF5 file (same shot as used for inoculate-shot)")
    p.add_argument("--outdir", type=str, default="inoculate_out", help="Output directory containing shot artifacts (default: inoculate_out)")
    p.add_argument("--resume", action="store_true", help="Skip IFU outputs that already exist")
    p.add_argument("--ifus", type=str, default=None, help="Optional list of amplifier indices to process (e.g., '0,1,2' or '0 1 2')")
    p.add_argument("--wave-mask-frac", type=float, default=0.8, help="Central fraction of wavelengths to use for robust per-fiber ratios (default: 0.8)")
    p.add_argument("--k-source", type=float, default=5.0, help="MAD threshold multiplier for source masking across fibers (default: 5.0)")
    p.add_argument("--log-level", default="INFO", help="Logging level (default: INFO)")
    p.add_argument("--make-plots", action="store_true", help="Write example IFU diagnostic plots (delta_mult vs fiber)")
    p.add_argument("--max-plots", type=int, default=6, help="Maximum number of diagnostic plots to write (default: 6)")
    return p


essential_keys = ("ifu_dir", "n_written", "n_skipped")


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    setup_logging(args.log_level)

    # Validate inputs
    h5 = Path(args.h5file)
    if not h5.exists():
        parser.error(f"HDF5 file not found: {h5}")
    outdir = Path(args.outdir)
    if not outdir.exists():
        logger.warning("Output directory %s does not exist yet; proceeding (it will be created if needed)", outdir)

    ifu_list = _parse_ifus(args.ifus)
    opts = IFUOptions(
        wave_mask_frac=float(args.wave_mask_frac),
        k_source=float(args.k_source),
        make_plots=bool(args.make_plots),
        max_plots=int(args.max_plots),
    )

    result = run_ifu(str(h5), str(outdir), ifu_indices=ifu_list, resume=bool(args.resume), options=opts)

    # Log a concise summary
    n_written = int(result.get("n_written", 0))
    n_skipped = int(result.get("n_skipped", 0))
    ifu_dir = result.get("ifu_dir", str(outdir / "ifu"))
    logger.info("IFU refinement complete: wrote %d, skipped %d; outputs in %s", n_written, n_skipped, ifu_dir)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
