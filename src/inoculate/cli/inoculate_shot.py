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

from ..io import amp_exposure_slice
from ..utils.logging import setup_logging


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="inoculate-shot", description="Inoculate shot runner (prototype)")
    p.add_argument("h5file", nargs="?", help="Path to VIRUS spectral HDF5 file (optional for --show-slice)")
    p.add_argument("--amp", type=int, default=0, help="Amplifier index (0-based)")
    p.add_argument("--exp", type=int, default=0, help="Exposure index (0..2)")
    p.add_argument("--show-slice", action="store_true", help="Print the fiber slice for amp/exp and exit")
    p.add_argument("--log-level", default="INFO", help="Logging level (default: INFO)")
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    setup_logging(args.log_level)

    if args.show_slice:
        s = amp_exposure_slice(args.amp, args.exp)
        print(f"amp={args.amp} exp={args.exp} -> slice({s.start}, {s.stop})")
        return 0

    if not args.h5file:
        parser.error("h5file is required unless --show-slice is used")

    # Placeholder for future pipeline execution.
    path = Path(args.h5file)
    if not path.exists():
        parser.error(f"HDF5 file not found: {path}")

    # For now, just acknowledge the file; real processing to come.
    print(f"Inoculate prototype: would process shot file: {path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
