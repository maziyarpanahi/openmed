#!/usr/bin/env python3
"""Regenerate the reviewed synthetic zh/hi/ta throughput corpora."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from openmed.eval.i18n_throughput import (
    DEFAULT_FIXTURE_DIR,
    I18N_THROUGHPUT_MIN_CHARS,
    write_synthetic_corpora,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_FIXTURE_DIR,
    )
    parser.add_argument(
        "--target-chars",
        type=int,
        default=I18N_THROUGHPUT_MIN_CHARS,
    )
    args = parser.parse_args(argv)

    for path in write_synthetic_corpora(
        args.output_dir,
        target_chars=args.target_chars,
    ):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
