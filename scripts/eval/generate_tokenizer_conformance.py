#!/usr/bin/env python3
"""Regenerate the portable tokenizer conformance fixtures.

The generator builds a tiny synthetic WordPiece ``tokenizer.json`` from the
invented text in :mod:`openmed.eval.tokenizer_conformance`, encodes every
synthetic case with the standard-library reference semantics, and rewrites the
vectors and the digest manifest. It reads no clock, network, environment or
model download, so repeated runs emit identical bytes.

``tokenizer_runtime_support_v1.json`` is hand-reviewed: the generator reads
and validates it and records its digest in the manifest, but never rewrites
it. ``--check`` fails when any committed file differs from a fresh generation
or when the support declarations reference unknown fields or features. Output
names fixture roles and reason codes only.

Usage::

    python scripts/eval/generate_tokenizer_conformance.py
    python scripts/eval/generate_tokenizer_conformance.py --check
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from openmed.eval.tokenizer_conformance import (  # noqa: E402
    DEFAULT_FIXTURE_DIR,
    check_fixture_dir,
    write_fixture_dir,
)


def main(argv: Sequence[str] | None = None) -> int:
    """Regenerate or check the tokenizer conformance fixtures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fixture-dir",
        type=Path,
        default=ROOT / DEFAULT_FIXTURE_DIR,
        help="tokenizer conformance fixture directory",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify the committed fixtures match a fresh generation",
    )
    args = parser.parse_args(argv)

    if args.check:
        problems = check_fixture_dir(args.fixture_dir)
        for problem in problems:
            print(f"tokenizer conformance fixture problem: {problem}")
        if problems:
            return 1
        print("tokenizer conformance fixtures match the generator")
        return 0

    try:
        changed = write_fixture_dir(args.fixture_dir)
    except ValueError as error:
        # Raised with controlled support-declaration codes only.
        print(f"tokenizer conformance generation failed: {error}")
        return 1
    except OSError as error:
        print(f"tokenizer conformance generation failed: {type(error).__name__}")
        return 1
    print(f"tokenizer conformance fixtures updated: {len(changed)} file(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
