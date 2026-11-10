#!/usr/bin/env python3
"""Run a pre-converted OpenMed token-classification model with MLX.

Install the runtime with ``python -m pip install "openmed[mlx,hf]"``.
The first run downloads the selected MLX repository from Hugging Face.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from openmed.mlx import create_mlx_pipeline

DEFAULT_MODEL = "OpenMed/OpenMed-NER-AnatomyDetect-ElectraMed-33M-mlx"
DEFAULT_TEXT = "Synthetic note: the biopsy sampled liver tissue near the portal vein."


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--text", default=DEFAULT_TEXT)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    pipeline = create_mlx_pipeline(args.model, aggregation_strategy="simple")

    for entity in pipeline(args.text):
        print(
            f"{entity['entity_group']:<20} "
            f"{entity['word']!r} score={entity['score']:.3f} "
            f"span=({entity['start']}, {entity['end']})"
        )


if __name__ == "__main__":
    main()
