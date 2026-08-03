#!/usr/bin/env python3
"""Run an OpenMed GLiNER zero-shot NER model with MLX.

Install the runtime with ``python -m pip install "openmed[mlx,hf]"``.
The first run downloads the selected MLX repository from Hugging Face.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from openmed.mlx import create_mlx_pipeline

DEFAULT_MODEL = "OpenMed/OpenMed-ZeroShot-NER-Anatomy-Small-166M-mlx"
DEFAULT_TEXT = "Synthetic note: a lesion was observed in the left kidney cortex."
DEFAULT_LABELS = ("organ", "tissue", "anatomy")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--labels", nargs="+", default=list(DEFAULT_LABELS))
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    pipeline = create_mlx_pipeline(args.model)
    entities = pipeline.predict_entities(
        args.text,
        args.labels,
        threshold=args.threshold,
    )

    for entity in entities:
        print(
            f"{entity['label']:<20} "
            f"{entity['text']!r} score={entity['score']:.3f} "
            f"span=({entity['start']}, {entity['end']})"
        )


if __name__ == "__main__":
    main()
