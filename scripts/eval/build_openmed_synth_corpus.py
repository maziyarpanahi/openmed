#!/usr/bin/env python3
"""CLI wrapper for the packaged OpenMed synthetic corpus generator."""

from openmed.eval.synth_corpus import (
    CORPUS_ID,
    CORPUS_LICENSE,
    CORPUS_VERSION,
    DEFAULT_CORPUS_SIZE,
    DEFAULT_SEED,
    corpus_content_hash,
    generate_corpus,
    label_distribution,
    main,
    render_corpus,
    write_corpus,
)

__all__ = [
    "CORPUS_ID",
    "CORPUS_LICENSE",
    "CORPUS_VERSION",
    "DEFAULT_CORPUS_SIZE",
    "DEFAULT_SEED",
    "corpus_content_hash",
    "generate_corpus",
    "label_distribution",
    "main",
    "render_corpus",
    "write_corpus",
]


if __name__ == "__main__":
    raise SystemExit(main())
