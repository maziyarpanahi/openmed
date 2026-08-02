"""DEFT loader for explicit, user-supplied local benchmark paths."""

from __future__ import annotations

from pathlib import Path

from openmed.eval.datasets.multilingual_ner import (
    DEFT,
    MultilingualNerLoadResult,
    load_multilingual_ner_benchmark,
    map_multilingual_ner_label,
    source_for,
)

DEFT_SOURCE = source_for(DEFT)


def load_deft(
    path: str | Path | None = None,
    *,
    split: str = "test",
    allow_repo_path: bool = False,
) -> MultilingualNerLoadResult:
    """Load DEFT records from an explicit local path.

    OpenMed does not bundle DEFT corpus text. Pass a local path to data you are
    licensed to use; synthetic smoke fixtures may opt into
    ``allow_repo_path=True`` in tests.
    """

    return load_multilingual_ner_benchmark(
        DEFT,
        path,
        split=split,
        allow_repo_path=allow_repo_path,
    )


def map_deft_label(label: str):
    """Map a DEFT source label to an OpenMed canonical label."""

    return map_multilingual_ner_label(DEFT, label)


__all__ = [
    "DEFT",
    "DEFT_SOURCE",
    "load_deft",
    "map_deft_label",
]
