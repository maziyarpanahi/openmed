"""CMeEE loader for explicit, user-supplied local benchmark paths."""

from __future__ import annotations

from pathlib import Path

from openmed.eval.datasets.multilingual_ner import (
    CMEEE,
    MultilingualNerLoadResult,
    load_multilingual_ner_benchmark,
    map_multilingual_ner_label,
    source_for,
)

CMEEE_SOURCE = source_for(CMEEE)


def load_cmeee(
    path: str | Path | None = None,
    *,
    split: str = "test",
    allow_repo_path: bool = False,
) -> MultilingualNerLoadResult:
    """Load CMeEE records from an explicit local path.

    OpenMed does not bundle CMeEE corpus text. Pass a local path to data you
    are licensed to use; synthetic smoke fixtures may opt into
    ``allow_repo_path=True`` in tests.
    """

    return load_multilingual_ner_benchmark(
        CMEEE,
        path,
        split=split,
        allow_repo_path=allow_repo_path,
    )


def map_cmeee_label(label: str):
    """Map a CMeEE source label to an OpenMed canonical label."""

    return map_multilingual_ner_label(CMEEE, label)


__all__ = [
    "CMEEE",
    "CMEEE_SOURCE",
    "load_cmeee",
    "map_cmeee_label",
]
