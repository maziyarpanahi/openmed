"""PharmaCoNER loader for explicit, user-supplied local benchmark paths."""

from __future__ import annotations

from pathlib import Path

from openmed.eval.datasets.multilingual_ner import (
    PHARMACONER,
    MultilingualNerLoadResult,
    load_multilingual_ner_benchmark,
    map_multilingual_ner_label,
    source_for,
)

PHARMACONER_SOURCE = source_for(PHARMACONER)


def load_pharmaconer(
    path: str | Path | None = None,
    *,
    split: str = "test",
    allow_repo_path: bool = False,
) -> MultilingualNerLoadResult:
    """Load PharmaCoNER records from an explicit local path.

    OpenMed does not bundle PharmaCoNER corpus text. Pass a local path to data
    you are licensed to use; synthetic smoke fixtures may opt into
    ``allow_repo_path=True`` in tests.
    """

    return load_multilingual_ner_benchmark(
        PHARMACONER,
        path,
        split=split,
        allow_repo_path=allow_repo_path,
    )


def map_pharmaconer_label(label: str):
    """Map a PharmaCoNER source label to an OpenMed canonical label."""

    return map_multilingual_ner_label(PHARMACONER, label)


__all__ = [
    "PHARMACONER",
    "PHARMACONER_SOURCE",
    "load_pharmaconer",
    "map_pharmaconer_label",
]
