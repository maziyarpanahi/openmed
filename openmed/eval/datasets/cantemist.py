"""CANTEMIST loader for explicit, user-supplied local benchmark paths."""

from __future__ import annotations

from pathlib import Path

from openmed.eval.datasets.multilingual_ner import (
    CANTEMIST,
    MultilingualNerLoadResult,
    load_multilingual_ner_benchmark,
    map_multilingual_ner_label,
    source_for,
)

CANTEMIST_SOURCE = source_for(CANTEMIST)


def load_cantemist(
    path: str | Path | None = None,
    *,
    split: str = "test",
    allow_repo_path: bool = False,
) -> MultilingualNerLoadResult:
    """Load CANTEMIST records from an explicit local path.

    OpenMed does not bundle CANTEMIST corpus text. Pass a local path to data
    you are licensed to use; synthetic smoke fixtures may opt into
    ``allow_repo_path=True`` in tests.
    """

    return load_multilingual_ner_benchmark(
        CANTEMIST,
        path,
        split=split,
        allow_repo_path=allow_repo_path,
    )


def map_cantemist_label(label: str):
    """Map a CANTEMIST source label to an OpenMed canonical label."""

    return map_multilingual_ner_label(CANTEMIST, label)


__all__ = [
    "CANTEMIST",
    "CANTEMIST_SOURCE",
    "load_cantemist",
    "map_cantemist_label",
]
