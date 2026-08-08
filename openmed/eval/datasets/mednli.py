"""Eval-only loader for credentialed MedNLI sentence pairs."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Mapping

from ._dua import (
    fixture_id,
    load_json_rows,
    require_credentialed_path,
    source_path_hash,
)
from ._task_fixtures import SentencePairFixture
from .dua_stubs import DUACredentialRequired
from .licenses import license_for

MEDNLI = "mednli"
MEDNLI_PATH_ENV = "OPENMED_MEDNLI_PATH"
MEDNLI_AUTHORITY = "PhysioNet"
MEDNLI_DUA = "PhysioNet credentialed clinical-data access"
MEDNLI_TASK = "sentence_pair_nli"
MEDNLI_LABELS: tuple[str, ...] = ("entailment", "contradiction", "neutral")

_JSON_SUFFIXES = frozenset({".json", ".jsonl", ".ndjson"})
_LABEL_ALIASES = {
    "contradiction": "contradiction",
    "contradicts": "contradiction",
    "entailment": "entailment",
    "entails": "entailment",
    "neutral": "neutral",
}

MedNLIFixture = SentencePairFixture
SentencePairNLIFixture = SentencePairFixture


def load_mednli(path: str | Path | None = None) -> list[SentencePairFixture]:
    """Load MedNLI JSON/JSONL rows from an authorized local path."""

    root = require_credentialed_path(
        path,
        dataset=MEDNLI,
        authority=MEDNLI_AUTHORITY,
        env_var=MEDNLI_PATH_ENV,
    )
    fixtures: list[SentencePairFixture] = []
    for source in _files(root):
        for row_number, row in enumerate(
            load_json_rows(source, dataset=MEDNLI, authority=MEDNLI_AUTHORITY),
            start=1,
        ):
            fixtures.append(
                _fixture_from_row(row, source=source, root=root, row_number=row_number)
            )
    if not fixtures:
        raise DUACredentialRequired(
            f"{MEDNLI_AUTHORITY} credentialed {MEDNLI} path contains no supported "
            "fixtures; no corpus rows were loaded"
        )
    _validate_unique_fixture_ids(fixtures)
    return fixtures


load_mednli_fixtures = load_mednli
load_mednli_sentence_pairs = load_mednli


def normalize_mednli_label(label: str) -> str:
    """Normalize a MedNLI gold label to the three-way task vocabulary."""

    key = re.sub(r"[^a-z]+", "_", str(label).strip().casefold()).strip("_")
    try:
        return _LABEL_ALIASES[key]
    except KeyError as exc:
        allowed = ", ".join(MEDNLI_LABELS)
        raise ValueError(
            f"unknown MedNLI label {label!r}; expected one of: {allowed}"
        ) from exc


def mednli_suite_metadata() -> dict[str, Any]:
    """Return row-free metadata for the MedNLI sentence-pair view."""

    return {
        "access": (
            f"credentialed local path only; pass path=... or set {MEDNLI_PATH_ENV}"
        ),
        "dataset": MEDNLI,
        "dua": MEDNLI_DUA,
        "eval_only": True,
        "license": license_for(MEDNLI).to_dict(),
        "network_fetch": False,
        "redistribution": "never; read-only from user-supplied credentialed path",
        "suite": MEDNLI,
        "task": "nli",
        "task_view": MEDNLI_TASK,
        "labels": list(MEDNLI_LABELS),
    }


def _fixture_from_row(
    row: Mapping[str, Any],
    *,
    source: Path,
    root: Path,
    row_number: int,
) -> SentencePairFixture:
    premise = _first_text(
        row,
        "premise",
        "sentence1",
        "sentence1_raw",
        "text1",
    )
    hypothesis = _first_text(
        row,
        "hypothesis",
        "sentence2",
        "sentence2_raw",
        "text2",
    )
    raw_label = row.get("label") or row.get("gold_label") or row.get("goldLabel")
    if raw_label is None:
        raise ValueError("MedNLI rows require label or gold_label")
    record_id = _record_id(row, fallback=f"row-{row_number}")
    return SentencePairFixture(
        fixture_id=fixture_id(MEDNLI, source, root, record_id),
        premise=premise,
        hypothesis=hypothesis,
        label=normalize_mednli_label(str(raw_label)),
        language=str(row.get("language") or row.get("lang") or "en"),
        metadata=_metadata(source, root),
    )


def _metadata(source: Path, root: Path) -> dict[str, Any]:
    return {
        "cache_corpus_rows": False,
        "dataset": MEDNLI,
        "dua": MEDNLI_DUA,
        "eval_only": True,
        "license": license_for(MEDNLI).to_dict(),
        "network_fetch": False,
        "redistribution": "never; read-only from user-supplied credentialed path",
        "source_path_hash": source_path_hash(source, root),
        "suite": MEDNLI,
        "task": "nli",
        "task_view": MEDNLI_TASK,
    }


def _first_text(row: Mapping[str, Any], *keys: str) -> str:
    for key in keys:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value
    raise ValueError(f"MedNLI rows require one of: {', '.join(keys)}")


def _record_id(row: Mapping[str, Any], *, fallback: str) -> str:
    for key in ("id", "pairID", "pair_id", "record_id", "uid"):
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return fallback


def _files(root: Path) -> tuple[Path, ...]:
    wanted = {suffix.casefold() for suffix in _JSON_SUFFIXES}
    if root.is_file():
        return (root,) if root.suffix.casefold() in wanted else tuple()
    return tuple(
        path
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.suffix.casefold() in wanted
    )


def _validate_unique_fixture_ids(fixtures: list[SentencePairFixture]) -> None:
    ids = [fixture.fixture_id for fixture in fixtures]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate {MEDNLI} fixture ids")


__all__ = [
    "MEDNLI",
    "MEDNLI_AUTHORITY",
    "MEDNLI_DUA",
    "MEDNLI_LABELS",
    "MEDNLI_PATH_ENV",
    "MEDNLI_TASK",
    "MedNLIFixture",
    "SentencePairFixture",
    "SentencePairNLIFixture",
    "load_mednli",
    "load_mednli_fixtures",
    "load_mednli_sentence_pairs",
    "mednli_suite_metadata",
    "normalize_mednli_label",
]
