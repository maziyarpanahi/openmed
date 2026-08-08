"""Eval-only loader for credentialed MIMIC-IV-Ext-BHC pairs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from ._dua import (
    fixture_id,
    load_json_rows,
    require_credentialed_path,
    source_path_hash,
)
from ._task_fixtures import DocumentSummaryFixture
from .dua_stubs import DUACredentialRequired
from .licenses import license_for

MIMIC_IV_BHC = "mimic-iv-bhc"
MIMIC_IV_BHC_PATH_ENV = "OPENMED_MIMIC_IV_BHC_PATH"
MIMIC_IV_BHC_AUTHORITY = "UW / PhysioNet"
MIMIC_IV_BHC_DUA = "UW / PhysioNet credentialed access"
MIMIC_IV_BHC_TASK = "document_summary_pair"

_JSON_SUFFIXES = frozenset({".json", ".jsonl", ".ndjson"})

MIMICIVBHCFixture = DocumentSummaryFixture
DocumentSummaryPairFixture = DocumentSummaryFixture


def load_mimic_iv_bhc(
    path: str | Path | None = None,
) -> list[DocumentSummaryFixture]:
    """Load document-summary pairs from an authorized local export."""

    root = require_credentialed_path(
        path,
        dataset=MIMIC_IV_BHC,
        authority=MIMIC_IV_BHC_AUTHORITY,
        env_var=MIMIC_IV_BHC_PATH_ENV,
    )
    fixtures: list[DocumentSummaryFixture] = []
    for source in _files(root):
        for row_number, row in enumerate(
            load_json_rows(
                source,
                dataset=MIMIC_IV_BHC,
                authority=MIMIC_IV_BHC_AUTHORITY,
            ),
            start=1,
        ):
            fixtures.append(
                _fixture_from_row(row, source=source, root=root, row_number=row_number)
            )
    if not fixtures:
        raise DUACredentialRequired(
            f"{MIMIC_IV_BHC_AUTHORITY} credentialed {MIMIC_IV_BHC} path contains "
            "no supported fixtures; no corpus rows were loaded"
        )
    _validate_unique_fixture_ids(fixtures)
    return fixtures


load_mimic_iv_bhc_fixtures = load_mimic_iv_bhc
load_mimic_bhc_fixtures = load_mimic_iv_bhc


def mimic_iv_bhc_suite_metadata() -> dict[str, Any]:
    """Return row-free metadata for the BHC summarization view."""

    return {
        "access": (
            f"credentialed local path only; pass path=... or set "
            f"{MIMIC_IV_BHC_PATH_ENV}"
        ),
        "dataset": MIMIC_IV_BHC,
        "dua": MIMIC_IV_BHC_DUA,
        "eval_only": True,
        "license": license_for(MIMIC_IV_BHC).to_dict(),
        "network_fetch": False,
        "redistribution": "never; read-only from user-supplied credentialed path",
        "suite": MIMIC_IV_BHC,
        "task": "summarization",
        "task_view": MIMIC_IV_BHC_TASK,
    }


def _fixture_from_row(
    row: Mapping[str, Any],
    *,
    source: Path,
    root: Path,
    row_number: int,
) -> DocumentSummaryFixture:
    document = _first_text(
        row,
        "document",
        "note",
        "text",
        "input",
        "source",
        "article",
    )
    summary = _first_text(
        row,
        "summary",
        "reference_summary",
        "target",
        "output",
        "abstract",
    )
    record_id = _record_id(row, fallback=f"row-{row_number}")
    return DocumentSummaryFixture(
        fixture_id=fixture_id(MIMIC_IV_BHC, source, root, record_id),
        document=document,
        summary=summary,
        language=str(row.get("language") or row.get("lang") or "en"),
        metadata=_metadata(source, root),
    )


def _metadata(source: Path, root: Path) -> dict[str, Any]:
    return {
        "cache_corpus_rows": False,
        "dataset": MIMIC_IV_BHC,
        "dua": MIMIC_IV_BHC_DUA,
        "eval_only": True,
        "license": license_for(MIMIC_IV_BHC).to_dict(),
        "network_fetch": False,
        "redistribution": "never; read-only from user-supplied credentialed path",
        "source_path_hash": source_path_hash(source, root),
        "suite": MIMIC_IV_BHC,
        "task": "summarization",
        "task_view": MIMIC_IV_BHC_TASK,
    }


def _first_text(row: Mapping[str, Any], *keys: str) -> str:
    for key in keys:
        value = row.get(key)
        if isinstance(value, Mapping):
            nested = value.get("text") or value.get("content") or value.get("value")
            if isinstance(nested, str) and nested.strip():
                return nested
        elif isinstance(value, str) and value.strip():
            return value
    raise ValueError(f"MIMIC-IV-Ext-BHC rows require one of: {', '.join(keys)}")


def _record_id(row: Mapping[str, Any], *, fallback: str) -> str:
    for key in ("id", "record_id", "document_id", "note_id", "uid"):
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


def _validate_unique_fixture_ids(fixtures: list[DocumentSummaryFixture]) -> None:
    ids = [fixture.fixture_id for fixture in fixtures]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate {MIMIC_IV_BHC} fixture ids")


__all__ = [
    "DocumentSummaryFixture",
    "DocumentSummaryPairFixture",
    "MIMICIVBHCFixture",
    "MIMIC_IV_BHC",
    "MIMIC_IV_BHC_AUTHORITY",
    "MIMIC_IV_BHC_DUA",
    "MIMIC_IV_BHC_PATH_ENV",
    "MIMIC_IV_BHC_TASK",
    "load_mimic_bhc_fixtures",
    "load_mimic_iv_bhc",
    "load_mimic_iv_bhc_fixtures",
    "mimic_iv_bhc_suite_metadata",
]
