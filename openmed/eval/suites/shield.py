"""SHIELD clinical PHI comparison corpus loader.

The suite is intentionally loaded by reference from the public dataset mirror.
No corpus rows are stored in this repository.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping
from urllib.parse import quote
from urllib.request import urlopen

from openmed.core.labels import (
    AGE,
    CANONICAL_LABELS,
    DATE,
    ID_NUM,
    LOCATION,
    ORGANIZATION,
    PERSON,
    PHONE,
    URL,
)
from openmed.eval.harness import BenchmarkFixture
from openmed.eval.metrics import EvalSpan

SHIELD = "shield"
CORPUS_ROLE = "comparison"
SUITE_ANNOTATION = "comparison corpus, not a high-recall gate target"
IS_HIGH_RECALL_GATE_TARGET = False

PUBLIC_SAMPLE_REPOSITORY = "tds-research-tech/shield-sample"
FULL_REPOSITORY = "tds-research-tech/shield"
PUBLIC_SAMPLE_NOTES_CONFIG = "sample_notes"
PUBLIC_SAMPLE_SPANS_CONFIG = "sample_spans"
FULL_NOTES_CONFIG = "full_notes"
FULL_SPANS_CONFIG = "full_spans"
DEFAULT_SPLIT = "train"
VERIFIED_LICENSE = "data-use-agreement"
VERIFIED_LICENSE_DATE = "2026-06-12"

SHIELD_LABEL_TO_CANONICAL: dict[str, str] = {
    "age": AGE,
    "date": DATE,
    "doctor": PERSON,
    "hospital": ORGANIZATION,
    "id": ID_NUM,
    "location": LOCATION,
    "patient": PERSON,
    "phone": PHONE,
    "web": URL,
}

RowsLoader = Callable[[str, str, str], list[Mapping[str, Any]]]


@dataclass(frozen=True)
class ShieldSource:
    """Dataset mirror coordinates for one SHIELD variant."""

    repository: str
    notes_config: str
    spans_config: str
    split: str
    variant: str
    requires_approval: bool


PUBLIC_SAMPLE_SOURCE = ShieldSource(
    repository=PUBLIC_SAMPLE_REPOSITORY,
    notes_config=PUBLIC_SAMPLE_NOTES_CONFIG,
    spans_config=PUBLIC_SAMPLE_SPANS_CONFIG,
    split=DEFAULT_SPLIT,
    variant="public_sample",
    requires_approval=False,
)

FULL_SOURCE = ShieldSource(
    repository=FULL_REPOSITORY,
    notes_config=FULL_NOTES_CONFIG,
    spans_config=FULL_SPANS_CONFIG,
    split=DEFAULT_SPLIT,
    variant="full_access_controlled",
    requires_approval=True,
)


def map_shield_label(label: str) -> str:
    """Map a SHIELD PHI category onto OpenMed's canonical label taxonomy."""
    canonical = SHIELD_LABEL_TO_CANONICAL.get(label.strip().lower())
    if canonical is None:
        allowed = ", ".join(sorted(SHIELD_LABEL_TO_CANONICAL))
        raise ValueError(f"unknown SHIELD label {label!r}; expected one of: {allowed}")
    return canonical


def shield_suite_metadata(*, use_sample: bool = True) -> dict[str, Any]:
    """Return source, license, and role metadata for SHIELD benchmark reports."""
    source = _source_for(use_sample=use_sample)
    return {
        "access": (
            "public sample is available without approval; full corpus requires "
            "approved access and a signed data-use agreement"
        ),
        "annotation": SUITE_ANNOTATION,
        "corpus_role": CORPUS_ROLE,
        "full_repository": FULL_REPOSITORY,
        "gate_target": IS_HIGH_RECALL_GATE_TARGET,
        "label_mapping": dict(sorted(SHIELD_LABEL_TO_CANONICAL.items())),
        "license": VERIFIED_LICENSE,
        "license_verified_at": VERIFIED_LICENSE_DATE,
        "notes_config": source.notes_config,
        "redistribution": "not vendored; loaded by reference",
        "repository": source.repository,
        "requires_approval": source.requires_approval,
        "source_url": f"https://huggingface.co/datasets/{source.repository}",
        "span_count_paper": 10505,
        "spans_config": source.spans_config,
        "split": source.split,
        "suite": SHIELD,
        "variant": source.variant,
    }


def load_shield_fixtures(
    *,
    use_sample: bool = True,
    rows_loader: RowsLoader | None = None,
) -> list[BenchmarkFixture]:
    """Load SHIELD notes and spans as benchmark fixtures.

    The default uses the public sample mirror. Set ``use_sample=False`` only
    on an approved machine with access to the full corpus.
    """
    source = _source_for(use_sample=use_sample)
    loader = rows_loader or _load_dataset_rows
    notes = loader(source.repository, source.notes_config, source.split)
    spans = loader(source.repository, source.spans_config, source.split)
    return fixtures_from_rows(notes, spans, source=source)


def fixtures_from_rows(
    notes: Iterable[Mapping[str, Any]],
    spans: Iterable[Mapping[str, Any]],
    *,
    source: ShieldSource = PUBLIC_SAMPLE_SOURCE,
) -> list[BenchmarkFixture]:
    """Build benchmark fixtures from SHIELD note and span table rows."""
    spans_by_note: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for span in spans:
        note_id = str(span.get("note_id", ""))
        if note_id:
            spans_by_note[note_id].append(span)

    metadata = shield_suite_metadata(use_sample=source.variant == "public_sample")
    fixtures: list[BenchmarkFixture] = []
    for note in notes:
        note_id = str(note.get("note_id", ""))
        text = str(note.get("note_text", ""))
        note_type = str(note.get("note_type") or "")
        gold_spans = tuple(
            _span_from_row(span, text=text)
            for span in sorted(
                spans_by_note.get(note_id, []),
                key=lambda row: (
                    int(row.get("span_start", 0)),
                    str(row.get("span_id", "")),
                ),
            )
        )
        fixture_metadata = dict(metadata)
        fixture_metadata.update(
            {
                "note_type": note_type,
                "source_note_id": note_id,
            }
        )
        fixtures.append(
            BenchmarkFixture(
                fixture_id=note_id,
                text=text,
                gold_spans=gold_spans,
                language="en",
                metadata=fixture_metadata,
            )
        )
    return fixtures


def _span_from_row(row: Mapping[str, Any], *, text: str) -> EvalSpan:
    raw_label = str(row.get("span_label", ""))
    canonical_label = map_shield_label(raw_label)
    start = _read_required_int(row, "span_start")
    end = _read_required_int(row, "span_end")
    if start < 0 or end < start or end > len(text):
        raise ValueError(
            f"invalid SHIELD span offsets {start}:{end} for text length {len(text)}"
        )
    return EvalSpan(
        start=start,
        end=end,
        label=canonical_label,
        text=text[start:end],
        language="en",
        metadata={
            "canonical_label": canonical_label,
            "shield_label": raw_label.strip().lower(),
            "span_id": str(row.get("span_id") or ""),
        },
    )


def _load_dataset_rows(
    repository: str, config: str, split: str
) -> list[Mapping[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError:
        return _load_dataset_rows_via_server(repository, config, split)

    try:
        dataset = load_dataset(repository, config, split=split)
    except Exception as exc:
        if repository == PUBLIC_SAMPLE_REPOSITORY:
            return _load_dataset_rows_via_server(repository, config, split)
        raise RuntimeError(
            f"failed to load approved SHIELD rows for {repository}/{config}/{split}: {exc}"
        ) from exc
    return [dict(row) for row in dataset]


def _load_dataset_rows_via_server(
    repository: str,
    config: str,
    split: str,
    *,
    page_size: int = 100,
) -> list[Mapping[str, Any]]:
    encoded_repository = quote(repository, safe="")
    encoded_config = quote(config, safe="")
    encoded_split = quote(split, safe="")
    rows: list[Mapping[str, Any]] = []
    offset = 0

    while True:
        url = (
            "https://datasets-server.huggingface.co/rows"
            f"?dataset={encoded_repository}"
            f"&config={encoded_config}"
            f"&split={encoded_split}"
            f"&offset={offset}"
            f"&length={page_size}"
        )
        try:
            with urlopen(url, timeout=30) as response:  # nosec: trusted fixed host
                payload = json.loads(response.read().decode("utf-8"))
        except OSError as exc:
            raise RuntimeError(
                f"failed to load SHIELD rows for {repository}/{config}/{split}: {exc}"
            ) from exc

        page_rows = [item["row"] for item in payload.get("rows", [])]
        rows.extend(page_rows)
        total = int(payload.get("num_rows_total") or len(rows))
        if not page_rows or len(rows) >= total:
            return rows
        offset += len(page_rows)


def _read_required_int(row: Mapping[str, Any], key: str) -> int:
    try:
        return int(row[key])
    except (KeyError, TypeError, ValueError):
        raise ValueError(f"SHIELD span row missing integer {key!r}: {row!r}") from None


def _source_for(*, use_sample: bool) -> ShieldSource:
    return PUBLIC_SAMPLE_SOURCE if use_sample else FULL_SOURCE


_invalid_mapping = {
    label: canonical
    for label, canonical in SHIELD_LABEL_TO_CANONICAL.items()
    if canonical not in CANONICAL_LABELS
}
if _invalid_mapping:
    raise RuntimeError(
        f"SHIELD mapping contains non-canonical labels: {_invalid_mapping}"
    )


__all__ = [
    "SHIELD",
    "CORPUS_ROLE",
    "SUITE_ANNOTATION",
    "IS_HIGH_RECALL_GATE_TARGET",
    "PUBLIC_SAMPLE_REPOSITORY",
    "FULL_REPOSITORY",
    "VERIFIED_LICENSE",
    "SHIELD_LABEL_TO_CANONICAL",
    "ShieldSource",
    "PUBLIC_SAMPLE_SOURCE",
    "FULL_SOURCE",
    "map_shield_label",
    "shield_suite_metadata",
    "load_shield_fixtures",
    "fixtures_from_rows",
]
