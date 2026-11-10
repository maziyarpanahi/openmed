"""Fixture-driven clinical context regression traps for OM-295."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from pathlib import Path

from openmed.clinical import (
    AFFIRMED,
    CERTAIN,
    HISTORICAL,
    HYPOTHETICAL,
    NEGATED,
    RECENT,
    UNCERTAIN,
    resolve_span_context,
)
from openmed.clinical.context import (
    HISTORICAL_CUES,
    HYPOTHETICAL_CUES,
    NEGATION_CUES,
    PSEUDO_NEGATION_CUES,
    UNCERTAINTY_CUES,
)

ROOT = Path(__file__).resolve().parents[3]
FIXTURE = ROOT / "tests" / "fixtures" / "clinical" / "context_traps.jsonl"
FORBIDDEN_FIXTURE_MARKERS = (
    "cpt",
    "dua",
    "i2b2",
    "mimic",
    "n2c2",
    "snomed",
    "umls",
)
REQUIRED_TRAPS = {
    "pseudo_cue",
    "conjunction_termination",
    "sentence_boundary",
    "cue_over_section",
    "hypothetical_over_historical",
}
SECTION_PRIOR_HITS = {
    HISTORICAL: "history of",
}
TERMINATOR_RE = re.compile(r"(?:[.!?;]|\b(?:and|but|however|or)\b)", re.IGNORECASE)


def test_context_trap_fixture_file_is_synthetic_and_complete() -> None:
    meta, rows = _load_jsonl_suite(FIXTURE)

    assert meta["version"] == 1
    assert meta["suite"] == "clinical_context_traps"
    assert rows
    assert REQUIRED_TRAPS <= {row["trap"] for row in rows}
    assert all(row.get("synthetic") is True for row in rows)

    fixture_text = FIXTURE.read_text(encoding="utf-8").casefold()
    for marker in FORBIDDEN_FIXTURE_MARKERS:
        assert (
            re.search(rf"(?<![a-z0-9]){re.escape(marker)}(?![a-z0-9])", fixture_text)
            is None
        )


def test_context_trap_fixtures_match_scoped_resolver_outputs() -> None:
    _, rows = _load_jsonl_suite(FIXTURE)

    for row in rows:
        included, excluded = _scan_scoped_modifier_hits(row)
        expected = row["expected"]
        assert included == expected["scope_modifier_hits"], row["case_id"]
        assert excluded == expected["excluded_modifier_hits"], row["case_id"]

        effective_hits = _apply_section_prior(row, included)
        assert effective_hits == expected["effective_modifier_hits"], row["case_id"]

        context = resolve_span_context(row["target"]["text"], effective_hits)
        assert context.temporality == expected["temporality"], row["case_id"]
        assert context.certainty == expected["certainty"], row["case_id"]
        assert context.negation == expected["negation"], row["case_id"]


def test_seeded_context_regression_fails_loudly() -> None:
    _, rows = _load_jsonl_suite(FIXTURE)
    seeded = dict(rows[0])
    seeded["expected"] = {
        **seeded["expected"],
        "negation": NEGATED,
    }

    included, _ = _scan_scoped_modifier_hits(seeded)
    effective_hits = _apply_section_prior(seeded, included)
    context = resolve_span_context(seeded["target"]["text"], effective_hits)

    assert context.negation != seeded["expected"]["negation"]


def _load_jsonl_suite(path: Path) -> tuple[dict, list[dict]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    meta = rows[0]
    assert meta["kind"] == "meta"
    return meta, rows[1:]


def _scan_scoped_modifier_hits(row: dict) -> tuple[list[str], list[str]]:
    text = row["text"]
    target_start, target_end = _target_offsets(text, row["target"])
    included: list[str] = []
    excluded: list[str] = []

    for cue, start, end in _iter_context_cues(text):
        destination = (
            included
            if _cue_reaches_target(text, start, end, target_start, target_end)
            else excluded
        )
        if cue not in destination:
            destination.append(cue)

    return included, excluded


def _target_offsets(text: str, target: dict) -> tuple[int, int]:
    occurrence = target.get("occurrence", 0)
    offset = -1
    for _ in range(occurrence + 1):
        offset = text.index(target["text"], offset + 1)
    return offset, offset + len(target["text"])


def _iter_context_cues(text: str) -> Iterable[tuple[str, int, int]]:
    cue_re = _cue_pattern(
        (
            *HISTORICAL_CUES,
            *HYPOTHETICAL_CUES,
            *UNCERTAINTY_CUES,
            *NEGATION_CUES,
            *PSEUDO_NEGATION_CUES,
        )
    )
    seen: set[tuple[int, int]] = set()
    for match in cue_re.finditer(text):
        span = match.span()
        if span in seen:
            continue
        if _is_section_header_cue(text, match.end()):
            continue
        seen.add(span)
        yield match.group(0).casefold(), *span


def _cue_pattern(cues: Iterable[str]) -> re.Pattern[str]:
    alternation = "|".join(
        r"\s+".join(re.escape(part) for part in cue.split())
        for cue in sorted(set(cues), key=len, reverse=True)
    )
    return re.compile(rf"(?<!\w)(?:{alternation})(?!\w)", re.IGNORECASE)


def _is_section_header_cue(text: str, cue_end: int) -> bool:
    return text[cue_end:].lstrip().startswith(":")


def _cue_reaches_target(
    text: str,
    cue_start: int,
    cue_end: int,
    target_start: int,
    target_end: int,
) -> bool:
    if cue_end <= target_start:
        between = text[cue_end:target_start]
    elif target_end <= cue_start:
        between = text[target_end:cue_start]
    else:
        between = ""
    return TERMINATOR_RE.search(between) is None


def _apply_section_prior(row: dict, scoped_hits: list[str]) -> list[str]:
    effective_hits = list(scoped_hits)
    section_prior = row.get("section", {}).get("temporality_prior")
    if section_prior and not _has_temporal_hit(scoped_hits):
        effective_hits.append(SECTION_PRIOR_HITS[section_prior])
    return effective_hits


def _has_temporal_hit(hits: list[str]) -> bool:
    temporal_cues = {cue.casefold() for cue in (*HISTORICAL_CUES, *HYPOTHETICAL_CUES)}
    return any(hit.casefold() in temporal_cues for hit in hits)


def test_expected_context_values_are_public_constants() -> None:
    _, rows = _load_jsonl_suite(FIXTURE)
    expected_values = {
        "temporality": {RECENT, HISTORICAL, HYPOTHETICAL},
        "certainty": {CERTAIN, UNCERTAIN},
        "negation": {AFFIRMED, NEGATED},
    }

    for row in rows:
        for axis, values in expected_values.items():
            assert row["expected"][axis] in values, (row["case_id"], axis)
