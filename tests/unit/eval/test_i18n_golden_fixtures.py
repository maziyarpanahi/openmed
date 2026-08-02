"""Per-language i18n golden fixture recovery tests (OM-100).

Every language-named JSONL fixture under ``openmed/eval/golden/fixtures/i18n/``
is a synthetic, no-DUA clinical note that exercises its language pack's own
date / phone / national-ID / address / postcode patterns. Relation fixture
packs are co-located under ``i18n`` but have their own schema and test suite.
This suite runs
:func:`get_patterns_for_language` over each fixture and asserts that every gold
span is recovered at its exact offset with the correct canonical label, and that
each national-ID span whose language has a checksum validator passes it. A new
language pack that mis-detects a native ID or shifts an offset fails loudly here.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from openmed.core.labels import ID_NUM, normalize_label
from openmed.core.pii_i18n import (
    LANGUAGE_PII_PATTERNS,
    get_patterns_for_language,
)

_I18N_DIR = Path("openmed/eval/golden/fixtures/i18n")
_SPECIALIZED_CORPUS_FIXTURES = frozenset({"india_clinical.jsonl"})

# Canonical gold label -> the pattern entity_type that must recover it.
_LABEL_TO_ENTITY_TYPE = {
    "DATE": "date",
    "PHONE": "phone_number",
    "ID_NUM": "national_id",
    "STREET_ADDRESS": "street_address",
    "ZIPCODE": "postcode",
}

# The five span types every per-language fixture is expected to exercise.
_REQUIRED_LABELS = frozenset(_LABEL_TO_ENTITY_TYPE)

# Languages seeded by OM-100. The issue contract requires 3-5 notes for each
# one; older and later-added fixture packs are intentionally not included in
# that historical row-count requirement.
_OM_100_LANGUAGES = frozenset(
    {"ar", "de", "es", "fr", "hi", "it", "ja", "nl", "pt", "te", "tr"}
)
_MIN_NOTES_PER_LANGUAGE = 3
_MAX_NOTES_PER_LANGUAGE = 5

# Legacy fixtures authored before the five-families convention (OM-100) that do
# not yet carry a postcode span. They are still covered by the offset-recovery
# and checksum tests; backfilling a ZIPCODE span for each is a separate task.
_LEGACY_INCOMPLETE_LANGUAGES = frozenset({"ko", "ms", "tl"})
_NON_PII_FIXTURE_PREFIXES = ("relations_",)


def _fixture_paths() -> list[Path]:
    return sorted(
        path
        for path in _I18N_DIR.glob("*.jsonl")
        if path.name not in _SPECIALIZED_CORPUS_FIXTURES
        and not path.stem.startswith(_NON_PII_FIXTURE_PREFIXES)
    )


def _load_rows(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _recovered_spans(language: str, text: str) -> set[tuple[str, int, int, str]]:
    """Return ``(entity_type, start, end, value)`` spans validator-filtered."""

    observed: set[tuple[str, int, int, str]] = set()
    for pattern in get_patterns_for_language(language):
        for match in re.finditer(pattern.pattern, text, pattern.flags):
            value = match.group(0)
            if pattern.validator is not None and not pattern.validator(value):
                continue
            observed.add((pattern.entity_type, match.start(), match.end(), value))
    return observed


def _national_id_validators(language: str):
    return [
        pattern.validator
        for pattern in LANGUAGE_PII_PATTERNS.get(language, [])
        if normalize_label(pattern.entity_type) == ID_NUM
        and pattern.validator is not None
    ]


def test_i18n_fixture_directory_is_non_empty():
    paths = _fixture_paths()
    assert paths, "no i18n golden fixtures found"
    available_languages = {path.stem for path in paths}
    missing = _OM_100_LANGUAGES - available_languages
    assert not missing, f"missing OM-100 language fixtures: {sorted(missing)}"


@pytest.mark.parametrize(
    "path",
    [path for path in _fixture_paths() if path.stem in _OM_100_LANGUAGES],
    ids=lambda p: p.stem,
)
def test_om_100_languages_have_three_to_five_notes(path: Path):
    note_count = len(_load_rows(path))
    assert _MIN_NOTES_PER_LANGUAGE <= note_count <= _MAX_NOTES_PER_LANGUAGE, (
        f"{path.name} must contain 3-5 synthetic notes, found {note_count}"
    )


@pytest.mark.parametrize("path", _fixture_paths(), ids=lambda p: p.stem)
def test_i18n_fixture_spans_recover_with_exact_offsets(path: Path):
    rows = _load_rows(path)
    assert rows, f"{path.name} is empty"

    for row in rows:
        language = row["language"]
        assert path.stem == language, f"{path.name} declares language {language!r}"
        assert row["metadata"]["synthetic"] is True
        text = row["text"]
        observed = _recovered_spans(language, text)

        for span in row["gold_spans"]:
            label = span["label"]
            # Every gold span text must sit at its declared offsets.
            assert text[span["start"] : span["end"]] == span["text"], (
                f"{path.name}: offset mismatch for {label} {span['text']!r}"
            )
            entity_type = _LABEL_TO_ENTITY_TYPE.get(label)
            if entity_type is None:
                continue
            key = (entity_type, span["start"], span["end"], span["text"])
            assert key in observed, (
                f"{path.name}: {label} {span['text']!r} at "
                f"({span['start']},{span['end']}) was not recovered by "
                f"get_patterns_for_language({language!r})"
            )


@pytest.mark.parametrize("path", _fixture_paths(), ids=lambda p: p.stem)
def test_i18n_fixture_exercises_the_five_pattern_families(path: Path):
    if path.stem in _LEGACY_INCOMPLETE_LANGUAGES:
        pytest.skip(f"{path.name} predates the five-families convention (OM-100)")
    labels = {span["label"] for row in _load_rows(path) for span in row["gold_spans"]}
    missing = _REQUIRED_LABELS - labels
    assert not missing, f"{path.name} is missing span families: {sorted(missing)}"


@pytest.mark.parametrize("path", _fixture_paths(), ids=lambda p: p.stem)
def test_i18n_fixture_national_ids_pass_checksum_where_a_validator_exists(path: Path):
    for row in _load_rows(path):
        language = row["language"]
        validators = _national_id_validators(language)
        for span in row["gold_spans"]:
            if span["label"] != "ID_NUM":
                continue
            if span.get("metadata", {}).get("checksum_status") == "unvalidated":
                continue
            if not validators:
                # Languages such as ar/ja have a regex-only national_id pattern
                # with no checksum validator; the span is still recovered above.
                assert span.get("metadata", {}).get("checksum_status") == "unvalidated"
                continue
            assert any(v(span["text"]) for v in validators), (
                f"{path.name}: national_id {span['text']!r} fails every "
                f"registered validator for {language!r}"
            )
