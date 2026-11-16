"""Cross-runtime clinical assertion vectors recomputed from Python ConText."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import pytest

from openmed.clinical.context import assert_context
from scripts.parity import export_context_cues

PARITY = Path(__file__).resolve().parents[2] / "fixtures" / "parity"
VECTORS = json.loads((PARITY / "clinical_assertion_v1.json").read_text("utf-8"))
CASES = VECTORS["cases"]
AXES = ("experiencer", "negation", "temporality", "uncertainty")
GAP_WORDS = {"negation_scope_ends_at_and": "and", "negation_scope_ends_at_or": "or"}


def test_header_marks_a_synthetic_versioned_english_reference() -> None:
    assert VECTORS["version"] == 1
    assert VECTORS["synthetic"] is True
    assert VECTORS["offset_unit"] == "unicode_scalar"
    assert VECTORS["reference"] == "openmed.clinical.context.assert_context"
    assert VECTORS["language"] == "en"
    assert tuple(sorted(VECTORS["axes"])) == AXES
    assert len({case["id"] for case in CASES}) == len(CASES)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["id"])
def test_case_recomputes_exactly_with_its_recorded_sentences(case: dict) -> None:
    text, target = case["text"], case["target"]
    assert text[target["start"] : target["end"]] == target["text"]
    assert case["non_ascii"] is any(ord(char) > 127 for char in text)
    for sentence in case["sentences"]:
        assert 0 <= sentence["start"] < sentence["end"] <= len(text)

    result = assert_context(text, [dict(target)], sentences=case["sentences"])[0]

    assert {axis: result[axis] for axis in AXES} == case["expected"]
    for axis, value in case["expected"].items():
        assert value in VECTORS["axes"][axis]


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["id"])
def test_reference_gap_is_recorded_exactly_when_output_contradicts_category(
    case: dict,
) -> None:
    intended = VECTORS["categories"][case["category"]]["intended"]
    contradicts = any(
        case["expected"].get(axis) != value for axis, value in intended.items()
    )
    gap = case["reference_gap"]

    assert (gap is not None) is contradicts
    if gap is None:
        return
    assert gap in VECTORS["reference_gap_codes"]
    if gap == "no_conditional_axis":
        assert case["category"] == "conditional"
    else:
        before_target = case["text"][: case["target"]["start"]]
        assert re.search(rf"\b{GAP_WORDS[gap]}\b", before_target)


def test_every_category_has_three_cases_and_required_controls_exist() -> None:
    counts = Counter(case["category"] for case in CASES)

    assert set(counts) == set(VECTORS["categories"])
    assert min(counts.values()) >= 3
    assert any(case["non_ascii"] for case in CASES)
    assert any(len(case["sentences"]) > 1 for case in CASES)
    assert {case["reference_gap"] for case in CASES} - {None} == set(
        VECTORS["reference_gap_codes"]
    )


def test_cue_table_matches_the_live_lexicon_and_its_recorded_digest() -> None:
    raw = (PARITY / "context_cues_en_v1.json").read_bytes()

    assert raw.decode("utf-8") == export_context_cues.cue_table_text()
    assert VECTORS["cue_table_sha256"] == hashlib.sha256(raw).hexdigest()
    table = json.loads(raw)
    for values in (*table["lexicon"].values(), *table["experiencer"].values()):
        if isinstance(values, list):
            assert values == sorted(values)


def test_editing_a_cue_without_regenerating_fails_the_check(monkeypatch) -> None:
    lexicon = export_context_cues.get_clinical_cue_lexicon("en")
    edited = type(lexicon)(
        **{**lexicon.__dict__, "negation": (*lexicon.negation, "nary")}
    )
    monkeypatch.setattr(
        export_context_cues, "get_clinical_cue_lexicon", lambda _language: edited
    )

    raw = (PARITY / "context_cues_en_v1.json").read_text("utf-8")
    assert raw != export_context_cues.cue_table_text()
    assert export_context_cues.main(["--check"]) == 1
