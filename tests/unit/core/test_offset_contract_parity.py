from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from openmed.core.decoding import (
    coerce_token_classification_spans,
    is_grapheme_boundary,
    snap_span_to_grapheme_boundaries,
    token_spans_to_char_spans,
)

FIXTURE_PATH = (
    Path(__file__).parents[2] / "fixtures" / "parity" / "offset_contract.json"
)
FIXTURE = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
CASES: list[dict[str, Any]] = FIXTURE["cases"]


def test_shared_offset_contract_fixture_has_required_coverage() -> None:
    assert FIXTURE["version"] == 1
    assert FIXTURE["offset_unit"] == "unicode_scalar"
    assert len(CASES) >= 40

    categories = {case["category"] for case in CASES}
    assert {"cjk", "indic", "joiner", "mixed"} <= categories

    indic_scripts = {case["script"] for case in CASES if case["category"] == "indic"}
    assert indic_scripts == {
        "Deva",
        "Beng",
        "Guru",
        "Gujr",
        "Orya",
        "Taml",
        "Telu",
        "Knda",
        "Mlym",
    }
    assert {"zh-Hans", "zh-Hant"} <= {
        case["script"] for case in CASES if case["category"] == "cjk"
    }
    assert any("\u200d" in case["text"] for case in CASES)
    assert any("\u200c" in case["text"] for case in CASES)


@pytest.mark.parametrize("case", CASES, ids=lambda case: str(case["id"]))
def test_python_snapping_and_redaction_match_shared_fixture(
    case: dict[str, Any],
) -> None:
    text = str(case["text"])
    start, end = snap_span_to_grapheme_boundaries(
        int(case["input_start"]),
        int(case["input_end"]),
        text,
    )

    assert (start, end) == (
        int(case["expected_start"]),
        int(case["expected_end"]),
    )
    assert is_grapheme_boundary(start, text)
    assert is_grapheme_boundary(end, text)

    redacted = text[:start] + str(case["replacement"]) + text[end:]
    assert redacted == case["expected_redacted"]
    assert redacted.encode("utf-8") == str(case["expected_redacted"]).encode("utf-8")


@pytest.mark.parametrize("case", CASES, ids=lambda case: str(case["id"]))
def test_python_decoders_enforce_shared_grapheme_boundaries(
    case: dict[str, Any],
) -> None:
    text = str(case["text"])
    expected_start = int(case["expected_start"])
    expected_end = int(case["expected_end"])

    viterbi_spans = token_spans_to_char_spans(
        [(1, 0, 1)],
        [(int(case["input_start"]), int(case["input_end"]))],
        text,
    )
    assert viterbi_spans == [(1, expected_start, expected_end)]

    coerced = coerce_token_classification_spans(
        [
            {
                "entity": "B-NAME",
                "score": 0.99,
                "start": case["input_start"],
                "end": case["input_end"],
            }
        ],
        text,
    )
    assert len(coerced) == 1
    assert (coerced[0].start, coerced[0].end) == (
        expected_start,
        expected_end,
    )
    assert coerced[0].text == text[expected_start:expected_end]
    assert coerced[0].byte_start == len(text[:expected_start].encode("utf-8"))
    assert coerced[0].byte_end == len(text[:expected_end].encode("utf-8"))
