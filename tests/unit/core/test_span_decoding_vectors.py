"""Run the shared span-decoding golden vectors through the Python reference."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from openmed.core.decoding import viterbi
from openmed.core.decoding.viterbi import (
    build_label_info,
    labels_to_char_spans,
    labels_to_token_spans,
    viterbi_decode,
)

FIXTURE_PATH = (
    Path(__file__).parents[2] / "fixtures" / "parity" / "span_decoding_v1.json"
)
FIXTURE = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
CASES: list[dict[str, Any]] = FIXTURE["cases"]
LABELS: list[str] = FIXTURE["labels"]


def _spans(triples: list[tuple[int, int, int]], label_info) -> list[dict[str, Any]]:
    names = label_info.span_class_names
    return [
        {"label": names[label], "start": start, "end": end}
        for label, start, end in triples
    ]


def test_fixture_is_synthetic_and_covers_required_categories() -> None:
    assert FIXTURE["version"] == 1
    assert FIXTURE["synthetic"] is True
    assert "synthetic" in FIXTURE["description"].lower()
    assert FIXTURE["offset_unit"] == "unicode_scalar"
    assert 15 <= len(CASES) <= 30
    ids = [case["id"] for case in CASES]
    assert len(ids) == len(set(ids))
    assert {
        "bio_repair",
        "invalid_start",
        "tie",
        "cjk",
        "emoji",
        "empty",
    } <= {case["category"] for case in CASES}


@pytest.mark.parametrize("numpy_path", [True, False], ids=["numpy", "pure-python"])
@pytest.mark.parametrize("case", CASES, ids=[case["id"] for case in CASES])
def test_span_decoding_vector_matches_python_reference(
    case: dict[str, Any],
    numpy_path: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not numpy_path:
        monkeypatch.setattr(viterbi, "_np", None)
    elif viterbi._np is None:
        pytest.skip("numpy not installed")

    case_id = case["id"]
    label_info = build_label_info(dict(enumerate(LABELS)))

    if "token_logprobs" in case:
        path = viterbi_decode(
            case["token_logprobs"],
            label_info=label_info,
            biases=case["biases"],
        )
        assert [LABELS[index] for index in path] == case["expected_label_path"], (
            f"{case_id}: label path"
        )
    else:
        path = [LABELS.index(label) for label in case["token_labels"]]

    labels_by_index = dict(enumerate(path))
    hint = case.get("language_hint")
    token_spans = labels_to_token_spans(
        labels_by_index,
        label_info,
        token_offsets=case["token_offsets"],
        text=case["text"],
        language_hint=hint,
    )
    char_spans = labels_to_char_spans(
        labels_by_index,
        label_info,
        case["token_offsets"],
        case["text"],
        language_hint=hint,
    )

    assert _spans(token_spans, label_info) == case["expected_token_spans"], (
        f"{case_id}: token spans"
    )
    assert _spans(char_spans, label_info) == case["expected_char_spans"], (
        f"{case_id}: char spans"
    )
