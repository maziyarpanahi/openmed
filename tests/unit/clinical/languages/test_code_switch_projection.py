"""Synthetic source-offset and overlap conflict regression controls."""

import json

import pytest

from openmed.clinical.languages.code_switch_projection import (
    LanguageWindow,
    WindowEntity,
    code_switch_windows,
    reconcile_code_switch_entities,
)
from openmed.core.language_router import LanguageRun


class SyntheticRouter:
    def route_runs(self, text):
        return (
            LanguageRun(0, 5, "Latin", "en", 0.8, "synthetic"),
            LanguageRun(5, len(text), "Latin", "de", 0.7, "synthetic"),
        )


def candidate(window, start, end, label="PERSON", confidence=0.8, digest="a" * 64):
    return WindowEntity(window.window_id, start, end, label, confidence, digest)


def test_windows_preserve_unicode_source_offsets_and_overlap():
    text = "A😀 B Ünicode"
    windows = code_switch_windows(text, router=SyntheticRouter(), overlap=2)
    assert [(w.start, w.end, w.language) for w in windows] == [
        (0, 7, "en"),
        (3, len(text), "de"),
    ]
    assert text[windows[1].start : windows[1].end] == "B Ünicode"


def test_boundary_entities_project_and_exact_duplicates_share_provenance():
    a, b = LanguageWindow(0, 10, "en", 0.8), LanguageWindow(5, 15, "de", 0.7)
    result = reconcile_code_switch_entities(
        [a, b],
        [candidate(a, 6, 9), candidate(b, 1, 4, confidence=0.9, digest="b" * 64)],
        text_length=15,
    )
    assert result["entities"] == [
        {
            "start": 6,
            "end": 9,
            "label": "PERSON",
            "confidence": 0.9,
            "languages": ["de", "en"],
            "evidence_digests": ["a" * 64, "b" * 64],
            "status": "candidate",
        }
    ]
    assert result["needs_review"] is False


def test_conflicting_labels_and_boundaries_require_review_not_score_selection():
    w = LanguageWindow(0, 20, "en", 0.8)
    result = reconcile_code_switch_entities(
        [w],
        [
            candidate(w, 1, 5, "PERSON", 0.99),
            candidate(w, 1, 5, "ORG", 0.1),
            candidate(w, 4, 8, "PERSON"),
        ],
        text_length=20,
    )
    assert result["conflict_count"] == 3
    assert all(e["status"] == "needs_review" for e in result["entities"])


def test_adjacent_spans_do_not_conflict_and_order_is_stable():
    w = LanguageWindow(0, 20, "en", 0.8)
    candidates = [candidate(w, 1, 5), candidate(w, 5, 8)]
    a = reconcile_code_switch_entities([w], candidates, text_length=20)
    assert a == reconcile_code_switch_entities(
        [w], reversed(candidates), text_length=20
    )
    assert a["conflict_count"] == 0
    assert "synthetic source" not in json.dumps(a)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"start": -1},
        {"end": 0},
        {"confidence": float("nan")},
        {"confidence": True},
        {"label": "raw patient name"},
    ],
)
def test_invalid_local_entity_rejects_without_source_echo(kwargs):
    w = LanguageWindow(0, 20, "en", 0.8)
    args = dict(
        window_id=w.window_id,
        start=1,
        end=5,
        label="PERSON",
        confidence=0.8,
        evidence_digest="a" * 64,
    )
    args.update(kwargs)
    with pytest.raises(ValueError) as exc:
        WindowEntity(**args)
    assert "raw patient name" not in str(exc.value)


def test_unknown_window_and_out_of_window_offsets_are_rejected():
    w = LanguageWindow(0, 10, "en", 0.8)
    with pytest.raises(ValueError, match="unknown window"):
        reconcile_code_switch_entities([], [candidate(w, 1, 2)], text_length=20)
    with pytest.raises(ValueError, match="exceeds window"):
        reconcile_code_switch_entities([w], [candidate(w, 1, 11)], text_length=20)
    with pytest.raises(ValueError, match="candidate limit"):
        reconcile_code_switch_entities(
            [w], [candidate(w, 1, 2)] * 2, text_length=20, max_candidates=1
        )


def test_empty_text_and_default_router_are_offline():
    assert code_switch_windows("") == ()
    text = "Synthetic text. दूसरा पाठ."
    assert all(0 <= w.start < w.end <= len(text) for w in code_switch_windows(text))


def test_signed_zero_confidence_has_one_window_identity_and_stable_run_order():
    positive = LanguageWindow(0, 4, "en", 0.0)
    negative = LanguageWindow(0, 4, "en", -0.0)
    assert positive == negative
    assert positive.window_id == negative.window_id
    assert candidate(negative, 0, 1, confidence=-0.0).confidence == 0.0

    class Router:
        def __init__(self, reverse):
            self.reverse = reverse

        def route_runs(self, text):
            runs = (
                LanguageRun(0, 4, "Latin", "en", -0.0, "synthetic"),
                LanguageRun(0, 4, "Latin", "en", 0.0, "synthetic"),
            )
            return runs[::-1] if self.reverse else runs

    first = code_switch_windows("text", router=Router(False), overlap=0)
    second = code_switch_windows("text", router=Router(True), overlap=0)
    assert [w.window_id for w in first] == [w.window_id for w in second]
