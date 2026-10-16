"""Explicit negative annotations must not fall back to model predictions."""

from types import SimpleNamespace

import pytest

from openmed.eval.data_provenance import build_training_data_manifest

SPAN = {"start": 0, "end": 3, "label": "ENTITY"}


def _manifest(fixture):
    return build_training_data_manifest(
        [fixture], dataset_id="synthetic", data_revision="test"
    )


@pytest.mark.parametrize("empty", [[], ()])
@pytest.mark.parametrize("field", ["gold_spans", "spans"])
def test_empty_authoritative_annotation_is_preserved(empty, field):
    fixture = {"id": "sample", "text": "abc", field: empty, "entities": [SPAN]}
    manifest = _manifest(fixture)
    entry = manifest["fixtures"][0]
    assert entry["span_count"] == 0
    assert entry["spans"] == []


def test_empty_gold_overrides_nonempty_spans_and_entities():
    entry = _manifest(
        {
            "id": "sample",
            "text": "abc",
            "gold_spans": [],
            "spans": [SPAN],
            "entities": [SPAN],
        }
    )["fixtures"][0]
    assert entry["span_count"] == 0


def test_mapping_and_object_negative_fixtures_match():
    mapping = {
        "fixture_id": "sample",
        "text": "abc",
        "gold_spans": [],
        "entities": [SPAN],
    }
    obj = SimpleNamespace(fixture_id="sample", text="abc", gold_spans=[])
    assert _manifest(mapping) == _manifest(obj)


@pytest.mark.parametrize(
    "missing", [{}, {"gold_spans": None}, {"gold_spans": None, "spans": None}]
)
def test_missing_or_none_annotation_still_falls_back(missing):
    fixture = {"id": "sample", "text": "abc", "entities": [SPAN], **missing}
    assert _manifest(fixture)["fixtures"][0]["span_count"] == 1


def test_nonempty_gold_still_has_precedence():
    gold = {**SPAN, "label": "GOLD"}
    fixture = {
        "id": "sample",
        "text": "abc",
        "gold_spans": [gold],
        "spans": [SPAN],
        "entities": [SPAN],
    }
    assert _manifest(fixture)["fixtures"][0]["spans"][0]["label"] == "GOLD"
