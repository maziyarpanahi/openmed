"""BIO prefixes must not corrupt the interior of a custom NER label."""

from copy import deepcopy

import pytest

from openmed.processing.outputs import OutputFormatter


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("HLA-B-27", "HLA-B-27"),
        ("MULTI-MORBIDITY", "MULTI-MORBIDITY"),
        ("B-HLA-B-27", "HLA-B-27"),
        ("I-MULTI-MORBIDITY", "MULTI-MORBIDITY"),
        ("B-I-CUSTOM", "I-CUSTOM"),
        ("B-DISEASE", "DISEASE"),
        ("I-DISEASE", "DISEASE"),
        ("DISEASE", "DISEASE"),
        ("", "UNKNOWN"),
        ("B-", "B-"),
        ("I-", "I-"),
        ("b-DISEASE", "b-DISEASE"),
    ],
)
def test_normalization_removes_only_one_leading_bio_marker(raw, expected):
    result = OutputFormatter().format_predictions(
        [{"entity": raw, "word": "synthetic", "score": 0.9}], ""
    )
    assert result.entities[0].label == expected


def test_entity_group_precedence_and_inputs_are_preserved():
    prediction = {
        "entity_group": "HLA-B-27",
        "entity": "B-OTHER",
        "word": "synthetic",
        "score": 0.9,
        "metadata": {"source": "test"},
    }
    before = deepcopy(prediction)
    result = OutputFormatter().format_predictions([prediction], "")
    assert result.entities[0].label == "HLA-B-27"
    assert prediction == before
    assert result.entities[0].confidence == 0.9
    assert result.entities[0].metadata == {"source": "test"}


def test_threshold_filter_is_unchanged():
    result = OutputFormatter(confidence_threshold=0.5).format_predictions(
        [{"entity": "HLA-B-27", "word": "synthetic", "score": 0.2}], ""
    )
    assert result.entities == []
