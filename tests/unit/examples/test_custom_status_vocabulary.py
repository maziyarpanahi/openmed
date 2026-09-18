"""Tests for the local status-vocabulary extension example."""

from __future__ import annotations

import json

import pytest

from examples import custom_status_vocabulary as example
from openmed.clinical import load_status_vocab


def test_custom_status_vocabulary_runs_end_to_end(capsys) -> None:
    summary = example.main()

    printed = json.loads(capsys.readouterr().out)
    assert printed == summary

    assert summary["normalized"] == {
        "uses a cane": "assisted",
        "walks independently": "ambulatory",
        "formerly used a wheelchair": "former",
        "walks independently (historical)": "former",
        "uses a cane (negated)": "never",
        "chart silent on mobility": "unknown",
    }
    assert "advisory disclaimer" in summary["invalid_provenance_rejected"]
    assert "uses a cane" in summary["duplicate_cue_rejected"]
    assert "assisted" in summary["duplicate_cue_rejected"]
    assert "never" in summary["duplicate_cue_rejected"]


def test_write_example_vocabulary_loads_through_the_public_path_api(tmp_path) -> None:
    path = example.write_example_vocabulary(tmp_path)

    payload = load_status_vocab(path)

    assert set(payload["vocabularies"]) == {"mobility"}
    assert "clinical decision" in payload["provenance"]["disclaimer"]


def test_find_duplicate_cues_is_empty_for_the_shipped_example_vocabulary(
    tmp_path,
) -> None:
    path = example.write_example_vocabulary(tmp_path)
    mobility = load_status_vocab(path)["vocabularies"]["mobility"]

    assert example.find_duplicate_cues(mobility) == []


def test_broken_provenance_yaml_is_rejected_by_load_status_vocab(tmp_path) -> None:
    path = tmp_path / "invalid.yaml"
    path.write_text(example.broken_provenance_yaml(), encoding="utf-8")

    with pytest.raises(ValueError, match="advisory disclaimer"):
        load_status_vocab(path)


def test_broken_duplicate_cue_yaml_is_rejected_by_validate_no_duplicate_cues(
    tmp_path,
) -> None:
    path = tmp_path / "duplicate.yaml"
    path.write_text(example.broken_duplicate_cue_yaml(), encoding="utf-8")
    mobility = load_status_vocab(path)["vocabularies"]["mobility"]

    with pytest.raises(ValueError, match="uses a cane"):
        example.validate_no_duplicate_cues(mobility, domain="mobility")


@pytest.mark.parametrize(
    ("phrase", "negated", "temporality", "expected"),
    [
        ("uses a walker", False, None, "assisted"),
        ("ambulates without assistance", False, None, "ambulatory"),
        ("no longer uses a cane", False, None, "former"),
        ("denies mobility limitation", False, None, "never"),
        ("uses a wheelchair", True, None, "never"),
        ("uses a wheelchair", False, "historical", "former"),
        ("", False, None, "unknown"),
    ],
)
def test_normalize_mobility_status(
    phrase, negated, temporality, expected, tmp_path
) -> None:
    path = example.write_example_vocabulary(tmp_path)
    mobility = load_status_vocab(path)["vocabularies"]["mobility"]

    assert (
        example.normalize_mobility_status(
            phrase, mobility, negated=negated, temporality=temporality
        )
        == expected
    )
