"""Offline coverage diff regression and malformed-input controls."""

import json
from copy import deepcopy

import pytest

from openmed.core.language_coverage_diff import (
    diff_language_coverage,
    render_language_coverage_diff,
)


def manifest(*packs):
    return {"schema_version": "1.0.0", "packs": list(packs)}


def pack(locale="en-US", **overrides):
    result = dict(
        locale=locale,
        scripts=["Latin"],
        identifier_classes=["DATE", "PERSON"],
        surrogate_support="supported",
        evidence_digest="a" * 64,
    )
    result.update(overrides)
    return result


def test_aliases_and_reordering_are_equal():
    before = manifest(pack(), pack("de-DE"))
    after = manifest(
        pack("DE_de", identifier_classes=["PERSON", "DATE"]), pack("EN_us")
    )
    assert diff_language_coverage(before, after) == {
        "schema_version": "1.0.0",
        "added": [],
        "removed": [],
        "improved": [],
        "regressed": [],
        "changed": [],
    }


def test_add_remove_golden():
    result = diff_language_coverage(manifest(pack()), manifest(pack("fr-FR")))
    assert render_language_coverage_diff(result) == (
        '{"added":["fr-FR"],"changed":[],"improved":[],"regressed":[],"removed":["en-US"],'
        '"schema_version":"1.0.0"}\n'
    )


def test_improvement_does_not_hide_regression():
    result = diff_language_coverage(
        manifest(pack()),
        manifest(
            pack(
                scripts=["Latin", "Cyrillic"],
                identifier_classes=["DATE"],
                surrogate_support="approximated",
                evidence_digest=None,
            )
        ),
    )
    assert result["improved"] == [{"locale": "en-US", "fields": ["scripts"]}]
    assert result["regressed"] == [
        {
            "locale": "en-US",
            "fields": ["identifier_classes", "surrogate_support", "evidence"],
        }
    ]


def test_changed_evidence_is_not_a_measured_improvement():
    result = diff_language_coverage(
        manifest(pack()), manifest(pack(evidence_digest="b" * 64))
    )
    assert result["changed"] == [{"locale": "en-US", "evidence_changed": True}]
    assert result["improved"] == result["regressed"] == []


def test_arbitrary_payloads_are_not_rendered():
    secret = "synthetic private note /private/example"
    result = diff_language_coverage(
        manifest(pack(note=secret)), manifest(pack(note=secret))
    )
    assert secret not in render_language_coverage_diff(result)
    with pytest.raises(ValueError, match="invalid coverage diff"):
        render_language_coverage_diff({**result, "private_note": secret})


@pytest.mark.parametrize(
    "bad",
    [
        manifest(pack(), pack("EN_us")),
        manifest(pack(surrogate_support="certified")),
        manifest(pack(surrogate_support=[])),
        manifest(pack(evidence_digest="private/raw")),
        manifest(pack(scripts=["Latin", "Latin"])),
        manifest(pack(identifier_classes=["patient name"])),
        {"schema_version": "2.0.0", "packs": []},
        {"schema_version": "1.0.0", "packs": [None]},
    ],
)
def test_malformed_manifests_fail_with_controlled_errors(bad):
    with pytest.raises(ValueError) as exc:
        diff_language_coverage(manifest(), bad)
    assert "patient name" not in str(exc.value)
    assert "private/raw" not in str(exc.value)


def test_render_roundtrip_is_stable():
    result = diff_language_coverage(
        manifest(pack()), manifest(pack(scripts=["Cyrillic"]))
    )
    original = deepcopy(result)
    assert json.loads(render_language_coverage_diff(result)) == result
    assert result == original


@pytest.mark.parametrize(
    "category", ["added", "removed", "changed", "improved", "regressed"]
)
def test_renderer_rejects_duplicate_aliases(category):
    result = diff_language_coverage(manifest(), manifest())
    if category in ("added", "removed"):
        result[category] = ["en-US", "EN_us"]
    elif category == "changed":
        result[category] = [
            {"locale": "en-US", "evidence_changed": True},
            {"locale": "EN_us", "evidence_changed": False},
        ]
    else:
        result[category] = [
            {"locale": "en-US", "fields": ["scripts"]},
            {"locale": "EN_us", "fields": ["evidence"]},
        ]
    with pytest.raises(ValueError, match="duplicate canonical"):
        render_language_coverage_diff(result)


def test_renderer_non_mapping_has_controlled_error():
    with pytest.raises(ValueError, match="invalid coverage diff"):
        render_language_coverage_diff(None)
