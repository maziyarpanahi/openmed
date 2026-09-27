"""Fingerprint set normalization preserves old ordering and handles mixed values."""

import json

import pytest

from openmed.processing.checkpoint import (
    _safe_parameter_identity,
    build_stream_fingerprint,
)


@pytest.mark.parametrize("values", [{1, "one"}, {None, 2, "two"}, {False, "zero"}])
def test_mixed_scalar_sets_have_canonical_fallback_order(values):
    actual = _safe_parameter_identity(values)
    expected = sorted(
        values, key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":"))
    )
    assert actual == expected


def test_public_fingerprint_accepts_mixed_sets_deterministically():
    options = {"labels": {1, "synthetic", None}}
    first = build_stream_fingerprint(policy_name="synthetic", deidentify_kwargs=options)
    second = build_stream_fingerprint(
        policy_name="synthetic",
        deidentify_kwargs={"labels": set(reversed(list(options["labels"])))},
    )
    assert first == second
    changed = build_stream_fingerprint(
        policy_name="synthetic", deidentify_kwargs={"labels": {2, "synthetic", None}}
    )
    assert first.model != changed.model
    assert first.policy == changed.policy


def test_normalized_object_sets_do_not_compare_dicts():
    values = {object(), object()}
    assert _safe_parameter_identity(values) == [{"type": "builtins.object"}] * 2


@pytest.mark.parametrize(
    "values,expected", [({2, 10}, [2, 10]), ({"z", "a"}, ["a", "z"]), (set(), [])]
)
def test_comparable_sets_keep_legacy_order(values, expected):
    assert _safe_parameter_identity(values) == expected


def test_nested_and_sensitive_parameter_paths_are_preserved():
    assert _safe_parameter_identity({"nested": [{"x", 1}]}) == {"nested": [["x", 1]]}
    result = _safe_parameter_identity({"secret": "synthetic-only"})
    assert result["secret"]["type"] == "str"
    assert len(result["secret"]["sha256"]) == 64
    assert "synthetic-only" not in json.dumps(result)
