"""Synthetic unit tests for SPDX identifier normalization."""

from __future__ import annotations

import json

import pytest

from openmed.training.synthetic.spdx_identifier import (
    PERMISSIVE_SPDX_IDENTIFIERS,
    SPDX_IDENTIFIER_SCHEMA_VERSION,
    SpdxIdentifierStatus,
    normalize_spdx_identifier,
)


def outcome(value):
    result = normalize_spdx_identifier(value)
    return (
        result.status,
        result.normalized,
        result.is_permissive,
        result.category,
    )


@pytest.mark.parametrize("identifier", sorted(PERMISSIVE_SPDX_IDENTIFIERS))
def test_every_allowlisted_identifier_is_canonical_and_permissive(identifier) -> None:
    assert outcome(identifier) == (
        SpdxIdentifierStatus.CANONICAL,
        identifier,
        True,
        "spdx_identifier_canonical",
    )


@pytest.mark.parametrize(
    "value,expected",
    [
        ("mit", "MIT"),
        ("apache-2.0", "Apache-2.0"),
        ("bsd-2-clause", "BSD-2-Clause"),
        ("bsd-2-clause-patent", "BSD-2-Clause-Patent"),
        ("isc", "ISC"),
        ("bsl-1.0", "BSL-1.0"),
        ("0bsd", "0BSD"),
        ("zlib", "Zlib"),
        ("unlicense", "Unlicense"),
        ("mpl-2.0", "MPL-2.0"),
        ("python-2.0", "Python-2.0"),
        ("blueoak-1.0.0", "BlueOak-1.0.0"),
    ],
)
def test_case_variants_normalize_to_allowlist_members(value, expected) -> None:
    assert outcome(value) == (
        SpdxIdentifierStatus.NORMALIZED,
        expected,
        True,
        "spdx_identifier_normalized_case",
    )


def test_surrounding_whitespace_normalizes() -> None:
    assert outcome(" MIT ") == (
        SpdxIdentifierStatus.NORMALIZED,
        "MIT",
        True,
        "spdx_identifier_normalized_whitespace",
    )


def test_case_and_whitespace_normalize_together() -> None:
    assert outcome("  mit  ") == (
        SpdxIdentifierStatus.NORMALIZED,
        "MIT",
        True,
        "spdx_identifier_normalized_case",
    )


def test_deprecated_alias_maps_to_documented_successor() -> None:
    assert outcome("BSD-2-Clause-NetBSD") == (
        SpdxIdentifierStatus.DEPRECATED_ALIAS,
        "BSD-2-Clause",
        True,
        "spdx_identifier_deprecated_alias",
    )
    assert outcome("bsd-2-clause-netbsd") == (
        SpdxIdentifierStatus.DEPRECATED_ALIAS,
        "BSD-2-Clause",
        True,
        "spdx_identifier_deprecated_alias",
    )


def test_alias_successors_are_allowlisted() -> None:
    result = normalize_spdx_identifier("BSD-2-Clause-NetBSD")
    assert result.normalized in PERMISSIVE_SPDX_IDENTIFIERS


@pytest.mark.parametrize(
    "value",
    [
        "GPL-3.0-only",
        "GPL-2.0",
        "Proprietary",
        "CC-BY-NC-4.0",
        "AGPL-1.0-only",
        " gpl-3.0 ",
        "licenseref-synthetic",
    ],
)
def test_unknown_values_stay_unknown_and_never_permissive(value) -> None:
    assert outcome(value) == (
        SpdxIdentifierStatus.UNKNOWN,
        None,
        False,
        "spdx_identifier_unknown",
    )


@pytest.mark.parametrize(
    "value",
    [
        "MIT OR Apache-2.0",
        "Apache-2.0 AND ISC",
        "GPL-3.0-only WITH Classpath-exception-2.0",
        "MIT+",
    ],
)
def test_expression_like_values_are_malformed(value) -> None:
    assert outcome(value) == (
        SpdxIdentifierStatus.MALFORMED,
        None,
        False,
        "spdx_identifier_malformed_expression",
    )


@pytest.mark.parametrize("value", ["", "   ", "\t\n"])
def test_empty_values_are_malformed(value) -> None:
    assert outcome(value) == (
        SpdxIdentifierStatus.MALFORMED,
        None,
        False,
        "spdx_identifier_malformed_empty",
    )


@pytest.mark.parametrize("value", [None, 42, True, [], 2.5])
def test_non_string_values_are_malformed(value) -> None:
    assert outcome(value) == (
        SpdxIdentifierStatus.MALFORMED,
        None,
        False,
        "spdx_identifier_malformed_type",
    )


@pytest.mark.parametrize("value", ["MIT!", "BSD 2-Clause", "Apache 2.0", "MIT/MIT"])
def test_invalid_characters_are_malformed(value) -> None:
    assert outcome(value) == (
        SpdxIdentifierStatus.MALFORMED,
        None,
        False,
        "spdx_identifier_malformed_characters",
    )


def test_licenseref_values_are_valid_but_never_permissive() -> None:
    assert outcome("LicenseRef-Synthetic-Clinical") == (
        SpdxIdentifierStatus.CANONICAL,
        "LicenseRef-Synthetic-Clinical",
        False,
        "spdx_identifier_canonical_licenseref",
    )
    assert outcome("LicenseRef-MIT") == (
        SpdxIdentifierStatus.CANONICAL,
        "LicenseRef-MIT",
        False,
        "spdx_identifier_canonical_licenseref",
    )
    assert outcome(" LicenseRef-Synthetic ") == (
        SpdxIdentifierStatus.NORMALIZED,
        "LicenseRef-Synthetic",
        False,
        "spdx_identifier_normalized_licenseref_whitespace",
    )


@pytest.mark.parametrize(
    "value", ["LicenseRef-", "LicenseRef-Synthetic Clinical", "LicenseRef-MIT*"]
)
def test_malformed_licenseref_values_fail_closed(value) -> None:
    assert outcome(value) == (
        SpdxIdentifierStatus.MALFORMED,
        None,
        False,
        "spdx_identifier_malformed_characters",
    )


def test_licenseref_prefix_is_case_sensitive() -> None:
    result = normalize_spdx_identifier("licenseref-synthetic")
    assert result.status is SpdxIdentifierStatus.UNKNOWN
    assert result.is_permissive is False


def test_serialization_is_deterministic() -> None:
    result = normalize_spdx_identifier("mit")
    assert result.schema_version == SPDX_IDENTIFIER_SCHEMA_VERSION
    data = result.to_dict()
    assert list(data) == [
        "schema_version",
        "status",
        "normalized",
        "is_permissive",
        "category",
    ]
    expected = (
        '{"schema_version":1,"status":"normalized","normalized":"MIT",'
        '"is_permissive":true,"category":"spdx_identifier_normalized_case"}'
    )
    assert result.to_json() == expected
    assert json.loads(result.to_json()) == data
    repeat = normalize_spdx_identifier("mit")
    assert repeat.to_json() == result.to_json()


def test_malformed_results_serialize_without_the_input_value() -> None:
    result = normalize_spdx_identifier("MIT OR Apache-2.0")
    assert "MIT OR Apache-2.0" not in result.to_json()
    assert result.to_dict()["normalized"] is None
