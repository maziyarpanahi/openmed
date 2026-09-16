"""Focused synthetic tests for deterministic summary-output normalization."""

from __future__ import annotations

import re
import socket

import pytest

from openmed.clinical import (
    SUMMARY_OUTPUT_NORMALIZATION_OPERATION_CODES,
    SUMMARY_OUTPUT_NORMALIZATION_SCHEMA_VERSION,
    SummaryOutputNormalization,
    SummaryOutputNormalizationError,
    SummaryOutputNormalizer,
    normalize_summary_output,
    normalize_summary_text,
)


def test_normalizes_formatting_without_rewriting_claim_words() -> None:
    source = (
        "\r\n##  Assessment   ##\r\n"
        "* Patient   improved[ 2 ] .  \r\n"
        "+ Follow-up is planned 【3】 [citation:4]\r\n"
        "3) Additional finding [1]\r\n"
        "\r\n\r\n"
    )

    result = normalize_summary_output(source)

    assert result.normalized_text == (
        "## Assessment\n"
        "- Patient improved. [2]\n"
        "- Follow-up is planned [3] [citation:4]\n"
        "1. Additional finding [1]"
    )
    assert result.operation_codes == (
        "whitespace",
        "heading",
        "list_marker",
        "citation_placement",
    )
    assert re.findall(r"[A-Za-z]+", source) == re.findall(
        r"[A-Za-z]+", result.normalized_text
    )


def test_setext_headings_and_ordered_lists_have_stable_markers() -> None:
    source = "Assessment\r\n----\r\n3) first finding\r\n9. second finding\r\n\r\nPlan"

    assert normalize_summary_text(source) == (
        "## Assessment\n1. first finding\n2. second finding\n\nPlan"
    )


def test_normalization_is_idempotent_and_facade_is_equivalent() -> None:
    source = "  * stable claim [ 1 ]  \n\n\n"

    first = normalize_summary_output(source)
    second = normalize_summary_output(first.text)
    facade = SummaryOutputNormalizer().normalize(source)

    assert first == facade
    assert second.normalized_text == first.normalized_text
    assert second.operation_codes == ()
    assert first.changed is True


def test_markdown_links_are_not_treated_as_citation_tokens() -> None:
    source = "See [1](synthetic-reference) for the supporting note."

    result = normalize_summary_output(source)

    assert result.normalized_text == source
    assert result.operation_codes == ()


def test_audit_artifact_and_repr_are_value_free() -> None:
    synthetic_value = "SYNTHETIC_PATIENT_IDENTIFIER_001"
    result = normalize_summary_output(f"Finding: {synthetic_value} [ 1 ]")

    assert result.to_dict() == {
        "schema_version": SUMMARY_OUTPUT_NORMALIZATION_SCHEMA_VERSION,
        "operation_codes": ["citation_placement"],
    }
    assert result.to_json() == (
        '{"operation_codes":["citation_placement"],"schema_version":1}\n'
    )
    assert synthetic_value not in result.to_json()
    assert synthetic_value not in repr(result)


def test_normalizer_is_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_network(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("network access is forbidden")

    monkeypatch.setattr(socket, "create_connection", fail_network)

    assert normalize_summary_text("- local claim") == "- local claim"


def test_invalid_input_and_artifact_schema_fail_without_echoing_values() -> None:
    sentinel = "SYNTHETIC_SECRET_SHOULD_NOT_BE_ECHOED"

    with pytest.raises(
        SummaryOutputNormalizationError,
        match="summary output must be text",
    ) as error:
        normalize_summary_output(42)  # type: ignore[arg-type]
    assert sentinel not in str(error.value)

    with pytest.raises(
        SummaryOutputNormalizationError,
        match="unsupported summary normalization schema",
    ):
        SummaryOutputNormalization("local claim", schema_version=2)


def test_operation_code_contract_is_fixed() -> None:
    assert SUMMARY_OUTPUT_NORMALIZATION_OPERATION_CODES == (
        "whitespace",
        "heading",
        "list_marker",
        "citation_placement",
    )
