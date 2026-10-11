"""Synthetic offline tests for SDOH category completeness audits."""

from __future__ import annotations

import json
import socket
from importlib import import_module

import pytest

from openmed.clinical.sdoh import (
    DeterminantExtractorRegistry,
    SDOHFinding,
    available_determinant_extractors,
    extract_sdoh,
    extract_sdoh_with_language,
)
from openmed.clinical.sdoh_completeness import (
    MISSING_PROCESSING_RESULT,
    SDOHCategoryResult,
    SDOHCategoryState,
    audit_sdoh_completeness,
)
from openmed.clinical.sections import detect_sections


def test_audit_reports_every_configured_category_and_all_states() -> None:
    audit = audit_sdoh_completeness(
        ("housing", "food", "transportation", "utilities"),
        (
            SDOHCategoryResult(
                category="housing",
                state=SDOHCategoryState.PROCESSED,
                finding_count=1,
            ),
            SDOHCategoryResult(
                category="food",
                state=SDOHCategoryState.SKIPPED,
                reason_code="policy_disabled",
            ),
            SDOHCategoryResult(
                category="transportation",
                state=SDOHCategoryState.UNSUPPORTED,
                reason_code="extractor_unavailable",
            ),
            SDOHCategoryResult(
                category="utilities",
                state=SDOHCategoryState.FAILED,
                reason_code="extractor_error",
            ),
        ),
    )
    payload = audit.to_dict()

    assert [item["category"] for item in payload["categories"]] == [
        "food",
        "housing",
        "transportation",
        "utilities",
    ]
    assert payload["state_counts"] == {
        "processed": 1,
        "skipped": 1,
        "unsupported": 1,
        "failed": 1,
    }


def test_processed_zero_findings_is_unmentioned_not_negative() -> None:
    audit = audit_sdoh_completeness(
        ("housing",),
        (
            SDOHCategoryResult(
                category="housing",
                state=SDOHCategoryState.PROCESSED,
            ),
        ),
    )

    category = audit.to_dict()["categories"][0]
    assert category["finding_count"] == 0
    assert category["absence_interpretation"] == "unmentioned_not_negative"
    assert "negative" not in category["state"]


def test_missing_processing_result_is_explicit_failure() -> None:
    audit = audit_sdoh_completeness(("food", "housing"), ())

    assert all(item.state is SDOHCategoryState.FAILED for item in audit.categories)
    assert audit.to_dict()["reason_counts"] == {MISSING_PROCESSING_RESULT: 2}


def test_counts_and_reason_codes_are_value_free() -> None:
    audit = audit_sdoh_completeness(
        ("housing",),
        (
            SDOHCategoryResult(
                category="housing",
                state=SDOHCategoryState.SKIPPED,
                reason_code="consent_unavailable",
            ),
        ),
    )
    rendered = json.dumps(audit.to_dict(), sort_keys=True)

    assert "raw" not in rendered.lower()
    assert "consent_unavailable" in rendered


def test_unconfigured_category_is_rejected_without_echoing_code() -> None:
    result = SDOHCategoryResult(
        category="private_marker",
        state=SDOHCategoryState.PROCESSED,
    )

    with pytest.raises(ValueError, match="unconfigured category") as error:
        audit_sdoh_completeness(("housing",), (result,))

    assert "private_marker" not in str(error.value)


def test_audit_is_input_order_independent() -> None:
    categories = ("housing", "food")
    results = (
        SDOHCategoryResult("housing", SDOHCategoryState.PROCESSED, 1),
        SDOHCategoryResult("food", SDOHCategoryState.PROCESSED, 0),
    )

    assert audit_sdoh_completeness(categories, results).to_dict() == (
        audit_sdoh_completeness(reversed(categories), reversed(results)).to_dict()
    )


def test_completeness_audit_performs_no_network_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reject_network(*args: object, **kwargs: object) -> None:
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket.socket, "connect", reject_network)

    assert audit_sdoh_completeness(("housing",), ()).categories


@pytest.mark.parametrize(
    "language,text",
    [
        ("es", "Social History: vive solo y no tiene empleo."),
        ("de", "Social History: lebt allein und ist arbeitslos."),
        ("es-MX", "Social History: alcohol, tobacco, drug."),
        (None, "Social History: current smoker."),
        ("", "Social History: current smoker."),
    ],
)
def test_explicit_unsupported_languages_are_never_processed(language, text):
    def unused():
        pytest.fail("unsupported input consumed caller spans or sections")
        yield

    report = extract_sdoh_with_language(text, unused(), unused(), language=language)
    assert report.findings == ()
    assert {item.category for item in report.category_results} == set(
        available_determinant_extractors()
    )
    assert all(
        item.state is SDOHCategoryState.UNSUPPORTED and item.finding_count == 0
        for item in report.category_results
    )
    expected = "language_undeclared" if not language else "unsupported_language"
    assert {item.reason_code for item in report.category_results} == {expected}
    assert text not in json.dumps(report.to_dict())


@pytest.mark.parametrize("language", ["en", "en-US", "EN_us"])
@pytest.mark.parametrize(
    "text",
    [
        "Social History: retired teacher. Lives alone. Current smoker, 7 drinks/week. Reports food insecurity.",
        "Social History: no social determinants documented.",
        "Social History: ask about smoking?",
    ],
)
def test_english_findings_match_legacy_and_feed_completeness(language, text):
    sections = detect_sections(text)
    expected = extract_sdoh(text, (), sections)
    report = extract_sdoh_with_language(text, (), sections, language=language)
    assert list(report.findings) == expected
    assert all(
        item.state is SDOHCategoryState.PROCESSED for item in report.category_results
    )
    audit = audit_sdoh_completeness(
        available_determinant_extractors(), report.category_results
    )
    assert sum(item.finding_count for item in audit.categories) == len(expected)
    for item in report.to_dict()["findings"]:
        assert set(item) == {"category", "span"}
    for item in report.category_results:
        if not item.finding_count:
            assert item.absence_interpretation == "unmentioned_not_negative"


def _isolated_registry(monkeypatch, entries):
    module = import_module("openmed.clinical.sdoh")
    registry = DeterminantExtractorRegistry()
    for category, extractor, languages in entries:
        registry.register(category, extractor, languages=languages)
    monkeypatch.setattr(module, "_DETERMINANT_EXTRACTORS", registry)
    return module, registry


def test_undeclared_custom_extractor_still_runs_only_in_legacy_dispatch(monkeypatch):
    calls = []

    def custom(text, spans):
        calls.append(True)
        return []

    _isolated_registry(monkeypatch, [("custom", custom, None)])
    report = extract_sdoh_with_language("synthetic", language="en")
    assert report.category_results[0].reason_code == "language_undeclared"
    assert calls == []
    assert extract_sdoh("synthetic", ()) == []
    assert calls == [True]


def test_declared_custom_language_and_value_free_audit(monkeypatch):
    secret = "PRIVATE_CLINICAL_SENTINEL"

    def custom(text, spans):
        return [SDOHFinding("custom", secret, secret, secret, secret, (0, 3), 0.9)]

    _isolated_registry(monkeypatch, [("custom", custom, ("es",))])
    report = extract_sdoh_with_language("synthetic", language="es")
    assert report.findings[0].value == secret
    assert report.category_results == (
        SDOHCategoryResult("custom", SDOHCategoryState.PROCESSED, 1),
    )
    assert secret not in repr(report)
    assert secret not in json.dumps(report.to_dict())
    assert report.to_dict()["findings"] == [{"category": "custom", "span": [0, 3]}]
    unsupported = extract_sdoh_with_language("synthetic", language="de")
    assert unsupported.findings == ()
    assert unsupported.category_results[0].state is SDOHCategoryState.UNSUPPORTED


@pytest.mark.parametrize(
    "language_declaration", [[], ["de"], ["en", "EN"], [None], "en"]
)
def test_cue_table_language_declaration_controls_builtin_support(
    monkeypatch, language_declaration
):
    module = import_module("openmed.clinical.sdoh")
    payload = module.load_sdoh_social_cues()
    payload["languages"] = language_declaration
    monkeypatch.setattr(module, "_load_default_sdoh_social_cues", lambda: payload)
    report = extract_sdoh_with_language("retired teacher", language="en")
    social = [
        item
        for item in report.category_results
        if item.category in {"employment", "food_insecurity", "living_status"}
    ]
    assert all(item.state is not SDOHCategoryState.PROCESSED for item in social)
    assert not any(
        item.category in {"employment", "food_insecurity", "living_status"}
        for item in report.findings
    )


def test_substance_cue_table_also_declares_english():
    module = import_module("openmed.clinical.sdoh")
    assert module._builtin_cue_languages(module._extract_tobacco) == ("en",)
    assert module.load_sdoh_social_cues()["languages"] == ["en"]


def test_failed_extractor_discards_partial_findings_and_continues(monkeypatch):
    def failed(text, spans):
        yield SDOHFinding("failed", "private", None, None, None, (0, 3), 0.9)
        raise RuntimeError("PRIVATE_EXCEPTION_SENTINEL")

    def valid(text, spans):
        return [SDOHFinding("valid", "private", None, None, None, (0, 3), 0.9)]

    _isolated_registry(
        monkeypatch, [("failed", failed, ("en",)), ("valid", valid, ("en",))]
    )
    report = extract_sdoh_with_language("synthetic", language="en")
    assert [item.category for item in report.findings] == ["valid"]
    assert report.category_results[0] == SDOHCategoryResult(
        "failed", SDOHCategoryState.FAILED, reason_code="extractor_failed"
    )
    assert "PRIVATE_EXCEPTION_SENTINEL" not in json.dumps(report.to_dict())


@pytest.mark.parametrize("kind", ["wrong_category", "outside_source", "untyped"])
def test_invalid_custom_findings_fail_value_free(monkeypatch, kind):
    def custom(text, spans):
        if kind == "untyped":
            return [{"text": "PRIVATE_SENTINEL"}]
        return [
            SDOHFinding(
                "other" if kind == "wrong_category" else "custom",
                "private",
                None,
                None,
                None,
                (0, 99) if kind == "outside_source" else (0, 3),
                0.9,
            )
        ]

    _isolated_registry(monkeypatch, [("custom", custom, ("en",))])
    report = extract_sdoh_with_language("synthetic", language="en")
    assert report.findings == ()
    assert report.category_results[0].state is SDOHCategoryState.FAILED


def test_unselected_sections_are_explicitly_skipped(monkeypatch):
    def unused(*args):
        pytest.fail("empty section selection invoked extractor")

    _isolated_registry(monkeypatch, [("custom", unused, ("en",))])
    report = extract_sdoh_with_language("synthetic", sections=(), language="en")
    assert report.category_results == (
        SDOHCategoryResult(
            "custom", SDOHCategoryState.SKIPPED, reason_code="section_not_selected"
        ),
    )


@pytest.mark.parametrize("language", [True, 1, "PRIVATE LANGUAGE", "en\n", "x" * 64])
def test_malformed_language_values_are_not_echoed(language):
    with pytest.raises(ValueError, match="^invalid_sdoh_language$"):
        extract_sdoh_with_language("synthetic", language=language)


def test_registry_replacement_resets_language_declaration():
    registry = DeterminantExtractorRegistry()
    callback = lambda *_: []
    registry.register("custom", callback, languages=("es",))
    assert registry.language_items()[0][2] == ("es",)
    registry.register("custom", callback, replace=True)
    assert registry.language_items()[0][2] == ()
    registry.unregister("custom")
    assert registry.language_items() == ()


@pytest.mark.parametrize("input_kind", ["spans", "sections"])
def test_invalid_scope_is_a_value_free_failure(monkeypatch, input_kind):
    def unused(*args):
        pytest.fail("invalid scope invoked extractor")

    def broken():
        raise RuntimeError("PRIVATE_SCOPE_SENTINEL")
        yield

    _isolated_registry(monkeypatch, [("custom", unused, ("en",))])
    kwargs = {input_kind: broken(), "language": "en"}
    report = extract_sdoh_with_language("synthetic", **kwargs)
    assert report.findings == ()
    assert report.category_results == (
        SDOHCategoryResult(
            "custom", SDOHCategoryState.FAILED, reason_code="scope_invalid"
        ),
    )
    assert "PRIVATE_SCOPE_SENTINEL" not in json.dumps(report.to_dict())


def test_report_constructor_rejects_inconsistent_counts():
    from openmed.clinical.sdoh import SDOHExtractionResult

    finding = SDOHFinding("custom", "private", None, None, None, (0, 3), 0.9)
    with pytest.raises(ValueError, match="^invalid_sdoh_finding_counts$"):
        SDOHExtractionResult(
            (finding,), (SDOHCategoryResult("custom", SDOHCategoryState.PROCESSED),)
        )
