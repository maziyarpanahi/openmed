"""Focused tests for the post-de-identification summarization stage."""

from __future__ import annotations

from datetime import datetime
from importlib import import_module

import pytest

from openmed.clinical.summarize import (
    SummarizationLeakageError,
    SummarizationOrderError,
    summarize,
    summarize_deidentified,
)
from openmed.core.pii import DeidentificationResult, PIIEntity

summarize_module = import_module("openmed.clinical.summarize")


SYNTHETIC_NOTE = (
    "Patient Casey Example presented with a cough. "
    "The synthetic admission was uncomplicated."
)


def _deidentified_result() -> DeidentificationResult:
    return DeidentificationResult(
        original_text=SYNTHETIC_NOTE,
        deidentified_text=(
            "Patient [NAME] presented with a cough. "
            "The synthetic admission was uncomplicated."
        ),
        pii_entities=[
            PIIEntity(
                text="Casey Example",
                label="NAME",
                start=8,
                end=21,
                confidence=0.99,
                redacted_text="[NAME]",
            )
        ],
        method="mask",
        timestamp=datetime(2026, 1, 1),
    )


def test_summarize_deidentifies_before_backend_and_returns_passing_check(monkeypatch):
    calls: list[tuple[str, str]] = []
    result = _deidentified_result()

    def fake_deidentify(text: str, *, method: str) -> DeidentificationResult:
        calls.append(("deidentify", text))
        assert method == "mask"
        return result

    def backend(text: str, *, mode: str) -> str:
        calls.append(("backend", text))
        assert mode == "bhc"
        assert "Casey Example" not in text
        return text.split(".", 1)[0] + "."

    monkeypatch.setattr(summarize_module, "deidentify", fake_deidentify)

    output = summarize(SYNTHETIC_NOTE, model=backend)

    assert calls == [
        ("deidentify", SYNTHETIC_NOTE),
        ("backend", result.deidentified_text),
    ]
    assert output.leakage_check.passed is True
    assert output.leakage_check.leaked_token_count == 0
    assert "Casey Example" not in output.summary


def test_default_stub_summary_contains_no_original_phi(monkeypatch):
    monkeypatch.setattr(
        summarize_module,
        "deidentify",
        lambda text, *, method: _deidentified_result(),
    )

    output = summarize(SYNTHETIC_NOTE)

    assert output.summary == (
        "Patient [NAME] presented with a cough. "
        "The synthetic admission was uncomplicated."
    )
    assert output.leakage_check.passed is True
    assert "Casey Example" not in output.summary


def test_ordering_guard_rejects_raw_input():
    with pytest.raises(SummarizationOrderError, match="requires a de-identification"):
        summarize_deidentified(SYNTHETIC_NOTE)  # type: ignore[arg-type]


def test_leakage_guard_rejects_backend_reemission_without_exposing_token():
    with pytest.raises(SummarizationLeakageError) as raised:
        summarize_deidentified(
            _deidentified_result(),
            model=lambda _text: "Casey Example returned for follow-up.",
        )

    assert raised.value.check.passed is False
    assert raised.value.check.leaked_token_count == 1
    assert "Casey Example" not in str(raised.value)


def test_result_can_be_unpacked_as_summary_and_leakage_check():
    summary, check = summarize_deidentified(_deidentified_result())

    assert summary.startswith("Patient [NAME]")
    assert check.passed is True
