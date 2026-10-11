"""Focused tests for the post-de-identification summarization stage."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace

import pytest

from openmed.clinical.summarize import (
    SummarizationLeakageError,
    SummarizationOrderError,
    summarize,
    summarize_deidentified,
)
from openmed.core.pii import DeidentificationResult, PIIEntity

summarize_module = import_module("openmed.clinical.summarize")


def _leakage_parity_cases():
    path = (
        Path(__file__).resolve().parents[2]
        / "fixtures/clinical/brief_parity/verified.json"
    )
    return json.loads(path.read_text())["leakage_cases"]


@pytest.mark.parametrize("case", _leakage_parity_cases(), ids=lambda case: case["id"])
def test_shared_unicode_leakage_backend_cases(case):
    surface = case["surface"]
    source = DeidentificationResult(
        original_text=surface,
        deidentified_text="[NAME] improved.",
        pii_entities=[
            PIIEntity(
                text=surface,
                label="NAME",
                start=0,
                end=len(surface),
                confidence=0.99,
                redacted_text="[NAME]",
            )
        ],
        method="mask",
        timestamp=datetime(2026, 1, 1),
    )
    if case["leaked"]:
        with pytest.raises(SummarizationLeakageError) as raised:
            summarize_deidentified(source, model=lambda _: case["candidate"])
        check = raised.value.check
        assert check.checked_token_count == check.leaked_token_count == 1
        assert len(check.leaked_token_hashes[0]) == 64
        expected_digest = hashlib.sha256(" ".join(surface.split()).encode()).hexdigest()
        assert check.to_dict() == {
            "passed": False,
            "checked_token_count": 1,
            "leaked_token_count": 1,
            "leaked_token_hashes": [expected_digest],
        }
        assert (
            str(raised.value)
            == "summary leakage guard rejected backend output: 1 source token(s) detected"
        )
        assert case["candidate"] not in json.dumps(check.to_dict(), ensure_ascii=False)
    else:
        result = summarize_deidentified(source, model=lambda _: case["candidate"])
        assert result.leakage_check.passed


def test_shared_native_normalization_mirrors_existing_detector_defenses():
    from openmed.core.script_detect import _CONFUSABLE_FOLD

    path = (
        Path(__file__).resolve().parents[2]
        / "fixtures/clinical/brief_parity/verified.json"
    )
    cases = json.loads(path.read_text())["normalization_cases"]
    assert {"x" + char + "y" for char in _CONFUSABLE_FOLD} <= {
        case["input"] for case in cases
    }
    for case in cases:
        assert (
            summarize_module._normalize_leakage_text(case["input"])
            == case["normalized"]
        )


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

    def fake_deidentify(text: str, *, method: str, config) -> DeidentificationResult:
        calls.append(("deidentify", text))
        assert method == "mask"
        assert config.local_only is True
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


def test_explicit_extractive_summary_contains_no_original_phi(monkeypatch):
    monkeypatch.setattr(
        summarize_module,
        "deidentify",
        lambda text, *, method, config: _deidentified_result(),
    )

    output = summarize(SYNTHETIC_NOTE, model="extractive")

    assert output.summary == (
        "Patient [NAME] presented with a cough. "
        "The synthetic admission was uncomplicated."
    )
    assert output.leakage_check.passed is True
    assert "Casey Example" not in output.summary


def test_ordering_guard_rejects_raw_input():
    with pytest.raises(SummarizationOrderError, match="requires a de-identification"):
        summarize_deidentified(SYNTHETIC_NOTE)  # type: ignore[arg-type]


def test_ordering_guard_rejects_lookalike_input():
    lookalike = SimpleNamespace(
        original_text="synthetic source",
        deidentified_text="synthetic output",
        pii_entities=[],
    )

    with pytest.raises(SummarizationOrderError, match="requires a de-identification"):
        summarize_deidentified(lookalike)  # type: ignore[arg-type]


def test_leakage_guard_rejects_backend_reemission_without_exposing_token():
    with pytest.raises(SummarizationLeakageError) as raised:
        summarize_deidentified(
            _deidentified_result(),
            model=lambda _text: "Casey Example returned for follow-up.",
        )

    assert raised.value.check.passed is False
    assert raised.value.check.leaked_token_count == 1
    assert "Casey Example" not in str(raised.value)


def test_leakage_guard_rejects_partial_source_name_token():
    with pytest.raises(SummarizationLeakageError) as raised:
        summarize_deidentified(
            _deidentified_result(),
            model=lambda _text: "Casey returned for follow-up.",
        )

    assert raised.value.check.passed is False
    assert raised.value.check.leaked_token_count == 1
    assert "Casey" not in str(raised.value)


def test_result_can_be_unpacked_as_summary_and_leakage_check():
    summary, check = summarize_deidentified(_deidentified_result(), model="extractive")

    assert summary.startswith("Patient [NAME]")
    assert check.passed is True
