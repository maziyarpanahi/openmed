"""Focused tests for the post-de-identification summarization stage."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from importlib import import_module
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


def test_default_serialization_matches_recorded_master_bytes(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("default summary resolved NLI")

    from openmed.clinical import nli_backends

    monkeypatch.setattr(nli_backends, "resolve_nli_backend", forbidden)
    result = summarize(_deidentified_result(), model="extractive")
    assert result.verification is None
    assert "verification" not in result.to_dict()
    encoded = json.dumps(
        result.to_dict(), sort_keys=True, separators=(",", ":")
    ).encode()
    assert (
        hashlib.sha256(encoded).hexdigest()
        == "ef144ee0928dd32f0a8f9ac8198bc3e249bf6544656a97d21d9ab56253b091e3"
    )


def _nli_source() -> DeidentificationResult:
    return DeidentificationResult(
        original_text="The patient has pneumonia.",
        deidentified_text="The patient has pneumonia.",
        pii_entities=[],
        method="mask",
        timestamp=datetime(2026, 1, 1),
    )


def test_summary_hook_retains_entailed_and_contradicted_claims():
    text = "The patient has pneumonia. The patient has no pneumonia."
    result = summarize(_nli_source(), model=lambda _: text, verify="heuristic")
    assert result.summary == text
    assert [check.label for check in result.verification] == [
        "entailment",
        "contradiction",
    ]
    assert result.verification[1].contradicted
    assert result.verification[1].review_required
    assert [text[slice(*check.claim_offset)] for check in result.verification] == [
        "The patient has pneumonia.",
        "The patient has no pneumonia.",
    ]
    for check in result.verification:
        assert check.source_offset == (0, len(_nli_source().deidentified_text))
        assert "pneumonia" not in json.dumps(check.to_dict())
        assert check.source_digest.startswith("sha256:")


def test_raw_summary_verification_receives_only_deidentified_source(monkeypatch):
    calls = []

    class Backend:
        backend_id = "synthetic-nli"

        def predict(self, premise, hypothesis):
            calls.append((premise, hypothesis))
            assert "Casey" not in premise + hypothesis
            return {"label": "abstention", "score": 0.0, "source": SYNTHETIC_NOTE}

    monkeypatch.setattr(
        summarize_module, "deidentify", lambda *args, **kwargs: _deidentified_result()
    )
    result = summarize(SYNTHETIC_NOTE, model="extractive", verify=Backend())
    assert calls
    assert all(
        check.label == "abstention" and check.review_required
        for check in result.verification
    )
    assert "Casey" not in json.dumps([check.to_dict() for check in result.verification])


def test_leakage_is_rejected_before_nli_is_called():
    class ForbiddenBackend:
        def predict(self, *args):
            raise AssertionError("NLI saw a leaking summary")

    with pytest.raises(SummarizationLeakageError):
        summarize(
            _deidentified_result(),
            model=lambda _: "Casey Example",
            verify=ForbiddenBackend(),
        )


def test_true_uses_configured_backend_and_missing_backend_is_typed(monkeypatch):
    import importlib

    nli_module = importlib.import_module("openmed.clinical.nli")
    from openmed.clinical.nli_backends import LocalNLIError

    monkeypatch.setattr(nli_module, "DEFAULT_NLI_BACKEND", "heuristic")
    result = summarize(_nli_source(), model="extractive", verify=True)
    assert result.verification[0].label == "entailment"
    with pytest.raises(LocalNLIError, match="not registered"):
        summarize(
            _nli_source(), model="extractive", verify="missing-synthetic-checkpoint"
        )


@pytest.mark.parametrize("option", [None, 0, 1, "https://private.invalid/payload"])
def test_invalid_verify_option_is_not_silently_treated_as_off(option):
    from openmed.clinical.nli_backends import LocalNLIError

    with pytest.raises(LocalNLIError) as raised:
        summarize(_nli_source(), model="extractive", verify=option)
    assert "private.invalid" not in str(raised.value)


def test_verification_backend_cannot_use_network_and_error_is_value_free():
    import socket

    from openmed.clinical.nli_backends import LocalNLIError

    class Backend:
        backend_id = "synthetic-nli"

        def predict(self, premise, hypothesis):
            socket.create_connection(("example.invalid", 443))
            raise AssertionError("network must be blocked")

    with pytest.raises(LocalNLIError, match="local NLI inference failed") as raised:
        summarize(_nli_source(), model="extractive", verify=Backend())
    assert "pneumonia" not in str(raised.value)


def test_unsegmentable_summary_claim_preserves_review_marker():
    result = summarize(
        _nli_source(),
        model=lambda _: "Pneumonia and fever.",
        verify=lambda *_: {"label": "entailment", "score": 0.9},
    )
    assert result.verification[0].label == "entailment"
    assert result.verification[0].review_required


def test_summary_verification_cannot_be_rebound_to_changed_claims():
    from dataclasses import replace

    result = summarize(_nli_source(), model="extractive", verify="heuristic")
    with pytest.raises(ValueError, match="invalid summary verification metadata"):
        replace(result, summary="A different synthetic claim.")


def test_true_with_unavailable_default_retains_local_nli_refusal(monkeypatch):
    from openmed.clinical import nli_backends
    from openmed.clinical.nli_backends import LocalNLIError

    monkeypatch.setattr(nli_backends, "get_default_nli_model", lambda: None)
    with pytest.raises(LocalNLIError, match="no released local NLI checkpoint"):
        summarize(_nli_source(), model="extractive", verify=True)


@pytest.mark.parametrize("failure", ["resolve", "infer", "identity"])
def test_verification_provider_properties_cannot_leak_exception_text(failure, capsys):
    from openmed.clinical.nli_backends import LocalNLIError

    private = "Synthetic private source and claim text"

    class Backend:
        accesses = 0

        @property
        def predict(self):
            self.accesses += 1
            if failure == "resolve" or (failure == "infer" and self.accesses > 1):
                raise RuntimeError(private)
            return lambda *_: {"label": "entailment", "score": 0.9}

        @property
        def backend_id(self):
            raise LocalNLIError(private)

    with pytest.raises(LocalNLIError) as raised:
        summarize(_nli_source(), model="extractive", verify=Backend())
    assert private not in str(raised.value)
    assert raised.value.__suppress_context__
    captured = capsys.readouterr()
    assert private not in captured.out + captured.err
