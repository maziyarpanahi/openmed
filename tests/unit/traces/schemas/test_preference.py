"""Focused tests for the local preference-pair schema adapter."""

from __future__ import annotations

from copy import deepcopy

import pytest

from openmed.traces.schemas.preference import (
    CONTENT_FIELDS,
    PreferencePair,
    PreferencePairAdapter,
    PreferenceRedactionReport,
    PreferenceRedactionState,
    PreferenceSchemaError,
    SensitiveSpan,
)


def test_span_repr_and_direct_report_do_not_expose_caller_metadata() -> None:
    sensitive = "PatientJaneDoe"
    span = SensitiveSpan(0, 4, sensitive)

    assert sensitive not in repr(span)
    with pytest.raises(PreferenceSchemaError) as caught:
        PreferenceRedactionReport(schema_version=sensitive)
    assert sensitive not in str(caught.value)


def test_redacts_three_branches_with_one_shared_pseudonym_state():
    record = {
        "pair_id": "synthetic-pair-01",
        "prompt": "Contact Ada Example at ada@example.test.",
        "chosen": "Ada Example confirmed the appointment at ada@example.test.",
        "rejected": "Please ask Ada Example at ada@example.test to confirm.",
        "scores": {"chosen": 0.91, "rejected": 0.09},
        "metadata": {"synthetic": True, "source": "offline-fixture"},
    }
    original = deepcopy(record)

    result = PreferencePairAdapter(seed=17).redact(record)

    assert record == original
    assert result["pair_id"] == record["pair_id"]
    assert result["scores"] == record["scores"]
    assert result["metadata"] == record["metadata"]
    assert "Ada Example" not in result["prompt"]
    assert "ada@example.test" not in result["prompt"]
    prompt_name = result["prompt"].split("Contact ", 1)[1].split(" at ", 1)[0]
    chosen_name = result["chosen"].split(" confirmed", 1)[0]
    assert prompt_name == chosen_name
    assert "ada@example.test" not in result["chosen"]
    assert "ada@example.test" not in result["rejected"]


def test_walks_message_content_but_preserves_roles_and_non_content_metadata():
    record = {
        "prompt": [
            {"role": "user", "content": "Email bob@example.test."},
            {"role": "system", "content": "Use the safe policy."},
        ],
        "chosen": {
            "messages": [{"role": "assistant", "content": "Call bob@example.test."}]
        },
        "rejected": [{"role": "assistant", "content": "Ignore bob@example.test."}],
        "metadata": {"note": "metadata is preserved"},
    }

    result = PreferencePairAdapter(seed=3).redact(record)

    assert result["prompt"][0]["role"] == "user"
    assert result["prompt"][1] == record["prompt"][1]
    assert result["chosen"]["messages"][0]["role"] == "assistant"
    assert result["metadata"] == record["metadata"]
    assert "bob@example.test" not in str(result["prompt"])
    assert "bob@example.test" not in str(result["chosen"])
    assert "bob@example.test" not in str(result["rejected"])


def test_custom_detector_and_two_argument_redactor_share_state():
    def detector(text: str):
        start = text.find("SYNTH-42")
        return () if start < 0 else (SensitiveSpan(start, start + 8, "ID_NUM"),)

    def redactor(text: str, state):
        return state.redact_spans(text, detector(text))

    record = {
        "prompt": "SYNTH-42 appears here.",
        "chosen": "The same SYNTH-42 appears here.",
        "rejected": "SYNTH-42 is repeated.",
    }

    result = PreferencePairAdapter(text_redactor=redactor).redact_with_report(record)

    assert result.record["prompt"]
    assert "SYNTH-42" not in str(result.record)
    replacement = result.record["prompt"].split(" appears", 1)[0]
    assert replacement in result.record["chosen"]
    assert replacement in result.record["rejected"]
    assert result.report.replacement_count == 3
    assert "SYNTH-42" not in str(result.report.to_dict())


def test_falsey_callable_redactor_is_not_discarded() -> None:
    class FalseyRedactor:
        def __bool__(self) -> bool:
            return False

        def __call__(self, text: str) -> str:
            return text.replace("SYNTH-42", "[ID]")

    record = {
        "prompt": "SYNTH-42 prompt",
        "chosen": "SYNTH-42 chosen",
        "rejected": "SYNTH-42 rejected",
    }

    result = PreferencePairAdapter(text_redactor=FalseyRedactor()).redact(record)

    assert all(
        "SYNTH-42" not in result[field] for field in ("prompt", "chosen", "rejected")
    )


def test_falsey_callable_detector_is_not_discarded() -> None:
    class FalseyDetector:
        calls = 0

        def __bool__(self) -> bool:
            return False

        def __call__(self, text: str):
            self.calls += 1
            return (SensitiveSpan(0, len(text), "ID_NUM"),)

    detector = FalseyDetector()
    record = {
        "prompt": "opaque prompt value",
        "chosen": "opaque chosen value",
        "rejected": "opaque rejected value",
    }

    result = PreferencePairAdapter(span_detector=detector).redact(record)

    assert detector.calls == 3
    assert all(result[field] != record[field] for field in CONTENT_FIELDS)


def test_hostile_anonymizer_property_error_is_sanitized() -> None:
    sensitive = "synthetic private anonymizer value"

    class HostileAnonymizer:
        @property
        def surrogate(self):
            raise RuntimeError(sensitive)

    with pytest.raises(PreferenceSchemaError) as caught:
        PreferenceRedactionState(anonymizer=HostileAnonymizer())

    assert sensitive not in str(caught.value)


def test_dataset_iterator_failures_do_not_echo_source_values() -> None:
    sensitive = "synthetic-secret@example.test"
    record = {"prompt": "safe", "chosen": "safe", "rejected": "safe"}

    def failing_records():
        yield record
        raise RuntimeError(sensitive)

    with pytest.raises(PreferenceSchemaError) as caught:
        PreferencePairAdapter().redact_dataset(failing_records())

    assert sensitive not in str(caught.value)
    assert "could not be read" in str(caught.value)


def test_determinism_is_stable_across_adapter_instances():
    record = {
        "prompt": "Call +1 415 555 0123.",
        "chosen": "The number is +1 415 555 0123.",
        "rejected": "Do not call +1 415 555 0123.",
    }

    first = PreferencePairAdapter(seed=99).redact(record)
    second = PreferencePairAdapter(seed=99).redact(record)

    assert first == second
    assert "+1 415 555 0123" not in str(first)


def test_pair_view_preserves_scores_and_extra_metadata_without_raw_repr():
    pair = PreferencePair.from_mapping(
        {
            "prompt": "Synthetic prompt",
            "chosen": "Synthetic chosen",
            "rejected": "Synthetic rejected",
            "score": 1.0,
            "metadata": {"synthetic": True},
        }
    )

    assert pair.scores == 1.0
    assert pair.metadata == {"synthetic": True}
    assert pair.to_mapping()["score"] == 1.0
    assert "Synthetic prompt" not in repr(pair)


def test_invalid_pair_errors_do_not_echo_content():
    secret = "synthetic-secret@example.test"

    with pytest.raises(PreferenceSchemaError) as exc_info:
        PreferencePairAdapter().redact({"prompt": secret, "chosen": "ok"})

    assert secret not in str(exc_info.value)
    assert "rejected" in str(exc_info.value)


def test_label_normalization_failure_does_not_echo_the_label(
    monkeypatch: pytest.MonkeyPatch,
):
    import openmed.core.labels as labels

    sensitive_label = "PatientJaneDoe"

    def fail_normalization(label: str, lang: str) -> str:
        del label, lang
        raise RuntimeError("synthetic normalization failure")

    class EchoAnonymizer:
        def surrogate(self, value: str, label: str, **kwargs: object) -> str:
            del label, kwargs
            return value

    monkeypatch.setattr(labels, "normalize_label", fail_normalization)
    state = PreferenceRedactionState(anonymizer=EchoAnonymizer())

    pseudonym = state.pseudonym("synthetic secret", sensitive_label)

    assert sensitive_label.casefold() not in pseudonym.casefold()
    assert pseudonym.startswith("[other-")
