"""Synthetic checkpoint, late-provider and privacy refusal regression tests."""

import asyncio
import importlib
import json
from dataclasses import replace

import pytest

from openmed.clinical.brief import STAGES, BriefRefusal, build_clinical_brief
from openmed.clinical.brief_cancellation import BriefCancellation, BriefInterrupted
from openmed.core.budget import RequestBudget
from tests.unit.clinical.test_brief import SENTENCES, fixture_context


@pytest.mark.parametrize("stage", STAGES)
@pytest.mark.parametrize("expired", [False, True])
def test_interrupt_at_every_stage_prevents_later_work(monkeypatch, stage, expired):
    module = importlib.import_module("openmed.clinical.brief")
    clock_time = [0.0]
    monkeypatch.setattr("openmed.core.budget.time.perf_counter", lambda: clock_time[0])
    cancellation = BriefCancellation(RequestBudget(max_wall_time=1).start())
    original = module.check_cancellation
    entered = []

    def checkpoint(context):
        # stage() calls the checkpoint before recording each stage.
        if len(entered) == STAGES.index(stage):
            if expired:
                clock_time[0] = 2.0
            else:
                cancellation.cancel()
        original(context)
        entered.append(STAGES[len(entered)])

    monkeypatch.setattr(module, "check_cancellation", checkpoint)
    value, context = fixture_context()
    result = build_clinical_brief(
        value, context=context, model="extractive", cancellation=cancellation
    )
    assert result.refusal_reason is (
        BriefRefusal.DEADLINE_EXCEEDED if expired else BriefRefusal.CANCELLED
    )
    assert result.to_dict()["stages"] == list(STAGES[: STAGES.index(stage)])
    assert result.summary == ""
    assert result.citations == result.verdicts == ()
    assert all(
        sentence not in json.dumps(result.to_response()) for sentence in SENTENCES
    )


@pytest.mark.parametrize("failure", [False, True])
@pytest.mark.parametrize("provider", ["generation", "nli", "privacy", "final_review"])
def test_late_results_and_errors_are_discarded(provider, failure):
    value, context = fixture_context()
    cancellation = BriefCancellation()
    calls = []

    def late(*args, **kwargs):
        calls.append(provider)
        cancellation.cancel()
        if failure:
            raise RuntimeError("SYNTHETIC_PRIVATE_FAILURE")
        return value.deidentified_text

    model = "extractive"
    if provider == "generation":
        model = late
    elif provider == "nli":
        context = replace(context, nli_predict=late)
    else:
        count = [0]

        def detector(text):
            count[0] += 1
            if count[0] == (2 if provider == "final_review" else 1):
                late(text)
            return []

        context = replace(context, privacy_detector=detector)
    result = build_clinical_brief(
        value, context=context, model=model, cancellation=cancellation
    )
    assert result.refusal_reason is BriefRefusal.CANCELLED
    assert calls == [provider]
    assert result.summary == ""
    assert "SYNTHETIC_PRIVATE" not in json.dumps(result.to_response())
    assert result.citations == result.verdicts == ()


def test_cooperative_callbacks_receive_same_context_and_uncancelled_output_matches():
    value, context = fixture_context()
    cancellation = BriefCancellation()
    observed = []
    original_nli = context.nli_predict

    def generate(text, *, cancellation):
        observed.append(cancellation)
        cancellation.check()
        return text

    def predict(premise, hypothesis, *, cancellation):
        observed.append(cancellation)
        return original_nli(premise, hypothesis)

    result = build_clinical_brief(
        value,
        context=replace(context, nli_predict=predict),
        model=generate,
        cancellation=cancellation,
    )
    baseline = build_clinical_brief(value, context=context, model=lambda text: text)
    assert result.to_response() == baseline.to_response()
    assert observed == [cancellation] * 4


@pytest.mark.parametrize(
    "error", [KeyboardInterrupt, asyncio.CancelledError, RuntimeError]
)
def test_interruption_and_backend_failure_are_distinct(error):
    value, context = fixture_context()

    def fail(_):
        raise error("SYNTHETIC_PRIVATE_FAILURE")

    result = build_clinical_brief(value, context=context, model=fail)
    assert result.refusal_reason is (
        BriefRefusal.STAGE_FAILED if error is RuntimeError else BriefRefusal.CANCELLED
    )
    assert result.summary == ""
    assert "SYNTHETIC_PRIVATE" not in json.dumps(result.to_response())


def test_expiry_is_terminal_and_explicit_cancel_wins(monkeypatch):
    now = [0.0]
    monkeypatch.setattr("openmed.core.budget.time.perf_counter", lambda: now[0])
    cancellation = BriefCancellation(RequestBudget(max_wall_time=1).start())
    now[0] = 2.0
    with pytest.raises(BriefInterrupted, match="deadline_exceeded"):
        cancellation.check()
    now[0] = 0.0
    with pytest.raises(BriefInterrupted, match="deadline_exceeded"):
        cancellation.check()
    cancellation.cancel()
    with pytest.raises(BriefInterrupted, match="cancelled"):
        cancellation.check()


def test_interruption_exception_does_not_chain_private_provider_failure():
    from openmed.clinical.brief_cancellation import call_with_cancellation

    cancellation = BriefCancellation()

    def fail():
        cancellation.cancel()
        raise RuntimeError("SYNTHETIC_PRIVATE_PROVIDER_FAILURE")

    with pytest.raises(BriefInterrupted) as caught:
        call_with_cancellation(fail, cancellation=cancellation)
    assert caught.value.__context__ is None
    assert caught.value.__cause__ is None
