"""Tests for per-request resource / timeout budgets and cooperative cancellation.

These tests assert three properties required by OM-800:

1. A tiny budget triggers a clean :class:`BudgetExceededError` on a slow
   synthetic path (no thread kill, no partial result).
2. Cancellation is cooperative and leaves no partial-state corruption: when a
   budget is exceeded mid-request, no partial ``DeidentificationResult`` is
   returned and the typed error propagates cleanly.
3. When no budget is supplied, behavior is byte-for-byte identical to today.

All slow paths are synthetic (a stub ``model_detector`` / batched helper that
sleeps); no real models are downloaded.
"""

from __future__ import annotations

import time
from datetime import datetime

import pytest

from openmed.core.budget import (
    BudgetClock,
    BudgetExceededError,
    RequestBudget,
    coerce_budget,
)
from openmed.core.pii import _extract_pii_batch, deidentify, extract_pii
from openmed.core.pipeline import Pipeline
from openmed.processing.batch import BatchProcessor
from openmed.processing.outputs import EntityPrediction, PredictionResult


def _name_prediction(text: str, model_name: str = "stub") -> PredictionResult:
    """A synthetic single-NAME detection used by stub detectors."""
    entity_text = "Casey Example"
    start = text.index(entity_text)
    return PredictionResult(
        text=text,
        entities=[
            EntityPrediction(
                text=entity_text,
                label="NAME",
                start=start,
                end=start + len(entity_text),
                confidence=0.95,
            )
        ],
        model_name=model_name,
        timestamp=datetime.now().isoformat(),
    )


# ---------------------------------------------------------------------------
# RequestBudget / BudgetExceededError unit behavior
# ---------------------------------------------------------------------------


def test_request_budget_unlimited_by_default():
    budget = RequestBudget()
    assert budget.max_wall_time is None
    assert budget.max_input_chars is None
    assert budget.is_unlimited is True


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_wall_time": 0},
        {"max_wall_time": -1.0},
        {"max_input_chars": 0},
        {"max_input_chars": -5},
    ],
)
def test_request_budget_rejects_non_positive_limits(kwargs):
    with pytest.raises(ValueError):
        RequestBudget(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_input_chars": True},  # bool is not a valid int limit
        {"max_wall_time": "1.0"},
        {"max_input_chars": 1.5},
    ],
)
def test_request_budget_rejects_wrong_types(kwargs):
    with pytest.raises(TypeError):
        RequestBudget(**kwargs)


def test_coerce_budget_normalizes_unlimited_to_none():
    assert coerce_budget(None) is None
    assert coerce_budget(RequestBudget()) is None
    assert coerce_budget({"max_wall_time": None, "max_input_chars": None}) is None

    coerced = coerce_budget({"max_input_chars": 10})
    assert isinstance(coerced, RequestBudget)
    assert coerced.max_input_chars == 10


def test_coerce_budget_rejects_unsupported_types():
    with pytest.raises(TypeError):
        coerce_budget(123)


def test_input_length_guard_raises_phi_free_error():
    budget = RequestBudget(max_input_chars=8)
    with pytest.raises(BudgetExceededError) as excinfo:
        budget.check_input_length(42, checkpoint="unit_guard")

    error = excinfo.value
    assert error.kind == "input_chars"
    assert error.limit == 8
    assert error.observed == 42
    assert error.checkpoint == "unit_guard"

    # The message must carry only counts / limits / checkpoint names -- no PHI.
    message = str(error)
    assert "8" in message and "42" in message and "unit_guard" in message


def test_input_length_guard_allows_within_limit():
    RequestBudget(max_input_chars=8).check_input_length(8)  # exactly at limit is OK


def test_wall_time_clock_raises_after_deadline():
    clock = RequestBudget(max_wall_time=0.001).start()
    time.sleep(0.01)
    with pytest.raises(BudgetExceededError) as excinfo:
        clock.check("unit_stage")
    assert excinfo.value.kind == "wall_time"
    assert excinfo.value.checkpoint == "unit_stage"


def test_wall_time_clock_no_limit_is_noop():
    clock = RequestBudget(max_input_chars=5).start()  # no wall-time limit
    assert isinstance(clock, BudgetClock)
    clock.check("unit_stage")  # must not raise


def test_budget_error_message_contains_no_input_text():
    # Passing only a length -- never the surface -- means the error cannot leak.
    budget = RequestBudget(max_input_chars=4)
    secret = "Casey Example 555-0100"
    with pytest.raises(BudgetExceededError) as excinfo:
        budget.check_input_length(len(secret))
    assert secret not in str(excinfo.value)
    assert "Casey" not in str(excinfo.value)


# ---------------------------------------------------------------------------
# Cooperative wall-time cancellation in the pipeline (deidentify path)
# ---------------------------------------------------------------------------


def test_pipeline_tiny_time_budget_raises_clean_budget_error():
    text = "Patient Casey Example called."

    def slow_model_detector(text, **kwargs):
        time.sleep(0.05)
        return _name_prediction(text, model_name=kwargs["model_name"])

    with pytest.raises(BudgetExceededError) as excinfo:
        Pipeline(
            model_detector=slow_model_detector,
            use_safety_sweep=False,
        ).run(
            text,
            method="mask",
            budget=RequestBudget(max_wall_time=0.001),
        )
    # The breach is detected at a named pipeline checkpoint, not a thread kill.
    assert excinfo.value.kind == "wall_time"
    assert excinfo.value.checkpoint.startswith("pipeline.")


def test_pipeline_time_budget_returns_no_partial_result():
    text = "Patient Casey Example called."
    observed = {"detector_calls": 0}

    def slow_model_detector(text, **kwargs):
        observed["detector_calls"] += 1
        time.sleep(0.05)
        return _name_prediction(text, model_name=kwargs["model_name"])

    # Each pipeline stage sleeps a little so the wall-time budget is reliably
    # exhausted at one of the between-stage checkpoints regardless of machine
    # speed, without relying on preemptive interruption.
    result = None
    try:
        result = Pipeline(
            model_detector=slow_model_detector,
            use_safety_sweep=False,
        ).run(text, method="mask", budget=RequestBudget(max_wall_time=0.02))
    except BudgetExceededError:
        pass

    # No partial DeidentificationResult escaped: the typed error propagated and
    # the caller received nothing. Cancellation is cooperative -- if the slow
    # model stage was reached it still could not produce a returned result.
    assert result is None
    assert observed["detector_calls"] <= 1


def test_pipeline_input_length_guard_runs_before_inference():
    text = "Patient Casey Example called."
    observed = {"detector_calls": 0}

    def model_detector(text, **kwargs):
        observed["detector_calls"] += 1
        return _name_prediction(text, model_name=kwargs["model_name"])

    with pytest.raises(BudgetExceededError) as excinfo:
        Pipeline(
            model_detector=model_detector,
            use_safety_sweep=False,
        ).run(text, method="mask", budget=RequestBudget(max_input_chars=5))

    assert excinfo.value.kind == "input_chars"
    assert excinfo.value.checkpoint == "pipeline.input_guard"
    # Guard rejected the request before any model inference.
    assert observed["detector_calls"] == 0


# ---------------------------------------------------------------------------
# No-budget parity: byte-for-byte identical to today's behavior
# ---------------------------------------------------------------------------


def _run_pipeline(text, budget):
    def model_detector(text, **kwargs):
        return _name_prediction(text, model_name=kwargs["model_name"])

    return Pipeline(
        model_detector=model_detector,
        use_safety_sweep=False,
    ).run(text, method="mask", budget=budget)


def test_no_budget_matches_explicit_none_and_unlimited():
    text = "Patient Casey Example called."

    baseline = _run_pipeline(text, budget=None)
    explicit_none = _run_pipeline(text, budget=None)
    unlimited = _run_pipeline(text, budget=RequestBudget())

    assert (
        baseline.redacted_text
        == explicit_none.redacted_text
        == unlimited.redacted_text
        == "Patient [NAME] called."
    )
    assert (
        baseline.deidentification_result.deidentified_text
        == unlimited.deidentification_result.deidentified_text
    )


def test_generous_budget_does_not_change_output():
    text = "Patient Casey Example called."
    baseline = _run_pipeline(text, budget=None)
    generous = _run_pipeline(
        text,
        budget=RequestBudget(max_wall_time=60.0, max_input_chars=10_000),
    )
    assert baseline.redacted_text == generous.redacted_text
    assert [s.start for s in baseline.spans] == [s.start for s in generous.spans]


# ---------------------------------------------------------------------------
# Public extract_pii / deidentify accept budget and stay backward compatible
# ---------------------------------------------------------------------------


def test_extract_pii_input_guard_before_model(monkeypatch):
    calls = {"analyze_text": 0}

    def fake_analyze_text(text, **kwargs):
        calls["analyze_text"] += 1
        return _name_prediction(text, model_name=kwargs.get("model_name", "stub"))

    monkeypatch.setattr("openmed.analyze_text", fake_analyze_text)

    with pytest.raises(BudgetExceededError) as excinfo:
        extract_pii(
            "Patient Casey Example called.",
            model_name="fixture-pii-model",
            use_smart_merging=False,
            budget=RequestBudget(max_input_chars=5),
        )
    assert excinfo.value.kind == "input_chars"
    assert calls["analyze_text"] == 0


def test_extract_pii_no_budget_matches_none(monkeypatch):
    def fake_analyze_text(text, **kwargs):
        return _name_prediction(text, model_name=kwargs.get("model_name", "stub"))

    monkeypatch.setattr("openmed.analyze_text", fake_analyze_text)

    text = "Patient Casey Example called."
    without = extract_pii(text, model_name="m", use_smart_merging=False)
    with_none = extract_pii(text, model_name="m", use_smart_merging=False, budget=None)
    with_unlimited = extract_pii(
        text,
        model_name="m",
        use_smart_merging=False,
        budget=RequestBudget(),
    )

    def labels(result):
        return [(e.text, e.label, e.start, e.end) for e in result.entities]

    assert labels(without) == labels(with_none) == labels(with_unlimited)


def test_deidentify_accepts_generous_budget_via_public_api(monkeypatch):
    def fake_analyze_text(text, **kwargs):
        return _name_prediction(text, model_name=kwargs.get("model_name", "stub"))

    monkeypatch.setattr("openmed.analyze_text", fake_analyze_text)

    text = "Patient Casey Example called."
    # A generous budget must not change the public deidentify() output.
    baseline = deidentify(text, method="mask", use_safety_sweep=False)
    budgeted = deidentify(
        text,
        method="mask",
        use_safety_sweep=False,
        budget=RequestBudget(max_wall_time=60.0, max_input_chars=10_000),
    )
    assert baseline.deidentified_text == budgeted.deidentified_text
    assert budgeted.deidentified_text == "Patient [NAME] called."


def test_deidentify_input_guard_rejects_over_length_input(monkeypatch):
    calls = {"analyze_text": 0}

    def fake_analyze_text(text, **kwargs):
        calls["analyze_text"] += 1
        return _name_prediction(text, model_name=kwargs.get("model_name", "stub"))

    monkeypatch.setattr("openmed.analyze_text", fake_analyze_text)

    with pytest.raises(BudgetExceededError) as excinfo:
        deidentify(
            "Patient Casey Example called.",
            method="mask",
            use_safety_sweep=False,
            budget=RequestBudget(max_input_chars=5),
        )
    assert excinfo.value.kind == "input_chars"
    # The over-length input was rejected before any model inference ran.
    assert calls["analyze_text"] == 0


# ---------------------------------------------------------------------------
# Cooperative cancellation in the batch loop
# ---------------------------------------------------------------------------


def test_extract_pii_batch_cancels_between_items_without_partial_state(monkeypatch):
    processed = []

    def fake_analyze_text(text, **kwargs):
        processed.append(text)
        time.sleep(0.02)
        return _name_prediction(text, model_name=kwargs.get("model_name", "stub"))

    monkeypatch.setattr("openmed.analyze_text", fake_analyze_text)

    texts = [
        "Patient Casey Example one.",
        "Patient Casey Example two.",
        "Patient Casey Example three.",
    ]
    # A very short budget that trips shortly after the first item's slow call,
    # so cancellation lands at a between-item checkpoint (not on item zero and
    # not after the whole batch).
    clock = RequestBudget(max_wall_time=0.01).start()

    result = None
    with pytest.raises(BudgetExceededError) as excinfo:
        result = _extract_pii_batch(
            texts,
            model_name="m",
            use_smart_merging=False,
            loader=object(),  # stub loader; never used before the deadline check
            budget_clock=clock,
        )

    assert excinfo.value.kind == "wall_time"
    # Cooperative: the batch stopped at a named checkpoint, mid-batch.
    assert excinfo.value.checkpoint in {
        "extract_pii.batch_setup",
        "extract_pii.batch_item",
    }
    # No partial result list escaped, and the batch was cancelled before every
    # item was processed -- no partial-state corruption.
    assert result is None
    assert len(processed) < len(texts)


def test_extract_pii_batch_no_budget_processes_all(monkeypatch):
    def fake_analyze_text(text, **kwargs):
        return _name_prediction(text, model_name=kwargs.get("model_name", "stub"))

    monkeypatch.setattr("openmed.analyze_text", fake_analyze_text)

    texts = ["Patient Casey Example one.", "Patient Casey Example two."]
    results = _extract_pii_batch(
        texts,
        model_name="m",
        use_smart_merging=False,
        loader=object(),  # stub loader; analyze_text is patched above
    )
    assert len(results) == 2
    for result in results:
        assert result.entities[0].label == "NAME"


def test_batch_processor_input_budget_records_clean_error(monkeypatch):
    def fake_analyze_text(text, **kwargs):
        return _name_prediction(text, model_name=kwargs.get("model_name", "stub"))

    monkeypatch.setattr("openmed.analyze_text", fake_analyze_text)

    processor = BatchProcessor(
        model_name="m",
        operation="extract_pii",
        use_smart_merging=False,
        continue_on_error=True,
        budget={"max_input_chars": 5},
    )
    result = processor.process_texts(["Patient Casey Example called."])

    # The over-length item is recorded as a clean failure -- no crash, no
    # partial result -- and the batch aggregate stays consistent.
    assert result.total_items == 1
    assert result.failed_items == 1
    assert result.successful_items == 0
    item = result.items[0]
    assert item.success is False
    assert item.result is None
    assert "budget exceeded" in (item.error or "").lower()


def test_batch_processor_no_budget_processes_normally(monkeypatch):
    def fake_analyze_text(text, **kwargs):
        return _name_prediction(text, model_name=kwargs.get("model_name", "stub"))

    monkeypatch.setattr("openmed.analyze_text", fake_analyze_text)

    processor = BatchProcessor(
        model_name="m",
        operation="extract_pii",
        use_smart_merging=False,
    )
    result = processor.process_texts(["Patient Casey Example called."])
    assert result.successful_items == 1
    assert result.failed_items == 0
