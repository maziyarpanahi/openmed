"""Shadow results and failures stay separate from the active response."""

import json
import threading
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from openmed.core.pii import DeidentificationResult, PIIEntity
from openmed.eval.shadow_mode import ShadowRunner
from openmed.processing.outputs import PredictionResult


def extraction(text, start=0, end=3):
    return PredictionResult(
        text,
        [PIIEntity(text[start:end], "PERSON", 1.0, start, end)],
        "synthetic",
        "time",
    )


def test_actual_result_shapes_are_preserved_and_aggregates_are_detached(capsys, caplog):
    text = "private-sentinel"
    prediction = extraction(text)
    redaction = DeidentificationResult(
        text, "[PERSON]", prediction.entities, "mask", datetime.now(timezone.utc)
    )
    calls = []

    def active(source, *, audit):
        assert source == text and audit
        return redaction

    def candidate(source):
        calls.append(source == text)
        return redaction

    runner = ShadowRunner(
        SimpleNamespace(extract_pii=lambda _: prediction, deidentify=active),
        SimpleNamespace(
            extract_pii=lambda _: extraction(text, 1, 4), deidentify=candidate
        ),
    )
    original = json.dumps(prediction.to_dict(), sort_keys=True)
    assert runner.extract_pii(text) is prediction
    assert original == json.dumps(prediction.to_dict(), sort_keys=True)
    assert runner.deidentify(text, audit=True) is redaction
    assert calls == [True]
    counts = runner.snapshot()
    assert counts["compared"] == 2
    assert counts["by_canonical_label"]["PERSON"] == {
        "exact": 1,
        "overlap": 1,
        "active_only": 0,
        "candidate_only": 0,
    }
    counts["by_canonical_label"].clear()
    assert "PERSON" in runner.snapshot()["by_canonical_label"]
    stdout, stderr = capsys.readouterr()
    assert text not in stdout + stderr + caplog.text + json.dumps(counts)


@pytest.mark.parametrize(
    ("error", "counter"),
    [
        (RuntimeError, "candidate_errors"),
        (TimeoutError, "timeouts"),
        (SystemExit, "candidate_errors"),
    ],
)
def test_candidate_errors_are_value_free(error, counter, caplog, capsys):
    def fail(_):
        raise error("private-candidate-sentinel")

    result = extraction("private input")
    runner = ShadowRunner(lambda _: result, fail)
    assert runner.extract_pii("private input") is result
    counts = runner.snapshot()
    assert counts[counter] == 1 and counts["compared"] == 0
    out, err = capsys.readouterr()
    assert "private" not in out + err + caplog.text + json.dumps(counts)


def test_timed_out_candidate_cannot_queue_or_publish_a_late_comparison(monkeypatch):
    from concurrent.futures import Future

    release = threading.Event()
    finished = threading.Event()
    result = extraction("private input")

    def delayed(_):
        release.wait()
        return result

    original = Future.set_result

    def settled(self, value):
        original(self, value)
        finished.set()

    monkeypatch.setattr(Future, "set_result", settled)
    runner = ShadowRunner(lambda _: result, delayed, timeout=0.001)
    try:
        assert runner.extract_pii("private input") is result
        assert runner.extract_pii("private input") is result
        assert runner.snapshot()["timeouts"] == runner.snapshot()["busy"] == 1
    finally:
        release.set()
    assert finished.wait(5)
    assert runner.snapshot()["compared"] == 0
