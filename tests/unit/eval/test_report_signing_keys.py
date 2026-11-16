"""Evaluation reports are signed only with caller-supplied keys."""

from __future__ import annotations

from pathlib import Path

import pytest

from openmed.eval import harness
from openmed.eval.harness import (
    BenchmarkFixture,
    run_federated_leakage_eval,
    run_relation_suite,
)
from openmed.eval.metrics import EvalSpan

RELATION_ENV = "OPENMED_RELATION_SCORECARD_KEY"
FEDERATED_ENV = "OPENMED_FEDERATED_EVAL_KEY"


def _gold_runner(fixture, _model, _device):
    return fixture.gold_relations


def _relation_suite(tmp_path: Path, **kwargs):
    return run_relation_suite(
        model_name="synthetic-relation-model",
        output_json=tmp_path / "scorecard.json",
        generated_at="2026-11-07T00:00:00Z",
        ci_resamples=5,
        ci_seed=3,
        **kwargs,
    )


def _federated_fixture() -> BenchmarkFixture:
    text = "Patient Jane Roe arrived for follow up."
    start = text.index("Jane Roe")
    return BenchmarkFixture(
        fixture_id="synthetic-1",
        text=text,
        gold_spans=(
            EvalSpan(start=start, end=start + len("Jane Roe"), label="PERSON"),
        ),
    )


def _write_detector(tmp_path: Path) -> Path:
    detector = tmp_path / "detector.py"
    detector.write_text(
        """
import json
import os
from pathlib import Path

Path(os.environ["OPENMED_DETECTOR_OUTPUT"]).write_text(json.dumps({"spans": []}))
""".lstrip(),
        encoding="utf-8",
    )
    return detector


def _forbid_detector_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*_args, **_kwargs):
        raise AssertionError("the detector must not start without a signing key")

    monkeypatch.setattr(harness, "_run_federated_fixture", fail)


def test_relation_suite_without_a_key_fails_before_running(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(RELATION_ENV, raising=False)
    calls: list[str] = []

    def runner(fixture, _model, _device):
        calls.append(fixture.fixture_id)
        return fixture.gold_relations

    with pytest.raises(ValueError, match=RELATION_ENV):
        _relation_suite(tmp_path, runner=runner)

    assert calls == []
    assert not (tmp_path / "scorecard.json").exists()


def test_relation_suite_rejects_an_empty_environment_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(RELATION_ENV, "")

    with pytest.raises(ValueError, match="no built-in signing key"):
        _relation_suite(tmp_path, runner=_gold_runner)


def test_relation_suite_signs_with_the_environment_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(RELATION_ENV, "synthetic-env-relation-key")

    scorecard = _relation_suite(tmp_path, runner=_gold_runner)

    assert scorecard.verify("synthetic-env-relation-key")
    assert not scorecard.verify("openmed-relation-scorecard-local-key")


def test_relation_suite_prefers_the_explicit_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(RELATION_ENV, "synthetic-env-relation-key")

    scorecard = _relation_suite(
        tmp_path, runner=_gold_runner, signing_key=b"synthetic-explicit-key"
    )

    assert scorecard.verify(b"synthetic-explicit-key")
    assert not scorecard.verify("synthetic-env-relation-key")


def test_federated_eval_without_a_key_fails_before_the_detector_starts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(FEDERATED_ENV, raising=False)
    _forbid_detector_runs(monkeypatch)

    with pytest.raises(ValueError, match=FEDERATED_ENV):
        run_federated_leakage_eval(
            [_federated_fixture()], detector=_write_detector(tmp_path)
        )


def test_federated_eval_rejects_an_empty_explicit_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _forbid_detector_runs(monkeypatch)

    with pytest.raises(ValueError, match="non-empty"):
        run_federated_leakage_eval(
            [_federated_fixture()],
            detector=_write_detector(tmp_path),
            signing_key="",
        )


def test_federated_eval_signs_with_the_environment_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(FEDERATED_ENV, "synthetic-env-federated-key")

    report = run_federated_leakage_eval(
        [_federated_fixture()],
        detector=_write_detector(tmp_path),
        generated_at="2026-11-07T00:00:00+00:00",
    )

    assert report.fixture_count == 1
    assert report.verify("synthetic-env-federated-key")
    assert not report.verify("openmed-federated-eval-local-key")
