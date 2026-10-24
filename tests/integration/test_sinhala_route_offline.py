"""Offline end-to-end Sinhala route over the synthetic ``si`` golden fixture.

Automatic routing selects the experimental ``si`` user-supplied-model route
for every synthetic note, the deterministic safety net masks each gold span
with an empty model detector, and ``method="replace"`` swaps every NIC for a
structurally valid synthetic NIC that the route recognizes again. Network
access is forbidden for the whole test.
"""

from __future__ import annotations

import json
import socket
import warnings
from pathlib import Path

import pytest

import openmed
from openmed.core.pii import deidentify
from openmed.core.pii_i18n import validate_sri_lanka_nic
from openmed.core.pipeline import Pipeline
from openmed.core.safety_sweep import safety_sweep
from openmed.processing.outputs import PredictionResult

pytestmark = pytest.mark.integration

FIXTURE_PATH = (
    Path(__file__).resolve().parents[2] / "openmed/eval/golden/fixtures/i18n/si.jsonl"
)
_MODEL = "local-user-supplied-sinhala-model"


def _empty_prediction(text, **kwargs):
    return PredictionResult(
        text=text, entities=[], model_name="synthetic", timestamp="2026-01-01"
    )


@pytest.fixture(autouse=True)
def _offline(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args, **kwargs):
        raise AssertionError("the Sinhala route must not open network connections")

    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(openmed, "analyze_text", _empty_prediction)


def _rows() -> list[dict]:
    return [
        json.loads(line)
        for line in FIXTURE_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_auto_routed_sinhala_notes_are_masked_without_a_model_entity() -> None:
    pipeline = Pipeline(
        lang="auto",
        model_name=_MODEL,
        model_detector=_empty_prediction,
    )

    for row in _rows():
        result = pipeline.run(row["text"])

        assert result.route.lang == "si"
        assert result.route.model_name == _MODEL
        assert result.route.metadata["available_default_model"] is None
        for span in row["gold_spans"]:
            assert span["text"] not in result.redacted_text, (row["id"], span["label"])


def test_replaced_nics_are_valid_synthetic_nics_in_the_source_layout() -> None:
    for row in _rows():
        text = row["text"]
        source_nics = [
            span["text"] for span in row["gold_spans"] if span["label"] == "ID_NUM"
        ]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            replaced = deidentify(
                text,
                lang="si",
                model_name=_MODEL,
                method="replace",
                consistent=True,
                seed=3890,
            ).deidentified_text

        for span in row["gold_spans"]:
            assert span["text"] not in replaced, (row["id"], span["label"])
        surrogate_nics = [
            entity.text
            for entity in safety_sweep(replaced, [], lang="si")
            if entity.label == "national_id"
        ]
        assert sorted(map(len, surrogate_nics)) == sorted(map(len, source_nics))
        assert all(validate_sri_lanka_nic(value) for value in surrogate_nics)
        assert set(surrogate_nics).isdisjoint(source_nics)
