"""Python reference checks for the synthetic WebGPU runtime fixture."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

FIXTURE = Path(__file__).with_name("fixtures") / (
    "webgpu_token_classification_golden.json"
)


def test_webgpu_fixture_matches_python_float32_head_and_spans() -> None:
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    head = payload["classification_head"]
    tokens = payload["tokens"]
    batch_size = tokens["batch_size"]
    sequence_length = tokens["sequence_length"]
    hidden_size = head["hidden_size"]
    label_count = head["label_count"]
    hidden_states = payload["hidden_states"]
    weights = head["weights"]
    bias = head["bias"]

    logits: list[float] = []
    for batch in range(batch_size):
        for token in range(sequence_length):
            row = batch * sequence_length + token
            for label in range(label_count):
                value = float(bias[label])
                for hidden in range(hidden_size):
                    value += float(hidden_states[row * hidden_size + hidden]) * float(
                        weights[hidden * label_count + label]
                    )
                logits.append(value)

    assert logits == pytest.approx(payload["reference_logits"], abs=1e-12)
    predicted = [
        max(
            range(label_count),
            key=lambda label: logits[token * label_count + label],
        )
        for token in range(sequence_length)
    ]
    labels = [payload["id2label"][str(label)] for label in predicted]
    assert labels == ["O", "B-PERSON", "I-PERSON", "O"]
    assert payload["reference_token_spans"] == [
        {
            "batch_index": 0,
            "label": "PERSON",
            "start_token": 1,
            "end_token": 3,
        }
    ]
    assert payload["reference_runtime"] == "python-float32"
    assert payload["note"].startswith("Synthetic ")
