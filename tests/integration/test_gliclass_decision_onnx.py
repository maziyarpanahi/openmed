"""Opt-in GLiClass v4 edge decision checks on CPU (q8 ONNX and torch).

The tests download ``knowledgator/gliclass-v4-edge`` at its pinned revision:
the int8 ``onnx/model_q8.onnx`` graph (about 650 MB) and, for the agreement
check, the torch weights (about 880 MB). They never download the fp32 ONNX
graph. Set ``OPENMED_RUN_GLICLASS_V4_TEST=1`` to run them.
"""

from __future__ import annotations

import math
import os
import socket
from typing import Any

import pytest
from jsonschema import Draft202012Validator

from openmed.structured.decision import (
    DecisionRequest,
    DecisionState,
    decision_result_schema,
)
from openmed.structured.decision_gliclass import (
    GLiClassDecisionBackend,
    decide_with_gliclass,
)

pytestmark = [pytest.mark.integration, pytest.mark.slow]

_RUN_ENV = "OPENMED_RUN_GLICLASS_V4_TEST"
# Synthetic, non-identifying fixtures with one unambiguous option each.
FIXTURES = (
    (
        "fixed_choice",
        "Synthetic message: I was charged twice for my subscription this month.",
        ("hardware issue", "billing question", "password reset"),
        None,
        "billing question",
    ),
    (
        "boolean_choice",
        "The synthetic review says the food was cold and the waiter was rude.",
        ("positive", "negative"),
        "What is the sentiment of the review?",
        "negative",
    ),
    (
        "boolean_choice",
        "The synthetic review says the staff were friendly and the soup was delicious.",
        ("positive", "negative"),
        "What is the sentiment of the review?",
        "positive",
    ),
    (
        "boolean_choice",
        "Synthetic schedule: the meeting moved from Monday to Friday afternoon.",
        ("yes", "no"),
        "Was the meeting rescheduled?",
        "yes",
    ),
    (
        "fixed_choice",
        "Synthetic weather report: heavy rain and strong winds expected all day.",
        ("sunny", "rainy", "snowy"),
        "What is the weather forecast?",
        "rainy",
    ),
    (
        "fixed_choice",
        "Synthetic sports news: the home team won the football final by two goals.",
        ("sports", "politics", "technology"),
        None,
        "sports",
    ),
    (
        "fixed_choice",
        "Synthetic referral note: the follow-up visit is a routine annual check.",
        ("urgent", "routine"),
        "How urgent is the follow-up?",
        "routine",
    ),
    (
        "fixed_choice",
        "Synthetic intake note: the reported allergy is to penicillin.",
        ("medication allergy", "food allergy", "no allergy"),
        "Which allergy is documented?",
        "medication allergy",
    ),
)


def _require_opt_in() -> None:
    if os.environ.get(_RUN_ENV) != "1":
        pytest.skip(f"set {_RUN_ENV}=1 to download and run gliclass-v4-edge")


def _block_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def blocked(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("network access attempted after download")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)


def _decide(backend: GLiClassDecisionBackend, fixture: tuple[Any, ...]) -> Any:
    mode, text, options, prompt, _ = fixture
    return decide_with_gliclass(
        DecisionRequest(
            mode=mode,
            input_text=text,
            options=options,
            timeout_ms=30000,
        ),
        backend=backend.with_instructions(prompt),
    )


@pytest.fixture(scope="module")
def onnx_backend() -> GLiClassDecisionBackend:
    _require_opt_in()
    pytest.importorskip("onnxruntime", reason="install openmed[onnx-runtime]")
    pytest.importorskip("tokenizers", reason="install openmed[onnx-runtime]")
    pytest.importorskip("huggingface_hub", reason="install openmed[onnx-runtime]")
    return GLiClassDecisionBackend.from_pretrained(runtime="onnx")


def test_q8_onnx_decides_synthetic_fixtures_on_cpu(
    onnx_backend: GLiClassDecisionBackend,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _block_network(monkeypatch)
    validator = Draft202012Validator(decision_result_schema())

    for fixture in FIXTURES:
        result = _decide(onnx_backend, fixture)
        metadata = result.extensions["gliclass"]

        assert result.state in {DecisionState.SUCCESS, DecisionState.ABSTAINED}
        assert result.ranking[0] == fixture[4]
        assert math.fsum(item["score"] for item in result.option_scores) == (
            pytest.approx(1.0, abs=1e-6)
        )
        assert metadata["runtime"] == "onnxruntime"
        assert metadata["classification_type"] == "single-label"
        assert metadata["initial_scores_available"] is False
        assert (
            result.backend["backend_id"] == "openmed.gliclass.v4.onnxruntime.model_q8"
        )
        validator.validate(result.to_dict())


def test_torch_and_q8_onnx_agree_on_the_top_option(
    onnx_backend: GLiClassDecisionBackend,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("gliclass", reason="install openmed[gliclass]")
    torch_backend = GLiClassDecisionBackend.from_pretrained(runtime="torch")
    _block_network(monkeypatch)
    # The private pipeline handle checks that the backend reproduces the
    # public gliclass API (``score`` and ``initial_score``) on the same model.
    from gliclass import ZeroShotClassificationPipeline

    pipe = torch_backend.runner._pipe  # type: ignore[attr-defined]
    public = ZeroShotClassificationPipeline(
        pipe.model,
        pipe.tokenizer,
        classification_type="single-label",
        max_length=pipe.max_length,
        device="cpu",
        progress_bar=False,
    )

    for fixture in FIXTURES:
        torch_result = _decide(torch_backend, fixture)
        onnx_result = _decide(onnx_backend, fixture)
        metadata = torch_result.extensions["gliclass"]

        assert torch_result.ranking[0] == onnx_result.ranking[0] == fixture[4]
        assert metadata["initial_scores_available"] is True
        top = public(
            fixture[1],
            list(fixture[2]),
            prompt=fixture[3],
            classification_type="single-label",
        )[0][0]
        index = fixture[2].index(top["label"])
        assert top["label"] == torch_result.ranking[0]
        assert top["score"] == pytest.approx(metadata["scores"][index], rel=1e-4)
        assert top["initial_score"] == pytest.approx(
            metadata["initial_scores"][index], rel=1e-4, abs=1e-9
        )
