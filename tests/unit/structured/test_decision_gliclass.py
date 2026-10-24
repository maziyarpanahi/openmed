"""GLiClass v4 decision backend tests with mocked pipelines and sessions."""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import math
import socket
import subprocess
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from jsonschema import Draft202012Validator

from openmed.core.capabilities import MissingOptionalDependencyError
from openmed.ner.exceptions import MissingDependencyError
from openmed.structured.decision import (
    DecisionAccessPolicy,
    DecisionBackendKind,
    DecisionError,
    DecisionMode,
    DecisionRequest,
    DecisionState,
    EncoderDecisionBackend,
    decide,
    decision_result_schema,
)
from openmed.structured.decision_gliclass import (
    GLICLASS_ONNX_Q8_FILE,
    GLICLASS_V4_EDGE_MODEL_ID,
    GLICLASS_V4_EDGE_REVISION,
    GLiClassDecisionBackend,
    GLiClassExample,
    GLiClassLogits,
    GLiClassOnnxRunner,
    GLiClassRunner,
    GLiClassTorchRunner,
    decide_with_gliclass,
    format_gliclass_input,
)

ROOT = Path(__file__).resolve().parents[3]
TEXT = (
    "Synthetic note: checkout errors rose after the deploy; rollback takes two minutes."
)
LABELS = ("roll back the build", "wait an hour", "page the team lead")
PROMPT = "What should the on-call engineer do right now?"
EXAMPLE = GLiClassExample(
    text="Synthetic: disk full alert on a test host.",
    labels=("wait an hour",),
)
# Logits and scores recorded from gliclass 0.1.21 running
# knowledgator/gliclass-v4-edge on TEXT, LABELS and PROMPT. ``score`` uses the
# calibrated logits and ``initial_score`` the uncalibrated ones.
CALIBRATED_LOGITS = (2.0625, -3.453125, -2.5)
UNCALIBRATED_LOGITS = (8.375, -7.625, -6.125)
SINGLE_LABEL_SCORES = (0.9857467412948608, 0.003966065123677254, 0.010287190787494183)
SINGLE_LABEL_INITIAL = (
    0.9999994039535522,
    1.1253511189579513e-07,
    5.04347383412096e-07,
)
MULTI_LABEL_SCORES = (0.8872045874595642, 0.030675802379846573, 0.07585817575454712)
MULTI_LABEL_INITIAL = (0.9997695088386536, 0.000487857119878754, 0.002182716503739357)
# gliclass 0.1.21 ``prepare_input`` output for TEXT, LABELS, PROMPT and EXAMPLE.
REFERENCE_INPUT = (
    "<<LABEL>>roll back the build<<LABEL>>wait an hour<<LABEL>>page the team lead"
    "<<SEP>>What should the on-call engineer do right now?" + TEXT + "<<EXAMPLE>>"
    "Synthetic: disk full alert on a test host. \nLabels:\n wait an hour<<SEP>>"
)
LABEL_TOKEN_ID = 128001


class _FakeModel:
    def __init__(
        self,
        logits: tuple[float, ...],
        uncalibrated: tuple[float, ...] | None,
    ) -> None:
        self.logits = logits
        self.uncalibrated = uncalibrated
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(kwargs)
        return SimpleNamespace(
            logits=[list(self.logits)],
            uncalibrated_logits=(
                None if self.uncalibrated is None else [list(self.uncalibrated)]
            ),
        )


class _FakeUniPipe:
    """Mimics gliclass ``UniEncoderZeroShotClassificationPipeline``."""

    def __init__(
        self,
        logits: tuple[float, ...] = CALIBRATED_LOGITS,
        uncalibrated: tuple[float, ...] | None = UNCALIBRATED_LOGITS,
        *,
        token_count: int = 44,
        max_length: int = 2048,
    ) -> None:
        self.model = _FakeModel(logits, uncalibrated)
        self.max_length = max_length
        self.token_count = token_count
        self.calls: list[dict[str, Any]] = []

    def prepare_inputs(
        self,
        texts: list[str],
        labels: list[str],
        same_labels: bool = False,
        examples: Any = None,
        prompt: Any = None,
    ) -> dict[str, Any]:
        self.calls.append(
            {
                "texts": texts,
                "labels": labels,
                "same_labels": same_labels,
                "examples": examples,
                "prompt": prompt,
            }
        )
        return {
            "input_ids": [[1] * self.token_count],
            "attention_mask": [[1] * self.token_count],
        }


class _FakePipeline:
    """Mimics ``gliclass.ZeroShotClassificationPipeline`` and its ``pipe``."""

    def __init__(self, pipe: _FakeUniPipe) -> None:
        self.pipe = pipe


class _FakeTokenizer:
    def __init__(self, *, filler: int = 10) -> None:
        self.encoded: list[str] = []
        self.disabled: list[str] = []
        self.filler = filler

    def no_truncation(self) -> None:
        self.disabled.append("truncation")

    def no_padding(self) -> None:
        self.disabled.append("padding")

    def token_to_id(self, token: str) -> int | None:
        return LABEL_TOKEN_ID if token == "<<LABEL>>" else None

    def encode(self, text: str) -> SimpleNamespace:
        self.encoded.append(text)
        ids = [1] + [LABEL_TOKEN_ID] * text.count("<<LABEL>>") + [5] * self.filler
        ids.append(2)
        return SimpleNamespace(ids=ids, attention_mask=[1] * len(ids))


class _FakeSession:
    def __init__(
        self,
        logits: tuple[float, ...] = CALIBRATED_LOGITS,
        *,
        inputs: tuple[str, ...] = ("input_ids", "attention_mask"),
        outputs: tuple[str, ...] = ("logits",),
    ) -> None:
        self.logits = logits
        self._inputs = [SimpleNamespace(name=name) for name in inputs]
        self._outputs = [SimpleNamespace(name=name) for name in outputs]
        self.runs: list[tuple[list[str], dict[str, Any]]] = []

    def get_inputs(self) -> list[SimpleNamespace]:
        return self._inputs

    def get_outputs(self) -> list[SimpleNamespace]:
        return self._outputs

    def run(self, names: list[str], feeds: dict[str, Any]) -> list[Any]:
        np = pytest.importorskip("numpy")
        self.runs.append((names, feeds))
        return [np.asarray([self.logits], dtype=np.float32)]


def _torch_backend(
    pipe: _FakeUniPipe | None = None, **kwargs: Any
) -> GLiClassDecisionBackend:
    runner = GLiClassTorchRunner(_FakePipeline(pipe or _FakeUniPipe()))
    return GLiClassDecisionBackend(runner, **kwargs)


def _onnx_backend(
    session: _FakeSession | None = None,
    tokenizer: _FakeTokenizer | None = None,
    **kwargs: Any,
) -> GLiClassDecisionBackend:
    runner = GLiClassOnnxRunner(
        session or _FakeSession(),
        tokenizer or _FakeTokenizer(),
        variant="model_q8",
    )
    return GLiClassDecisionBackend(runner, **kwargs)


def _request(
    mode: DecisionMode = DecisionMode.FIXED_CHOICE,
    options: tuple[str, ...] = LABELS,
    **kwargs: Any,
) -> DecisionRequest:
    return DecisionRequest(mode=mode, input_text=TEXT, options=options, **kwargs)


def _scores(result: Any) -> list[float]:
    return [item["score"] for item in result.option_scores]


def test_input_layout_matches_gliclass_reference() -> None:
    assert (
        format_gliclass_input(TEXT, LABELS, prompt=PROMPT, examples=(EXAMPLE,))
        == REFERENCE_INPUT
    )
    head = "<<LABEL>>yes<<LABEL>>no<<SEP>>"
    assert format_gliclass_input("t", ("yes", "no")) == head + "t"
    assert format_gliclass_input("t", ("yes", "no"), prompt_first=False) == "t" + head


def test_torch_runner_sends_labels_prompt_and_examples_to_the_pipeline() -> None:
    pipe = _FakeUniPipe()
    backend = _torch_backend(pipe, prompt=PROMPT, examples=(EXAMPLE,))

    result = decide_with_gliclass(_request(), backend=backend)

    assert result.state is DecisionState.SUCCESS
    assert pipe.calls == [
        {
            "texts": [TEXT],
            "labels": list(LABELS),
            "same_labels": True,
            "examples": [{"text": EXAMPLE.text, "labels": ["wait an hour"]}],
            "prompt": PROMPT,
        }
    ]
    assert pipe.model.calls[0]["max_num_classes"] == 3
    assert pipe.model.calls[0]["input_ids"] == [[1] * 44]


@pytest.mark.parametrize(
    "mode",
    [DecisionMode.FIXED_CHOICE, DecisionMode.ORDERED_PREFERENCE],
)
def test_single_label_modes_use_calibrated_softmax_and_keep_initial_scores(
    mode: DecisionMode,
) -> None:
    result = decide_with_gliclass(_request(mode), backend=_torch_backend())
    metadata = result.extensions["gliclass"]

    assert result.state is DecisionState.SUCCESS
    assert result.choice == "roll back the build"
    assert result.ranking == (
        "roll back the build",
        "page the team lead",
        "wait an hour",
    )
    assert _scores(result) == pytest.approx(SINGLE_LABEL_SCORES, rel=1e-6)
    assert result.confidence == pytest.approx(SINGLE_LABEL_SCORES[0], rel=1e-6)
    assert metadata["classification_type"] == "single-label"
    assert metadata["label_source"] == "options"
    assert metadata["scores"] == pytest.approx(SINGLE_LABEL_SCORES, rel=1e-6)
    assert metadata["initial_scores"] == pytest.approx(
        SINGLE_LABEL_INITIAL, rel=1e-5, abs=1e-12
    )
    assert metadata["initial_scores_available"] is True
    assert math.fsum(_scores(result)) == pytest.approx(1.0)


def test_boolean_choice_scores_the_two_caller_options_as_one_question() -> None:
    pipe = _FakeUniPipe(logits=(-1.5, 2.5), uncalibrated=(-4.0, 4.0))
    backend = _torch_backend(pipe, prompt="Does this need action within an hour?")

    result = decide_with_gliclass(
        _request(DecisionMode.BOOLEAN_CHOICE, options=("no", "yes")),
        backend=backend,
    )

    assert pipe.calls[0]["labels"] == ["no", "yes"]
    assert result.state is DecisionState.SUCCESS
    assert result.choice == "yes"
    expected_yes = 1.0 / (1.0 + math.exp(-4.0))
    assert _scores(result) == pytest.approx([1.0 - expected_yes, expected_yes])
    assert result.extensions["gliclass"]["classification_type"] == "single-label"


def test_multi_label_uses_independent_calibrated_sigmoid_scores() -> None:
    result = decide_with_gliclass(
        _request(DecisionMode.MULTI_LABEL),
        backend=_torch_backend(),
    )
    metadata = result.extensions["gliclass"]

    assert result.state is DecisionState.SUCCESS
    assert result.choices == ("roll back the build",)
    assert _scores(result) == pytest.approx(MULTI_LABEL_SCORES, rel=1e-6)
    assert metadata["classification_type"] == "multi-label"
    assert metadata["initial_scores"] == pytest.approx(MULTI_LABEL_INITIAL, rel=1e-6)


def test_scalar_score_uses_ordered_levels_and_expected_level() -> None:
    pipe = _FakeUniPipe(logits=(-2.0, 0.0, 3.0), uncalibrated=(0.0, 0.0, 0.0))
    backend = _torch_backend(
        pipe,
        prompt="Rate the urgency of the synthetic incident.",
        score_levels=("low", "medium", "high"),
    )

    result = decide_with_gliclass(
        _request(DecisionMode.SCALAR_SCORE, options=()),
        backend=backend,
    )
    metadata = result.extensions["gliclass"]
    exps = [math.exp(-5.0), math.exp(-3.0), 1.0]
    probabilities = [value / sum(exps) for value in exps]

    assert pipe.calls[0]["labels"] == ["low", "medium", "high"]
    assert result.state is DecisionState.SUCCESS
    assert result.option_scores == ()
    assert result.scalar_score == pytest.approx(
        (probabilities[1] + 2 * probabilities[2]) / 2
    )
    assert result.confidence == pytest.approx(probabilities[2])
    assert metadata["label_source"] == "score_levels"
    assert metadata["scores"] == pytest.approx(probabilities)
    assert metadata["initial_scalar_score"] == pytest.approx(0.5)


def test_default_score_levels_cover_scalar_requests() -> None:
    pipe = _FakeUniPipe(logits=(0.0, 0.0, 0.0, 0.0, 9.0), uncalibrated=None)

    result = decide_with_gliclass(
        _request(DecisionMode.SCALAR_SCORE, options=()),
        backend=_torch_backend(pipe),
    )

    assert pipe.calls[0]["labels"] == [
        "very low",
        "low",
        "moderate",
        "high",
        "very high",
    ]
    assert result.state is DecisionState.SUCCESS
    assert result.scalar_score == pytest.approx(1.0, abs=1e-3)


def test_calibrated_scores_drive_abstention_and_initial_scores_are_kept() -> None:
    hesitant = _FakeUniPipe(logits=(0.2, 0.0), uncalibrated=(6.0, -6.0))
    abstained = decide_with_gliclass(
        _request(DecisionMode.BOOLEAN_CHOICE, options=("yes", "no")),
        backend=_torch_backend(hesitant),
    )
    initial = abstained.extensions["gliclass"]["initial_scores"]

    assert abstained.state is DecisionState.ABSTAINED
    assert abstained.code == "low_confidence"
    assert abstained.choice is None
    assert max(_scores(abstained)) < 0.7
    assert initial[0] > 0.99

    confident = _FakeUniPipe(logits=(6.0, -6.0), uncalibrated=(0.2, 0.0))
    success = decide_with_gliclass(
        _request(DecisionMode.BOOLEAN_CHOICE, options=("yes", "no")),
        backend=_torch_backend(confident),
    )

    assert success.state is DecisionState.SUCCESS
    assert success.choice == "yes"
    assert success.extensions["gliclass"]["initial_scores"][0] < 0.6


def test_instructions_are_backend_settings_that_never_leak(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG)
    pipe = _FakeUniPipe()
    backend = _torch_backend(pipe, prompt=PROMPT, examples=(EXAMPLE,))
    other_prompt = "Which synthetic action is cheapest?"
    other = backend.with_instructions(other_prompt)

    first = decide_with_gliclass(_request(), backend=backend)
    second = decide_with_gliclass(_request(), backend=other)

    assert other.runner is backend.runner
    assert other.identity == backend.identity
    assert other.example_count == 1
    assert [call["prompt"] for call in pipe.calls] == [PROMPT, other_prompt]
    assert backend.prompt_digest == (
        "sha256:" + hashlib.sha256(PROMPT.encode("utf-8")).hexdigest()
    )
    assert first.extensions["gliclass"]["prompt_digest"] == backend.prompt_digest
    assert second.extensions["gliclass"]["prompt_digest"] == other.prompt_digest
    serialized = json.dumps([first.to_dict(), second.to_dict()])
    for secret in (TEXT, PROMPT, other_prompt, EXAMPLE.text):
        assert secret not in serialized
        assert secret not in caplog.text
        assert secret not in repr(backend)
        assert secret not in repr(EXAMPLE)
    assert first.extensions["gliclass"]["example_count"] == 1
    assert backend.with_instructions(None).prompt_digest is None


def test_onnx_runner_formats_tokenizes_and_runs_the_q8_graph() -> None:
    np = pytest.importorskip("numpy")
    session = _FakeSession()
    tokenizer = _FakeTokenizer()
    backend = _onnx_backend(session, tokenizer, prompt=PROMPT, examples=(EXAMPLE,))

    result = decide_with_gliclass(_request(), backend=backend)
    names, feeds = session.runs[0]
    metadata = result.extensions["gliclass"]

    assert tokenizer.disabled == ["truncation", "padding"]
    assert tokenizer.encoded == [REFERENCE_INPUT]
    assert names == ["logits"]
    assert set(feeds) == {"input_ids", "attention_mask"}
    assert feeds["input_ids"].dtype == np.int64
    assert feeds["attention_mask"].dtype == np.int64
    assert feeds["input_ids"].shape == (1, 15)
    assert result.state is DecisionState.SUCCESS
    assert _scores(result) == pytest.approx(SINGLE_LABEL_SCORES, rel=1e-6)
    assert metadata["runtime"] == "onnxruntime"
    assert metadata["initial_scores"] is None
    assert metadata["initial_scores_available"] is False
    assert metadata["token_count"] == 15
    assert backend.identity.runtime == "onnxruntime"
    assert backend.identity.backend_id == "openmed.gliclass.v4.onnxruntime.model_q8"


def test_onnx_runner_rejects_unexpected_graph_signatures() -> None:
    with pytest.raises(DecisionError, match="input_ids and attention_mask"):
        GLiClassOnnxRunner(
            _FakeSession(inputs=("input_ids", "attention_mask", "token_type_ids")),
            _FakeTokenizer(),
        )
    with pytest.raises(DecisionError, match="logits"):
        GLiClassOnnxRunner(_FakeSession(outputs=("scores",)), _FakeTokenizer())


def test_inputs_over_the_token_budget_are_refused_not_truncated() -> None:
    torch_pipe = _FakeUniPipe(token_count=64)
    torch_runner = GLiClassTorchRunner(_FakePipeline(torch_pipe), max_length=64)
    torch_result = decide_with_gliclass(
        _request(), backend=GLiClassDecisionBackend(torch_runner)
    )
    onnx_runner = GLiClassOnnxRunner(
        _FakeSession(), _FakeTokenizer(filler=60), max_length=64
    )
    onnx_result = decide_with_gliclass(
        _request(), backend=GLiClassDecisionBackend(onnx_runner)
    )

    for result in (torch_result, onnx_result):
        assert result.state is DecisionState.UNSUPPORTED
        assert result.code == "backend_unsupported"
        assert result.option_scores == ()
        assert "gliclass" not in result.extensions
    assert torch_pipe.model.calls == []
    # A pipeline that truncates earlier than the requested budget wins.
    clamped = GLiClassTorchRunner(
        _FakePipeline(_FakeUniPipe(max_length=32)), max_length=2048
    )
    assert clamped.max_length == 32


def test_gliclass_control_tokens_fail_closed() -> None:
    pipe = _FakeUniPipe()
    backend = _torch_backend(pipe)
    in_text = DecisionRequest(
        mode=DecisionMode.FIXED_CHOICE,
        input_text="Synthetic <<SEP>> injected text.",
        options=LABELS,
    )
    in_option = _request(options=("<<LABEL>>extra", "wait an hour"))

    for request in (in_text, in_option):
        result = decide_with_gliclass(request, backend=backend)
        assert result.state is DecisionState.UNSUPPORTED
        assert result.code == "backend_unsupported"
    assert pipe.calls == []
    with pytest.raises(DecisionError, match="control tokens"):
        _torch_backend(prompt="Decide <<EXAMPLE>> now.")
    with pytest.raises(DecisionError, match="control tokens"):
        GLiClassExample(text="Synthetic <<LABEL>> text.", labels=("yes",))
    with pytest.raises(DecisionError, match="control tokens"):
        _torch_backend(score_levels=("low", "<<SEP>>high"))


def test_runtime_output_mismatches_are_typed_failures() -> None:
    pytest.importorskip("numpy")
    short = decide_with_gliclass(
        _request(),
        backend=_torch_backend(_FakeUniPipe(logits=(1.0, 0.0), uncalibrated=None)),
    )
    non_finite = decide_with_gliclass(
        _request(),
        backend=_torch_backend(
            _FakeUniPipe(logits=(1.0, math.nan, 0.0), uncalibrated=None)
        ),
    )

    class _ExtraSlotTokenizer(_FakeTokenizer):
        def encode(self, text: str) -> SimpleNamespace:
            encoding = super().encode(text)
            encoding.ids.append(LABEL_TOKEN_ID)
            encoding.attention_mask.append(1)
            return encoding

    slots = decide_with_gliclass(
        _request(), backend=_onnx_backend(tokenizer=_ExtraSlotTokenizer())
    )

    for result in (short, non_finite, slots):
        assert result.state is DecisionState.CONFLICT
        assert result.code == "backend_conflict"
        assert result.choice is None


def test_failures_and_denials_do_not_attach_gliclass_metadata() -> None:
    class _BrokenRunner:
        runtime = "torch"

        def classify(self, *args: Any, **kwargs: Any) -> GLiClassLogits:
            raise RuntimeError("synthetic runtime failure")

    failed = decide_with_gliclass(
        _request(), backend=GLiClassDecisionBackend(_BrokenRunner())
    )
    denied = decide_with_gliclass(
        _request(namespace="restricted"), backend=_torch_backend()
    )

    assert failed.state is DecisionState.FAILURE
    assert failed.code == "backend_failure"
    assert denied.state is DecisionState.DENIED
    assert denied.code == "namespace_denied"
    assert "gliclass" not in failed.extensions
    assert "gliclass" not in denied.extensions


def test_backend_is_an_encoder_with_a_valid_result_contract() -> None:
    backend = _torch_backend()
    result = decide_with_gliclass(
        _request().to_dict(),
        backend=backend,
        policy=DecisionAccessPolicy(),
    )
    plain = decide(_request(), backend=backend)

    assert isinstance(backend, EncoderDecisionBackend)
    assert isinstance(backend.runner, GLiClassRunner)
    assert backend.identity.kind is DecisionBackendKind.ENCODER
    assert backend.identity.runtime == "torch"
    assert backend.identity.backend_id == "openmed.gliclass.v4.torch"
    assert backend.identity.model_id == GLICLASS_V4_EDGE_MODEL_ID
    assert backend.identity.revision == GLICLASS_V4_EDGE_REVISION
    assert backend.identity.license_id == "Apache-2.0"
    Draft202012Validator(decision_result_schema()).validate(result.to_dict())
    assert result.to_dict()["autonomous_action"] is False
    assert result.review["required"] is True
    assert plain.state is DecisionState.SUCCESS
    assert plain.option_scores == result.option_scores
    assert dict(plain.extensions) == {}
    with pytest.raises(TypeError):
        decide_with_gliclass(_request(), backend=object())  # type: ignore[arg-type]


def test_backend_settings_are_validated() -> None:
    with pytest.raises(DecisionError, match="score levels"):
        _torch_backend(score_levels=("only",))
    with pytest.raises(DecisionError, match="unique"):
        _torch_backend(score_levels=("low", "LOW"))
    with pytest.raises(DecisionError, match="examples require exactly"):
        _torch_backend(examples=({"text": "Synthetic.", "labels": ["a"], "x": 1},))
    with pytest.raises(DecisionError, match="permissive"):
        _torch_backend(license_id="GPL-3.0-only")
    with pytest.raises(DecisionError, match="instruction prompt"):
        _torch_backend(prompt="x" * 5000)
    with pytest.raises(TypeError, match="classify"):
        GLiClassDecisionBackend(object())  # type: ignore[arg-type]
    mapped = _torch_backend(examples=({"text": "Synthetic.", "labels": ["a"]},))
    assert mapped.example_count == 1


def _block_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def blocked(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)


def _fake_onnx_modules(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    created: dict[str, Any] = {}
    onnxruntime = types.ModuleType("onnxruntime")
    tokenizers = types.ModuleType("tokenizers")

    def inference_session(path: str, sess_options: Any, providers: list[str]) -> Any:
        created["session"] = (path, providers)
        return _FakeSession()

    class _Tokenizer:
        @staticmethod
        def from_file(path: str) -> _FakeTokenizer:
            created["tokenizer"] = path
            return _FakeTokenizer()

    onnxruntime.InferenceSession = inference_session  # type: ignore[attr-defined]
    tokenizers.Tokenizer = _Tokenizer  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "onnxruntime", onnxruntime)
    monkeypatch.setitem(sys.modules, "tokenizers", tokenizers)
    return created


def _fake_torch_modules(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    created: dict[str, Any] = {}
    torch = types.ModuleType("torch")
    torch.float32 = "float32"  # type: ignore[attr-defined]
    torch.no_grad = contextlib.nullcontext  # type: ignore[attr-defined]
    gliclass = types.ModuleType("gliclass")

    class _Model:
        @staticmethod
        def from_pretrained(path: str, dtype: Any = None) -> SimpleNamespace:
            created["model"] = (path, dtype)
            return SimpleNamespace(
                config=SimpleNamespace(architecture_type="uni-encoder")
            )

    class _Pipeline(_FakePipeline):
        def __init__(self, model: Any, tokenizer: str, **kwargs: Any) -> None:
            created["pipeline"] = (tokenizer, kwargs)
            super().__init__(_FakeUniPipe(max_length=kwargs["max_length"]))

    gliclass.GLiClassModel = _Model  # type: ignore[attr-defined]
    gliclass.ZeroShotClassificationPipeline = _Pipeline  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "gliclass", gliclass)
    return created


def test_local_checkpoints_load_and_score_without_network(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pytest.importorskip("numpy")
    (tmp_path / "onnx").mkdir()
    (tmp_path / GLICLASS_ONNX_Q8_FILE).write_bytes(b"synthetic graph placeholder")
    (tmp_path / "onnx" / "tokenizer.json").write_text("{}", encoding="utf-8")
    (tmp_path / "config.json").write_text(
        json.dumps({"architecture_type": "uni-encoder", "prompt_first": True}),
        encoding="utf-8",
    )
    onnx_created = _fake_onnx_modules(monkeypatch)
    torch_created = _fake_torch_modules(monkeypatch)
    _block_network(monkeypatch)

    onnx_backend = GLiClassDecisionBackend.from_pretrained(
        runtime="onnx", local_dir=tmp_path, prompt=PROMPT
    )
    torch_backend = GLiClassDecisionBackend.from_pretrained(
        runtime="torch", local_dir=tmp_path, max_length=512
    )
    results = [
        decide_with_gliclass(_request(), backend=backend)
        for backend in (onnx_backend, torch_backend)
    ]

    assert [result.state for result in results] == [DecisionState.SUCCESS] * 2
    assert onnx_created["session"] == (
        str(tmp_path / GLICLASS_ONNX_Q8_FILE),
        ["CPUExecutionProvider"],
    )
    assert onnx_created["tokenizer"] == str(tmp_path / "onnx" / "tokenizer.json")
    assert torch_created["model"] == (str(tmp_path), "float32")
    assert torch_created["pipeline"][0] == str(tmp_path)
    assert torch_created["pipeline"][1]["progress_bar"] is False
    for backend in (onnx_backend, torch_backend):
        identity = json.dumps(backend.identity.to_dict())
        assert str(tmp_path) not in identity
        assert backend.identity.model_id == GLICLASS_V4_EDGE_MODEL_ID
        assert backend.identity.revision == GLICLASS_V4_EDGE_REVISION


def test_hub_downloads_are_pinned_and_skip_the_fp32_graph(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pytest.importorskip("numpy")
    (tmp_path / "onnx").mkdir()
    (tmp_path / GLICLASS_ONNX_Q8_FILE).write_bytes(b"synthetic graph placeholder")
    (tmp_path / "onnx" / "tokenizer.json").write_text("{}", encoding="utf-8")
    _fake_onnx_modules(monkeypatch)
    calls: list[dict[str, Any]] = []
    hub = types.ModuleType("huggingface_hub")

    def snapshot_download(**kwargs: Any) -> str:
        calls.append(kwargs)
        return str(tmp_path)

    hub.snapshot_download = snapshot_download  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)

    backend = GLiClassDecisionBackend.from_pretrained(runtime="onnx")

    assert calls == [
        {
            "repo_id": GLICLASS_V4_EDGE_MODEL_ID,
            "revision": GLICLASS_V4_EDGE_REVISION,
            "cache_dir": None,
            "local_files_only": False,
            "allow_patterns": [
                GLICLASS_ONNX_Q8_FILE,
                "onnx/tokenizer.json",
                "config.json",
            ],
        }
    ]
    assert "onnx/model.onnx" not in calls[0]["allow_patterns"]
    assert backend.identity.backend_id.endswith(".model_q8")
    with pytest.raises(DecisionError, match="revision"):
        GLiClassDecisionBackend.from_pretrained("synthetic/other-gliclass")
    with pytest.raises(DecisionError, match="inside the checkpoint"):
        GLiClassOnnxRunner.from_pretrained(local_dir=tmp_path, onnx_file="../x.onnx")
    with pytest.raises(DecisionError, match="runtime"):
        GLiClassDecisionBackend.from_pretrained(runtime="tensorflow")


def test_missing_torch_extra_raises_install_hint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = types.ModuleType("torch")
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "gliclass", None)

    with pytest.raises(MissingDependencyError) as caught:
        GLiClassDecisionBackend.from_pretrained(runtime="torch")

    error = caught.value
    assert isinstance(error, ImportError)
    assert isinstance(error, MissingOptionalDependencyError)
    assert error.dependency == "gliclass"
    assert "pip install openmed[gliclass]" in str(error)
    assert "transformers>=5" in str(error)


@pytest.mark.parametrize("missing", ["onnxruntime", "tokenizers"])
def test_missing_onnx_extra_raises_install_hint(
    monkeypatch: pytest.MonkeyPatch, missing: str
) -> None:
    pytest.importorskip("numpy")
    _fake_onnx_modules(monkeypatch)
    monkeypatch.setitem(sys.modules, missing, None)

    with pytest.raises(MissingDependencyError) as caught:
        GLiClassOnnxRunner.from_pretrained()

    assert caught.value.dependency == missing
    assert "pip install openmed[onnx-runtime]" in str(caught.value)


def test_missing_hub_client_names_the_runtime_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fake_torch_modules(monkeypatch)
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)

    with pytest.raises(MissingDependencyError) as caught:
        GLiClassTorchRunner.from_pretrained()

    assert caught.value.dependency == "huggingface_hub"
    assert "pip install openmed[gliclass]" in str(caught.value)


def test_gliclass_extra_stays_out_of_the_base_install() -> None:
    try:
        import tomllib
    except ModuleNotFoundError:  # pragma: no cover - Python 3.10
        import tomli as tomllib

    with (ROOT / "pyproject.toml").open("rb") as handle:
        project = tomllib.load(handle)
    extra = project["project"]["optional-dependencies"]["gliclass"]
    base = " ".join(project["project"]["dependencies"])

    assert "gliclass>=0.1.21,<0.2" in extra
    assert "transformers>=5.0" in extra
    assert "scikit-learn>=1.0" in extra
    assert "torch>=2.13.0" in extra
    for name in ("gliclass", "transformers", "torch", "scikit-learn", "onnxruntime"):
        assert name not in base
    assert [{"extra": "gliclass"}, {"extra": "scispacy"}] in project["tool"]["uv"][
        "conflicts"
    ]
    probe = (
        "import sys\n"
        "import openmed.structured.decision\n"
        "before = set(sys.modules)\n"
        "import openmed.structured.decision_gliclass\n"
        "heavy = {'gliclass', 'onnxruntime', 'sklearn', 'tokenizers', 'torch',"
        " 'transformers', 'huggingface_hub'}\n"
        "print(sorted(heavy & {name.split('.')[0] for name in set(sys.modules) - before}))\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        check=True,
        cwd=ROOT,
        text=True,
    )
    assert completed.stdout.strip() == "[]"
