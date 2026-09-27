"""Offline acceptance tests for the local clinical NLI backend."""

from __future__ import annotations

import socket
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from openmed.clinical.nli import nli, verify
from openmed.clinical.nli_backends import (
    EncoderNLIBackend,
    LocalNLIError,
    RemoteNLIBackendError,
    resolve_nli_backend,
)
from openmed.clinical.nli_gate import NLIThresholds
from openmed.core.models import ModelLoader


class _FakeLogits:
    def __init__(self, values: tuple[float, float, float]):
        self.values = values

    def __getitem__(self, index: int) -> "_FakeLogits":
        assert index == 0
        return self

    def tolist(self) -> list[float]:
        return list(self.values)


class _FakeModel:
    def __init__(self, values: tuple[float, float, float]):
        self.values = values

    def __call__(self, **_encoded: object) -> SimpleNamespace:
        return SimpleNamespace(logits=_FakeLogits(self.values))


class _FakeLoader:
    def __init__(self, values: tuple[float, float, float]):
        self.values = values
        self.calls: list[tuple[str, str | None, str]] = []

    def load_local_sequence_classifier(
        self, model_ref: str, *, revision: str | None, runtime: str
    ) -> dict[str, object]:
        self.calls.append((model_ref, revision, runtime))
        return {
            "tokenizer": lambda *_args, **_kwargs: {"input_ids": [1]},
            "model": _FakeModel(self.values),
        }


class _FakeOnnxSession:
    def get_inputs(self) -> list[SimpleNamespace]:
        return [SimpleNamespace(name="input_ids")]

    def run(
        self, _outputs: object, feeds: dict[str, object]
    ) -> list[list[_FakeLogits]]:
        assert set(feeds) == {"input_ids"}
        return [[_FakeLogits((5.0, 0.0, 0.0))]]


def _backend(
    values: tuple[float, float, float],
) -> tuple[EncoderNLIBackend, _FakeLoader]:
    loader = _FakeLoader(values)
    backend = EncoderNLIBackend(
        "synthetic-local-checkpoint",
        revision="a" * 40,
        label_mapping={"0": "entailment", "1": "neutral", "2": "contradiction"},
        thresholds=NLIThresholds(entailment=0.9, contradiction=0.9, margin=0.05),
        loader=loader,
    )
    return backend, loader


@pytest.mark.parametrize(
    ("logits", "expected"),
    [
        ((5.0, 0.0, 0.0), "entailment"),
        ((0.0, 0.0, 5.0), "contradiction"),
        ((0.0, 0.0, 0.0), "abstention"),
    ],
)
def test_fake_checkpoint_produces_four_state_decisions_without_text(
    logits: tuple[float, float, float], expected: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_network(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("NLI attempted network access")

    monkeypatch.setattr(socket, "create_connection", fail_network)
    backend, loader = _backend(logits)
    result = nli("synthetic source 555-0101", "synthetic claim", backend=backend)
    assert result["label"] == expected
    assert 0 <= result["score"] <= 1
    assert result["backend_id"] == "local-encoder"
    assert "555-0101" not in str(result)
    assert loader.calls == [("synthetic-local-checkpoint", "a" * 40, "torch")]


def test_verify_returns_value_free_metadata_and_prechecks_before_model() -> None:
    backend, loader = _backend((5.0, 0.0, 0.0))
    results = verify(
        [{"text": "synthetic claim", "numeric": {"value": 11, "unit": "mg/dL"}}],
        {"text": "synthetic source", "numeric": {"value": 10, "unit": "mg/dL"}},
        backend=backend,
    )
    assert results == [
        {
            "claim_index": 0,
            "label": "contradiction",
            "score": 1.0,
            "backend_id": "precheck",
            "contradicted": True,
            "review_required": False,
        }
    ]
    assert loader.calls == []


def test_fake_onnx_checkpoint_uses_the_same_label_and_abstention_contract() -> None:
    class OnnxLoader(_FakeLoader):
        def load_local_sequence_classifier(
            self, model_ref: str, *, revision: str | None, runtime: str
        ) -> dict[str, object]:
            self.calls.append((model_ref, revision, runtime))
            return {
                "tokenizer": lambda *_args, **_kwargs: {"input_ids": [1]},
                "model": _FakeOnnxSession(),
            }

    loader = OnnxLoader((0.0, 0.0, 0.0))
    backend = EncoderNLIBackend(
        "synthetic-local-onnx",
        revision="a" * 40,
        runtime="onnx",
        label_mapping={"0": "entailment", "1": "neutral", "2": "contradiction"},
        thresholds=NLIThresholds(),
        loader=loader,
    )
    assert (
        backend.predict("synthetic source", "synthetic claim")["label"] == "entailment"
    )
    assert loader.calls == [("synthetic-local-onnx", "a" * 40, "onnx")]


def test_unreleased_default_fails_closed_and_remote_selection_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "openmed.clinical.nli_backends.get_default_nli_model", lambda: None
    )
    with pytest.raises(LocalNLIError, match="no released local NLI checkpoint"):
        nli("synthetic source", "synthetic claim")
    with pytest.raises(RemoteNLIBackendError):
        resolve_nli_backend("https://example.test/model")
    with pytest.raises(RemoteNLIBackendError):
        resolve_nli_backend("openai")


def test_verify_local_alias_uses_pinned_fake_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_loader = _FakeLoader((0.0, 0.0, 5.0))
    info = SimpleNamespace(
        model_id="OpenMed/Synthetic-NLI",
        category="Clinical NLI",
        task="text-classification",
        released="2026-09-27",
        license="Apache-2.0",
        provenance={
            "revision": "b" * 40,
            "nli_label_mapping": {
                "0": "entailment",
                "1": "neutral",
                "2": "contradiction",
            },
            "nli_calibration": NLIThresholds().to_dict(),
        },
        formats=["pytorch"],
    )
    monkeypatch.setattr(
        "openmed.clinical.nli_backends.get_default_nli_model",
        lambda: info.model_id,
    )
    monkeypatch.setattr(
        "openmed.clinical.nli_backends.get_model_info", lambda _name: info
    )
    monkeypatch.setattr("openmed.core.models.ModelLoader", lambda: fake_loader)

    result = verify(["synthetic claim"], "synthetic source", backend="local")
    assert result[0]["label"] == "contradiction"
    assert result[0]["contradicted"] is True
    assert result[0]["backend_id"] == "local-encoder"
    assert fake_loader.calls == [(info.model_id, "b" * 40, "torch")]


def test_loader_failure_does_not_echo_sensitive_exception() -> None:
    class FailingLoader:
        def load_local_sequence_classifier(self, *_args: object, **_kwargs: object):
            raise RuntimeError("synthetic identifier 555-0199")

    backend = EncoderNLIBackend(
        "synthetic-local-checkpoint",
        revision="a" * 40,
        label_mapping={"0": "entailment", "1": "neutral", "2": "contradiction"},
        thresholds=NLIThresholds(),
        loader=FailingLoader(),
    )
    with pytest.raises(LocalNLIError) as error:
        backend.predict("synthetic source", "synthetic claim")
    assert "555-0199" not in str(error.value)


def test_loader_encloses_local_resolution_in_offline_guard(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    loader = ModelLoader.__new__(ModelLoader)
    loader.config = SimpleNamespace(cache_dir=tmp_path)
    events: list[bool] = []

    @contextmanager
    def guard(_config: object, *, local_only: bool):
        events.append(local_only)
        yield

    monkeypatch.setattr("openmed.core.models.network_blocked_if_offline", guard)
    monkeypatch.setattr(loader, "_as_existing_local_path", lambda _name: None)
    monkeypatch.setattr(loader, "_resolve_model_name", lambda name: name)

    def stop_before_loading(*_args: object, **_kwargs: object) -> str:
        raise LocalNLIError("synthetic stop")

    monkeypatch.setattr(loader, "_prepare_model_reference", stop_before_loading)
    with pytest.raises(LocalNLIError, match="synthetic stop"):
        loader.load_local_sequence_classifier(
            "OpenMed/Synthetic-NLI", revision="a" * 40
        )
    assert events == [True]
