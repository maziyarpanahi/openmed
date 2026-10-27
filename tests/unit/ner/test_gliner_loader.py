from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

import pytest

from openmed.ner.exceptions import MissingDependencyError
from openmed.ner.families import gliner as gliner_module
from openmed.ner.families import gliner2 as gliner2_module


@pytest.fixture(autouse=True)
def reset_cache():
    gliner_module._availability_cache = None
    gliner_module.clear_gliner_cache()
    yield
    gliner_module._availability_cache = None
    gliner_module.clear_gliner_cache()


def test_load_gliner_handle_uses_from_pretrained(monkeypatch):
    fake_model = mock.MagicMock()

    class FakeGLiNER:
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            FakeGLiNER.calls.append((model_id, kwargs))
            return fake_model

    FakeGLiNER.calls = []

    fake_module = mock.MagicMock()
    fake_module.GLiNER = FakeGLiNER

    monkeypatch.setattr(gliner_module, "ensure_gliner_available", lambda: None)
    monkeypatch.setattr("importlib.import_module", lambda name: fake_module)

    handle = gliner_module.load_gliner_handle(
        "gliner-model", cache_dir="/tmp/cache", token="abc"
    )
    assert handle.model is fake_model
    assert FakeGLiNER.calls[0][0] == "gliner-model"
    assert FakeGLiNER.calls[0][1]["cache_dir"] == "/tmp/cache"
    assert FakeGLiNER.calls[0][1]["token"] == "abc"


def test_clear_gliner_cache(monkeypatch):
    fake_model = mock.MagicMock()

    class FakeGLiNER:
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            return fake_model

    fake_module = mock.MagicMock()
    fake_module.GLiNER = FakeGLiNER

    monkeypatch.setattr(gliner_module, "ensure_gliner_available", lambda: None)
    monkeypatch.setattr("importlib.import_module", lambda name: fake_module)

    first = gliner_module.load_gliner_handle("gliner-model")
    second = gliner_module.load_gliner_handle("gliner-model")
    assert first.model is second.model

    gliner_module.clear_gliner_cache()
    third = gliner_module.load_gliner_handle("gliner-model")
    assert third.model is fake_model


class NativeModelDouble:
    """Match native APIs while retaining state for cache/device assertions."""

    def __init__(self):
        self.device = "cpu"
        self.moves = []
        self.evaluations = 0
        self.calls = []

    def to(self, device):
        if device == "invalid":
            raise RuntimeError("invalid device")
        self.device = device
        self.moves.append(device)
        return self

    def eval(self):
        self.evaluations += 1
        return self

    def predict_entities(self, text, entities, *, threshold, flat_ner):
        self.calls.append((text, entities, threshold, flat_ner))
        return [
            {"text": "asthma", "start": 0, "end": 6, "label": "disease", "score": 0.9}
        ]

    def extract_entities(
        self,
        text,
        entity_types,
        *,
        threshold,
        include_spans,
        include_confidence,
        overlap_policy,
    ):
        self.calls.append(
            (
                text,
                entity_types,
                threshold,
                include_spans,
                include_confidence,
                overlap_policy,
            )
        )
        return {
            "entities": {
                "disease": [
                    {"text": "asthma", "start": 8, "end": 14, "confidence": 0.9},
                    {"text": "asthma", "start": 0, "end": 6, "confidence": 0.8},
                ]
            }
        }


@pytest.fixture(params=["gliformer", "gliner2"])
def native_loader(request, monkeypatch):
    family = request.param
    module = gliner_module if family == "gliformer" else gliner2_module
    clear = getattr(module, f"clear_{family}_cache")
    clear()
    monkeypatch.setattr(module, f"ensure_{family}_available", lambda: None)
    factory = mock.Mock(side_effect=lambda *_a, **_k: NativeModelDouble())
    snapshot = mock.Mock(return_value="synthetic-snapshot")
    factory.snapshot_download = snapshot
    class_name = "GLiFormer" if family == "gliformer" else "AutoExtractor"
    runtime = SimpleNamespace(**{class_name: SimpleNamespace(from_pretrained=factory)})
    original = module.importlib.import_module
    monkeypatch.setattr(
        module.importlib,
        "import_module",
        lambda name, *a, **kw: (
            runtime
            if name == family
            else SimpleNamespace(snapshot_download=snapshot)
            if name == "huggingface_hub"
            else original(name, *a, **kw)
        ),
    )
    yield family, getattr(module, f"load_{family}_handle"), factory
    clear()


def test_native_loader_uses_real_package_contract(native_loader):
    family, load, factory = native_loader
    handle = load(
        "synthetic/checkpoint",
        cache_dir="synthetic-cache",
        token="synthetic-token",
        device="cpu",
        local_files_only=True,
        revision="synthetic-revision",
    )
    kwargs = {
        "cache_dir": "synthetic-cache",
        "token": "synthetic-token",
        "local_files_only": True,
        "revision": "synthetic-revision",
    }
    if family == "gliformer":
        kwargs["load_tokenizer"] = True
        factory.assert_called_once_with("synthetic/checkpoint", **kwargs)
    else:
        factory.snapshot_download.assert_called_once_with(
            repo_id="synthetic/checkpoint",
            repo_type="model",
            **kwargs,
            allow_patterns=[
                "*.json",
                "*.safetensors",
                "*.bin",
                "*.model",
                "*.txt",
                "*.vocab",
                "*.merges",
            ],
        )
        factory.assert_called_once_with("synthetic-snapshot", local_files_only=True)
    assert handle.model.device == "cpu"
    assert handle.model.evaluations == 1


def test_native_cache_does_not_move_another_device(native_loader):
    _, load, factory = native_loader
    cpu = load("synthetic/checkpoint", device="cpu")
    cuda = load("synthetic/checkpoint", device="cuda:0")
    again = load("synthetic/checkpoint", device="cpu")
    assert cpu.model is again.model
    assert cpu.model is not cuda.model
    assert cpu.model.moves == ["cpu"]
    assert cuda.model.moves == ["cuda:0"]
    assert cpu.model.evaluations == 1
    assert factory.call_count == 2


@pytest.mark.parametrize(
    "changed",
    [
        {"revision": "another-revision"},
        {"local_files_only": True},
        {"cache_dir": "another-cache"},
        {"token": "another-synthetic-token"},
    ],
)
def test_native_cache_separates_loading_options(native_loader, changed):
    _, load, factory = native_loader
    first = load("synthetic/checkpoint")
    second = load("synthetic/checkpoint", **changed)
    assert first.model is not second.model
    assert factory.call_count == 2


def test_native_invalid_device_fails_visibly(native_loader):
    _, load, _ = native_loader
    with pytest.raises(RuntimeError, match="invalid device"):
        load("synthetic/checkpoint", device="invalid")


def test_native_predictor_forwards_labels_threshold_and_overlap(native_loader):
    family, load, _ = native_loader
    handle = load("synthetic/checkpoint")
    entities = handle.predict_entities(
        "asthma; asthma", ["disease"], threshold=0.75, flat_ner=False
    )
    if family == "gliformer":
        assert handle.model.calls == [("asthma; asthma", ["disease"], 0.75, False)]
    else:
        assert handle.model.calls == [
            ("asthma; asthma", ["disease"], 0.75, True, True, "allow")
        ]
        assert [(entity["start"], entity["score"]) for entity in entities] == [
            (0, 0.8),
            (8, 0.9),
        ]


@pytest.mark.parametrize(
    "module,family",
    [
        (gliner_module, "gliformer"),
        (gliner2_module, "gliner2"),
    ],
)
def test_native_missing_dependencies_explain_exact_extra(monkeypatch, module, family):
    monkeypatch.setattr(module, "is_backend_available", lambda _name: False)
    with pytest.raises(MissingDependencyError, match=rf"openmed\[{family}\]"):
        getattr(module, f"ensure_{family}_available")()


@pytest.mark.parametrize(
    "payload",
    [
        {"entities": {"disease": ["asthma"]}},
        {"entities": {"disease": [{"text": "asthma", "start": 0, "end": 6}]}},
        {"entities": []},
    ],
)
def test_gliner2_never_guesses_missing_spans_or_confidence(payload):
    handle = gliner2_module.GLiNER2Handle(
        model_id="synthetic/checkpoint",
        model=SimpleNamespace(extract_entities=lambda *_a, **_k: payload),
    )
    with pytest.raises(ValueError, match="GLiNER2"):
        handle.predict_entities("asthma; asthma", ["disease"])
