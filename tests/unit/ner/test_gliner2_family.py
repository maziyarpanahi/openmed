"""Offline tests for the GLiNER2.5 family built on ``gliner2.AutoExtractor``.

Every test uses a fake ``gliner2`` module and fake Hub resolution, so nothing
downloads weights or opens a socket.
"""

from __future__ import annotations

import importlib
import json
import logging
import os
import socket
import subprocess
import sys
import textwrap
import types
from pathlib import Path
from typing import Any

import pytest

from openmed.core import capabilities
from openmed.core.errors import ModelLoadError
from openmed.core.offline import OfflineModeError
from openmed.ner.exceptions import MissingDependencyError
from openmed.ner.families import gliner as gliner_module
from openmed.ner.families import gliner2 as gliner2_module
from openmed.ner.families.gliner2 import (
    GLINER2_CHECKPOINTS,
    GLiNER2Handle,
    GLiNER2OutputError,
    get_gliner2_checkpoint,
    list_gliner2_checkpoints,
    load_gliner2_handle,
)
from openmed.ner.indexing import load_index
from openmed.ner.infer import Entity, NerRequest, infer

COMMIT = "0123456789abcdef0123456789abcdef01234567"
MARKER = "Zephyrine Quillfeather"
TEXT = f"Synthetic patient {MARKER} was started on metformin 500 mg in Springfield."
REPO_ROOT = Path(__file__).resolve().parents[3]


def _span(text: str, surface: str, score: float, *, pad: int = 0) -> dict[str, Any]:
    start = text.index(surface)
    return {
        "text": surface,
        "confidence": score,
        "start": start - pad,
        "end": start + len(surface),
    }


class FakeExtractor:
    """Minimal stand-in for a ``gliner2`` span or boundary extractor."""

    architecture = "boundary"

    def __init__(
        self,
        *,
        entities: Any = None,
        classification_scores: dict[str, dict[str, float]] | None = None,
        relations: Any = None,
        records: Any = None,
        on_call: Any = None,
    ) -> None:
        self.entities = entities if entities is not None else {}
        self.classification_scores = classification_scores or {}
        self.relations = relations if relations is not None else {}
        self.records = records if records is not None else {}
        self.on_call = on_call
        self.calls: list[tuple[str, Any, dict[str, Any]]] = []

    def eval(self) -> "FakeExtractor":
        return self

    def _record(self, name: str, spec: Any, kwargs: dict[str, Any]) -> None:
        self.calls.append((name, spec, kwargs))
        if self.on_call is not None:
            self.on_call()

    def extract_entities(self, text: str, entity_types: Any, **kwargs: Any) -> Any:
        self._record("extract_entities", entity_types, kwargs)
        return {"entities": self.entities}

    def extract_entities_long(self, text: str, entity_types: Any, **kwargs: Any) -> Any:
        self._record("extract_entities_long", entity_types, kwargs)
        return {"entities": self.entities}

    def classify_text(self, text: str, tasks: Any, **kwargs: Any) -> Any:
        self._record("classify_text", tasks, kwargs)
        payload = {}
        for task, config in tasks.items():
            labels = list(config["labels"])
            scores = self.classification_scores.get(task, {})
            payload[task] = [
                {"label": label, "confidence": scores[label]}
                for label in labels
                if label in scores
            ]
        return payload

    def extract_relations(self, text: str, relation_types: Any, **kwargs: Any) -> Any:
        self._record("extract_relations", relation_types, kwargs)
        return {"relation_extraction": self.relations}

    def extract_relations_long(
        self, text: str, relation_types: Any, **kwargs: Any
    ) -> Any:
        self._record("extract_relations_long", relation_types, kwargs)
        return {"relation_extraction": self.relations}

    def extract_json(self, text: str, structures: Any, **kwargs: Any) -> Any:
        self._record("extract_json", structures, kwargs)
        return self.records

    def extract_json_long(self, text: str, structures: Any, **kwargs: Any) -> Any:
        self._record("extract_json_long", structures, kwargs)
        return self.records


def _handle(model: Any, model_id: str = "fastino/gliner2.5-small-v1") -> GLiNER2Handle:
    return GLiNER2Handle(
        model_id=model_id,
        model=model,
        revision=COMMIT,
        license_id="Apache-2.0",
        architecture="boundary",
    )


def _write_checkpoint(directory: Path, **overrides: Any) -> Path:
    (directory / "encoder_config").mkdir(parents=True)
    config = {"architecture": "boundary", "model_type": "gliner2"}
    config.update(overrides.get("config", {}))
    (directory / "config.json").write_text(json.dumps(config), encoding="utf-8")
    encoder = {"model_type": "deberta-v2"}
    encoder.update(overrides.get("encoder", {}))
    (directory / "encoder_config" / "config.json").write_text(
        json.dumps(encoder), encoding="utf-8"
    )
    (directory / "tokenizer.json").write_text("{}", encoding="utf-8")
    if overrides.get("weights", True):
        (directory / "model.safetensors").write_bytes(b"synthetic")
    return directory


class LocalEntryNotFoundError(FileNotFoundError):
    """Fake ``huggingface_hub`` cache-miss error."""


@pytest.fixture(autouse=True)
def reset_gliner2_state(monkeypatch):
    monkeypatch.delenv("OPENMED_OFFLINE", raising=False)
    gliner2_module._availability_cache = None
    gliner2_module.clear_gliner2_cache()
    yield
    gliner2_module._availability_cache = None
    gliner2_module.clear_gliner2_cache()


@pytest.fixture
def fake_runtime(monkeypatch, tmp_path):
    """Install fake ``gliner2`` and ``huggingface_hub`` modules."""

    snapshot = _write_checkpoint(
        tmp_path / "hub" / "models--fastino--gliner2.5-small-v1" / "snapshots" / COMMIT
    )
    state: dict[str, Any] = {
        "snapshot": snapshot,
        "cached": True,
        "hub_calls": [],
        "load_calls": [],
        "models": [],
    }

    def snapshot_download(**kwargs: Any) -> str:
        state["hub_calls"].append(kwargs)
        if kwargs["local_files_only"] and not state["cached"]:
            raise LocalEntryNotFoundError("not cached")
        return str(state["snapshot"])

    class AutoExtractor:
        @staticmethod
        def from_pretrained(path: str, **kwargs: Any) -> FakeExtractor:
            state["load_calls"].append((path, kwargs))
            model = FakeExtractor()
            state["models"].append(model)
            return model

    fake_gliner2 = types.ModuleType("gliner2")
    fake_gliner2.AutoExtractor = AutoExtractor
    fake_hub = types.ModuleType("huggingface_hub")
    fake_hub.snapshot_download = snapshot_download
    fake_errors = types.ModuleType("huggingface_hub.errors")
    fake_errors.LocalEntryNotFoundError = LocalEntryNotFoundError
    monkeypatch.setitem(sys.modules, "gliner2", fake_gliner2)
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)
    monkeypatch.setitem(sys.modules, "huggingface_hub.errors", fake_errors)
    monkeypatch.setattr(gliner2_module, "is_backend_available", lambda name: True)
    return state


@pytest.fixture
def socket_guard(monkeypatch):
    """Fail the test on any outbound socket use."""

    attempts: list[str] = []

    def blocked(*args: Any, **kwargs: Any) -> Any:
        attempts.append("socket")
        raise AssertionError("network access attempted during GLiNER2 inference")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setattr(socket, "getaddrinfo", blocked)
    return attempts


# ---------------------------------------------------------------------------
# Availability and install hints
# ---------------------------------------------------------------------------


def test_missing_gliner2_raises_install_hint(monkeypatch) -> None:
    monkeypatch.setattr(gliner2_module, "is_backend_available", lambda name: False)
    real_find_spec = capabilities.importlib.util.find_spec

    def fake_find_spec(name, package=None):
        if name == "gliner2":
            return None
        return real_find_spec(name, package)

    monkeypatch.setattr(capabilities.importlib.util, "find_spec", fake_find_spec)

    assert gliner2_module.is_gliner2_available(force_refresh=True) is False
    with pytest.raises(MissingDependencyError) as caught:
        gliner2_module.ensure_gliner2_available()
    assert caught.value.dependency == "gliner2"
    assert "pip install openmed[gliner2]" in str(caught.value)
    with pytest.raises(MissingDependencyError, match=r"openmed\[gliner2\]"):
        load_gliner2_handle("fastino/gliner2.5-small-v1")


def test_capability_registry_describes_gliner2_extra() -> None:
    spec = capabilities.backend_spec("gliner2")
    status = capabilities.backend_status("gliner2")

    assert spec.extra == "gliner2"
    assert spec.modules[0] == "gliner2"
    assert {"torch", "transformers", "safetensors", "huggingface_hub"} <= set(
        spec.modules
    )
    assert "openmed[gliner2]" in status.install_hint
    assert "gliner2[local]" in status.install_hint


def test_v1_gliner_family_is_unchanged(monkeypatch) -> None:
    assert capabilities.backend_spec("gliner").modules == ("gliner",)
    assert gliner_module._GLINER_HINT == "Install with `pip install openmed[gliner]`."
    monkeypatch.setattr(gliner2_module, "is_backend_available", lambda name: False)
    monkeypatch.setattr(gliner_module, "is_backend_available", lambda name: True)
    gliner_module._availability_cache = None
    try:
        assert gliner_module.is_gliner_available(force_refresh=True) is True
        assert gliner2_module.is_gliner2_available(force_refresh=True) is False
    finally:
        gliner_module._availability_cache = None


def test_availability_probe_never_imports_gliner2(monkeypatch) -> None:
    monkeypatch.delitem(sys.modules, "gliner2", raising=False)
    monkeypatch.setattr(
        capabilities.importlib.util, "find_spec", lambda name, package=None: object()
    )

    assert gliner2_module.is_gliner2_available(force_refresh=True) is True
    assert "gliner2" not in sys.modules


def test_import_openmed_does_not_import_gliner2(tmp_path) -> None:
    fake_package = tmp_path / "gliner2"
    fake_package.mkdir()
    (fake_package / "__init__.py").write_text(
        "raise RuntimeError('gliner2 must not be imported eagerly')\n",
        encoding="utf-8",
    )
    script = textwrap.dedent(
        """
        import sys
        import openmed
        import openmed.ner
        from openmed.ner.families import gliner2, gliner2_decision
        assert gliner2.is_gliner2_available(force_refresh=True) in (True, False)
        assert gliner2.list_gliner2_checkpoints()
        assert "gliner2" not in sys.modules, "gliner2 was imported"
        print("ok")
        """
    )
    environment = dict(os.environ)
    environment.pop("OPENMED_OFFLINE", None)
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(None, (str(tmp_path), str(REPO_ROOT), environment.get("PYTHONPATH")))
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=environment,
        timeout=120,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr[-2000:]
    assert completed.stdout.strip().endswith("ok")


# ---------------------------------------------------------------------------
# Capability metadata
# ---------------------------------------------------------------------------


def test_all_six_checkpoints_have_language_and_size_metadata() -> None:
    expected = {
        "fastino/gliner2.5-small-v1": ("74M", ("en",), False),
        "fastino/gliner2.5-base-v1": ("194M", ("en",), False),
        "fastino/gliner2.5-multi-v1": ("287M", ("multi",), False),
        "fastino/GLiNER2.5-multi-Decide": ("287M", ("multi",), True),
        "fastino/GLiNER2.5-Decide": ("486M", ("en",), True),
        "fastino/GLiNER2.5-Decide-1B": ("1.19B", ("en",), True),
    }

    specs = list_gliner2_checkpoints()
    assert [spec.model_id for spec in specs] == list(expected)
    for spec in specs:
        size, languages, decision_tuned = expected[spec.model_id]
        payload = spec.to_dict()
        assert payload["size"] == size
        assert payload["languages"] == list(languages)
        assert payload["decision_tuned"] is decision_tuned
        assert payload["license_id"] == "Apache-2.0"
        assert payload["requires_remote_code"] is False
        assert payload["architecture"] in {"span", "boundary"}
        assert set(payload["tasks"]) == {
            "entities",
            "classification",
            "relations",
            "json_extraction",
        }
    assert get_gliner2_checkpoint("FASTINO/GLINER2.5-SMALL-V1") is specs[0]
    assert get_gliner2_checkpoint("someone/other-model") is None


def test_builtin_index_lists_gliner2_checkpoints() -> None:
    records = {record.id: record for record in load_index().models}

    for model_id, spec in GLINER2_CHECKPOINTS.items():
        record = records[model_id]
        assert record.family == "gliner2"
        assert record.languages == spec.languages
        assert record.license == "Apache-2.0"
        assert spec.size_label in (record.notes or "")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def test_loader_uses_autoextractor_on_cached_snapshot(fake_runtime) -> None:
    handle = load_gliner2_handle(
        "fastino/gliner2.5-small-v1", cache_dir="/synthetic/cache", device="cpu"
    )

    assert len(fake_runtime["hub_calls"]) == 1
    call = fake_runtime["hub_calls"][0]
    assert call["local_files_only"] is True
    assert call["repo_id"] == "fastino/gliner2.5-small-v1"
    assert call["cache_dir"] == "/synthetic/cache"
    assert not any(pattern.endswith(".bin") for pattern in call["allow_patterns"])
    assert fake_runtime["load_calls"] == [
        (str(fake_runtime["snapshot"]), {"map_location": "cpu"})
    ]
    assert handle.model is fake_runtime["models"][0]
    assert handle.revision == COMMIT
    assert handle.license_id == "Apache-2.0"
    assert handle.architecture == "boundary"
    assert handle.metadata()["checkpoint"]["size"] == "74M"


def test_loader_downloads_only_when_snapshot_is_missing(fake_runtime) -> None:
    fake_runtime["cached"] = False

    load_gliner2_handle("fastino/gliner2.5-small-v1", revision="main")

    assert [call["local_files_only"] for call in fake_runtime["hub_calls"]] == [
        True,
        False,
    ]
    assert all(call["revision"] == "main" for call in fake_runtime["hub_calls"])


@pytest.mark.parametrize("use_env", [False, True])
def test_loader_offline_mode_never_downloads(
    fake_runtime, monkeypatch, use_env
) -> None:
    fake_runtime["cached"] = False
    if use_env:
        monkeypatch.setenv("OPENMED_OFFLINE", "1")
        kwargs: dict[str, Any] = {}
    else:
        kwargs = {"local_files_only": True}

    with pytest.raises(OfflineModeError):
        load_gliner2_handle("fastino/gliner2.5-small-v1", **kwargs)
    assert [call["local_files_only"] for call in fake_runtime["hub_calls"]] == [True]
    assert fake_runtime["load_calls"] == []


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"config": {"auto_map": {"AutoModel": "remote.Model"}}}, "remote code"),
        ({"encoder": {"auto_map": {"AutoModel": "remote.Model"}}}, "remote code"),
        ({"weights": False}, "safetensors"),
    ],
)
def test_loader_refuses_remote_code_and_pickle_weights(
    fake_runtime, tmp_path, overrides, message
) -> None:
    checkpoint = _write_checkpoint(tmp_path / "unsafe-checkpoint", **overrides)

    with pytest.raises(ModelLoadError, match=message) as caught:
        load_gliner2_handle(str(checkpoint), license_id="Apache-2.0")
    assert str(tmp_path) not in str(caught.value)
    assert fake_runtime["load_calls"] == []


def test_loader_reads_local_directories_without_hub_calls(
    fake_runtime, tmp_path
) -> None:
    checkpoint = _write_checkpoint(tmp_path / "private-site" / "my-gliner2")

    handle = load_gliner2_handle(str(checkpoint), license_id="Apache-2.0")

    assert fake_runtime["hub_calls"] == []
    assert fake_runtime["load_calls"][0][0] == str(checkpoint)
    assert handle.source == "local"
    assert handle.revision.startswith("local-config-sha256:")
    assert handle.public_model_id == "local/my-gliner2"
    assert str(tmp_path) not in json.dumps(handle.metadata())


def test_loader_caches_models_until_cleared(fake_runtime) -> None:
    first = load_gliner2_handle("fastino/gliner2.5-small-v1")
    second = load_gliner2_handle("fastino/gliner2.5-small-v1")
    assert first.model is second.model

    gliner2_module.clear_gliner2_cache()
    third = load_gliner2_handle("fastino/gliner2.5-small-v1")
    assert third.model is not first.model
    assert len(fake_runtime["load_calls"]) == 2


def test_handle_refuses_cloud_api_clients() -> None:
    cloud_client = type("GLiNER2API", (), {"__module__": "gliner2.api_client"})()

    with pytest.raises(ValueError, match="cloud API"):
        GLiNER2Handle(model_id="fastino/gliner2.5-small-v1", model=cloud_client)


# ---------------------------------------------------------------------------
# Entities
# ---------------------------------------------------------------------------


def test_extract_entities_returns_openmed_entities_with_offsets() -> None:
    person = _span(TEXT, MARKER, 0.93)
    padded = _span(TEXT, "metformin", 0.81, pad=1)
    padded["text"] = " metformin"
    model = FakeExtractor(
        entities={
            "medication": [padded, dict(padded, confidence=0.6)],
            "person": [person],
            "city": [],
        }
    )

    entities = _handle(model).extract_entities(
        TEXT, ["person", "medication", "city"], threshold=0.4
    )

    assert all(isinstance(entity, Entity) for entity in entities)
    assert [(entity.label, entity.text) for entity in entities] == [
        ("person", MARKER),
        ("medication", "metformin"),
    ]
    for entity in entities:
        assert TEXT[entity.start : entity.end] == entity.text
    assert entities[1].score == pytest.approx(0.81)
    name, labels, kwargs = model.calls[0]
    assert name == "extract_entities"
    assert labels == ["person", "medication", "city"]
    assert kwargs["include_spans"] is True
    assert kwargs["include_confidence"] is True
    assert kwargs["threshold"] == 0.4


def test_extract_entities_long_document_uses_absolute_offsets() -> None:
    long_text = ("Synthetic filler sentence. " * 200) + TEXT
    model = FakeExtractor(entities={"person": [_span(long_text, MARKER, 0.9)]})

    entities = _handle(model).extract_entities(
        long_text, {"person": "a patient name"}, chunk_size=128, chunk_overlap=16
    )

    name, labels, kwargs = model.calls[0]
    assert name == "extract_entities_long"
    assert labels == {"person": "a patient name"}
    assert kwargs["chunk_size"] == 128
    assert kwargs["chunk_overlap"] == 16
    assert entities[0].start == long_text.index(MARKER) > 5000
    assert long_text[entities[0].start : entities[0].end] == MARKER


def test_short_text_can_force_the_long_document_helper() -> None:
    model = FakeExtractor(entities={"person": [_span(TEXT, MARKER, 0.9)]})

    _handle(model).extract_entities(TEXT, ["person"], long_document=True)

    assert model.calls[0][0] == "extract_entities_long"


@pytest.mark.parametrize(
    "item",
    [
        {"text": MARKER, "confidence": 0.9, "start": 0, "end": 10_000},
        {"text": MARKER, "confidence": 0.9, "start": True, "end": 5},
        {"text": MARKER, "confidence": 0.9, "start": 0, "end": 7},
        {"text": MARKER, "confidence": 1.5, **{"start": 18, "end": 40}},
        {"text": MARKER, "confidence": float("nan"), "start": 18, "end": 40},
    ],
)
def test_malformed_entity_payloads_fail_without_echoing_text(item) -> None:
    model = FakeExtractor(entities={"person": [item]})

    with pytest.raises(GLiNER2OutputError) as caught:
        _handle(model).extract_entities(TEXT, ["person"])
    assert MARKER not in str(caught.value)
    assert MARKER not in repr(caught.value.args)


def test_unrequested_entity_labels_fail_closed() -> None:
    model = FakeExtractor(entities={"diagnosis": [_span(TEXT, MARKER, 0.9)]})

    with pytest.raises(GLiNER2OutputError, match="unrequested"):
        _handle(model).extract_entities(TEXT, ["person"])


def test_predict_entities_keeps_the_v1_dictionary_shape(monkeypatch) -> None:
    monkeypatch.setattr(gliner2_module, "is_backend_available", lambda name: True)
    model = FakeExtractor(entities={"person": [_span(TEXT, MARKER, 0.9)]})

    rows = _handle(model).predict_entities(TEXT, ["person"], flat_ner=False)

    assert rows == [
        {
            "text": MARKER,
            "start": TEXT.index(MARKER),
            "end": TEXT.index(MARKER) + len(MARKER),
            "label": "person",
            "score": 0.9,
        }
    ]
    assert model.calls[0][2]["overlap_policy"] == "nested"


# ---------------------------------------------------------------------------
# Classification, relations and JSON extraction
# ---------------------------------------------------------------------------


def test_classify_text_requests_and_returns_every_label_score() -> None:
    model = FakeExtractor(
        classification_scores={
            "priority": {"urgent": 0.2, "routine": 0.7, "deferred": 0.1},
            "topics": {"diabetes": 0.91, "oncology": 0.03, "renal": 0.55},
        }
    )

    result = _handle(model).classify_text(
        TEXT,
        {
            "priority": ["urgent", "routine", "deferred"],
            "topics": {
                "labels": ["diabetes", "oncology", "renal"],
                "multi_label": True,
                "cls_threshold": 0.5,
            },
        },
    )

    _, tasks, kwargs = model.calls[0]
    assert kwargs == {"include_confidence": True}
    assert tasks["priority"] == {
        "labels": ["urgent", "routine", "deferred"],
        "multi_label": True,
        "cls_threshold": 0.0,
        "class_act": "softmax",
    }
    assert tasks["topics"]["class_act"] == "sigmoid"
    assert tasks["topics"]["cls_threshold"] == 0.0
    priority = result["priority"]
    assert [item.label for item in priority.scores] == ["urgent", "routine", "deferred"]
    assert priority.selected == ("routine",)
    assert priority.score_for("urgent") == pytest.approx(0.2)
    topics = result["topics"]
    assert topics.multi_label is True
    assert topics.selected == ("diabetes", "renal")
    assert topics.score_for("oncology") == pytest.approx(0.03)


def test_classification_missing_label_scores_fail_closed() -> None:
    model = FakeExtractor(classification_scores={"priority": {"urgent": 0.9}})

    with pytest.raises(GLiNER2OutputError, match="every requested label"):
        _handle(model).classify_text(TEXT, {"priority": ["urgent", "routine"]})


def test_extract_relations_converts_heads_and_tails() -> None:
    head = _span(TEXT, MARKER, 0.95)
    tail = _span(TEXT, "metformin", 0.8)
    model = FakeExtractor(
        relations={
            "treated_with": [
                {"head": head, "tail": tail, "confidence": 0.77},
                {"head": head, "tail": _span(TEXT, "Springfield", 0.6)},
            ],
            "located_in": [],
        }
    )

    relations = _handle(model).extract_relations(TEXT, ["treated_with", "located_in"])

    assert [relation.tail.text for relation in relations] == [
        "metformin",
        "Springfield",
    ]
    first, second = relations
    assert first.score == pytest.approx(0.77)
    assert second.score == pytest.approx(0.6)
    assert first.head.group == "head" and first.tail.group == "tail"
    assert first.head.label == "treated_with"
    assert TEXT[first.tail.start : first.tail.end] == "metformin"
    assert model.calls[0][2]["include_spans"] is True


def test_extract_json_maps_fields_to_spans_and_choices() -> None:
    model = FakeExtractor(
        records={
            "prescription": [
                {
                    "drug": _span(TEXT, "metformin", 0.94),
                    "dose": _span(TEXT, "500 mg", 0.83),
                    "route": {
                        "text": "oral",
                        "confidence": 0.37,
                        "start": TEXT.index("metformin"),
                        "end": TEXT.index("metformin") + len("metformin"),
                    },
                    "side_effects": [],
                    "confidence": 0.94,
                }
            ]
        }
    )

    records = _handle(model).extract_json(
        TEXT,
        {
            "prescription": [
                "drug::str",
                "dose::str",
                "route::[oral|iv]::str",
                "side_effects::list",
            ]
        },
    )

    (record,) = records["prescription"]
    assert record.score == pytest.approx(0.94)
    assert record.fields["drug"][0].text == "metformin"
    assert record.fields["dose"][0].label == "dose"
    assert record.fields["side_effects"] == ()
    assert record.choices["route"][0].label == "oral"
    assert record.choices["route"][0].score == pytest.approx(0.37)
    assert "route" not in record.fields


# ---------------------------------------------------------------------------
# infer() routing, privacy and network posture
# ---------------------------------------------------------------------------


def test_infer_routes_builtin_gliner2_checkpoints(monkeypatch) -> None:
    infer_module = importlib.import_module("openmed.ner.infer")
    model = FakeExtractor(entities={"Drug": [_span(TEXT, "metformin", 0.88)]})
    seen: dict[str, Any] = {}

    def fake_loader(source: str, **kwargs: Any) -> GLiNER2Handle:
        seen["source"] = source
        seen.update(kwargs)
        return _handle(model, model_id=source)

    monkeypatch.setattr(infer_module, "ensure_gliner2_available", lambda: None)
    monkeypatch.setattr(infer_module, "load_gliner2_handle", fake_loader)

    response = infer(
        NerRequest(
            model_id="fastino/gliner2.5-small-v1",
            text=TEXT,
            labels=["Drug"],
            threshold=0.5,
        )
    )

    assert seen["source"] == "fastino/gliner2.5-small-v1"
    assert response.meta["family"] == "gliner2"
    assert [(entity.label, entity.text) for entity in response.entities] == [
        ("DRUG", "metformin")
    ]


def test_inference_makes_no_socket_calls(socket_guard) -> None:
    model = FakeExtractor(
        entities={"person": [_span(TEXT, MARKER, 0.9)]},
        classification_scores={"priority": {"urgent": 0.8, "routine": 0.2}},
        relations={"treated_with": []},
        records={"prescription": []},
    )
    handle = _handle(model)

    handle.extract_entities(TEXT, ["person"])
    handle.classify_text(TEXT, {"priority": ["urgent", "routine"]})
    handle.extract_relations(TEXT, ["treated_with"])
    handle.extract_json(TEXT, {"prescription": ["drug::str"]})

    assert socket_guard == []


def test_socket_guard_detects_network_use(socket_guard) -> None:
    def phone_home() -> None:
        socket.create_connection(("example.invalid", 443))

    model = FakeExtractor(entities={}, on_call=phone_home)

    with pytest.raises(AssertionError, match="network access"):
        _handle(model).extract_entities(TEXT, ["person"])
    assert socket_guard == ["socket"]


def test_offline_mode_guards_inference_sockets(monkeypatch) -> None:
    monkeypatch.setenv("OPENMED_OFFLINE", "1")
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        monkeypatch.setenv(name, "0")

    def phone_home() -> None:
        socket.create_connection(("example.invalid", 443))

    model = FakeExtractor(entities={}, on_call=phone_home)

    with pytest.raises(OfflineModeError):
        _handle(model).extract_entities(TEXT, ["person"])


def test_no_request_text_is_logged_or_retained(fake_runtime, caplog) -> None:
    caplog.set_level(logging.DEBUG)
    handle = load_gliner2_handle("fastino/gliner2.5-small-v1")
    handle.model.entities = {"person": [_span(TEXT, MARKER, 0.9)]}
    handle.model.classification_scores = {"flag": {"yes": 0.9, "no": 0.1}}

    handle.extract_entities(TEXT, ["person"])
    handle.classify_text(TEXT, {"flag": ["yes", "no"]})

    assert MARKER not in caplog.text
    assert MARKER not in json.dumps(handle.metadata())
    retained = {key: value for key, value in vars(handle).items() if key != "model"}
    assert MARKER not in repr(retained)
    assert MARKER not in repr(gliner2_module._load_model.cache_info())
