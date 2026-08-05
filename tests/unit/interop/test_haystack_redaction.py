from __future__ import annotations

import sys
from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest

from openmed.interop import adapter_spec, available_adapters, get_adapter
from openmed.interop import haystack as haystack_adapter


@dataclass
class FixtureDocument:
    content: str | None = None
    meta: dict[str, object] = field(default_factory=dict)
    id: str = "fixture-document"
    score: float | None = None

    def to_dict(self, *, flatten: bool = True) -> dict[str, object]:
        assert flatten is False
        return {
            "content": self.content,
            "meta": dict(self.meta),
            "id": self.id,
            "score": self.score,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> FixtureDocument:
        return cls(**payload)


class ComponentDecorator:
    def __call__(self, cls):
        cls.__haystack_component__ = True
        return cls

    @staticmethod
    def output_types(**output_types):
        def decorate(func):
            func.__haystack_output_types__ = output_types
            return func

        return decorate


@pytest.fixture(autouse=True)
def clear_component_class_cache():
    haystack_adapter._component_class.cache_clear()
    yield
    haystack_adapter._component_class.cache_clear()


def fake_haystack_import(name: str):
    assert name == "haystack"
    return SimpleNamespace(
        component=ComponentDecorator(),
        Document=FixtureDocument,
    )


def fake_deidentify(text: str, **kwargs):
    assert kwargs == {
        "method": "mask",
        "keep_mapping": False,
        "use_safety_sweep": True,
    }
    redacted = (
        text.replace("Jane Roe", "[PERSON]")
        .replace("jane.roe@example.com", "[EMAIL]")
        .replace("555-0100", "[PHONE]")
    )
    return SimpleNamespace(deidentified_text=redacted)


def test_registry_loads_haystack_adapter_without_importing_haystack() -> None:
    for name in list(sys.modules):
        if name == "haystack" or name.startswith("haystack."):
            sys.modules.pop(name, None)

    adapter = get_adapter("haystack")

    assert adapter is haystack_adapter
    assert "haystack" in available_adapters()
    assert adapter_spec("haystack").extra == "haystack"
    assert not any(
        name == "haystack" or name.startswith("haystack.") for name in sys.modules
    )


def test_component_redacts_fixture_documents_without_a_model_server(
    monkeypatch,
) -> None:
    monkeypatch.setattr(haystack_adapter, "_import_module", fake_haystack_import)
    monkeypatch.setattr(haystack_adapter, "_deidentify", fake_deidentify)
    original = FixtureDocument(
        content="Patient Jane Roe called jane.roe@example.com or 555-0100.",
        meta={"source": "synthetic-fixture"},
        score=0.91,
    )

    redactor_type = haystack_adapter.OpenMedRedactor
    result = redactor_type().run(documents=[original])
    processed = result["documents"][0]

    assert redactor_type.__haystack_component__ is True
    assert redactor_type.run.__haystack_output_types__ == {
        "documents": list[FixtureDocument]
    }
    assert processed.content == "Patient [PERSON] called [EMAIL] or [PHONE]."
    assert processed.meta == {"source": "synthetic-fixture"}
    assert processed.id == "fixture-document"
    assert processed.score == pytest.approx(0.91)
    assert original.content == (
        "Patient Jane Roe called jane.roe@example.com or 555-0100."
    )


def test_component_preserves_documents_without_text(monkeypatch) -> None:
    monkeypatch.setattr(haystack_adapter, "_import_module", fake_haystack_import)

    result = haystack_adapter.OpenMedRedactor().run(
        documents=[FixtureDocument(content=None, meta={"kind": "blob"})]
    )

    assert result["documents"][0].content is None
    assert result["documents"][0].meta == {"kind": "blob"}


def test_component_raises_clear_error_without_haystack_extra(monkeypatch) -> None:
    def missing_dependency(name: str):
        raise ImportError(name)

    monkeypatch.setattr(haystack_adapter, "_import_module", missing_dependency)

    with pytest.raises(ImportError, match=r"openmed\[haystack\]"):
        haystack_adapter.OpenMedRedactor


def test_real_haystack_pipeline_when_extra_is_installed(monkeypatch) -> None:
    haystack = pytest.importorskip("haystack")
    monkeypatch.setattr(haystack_adapter, "_deidentify", fake_deidentify)
    original = haystack.Document(
        content="Patient Jane Roe called 555-0100.",
        meta={"source": "synthetic-fixture"},
    )
    pipeline = haystack.Pipeline()
    pipeline.add_component("redactor", haystack_adapter.OpenMedRedactor())

    result = pipeline.run({"redactor": {"documents": [original]}})
    processed = result["redactor"]["documents"][0]

    assert processed.content == "Patient [PERSON] called [PHONE]."
    assert processed.meta == {"source": "synthetic-fixture"}
    assert original.content == "Patient Jane Roe called 555-0100."
