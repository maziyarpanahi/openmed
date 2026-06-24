from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from openmed.interop import adapter_spec, available_adapters, get_adapter
from openmed.interop import langchain as langchain_adapter


class RunnableLambdaLike:
    def __init__(self, func, name=None):
        self.func = func
        self.name = name

    def invoke(self, value, config=None, **kwargs):
        return self.func(value, config=config, **kwargs)

    def __or__(self, other):
        return ChainLike(self, other)


class ChainLike:
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def invoke(self, value):
        redacted = self.left.invoke(value)
        return self.right.invoke(redacted)


class CaptureModel:
    def __init__(self):
        self.seen = None

    def invoke(self, value):
        self.seen = value
        return "model-result"


@dataclass
class DocumentLike:
    page_content: str
    metadata: dict[str, str]

    def copy(self, update=None):
        values = {
            "page_content": self.page_content,
            "metadata": dict(self.metadata),
        }
        values.update(update or {})
        return DocumentLike(**values)


def fake_deidentify(text: str, **kwargs):
    assert kwargs["method"] == "mask"
    redacted = (
        text.replace("Jane Roe", "[PERSON]")
        .replace("jane.roe@example.com", "[EMAIL]")
        .replace("555-0100", "[PHONE]")
    )
    return SimpleNamespace(deidentified_text=redacted)


def test_registry_loads_langchain_adapter_lazily():
    adapter = get_adapter("langchain")

    assert adapter is langchain_adapter
    assert "langchain" in available_adapters()
    assert adapter_spec("langchain").extra == "langchain"
    assert hasattr(adapter, "create_redaction_runnable")


def test_create_redaction_runnable_raises_clear_error_without_extra(monkeypatch):
    def missing_dependency(name: str):
        raise ImportError(name)

    monkeypatch.setattr(langchain_adapter, "_import_module", missing_dependency)

    with pytest.raises(ImportError, match=r"openmed\[langchain\]"):
        langchain_adapter.create_redaction_runnable(deidentifier=fake_deidentify)


def test_chain_style_flow_redacts_text_before_model(monkeypatch):
    monkeypatch.setattr(
        langchain_adapter,
        "_import_module",
        lambda name: SimpleNamespace(RunnableLambda=RunnableLambdaLike),
    )
    model = CaptureModel()
    runnable = langchain_adapter.create_redaction_runnable(
        deidentifier=fake_deidentify,
    )

    chain = runnable | model
    result = chain.invoke(
        "Patient Jane Roe can be reached at jane.roe@example.com or 555-0100."
    )

    assert result == "model-result"
    assert model.seen == "Patient [PERSON] can be reached at [EMAIL] or [PHONE]."
    assert "Jane Roe" not in model.seen
    assert "jane.roe@example.com" not in model.seen


def test_redaction_transform_handles_rag_document_payloads():
    transform = langchain_adapter.create_redaction_transform(
        input_key="context",
        deidentifier=fake_deidentify,
    )
    document = DocumentLike(
        page_content="Patient Jane Roe called from 555-0100.",
        metadata={"source": "fixture"},
    )

    payload = transform.invoke(
        {
            "context": [document],
            "question": "Summarize follow-up.",
        }
    )

    assert payload["context"][0].page_content == "Patient [PERSON] called from [PHONE]."
    assert payload["context"][0].metadata == {"source": "fixture"}
    assert payload["question"] == "Summarize follow-up."
    assert document.page_content == "Patient Jane Roe called from 555-0100."
