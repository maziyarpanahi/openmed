from __future__ import annotations

import asyncio
import pickle
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID

import pytest

from openmed.interop import adapter_spec, get_adapter
from openmed.interop import llamaindex as llamaindex_adapter


@dataclass
class FixtureNode:
    text: str
    metadata: dict[str, object]
    id_: str = "fixture-node"
    excluded_llm_metadata_keys: list[str] = field(default_factory=list)
    excluded_embed_metadata_keys: list[str] = field(default_factory=list)

    def set_content(self, text: str) -> None:
        self.text = text


@dataclass
class FixtureNodeWithScore:
    node: FixtureNode
    score: float


class BaseNodePostprocessorLike:
    callback_manager = None

    def postprocess_nodes(self, nodes, query_bundle=None, query_str=None):
        del query_str
        return self._postprocess_nodes(nodes, query_bundle=query_bundle)


class TransformComponentLike:
    async def acall(self, nodes, **kwargs):
        return self(nodes, **kwargs)


def fake_import(name: str):
    if name == "llama_index.core.postprocessor.types":
        return SimpleNamespace(BaseNodePostprocessor=BaseNodePostprocessorLike)
    if name == "llama_index.core.schema":
        return SimpleNamespace(TransformComponent=TransformComponentLike)
    raise ImportError(name)


def fake_deidentify(text: str, **kwargs):
    assert kwargs["method"] == "mask"
    assert kwargs["keep_year"] is False
    assert kwargs["use_safety_sweep"] is True
    redacted = (
        text.replace("Jane Roe", "[PERSON]")
        .replace("jane.roe@example.com", "[EMAIL]")
        .replace("555-0100", "[PHONE]")
    )
    return SimpleNamespace(deidentified_text=redacted)


def fake_mask_deidentify(text: str, **kwargs):
    del kwargs
    return SimpleNamespace(deidentified_text=f"MASKED:{text}")


def fake_remove_deidentify(text: str, **kwargs):
    del kwargs
    return SimpleNamespace(deidentified_text=f"REMOVED:{text}")


def test_registry_loads_llamaindex_redaction_adapter_lazily() -> None:
    for name in list(sys.modules):
        if name == "llama_index" or name.startswith("llama_index."):
            sys.modules.pop(name, None)

    adapter = get_adapter("llamaindex")

    assert adapter is llamaindex_adapter
    assert adapter_spec("llamaindex").extra == "llamaindex"
    assert hasattr(adapter, "create_redaction_postprocessor")
    assert not any(
        name == "llama_index" or name.startswith("llama_index.") for name in sys.modules
    )


def test_postprocessor_redacts_fixture_nodes_without_an_llm(monkeypatch) -> None:
    monkeypatch.setattr(llamaindex_adapter, "_import_module", fake_import)
    original = FixtureNodeWithScore(
        node=FixtureNode(
            text="Patient Jane Roe called jane.roe@example.com or 555-0100.",
            metadata={"source": "synthetic-fixture"},
        ),
        score=0.93,
    )

    postprocessor = llamaindex_adapter.create_redaction_postprocessor(
        deidentifier=fake_deidentify,
    )
    processed = postprocessor.postprocess_nodes([original])

    assert isinstance(postprocessor, BaseNodePostprocessorLike)
    assert processed[0].node.text == ("Patient [PERSON] called [EMAIL] or [PHONE].")
    assert processed[0].node.metadata == {"source": "synthetic-fixture"}
    assert processed[0].score == pytest.approx(0.93)
    assert original.node.text == (
        "Patient Jane Roe called jane.roe@example.com or 555-0100."
    )


def test_postprocessor_supports_async_retrieval(monkeypatch) -> None:
    monkeypatch.setattr(llamaindex_adapter, "_import_module", fake_import)
    postprocessor = llamaindex_adapter.create_redaction_postprocessor(
        deidentifier=fake_deidentify,
    )
    original = FixtureNodeWithScore(
        node=FixtureNode("Jane Roe", {}),
        score=0.7,
    )

    processed = asyncio.run(postprocessor.apostprocess_nodes([original]))

    assert processed[0].node.text == "[PERSON]"


def test_postprocessor_redacts_string_metadata_without_mutating_source(
    monkeypatch,
) -> None:
    monkeypatch.setattr(llamaindex_adapter, "_import_module", fake_import)
    original = FixtureNodeWithScore(
        node=FixtureNode(
            text="Jane Roe",
            metadata={
                "patient_name": "Jane Roe",
                "contacts": ["jane.roe@example.com", "555-0100"],
                "Jane Roe": "primary patient",
                "sequence": 7,
            },
        ),
        score=0.7,
    )

    postprocessor = llamaindex_adapter.create_redaction_postprocessor(
        deidentifier=fake_deidentify,
    )
    processed = postprocessor.postprocess_nodes([original])

    assert processed[0].node.metadata == {
        "patient_name": "[PERSON]",
        "contacts": ["[EMAIL]", "[PHONE]"],
        "[PERSON]": "primary patient",
        "sequence": 7,
    }
    assert original.node.metadata["patient_name"] == "Jane Roe"
    assert original.node.metadata["contacts"] == [
        "jane.roe@example.com",
        "555-0100",
    ]
    expected_exclusions = [
        "patient_name",
        "contacts",
        "[PERSON]",
        "sequence",
    ]
    assert processed[0].node.excluded_llm_metadata_keys == expected_exclusions
    assert processed[0].node.excluded_embed_metadata_keys == expected_exclusions
    assert original.node.excluded_llm_metadata_keys == []
    assert original.node.excluded_embed_metadata_keys == []


def test_postprocessor_rejects_unsupported_or_cyclic_metadata(monkeypatch) -> None:
    monkeypatch.setattr(llamaindex_adapter, "_import_module", fake_import)
    postprocessor = llamaindex_adapter.create_redaction_postprocessor(
        deidentifier=fake_deidentify,
    )
    unsupported = FixtureNodeWithScore(
        node=FixtureNode("Jane Roe", {"source": Path("Jane Roe.txt")}),
        score=0.7,
    )

    with pytest.raises(TypeError, match="metadata values must be"):
        postprocessor.postprocess_nodes([unsupported])

    cyclic_metadata: dict[str, object] = {}
    cyclic_metadata["self"] = cyclic_metadata
    cyclic = FixtureNodeWithScore(
        node=FixtureNode("Jane Roe", cyclic_metadata),
        score=0.7,
    )
    with pytest.raises(ValueError, match="metadata must not contain cycles"):
        postprocessor.postprocess_nodes([cyclic])


def test_real_llamaindex_content_excludes_metadata_when_extra_is_installed() -> None:
    schema = pytest.importorskip("llama_index.core.schema")
    original = schema.TextNode(
        text="Patient Jane Roe",
        metadata={"patient_name": "Jane Roe", "record_number": 1234567},
    )
    scored = schema.NodeWithScore(node=original, score=0.9)

    postprocessor = llamaindex_adapter.create_redaction_postprocessor(
        deidentifier=fake_deidentify,
    )
    processed = postprocessor.postprocess_nodes([scored])[0].node

    assert processed.get_content(metadata_mode=schema.MetadataMode.LLM) == (
        "Patient [PERSON]"
    )
    assert processed.get_content(metadata_mode=schema.MetadataMode.EMBED) == (
        "Patient [PERSON]"
    )
    assert processed.metadata == {
        "patient_name": "[PERSON]",
        "record_number": 1234567,
    }
    assert original.metadata == {
        "patient_name": "Jane Roe",
        "record_number": 1234567,
    }


def test_real_llamaindex_transform_is_picklable_when_extra_is_installed() -> None:
    schema = pytest.importorskip("llama_index.core.schema")
    transform = llamaindex_adapter.create_redaction_transform(
        deidentifier=fake_mask_deidentify,
    )

    restored = pickle.loads(pickle.dumps(transform))
    processed = restored([schema.TextNode(text="Jane Roe")])

    assert processed[0].text == "MASKED:Jane Roe"
    assert restored.to_dict() == transform.to_dict()


def test_real_llamaindex_parallel_ingestion_when_extra_is_installed() -> None:
    ingestion = pytest.importorskip("llama_index.core.ingestion")
    schema = pytest.importorskip("llama_index.core.schema")
    pipeline = ingestion.IngestionPipeline(
        transformations=[
            llamaindex_adapter.create_redaction_transform(
                deidentifier=fake_mask_deidentify,
            )
        ],
        disable_cache=True,
    )

    processed = pipeline.run(
        nodes=[schema.TextNode(text="Jane Roe")],
        num_workers=2,
    )

    assert processed[0].text == "MASKED:Jane Roe"


def test_real_ingestion_sanitizes_related_node_metadata() -> None:
    schema = pytest.importorskip("llama_index.core.schema")
    vector_utils = pytest.importorskip("llama_index.core.vector_stores.utils")
    original = schema.TextNode(
        text="Jane Roe",
        id_="patient-1234567",
        relationships={
            schema.NodeRelationship.SOURCE: schema.RelatedNodeInfo(
                node_id="document-1234567",
                metadata={"patient_name": "Jane Roe", "numeric_mrn": 1234567},
            )
        },
    )

    transformed = llamaindex_adapter.create_redaction_transform(
        deidentifier=fake_deidentify,
    )([original])[0]
    stored_metadata = vector_utils.node_to_metadata_dict(
        transformed,
        remove_text=True,
    )
    serialized = repr(stored_metadata)

    assert "Jane Roe" not in serialized
    assert "patient-1234567" not in serialized
    assert "document-1234567" not in serialized
    assert "1234567" not in serialized
    UUID(transformed.id_)
    UUID(transformed.source_node.node_id)
    assert transformed.source_node.metadata["patient_name"] == "[PERSON]"
    assert transformed.source_node.metadata["numeric_mrn"].startswith("openmed-")
    assert original.source_node.metadata["patient_name"] == "Jane Roe"
    assert original.source_node.metadata["numeric_mrn"] == 1234567


def test_real_llamaindex_cache_separates_configs_when_extra_is_installed() -> None:
    ingestion = pytest.importorskip("llama_index.core.ingestion")
    schema = pytest.importorskip("llama_index.core.schema")
    cache = ingestion.IngestionCache()
    original = schema.TextNode(text="Jane Roe")

    masked_pipeline = ingestion.IngestionPipeline(
        transformations=[
            llamaindex_adapter.create_redaction_transform(
                config=llamaindex_adapter.LlamaIndexRedactionConfig(method="mask"),
                deidentifier=fake_mask_deidentify,
            )
        ],
        cache=cache,
    )
    removed_pipeline = ingestion.IngestionPipeline(
        transformations=[
            llamaindex_adapter.create_redaction_transform(
                config=llamaindex_adapter.LlamaIndexRedactionConfig(method="remove"),
                deidentifier=fake_remove_deidentify,
            )
        ],
        cache=cache,
    )

    masked = masked_pipeline.run(nodes=[original])
    removed = removed_pipeline.run(nodes=[original])

    assert masked[0].text == "MASKED:Jane Roe"
    assert removed[0].text == "REMOVED:Jane Roe"
    assert masked_pipeline.transformations[0].to_dict() != (
        removed_pipeline.transformations[0].to_dict()
    )


def test_default_cache_identity_is_stable_and_config_specific(monkeypatch) -> None:
    monkeypatch.setattr(llamaindex_adapter, "_import_module", fake_import)
    mask_config = llamaindex_adapter.LlamaIndexRedactionConfig(method="mask")
    remove_config = llamaindex_adapter.LlamaIndexRedactionConfig(method="remove")

    first_mask = llamaindex_adapter.create_redaction_transform(config=mask_config)
    second_mask = llamaindex_adapter.create_redaction_transform(config=mask_config)
    remove = llamaindex_adapter.create_redaction_transform(config=remove_config)

    assert first_mask.to_dict() == second_mask.to_dict()
    assert first_mask.to_dict() != remove.to_dict()


def test_ingestion_transform_redacts_copies_before_storage(monkeypatch) -> None:
    monkeypatch.setattr(llamaindex_adapter, "_import_module", fake_import)
    original = FixtureNode(
        text="Jane Roe can be reached at 555-0100.",
        metadata={
            "source": "synthetic-fixture",
            "record_number": 1234567,
            "page_number": 2,
        },
        id_="patient-1234567",
    )

    transform = llamaindex_adapter.create_redaction_transform(
        config=llamaindex_adapter.LlamaIndexRedactionConfig(
            numeric_metadata_allowlist=("page_number",),
        ),
        deidentifier=fake_deidentify,
    )
    processed = transform([original])

    assert isinstance(transform, TransformComponentLike)
    assert processed[0].text == "[PERSON] can be reached at [PHONE]."
    assert processed[0].metadata["source"] == "synthetic-fixture"
    assert processed[0].metadata["record_number"].startswith("openmed-")
    assert processed[0].metadata["record_number"] != "1234567"
    assert processed[0].metadata["page_number"] == 2
    assert processed[0].id_ != "patient-1234567"
    UUID(processed[0].id_)
    assert original.text == "Jane Roe can be reached at 555-0100."
    assert original.metadata["record_number"] == 1234567
    assert original.id_ == "patient-1234567"


def test_redaction_factories_raise_clear_error_without_extra(monkeypatch) -> None:
    def missing_dependency(name: str):
        raise ImportError(name)

    monkeypatch.setattr(llamaindex_adapter, "_import_module", missing_dependency)

    with pytest.raises(ImportError, match=r"openmed\[llamaindex\]"):
        llamaindex_adapter.create_redaction_postprocessor(
            deidentifier=fake_deidentify,
        )
    with pytest.raises(ImportError, match=r"openmed\[llamaindex\]"):
        llamaindex_adapter.create_redaction_transform(deidentifier=fake_deidentify)


def test_extra_kwargs_cannot_override_named_safety_configuration(
    monkeypatch,
) -> None:
    monkeypatch.setattr(llamaindex_adapter, "_import_module", fake_import)
    config = llamaindex_adapter.LlamaIndexRedactionConfig(
        use_safety_sweep=True,
        extra_kwargs={"use_safety_sweep": False},
    )
    postprocessor = llamaindex_adapter.create_redaction_postprocessor(
        config=config,
        deidentifier=fake_deidentify,
    )
    original = FixtureNodeWithScore(
        node=FixtureNode("Jane Roe", {}),
        score=0.7,
    )

    with pytest.raises(
        ValueError,
        match="cannot override named configuration fields: use_safety_sweep",
    ):
        postprocessor.postprocess_nodes([original])

    with pytest.raises(
        ValueError,
        match="cannot override named configuration fields: model_name",
    ):
        llamaindex_adapter.LlamaIndexRedactionConfig(
            extra_kwargs={"model_name": "unreviewed-model"},
        ).to_deidentify_kwargs()
