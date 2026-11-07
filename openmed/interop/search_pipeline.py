"""Redaction components for modular document-search pipelines."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from copy import copy
from dataclasses import dataclass
from importlib import import_module as _import_module
from typing import Any

from openmed.core.schemas import OpenMedSpan
from openmed.interop._pii import canonical_redaction
from openmed.interop.gateway import PrivacyGatewayConfig, RedactionMapping
from openmed.mcp.tool_registry import render_search_pipeline_tool_definitions

Deidentifier = Callable[..., Any]


@dataclass(frozen=True)
class RedactedSearchItem:
    """One redacted value plus text-free canonical spans and local mapping."""

    value: Any
    spans: tuple[OpenMedSpan, ...]
    mapping: RedactionMapping


class RedactionComponent:
    """Redact text or document-like values before indexing or generation."""

    def __init__(
        self,
        *,
        config: PrivacyGatewayConfig | None = None,
        deidentify_kwargs: Mapping[str, Any] | None = None,
        deidentifier: Deidentifier | None = None,
        default_doc_id: str = "search-pipeline",
    ) -> None:
        self.config = config or PrivacyGatewayConfig()
        kwargs = self.config.to_deidentify_kwargs()
        if deidentify_kwargs is not None:
            kwargs.update(dict(deidentify_kwargs))
        kwargs["keep_mapping"] = True
        kwargs["audit"] = False
        kwargs["cache_results"] = False
        self.deidentify_kwargs = kwargs
        self._deidentifier = deidentifier
        self.default_doc_id = default_doc_id

    def redact(self, value: Any, *, doc_id: str | None = None) -> RedactedSearchItem:
        """Return the transformed value and canonical redaction artifacts."""

        text = _document_text(value)
        if text == "":
            return RedactedSearchItem(value, (), RedactionMapping())
        result = self._deidentifier_or_default()(text, **self.deidentify_kwargs)
        artifact = canonical_redaction(
            result,
            source_text=text,
            doc_id=doc_id or self.default_doc_id,
            lang=str(self.deidentify_kwargs.get("lang", self.config.lang)),
            method=str(self.deidentify_kwargs.get("method", self.config.method)),
        )
        return RedactedSearchItem(
            value=_replace_document_text(value, artifact.redacted_text),
            spans=artifact.spans,
            mapping=RedactionMapping(artifact.mapping),
        )

    def invoke(
        self,
        input: Any,
        config: Any | None = None,
        **kwargs: Any,
    ) -> Any:
        """Redact one value using a synchronous Runnable-style signature."""

        del config, kwargs
        return self.redact(input).value

    def batch(
        self,
        inputs: Sequence[Any],
        config: Any | None = None,
        **kwargs: Any,
    ) -> list[Any]:
        """Redact a batch using a synchronous Runnable-style signature."""

        del config, kwargs
        return [
            self.redact(value, doc_id=f"{self.default_doc_id}:{index}").value
            for index, value in enumerate(inputs)
        ]

    def transform(
        self,
        input: Iterable[Any],
        config: Any | None = None,
        **kwargs: Any,
    ) -> Iterable[Any]:
        """Yield redacted values for streaming pipeline stages."""

        del config, kwargs
        for index, value in enumerate(input):
            yield self.redact(
                value,
                doc_id=f"{self.default_doc_id}:{index}",
            ).value

    def run(self, documents: Sequence[Any]) -> dict[str, list[Any]]:
        """Return a modular-search component output with canonical spans."""

        redactions = [
            self.redact(document, doc_id=f"{self.default_doc_id}:{index}")
            for index, document in enumerate(documents)
        ]
        return {
            "documents": [item.value for item in redactions],
            "spans": [span for item in redactions for span in item.spans],
        }

    def _deidentifier_or_default(self) -> Deidentifier:
        if self._deidentifier is not None:
            return self._deidentifier
        from openmed.core.pii import deidentify

        return deidentify


class RetrievalRedactionFilter(RedactionComponent):
    """Redact documents returned by a retriever before downstream generation."""

    def __init__(
        self,
        *,
        documents_key: str = "documents",
        spans_key: str = "spans",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.documents_key = documents_key
        self.spans_key = spans_key

    def invoke(
        self,
        input: Any,
        config: Any | None = None,
        **kwargs: Any,
    ) -> Any:
        """Redact a retriever mapping while preserving its other fields."""

        del config, kwargs
        if not isinstance(input, Mapping):
            return super().invoke(input)
        if self.documents_key not in input:
            raise KeyError(f"retrieval payload is missing {self.documents_key!r}")
        documents = input[self.documents_key]
        if not isinstance(documents, Sequence) or isinstance(documents, (str, bytes)):
            raise TypeError(
                f"retrieval payload {self.documents_key!r} must be a sequence"
            )
        output = dict(input)
        component_output = self.run(documents)
        output[self.documents_key] = component_output["documents"]
        output[self.spans_key] = component_output["spans"]
        return output

    filter = invoke


def create_haystack_component(
    *,
    retrieval_filter: bool = False,
    config: PrivacyGatewayConfig | None = None,
    deidentify_kwargs: Mapping[str, Any] | None = None,
    deidentifier: Deidentifier | None = None,
    default_doc_id: str = "search-pipeline",
) -> Any:
    """Return a lazily decorated Haystack component instance."""

    try:
        haystack = _import_module("haystack")
        component = haystack.component
        document_type = haystack.Document
    except (ImportError, AttributeError) as exc:
        raise ImportError(
            "Search-pipeline support requires the 'haystack' extra. "
            "Install with `pip install openmed[haystack]`."
        ) from exc

    component_type = (
        RetrievalRedactionFilter if retrieval_filter else RedactionComponent
    )
    implementation = component_type(
        config=config,
        deidentify_kwargs=deidentify_kwargs,
        deidentifier=deidentifier,
        default_doc_id=default_doc_id,
    )

    class OpenMedSearchRedactor:
        """Haystack binding for the dependency-light OpenMed component."""

        def run(self, documents: list[Any]) -> dict[str, list[Any]]:
            return implementation.run(documents)

    document_list_type = list[document_type]
    span_list_type = list[OpenMedSpan]
    OpenMedSearchRedactor.run.__annotations__ = {
        "documents": document_list_type,
        "return": dict[str, list[Any]],
    }
    OpenMedSearchRedactor.run = component.output_types(
        documents=document_list_type,
        spans=span_list_type,
    )(OpenMedSearchRedactor.run)
    OpenMedSearchRedactor.__module__ = __name__
    OpenMedSearchRedactor.__qualname__ = "OpenMedSearchRedactor"
    return component(OpenMedSearchRedactor)()


def bind_search_pipeline(
    pipeline: Any,
    *,
    name: str = "openmed_redaction",
    retrieval_filter: bool = False,
    **component_kwargs: Any,
) -> Any:
    """Add a lazily created redaction component to a Haystack pipeline."""

    add_component = getattr(pipeline, "add_component", None)
    if not callable(add_component):
        raise TypeError("pipeline must expose add_component()")
    add_component(
        name,
        create_haystack_component(
            retrieval_filter=retrieval_filter,
            **component_kwargs,
        ),
    )
    return pipeline


def create_tool_definitions() -> tuple[dict[str, Any], ...]:
    """Return search-pipeline tool definitions from the shared registry."""

    return render_search_pipeline_tool_definitions()


def _document_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        for key in ("content", "page_content", "text"):
            if key in value:
                text = value[key]
                if text is None:
                    return ""
                if isinstance(text, str):
                    return text
                raise TypeError(f"document field {key!r} must be a string")
    for attribute in ("content", "page_content", "text"):
        if hasattr(value, attribute):
            text = getattr(value, attribute)
            if text is None:
                return ""
            if isinstance(text, str):
                return text
            raise TypeError(f"document attribute {attribute!r} must be a string")
    raise TypeError("search values must be strings or text-bearing documents")


def _replace_document_text(value: Any, text: str) -> Any:
    if isinstance(value, str):
        return text
    key = _document_text_key(value)
    if isinstance(value, Mapping):
        output = dict(value)
        output[key] = text
        return output
    if hasattr(value, "model_copy"):
        return value.model_copy(update={key: text})
    if hasattr(value, "to_dict") and hasattr(type(value), "from_dict"):
        payload = value.to_dict(flatten=False)
        payload[key] = text
        return type(value).from_dict(payload)
    if hasattr(value, "copy"):
        try:
            return value.copy(update={key: text})
        except TypeError:
            pass
    output = copy(value)
    setattr(output, key, text)
    return output


def _document_text_key(value: Any) -> str:
    for key in ("content", "page_content", "text"):
        if (isinstance(value, Mapping) and key in value) or hasattr(value, key):
            return key
    raise TypeError("search values must be strings or text-bearing documents")


OpenMedRedactionComponent = RedactionComponent
OpenMedRetrievalRedactionFilter = RetrievalRedactionFilter

__all__ = [
    "Deidentifier",
    "OpenMedRedactionComponent",
    "OpenMedRetrievalRedactionFilter",
    "RedactedSearchItem",
    "RedactionComponent",
    "RetrievalRedactionFilter",
    "bind_search_pipeline",
    "create_haystack_component",
    "create_tool_definitions",
]
