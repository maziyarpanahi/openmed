"""LangChain-compatible redaction transforms backed by OpenMed de-identification."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from copy import copy
from dataclasses import dataclass, field
from importlib import import_module as _import_module
from pathlib import Path
from typing import Any

from openmed.interop.function_tools import (
    RuntimeProvider,
    create_tool_callable,
    registry_tool_specs,
)
from openmed.mcp.tool_registry import render_langchain_tool_definitions

Deidentifier = Callable[..., Any]


@dataclass(frozen=True)
class LangChainRedactionConfig:
    """Runtime options forwarded to OpenMed's de-identification engine."""

    method: str = "mask"
    model_name: str | None = None
    confidence_threshold: float = 0.7
    keep_year: bool = False
    keep_mapping: bool = False
    use_smart_merging: bool = True
    lang: str = "en"
    normalize_accents: bool | None = None
    use_safety_sweep: bool = True
    consistent: bool = False
    seed: int | None = None
    locale: str | None = None
    policy: str | None = None
    calibration_thresholds_path: str | Path | None = None
    extra_kwargs: Mapping[str, Any] = field(default_factory=dict)

    def to_deidentify_kwargs(self) -> dict[str, Any]:
        """Return keyword arguments for ``openmed.core.pii.deidentify``."""

        kwargs: dict[str, Any] = {
            "method": self.method,
            "confidence_threshold": self.confidence_threshold,
            "keep_year": self.keep_year,
            "keep_mapping": self.keep_mapping,
            "use_smart_merging": self.use_smart_merging,
            "lang": self.lang,
            "normalize_accents": self.normalize_accents,
            "use_safety_sweep": self.use_safety_sweep,
            "consistent": self.consistent,
            "seed": self.seed,
            "locale": self.locale,
            "policy": self.policy,
            "calibration_thresholds_path": self.calibration_thresholds_path,
        }
        if self.model_name is not None:
            kwargs["model_name"] = self.model_name

        kwargs.update(dict(self.extra_kwargs))
        return {key: value for key, value in kwargs.items() if value is not None}


class OpenMedRedactionTransform:
    """Redact strings, documents, lists, and mapping payloads before a chain call."""

    def __init__(
        self,
        *,
        config: LangChainRedactionConfig | None = None,
        input_key: str | None = None,
        output_key: str | None = None,
        deidentifier: Deidentifier | None = None,
    ) -> None:
        self.config = config or LangChainRedactionConfig()
        self.input_key = input_key
        self.output_key = output_key
        self._deidentifier = deidentifier

    def invoke(
        self,
        input: Any,
        config: Any | None = None,
        **kwargs: Any,
    ) -> Any:
        """Redact one input payload using the LangChain ``Runnable`` signature."""

        del config, kwargs
        return self._redact_value(input)

    def batch(
        self,
        inputs: Sequence[Any],
        config: Any | None = None,
        **kwargs: Any,
    ) -> list[Any]:
        """Redact a batch of payloads using LangChain's synchronous batch shape."""

        del config, kwargs
        return [self.invoke(item) for item in inputs]

    def transform(
        self,
        input: Iterable[Any],
        config: Any | None = None,
        **kwargs: Any,
    ) -> Iterable[Any]:
        """Yield redacted payloads for streaming-style chain stages."""

        del config, kwargs
        for item in input:
            yield self.invoke(item)

    def as_runnable(self, *, name: str = "openmed_redaction") -> Any:
        """Return a LangChain ``RunnableLambda`` wrapping this transform."""

        runnable_lambda = _load_runnable_lambda()
        return runnable_lambda(self.invoke, name=name)

    def index_document(
        self,
        index: Any,
        vault: Any,
        *,
        document_id: str,
        text: str,
        chunk_size: int = 1000,
    ) -> Any:
        """Redact and ingest one note into a redaction-preserving index.

        This method keeps the existing LangChain redaction configuration as the
        single detector configuration while the retrieval index owns unique
        placeholders, encrypted mapping storage, and chunk offsets.
        """

        index_document = getattr(index, "index_document", None)
        if not callable(index_document):
            raise TypeError("index must expose index_document()")
        return index_document(
            document_id,
            text,
            vault=vault,
            chunk_size=chunk_size,
            deidentifier=self._deidentifier_or_default(),
            deidentify_kwargs=self.config.to_deidentify_kwargs(),
        )

    def _redact_value(self, value: Any) -> Any:
        if isinstance(value, str):
            return self._redact_text(value)
        if _is_document_like(value):
            return self._redact_document(value)
        if isinstance(value, Mapping):
            return self._redact_mapping(value)
        if isinstance(value, list):
            return [self._redact_value(item) for item in value]
        if isinstance(value, tuple):
            return tuple(self._redact_value(item) for item in value)
        return value

    def _redact_mapping(self, value: Mapping[str, Any]) -> dict[str, Any]:
        payload = dict(value)
        if self.input_key is None:
            return {key: self._redact_value(item) for key, item in payload.items()}

        if self.input_key not in payload:
            raise KeyError(
                f"input key {self.input_key!r} not found in LangChain payload"
            )

        target_key = self.output_key or self.input_key
        payload[target_key] = self._redact_value(payload[self.input_key])
        return payload

    def _redact_document(self, value: Any) -> Any:
        redacted_content = self._redact_text(str(value.page_content))
        if hasattr(value, "model_copy"):
            return value.model_copy(update={"page_content": redacted_content})
        if hasattr(value, "copy"):
            return value.copy(update={"page_content": redacted_content})

        cloned = copy(value)
        cloned.page_content = redacted_content
        return cloned

    def _redact_text(self, text: str) -> str:
        if text == "":
            return text

        result = self._deidentifier_or_default()(
            text,
            **self.config.to_deidentify_kwargs(),
        )
        if isinstance(result, str):
            return result

        try:
            return str(result.deidentified_text)
        except AttributeError as exc:
            raise TypeError(
                "deidentifier must return a string or an object with deidentified_text"
            ) from exc

    def _deidentifier_or_default(self) -> Deidentifier:
        if self._deidentifier is not None:
            return self._deidentifier

        from openmed.core.pii import deidentify

        return deidentify


class OpenMedRetrievalChain:
    """Compose local redaction/retrieval with gateway-only model boundaries."""

    def __init__(
        self,
        *,
        redaction_transform: OpenMedRedactionTransform,
        index: Any,
        vault: Any,
        retriever: Any,
        external_llm: Any,
        reidentifier: Any,
    ) -> None:
        from openmed.interop.retrieval import (
            AuthorizedReidentifier,
            GatewayBoundExternalLLM,
        )

        if not isinstance(external_llm, GatewayBoundExternalLLM):
            raise TypeError(
                "external_llm must be a gateway-bound retrieval model wrapper"
            )
        if not isinstance(reidentifier, AuthorizedReidentifier):
            raise TypeError(
                "reidentifier must enforce gateway authorization and auditing"
            )
        self.redaction_transform = redaction_transform
        self.index = index
        self.vault = vault
        self.retriever = retriever
        self.external_llm = external_llm
        self.reidentifier = reidentifier

    def index_document(
        self,
        document_id: str,
        text: str,
        *,
        chunk_size: int = 1000,
    ) -> Any:
        """Run the configured local redactor before retrieval ingestion."""

        return self.redaction_transform.index_document(
            self.index,
            self.vault,
            document_id=document_id,
            text=text,
            chunk_size=chunk_size,
        )

    def invoke(
        self,
        input: Mapping[str, Any],
        config: Any | None = None,
        **kwargs: Any,
    ) -> str:
        """Retrieve redacted context, call the gateway, then authorize restore."""

        del config, kwargs
        if not isinstance(input, Mapping):
            raise TypeError("retrieval chain input must be a mapping")
        try:
            query = input["query"]
            principal = input["principal"]
        except KeyError as exc:
            raise KeyError(
                "retrieval chain input requires query and principal"
            ) from exc
        if not isinstance(query, str) or not isinstance(principal, str):
            raise TypeError("query and principal must be strings")
        k = input.get("k", 4)
        if not isinstance(k, int) or isinstance(k, bool):
            raise TypeError("k must be an integer")

        redacted_query = self.redaction_transform.invoke(query)
        if not isinstance(redacted_query, str):
            raise TypeError("redaction transform must return a string for query input")
        passages = self.retriever.retrieve(redacted_query, k=k)
        response = self.external_llm.invoke(redacted_query, passages)
        return self.reidentifier.reidentify(
            response.text,
            document_keys=response.document_keys,
            principal=principal,
        )

    def as_runnable(self, *, name: str = "openmed_redacted_retrieval") -> Any:
        """Return a LangChain runnable wrapping the complete retrieval flow."""

        runnable_lambda = _load_runnable_lambda()
        return runnable_lambda(self.invoke, name=name)


def create_redaction_transform(
    *,
    config: LangChainRedactionConfig | None = None,
    input_key: str | None = None,
    output_key: str | None = None,
    deidentifier: Deidentifier | None = None,
) -> OpenMedRedactionTransform:
    """Create a dependency-light transform that can be wrapped as a runnable."""

    return OpenMedRedactionTransform(
        config=config,
        input_key=input_key,
        output_key=output_key,
        deidentifier=deidentifier,
    )


def create_redaction_runnable(
    *,
    config: LangChainRedactionConfig | None = None,
    input_key: str | None = None,
    output_key: str | None = None,
    deidentifier: Deidentifier | None = None,
    name: str = "openmed_redaction",
) -> Any:
    """Create a LangChain runnable that redacts payloads before downstream steps."""

    transform = create_redaction_transform(
        config=config,
        input_key=input_key,
        output_key=output_key,
        deidentifier=deidentifier,
    )
    return transform.as_runnable(name=name)


def create_retrieval_chain(
    *,
    index: Any,
    vault: Any,
    retriever: Any,
    external_llm: Any,
    reidentifier: Any,
    redaction_transform: OpenMedRedactionTransform | None = None,
    config: LangChainRedactionConfig | None = None,
    deidentifier: Deidentifier | None = None,
) -> OpenMedRetrievalChain:
    """Create a dependency-light redacted retrieval composition."""

    transform = redaction_transform or create_redaction_transform(
        config=config,
        deidentifier=deidentifier,
    )
    return OpenMedRetrievalChain(
        redaction_transform=transform,
        index=index,
        vault=vault,
        retriever=retriever,
        external_llm=external_llm,
        reidentifier=reidentifier,
    )


def create_tool_definitions() -> tuple[dict[str, Any], ...]:
    """Return LangChain-facing OpenMed tool definitions from the registry."""

    return render_langchain_tool_definitions()


def get_langchain_tools(
    *,
    runtime_provider: RuntimeProvider | None = None,
) -> tuple[Any, ...]:
    """Return LangChain ``StructuredTool`` objects for every registry tool."""

    structured_tool = _load_structured_tool()
    return tuple(
        _structured_tool_from_spec(structured_tool, spec, runtime_provider)
        for spec in registry_tool_specs()
    )


def _structured_tool_from_spec(
    structured_tool: Any,
    spec: Any,
    runtime_provider: RuntimeProvider | None,
) -> Any:
    func = create_tool_callable(spec, runtime_provider=runtime_provider)
    try:
        return structured_tool.from_function(
            func=func,
            name=spec.name,
            description=spec.description,
        )
    except TypeError:
        return structured_tool.from_function(
            func,
            name=spec.name,
            description=spec.description,
        )


def _load_structured_tool() -> Any:
    try:
        module = _import_module("langchain_core.tools")
    except ImportError as exc:
        raise ImportError(
            "LangChain tools require the 'langchain' extra. "
            "Install with `pip install openmed[langchain]`."
        ) from exc

    try:
        return module.StructuredTool
    except AttributeError as exc:
        raise ImportError(
            "LangChain tools require langchain-core with StructuredTool."
        ) from exc


def _load_runnable_lambda() -> Any:
    try:
        module = _import_module("langchain_core.runnables")
    except ImportError as exc:
        raise ImportError(
            "LangChain support requires the 'langchain' extra. "
            "Install with `pip install openmed[langchain]`."
        ) from exc

    try:
        return module.RunnableLambda
    except AttributeError as exc:
        raise ImportError(
            "LangChain support requires langchain-core with RunnableLambda."
        ) from exc


def _is_document_like(value: Any) -> bool:
    return hasattr(value, "page_content") and isinstance(value.page_content, str)


__all__ = [
    "Deidentifier",
    "LangChainRedactionConfig",
    "OpenMedRedactionTransform",
    "OpenMedRetrievalChain",
    "create_retrieval_chain",
    "create_tool_definitions",
    "create_redaction_runnable",
    "create_redaction_transform",
    "get_langchain_tools",
]
