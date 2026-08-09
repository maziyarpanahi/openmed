"""LangChain-compatible redaction transforms backed by OpenMed de-identification."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from copy import copy
from dataclasses import dataclass, field, replace
from importlib import import_module as _import_module
from pathlib import Path
from typing import Any

from openmed.core.capabilities import raise_missing_backend
from openmed.interop.function_tools import (
    RuntimeProvider,
    create_tool_callable,
    registry_tool_specs,
)
from openmed.mcp.tool_registry import render_langchain_tool_definitions

Deidentifier = Callable[..., Any]
_DEFAULT_POLICY = "hipaa_safe_harbor"
_REPLACEMENT_METHODS = frozenset({"replace", "format_preserve"})
_MESSAGE_METADATA_KEYS = frozenset(
    {"metadata", "additional_kwargs", "response_metadata"}
)


class LangChainRedactionError(RuntimeError):
    """Raised when a LangChain payload cannot be redacted safely.

    The adapter deliberately omits payload values from this exception. A
    deidentifier can be supplied by an application, so its exception text is
    not assumed to be safe for a chain log or callback trace.
    """


@dataclass
class LangChainRedactionState:
    """Request-local replacement state for deterministic chain redaction.

    ``replace`` and ``format_preserve`` can otherwise produce a different
    surrogate each time a message is processed. This state carries only the
    deterministic controls and aggregate counters; it never stores source
    text, mappings, or payloads. Reuse one state instance for a chain request
    when separate messages should share replacement behavior.
    """

    seed: int | None = None
    consistent: bool = True
    redacted_items: int = field(default=0, init=False)
    replacement_items: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.consistent, bool):
            raise TypeError("replacement_state.consistent must be a boolean")
        if self.seed is not None and (
            not isinstance(self.seed, int) or isinstance(self.seed, bool)
        ):
            raise TypeError("replacement_state.seed must be an integer or None")

    def deidentify_kwargs(self, *, method: str) -> dict[str, Any]:
        """Return safe replacement controls for one deidentifier call."""

        if method not in _REPLACEMENT_METHODS:
            return {}
        kwargs: dict[str, Any] = {"consistent": self.consistent}
        if self.seed is not None:
            kwargs["seed"] = self.seed
        return kwargs

    def record(self, *, method: str) -> None:
        """Record aggregate work without retaining any input value."""

        self.redacted_items += 1
        if method in _REPLACEMENT_METHODS:
            self.replacement_items += 1


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
    consistent: bool = True
    seed: int | None = None
    locale: str | None = None
    policy: str | None = _DEFAULT_POLICY
    calibration_thresholds_path: str | Path | None = None
    extra_kwargs: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.policy is None:
            object.__setattr__(self, "policy", _DEFAULT_POLICY)
        elif not isinstance(self.policy, str) or not self.policy.strip():
            raise ValueError("policy must be a non-empty string")
        if not isinstance(self.consistent, bool):
            raise TypeError("consistent must be a boolean")
        if self.seed is not None and (
            not isinstance(self.seed, int) or isinstance(self.seed, bool)
        ):
            raise TypeError("seed must be an integer or None")

    def to_deidentify_kwargs(self) -> dict[str, Any]:
        """Return keyword arguments for ``openmed.core.pii.deidentify``."""

        kwargs: dict[str, Any] = {
            "method": self.method,
            "model_name": self.model_name,
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
        extras = dict(self.extra_kwargs)
        collisions = sorted(kwargs.keys() & extras.keys())
        if collisions:
            fields = ", ".join(collisions)
            raise ValueError(
                f"extra_kwargs cannot override named configuration fields: {fields}"
            )
        kwargs.update(extras)
        return {key: value for key, value in kwargs.items() if value is not None}


class OpenMedRedactionTransform:
    """Redact chain payloads while preserving their container structure.

    Strings are redacted directly. LangChain ``Document`` and message-like
    objects are copied with only their textual content changed, so metadata,
    message attributes, and list ordering remain untouched.
    """

    def __init__(
        self,
        *,
        config: LangChainRedactionConfig | None = None,
        input_key: str | None = None,
        output_key: str | None = None,
        deidentifier: Deidentifier | None = None,
        policy: str | None = None,
        replacement_state: LangChainRedactionState | Mapping[str, Any] | None = None,
    ) -> None:
        self.config = _config_with_policy(config, policy) or LangChainRedactionConfig()
        self.input_key = input_key
        self.output_key = output_key
        self._deidentifier = deidentifier
        self.replacement_state = _coerce_replacement_state(replacement_state)

    def __call__(self, input: Any) -> Any:
        """Redact one payload when the transform is used as a plain callable."""

        return self.invoke(input)

    def redact(self, input: Any) -> Any:
        """Redact one payload using a descriptive node-style method name."""

        return self.invoke(input)

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
        if _is_message_like(value):
            return self._redact_message(value)
        if _is_prompt_like(value):
            return self._redact_prompt(value)
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
            if _is_message_mapping(payload):
                payload["content"] = self._redact_message_content(payload["content"])
                return payload
            return {
                key: item if key in _MESSAGE_METADATA_KEYS else self._redact_value(item)
                for key, item in payload.items()
            }

        if self.input_key not in payload:
            raise KeyError(
                f"input key {self.input_key!r} not found in LangChain payload"
            )

        target_key = self.output_key or self.input_key
        payload[target_key] = self._redact_value(payload[self.input_key])
        return payload

    def _redact_document(self, value: Any) -> Any:
        redacted_content = self._redact_text(value.page_content)
        return _copy_with_field(
            value,
            field_name="page_content",
            field_value=redacted_content,
            error_message="LangChain document cannot be copied with redacted content",
        )

    def _redact_message(self, value: Any) -> Any:
        redacted_content = _redact_message_content(
            value.content,
            redact_text=self._redact_text,
        )
        return _copy_with_field(
            value,
            field_name="content",
            field_value=redacted_content,
            error_message="LangChain message cannot be copied with redacted content",
        )

    def _redact_prompt(self, value: Any) -> Any:
        messages = [self._redact_value(message) for message in value.messages]
        if isinstance(value.messages, tuple):
            messages = tuple(messages)
        return _copy_with_field(
            value,
            field_name="messages",
            field_value=messages,
            error_message="LangChain prompt cannot be copied with redacted messages",
        )

    def _redact_text(self, text: str) -> str:
        if text == "":
            return text

        kwargs = self.config.to_deidentify_kwargs()
        if self.replacement_state is not None:
            kwargs.update(
                self.replacement_state.deidentify_kwargs(method=self.config.method)
            )

        try:
            result = self._deidentifier_or_default()(text, **kwargs)
        except Exception as exc:  # noqa: BLE001 - do not expose payload-bearing errors
            raise LangChainRedactionError(
                "LangChain redaction failed in the local deidentifier "
                f"({exc.__class__.__name__})"
            ) from None

        if isinstance(result, str):
            redacted_text = result
        else:
            redacted_text = getattr(result, "deidentified_text", None)
            if not isinstance(redacted_text, str):
                raise LangChainRedactionError(
                    "deidentifier must return a string or an object with "
                    "deidentified_text"
                )

        if self.replacement_state is not None:
            self.replacement_state.record(method=self.config.method)
        return redacted_text

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
    policy: str | None = None,
    replacement_state: LangChainRedactionState | Mapping[str, Any] | None = None,
) -> OpenMedRedactionTransform:
    """Create a dependency-light transform that can be wrapped as a runnable."""

    resolved_config = _config_with_policy(config, policy)
    return OpenMedRedactionTransform(
        config=resolved_config,
        input_key=input_key,
        output_key=output_key,
        deidentifier=deidentifier,
        replacement_state=replacement_state,
    )


def create_redaction_runnable(
    *,
    config: LangChainRedactionConfig | None = None,
    input_key: str | None = None,
    output_key: str | None = None,
    deidentifier: Deidentifier | None = None,
    policy: str | None = None,
    replacement_state: LangChainRedactionState | Mapping[str, Any] | None = None,
    name: str = "openmed_redaction",
) -> Any:
    """Create a LangChain runnable that redacts payloads before downstream steps."""

    transform = create_redaction_transform(
        config=config,
        input_key=input_key,
        output_key=output_key,
        deidentifier=deidentifier,
        policy=policy,
        replacement_state=replacement_state,
    )
    return transform.as_runnable(name=name)


def create_redaction_node(
    *,
    config: LangChainRedactionConfig | None = None,
    input_key: str | None = None,
    output_key: str | None = None,
    deidentifier: Deidentifier | None = None,
    policy: str | None = None,
    replacement_state: LangChainRedactionState | Mapping[str, Any] | None = None,
    name: str = "openmed_redaction",
) -> Any:
    """Create an optional LangChain node that redacts payloads locally.

    The `langchain-core` extra is loaded only by this factory. Use
    :func:`create_redaction_transform` when the dependency-light transform is
    sufficient for a test or another local orchestration layer.
    """

    return create_redaction_runnable(
        config=config,
        input_key=input_key,
        output_key=output_key,
        deidentifier=deidentifier,
        policy=policy,
        replacement_state=replacement_state,
        name=name,
    )


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
        raise_missing_backend("langchain", feature="LangChain tools", cause=exc)

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
        raise_missing_backend("langchain", feature="LangChain support", cause=exc)

    try:
        return module.RunnableLambda
    except AttributeError as exc:
        raise ImportError(
            "LangChain support requires langchain-core with RunnableLambda."
        ) from exc


def _is_document_like(value: Any) -> bool:
    return hasattr(value, "page_content") and isinstance(value.page_content, str)


def _is_message_like(value: Any) -> bool:
    """Return whether *value* resembles a LangChain message without importing it."""

    return not _is_document_like(value) and hasattr(value, "content")


def _is_prompt_like(value: Any) -> bool:
    """Return whether *value* resembles a LangChain prompt value."""

    if _is_message_like(value) or not hasattr(value, "messages"):
        return False
    messages = value.messages
    return isinstance(messages, Sequence) and not isinstance(messages, (str, bytes))


def _is_message_mapping(value: Mapping[str, Any]) -> bool:
    """Return whether a mapping has the standard message-content shape."""

    return "content" in value


def _redact_message_content(
    content: Any,
    *,
    redact_text: Callable[[str], str] | None = None,
) -> Any:
    """Redact text blocks while retaining non-text message content verbatim."""

    redact = redact_text or (lambda text: text)
    if isinstance(content, str):
        return redact(content)
    if isinstance(content, list):
        return [
            _redact_message_content_item(item, redact_text=redact) for item in content
        ]
    if isinstance(content, tuple):
        return tuple(
            _redact_message_content_item(item, redact_text=redact) for item in content
        )
    if isinstance(content, Mapping):
        return _redact_message_block(content, redact_text=redact)
    return content


def _redact_message_content_item(
    item: Any, *, redact_text: Callable[[str], str]
) -> Any:
    if isinstance(item, str):
        return redact_text(item)
    if isinstance(item, Mapping):
        return _redact_message_block(item, redact_text=redact_text)
    if hasattr(item, "text") and isinstance(item.text, str):
        return _copy_with_field(
            item,
            field_name="text",
            field_value=redact_text(item.text),
            error_message="LangChain content block cannot be copied safely",
        )
    return item


def _redact_message_block(
    block: Mapping[str, Any], *, redact_text: Callable[[str], str]
) -> dict[str, Any]:
    """Redact only text-bearing keys in a multimodal message block."""

    redacted = dict(block)
    if isinstance(redacted.get("text"), str):
        redacted["text"] = redact_text(redacted["text"])
    elif isinstance(redacted.get("content"), str):
        redacted["content"] = redact_text(redacted["content"])
    return redacted


def _copy_with_field(
    value: Any,
    *,
    field_name: str,
    field_value: Any,
    error_message: str,
) -> Any:
    """Copy a framework object and replace one field without mutating it."""

    for copier_name in ("model_copy", "copy"):
        copier = getattr(value, copier_name, None)
        if not callable(copier):
            continue
        try:
            return copier(update={field_name: field_value})
        except TypeError:
            try:
                cloned = copier()
                setattr(cloned, field_name, field_value)
                return cloned
            except Exception:  # noqa: BLE001 - sanitize framework copy failures
                continue

    try:
        cloned = copy(value)
        setattr(cloned, field_name, field_value)
        return cloned
    except Exception:  # noqa: BLE001 - sanitize framework copy failures
        raise TypeError(error_message) from None


def _coerce_replacement_state(
    value: LangChainRedactionState | Mapping[str, Any] | None,
) -> LangChainRedactionState | None:
    if value is None or isinstance(value, LangChainRedactionState):
        return value
    if not isinstance(value, Mapping):
        raise TypeError(
            "replacement_state must be LangChainRedactionState, a mapping, or None"
        )
    unknown = sorted(set(value) - {"seed", "consistent"})
    if unknown:
        raise ValueError(
            "replacement_state supports only the 'seed' and 'consistent' fields"
        )
    return LangChainRedactionState(
        seed=value.get("seed"),
        consistent=value.get("consistent", True),
    )


def _config_with_policy(
    config: LangChainRedactionConfig | None,
    policy: str | None,
) -> LangChainRedactionConfig | None:
    if policy is None:
        return config
    if config is None:
        return LangChainRedactionConfig(policy=policy)
    return replace(config, policy=policy)


LangChainRedactionNode = OpenMedRedactionTransform
OpenMedRedactionNode = OpenMedRedactionTransform
RedactionNode = OpenMedRedactionTransform
ReplacementState = LangChainRedactionState


__all__ = [
    "Deidentifier",
    "LangChainRedactionError",
    "LangChainRedactionConfig",
    "LangChainRedactionNode",
    "LangChainRedactionState",
    "OpenMedRedactionTransform",
    "OpenMedRetrievalChain",
    "OpenMedRedactionNode",
    "RedactionNode",
    "ReplacementState",
    "create_retrieval_chain",
    "create_tool_definitions",
    "create_redaction_node",
    "create_redaction_runnable",
    "create_redaction_transform",
    "get_langchain_tools",
]
