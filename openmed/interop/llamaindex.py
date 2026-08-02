"""LlamaIndex redaction and tool adapters backed by OpenMed."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from hashlib import sha256
from importlib import import_module as _import_module
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, uuid4, uuid5

from openmed.interop.function_tools import (
    RuntimeProvider,
    create_tool_callable,
    registry_tool_specs,
)
from openmed.mcp.tool_registry import render_adapter_tool_definitions

Deidentifier = Callable[..., Any]
_NODE_ID_NAMESPACE = uuid5(NAMESPACE_URL, "https://openmed.ai/llamaindex-redaction")


@dataclass(frozen=True)
class LlamaIndexRedactionConfig:
    """Runtime options forwarded to OpenMed's de-identification engine."""

    method: str = "mask"
    model_name: str | None = None
    confidence_threshold: float = 0.7
    keep_year: bool = False
    keep_mapping: bool = False
    use_smart_merging: bool = True
    lang: str = "en"
    redact_metadata: bool = True
    numeric_metadata_allowlist: tuple[str, ...] = ()
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


class _NodeRedactor:
    def __init__(
        self,
        *,
        config: LlamaIndexRedactionConfig,
        deidentifier: Deidentifier | None,
        storage_safe: bool,
    ) -> None:
        self.config = config
        self._deidentifier = deidentifier
        self._storage_safe = storage_safe

    def redact_scored_nodes(self, nodes: Sequence[Any]) -> list[Any]:
        return [self._redact_scored_node(node) for node in nodes]

    def redact_nodes(self, nodes: Sequence[Any]) -> list[Any]:
        return [self._redact_node(node) for node in nodes]

    def _redact_scored_node(self, scored_node: Any) -> Any:
        try:
            redacted = _clone(scored_node)
            self._redact_node(redacted.node, clone=False)
        except AttributeError as exc:
            raise TypeError(
                "LlamaIndex postprocessor inputs must expose a node attribute"
            ) from exc
        return redacted

    def _redact_node(self, node: Any, *, clone: bool = True) -> Any:
        redacted = _clone(node) if clone else node
        text = _node_text(redacted)
        if text:
            _set_node_text(redacted, self._redact_text(text))

        metadata = getattr(redacted, "metadata", None)
        if self.config.redact_metadata and isinstance(metadata, Mapping):
            sanitized_metadata = self._redact_metadata_value(metadata)
            _set_node_metadata(redacted, sanitized_metadata)
            _exclude_metadata_from_content(redacted, sanitized_metadata)
        if self._storage_safe:
            self._pseudonymize_node_identifiers(redacted)
        return redacted

    def _redact_text(self, text: str) -> str:
        result = self._deidentifier_or_default()(
            text,
            **self.config.to_deidentify_kwargs(),
        )
        return _deidentified_text(result)

    def _redact_metadata_value(
        self,
        value: Any,
        *,
        ancestors: frozenset[int] = frozenset(),
        field_name: str | None = None,
    ) -> Any:
        if isinstance(value, str):
            return self._redact_text(value) if value else value
        if isinstance(value, Mapping):
            next_ancestors = _metadata_ancestors(value, ancestors)
            redacted: dict[str, Any] = {}
            for key, item in value.items():
                if not isinstance(key, str):
                    raise TypeError(
                        "LlamaIndex metadata keys must be strings when "
                        "redact_metadata=True"
                    )
                redacted_key = self._redact_text(key) if key else key
                if redacted_key in redacted:
                    raise ValueError("LlamaIndex metadata keys collide after redaction")
                redacted[redacted_key] = self._redact_metadata_value(
                    item,
                    ancestors=next_ancestors,
                    field_name=key,
                )
            return redacted
        if isinstance(value, list):
            next_ancestors = _metadata_ancestors(value, ancestors)
            return [
                self._redact_metadata_value(
                    item,
                    ancestors=next_ancestors,
                    field_name=field_name,
                )
                for item in value
            ]
        if isinstance(value, tuple):
            next_ancestors = _metadata_ancestors(value, ancestors)
            return tuple(
                self._redact_metadata_value(
                    item,
                    ancestors=next_ancestors,
                    field_name=field_name,
                )
                for item in value
            )
        if value is None or isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            if (
                self._storage_safe
                and field_name not in self.config.numeric_metadata_allowlist
            ):
                return _pseudonymized_identifier(value)
            return value
        raise TypeError(
            "LlamaIndex metadata values must be strings, numbers, booleans, "
            "null, mappings, lists, or tuples when redact_metadata=True"
        )

    def _pseudonymize_node_identifiers(self, node: Any) -> None:
        node_id = getattr(node, "id_", None)
        if isinstance(node_id, str) and node_id:
            try:
                node.id_ = _pseudonymized_node_id(node_id)
            except (AttributeError, TypeError, ValueError) as exc:
                raise TypeError("LlamaIndex node id is not mutable") from exc

        relationships = getattr(node, "relationships", None)
        if isinstance(relationships, Mapping):
            for related in relationships.values():
                self._pseudonymize_related_node_ids(related)

    def _pseudonymize_related_node_ids(self, related: Any) -> None:
        if isinstance(related, list):
            for item in related:
                self._pseudonymize_related_node_ids(item)
            return

        metadata = getattr(related, "metadata", None)
        if self.config.redact_metadata and isinstance(metadata, Mapping):
            sanitized_metadata = self._redact_metadata_value(metadata)
            try:
                related.metadata = sanitized_metadata
            except (AttributeError, TypeError, ValueError) as exc:
                raise TypeError(
                    "LlamaIndex related-node metadata is not mutable"
                ) from exc

        node_id = getattr(related, "node_id", None)
        if not isinstance(node_id, str) or not node_id:
            return
        try:
            related.node_id = _pseudonymized_node_id(node_id)
        except (AttributeError, TypeError, ValueError) as exc:
            raise TypeError("LlamaIndex related-node id is not mutable") from exc

    def _deidentifier_or_default(self) -> Deidentifier:
        if self._deidentifier is not None:
            return self._deidentifier

        from openmed.core.pii import deidentify

        return deidentify


def create_redaction_postprocessor(
    *,
    config: LlamaIndexRedactionConfig | None = None,
    deidentifier: Deidentifier | None = None,
) -> Any:
    """Create a LlamaIndex node postprocessor that redacts retrieved text."""

    base = _load_optional_class(
        "llama_index.core.postprocessor.types",
        "BaseNodePostprocessor",
        feature="node postprocessors",
    )
    redactor = _NodeRedactor(
        config=config or LlamaIndexRedactionConfig(),
        deidentifier=deidentifier,
        storage_safe=False,
    )

    class OpenMedRedactionPostprocessor(base):
        @classmethod
        def class_name(cls) -> str:
            return "OpenMedRedactionPostprocessor"

        def __init__(self) -> None:
            super().__init__()
            object.__setattr__(self, "_openmed_redactor", redactor)

        def _postprocess_nodes(
            self,
            nodes: list[Any],
            query_bundle: Any | None = None,
        ) -> list[Any]:
            del query_bundle
            return self._openmed_redactor.redact_scored_nodes(nodes)

        async def apostprocess_nodes(
            self,
            nodes: list[Any],
            query_bundle: Any | None = None,
            query_str: str | None = None,
        ) -> list[Any]:
            return await asyncio.to_thread(
                self.postprocess_nodes,
                nodes,
                query_bundle=query_bundle,
                query_str=query_str,
            )

    OpenMedRedactionPostprocessor.__module__ = __name__
    return OpenMedRedactionPostprocessor()


def create_redaction_transform(
    *,
    config: LlamaIndexRedactionConfig | None = None,
    deidentifier: Deidentifier | None = None,
    _cache_identity: str | None = None,
) -> Any:
    """Create an optional ingestion transform that redacts nodes before storage."""

    base = _load_optional_class(
        "llama_index.core.schema",
        "TransformComponent",
        feature="ingestion transforms",
    )
    resolved_config = config or LlamaIndexRedactionConfig()
    redactor = _NodeRedactor(
        config=resolved_config,
        deidentifier=deidentifier,
        storage_safe=True,
    )
    cache_identity = _cache_identity or _redaction_cache_identity(
        resolved_config,
        deidentifier,
    )

    class OpenMedRedactionTransform(base):
        @classmethod
        def class_name(cls) -> str:
            return "OpenMedRedactionTransform"

        def __init__(self) -> None:
            super().__init__()
            object.__setattr__(self, "_openmed_redactor", redactor)
            object.__setattr__(self, "_openmed_cache_identity", cache_identity)

        def __call__(self, nodes: Sequence[Any], **kwargs: Any) -> list[Any]:
            del kwargs
            return self._openmed_redactor.redact_nodes(nodes)

        def to_dict(self, **kwargs: Any) -> dict[str, Any]:
            del kwargs
            return {
                "class_name": self.class_name(),
                "openmed_cache_identity": self._openmed_cache_identity,
            }

        def __reduce__(self) -> tuple[Any, tuple[Any, ...]]:
            return (
                _rebuild_redaction_transform,
                (
                    self._openmed_redactor.config,
                    self._openmed_redactor._deidentifier,
                    self._openmed_cache_identity,
                ),
            )

    OpenMedRedactionTransform.__module__ = __name__
    return OpenMedRedactionTransform()


def _rebuild_redaction_transform(
    config: LlamaIndexRedactionConfig,
    deidentifier: Deidentifier | None,
    cache_identity: str,
) -> Any:
    return create_redaction_transform(
        config=config,
        deidentifier=deidentifier,
        _cache_identity=cache_identity,
    )


def _redaction_cache_identity(
    config: LlamaIndexRedactionConfig,
    deidentifier: Deidentifier | None,
) -> str:
    if deidentifier is not None:
        return f"custom:{uuid4().hex}"
    digest = sha256(repr(config).encode("utf-8")).hexdigest()
    return f"default:{digest}"


def create_tool_definitions() -> tuple[dict[str, Any], ...]:
    """Return LlamaIndex-facing OpenMed tool definitions from the registry."""

    return render_adapter_tool_definitions("llamaindex")


def get_llamaindex_tools(
    *,
    runtime_provider: RuntimeProvider | None = None,
) -> tuple[Any, ...]:
    """Return LlamaIndex ``FunctionTool`` objects for every registry tool."""

    function_tool = _load_function_tool()
    return tuple(
        _function_tool_from_spec(function_tool, spec, runtime_provider)
        for spec in registry_tool_specs()
    )


def _function_tool_from_spec(
    function_tool: Any,
    spec: Any,
    runtime_provider: RuntimeProvider | None,
) -> Any:
    func = create_tool_callable(spec, runtime_provider=runtime_provider)
    if hasattr(function_tool, "from_defaults"):
        return function_tool.from_defaults(
            fn=func,
            name=spec.name,
            description=spec.description,
        )
    if hasattr(function_tool, "from_function"):
        return function_tool.from_function(
            func=func,
            name=spec.name,
            description=spec.description,
        )
    raise ImportError("LlamaIndex tools require llama-index-core with FunctionTool.")


def _load_function_tool() -> Any:
    return _load_optional_class(
        "llama_index.core.tools",
        "FunctionTool",
        feature="tools",
    )


def _load_optional_class(module_name: str, class_name: str, *, feature: str) -> Any:
    try:
        module = _import_module(module_name)
    except ImportError as exc:
        raise ImportError(
            f"LlamaIndex {feature} require the 'llamaindex' extra. "
            "Install with `pip install openmed[llamaindex]`."
        ) from exc

    try:
        return getattr(module, class_name)
    except AttributeError as exc:
        raise ImportError(
            f"LlamaIndex {feature} require llama-index-core with {class_name}."
        ) from exc


def _clone(value: Any) -> Any:
    model_copy = getattr(value, "model_copy", None)
    if callable(model_copy):
        return model_copy(deep=True)
    return deepcopy(value)


def _node_text(node: Any) -> str | None:
    text = getattr(node, "text", None)
    if isinstance(text, str):
        return text

    get_content = getattr(node, "get_content", None)
    if callable(get_content):
        content = get_content()
        if isinstance(content, str):
            return content
    return None


def _set_node_text(node: Any, text: str) -> None:
    set_content = getattr(node, "set_content", None)
    if callable(set_content):
        set_content(text)
        return
    if hasattr(node, "text"):
        node.text = text
        return
    raise TypeError("LlamaIndex node does not expose mutable text content")


def _set_node_metadata(node: Any, metadata: dict[Any, Any]) -> None:
    try:
        node.metadata = metadata
    except (AttributeError, TypeError, ValueError) as exc:
        raise TypeError(
            "LlamaIndex node does not expose mutable metadata content"
        ) from exc


def _exclude_metadata_from_content(node: Any, metadata: Mapping[Any, Any]) -> None:
    keys = [key for key in metadata if isinstance(key, str)]
    for attribute in (
        "excluded_llm_metadata_keys",
        "excluded_embed_metadata_keys",
    ):
        current = getattr(node, attribute, None)
        if not isinstance(current, list):
            continue
        setattr(node, attribute, list(dict.fromkeys([*current, *keys])))


def _metadata_ancestors(
    value: Any,
    ancestors: frozenset[int],
) -> frozenset[int]:
    identity = id(value)
    if identity in ancestors:
        raise ValueError("LlamaIndex metadata must not contain cycles")
    return ancestors | {identity}


def _pseudonymized_identifier(value: str | int | float) -> str:
    typed_value = f"{type(value).__name__}:{value}"
    digest = sha256(typed_value.encode("utf-8")).hexdigest()
    return f"openmed-{digest}"


def _pseudonymized_node_id(value: str) -> str:
    return str(uuid5(_NODE_ID_NAMESPACE, value))


def _deidentified_text(result: Any) -> str:
    if isinstance(result, str):
        return result
    try:
        return str(result.deidentified_text)
    except AttributeError as exc:
        raise TypeError(
            "deidentifier must return a string or an object with deidentified_text"
        ) from exc


__all__ = [
    "Deidentifier",
    "LlamaIndexRedactionConfig",
    "create_redaction_postprocessor",
    "create_redaction_transform",
    "create_tool_definitions",
    "get_llamaindex_tools",
]
