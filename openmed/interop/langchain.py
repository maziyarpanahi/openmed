"""LangChain-compatible redaction transforms backed by OpenMed de-identification."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from copy import copy
from dataclasses import dataclass, field
from importlib import import_module as _import_module
from pathlib import Path
from typing import Any

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
    "create_redaction_runnable",
    "create_redaction_transform",
]
