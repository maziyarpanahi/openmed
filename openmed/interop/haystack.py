"""Haystack 2.x document redaction component backed by OpenMed."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from functools import lru_cache
from importlib import import_module as _import_module
from typing import Any


class _OpenMedRedactorBase:
    """Shared implementation for the lazily decorated Haystack component."""

    def __init__(
        self,
        *,
        deidentify_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        """Configure keyword arguments forwarded to OpenMed de-identification.

        Args:
            deidentify_kwargs: Optional overrides for
                :func:`openmed.core.pii.deidentify`. Masking, disabled mapping
                retention, and the deterministic safety sweep are enabled by
                default.
        """

        kwargs: dict[str, Any] = {
            "method": "mask",
            "keep_mapping": False,
            "use_safety_sweep": True,
        }
        if deidentify_kwargs is not None:
            kwargs.update(dict(deidentify_kwargs))
        self.deidentify_kwargs = kwargs

    def _run(
        self,
        documents: Sequence[Any],
        *,
        document_type: type[Any],
    ) -> dict[str, list[Any]]:
        redacted_documents = [
            self._redact_document(document, document_type=document_type)
            for document in documents
        ]
        return {"documents": redacted_documents}

    def _redact_document(
        self,
        document: Any,
        *,
        document_type: type[Any],
    ) -> Any:
        if not isinstance(document, document_type):
            raise TypeError("documents must contain Haystack Document instances")

        content = document.content
        redacted_content = content
        if content:
            result = _deidentify(content, **self.deidentify_kwargs)
            redacted_content = _deidentified_text(result)

        payload = document.to_dict(flatten=False)
        payload["content"] = redacted_content
        return document_type.from_dict(payload)


@lru_cache(maxsize=1)
def _component_class() -> type[Any]:
    """Build and cache the Haystack-decorated redactor class on first use."""

    try:
        haystack = _import_module("haystack")
        component = haystack.component
        document_type = haystack.Document
    except (ImportError, AttributeError) as exc:
        raise ImportError(
            "Haystack support requires the 'haystack' extra. "
            "Install with `pip install openmed[haystack]`."
        ) from exc

    class OpenMedRedactor(_OpenMedRedactorBase):
        """Redact Haystack document content before indexing or generation."""

        def run(self, documents: list[Any]) -> dict[str, list[Any]]:
            """Return copied Documents whose textual content is redacted."""

            return self._run(documents, document_type=document_type)

    document_list_type = list[document_type]
    OpenMedRedactor.run.__annotations__ = {
        "documents": document_list_type,
        "return": dict[str, document_list_type],
    }
    OpenMedRedactor.run = component.output_types(documents=document_list_type)(
        OpenMedRedactor.run
    )
    OpenMedRedactor.__module__ = __name__
    OpenMedRedactor.__qualname__ = "OpenMedRedactor"
    return component(OpenMedRedactor)


def _deidentify(text: str, **kwargs: Any) -> Any:
    from openmed.core.pii import deidentify

    return deidentify(text, **kwargs)


def _deidentified_text(result: Any) -> str:
    if isinstance(result, str):
        return result

    try:
        return str(result.deidentified_text)
    except AttributeError as exc:
        raise TypeError(
            "deidentify must return a string or an object with deidentified_text"
        ) from exc


OpenMedRedactor: Any


def __getattr__(name: str) -> Any:
    if name == "OpenMedRedactor":
        return _component_class()
    raise AttributeError(name)


__all__ = ["OpenMedRedactor"]
