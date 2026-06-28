"""Shared ingest/redact contract for the multimodal subsystem.

This module is the stable foundation every document/image/DICOM ingester builds
on. It deliberately avoids importing heavy ingestion dependencies
(pdfplumber/python-docx/Pillow) at module load time so that ``import openmed``
and ``import openmed.multimodal`` stay lightweight; ingesters import their own
dependencies lazily and surface a clear error when the ``multimodal`` extra is
absent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from .exceptions import MissingDependencyError, UnsupportedDocumentError

# Module name -> distribution name for the dependencies in the [multimodal] extra.
_MULTIMODAL_DEPENDENCIES: tuple[tuple[str, str], ...] = (
    ("pdfplumber", "pdfplumber"),
    ("docx", "python-docx"),
    ("PIL", "Pillow"),
)

_MULTIMODAL_INSTALL_HINT = 'Install with: pip install "openmed[multimodal]".'


def _missing_multimodal_dependencies() -> list[str]:
    """Return the distribution names of any missing multimodal dependencies."""
    import importlib.util

    return [
        distribution
        for module, distribution in _MULTIMODAL_DEPENDENCIES
        if importlib.util.find_spec(module) is None
    ]


def ensure_multimodal_available() -> None:
    """Raise a clear error if the ``multimodal`` extra is not installed."""
    missing = _missing_multimodal_dependencies()
    if missing:
        raise MissingDependencyError(
            dependency=", ".join(missing),
            instruction=_MULTIMODAL_INSTALL_HINT,
        )


@dataclass(frozen=True)
class SourceSpan:
    """Maps a character range in normalized text to a location in the source.

    ``start``/``end`` are character offsets into :attr:`ExtractedDocument.text`
    (``end`` exclusive). ``page`` is the 0-based source page and ``bbox`` is an
    optional ``(x0, y0, x1, y1)`` bounding box. ``metadata`` carries per-block
    details (block type, font, confidence, ...).
    """

    start: int
    end: int
    page: int = 0
    bbox: tuple[float, float, float, float] | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExtractedDocument:
    """Normalized text plus a char-offset -> source-location map.

    This is the common contract returned by every per-format ingester. It lets
    downstream redaction work on a single normalized string while still being
    able to map any character offset back to its location in the source.
    """

    text: str
    spans: tuple[SourceSpan, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_blocks(
        cls,
        blocks: Iterable[Mapping[str, Any]],
        *,
        separator: str = "\n",
        metadata: Mapping[str, Any] | None = None,
    ) -> "ExtractedDocument":
        """Assemble a document from ordered source blocks.

        Each block is a mapping with a required ``text`` key and optional
        ``page``, ``bbox`` and ``metadata`` keys. Blocks are joined with
        ``separator`` and a :class:`SourceSpan` is recorded for each.
        """
        parts: list[str] = []
        spans: list[SourceSpan] = []
        cursor = 0
        for index, block in enumerate(blocks):
            if index > 0:
                parts.append(separator)
                cursor += len(separator)
            block_text = str(block["text"])
            start = cursor
            cursor += len(block_text)
            parts.append(block_text)
            spans.append(
                SourceSpan(
                    start=start,
                    end=cursor,
                    page=int(block.get("page", 0)),
                    bbox=block.get("bbox"),
                    metadata=dict(block.get("metadata", {})),
                )
            )
        return cls(
            text="".join(parts),
            spans=tuple(spans),
            metadata=dict(metadata or {}),
        )

    def location_at(self, offset: int) -> SourceSpan | None:
        """Return the source span covering ``offset``, or ``None`` if unmapped."""
        if offset < 0 or offset >= len(self.text):
            return None
        for span in self.spans:
            if span.start <= offset < span.end:
                return span
        return None

    def text_for(self, span: SourceSpan) -> str:
        """Return the normalized text covered by ``span``."""
        return self.text[span.start : span.end]


# Document handlers are registered lazily by per-format ingester modules so the
# dispatcher never needs editing when a new format lands.
DocumentHandler = Callable[..., ExtractedDocument]
DocumentDetector = Callable[[str | Path], bool]


@dataclass(frozen=True)
class _HandlerSpec:
    handler: DocumentHandler
    detector: DocumentDetector | None = None
    requires_multimodal: bool = True


_HANDLERS: dict[str, list[_HandlerSpec]] = {}


def _normalize_extension(extension: str) -> str:
    extension = extension.lower()
    return extension if extension.startswith(".") else f".{extension}"


def register_handler(
    extensions: str | Iterable[str],
    handler: DocumentHandler,
    *,
    detector: DocumentDetector | None = None,
    requires_multimodal: bool = True,
) -> None:
    """Register ``handler`` for one or more file extensions (e.g. ``".pdf"``)."""
    if isinstance(extensions, str):
        extensions = [extensions]
    spec = _HandlerSpec(
        handler=handler,
        detector=detector,
        requires_multimodal=requires_multimodal,
    )
    for extension in extensions:
        _HANDLERS.setdefault(_normalize_extension(extension), []).append(spec)


def _select_handler(
    path: str | Path, specs: Iterable[_HandlerSpec]
) -> _HandlerSpec | None:
    ordered = sorted(tuple(specs), key=lambda spec: spec.detector is None)
    for spec in ordered:
        if spec.detector is None or spec.detector(path):
            return spec
    return None


def redact_document(
    path: str | Path,
    *,
    policy: Any | None = None,
    models: Any | None = None,
    lang: str | None = None,
) -> ExtractedDocument:
    """De-identify a document, dispatching by file extension to its ingester.

    Registered stdlib-only handlers may run without the full ``multimodal``
    extra. Unknown extensions still check the optional dependency set first so
    installs missing the extra keep surfacing the actionable install hint before
    reporting unsupported formats.

    ``lang`` is an optional OpenMed language code (e.g. ``"fr"``) configuring
    language-aware ingestion: image handlers pass it through to OCR so scanned
    documents are read in the right language. Handlers that do not use it accept
    and ignore it. Defaults to English-equivalent behavior when unset.
    """
    extension = Path(str(path)).suffix.lower()
    specs = _HANDLERS.get(extension)
    if specs is None:
        ensure_multimodal_available()
        supported = ", ".join(sorted(_HANDLERS)) or "none"
        raise UnsupportedDocumentError(
            f"No multimodal handler registered for extension "
            f"{extension or '(none)'!r}. Supported extensions: {supported}."
        )

    spec = _select_handler(path, specs)
    if spec is None:
        supported = ", ".join(sorted(_HANDLERS)) or "none"
        raise UnsupportedDocumentError(
            f"No registered handler matched content for extension "
            f"{extension or '(none)'!r}. Supported extensions: {supported}."
        )

    if spec.requires_multimodal:
        ensure_multimodal_available()
    return spec.handler(path, policy=policy, models=models, lang=lang)
