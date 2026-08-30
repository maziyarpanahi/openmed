"""Local EML and optional Outlook MSG ingestion with PHI redaction.

EML parsing uses only Python's standard-library :mod:`email` package. Outlook
MSG parsing is an explicit optional bridge to the GPL-licensed ``extract-msg``
package. The bridge runs in an isolated subprocess and exchanges message bytes
through pipes, so OpenMed neither imports GPL code into its process nor writes
raw message or attachment PHI to temporary files.
"""

from __future__ import annotations

import hashlib
import html as html_lib
import importlib.util
import re
import subprocess
import sys
import zipfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher
from email import encoders
from email import policy as email_policy
from email.errors import HeaderParseError
from email.header import decode_header
from email.message import Message
from email.parser import BytesParser
from html.parser import HTMLParser
from io import BytesIO
from pathlib import Path
from typing import Any, BinaryIO

from .base import ExtractedDocument, SourceSpan, redact_document, register_handler
from .exceptions import MissingDependencyError, UnsupportedDocumentError

_MSG_INSTALL_HINT = (
    'Install the isolated MSG bridge with: pip install "openmed[email-msg-gpl]".'
)
_MSG_BRIDGE_TIMEOUT_SECONDS = 60

_EXTRACTED_HEADERS = (
    "From",
    "To",
    "Cc",
    "Bcc",
    "Reply-To",
    "Sender",
    "Subject",
    "Received",
    "Return-Path",
    "Message-ID",
    "In-Reply-To",
    "References",
    "Date",
)
_STRUCTURAL_HEADERS = frozenset(
    {
        "content-type",
        "content-transfer-encoding",
        "content-disposition",
        "content-id",
        "content-location",
        "mime-version",
    }
)
_REMOVED_AUTH_HEADERS = frozenset(
    {
        "authentication-results",
        "dkim-signature",
        "domainkey-signature",
        "received-spf",
    }
)
_REMOVED_DATE_HEADERS = frozenset({"date", "delivery-date", "orig-date", "resent-date"})
_REMOVED_INTEGRITY_HEADERS = frozenset({"content-length", "content-md5"})
_HASHED_ID_HEADERS = frozenset({"message-id", "in-reply-to", "references"})
_PRESERVED_MESSAGE_HEADERS = frozenset(
    name.lower() for name in _EXTRACTED_HEADERS if name.lower() != "date"
)
_ADDRESS_HEADERS = frozenset({"bcc", "cc", "from", "reply-to", "sender", "to"})
_MESSAGE_ID_RE = re.compile(r"<[^<>]+>")

_HTML_BLOCK_TAGS = frozenset(
    {
        "address",
        "article",
        "aside",
        "blockquote",
        "br",
        "dd",
        "div",
        "dl",
        "dt",
        "figcaption",
        "figure",
        "footer",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "header",
        "hr",
        "li",
        "main",
        "nav",
        "ol",
        "p",
        "pre",
        "section",
        "table",
        "td",
        "th",
        "tr",
        "ul",
    }
)
_HTML_IGNORED_TAGS = frozenset({"script", "style", "template"})
_HTML_DROPPED_ATTRIBUTES = frozenset({"srcdoc", "style"})
_HTML_SAFE_TAGS = frozenset(
    {
        "a",
        "abbr",
        "address",
        "article",
        "aside",
        "b",
        "blockquote",
        "body",
        "br",
        "caption",
        "cite",
        "code",
        "col",
        "colgroup",
        "dd",
        "del",
        "details",
        "dfn",
        "div",
        "dl",
        "dt",
        "em",
        "figcaption",
        "figure",
        "footer",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "head",
        "header",
        "hr",
        "html",
        "i",
        "img",
        "ins",
        "kbd",
        "li",
        "main",
        "mark",
        "nav",
        "ol",
        "p",
        "pre",
        "q",
        "s",
        "samp",
        "section",
        "small",
        "span",
        "strong",
        "sub",
        "summary",
        "sup",
        "table",
        "tbody",
        "td",
        "tfoot",
        "th",
        "thead",
        "time",
        "title",
        "tr",
        "u",
        "ul",
        "var",
        "wbr",
    }
)
_HTML_SAFE_ATTRIBUTES = frozenset(
    {
        "align",
        "alt",
        "aria-describedby",
        "aria-hidden",
        "aria-label",
        "aria-labelledby",
        "class",
        "colspan",
        "dir",
        "height",
        "href",
        "id",
        "lang",
        "role",
        "rowspan",
        "src",
        "target",
        "title",
        "width",
    }
)

_ATTACHMENT_MIME_SUFFIXES = {
    "application/pdf": ".pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": (
        ".docx"
    ),
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": (
        ".pptx"
    ),
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/tiff": ".tiff",
    "message/rfc822": ".eml",
}
_SAFE_MIME_BY_SUFFIX = {
    ".docx": (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    ),
    ".eml": "message/rfc822",
    ".jpeg": "image/jpeg",
    ".jpg": "image/jpeg",
    ".msg": "application/vnd.ms-outlook",
    ".pdf": "application/pdf",
    ".png": "image/png",
    ".pptx": (
        "application/vnd.openxmlformats-officedocument.presentationml.presentation"
    ),
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
}
_MATERIALIZABLE_ATTACHMENT_SUFFIXES = frozenset(
    {
        ".docx",
        ".eml",
        ".jpeg",
        ".jpg",
        ".msg",
        ".pdf",
        ".png",
        ".pptx",
        ".tif",
        ".tiff",
    }
)
_REDACTED_BYTES_KEYS = (
    "redacted_attachment_bytes",
    "redacted_email_bytes",
    "redacted_image_bytes",
    "redacted_pdf_bytes",
)

TextRedactor = Callable[[str], Any]


@dataclass(frozen=True)
class EmailAttachmentReport:
    """PHI-safe processing evidence for one MIME attachment."""

    attachment_index: int
    extension: str
    content_type: str
    handler_format: str
    detected_span_count: int
    output_sha256: str

    def to_dict(self) -> dict[str, Any]:
        """Return counts and digests without filenames, paths, or content."""
        return {
            "attachment_index": self.attachment_index,
            "extension": self.extension,
            "content_type": self.content_type,
            "handler_format": self.handler_format,
            "detected_span_count": self.detected_span_count,
            "output_sha256": self.output_sha256,
        }


@dataclass(frozen=True)
class RedactedEmail:
    """A serialized clean EML plus normalized redacted text and safe evidence."""

    email_bytes: bytes
    document: ExtractedDocument
    source_format: str
    header_redaction_count: int
    body_redaction_count: int
    attachments: tuple[EmailAttachmentReport, ...] = ()
    output_path: Path | None = None

    def to_document(self) -> ExtractedDocument:
        """Bridge the redacted MIME result into the multimodal contract."""
        metadata = dict(self.document.metadata)
        metadata.update(
            {
                "format": self.source_format,
                "output_format": "eml",
                "header_redaction_count": self.header_redaction_count,
                "body_redaction_count": self.body_redaction_count,
                "attachment_count": len(self.attachments),
                "attachments": [item.to_dict() for item in self.attachments],
                "redacted_email_sha256": hashlib.sha256(self.email_bytes).hexdigest(),
                "redacted_email_bytes": self.email_bytes,
            }
        )
        if self.output_path is not None:
            metadata["output_suffix"] = self.output_path.suffix.lower()
        return ExtractedDocument(
            text=self.document.text,
            spans=self.document.spans,
            metadata=metadata,
        )


@dataclass(frozen=True)
class _TextEdit:
    start: int
    end: int
    replacement: str


@dataclass(frozen=True)
class _TextOutcome:
    text: str
    edits: tuple[_TextEdit, ...] = ()

    @property
    def changed(self) -> bool:
        return bool(self.edits)


@dataclass(frozen=True)
class _HtmlChunk:
    part_index: int
    visible_start: int
    visible_end: int
    raw_text: str
    entity_encoded: bool = False


class _NamedBytesIO(BytesIO):
    """Seekable in-memory attachment source with a dispatcher-safe name."""

    def __init__(self, data: bytes = b"", *, name: str) -> None:
        super().__init__(data)
        self.name = name


def extract_email(
    source: str | Path | bytes | bytearray | BinaryIO,
) -> ExtractedDocument:
    """Extract decoded email headers and text bodies with character maps.

    ``.eml`` inputs are parsed with the standard library. ``.msg`` inputs use
    the optional isolated ``extract-msg`` bridge and are normalized to the same
    RFC 5322/MIME representation before extraction.

    Args:
        source: EML/MSG path, raw EML bytes, or a named binary stream.

    Returns:
        Normalized decoded header, ``text/plain``, and visible ``text/html``
        segments. Every mapped span identifies its header or MIME part without
        copying raw PHI into metadata.

    Raises:
        MissingDependencyError: If a ``.msg`` input is used without the
            ``email-msg-gpl`` bridge extra.
    """
    payload, source_format = _email_payload(source)
    message = _parse_eml(payload)
    return _document_from_message(message, source_format=source_format)


def redact_email(
    source: str | Path | bytes | bytearray | BinaryIO,
    output_path: str | Path | BinaryIO | None = None,
    *,
    policy: Any | None = None,
    models: Any | None = None,
    lang: str | None = None,
) -> RedactedEmail:
    """Redact headers, multipart bodies, and supported email attachments.

    Header values and decoded text bodies are routed through a caller-supplied
    OpenMed detector/redactor when present. The deterministic safety sweep is
    always applied as a residual structured-identifier backstop. Attachments
    are dispatched entirely in memory through :func:`redact_document`; no raw
    attachment is written to a temporary file.

    Args:
        source: EML/MSG path, raw EML bytes, or a named binary stream.
        output_path: Optional EML destination or writable binary stream. MSG
            input is intentionally emitted as EML because ``extract-msg`` is a
            read-only parser.
        policy: OpenMed policy name or mapping. ``method``/
            ``deidentify_method`` and ``deidentify_policy`` are honored.
        models: OpenMed PII detector/redactor, or a mapping exposing
            ``detector``/``extract_pii`` and optionally ``text_redactor``.
        lang: OpenMed language code; defaults to ``"en"``.

    Returns:
        A :class:`RedactedEmail` containing clean MIME bytes and PHI-safe
        processing evidence.

    Raises:
        UnsupportedDocumentError: If an attachment cannot produce a clean
            replacement artifact. Unsupported attachments fail closed.
        MissingDependencyError: If an optional format dependency is absent.
    """
    payload, source_format = _email_payload(source)
    message = _parse_eml(payload)
    resolved_lang = lang or "en"
    processor = _TextProcessor(models=models, policy=policy, lang=resolved_lang)

    header_redactions = sum(
        _redact_headers(part, processor) for part in _iter_mime_parts(message)
    )
    attachments, cid_map = _redact_attachments(
        message,
        processor=processor,
        policy=policy,
        models=models,
        lang=resolved_lang,
    )
    body_redactions = _redact_body_parts(message, processor, cid_map=cid_map)
    _sanitize_mime_metadata(message)

    email_bytes = message.as_bytes(policy=email_policy.SMTP)
    resolved_output = _write_email_output(output_path, email_bytes)
    redacted_message = _parse_eml(email_bytes)
    document = _document_from_message(
        redacted_message,
        source_format=source_format,
    )
    return RedactedEmail(
        email_bytes=email_bytes,
        document=document,
        source_format=source_format,
        header_redaction_count=header_redactions,
        body_redaction_count=body_redactions,
        attachments=attachments,
        output_path=resolved_output,
    )


def _email_payload(
    source: str | Path | bytes | bytearray | BinaryIO,
) -> tuple[bytes, str]:
    payload, name = _read_source_bytes(source)
    suffix = Path(name).suffix.lower() if name else ".eml"
    if suffix == ".msg":
        return _convert_msg_to_eml_bytes(payload), "msg"
    return payload, "eml"


def _read_source_bytes(
    source: str | Path | bytes | bytearray | BinaryIO,
) -> tuple[bytes, str | None]:
    if isinstance(source, (bytes, bytearray)):
        return bytes(source), None
    if hasattr(source, "read"):
        stream = source
        try:
            stream.seek(0)
        except (AttributeError, OSError):
            pass
        data = stream.read()
        if isinstance(data, str):
            data = data.encode("utf-8")
        if not isinstance(data, (bytes, bytearray)):
            raise TypeError("email stream must return bytes or text")
        return bytes(data), _safe_stream_name(stream)
    path = Path(source)
    return path.read_bytes(), path.name


def _safe_stream_name(stream: Any) -> str | None:
    name = getattr(stream, "name", None)
    if isinstance(name, (str, Path)):
        return Path(str(name)).name
    return None


def _parse_eml(payload: bytes) -> Message:
    return BytesParser(policy=email_policy.default).parsebytes(payload)


def _raw_header_values(message: Message, header_name: str) -> tuple[Any, ...]:
    """Return header values without invoking strict structured-header parsing."""

    normalized = header_name.lower()
    return tuple(
        value for name, value in message.raw_items() if name.lower() == normalized
    )


def _ensure_msg_available() -> None:
    if importlib.util.find_spec("extract_msg") is None:
        raise MissingDependencyError(
            dependency="extract-msg",
            instruction=_MSG_INSTALL_HINT,
        )


def _convert_msg_to_eml_bytes(payload: bytes) -> bytes:
    _ensure_msg_available()
    bridge = """
import sys
from email import policy
from io import BytesIO

import extract_msg

message = extract_msg.openMsg(BytesIO(sys.stdin.buffer.read()))
try:
    converter = getattr(message, "asEmailMessage")
    converted = converter() if callable(converter) else converter
    sys.stdout.buffer.write(converted.as_bytes(policy=policy.SMTP))
finally:
    close = getattr(message, "close", None)
    if callable(close):
        close()
"""
    try:
        completed = subprocess.run(  # nosec B603 - fixed interpreter/script.
            [sys.executable, "-I", "-c", bridge],
            input=payload,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=_MSG_BRIDGE_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise UnsupportedDocumentError(
            "The isolated extract-msg bridge could not parse this MSG input."
        ) from exc
    if completed.returncode != 0 or not completed.stdout:
        raise UnsupportedDocumentError(
            "The isolated extract-msg bridge rejected this MSG input."
        )
    return bytes(completed.stdout)


def _document_from_message(
    message: Message, *, source_format: str
) -> ExtractedDocument:
    parts: list[str] = []
    spans: list[SourceSpan] = []
    sections: list[dict[str, Any]] = []
    cursor = 0

    def append_segment(
        text: str,
        metadata: Mapping[str, Any],
        *,
        local_spans: Sequence[SourceSpan] | None = None,
    ) -> None:
        nonlocal cursor
        if not text:
            return
        if parts:
            parts.append("\n")
            cursor += 1
        section_start = cursor
        parts.append(text)
        cursor += len(text)
        section_end = cursor
        sections.append(
            {
                "start": section_start,
                "end": section_end,
                "block_type": str(metadata.get("block_type", "segment")),
            }
        )
        if local_spans is None:
            spans.append(
                SourceSpan(
                    start=section_start,
                    end=section_end,
                    metadata=dict(metadata),
                )
            )
            return
        for local in local_spans:
            local_start = max(0, min(local.start, len(text)))
            local_end = max(0, min(local.end, len(text)))
            if local_end <= local_start:
                continue
            spans.append(
                SourceSpan(
                    start=section_start + local_start,
                    end=section_start + local_end,
                    metadata={**dict(metadata), **dict(local.metadata)},
                )
            )

    for header_name in _EXTRACTED_HEADERS:
        for header_index, value in enumerate(_raw_header_values(message, header_name)):
            decoded = _decoded_header_value(value)
            if decoded:
                append_segment(
                    f"{header_name}: {decoded}",
                    {
                        "format": source_format,
                        "block_type": "header",
                        "header_name": header_name.lower(),
                        "header_index": header_index,
                    },
                )

    body_count = 0
    attachment_count = 0
    for part_index, part in enumerate(_iter_content_parts(message)):
        if _is_attachment(part):
            attachment_count += 1
            continue
        content_type = part.get_content_type().lower()
        if content_type not in {"text/plain", "text/html"}:
            continue
        decoded = _decoded_text_part(part)
        if not decoded:
            continue
        body_count += 1
        metadata = {
            "format": source_format,
            "block_type": "body",
            "part_index": part_index,
            "content_type": content_type,
        }
        if content_type == "text/html":
            parsed = _parse_html_text(decoded)
            append_segment(parsed.text, metadata, local_spans=parsed.spans)
        else:
            append_segment(decoded, metadata)

    return ExtractedDocument(
        text="".join(parts),
        spans=tuple(spans),
        metadata={
            "format": source_format,
            "header_count": sum(
                len(_raw_header_values(message, name)) for name in _EXTRACTED_HEADERS
            ),
            "body_part_count": body_count,
            "attachment_count": attachment_count,
            "sections": sections,
        },
    )


def _decoded_header_value(value: Any) -> str:
    # ``raw_items`` preserves legal folding whitespace. Unfold before decoding
    # so the redacted value can be stored without permitting header injection.
    text = re.sub(r"(?:\r\n|\r|\n)[ \t]+", " ", str(value))
    text = text.replace("\r", " ").replace("\n", " ")
    decoded_parts: list[str] = []
    try:
        fragments = decode_header(text)
    except (LookupError, ValueError):
        return text
    for fragment, charset in fragments:
        if isinstance(fragment, bytes):
            encoding = charset or "utf-8"
            try:
                decoded_parts.append(fragment.decode(encoding, errors="replace"))
            except LookupError:
                decoded_parts.append(fragment.decode("utf-8", errors="replace"))
        else:
            decoded_parts.append(fragment)
    return "".join(decoded_parts)


def _decoded_text_part(part: Message) -> str:
    try:
        content = part.get_content()
    except (LookupError, UnicodeError):
        content = None
    if isinstance(content, str):
        return content
    payload = part.get_payload(decode=True)
    if not isinstance(payload, bytes):
        return ""
    charset = part.get_content_charset() or "utf-8"
    try:
        return payload.decode(charset, errors="replace")
    except LookupError:
        return payload.decode("utf-8", errors="replace")


def _address_header_is_valid(header_name: str, value: str) -> bool:
    """Return whether the stdlib accepts an address without parser defects."""

    candidate = Message(policy=email_policy.default)
    try:
        candidate[header_name] = value
        parsed = candidate[header_name]
    except (HeaderParseError, IndexError, KeyError, TypeError, ValueError):
        return False
    return not getattr(parsed, "defects", ())


def _redact_headers(message: Message, processor: "_TextProcessor") -> int:
    changed = 0
    raw_headers = tuple(message.raw_items())
    ordered_names: dict[str, str] = {}
    for header_name, _ in raw_headers:
        ordered_names.setdefault(header_name.lower(), header_name)
    for normalized, header_name in ordered_names.items():
        values = tuple(
            value for name, value in raw_headers if name.lower() == normalized
        )
        if normalized in _STRUCTURAL_HEADERS:
            continue
        del message[header_name]
        if (
            normalized in _REMOVED_AUTH_HEADERS
            or normalized in _REMOVED_DATE_HEADERS
            or normalized in _REMOVED_INTEGRITY_HEADERS
            or normalized.startswith("arc-")
        ):
            changed += len(values)
            continue
        if normalized not in _PRESERVED_MESSAGE_HEADERS:
            changed += len(values)
            continue
        for value in values:
            decoded = _decoded_header_value(value)
            if normalized in _HASHED_ID_HEADERS:
                redacted = _redact_message_ids(decoded, processor)
            else:
                redacted = processor.redact(decoded).text
            value_changed = redacted != decoded
            if normalized in _ADDRESS_HEADERS and not _address_header_is_valid(
                header_name, redacted
            ):
                redacted = "redacted-address@openmed.invalid"
                value_changed = True
            if value_changed:
                changed += 1
            try:
                message[header_name] = redacted
            except (HeaderParseError, IndexError, KeyError, TypeError, ValueError):
                # Some malformed address values trigger exceptions inside the
                # standard-library header registry. Emit a safe valid address
                # instead of preserving an unparsable or potentially private
                # raw value in the clean message.
                fallback = (
                    "redacted-address@openmed.invalid"
                    if normalized in _ADDRESS_HEADERS
                    else "[REDACTED]"
                )
                message[header_name] = fallback
                if not value_changed:
                    changed += 1
    return changed


def _redact_message_ids(value: str, processor: "_TextProcessor") -> str:
    matches = tuple(_MESSAGE_ID_RE.finditer(value))
    if matches:
        redacted_ids = []
        for match in matches:
            digest = hashlib.sha256(match.group(0).encode("utf-8")).hexdigest()[:16]
            redacted_ids.append(f"<message-{digest}@openmed.invalid>")
        return " ".join(redacted_ids)
    return processor.redact(value).text


def _redact_body_parts(
    message: Message,
    processor: "_TextProcessor",
    *,
    cid_map: Mapping[str, str],
) -> int:
    changed = 0
    for part in _iter_content_parts(message):
        if _is_attachment(part):
            continue
        content_type = part.get_content_type().lower()
        if content_type not in {"text/plain", "text/html"}:
            continue
        source = _decoded_text_part(part)
        if content_type == "text/html":
            redacted = _redact_html(source, processor, cid_map=cid_map)
        else:
            redacted = processor.redact(source).text
        if redacted != source:
            changed += 1
        _set_text_part(part, redacted)
    return changed


def _set_text_part(part: Message, content: str) -> None:
    subtype = part.get_content_subtype() or "plain"
    cte = str(part.get("Content-Transfer-Encoding", "")).lower()
    kwargs: dict[str, Any] = {"subtype": subtype, "charset": "utf-8"}
    if cte in {"7bit", "8bit", "base64", "quoted-printable"}:
        kwargs["cte"] = cte
    part.set_content(content, **kwargs)


def _redact_attachments(
    message: Message,
    *,
    processor: "_TextProcessor",
    policy: Any | None,
    models: Any | None,
    lang: str,
) -> tuple[tuple[EmailAttachmentReport, ...], dict[str, str]]:
    reports: list[EmailAttachmentReport] = []
    cid_map: dict[str, str] = {}
    attachment_index = 0
    for part in _iter_content_parts(message):
        if not _is_attachment(part):
            continue
        attachment_index += 1
        payload = _attachment_payload(part)
        if not isinstance(payload, bytes):
            raise UnsupportedDocumentError(
                f"Attachment {attachment_index} has no decodable binary payload."
            )
        extension = _attachment_extension(part)
        if extension not in _MATERIALIZABLE_ATTACHMENT_SUFFIXES:
            raise UnsupportedDocumentError(
                f"Attachment {attachment_index} has no supported clean-output "
                "handler; remove it or register one before emitting a clean email."
            )

        output_extension = ".eml" if extension == ".msg" else extension
        source_filename = f"attachment-{attachment_index:04d}{extension}"
        safe_filename = f"attachment-{attachment_index:04d}{output_extension}"
        source_buffer = _NamedBytesIO(payload, name=source_filename)
        output_buffer = _NamedBytesIO(
            name=f"attachment-{attachment_index:04d}.redacted{output_extension}"
        )
        attachment_policy = _attachment_policy(policy, output_buffer=output_buffer)
        attachment_models = _attachment_models(models, lang=lang)
        document = redact_document(
            source_buffer,
            policy=attachment_policy,
            models=attachment_models,
            lang=lang,
        )
        redacted_bytes = _redacted_attachment_bytes(document, output_buffer)
        if redacted_bytes is None:
            raise UnsupportedDocumentError(
                f"Attachment {attachment_index} was inspected but its "
                "handler did not emit a clean replacement artifact."
            )

        if extension in {".docx", ".pptx"}:
            redacted_bytes = _scrub_office_metadata(redacted_bytes)

        original_cid = str(part.get("Content-ID", "")).strip().strip("<>")
        if original_cid:
            safe_cid = f"attachment-{attachment_index:04d}@openmed.invalid"
            cid_map[original_cid] = safe_cid
        else:
            safe_cid = None
        _replace_attachment_payload(
            part,
            redacted_bytes,
            safe_filename=safe_filename,
            safe_cid=safe_cid,
        )

        metadata = document.metadata
        reports.append(
            EmailAttachmentReport(
                attachment_index=attachment_index,
                extension=output_extension,
                content_type=part.get_content_type().lower(),
                handler_format=str(metadata.get("format", extension.lstrip("."))),
                detected_span_count=_safe_count(metadata.get("detected_span_count", 0)),
                output_sha256=hashlib.sha256(redacted_bytes).hexdigest(),
            )
        )

    return tuple(reports), cid_map


def _iter_content_parts(message: Message) -> Iterable[Message]:
    """Yield MIME leaves without descending into attached messages."""
    payload = message.get_payload()
    children = payload if message.is_multipart() and isinstance(payload, list) else ()
    for part in children or (message,):
        if _is_attachment(part):
            yield part
        elif part.is_multipart():
            yield from _iter_content_parts(part)
        else:
            yield part


def _iter_mime_parts(message: Message) -> Iterable[Message]:
    """Yield a message and its non-attachment MIME descendants."""
    yield message
    payload = message.get_payload()
    if not message.is_multipart() or _is_attachment(message):
        return
    if isinstance(payload, list):
        for child in payload:
            yield from _iter_mime_parts(child)


def _attachment_payload(part: Message) -> bytes | None:
    payload = part.get_payload(decode=True)
    if isinstance(payload, bytes):
        return payload
    if part.get_content_type().lower() == "message/rfc822":
        nested = part.get_payload()
        if isinstance(nested, list) and nested and isinstance(nested[0], Message):
            return nested[0].as_bytes(policy=email_policy.SMTP)
    return None


def _is_attachment(part: Message) -> bool:
    return (
        part.get_content_disposition() == "attachment"
        or part.get_filename() is not None
        or (
            not part.is_multipart()
            and part.get_content_type().lower() not in {"text/html", "text/plain"}
        )
    )


def _attachment_extension(part: Message) -> str:
    filename = part.get_filename()
    suffix = Path(filename).suffix.lower() if filename else ""
    if suffix in _MATERIALIZABLE_ATTACHMENT_SUFFIXES:
        return suffix
    return _ATTACHMENT_MIME_SUFFIXES.get(part.get_content_type().lower(), suffix)


def _attachment_policy(
    policy: Any | None, *, output_buffer: BinaryIO
) -> dict[str, Any]:
    if isinstance(policy, Mapping):
        resolved = {
            str(key): value
            for key, value in policy.items()
            if str(key) not in {"output_path", "redacted_path", "destination_path"}
        }
    elif isinstance(policy, str):
        resolved = {"deidentify_policy": policy}
    else:
        resolved = {}
    resolved["output_path"] = output_buffer
    resolved["return_bytes"] = True
    return resolved


def _attachment_models(models: Any | None, *, lang: str) -> dict[str, Any]:
    resolved: dict[str, Any] = dict(models) if isinstance(models, Mapping) else {}
    for name in ("ocr_engine", "verification_ocr_engine", "verify"):
        if name not in resolved:
            value = getattr(models, name, None)
            if value is not None:
                resolved[name] = value
    resolved["detector"] = _swept_detector(models, lang=lang)
    return resolved


def _swept_detector(models: Any | None, *, lang: str) -> Callable[..., Any]:
    detector = _resolve_detector(models)
    default_lang = lang

    def detect(text: str, *, lang: str | None = None) -> dict[str, Any]:
        resolved_lang = lang or default_lang
        if detector is None:
            if models is not None:
                raise TypeError(
                    "Binary attachment redaction requires a detector via "
                    "models['detector'] or an object exposing extract_pii."
                )
            from openmed.core.pii import extract_pii

            result = extract_pii(text, lang=resolved_lang)
        else:
            result = _call_text_callable(detector, text, lang=resolved_lang)
        entities = _iter_entities(result)
        from openmed.core.safety_sweep import safety_sweep

        return {"entities": safety_sweep(text, entities, lang=resolved_lang)}

    return detect


def _redacted_attachment_bytes(
    document: ExtractedDocument,
    output_buffer: _NamedBytesIO,
) -> bytes | None:
    for key in _REDACTED_BYTES_KEYS:
        value = document.metadata.get(key)
        if isinstance(value, (bytes, bytearray)):
            return bytes(value)
    materialized = output_buffer.getvalue()
    return materialized or None


def _scrub_office_metadata(payload: bytes) -> bytes:
    """Strip OOXML core/app/custom properties without a PHI-bearing temp file."""
    from .metadata_scrub import _DOCX_PARTS, _scrub_docx_part

    source = BytesIO(payload)
    output = BytesIO()
    try:
        with (
            zipfile.ZipFile(source) as archive,
            zipfile.ZipFile(
                output,
                "w",
                compression=zipfile.ZIP_DEFLATED,
            ) as scrubbed,
        ):
            for member in archive.infolist():
                data = archive.read(member.filename)
                container = _DOCX_PARTS.get(member.filename)
                if container is not None:
                    data = _scrub_docx_part(
                        data,
                        container=container,
                        allowlist=frozenset(),
                    )
                scrubbed.writestr(member.filename, data)
    except (OSError, zipfile.BadZipFile) as exc:
        raise UnsupportedDocumentError(
            "An Office attachment could not be safely metadata-scrubbed."
        ) from exc
    return output.getvalue()


def _replace_attachment_payload(
    part: Message,
    payload: bytes,
    *,
    safe_filename: str,
    safe_cid: str | None,
) -> None:
    disposition = part.get_content_disposition() or "attachment"
    extension = Path(safe_filename).suffix.lower()
    # Attachment headers are untrusted metadata and can carry filenames,
    # descriptions, routing data, or arbitrary PHI. Rebuild the safe minimum.
    for header_name in tuple(dict.fromkeys(part.keys())):
        del part[header_name]
    part["Content-Type"] = _SAFE_MIME_BY_SUFFIX[extension]
    if extension == ".eml":
        part.set_payload([_parse_eml(payload)])
    else:
        part.set_payload(payload)
        encoders.encode_base64(part)
    part.add_header("Content-Disposition", disposition, filename=safe_filename)
    part.set_param("name", safe_filename, header="Content-Type", replace=True)
    if safe_cid is not None:
        part["Content-ID"] = f"<{safe_cid}>"


def _sanitize_mime_metadata(message: Message) -> None:
    part_index = 0

    def sanitize(part: Message) -> None:
        nonlocal part_index
        current_index = part_index
        part_index += 1
        if part.is_multipart():
            part.preamble = None
            part.epilogue = None
        if part.is_multipart() and not _is_attachment(part):
            content_type = part.get_content_type().lower()
            subtype = content_type.partition("/")[2]
            if subtype not in {"alternative", "digest", "mixed", "related"}:
                content_type = "multipart/mixed"
            if part.get("Content-Type") is not None:
                del part["Content-Type"]
            part["Content-Type"] = content_type
        if not _is_attachment(part):
            for header_name in (
                "Content-Description",
                "Content-Disposition",
                "Content-Location",
            ):
                if part.get(header_name) is not None:
                    del part[header_name]
            if part.get("Content-ID") is not None:
                del part["Content-ID"]
                part["Content-ID"] = f"<body-{current_index:04d}@openmed.invalid>"

        payload = part.get_payload()
        if part.is_multipart() and isinstance(payload, list):
            for child in payload:
                sanitize(child)

    sanitize(message)


def _safe_count(value: Any) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _write_email_output(
    output: str | Path | BinaryIO | None,
    payload: bytes,
) -> Path | None:
    if output is None:
        return None
    if hasattr(output, "write"):
        try:
            output.seek(0)
            output.truncate()
        except (AttributeError, OSError):
            pass
        output.write(payload)
        try:
            output.seek(0)
        except (AttributeError, OSError):
            pass
        return None
    path = Path(output)
    if path.suffix.lower() != ".eml":
        raise ValueError("redacted email output must use the .eml extension")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


class _TextProcessor:
    def __init__(self, *, models: Any | None, policy: Any | None, lang: str) -> None:
        self.models = models
        self.lang = lang
        self.method, self.deidentify_policy = _redaction_policy(policy)

    def redact(self, text: str) -> _TextOutcome:
        if not text or not text.strip():
            return _TextOutcome(text=text)

        redactor = _resolve_text_redactor(self.models)
        detector = _resolve_detector(self.models)
        if redactor is not None:
            result = _call_text_callable(redactor, text, lang=self.lang)
            transformed = _result_redacted_text(result)
            if transformed is None:
                entities = _iter_entities(result)
                transformed = _apply_entity_redactions(
                    text,
                    _sweep_entities(text, entities, lang=self.lang),
                )
        elif detector is not None:
            result = _call_text_callable(detector, text, lang=self.lang)
            transformed = _apply_entity_redactions(
                text,
                _sweep_entities(text, _iter_entities(result), lang=self.lang),
            )
        else:
            from openmed.core.pii import deidentify

            kwargs: dict[str, Any] = {
                "method": self.method,
                "lang": self.lang,
                "use_safety_sweep": True,
            }
            if self.deidentify_policy is not None:
                kwargs["policy"] = self.deidentify_policy
            model_name = _model_option(self.models, "model_name")
            if model_name is not None:
                kwargs["model_name"] = str(model_name)
            result = deidentify(text, **kwargs)
            transformed = _result_redacted_text(result)
            if transformed is None:
                raise TypeError("OpenMed deidentify did not return deidentified text")

        transformed = _apply_entity_redactions(
            transformed,
            _sweep_entities(transformed, (), lang=self.lang),
        )
        return _TextOutcome(
            text=transformed,
            edits=_text_edits(text, transformed),
        )


def _redaction_policy(policy: Any | None) -> tuple[str, str | None]:
    if isinstance(policy, str):
        return "mask", policy
    if isinstance(policy, Mapping):
        method = policy.get("deidentify_method", policy.get("method", "mask"))
        profile = policy.get("deidentify_policy")
        return str(method), str(profile) if profile is not None else None
    return "mask", None


def _resolve_text_redactor(models: Any | None) -> TextRedactor | None:
    if callable(models):
        return models
    if isinstance(models, Mapping):
        for name in ("text_redactor", "deidentifier", "redactor"):
            candidate = models.get(name)
            if callable(candidate):
                return candidate
        return None
    for name in ("text_redactor", "deidentifier", "redactor"):
        candidate = getattr(models, name, None)
        if callable(candidate):
            return candidate
    return None


def _resolve_detector(models: Any | None) -> Callable[..., Any] | None:
    if models is None:
        return None
    if callable(models):
        return models
    if isinstance(models, Mapping):
        for name in ("detector", "extract_pii", "analyze_text", "predict_entities"):
            candidate = models.get(name)
            if callable(candidate):
                return candidate
        return None
    for name in (
        "detect",
        "extract_pii",
        "analyze_text",
        "predict_entities",
        "predict",
    ):
        candidate = getattr(models, name, None)
        if callable(candidate):
            return candidate
    return None


def _model_option(models: Any | None, name: str) -> Any:
    if isinstance(models, Mapping):
        return models.get(name)
    return getattr(models, name, None)


def _call_text_callable(
    callable_: Callable[..., Any],
    text: str,
    *,
    lang: str,
) -> Any:
    try:
        return callable_(text, lang=lang)
    except TypeError:
        return callable_(text)


def _result_redacted_text(result: Any) -> str | None:
    if isinstance(result, str):
        return result
    if isinstance(result, Mapping):
        value = result.get("deidentified_text", result.get("redacted_text"))
    else:
        value = getattr(
            result,
            "deidentified_text",
            getattr(result, "redacted_text", None),
        )
    return str(value) if value is not None else None


def _iter_entities(result: Any) -> tuple[Any, ...]:
    if result is None or isinstance(result, str):
        return ()
    if isinstance(result, Mapping):
        for name in ("entities", "pii_entities", "spans"):
            entities = result.get(name)
            if entities is not None:
                return tuple(entities)
        if "start" in result and "end" in result:
            return (result,)
        return ()
    for name in ("entities", "pii_entities", "spans"):
        entities = getattr(result, name, None)
        if entities is not None:
            return tuple(entities)
    if isinstance(result, Iterable) and not isinstance(result, (bytes, bytearray, str)):
        return tuple(result)
    return ()


def _sweep_entities(text: str, entities: Sequence[Any], *, lang: str) -> list[Any]:
    from openmed.core.safety_sweep import safety_sweep

    return safety_sweep(text, entities, lang=lang)


def _apply_entity_redactions(text: str, entities: Sequence[Any]) -> str:
    edits: list[_TextEdit] = []
    for entity in entities:
        edit = _entity_edit(entity, text_length=len(text))
        if edit is not None:
            edits.append(edit)
    redacted = text
    for edit in sorted(edits, key=lambda item: (item.start, item.end), reverse=True):
        redacted = redacted[: edit.start] + edit.replacement + redacted[edit.end :]
    return redacted


def _entity_edit(entity: Any, *, text_length: int) -> _TextEdit | None:
    if isinstance(entity, Mapping):
        start = entity.get("start")
        end = entity.get("end")
        label = entity.get(
            "label", entity.get("entity_type", entity.get("entity_group"))
        )
        replacement = entity.get(
            "redacted_text",
            entity.get("replacement", entity.get("surrogate")),
        )
    else:
        start = getattr(entity, "start", None)
        end = getattr(entity, "end", None)
        label = getattr(
            entity,
            "label",
            getattr(entity, "entity_type", getattr(entity, "entity_group", None)),
        )
        replacement = getattr(
            entity,
            "redacted_text",
            getattr(entity, "replacement", getattr(entity, "surrogate", None)),
        )
    try:
        start_int = int(start)
        end_int = int(end)
    except (TypeError, ValueError):
        return None
    if start_int < 0 or end_int <= start_int or end_int > text_length:
        return None
    replacement_text = str(replacement) if replacement is not None else _mask(label)
    return _TextEdit(start=start_int, end=end_int, replacement=replacement_text)


def _mask(label: Any) -> str:
    safe = re.sub(r"[^A-Za-z0-9]+", "_", str(label or "PHI").upper()).strip("_")
    return f"[{safe or 'PHI'}]"


def _text_edits(original: str, transformed: str) -> tuple[_TextEdit, ...]:
    if transformed == original:
        return ()
    matcher = SequenceMatcher(a=original, b=transformed, autojunk=False)
    return tuple(
        _TextEdit(start=i1, end=i2, replacement=transformed[j1:j2])
        for tag, i1, i2, j1, j2 in matcher.get_opcodes()
        if tag != "equal"
    )


@dataclass(frozen=True)
class _ParsedHtml:
    text: str
    spans: tuple[SourceSpan, ...] = ()


def _parse_html_text(source: str) -> _ParsedHtml:
    parser = _HtmlTextParser(source)
    parser.feed(source)
    parser.close()
    return parser.document()


class _HtmlTextParser(HTMLParser):
    def __init__(self, source: str) -> None:
        super().__init__(convert_charrefs=False)
        self.source = source
        self.line_offsets = _line_offsets(source)
        self.parts: list[str] = []
        self.spans: list[SourceSpan] = []
        self.cursor = 0
        self.ignore_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        normalized = tag.lower()
        if normalized in _HTML_IGNORED_TAGS:
            self.ignore_depth += 1
            return
        if not self.ignore_depth and normalized in _HTML_BLOCK_TAGS:
            self._append_break()

    def handle_startendtag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        if not self.ignore_depth and tag.lower() in _HTML_BLOCK_TAGS:
            self._append_break()

    def handle_endtag(self, tag: str) -> None:
        normalized = tag.lower()
        if normalized in _HTML_IGNORED_TAGS and self.ignore_depth:
            self.ignore_depth -= 1
            return
        if not self.ignore_depth and normalized in _HTML_BLOCK_TAGS:
            self._append_break()

    def handle_data(self, data: str) -> None:
        if self.ignore_depth or not data:
            return
        source_start = self._source_offset()
        self._append_mapped(data, source_start, source_start + len(data))

    def handle_entityref(self, name: str) -> None:
        self._append_reference(f"&{name};")

    def handle_charref(self, name: str) -> None:
        self._append_reference(f"&#{name};")

    def document(self) -> _ParsedHtml:
        while self.parts and self.parts[-1] == "\n":
            self.parts.pop()
            self.cursor -= 1
        return _ParsedHtml(text="".join(self.parts), spans=tuple(self.spans))

    def _append_reference(self, raw: str) -> None:
        if self.ignore_depth:
            return
        source_start = self._source_offset()
        self._append_mapped(
            html_lib.unescape(raw),
            source_start,
            source_start + len(raw),
        )

    def _append_break(self) -> None:
        if not self.parts or self.parts[-1].endswith("\n"):
            return
        self.parts.append("\n")
        self.cursor += 1

    def _append_mapped(self, text: str, source_start: int, source_end: int) -> None:
        if not text:
            return
        start = self.cursor
        self.parts.append(text)
        self.cursor += len(text)
        self.spans.append(
            SourceSpan(
                start=start,
                end=self.cursor,
                metadata={
                    "html_source_start": source_start,
                    "html_source_end": source_end,
                },
            )
        )

    def _source_offset(self) -> int:
        line, column = self.getpos()
        line_index = max(0, min(line - 1, len(self.line_offsets) - 1))
        return min(len(self.source), self.line_offsets[line_index] + column)


def _line_offsets(text: str) -> tuple[int, ...]:
    offsets = [0]
    for match in re.finditer(r"\n", text):
        offsets.append(match.end())
    return tuple(offsets)


def _redact_html(
    source: str,
    processor: _TextProcessor,
    *,
    cid_map: Mapping[str, str],
) -> str:
    parser = _HtmlRedactor(source, processor, cid_map=cid_map)
    parser.feed(source)
    parser.close()
    return parser.result()


class _HtmlRedactor(HTMLParser):
    def __init__(
        self,
        source: str,
        processor: _TextProcessor,
        *,
        cid_map: Mapping[str, str],
    ) -> None:
        super().__init__(convert_charrefs=False)
        self.source = source
        self.processor = processor
        self.cid_map = cid_map
        self.parts: list[str] = []
        self.visible_parts: list[str] = []
        self.chunks: list[_HtmlChunk] = []
        self.visible_cursor = 0
        self.ignore_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        normalized = tag.lower()
        if normalized in _HTML_IGNORED_TAGS:
            self.ignore_depth += 1
        if not self.ignore_depth and normalized in _HTML_BLOCK_TAGS:
            self._append_visible_break()
        self.parts.append(self._format_tag(tag, attrs, self_closing=False))

    def handle_startendtag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        if not self.ignore_depth and tag.lower() in _HTML_BLOCK_TAGS:
            self._append_visible_break()
        self.parts.append(self._format_tag(tag, attrs, self_closing=True))

    def handle_endtag(self, tag: str) -> None:
        normalized = tag.lower()
        safe_tag = (
            normalized
            if normalized in _HTML_SAFE_TAGS or normalized in _HTML_IGNORED_TAGS
            else "span"
        )
        self.parts.append(f"</{safe_tag}>")
        if normalized in _HTML_IGNORED_TAGS and self.ignore_depth:
            self.ignore_depth -= 1
            return
        if not self.ignore_depth and normalized in _HTML_BLOCK_TAGS:
            self._append_visible_break()

    def handle_data(self, data: str) -> None:
        part_index = len(self.parts)
        if self.ignore_depth:
            self.parts.append("")
            return
        self.parts.append(data)
        self._append_chunk(part_index, data, raw=data)

    def handle_entityref(self, name: str) -> None:
        self._append_entity(f"&{name};")

    def handle_charref(self, name: str) -> None:
        self._append_entity(f"&#{name};")

    def handle_comment(self, data: str) -> None:
        redacted = self.processor.redact(data).text
        self.parts.append(f"<!--{redacted}-->")

    def handle_decl(self, decl: str) -> None:
        if decl.lower().startswith("doctype"):
            self.parts.append("<!DOCTYPE html>")

    def handle_pi(self, data: str) -> None:
        return

    def result(self) -> str:
        visible = "".join(self.visible_parts)
        outcome = self.processor.redact(visible)
        self._apply_visible_edits(outcome.edits)
        return "".join(self.parts)

    def _format_tag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
        *,
        self_closing: bool,
    ) -> str:
        normalized_tag = tag.lower()
        safe_tag = (
            normalized_tag
            if normalized_tag in _HTML_SAFE_TAGS or normalized_tag in _HTML_IGNORED_TAGS
            else "span"
        )
        pieces = [safe_tag]
        for name, value in attrs:
            normalized = name.lower()
            if (
                normalized.startswith("on")
                or normalized in _HTML_DROPPED_ATTRIBUTES
                or normalized not in _HTML_SAFE_ATTRIBUTES
            ):
                continue
            if value is None:
                pieces.append(normalized)
                continue
            transformed = self._redact_attribute(normalized, value)
            if transformed is None:
                continue
            pieces.append(f'{normalized}="{html_lib.escape(transformed, quote=True)}"')
        inner = " ".join(pieces)
        return f"<{inner}/>" if self_closing else f"<{inner}>"

    def _redact_attribute(self, name: str, value: str) -> str | None:
        stripped = value.strip()
        lowered = stripped.lower()
        if lowered.startswith("cid:"):
            cid = stripped[4:].strip().strip("<>")
            replacement = self.cid_map.get(cid)
            if replacement is not None:
                return f"cid:{replacement}"
            return None
        if name == "src":
            # Remote images can disclose that a clean message was opened. Only
            # regenerated local Content-IDs are safe to retain.
            return None
        if name == "href" and not lowered.startswith(
            ("#", "http://", "https://", "mailto:")
        ):
            return None
        return self.processor.redact(value).text

    def _append_entity(self, raw: str) -> None:
        part_index = len(self.parts)
        if self.ignore_depth:
            self.parts.append("")
            return
        self.parts.append(raw)
        self._append_chunk(
            part_index,
            html_lib.unescape(raw),
            raw=raw,
            entity_encoded=True,
        )

    def _append_chunk(
        self,
        part_index: int,
        text: str,
        *,
        raw: str,
        entity_encoded: bool = False,
    ) -> None:
        if not text:
            return
        start = self.visible_cursor
        self.visible_parts.append(text)
        self.visible_cursor += len(text)
        self.chunks.append(
            _HtmlChunk(
                part_index=part_index,
                visible_start=start,
                visible_end=self.visible_cursor,
                raw_text=raw,
                entity_encoded=entity_encoded,
            )
        )

    def _append_visible_break(self) -> None:
        if not self.visible_parts or self.visible_parts[-1].endswith("\n"):
            return
        self.visible_parts.append("\n")
        self.visible_cursor += 1

    def _apply_visible_edits(self, edits: Sequence[_TextEdit]) -> None:
        edits_by_part: dict[int, list[_TextEdit]] = {}
        for edit in edits:
            covered = [
                chunk
                for chunk in self.chunks
                if chunk.visible_end > edit.start and chunk.visible_start < edit.end
            ]
            if not covered and edit.start == edit.end:
                covered = [
                    chunk
                    for chunk in self.chunks
                    if chunk.visible_start <= edit.start <= chunk.visible_end
                ][:1]
            replacement_pending = edit.replacement
            for chunk in covered:
                if chunk.entity_encoded:
                    local_start = 0
                    local_end = len(chunk.raw_text)
                else:
                    local_start = (
                        max(edit.start, chunk.visible_start) - chunk.visible_start
                    )
                    local_end = min(edit.end, chunk.visible_end) - chunk.visible_start
                replacement = replacement_pending
                replacement_pending = ""
                edits_by_part.setdefault(chunk.part_index, []).append(
                    _TextEdit(
                        start=local_start,
                        end=local_end,
                        replacement=html_lib.escape(replacement, quote=False),
                    )
                )

        for part_index, part_edits in edits_by_part.items():
            value = self.parts[part_index]
            for edit in sorted(
                part_edits,
                key=lambda item: (item.start, item.end),
                reverse=True,
            ):
                value = value[: edit.start] + edit.replacement + value[edit.end :]
            self.parts[part_index] = value


def _email_handler(
    path: str | Path | BinaryIO,
    *,
    policy: Any = None,
    models: Any = None,
    lang: str | None = None,
) -> ExtractedDocument:
    output_path = _policy_value(
        policy,
        "output_path",
        "redacted_path",
        "destination_path",
    )
    if models is None and output_path is None:
        return extract_email(path)
    return redact_email(
        path,
        output_path=output_path,
        policy=policy,
        models=models,
        lang=lang,
    ).to_document()


def _policy_value(policy: Any, *names: str) -> Any:
    if isinstance(policy, Mapping):
        for name in names:
            if name in policy:
                return policy[name]
        return None
    for name in names:
        value = getattr(policy, name, None)
        if value is not None:
            return value
    return None


register_handler((".eml", ".msg"), _email_handler, requires_multimodal=False)


__all__ = [
    "EmailAttachmentReport",
    "RedactedEmail",
    "extract_email",
    "redact_email",
]
