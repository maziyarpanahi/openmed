"""CDA/C-CDA XML de-identification utilities.

The CDA path is intentionally XML-aware: structured header elements are handled
with namespace-qualified rules, while section narrative text is redacted in
text nodes only so the surrounding CDA markup remains parseable.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from io import BytesIO
from pathlib import Path
from typing import Any, Literal
from xml.etree import ElementTree as ET

from openmed.multimodal.base import ExtractedDocument, register_handler

CDA_NAMESPACE = "urn:hl7-org:v3"

ElementAction = Literal["clear_text", "clear", "null_flavor", "hash", "shift_date"]
TextRedactor = Callable[[str], str]


@dataclass(frozen=True)
class PhiElementRule:
    """Data-driven CDA PHI element action.

    Attributes:
        xpath: ElementTree-compatible XPath using the ``hl7`` CDA namespace
            prefix.
        action: Redaction action to apply to matched elements.
        attribute: Optional attribute target for hash and date-shift actions.
        label: Placeholder label used when matched surfaces appear in section
            narrative text.
        replacement: Text used by the ``clear_text`` action.
        null_flavor: CDA ``nullFlavor`` value used by ``null_flavor``.
    """

    xpath: str
    action: ElementAction
    attribute: str | None = None
    label: str = "PHI"
    replacement: str = ""
    null_flavor: str = "UNK"


DEFAULT_PHI_ELEMENT_MAP: tuple[PhiElementRule, ...] = (
    PhiElementRule(
        ".//hl7:recordTarget/hl7:patientRole/hl7:id",
        "hash",
        attribute="extension",
        label="ID",
    ),
    PhiElementRule(
        ".//hl7:recordTarget/hl7:patientRole/hl7:addr",
        "null_flavor",
        label="ADDRESS",
    ),
    PhiElementRule(
        ".//hl7:recordTarget/hl7:patientRole/hl7:telecom",
        "null_flavor",
        attribute="value",
        label="PHONE",
    ),
    PhiElementRule(
        ".//hl7:recordTarget/hl7:patientRole/hl7:patient/hl7:name",
        "null_flavor",
        label="PERSON",
    ),
    PhiElementRule(
        ".//hl7:recordTarget/hl7:patientRole/hl7:patient/hl7:birthTime",
        "shift_date",
        attribute="value",
        label="DATE_OF_BIRTH",
    ),
    PhiElementRule(".//hl7:author//hl7:assignedPerson/hl7:name", "null_flavor"),
    PhiElementRule(
        ".//hl7:authenticator//hl7:assignedPerson/hl7:name",
        "null_flavor",
    ),
    PhiElementRule(
        ".//hl7:legalAuthenticator//hl7:assignedPerson/hl7:name",
        "null_flavor",
    ),
    PhiElementRule(".//hl7:custodian//hl7:assignedPerson/hl7:name", "null_flavor"),
    PhiElementRule(".//hl7:effectiveTime", "shift_date", attribute="value"),
    PhiElementRule(".//hl7:effectiveTime/hl7:low", "shift_date", attribute="value"),
    PhiElementRule(".//hl7:effectiveTime/hl7:high", "shift_date", attribute="value"),
)

_TEXT_SWEEP_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "EMAIL",
        re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE),
    ),
    (
        "SSN",
        re.compile(r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)"),
    ),
    (
        "PHONE",
        re.compile(
            r"(?<!\w)(?:\+?1[\s.-]?)?(?:\(\d{3}\)|\d{3})"
            r"[\s.-]?\d{3}[\s.-]?\d{4}(?!\w)"
        ),
    ),
    (
        "ID",
        re.compile(
            r"\b(?:MRN|medical record(?: number)?|patient id)"
            r"[\s:#-]*[A-Z0-9-]{4,}\b",
            re.IGNORECASE,
        ),
    ),
    (
        "DATE",
        re.compile(
            r"\b(?:\d{4}-\d{2}-\d{2}|\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|"
            r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* "
            r"\d{1,2},? \d{4}|\d{1,2} "
            r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* "
            r"\d{4})\b",
            re.IGNORECASE,
        ),
    ),
    (
        "ADDRESS",
        re.compile(
            r"\b\d{1,5}\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+"
            r"(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|"
            r"Drive|Dr|Court|Ct|Way)\b",
            re.IGNORECASE,
        ),
    ),
)
_UNSAFE_XML_DECLARATION = re.compile(rb"<!\s*(?:DOCTYPE|ENTITY)\b", re.IGNORECASE)


@dataclass(frozen=True)
class _TextPiece:
    node: ET.Element
    attribute: Literal["text", "tail"]
    start: int
    end: int


def is_cda_document(document_or_path: str | bytes | Path) -> bool:
    """Return whether the XML input looks like a CDA ClinicalDocument."""

    try:
        data = _read_xml_source(document_or_path)
        _reject_unsafe_xml(data)
        root = ET.fromstring(data)
    except (ET.ParseError, OSError, UnicodeEncodeError, ValueError):
        return False

    return _is_cda_root(root)


def redact_cda(
    document_or_path: str | bytes | Path,
    *,
    element_map: Sequence[PhiElementRule] | None = None,
    text_redactor: TextRedactor | None = None,
    date_shift_days: int | None = None,
    keep_year: bool = False,
    hash_salt: str = "",
    encoding: str = "unicode",
) -> str | bytes:
    """Redact PHI from a CDA/C-CDA XML document.

    Args:
        document_or_path: XML string/bytes or a filesystem path.
        element_map: XPath-style PHI element map. Defaults to
            :data:`DEFAULT_PHI_ELEMENT_MAP`.
        text_redactor: Optional callback for additional narrative free-text
            redaction. It receives and returns a single text-node string.
        date_shift_days: Day offset used for all CDA date shifts. When omitted,
            a non-zero offset is selected once per document.
        keep_year: Preserve original years while shifting month/day values.
        hash_salt: Optional salt for hashed identifier attributes.
        encoding: Passed through to :func:`xml.etree.ElementTree.tostring`.

    Returns:
        Redacted XML as ``str`` for ``encoding="unicode"``, otherwise ``bytes``.

    Raises:
        ValueError: If the XML root is not a CDA ``ClinicalDocument``.
    """

    data = _read_xml_source(document_or_path)
    _reject_unsafe_xml(data)
    _register_input_namespaces(data)
    root = ET.fromstring(data)
    if not _is_cda_root(root):
        raise ValueError("XML document is not a CDA ClinicalDocument")

    rules = tuple(element_map or DEFAULT_PHI_ELEMENT_MAP)
    namespaces = _namespace_map(root)
    surfaces = _collect_phi_surfaces(root, rules, namespaces)
    shift_days = date_shift_days if date_shift_days is not None else _nonzero_shift()
    date_cache: dict[str, str] = {}

    for rule in rules:
        for element in root.findall(rule.xpath, namespaces):
            _apply_rule(
                element,
                rule,
                date_shift_days=shift_days,
                keep_year=keep_year,
                hash_salt=hash_salt,
                date_cache=date_cache,
            )

    for narrative in _section_text_elements(root, namespaces):
        _redact_narrative(narrative, surfaces, text_redactor=text_redactor)

    return ET.tostring(root, encoding=encoding)


def _redact_document_handler(
    path: str | Path,
    *,
    policy: Any | None = None,
    models: Any | None = None,
) -> ExtractedDocument:
    text_redactor = models if callable(models) else None
    redacted = redact_cda(path, text_redactor=text_redactor)
    return ExtractedDocument(
        text=str(redacted),
        metadata={
            "format": "cda",
            "policy": policy,
            "handler": "openmed.interop.cda",
        },
    )


def _read_xml_source(document_or_path: str | bytes | Path) -> bytes:
    if isinstance(document_or_path, bytes):
        return document_or_path

    if isinstance(document_or_path, Path):
        return document_or_path.read_bytes()

    value = str(document_or_path)
    if not value.lstrip().startswith("<"):
        try:
            candidate = Path(value)
            if candidate.exists():
                return candidate.read_bytes()
        except OSError:
            pass
    return value.encode()


def _reject_unsafe_xml(data: bytes) -> None:
    if _UNSAFE_XML_DECLARATION.search(data):
        raise ValueError("CDA XML with DOCTYPE or ENTITY declarations is unsupported")


def _register_input_namespaces(data: bytes) -> None:
    for prefix, uri in _iter_namespace_declarations(data):
        if prefix == "xml" or re.fullmatch(r"ns\d+", prefix or ""):
            continue
        ET.register_namespace(prefix, uri)


def _iter_namespace_declarations(data: bytes) -> Iterable[tuple[str, str]]:
    seen: set[tuple[str, str]] = set()
    try:
        events = ET.iterparse(BytesIO(data), events=("start-ns",))
        for _event, namespace in events:
            prefix, uri = namespace
            item = (prefix or "", uri)
            if item in seen:
                continue
            seen.add(item)
            yield item
    except ET.ParseError:
        return


def _local_name(tag: str) -> str:
    if tag.startswith("{"):
        return tag.rsplit("}", 1)[1]
    return tag


def _namespace_uri(tag: str) -> str:
    if tag.startswith("{"):
        return tag[1:].split("}", 1)[0]
    return ""


def _is_cda_root(root: ET.Element) -> bool:
    if _local_name(root.tag) != "ClinicalDocument":
        return False
    if _namespace_uri(root.tag) == CDA_NAMESPACE:
        return True
    return any(_local_name(child.tag) == "templateId" for child in list(root))


def _namespace_map(root: ET.Element) -> dict[str, str]:
    return {"hl7": _namespace_uri(root.tag)}


def _collect_phi_surfaces(
    root: ET.Element,
    rules: Sequence[PhiElementRule],
    namespaces: Mapping[str, str],
) -> dict[str, str]:
    surfaces: dict[str, str] = {}
    for rule in rules:
        placeholder = f"[{rule.label}]"
        for element in root.findall(rule.xpath, namespaces):
            for surface in _element_surfaces(element, rule.attribute):
                if len(surface) >= 2:
                    surfaces.setdefault(surface, placeholder)
    return surfaces


def _element_surfaces(element: ET.Element, attribute: str | None) -> Iterable[str]:
    if attribute and element.get(attribute):
        yield from _surface_variants(str(element.get(attribute)))

    pieces = [piece.strip() for piece in element.itertext() if piece.strip()]
    if pieces:
        yield from _surface_variants(" ".join(pieces))
    for piece in pieces:
        yield from _surface_variants(piece)


def _surface_variants(value: str) -> Iterable[str]:
    stripped = value.strip()
    if not stripped:
        return
    yield stripped
    for prefix in ("tel:", "mailto:"):
        if stripped.lower().startswith(prefix):
            yield stripped[len(prefix) :]


def _apply_rule(
    element: ET.Element,
    rule: PhiElementRule,
    *,
    date_shift_days: int,
    keep_year: bool,
    hash_salt: str,
    date_cache: dict[str, str],
) -> None:
    if rule.action in {"clear", "clear_text"}:
        _clear_text(element, replacement=rule.replacement, attribute=rule.attribute)
    elif rule.action == "null_flavor":
        _null_flavor(element, rule.null_flavor)
    elif rule.action == "hash":
        _hash_element_value(element, attribute=rule.attribute, hash_salt=hash_salt)
    elif rule.action == "shift_date":
        _shift_element_date(
            element,
            attribute=rule.attribute,
            date_shift_days=date_shift_days,
            keep_year=keep_year,
            date_cache=date_cache,
        )


def _clear_text(
    element: ET.Element,
    *,
    replacement: str,
    attribute: str | None,
) -> None:
    if attribute is not None:
        element.set(attribute, replacement)
        return

    first = True
    for node in element.iter():
        node.text = replacement if first else None
        first = False


def _null_flavor(element: ET.Element, value: str) -> None:
    tail = element.tail
    element.clear()
    element.tail = tail
    element.set("nullFlavor", value)


def _hash_element_value(
    element: ET.Element,
    *,
    attribute: str | None,
    hash_salt: str,
) -> None:
    if attribute is not None:
        value = element.get(attribute)
        if value:
            element.set(attribute, _hash_value(value, hash_salt=hash_salt))
        return

    if element.text:
        element.text = _hash_value(element.text, hash_salt=hash_salt)


def _hash_value(value: str, *, hash_salt: str) -> str:
    digest = hashlib.sha256(f"{hash_salt}{value}".encode()).hexdigest()[:16]
    return f"h{digest}"


def _shift_element_date(
    element: ET.Element,
    *,
    attribute: str | None,
    date_shift_days: int,
    keep_year: bool,
    date_cache: dict[str, str],
) -> None:
    if attribute is not None:
        value = element.get(attribute)
        if value:
            element.set(
                attribute,
                _shift_cda_date(
                    value,
                    date_shift_days,
                    keep_year=keep_year,
                    date_cache=date_cache,
                ),
            )
        return

    if element.text:
        element.text = _shift_cda_date(
            element.text,
            date_shift_days,
            keep_year=keep_year,
            date_cache=date_cache,
        )


def _shift_cda_date(
    value: str,
    shift_days: int,
    *,
    keep_year: bool,
    date_cache: dict[str, str],
) -> str:
    cached = date_cache.get(value)
    if cached is not None:
        return cached

    shifted = _shift_hl7_timestamp(value, shift_days, keep_year=keep_year)
    if shifted is None:
        shifted = _shift_general_date(value, shift_days, keep_year=keep_year)
    date_cache[value] = shifted
    return shifted


def _shift_hl7_timestamp(
    value: str,
    shift_days: int,
    *,
    keep_year: bool,
) -> str | None:
    match = re.fullmatch(
        r"(?P<date>\d{8})(?P<time>\d{0,6})(?P<fraction>\.\d+)?"
        r"(?P<zone>[+-]\d{4})?",
        value,
    )
    if match is None:
        return None

    try:
        parsed = datetime.strptime(match.group("date"), "%Y%m%d")
    except ValueError:
        return "[DATE_SHIFTED]"

    shifted = parsed + timedelta(days=shift_days)
    if keep_year:
        shifted = _replace_year_safe(shifted, parsed.year)

    return "".join(
        (
            shifted.strftime("%Y%m%d"),
            match.group("time") or "",
            match.group("fraction") or "",
            match.group("zone") or "",
        )
    )


def _shift_general_date(value: str, shift_days: int, *, keep_year: bool) -> str:
    from openmed.core.pii import _shift_date

    return _shift_date(value, shift_days, keep_year=keep_year)


def _replace_year_safe(value: datetime, year: int) -> datetime:
    try:
        return value.replace(year=year)
    except ValueError:
        return value.replace(year=year, month=2, day=28)


def _nonzero_shift() -> int:
    from openmed.core.pii import _random_nonzero_shift

    return _random_nonzero_shift()


def _section_text_elements(
    root: ET.Element, namespaces: Mapping[str, str]
) -> Iterable[ET.Element]:
    yield from root.findall(
        ".//hl7:component/hl7:structuredBody//hl7:section/hl7:text", namespaces
    )


def _redact_narrative(
    element: ET.Element,
    surfaces: Mapping[str, str],
    *,
    text_redactor: TextRedactor | None,
) -> None:
    pieces = _collect_text_pieces(element)
    _redact_piece_spans(pieces, _collect_text_spans(pieces, surfaces))

    for node in element.iter():
        if node.text:
            node.text = _redact_text_piece(
                node.text,
                surfaces,
                text_redactor=text_redactor,
            )
        if node is not element and node.tail:
            node.tail = _redact_text_piece(
                node.tail,
                surfaces,
                text_redactor=text_redactor,
            )


def _collect_text_pieces(element: ET.Element) -> list[_TextPiece]:
    pieces: list[_TextPiece] = []
    offset = 0
    for node in element.iter():
        if node.text:
            end = offset + len(node.text)
            pieces.append(_TextPiece(node, "text", offset, end))
            offset = end
        if node is not element and node.tail:
            end = offset + len(node.tail)
            pieces.append(_TextPiece(node, "tail", offset, end))
            offset = end
    return pieces


def _collect_text_spans(
    pieces: Sequence[_TextPiece],
    surfaces: Mapping[str, str],
) -> list[tuple[int, int, str]]:
    if not pieces:
        return []

    text = "".join(str(getattr(piece.node, piece.attribute) or "") for piece in pieces)
    candidates: list[tuple[int, int, str]] = []
    for surface, placeholder in sorted(
        surfaces.items(),
        key=lambda item: len(item[0]),
        reverse=True,
    ):
        for match in re.finditer(re.escape(surface), text, re.IGNORECASE):
            candidates.append((match.start(), match.end(), placeholder))

    for label, pattern in _TEXT_SWEEP_PATTERNS:
        for match in pattern.finditer(text):
            candidates.append((match.start(), match.end(), f"[{label}]"))

    return _non_overlapping_spans(candidates)


def _non_overlapping_spans(
    candidates: Sequence[tuple[int, int, str]],
) -> list[tuple[int, int, str]]:
    ordered = sorted(
        candidates,
        key=lambda item: (item[0], -(item[1] - item[0])),
    )
    selected: list[tuple[int, int, str]] = []
    for start, end, placeholder in ordered:
        if start >= end:
            continue
        if any(
            start < active_end and end > active_start
            for active_start, active_end, _ in selected
        ):
            continue
        selected.append((start, end, placeholder))
    return selected


def _redact_piece_spans(
    pieces: Sequence[_TextPiece],
    spans: Sequence[tuple[int, int, str]],
) -> None:
    if not pieces or not spans:
        return

    for piece in pieces:
        value = str(getattr(piece.node, piece.attribute) or "")
        cursor = piece.start
        parts: list[str] = []
        for start, end, placeholder in spans:
            if end <= piece.start or start >= piece.end:
                continue
            if start > cursor:
                parts.append(value[cursor - piece.start : start - piece.start])
            if piece.start <= start < piece.end:
                parts.append(placeholder)
            cursor = max(cursor, min(end, piece.end))
        parts.append(value[cursor - piece.start :])
        setattr(piece.node, piece.attribute, "".join(parts))


def _redact_text_piece(
    value: str,
    surfaces: Mapping[str, str],
    *,
    text_redactor: TextRedactor | None,
) -> str:
    if not value.strip():
        return value

    redacted = _redact_known_surfaces(value, surfaces)
    redacted = _deterministic_text_sweep(redacted)
    if text_redactor is not None:
        redacted = text_redactor(redacted)
    return redacted


def _redact_known_surfaces(value: str, surfaces: Mapping[str, str]) -> str:
    redacted = value
    for surface, placeholder in sorted(
        surfaces.items(),
        key=lambda item: len(item[0]),
        reverse=True,
    ):
        redacted = redacted.replace(surface, placeholder)
    return redacted


def _deterministic_text_sweep(value: str) -> str:
    redacted = value
    for label, pattern in _TEXT_SWEEP_PATTERNS:
        redacted = pattern.sub(f"[{label}]", redacted)
    return redacted


register_handler(
    ".xml",
    _redact_document_handler,
    detector=is_cda_document,
    requires_multimodal=False,
)


__all__ = [
    "CDA_NAMESPACE",
    "DEFAULT_PHI_ELEMENT_MAP",
    "PhiElementRule",
    "is_cda_document",
    "redact_cda",
]
