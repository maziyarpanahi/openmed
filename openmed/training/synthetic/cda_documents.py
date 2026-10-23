"""Seeded synthetic C-CDA documents with element-level PHI truth.

The generator renders continuity of care documents, discharge summaries and
progress notes from repository-authored synthetic pools. Every PHI value it
places is recorded as a truth value addressed by an indexed element path, the
node that holds it (an attribute, the element text or the element tail), the
half-open character offsets inside that node value, the half-open offsets into
the serialized document and a canonical OpenMed label. Narrative mentions are
linked to the structured header or entry values they repeat.

Paths use CDA local names with one-based positions among same-named siblings,
for example ``/ClinicalDocument/recordTarget[1]/patientRole[1]/telecom[2]``.
The ``tail`` node is the text that follows an element's end tag inside its
parent, as in :mod:`xml.etree.ElementTree`.

Documents never contain a DOCTYPE, entity declarations, entity references or
remote schema locations. Terminology codes are placeholders from
:data:`CDA_PLACEHOLDER_CODE_SYSTEM`; no restricted vocabulary is bundled. The
leakage harness runs :func:`openmed.interop.cda.redact_cda`, checks that the
output is well-formed CDA without fetching schemas, and reports residual truth
values by index-free element path and label as counts only.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Final
from xml.etree import ElementTree as ET

from openmed.core import labels as L
from openmed.interop.cda import (
    CDA_NAMESPACE,
    PhiElementRule,
    TextRedactor,
    is_cda_document,
    redact_cda,
)

CDA_SYNTHETIC_SCHEMA_VERSION: Final = 1
SYNTHETIC_CDA_SOURCE: Final = "openmed.synthetic.cda_documents"
CDA_SYNTHETIC_DOCUMENT_TYPES: Final[tuple[str, ...]] = (
    "ccd",
    "discharge_summary",
    "progress_note",
)
CDA_PLACEHOLDER_CODE_SYSTEM: Final = "2.16.840.1.113883.19.999"
CDA_PLACEHOLDER_CODE_SYSTEM_NAME: Final = "OpenMed synthetic placeholder"
MAX_CDA_DOCUMENTS: Final = 1000
MAX_CDA_PARTICIPANTS: Final = 3
MAX_CDA_ENTRIES: Final = 5

_EXAMPLE_ID_ROOT: Final = "2.16.840.1.113883.19.5"
_SSN_ID_ROOT: Final = "2.16.840.1.113883.4.1"
_XSI_NAMESPACE: Final = "http://www.w3.org/2001/XMLSchema-instance"
_LOCAL_SCHEMA: Final = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}\.xsd")
_LOCAL_STYLESHEET: Final = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}\.xsl")
_UNSAFE_XML_DECLARATION: Final = re.compile(rb"<!\s*(?:DOCTYPE|ENTITY)\b", re.I)
_SAFE_VALUE: Final = re.compile(r"[A-Za-z0-9 .,:@+()/-]+")
_PATH_SEGMENT: Final = re.compile(r"(?P<name>[A-Za-z]+)(?:\[(?P<index>[1-9]\d*)\])?")
_POSITION: Final = re.compile(r"\[\d+\]")

_DOCUMENT_TEMPLATES: Final[dict[str, tuple[str, str, str, tuple[str, ...]]]] = {
    "ccd": (
        "2.16.840.1.113883.10.20.22.1.2",
        "SYN-DOC-001",
        "Synthetic continuity of care document",
        ("allergies", "problems", "procedures", "contacts"),
    ),
    "discharge_summary": (
        "2.16.840.1.113883.10.20.22.1.8",
        "SYN-DOC-002",
        "Synthetic discharge summary",
        ("hospital_course", "problems", "procedures", "allergies"),
    ),
    "progress_note": (
        "2.16.840.1.113883.10.20.22.1.9",
        "SYN-DOC-003",
        "Synthetic progress note",
        ("assessment", "allergies", "problems", "procedures"),
    ),
}
_US_REALM_HEADER: Final = "2.16.840.1.113883.10.20.22.1.1"
_SECTIONS: Final[dict[str, tuple[str | None, str, str]]] = {
    "allergies": ("2.16.840.1.113883.10.20.22.2.6.1", "SYN-SEC-001", "Allergies"),
    "problems": ("2.16.840.1.113883.10.20.22.2.5.1", "SYN-SEC-002", "Problems"),
    "procedures": ("2.16.840.1.113883.10.20.22.2.7.1", "SYN-SEC-003", "Procedures"),
    "hospital_course": (
        "1.3.6.1.4.1.19376.1.5.3.1.3.5",
        "SYN-SEC-004",
        "Hospital course",
    ),
    "assessment": ("2.16.840.1.113883.10.20.22.2.8", "SYN-SEC-005", "Assessment"),
    "contacts": (None, "SYN-SEC-006", "Patient contacts"),
}

_GIVEN_NAMES: Final[tuple[str, ...]] = tuple(
    (
        "Avery Briar Calder Dorian Elowen Fenwick Greer Hollis Isolde "
        "Jasper Keira Linden Marlow Nerys Orrin Perrin Quilla Rowan Sable "
        "Tamsin Ulric Verity Wren Yara"
    ).split()
)
_FAMILY_NAMES: Final[tuple[str, ...]] = tuple(
    (
        "Ashdown Blackwood Carrow Dunmore Ellery Fairbairn Galloway "
        "Hartigan Ingleby Jessop Kittredge Lockhart Merriweather "
        "Northcott Oakhurst Pembury Quenby Ravenscroft Stanhope "
        "Thistlewood Underhill Valcourt Whitlock Yardley"
    ).split()
)
_STREET_NAMES: Final[tuple[str, ...]] = tuple(
    (
        "Alder Birchwood Cobalt Driftwood Elmstead Foxglove Granite Heron "
        "Juniper Kestrel Larkspur Millpond"
    ).split()
)
_STREET_SUFFIXES: Final[tuple[str, ...]] = (
    "Street",
    "Avenue",
    "Road",
    "Lane",
    "Drive",
    "Court",
)
_CITIES: Final[tuple[str, ...]] = (
    "Alder Crossing",
    "Birch Harbor",
    "Cinder Falls",
    "Dovetail",
    "Ember Ridge",
    "Fernhollow",
    "Glimmer Bay",
    "Hazelmoor",
    "Ivory Springs",
    "Juniper Point",
)
_STATES: Final[tuple[str, ...]] = ("CA", "MA", "NY", "TX", "WA")
_ORGANIZATION_PREFIXES: Final[tuple[str, ...]] = (
    "Lanternfield",
    "Mossbank",
    "Quarrystone",
    "Silverleaf",
    "Tidewell",
    "Wildbrook",
)
_ORGANIZATION_SUFFIXES: Final[tuple[str, ...]] = (
    "Community Health Center",
    "Family Clinic",
    "Regional Medical Group",
)
_AREA_CODES: Final[tuple[str, ...]] = ("201", "212", "312", "415", "617", "713")
_TIME_ZONES: Final[tuple[str, ...]] = ("", "-0500", "+0000")
_MONTHS: Final[tuple[str, ...]] = tuple(
    (
        "January February March April May June July August September "
        "October November December"
    ).split()
)
_PLACEHOLDER_LETTERS: Final = "ABCDE"


class CdaSynthesisError(ValueError):
    """Raised for invalid, unsafe or unbounded synthetic C-CDA settings.

    Attributes:
        code: Stable, controlled failure code. Messages never echo the
            rejected input.
    """

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"{code}: {message}")


class CdaLeakageError(RuntimeError):
    """Raised when the leakage harness cannot evaluate a document.

    Attributes:
        code: Stable, controlled failure code.
        document_index: Zero-based index of the failing document, when known.
    """

    def __init__(self, code: str, document_index: int | None = None) -> None:
        self.code = code
        self.document_index = document_index
        where = "" if document_index is None else f" at index {document_index}"
        super().__init__(f"{code}: C-CDA leakage evaluation failed{where}")


@dataclass(frozen=True)
class CdaSynthesisConfig:
    """Bounded, local-only settings for synthetic C-CDA generation.

    Args:
        document_types: Document types to rotate through, drawn from
            :data:`CDA_SYNTHETIC_DOCUMENT_TYPES`.
        max_authors: Upper bound for header authors, from 1 to
            :data:`MAX_CDA_PARTICIPANTS`.
        max_informants: Upper bound for related-person informants, from 0 to
            :data:`MAX_CDA_PARTICIPANTS`. Zero omits informants.
        max_next_of_kin: Upper bound for next-of-kin participants, from 0 to
            :data:`MAX_CDA_PARTICIPANTS`. Zero omits them.
        max_allergies: Upper bound for allergy entries, from 1 to
            :data:`MAX_CDA_ENTRIES`.
        max_problems: Upper bound for problem entries, from 1 to
            :data:`MAX_CDA_ENTRIES`.
        max_procedures: Upper bound for procedure entries, from 1 to
            :data:`MAX_CDA_ENTRIES`.
        guardian: Whether the patient has a guardian.
        birthplace: Whether the patient has a birthplace.
        schema_location: Optional local ``.xsd`` file name emitted in
            ``xsi:schemaLocation``. URLs, paths and markup are rejected; the
            harness never fetches it.
        stylesheet: Optional local ``.xsl`` file name emitted in an
            ``xml-stylesheet`` processing instruction. URLs, paths and markup
            are rejected.

    Raises:
        CdaSynthesisError: If any setting is unknown, unsafe or outside its
            bound.
    """

    document_types: tuple[str, ...] = CDA_SYNTHETIC_DOCUMENT_TYPES
    max_authors: int = 2
    max_informants: int = 2
    max_next_of_kin: int = 2
    max_allergies: int = 2
    max_problems: int = 3
    max_procedures: int = 2
    guardian: bool = True
    birthplace: bool = True
    schema_location: str | None = None
    stylesheet: str | None = None

    def __post_init__(self) -> None:
        types = self.document_types
        if (
            not isinstance(types, (tuple, list))
            or not types
            or any(item not in CDA_SYNTHETIC_DOCUMENT_TYPES for item in types)
        ):
            raise CdaSynthesisError(
                "unknown_document_type",
                "document types must be drawn from "
                + ", ".join(CDA_SYNTHETIC_DOCUMENT_TYPES),
            )
        object.__setattr__(self, "document_types", tuple(types))
        _bounded_int(self.max_authors, 1, MAX_CDA_PARTICIPANTS)
        _bounded_int(self.max_informants, 0, MAX_CDA_PARTICIPANTS)
        _bounded_int(self.max_next_of_kin, 0, MAX_CDA_PARTICIPANTS)
        _bounded_int(self.max_allergies, 1, MAX_CDA_ENTRIES)
        _bounded_int(self.max_problems, 1, MAX_CDA_ENTRIES)
        _bounded_int(self.max_procedures, 1, MAX_CDA_ENTRIES)
        for flag in (self.guardian, self.birthplace):
            if not isinstance(flag, bool):
                raise CdaSynthesisError("invalid_flag", "flags must be booleans")
        if self.schema_location is not None and not _is_local_name(
            self.schema_location, _LOCAL_SCHEMA
        ):
            raise CdaSynthesisError(
                "unsafe_schema_location",
                "schema_location must be a local .xsd file name without a "
                "scheme, path or markup",
            )
        if self.stylesheet is not None and not _is_local_name(
            self.stylesheet, _LOCAL_STYLESHEET
        ):
            raise CdaSynthesisError(
                "unsafe_stylesheet",
                "stylesheet must be a local .xsl file name without a scheme, "
                "path or markup",
            )


@dataclass(frozen=True)
class CdaTruthValue:
    """One PHI value placed in a synthetic C-CDA document.

    Attributes:
        truth_id: Document-unique identifier in document order.
        label: Canonical OpenMed PHI label.
        role: Controlled role of the person, place or event, such as
            ``patient``, ``guardian`` or ``custodian``.
        context: ``header``, ``entry`` or ``narrative``.
        path: Indexed element path from ``/ClinicalDocument``.
        node: ``attribute``, ``text`` or ``tail``.
        attribute: Attribute name for ``attribute`` nodes, otherwise ``None``.
        start: Inclusive offset into the node value.
        end: Exclusive offset into the node value.
        document_start: Inclusive offset into the serialized document text.
        document_end: Exclusive offset into the serialized document text.
        value: The synthetic PHI value.
    """

    truth_id: str
    label: str
    role: str
    context: str
    path: str
    node: str
    attribute: str | None
    start: int
    end: int
    document_start: int
    document_end: int
    value: str

    @property
    def locator(self) -> str:
        """Indexed path plus node, such as ``.../telecom[1]/@value``."""

        return _locator(self.path, self.node, self.attribute)

    @property
    def element_path(self) -> str:
        """Index-free locator used to aggregate harness results."""

        return _POSITION.sub("", self.locator)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible truth record with a fixed key order."""

        return {
            "truth_id": self.truth_id,
            "label": self.label,
            "role": self.role,
            "context": self.context,
            "path": self.path,
            "node": self.node,
            "attribute": self.attribute,
            "start": self.start,
            "end": self.end,
            "document_start": self.document_start,
            "document_end": self.document_end,
            "value": self.value,
        }


@dataclass(frozen=True)
class CdaMentionLink:
    """Link from a narrative mention to the structured values it repeats.

    Attributes:
        mention_id: Truth identifier of the narrative mention.
        value_ids: Truth identifiers of the linked header or entry values.
        content_id: ``ID`` of the enclosing narrative ``content`` element
            when the mention is its text, otherwise ``None``.
    """

    mention_id: str
    value_ids: tuple[str, ...]
    content_id: str | None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible link record."""

        return {
            "mention_id": self.mention_id,
            "value_ids": list(self.value_ids),
            "content_id": self.content_id,
        }


@dataclass(frozen=True)
class CdaContentReference:
    """Entry ``reference`` that points at a narrative ``content`` ID.

    Attributes:
        content_id: Narrative ``content/@ID`` value.
        content_path: Indexed path of the narrative ``content`` element.
        reference_path: Indexed path of the entry ``reference`` element whose
            ``value`` is ``#`` plus ``content_id``.
    """

    content_id: str
    content_path: str
    reference_path: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible reference record."""

        return {
            "content_id": self.content_id,
            "content_path": self.content_path,
            "reference_path": self.reference_path,
        }


@dataclass(frozen=True)
class SyntheticCdaDocument:
    """One synthetic C-CDA document and its element-level PHI truth.

    Attributes:
        xml: Serialized document text, ASCII only.
        document_type: One of :data:`CDA_SYNTHETIC_DOCUMENT_TYPES`.
        truth: PHI truth values in document order.
        links: Narrative mention links in document order.
        content_references: Entry references to narrative content IDs.
        seed: Generator seed.
        index: Zero-based document index within the generator.
    """

    xml: str
    document_type: str
    truth: tuple[CdaTruthValue, ...]
    links: tuple[CdaMentionLink, ...]
    content_references: tuple[CdaContentReference, ...]
    seed: int
    index: int

    def to_bytes(self) -> bytes:
        """Return the UTF-8 encoded document bytes."""

        return self.xml.encode("utf-8")

    @property
    def xml_sha256(self) -> str:
        """SHA-256 hex digest of the UTF-8 document bytes."""

        return hashlib.sha256(self.to_bytes()).hexdigest()

    def manifest(self) -> dict[str, Any]:
        """Return the JSON-compatible truth manifest for this document."""

        return {
            "schema_version": CDA_SYNTHETIC_SCHEMA_VERSION,
            "source": SYNTHETIC_CDA_SOURCE,
            "synthetic": True,
            "contains_real_phi": False,
            "seed": self.seed,
            "index": self.index,
            "document_type": self.document_type,
            "code_system": CDA_PLACEHOLDER_CODE_SYSTEM,
            "offset_basis": "node_value_and_document_code_points",
            "xml_sha256": self.xml_sha256,
            "truth": [truth.to_dict() for truth in self.truth],
            "links": [link.to_dict() for link in self.links],
            "content_references": [
                reference.to_dict() for reference in self.content_references
            ],
        }

    @property
    def manifest_sha256(self) -> str:
        """SHA-256 hex digest of the canonical JSON manifest."""

        payload = json.dumps(
            self.manifest(), sort_keys=True, separators=(",", ":"), ensure_ascii=True
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class CdaDocumentGenerator:
    """Generate deterministic synthetic C-CDA documents with PHI truth.

    Args:
        config: Generation settings. Defaults to :class:`CdaSynthesisConfig`.
        seed: Integer seed. Identical seeds and settings produce byte-identical
            documents and manifests.

    Raises:
        CdaSynthesisError: If ``seed`` is not an integer or ``config`` is not a
            :class:`CdaSynthesisConfig`.
    """

    def __init__(
        self,
        config: CdaSynthesisConfig | None = None,
        *,
        seed: int = 0,
    ) -> None:
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise CdaSynthesisError("invalid_seed", "seed must be an integer")
        if config is not None and not isinstance(config, CdaSynthesisConfig):
            raise CdaSynthesisError(
                "invalid_config", "config must be a CdaSynthesisConfig"
            )
        self.config = config or CdaSynthesisConfig()
        self.seed = seed
        self._index = 0

    def generate(self, document_type: str | None = None) -> SyntheticCdaDocument:
        """Generate the next document.

        Args:
            document_type: Optional type override. By default the configured
                document types rotate in order.

        Returns:
            The rendered document with its truth manifest.

        Raises:
            CdaSynthesisError: If ``document_type`` is unsupported or the
                generator already produced :data:`MAX_CDA_DOCUMENTS` documents.
        """

        index = self._index
        if index >= MAX_CDA_DOCUMENTS:
            raise CdaSynthesisError(
                "count_out_of_bounds",
                f"a generator produces at most {MAX_CDA_DOCUMENTS} documents",
            )
        if document_type is None:
            types = self.config.document_types
            document_type = types[index % len(types)]
        elif document_type not in CDA_SYNTHETIC_DOCUMENT_TYPES:
            raise CdaSynthesisError(
                "unknown_document_type",
                "document types must be drawn from "
                + ", ".join(CDA_SYNTHETIC_DOCUMENT_TYPES),
            )
        self._index += 1
        return _DocumentBuilder(self.config, self.seed, index, document_type).build()

    def generate_batch(self, count: int) -> tuple[SyntheticCdaDocument, ...]:
        """Generate ``count`` documents, bounded by :data:`MAX_CDA_DOCUMENTS`.

        Args:
            count: Number of documents to generate.

        Returns:
            Documents in generation order.

        Raises:
            CdaSynthesisError: If ``count`` is not an integer within bounds.
        """

        _bounded_int(count, 1, MAX_CDA_DOCUMENTS)
        return tuple(self.generate() for _ in range(count))


def generate_cda_documents(
    count: int = 3,
    *,
    seed: int = 0,
    config: CdaSynthesisConfig | None = None,
) -> tuple[SyntheticCdaDocument, ...]:
    """Generate synthetic C-CDA documents with element-level PHI truth.

    Args:
        count: Number of documents, from 1 to :data:`MAX_CDA_DOCUMENTS`.
        seed: Integer seed for deterministic output.
        config: Optional generation settings.

    Returns:
        Documents in generation order.

    Raises:
        CdaSynthesisError: If ``count``, ``seed`` or ``config`` is invalid.

    Example:
        >>> from openmed.training.synthetic.cda_documents import (
        ...     generate_cda_documents,
        ...     resolve_cda_locator,
        ... )
        >>> doc = generate_cda_documents(1, seed=7)[0]
        >>> truth = doc.truth[0]
        >>> resolve_cda_locator(doc.xml, truth)[truth.start:truth.end] == truth.value
        True
    """

    _bounded_int(count, 1, MAX_CDA_DOCUMENTS)
    return CdaDocumentGenerator(config, seed=seed).generate_batch(count)


def resolve_cda_locator(
    document: str | bytes | ET.Element, truth: CdaTruthValue
) -> str | None:
    """Return the node value addressed by a truth locator.

    Args:
        document: Serialized CDA document or a parsed ``ClinicalDocument``
            root. Serialized input is parsed without DTD or entity support.
        truth: Truth value whose ``path``, ``node`` and ``attribute`` select
            the node.

    Returns:
        The attribute value, element text or element tail, or ``None`` when
        the path no longer resolves or the node is empty.

    Raises:
        ValueError: If serialized input declares a DOCTYPE or entities, or is
            not well-formed XML.
    """

    root = document if isinstance(document, ET.Element) else _safe_parse(document)
    element = _find_path(root, truth.path)
    if element is None:
        return None
    if truth.node == "attribute":
        return element.get(truth.attribute or "")
    if truth.node == "text":
        return element.text
    return element.tail


# Document construction ------------------------------------------------------


@dataclass(frozen=True)
class _Phi:
    value: str
    label: str
    role: str
    context: str
    key: str | None = None
    refers_to: tuple[str, ...] = ()


_Piece = str | _Phi
_Pieces = tuple[_Piece, ...]


@dataclass
class _Node:
    tag: str
    attrs: list[tuple[str, _Pieces]]
    children: list[_Node]
    text: _Pieces = ()
    tail: _Pieces = ()

    def get(self, name: str) -> str | None:
        for key, pieces in self.attrs:
            if key == name:
                return "".join(_piece_text(piece) for piece in pieces)
        return None


def _pieces(value: _Piece | _Pieces | None) -> _Pieces:
    if value is None:
        return ()
    if isinstance(value, tuple):
        return value
    return (value,)


def _el(
    tag: str,
    *children: _Node,
    text: _Piece | _Pieces | None = None,
    **attrs: _Piece | _Pieces,
) -> _Node:
    return _Node(
        tag,
        [(name, _pieces(value)) for name, value in attrs.items()],
        list(children),
        _pieces(text),
    )


def _mixed(tag: str, *parts: _Piece | _Node, **attrs: _Piece | _Pieces) -> _Node:
    """Build a mixed-content element from text pieces and child elements."""

    node = _el(tag, **attrs)
    leading: list[_Piece] = []
    for part in parts:
        if isinstance(part, _Node):
            node.children.append(part)
        elif node.children:
            last = node.children[-1]
            last.tail = (*last.tail, part)
        else:
            leading.append(part)
    node.text = tuple(leading)
    return node


def _piece_text(piece: _Piece) -> str:
    return piece.value if isinstance(piece, _Phi) else piece


def _table(headers: Sequence[str], rows: Sequence[_Node]) -> _Node:
    head = _el("thead", _el("tr", *(_el("th", text=title) for title in headers)))
    return _el("table", head, _el("tbody", *rows))


@dataclass
class _Person:
    given: _Phi
    family: _Phi
    role: str

    @property
    def full(self) -> str:
        return f"{self.given.value} {self.family.value}"


class _DocumentBuilder:
    def __init__(
        self,
        config: CdaSynthesisConfig,
        seed: int,
        index: int,
        document_type: str,
    ) -> None:
        self.config = config
        self.seed = seed
        self.index = index
        self.document_type = document_type
        self.rng = random.Random(_derive(seed, index, document_type))
        self.values: dict[str, str] = {}
        self.content_ids: set[str] = set()

    # Values -----------------------------------------------------------------

    def phi(self, value: str, label: str, role: str, key: str) -> _Phi:
        if key in self.values:
            raise CdaSynthesisError("internal_error", "duplicate truth key")
        self.values[key] = value
        context = "entry" if role in {"entry", "performer"} else "header"
        return _Phi(_checked(value), label, role, context, key=key)

    def mention(self, label: str, role: str, *refers_to: str, value: str = "") -> _Phi:
        text = value or " ".join(self.values[key] for key in refers_to)
        return _Phi(_checked(text), label, role, "narrative", refers_to=refers_to)

    def person(self, role: str, key: str, *, family: str | None = None) -> _Person:
        return _Person(
            self.phi(self.rng.choice(_GIVEN_NAMES), L.FIRST_NAME, role, f"{key}.given"),
            self.phi(
                family or self.rng.choice(_FAMILY_NAMES),
                L.LAST_NAME,
                role,
                f"{key}.family",
            ),
            role,
        )

    def date_value(self, start: date, span_days: int) -> date:
        return start + timedelta(days=self.rng.randint(0, span_days))

    def timestamp(self, day: date) -> str:
        hour = self.rng.randint(7, 19)
        minute = self.rng.choice((0, 15, 30, 45))
        zone = self.rng.choice(_TIME_ZONES)
        return f"{day:%Y%m%d}{hour:02d}{minute:02d}00{zone}"

    def name_node(self, person: _Person) -> _Node:
        return _el(
            "name", _el("given", text=person.given), _el("family", text=person.family)
        )

    def addr_node(self, role: str, key: str, *, use: str = "HP") -> _Node:
        rng = self.rng
        street = (
            f"{rng.randint(10, 9899)} {rng.choice(_STREET_NAMES)} "
            f"{rng.choice(_STREET_SUFFIXES)}"
        )
        city = rng.choice(_CITIES)
        postal = f"000{rng.randint(0, 99):02d}"
        return _el(
            "addr",
            _el(
                "streetAddressLine",
                text=self.phi(street, L.STREET_ADDRESS, role, f"{key}.street"),
            ),
            _el("city", text=self.phi(city, L.LOCATION, role, f"{key}.city")),
            _el("state", text=rng.choice(_STATES)),
            _el("postalCode", text=self.phi(postal, L.ZIPCODE, role, f"{key}.postal")),
            _el("country", text="US"),
            use=use,
        )

    def telecom_node(self, role: str, key: str, *, use: str = "HP") -> _Node:
        phone = f"+1-{self.rng.choice(_AREA_CODES)}-555-01{self.rng.randint(0, 99):02d}"
        return _el(
            "telecom", use=use, value=("tel:", self.phi(phone, L.PHONE, role, key))
        )

    def code_node(self, tag: str, category: str, number: int, display: str) -> _Node:
        return _el(
            tag,
            code=f"SYN-{category}-{number:03d}",
            codeSystem=CDA_PLACEHOLDER_CODE_SYSTEM,
            codeSystemName=CDA_PLACEHOLDER_CODE_SYSTEM_NAME,
            displayName=display,
        )

    def id_node(self, value: str, role: str, key: str, label: str = L.ID_NUM) -> _Node:
        root = _SSN_ID_ROOT if label == L.SSN else _EXAMPLE_ID_ROOT
        return _el("id", root=root, extension=self.phi(value, label, role, key))

    def time_node(self, tag: str, role: str, key: str) -> _Node:
        return _el(tag, value=self.phi(self.document_time, L.DATE, role, key))

    def party(
        self,
        person: _Person,
        key: str,
        person_tag: str,
        relationship: tuple[int, str] | None,
        *,
        use: str = "HP",
    ) -> list[_Node]:
        """Return the optional code, addr, telecom and person children."""

        code = (
            []
            if relationship is None
            else [self.code_node("code", "REL", *relationship)]
        )
        return [
            *code,
            self.addr_node(person.role, f"{key}.addr", use=use),
            self.telecom_node(person.role, f"{key}.phone", use=use),
            _el(person_tag, self.name_node(person)),
        ]

    def content(self, content_id: str, *parts: _Piece | _Node) -> _Node:
        if content_id in self.content_ids:
            raise CdaSynthesisError("internal_error", "duplicate content ID")
        self.content_ids.add(content_id)
        return _mixed("content", *parts, ID=content_id)

    # Document ---------------------------------------------------------------

    def build(self) -> SyntheticCdaDocument:
        config = self.config
        rng = self.rng
        template, doc_code, title, sections = _DOCUMENT_TEMPLATES[self.document_type]

        self.encounter_low = self.date_value(date(2021, 1, 1), 1400)
        self.encounter_high = self.encounter_low
        if self.document_type == "discharge_summary":
            self.encounter_high += timedelta(rng.randint(1, 9))
        self.document_time = self.timestamp(self.encounter_high)
        birth = self.date_value(date(1940, 1, 1), 27000)

        self.patient = self.person("patient", "patient")
        shared_family = self.patient.family.value if rng.random() < 0.5 else None
        self.guardian = (
            self.person("guardian", "guardian", family=shared_family)
            if config.guardian
            else None
        )
        self.next_of_kin = [
            self.person("next_of_kin", f"next_of_kin{number}")
            for number in range(1, _count(rng, config.max_next_of_kin) + 1)
        ]
        self.informants = [
            self.person("informant", f"informant{number}")
            for number in range(1, _count(rng, config.max_informants) + 1)
        ]
        self.authors = [
            self.person("author", f"author{number}")
            for number in range(1, rng.randint(1, config.max_authors) + 1)
        ]
        organization = (
            f"{rng.choice(_ORGANIZATION_PREFIXES)} {rng.choice(_ORGANIZATION_SUFFIXES)}"
        )
        self.organization = self.phi(
            organization, L.ORGANIZATION, "custodian", "custodian.name"
        )

        document_id = _derive(self.seed, self.index, "id") % 10**8
        children: list[_Node] = [
            _el("typeId", root="2.16.840.1.113883.1.3", extension="POCD_HD000040"),
            _el("templateId", root=_US_REALM_HEADER),
            _el("templateId", root=template),
            _el("id", root=_EXAMPLE_ID_ROOT, extension=f"SYN-DOC-{document_id:08d}"),
            self.code_node("code", "DOC", int(doc_code[-3:]), title),
            _el("title", text=title),
            self.time_node("effectiveTime", "document", "document.time"),
            self.code_node("confidentialityCode", "CONF", 1, "Synthetic normal"),
            self.record_target(birth),
            *(self.author_node(number) for number in range(1, len(self.authors) + 1)),
            *(
                _el(
                    "informant",
                    _el(
                        "relatedEntity",
                        *self.party(
                            person,
                            f"informant{number}",
                            "relatedPerson",
                            (2, "Synthetic related person"),
                        ),
                        classCode="PRS",
                    ),
                )
                for number, person in enumerate(self.informants, 1)
            ),
            self.custodian_node(),
            self.legal_authenticator_node(),
            *(
                _el(
                    "participant",
                    _el(
                        "associatedEntity",
                        *self.party(
                            person,
                            f"next_of_kin{number}",
                            "associatedPerson",
                            (3, "Synthetic next of kin"),
                        ),
                        classCode="NOK",
                    ),
                    typeCode="IND",
                )
                for number, person in enumerate(self.next_of_kin, 1)
            ),
        ]
        if self.document_type == "ccd":
            service_low = self.encounter_low - timedelta(rng.randint(365, 3650))
            interval = self.interval(service_low, self.encounter_high, "service")
            children.append(
                _el("documentationOf", _el("serviceEvent", interval, classCode="PCPR"))
            )
        else:
            interval = self.interval(
                self.encounter_low, self.encounter_high, "encounter"
            )
            children.append(
                _el(
                    "componentOf",
                    _el(
                        "encompassingEncounter",
                        _el("id", root=_EXAMPLE_ID_ROOT),
                        interval,
                    ),
                )
            )
        components = [
            _el("component", getattr(self, f"section_{name}")()) for name in sections
        ]
        children.append(_el("component", _el("structuredBody", *components)))
        root = _el("ClinicalDocument", *children, xmlns=CDA_NAMESPACE)
        if config.schema_location is not None:
            location = f"{CDA_NAMESPACE} {config.schema_location}"
            root.attrs.append(("xmlns:xsi", (_XSI_NAMESPACE,)))
            root.attrs.append(("xsi:schemaLocation", (location,)))
        return self.render(root)

    def interval(self, low: date, high: date, key: str) -> _Node:
        return _el(
            "effectiveTime",
            _el(
                "low", value=self.phi(f"{low:%Y%m%d}", L.DATE, "document", f"{key}.low")
            ),
            _el(
                "high",
                value=self.phi(f"{high:%Y%m%d}", L.DATE, "document", f"{key}.high"),
            ),
        )

    def record_target(self, birth: date) -> _Node:
        rng = self.rng
        patient = self.patient
        email = (
            f"{patient.given.value.lower()}.{patient.family.value.lower()}"
            f"{rng.randint(10, 99)}@example.test"
        )
        ssn = (
            f"9{rng.randint(0, 99):02d}-{rng.randint(1, 99):02d}-"
            f"{rng.randint(1, 9999):04d}"
        )
        birth_value = self.phi(
            f"{birth:%Y%m%d}", L.DATE_OF_BIRTH, "patient", "patient.birth"
        )
        patient_children = [
            self.name_node(patient),
            self.code_node("administrativeGenderCode", "SEX", rng.randint(1, 3), "X"),
            _el("birthTime", value=birth_value),
        ]
        if self.guardian is not None:
            patient_children.append(
                _el(
                    "guardian",
                    *self.party(
                        self.guardian,
                        "guardian",
                        "guardianPerson",
                        (1, "Synthetic guardian"),
                    ),
                )
            )
        if self.config.birthplace:
            city = self.phi(
                rng.choice(_CITIES), L.LOCATION, "birthplace", "birthplace.city"
            )
            address = _el(
                "addr",
                _el("city", text=city),
                _el("state", text=rng.choice(_STATES)),
                _el("country", text="US"),
            )
            patient_children.append(_el("birthplace", _el("place", address)))
        mrn = f"SYN{rng.randint(0, 9_999_999):07d}"
        return _el(
            "recordTarget",
            _el(
                "patientRole",
                self.id_node(mrn, "patient", "patient.mrn"),
                self.id_node(ssn, "patient", "patient.ssn", L.SSN),
                self.addr_node("patient", "patient.addr"),
                self.telecom_node("patient", "patient.phone"),
                _el(
                    "telecom",
                    use="HP",
                    value=(
                        "mailto:",
                        self.phi(email, L.EMAIL, "patient", "patient.email"),
                    ),
                ),
                _el("patient", *patient_children),
            ),
        )

    def author_node(self, number: int) -> _Node:
        author = self.authors[number - 1]
        key = f"author{number}"
        identifier = f"0{self.rng.randint(0, 999_999_999):09d}"
        return _el(
            "author",
            self.time_node("time", "author", f"{key}.time"),
            _el(
                "assignedAuthor",
                self.id_node(identifier, "author", f"{key}.id"),
                *self.party(author, key, "assignedPerson", None, use="WP"),
            ),
        )

    def custodian_node(self) -> _Node:
        organization = _el(
            "representedCustodianOrganization",
            _el("id", root=_EXAMPLE_ID_ROOT),
            _el("name", text=self.organization),
            self.telecom_node("custodian", "custodian.phone", use="WP"),
            self.addr_node("custodian", "custodian.addr", use="WP"),
        )
        return _el("custodian", _el("assignedCustodian", organization))

    def legal_authenticator_node(self) -> _Node:
        author = self.authors[0]
        role = "legal_authenticator"
        signer = _Person(
            self.phi(author.given.value, L.FIRST_NAME, role, "legal.given"),
            self.phi(author.family.value, L.LAST_NAME, role, "legal.family"),
            role,
        )
        return _el(
            "legalAuthenticator",
            self.time_node("time", role, "legal.time"),
            self.code_node("signatureCode", "SIG", 1, "Synthetic signed"),
            _el(
                "assignedEntity",
                self.id_node(self.values["author1.id"], role, "legal.id"),
                _el("assignedPerson", self.name_node(signer)),
            ),
        )

    # Sections ---------------------------------------------------------------

    def section(self, name: str, text: _Node, *entries: _Node) -> _Node:
        template, code, title = _SECTIONS[name]
        children = [] if template is None else [_el("templateId", root=template)]
        children += [
            self.code_node("code", "SEC", int(code[-3:]), title),
            _el("title", text=title),
            text,
            *entries,
        ]
        return _el("section", *children)

    def person_mention(self, person: _Person) -> _Phi:
        return self.mention(
            L.PERSON, person.role, person.given.key or "", person.family.key or ""
        )

    def date_mention(self, day: date, key: str, role: str = "entry") -> _Phi:
        if self.rng.random() < 0.5:
            text = day.isoformat()
        else:
            text = f"{_MONTHS[day.month - 1]} {day.day}, {day.year}"
        return self.mention(L.DATE, role, key, value=text)

    def phone_mention(self, role: str, key: str) -> _Phi:
        return self.mention(L.PHONE, role, key)

    def concern_entry(
        self, number: int, display: str, observation: _Node, content_id: str
    ) -> _Node:
        observation.children.insert(
            1, _el("text", _el("reference", value=f"#{content_id}"))
        )
        act = _el(
            "act",
            self.code_node("code", "ACT", number, display),
            _el("entryRelationship", observation, typeCode="SUBJ"),
            classCode="ACT",
            moodCode="EVN",
        )
        return _el("entry", act, typeCode="DRIV")

    def section_allergies(self) -> _Node:
        rng = self.rng
        rows: list[_Node] = []
        entries: list[_Node] = []
        for number in range(1, rng.randint(1, self.config.max_allergies) + 1):
            onset = self.encounter_low - timedelta(rng.randint(30, 3000))
            content_id = f"allergy-{number}"
            label = f"Synthetic allergen {_PLACEHOLDER_LETTERS[number - 1]}"
            key = f"allergy{number}.onset"
            rows.append(
                _el(
                    "tr",
                    _el("td", self.content(content_id, label)),
                    _el("td", text=f"Synthetic reaction {number}"),
                    _el("td", text=self.date_mention(onset, key)),
                )
            )
            observation = _el(
                "observation",
                self.code_node("code", "OBS", 1, "Synthetic allergy observation"),
                _el("effectiveTime", _el("low", value=self.entry_date(onset, key))),
                self.code_node("value", "VAL", number, label),
                classCode="OBS",
                moodCode="EVN",
            )
            entries.append(
                self.concern_entry(1, "Synthetic concern", observation, content_id)
            )
        parts: list[_Piece | _Node] = [
            "Allergy history for ",
            self.content("allergies-patient", self.person_mention(self.patient)),
        ]
        if self.informants:
            informant = self.person_mention(self.informants[0])
            parts += [
                " was confirmed with ",
                self.content("allergies-informant", informant),
            ]
        table = _table(("Allergen", "Reaction", "Onset"), rows)
        text = _el("text", _mixed("paragraph", *parts, "."), table)
        return self.section("allergies", text, *entries)

    def entry_date(self, day: date, key: str) -> _Phi:
        return self.phi(f"{day:%Y%m%d}", L.DATE, "entry", key)

    def section_problems(self) -> _Node:
        rng = self.rng
        items: list[_Node] = []
        entries: list[_Node] = []
        for number in range(1, rng.randint(1, self.config.max_problems) + 1):
            onset = self.encounter_low - timedelta(rng.randint(30, 3000))
            content_id = f"problem-{number}"
            label = f"Synthetic problem {_PLACEHOLDER_LETTERS[number - 1]}"
            key = f"problem{number}.onset"
            items.append(
                _mixed(
                    "item",
                    self.content(content_id, label),
                    ", noted on ",
                    self.date_mention(onset, key),
                )
            )
            observation = _el(
                "observation",
                self.code_node("code", "OBS", 2, "Synthetic problem observation"),
                _el("effectiveTime", _el("low", value=self.entry_date(onset, key))),
                self.code_node("value", "PROB", number, label),
                classCode="OBS",
                moodCode="EVN",
            )
            entries.append(
                self.concern_entry(
                    2, "Synthetic problem concern", observation, content_id
                )
            )
        text = _el(
            "text",
            _el("paragraph", text="Active problems:"),
            _el("list", *items, listType="unordered"),
        )
        return self.section("problems", text, *entries)

    def section_procedures(self) -> _Node:
        rng = self.rng
        rows: list[_Node] = []
        entries: list[_Node] = []
        for number in range(1, rng.randint(1, self.config.max_procedures) + 1):
            day = self.encounter_low - timedelta(rng.randint(0, 900))
            content_id = f"procedure-{number}"
            label = f"Synthetic procedure {_PLACEHOLDER_LETTERS[number - 1]}"
            performer = self.person("performer", f"performer{number}")
            key = f"procedure{number}.date"
            performer_mention = self.content(
                f"{content_id}-performer", self.person_mention(performer)
            )
            rows.append(
                _el(
                    "tr",
                    _el("td", self.content(content_id, label)),
                    _el("td", text=self.date_mention(day, key)),
                    _el("td", performer_mention),
                )
            )
            assigned = _el(
                "assignedEntity",
                _el("id", root=_EXAMPLE_ID_ROOT),
                _el("assignedPerson", self.name_node(performer)),
            )
            procedure = _el(
                "procedure",
                self.code_node("code", "PROC", number, label),
                _el("text", _el("reference", value=f"#{content_id}")),
                _el("effectiveTime", value=self.entry_date(day, key)),
                _el("performer", assigned),
                classCode="PROC",
                moodCode="EVN",
            )
            entries.append(_el("entry", procedure, typeCode="DRIV"))
        table = _table(("Procedure", "Date", "Performer"), rows)
        return self.section("procedures", _el("text", table), *entries)

    def section_contacts(self) -> _Node:
        contacts: list[tuple[str, _Person, str]] = []
        if self.guardian is not None:
            contacts.append(("Guardian", self.guardian, "guardian"))
        contacts += [
            ("Next of kin", person, f"next_of_kin{number}")
            for number, person in enumerate(self.next_of_kin, 1)
        ]
        contacts += [
            ("Informant", person, f"informant{number}")
            for number, person in enumerate(self.informants, 1)
        ]
        rows = [
            _el(
                "tr",
                _el("td", text=relationship),
                _el(
                    "td", self.content(f"contact-{number}", self.person_mention(person))
                ),
                _el("td", text=self.phone_mention(person.role, f"{key}.phone")),
            )
            for number, (relationship, person, key) in enumerate(contacts, 1)
        ]
        custodian = _mixed(
            "item",
            "Records custodian: ",
            self.content("contact-custodian", self.custodian_mention()),
        )
        text = _el(
            "text",
            _el("paragraph", text="Contacts on file:"),
            _table(("Relationship", "Name", "Telephone"), rows),
            _el("list", custodian, listType="unordered"),
        )
        return self.section("contacts", text)

    def section_hospital_course(self) -> _Node:
        paragraphs = [
            _mixed(
                "paragraph",
                self.person_mention(self.patient),
                " was admitted on ",
                self.date_mention(self.encounter_low, "encounter.low", "document"),
                " and discharged on ",
                self.date_mention(self.encounter_high, "encounter.high", "document"),
                ".",
            )
        ]
        if self.guardian is not None:
            paragraphs.append(
                _mixed(
                    "paragraph",
                    "Discharge instructions were reviewed with guardian ",
                    self.content("course-guardian", self.person_mention(self.guardian)),
                    ", reachable at ",
                    self.phone_mention("guardian", "guardian.phone"),
                    ".",
                )
            )
        for number, person in enumerate(self.next_of_kin, 1):
            paragraphs.append(
                _mixed(
                    "paragraph",
                    "Next of kin ",
                    self.content(f"course-kin-{number}", self.person_mention(person)),
                    " was updated.",
                )
            )
        paragraphs.append(self.follow_up_paragraph("course"))
        text = _el("text", *paragraphs, self.care_team_list("course"))
        return self.section("hospital_course", text)

    def section_assessment(self) -> _Node:
        paragraphs = [
            _mixed(
                "paragraph",
                self.person_mention(self.patient),
                " (MRN ",
                self.mention(L.ID_NUM, "patient", "patient.mrn"),
                ") was seen on ",
                self.date_mention(self.encounter_low, "encounter.low", "document"),
                ".",
            )
        ]
        for number, person in enumerate(self.informants, 1):
            mention = self.person_mention(person)
            paragraphs.append(
                _mixed(
                    "paragraph",
                    "History was provided by ",
                    self.content(f"assessment-informant-{number}", mention),
                    ", reachable at ",
                    self.phone_mention("informant", f"informant{number}.phone"),
                    ".",
                )
            )
        paragraphs.append(self.follow_up_paragraph("assessment"))
        text = _el("text", *paragraphs, self.care_team_list("assessment"))
        return self.section("assessment", text)

    def follow_up_paragraph(self, prefix: str) -> _Node:
        return _mixed(
            "paragraph",
            "Follow up with ",
            self.content(f"{prefix}-author", self.person_mention(self.authors[0])),
            " at ",
            self.content(f"{prefix}-custodian", self.custodian_mention()),
            ", telephone ",
            self.phone_mention("custodian", "custodian.phone"),
            ".",
        )

    def care_team_list(self, prefix: str) -> _Node:
        members = [
            ("Author: ", f"{prefix}-team-{number}", person)
            for number, person in enumerate(self.authors, 1)
        ]
        if self.guardian is not None:
            members.append(("Guardian: ", f"{prefix}-team-guardian", self.guardian))
        items = [
            _mixed(
                "item", caption, self.content(content_id, self.person_mention(person))
            )
            for caption, content_id, person in members
        ]
        return _el("list", *items, listType="unordered")

    def custodian_mention(self) -> _Phi:
        return self.mention(L.ORGANIZATION, "custodian", "custodian.name")

    # Rendering --------------------------------------------------------------

    def render(self, root: _Node) -> SyntheticCdaDocument:
        renderer = _Renderer()
        if self.config.stylesheet is not None:
            renderer.prolog = (
                f'<?xml-stylesheet type="text/xsl" href="{self.config.stylesheet}"?>\n'
            )
        xml, raw_truth, content_paths, references = renderer.render(root)

        truth: list[CdaTruthValue] = []
        key_ids: dict[str, str] = {}
        mentions: list[tuple[str, _Phi, str | None]] = []
        for number, item in enumerate(raw_truth, 1):
            truth_id = f"t{number:04d}"
            phi = item.phi
            truth.append(
                CdaTruthValue(
                    truth_id=truth_id,
                    label=phi.label,
                    role=phi.role,
                    context=phi.context,
                    path=item.path,
                    node=item.node,
                    attribute=item.attribute,
                    start=item.start,
                    end=item.start + len(phi.value),
                    document_start=item.document_start,
                    document_end=item.document_start + len(phi.value),
                    value=phi.value,
                )
            )
            if phi.key is not None:
                key_ids[phi.key] = truth_id
            if phi.refers_to:
                mentions.append((truth_id, phi, item.content_id))
        links = tuple(
            CdaMentionLink(
                mention_id=truth_id,
                value_ids=tuple(key_ids[key] for key in phi.refers_to),
                content_id=content_id,
            )
            for truth_id, phi, content_id in mentions
        )
        content_references = tuple(
            CdaContentReference(
                content_id=content_id,
                content_path=content_paths[content_id],
                reference_path=reference_path,
            )
            for content_id, reference_path in references
        )
        try:
            parsed = _safe_parse(xml.encode("utf-8"))
        except ValueError:
            raise CdaSynthesisError("unsafe_output", "rendered XML failed") from None
        if not is_cda_document(xml) or _local_name(parsed.tag) != "ClinicalDocument":
            raise CdaSynthesisError("unsafe_output", "rendered XML is not CDA")
        return SyntheticCdaDocument(
            xml=xml,
            document_type=self.document_type,
            truth=tuple(truth),
            links=links,
            content_references=content_references,
            seed=self.seed,
            index=self.index,
        )


@dataclass(frozen=True)
class _RawTruth:
    phi: _Phi
    path: str
    node: str
    attribute: str | None
    start: int
    document_start: int
    content_id: str | None


class _Renderer:
    def __init__(self) -> None:
        self.prolog = ""
        self.chunks: list[str] = []
        self.cursor = 0
        self.truth: list[_RawTruth] = []
        self.content_paths: dict[str, str] = {}
        self.references: list[tuple[str, str]] = []

    def render(
        self, root: _Node
    ) -> tuple[str, list[_RawTruth], dict[str, str], list[tuple[str, str]]]:
        self.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        self.write(self.prolog)
        self.element(root, "/" + root.tag, 0, inline=False)
        self.write("\n")
        return "".join(self.chunks), self.truth, self.content_paths, self.references

    def write(self, text: str) -> None:
        self.chunks.append(text)
        self.cursor += len(text)

    def pieces(
        self,
        pieces: _Pieces,
        *,
        path: str,
        node: str,
        attribute: str | None = None,
        content_id: str | None = None,
    ) -> None:
        offset = 0
        for piece in pieces:
            if isinstance(piece, _Phi):
                self.truth.append(
                    _RawTruth(
                        piece, path, node, attribute, offset, self.cursor, content_id
                    )
                )
                self.write(piece.value)
                offset += len(piece.value)
            else:
                self.write(_escape(piece, attribute is not None))
                offset += len(piece)

    def element(self, node: _Node, path: str, depth: int, *, inline: bool) -> None:
        self.write("<" + node.tag)
        for name, value in node.attrs:
            self.write(f' {name}="')
            self.pieces(value, path=path, node="attribute", attribute=name)
            self.write('"')
        if node.tag == "content" and node.get("ID"):
            self.content_paths[str(node.get("ID"))] = path
        reference = node.get("value") if node.tag == "reference" else None
        if reference and reference.startswith("#"):
            self.references.append((reference[1:], path))
        if not node.children and not node.text:
            self.write("/>")
            return
        self.write(">")
        content_id = node.get("ID") if node.tag == "content" else None
        self.pieces(node.text, path=path, node="text", content_id=content_id)
        mixed = inline or bool(node.text) or node.tag == "text"
        counts: dict[str, int] = {}
        for child in node.children:
            counts[child.tag] = counts.get(child.tag, 0) + 1
            child_path = f"{path}/{child.tag}[{counts[child.tag]}]"
            if not mixed:
                if child.tail:
                    raise CdaSynthesisError("internal_error", "unexpected tail text")
                self.write("\n" + "  " * (depth + 1))
            self.element(child, child_path, depth + 1, inline=mixed)
            self.pieces(child.tail, path=child_path, node="tail")
        if not mixed:
            self.write("\n" + "  " * depth)
        self.write(f"</{node.tag}>")


# Leakage harness ------------------------------------------------------------


@dataclass(frozen=True)
class CdaResidualPosition:
    """Value-free residual counts for one element path and label.

    Attributes:
        path: Index-free locator such as
            ``/ClinicalDocument/author/time/@value`` or
            ``.../section/text/paragraph/text()``.
        label: Canonical PHI label.
        truth_count: Truth values placed at this path with this label.
        residual_count: Truth values still present in full after redaction.
        partial_count: Multi-token truth values not present in full whose
            tokens of three or more characters partly remain, such as a
            family name left beside a redacted given name.
    """

    path: str
    label: str
    truth_count: int
    residual_count: int
    partial_count: int = 0

    @property
    def residual_rate(self) -> float:
        """Residual share of truth values at this path and label."""

        return self.residual_count / self.truth_count if self.truth_count else 0.0

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible, value-free position record."""

        return {
            "path": self.path,
            "label": self.label,
            "truth_count": self.truth_count,
            "residual_count": self.residual_count,
            "partial_count": self.partial_count,
        }


@dataclass(frozen=True)
class CdaLeakageReport:
    """Value-free residual PHI counts after C-CDA redaction.

    The report holds only index-free element paths, canonical labels and
    counts. It never stores, renders or raises with truth values or XML.

    Attributes:
        document_count: Documents evaluated.
        well_formed_count: Redacted outputs that parsed as CDA without DTD or
            entity declarations.
        truth_count: Truth values evaluated.
        residual_count: Truth values still present in full at their
            locator. Every truth value of a malformed output counts as
            residual.
        partial_count: Multi-token truth values with some tokens remaining.
        positions: Counts for every truth path and label, sorted.
    """

    document_count: int
    well_formed_count: int
    truth_count: int
    residual_count: int
    partial_count: int
    positions: tuple[CdaResidualPosition, ...]

    @property
    def residual_positions(self) -> tuple[CdaResidualPosition, ...]:
        """Positions with at least one residual truth value."""

        return tuple(item for item in self.positions if item.residual_count)

    @property
    def residual_by_path_and_label(self) -> dict[tuple[str, str], int]:
        """Residual counts keyed by ``(path, label)``."""

        return {
            (item.path, item.label): item.residual_count
            for item in self.residual_positions
        }

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible, value-free report."""

        return {
            "schema_version": CDA_SYNTHETIC_SCHEMA_VERSION,
            "document_count": self.document_count,
            "well_formed_count": self.well_formed_count,
            "truth_count": self.truth_count,
            "residual_count": self.residual_count,
            "partial_count": self.partial_count,
            "positions": [item.to_dict() for item in self.positions],
        }

    def to_json(self) -> str:
        """Return the report as deterministic JSON."""

        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))


def evaluate_cda_leakage(
    documents: Iterable[SyntheticCdaDocument],
    *,
    element_map: Sequence[PhiElementRule] | None = None,
    text_redactor: TextRedactor | None = None,
    date_shift_days: int | None = None,
    keep_year: bool = False,
    hash_salt: str = "",
) -> CdaLeakageReport:
    """Redact synthetic documents and count residual truth values.

    Each document runs through :func:`openmed.interop.cda.redact_cda`. The
    output must parse without DTD or entity declarations and pass
    :func:`openmed.interop.cda.is_cda_document`; schemas are never fetched.
    A truth value is residual when its value is still present, as a whole
    token, in the node at the same indexed path. A multi-token value that is
    not residual is partial when one of its tokens of three or more
    characters remains there as a whole token. When a malformed output is
    found, every truth value of that document counts as residual.

    Args:
        documents: Generated documents, bounded by :data:`MAX_CDA_DOCUMENTS`.
        element_map: Optional rules forwarded to ``redact_cda``.
        text_redactor: Optional narrative text callback forwarded to
            ``redact_cda``.
        date_shift_days: Optional fixed date shift forwarded to redaction.
        keep_year: Forwarded to ``redact_cda``.
        hash_salt: Forwarded to ``redact_cda``.

    Returns:
        Value-free residual counts by element path and label.

    Raises:
        CdaLeakageError: If an input is not a generated document, the batch
            is out of bounds, or redaction fails. The error carries a
            controlled code and document index only.
    """

    items = tuple(documents)
    if not 1 <= len(items) <= MAX_CDA_DOCUMENTS:
        raise CdaLeakageError("count_out_of_bounds")
    counts: dict[tuple[str, str], list[int]] = {}
    well_formed = 0
    for index, document in enumerate(items):
        if not isinstance(document, SyntheticCdaDocument):
            raise CdaLeakageError("invalid_document", index)
        try:
            redacted = redact_cda(
                document.xml,
                element_map=element_map,
                text_redactor=text_redactor,
                date_shift_days=date_shift_days,
                keep_year=keep_year,
                hash_salt=hash_salt,
            )
        except Exception:
            # The underlying error may quote document content; drop it.
            raise CdaLeakageError("redaction_failed", index) from None
        root = _redacted_root(redacted)
        if root is not None:
            well_formed += 1
        for truth in document.truth:
            bucket = counts.setdefault((truth.element_path, truth.label), [0, 0, 0])
            bucket[0] += 1
            state = 2 if root is None else _residual_state(root, truth)
            if state:
                bucket[state] += 1

    positions = tuple(
        CdaResidualPosition(path, label, total, residual, partial)
        for (path, label), (total, partial, residual) in sorted(counts.items())
    )
    return CdaLeakageReport(
        document_count=len(items),
        well_formed_count=well_formed,
        truth_count=sum(item.truth_count for item in positions),
        residual_count=sum(item.residual_count for item in positions),
        partial_count=sum(item.partial_count for item in positions),
        positions=positions,
    )


def _redacted_root(redacted: str | bytes) -> ET.Element | None:
    try:
        data = redacted if isinstance(redacted, bytes) else redacted.encode("utf-8")
        root = _safe_parse(data)
    except (ValueError, TypeError, AttributeError):
        return None
    if root.tag != f"{{{CDA_NAMESPACE}}}ClinicalDocument" or not is_cda_document(data):
        return None
    return root


def _residual_state(root: ET.Element, truth: CdaTruthValue) -> int:
    """Return 2 for a full residual, 1 for a partial one, otherwise 0."""

    value = resolve_cda_locator(root, truth)
    if not value:
        return 0
    if _contains_token(value, truth.value):
        return 2
    tokens = [token.strip(",.") for token in truth.value.split()]
    if len(tokens) > 1 and any(
        len(token) >= 3 and _contains_token(value, token) for token in tokens
    ):
        return 1
    return 0


def _contains_token(haystack: str, needle: str) -> bool:
    # A value counts only as a whole token, so "Wren" inside "Wrenfield" does
    # not count, while "Wren" inside "Wren Ashdown" does.
    pattern = rf"(?<![^\W_]){re.escape(needle)}(?![^\W_])"
    return re.search(pattern, haystack) is not None


# Helpers ----------------------------------------------------------------------


def _safe_parse(document: str | bytes) -> ET.Element:
    """Parse XML after refusing DTD and entity declarations, like ``cda.py``."""

    data = document.encode("utf-8") if isinstance(document, str) else document
    if _UNSAFE_XML_DECLARATION.search(data):
        raise ValueError("XML with DOCTYPE or ENTITY declarations is unsupported")
    try:
        return ET.fromstring(data)
    except ET.ParseError:
        raise ValueError("XML is not well-formed") from None


def _find_path(root: ET.Element, path: str) -> ET.Element | None:
    segments = path.strip("/").split("/")
    if not segments or _local_name(root.tag) != segments[0]:
        return None
    namespace = root.tag[1:].split("}", 1)[0] if root.tag.startswith("{") else ""
    element = root
    for segment in segments[1:]:
        match = _PATH_SEGMENT.fullmatch(segment)
        if match is None:
            return None
        tag = match.group("name")
        if namespace:
            tag = f"{{{namespace}}}{tag}"
        position = int(match.group("index") or 1)
        matches = [child for child in element if child.tag == tag]
        if len(matches) < position:
            return None
        element = matches[position - 1]
    return element


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[1] if tag.startswith("{") else tag


def _locator(path: str, node: str, attribute: str | None) -> str:
    if node == "attribute":
        return f"{path}/@{attribute}"
    return f"{path}/{node}()"


def _escape(text: str, attribute: bool) -> str:
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return text.replace('"', "&quot;") if attribute else text


def _checked(value: str) -> str:
    if not _SAFE_VALUE.fullmatch(value) or not value.isascii():
        raise CdaSynthesisError("provider_format", "synthetic value is not XML-safe")
    return value


def _is_local_name(value: Any, pattern: re.Pattern[str]) -> bool:
    return isinstance(value, str) and pattern.fullmatch(value) is not None


def _bounded_int(value: Any, minimum: int, maximum: int) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not minimum <= value <= maximum
    ):
        raise CdaSynthesisError(
            "count_out_of_bounds",
            f"value must be an integer from {minimum} to {maximum}",
        )
    return value


def _count(rng: random.Random, maximum: int) -> int:
    return rng.randint(1, maximum) if maximum else 0


def _derive(seed: int, index: int, name: str) -> int:
    material = f"{SYNTHETIC_CDA_SOURCE}|{seed}|{index}|{name}".encode("utf-8")
    digest = hashlib.blake2b(material, digest_size=8).digest()
    return int.from_bytes(digest, "big", signed=False)


__all__ = [
    "CDA_PLACEHOLDER_CODE_SYSTEM",
    "CDA_PLACEHOLDER_CODE_SYSTEM_NAME",
    "CDA_SYNTHETIC_DOCUMENT_TYPES",
    "CDA_SYNTHETIC_SCHEMA_VERSION",
    "MAX_CDA_DOCUMENTS",
    "MAX_CDA_ENTRIES",
    "MAX_CDA_PARTICIPANTS",
    "SYNTHETIC_CDA_SOURCE",
    "CdaContentReference",
    "CdaDocumentGenerator",
    "CdaLeakageError",
    "CdaLeakageReport",
    "CdaMentionLink",
    "CdaResidualPosition",
    "CdaSynthesisConfig",
    "CdaSynthesisError",
    "CdaTruthValue",
    "SyntheticCdaDocument",
    "evaluate_cda_leakage",
    "generate_cda_documents",
    "resolve_cda_locator",
]
