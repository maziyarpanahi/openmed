"""Bounded, offline HL7 aECG waveform reading without identifying metadata.

This reader deliberately does not normalize leads, redact XML, interpret signals,
or depend on the separate ECG normalization contract.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass, field
from xml.etree import ElementTree as ET

__all__ = [
    "AECG_NOTICE",
    "AecgError",
    "AecgLimits",
    "AecgLead",
    "AecgSeries",
    "AecgRecord",
    "read_aecg",
]

AECG_NOTICE = (
    "Non-diagnostic waveform for review only. Explicit reviewer confirmation "
    "is required before consequential use. No autonomous clinical decisions."
)
_HL7 = "{urn:hl7-org:v3}"
_XSI_TYPE = "{http://www.w3.org/2001/XMLSchema-instance}type"
_ACT_SYSTEM = "2.16.840.1.113883.5.4"
_LEAD_SYSTEM = "2.16.840.1.113883.6.24"
_LEADS = frozenset(
    "MDC_ECG_LEAD_" + lead
    for lead in (
        "I",
        "II",
        "III",
        "AVR",
        "AVL",
        "AVF",
        "V1",
        "V2",
        "V3",
        "V4",
        "V5",
        "V6",
    )
)
_UNITS = frozenset(("uV", "mV", "V"))
_NUMBER = re.compile(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?\Z")
_INTEGER = re.compile(r"[+-]?[0-9]{1,10}\Z")
_UNSAFE = re.compile(r"<!\s*(?:DOCTYPE|ENTITY)\b", re.IGNORECASE)
_ENCODING = re.compile(
    r"^\ufeff?<\?xml\s+[^?]*\bencoding\s*=\s*['\"]([^'\"]+)['\"]", re.IGNORECASE
)
_DEMOGRAPHICS = frozenset(
    _HL7 + name
    for name in (
        "subject",
        "trialSubject",
        "subjectAssignment",
        "subjectDemographicPerson",
    )
)


class AecgError(ValueError):
    """A controlled reason code that never includes submitted XML or paths."""

    def __init__(self, reason_code: str) -> None:
        """Initialize a reader rejection with a controlled code."""
        self.reason_code = reason_code
        super().__init__(reason_code)


@dataclass(frozen=True)
class AecgLimits:
    """Positive resource bounds enforced before decoding complete waveforms."""

    max_bytes: int = 16 * 1024 * 1024
    max_elements: int = 100_000
    max_depth: int = 64
    max_series: int = 16
    max_leads_per_series: int = 64
    max_samples_per_lead: int = 1_000_000
    max_total_samples: int = 2_000_000

    def __post_init__(self) -> None:
        for value in vars(self).values():
            if type(value) is not int or value <= 0:
                raise AecgError("invalid_limits")


@dataclass(frozen=True)
class AecgLead:
    """Original controlled MDC lead code and decoded samples in the source unit."""

    code: str
    unit: str
    samples: tuple[float, ...] = field(repr=False)


@dataclass(frozen=True)
class AecgSeries:
    """Aligned rhythm or derived representative-beat signals with relative timing."""

    kind: str
    sample_interval_seconds: float
    leads: tuple[AecgLead, ...]


@dataclass(frozen=True)
class AecgRecord:
    """Reader-local signals, digest provenance and identifying-surface counts only."""

    rhythm: tuple[AecgSeries, ...]
    derived: tuple[AecgSeries, ...]
    source_digest: str
    demographic_element_count: int
    annotation_count: int

    @property
    def notice(self) -> str:
        """Return the mandatory non-diagnostic boundary."""
        return AECG_NOTICE

    @property
    def demographics_present(self) -> bool:
        """Indicate withheld subject surfaces without returning their contents."""
        return self.demographic_element_count > 0

    @property
    def annotations_present(self) -> bool:
        """Indicate withheld annotations without returning their contents."""
        return self.annotation_count > 0

    def confirm_review(self, *, reviewer_confirmed: bool) -> AecgRecord:
        """Require explicit confirmation before handing signals to consequential use."""
        if reviewer_confirmed is not True:
            raise AecgError("review_required")
        return self

    def report(self) -> dict[str, object]:
        """Return controlled metadata without samples, identifiers or absolute times."""
        return {
            "source_digest": self.source_digest,
            "demographics_present": self.demographics_present,
            "demographic_element_count": self.demographic_element_count,
            "annotations_present": self.annotations_present,
            "annotation_count": self.annotation_count,
            "notice": self.notice,
            "review_required": True,
            "series": [
                {
                    "kind": series.kind,
                    "lead_count": len(series.leads),
                    "sample_count": len(series.leads[0].samples),
                }
                for series in (*self.rhythm, *self.derived)
            ],
        }


class _BoundedTree(ET.TreeBuilder):
    def __init__(self, limits: AecgLimits) -> None:
        super().__init__()
        self.limits = limits
        self.elements = self.depth = self.series = 0

    def start(self, tag: str, attrs: dict[str, str]) -> ET.Element:
        self.elements += 1
        self.depth += 1
        if self.elements > self.limits.max_elements:
            raise AecgError("element_limit")
        if self.depth > self.limits.max_depth:
            raise AecgError("depth_limit")
        if tag in (_HL7 + "series", _HL7 + "derivedSeries"):
            self.series += 1
            if self.series > self.limits.max_series:
                raise AecgError("series_limit")
        if tag in (_HL7 + "reference", "{http://www.w3.org/2001/XInclude}include"):
            raise AecgError("external_reference")
        return super().start(tag, attrs)

    def end(self, tag: str) -> ET.Element:
        self.depth -= 1
        return super().end(tag)

    def doctype(self, name: str, pubid: str, system: str) -> None:
        raise AecgError("unsafe_xml")


def _one(node: ET.Element, name: str) -> ET.Element:
    children = node.findall(_HL7 + name)
    if len(children) != 1:
        raise AecgError("unsupported_layout")
    return children[0]


def _number(text: str | None) -> float:
    if text is None or len(text) > 64 or not _NUMBER.fullmatch(text):
        raise AecgError("invalid_numeric")
    number = float(text)
    if not math.isfinite(number):
        raise AecgError("invalid_numeric")
    return number


def _leaf(node: ET.Element, attributes: frozenset[str]) -> None:
    if len(node) or not set(node.attrib) <= attributes:
        raise AecgError("unsupported_encoding")


def _series(
    node: ET.Element, kind: str, limits: AecgLimits, remaining: int
) -> AecgSeries:
    sequence_sets = node.findall(_HL7 + "component/" + _HL7 + "sequenceSet")
    if len(sequence_sets) != 1:
        raise AecgError("unsupported_layout")
    sequence_set = sequence_sets[0]
    if any(
        child.tag != _HL7 + "component"
        or len(child) != 1
        or child[0].tag != _HL7 + "sequence"
        for child in sequence_set
    ):
        raise AecgError("unsupported_layout")
    sequences = sequence_set.findall(_HL7 + "component/" + _HL7 + "sequence")
    if len(sequences) < 2:
        raise AecgError("unsupported_layout")
    if len(sequences) - 1 > limits.max_leads_per_series:
        raise AecgError("lead_limit")
    leads: list[AecgLead] = []
    interval = None
    used = 0
    for sequence in sequences:
        if len(sequence) != 2:
            raise AecgError("unsupported_layout")
        code = _one(sequence, "code")
        value = _one(sequence, "value")
        if set(value.attrib) != {_XSI_TYPE}:
            raise AecgError("unsupported_encoding")
        name = code.get("code")
        if name in ("TIME_ABSOLUTE", "TIME_RELATIVE"):
            if interval is not None or code.get("codeSystem") != _ACT_SYSTEM:
                raise AecgError("unsupported_layout")
            expected = "GLIST_TS" if name == "TIME_ABSOLUTE" else "GLIST_PQ"
            if value.get(_XSI_TYPE) != expected:
                raise AecgError("unsupported_encoding")
            if {child.tag for child in value} != {_HL7 + "head", _HL7 + "increment"}:
                raise AecgError("unsupported_layout")
            head = _one(value, "head")
            _leaf(head, frozenset(("value", "unit")))
            if head.get("value") is None:
                raise AecgError("unsupported_layout")
            # Absolute acquisition time is deliberately withheld.
            increment = _one(value, "increment")
            _leaf(increment, frozenset(("value", "unit")))
            if increment.get("unit") != "s":
                raise AecgError("unsupported_encoding")
            interval = _number(increment.get("value"))
            if interval <= 0:
                raise AecgError("invalid_numeric")
            continue
        if name not in _LEADS or code.get("codeSystem") != _LEAD_SYSTEM:
            raise AecgError("unsupported_lead")
        if any(lead.code == name for lead in leads):
            raise AecgError("unsupported_layout")
        if value.get(_XSI_TYPE) != "SLIST_PQ":
            raise AecgError("unsupported_encoding")
        if {child.tag for child in value} != {
            _HL7 + "origin",
            _HL7 + "scale",
            _HL7 + "digits",
        }:
            raise AecgError("unsupported_layout")
        origin_node, scale_node = _one(value, "origin"), _one(value, "scale")
        _leaf(origin_node, frozenset(("value", "unit")))
        _leaf(scale_node, frozenset(("value", "unit")))
        unit = origin_node.get("unit")
        if unit not in _UNITS or scale_node.get("unit") != unit:
            raise AecgError("unsupported_encoding")
        origin, scale = (
            _number(origin_node.get("value")),
            _number(scale_node.get("value")),
        )
        if scale == 0:
            raise AecgError("invalid_numeric")
        digits = _one(value, "digits")
        _leaf(digits, frozenset())
        samples: list[float] = []
        for token in re.finditer(r"\S+", digits.text or "", flags=re.ASCII):
            if len(samples) >= limits.max_samples_per_lead or used >= remaining:
                raise AecgError("sample_limit")
            text = token.group()
            if not _INTEGER.fullmatch(text) or not -(2**31) <= int(text) < 2**31:
                raise AecgError("invalid_numeric")
            sample = origin + scale * int(text)
            if not math.isfinite(sample):
                raise AecgError("invalid_numeric")
            samples.append(sample)
            used += 1
        if not samples:
            raise AecgError("unsupported_layout")
        if leads and len(samples) != len(leads[0].samples):
            raise AecgError("inconsistent_lengths")
        leads.append(AecgLead(name, unit, tuple(samples)))
    if interval is None or not leads:
        raise AecgError("unsupported_layout")
    return AecgSeries(kind, interval, tuple(leads))


def read_aecg(data: bytes, *, limits: AecgLimits | None = None) -> AecgRecord:
    """Read a supported aECG XML subset from caller-supplied UTF-8 bytes.

    Args:
        data: In-memory XML, never a path or URL. Only UTF-8 is supported.
        limits: Positive byte, tree, series, lead and sample bounds.

    Returns:
        Separate rhythm and derived records with identifying metadata withheld.

    Raises:
        AecgError: A controlled rejection code for unsafe or unsupported input.
    """
    limits = limits if limits is not None else AecgLimits()
    if not isinstance(limits, AecgLimits):
        raise AecgError("invalid_limits")
    if type(data) is not bytes:
        raise AecgError("invalid_input")
    if len(data) > limits.max_bytes:
        raise AecgError("byte_limit")
    text = None
    try:
        text = data.decode("utf-8")
    except UnicodeError:
        pass
    # Raise outside the handler so the public error does not retain a decoder
    # exception whose .object contains the source bytes.
    if text is None:
        raise AecgError("unsupported_encoding")
    if "\x00" in text:
        raise AecgError("unsupported_encoding")
    declaration = _ENCODING.match(text)
    if declaration and declaration.group(1).lower() not in ("utf-8", "utf8"):
        raise AecgError("unsupported_encoding")
    if _UNSAFE.search(text):
        raise AecgError("unsafe_xml")
    reason = None
    try:
        # Parse decoded text: an XML declaration cannot change the byte encoding.
        root = ET.fromstring(text, parser=ET.XMLParser(target=_BoundedTree(limits)))
    except (ET.ParseError, ValueError) as exc:
        reason = exc.reason_code if isinstance(exc, AecgError) else "malformed_xml"
    if reason is not None:
        raise AecgError(reason)
    if root.tag != _HL7 + "AnnotatedECG":
        raise AecgError("unsupported_layout")
    rhythm_nodes = root.findall(_HL7 + "component/" + _HL7 + "series")
    derived_nodes = [
        derived
        for rhythm in rhythm_nodes
        for derived in rhythm.findall(_HL7 + "derivation/" + _HL7 + "derivedSeries")
    ]
    all_nodes = (*rhythm_nodes, *derived_nodes)
    elements = list(root.iter())
    if not rhythm_nodes or sum(
        element.tag in (_HL7 + "series", _HL7 + "derivedSeries") for element in elements
    ) != len(all_nodes):
        raise AecgError("unsupported_layout")
    # Reject misplaced/nested sequence sets and sequences rather than ignoring them.
    owned_sets = [
        item
        for node in all_nodes
        for item in node.findall(_HL7 + "component/" + _HL7 + "sequenceSet")
    ]
    owned_sequences = [
        item
        for node in owned_sets
        for item in node.findall(_HL7 + "component/" + _HL7 + "sequence")
    ]
    for name, owned in (("sequenceSet", owned_sets), ("sequence", owned_sequences)):
        if sum(element.tag == _HL7 + name for element in elements) != len(owned):
            raise AecgError("unsupported_layout")
    rhythm, derived = [], []
    remaining = limits.max_total_samples
    for node in all_nodes:
        is_rhythm = node.tag == _HL7 + "series"
        code = _one(node, "code")
        if code.get("codeSystem") != _ACT_SYSTEM or code.get("code") != (
            "RHYTHM" if is_rhythm else "REPRESENTATIVE_BEAT"
        ):
            raise AecgError("unsupported_layout")
        series = _series(
            node,
            "rhythm" if is_rhythm else "derived_representative_beat",
            limits,
            remaining,
        )
        remaining -= sum(len(lead.samples) for lead in series.leads)
        (rhythm if is_rhythm else derived).append(series)
    return AecgRecord(
        tuple(rhythm),
        tuple(derived),
        hashlib.sha256(data).hexdigest(),
        sum(element.tag in _DEMOGRAPHICS for element in elements),
        sum(
            element.tag in (_HL7 + "annotation", _HL7 + "annotationSet")
            for element in elements
        ),
    )
