"""Deterministic laboratory reference-range helpers."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from typing import Literal, TypedDict

from openmed.core.decoding import (
    EdgeCardinality,
    SpanEdge,
    SpanGraph,
    SpanGraphConstraints,
    SpanNode,
    decode_span_graph,
)

from .units import parse_measurement

AbnormalFlag = Literal["low", "normal", "high", "critical", "unknown"]
LabValueAttributeRole = Literal[
    "lab_name",
    "lab_value",
    "reference_range",
    "abnormal_flag",
]


class ReferenceRange(TypedDict, total=False):
    """Parsed numeric reference-range bounds."""

    low: float | None
    high: float | None
    low_inclusive: bool
    high_inclusive: bool
    unit: str


class LabValueAttributeMention(TypedDict, total=False):
    """Already-detected lab mention used by the lab attribute graph linker."""

    id: str
    label: LabValueAttributeRole
    role: LabValueAttributeRole
    type: LabValueAttributeRole
    start: int
    end: int
    score: float
    text_hash: str


class LabValueEventMention(TypedDict, total=False):
    """Lab mention projected into the clinical event role-filling vocabulary."""

    id: str
    label: str
    start: int
    end: int
    score: float
    text_hash: str
    metadata: Mapping[str, object]


LAB_FLAG_ADVISORY = (
    "Derived lab abnormal flags are heuristic and are not a substitute for the "
    "originating laboratory's own formal diagnostic flagging."
)

_EN_DASH = "\u2013"
_EM_DASH = "\u2014"
_LESS_THAN_OR_EQUAL = "\u2264"
_GREATER_THAN_OR_EQUAL = "\u2265"
_NUMERIC = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)"
_UNIT_SUFFIX = r"(?:\s+(?P<unit>\S.*))?"
_RANGE_RE = re.compile(
    rf"^(?P<low>{_NUMERIC})\s*(?:-|to|{_EN_DASH}|{_EM_DASH})\s*"
    rf"(?P<high>{_NUMERIC}){_UNIT_SUFFIX}$",
    re.IGNORECASE,
)
_ONE_SIDED_RE = re.compile(
    rf"^(?P<operator><=|>=|<|>|{_LESS_THAN_OR_EQUAL}|"
    rf"{_GREATER_THAN_OR_EQUAL})\s*(?P<bound>{_NUMERIC}){_UNIT_SUFFIX}$"
)

_EXPLICIT_FLAGS: Mapping[str, AbnormalFlag] = {
    "H": "high",
    "HIGH": "high",
    "L": "low",
    "LOW": "low",
    "C": "critical",
    "CRIT": "critical",
    "CRITICAL": "critical",
    "CRITICAL HIGH": "critical",
    "CRITICAL LOW": "critical",
    "HH": "critical",
    "LL": "critical",
    "N": "normal",
    "NORMAL": "normal",
}

_LAB_VALUE_GRAPH_CONSTRAINTS = SpanGraphConstraints(
    type_compatibility={
        "has_value": (("lab_name", "lab_value"),),
        "has_reference_range": (("lab_value", "reference_range"),),
        "has_abnormal_flag": (("lab_value", "abnormal_flag"),),
    },
    cardinality={
        "has_value": EdgeCardinality.one_to_one(),
        "has_reference_range": EdgeCardinality.one_to_one(),
        "has_abnormal_flag": EdgeCardinality.one_to_one(),
    },
)


def _empty_reference_range() -> ReferenceRange:
    return {
        "low": None,
        "high": None,
        "low_inclusive": True,
        "high_inclusive": True,
    }


def _finite_float(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _set_unit_if_present(result: ReferenceRange, unit: str | None) -> None:
    if unit is None:
        return
    cleaned = unit.strip()
    if cleaned:
        result["unit"] = cleaned


def parse_reference_range(text: object) -> ReferenceRange:
    """Parse a laboratory reference range into numeric bounds.

    Supported forms are closed ranges such as ``"135-145"``, ``"0.5 - 1.2"``,
    and ``"135 to 145"``, plus one-sided bounds such as ``"<5"``, ``"<=5"``,
    ``">10"``, and ``">=10"``. A trailing unit, for example
    ``"70-99 mg/dL"``, is preserved for explicit unit-aware comparison by
    :func:`derive_abnormal_flag`.

    Args:
        text: Raw reference-range text.

    Returns:
        A mapping with ``low``, ``high``, ``low_inclusive``, and
        ``high_inclusive`` keys. Unparseable or contradictory ranges return
        empty bounds instead of guessing.
    """

    result = _empty_reference_range()
    if not isinstance(text, str):
        return result

    normalized = text.strip()
    if not normalized:
        return result

    if range_match := _RANGE_RE.fullmatch(normalized):
        low = _finite_float(range_match.group("low"))
        high = _finite_float(range_match.group("high"))
        if low is None or high is None or low > high:
            return result
        result["low"] = low
        result["high"] = high
        _set_unit_if_present(result, range_match.groupdict().get("unit"))
        return result

    if one_sided_match := _ONE_SIDED_RE.fullmatch(normalized):
        bound = _finite_float(one_sided_match.group("bound"))
        if bound is None:
            return result

        operator = one_sided_match.group("operator")
        if operator in {"<", "<=", _LESS_THAN_OR_EQUAL}:
            result["high"] = bound
            result["high_inclusive"] = operator in {"<=", _LESS_THAN_OR_EQUAL}
        else:
            result["low"] = bound
            result["low_inclusive"] = operator in {">=", _GREATER_THAN_OR_EQUAL}
        _set_unit_if_present(result, one_sided_match.groupdict().get("unit"))
        return result

    return result


def _bool_or_default(value: object, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    return default


def _unit_or_none(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    return cleaned or None


def _normalize_reference_range(
    reference_range: Mapping[str, object] | str | None,
) -> ReferenceRange:
    if isinstance(reference_range, str):
        return parse_reference_range(reference_range)
    if not isinstance(reference_range, Mapping):
        return _empty_reference_range()

    raw_low = reference_range.get("low")
    raw_high = reference_range.get("high")
    low = _finite_float(raw_low)
    high = _finite_float(raw_high)

    if raw_low is not None and low is None:
        return _empty_reference_range()
    if raw_high is not None and high is None:
        return _empty_reference_range()
    if low is not None and high is not None and low > high:
        return _empty_reference_range()

    normalized: ReferenceRange = {
        "low": low,
        "high": high,
        "low_inclusive": _bool_or_default(
            reference_range.get("low_inclusive"),
            True,
        ),
        "high_inclusive": _bool_or_default(
            reference_range.get("high_inclusive"),
            True,
        ),
    }
    unit = _unit_or_none(
        reference_range.get("unit")
        or reference_range.get("units")
        or reference_range.get("reference_unit")
    )
    if unit is not None:
        normalized["unit"] = unit
    return normalized


def _range_unit(
    parsed_range: ReferenceRange,
    explicit_reference_unit: object | None,
) -> str | None:
    return _unit_or_none(explicit_reference_unit) or _unit_or_none(
        parsed_range.get("unit")
    )


def _canonicalize_for_comparison(
    numeric_value: object,
    parsed_range: ReferenceRange,
    *,
    value_unit: object | None,
    reference_unit: object | None,
) -> tuple[float, float | None, float | None] | None:
    range_unit = _range_unit(parsed_range, reference_unit)
    if value_unit is None and range_unit is None:
        value = _finite_float(numeric_value)
        if value is None:
            return None
        return value, parsed_range["low"], parsed_range["high"]

    if range_unit is None:
        return None

    if value_unit is None:
        value_measurement = parse_measurement(numeric_value)
    else:
        value_measurement = parse_measurement(numeric_value, value_unit)
    if value_measurement["status"] != "ok":
        return None

    dimension = value_measurement["dimension"]
    low = parsed_range["low"]
    high = parsed_range["high"]
    canonical_low: float | None = None
    canonical_high: float | None = None

    if low is not None:
        low_measurement = parse_measurement(low, range_unit)
        if (
            low_measurement["status"] != "ok"
            or low_measurement["dimension"] != dimension
        ):
            return None
        canonical_low = low_measurement["canonical_magnitude"]
    if high is not None:
        high_measurement = parse_measurement(high, range_unit)
        if (
            high_measurement["status"] != "ok"
            or high_measurement["dimension"] != dimension
        ):
            return None
        canonical_high = high_measurement["canonical_magnitude"]

    canonical_value = value_measurement["canonical_magnitude"]
    if canonical_value is None:
        return None
    return canonical_value, canonical_low, canonical_high


def derive_abnormal_flag(
    value: object,
    reference_range: Mapping[str, object] | str | None,
    explicit_flag: str | None = None,
    *,
    value_unit: object | None = None,
    reference_unit: object | None = None,
) -> AbnormalFlag:
    """Derive a laboratory abnormal flag from a value and reference range.

    Explicit laboratory flags for high, low, normal, or critical values are
    honored before derived comparisons. Unknown explicit flags return
    ``"unknown"`` instead of being ignored. Critical thresholds beyond the
    reference range are out of scope unless the explicit flag marks the value
    critical. When both value and range units are supplied, comparisons are
    made after deterministic dimensional conversion into shared canonical
    units. Missing, ambiguous, incommensurable, or analyte-dependent units
    return ``"unknown"`` rather than guessing.

    Args:
        value: Numeric lab result value.
        reference_range: Parsed range mapping or raw range text.
        explicit_flag: Optional originating-lab flag such as ``"H"``, ``"L"``,
            or ``"critical"``.
        value_unit: Optional unit for ``value``.
        reference_unit: Optional unit for ``reference_range`` bounds. A unit
            embedded in parsed range mappings or raw range text is also used.

    Returns:
        ``"low"``, ``"normal"``, ``"high"``, ``"critical"``, or ``"unknown"``.
        Non-numeric values and unparseable ranges return ``"unknown"``.
    """

    if explicit_flag is not None:
        normalized_flag = explicit_flag.strip().upper()
        if normalized_flag:
            return _EXPLICIT_FLAGS.get(normalized_flag, "unknown")

    parsed_range = _normalize_reference_range(reference_range)
    low = parsed_range["low"]
    high = parsed_range["high"]
    if low is None and high is None:
        return "unknown"

    canonical = _canonicalize_for_comparison(
        value,
        parsed_range,
        value_unit=value_unit,
        reference_unit=reference_unit,
    )
    if canonical is None:
        return "unknown"

    numeric_value, low, high = canonical

    if low is not None:
        if parsed_range["low_inclusive"] and numeric_value < low:
            return "low"
        if not parsed_range["low_inclusive"] and numeric_value <= low:
            return "low"

    if high is not None:
        if parsed_range["high_inclusive"] and numeric_value > high:
            return "high"
        if not parsed_range["high_inclusive"] and numeric_value >= high:
            return "high"

    return "normal"


def link_lab_value_attributes(
    mentions: list[LabValueAttributeMention | Mapping[str, object]],
    *,
    max_distance: int = 80,
) -> SpanGraph:
    """Link lab name/value/range/flag mentions into a constrained span graph.

    The helper consumes mentions already found by an upstream extractor and
    uses deterministic proximity scores plus graph constraints to attach one
    value to one lab name, and optional range / abnormal flag attributes to one
    value. It does not perform clinical interpretation or unit conversion.
    """

    if max_distance < 0:
        raise ValueError("max_distance must be non-negative")

    nodes = [
        _coerce_lab_attribute_node(raw_mention, index)
        for index, raw_mention in enumerate(mentions)
    ]
    candidates: list[SpanEdge] = []
    candidates.extend(
        _lab_attribute_edges(
            nodes,
            head_label="lab_name",
            tail_label="lab_value",
            edge_label="has_value",
            max_distance=max_distance,
        )
    )
    candidates.extend(
        _lab_attribute_edges(
            nodes,
            head_label="lab_value",
            tail_label="reference_range",
            edge_label="has_reference_range",
            max_distance=max_distance,
        )
    )
    candidates.extend(
        _lab_attribute_edges(
            nodes,
            head_label="lab_value",
            tail_label="abnormal_flag",
            edge_label="has_abnormal_flag",
            max_distance=max_distance,
        )
    )
    return decode_span_graph(
        nodes,
        candidates,
        constraints=_LAB_VALUE_GRAPH_CONSTRAINTS,
    )


def lab_value_event_mentions(graph: SpanGraph) -> list[LabValueEventMention]:
    """Project lab name/value graph nodes into event-extraction mentions.

    The event extractor consumes the generic labels ``analyte`` and
    ``lab_value``. This adapter preserves safe span offsets, node ids, scores,
    hashes, and graph provenance without adding raw source text to the output.
    """

    mentions: list[LabValueEventMention] = []
    for node in graph.nodes:
        if node.label == "lab_name":
            event_label = "analyte"
        elif node.label == "lab_value":
            event_label = "lab_value"
        else:
            continue

        mention: LabValueEventMention = {
            "id": node.node_id,
            "label": event_label,
            "start": node.start,
            "end": node.end,
            "metadata": {
                "source": "lab_value_attribute_graph",
                "graph_node_label": node.label,
            },
        }
        if node.score is not None:
            mention["score"] = node.score
        if node.text_hash is not None:
            mention["text_hash"] = node.text_hash
        mentions.append(mention)
    return mentions


def _coerce_lab_attribute_node(
    raw_mention: LabValueAttributeMention | Mapping[str, object],
    index: int,
) -> SpanNode:
    if not isinstance(raw_mention, Mapping):
        raise TypeError("lab attribute mentions must be mappings")

    raw_label = (
        raw_mention.get("label") or raw_mention.get("role") or raw_mention.get("type")
    )
    label = _lab_attribute_label(raw_label)
    try:
        start = int(raw_mention["start"])
        end = int(raw_mention["end"])
    except KeyError as exc:
        raise KeyError("lab attribute mentions require start and end offsets") from exc
    score = raw_mention.get("score")
    text_hash = raw_mention.get("text_hash")
    node_id = raw_mention.get("id") or f"{label}:{start}:{end}:{index}"
    return SpanNode(
        node_id=str(node_id),
        start=start,
        end=end,
        label=label,
        score=float(score) if score is not None else None,
        text_hash=str(text_hash) if text_hash is not None else None,
    )


def _lab_attribute_label(raw_label: object) -> LabValueAttributeRole:
    if not isinstance(raw_label, str):
        raise TypeError("lab attribute mention label must be a string")
    normalized = raw_label.strip().casefold()
    if normalized in {"lab", "lab_name", "test", "analyte"}:
        return "lab_name"
    if normalized in {"value", "lab_value", "result"}:
        return "lab_value"
    if normalized in {"range", "reference_range", "ref_range"}:
        return "reference_range"
    if normalized in {"flag", "abnormal_flag", "abnormality"}:
        return "abnormal_flag"
    allowed = "lab_name, lab_value, reference_range, abnormal_flag"
    raise ValueError(f"unknown lab attribute label {raw_label!r}; expected {allowed}")


def _lab_attribute_edges(
    nodes: list[SpanNode],
    *,
    head_label: LabValueAttributeRole,
    tail_label: LabValueAttributeRole,
    edge_label: str,
    max_distance: int,
) -> list[SpanEdge]:
    heads = [node for node in nodes if node.label == head_label]
    tails = [node for node in nodes if node.label == tail_label]
    edges: list[SpanEdge] = []
    for head in heads:
        for tail in tails:
            distance = _span_gap(head, tail)
            if distance > max_distance:
                continue
            edges.append(
                SpanEdge(
                    head=head.node_id,
                    tail=tail.node_id,
                    label=edge_label,
                    score=_lab_attribute_score(head, tail, distance, max_distance),
                )
            )
    return edges


def _span_gap(head: SpanNode, tail: SpanNode) -> int:
    if head.end < tail.start:
        return tail.start - head.end
    if tail.end < head.start:
        return head.start - tail.end
    return 0


def _lab_attribute_score(
    head: SpanNode,
    tail: SpanNode,
    distance: int,
    max_distance: int,
) -> float:
    head_score = float(head.score if head.score is not None else 1.0)
    tail_score = float(tail.score if tail.score is not None else 1.0)
    if max_distance == 0:
        proximity = 1.0 if distance == 0 else 0.0
    else:
        proximity = max(0.0, 1.0 - (distance / (max_distance + 1)))
    return (0.7 * ((head_score + tail_score) / 2.0)) + (0.3 * proximity)


__all__ = [
    "AbnormalFlag",
    "LAB_FLAG_ADVISORY",
    "LabValueEventMention",
    "LabValueAttributeMention",
    "LabValueAttributeRole",
    "ReferenceRange",
    "derive_abnormal_flag",
    "lab_value_event_mentions",
    "link_lab_value_attributes",
    "parse_reference_range",
]
