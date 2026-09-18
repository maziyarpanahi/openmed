"""Deterministic linking of already-detected biomarker result mentions.

The linker accepts character-offset mentions from an upstream genomic or
biomarker recognizer and uses the shared constrained span-graph decoder to
assemble clause-local result tuples. It performs no entity recognition,
pathogenicity interpretation, actionability lookup, or therapy matching.

Returned provenance uses half-open UTF-8 byte offsets. This is deliberately
different from the character offsets accepted for mentions: byte offsets make
the emitted record directly auditable against an encoded source document.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from numbers import Real
from types import MappingProxyType
from typing import Literal, TypedDict

from openmed.core.decoding import (
    EdgeCardinality,
    SpanEdge,
    SpanGraphConstraints,
    SpanNode,
    decode_span_graph,
)

ResultPolarity = Literal["detected", "not_detected", "equivocal"]
BiomarkerMentionRole = Literal[
    "gene",
    "variant_or_finding",
    "result_value",
    "method",
]

BIOMARKER_RESULT_ADVISORY = (
    "Biomarker results are deterministic extraction aids for clinician review, "
    "not biological interpretations, pathogenicity or actionability calls, or "
    "therapy recommendations."
)

#: Small, documented, case-insensitive assay lexicon. Values are the only
#: normalizations performed; caller-supplied methods outside the lexicon are
#: preserved exactly rather than mapped to an unsupported assay.
BIOMARKER_METHOD_LEXICON: Mapping[str, str] = MappingProxyType(
    {
        "ngs": "NGS",
        "next generation sequencing": "NGS",
        "next-generation sequencing": "NGS",
        "ihc": "IHC",
        "immunohistochemistry": "IHC",
        "immunohistochemical staining": "IHC",
        "fish": "FISH",
        "fluorescence in situ hybridization": "FISH",
        "fluorescence in-situ hybridization": "FISH",
        "pcr": "PCR",
        "polymerase chain reaction": "PCR",
    }
)

#: Small, documented result lexicon. Longer negated phrases are exact entries,
#: so ``not detected`` can never be mistaken for ``detected``.
BIOMARKER_RESULT_POLARITY_LEXICON: Mapping[str, ResultPolarity] = MappingProxyType(
    {
        "detected": "detected",
        "positive": "detected",
        "present": "detected",
        "amplified": "detected",
        "high": "detected",
        "3+": "detected",
        "mutated": "detected",
        "not detected": "not_detected",
        "negative": "not_detected",
        "absent": "not_detected",
        "wild type": "not_detected",
        "wild-type": "not_detected",
        "no mutation detected": "not_detected",
        "equivocal": "equivocal",
        "indeterminate": "equivocal",
        "borderline": "equivocal",
        "uncertain": "equivocal",
    }
)


class ByteProvenanceSpan(TypedDict):
    """A half-open UTF-8 byte range into the source text."""

    start: int
    end: int
    unit: Literal["utf8_byte"]


class BiomarkerMention(TypedDict, total=False):
    """One already-detected mention supplied to the tuple linker.

    ``start`` and ``end`` are half-open Python character offsets. A mention may
    declare its role through ``label``, ``role``, or ``type``. Result mentions
    with a surface outside the documented polarity lexicon may provide an
    explicit ``polarity`` from an upstream assertion component.
    """

    id: str
    label: str
    role: str
    type: str
    start: int
    end: int
    score: float
    text: str
    text_hash: str
    polarity: ResultPolarity


class BiomarkerResult(TypedDict):
    """One linked biomarker result with source-byte provenance."""

    gene: str | None
    variant_or_finding: str | None
    result_value: str
    method: str | None
    result_polarity: ResultPolarity
    provenance_spans: dict[str, ByteProvenanceSpan]
    advisory: str


_ROLE_ALIASES: Mapping[str, BiomarkerMentionRole] = MappingProxyType(
    {
        "gene": "gene",
        "gene_name": "gene",
        "gene_symbol": "gene",
        "variant": "variant_or_finding",
        "genomic_variant": "variant_or_finding",
        "variant_or_finding": "variant_or_finding",
        "finding": "variant_or_finding",
        "biomarker": "variant_or_finding",
        "result": "result_value",
        "result_value": "result_value",
        "value": "result_value",
        "method": "method",
        "assay": "method",
        "assay_method": "method",
        "test_method": "method",
    }
)

_EDGE_TO_ROLE: Mapping[str, BiomarkerMentionRole] = MappingProxyType(
    {
        "has_gene": "gene",
        "has_variant_or_finding": "variant_or_finding",
        "has_method": "method",
    }
)
_ROLE_TO_EDGE = {role: edge for edge, role in _EDGE_TO_ROLE.items()}
_CLAUSE_BOUNDARY_RE = re.compile(r"[;\n]|(?<!\d)[!?](?!\d)|\.(?=\s+[A-Z])")
_TRIM_RESULT_PUNCTUATION = " \t\r\n.,:;()[]{}"
_DEFAULT_MAX_DISTANCE = 120

_GRAPH_CONSTRAINTS = SpanGraphConstraints(
    allowed_edge_labels=_EDGE_TO_ROLE,
    type_compatibility={
        "has_gene": (("result_value", "gene"),),
        "has_variant_or_finding": (("result_value", "variant_or_finding"),),
        "has_method": (("result_value", "method"),),
    },
    cardinality={
        "has_gene": EdgeCardinality.one_to_one(),
        "has_variant_or_finding": EdgeCardinality.one_to_one(),
        # One explicitly written assay may qualify multiple nearby results.
        "has_method": EdgeCardinality.many_to_one(),
    },
)


def normalize_biomarker_method(value: str) -> str:
    """Normalize a documented assay surface, preserving unknown methods."""

    cleaned = " ".join(value.strip().split())
    return BIOMARKER_METHOD_LEXICON.get(cleaned.casefold(), cleaned)


def normalize_result_polarity(
    value: str,
    *,
    explicit: object | None = None,
) -> ResultPolarity:
    """Return a supported polarity without guessing an unknown assertion.

    Args:
        value: Exact result-value surface from the source text.
        explicit: Optional upstream normalized polarity. This is required when
            ``value`` is outside :data:`BIOMARKER_RESULT_POLARITY_LEXICON`.

    Raises:
        ValueError: If an explicit polarity is invalid or no documented
            normalization exists for the result surface.
    """

    if explicit is not None:
        if not isinstance(explicit, str) or explicit not in {
            "detected",
            "not_detected",
            "equivocal",
        }:
            raise ValueError(
                "result polarity must be detected, not_detected, or equivocal"
            )
        return explicit

    normalized = " ".join(value.strip(_TRIM_RESULT_PUNCTUATION).split()).casefold()
    try:
        return BIOMARKER_RESULT_POLARITY_LEXICON[normalized]
    except KeyError:
        raise ValueError(
            "no documented biomarker result polarity; "
            "provide an explicit polarity on the result mention"
        ) from None


def assemble_biomarker_results(
    text: str,
    mentions: Sequence[BiomarkerMention | Mapping[str, object]],
    *,
    max_distance: int = _DEFAULT_MAX_DISTANCE,
) -> list[BiomarkerResult]:
    """Assemble already-detected mentions into auditable biomarker tuples.

    Result-value mentions anchor tuples. Candidate gene, variant/finding, and
    method mentions must be in the same sentence-like clause and no farther
    than ``max_distance`` characters from the result. The shared span-graph
    decoder chooses the globally optimal role links under deterministic
    cardinality constraints.

    Args:
        text: Source document. Mention offsets index this Python string.
        mentions: Already-detected mention mappings; this function does not run
            NER or enrich mentions from an external knowledge base.
        max_distance: Maximum character gap for candidate graph edges.

    Returns:
        Results ordered by result-value source position. A result is emitted
        only when it has at least one linked gene or variant/finding. Every
        populated clinical field and the derived polarity carry UTF-8 byte
        provenance into ``text``.

    Raises:
        TypeError: If the source or a mention has an invalid type.
        ValueError: If offsets, roles, mention text, or polarity are invalid.
    """

    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if isinstance(max_distance, bool) or not isinstance(max_distance, int):
        raise TypeError("max_distance must be an integer")
    if max_distance < 0:
        raise ValueError("max_distance must be non-negative")

    nodes: list[SpanNode] = []
    mentions_by_id: dict[str, Mapping[str, object]] = {}
    for index, raw_mention in enumerate(mentions):
        node, normalized_mention = _coerce_mention(text, raw_mention, index)
        if node.node_id in mentions_by_id:
            raise ValueError("biomarker mention ids must be unique")
        nodes.append(node)
        mentions_by_id[node.node_id] = normalized_mention

    candidates = _candidate_edges(text, nodes, max_distance=max_distance)
    graph = decode_span_graph(nodes, candidates, constraints=_GRAPH_CONSTRAINTS)
    nodes_by_id = {node.node_id: node for node in graph.nodes}
    linked: dict[str, dict[BiomarkerMentionRole, SpanNode]] = {}
    for edge in graph.edges:
        role = _EDGE_TO_ROLE[edge.label]
        linked.setdefault(edge.head, {})[role] = nodes_by_id[edge.tail]

    results: list[tuple[int, BiomarkerResult]] = []
    for result_node in graph.nodes:
        if result_node.label != "result_value":
            continue
        attributes = linked.get(result_node.node_id, {})
        if not ({"gene", "variant_or_finding"} & attributes.keys()):
            continue
        result_surface = text[result_node.start : result_node.end]
        explicit_polarity = mentions_by_id[result_node.node_id].get("polarity")
        polarity = normalize_result_polarity(
            result_surface,
            explicit=explicit_polarity,
        )
        record = _build_result(
            text,
            result_node=result_node,
            attributes=attributes,
            polarity=polarity,
        )
        results.append((result_node.start, record))
    return [record for _, record in sorted(results, key=lambda item: item[0])]


def _coerce_mention(
    text: str,
    raw_mention: BiomarkerMention | Mapping[str, object],
    index: int,
) -> tuple[SpanNode, Mapping[str, object]]:
    if not isinstance(raw_mention, Mapping):
        raise TypeError("biomarker mentions must be mappings")
    raw_role = (
        raw_mention.get("label") or raw_mention.get("role") or raw_mention.get("type")
    )
    role = _normalize_role(raw_role)
    start = _required_offset(raw_mention, "start")
    end = _required_offset(raw_mention, "end")
    if start < 0 or end <= start or end > len(text):
        raise ValueError(
            "biomarker mention offsets must satisfy 0 <= start < end <= len(text)"
        )
    surface = text[start:end]
    supplied_text = raw_mention.get("text")
    if supplied_text is not None and supplied_text != surface:
        raise ValueError("biomarker mention text does not match its source span")

    node_id = str(raw_mention.get("id") or f"{role}:{start}:{end}:{index}")
    score = raw_mention.get("score")
    if score is not None and (
        isinstance(score, bool) or not isinstance(score, Real) or not 0 <= score <= 1
    ):
        raise ValueError("biomarker mention confidence must be between zero and one")
    text_hash = raw_mention.get("text_hash")
    normalized_mention = dict(raw_mention)
    normalized_mention["role"] = role
    return (
        SpanNode(
            node_id=node_id,
            start=start,
            end=end,
            label=role,
            score=float(score) if score is not None else None,
            text_hash=str(text_hash) if text_hash is not None else None,
        ),
        normalized_mention,
    )


def _required_offset(mention: Mapping[str, object], field: str) -> int:
    try:
        value = mention[field]
    except KeyError:
        raise KeyError(f"biomarker mentions require {field} offsets") from None
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"biomarker mention {field} must be an integer")
    return value


def _normalize_role(raw_role: object) -> BiomarkerMentionRole:
    if not isinstance(raw_role, str):
        raise TypeError("biomarker mention role must be a string")
    normalized = raw_role.strip().casefold().replace("-", "_").replace(" ", "_")
    try:
        return _ROLE_ALIASES[normalized]
    except KeyError:
        allowed = "gene, variant_or_finding, result_value, method"
        raise ValueError(
            f"unknown biomarker mention role; expected {allowed}"
        ) from None


def _candidate_edges(
    text: str,
    nodes: Sequence[SpanNode],
    *,
    max_distance: int,
) -> list[SpanEdge]:
    results = [node for node in nodes if node.label == "result_value"]
    attributes = [node for node in nodes if node.label != "result_value"]
    candidates: list[SpanEdge] = []
    for result in results:
        for attribute in attributes:
            distance = _span_gap(result, attribute)
            if distance > max_distance or not _same_clause(text, result, attribute):
                continue
            role = _normalize_role(attribute.label)
            candidates.append(
                SpanEdge(
                    head=result.node_id,
                    tail=attribute.node_id,
                    label=_ROLE_TO_EDGE[role],
                    score=_edge_score(result, attribute, distance, max_distance),
                    metadata={"distance": distance},
                )
            )
    return candidates


def _span_gap(first: SpanNode, second: SpanNode) -> int:
    return max(first.start - second.end, second.start - first.end, 0)


def _same_clause(text: str, first: SpanNode, second: SpanNode) -> bool:
    between_start = min(first.end, second.end)
    between_end = max(first.start, second.start)
    if between_end <= between_start:
        return True
    # Include the first character of the second mention so the sentence-stop
    # lookahead can distinguish ``. KRAS`` from an in-token period.
    search_end = min(len(text), between_end + 1)
    return _CLAUSE_BOUNDARY_RE.search(text[between_start:search_end]) is None


def _edge_score(
    result: SpanNode,
    attribute: SpanNode,
    distance: int,
    max_distance: int,
) -> float:
    result_score = float(result.score if result.score is not None else 1.0)
    attribute_score = float(attribute.score if attribute.score is not None else 1.0)
    proximity = 1.0 if max_distance == 0 else 1.0 - distance / (max_distance + 1)
    direction_bonus = 0.0
    if (
        attribute.label in {"gene", "variant_or_finding"}
        and attribute.end <= result.start
    ):
        direction_bonus = 0.02
    if attribute.label == "method" and result.end <= attribute.start:
        direction_bonus = 0.02
    return (
        0.58 * ((result_score + attribute_score) / 2.0)
        + 0.4 * proximity
        + direction_bonus
    )


def _build_result(
    text: str,
    *,
    result_node: SpanNode,
    attributes: Mapping[BiomarkerMentionRole, SpanNode],
    polarity: ResultPolarity,
) -> BiomarkerResult:
    gene_node = attributes.get("gene")
    finding_node = attributes.get("variant_or_finding")
    method_node = attributes.get("method")
    result_surface = text[result_node.start : result_node.end]
    provenance: dict[str, ByteProvenanceSpan] = {
        "result_value": _byte_span(text, result_node),
        "result_polarity": _byte_span(text, result_node),
    }
    if gene_node is not None:
        provenance["gene"] = _byte_span(text, gene_node)
    if finding_node is not None:
        provenance["variant_or_finding"] = _byte_span(text, finding_node)
    if method_node is not None:
        provenance["method"] = _byte_span(text, method_node)
    return {
        "gene": _surface(text, gene_node),
        "variant_or_finding": _surface(text, finding_node),
        "result_value": result_surface,
        "method": (
            normalize_biomarker_method(text[method_node.start : method_node.end])
            if method_node is not None
            else None
        ),
        "result_polarity": polarity,
        "provenance_spans": provenance,
        "advisory": BIOMARKER_RESULT_ADVISORY,
    }


def _surface(text: str, node: SpanNode | None) -> str | None:
    if node is None:
        return None
    return text[node.start : node.end]


def _byte_span(text: str, node: SpanNode) -> ByteProvenanceSpan:
    return {
        "start": len(text[: node.start].encode("utf-8")),
        "end": len(text[: node.end].encode("utf-8")),
        "unit": "utf8_byte",
    }


__all__ = [
    "BIOMARKER_METHOD_LEXICON",
    "BIOMARKER_RESULT_ADVISORY",
    "BIOMARKER_RESULT_POLARITY_LEXICON",
    "BiomarkerMention",
    "BiomarkerMentionRole",
    "BiomarkerResult",
    "ByteProvenanceSpan",
    "ResultPolarity",
    "assemble_biomarker_results",
    "normalize_biomarker_method",
    "normalize_result_polarity",
]
