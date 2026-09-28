"""Standalone FHIR R4 CodeableConcept builder for a single coded entity.

Callers frequently have a single (system, code, display, text) tuple — e.g.
from a comparator adapter or a hand-coded mapping — and want a correctly shaped
FHIR ``CodeableConcept`` without going through grounding or the Athena index.

Two public entry points are exposed:

* :func:`coding` builds one ``{"system", "code", "display"}`` Coding dict from
  a vocabulary id (or already-canonical URI) plus a code string.
* :func:`codeable_concept` wraps one or more Codings into a ``CodeableConcept``
  dict, ordering them deterministically by a configurable system priority.

The canonical SYSTEM_URI map is the **single source of truth** for vocabulary
id → HL7 system URI mapping across OpenMed. Import :func:`system_uri` when you
need a canonical URI but not the full ``CodeableConcept`` machinery.

Out of scope: looking up codes against an Athena index or grounding. This module is the lower-level, purely mechanical
helper that higher-level builders can delegate to.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from openmed.clinical.data.doctype_loinc_ontology import (
    get_document_type_mapping,
)

__all__ = [
    "system_uri",
    "coding",
    "codeable_concept",
    "codeable_concept_from_grounded_concept",
    "codeable_concept_from_grounded_span",
    "grounded_concept_to_codeable_concept",
    "document_type_codeable_concept",
    "codeable_concept_from_document_type",
    "codeable_concept_from_document_classification",
]

# Canonical HL7 FHIR R4 system URIs
# https://www.hl7.org/fhir/terminologies-systems.html

_SYSTEM_URI: dict[str, str] = {
    "cvx": "http://hl7.org/fhir/sid/cvx",
    "rxnorm": "http://www.nlm.nih.gov/research/umls/rxnorm",
    "icd10cm": "http://hl7.org/fhir/sid/icd-10-cm",
    "icd-11-mms": "http://id.who.int/icd/release/11/mms",
    "loinc": "http://loinc.org",
    "snomed": "http://snomed.info/sct",
    "hpo": "http://human-phenotype-ontology.org",
    "mesh": "https://www.nlm.nih.gov/mesh",
}

_SYSTEM_ALIASES: dict[str, str] = {
    "cvx": "cvx",
    "rxnorm": "rxnorm",
    "rx-norm": "rxnorm",
    "rx_norm": "rxnorm",
    "icd10": "icd10cm",
    "icd10cm": "icd10cm",
    "icd-10": "icd10cm",
    "icd-10-cm": "icd10cm",
    "icd_10_cm": "icd10cm",
    "loinc": "loinc",
    "snomed": "snomed",
    "snomed-ct": "snomed",
    "snomedct": "snomed",
    "hpo": "hpo",
    "hp": "hpo",
    "mesh": "mesh",
}

# Deterministic ordering for codings inside a CodeableConcept.
# Systems listed earlier sort first; systems absent from this list sort last
# (alphabetically among themselves so the output is still stable).
_DEFAULT_SYSTEM_PRIORITY: tuple[str, ...] = (
    "http://snomed.info/sct",
    "http://hl7.org/fhir/sid/cvx",
    "http://loinc.org",
    "http://www.nlm.nih.gov/research/umls/rxnorm",
    "http://hl7.org/fhir/sid/icd-10-cm",
    "http://id.who.int/icd/release/11/mms",
    "http://human-phenotype-ontology.org",
    "https://www.nlm.nih.gov/mesh",
)

_GROUNDING_ASSIST_EXTENSION_URL = (
    "https://openmed.ai/fhir/StructureDefinition/medical-device-assist"
)
_GROUNDING_ASSIST_ONLY_DISCLAIMER = (
    "Assist-only terminology grounding for human review; not an autonomous "
    "clinical coding, diagnosis, treatment, or billing decision."
)


def system_uri(vocabulary_id: str) -> str:
    """Return the canonical HL7 FHIR R4 system URI for *vocabulary_id*.

    Accepts either a short vocabulary id (case-insensitive) or an
    already-canonical URI.  Already-canonical URIs (those that start with
    ``http://`` or ``https://``) are returned unchanged so callers can
    safely pass either form without checking first.

    Args:
        vocabulary_id: A short vocabulary id such as ``"rxnorm"``,
            ``"cvx"``, ``"loinc"``, ``"snomed"``, ``"icd-10-cm"``, ``"icd-11-mms"``,
            ``"hpo"``, or ``"mesh"``; **or** an already-canonical system URI
            such as ``"http://loinc.org"``.

    Returns:
        The canonical HL7 system URI string.

    Raises:
        ValueError: If *vocabulary_id* is not a recognised short id and does
            not look like a canonical URI (i.e. does not start with
            ``http://`` or ``https://``).
    """
    # Already-canonical URI — pass through unchanged.
    if vocabulary_id.startswith(("http://", "https://")):
        return vocabulary_id

    key = vocabulary_id.lower().strip().replace(" ", "-")
    canonical_key = _SYSTEM_ALIASES.get(key, key)
    if canonical_key not in _SYSTEM_URI:
        raise ValueError(
            f"Unknown vocabulary id: {vocabulary_id!r}. "
            f"Expected one of {sorted(_SYSTEM_URI)} or an already-canonical URI."
        )
    return _SYSTEM_URI[canonical_key]


def coding(
    system: str,
    code: str,
    display: str | None = None,
) -> dict[str, Any]:
    """Build one FHIR R4 Coding dict.

    Args:
        system: A short vocabulary id or
            an already-canonical system URI.  Resolved via :func:`system_uri`.
        code: The code string within the system (e.g. ``"1049502"``).
        display: Optional human-readable display label for the code.

    Returns:
        A ``{"system": ..., "code": ..., "display": ...}`` mapping.
        The ``"display"`` key is omitted when *display* is ``None``.

    Raises:
        ValueError: If *system* is not a recognised vocabulary id or
            canonical URI.
    """
    result: dict[str, Any] = {
        "system": system_uri(system),
        "code": code,
    }
    if display is not None:
        result["display"] = display
    return result


def codeable_concept(
    codings: list[dict[str, Any]],
    text: str | None = None,
    *,
    system_priority: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Build an R4-shaped CodeableConcept from one or more Codings.

    Codings are ordered deterministically by *system_priority* so that output
    is stable regardless of the order in which codings were passed.  Systems
    not present in the priority list are sorted alphabetically after those that
    are listed.

    Args:
        codings: One or more Coding dicts, typically produced by
            :func:`coding`.  Must not be empty.
        text: Optional free-text representation of the concept.  Emitted as
            ``CodeableConcept.text`` when provided.
        system_priority: Ordered tuple of canonical system URIs that controls
            the sort order of codings.  Defaults to
            :data:`_DEFAULT_SYSTEM_PRIORITY`.

    Returns:
        A ``{"coding": [...], "text": ...}`` mapping conforming to the FHIR R4
        ``CodeableConcept`` data type.  The ``"text"`` key is omitted when
        *text* is ``None``.

    Raises:
        ValueError: If *codings* is empty.
    """
    if not codings:
        raise ValueError("codeable_concept requires at least one coding")

    priority = (
        system_priority if system_priority is not None else _DEFAULT_SYSTEM_PRIORITY
    )
    priority_index = {uri: idx for idx, uri in enumerate(priority)}

    def _sort_key(c: dict[str, Any]) -> tuple[int, str, str, str]:
        uri = c.get("system", "")
        return (
            priority_index.get(uri, len(priority)),
            str(uri),
            str(c.get("code", "")),
            str(c.get("display", "")),
        )

    ordered = sorted(codings, key=_sort_key)

    result: dict[str, Any] = {"coding": ordered}
    if text is not None:
        result["text"] = text
    return result


def codeable_concept_from_grounded_concept(
    grounded_concept: Any,
    *,
    text: str | None = None,
    max_codings: int | None = None,
) -> dict[str, Any]:
    """Convert a grounded result into a canonical FHIR ``CodeableConcept``.

    The checked-in grounding facade currently returns ``GroundedSpan`` objects.
    This adapter also accepts the one-system ``GroundedConcept`` shape used by
    newer facade callers through its public attributes, without importing an
    optional or not-yet-available result class. One selected coding per source
    vocabulary is emitted; ranked alternatives remain grounding-layer data and
    are never presented as selected FHIR codes.

    Args:
        grounded_concept: A ``GroundedSpan`` or a compatible grounded concept
            exposing ``text``/``surface_text``, offsets, and candidates.
        text: Optional ``CodeableConcept.text`` override. By default the
            de-identified source surface is used.
        max_codings: Optional positive limit applied after one-per-system
            selection.

    Returns:
        A JSON-serializable FHIR R4 ``CodeableConcept``. Abstentions are
        represented as text-only concepts.

    Raises:
        TypeError: If the grounded result or candidate records are malformed.
        ValueError: If a coded result has invalid offsets, score, system, code,
            or display data.
    """

    if max_codings is not None and max_codings <= 0:
        raise ValueError("max_codings must be positive when provided")

    surface, start, end, candidates, abstained, concept_provenance = (
        _grounded_concept_fields(grounded_concept)
    )
    if end <= start:
        raise ValueError("grounded concept export requires non-empty evidence offsets")
    concept_text = surface if text is None else text

    codings: list[dict[str, Any]] = []
    seen_systems: set[str] = set()
    for candidate in candidates:
        fields = _candidate_fields(candidate)
        system = _grounding_system_uri(fields["system"])
        if system in seen_systems:
            continue
        code = fields["code"]
        display = fields["display"]
        if not isinstance(code, str) or not code.strip():
            raise ValueError("grounded candidate code must be a non-empty string")
        if not isinstance(display, str) or not display.strip():
            raise ValueError("grounded candidate display must be a non-empty string")
        score = float(fields["score"])
        if not math.isfinite(score):
            raise ValueError("grounded candidate score must be finite")

        coding_value = coding(system, code, display)
        version = _first_non_empty(
            fields.get("vocab_version"),
            concept_provenance.get("vocab_version"),
            concept_provenance.get("vocabulary_snapshot_version"),
            concept_provenance.get("snapshot_version"),
        )
        if version is not None:
            coding_value["version"] = version

        grounding_record = {
            "linker": _first_non_empty(
                fields.get("source"),
                concept_provenance.get("linker"),
                concept_provenance.get("linker_name"),
            ),
            "score": score,
            "matched_alias": fields.get("matched_alias"),
            "vocab_version": version,
        }
        # Import lazily: code_provenance imports this module for system_uri.
        from .code_provenance import stamp_grounding_provenance

        coding_value = stamp_grounding_provenance(
            coding_value,
            grounding_record,
            evidence_start=start,
            evidence_end=end,
        )
        codings.append(coding_value)
        seen_systems.add(system)
        if max_codings is not None and len(codings) >= max_codings:
            break

    result = (
        {"text": concept_text}
        if abstained or not codings
        else codeable_concept(codings, text=concept_text)
    )
    result["extension"] = [_grounding_assist_only_extension(start, end)]
    return result


# Explicit names make the adapter discoverable from both vocabulary-oriented
# and grounding-oriented call sites while keeping one implementation.
codeable_concept_from_grounded_span = codeable_concept_from_grounded_concept
grounded_concept_to_codeable_concept = codeable_concept_from_grounded_concept


def _grounded_concept_fields(
    value: Any,
) -> tuple[str, int, int, tuple[Any, ...], bool, dict[str, Any]]:
    """Read GroundedSpan and one-system GroundedConcept shapes safely."""

    if isinstance(value, Mapping):
        source: Any = value
    else:
        source = value
    span = _field(source, "span")
    start = _field(source, "start")
    end = _field(source, "end")
    if span is not None:
        start = start if start is not None else _field(span, "start")
        end = end if end is not None else _field(span, "end")
    surface = _field(source, "text", "surface_text", "surface")
    if surface is None:
        raise TypeError("grounded concept must expose text or surface_text")
    if not isinstance(surface, str):
        raise TypeError("grounded concept surface must be a string")
    if type(start) is not int or start < 0:
        raise ValueError("grounded concept start must be a non-negative integer")
    if type(end) is not int or end < start:
        raise ValueError("grounded concept end must be at or after start")

    raw_provenance = _field(source, "provenance")
    provenance = dict(raw_provenance) if isinstance(raw_provenance, Mapping) else {}
    raw_candidates = _field(source, "candidates")
    candidates: tuple[Any, ...]
    # GroundedConcept is one-system-per-concept. Its ``candidates`` sequence is
    # a top-k review list, so use the selected code on the concept itself.
    selected_system = _field(source, "system")
    selected_code = _field(source, "code")
    selected_display = _field(source, "display")
    is_one_system_concept = span is not None and selected_system is not None
    if is_one_system_concept and selected_code is not None:
        candidates = (
            {
                "system": selected_system,
                "code": selected_code,
                "display": selected_display,
                "confidence": _field(source, "confidence", "score", default=0.0),
                "source": _field(
                    source,
                    "source",
                    default=provenance.get("linker", provenance.get("linker_name")),
                ),
                "matched_alias": provenance.get("matched_alias"),
                "vocabulary_snapshot_version": _field(
                    source,
                    "vocabulary_snapshot_version",
                    default=provenance.get("vocabulary_snapshot_version"),
                ),
            },
        )
    elif raw_candidates is None:
        candidates = ()
    elif isinstance(raw_candidates, (str, bytes)) or not isinstance(
        raw_candidates, Sequence
    ):
        raise TypeError("grounded concept candidates must be a sequence")
    else:
        candidates = tuple(raw_candidates)

    return (
        surface,
        start,
        end,
        candidates,
        bool(_field(source, "abstained", default=False)),
        provenance,
    )


def _candidate_fields(candidate: Any) -> dict[str, Any]:
    """Normalize Candidate and facade candidate attributes for the adapter."""

    system = _field(candidate, "system", "system_uri")
    code = _field(candidate, "code", "concept_id")
    display = _field(candidate, "display", "preferred_term")
    score = _field(candidate, "score", "confidence", default=0.0)
    if system is None or code is None or display is None:
        raise TypeError("grounded candidate must include system, code, and display")
    return {
        "system": system,
        "code": code,
        "display": display,
        "score": score,
        "source": _field(candidate, "source", default=""),
        "matched_alias": _field(candidate, "matched_alias"),
        "vocab_version": _field(
            candidate,
            "vocab_version",
            "vocabulary_snapshot_version",
        ),
    }


def _grounding_system_uri(value: Any) -> str:
    """Resolve grounding system tokens such as ``ICD10CM`` to FHIR URIs."""

    if not isinstance(value, str):
        raise TypeError("grounded candidate system must be a string")
    return system_uri(value)


def _field(source: Any, *names: str, default: Any = None) -> Any:
    """Read the first present field from a mapping or object."""

    for name in names:
        if isinstance(source, Mapping) and name in source:
            return source[name]
        value = getattr(source, name, None)
        if value is not None:
            return value
    return default


def _first_non_empty(*values: Any) -> str | None:
    """Return the first non-empty value as text."""

    for value in values:
        if value is not None and str(value).strip():
            return str(value)
    return None


def _grounding_assist_only_extension(start: int, end: int) -> dict[str, Any]:
    """Return the shared assist-only marker for grounded FHIR output."""

    return {
        "url": _GROUNDING_ASSIST_EXTENSION_URL,
        "extension": [
            {"url": "assist_only", "valueBoolean": True},
            {"url": "autonomous_decision", "valueBoolean": False},
            {"url": "evidence_start", "valueUnsignedInt": start},
            {"url": "evidence_end", "valueUnsignedInt": end},
            {"url": "disclaimer", "valueString": _GROUNDING_ASSIST_ONLY_DISCLAIMER},
        ],
    }


def document_type_codeable_concept(
    document_type: str | Mapping[str, Any],
    *,
    text: str | None = None,
    confidence: float | None = None,
) -> dict[str, Any]:
    """Build a FHIR ``DocumentReference.type`` CodeableConcept.

    The mapping is resolved from the bundled document-type subset rather than
    trusting a caller-supplied code. A classification mapping may be passed
    directly; its confidence is honored when the explicit ``confidence``
    argument is omitted. Unknown or low-confidence values return a valid
    text-only CodeableConcept using the documented no-code sentinel.

    Args:
        document_type: Canonical document type, supported alias, or the mapping
            returned by ``classify_document``.
        text: Optional text override for the CodeableConcept. Supported mapped
            types use their LOINC display label when omitted.
        confidence: Optional confidence threshold input. Values below the local
            LOINC mapping threshold abstain from emitting a Coding.

    Returns:
        A CodeableConcept directly usable as ``DocumentReference.type``.
    """

    raw_type: object = document_type
    if isinstance(document_type, Mapping):
        raw_type = document_type.get("type")
        if confidence is None:
            raw_confidence = document_type.get("confidence")
            if isinstance(raw_confidence, (int, float)) and not isinstance(
                raw_confidence, bool
            ):
                confidence = float(raw_confidence)

    mapping = get_document_type_mapping(raw_type, confidence=confidence)
    if mapping is None:
        fallback_text = text
        if fallback_text is None:
            fallback_text = raw_type.strip() if isinstance(raw_type, str) else "unknown"
        return {"text": fallback_text or "unknown"}

    concept_text = text if text is not None else mapping["label"]
    return codeable_concept(
        [coding("loinc", mapping["code"], mapping["label"])],
        text=concept_text,
    )


codeable_concept_from_document_type = document_type_codeable_concept
codeable_concept_from_document_classification = document_type_codeable_concept
