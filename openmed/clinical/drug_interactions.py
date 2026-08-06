"""Offline drug-drug interaction flagging from caller-supplied data.

The module deliberately ships no clinical interaction knowledge. Production
interaction severity and descriptions are license-sensitive and must be
provided by the caller as an in-memory mapping or a local UTF-8 JSON file. An
accepted JSON document has this shape::

    {
      "schema_version": 1,
      "interactions": [
        {
          "rxcui_1": "123",
          "rxcui_2": "456",
          "severity": "caller-defined",
          "description": "Description from the caller's licensed source.",
          "source_citation": "Dataset name and version"
        }
      ]
    }

``find_interactions`` compares normalized RxCUIs only against that local table.
It has no HTTP client and never uses an online service for an interaction
verdict. A caller-owned RxNorm normalization callback can be enabled explicitly
for otherwise unnormalized medication names; offline mode is the default and
guarantees that callback is not invoked.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Any, TypeAlias

from .decision_support import GuardedSuggestion, SourceSpan, build_guarded_suggestion

INTERACTION_TABLE_SCHEMA_VERSION = 1

DRUG_INTERACTION_ADVISORY = (
    "Potential interaction listed in caller-supplied data. This is a "
    "clinician-review advisory only; independently verify the cited source "
    "before any clinical action."
)

INTERACTION_DATA_NOTICE = (
    "OpenMed includes no production drug-interaction severity data. Callers must "
    "supply a local, license-cleared table and remain responsible for its scope, "
    "currency, and permitted use."
)

InteractionTableSource: TypeAlias = (
    Mapping[Any, Any] | Sequence[Mapping[str, Any]] | str | PathLike[str] | None
)
RxNormLookup: TypeAlias = Callable[[str], Any]

_RXNORM_SYSTEMS = frozenset(
    {
        "rxnorm",
        "rxnorm/rxcui",
        "http://www.nlm.nih.gov/research/umls/rxnorm",
        "https://www.nlm.nih.gov/research/umls/rxnorm",
        "http://purl.bioontology.org/ontology/rxn",
        "https://purl.bioontology.org/ontology/rxn",
        "http://purl.bioontology.org/ontology/rxnorm",
        "https://purl.bioontology.org/ontology/rxnorm",
        "http://rxnav.nlm.nih.gov/rest/rxcui",
        "https://rxnav.nlm.nih.gov/rest/rxcui",
    }
)


class InteractionTableError(ValueError):
    """Raised when a caller-supplied interaction table is malformed."""


@dataclass(frozen=True)
class _Medication:
    rxcui: str | None
    display: str
    source_spans: tuple[SourceSpan, ...]
    index: int


@dataclass(frozen=True)
class _Interaction:
    rxcui_1: str
    rxcui_2: str
    severity: str
    description: str
    source_citation: str


def find_interactions(
    medications: Iterable[Any],
    interaction_table: InteractionTableSource = None,
    *,
    offline: bool = True,
    rxnorm_lookup: RxNormLookup | None = None,
) -> list[GuardedSuggestion]:
    """Flag exact RxCUI pairs found in a caller-supplied local table.

    Bare positive integer strings or integers are treated as RxCUIs. Mapping and
    object inputs may expose ``rxcui`` directly, an RxNorm entry in ``codes``, or
    an RxNorm ``system``/``code`` pair. ``GroundedSpan`` values from the existing
    clinical normalization path therefore work without an adapter. Optional
    ``start``/``end`` or ``source_spans`` values preserve document offsets; for
    bare RxCUIs, deterministic offsets into the normalized medication-list
    representation provide the guardrail's required traceability.

    Unnormalized inputs produce guarded ``normalization_note`` suggestions with
    an explicit ``not_checked`` status. They are never silently interpreted as
    non-interacting. When ``offline`` is ``False``, a caller may provide an
    ``rxnorm_lookup`` callback for normalization only. That callback is never
    used to decide whether an interaction exists.

    Args:
        medications: RxNorm-normalized medications or medication-like values.
        interaction_table: Caller-owned table data, a local JSON file path, or
            ``None``/an empty mapping for no interaction data.
        offline: When ``True`` (the default), never invoke ``rxnorm_lookup``.
        rxnorm_lookup: Optional caller-owned out-of-process normalization
            callback. It is eligible only when ``offline`` is ``False``.

    Returns:
        Guarded clinician-review suggestions in deterministic canonical RxCUI-pair
        order.
        Interaction matches have ``kind == "drug_drug_interaction"``;
        unnormalized inputs have ``kind == "normalization_note"``.

    Raises:
        InteractionTableError: If supplied interaction data is malformed.
        TypeError: If an argument has an unsupported type.

    Note:
        This function flags table matches for review only. It does not suppress,
        substitute, prescribe, or override any medication decision.
    """

    if isinstance(medications, (str, bytes, Mapping)):
        raise TypeError("medications must be an iterable of medication values")
    if not isinstance(offline, bool):
        raise TypeError("offline must be a boolean")
    if rxnorm_lookup is not None and not callable(rxnorm_lookup):
        raise TypeError("rxnorm_lookup must be callable when provided")

    medication_values = list(medications)
    normalized = _normalize_medications(
        medication_values,
        offline=offline,
        rxnorm_lookup=rxnorm_lookup,
    )
    suggestions = [
        _normalization_note(medication)
        for medication in normalized
        if medication.rxcui is None
    ]

    interactions = _load_interaction_table(interaction_table)
    if not interactions:
        return suggestions

    unique_medications: list[_Medication] = []
    seen_rxcuis: set[str] = set()
    for medication in normalized:
        if medication.rxcui is None or medication.rxcui in seen_rxcuis:
            continue
        seen_rxcuis.add(medication.rxcui)
        unique_medications.append(medication)
    unique_medications.sort(
        key=lambda medication: (medication.rxcui or "", medication.index)
    )

    for first_index, first in enumerate(unique_medications):
        for second in unique_medications[first_index + 1 :]:
            first_rxcui = first.rxcui
            second_rxcui = second.rxcui
            if first_rxcui is None or second_rxcui is None:
                continue
            interaction = interactions.get(_pair_key(first_rxcui, second_rxcui))
            if interaction is not None:
                suggestions.append(_interaction_flag(first, second, interaction))

    return suggestions


def _normalize_medications(
    values: Sequence[Any],
    *,
    offline: bool,
    rxnorm_lookup: RxNormLookup | None,
) -> list[_Medication]:
    normalized: list[_Medication] = []
    offset = 0
    for index, value in enumerate(values):
        rxcui = _extract_rxcui(value)
        display = _extract_display(value, index=index, rxcui=rxcui)
        fallback_span = SourceSpan(
            start=offset,
            end=offset + max(len(display), 1),
            label=f"medication_list[{index}]",
        )
        source_spans = _extract_source_spans(value) or (fallback_span,)
        offset = fallback_span.end + 1

        if rxcui is None and not offline and rxnorm_lookup is not None:
            lookup_result = rxnorm_lookup(display)
            rxcui = _extract_rxcui(lookup_result)

        normalized.append(
            _Medication(
                rxcui=rxcui,
                display=display,
                source_spans=source_spans,
                index=index,
            )
        )
    return normalized


def _extract_rxcui(value: Any) -> str | None:
    direct = _as_rxcui(value)
    if direct is not None:
        return direct

    if isinstance(value, Mapping):
        for key in ("rxcui", "rxnorm_code"):
            direct = _as_rxcui(value.get(key))
            if direct is not None:
                return direct

        code = _code_for_system(value.get("system"), value.get("code"))
        if code is not None:
            return code

        code = _rxcui_from_codes(value.get("codes"))
        if code is not None:
            return code
        return _rxcui_from_candidates(value.get("candidates"))

    for attribute in ("rxcui", "rxnorm_code"):
        direct = _as_rxcui(getattr(value, attribute, None))
        if direct is not None:
            return direct

    code = _code_for_system(
        getattr(value, "system", None), getattr(value, "code", None)
    )
    if code is not None:
        return code

    code = _rxcui_from_codes(getattr(value, "codes", None))
    if code is not None:
        return code
    return _rxcui_from_candidates(getattr(value, "candidates", None))


def _as_rxcui(value: Any) -> str | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return str(value) if value > 0 else None
    if not isinstance(value, str):
        return None
    candidate = value.strip()
    if not candidate.isascii() or not candidate.isdecimal():
        return None
    canonical = candidate.lstrip("0")
    return canonical or None


def _is_rxnorm_system(system: Any) -> bool:
    if not isinstance(system, str):
        return False
    return system.strip().casefold().rstrip("/") in _RXNORM_SYSTEMS


def _code_for_system(system: Any, code: Any) -> str | None:
    if _is_rxnorm_system(system):
        return _as_rxcui(code)
    return None


def _rxcui_from_codes(codes: Any) -> str | None:
    if isinstance(codes, Mapping):
        direct = _as_rxcui(codes.get("rxcui"))
        if direct is not None:
            return direct
        direct = _as_rxcui(codes.get("rxnorm_code"))
        if direct is not None:
            return direct
        direct = _code_for_system(codes.get("system"), codes.get("code"))
        if direct is not None:
            return direct
        for system, code in codes.items():
            if _is_rxnorm_system(system):
                direct = _as_rxcui(code)
                if direct is not None:
                    return direct
        return None

    if isinstance(codes, Iterable) and not isinstance(codes, (str, bytes)):
        for coding in codes:
            direct = _extract_rxcui(coding)
            if direct is not None:
                return direct
    return None


def _rxcui_from_candidates(candidates: Any) -> str | None:
    if not isinstance(candidates, Iterable) or isinstance(
        candidates, (str, bytes, Mapping)
    ):
        return None
    for candidate in candidates:
        code = _extract_rxcui(candidate)
        if code is not None:
            return code
    return None


def _extract_display(value: Any, *, index: int, rxcui: str | None) -> str:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped:
            return stripped
    if isinstance(value, Mapping):
        for key in ("name", "text", "display"):
            candidate = value.get(key)
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()
    else:
        for attribute in ("name", "text", "display"):
            candidate = getattr(value, attribute, None)
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()
    if rxcui is not None:
        return f"RxCUI {rxcui}"
    return f"medication at index {index}"


def _extract_source_spans(value: Any) -> tuple[SourceSpan, ...]:
    if isinstance(value, Mapping):
        if "source_spans" in value:
            return _coerce_source_spans(value["source_spans"])
        if "source_span" in value:
            return _coerce_source_spans(value["source_span"])
        if "start" in value or "end" in value:
            return _coerce_source_spans(value)
        return ()

    source_spans = getattr(value, "source_spans", None)
    if source_spans is not None:
        return _coerce_source_spans(source_spans)
    start = getattr(value, "start", None)
    end = getattr(value, "end", None)
    if type(start) is int and type(end) is int and end > start >= 0:
        return (SourceSpan.from_obj(value),)
    return ()


def _coerce_source_spans(value: Any) -> tuple[SourceSpan, ...]:
    if value is None:
        return ()
    if isinstance(value, (SourceSpan, Mapping)):
        values = (value,)
    elif isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        values = tuple(value)
    else:
        values = (value,)
    return tuple(SourceSpan.from_obj(span) for span in values)


def _normalization_note(medication: _Medication) -> GuardedSuggestion:
    return build_guarded_suggestion(
        {
            "kind": "normalization_note",
            "status": "not_checked",
            "medication": medication.display,
            "medication_index": medication.index,
            "note": (
                "Could not normalize this medication to an RxCUI; drug-drug "
                "interactions were not checked for it."
            ),
            "review": "clinician_review_required",
        },
        medication.source_spans,
        1.0,
        provenance={
            "producer": "openmed.clinical.drug_interactions",
            "reason": "missing_rxcui",
        },
    )


def _interaction_flag(
    first: _Medication,
    second: _Medication,
    interaction: _Interaction,
) -> GuardedSuggestion:
    return build_guarded_suggestion(
        {
            "kind": "drug_drug_interaction",
            "medications": [
                {"rxcui": first.rxcui, "display": first.display},
                {"rxcui": second.rxcui, "display": second.display},
            ],
            "severity": interaction.severity,
            "source_description": interaction.description,
            "source_citation": interaction.source_citation,
            "advisory": DRUG_INTERACTION_ADVISORY,
            "review": "clinician_review_required",
        },
        (*first.source_spans, *second.source_spans),
        1.0,
        provenance={
            "producer": "openmed.clinical.drug_interactions",
            "match": "exact_user_supplied_rxcui_pair",
            "source_citation": interaction.source_citation,
            "interaction_data": "caller_supplied",
        },
    )


def _load_interaction_table(
    source: InteractionTableSource,
) -> dict[tuple[str, str], _Interaction]:
    if source is None:
        return {}

    payload: Any = source
    if isinstance(source, (str, PathLike)):
        path_text = str(source)
        if "://" in path_text:
            raise InteractionTableError(
                "interaction_table must be a local file path, not a URL"
            )
        path = Path(source)
        try:
            with path.open(encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise InteractionTableError(
                f"could not load local interaction table {path}"
            ) from exc

    rows, default_citation = _interaction_rows(payload)
    interactions: dict[tuple[str, str], _Interaction] = {}
    for index, row in enumerate(rows):
        interaction = _parse_interaction(row, index, default_citation)
        key = _pair_key(interaction.rxcui_1, interaction.rxcui_2)
        existing = interactions.get(key)
        if existing is not None and existing != interaction:
            raise InteractionTableError(
                f"interaction table contains conflicting records for pair {key}"
            )
        interactions[key] = interaction
    return interactions


def _interaction_rows(
    payload: Any,
) -> tuple[list[Mapping[str, Any]], str | None]:
    if isinstance(payload, Mapping):
        _validate_schema_version(payload)
        default_citation = _table_citation(payload)
        if "interactions" in payload:
            raw_rows = payload["interactions"]
            if not isinstance(raw_rows, Sequence) or isinstance(raw_rows, (str, bytes)):
                raise InteractionTableError("interactions must be a JSON array")
            rows = list(raw_rows)
        elif not payload:
            rows = []
        elif _looks_like_interaction(payload):
            rows = [payload]
        else:
            rows = _pair_mapping_rows(payload)
    elif isinstance(payload, Sequence) and not isinstance(payload, (str, bytes)):
        default_citation = None
        rows = list(payload)
    else:
        raise InteractionTableError(
            "interaction_table must be a mapping, record sequence, or local JSON path"
        )

    if any(not isinstance(row, Mapping) for row in rows):
        raise InteractionTableError("every interaction record must be a mapping")
    return rows, default_citation


def _validate_schema_version(payload: Mapping[Any, Any]) -> None:
    if "schema_version" not in payload:
        return
    version = payload["schema_version"]
    if type(version) is not int or version != INTERACTION_TABLE_SCHEMA_VERSION:
        raise InteractionTableError(
            "interaction table schema_version must be "
            f"{INTERACTION_TABLE_SCHEMA_VERSION}"
        )


def _table_citation(payload: Mapping[Any, Any]) -> str | None:
    direct = payload.get("source_citation")
    if isinstance(direct, str) and direct.strip():
        return direct.strip()
    metadata = payload.get("metadata")
    if isinstance(metadata, Mapping):
        for key in ("source_citation", "citation"):
            value = metadata.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def _looks_like_interaction(payload: Mapping[Any, Any]) -> bool:
    return any(
        key in payload
        for key in ("rxcuis", "rxcui_pair", "rxcui_1", "rxcui1", "rxcui_a")
    )


def _pair_mapping_rows(payload: Mapping[Any, Any]) -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    metadata_keys = {
        "schema_version",
        "metadata",
        "source_citation",
        "notice",
    }
    for raw_pair, raw_record in payload.items():
        if raw_pair in metadata_keys:
            continue
        first, second = _split_pair(raw_pair)
        if not isinstance(raw_record, Mapping):
            raise InteractionTableError(
                f"interaction record for pair {raw_pair!r} must be a mapping"
            )
        row = dict(raw_record)
        row.setdefault("rxcui_1", first)
        row.setdefault("rxcui_2", second)
        rows.append(row)
    return rows


def _split_pair(value: Any) -> tuple[Any, Any]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        parts = list(value)
        if len(parts) == 2:
            return parts[0], parts[1]
    if isinstance(value, str):
        for separator in ("|", ",", "+"):
            parts = value.split(separator)
            if len(parts) == 2:
                return parts[0].strip(), parts[1].strip()
    raise InteractionTableError(
        f"interaction pair key {value!r} must contain two RxCUIs"
    )


def _parse_interaction(
    row: Mapping[str, Any],
    index: int,
    default_citation: str | None,
) -> _Interaction:
    first, second = _record_pair(row, index)
    if first == second:
        raise InteractionTableError(
            f"interaction record {index} must contain two different RxCUIs"
        )
    severity = _required_text(row.get("severity"), index, "severity")
    description = _required_text(row.get("description"), index, "description")
    citation = _record_citation(row) or default_citation
    if citation is None:
        raise InteractionTableError(
            f"interaction record {index} requires a source citation"
        )
    canonical_first, canonical_second = _pair_key(first, second)
    return _Interaction(
        canonical_first,
        canonical_second,
        severity,
        description,
        citation,
    )


def _record_pair(row: Mapping[str, Any], index: int) -> tuple[str, str]:
    pair = row.get("rxcuis", row.get("rxcui_pair"))
    if pair is not None:
        try:
            raw_first, raw_second = _split_pair(pair)
        except InteractionTableError as exc:
            raise InteractionTableError(
                f"interaction record {index} requires exactly two RxCUIs"
            ) from exc
    else:
        raw_first = row.get("rxcui_1", row.get("rxcui1", row.get("rxcui_a")))
        raw_second = row.get("rxcui_2", row.get("rxcui2", row.get("rxcui_b")))

    first = _as_rxcui(raw_first)
    second = _as_rxcui(raw_second)
    if first is None or second is None:
        raise InteractionTableError(
            f"interaction record {index} requires two positive integer RxCUIs"
        )
    return first, second


def _record_citation(row: Mapping[str, Any]) -> str | None:
    direct = row.get("source_citation")
    if isinstance(direct, str) and direct.strip():
        return direct.strip()
    source = row.get("source")
    if isinstance(source, str) and source.strip():
        return source.strip()
    if isinstance(source, Mapping):
        citation = source.get("citation")
        if isinstance(citation, str) and citation.strip():
            return citation.strip()
    return None


def _required_text(value: Any, index: int, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InteractionTableError(
            f"interaction record {index} requires a non-empty {field}"
        )
    return value.strip()


def _pair_key(first: str, second: str) -> tuple[str, str]:
    if first <= second:
        return first, second
    return second, first


__all__ = [
    "DRUG_INTERACTION_ADVISORY",
    "INTERACTION_DATA_NOTICE",
    "INTERACTION_TABLE_SCHEMA_VERSION",
    "InteractionTableError",
    "InteractionTableSource",
    "RxNormLookup",
    "find_interactions",
]
