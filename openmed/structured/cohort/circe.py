"""Offline, loss-aware mapping of a declared OHDSI Circe expression subset.

The wire fields follow circe-be at 43c407c9ec514d345c35cb848a0cc16e8dac79e8.
This maps candidate patient predicates, not OHDSI execution or cohort eras.
No vocabulary, SQL generator, transport, or external runtime is included.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from datetime import date
from itertools import combinations
from typing import Any, Final

from openmed.clinical.journey_contracts import canonical_digest, canonical_json
from openmed.structured.store import StoreResult, StoreState

from .dsl import (
    ConceptSet,
    Criterion,
    Expression,
    OccurrenceCount,
    PhenotypeDefinition,
    PhenotypeDefinitionError,
    TemporalWindow,
)
from .exchange import (
    CohortConversionLoss,
    CohortDefinitionExchange,
    CohortExchangeError,
    CohortSourceSnapshot,
    export_cohort_definition,
)

CIRCE_MAPPING_VERSION: Final = "openmed.cohort.circe.v1"
MAX_CIRCE_BYTES: Final = 1_048_576
MAX_CIRCE_DEPTH: Final = 32
MAX_CIRCE_NODES: Final = 20_000
MAX_EXPANDED_CRITERIA: Final = 256
CIRCE_DOMAINS: Final = frozenset(
    {
        "ConditionOccurrence",
        "DrugExposure",
        "Measurement",
        "Observation",
        "ProcedureOccurrence",
    }
)


class CirceInputError(ValueError):
    """Malformed input, with a controlled code and no source values."""

    def __init__(self, code: str = "circe_invalid") -> None:
        self.code = code
        super().__init__(code)


class CirceLimitError(CirceInputError):
    """A JSON or boolean-expansion resource bound was exceeded."""


@dataclass(frozen=True, slots=True)
class CirceConversion:
    """Conversion targets and value-free custody for human loss review.

    Target payloads are protected data, excluded from repr and audit reports.
    ``source_digest``/``target_digest`` bind the full exchange envelope and
    Circe JSON in direction order. Import losses also live in the envelope.
    """

    direction: str
    source_digest: str
    target_digest: str | None
    losses: tuple[CohortConversionLoss, ...]
    exchange: CohortDefinitionExchange | None = field(repr=False)
    _expression_bytes: bytes = field(repr=False)

    def __post_init__(self) -> None:
        expression_digest = canonical_digest(_bounded_json(self._expression_bytes))
        exchange_digest = (
            None if self.exchange is None else self.exchange.canonical_hash
        )
        if self.direction == "export":
            valid = (
                self.source_digest == exchange_digest
                and self.target_digest == expression_digest
            )
        elif self.direction == "import":
            valid = (
                self.source_digest == expression_digest
                and self.target_digest == exchange_digest
            )
            valid = valid and (
                self.exchange is None or self.exchange.losses == self.losses
            )
        else:
            valid = False
        if not valid:
            raise CirceInputError("circe_custody_invalid")

    @property
    def expression(self) -> dict[str, Any]:
        """Return a fresh protected JSON object; mutations cannot alter custody."""
        return json.loads(self._expression_bytes)

    def to_report(self) -> dict[str, Any]:
        """Return only controlled paths, codes, counts, and digests."""
        return {
            "mapping_version": CIRCE_MAPPING_VERSION,
            "direction": self.direction,
            "source_digest": self.source_digest,
            "target_digest": self.target_digest,
            "exchange_digest": None
            if self.exchange is None
            else self.exchange.canonical_hash,
            "definition_digest": None
            if self.exchange is None
            else self.exchange.definition_digest,
            "loss_count": len(self.losses),
            "losses": [loss.to_dict() for loss in self.losses],
        }


def _object(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise CirceInputError()
    return value


def _array(value: Any) -> list[Any]:
    if not isinstance(value, list):
        raise CirceInputError()
    return value


def _integer(value: Any, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise CirceInputError()
    return value


def _boolean(value: Any) -> bool:
    if type(value) is not bool:
        raise CirceInputError()
    return value


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CirceInputError("circe_duplicate_key")
        result[key] = value
    return result


def _bounded_json(value: Mapping[str, Any] | str | bytes | bytearray) -> dict[str, Any]:
    try:
        if isinstance(value, (str, bytes, bytearray)):
            raw = value.encode("utf-8") if isinstance(value, str) else bytes(value)
            if len(raw) > MAX_CIRCE_BYTES:
                raise CirceLimitError("circe_size_limit")
            # Bound nesting before the recursive JSON decoder, ignoring string contents.
            nesting, quoted, escaped = 0, False, False
            for char in raw.decode("utf-8"):
                if quoted:
                    if escaped:
                        escaped = False
                    elif char == "\\":
                        escaped = True
                    elif char == '"':
                        quoted = False
                elif char == '"':
                    quoted = True
                elif char in "[{":
                    nesting += 1
                    if nesting > MAX_CIRCE_DEPTH:
                        raise CirceLimitError("circe_depth_limit")
                elif char in "]}":
                    nesting -= 1
            value = json.loads(raw, object_pairs_hook=_unique_object)
        if not isinstance(value, Mapping):
            raise CirceInputError()
        # Copy and validate mappings too, without serializing unbounded/cyclic objects.
        nodes = 0
        size = 0

        def copy(item: Any, depth: int) -> Any:
            nonlocal nodes, size
            nodes += 1
            size += 2
            if depth > MAX_CIRCE_DEPTH:
                raise CirceLimitError("circe_depth_limit")
            if nodes > MAX_CIRCE_NODES:
                raise CirceLimitError("circe_node_limit")
            result: Any
            if isinstance(item, Mapping):
                result = {}
                for key, child in item.items():
                    if not isinstance(key, str):
                        raise CirceInputError()
                    size += len(key.encode("utf-8"))
                    result[key] = copy(child, depth + 1)
            elif isinstance(item, list):
                result = [copy(child, depth + 1) for child in item]
            elif (
                item is None
                or type(item) in {str, int, bool}
                or (type(item) is float and math.isfinite(item))
            ):
                size += len(str(item).encode("utf-8"))
                result = item
            else:
                raise CirceInputError()
            if size > MAX_CIRCE_BYTES:
                raise CirceLimitError("circe_size_limit")
            return result

        result = copy(value, 1)
        if len(canonical_json(result).encode("utf-8")) > MAX_CIRCE_BYTES:
            raise CirceLimitError("circe_size_limit")
        return result
    except CirceInputError:
        raise
    except (ValueError, TypeError, UnicodeError, RecursionError, OverflowError):
        raise CirceInputError() from None


class _Losses:
    def __init__(self) -> None:
        self.items: list[CohortConversionLoss] = []

    def add(self, path: str, code: str) -> None:
        self.items.append(CohortConversionLoss(path=path, reason_code=code))

    def extras(self, data: dict[str, Any], allowed: set[str], path: str) -> None:
        # Unknown key names can themselves contain PHI: expose only field offsets.
        for index, key in enumerate(sorted(data)):
            if key not in allowed:
                self.add(f"{path}/fields/{index}", "field_unsupported")


def _join(operator: str, children: list[Expression]) -> Expression | None:
    if not children:
        return None
    if len(children) == 1:
        return children[0]
    return Expression(operator=operator, children=tuple(children))  # type: ignore[arg-type]


def _finish(
    direction: str,
    expression: dict[str, Any],
    exchange: CohortDefinitionExchange | None,
    losses: _Losses,
    strict: bool,
    source_digest: str,
) -> StoreResult[CirceConversion]:
    ordered = tuple(
        sorted(set(losses.items), key=lambda loss: (loss.path, loss.reason_code))
    )
    if direction == "import" and exchange is not None:
        exchange = replace(exchange, losses=ordered)
    target_digest = (
        canonical_digest(expression)
        if direction == "export"
        else None
        if exchange is None
        else exchange.canonical_hash
    )
    result = CirceConversion(
        direction,
        source_digest,
        target_digest,
        ordered,
        exchange,
        canonical_json(expression).encode("utf-8"),
    )
    if ordered or exchange is None:
        state = (
            StoreState.UNSUPPORTED if strict or exchange is None else StoreState.PARTIAL
        )
        return StoreResult.outcome(
            state,
            "circe_unsupported" if state is StoreState.UNSUPPORTED else "circe_partial",
            value=result,
        )
    return StoreResult.success(result)


def _date_range(temporal: TemporalWindow) -> dict[str, str] | None:
    start, end = temporal.start_date, temporal.end_date
    if start and end:
        return {"Op": "bt", "Value": start, "Extent": end}
    if start:
        return {"Op": "gte", "Value": start}
    if end:
        return {"Op": "lte", "Value": end}
    return None


def export_circe_expression(
    exchange: CohortDefinitionExchange,
    *,
    criterion_domains: Mapping[str, str],
    primary_criterion_id: str,
    strict: bool = True,
) -> StoreResult[CirceConversion]:
    """Export a protected Circe candidate expression without SQL or vocabulary.

    Args:
        exchange: Existing digest-bound phenotype exchange envelope.
        criterion_domains: Caller-declared Circe domain for every criterion ID.
        primary_criterion_id: Explicit index-event criterion (not inferred).
        strict: Return UNSUPPORTED rather than PARTIAL whenever losses exist.

    Returns:
        A typed result retaining the expression and all semantic losses.

    Raises:
        CirceInputError: Invalid bindings or malformed phenotype.
        CirceLimitError: The bounded JSON or expression budget is exceeded.
    """
    if not isinstance(exchange, CohortDefinitionExchange) or type(strict) is not bool:
        raise CirceInputError()
    definition = exchange.definition
    try:
        _bounded_json(definition.to_dict())
    except RecursionError:
        raise CirceLimitError("circe_depth_limit") from None
    criteria = {criterion.id: criterion for criterion in definition.criteria()}
    if len(criteria) > MAX_EXPANDED_CRITERIA:
        raise CirceLimitError("circe_expansion_limit")
    if set(criterion_domains) != set(criteria) or primary_criterion_id not in criteria:
        raise CirceInputError("circe_domain_binding_invalid")
    if any(domain not in CIRCE_DOMAINS for domain in criterion_domains.values()):
        raise CirceInputError("circe_domain_binding_invalid")
    losses = _Losses()
    # Existing adapter losses may contain caller-authored paths. Retain their
    # custody in the source envelope, but use offsets in this mapper's report.
    for index, _ in enumerate(exchange.losses):
        losses.add(f"/source_losses/{index}", "source_conversion_loss")
    codesets = {item.id: index for index, item in enumerate(definition.concept_sets)}
    for index, _ in enumerate(definition.concept_sets):
        losses.add(f"/ConceptSets/{index}", "vocabulary_filter_unsupported")
    losses.add("/PrimaryCriteria", "observation_period_membership_added")
    losses.add("/definition", "native_metadata_not_preserved")

    def raw(criterion: Criterion, path: str) -> dict[str, Any]:
        domain = criterion_domains[criterion.id]
        body: dict[str, Any] = {"CodesetId": codesets[criterion.concept_set]}
        losses.add(path, "domain_scope_added")
        for axis, values in criterion.assertion.to_dict().items():
            if values:
                losses.add(f"{path}/assertion/{axis}", "assertion_filter_unsupported")
        if criterion.temporal:
            dates = _date_range(criterion.temporal)
            if dates:
                body["OccurrenceStartDate"] = dates
        return {domain: body}

    def leaf(criterion: Criterion, path: str) -> dict[str, Any]:
        window: dict[str, Any] = {"Start": {"Coeff": -1}, "End": {"Coeff": 1}}
        temporal = criterion.temporal
        if temporal and temporal.anchor_criterion:
            if temporal.anchor_criterion != primary_criterion_id:
                losses.add(path + "/StartWindow", "non_primary_anchor_unsupported")
            else:
                window["Start"]["Days"] = temporal.days_before or 0
                window["End"]["Days"] = temporal.days_after or 0
                losses.add(path + "/StartWindow", "index_event_correlation_changed")
        result = {
            "Criteria": raw(criterion, path),
            "StartWindow": window,
            "IgnoreObservationPeriod": True,
            "Occurrence": {"Type": 2, "Count": criterion.occurrence.minimum},
        }
        maximum = criterion.occurrence.maximum
        if maximum is None:
            return {"Type": "ALL", "CriteriaList": [result], "Groups": []}
        upper = {**result, "Occurrence": {"Type": 1, "Count": maximum}}
        return {"Type": "ALL", "CriteriaList": [result, upper], "Groups": []}

    def group(node: Expression, path: str) -> dict[str, Any]:
        if node.criterion:
            return leaf(node.criterion, path)
        children = [
            group(child, f"{path}/Groups/{index}")
            for index, child in enumerate(node.children)
        ]
        if node.operator == "not":
            # Circe AT_MOST(0) is the exact boolean complement of one group.
            return {
                "Type": "AT_MOST",
                "Count": 0,
                "CriteriaList": [],
                "Groups": children,
            }
        return {
            "Type": "ALL" if node.operator == "and" else "ANY",
            "CriteriaList": [],
            "Groups": children,
        }

    def required(node: Expression) -> bool:
        if node.criterion:
            return node.criterion.id == primary_criterion_id
        if node.operator == "not":
            return False
        predicate = any if node.operator == "and" else all
        return predicate(required(child) for child in node.children)

    if not required(definition.expression):
        losses.add("/PrimaryCriteria", "primary_requirement_added")
    primary = criteria[primary_criterion_id]
    if primary.temporal and primary.temporal.anchor_criterion:
        losses.add("/PrimaryCriteria", "primary_relative_window_unsupported")
    expression = {
        "ConceptSets": [
            {
                "id": codesets[item.id],
                "name": f"concept_set_{index}",
                "expression": {
                    "items": [
                        {
                            "concept": {
                                "CONCEPT_ID": concept_id,
                                "VOCABULARY_ID": item.vocabulary,
                            },
                            "includeDescendants": item.include_descendants,
                            "includeMapped": False,
                            "isExcluded": False,
                        }
                        for concept_id in item.concept_ids
                    ]
                },
            }
            for index, item in enumerate(definition.concept_sets)
        ],
        "PrimaryCriteria": {
            "CriteriaList": [raw(primary, "/PrimaryCriteria/CriteriaList/0")],
            "ObservationWindow": {"PriorDays": 0, "PostDays": 0},
            "PrimaryCriteriaLimit": {"Type": "All"},
        },
        "InclusionRules": [
            {
                "name": "predicate",
                "expression": group(
                    definition.expression, "/InclusionRules/0/expression"
                ),
            }
        ],
        "QualifiedLimit": {"Type": "All"},
        "ExpressionLimit": {"Type": "All"},
    }
    expression = _bounded_json(expression)
    return _finish(
        "export", expression, exchange, losses, strict, exchange.canonical_hash
    )


class _Importer:
    def __init__(self, losses: _Losses) -> None:
        self.losses = losses
        self.sets: dict[int, ConceptSet] = {}
        self.declared_sets: set[int] = set()
        self.counter = 0
        self.anchor: str | None = None

    def concept_sets(self, value: Any) -> None:
        for index, item in enumerate(_array(value)):
            path = f"/ConceptSets/{index}"
            data = _object(item)
            self.losses.extras(data, {"id", "name", "expression"}, path)
            if "name" in data:
                self.losses.add(path + "/name", "metadata_not_preserved")
            identifier = _integer(data.get("id"))
            if identifier in self.declared_sets:
                raise CirceInputError("circe_duplicate_codeset")
            self.declared_sets.add(identifier)
            expression = _object(data.get("expression"))
            self.losses.extras(expression, {"items"}, path + "/expression")
            ids, vocabularies, descendants = [], set(), set()
            usable = True
            for offset, entry in enumerate(_array(expression.get("items"))):
                entry_path = f"{path}/expression/items/{offset}"
                entry = _object(entry)
                self.losses.extras(
                    entry,
                    {"concept", "includeDescendants", "includeMapped", "isExcluded"},
                    entry_path,
                )
                concept = _object(entry.get("concept"))
                self.losses.extras(
                    concept, {"CONCEPT_ID", "VOCABULARY_ID"}, entry_path + "/concept"
                )
                ids.append(_integer(concept.get("CONCEPT_ID"), 1))
                vocabulary = concept.get("VOCABULARY_ID")
                if not isinstance(vocabulary, str) or not vocabulary.strip():
                    raise CirceInputError("circe_vocabulary_missing")
                vocabularies.add(vocabulary)
                descendants.add(_boolean(entry.get("includeDescendants", False)))
                for flag in ("includeMapped", "isExcluded"):
                    if _boolean(entry.get(flag, False)):
                        self.losses.add(
                            entry_path + "/" + flag, "concept_set_operation_unsupported"
                        )
                        usable = False
            if not ids:
                raise CirceInputError("circe_empty_codeset")
            if len(vocabularies) != 1 or len(descendants) != 1:
                self.losses.add(path, "mixed_concept_set_unsupported")
                usable = False
            if usable:
                self.sets[identifier] = ConceptSet(
                    f"cs_{identifier}",
                    vocabularies.pop(),
                    tuple(ids),
                    descendants.pop(),
                )
                self.losses.add(path, "vocabulary_filter_added")

    def dates(self, value: Any, path: str) -> dict[str, str]:
        data = _object(value)
        self.losses.extras(data, {"Op", "Value", "Extent"}, path)
        operator = data.get("Op")
        candidates = [data.get("Value")]
        if operator == "bt":
            candidates.append(data.get("Extent"))
        values: list[str] = []
        try:
            for item in candidates:
                if (
                    not isinstance(item, str)
                    or date.fromisoformat(item).isoformat() != item
                ):
                    raise CirceInputError()
                values.append(item)
        except ValueError:
            raise CirceInputError("circe_date_invalid") from None
        if operator == "bt":
            return {"start_date": values[0], "end_date": values[1]}
        if operator in {"gte", "lte", "eq"}:
            result = {}
            if operator != "lte":
                result["start_date"] = values[0]
            if operator != "gte":
                result["end_date"] = values[0]
            if "Extent" in data:
                self.losses.add(path + "/Extent", "date_extent_unsupported")
            return result
        self.losses.add(path, "date_operator_unsupported")
        return {}

    def raw(
        self,
        value: Any,
        path: str,
        temporal: dict[str, Any] | None = None,
        occurrence: OccurrenceCount | None = None,
    ) -> Expression | None:
        data = _object(value)
        if len(data) != 1:
            raise CirceInputError("circe_domain_invalid")
        domain, body = next(iter(data.items()))
        body = _object(body)
        if domain not in CIRCE_DOMAINS:
            code = (
                "era_unsupported"
                if domain in {"ConditionEra", "DrugEra", "DoseEra"}
                else "domain_unsupported"
            )
            self.losses.add(path, code)
            return None
        self.losses.extras(
            body, {"CodesetId", "OccurrenceStartDate"}, path + "/" + domain
        )
        codeset = _integer(body.get("CodesetId"))
        if codeset not in self.declared_sets:
            raise CirceInputError("circe_codeset_reference_invalid")
        window = dict(temporal or {})
        if "OccurrenceStartDate" in body:
            window.update(
                self.dates(
                    body["OccurrenceStartDate"],
                    path + "/" + domain + "/OccurrenceStartDate",
                )
            )
        if codeset not in self.sets:
            self.losses.add(path, "concept_set_unavailable")
            return None
        self.counter += 1
        if self.counter > MAX_EXPANDED_CRITERIA:
            raise CirceLimitError("circe_expansion_limit")
        self.losses.add(path, "domain_scope_not_enforced")
        self.losses.add(path, "assertion_default_added")
        return Expression.leaf(
            Criterion(
                f"criterion_{self.counter}",
                self.sets[codeset].id,
                occurrence=occurrence or OccurrenceCount(),
                temporal=TemporalWindow(**window) if window else None,
            )
        )

    def correlated(self, value: Any, path: str) -> Expression | None:
        data = _object(value)
        self.losses.extras(
            data,
            {
                "Criteria",
                "StartWindow",
                "EndWindow",
                "Occurrence",
                "RestrictVisit",
                "IgnoreObservationPeriod",
            },
            path,
        )
        for flag in ("RestrictVisit", "IgnoreObservationPeriod"):
            if flag in data:
                _boolean(data[flag])
        if data.get("RestrictVisit", False):
            self.losses.add(path + "/RestrictVisit", "visit_restriction_unsupported")
        if not data.get("IgnoreObservationPeriod", False):
            self.losses.add(path, "observation_period_window_unsupported")
        if data.get("EndWindow") is not None:
            self.losses.add(path + "/EndWindow", "end_window_unsupported")
        temporal: dict[str, Any] = {}
        window = _object(data.get("StartWindow"))
        self.losses.extras(
            window,
            {"Start", "End", "UseIndexEnd", "UseEventEnd"},
            path + "/StartWindow",
        )
        endpoints = []
        for field_name in ("Start", "End"):
            endpoint = _object(window.get(field_name))
            self.losses.extras(
                endpoint, {"Days", "Coeff"}, path + "/StartWindow/" + field_name
            )
            coeff = endpoint.get("Coeff")
            if type(coeff) is not int or coeff not in {-1, 1}:
                raise CirceInputError("circe_window_invalid")
            days = endpoint.get("Days")
            endpoints.append(None if days is None else coeff * _integer(days))
        end_based = False
        for flag in ("UseIndexEnd", "UseEventEnd"):
            if flag in window and window[flag] is not None and _boolean(window[flag]):
                self.losses.add(
                    path + "/StartWindow/" + flag, "end_date_anchor_unsupported"
                )
                end_based = True
        start, end = endpoints
        if start is not None or end is not None:
            if start is not None and end is not None and start > end:
                raise CirceInputError("circe_window_invalid")
            if (
                self.anchor
                and start is not None
                and end is not None
                and start <= 0 <= end
                and not end_based
            ):
                temporal = {
                    "anchor_criterion": self.anchor,
                    "days_before": -start,
                    "days_after": end,
                }
                self.losses.add(
                    path + "/StartWindow", "index_event_correlation_changed"
                )
            else:
                self.losses.add(path + "/StartWindow", "start_window_unsupported")
        elif not data.get("IgnoreObservationPeriod", False):
            self.losses.add(
                path + "/StartWindow", "observation_period_window_unsupported"
            )
        count = _object(data.get("Occurrence"))
        self.losses.extras(
            count, {"Type", "Count", "IsDistinct", "CountColumn"}, path + "/Occurrence"
        )
        if "IsDistinct" in count and _boolean(count["IsDistinct"]):
            self.losses.add(
                path + "/Occurrence/IsDistinct", "distinct_count_unsupported"
            )
        if count.get("CountColumn") is not None:
            self.losses.add(
                path + "/Occurrence/CountColumn", "count_column_unsupported"
            )
        kind, amount = _integer(count.get("Type")), _integer(count.get("Count"))
        if kind not in {0, 1, 2}:
            self.losses.add(path + "/Occurrence", "occurrence_type_unsupported")
            self.raw(data.get("Criteria"), path + "/Criteria", temporal)
            return None
        if amount == 0:
            if kind == 2:
                self.losses.add(path + "/Occurrence", "tautology_unsupported")
                self.raw(data.get("Criteria"), path + "/Criteria", temporal)
                return None
            leaf = self.raw(data.get("Criteria"), path + "/Criteria", temporal)
            return None if leaf is None else Expression.exclude(leaf)
        if kind == 0:
            occurrence = OccurrenceCount(amount, amount)
        elif kind == 1:
            # At most N includes zero, which the native positive criterion excludes.
            leaf = self.raw(
                data.get("Criteria"),
                path + "/Criteria",
                temporal,
                OccurrenceCount(amount + 1),
            )
            return None if leaf is None else Expression.exclude(leaf)
        else:
            occurrence = OccurrenceCount(amount)
        return self.raw(data.get("Criteria"), path + "/Criteria", temporal, occurrence)

    def clone(self, node: Expression) -> Expression:
        if node.criterion:
            self.counter += 1
            if self.counter > MAX_EXPANDED_CRITERIA:
                raise CirceLimitError("circe_expansion_limit")
            return Expression.leaf(
                replace(node.criterion, id=f"criterion_{self.counter}")
            )
        return replace(
            node, children=tuple(self.clone(child) for child in node.children)
        )

    def group(self, value: Any, path: str) -> Expression | None:
        data = _object(value)
        self.losses.extras(
            data,
            {"Type", "Count", "CriteriaList", "DemographicCriteriaList", "Groups"},
            path,
        )
        children = [
            self.correlated(item, f"{path}/CriteriaList/{index}")
            for index, item in enumerate(_array(data.get("CriteriaList", [])))
        ]
        children.extend(
            self.group(item, f"{path}/Groups/{index}")
            for index, item in enumerate(_array(data.get("Groups", [])))
        )
        demographics = _array(data.get("DemographicCriteriaList", []))
        for index, item in enumerate(demographics):
            _object(item)
            self.losses.add(
                f"{path}/DemographicCriteriaList/{index}", "demographic_unsupported"
            )
        kind = data.get("Type")
        if not isinstance(kind, str):
            raise CirceInputError("circe_group_invalid")
        if kind not in {"ALL", "ANY", "AT_LEAST", "AT_MOST"}:
            self.losses.add(path, "group_type_unsupported")
            return None
        if demographics or any(child is None for child in children) or not children:
            self.losses.add(path, "group_unrepresentable")
            return None
        nodes = [child for child in children if child is not None]
        if kind in {"ALL", "ANY"}:
            if "Count" in data and data["Count"] is not None:
                self.losses.add(path + "/Count", "unused_group_count")
            return _join("and" if kind == "ALL" else "or", nodes)
        amount = _integer(data.get("Count"))
        if kind == "AT_MOST" and amount == 0:
            child = _join("or", nodes)
            return None if child is None else Expression.exclude(child)
        if kind != "AT_LEAST" or not 1 <= amount <= len(nodes):
            self.losses.add(path, "group_type_unsupported")
            return None
        # Expand the threshold into exact boolean composition, without a new DSL operator.
        budget = math.comb(len(nodes), amount) * sum(
            len(tuple(node.iter_criteria())) for node in nodes
        )
        if budget > MAX_EXPANDED_CRITERIA:
            raise CirceLimitError("circe_expansion_limit")
        terms = [
            _join("and", [self.clone(node) for node in term])
            for term in combinations(nodes, amount)
        ]
        return _join("or", [term for term in terms if term is not None])


def import_circe_expression(
    value: Mapping[str, Any] | str | bytes | bytearray,
    *,
    source_snapshot: CohortSourceSnapshot,
    definition_id: str = "circe_cohort",
    expected_expression_digest: str | None = None,
    strict: bool = True,
) -> StoreResult[CirceConversion]:
    """Import bounded Circe JSON into an existing cohort exchange envelope.

    Args:
        value: Protected JSON bytes/text or a JSON-compatible mapping.
        source_snapshot: Caller-owned vocabulary/data snapshot custody.
        definition_id: Caller-owned native identifier; source names are not copied.
        expected_expression_digest: Optional canonical Circe digest to verify.
        strict: Return UNSUPPORTED for any loss; False permits reviewable PARTIAL.

    Returns:
        A digest-bound result. Unrepresentable boolean groups retain losses but
        have no native exchange; supported lossy predicates retain an envelope.

    Raises:
        CirceInputError: Malformed input, including duplicate keys/references.
        CirceLimitError: Byte, depth, node, or threshold-expansion limit exceeded.
    """
    if (
        not isinstance(source_snapshot, CohortSourceSnapshot)
        or type(strict) is not bool
    ):
        raise CirceInputError()
    expression = _bounded_json(value)
    digest = canonical_digest(expression)
    if expected_expression_digest is not None and expected_expression_digest != digest:
        return StoreResult.outcome(StoreState.CONFLICT, "circe_digest_conflict")
    losses = _Losses()
    importer = _Importer(losses)
    losses.extras(
        expression,
        {
            "ConceptSets",
            "PrimaryCriteria",
            "AdditionalCriteria",
            "InclusionRules",
            "Title",
            "QualifiedLimit",
            "ExpressionLimit",
            "EndStrategy",
            "CensoringCriteria",
            "CollapseSettings",
            "CensorWindow",
            "cdmVersionRange",
        },
        "",
    )
    for key, code in {
        "Title": "metadata_not_preserved",
        "cdmVersionRange": "cdm_version_not_enforced",
        "EndStrategy": "end_strategy_unsupported",
        "CollapseSettings": "era_unsupported",
        "CensorWindow": "censoring_unsupported",
    }.items():
        if key in expression and expression[key] is not None:
            losses.add("/" + key, code)
    for index, item in enumerate(_array(expression.get("CensoringCriteria", []))):
        _object(item)
        losses.add(f"/CensoringCriteria/{index}", "censoring_unsupported")
    try:
        importer.concept_sets(expression.get("ConceptSets"))
        primary = _object(expression.get("PrimaryCriteria"))
        losses.extras(
            primary,
            {"CriteriaList", "ObservationWindow", "PrimaryCriteriaLimit"},
            "/PrimaryCriteria",
        )
        if primary.get("ObservationWindow") is not None:
            losses.add(
                "/PrimaryCriteria/ObservationWindow",
                "observation_period_window_unsupported",
            )
        losses.add("/PrimaryCriteria", "observation_period_membership_not_enforced")
        for path, limit in [
            (
                "/PrimaryCriteria/PrimaryCriteriaLimit",
                primary.get("PrimaryCriteriaLimit"),
            ),
            ("/QualifiedLimit", expression.get("QualifiedLimit")),
            ("/ExpressionLimit", expression.get("ExpressionLimit")),
        ]:
            if limit is not None:
                limit = _object(limit)
                losses.extras(limit, {"Type"}, path)
                if limit.get("Type") != "All":
                    losses.add(path, "event_limit_unsupported")
        primary_nodes = [
            importer.raw(item, f"/PrimaryCriteria/CriteriaList/{index}")
            for index, item in enumerate(_array(primary.get("CriteriaList")))
        ]
        if not primary_nodes:
            raise CirceInputError("circe_primary_missing")
        if len(primary_nodes) == 1 and primary_nodes[0] is not None:
            criterion = primary_nodes[0].criterion
            assert criterion is not None
            importer.anchor = criterion.id
        children = [_join("or", [node for node in primary_nodes if node is not None])]
        representable = all(node is not None for node in primary_nodes)
        if expression.get("AdditionalCriteria") is not None:
            children.append(
                importer.group(expression["AdditionalCriteria"], "/AdditionalCriteria")
            )
        for index, rule in enumerate(_array(expression.get("InclusionRules", []))):
            rule = _object(rule)
            path = f"/InclusionRules/{index}"
            losses.extras(rule, {"name", "description", "expression"}, path)
            for field_name in ("name", "description"):
                if field_name in rule:
                    losses.add(path + "/" + field_name, "metadata_not_preserved")
            children.append(
                importer.group(rule.get("expression"), path + "/expression")
            )
        representable = representable and all(child is not None for child in children)
        native = _join("and", [child for child in children if child is not None])
        exchange = None
        if representable and native is not None and importer.sets:
            definition = PhenotypeDefinition(
                definition_id,
                "Imported Circe candidate",
                tuple(importer.sets.values()),
                native,
            )
            result = export_cohort_definition(
                definition, source_snapshot=source_snapshot
            )
            exchange = result.value
            if exchange is None:
                raise CirceInputError("circe_envelope_invalid")
    except (PhenotypeDefinitionError, CohortExchangeError, TypeError, KeyError):
        raise CirceInputError() from None
    return _finish("import", expression, exchange, losses, strict, digest)
