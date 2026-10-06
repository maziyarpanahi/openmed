"""Synthetic Circe contract, resource-bound, loss, and privacy controls."""

from __future__ import annotations

import json
from copy import deepcopy

import pytest

from openmed.clinical.journey_contracts import canonical_digest
from openmed.structured.cohort import (
    AssertionFilter,
    CirceInputError,
    CirceLimitError,
    CohortSourceSnapshot,
    ConceptSet,
    Criterion,
    Expression,
    OccurrenceCount,
    PhenotypeDefinition,
    TemporalWindow,
    export_circe_expression,
    export_cohort_definition,
    import_circe_expression,
)
from openmed.structured.cohort.circe import MAX_CIRCE_BYTES, MAX_CIRCE_DEPTH
from openmed.structured.store import StoreState


def snapshot():
    return CohortSourceSnapshot(
        "snapshot_circesynthetic01",
        "sha256:" + "a" * 64,
        "synthetic-v1",
        ("synthetic",),
    )


def expression():
    return {
        "ConceptSets": [
            {
                "id": 0,
                "expression": {
                    "items": [
                        {
                            "concept": {
                                "CONCEPT_ID": 900001,
                                "VOCABULARY_ID": "Synthetic",
                            },
                            "includeDescendants": True,
                        }
                    ]
                },
            }
        ],
        "PrimaryCriteria": {
            "CriteriaList": [{"ConditionOccurrence": {"CodesetId": 0}}],
            "PrimaryCriteriaLimit": {"Type": "All"},
        },
        "InclusionRules": [],
    }


def correlated(domain="ConditionOccurrence", kind=2, count=1, start=None, end=None):
    return {
        "Criteria": {domain: {"CodesetId": 0}},
        "Occurrence": {"Type": kind, "Count": count},
        "StartWindow": {
            "Start": {"Coeff": -1, "Days": start},
            "End": {"Coeff": 1, "Days": end},
        },
        "IgnoreObservationPeriod": True,
    }


def group(*criteria, kind="ALL", count=None):
    result = {
        "Type": kind,
        "CriteriaList": list(criteria),
        "Groups": [],
        "DemographicCriteriaList": [],
    }
    if count is not None:
        result["Count"] = count
    return result


def import_value(value, **kwargs):
    return import_circe_expression(
        value, source_snapshot=snapshot(), strict=False, **kwargs
    )


def codes(result):
    return {loss.reason_code for loss in result.value.losses}


@pytest.mark.parametrize(
    "domain",
    [
        "ConditionOccurrence",
        "DrugExposure",
        "Measurement",
        "Observation",
        "ProcedureOccurrence",
    ],
)
def test_domains_ids_descendants_and_custody(domain):
    value = expression()
    value["PrimaryCriteria"]["CriteriaList"] = [{domain: {"CodesetId": 0}}]
    imported = import_value(json.dumps(value))
    assert imported.state is StoreState.PARTIAL
    conversion = imported.value
    envelope = conversion.exchange
    assert envelope.definition.concept_sets[0].concept_ids == (900001,)
    assert envelope.definition.concept_sets[0].include_descendants
    assert conversion.source_digest == canonical_digest(value)
    assert conversion.target_digest == envelope.canonical_hash
    assert conversion.losses == envelope.losses
    assert "domain_scope_not_enforced" in codes(imported)
    assert "assertion_default_added" in codes(imported)
    assert (
        import_circe_expression(value, source_snapshot=snapshot()).state
        is StoreState.UNSUPPORTED
    )
    assert (
        import_value(value, expected_expression_digest="sha256:" + "b" * 64).state
        is StoreState.CONFLICT
    )
    conversion.expression["ConceptSets"].clear()
    assert conversion.expression == value


@pytest.mark.parametrize(
    "kind,count,expected,excluded",
    [
        (0, 3, (3, 3), False),
        (2, 4, (4, None), False),
        (1, 2, (3, None), True),
        (0, 0, (1, None), True),
    ],
)
def test_occurrence_semantics_include_zero(kind, count, expected, excluded):
    value = expression()
    value["AdditionalCriteria"] = group(correlated(kind=kind, count=count))
    result = import_value(value)
    criterion = result.value.exchange.definition.criteria()[-1]
    assert (criterion.occurrence.minimum, criterion.occurrence.maximum) == expected
    node = result.value.exchange.definition.expression.children[-1]
    assert (node.operator == "not") is excluded


@pytest.mark.parametrize(
    "kind,operator", [("ALL", "and"), ("ANY", "or"), ("AT_LEAST", "or")]
)
def test_groups_lower_to_existing_dsl_with_unique_ids(kind, operator):
    value = expression()
    value["AdditionalCriteria"] = group(
        correlated(),
        correlated("DrugExposure"),
        correlated("Measurement"),
        kind=kind,
        count=2 if kind == "AT_LEAST" else None,
    )
    result = import_value(value)
    definition = result.value.exchange.definition
    assert definition.expression.children[-1].operator == operator
    assert len({item.id for item in definition.criteria()}) == len(
        definition.criteria()
    )
    assert len(definition.criteria()) == (7 if kind == "AT_LEAST" else 4)


def test_start_window_anchors_to_single_primary_and_dates():
    value = expression()
    value["AdditionalCriteria"] = group(correlated(start=5, end=10))
    value["AdditionalCriteria"]["CriteriaList"][0]["Criteria"]["ConditionOccurrence"][
        "OccurrenceStartDate"
    ] = {"Op": "bt", "Value": "2026-01-01", "Extent": "2026-12-31"}
    result = import_value(value)
    primary, inclusion = result.value.exchange.definition.criteria()
    assert inclusion.temporal == TemporalWindow(
        "2026-01-01", "2026-12-31", primary.id, 5, 10
    )
    assert "index_event_correlation_changed" in codes(result)


@pytest.mark.parametrize(
    "construct,path,code",
    [
        (
            {"EndStrategy": {"CustomEra": {"DrugCodesetId": 0}}},
            "/EndStrategy",
            "end_strategy_unsupported",
        ),
        (
            {"CollapseSettings": {"CollapseType": "ERA", "EraPad": 0}},
            "/CollapseSettings",
            "era_unsupported",
        ),
        (
            {"CensoringCriteria": [{"ConditionOccurrence": {"CodesetId": 0}}]},
            "/CensoringCriteria/0",
            "censoring_unsupported",
        ),
        (
            {"CensorWindow": {"StartDate": "2026-01-01"}},
            "/CensorWindow",
            "censoring_unsupported",
        ),
        (
            {"private-name-Иван": {"nested": "private-value"}},
            "/fields/3",
            "field_unsupported",
        ),
    ],
)
def test_unsupported_top_level_constructs_are_reported(construct, path, code):
    value = {**expression(), **construct}
    result = import_value(value)
    assert any(
        loss.path == path and loss.reason_code == code for loss in result.value.losses
    )


def test_all_losses_collected_even_when_boolean_group_cannot_be_represented():
    value = expression()
    value["AdditionalCriteria"] = group(correlated("DrugEra"))
    value["AdditionalCriteria"]["DemographicCriteriaList"] = [
        {"Age": {"Value": 50, "Op": "gte"}},
        {"Gender": []},
    ]
    value["EndStrategy"] = {"DateOffset": {"Offset": 20}}
    value["CensoringCriteria"] = [{"Death": {}}, {"ObservationPeriod": {}}]
    result = import_value(value)
    assert result.state is StoreState.UNSUPPORTED
    assert result.value.exchange is None
    assert {
        "era_unsupported",
        "demographic_unsupported",
        "group_unrepresentable",
        "end_strategy_unsupported",
        "censoring_unsupported",
    } <= codes(result)
    assert (
        sum(
            loss.reason_code == "demographic_unsupported"
            for loss in result.value.losses
        )
        == 2
    )
    assert (
        sum(loss.reason_code == "censoring_unsupported" for loss in result.value.losses)
        == 2
    )


@pytest.mark.parametrize(
    "mutate,code",
    [
        (lambda x: x.update(EndWindow={}), "end_window_unsupported"),
        (lambda x: x.update(RestrictVisit=True), "visit_restriction_unsupported"),
        (
            lambda x: x.update(IgnoreObservationPeriod=False),
            "observation_period_window_unsupported",
        ),
        (
            lambda x: x["StartWindow"].update(UseIndexEnd=True),
            "end_date_anchor_unsupported",
        ),
        (
            lambda x: x["Occurrence"].update(IsDistinct=True),
            "distinct_count_unsupported",
        ),
        (
            lambda x: x["Occurrence"].update(CountColumn="start_date"),
            "count_column_unsupported",
        ),
        (
            lambda x: x["Criteria"]["ConditionOccurrence"].update(Age={"Value": 50.5}),
            "field_unsupported",
        ),
    ],
)
def test_unsupported_nested_filters_never_become_success(mutate, code):
    value = expression()
    criterion = correlated(start=2, end=3)
    mutate(criterion)
    value["AdditionalCriteria"] = group(criterion)
    result = import_value(value)
    assert code in codes(result)
    assert not result.ok


@pytest.mark.parametrize("flag", ["isExcluded", "includeMapped"])
def test_unsupported_concept_set_operations_do_not_widen_membership(flag):
    value = expression()
    value["ConceptSets"][0]["expression"]["items"][0][flag] = True
    result = import_value(value)
    assert result.value.exchange is None
    assert "concept_set_operation_unsupported" in codes(result)


def test_export_count_range_assertions_domains_and_binding():
    assertions = AssertionFilter(
        temporality=("recent",), certainty=("certain",), experiencer=("patient",)
    )
    primary = Criterion("condition", "cs", OccurrenceCount(2, 4), assertion=assertions)
    secondary = Criterion(
        "drug",
        "cs",
        temporal=TemporalWindow(anchor_criterion="condition", days_after=30),
    )
    definition = PhenotypeDefinition(
        "synthetic",
        "Synthetic",
        (ConceptSet("cs", "Synthetic", (900001,)),),
        Expression.all_of(Expression.leaf(primary), Expression.leaf(secondary)),
    )
    envelope = export_cohort_definition(definition, source_snapshot=snapshot()).value
    result = export_circe_expression(
        envelope,
        criterion_domains={"condition": "ConditionOccurrence", "drug": "DrugExposure"},
        primary_criterion_id="condition",
        strict=False,
    )
    assert result.state is StoreState.PARTIAL
    assert result.value.source_digest == envelope.canonical_hash
    assert result.value.target_digest == canonical_digest(result.value.expression)
    first = result.value.expression["InclusionRules"][0]["expression"]["Groups"][0]
    assert [item["Occurrence"] for item in first["CriteriaList"]] == [
        {"Type": 2, "Count": 2},
        {"Type": 1, "Count": 4},
    ]
    assert {
        loss.path.rsplit("/", 1)[-1]
        for loss in result.value.losses
        if loss.reason_code == "assertion_filter_unsupported"
    } == {"negation", "temporality", "certainty", "experiencer"}
    assert "SQL" not in json.dumps(result.value.expression)
    imported = import_value(result.value.expression)
    assert imported.value.exchange is not None


@pytest.mark.parametrize(
    "value",
    [
        "{",
        "[]",
        b"\xff",
        '{"ConceptSets":[],"ConceptSets":[]}',
        '{"x":NaN}',
        {"x": float("inf")},
        {"x": object()},
        {1: "private"},
    ],
)
def test_malformed_json_is_typed_and_value_free(value):
    with pytest.raises(CirceInputError) as error:
        import_value(value)
    assert "private" not in str(error.value)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda x: x["PrimaryCriteria"]["CriteriaList"][0]["ConditionOccurrence"].update(
            CodesetId=42
        ),
        lambda x: x["ConceptSets"][0]["expression"]["items"][0]["concept"].update(
            CONCEPT_ID=True
        ),
        lambda x: x["ConceptSets"].append(deepcopy(x["ConceptSets"][0])),
        lambda x: x["PrimaryCriteria"].update(CriteriaList=[]),
    ],
)
def test_malformed_contract_references_and_integers(mutate):
    value = expression()
    mutate(value)
    with pytest.raises(CirceInputError):
        import_value(value)


def test_size_depth_mapping_cycles_and_threshold_expansion_bounded():
    with pytest.raises(CirceLimitError, match="circe_size_limit"):
        import_value('"' + "x" * MAX_CIRCE_BYTES + '"')
    with pytest.raises(CirceLimitError, match="circe_depth_limit"):
        import_value("[" * (MAX_CIRCE_DEPTH + 1) + "0" + "]" * (MAX_CIRCE_DEPTH + 1))
    cyclic = {}
    cyclic["cycle"] = cyclic
    with pytest.raises(CirceLimitError):
        import_value(cyclic)
    value = expression()
    value["AdditionalCriteria"] = group(
        *[correlated() for _ in range(20)], kind="AT_LEAST", count=10
    )
    with pytest.raises(CirceLimitError, match="circe_expansion_limit"):
        import_value(value)


def test_privacy_canaries_never_enter_reports_repr_or_exceptions():
    canary = "Иван-علي-555-01-2345-/private/patient-token"
    value = expression()
    value[canary] = canary
    value["Title"] = canary
    value["ConceptSets"][0]["name"] = canary
    value["ConceptSets"][0]["expression"]["items"][0]["concept"]["CONCEPT_NAME"] = (
        canary
    )
    result = import_value(value)
    assert canary not in json.dumps(result.value.to_report(), ensure_ascii=False)
    assert canary not in repr(result)
    assert canary not in repr(result.value)
    assert canary not in result.value.exchange.to_json()
    value["PrimaryCriteria"]["CriteriaList"][0]["ConditionOccurrence"]["CodesetId"] = (
        canary
    )
    with pytest.raises(CirceInputError) as error:
        import_value(value)
    assert canary not in str(error.value)


def test_combined_unsupported_occurrence_and_domain_both_reported():
    value = expression()
    value["AdditionalCriteria"] = group(correlated("DrugEra", kind=3))
    result = import_value(value)
    assert {"era_unsupported", "occurrence_type_unsupported"} <= codes(result)
    assert result.value.exchange is None


@pytest.mark.parametrize(
    "op,expected",
    [
        ("gte", {"start_date": "2026-01-01"}),
        ("lte", {"end_date": "2026-01-01"}),
        ("eq", {"start_date": "2026-01-01", "end_date": "2026-01-01"}),
    ],
)
def test_absolute_start_date_operators(op, expected):
    value = expression()
    value["PrimaryCriteria"]["CriteriaList"][0]["ConditionOccurrence"][
        "OccurrenceStartDate"
    ] = {"Op": op, "Value": "2026-01-01"}
    result = import_value(value)
    assert result.value.exchange.definition.criteria()[0].temporal.to_dict() == expected


def test_one_sided_window_and_multiple_primary_anchors_are_explicit_losses():
    value = expression()
    value["AdditionalCriteria"] = group(correlated(start=5, end=None))
    assert "start_window_unsupported" in codes(import_value(value))
    value["AdditionalCriteria"] = group(correlated(start=5, end=10))
    value["PrimaryCriteria"]["CriteriaList"].append({"DrugExposure": {"CodesetId": 0}})
    result = import_value(value)
    assert "start_window_unsupported" in codes(result)
    assert result.value.exchange.definition.criteria()[-1].temporal is None


def test_mixed_concept_descendants_unavailable_and_unknown_restrictions_counted():
    value = expression()
    items = value["ConceptSets"][0]["expression"]["items"]
    items.append(
        {
            "concept": {"CONCEPT_ID": 900002, "VOCABULARY_ID": "Synthetic"},
            "includeDescendants": False,
        }
    )
    value["PrimaryCriteria"]["ObservationWindow"] = {"PriorDays": 90, "PostDays": 30}
    value["QualifiedLimit"] = {"Type": "First"}
    result = import_value(value)
    assert result.value.exchange is None
    assert {
        "mixed_concept_set_unsupported",
        "event_limit_unsupported",
        "observation_period_window_unsupported",
    } <= codes(result)


def test_mapping_size_nodes_and_string_depth_scanner():
    from openmed.structured.cohort.circe import MAX_CIRCE_NODES

    with pytest.raises(CirceLimitError, match="circe_size_limit"):
        import_value({"ignored": "x" * MAX_CIRCE_BYTES})
    with pytest.raises(CirceLimitError, match="circe_node_limit"):
        import_value({"ignored": [None] * MAX_CIRCE_NODES})
    value = expression()
    value["Title"] = '[\\"' * (MAX_CIRCE_DEPTH + 10)
    assert import_value(json.dumps(value)).value.exchange is not None


def test_custody_rejects_tampered_conversion_and_envelope_roundtrip():
    from dataclasses import replace

    from openmed.structured.cohort import import_cohort_definition

    conversion = import_value(expression()).value
    with pytest.raises(CirceInputError, match="circe_custody_invalid"):
        replace(conversion, target_digest="sha256:" + "b" * 64)
    restored = import_cohort_definition(conversion.exchange.to_json_bytes())
    assert restored.state is StoreState.PARTIAL
    assert restored.value.canonical_hash == conversion.target_digest
    assert restored.value.losses == conversion.losses


def test_export_source_losses_are_offset_references_not_private_paths():
    from dataclasses import replace

    from openmed.structured.cohort import CohortConversionLoss

    imported = import_value(expression()).value.exchange
    canary = "/private/Иван/555-01-2345"
    source = replace(
        imported, losses=(CohortConversionLoss(canary, "adapter_unsupported"),)
    )
    criterion = source.definition.criteria()[0]
    conversion = export_circe_expression(
        source,
        criterion_domains={criterion.id: "ConditionOccurrence"},
        primary_criterion_id=criterion.id,
        strict=False,
    ).value
    assert canary not in repr(conversion)
    assert canary not in json.dumps(conversion.to_report(), ensure_ascii=False)
    assert any(
        loss.path == "/source_losses/0" and loss.reason_code == "source_conversion_loss"
        for loss in conversion.losses
    )
    assert conversion.source_digest == source.canonical_hash


@pytest.mark.parametrize(
    "negation", [("affirmed",), ("negated",), ("affirmed", "negated")]
)
def test_every_native_negation_filter_is_an_explicit_export_loss(negation):
    criterion = Criterion("primary", "cs", assertion=AssertionFilter(negation=negation))
    definition = PhenotypeDefinition(
        "synthetic",
        "Synthetic",
        (ConceptSet("cs", "Synthetic", (900001,)),),
        Expression.leaf(criterion),
    )
    envelope = export_cohort_definition(definition, source_snapshot=snapshot()).value
    result = export_circe_expression(
        envelope,
        criterion_domains={"primary": "ConditionOccurrence"},
        primary_criterion_id="primary",
        strict=False,
    )
    assert any(
        loss.path.endswith("/assertion/negation")
        and loss.reason_code == "assertion_filter_unsupported"
        for loss in result.value.losses
    )
