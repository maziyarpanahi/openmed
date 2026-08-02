"""Tests for PHI-safe qualified-expert review evidence."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from openmed.compliance import (
    EXPERT_REVIEW_EVIDENCE_DISCLAIMER,
    EXPERT_REVIEW_EVIDENCE_TITLE,
    AggregateRiskMetrics,
    AttributeRoleReview,
    ClassSizeBin,
    CompositionEvidence,
    EvidenceDigests,
    ExpertReviewEvidenceReport,
    PrivacyModelEvidence,
    ReleaseAssumptions,
    SearchEvidence,
    SuppressionAggregate,
    TransformationAggregate,
    UtilityAggregate,
    build_expert_review_evidence,
)
from openmed.core.audit import stable_hash

CANARY_NAME = "Patient Canary Meridian"
CANARY_RECORD_ID = "medical-record-999-88-7777"
CANARY_PATH = "/private/cohort/canary-record.jsonl"


def _digest(value: str) -> str:
    return stable_hash({"synthetic": value})


def _metrics(
    *,
    privacy_units: int,
    classes: int,
    smallest: int,
    largest: int,
    mean: float,
    bins: tuple[ClassSizeBin, ...],
    k_violations: int,
    l_violations: int,
    t_violations: int,
    any_violations: int,
    violating_privacy_units: int,
) -> AggregateRiskMetrics:
    return AggregateRiskMetrics(
        privacy_unit_count=privacy_units,
        equivalence_class_count=classes,
        smallest_class_size=smallest,
        largest_class_size=largest,
        mean_class_size=mean,
        class_size_histogram=bins,
        k_violating_class_count=k_violations,
        l_violating_class_count=l_violations,
        t_violating_class_count=t_violations,
        any_violating_class_count=any_violations,
        violating_privacy_unit_count=violating_privacy_units,
    )


def _report(*, reverse: bool = False) -> ExpertReviewEvidenceReport:
    reviews = (
        AttributeRoleReview("age", ("quasi_identifier",)),
        AttributeRoleReview("diagnosis", ("sensitive_attribute",)),
        AttributeRoleReview(
            "patient_id",
            ("direct_identifier", "privacy_unit"),
        ),
        AttributeRoleReview(
            "region",
            ("sensitive_attribute", "quasi_identifier"),
            override_applied=True,
            override_reason="sensitive_attribute_dual_role",
        ),
        AttributeRoleReview("patient_name", ("direct_identifier",)),
    )
    transformations = (
        TransformationAggregate("age", "bucket", 90, 0, 2),
        TransformationAggregate("region", "generalize", 80, 0, 1),
    )
    utility = (
        UtilityAggregate("classification_f1", 0.91, 0.89, "score", True),
        UtilityAggregate("information_loss", 0.0, 0.14, "ratio", False),
    )
    if reverse:
        reviews = tuple(reversed(reviews))
        transformations = tuple(reversed(transformations))
        utility = tuple(reversed(utility))
    return build_expert_review_evidence(
        digests=EvidenceDigests(
            source_dataset=_digest("source dataset"),
            dataset=_digest("dataset"),
            schema=_digest("schema"),
            policy=_digest("policy"),
            hierarchy=_digest("hierarchy"),
            config=_digest("config"),
            software=_digest("software"),
        ),
        assumptions=ReleaseAssumptions(
            privacy_unit="patient",
            population_scope="release_cohort",
            release_model="restricted",
            recipient_model="named_researchers",
            auxiliary_data_model="reasonably_available",
            notes_digest=_digest("expert assumption notes"),
        ),
        attribute_reviews=reviews,
        selected_quasi_identifiers=("region", "age"),
        sensitive_attributes=("region", "diagnosis"),
        privacy_models=PrivacyModelEvidence(
            configured_k=5,
            pre_achieved_k=1,
            achieved_k=5,
            l_variant="distinct",
            configured_l=2,
            pre_achieved_l=1,
            achieved_l=2,
            t_variant="variational",
            configured_t=0.2,
            pre_achieved_t=0.6,
            achieved_t=0.18,
        ),
        pre_metrics=_metrics(
            privacy_units=100,
            classes=25,
            smallest=1,
            largest=12,
            mean=4.0,
            bins=(
                ClassSizeBin(1, 4, 20, 60),
                ClassSizeBin(5, 12, 5, 40),
            ),
            k_violations=20,
            l_violations=3,
            t_violations=5,
            any_violations=21,
            violating_privacy_units=64,
        ),
        post_metrics=_metrics(
            privacy_units=95,
            classes=12,
            smallest=5,
            largest=12,
            mean=95 / 12,
            bins=(
                ClassSizeBin(5, 8, 8, 55),
                ClassSizeBin(9, 12, 4, 40),
            ),
            k_violations=0,
            l_violations=0,
            t_violations=0,
            any_violations=0,
            violating_privacy_units=0,
        ),
        transformations=transformations,
        suppression=SuppressionAggregate(
            privacy_units_suppressed=5,
            rows_suppressed=7,
            cells_suppressed=10,
            suppression_rate=0.05,
            privacy_unit_limit=5,
        ),
        utility=utility,
        search=SearchEvidence(
            strategy="exhaustive_lattice",
            complete=True,
            optimality_proven=True,
            evaluated_candidates=25,
            total_candidates=25,
            maximum_quasi_identifiers=8,
            candidate_limit=None,
            suppression_subsets_evaluated=25,
            suppression_subsets_total=25,
            suppression_subset_limit=None,
            time_limit_seconds=None,
            termination_reason="optimal_candidate_found",
        ),
        composition=CompositionEvidence(
            release_count=2,
            longitudinal_linkage_assessed=True,
            prior_release_overlap_assessed=True,
            risk_status="no_material_increase_observed",
            evidence_digest=_digest("composition"),
        ),
        limitations=(
            "external_population_frequency_not_estimated",
            "expert_judgment_required",
        ),
        unsupported_modalities=("audio", "images"),
    )


@pytest.mark.parametrize("missing_model", ["l_diversity", "t_closeness", "both"])
def test_sensitive_attributes_require_complete_l_and_t_evidence(
    missing_model: str,
) -> None:
    complete = _report().privacy_models
    omit_l = missing_model in {"l_diversity", "both"}
    omit_t = missing_model in {"t_closeness", "both"}
    models = PrivacyModelEvidence(
        configured_k=complete.configured_k,
        pre_achieved_k=complete.pre_achieved_k,
        achieved_k=complete.achieved_k,
        l_variant=None if omit_l else "distinct",
        configured_l=None if omit_l else complete.configured_l,
        pre_achieved_l=None if omit_l else complete.pre_achieved_l,
        achieved_l=None if omit_l else complete.achieved_l,
        t_variant=None if omit_t else "variational",
        configured_t=None if omit_t else complete.configured_t,
        pre_achieved_t=None if omit_t else complete.pre_achieved_t,
        achieved_t=None if omit_t else complete.achieved_t,
    )

    with pytest.raises(
        ValueError,
        match="sensitive attributes require both l-diversity and t-closeness",
    ):
        replace(_report(), privacy_models=models)


@pytest.mark.parametrize(
    ("longitudinal", "overlap", "status"),
    [
        (True, False, "not_assessed"),
        (False, True, "not_assessed"),
        (False, False, "increase_observed"),
        (True, True, "inconclusive"),
    ],
)
def test_single_release_composition_must_remain_unassessed(
    longitudinal: bool,
    overlap: bool,
    status: str,
) -> None:
    with pytest.raises(ValueError, match="single-release"):
        CompositionEvidence(
            release_count=1,
            longitudinal_linkage_assessed=longitudinal,
            prior_release_overlap_assessed=overlap,
            risk_status=status,
            evidence_digest=_digest("single release"),
        )


def test_bundle_contains_required_aggregate_evidence_and_placeholders() -> None:
    report = _report()
    payload = report.to_dict()

    assert payload["title"] == EXPERT_REVIEW_EVIDENCE_TITLE
    assert payload["disclaimer"] == EXPERT_REVIEW_EVIDENCE_DISCLAIMER
    assert payload["schema_version"] == 3
    assert payload["search"]["optimality_proven"] is True
    assert "- Optimality proven: `true`" in report.to_markdown()
    assert set(payload["digests"]) == {
        "source_dataset",
        "dataset",
        "schema",
        "policy",
        "hierarchy",
        "config",
        "software",
    }
    assert payload["privacy_models"]["k_anonymity"] == {
        "configured_k": 5,
        "pre_achieved_k": 1,
        "achieved_k": 5,
    }
    assert payload["privacy_models"]["l_diversity"]["variant"] == "distinct"
    assert payload["privacy_models"]["t_closeness"]["variant"] == "variational"
    assert payload["metrics"]["pre_transform"]["violations"]["privacy_unit_count"] == 64
    assert payload["metrics"]["post_transform"]["class_sizes"]["smallest"] == 5
    assert payload["qualified_expert_review"] == {
        "status": "pending_qualified_expert_review",
        "qualified_expert_name": None,
        "qualifications": None,
        "methodology_review": None,
        "risk_conclusion": None,
        "review_date": None,
        "signature": None,
    }
    assert report.verify()


def test_serialization_is_deterministic_for_semantically_equal_inputs() -> None:
    first = _report()
    second = _report(reverse=True)

    assert first.integrity_hash == second.integrity_hash
    assert first.to_json(indent=None) == second.to_json(indent=None)
    assert first.to_markdown() == second.to_markdown()
    assert ExpertReviewEvidenceReport.from_json(first.to_json()).to_dict() == (
        first.to_dict()
    )


def test_schema_two_evidence_remains_verifiable_without_optimality_field() -> None:
    legacy = _report().to_dict()
    legacy["schema_version"] = 2
    legacy["search"].pop("optimality_proven")
    legacy["integrity_hash"] = stable_hash(
        {key: value for key, value in legacy.items() if key != "integrity_hash"}
    )

    restored = ExpertReviewEvidenceReport.from_dict(legacy)

    assert restored.schema_version == 2
    assert restored.search.optimality_proven is True
    assert restored.verify() is True
    assert "optimality_proven" not in restored.to_dict()["search"]


def test_schema_two_bounded_optimal_termination_remains_parseable() -> None:
    legacy = _report().to_dict()
    legacy["schema_version"] = 2
    legacy["search"].update(
        {
            "strategy": "bounded_lattice",
            "complete": False,
            "evaluated_candidates": 4,
            "suppression_subsets_evaluated": 4,
            "termination_reason": "optimal_candidate_found",
        }
    )
    legacy["search"].pop("optimality_proven")
    legacy["integrity_hash"] = stable_hash(
        {key: value for key, value in legacy.items() if key != "integrity_hash"}
    )

    restored = ExpertReviewEvidenceReport.from_dict(legacy)

    assert restored.schema_version == 2
    assert restored.search.complete is False
    assert restored.search.optimality_proven is True
    assert restored.verify() is True


def test_schema_three_rejects_inconsistent_pruned_optimality_claim() -> None:
    payload = _report().to_dict()
    payload["search"].update(
        {
            "strategy": "bounded_lattice",
            "complete": False,
            "optimality_proven": True,
            "evaluated_candidates": 1,
            "suppression_subsets_evaluated": 1,
            "suppression_subsets_total": None,
            "termination_reason": "optimal_candidate_found",
        }
    )
    payload["integrity_hash"] = stable_hash(
        {key: value for key, value in payload.items() if key != "integrity_hash"}
    )

    with pytest.raises(ValueError, match="exact zero-loss lower-bound proof"):
        ExpertReviewEvidenceReport.from_dict(payload)


def test_schema_version_requires_an_exact_integer() -> None:
    payload = _report().to_dict()
    payload["schema_version"] = 3.0

    with pytest.raises(ValueError, match="schema version"):
        ExpertReviewEvidenceReport.from_dict(payload, verify=False)


def test_search_evidence_preserves_schema_two_positional_constructor() -> None:
    search = SearchEvidence(
        "bounded_lattice",
        False,
        4,
        10,
        4,
        4,
        4,
        10,
        4,
        None,
        "candidate_limit_reached",
    )

    assert search.complete is False
    assert search.optimality_proven is False


@pytest.mark.parametrize(
    "duplicate_prefix",
    [
        '"schema_version":999,',
        '"title":"discarded-sensitive-canary",',
    ],
)
def test_json_parser_rejects_duplicate_keys_before_integrity_verification(
    duplicate_prefix: str,
) -> None:
    rendered = _report().to_json(indent=None)
    ambiguous = "{" + duplicate_prefix + rendered[1:]

    with pytest.raises(ValueError, match="invalid JSON") as raised:
        ExpertReviewEvidenceReport.from_json(ambiguous)

    assert "discarded-sensitive-canary" not in str(raised.value)


def test_json_parser_rejects_nested_duplicate_keys() -> None:
    report = _report()
    rendered = report.to_json(indent=None)
    source_digest = report.digests.source_dataset
    source_field = f'"source_dataset":"{source_digest}"'
    ambiguous = rendered.replace(
        source_field,
        f'"source_dataset":"discarded-sensitive-canary",{source_field}',
        1,
    )

    with pytest.raises(ValueError, match="invalid JSON") as raised:
        ExpertReviewEvidenceReport.from_json(ambiguous)

    assert "discarded-sensitive-canary" not in str(raised.value)


@pytest.mark.parametrize(
    "constant",
    ["NaN", "Infinity", "-Infinity", "1e999", "-1e999"],
)
def test_json_parser_rejects_non_finite_numbers(constant: str) -> None:
    rendered = _report().to_json(indent=None)
    malformed = rendered.replace(
        '"schema_version":3',
        f'"schema_version":{constant}',
        1,
    )

    with pytest.raises(ValueError, match="invalid JSON"):
        ExpertReviewEvidenceReport.from_json(malformed)


def test_bundle_does_not_accept_or_leak_row_level_canaries() -> None:
    report = _report()
    rendered = report.to_json() + report.to_markdown()

    assert CANARY_NAME not in rendered
    assert CANARY_RECORD_ID not in rendered
    assert CANARY_PATH not in rendered
    forbidden_schema_fields = {
        "records",
        "record_ids",
        "samples",
        "class_keys",
        "equivalence_classes",
        "source_path",
        "transformed_records",
        "record_count",
        "records_suppressed",
        "record_limit",
        "affected_record_count",
    }
    assert forbidden_schema_fields.isdisjoint(_all_keys(report.to_dict()))

    with pytest.raises(TypeError, match="unexpected keyword argument") as error:
        build_expert_review_evidence(  # type: ignore[call-arg]
            records=[{"name": CANARY_NAME}],
        )
    assert CANARY_NAME not in str(error.value)


def test_tampering_is_rejected_by_default_and_inspectable_fail_closed() -> None:
    payload = _report().to_dict()
    payload["privacy_models"]["k_anonymity"]["configured_k"] = 6

    with pytest.raises(ValueError, match="integrity verification failed"):
        ExpertReviewEvidenceReport.from_dict(payload)

    restored = ExpertReviewEvidenceReport.from_dict(payload, verify=False)
    assert restored.verify() is False
    assert restored.integrity_hash_matches() is False


def test_unknown_or_row_level_fields_are_rejected() -> None:
    payload = _report().to_dict()
    payload["sample_records"] = [{"record_id": CANARY_RECORD_ID}]

    with pytest.raises(ValueError, match="missing or unsupported fields") as error:
        ExpertReviewEvidenceReport.from_dict(payload)
    assert CANARY_RECORD_ID not in str(error.value)


def test_markdown_has_required_review_and_method_sections() -> None:
    markdown = _report().to_markdown()

    assert markdown.startswith(f"# {EXPERT_REVIEW_EVIDENCE_TITLE}")
    assert "not an Expert Determination" in markdown
    assert "A qualified expert must independently evaluate" in markdown
    assert "## Configured and achieved privacy models" in markdown
    assert "## Aggregate class and violation metrics" in markdown
    assert "| Stage | Privacy units |" in markdown
    assert "Violating privacy units" in markdown
    assert "Affected privacy units" in markdown
    assert "Privacy units suppressed" in markdown
    assert "## Search completeness and limits" in markdown
    assert "## Composition evidence" in markdown
    assert "## Limitations and unsupported modalities" in markdown
    assert "## Qualified expert review" in markdown
    assert "Qualified expert name: ____________________" in markdown


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (
            lambda: EvidenceDigests(
                source_dataset=_digest("source dataset"),
                dataset=CANARY_PATH,
                schema=_digest("schema"),
                policy=_digest("policy"),
                hierarchy=_digest("hierarchy"),
                config=_digest("config"),
                software=_digest("software"),
            ),
            "canonical sha256",
        ),
        (
            lambda: AttributeRoleReview("age", ("quasi_identifier", "excluded")),
            "excluded must be the only role",
        ),
        (
            lambda: PrivacyModelEvidence(
                configured_k=5,
                pre_achieved_k=1,
                achieved_k=5,
                configured_l=2,
            ),
            "explicit variant",
        ),
        (
            lambda: SearchEvidence(
                strategy="exhaustive_lattice",
                complete=False,
                optimality_proven=False,
                evaluated_candidates=4,
                total_candidates=10,
                maximum_quasi_identifiers=4,
                candidate_limit=4,
                suppression_subsets_evaluated=4,
                suppression_subsets_total=10,
                suppression_subset_limit=4,
                time_limit_seconds=None,
                termination_reason="candidate_limit_reached",
            ),
            "must be complete",
        ),
        (
            lambda: AttributeRoleReview("patient_id", ("privacy_unit",)),
            "paired only with direct_identifier",
        ),
    ],
)
def test_invalid_aggregate_inputs_are_rejected(
    factory: object,
    message: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        factory()  # type: ignore[operator]


def test_disclaimer_and_review_placeholders_cannot_be_removed() -> None:
    payload = _report().to_dict()
    payload["disclaimer"] = "approved"

    with pytest.raises(ValueError, match="qualified-expert disclaimer"):
        ExpertReviewEvidenceReport.from_dict(payload, verify=False)

    payload = _report().to_dict()
    payload["qualified_expert_review"]["risk_conclusion"] = "very small"
    with pytest.raises(ValueError, match="must remain placeholders"):
        ExpertReviewEvidenceReport.from_dict(payload, verify=False)


def test_cross_field_counts_and_achieved_k_must_agree() -> None:
    payload = _report().to_dict()
    payload["suppression"]["suppression_rate"] = 0.04
    payload["integrity_hash"] = stable_hash(
        {key: value for key, value in payload.items() if key != "integrity_hash"}
    )
    with pytest.raises(ValueError, match="suppression rate"):
        ExpertReviewEvidenceReport.from_dict(payload, verify=False)

    payload = _report().to_dict()
    payload["privacy_models"]["k_anonymity"]["achieved_k"] = 6
    payload["integrity_hash"] = stable_hash(
        {key: value for key, value in payload.items() if key != "integrity_hash"}
    )
    with pytest.raises(ValueError, match="achieved k"):
        ExpertReviewEvidenceReport.from_dict(payload, verify=False)


def test_json_is_valid_and_never_serializes_non_finite_numbers() -> None:
    payload = json.loads(_report().to_json(indent=None))
    assert payload["integrity_hash"].startswith("sha256:")

    with pytest.raises(ValueError, match="must be finite"):
        UtilityAggregate("information_loss", 0.0, float("nan"), "ratio", False)


def test_row_is_an_explicit_supported_privacy_unit_assumption() -> None:
    assumptions = ReleaseAssumptions(
        privacy_unit="row",
        population_scope="release_cohort",
        release_model="restricted",
        recipient_model="named_researchers",
        auxiliary_data_model="reasonably_available",
        notes_digest=_digest("row-level assumptions"),
    )

    assert assumptions.to_dict()["privacy_unit"] == "row"


def test_privacy_unit_role_matches_keyed_or_row_level_assumptions() -> None:
    payload = _report().to_dict()
    patient_id = next(
        item
        for item in payload["attribute_review"]
        if item["attribute"] == "patient_id"
    )
    patient_id["roles"] = ["direct_identifier"]
    with pytest.raises(ValueError, match="exactly one reviewed privacy-unit"):
        ExpertReviewEvidenceReport.from_dict(payload, verify=False)

    payload = _report().to_dict()
    payload["assumptions"]["privacy_unit"] = "row"
    with pytest.raises(ValueError, match="row-level privacy assumptions"):
        ExpertReviewEvidenceReport.from_dict(payload, verify=False)


def _all_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        return set(value).union(
            *(set() if not value else (_all_keys(item) for item in value.values()))
        )
    if isinstance(value, list):
        return set().union(*(_all_keys(item) for item in value), set())
    return set()
