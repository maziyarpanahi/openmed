"""Synthetic unit tests for synthetic-artifact release controls."""

from __future__ import annotations

import json
import socket
from dataclasses import replace

import pytest

from openmed.compliance.artifact_lineage import (
    ArtifactLineageNode,
    build_artifact_lineage_manifest,
)
from openmed.eval.release_gates import GateCheck
from openmed.training.synthetic.governance import (
    DEFAULT_MINIMUM_CELL_SIZE,
    MANDATORY_PROHIBITED_USES,
    SYNTHETIC_LABEL_DISCLOSURES,
    SYNTHETIC_RELEASE_GATE,
    BenchmarkContaminationEvidence,
    CountBand,
    DownstreamComparison,
    GenerationMethod,
    GeneratorDisclosure,
    LicenseRecord,
    ReleaseControl,
    SubgroupMetric,
    SyntheticArtifactLabel,
    SyntheticGovernanceError,
    SyntheticReleaseBlockedError,
    SyntheticReleasePolicy,
    SyntheticUse,
    WithdrawalNotice,
    WithdrawalReason,
    assert_synthetic_release,
    build_downstream_regression_report,
    build_subgroup_quality_report,
    fingerprint_record,
    measure_benchmark_contamination,
    propagate_withdrawal,
    validate_synthetic_release,
)

CONTENT_DIGEST = "sha256:" + "a" * 64
GENERATOR_DIGEST = "sha256:" + "b" * 64
BENCHMARK_DIGEST = "sha256:" + "c" * 64
MODEL_DIGEST = "sha256:" + "d" * 64
CANARY = "openmed-canary-7f3c9e21"
SYNTHETIC_RECORDS = (
    "Synthetic note one: patient reports mild headache.",
    "Synthetic note two: follow-up scheduled in two weeks.",
    "Synthetic note three: no known drug allergies.",
)
BENCHMARK_RECORDS = (
    "Benchmark row alpha: discharge summary placeholder.",
    "Benchmark row beta: radiology placeholder.",
)


def ref(node: ArtifactLineageNode) -> dict[str, str]:
    return {"type": node.artifact_type, "hash": node.artifact_hash}


def lineage_nodes() -> dict[str, ArtifactLineageNode]:
    """Return a small synthetic lineage: seed + generator -> dataset -> model."""

    seed = ArtifactLineageNode.create(
        artifact_type="seed_vocabulary", transformation="curate"
    )
    weights = ArtifactLineageNode.create(
        artifact_type="generator_weights", transformation="publish"
    )
    generator = ArtifactLineageNode.create(
        artifact_type="synthetic_generator",
        parents=[ref(weights)],
        transformation="configure",
    )
    dataset = ArtifactLineageNode.create(
        artifact_type="synthetic_dataset",
        parents=[ref(seed), ref(generator)],
        transformation="generate",
    )
    model = ArtifactLineageNode.create(
        artifact_type="trained_model",
        parents=[ref(dataset)],
        transformation="train",
    )
    split = ArtifactLineageNode.create(
        artifact_type="dataset_split",
        parents=[ref(dataset)],
        transformation="split",
    )
    unrelated = ArtifactLineageNode.create(
        artifact_type="unrelated_corpus", transformation="curate"
    )
    return {
        "seed": seed,
        "weights": weights,
        "generator": generator,
        "dataset": dataset,
        "model": model,
        "split": split,
        "unrelated": unrelated,
    }


def make_label(nodes, **overrides) -> SyntheticArtifactLabel:
    fields = {
        "artifact_hash": nodes["dataset"].artifact_hash,
        "content_digest": CONTENT_DIGEST,
        "content_kind": "fully_synthetic",
        "generators": [
            GeneratorDisclosure(
                generator_id="openmed.locale_phi",
                generator_version="1.2.0",
                method=GenerationMethod.NEURAL_MODEL,
                generator_digest=GENERATOR_DIGEST,
                lineage_node_hash=nodes["generator"].artifact_hash,
                model_digests=[MODEL_DIGEST],
            )
        ],
        "intended_uses": [SyntheticUse.MODEL_TRAINING, SyntheticUse.SOFTWARE_TESTING],
    }
    fields.update(overrides)
    return SyntheticArtifactLabel(**fields)


def make_licenses(nodes, license_id="Apache-2.0", **overrides):
    names = ("seed", "weights", "generator", "dataset")
    records = {
        name: LicenseRecord(
            node_hash=nodes[name].artifact_hash,
            license_id=license_id,
            redistribution_allowed=True,
        )
        for name in names
    }
    records.update(overrides)
    return list(records.values())


def make_evidence(records=SYNTHETIC_RECORDS, **overrides):
    kwargs = {
        "benchmark_id": "openmed.privacy_corpus",
        "benchmark_digest": BENCHMARK_DIGEST,
        "artifact_content_digest": CONTENT_DIGEST,
        "benchmark_fingerprints": [fingerprint_record(r) for r in BENCHMARK_RECORDS],
        "canaries": [CANARY],
    }
    kwargs.update(overrides)
    return measure_benchmark_contamination(records, **kwargs)


def make_regression(candidate_overall=88, candidate_group=44):
    return [
        DownstreamComparison(
            task_id="pii_ner",
            metric_id="direct_identifier_recall",
            baseline_successes=90,
            baseline_trials=100,
            candidate_successes=candidate_overall,
            candidate_trials=100,
        ),
        DownstreamComparison(
            task_id="pii_ner",
            metric_id="direct_identifier_recall",
            baseline_successes=45,
            baseline_trials=50,
            candidate_successes=candidate_group,
            candidate_trials=50,
            axis="language",
            group="en",
        ),
        DownstreamComparison(
            task_id="pii_ner",
            metric_id="direct_identifier_recall",
            baseline_successes=45,
            baseline_trials=50,
            candidate_successes=44,
            candidate_trials=50,
            axis="language",
            group="fr",
        ),
    ]


def make_subgroups():
    return [
        SubgroupMetric("label_validity", "overall", "all", 190, 200),
        SubgroupMetric("label_validity", "language", "en", 96, 100),
        SubgroupMetric("label_validity", "language", "fr", 94, 100),
    ]


def release_inputs(**overrides):
    nodes = overrides.pop("nodes", None) or lineage_nodes()
    inputs = {
        "label": make_label(nodes),
        "lineage": build_artifact_lineage_manifest(nodes.values()),
        "licenses": make_licenses(nodes),
        "benchmark_evidence": [make_evidence()],
        "regression": make_regression(),
        "subgroup_metrics": make_subgroups(),
        "withdrawals": [],
    }
    inputs.update(overrides)
    return nodes, inputs


def decide(**overrides):
    _, inputs = release_inputs(**overrides)
    label = inputs.pop("label")
    lineage = inputs.pop("lineage")
    return validate_synthetic_release(label, lineage, **inputs)


# ---------------------------------------------------------------------------
# Complete releases and machine-readable labels.
# ---------------------------------------------------------------------------


def test_complete_release_passes_with_content_free_evidence() -> None:
    decision = decide()

    assert decision.passed is True
    assert decision.findings == ()
    assert decision.lineage_closure_count == 4
    payload = decision.to_json()
    for text in SYNTHETIC_RECORDS + BENCHMARK_RECORDS + (CANARY,):
        assert text not in payload
    assert json.loads(payload)["passed"] is True
    assert decision.to_capsule_check() == {
        "checked_count": len(ReleaseControl),
        "evidence_digest": decision.evidence_digest.removeprefix("sha256:"),
        "status": "pass",
    }


def test_decision_is_deterministic_across_input_order() -> None:
    nodes, inputs = release_inputs()
    first = validate_synthetic_release(
        inputs.pop("label"), inputs.pop("lineage"), **inputs
    )
    _, shuffled = release_inputs(nodes=nodes)
    shuffled["licenses"] = list(reversed(shuffled["licenses"]))
    shuffled["regression"] = list(reversed(shuffled["regression"]))
    shuffled["subgroup_metrics"] = list(reversed(shuffled["subgroup_metrics"]))
    second = validate_synthetic_release(
        shuffled.pop("label"), shuffled.pop("lineage"), **shuffled
    )

    assert first.to_json() == second.to_json()
    assert first.evidence_digest == second.evidence_digest


def test_decision_binds_to_release_gate_check() -> None:
    passing = decide().to_gate_check()
    failing = decide(label=None).to_gate_check()

    assert isinstance(passing, GateCheck)
    assert passing.gate == SYNTHETIC_RELEASE_GATE
    assert passing.passed is True
    assert failing.passed is False
    assert failing.reason == "synthetic_release_controls_failed"
    assert failing.details["findings"][0]["code"] == "label_missing"


def test_label_round_trips_and_always_carries_disclosures() -> None:
    label = make_label(lineage_nodes(), prohibited_uses=["public_redistribution"])
    payload = label.to_dict()

    assert payload["synthetic"] is True
    assert payload["disclosures"] == list(SYNTHETIC_LABEL_DISCLOSURES)
    assert set(payload["prohibited_uses"]) == {
        use.value for use in MANDATORY_PROHIBITED_USES
    } | {"public_redistribution"}
    restored = SyntheticArtifactLabel.from_json(label.to_json())
    assert restored == label
    assert restored.label_digest == label.label_digest


@pytest.mark.parametrize(
    "mutation",
    [
        lambda payload: payload.pop("synthetic"),
        lambda payload: payload.update(synthetic=False),
        lambda payload: payload.update(disclosures=["not_real_patient_data"]),
        lambda payload: payload.update(note_text="Synthetic note one"),
        lambda payload: payload.update(schema_version="openmed.label.v0"),
    ],
)
def test_label_loader_rejects_unmarked_or_altered_labels(mutation) -> None:
    payload = make_label(lineage_nodes()).to_dict()
    mutation(payload)

    with pytest.raises(SyntheticGovernanceError) as error:
        SyntheticArtifactLabel.from_mapping(payload)
    assert "Synthetic note one" not in str(error.value)


def test_label_loader_rejects_invalid_json_without_echo() -> None:
    with pytest.raises(SyntheticGovernanceError) as error:
        SyntheticArtifactLabel.from_json("{not json: Synthetic note one")
    assert "Synthetic note one" not in str(error.value)


# ---------------------------------------------------------------------------
# Unlabeled and incomplete labels.
# ---------------------------------------------------------------------------


def test_unlabeled_artifact_cannot_pass() -> None:
    decision = decide(label=None)

    assert decision.passed is False
    assert "label_missing" in decision.codes
    assert ReleaseControl.LABEL in decision.failed_controls
    assert decision.artifact_hash is None


def test_label_missing_intended_use_or_using_a_prohibited_use_fails() -> None:
    nodes = lineage_nodes()
    missing = decide(nodes=nodes, label=make_label(nodes, intended_uses=[]))
    prohibited = decide(
        nodes=nodes,
        label=make_label(
            nodes,
            intended_uses=["model_training", "clinical_decision_making"],
        ),
    )

    assert missing.codes == ("intended_use_missing",)
    assert prohibited.codes == ("intended_use_prohibited",)


def test_mandatory_prohibitions_cannot_be_removed() -> None:
    label = make_label(lineage_nodes(), prohibited_uses=[])

    assert MANDATORY_PROHIBITED_USES <= set(label.prohibited_uses)


def test_generator_disclosure_is_required() -> None:
    nodes = lineage_nodes()
    decision = decide(nodes=nodes, label=make_label(nodes, generators=[]))

    assert decision.codes == ("generator_disclosure_missing",)


def test_model_backed_generator_must_disclose_model_digests() -> None:
    nodes = lineage_nodes()
    generator = GeneratorDisclosure(
        generator_id="openmed.locale_phi",
        generator_version="1.2.0",
        method="statistical_model",
        generator_digest=GENERATOR_DIGEST,
        lineage_node_hash=nodes["generator"].artifact_hash,
    )
    template = replace(generator, method=GenerationMethod.TEMPLATE)

    assert decide(
        nodes=nodes, label=make_label(nodes, generators=[generator])
    ).codes == ("model_digest_missing",)
    assert decide(nodes=nodes, label=make_label(nodes, generators=[template])).passed


def test_generator_must_be_an_ancestor_in_lineage() -> None:
    nodes = lineage_nodes()
    generator = GeneratorDisclosure(
        generator_id="openmed.locale_phi",
        generator_version="1.2.0",
        method="template",
        generator_digest=GENERATOR_DIGEST,
        lineage_node_hash=nodes["unrelated"].artifact_hash,
    )
    decision = decide(nodes=nodes, label=make_label(nodes, generators=[generator]))

    assert decision.codes == ("generator_not_in_lineage",)


# ---------------------------------------------------------------------------
# Lineage and licenses.
# ---------------------------------------------------------------------------


def test_missing_lineage_fails_closed() -> None:
    decision = decide(lineage=None)

    assert decision.passed is False
    assert decision.codes == ("lineage_missing",)


def test_artifact_absent_from_lineage_fails() -> None:
    nodes = lineage_nodes()
    others = [node for name, node in nodes.items() if name != "dataset"]
    decision = decide(nodes=nodes, lineage=build_artifact_lineage_manifest(others))

    assert "lineage_artifact_absent" in decision.codes


def test_missing_ancestor_makes_lineage_incomplete() -> None:
    nodes = lineage_nodes()
    others = [node for name, node in nodes.items() if name != "weights"]
    decision = decide(nodes=nodes, lineage=build_artifact_lineage_manifest(others))

    assert decision.passed is False
    assert "lineage_ancestor_missing" in decision.codes
    assert "lineage_integrity_failed" in decision.codes


def test_tampered_lineage_node_fails_integrity() -> None:
    nodes = lineage_nodes()
    tampered = replace(nodes["seed"], transformation="tampered")
    lineage = build_artifact_lineage_manifest(
        [tampered if name == "seed" else node for name, node in nodes.items()]
    )
    decision = decide(nodes=nodes, lineage=lineage)

    assert decision.codes == ("lineage_integrity_failed",)


def test_every_lineage_node_needs_a_license() -> None:
    nodes = lineage_nodes()
    licenses = [
        record
        for record in make_licenses(nodes)
        if record.node_hash != nodes["weights"].artifact_hash
    ]
    decision = decide(nodes=nodes, licenses=licenses)

    assert decision.codes == ("license_missing",)
    assert decision.findings[0].count == 1


@pytest.mark.parametrize(
    "license_id",
    ["GPL-3.0-only", "LicenseRef-Proprietary", "MIT OR GPL-2.0", "CC-BY-SA-4.0"],
)
def test_restricted_or_unknown_licenses_fail_closed(license_id) -> None:
    nodes = lineage_nodes()
    restricted = LicenseRecord(
        node_hash=nodes["seed"].artifact_hash,
        license_id=license_id,
        redistribution_allowed=True,
    )
    decision = decide(nodes=nodes, licenses=make_licenses(nodes, seed=restricted))

    assert decision.codes == ("license_not_permitted",)
    assert license_id not in decision.to_json()


def test_permissive_data_licenses_are_accepted() -> None:
    nodes = lineage_nodes()

    assert decide(nodes=nodes, licenses=make_licenses(nodes, "CC-BY-4.0")).passed
    assert decide(nodes=nodes, licenses=make_licenses(nodes, "cc0-1.0")).passed
    assert decide(nodes=nodes, licenses=make_licenses(nodes, "mit")).passed


def test_redistribution_and_conflicting_license_records_fail() -> None:
    nodes = lineage_nodes()
    no_redistribution = LicenseRecord(
        node_hash=nodes["seed"].artifact_hash,
        license_id="MIT",
        redistribution_allowed=False,
    )
    blocked = decide(nodes=nodes, licenses=make_licenses(nodes, seed=no_redistribution))
    conflict = decide(nodes=nodes, licenses=make_licenses(nodes) + [no_redistribution])

    assert blocked.codes == ("license_redistribution_disallowed",)
    assert conflict.codes == ("license_conflict",)


# ---------------------------------------------------------------------------
# Withdrawal propagation.
# ---------------------------------------------------------------------------


def test_withdrawal_invalidates_every_descendant() -> None:
    nodes = lineage_nodes()
    lineage = build_artifact_lineage_manifest(nodes.values())
    notice = WithdrawalNotice(
        artifact_hash=nodes["weights"].artifact_hash,
        reason=WithdrawalReason.MEMORIZATION_DETECTED,
        effective_date="2026-10-01",
    )
    evidence = propagate_withdrawal(lineage, [notice])

    distances = {item.artifact_hash: item.distance for item in evidence.invalidated}
    assert distances == {
        nodes["weights"].artifact_hash: 0,
        nodes["generator"].artifact_hash: 1,
        nodes["dataset"].artifact_hash: 2,
        nodes["model"].artifact_hash: 3,
        nodes["split"].artifact_hash: 3,
    }
    assert evidence.descendant_count == 4
    assert not evidence.invalidates(nodes["seed"].artifact_hash)
    assert not evidence.invalidates(nodes["unrelated"].artifact_hash)
    assert all(
        item.reasons == (WithdrawalReason.MEMORIZATION_DETECTED,)
        and item.withdrawn_by == (nodes["weights"].artifact_hash,)
        for item in evidence.invalidated
    )


def test_withdrawal_evidence_is_content_free_and_order_independent() -> None:
    nodes = lineage_nodes()
    lineage = build_artifact_lineage_manifest(nodes.values())
    notices = [
        WithdrawalNotice(nodes["seed"].artifact_hash, "consent_withdrawn"),
        WithdrawalNotice(nodes["generator"].artifact_hash, "generator_defect"),
    ]
    first = propagate_withdrawal(lineage, notices)
    second = propagate_withdrawal(lineage.to_dict(), list(reversed(notices)))

    assert first.to_json() == second.to_json()
    assert first.evidence_digest == second.evidence_digest
    payload = first.to_dict()
    assert set(payload) == {
        "descendant_count",
        "invalidated",
        "invalidated_count",
        "lineage_manifest_hash",
        "notice_digests",
        "notices",
        "schema_version",
        "unresolved_notice_count",
    }
    dataset = next(
        item
        for item in first.invalidated
        if item.artifact_hash == nodes["dataset"].artifact_hash
    )
    assert dataset.reasons == (
        WithdrawalReason.CONSENT_WITHDRAWN,
        WithdrawalReason.GENERATOR_DEFECT,
    )
    assert len(dataset.withdrawn_by) == 2


def test_withdrawn_ancestor_blocks_release_of_descendants() -> None:
    nodes = lineage_nodes()
    notice = WithdrawalNotice(nodes["seed"].artifact_hash, "source_license_revoked")

    decision = decide(nodes=nodes, withdrawals=[notice])

    assert decision.codes == ("artifact_withdrawn",)
    assert decision.to_dict()["withdrawal"]["invalidated_count"] == 4


def test_withdrawal_of_unrelated_or_descendant_node_does_not_block() -> None:
    nodes = lineage_nodes()
    notices = [
        WithdrawalNotice(nodes["unrelated"].artifact_hash, "quality_defect"),
        WithdrawalNotice(nodes["model"].artifact_hash, "quality_defect"),
    ]

    assert decide(nodes=nodes, withdrawals=notices).passed


def test_unresolved_notice_still_withdraws_its_own_hash() -> None:
    nodes = lineage_nodes()
    notice = WithdrawalNotice(nodes["dataset"].artifact_hash, "legal_request")

    evidence = propagate_withdrawal(None, [notice])
    decision = decide(nodes=nodes, lineage=None, withdrawals=[notice])

    assert evidence.unresolved_notice_count == 1
    assert evidence.invalidates(nodes["dataset"].artifact_hash)
    assert "artifact_withdrawn" in decision.codes


def test_withdrawal_notice_rejects_free_text() -> None:
    nodes = lineage_nodes()

    with pytest.raises(SyntheticGovernanceError) as error:
        WithdrawalNotice(nodes["seed"].artifact_hash, "Patient asked to withdraw")
    assert "Patient" not in str(error.value)
    with pytest.raises(SyntheticGovernanceError):
        WithdrawalNotice(nodes["seed"].artifact_hash, "legal_request", "10/01/2026")


# ---------------------------------------------------------------------------
# Benchmark contamination.
# ---------------------------------------------------------------------------


def test_benchmark_canary_blocks_release() -> None:
    contaminated = SYNTHETIC_RECORDS + (f"Leaked row with {CANARY.upper()} inside.",)
    evidence = make_evidence(records=contaminated)
    decision = decide(benchmark_evidence=[evidence])

    assert evidence.canary_hit_count == 1
    assert decision.codes == ("benchmark_canary_detected",)
    assert CANARY not in decision.to_json()


def test_benchmark_overlap_blocks_release_unless_policy_tolerates_it() -> None:
    overlapping = SYNTHETIC_RECORDS + (
        "  BENCHMARK row alpha:  discharge summary placeholder.",
    )
    evidence = make_evidence(records=overlapping)

    assert evidence.overlapping_record_count == 1
    assert decide(benchmark_evidence=[evidence]).codes == (
        "benchmark_overlap_exceeded",
    )
    tolerant = SyntheticReleasePolicy(max_benchmark_overlap_rate=0.25)
    assert decide(benchmark_evidence=[evidence], policy=tolerant).passed


def test_benchmark_evidence_must_exist_and_match_the_artifact() -> None:
    other_content = make_evidence(artifact_content_digest="e" * 64)
    empty = make_evidence(records=[])

    assert decide(benchmark_evidence=[]).codes == ("benchmark_evidence_missing",)
    assert decide(benchmark_evidence=[other_content]).codes == (
        "benchmark_evidence_mismatch",
    )
    assert decide(benchmark_evidence=[empty]).codes == ("benchmark_evidence_empty",)


def test_required_benchmarks_must_be_checked() -> None:
    policy = SyntheticReleasePolicy(
        required_benchmarks=("openmed.privacy_corpus", "openmed.shield")
    )
    decision = decide(policy=policy)

    assert decision.codes == ("required_benchmark_missing",)


def test_contamination_inputs_never_echo_values() -> None:
    with pytest.raises(SyntheticGovernanceError) as short:
        make_evidence(canaries=["tiny"])
    with pytest.raises(SyntheticGovernanceError) as bad_record:
        make_evidence(records=["Synthetic note one", 7])
    with pytest.raises(SyntheticGovernanceError) as bad_counts:
        BenchmarkContaminationEvidence(
            benchmark_id="openmed.privacy_corpus",
            benchmark_digest=BENCHMARK_DIGEST,
            artifact_content_digest=CONTENT_DIGEST,
            checked_record_count=1,
            overlapping_record_count=2,
            canary_count=0,
            canary_hit_count=0,
        )

    assert "tiny" not in str(short.value)
    assert "record 1" in str(bad_record.value)
    assert "Synthetic note one" not in str(bad_record.value)
    assert "cannot exceed" in str(bad_counts.value)


# ---------------------------------------------------------------------------
# Downstream regression.
# ---------------------------------------------------------------------------


def test_critical_downstream_regression_blocks_release() -> None:
    overall = decide(regression=make_regression(candidate_overall=80))
    subgroup = decide(regression=make_regression(candidate_group=40))
    minor = decide(regression=make_regression(candidate_overall=87))

    assert overall.codes == ("critical_regression",)
    assert subgroup.codes == ("critical_regression",)
    assert minor.passed


def test_regression_evidence_is_required_and_sufficient() -> None:
    small_overall = [
        DownstreamComparison("pii_ner", "recall", 9, 10, 9, 10),
    ]
    subgroup_only = [
        DownstreamComparison("pii_ner", "recall", 45, 50, 45, 50, "language", "en"),
    ]

    assert decide(regression=None).codes == ("regression_evidence_missing",)
    assert decide(regression=[]).codes == ("regression_evidence_missing",)
    assert decide(regression=small_overall).codes == (
        "regression_evidence_insufficient",
    )
    assert decide(regression=subgroup_only).codes == (
        "regression_evidence_insufficient",
    )


def test_regression_interval_matches_newcombe_reference() -> None:
    # Newcombe (1998), example (a): 56/70 versus 48/80, hybrid score method.
    report = build_downstream_regression_report(
        [DownstreamComparison("task", "metric", 48, 80, 56, 70)],
        critical_drop=0.05,
    )
    (item,) = report.slices

    assert item.delta == pytest.approx(0.2, abs=1e-4)
    assert item.delta_lower == pytest.approx(0.0524, abs=1e-4)
    assert item.delta_upper == pytest.approx(0.3339, abs=1e-4)
    assert item.critical is False


def test_small_regression_slices_are_suppressed_not_assessed() -> None:
    comparisons = make_regression() + [
        DownstreamComparison(
            "pii_ner", "direct_identifier_recall", 5, 5, 0, 5, "language", "sw"
        )
    ]
    report = build_downstream_regression_report(comparisons)

    groups = {item.group for item in report.slices}
    # The small "sw" slice is suppressed, and complementary suppression hides
    # the smallest remaining language slice (ties break by group name).
    assert groups == {"all", "fr"}
    assert report.suppressed_slice_count == 2
    assert report.critical_slice_count == 0
    assert "sw" not in report.to_json()


# ---------------------------------------------------------------------------
# Subgroup quality and bias.
# ---------------------------------------------------------------------------


def test_wilson_interval_matches_reference_values() -> None:
    report = build_subgroup_quality_report(
        [SubgroupMetric("metric", "overall", "all", 81, 263)]
    )
    (item,) = report.slices

    assert item.estimate == pytest.approx(0.3080, abs=1e-4)
    assert item.lower == pytest.approx(0.2553, abs=1e-4)
    assert item.upper == pytest.approx(0.3662, abs=1e-4)
    assert report.interval_method == "wilson_score"


def test_small_groups_are_suppressed_with_a_complementary_cell() -> None:
    metrics = [
        SubgroupMetric("label_validity", "overall", "all", 300, 320),
        SubgroupMetric("label_validity", "language", "en", 150, 160),
        SubgroupMetric("label_validity", "language", "fr", 120, 130),
        SubgroupMetric("label_validity", "language", "yo", 3, 4),
        SubgroupMetric("label_validity", "language", "ha", 25, 26),
    ]
    report = build_subgroup_quality_report(metrics)

    released = {(item.axis, item.group) for item in report.slices}
    assert released == {("overall", "all"), ("language", "en"), ("language", "fr")}
    assert report.suppressed_slice_count == 2
    assert report.minimum_cell_size == DEFAULT_MINIMUM_CELL_SIZE
    payload = report.to_json()
    assert '"yo"' not in payload and '"ha"' not in payload
    assert "trials" not in payload and "successes" not in payload
    (disparity,) = report.disparities
    assert disparity.suppressed_group_count == 2
    assert disparity.released_group_count == 2


def test_count_bands_and_declared_cell_size() -> None:
    metrics = [
        SubgroupMetric("m", "site", "a", 10, 12),
        SubgroupMetric("m", "site", "b", 20, 30),
        SubgroupMetric("m", "site", "c", 40, 50),
    ]
    report = build_subgroup_quality_report(metrics, minimum_cell_size=12)

    bands = {item.group: item.count_band for item in report.slices}
    assert bands == {
        "a": CountBand.MINIMUM_TO_UNDER_DOUBLE,
        "b": CountBand.DOUBLE_TO_UNDER_FOURFOLD,
        "c": CountBand.FOURFOLD_OR_MORE,
    }
    assert report.to_dict()["minimum_cell_size"] == 12
    with pytest.raises(SyntheticGovernanceError):
        build_subgroup_quality_report(metrics, minimum_cell_size=1)


def test_supported_disparity_blocks_only_when_policy_sets_a_limit() -> None:
    biased = [
        SubgroupMetric("label_validity", "overall", "all", 350, 400),
        SubgroupMetric("label_validity", "language", "en", 195, 200),
        SubgroupMetric("label_validity", "language", "sw", 155, 200),
    ]
    report = build_subgroup_quality_report(biased)
    (disparity,) = report.disparities

    assert disparity.gap_supported is True
    assert disparity.highest_group == "en"
    assert disparity.lowest_group == "sw"
    assert decide(subgroup_metrics=biased).passed
    strict = SyntheticReleasePolicy(max_subgroup_gap=0.1)
    assert decide(subgroup_metrics=biased, policy=strict).codes == (
        "subgroup_disparity_exceeded",
    )


def test_unsupported_gap_is_not_reported_as_disparity() -> None:
    noisy = [
        SubgroupMetric("m", "language", "en", 12, 15),
        SubgroupMetric("m", "language", "fr", 9, 15),
    ]
    (disparity,) = build_subgroup_quality_report(noisy).disparities

    assert disparity.gap == pytest.approx(0.2)
    assert disparity.gap_supported is False
    strict = SyntheticReleasePolicy(max_subgroup_gap=0.1)
    assert decide(subgroup_metrics=noisy, policy=strict).passed


def test_subgroup_evidence_is_required_and_must_release_a_slice() -> None:
    tiny = [SubgroupMetric("m", "overall", "all", 3, 5)]
    optional = SyntheticReleasePolicy(require_subgroup_report=False)

    assert decide(subgroup_metrics=None).codes == ("subgroup_report_missing",)
    assert decide(subgroup_metrics=tiny).codes == ("subgroup_evidence_insufficient",)
    assert decide(subgroup_metrics=None, policy=optional).passed


def test_duplicate_or_invalid_slices_are_rejected() -> None:
    with pytest.raises(SyntheticGovernanceError):
        build_subgroup_quality_report(make_subgroups() + make_subgroups()[:1])
    with pytest.raises(SyntheticGovernanceError):
        SubgroupMetric("m", "overall", "en", 1, 2)
    with pytest.raises(SyntheticGovernanceError):
        SubgroupMetric("m", "language", "en", 3, 2)
    with pytest.raises(SyntheticGovernanceError) as error:
        SubgroupMetric("m", "name", "Jane Doe", 1, 20)
    assert "Jane" not in str(error.value)


# ---------------------------------------------------------------------------
# Enforcement and local-first behavior.
# ---------------------------------------------------------------------------


def test_assert_release_raises_with_codes_only() -> None:
    nodes, inputs = release_inputs()
    inputs["withdrawals"] = [
        WithdrawalNotice(nodes["dataset"].artifact_hash, "privacy_incident")
    ]
    label = inputs.pop("label")
    lineage = inputs.pop("lineage")

    with pytest.raises(SyntheticReleaseBlockedError) as error:
        assert_synthetic_release(label, lineage, **inputs)
    assert str(error.value) == "synthetic release blocked: artifact_withdrawn"
    assert error.value.decision.to_capsule_check()["status"] == "fail"

    _, clean = release_inputs(nodes=nodes)
    assert assert_synthetic_release(
        clean.pop("label"), clean.pop("lineage"), **clean
    ).passed


def test_multiple_failures_are_all_reported() -> None:
    nodes = lineage_nodes()
    decision = decide(
        nodes=nodes,
        label=None,
        benchmark_evidence=[],
        regression=None,
        subgroup_metrics=None,
    )

    assert decision.codes == (
        "label_missing",
        "benchmark_evidence_missing",
        "regression_evidence_missing",
        "subgroup_report_missing",
    )
    assert set(decision.failed_controls) == {
        ReleaseControl.LABEL,
        ReleaseControl.CONTAMINATION,
        ReleaseControl.REGRESSION,
        ReleaseControl.SUBGROUP,
    }


def test_validation_makes_no_network_calls(monkeypatch) -> None:
    def blocked(*args, **kwargs):
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket, "socket", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)

    assert decide().passed


@pytest.mark.parametrize(
    "factory",
    [
        lambda: LicenseRecord("not-a-digest Jane Doe", "MIT", True),
        lambda: LicenseRecord("a" * 64, "MIT", "yes"),
        lambda: SyntheticReleasePolicy(minimum_cell_size=True),
        lambda: SyntheticReleasePolicy(critical_regression_drop=0),
        lambda: SyntheticReleasePolicy(max_benchmark_overlap_rate=1.5),
        lambda: SyntheticReleasePolicy(confidence_level=1.0),
        lambda: DownstreamComparison("Task Name", "m", 1, 2, 1, 2),
    ],
)
def test_malformed_inputs_raise_without_echo(factory) -> None:
    with pytest.raises(SyntheticGovernanceError) as error:
        factory()
    assert "Jane" not in str(error.value)
    assert "Task Name" not in str(error.value)
