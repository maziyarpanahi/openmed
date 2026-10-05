"""Synthetic local review vectors and fail-closed negative controls."""

import copy
import json
from dataclasses import replace

import pytest

from openmed.eval.governance.blinded_adjudication import (
    ComparisonCase,
    ConflictOfInterestMetadata,
    RubricCriterion,
    SourceEvidence,
    render_blinded_adjudication_packets,
)
from openmed.eval.summary_review import (
    ReviewBinding,
    SummaryReviewError,
    artifact_digest,
    export_summary_review,
    import_summary_review,
)


def vector(*, evidence_kind="synthetic", count=2):
    """Create an offline synthetic packet/export/return vector."""
    source = "Dehydration improved."
    summary = source
    key = b"synthetic-evaluator-seal-key-00000"
    claims = [
        dict(
            claim_id=f"private-claim-{i}",
            claim_class="finding",
            start=0,
            end=len(summary),
            source_length=len(summary),
            citations=[f"evidence-{i}"],
        )
        for i in range(count)
    ]
    evidence = [
        dict(
            evidence_id=f"evidence-{i}",
            start=0,
            end=len(source),
            source_length=len(source),
        )
        for i in range(count)
    ]
    packets, mapping = render_blinded_adjudication_packets(
        packet_set_ref="packet-set-1",
        cases=[
            ComparisonCase(
                f"private-case-{i}",
                [SourceEvidence(f"evidence-{i}", source)],
                {
                    "hidden-model-one": summary,
                    "hidden-model-two": "Other synthetic output.",
                },
            )
            for i in range(count)
        ],
        rubric=[
            RubricCriterion("support", "Does the evidence support the claim?", 0, 1)
        ],
        conflict_of_interest=ConflictOfInterestMetadata(
            "reviewer-screening", artifact_digest("synthetic declaration")
        ),
        holdout_commitment_digest=artifact_digest("synthetic holdout"),
        submission_manifest_digests={
            name: artifact_digest(name)
            for name in ("hidden-model-one", "hidden-model-two")
        },
        randomization_key=key,
    )
    inputs = dict(
        packets=packets,
        mapping=mapping,
        sealing_key=key,
        bindings=[
            ReviewBinding(
                f"private-case-{i}",
                "hidden-model-one",
                f"private-claim-{i}",
                f"evidence-{i}",
            )
            for i in range(count)
        ],
        claims=claims,
        evidence=evidence,
        summary=summary,
        source=source,
        rubric_version=1,
        evidence_kind=evidence_kind,
    )
    request, sealed = export_summary_review(**inputs)
    bundle = {
        field: request[field]
        for field in (
            "schema_version",
            "request_digest",
            "rubric_version",
            "rubric_digest",
            "evidence_kind",
        )
    }
    bundle.update(
        review_evidence_digest=artifact_digest("synthetic review receipt"),
        decisions=[
            dict(
                review_ref=row["review_ref"],
                reviewer_ref=artifact_digest(reviewer),
                label="supports",
                reason=None,
            )
            for row in request["reviews"]
            for reviewer in ("reviewer-one", "reviewer-two")
        ],
        resolutions=[],
    )
    arguments = dict(
        sealed=sealed,
        sealing_key=key,
        claims=claims,
        evidence=evidence,
        summary=summary,
        source=source,
        minimum_cell_size=2,
    )
    return inputs, request, bundle, arguments


def test_synthetic_round_trip_is_stable_and_value_free():
    inputs, request, bundle, args = vector()
    imported = import_summary_review(json.loads(json.dumps(bundle)), **args)
    assert imported.complete
    assert not imported.reviewer_evidence_available
    assert imported.citation_support.support_recall == 1
    assert imported.reviewer_agreement.agreement.rate == 1
    assert export_summary_review(**inputs)[0] == request
    assert import_summary_review(bundle, **args).to_dict() == imported.to_dict()
    public = json.dumps(imported.to_dict()) + repr(imported) + json.dumps(request)
    for secret in (
        inputs["source"],
        "hidden-model-one",
        "private-claim",
        "private-case",
        "reviewer-one",
        "evidence-0",
    ):
        assert secret not in public
    assert repr(args["sealed"]) == "SealedSummaryReview(<redacted>)"


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", "unknown"),
        ("rubric_version", 2),
        ("rubric_version", True),
        ("rubric_digest", artifact_digest("wrong rubric")),
        ("request_digest", artifact_digest("stale request")),
        ("evidence_kind", "reviewer"),
        ("review_evidence_digest", "missing"),
    ],
)
def test_wrong_metadata_is_rejected(field, value):
    _, _, bundle, args = vector()
    bundle[field] = value
    with pytest.raises(SummaryReviewError, match="^invalid_review_import$"):
        import_summary_review(bundle, **args)


@pytest.mark.parametrize("field", ["claims", "evidence", "summary", "source"])
def test_stale_evaluated_artifacts_are_rejected(field):
    _, _, bundle, args = vector()
    args = copy.deepcopy(args)
    if field in ("summary", "source"):
        args[field] += " Changed."
    else:
        args[field][0]["end"] -= 1
    with pytest.raises(SummaryReviewError):
        import_summary_review(bundle, **args)


def test_duplicates_unknown_references_and_unblinded_fields_fail_closed():
    _, _, original, args = vector()
    for change in ("duplicate", "unknown", "identity", "free_text", "reviewer"):
        bundle = copy.deepcopy(original)
        if change == "duplicate":
            bundle["decisions"].append(bundle["decisions"][0])
        elif change == "unknown":
            bundle["decisions"][0]["review_ref"] = artifact_digest("unknown")
        elif change in ("identity", "free_text"):
            bundle["decisions"][0][change] = "Synthetic Patient 张三 /private/example"
        else:
            bundle["decisions"][0]["reviewer_ref"] = "Synthetic Patient"
        with pytest.raises(SummaryReviewError) as caught:
            import_summary_review(bundle, **args)
        assert "Synthetic Patient" not in str(caught.value)
        assert caught.value.__context__ is None


@pytest.mark.parametrize(
    "kind,state",
    [
        ("missing", "missing"),
        ("single", "incomplete"),
        ("disputed", "disputed"),
        ("unclear", "unclear"),
    ],
)
def test_unevaluable_states_do_not_become_support(kind, state):
    _, request, bundle, args = vector()
    ref = request["reviews"][0]["review_ref"]
    if kind in ("missing", "single"):
        bundle["decisions"] = [
            row
            for row in bundle["decisions"]
            if row["review_ref"] != ref
            or (
                kind == "single"
                and row["reviewer_ref"] == artifact_digest("reviewer-one")
            )
        ]
    else:
        for row in bundle["decisions"]:
            if row["review_ref"] == ref:
                row["label"] = (
                    "unclear"
                    if kind == "unclear"
                    else (
                        "supports"
                        if row["reviewer_ref"] == artifact_digest("reviewer-one")
                        else "contradicts"
                    )
                )
                row["reason"] = (
                    "clinical_interpretation" if kind == "disputed" else None
                )
    result = import_summary_review(bundle, **args)
    assert not result.complete and not result.reviewer_evidence_available
    assert result.to_dict()["states"][state] == 1
    assert result.citation_support.support_recall == 0.5


def test_disagreement_requires_separate_complete_adjudication():
    _, request, bundle, args = vector(evidence_kind="reviewer")
    ref = request["reviews"][0]["review_ref"]
    for row in bundle["decisions"]:
        if row["review_ref"] == ref:
            row["reason"] = "evidence_quality"
            if row["reviewer_ref"] == artifact_digest("reviewer-two"):
                row["label"] = "contradicts"
    bundle["resolutions"] = [
        dict(
            review_ref=ref,
            label="supports",
            reason="evidence_quality",
            adjudicator_ref=artifact_digest("separate-adjudicator"),
        )
    ]
    result = import_summary_review(bundle, **args)
    assert result.complete and result.reviewer_evidence_available
    assert result.reviewer_agreement.has_suppressed_cells
    bundle["resolutions"][0]["adjudicator_ref"] = artifact_digest("reviewer-one")
    with pytest.raises(SummaryReviewError):
        import_summary_review(bundle, **args)


@pytest.mark.parametrize(
    "change",
    [
        "key",
        "seal",
        "private",
        "mapping",
        "packet",
        "rubric",
        "output",
        "evidence",
        "binding",
        "recused",
    ],
)
def test_broken_seals_and_packet_bindings_fail_closed(change):
    inputs, _, bundle, args = vector()
    if change in ("key", "seal", "private"):
        if change == "key":
            args["sealing_key"] = b"different-synthetic-key-0000000000"
        elif change == "seal":
            args["sealed"] = replace(
                args["sealed"], commitment=artifact_digest("wrong seal")
            )
        else:
            private = json.loads(args["sealed"].private_json)
            private["bindings"][0]["claim_id"] = "changed"
            args["sealed"] = replace(args["sealed"], private_json=json.dumps(private))
        with pytest.raises(SummaryReviewError):
            import_summary_review(bundle, **args)
        return
    if change == "mapping":
        inputs["mapping"] = replace(
            inputs["mapping"], mapping_commitment=artifact_digest("wrong")
        )
    elif change == "packet":
        inputs["packets"] = inputs["packets"][:-1]
    elif change == "rubric":
        inputs["rubric_version"] = True
    elif change == "output":
        inputs["summary"] += " changed"
    elif change == "evidence":
        inputs["source"] = "X" * len(inputs["source"])
    elif change == "binding":
        inputs["bindings"] = inputs["bindings"][:-1]
    else:
        inputs["packets"] = tuple(
            replace(
                packet,
                conflict_of_interest=replace(
                    packet.conflict_of_interest, status="recused"
                ),
            )
            for packet in inputs["packets"]
        )
    with pytest.raises(SummaryReviewError):
        export_summary_review(**inputs)


def test_embedded_labels_cannot_bypass_review_import():
    inputs, _, bundle, args = vector()
    args["evidence"][0].update(claim_id="private-claim-0", label="supports")
    with pytest.raises(SummaryReviewError):
        import_summary_review(bundle, **args)
    with pytest.raises(SummaryReviewError):
        export_summary_review(**inputs)


def test_sealed_document_can_be_reloaded_without_public_mapping_disclosure():
    from openmed.eval.summary_review import SealedSummaryReview

    _, _, bundle, args = vector()
    stored = json.loads(
        json.dumps(
            {
                "private_json": args["sealed"].private_json,
                "commitment": args["sealed"].commitment,
            }
        )
    )
    args["sealed"] = SealedSummaryReview(**stored)
    result = import_summary_review(bundle, **args)
    bundle["decisions"].reverse()
    assert import_summary_review(bundle, **args).to_dict() == result.to_dict()
    bundle["decisions"][0]["label"] = "contradicts"
    for row in bundle["decisions"]:
        if row["review_ref"] == bundle["decisions"][0]["review_ref"]:
            row["reason"] = "evidence_quality"
    assert (
        import_summary_review(bundle, **args).decisions_digest
        != result.decisions_digest
    )


@pytest.mark.parametrize(
    "change",
    ["duplicate", "agreed", "missing", "single", "reason", "unknown", "malformed"],
)
def test_invalid_resolutions_and_disagreement_reasons_are_rejected(change):
    _, request, bundle, args = vector()
    ref = request["reviews"][0]["review_ref"]
    resolution = dict(
        review_ref=ref,
        label="supports",
        reason="evidence_quality",
        adjudicator_ref=artifact_digest("separate-reviewer"),
    )
    for index, row in enumerate(bundle["decisions"]):
        if row["review_ref"] == ref:
            row["label"] = "supports" if index % 2 == 0 else "contradicts"
            row["reason"] = "evidence_quality"
    bundle["resolutions"] = [resolution]
    if change == "duplicate":
        bundle["resolutions"].append(resolution)
    elif change == "agreed":
        for row in bundle["decisions"]:
            row["label"] = "supports"
            row["reason"] = None
    elif change in ("missing", "single"):
        bundle["decisions"] = [] if change == "missing" else bundle["decisions"][:1]
    elif change == "reason":
        bundle["decisions"][0]["reason"] = None
    elif change == "unknown":
        resolution["review_ref"] = artifact_digest("unknown")
    else:
        resolution.pop("adjudicator_ref")
    with pytest.raises(SummaryReviewError):
        import_summary_review(bundle, **args)


@pytest.mark.parametrize("label", ["contradicts", "irrelevant"])
def test_non_supporting_consensus_is_retained(label):
    _, _, bundle, args = vector(evidence_kind="reviewer")
    for row in bundle["decisions"]:
        row["label"] = label
    result = import_summary_review(bundle, **args)
    assert result.complete
    assert result.citation_support.support_recall == 0
    assert result.citation_support.adjudication.label_counts[label] == 2


def test_same_length_artifact_changes_and_out_of_artifact_spans_are_rejected():
    inputs, _, bundle, args = vector()
    for field in ("source", "summary"):
        changed = {**args, field: "X" * len(args[field])}
        with pytest.raises(SummaryReviewError):
            import_summary_review(bundle, **changed)
    inputs["claims"][0]["source_length"] += 10
    inputs["claims"][0]["end"] += 5
    with pytest.raises(SummaryReviewError):
        export_summary_review(**inputs)


@pytest.mark.parametrize(
    "marker",
    [
        "Synthetic Patient",
        "张三合成病人",
        "مريض تجريبي",
        "/private/synthetic/patient",
        "synthetic@example.invalid",
    ],
)
def test_multilingual_private_values_do_not_reach_diagnostics(marker):
    import traceback

    inputs, _, bundle, args = vector()
    inputs["bindings"][0] = replace(inputs["bindings"][0], candidate_identity=marker)
    with pytest.raises(SummaryReviewError) as caught:
        export_summary_review(**inputs)
    assert marker not in "".join(traceback.format_exception(caught.value))
    bundle["decisions"][0]["label"] = marker
    with pytest.raises(SummaryReviewError) as caught:
        import_summary_review(bundle, **args)
    assert marker not in "".join(traceback.format_exception(caught.value))


def test_no_network_and_bounded_collections(monkeypatch):
    import socket

    def forbidden(*args, **kwargs):
        raise AssertionError("network forbidden in offline review")

    monkeypatch.setattr(socket, "create_connection", forbidden)
    inputs, _, bundle, args = vector()
    assert import_summary_review(bundle, **args).complete
    monkeypatch.setattr("openmed.eval.summary_review._MAX_ROWS", 1)
    with pytest.raises(SummaryReviewError):
        export_summary_review(**inputs)
    with pytest.raises(SummaryReviewError):
        import_summary_review(bundle, **args)


def test_uncited_claims_cannot_be_hidden_by_complete_reviewed_edges():
    inputs, _, _, _ = vector()
    inputs["claims"].append(
        dict(
            claim_id="uncited-claim", start=0, end=len(inputs["summary"]), citations=[]
        )
    )
    with pytest.raises(SummaryReviewError):
        export_summary_review(**inputs)


def test_authenticated_unknown_request_schema_is_still_rejected():
    import hashlib
    import hmac

    _, _, bundle, args = vector()
    private = json.loads(args["sealed"].private_json)
    private["request"]["schema_version"] = "unsupported-review-version"
    payload = json.dumps(
        private, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    seal = (
        "sha256:"
        + hmac.new(
            args["sealing_key"],
            b"openmed.eval.summary_review.v1\0" + payload.encode(),
            hashlib.sha256,
        ).hexdigest()
    )
    args["sealed"] = replace(args["sealed"], private_json=payload, commitment=seal)
    bundle["schema_version"] = "unsupported-review-version"
    with pytest.raises(SummaryReviewError):
        import_summary_review(bundle, **args)
