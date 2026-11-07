"""Synthetic review workflow, exact binding and non-compensable safety controls."""

import hashlib
import json
from dataclasses import replace

import pytest

from openmed.eval.reviewer_disagreement import DisagreementReason
from openmed.eval.suites.multimodal_clinical import (
    Adjudication,
    CandidateLineage,
    CandidateObservation,
    ClinicalValidationError,
    CompletedReview,
    PacketRating,
    ReviewAsset,
    ReviewerQualification,
    ReviewOrigin,
    ValidationCase,
    ValidationPack,
    ValidationSlice,
    ValidationTask,
    assess_validation,
    packet_digest,
    prepare_review,
)
from tests.fixtures.eval.multimodal_clinical import development_pack

KEY = b"synthetic-evaluator-key-never-production"
PRIVATE = "SYNTHETIC-PRIVATE-SOURCE-3207"


def sha(value):
    return hashlib.sha256(str(value).encode()).hexdigest()


def pack(count=15):
    candidates = tuple(
        CandidateLineage(
            model_digest=sha(index),
            preprocessing_digest=sha("p"),
            runtime_digest=sha("r"),
            hardware_digest=sha("h"),
            harness_digest=sha("code"),
            license_digest=sha("terms"),
        )
        for index in range(2)
    )
    observations = tuple(
        CandidateObservation(
            content=PRIVATE + str(index),
            abstained=False,
            critical_privacy_failure=False,
            provenance_failure=False,
        )
        for index in range(2)
    )
    return ValidationPack(
        task=ValidationTask.FORM,
        candidates=candidates,
        cases=tuple(
            ValidationCase(
                assets=(
                    ReviewAsset(
                        media_type="application/json",
                        payload=json.dumps({"synthetic": i}).encode(),
                    ),
                ),
                context=PRIVATE,
                candidates=observations,
                slice=tuple(ValidationSlice)[i % 3],
            )
            for i in range(count)
        ),
        sampling_manifest_digest=sha("sampling"),
        dataset_license_digest=sha("terms"),
    )


def reviewer(index, **changes):
    return replace(
        ReviewerQualification(
            reviewer_digest=sha(f"reviewer {index}"),
            tasks=tuple(ValidationTask),
            qualification_digest=sha(f"simulated qualification {index}"),
            conflict_screening_digest=sha(f"conflict {index}"),
            origin=ReviewOrigin.SIMULATED,
        ),
        **changes,
    )


def completed(source, index, changes=None):
    prepared = prepare_review(source, reviewer(index), key=KEY)
    ratings = []
    for packet, entry in zip(prepared.packets, prepared.mapping.entries):
        for assignment in entry.assignments:
            rating = PacketRating(
                packet_digest=packet_digest(packet),
                alias=assignment.alias,
                quality=True,
                utility=True,
                privacy=True,
                provenance=True,
            )
            if changes and assignment.candidate_identity == source.candidates[0].digest:
                rating = replace(rating, **changes)
            ratings.append(rating)
    return CompletedReview(prepared=prepared, ratings=tuple(ratings))


def assess(source, reviews=None, **kwargs):
    return assess_validation(
        source,
        reviews or (completed(source, 1), completed(source, 2)),
        candidate_index=0,
        key=KEY,
        **kwargs,
    )


def target_rating(review, source, case_index=0):
    case = source.cases[case_index]
    packet = next(p for p in review.prepared.packets if p.case_ref == case.digest)
    entry = next(
        e for e in review.prepared.mapping.entries if e.case_ref == case.digest
    )
    alias = next(
        a.alias
        for a in entry.assignments
        if a.candidate_identity == source.candidates[0].digest
    )
    return next(
        r
        for r in review.ratings
        if r.packet_digest == packet_digest(packet) and r.alias == alias
    )


def resolution(source, reviews, index=3, **changes):
    prepared = completed(source, index)
    return Adjudication(
        prepared=prepared.prepared,
        rating=replace(target_rating(prepared, source), **changes),
        conflicting_rating_digests=tuple(
            target_rating(review, source).digest for review in reviews
        ),
        reason=DisagreementReason.EVIDENCE_QUALITY,
    )


def test_complete_case_level_aggregate_has_intervals_and_no_source_or_identity():
    source = pack()
    result = assess(source)
    report = json.loads(result.to_json())
    assert result.decision == "rubric_satisfied"
    assert report["overall"]["case_count"] == 15
    assert report["overall"]["metrics"]["quality"]["trials"] == 15
    assert report["overall"]["metrics"]["quality"]["wilson_95"][0] > 0.79
    assert report["review_origin"] == "development_simulation"
    assert report["independent_clinical_validation"] is False
    public = result.to_json() + repr(result)
    for value in (
        PRIVATE,
        source.digest,
        source.candidates[0].digest,
        reviewer(1).reviewer_digest,
    ):
        assert value not in public
    assert "decision" not in report


def test_single_small_slice_suppresses_overall_to_prevent_subtraction():
    result = assess(pack(14))
    report = json.loads(result.to_json())
    assert report["slices"]["nominal"]["case_count"] == 5
    assert report["slices"]["ambiguity_or_conflict"]["suppressed"]
    assert report["overall"]["case_count"] is None


@pytest.mark.parametrize("flag", ["critical_privacy_failure", "provenance_failure"])
def test_one_automatic_critical_failure_blocks_and_suppresses_complement(flag):
    source = pack()
    case = source.cases[0]
    changed = replace(
        case,
        candidates=(replace(case.candidates[0], **{flag: True}), case.candidates[1]),
    )
    source = replace(source, cases=(changed,) + source.cases[1:])
    result = assess(source)
    report = json.loads(result.to_json())
    assert result.decision == "blocked"
    assert report["slices"]["nominal"]["metrics"] is None
    assert report["overall"]["metrics"] is None


def test_minority_privacy_failure_survives_positive_adjudication():
    source = pack(1)
    reviews = (completed(source, 1), completed(source, 2, {"privacy": False}))
    result = assess(source, reviews, adjudications=(resolution(source, reviews),))
    assert result.decision == "blocked"
    assert json.loads(result.to_json())["overall"]["suppressed"]


def test_quality_disagreement_holds_until_independent_artifact_bound_resolution():
    source = pack(1)
    reviews = (completed(source, 1), completed(source, 2, {"quality": False}))
    assert assess(source, reviews).decision == "hold"
    assert (
        assess(source, reviews, adjudications=(resolution(source, reviews),)).decision
        == "rubric_satisfied"
    )
    assert (
        assess(
            source, reviews, adjudications=(resolution(source, reviews, utility=False),)
        ).decision
        == "hold"
    )


@pytest.mark.parametrize(
    "change", ["self_review", "wrong_ratings", "wrong_key", "different_candidate"]
)
def test_adjudication_rejects_self_review_or_stale_conflict_evidence(change):
    source = pack(1)
    reviews = (completed(source, 1), completed(source, 2, {"quality": False}))
    item = resolution(source, reviews)
    if change == "self_review":
        item = resolution(source, reviews, index=1)
    elif change == "wrong_ratings":
        item = replace(item, conflicting_rating_digests=(sha("wrong1"), sha("wrong2")))
    elif change == "wrong_key":
        item = replace(
            item,
            prepared=prepare_review(
                source, reviewer(3), key=b"other-evaluator-key-for-synthetic-data"
            ),
        )
    else:
        item = replace(
            item,
            rating=replace(
                item.rating,
                alias="candidate-b"
                if item.rating.alias == "candidate-a"
                else "candidate-a",
            ),
        )
    with pytest.raises(ClinicalValidationError):
        assess(source, reviews, adjudications=(item,))


@pytest.mark.parametrize(
    "change",
    [
        "missing",
        "duplicate",
        "wrong_packet",
        "changed_rubric",
        "stale_source",
        "stale_terms",
        "stale_output",
        "stale_qualification",
    ],
)
def test_missing_duplicate_and_stale_review_evidence_is_rejected(change):
    source = pack(1)
    first = completed(source, 1)
    second = completed(source, 2)
    if change == "missing":
        # A larger pack permits a shape-valid but incomplete review.
        source = pack(2)
        first = completed(source, 1)
        second = completed(source, 2)
        first = replace(first, ratings=first.ratings[:-1])
    elif change == "duplicate":
        first = replace(first, ratings=(first.ratings[0], first.ratings[0]))
    elif change == "wrong_packet":
        first = replace(
            first,
            ratings=(
                replace(first.ratings[0], packet_digest=sha("stale")),
                first.ratings[1],
            ),
        )
    elif change == "changed_rubric":
        packet = first.prepared.packets[0]
        packet = replace(
            packet,
            rubric=(replace(packet.rubric[0], question="Changed question"),)
            + packet.rubric[1:],
        )
        first = replace(first, prepared=replace(first.prepared, packets=(packet,)))
    elif change == "stale_source":
        case = source.cases[0]
        source = replace(
            source,
            cases=(
                replace(
                    case,
                    assets=(
                        ReviewAsset(media_type="application/json", payload=b"changed"),
                    ),
                ),
            ),
        )
    elif change == "stale_terms":
        source = replace(source, dataset_license_digest=sha("new license snapshot"))
    elif change == "stale_output":
        case = source.cases[0]
        source = replace(
            source,
            cases=(
                replace(
                    case,
                    candidates=(
                        replace(case.candidates[0], content="changed"),
                        case.candidates[1],
                    ),
                ),
            ),
        )
    else:
        first = replace(
            first,
            prepared=replace(
                first.prepared,
                reviewer=replace(
                    first.prepared.reviewer, qualification_digest=sha("new credential")
                ),
            ),
        )
    with pytest.raises(ClinicalValidationError) as caught:
        assess(source, (first, second))
    assert PRIVATE not in str(caught.value)


def test_reviewers_must_be_distinct_qualified_non_recused_and_scoped():
    source = pack(1)
    review = completed(source, 1)
    with pytest.raises(ClinicalValidationError):
        assess(source, (review, review))
    for declaration in (
        reviewer(1, recused=True),
        reviewer(1, tasks=(ValidationTask.WSI,)),
    ):
        with pytest.raises(ClinicalValidationError):
            prepare_review(source, declaration, key=KEY)


def test_all_abstention_never_satisfies_rubric_even_with_positive_ratings():
    source = pack(1)
    case = source.cases[0]
    source = replace(
        source,
        cases=(
            replace(
                case,
                candidates=(
                    replace(case.candidates[0], abstained=True),
                    case.candidates[1],
                ),
            ),
        ),
    )
    assert assess(source).decision == "hold"


def test_license_snapshots_record_changes_without_license_allowlist():
    source = replace(
        pack(1), dataset_license_digest=sha("caller terms of any chosen model")
    )
    assert assess(source).decision == "rubric_satisfied"
    assert source.describe()["dataset_license_digest"] == source.dataset_license_digest


def test_duplicate_source_cannot_be_counted_as_independent_case():
    source = pack(1)
    with pytest.raises(ClinicalValidationError):
        replace(
            source,
            cases=(source.cases[0], replace(source.cases[0], context="relabelled")),
        )


@pytest.mark.parametrize("task", list(ValidationTask))
def test_real_synthetic_modality_controls_execute_and_bind_assets(task, monkeypatch):
    pytest.importorskip("PIL")
    pytest.importorskip("numpy")
    pytest.importorskip("pydicom")
    import socket

    def denied(*args, **kwargs):
        raise AssertionError("synthetic controls must stay offline")

    monkeypatch.setattr(socket, "create_connection", denied)
    source = development_pack(task)
    assert source.digest == development_pack(task).digest
    assert all(
        asset.digest == hashlib.sha256(asset.payload).hexdigest()
        for asset in source.cases[0].assets
    )
    assert (
        source.cases[0].candidates[0].content != source.cases[0].candidates[1].content
    )
    assert source.describe()["independent_validation"] is False
    assert json.loads(assess(source).to_json())["overall"]["suppressed"]
    if task is ValidationTask.PIXEL_PHI:
        assert source.cases[0].candidates[1].critical_privacy_failure
    elif task is ValidationTask.ECG_INTERVAL:
        report = json.loads(source.cases[0].candidates[0].content)
        assert len(report["measurements"]) == 4
        assert report["leads"]


def test_new_review_origin_cannot_hide_simulated_adjudication():
    source = pack(1)
    originals = (completed(source, 1), completed(source, 2, {"quality": False}))
    reviews = []
    for original in originals:
        declaration = replace(original.prepared.reviewer, origin=ReviewOrigin.ATTESTED)
        prepared = prepare_review(source, declaration, key=KEY)
        ratings = tuple(
            PacketRating(
                packet_digest=packet_digest(packet),
                alias=candidate.alias,
                quality=original.ratings[0].quality,
                utility=True,
                privacy=True,
                provenance=True,
            )
            for packet in prepared.packets
            for candidate in packet.candidates
        )
        reviews.append(CompletedReview(prepared=prepared, ratings=ratings))
    # Set the second target explicitly after alias randomization.
    target = target_rating(reviews[1], source)
    reviews[1] = replace(
        reviews[1],
        ratings=tuple(
            replace(rating, quality=False) if rating is target else rating
            for rating in reviews[1].ratings
        ),
    )
    paired = tuple(reviews)
    result = assess(source, paired, adjudications=(resolution(source, paired),))
    assert result.decision == "rubric_satisfied"
    assert json.loads(result.to_json())["review_origin"] == "development_simulation"


def test_safety_failure_by_adjudicator_is_non_compensable():
    source = pack(1)
    reviews = (completed(source, 1), completed(source, 2, {"utility": False}))
    assert (
        assess(
            source,
            reviews,
            adjudications=(resolution(source, reviews, provenance=False),),
        ).decision
        == "blocked"
    )


def test_unicode_refusal_and_invalid_publication_thresholds_never_reflect_input():
    source = pack(1)
    with pytest.raises(ClinicalValidationError) as caught:
        replace(source.cases[0], context=PRIVATE + "\ud800")
    assert PRIVATE not in str(caught.value)
    for minimum in (True, 1, 4, 257):
        with pytest.raises(ClinicalValidationError):
            assess(source, minimum_cell_size=minimum)
