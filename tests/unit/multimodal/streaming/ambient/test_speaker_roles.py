"""Synthetic attribution hypotheses, stale-review rejection and actual use gates."""

import json
import socket
from dataclasses import replace

import pytest

from openmed.multimodal.streaming.ambient.speaker_roles import (
    RoleAttributionReview,
    RoleHypothesis,
    RoleObservation,
    RoleState,
    SpeakerRoleError,
    TranscriptSource,
    assess_speaker_roles,
    extract_with_reviewed_roles,
    finalize_role_bound_note,
)
from openmed.multimodal.streaming.speech.speaker_aliases import (
    ClinicalRole,
    LineageEvent,
    LineageKind,
    RoleReview,
    SpeakerAliases,
)
from openmed.multimodal.streaming.transcript.revision_ledger import (
    RevisionLedger,
    TranscriptToken,
)


def ref(index):
    return f"{index:032x}"


def context(count=1, *, finalized=True, text=None):
    aliases = SpeakerAliases(ref(100))
    aliases.apply_revision(0, (ref(200),))
    sources, observations = [], []
    for index in range(count):
        content = text or f"Synthetic statement {index}"
        ledger = RevisionLedger(
            ref(100),
            ref(300 + index),
            len(content),
            (
                TranscriptToken(
                    token_ref=ref(400 + index),
                    start=0,
                    end=len(content),
                    start_ms=100 * index,
                    end_ms=100 * (index + 1),
                ),
            ),
        )
        citation = ledger.capture_span(ref(500 + index), 0, len(content))
        if finalized:
            ledger.finalize(expected_revision=0)
        sources.append(TranscriptSource(ledger=ledger, text=content))
        observations.append(
            RoleObservation(
                observation_id=ref(600 + index),
                cluster_ref=ref(200),
                diarization_revision=0,
                citation=citation,
                hypotheses=(
                    RoleHypothesis(
                        role=ClinicalRole.PATIENT,
                        evidence=(citation,),
                        confidence_ppm=600_000,
                    ),
                    RoleHypothesis(
                        role=ClinicalRole.CLINICIAN,
                        evidence=(citation,),
                        confidence_ppm=990_000,
                    ),
                ),
            )
        )
    return aliases, tuple(sources), tuple(observations)


def approvals(aliases, sources, observations, *, role=ClinicalRole.PATIENT):
    report = assess_speaker_roles(aliases, sources, observations)
    return tuple(
        RoleAttributionReview(
            observation_id=item.observation_id,
            context_digest=report.context_digest,
            role=role,
            review_id=ref(700 + index),
            reviewer_id=ref(800),
            decision_digest="b" * 64,
            reviewer_confirmed=True,
        )
        for index, item in enumerate(observations)
    )


def test_multiple_role_hypotheses_are_never_selected_by_score_or_alias_order():
    aliases, sources, observations = context()
    report = assess_speaker_roles(aliases, sources, observations)
    assessment = report.assessments[0]
    assert assessment.state is RoleState.MULTIPLE_ROLES
    assert assessment.alias == "speaker_0001"
    assert assessment.blockers == ()
    assert [h.confidence_ppm for h in assessment.hypotheses] == [600_000, 990_000]
    called = []
    with pytest.raises(SpeakerRoleError):
        extract_with_reviewed_roles(
            aliases, sources, observations, (), lambda spans: called.append(spans)
        )
    assert called == []


@pytest.mark.parametrize(
    ("count", "expected"),
    [
        (0, RoleState.UNKNOWN),
        (1, RoleState.SINGLE_HYPOTHESIS),
        (2, RoleState.MULTIPLE_ROLES),
    ],
)
def test_all_hypothesis_states_require_explicit_current_review(count, expected):
    aliases, sources, observations = context()
    observations = (
        replace(observations[0], hypotheses=observations[0].hypotheses[:count]),
    )
    assert (
        assess_speaker_roles(aliases, sources, observations).assessments[0].state
        is expected
    )
    reviews = approvals(aliases, sources, observations)
    with pytest.raises(SpeakerRoleError, match="explicit role review"):
        extract_with_reviewed_roles(
            aliases,
            sources,
            observations,
            (replace(reviews[0], reviewer_confirmed=False),),
            lambda spans: "Synthetic draft",
        )
    draft = extract_with_reviewed_roles(
        aliases, sources, observations, reviews, lambda spans: "Synthetic draft"
    )
    assert draft.text == "Synthetic draft"


def test_reviewer_can_choose_lower_confidence_role_and_original_text_reaches_extractor():
    aliases, sources, observations = context()
    reviews = approvals(aliases, sources, observations)
    seen = []

    def extractor(spans):
        seen.extend(spans)
        return "Synthetic reviewed draft"

    draft = extract_with_reviewed_roles(
        aliases, sources, observations, reviews, extractor
    )
    assert len(seen) == 1 and seen[0].role is ClinicalRole.PATIENT
    assert seen[0].text == sources[0].text
    assert seen[0].citation.reference == observations[0].citation
    assert seen[0].citation.finalized and seen[0].citation.start_ms == 0
    note = finalize_role_bound_note(draft, aliases, sources, observations, reviews)
    assert json.loads(note.data_json)["note"] == "Synthetic reviewed draft"
    assert json.loads(note.data_json)["state"] == "role_attribution_checked"
    assert json.loads(note.to_json())["clinical_review_required"] is True


def test_explicit_correction_can_resolve_unknown_or_unlisted_role():
    aliases, sources, observations = context()
    observations = (replace(observations[0], hypotheses=()),)
    reviews = approvals(aliases, sources, observations, role=ClinicalRole.INTERPRETER)
    seen = []
    extract_with_reviewed_roles(
        aliases,
        sources,
        observations,
        reviews,
        lambda spans: seen.extend(spans) or "Synthetic draft",
    )
    assert seen[0].role is ClinicalRole.INTERPRETER
    assert (
        assess_speaker_roles(aliases, sources, observations).assessments[0].state
        is RoleState.UNKNOWN
    )


def test_same_alias_can_have_reviewed_roles_for_different_spans_without_global_guessing():
    aliases, sources, observations = context(2)
    reviews = approvals(aliases, sources, observations)
    reviews = (reviews[0], replace(reviews[1], role=ClinicalRole.CAREGIVER))
    seen = []
    draft = extract_with_reviewed_roles(
        aliases,
        sources,
        observations,
        reviews[::-1],
        lambda spans: seen.extend(spans) or "Synthetic draft",
    )
    assert seen[0].alias == seen[1].alias
    assert [span.role for span in seen] == [
        ClinicalRole.PATIENT,
        ClinicalRole.CAREGIVER,
    ]
    note = finalize_role_bound_note(draft, aliases, sources, observations, reviews)
    assert len(json.loads(note.to_json())["reviews"]) == 2


def test_old_alias_level_approval_does_not_review_current_span_hypotheses():
    aliases, sources, observations = context()
    aliases.confirm_role(
        RoleReview(
            session_ref=ref(100),
            alias="speaker_0001",
            role=ClinicalRole.PATIENT,
            revision=0,
            review_ref=ref(900),
        )
    )
    assert aliases.to_dict()["active"][0]["role"] == "patient"
    with pytest.raises(SpeakerRoleError):
        extract_with_reviewed_roles(
            aliases, sources, observations, (), lambda spans: "Synthetic draft"
        )


def test_any_missing_duplicate_or_unconfirmed_decision_blocks_entire_callback():
    aliases, sources, observations = context(2)
    reviews = approvals(aliases, sources, observations)
    malformed = (
        reviews[:1],
        reviews + (reviews[0],),
        (reviews[0], reviews[0]),
        (reviews[0], replace(reviews[1], review_id=reviews[0].review_id)),
        (reviews[0], replace(reviews[1], reviewer_confirmed=False)),
    )
    called = []
    for decisions in malformed:
        with pytest.raises(SpeakerRoleError):
            extract_with_reviewed_roles(
                aliases,
                sources,
                observations,
                decisions,
                lambda spans: called.append(spans),
            )
    assert called == []


def test_same_length_transcript_changes_invalidate_review_and_finalization():
    aliases, sources, observations = context()
    reviews = approvals(aliases, sources, observations)
    draft = extract_with_reviewed_roles(
        aliases, sources, observations, reviews, lambda spans: "Synthetic draft"
    )
    changed = (replace(sources[0], text=sources[0].text.replace("0", "1")),)
    assert len(changed[0].text) == len(sources[0].text)
    assert (
        assess_speaker_roles(aliases, changed, observations).context_digest
        != draft.context_digest
    )
    with pytest.raises(SpeakerRoleError, match="explicit role review"):
        finalize_role_bound_note(draft, aliases, changed, observations, reviews)
    fresh_reviews = approvals(aliases, changed, observations)
    with pytest.raises(SpeakerRoleError, match="unchanged reviewed role context"):
        finalize_role_bound_note(draft, aliases, changed, observations, fresh_reviews)


def test_changed_role_choice_or_note_text_cannot_finalize_old_extraction():
    aliases, sources, observations = context()
    reviews = approvals(aliases, sources, observations)
    draft = extract_with_reviewed_roles(
        aliases, sources, observations, reviews, lambda spans: "Synthetic draft"
    )
    for altered, decisions in (
        (replace(draft, text="Synthetic altered draft"), reviews),
        (draft, (replace(reviews[0], role=ClinicalRole.CLINICIAN),)),
    ):
        with pytest.raises(SpeakerRoleError, match="unchanged reviewed role context"):
            finalize_role_bound_note(altered, aliases, sources, observations, decisions)


@pytest.mark.parametrize(
    "kind", [LineageKind.RELABEL, LineageKind.MERGE, LineageKind.SPLIT]
)
def test_alias_relabel_merge_split_and_revision_changes_invalidate_previous_role_review(
    kind,
):
    aliases, sources, observations = context()
    if kind is LineageKind.MERGE:
        aliases.apply_revision(1, (ref(200), ref(201)))
        observations = (replace(observations[0], diarization_revision=1),)
    reviews = approvals(aliases, sources, observations)
    draft = extract_with_reviewed_roles(
        aliases, sources, observations, reviews, lambda spans: "Synthetic draft"
    )
    revision = aliases.to_dict()["revision"] + 1
    parents = (ref(200), ref(201)) if kind is LineageKind.MERGE else (ref(200),)
    children = (ref(202), ref(203)) if kind is LineageKind.SPLIT else (ref(202),)
    aliases.apply_revision(
        revision,
        children,
        (LineageEvent(kind=kind, parents=parents, children=children),),
    )
    report = assess_speaker_roles(aliases, sources, observations)
    assert "diarization_revision_changed" in report.assessments[0].blockers
    assert "speaker_not_active" in report.assessments[0].blockers
    with pytest.raises(SpeakerRoleError):
        finalize_role_bound_note(draft, aliases, sources, observations, reviews)
    current = (
        replace(
            observations[0], cluster_ref=children[0], diarization_revision=revision
        ),
    )
    with pytest.raises(SpeakerRoleError, match="explicit role review"):
        extract_with_reviewed_roles(
            aliases, sources, current, reviews, lambda spans: "Synthetic draft"
        )


def test_unfinalized_or_invalidated_evidence_cannot_be_overridden_by_role_review():
    aliases, sources, observations = context(finalized=False)
    report = assess_speaker_roles(aliases, sources, observations)
    assert "transcript_not_finalized" in report.assessments[0].blockers
    with pytest.raises(SpeakerRoleError, match="finalized source"):
        extract_with_reviewed_roles(
            aliases,
            sources,
            observations,
            approvals(aliases, sources, observations),
            lambda spans: "Synthetic draft",
        )
    ledger = sources[0].ledger
    length = len(sources[0].text)
    ledger.replace(
        0,
        length,
        length,
        (
            TranscriptToken(
                token_ref=ref(999), start=0, end=length, start_ms=0, end_ms=100
            ),
        ),
        expected_revision=0,
    )
    ledger.finalize(expected_revision=1)
    assert (
        "evidence_invalidated"
        in assess_speaker_roles(aliases, sources, observations).assessments[0].blockers
    )
    with pytest.raises(SpeakerRoleError):
        extract_with_reviewed_roles(
            aliases,
            sources,
            observations,
            approvals(aliases, sources, observations),
            lambda spans: "Synthetic draft",
        )


def test_timestamp_and_finalization_changes_require_new_role_binding():
    aliases, sources, observations = context(finalized=False)
    old = approvals(aliases, sources, observations)
    ledger = sources[0].ledger
    ledger.shift_timestamps((ref(400),), 50, expected_revision=0)
    ledger.finalize(expected_revision=1)
    report = assess_speaker_roles(aliases, sources, observations)
    assert report.assessments[0].primary.start_ms == 50
    with pytest.raises(SpeakerRoleError, match="explicit role review"):
        extract_with_reviewed_roles(
            aliases, sources, observations, old, lambda spans: "Synthetic draft"
        )


def test_explicit_provisional_extraction_cannot_bypass_finalized_note_gate():
    aliases, sources, observations = context(finalized=False)
    reviews = approvals(aliases, sources, observations)
    draft = extract_with_reviewed_roles(
        aliases,
        sources,
        observations,
        reviews,
        lambda spans: "Synthetic partial draft",
        allow_provisional=True,
    )
    assert draft.provisional and json.loads(draft.to_json())["provisional_sources"]
    with pytest.raises(SpeakerRoleError, match="finalized source"):
        finalize_role_bound_note(draft, aliases, sources, observations, reviews)
    sources[0].ledger.finalize(expected_revision=0)
    fresh = approvals(aliases, sources, observations)
    with pytest.raises(SpeakerRoleError, match="unchanged reviewed role context"):
        finalize_role_bound_note(draft, aliases, sources, observations, fresh)
    current = extract_with_reviewed_roles(
        aliases, sources, observations, fresh, lambda spans: "Synthetic final draft"
    )
    assert not current.provisional
    assert (
        json.loads(
            finalize_role_bound_note(
                current, aliases, sources, observations, fresh
            ).data_json
        )["note"]
        == "Synthetic final draft"
    )


def test_supporting_hypothesis_evidence_is_validated_even_when_primary_is_finalized():
    aliases, sources, observations = context(2, finalized=False)
    sources[0].ledger.finalize(expected_revision=0)
    first = replace(
        observations[0],
        hypotheses=(
            RoleHypothesis(
                role=ClinicalRole.PATIENT, evidence=(observations[1].citation,)
            ),
        ),
    )
    report = assess_speaker_roles(aliases, sources, (first,))
    assert report.assessments[0].primary.finalized
    assert report.assessments[0].hypotheses[0].confidence_ppm is None
    assert "transcript_not_finalized" in report.assessments[0].blockers


def test_closed_sessions_and_wrong_session_sources_fail_closed():
    aliases, sources, observations = context()
    reviews = approvals(aliases, sources, observations)
    aliases.close()
    assert (
        "session_closed"
        in assess_speaker_roles(aliases, sources, observations).assessments[0].blockers
    )
    with pytest.raises(SpeakerRoleError):
        extract_with_reviewed_roles(
            aliases, sources, observations, reviews, lambda spans: "Synthetic draft"
        )
    other = SpeakerAliases(ref(999))
    other.apply_revision(0, (ref(200),))
    with pytest.raises(SpeakerRoleError):
        assess_speaker_roles(other, sources, observations)


def test_fabricated_or_overlapping_registered_citations_are_rejected():
    aliases, sources, observations = context()
    forged = replace(
        observations[0], citation=replace(observations[0].citation, span_ref=ref(999))
    )
    with pytest.raises(SpeakerRoleError, match="registered") as raised:
        assess_speaker_roles(aliases, sources, (forged,))
    assert raised.value.__context__ is None
    duplicate = replace(observations[0], observation_id=ref(999))
    with pytest.raises(SpeakerRoleError, match="must not overlap"):
        assess_speaker_roles(aliases, sources, (observations[0], duplicate))


def test_callback_context_changes_discard_its_result():
    aliases, sources, observations = context()
    reviews = approvals(aliases, sources, observations)
    called = []

    def mutate(spans):
        called.append(True)
        aliases.apply_revision(1, (ref(200),))
        return "Synthetic stale draft"

    with pytest.raises(SpeakerRoleError):
        extract_with_reviewed_roles(aliases, sources, observations, reviews, mutate)
    assert called == [True]


def test_callback_errors_and_protected_content_stay_out_of_diagnostics():
    aliases, sources, observations = context()
    reviews = approvals(aliases, sources, observations)
    sensitive = "Synthetic confidential callback content"

    def fail(spans):
        raise RuntimeError(sensitive)

    with pytest.raises(SpeakerRoleError) as raised:
        extract_with_reviewed_roles(aliases, sources, observations, reviews, fail)
    assert sensitive not in str(raised.value) and raised.value.__context__ is None
    draft = extract_with_reviewed_roles(
        aliases, sources, observations, reviews, lambda spans: sensitive
    )
    note = finalize_role_bound_note(draft, aliases, sources, observations, reviews)
    report = assess_speaker_roles(aliases, sources, observations)
    diagnostics = (
        repr(sources)
        + repr(report)
        + report.to_json()
        + repr(draft)
        + draft.to_json()
        + repr(note)
        + note.to_json()
    )
    assert sensitive not in diagnostics and sources[0].text not in diagnostics
    changed_report = replace(draft, report_json=sensitive)
    assert (
        sensitive
        not in finalize_role_bound_note(
            changed_report, aliases, sources, observations, reviews
        ).to_json()
    )


def test_offline_determinism_and_source_order_invariance(monkeypatch):
    aliases, sources, observations = context(2)
    monkeypatch.setattr(
        socket, "socket", lambda *a, **k: pytest.fail("unexpected network")
    )
    first = assess_speaker_roles(aliases, sources, observations)
    assert (
        first.to_json()
        == assess_speaker_roles(aliases, sources, observations).to_json()
    )
    assert (
        first.context_digest
        == assess_speaker_roles(aliases, sources[::-1], observations).context_digest
    )


def test_bounds_and_hypothesis_contracts_fail_with_fixed_messages():
    aliases, sources, observations = context()
    for malformed in ((), sources * 33):
        with pytest.raises(SpeakerRoleError):
            assess_speaker_roles(aliases, malformed, observations)
    with pytest.raises(SpeakerRoleError):
        assess_speaker_roles(aliases, sources, observations * 129)
    for text in ("", " ", "x" * 65_537, "Synthetic\ud800"):
        with pytest.raises(SpeakerRoleError) as raised:
            replace(sources[0], text=text)
        assert "Synthetic" not in str(raised.value)
    for score in (-1, 1_000_001, True):
        with pytest.raises(SpeakerRoleError):
            replace(observations[0].hypotheses[0], confidence_ppm=score)
    with pytest.raises(SpeakerRoleError):
        replace(observations[0], hypotheses=(observations[0].hypotheses[0],) * 2)
    with pytest.raises(SpeakerRoleError):
        replace(observations[0].hypotheses[0], evidence=())
    many = context(5, text="x" * 65_536)
    with pytest.raises(SpeakerRoleError):
        assess_speaker_roles(*many)


@pytest.mark.parametrize("output", [None, b"Synthetic bytes", "", "x" * 65_537])
def test_extractor_output_is_bounded_protected_text(output):
    aliases, sources, observations = context()
    with pytest.raises(SpeakerRoleError):
        extract_with_reviewed_roles(
            aliases,
            sources,
            observations,
            approvals(aliases, sources, observations),
            lambda spans: output,
        )
