"""Synthetic acoustic lineage and reviewer-confirmation controls."""

import pytest

from openmed.multimodal.streaming.speech.speaker_aliases import (
    MAX_ACTIVE_CLUSTERS,
    MAX_CLUSTERS,
    ClinicalRole,
    LineageEvent,
    LineageKind,
    RoleReview,
    SpeakerAliasError,
    SpeakerAliases,
)


def ref(index):
    return f"{index:032x}"


def event(kind, parents, children):
    return LineageEvent(
        kind=kind,
        parents=tuple(ref(i) for i in parents),
        children=tuple(ref(i) for i in children),
    )


@pytest.fixture
def session():
    state = SpeakerAliases(ref(100))
    state.apply_revision(0, (ref(1), ref(2)))
    return state


def review(session, alias="speaker_0001", role=ClinicalRole.PATIENT, revision=0):
    return RoleReview(
        session_ref=ref(100),
        alias=alias,
        role=role,
        revision=revision,
        review_ref=ref(101),
    )


def test_reordering_and_temporary_absence_preserve_aliases(session):
    first = session.alias_for(ref(1))
    second = session.alias_for(ref(2))
    session.apply_revision(1, (ref(2), ref(1)))
    assert session.alias_for(ref(1)) == first
    session.apply_revision(2, (ref(2),))
    with pytest.raises(SpeakerAliasError):
        session.alias_for(ref(1))
    session.apply_revision(3, (ref(1), ref(2)))
    assert session.alias_for(ref(1)) == first
    assert session.alias_for(ref(2)) == second


def test_input_order_does_not_allocate_different_aliases():
    a, b = SpeakerAliases(ref(100)), SpeakerAliases(ref(100))
    a.apply_revision(0, (ref(3), ref(1), ref(2)))
    b.apply_revision(0, (ref(2), ref(3), ref(1)))
    assert a.to_json() == b.to_json()


def test_unknown_clusters_do_not_inherit_an_absent_speakers_role(session):
    session.confirm_role(review(session))
    session.apply_revision(1, (ref(3),))
    assert session.alias_for(ref(3)) == "speaker_0003"
    assert session.to_dict()["active"][0]["role"] is None


def test_explicit_relabel_preserves_alias_and_confirmed_role(session):
    session.confirm_role(review(session))
    session.apply_revision(
        1, (ref(2), ref(3)), (event(LineageKind.RELABEL, (1,), (3,)),)
    )
    assert session.alias_for(ref(3)) == "speaker_0001"
    assert session.to_dict()["active"][1]["role"] == "patient"
    assert session.to_dict()["lineage"][0]["parent_aliases"] == ("speaker_0001",)
    with pytest.raises(SpeakerAliasError):
        session.apply_revision(2, (ref(1), ref(2), ref(3)))


def test_merge_keeps_earliest_alias_but_clears_role_attestations(session):
    session.confirm_role(review(session))
    session.confirm_role(
        review(session, alias="speaker_0002", role=ClinicalRole.CLINICIAN)
    )
    session.apply_revision(1, (ref(3),), (event(LineageKind.MERGE, (2, 1), (3,)),))
    assert session.alias_for(ref(3)) == "speaker_0001"
    assert session.to_dict()["active"][0]["role"] is None
    record = session.to_dict()["lineage"][0]
    assert record["parent_aliases"] == ("speaker_0001", "speaker_0002")
    assert record["child_aliases"] == ("speaker_0001",)
    session.confirm_role(review(session, revision=1))
    assert session.to_dict()["active"][0]["role"] == "patient"


def test_split_retires_ambiguous_alias_and_never_copies_role(session):
    session.confirm_role(review(session))
    session.apply_revision(
        1, (ref(2), ref(3), ref(4)), (event(LineageKind.SPLIT, (1,), (4, 3)),)
    )
    assert session.alias_for(ref(3)) == "speaker_0003"
    assert session.alias_for(ref(4)) == "speaker_0004"
    assert all(x["role"] is None for x in session.to_dict()["active"])
    with pytest.raises(SpeakerAliasError):
        session.confirm_role(review(session, revision=1))
    assert session.to_dict()["lineage"][0]["parent_aliases"] == ("speaker_0001",)


def test_role_requires_explicit_current_session_review(session):
    assert all(x["role"] is None for x in session.to_dict()["active"])
    with pytest.raises(SpeakerAliasError):
        session.confirm_role(ClinicalRole.PATIENT)
    wrong_session = RoleReview(
        session_ref=ref(999),
        alias="speaker_0001",
        role=ClinicalRole.PATIENT,
        revision=0,
        review_ref=ref(101),
    )
    with pytest.raises(SpeakerAliasError):
        session.confirm_role(wrong_session)
    session.apply_revision(1, (ref(1), ref(2)))
    with pytest.raises(SpeakerAliasError):
        session.confirm_role(review(session))
    session.confirm_role(review(session, revision=1))
    assert session.to_dict()["active"][0]["role_review_ref"] == ref(101)


def test_independent_sessions_never_share_roles(session):
    session.confirm_role(review(session))
    other = SpeakerAliases(ref(200))
    other.apply_revision(0, (ref(1),))
    assert other.to_dict()["active"][0]["role"] is None
    assert other.to_dict()["session_ref"] != session.to_dict()["session_ref"]


@pytest.mark.parametrize(
    "revision,clusters,events",
    [
        (0, (ref(1),), ()),
        (-1, (ref(1),), ()),
        (True, (ref(1),), ()),
        (1, (ref(1), ref(1)), ()),
        (1, (ref(3),), (event(LineageKind.RELABEL, (99,), (3,)),)),
        (1, (ref(1), ref(3)), (event(LineageKind.RELABEL, (1,), (3,)),)),
        (
            1,
            (ref(3), ref(4)),
            (
                event(LineageKind.RELABEL, (1,), (3,)),
                event(LineageKind.RELABEL, (1,), (4,)),
            ),
        ),
        (
            1,
            (ref(3),),
            (
                event(LineageKind.RELABEL, (1,), (3,)),
                event(LineageKind.RELABEL, (2,), (3,)),
            ),
        ),
        (1, (ref(3),), (event(LineageKind.RELABEL, (1,), (4,)),)),
    ],
)
def test_invalid_revision_is_atomic(session, revision, clusters, events):
    session.confirm_role(review(session))
    before = session.to_json()
    with pytest.raises(SpeakerAliasError, match="^invalid speaker alias transition$"):
        session.apply_revision(revision, clusters, events)
    assert session.to_json() == before
    session.apply_revision(2, (ref(1), ref(2), ref(8)))
    assert session.alias_for(ref(8)) == "speaker_0003"


@pytest.mark.parametrize(
    "kind,parents,children",
    [
        (LineageKind.MERGE, (1,), (2,)),
        (LineageKind.SPLIT, (1,), (2,)),
        (LineageKind.RELABEL, (1, 2), (3,)),
        (LineageKind.MERGE, (1, 1), (2,)),
        (LineageKind.SPLIT, (1,), (1, 2)),
    ],
)
def test_explicit_lineage_shapes_are_required(kind, parents, children):
    with pytest.raises(SpeakerAliasError):
        event(kind, parents, children)


def test_disjoint_events_are_order_independent(session):
    other = SpeakerAliases(ref(100))
    other.apply_revision(0, (ref(1), ref(2)))
    events = (
        event(LineageKind.SPLIT, (1,), (3, 4)),
        event(LineageKind.RELABEL, (2,), (5,)),
    )
    session.apply_revision(1, (ref(3), ref(4), ref(5)), events)
    other.apply_revision(1, (ref(5), ref(4), ref(3)), tuple(reversed(events)))
    assert session.to_json() == other.to_json()


def test_snapshot_mutation_does_not_mutate_the_registry(session):
    session.apply_revision(
        1, (ref(2), ref(3)), (event(LineageKind.RELABEL, (1,), (3,)),)
    )
    before = session.to_json()
    report = session.to_dict()
    report["active"][0]["alias"] = "changed"
    report["lineage"][0]["kind"] = "changed"
    assert session.to_json() == before


@pytest.mark.parametrize(
    "value",
    ["synthetic-private-name", "patient@example.invalid", "../synthetic-path", 1, True],
)
def test_free_text_labels_are_rejected_without_echo(value, session, caplog):
    with pytest.raises(SpeakerAliasError) as caught:
        session.apply_revision(1, (value,))
    assert str(caught.value) == "invalid speaker alias transition"
    assert caplog.records == []


def test_resource_limits_do_not_reset_or_evict_previous_aliases():
    session = SpeakerAliases(ref(10000))
    for batch in range(MAX_CLUSTERS // MAX_ACTIVE_CLUSTERS):
        ids = tuple(
            ref(batch * MAX_ACTIVE_CLUSTERS + i) for i in range(MAX_ACTIVE_CLUSTERS)
        )
        session.apply_revision(batch, ids)
    before = session.to_json()
    with pytest.raises(SpeakerAliasError):
        session.apply_revision(100, (ref(MAX_CLUSTERS),))
    assert session.to_json() == before
    session.apply_revision(101, (ref(0),))
    assert session.alias_for(ref(0)) == "speaker_0001"


def test_close_discards_linkage_and_rejects_new_work(session):
    session.confirm_role(review(session))
    session.close()
    session.close()
    assert session.to_dict()["active"] == []
    assert session.to_dict()["lineage"] == []
    assert session.to_dict()["closed"] is True
    with pytest.raises(SpeakerAliasError):
        session.apply_revision(1, (ref(1),))
    with pytest.raises(SpeakerAliasError):
        session.confirm_role(review(session))


def test_report_contains_no_identity_or_biometric_fields(session):
    report = session.to_json()
    assert all(
        key not in report
        for key in ('"name":', '"identity":', '"embedding":', '"audio":')
    )
