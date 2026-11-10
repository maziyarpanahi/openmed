from __future__ import annotations

import copy
import hashlib
import json
import re
import socket

import pytest

from openmed.eval.annotation.interchange import CoordinateConvention
from openmed.eval.native_review_packets import (
    DEFAULT_REVIEWER_ROLES,
    NATIVE_REVIEW_NOTICE,
    NATIVE_REVIEW_SUBMISSION_SCHEMA_VERSION,
    NativeReviewDecision,
    NativeReviewIncompleteError,
    NativeReviewItem,
    NativeReviewManifest,
    NativeReviewPacket,
    NativeReviewPacketError,
    NativeReviewSpan,
    NativeReviewTask,
    ReviewPrivacyStatus,
    build_native_review_packet,
    import_native_review_decisions,
    verify_native_review_manifest,
)

# Synthetic marker values. None of them may surface in errors or packets.
MARKER_SOURCE_PREFIX = "synthetic-ward7-bed"
MARKER_NAME = "Zuri Amani"
MARKER_REVIEWER = "Synthetic Reviewer Okafor"


def _digest(label: str) -> str:
    return f"sha256:{hashlib.sha256(label.encode('utf-8')).hexdigest()}"


def _label_task() -> NativeReviewTask:
    return NativeReviewTask(
        task_code="label_check",
        instructions=(
            "Confirm whether each highlighted span carries the stated label "
            "in natural Swahili clinical usage."
        ),
        decision_codes=("accept", "reject", "revise", "unsure"),
        label_codes=("CONDITION", "MEDICATION", "PERSON"),
        reason_codes=("boundary", "label", "morphology"),
        correction_decision_code="revise",
    )


def _fluency_task() -> NativeReviewTask:
    return NativeReviewTask(
        task_code="fluency_check",
        instructions="Decide whether the sentence reads as natural clinical Swahili.",
        decision_codes=("natural", "awkward", "unsure"),
    )


def _span(text: str, surface: str, label: str) -> NativeReviewSpan:
    start = text.index(surface)
    return NativeReviewSpan(start=start, end=start + len(surface), label=label)


def _items(count: int = 6) -> tuple[NativeReviewItem, ...]:
    items = []
    for index in range(count):
        if index % 2 == 0:
            text = f"Mgonjwa {MARKER_NAME} ana homa siku {index + 2}."
            spans = (
                _span(text, MARKER_NAME, "PERSON"),
                _span(text, "homa", "CONDITION"),
            )
            task_code = "label_check"
        else:
            text = f"Alipewa dawa ya paracetamol mara {index} kwa siku."
            spans = ()
            task_code = "fluency_check"
        items.append(
            NativeReviewItem(
                source_ref=f"{MARKER_SOURCE_PREFIX}-{index:02d}",
                task_code=task_code,
                privacy_status=(
                    ReviewPrivacyStatus.SYNTHETIC
                    if index < 4
                    else ReviewPrivacyStatus.DEIDENTIFIED
                ),
                provenance_digest=_digest(f"provenance-{index}"),
                text=text,
                spans=spans,
            )
        )
    return tuple(items)


def _build(**overrides):
    arguments = {
        "items": _items(),
        "packet_ref": "sw-review-assignment-001",
        "language": "sw",
        "tasks": (_label_task(), _fluency_task()),
        "seed": 20261024,
    }
    arguments.update(overrides)
    return build_native_review_packet(**arguments)


def _filled_submission(packet: NativeReviewPacket, role: str = "native_clinician"):
    submission = packet.submission_template()
    submission["reviewer_role"] = role
    for row in submission["decisions"]:
        item = next(item for item in packet.items if item.item_id == row["item_id"])
        row["decision"] = "accept" if item.task_code == "label_check" else "natural"
    return submission


def _assert_value_free(error: BaseException, *values: str) -> None:
    rendered = f"{error} {error!r} {error.args!r}"
    for value in values:
        assert value not in rendered
    assert error.__cause__ is None


def test_packet_is_deterministic_canonical_and_input_order_independent() -> None:
    first_packet, first_manifest = _build()
    second_packet, second_manifest = _build(
        items=tuple(reversed(_items())),
        tasks=(_fluency_task(), _label_task()),
    )

    assert first_packet == second_packet
    assert first_manifest == second_manifest
    assert first_packet.to_json() == second_packet.to_json()
    assert first_manifest.to_json() == second_manifest.to_json()
    assert first_packet.to_json() == json.dumps(
        first_packet.to_dict(),
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    assert first_manifest.packet_digest == first_packet.packet_digest
    assert first_packet.packet_digest == (
        "sha256:" + hashlib.sha256(first_packet.to_json().encode()).hexdigest()
    )


def test_seed_randomizes_order_and_item_ids() -> None:
    items = _items(12)
    packet_a, manifest_a = _build(items=items, seed=1)
    packet_b, manifest_b = _build(items=items, seed=2)

    order_a = [entry.source_ref for entry in manifest_a.entries]
    order_b = [entry.source_ref for entry in manifest_b.entries]
    input_order = [item.source_ref for item in items]
    assert sorted(order_a) == sorted(order_b) == sorted(input_order)
    assert order_a != order_b
    assert order_a != input_order
    assert {item.item_id for item in packet_a.items}.isdisjoint(
        {item.item_id for item in packet_b.items}
    )
    assert all(
        re.fullmatch(r"nri_[0-9a-f]{32}", item.item_id) for item in packet_a.items
    )


def test_packet_carries_instructions_provenance_and_no_private_values() -> None:
    packet, manifest = _build()
    document = packet.to_dict()
    serialized = packet.to_json()

    assert document["language"] == "sw"
    assert document["notice"] == NATIVE_REVIEW_NOTICE
    assert document["reviewer_roles"] == sorted(DEFAULT_REVIEWER_ROLES)
    assert [task["task_code"] for task in document["tasks"]] == [
        "fluency_check",
        "label_check",
    ]
    assert all(task["instructions"] for task in document["tasks"])
    assert re.fullmatch(r"sha256:[0-9a-f]{64}", document["provenance_commitment"])
    assert document["coordinate_convention"] == (
        CoordinateConvention.UNICODE_CODEPOINT.value
    )
    assert {item["privacy_status"] for item in document["items"]} == {
        "deidentified",
        "synthetic",
    }

    assert MARKER_SOURCE_PREFIX not in serialized
    assert "20261024" not in serialized
    for entry in manifest.entries:
        assert entry.provenance_digest not in serialized
    for item in document["items"]:
        assert set(item) == {"item_id", "privacy_status", "spans", "task_code", "text"}


def test_only_used_tasks_enter_the_packet() -> None:
    unused = NativeReviewTask(
        task_code="terminology_check",
        instructions="Confirm the preferred local term.",
    )
    packet, _ = _build(
        items=_items()[:1],
        tasks=(_label_task(), _fluency_task(), unused),
    )

    assert [task.task_code for task in packet.tasks] == ["label_check"]


@pytest.mark.parametrize(
    "status",
    ["raw", "identified", "unknown", "", "SYNTHETIC", None, 1],
)
def test_items_not_marked_synthetic_or_deidentified_are_refused(status) -> None:
    with pytest.raises(NativeReviewPacketError, match="not_reviewable") as caught:
        NativeReviewItem(
            source_ref=f"{MARKER_SOURCE_PREFIX}-99",
            task_code="fluency_check",
            privacy_status=status,
            provenance_digest=_digest("provenance-99"),
            text=f"Mgonjwa {MARKER_NAME} ana homa.",
        )

    _assert_value_free(caught.value, MARKER_NAME, MARKER_SOURCE_PREFIX)


def test_items_must_stay_within_minimal_context() -> None:
    long_item = NativeReviewItem(
        source_ref="long-item",
        task_code="fluency_check",
        privacy_status="synthetic",
        provenance_digest=_digest("provenance-long"),
        text="homa " * 40,
    )

    with pytest.raises(NativeReviewPacketError) as caught:
        _build(items=(long_item,), max_item_characters=100)
    assert str(caught.value) == "items[0].text: exceeds_minimal_context"

    packet, _ = _build(items=(long_item,), max_item_characters=200)
    assert len(packet.items) == 1

    with pytest.raises(NativeReviewPacketError, match="exceeds_minimal_context"):
        NativeReviewItem(
            source_ref="huge-item",
            task_code="fluency_check",
            privacy_status="synthetic",
            provenance_digest=_digest("provenance-huge"),
            text="a" * 4_001,
        )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"packet_ref": "not a ref"}, "packet_ref: invalid_reference"),
        ({"language": "Swahili"}, "language: invalid_language"),
        ({"seed": -1}, "seed: invalid_seed"),
        ({"seed": True}, "seed: invalid_seed"),
        ({"items": ()}, "items: invalid_sequence"),
        ({"tasks": (_label_task(),)}, "unknown_task"),
        ({"tasks": (_label_task(), _label_task())}, "tasks: duplicate_code"),
        ({"reviewer_roles": "native_speaker"}, "reviewer_roles: invalid_sequence"),
        ({"reviewer_roles": ("Dr Okafor",)}, "reviewer_roles: invalid_code"),
        ({"max_item_characters": 0}, "max_item_characters: invalid_limit"),
    ],
)
def test_build_rejects_invalid_inputs(overrides, message) -> None:
    with pytest.raises(NativeReviewPacketError, match=message) as caught:
        _build(**overrides)

    _assert_value_free(caught.value, MARKER_NAME, MARKER_SOURCE_PREFIX, "Okafor")


def test_build_rejects_duplicate_sources_and_unknown_labels() -> None:
    items = _items()
    with pytest.raises(NativeReviewPacketError, match="duplicate_source_ref"):
        _build(items=items + (items[0],))

    narrow = NativeReviewTask(
        task_code="label_check",
        instructions="Confirm condition labels only.",
        label_codes=("CONDITION",),
    )
    with pytest.raises(NativeReviewPacketError, match="spans: unknown_label"):
        _build(tasks=(narrow, _fluency_task()))


def test_span_integrity_is_enforced_for_multilingual_text() -> None:
    hindi = "रोगी को बुखार है"
    start = hindi.index("बुखार")
    hindi_task = NativeReviewTask(
        task_code="label_check",
        instructions="Confirm condition labels in natural Hindi clinical usage.",
        label_codes=("CONDITION",),
    )

    def hindi_item(span: NativeReviewSpan) -> NativeReviewItem:
        return NativeReviewItem(
            source_ref="hi-item-01",
            task_code="label_check",
            privacy_status="synthetic",
            provenance_digest=_digest("provenance-hi"),
            text=hindi,
            spans=(span,),
        )

    packet, _ = _build(
        items=(hindi_item(NativeReviewSpan(start, start + 5, "CONDITION")),),
        language="hi",
        tasks=(hindi_task,),
    )
    assert packet.items[0].spans[0].end - packet.items[0].spans[0].start == 5

    with pytest.raises(NativeReviewPacketError, match="spans: split_grapheme"):
        hindi_item(NativeReviewSpan(start, start + 1, "CONDITION"))
    with pytest.raises(NativeReviewPacketError, match="spans: split_grapheme"):
        # Ends before a spacing vowel sign (category Mc).
        hindi_item(NativeReviewSpan(start, start + 3, "CONDITION"))
    with pytest.raises(NativeReviewPacketError, match="spans: offset_out_of_range"):
        hindi_item(NativeReviewSpan(start, len(hindi) + 1, "CONDITION"))

    emoji = "\U0001f469\u200d\u2695\ufe0f homa"
    with pytest.raises(NativeReviewPacketError, match="spans: split_grapheme"):
        NativeReviewItem(
            source_ref="emoji-item",
            task_code="label_check",
            privacy_status="synthetic",
            provenance_digest=_digest("provenance-emoji"),
            text=emoji,
            spans=(NativeReviewSpan(0, 1, "PERSON"),),
        )
    with pytest.raises(NativeReviewPacketError, match="spans: duplicate_span"):
        NativeReviewItem(
            source_ref="dup-span",
            task_code="label_check",
            privacy_status="synthetic",
            provenance_digest=_digest("provenance-dup"),
            text="homa kali",
            spans=(
                NativeReviewSpan(0, 4, "CONDITION"),
                NativeReviewSpan(0, 4, "CONDITION"),
            ),
        )


def test_packet_and_manifest_round_trip_through_strict_json() -> None:
    packet, manifest = _build()

    restored_packet = NativeReviewPacket.from_json(packet.to_json().encode("utf-8"))
    restored_manifest = NativeReviewManifest.from_json(manifest.to_json())

    assert restored_packet == packet
    assert restored_manifest == manifest
    assert restored_packet.packet_digest == packet.packet_digest
    assert restored_manifest.manifest_digest == manifest.manifest_digest
    verify_native_review_manifest(restored_packet, restored_manifest)

    relabelled = packet.to_dict()
    relabelled["items"][0]["privacy_status"] = "identified"
    with pytest.raises(NativeReviewPacketError, match="not_reviewable"):
        NativeReviewPacket.from_json(json.dumps(relabelled))

    extended = packet.to_dict()
    extended["source_refs"] = [MARKER_SOURCE_PREFIX]
    with pytest.raises(NativeReviewPacketError) as caught:
        NativeReviewPacket.from_dict(extended)
    assert str(caught.value) == "packet: unknown_field"
    _assert_value_free(caught.value, MARKER_SOURCE_PREFIX)


def test_manifest_verification_detects_tampering() -> None:
    packet, manifest = _build()

    tampered_packet = packet.to_dict()
    tampered_packet["items"][0]["text"] = "Mgonjwa ana kikohozi."
    tampered_packet["items"][0]["spans"] = []
    with pytest.raises(NativeReviewPacketError, match="packet_digest_mismatch"):
        verify_native_review_manifest(
            NativeReviewPacket.from_dict(tampered_packet), manifest
        )

    swapped = manifest.to_dict()
    first, second = swapped["entries"][0], swapped["entries"][1]
    first["source_ref"], second["source_ref"] = (
        second["source_ref"],
        first["source_ref"],
    )
    with pytest.raises(NativeReviewPacketError, match="provenance_mismatch") as caught:
        verify_native_review_manifest(packet, NativeReviewManifest.from_dict(swapped))
    _assert_value_free(caught.value, MARKER_SOURCE_PREFIX)

    other_packet, other_manifest = _build(packet_ref="sw-review-assignment-002")
    with pytest.raises(NativeReviewPacketError, match="manifest: packet_mismatch"):
        verify_native_review_manifest(packet, other_manifest)
    with pytest.raises(NativeReviewPacketError, match="manifest: packet_mismatch"):
        verify_native_review_manifest(other_packet, manifest)


def test_import_resolves_opaque_items_without_reviewer_identity() -> None:
    packet, manifest = _build()
    submission = _filled_submission(packet)

    imported = import_native_review_decisions(
        json.dumps(submission), packet=packet, manifest=manifest
    )

    assert imported.completeness.complete
    assert imported.completeness.expected_count == len(packet.items)
    assert imported.reviewer_role == "native_clinician"
    assert imported.manifest_digest == manifest.manifest_digest
    assert [decision.item_id for decision in imported.decisions] == [
        item.item_id for item in packet.items
    ]
    assert [decision.source_ref for decision in imported.decisions] == [
        entry.source_ref for entry in manifest.entries
    ]
    for decision, entry in zip(imported.decisions, manifest.entries):
        assert re.fullmatch(r"nrd_[0-9a-f]{32}", decision.decision_id)
        assert decision.language == "sw"
        assert decision.provenance_digest == entry.provenance_digest
        assert decision.text_digest == entry.text_digest
        assert decision.projection_digest == entry.projection_digest
        assert decision.privacy_status is entry.privacy_status
        assert set(decision.to_dict()) == {
            "corrections",
            "decision",
            "decision_id",
            "item_id",
            "language",
            "packet_digest",
            "packet_ref",
            "privacy_status",
            "projection_digest",
            "provenance_digest",
            "reason_code",
            "reviewer_role",
            "schema_version",
            "source_ref",
            "submission_digest",
            "task_code",
            "text_digest",
        }

    serialized = imported.to_json()
    assert MARKER_NAME not in serialized
    assert not re.search(r'"(reviewer_id|reviewer_name|email|name)"', serialized)


def test_import_is_idempotent_and_independent_of_decision_order() -> None:
    packet, manifest = _build()
    submission = _filled_submission(packet)
    reordered = copy.deepcopy(submission)
    reordered["decisions"].reverse()

    first = import_native_review_decisions(submission, packet=packet, manifest=manifest)
    second = import_native_review_decisions(reordered, packet=packet, manifest=manifest)
    assert first == second
    assert first.to_json() == second.to_json()

    changed = copy.deepcopy(submission)
    changed["decisions"][0]["decision"] = "unsure"
    third = import_native_review_decisions(changed, packet=packet, manifest=manifest)
    assert third.submission_digest != first.submission_digest
    assert third.decisions[0].decision_id != first.decisions[0].decision_id
    assert (third.decisions[0].packet_ref, third.decisions[0].item_id) == (
        first.decisions[0].packet_ref,
        first.decisions[0].item_id,
    )


@pytest.mark.parametrize(
    ("location", "key", "message"),
    [
        ("top", "reviewer_name", "submission: reviewer_identity_forbidden"),
        ("top", "reviewer_id", "submission: reviewer_identity_forbidden"),
        ("top", "email", "submission: reviewer_identity_forbidden"),
        ("top", "comments", "submission: free_text_forbidden"),
        ("top", "created_by_tool", "submission: unknown_field"),
        ("decision", "annotator_id", "decisions: reviewer_identity_forbidden"),
        ("decision", "note", "decisions: free_text_forbidden"),
        ("decision", "text", "decisions: free_text_forbidden"),
        ("decision", "confidence", "decisions: unknown_field"),
    ],
)
def test_import_rejects_identity_free_text_and_unknown_fields(
    location, key, message
) -> None:
    packet, manifest = _build()
    submission = _filled_submission(packet)
    target = submission if location == "top" else submission["decisions"][0]
    target[key] = MARKER_REVIEWER

    with pytest.raises(NativeReviewPacketError) as caught:
        import_native_review_decisions(submission, packet=packet, manifest=manifest)

    assert str(caught.value) == message
    _assert_value_free(caught.value, MARKER_REVIEWER)


def test_import_validates_completeness_against_the_manifest() -> None:
    packet, manifest = _build()
    submission = _filled_submission(packet)
    removed = submission["decisions"].pop(1)["item_id"]
    submission["decisions"][0]["decision"] = None
    undecided = submission["decisions"][0]["item_id"]

    with pytest.raises(NativeReviewIncompleteError) as caught:
        import_native_review_decisions(submission, packet=packet, manifest=manifest)
    assert str(caught.value) == "decisions: incomplete"
    completeness = caught.value.completeness
    assert not completeness.complete
    assert completeness.expected_count == len(packet.items)
    assert completeness.decided_count == len(packet.items) - 2
    assert set(completeness.missing_item_ids) == {removed, undecided}

    partial = import_native_review_decisions(
        submission, packet=packet, manifest=manifest, require_complete=False
    )
    assert partial.completeness == completeness
    assert len(partial.decisions) == len(packet.items) - 2
    assert {removed, undecided}.isdisjoint(
        decision.item_id for decision in partial.decisions
    )

    template = import_native_review_decisions(
        {**packet.submission_template(), "reviewer_role": "native_speaker"},
        packet=packet,
        manifest=manifest,
        require_complete=False,
    )
    assert template.decisions == ()
    assert template.completeness.decided_count == 0


def test_import_rejects_unknown_duplicate_and_mismatched_submissions() -> None:
    packet, manifest = _build()

    unknown = _filled_submission(packet)
    unknown["decisions"][0]["item_id"] = f"nri_{MARKER_SOURCE_PREFIX}"
    with pytest.raises(NativeReviewPacketError, match="decisions: unknown_item") as c:
        import_native_review_decisions(unknown, packet=packet, manifest=manifest)
    _assert_value_free(c.value, MARKER_SOURCE_PREFIX)

    duplicate = _filled_submission(packet)
    duplicate["decisions"].append(dict(duplicate["decisions"][0]))
    with pytest.raises(NativeReviewPacketError, match="decisions: duplicate_item"):
        import_native_review_decisions(duplicate, packet=packet, manifest=manifest)

    for digest in (_digest("an-older-packet"), "sha256:é", None):
        stale = _filled_submission(packet)
        stale["packet_digest"] = digest
        with pytest.raises(
            NativeReviewPacketError, match="submission: packet_mismatch"
        ):
            import_native_review_decisions(stale, packet=packet, manifest=manifest)

    wrong_schema = _filled_submission(packet)
    wrong_schema["schema_version"] = "openmed.eval.native_review.submission.v0"
    with pytest.raises(NativeReviewPacketError, match="unsupported_version"):
        import_native_review_decisions(wrong_schema, packet=packet, manifest=manifest)

    other_packet, other_manifest = _build(seed=7)
    with pytest.raises(NativeReviewPacketError, match="manifest: packet_digest"):
        import_native_review_decisions(
            _filled_submission(packet), packet=packet, manifest=other_manifest
        )
    with pytest.raises(NativeReviewPacketError, match="submission: packet_mismatch"):
        import_native_review_decisions(
            _filled_submission(packet), packet=other_packet, manifest=other_manifest
        )


@pytest.mark.parametrize("role", [None, "Dr Okafor", "okafor_reviewer", 3])
def test_import_accepts_only_declared_reviewer_roles(role) -> None:
    packet, manifest = _build()
    submission = _filled_submission(packet, role=role)

    with pytest.raises(NativeReviewPacketError) as caught:
        import_native_review_decisions(submission, packet=packet, manifest=manifest)

    assert str(caught.value) == "submission.reviewer_role: not_allowed"
    _assert_value_free(caught.value, "Okafor", "okafor")


def _label_row(packet: NativeReviewPacket, submission: dict) -> tuple[dict, str]:
    item = next(item for item in packet.items if item.task_code == "label_check")
    row = next(row for row in submission["decisions"] if row["item_id"] == item.item_id)
    return row, item.text


def test_import_accepts_valid_corrections_and_reason_codes() -> None:
    packet, manifest = _build()
    submission = _filled_submission(packet)
    row, text = _label_row(packet, submission)
    start = text.index("homa")
    row["decision"] = "revise"
    row["reason_code"] = "boundary"
    row["corrections"] = [
        {"start": start, "end": start + 4, "label": "CONDITION"},
        {"start": 0, "end": 7, "label": "PERSON"},
    ]

    imported = import_native_review_decisions(
        submission, packet=packet, manifest=manifest
    )
    decision = next(item for item in imported.decisions if item.decision == "revise")

    assert decision.reason_code == "boundary"
    assert [span.to_dict() for span in decision.corrections] == [
        {"end": 7, "label": "PERSON", "start": 0},
        {"end": start + 4, "label": "CONDITION", "start": start},
    ]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"decision": "maybe"}, "decisions.decision: unknown_code"),
        ({"decision": 1}, "decisions.decision: unknown_code"),
        ({"reason_code": "typo"}, "decisions.reason_code: unknown_code"),
        ({"decision": "revise"}, "decisions.corrections: required"),
        (
            {"corrections": [{"start": 0, "end": 7, "label": "PERSON"}]},
            "decisions.corrections: not_allowed",
        ),
        (
            {
                "decision": "revise",
                "corrections": [{"start": 0, "end": 7, "label": "DIAGNOSIS"}],
            },
            "decisions.corrections: unknown_label",
        ),
        (
            {
                "decision": "revise",
                "corrections": [{"start": 0, "end": 900, "label": "PERSON"}],
            },
            "decisions.corrections: offset_out_of_range",
        ),
        (
            {
                "decision": "revise",
                "corrections": [{"start": 0, "end": 7, "label": "PERSON", "x": 1}],
            },
            "span: unknown_field",
        ),
        (
            {"decision": None, "reason_code": "boundary"},
            "decisions: undecided_with_details",
        ),
    ],
)
def test_import_rejects_invalid_codes_and_corrections(mutation, message) -> None:
    packet, manifest = _build()
    submission = _filled_submission(packet)
    row, _ = _label_row(packet, submission)
    row.update(mutation)

    with pytest.raises(NativeReviewPacketError) as caught:
        import_native_review_decisions(submission, packet=packet, manifest=manifest)

    assert str(caught.value) == message


def test_fluency_tasks_do_not_accept_corrections() -> None:
    packet, manifest = _build()
    submission = _filled_submission(packet)
    item = next(item for item in packet.items if item.task_code == "fluency_check")
    row = next(row for row in submission["decisions"] if row["item_id"] == item.item_id)
    row["decision"] = "accept"

    with pytest.raises(NativeReviewPacketError, match="decisions.decision"):
        import_native_review_decisions(submission, packet=packet, manifest=manifest)


def test_imported_decisions_round_trip_and_bind_lineage() -> None:
    packet, manifest = _build()
    imported = import_native_review_decisions(
        _filled_submission(packet), packet=packet, manifest=manifest
    )
    decision = imported.decisions[0]

    assert NativeReviewDecision.from_dict(decision.to_dict()) == decision
    restored = NativeReviewDecision.from_dict(
        json.loads(json.dumps(decision.to_dict()))
    )
    assert restored == decision

    moved = decision.to_dict()
    moved["item_id"] = imported.decisions[1].item_id
    with pytest.raises(NativeReviewPacketError, match="decision_id: lineage_mismatch"):
        NativeReviewDecision.from_dict(moved)

    with_identity = decision.to_dict()
    with_identity["reviewer_name"] = MARKER_REVIEWER
    with pytest.raises(NativeReviewPacketError) as caught:
        NativeReviewDecision.from_dict(with_identity)
    assert str(caught.value) == "decision_record: reviewer_identity_forbidden"
    _assert_value_free(caught.value, MARKER_REVIEWER)


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (b'{"a": 1, "a": 2}', "submission: invalid_json"),
        (b'{"a": NaN}', "submission: invalid_json"),
        (b"\xff\xfe", "submission: invalid_encoding"),
        (b"[1, 2]", "submission: invalid_object"),
        (b" " * (8 * 1024 * 1024 + 1), "submission: byte_limit_exceeded"),
        (f'{{"reviewer_note": "{MARKER_NAME}"'.encode(), "submission: invalid_json"),
    ],
)
def test_submission_json_is_parsed_strictly(payload, message) -> None:
    packet, manifest = _build()

    with pytest.raises(NativeReviewPacketError) as caught:
        import_native_review_decisions(payload, packet=packet, manifest=manifest)

    assert str(caught.value) == message
    assert caught.value.__context__ is None
    _assert_value_free(caught.value, MARKER_NAME)


def test_task_vocabularies_are_validated() -> None:
    with pytest.raises(NativeReviewPacketError, match="required_for_corrections"):
        NativeReviewTask(
            task_code="label_check",
            instructions="Confirm labels.",
            decision_codes=("accept", "revise"),
            correction_decision_code="revise",
        )
    with pytest.raises(NativeReviewPacketError, match="correction_decision_code"):
        NativeReviewTask(
            task_code="label_check",
            instructions="Confirm labels.",
            label_codes=("CONDITION",),
            correction_decision_code="revise",
        )
    with pytest.raises(NativeReviewPacketError, match="decision_codes: invalid_count"):
        NativeReviewTask(
            task_code="label_check",
            instructions="Confirm labels.",
            decision_codes=("accept",),
        )
    with pytest.raises(NativeReviewPacketError, match="instructions: control_char"):
        NativeReviewTask(task_code="label_check", instructions="Confirm\x00labels.")


def test_reprs_do_not_expose_review_text() -> None:
    packet, _ = _build()
    item = _items()[0]

    assert MARKER_NAME not in repr(item)
    assert MARKER_NAME not in repr(packet.items[0])
    assert MARKER_NAME not in repr(packet)


def test_build_and_import_make_no_network_calls(monkeypatch) -> None:
    def _blocked(*_args, **_kwargs):
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket, "create_connection", _blocked)
    monkeypatch.setattr(socket.socket, "connect", _blocked)
    monkeypatch.setattr(socket, "getaddrinfo", _blocked)

    packet, manifest = _build()
    imported = import_native_review_decisions(
        _filled_submission(packet), packet=packet, manifest=manifest
    )
    assert imported.completeness.complete


def test_submission_template_matches_the_packet() -> None:
    packet, _ = _build()
    template = packet.submission_template()

    assert template["schema_version"] == NATIVE_REVIEW_SUBMISSION_SCHEMA_VERSION
    assert template["packet_ref"] == packet.packet_ref
    assert template["packet_digest"] == packet.packet_digest
    assert template["reviewer_role"] is None
    assert [row["item_id"] for row in template["decisions"]] == [
        item.item_id for item in packet.items
    ]
    assert all(row["decision"] is None for row in template["decisions"])
