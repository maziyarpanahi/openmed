"""Synthetic intake conversations and their causal turn-level gold labels."""

from __future__ import annotations

import hashlib
import json
import socket
from collections import Counter
from dataclasses import replace

import pytest

from openmed.core.labels import CANONICAL_LABELS
from openmed.core.schemas.span import OpenMedSpan, hmac_text_hash
from openmed.multimodal.chatlog_jsonl import (
    iter_redacted_chatlog_jsonl_with_summary,
)
from openmed.training.synthetic.intake_dialogue import (
    ASSERTION_STATUSES,
    CONSENT_ACTIONS,
    INTAKE_CLINICAL_LABELS,
    INTAKE_DIALOGUE_SCHEMA_VERSION,
    INTAKE_PHI_LABELS,
    INTAKE_SCENARIOS,
    SPEAKER_ROLES,
    STREAM_TURN_SEPARATOR,
    SYNTHETIC_INTAKE_DIALOGUE_SOURCE,
    IntakeConversation,
    IntakeSpan,
    build_intake_dialogue_manifest,
    generate_intake_conversations,
)

SEED = 2847


@pytest.fixture(scope="module")
def conversations() -> tuple[IntakeConversation, ...]:
    return generate_intake_conversations(12, seed=SEED)


def _by_scenario(conversations, scenario):
    return [item for item in conversations if item.scenario == scenario]


def _surface_values(conversation, *, min_length=1):
    return {
        turn.text[span.start : span.end]
        for turn in conversation.turns
        for span in turn.spans
        if span.end - span.start >= min_length
    }


def _consent_actions(conversation):
    return [
        (turn.index, event.attributes["action"])
        for turn in conversation.turns
        for event in turn.events
        if event.type == "consent"
    ]


def test_scenarios_rotate_and_cover_every_speaker_role(conversations) -> None:
    assert [item.scenario for item in conversations[:3]] == list(INTAKE_SCENARIOS)
    for conversation in _by_scenario(conversations, "patient_only"):
        assert conversation.participants == ("patient", "clinician")
    for conversation in _by_scenario(conversations, "caregiver"):
        assert conversation.participants == ("patient", "caregiver", "clinician")
    for conversation in _by_scenario(conversations, "interpreter"):
        assert conversation.participants == ("patient", "clinician", "interpreter")
    assert {
        role for conversation in conversations for role in conversation.participants
    } == set(SPEAKER_ROLES)


def test_every_conversation_has_hesitation_repair_correction_and_consent(
    conversations,
) -> None:
    batch_events: Counter[str] = Counter()
    batch_revisions: Counter[str] = Counter()
    for conversation in conversations:
        events = {event.type for turn in conversation.turns for event in turn.events}
        assert {
            "hesitation",
            "uncertainty",
            "self_repair",
            "correction",
            "consent",
        } <= events
        actions = {action for _turn, action in _consent_actions(conversation)}
        assert "request" in actions
        assert actions & {"grant", "decline"}

        supersessions = conversation.supersessions()
        revisions = {item["revision"] for item in supersessions}
        assert {"self_repair", "correction"} <= revisions
        # A later turn supersedes a fact stated in an earlier turn.
        assert any(
            item["revision"] == "correction"
            and item["turn_index"] < item["superseded_in_turn"]
            for item in supersessions
        )
        # Self-repairs replace a value inside the same utterance.
        assert any(
            item["revision"] == "self_repair"
            and item["turn_index"] == item["superseded_in_turn"]
            for item in supersessions
        )
        assert {
            span.assertion for turn in conversation.turns for span in turn.spans
        } >= {"affirmed", "uncertain"}
        batch_events.update(events)
        batch_revisions.update(revisions)

    assert batch_events["interruption"] > 0
    assert batch_revisions["completion"] > 0
    assert {
        span.assertion
        for conversation in conversations
        for turn in conversation.turns
        for span in turn.spans
    } == set(ASSERTION_STATUSES)


def test_gold_spans_have_exact_turn_and_stream_offsets(conversations) -> None:
    for conversation in conversations:
        stream = conversation.stream_text
        assert stream == "".join(conversation.iter_stream_chunks())
        for turn in conversation.turns:
            assert (
                stream[turn.stream_offset : turn.stream_offset + len(turn.text)]
                == turn.text
            )
            assert stream[turn.stream_offset + len(turn.text)] == (
                STREAM_TURN_SEPARATOR
            )
            record = turn.to_dict()
            gold = record["gold"]
            for kind, spans in (
                ("phi", gold["phi_spans"]),
                ("clinical", gold["clinical_spans"]),
            ):
                for span in spans:
                    surface = record["content"][span["start"] : span["end"]]
                    assert surface
                    assert surface == surface.strip()
                    assert (
                        hashlib.sha256(surface.encode()).hexdigest()
                        == (span["text_sha256"])
                    )
                    assert stream[span["stream_start"] : span["stream_end"]] == (
                        surface
                    )
                    assert span["kind"] == kind
                    assert span["label"] in CANONICAL_LABELS
                    allowed = (
                        INTAKE_PHI_LABELS if kind == "phi" else INTAKE_CLINICAL_LABELS
                    )
                    assert span["label"] in allowed
                    assert span["slot"] == (
                        f"{span['subject_role']}.{span['attribute']}"
                    )
            for event in gold["events"]:
                assert 0 <= event["start"] < event["end"] <= len(turn.text)
                assert (
                    stream[event["stream_start"] : event["stream_end"]]
                    == (turn.text[event["start"] : event["end"]])
                )


def test_labels_are_causal_and_corrections_supersede_earlier_facts(
    conversations,
) -> None:
    for conversation in conversations:
        seen: dict[str, IntakeSpan] = {}
        for turn in conversation.turns:
            for span in turn.spans:
                for reference in (*span.conflicts_with, span.supersedes):
                    if reference is not None:
                        assert reference in seen
                        assert seen[reference].slot == span.slot
                seen[span.span_id] = span
            for event in turn.events:
                for key in ("span_id", "supersedes", "reparandum", "repair"):
                    if key in event.attributes:
                        assert event.attributes[key] in seen
                if "interrupts_turn" in event.attributes:
                    assert event.attributes["interrupts_turn"] == turn.index - 1

        final = conversation.final_facts()
        superseded = {item["span_id"] for item in conversation.supersessions()}
        assert superseded
        assert not superseded & {fact["span_id"] for fact in final.values()}
        assert all(not fact["partial"] for fact in final.values())

    for conversation in _by_scenario(conversations, "patient_only"):
        corrected_phone = [
            item
            for item in conversation.supersessions()
            if item["slot"] == "patient.phone"
        ]
        assert len(corrected_phone) == 1
        assert corrected_phone[0]["revision"] == "correction"
        assert (
            conversation.final_facts()["patient.phone"]["span_id"]
            == (corrected_phone[0]["superseded_by"])
        )


def test_streaming_fact_updates_replay_to_the_fact_timeline(conversations) -> None:
    for conversation in conversations:
        state: dict[str, str] = {}
        records = list(conversation.iter_stream_records())
        assert [record["turn_index"] for record in records] == list(
            range(len(conversation.turns))
        )
        for record, snapshot in zip(records, conversation.fact_timeline(), strict=True):
            assert record["schema_version"] == INTAKE_DIALOGUE_SCHEMA_VERSION
            assert record["conversation_id"] == conversation.conversation_id
            assert record["provenance"]["synthetic"] is True
            assert record["provenance"]["source"] == SYNTHETIC_INTAKE_DIALOGUE_SOURCE
            assert record["provenance"]["contains_real_phi"] is False
            state.update(record["gold"]["fact_updates"])
            assert state == snapshot
        assert {
            slot: fact["span_id"] for slot, fact in conversation.final_facts().items()
        } == state


def test_speaker_specific_facts_keep_subject_separate_from_speaker(
    conversations,
) -> None:
    caregiver = _by_scenario(conversations, "caregiver")[0]
    spoken = [
        (turn.speaker_role, span) for turn in caregiver.turns for span in turn.spans
    ]
    assert any(
        role == "caregiver"
        and span.subject_role == "patient"
        and span.label == "DATE_OF_BIRTH"
        and span.assertion == "uncertain"
        for role, span in spoken
    )
    assert any(
        role == "patient"
        and span.subject_role == "caregiver"
        and span.label == "PERSON"
        for role, span in spoken
    )
    final = caregiver.final_facts()
    assert final["patient.phone"]["span_id"] != final["caregiver.phone"]["span_id"]
    assert "caregiver.email" in final and "patient.email" not in final

    interpreter = _by_scenario(conversations, "interpreter")[0]
    interpreted = [
        span
        for turn in interpreter.turns
        if turn.speaker_role == "interpreter"
        for span in turn.spans
    ]
    # First-person interpreting: "my phone number" is the patient's.
    assert {span.subject_role for span in interpreted} == {"patient", "interpreter"}
    assert {
        span.attribute for span in interpreted if span.subject_role == "interpreter"
    } == {"name", "badge_number"}
    assert any(
        span.subject_role == "patient" and span.label == "PHONE" for span in interpreted
    )


def test_consent_state_gates_capture_turn_by_turn(conversations) -> None:
    observed: Counter[str] = Counter()
    for conversation in conversations:
        turns = conversation.turns
        actions = _consent_actions(conversation)
        observed.update(action for _turn, action in actions)
        assert actions[0][1] == "request"
        decision_turn, decision = next(
            (index, action)
            for index, action in actions
            if action in {"grant", "decline"}
        )
        assert all(
            turn.consent_state == "pending" and not turn.capture_permitted
            for turn in turns[: decision_turn + 1]
        )
        if decision == "decline":
            assert not any(turn.capture_permitted for turn in turns)
        pauses = [index for index, action in actions if action == "pause"]
        resumes = [index for index, action in actions if action == "resume"]
        assert len(pauses) == len(resumes)
        for pause_turn, resume_turn in zip(pauses, resumes, strict=True):
            assert turns[pause_turn].capture_permitted
            window = turns[pause_turn + 1 : resume_turn + 1]
            assert window
            assert all(turn.consent_state == "paused" for turn in window)
            # Identifiers spoken off the record are still redaction targets.
            assert any(turn.phi_spans for turn in window)
            assert turns[resume_turn + 1].capture_permitted
        for record, turn in zip(conversation.iter_stream_records(), turns):
            assert record["gold"]["capture_permitted"] is turn.capture_permitted
            assert record["gold"]["consent_state"] == turn.consent_state

    assert set(observed) == set(CONSENT_ACTIONS)


def test_speaker_disagreement_is_flagged_and_never_auto_resolved(
    conversations,
) -> None:
    for conversation in _by_scenario(conversations, "caregiver"):
        items = conversation.adjudication_items()
        assert len(items) == 1
        located = {
            span.span_id: (turn, span)
            for turn in conversation.turns
            for span in turn.spans
        }
        flagged_turn, flagged = located[items[0]["span_id"]]
        earlier_turn, earlier = located[items[0]["conflicts_with"][0]]
        assert flagged_turn.speaker_role == "patient"
        assert earlier_turn.speaker_role == "caregiver"
        assert (flagged.assertion, earlier.assertion) == ("negated", "affirmed")
        assert flagged.supersedes is None
        assert conversation.final_facts()[flagged.slot]["needs_adjudication"]

    for conversation in _by_scenario(conversations, "patient_only"):
        assert conversation.adjudication_items() == ()


def test_generation_is_deterministic_and_offline(monkeypatch) -> None:
    def no_network(*_args, **_kwargs):
        raise AssertionError("intake dialogue generation attempted network access")

    monkeypatch.setattr(socket, "socket", no_network)
    monkeypatch.setattr(socket, "create_connection", no_network)

    first = generate_intake_conversations(9, seed=11)
    second = generate_intake_conversations(9, seed=11)
    other = generate_intake_conversations(9, seed=12)

    assert first == second
    assert first != other
    assert [json.dumps(item.to_dict(), sort_keys=True) for item in first] == [
        json.dumps(item.to_dict(), sort_keys=True) for item in second
    ]
    assert build_intake_dialogue_manifest(first) == build_intake_dialogue_manifest(
        second
    )
    assert (
        build_intake_dialogue_manifest(first)["dataset_sha256"]
        != build_intake_dialogue_manifest(other)["dataset_sha256"]
    )
    assert generate_intake_conversations(0, seed=11) == ()
    # A conversation depends only on seed, index and scenario, never on the
    # rest of the batch.
    only_interpreter = generate_intake_conversations(
        3, seed=11, scenarios=("interpreter",)
    )
    assert [item.scenario for item in only_interpreter] == ["interpreter"] * 3
    assert only_interpreter[2] == first[2]


def test_reports_manifest_and_exported_gold_never_repeat_values(
    conversations,
) -> None:
    manifest = build_intake_dialogue_manifest(conversations)
    assert manifest["synthetic"] is True
    assert manifest["contains_real_phi"] is False
    assert manifest["restricted_data"] is False
    assert manifest["conversation_count"] == len(conversations)
    assert manifest["seeds"] == [SEED]
    serialized_manifest = json.dumps(manifest)

    for conversation in conversations:
        # Short values such as two-digit ages could collide with hex digests,
        # so only check surfaces long enough to be distinctive.
        values = _surface_values(conversation, min_length=6)
        assert values
        payloads = [json.dumps(conversation.report()), serialized_manifest]
        payloads.extend(
            json.dumps(record["gold"]) for record in conversation.iter_stream_records()
        )
        exported = conversation.to_dict()
        payloads.append(
            json.dumps(
                {
                    key: value
                    for key, value in exported.items()
                    if key != "turns"  # turn content is the synthetic input itself
                }
            )
        )
        for payload in payloads:
            assert not [value for value in values if value in payload]


def test_turn_records_fit_streaming_chatlog_redaction(conversations) -> None:
    conversation = conversations[1]
    records = list(conversation.iter_stream_records())

    def oracle_mask(record):
        content = record["content"]
        for span in sorted(
            record["gold"]["phi_spans"], key=lambda item: -item["start"]
        ):
            content = (
                content[: span["start"]] + f"[{span['label']}]" + content[span["end"] :]
            )
        return content

    masked = {record["content"]: oracle_mask(record) for record in records}
    stream = iter_redacted_chatlog_jsonl_with_summary(
        (json.dumps(record) + "\n" for record in records),
        text_redactor=masked.__getitem__,
    )
    redacted = [json.loads(line) for line in stream]

    assert stream.summary.message_count == len(records)
    assert stream.summary.schema_counts == {"turn": len(records)}
    for original, output, turn in zip(
        records, redacted, conversation.turns, strict=True
    ):
        assert output["gold"] == original["gold"]
        assert output["speaker_role"] == original["speaker_role"]
        for span in turn.phi_spans:
            assert turn.text[span.start : span.end] not in output["content"]


def test_gold_openmed_spans_use_stream_offsets_and_keyed_hashes(
    conversations,
) -> None:
    conversation = conversations[2]
    secret = "intake-unit-test-key"
    spans = conversation.gold_openmed_spans(secret=secret)
    stream = conversation.stream_text

    expected = [span for turn in conversation.turns for span in turn.phi_spans]
    assert len(spans) == len(expected)
    for gold, span in zip(spans, expected, strict=True):
        assert isinstance(gold, OpenMedSpan)
        assert gold.doc_id == conversation.conversation_id
        assert gold.canonical_label == span.label
        assert gold.metadata["span_id"] == span.span_id
        assert gold.text_hash == hmac_text_hash(stream[gold.start : gold.end], secret)
        assert stream[gold.start : gold.end] not in json.dumps(gold.metadata)
    with pytest.raises((TypeError, ValueError)):
        conversation.gold_openmed_spans(secret="")


def test_assertion_vocabulary_matches_clinical_context_axes() -> None:
    from openmed.clinical.context import AFFIRMED, NEGATED, UNCERTAIN

    assert {AFFIRMED, NEGATED, UNCERTAIN} == set(ASSERTION_STATUSES)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"count": -1},
        {"count": True},
        {"count": 1, "seed": -1},
        {"count": 1, "seed": False},
        {"count": 1, "scenarios": "patient_only"},
        {"count": 1, "scenarios": ()},
        {"count": 1, "scenarios": ("unknown",)},
    ],
)
def test_invalid_arguments_raise_value_errors(kwargs) -> None:
    count = kwargs.pop("count")
    with pytest.raises(ValueError):
        generate_intake_conversations(count, **kwargs)


def test_manifest_rejects_non_conversation_batches() -> None:
    with pytest.raises(ValueError):
        build_intake_dialogue_manifest("not a batch")
    with pytest.raises(ValueError):
        build_intake_dialogue_manifest([{"turns": []}])


def test_tampered_gold_fails_closed_without_echoing_values(conversations) -> None:
    conversation = conversations[0]
    turn = next(turn for turn in conversation.turns if turn.phi_spans)
    span = turn.phi_spans[0]
    value = turn.text[span.start : span.end]
    tampered = turn.text[: span.start] + "x" * len(value) + turn.text[span.end :]

    with pytest.raises(ValueError, match="digest") as error:
        replace(turn, text=tampered)
    assert value not in str(error.value)

    with pytest.raises(ValueError, match="adjudication"):
        replace(span, needs_adjudication=True)
    with pytest.raises(ValueError, match="paired"):
        replace(span, revision="correction")

    last = conversation.turns[-1]
    forward_link = replace(last.spans[0], revision="correction", supersedes="t99.s00")
    with pytest.raises(ValueError, match="earlier span"):
        replace(
            conversation,
            turns=conversation.turns[:-1]
            + (replace(last, spans=(forward_link,) + last.spans[1:]),),
        )

    with pytest.raises(ValueError, match="consent_state"):
        replace(
            conversation,
            turns=(replace(conversation.turns[0], consent_state="granted"),)
            + conversation.turns[1:],
        )
