"""Tests for the non-compensable v3.3 private-learning release packet."""

from __future__ import annotations

import copy
import hashlib
import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from openmed.eval.suites.private_learning_release import (
    ITEM_REASON_CODES,
    MOVED_REASON_CODES,
    NOT_READY,
    PRIVATE_LEARNING_RELEASE_CATALOGS,
    PRIVATE_LEARNING_RELEASE_EVIDENCE_SCHEMA_VERSION,
    PRIVATE_LEARNING_RELEASE_MANIFEST_SCHEMA_VERSION,
    PRIVATE_LEARNING_RELEASE_PACKET_SCHEMA_VERSION,
    READY,
    RELEASE_REASON_CODES,
    PrivateLearningReleaseError,
    PrivateLearningReleasePacket,
    evaluate_private_learning_release,
    sign_private_learning_evidence,
)
from tests.fixtures.private_learning_forbidden import FORBIDDEN_FIELD_CASES

_COMMIT = "a" * 40
_OTHER_COMMIT = "b" * 40
_EVIDENCE_KEY = b"synthetic-evidence-key-0123456789abcdef"
_SIGNING_KEY = b"synthetic-packet-key-0123456789abcdefgh"
_NOW = datetime(2026, 11, 1, 12, 0, 0, tzinfo=timezone.utc)
_GENERATED = "2026-11-01T11:00:00Z"
_CATALOG = PRIVATE_LEARNING_RELEASE_CATALOGS["v3.3"]
_P1_ITEM = next(spec.item_id for spec in _CATALOG if spec.priority == "P1")
_P0_ITEM = next(spec.item_id for spec in _CATALOG if spec.priority == "P0")
_PRIVACY_ITEM = next(spec.item_id for spec in _CATALOG if spec.privacy_invariant)
_PLAIN_P0_ITEM = next(
    spec.item_id
    for spec in _CATALOG
    if spec.priority == "P0" and not spec.privacy_invariant
)


def _digest(label: str) -> str:
    return "sha256:" + hashlib.sha256(label.encode("utf-8")).hexdigest()


def _schema(item_id: str) -> str:
    return f"openmed.release.{item_id}.v1"


def _clock() -> datetime:
    return _NOW


def _manifest() -> dict:
    return {
        "schema_version": PRIVATE_LEARNING_RELEASE_MANIFEST_SCHEMA_VERSION,
        "release": {"release_id": "v3.3", "git_commit": _COMMIT},
        "policy": {"max_evidence_age_seconds": 86_400},
        "items": [
            {
                "item_id": spec.item_id,
                "priority": spec.priority,
                "privacy_invariant": spec.privacy_invariant,
                "non_compensable": True,
                "required_schema_version": _schema(spec.item_id),
                "required_evidence_digest": _digest(spec.item_id),
            }
            for spec in _CATALOG
        ],
        "moved_work": [
            {
                "work_ref": "gh-3877",
                "priority": "P2",
                "privacy_invariant": False,
                "target_milestone": "v4.3",
                "rationale_code": "deferred_by_scope_policy",
                "rationale_digest": _digest("rationale-3877"),
            }
        ],
    }


def _entry(item_id: str, **overrides) -> dict:
    entry = {
        "schema_version": PRIVATE_LEARNING_RELEASE_EVIDENCE_SCHEMA_VERSION,
        "item_id": item_id,
        "evidence_schema_version": _schema(item_id),
        "evidence_digest": _digest(item_id),
        "source_commit": _COMMIT,
        "generated_at": _GENERATED,
        "status": "passed",
    }
    entry.update(overrides)
    return sign_private_learning_evidence(entry, _EVIDENCE_KEY)


def _evidence() -> list[dict]:
    return [_entry(spec.item_id) for spec in _CATALOG]


def _evaluate(manifest=None, evidence=None, **kwargs) -> PrivateLearningReleasePacket:
    options = {
        "source_commit": _COMMIT,
        "evidence_key": _EVIDENCE_KEY,
        "signing_key": _SIGNING_KEY,
        "clock": _clock,
    }
    options.update(kwargs)
    return evaluate_private_learning_release(
        _manifest() if manifest is None else manifest,
        _evidence() if evidence is None else evidence,
        **options,
    )


def _item(packet: PrivateLearningReleasePacket, item_id: str) -> dict:
    return next(row for row in packet.to_dict()["items"] if row["item_id"] == item_id)


def _replace_entry(evidence: list[dict], item_id: str, entry: dict) -> list[dict]:
    return [entry if row["item_id"] == item_id else row for row in evidence]


def _assert_only_item_fails(packet, item_id: str, codes: set[str]) -> None:
    payload = packet.to_dict()
    assert packet.decision == NOT_READY
    assert payload["reason_codes"] == []
    for row in payload["items"]:
        if row["item_id"] == item_id:
            assert row["status"] == "failed"
            assert set(row["reason_codes"]) == codes
        else:
            assert row["status"] == "passed", row["item_id"]
            assert row["reason_codes"] == []


# Ready path and offline verification -------------------------------------


def test_complete_synthetic_manifest_yields_ready_packet_that_verifies() -> None:
    packet = _evaluate()

    payload = packet.to_dict()
    assert packet.decision == READY
    assert payload["schema_version"] == PRIVATE_LEARNING_RELEASE_PACKET_SCHEMA_VERSION
    assert payload["release"] == {
        "evaluated_at": "2026-11-01T12:00:00Z",
        "git_commit": _COMMIT,
        "release_id": "v3.3",
    }
    assert payload["source_commit"] == _COMMIT
    assert payload["release_authority"] == "not_granted"
    assert [row["item_id"] for row in payload["items"]] == [
        spec.item_id for spec in _CATALOG
    ]
    assert all(row["status"] == "passed" for row in payload["items"])
    assert payload["moved_work"][0]["accepted"] is True
    assert packet.verify(_SIGNING_KEY)
    assert not packet.verify(b"x" * 40)


def test_packet_round_trips_through_json_and_disk_offline(tmp_path: Path) -> None:
    packet = _evaluate()
    path = tmp_path / "packet.json"
    path.write_text(packet.to_json(), encoding="utf-8")

    restored = PrivateLearningReleasePacket.from_json(path.read_text("utf-8"))

    assert restored.to_json() == packet.to_json()
    assert restored.verify(_SIGNING_KEY)
    assert restored.packet_digest == packet.packet_digest


def test_packet_is_deterministic_and_ignores_evidence_order() -> None:
    first = _evaluate()
    second = _evaluate(evidence=list(reversed(_evidence())))

    assert first.to_json() == second.to_json()


def test_items_are_never_collapsed_into_one_score() -> None:
    payload = _evaluate().to_dict()

    assert len(payload["items"]) == len(_CATALOG)
    assert not {"score", "pass_rate", "overall_score"} & set(payload)
    for row in payload["items"]:
        assert set(row) == {
            "entry_digest",
            "item_id",
            "priority",
            "privacy_invariant",
            "reason_codes",
            "required_evidence_digest",
            "required_schema_version",
            "status",
        }


# Independent failures with stable codes ----------------------------------


def test_missing_entry_fails_only_that_item() -> None:
    evidence = [row for row in _evidence() if row["item_id"] != _P0_ITEM]

    _assert_only_item_fails(
        _evaluate(evidence=evidence), _P0_ITEM, {"evidence_missing"}
    )


def test_failed_entry_fails_only_that_item() -> None:
    evidence = _replace_entry(_evidence(), _P0_ITEM, _entry(_P0_ITEM, status="failed"))

    _assert_only_item_fails(_evaluate(evidence=evidence), _P0_ITEM, {"evidence_failed"})


def test_unsigned_entry_fails_only_that_item() -> None:
    unsigned = _entry(_P0_ITEM)
    unsigned["signature"] = None
    evidence = _replace_entry(_evidence(), _P0_ITEM, unsigned)

    _assert_only_item_fails(
        _evaluate(evidence=evidence), _P0_ITEM, {"evidence_unsigned"}
    )


def test_tampered_entry_fails_signature_and_digest_binding() -> None:
    tampered = _entry(_P0_ITEM)
    tampered["evidence_digest"] = _digest("tampered")
    evidence = _replace_entry(_evidence(), _P0_ITEM, tampered)

    _assert_only_item_fails(
        _evaluate(evidence=evidence),
        _P0_ITEM,
        {"evidence_signature_invalid", "evidence_digest_mismatch"},
    )


def test_resigned_entry_with_wrong_digest_fails_digest_binding_only() -> None:
    evidence = _replace_entry(
        _evidence(), _P0_ITEM, _entry(_P0_ITEM, evidence_digest=_digest("other"))
    )

    _assert_only_item_fails(
        _evaluate(evidence=evidence), _P0_ITEM, {"evidence_digest_mismatch"}
    )


def test_entry_signed_with_another_key_fails_signature() -> None:
    entry = _entry(_P0_ITEM)
    del entry["signature"]
    foreign = sign_private_learning_evidence(entry, b"f" * 40)
    evidence = _replace_entry(_evidence(), _P0_ITEM, foreign)

    _assert_only_item_fails(
        _evaluate(evidence=evidence), _P0_ITEM, {"evidence_signature_invalid"}
    )


def test_wrong_commit_entry_fails_only_that_item() -> None:
    evidence = _replace_entry(
        _evidence(), _P0_ITEM, _entry(_P0_ITEM, source_commit=_OTHER_COMMIT)
    )

    _assert_only_item_fails(
        _evaluate(evidence=evidence), _P0_ITEM, {"evidence_commit_mismatch"}
    )


def test_stale_entry_fails_only_that_item() -> None:
    evidence = _replace_entry(
        _evidence(), _P0_ITEM, _entry(_P0_ITEM, generated_at="2026-10-01T00:00:00Z")
    )

    _assert_only_item_fails(_evaluate(evidence=evidence), _P0_ITEM, {"evidence_stale"})


def test_future_entry_fails_only_that_item() -> None:
    evidence = _replace_entry(
        _evidence(), _P0_ITEM, _entry(_P0_ITEM, generated_at="2026-11-02T00:00:00Z")
    )

    _assert_only_item_fails(
        _evaluate(evidence=evidence), _P0_ITEM, {"evidence_from_future"}
    )


def test_schema_version_mismatch_fails_only_that_item() -> None:
    evidence = _replace_entry(
        _evidence(),
        _P0_ITEM,
        _entry(_P0_ITEM, evidence_schema_version="openmed.release.other.v2"),
    )

    _assert_only_item_fails(
        _evaluate(evidence=evidence), _P0_ITEM, {"evidence_schema_mismatch"}
    )


def test_duplicate_entries_are_all_excluded() -> None:
    evidence = [*_evidence(), _entry(_P0_ITEM)]

    _assert_only_item_fails(
        _evaluate(evidence=evidence), _P0_ITEM, {"evidence_duplicate"}
    )


def test_malformed_entry_for_known_item_is_invalid_not_raised() -> None:
    entry = _entry(_P0_ITEM)
    entry["status"] = "mostly_passed"
    evidence = _replace_entry(_evidence(), _P0_ITEM, entry)

    _assert_only_item_fails(
        _evaluate(evidence=evidence), _P0_ITEM, {"evidence_invalid"}
    )


def test_unknown_or_unattributable_entries_block_at_release_level() -> None:
    unknown = {**_entry(_P0_ITEM), "item_id": "not_a_catalog_item"}
    packet = _evaluate(evidence=[*_evidence(), unknown, "not-an-object"])

    payload = packet.to_dict()
    assert packet.decision == NOT_READY
    assert payload["reason_codes"] == ["evidence_invalid", "evidence_unexpected_item"]
    assert all(row["status"] == "passed" for row in payload["items"])


def test_multiple_failures_are_reported_together_without_masking() -> None:
    evidence = _replace_entry(
        _evidence(),
        _P0_ITEM,
        _entry(
            _P0_ITEM,
            status="failed",
            source_commit=_OTHER_COMMIT,
            generated_at="2026-10-01T00:00:00Z",
        ),
    )
    evidence = [row for row in evidence if row["item_id"] != _PRIVACY_ITEM]

    packet = _evaluate(evidence=evidence)

    assert set(_item(packet, _P0_ITEM)["reason_codes"]) == {
        "evidence_commit_mismatch",
        "evidence_failed",
        "evidence_stale",
    }
    assert _item(packet, _PRIVACY_ITEM)["reason_codes"] == ["evidence_missing"]
    assert [row["item_id"] for row in packet.failing_items()] == [
        spec.item_id for spec in _CATALOG if spec.item_id in {_P0_ITEM, _PRIVACY_ITEM}
    ]


def test_release_commit_mismatch_blocks_even_when_items_pass() -> None:
    packet = _evaluate(source_commit=_OTHER_COMMIT)

    payload = packet.to_dict()
    assert packet.decision == NOT_READY
    assert payload["reason_codes"] == ["release_commit_mismatch"]
    assert payload["source_commit"] == _OTHER_COMMIT
    assert payload["release"]["git_commit"] == _COMMIT
    assert packet.verify(_SIGNING_KEY)


def test_reordered_manifest_items_fail_with_stable_code() -> None:
    manifest = _manifest()
    manifest["items"][0], manifest["items"][1] = (
        manifest["items"][1],
        manifest["items"][0],
    )

    packet = _evaluate(manifest=manifest)

    assert packet.decision == NOT_READY
    assert packet.to_dict()["reason_codes"] == ["manifest_item_order_invalid"]
    assert all(row["status"] == "passed" for row in packet.to_dict()["items"])


def test_undeclared_item_cannot_be_dropped_from_the_manifest() -> None:
    manifest = _manifest()
    manifest["items"] = [
        row for row in manifest["items"] if row["item_id"] != _PLAIN_P0_ITEM
    ]

    packet = _evaluate(manifest=manifest)

    _assert_only_item_fails(packet, _PLAIN_P0_ITEM, {"item_undeclared"})


def test_downgraded_or_compensable_item_contract_fails() -> None:
    manifest = _manifest()
    for row in manifest["items"]:
        if row["item_id"] == _PRIVACY_ITEM:
            row["privacy_invariant"] = False
            row["priority"] = "P2"
            row["non_compensable"] = False

    packet = _evaluate(manifest=manifest)

    _assert_only_item_fails(
        packet, _PRIVACY_ITEM, {"item_contract_mismatch", "item_compensable"}
    )


def test_duplicate_and_unknown_manifest_items_block() -> None:
    manifest = _manifest()
    duplicate = copy.deepcopy(manifest["items"][-1])
    unknown = {**copy.deepcopy(manifest["items"][-1]), "item_id": "unknown_item"}
    manifest["items"] = [*manifest["items"], duplicate, unknown]

    packet = _evaluate(manifest=manifest)

    payload = packet.to_dict()
    assert set(payload["reason_codes"]) == {
        "manifest_item_duplicate",
        "manifest_item_unknown",
    }
    assert _item(packet, duplicate["item_id"])["reason_codes"] == [
        "item_declared_twice"
    ]


# Tampering and reordering of stored packets -------------------------------


def test_tampered_packet_fails_offline_verification() -> None:
    payload = _evaluate().to_dict()
    payload["items"][0]["reason_codes"] = []
    payload["items"][0]["required_evidence_digest"] = _digest("swapped")

    restored = PrivateLearningReleasePacket.from_dict(payload)

    assert not restored.verify(_SIGNING_KEY)


def test_not_ready_packet_cannot_be_flipped_to_ready() -> None:
    packet = _evaluate(evidence=[row for row in _evidence()[1:]])
    payload = packet.to_dict()
    payload["items"][0]["status"] = "passed"
    payload["items"][0]["reason_codes"] = []
    payload["items"][0]["entry_digest"] = _digest("forged-entry")
    payload["decision"] = READY

    restored = PrivateLearningReleasePacket.from_dict(payload)

    assert not restored.verify(_SIGNING_KEY)


def test_reordered_packet_items_are_rejected() -> None:
    payload = _evaluate().to_dict()
    payload["items"][0], payload["items"][1] = payload["items"][1], payload["items"][0]

    with pytest.raises(PrivateLearningReleaseError, match="^packet_invalid$"):
        PrivateLearningReleasePacket.from_dict(payload)


def test_inconsistent_packet_decision_is_rejected() -> None:
    payload = _evaluate(evidence=_evidence()[1:]).to_dict()
    payload["decision"] = READY

    with pytest.raises(PrivateLearningReleaseError, match="^packet_invalid$"):
        PrivateLearningReleasePacket.from_dict(payload)


def test_unsigned_packet_is_rejected() -> None:
    payload = _evaluate().to_dict()
    payload["signature"] = None

    with pytest.raises(PrivateLearningReleaseError, match="^packet_invalid$"):
        PrivateLearningReleasePacket.from_dict(payload)


def test_packet_json_rejects_duplicate_keys_and_non_finite_numbers() -> None:
    text = _evaluate().to_json()
    duplicated = text[:-1] + ',"decision":"READY"}'

    with pytest.raises(PrivateLearningReleaseError, match="^packet_invalid$"):
        PrivateLearningReleasePacket.from_json(duplicated)
    with pytest.raises(PrivateLearningReleaseError, match="^packet_invalid$"):
        PrivateLearningReleasePacket.from_json('{"x": NaN}')


# Moved work --------------------------------------------------------------


def _moved(**overrides) -> dict:
    row = {
        "work_ref": _P1_ITEM,
        "priority": "P1",
        "privacy_invariant": False,
        "target_milestone": "v4.2",
        "rationale_code": "blocked_by_external_dependency",
        "rationale_digest": _digest("rationale"),
    }
    row.update(overrides)
    return row


def _manifest_moving(*rows: dict, drop: tuple[str, ...] = ()) -> dict:
    manifest = _manifest()
    manifest["items"] = [row for row in manifest["items"] if row["item_id"] not in drop]
    manifest["moved_work"] = [*manifest["moved_work"], *rows]
    return manifest


def test_moved_p1_item_with_rationale_is_accepted() -> None:
    manifest = _manifest_moving(_moved(), drop=(_P1_ITEM,))
    evidence = [row for row in _evidence() if row["item_id"] != _P1_ITEM]

    packet = _evaluate(manifest=manifest, evidence=evidence)

    assert packet.decision == READY
    assert _item(packet, _P1_ITEM)["status"] == "moved"
    moved = next(
        row for row in packet.to_dict()["moved_work"] if row["work_ref"] == _P1_ITEM
    )
    assert moved["accepted"] is True
    assert moved["rationale_digest"] == _digest("rationale")
    assert packet.verify(_SIGNING_KEY)


@pytest.mark.parametrize(
    "overrides",
    [{"rationale_code": None}, {"rationale_digest": None}],
)
def test_moved_work_requires_recorded_rationale(overrides: dict) -> None:
    manifest = _manifest_moving(_moved(**overrides), drop=(_P1_ITEM,))
    evidence = [row for row in _evidence() if row["item_id"] != _P1_ITEM]

    packet = _evaluate(manifest=manifest, evidence=evidence)

    moved = next(
        row for row in packet.to_dict()["moved_work"] if row["work_ref"] == _P1_ITEM
    )
    assert packet.decision == NOT_READY
    assert moved["accepted"] is False
    assert moved["reason_codes"] == ["moved_rationale_missing"]
    assert "moved_work_refused" in packet.to_dict()["reason_codes"]
    # A refused move keeps the item required; it is not silently dropped.
    assert _item(packet, _P1_ITEM)["status"] == "failed"
    assert set(_item(packet, _P1_ITEM)["reason_codes"]) == {
        "evidence_missing",
        "item_move_refused",
        "item_undeclared",
    }


def test_unknown_rationale_code_and_non_later_target_are_refused() -> None:
    manifest = _manifest_moving(
        _moved(rationale_code="because", target_milestone="v3.3"),
        drop=(_P1_ITEM,),
    )
    evidence = [row for row in _evidence() if row["item_id"] != _P1_ITEM]

    packet = _evaluate(manifest=manifest, evidence=evidence)

    moved = next(
        row for row in packet.to_dict()["moved_work"] if row["work_ref"] == _P1_ITEM
    )
    assert moved["reason_codes"] == [
        "moved_rationale_invalid",
        "moved_target_not_later",
    ]


@pytest.mark.parametrize("item_id", [_PLAIN_P0_ITEM, _PRIVACY_ITEM])
def test_moving_p0_or_privacy_invariant_item_cannot_remove_it(item_id: str) -> None:
    spec = next(row for row in _CATALOG if row.item_id == item_id)
    manifest = _manifest_moving(
        _moved(
            work_ref=item_id,
            priority=spec.priority,
            privacy_invariant=spec.privacy_invariant,
        ),
        drop=(item_id,),
    )
    evidence = [row for row in _evidence() if row["item_id"] != item_id]

    packet = _evaluate(manifest=manifest, evidence=evidence)

    moved = next(
        row for row in packet.to_dict()["moved_work"] if row["work_ref"] == item_id
    )
    expected = {"moved_p0_refused"}
    if spec.privacy_invariant:
        expected.add("moved_privacy_invariant_refused")
    assert set(moved["reason_codes"]) == expected
    assert packet.decision == NOT_READY
    row = _item(packet, item_id)
    assert row["status"] == "failed"
    assert "item_move_refused" in row["reason_codes"]


def test_moving_p0_by_declaring_it_p1_is_refused() -> None:
    manifest = _manifest_moving(
        _moved(work_ref=_PLAIN_P0_ITEM, priority="P1"), drop=(_PLAIN_P0_ITEM,)
    )
    evidence = [row for row in _evidence() if row["item_id"] != _PLAIN_P0_ITEM]

    packet = _evaluate(manifest=manifest, evidence=evidence)

    moved = next(
        row
        for row in packet.to_dict()["moved_work"]
        if row["work_ref"] == _PLAIN_P0_ITEM
    )
    assert set(moved["reason_codes"]) == {"moved_contract_mismatch", "moved_p0_refused"}
    assert _item(packet, _PLAIN_P0_ITEM)["status"] == "failed"


def test_workstream_p0_or_privacy_invariant_moves_are_refused() -> None:
    manifest = _manifest_moving(
        _moved(work_ref="gh-2821", priority="P0"),
        _moved(work_ref="gh-3875", priority="P2", privacy_invariant=True),
    )

    packet = _evaluate(manifest=manifest)

    rows = {row["work_ref"]: row for row in packet.to_dict()["moved_work"]}
    assert rows["gh-2821"]["reason_codes"] == ["moved_p0_refused"]
    assert rows["gh-3875"]["reason_codes"] == ["moved_privacy_invariant_refused"]
    assert packet.decision == NOT_READY


def test_moved_item_still_declared_and_duplicate_moves_are_refused() -> None:
    manifest = _manifest_moving(_moved(), _moved())

    packet = _evaluate(manifest=manifest)

    rows = [
        row for row in packet.to_dict()["moved_work"] if row["work_ref"] == _P1_ITEM
    ]
    assert len(rows) == 2
    for row in rows:
        assert set(row["reason_codes"]) == {
            "moved_duplicate",
            "moved_item_still_declared",
        }
    assert _item(packet, _P1_ITEM)["status"] == "passed"
    assert packet.decision == NOT_READY


# Sentinel scan -----------------------------------------------------------

_SENTINELS = (
    "SENTINEL-PATIENT-Jane-Roe",
    "SENTINEL-SITE-General-Hospital",
    "/Users/sentinel/private/evidence.json",
    "C:\\sentinel\\private",
    "MRN-5550199",
)
_ALLOWED_STRING = re.compile(
    r"(?:sha256:[0-9a-f]{64}|[0-9a-f]{40}|[0-9a-f]{64}"
    r"|[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z"
    r"|v[0-9]+\.[0-9]+|P[0-2]|gh-[0-9]{1,7}|READY|NOT_READY"
    r"|[a-z0-9_]+(?:\.[a-z0-9_]+)*)"
)


def _strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from _strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)


def _assert_value_free(text: str) -> None:
    for marker in (
        *_SENTINELS,
        *(case.marker for case in FORBIDDEN_FIELD_CASES if case.marker),
        _EVIDENCE_KEY.decode(),
        _SIGNING_KEY.decode(),
        _EVIDENCE_KEY.hex(),
        _SIGNING_KEY.hex(),
        "/Users/",
        "/home/",
    ):
        assert marker not in text


def test_sentinel_scan_finds_no_phi_site_identity_keys_or_paths() -> None:
    evidence = _evidence()
    for index, case in enumerate(FORBIDDEN_FIELD_CASES):
        row = copy.deepcopy(evidence[index % len(evidence)])
        row[case.field] = case.value
        evidence.append(row)
    evidence.extend(
        [
            {**_entry(_P0_ITEM), "item_id": _SENTINELS[0]},
            {**_entry(_P0_ITEM), "evidence_schema_version": _SENTINELS[1]},
            {**_entry(_P0_ITEM), "generated_at": _SENTINELS[2]},
            {"site_name": _SENTINELS[1], "path": _SENTINELS[3]},
        ]
    )
    manifest = _manifest()
    manifest["moved_work"].append(
        _moved(work_ref="gh-3878", rationale_code="unregistered_rationale")
    )

    packet = _evaluate(manifest=manifest, evidence=evidence)

    text = packet.to_json()
    _assert_value_free(text)
    assert "unregistered_rationale" not in text
    for value in _strings(json.loads(text)):
        assert _ALLOWED_STRING.fullmatch(value), "unexpected packet string"
    reasons = set(packet.to_dict()["reason_codes"])
    assert reasons <= RELEASE_REASON_CODES
    for row in packet.to_dict()["items"]:
        assert set(row["reason_codes"]) <= ITEM_REASON_CODES
    for row in packet.to_dict()["moved_work"]:
        assert set(row["reason_codes"]) <= MOVED_REASON_CODES


@pytest.mark.parametrize("case", FORBIDDEN_FIELD_CASES, ids=lambda case: case.field)
def test_manifest_with_identifying_fields_is_rejected_without_echo(case) -> None:
    manifest = _manifest()
    manifest["items"][0][case.field] = case.value

    with pytest.raises(PrivateLearningReleaseError) as excinfo:
        _evaluate(manifest=manifest)

    assert str(excinfo.value) == "manifest_invalid"
    _assert_value_free(str(excinfo.value))
    assert excinfo.value.__cause__ is None
    assert excinfo.value.__context__ is None


def test_manifest_with_free_text_refs_is_rejected_without_echo() -> None:
    manifest = _manifest()
    manifest["moved_work"][0]["work_ref"] = _SENTINELS[1]

    with pytest.raises(PrivateLearningReleaseError) as excinfo:
        _evaluate(manifest=manifest)

    assert str(excinfo.value) == "manifest_invalid"
    _assert_value_free(repr(excinfo.value))


# Configuration errors ----------------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "code"),
    [
        ({"signing_key": b"short"}, "signing_key_invalid"),
        ({"signing_key": "a" * 40}, "signing_key_invalid"),
        ({"evidence_key": b"short"}, "evidence_key_invalid"),
        ({"source_commit": "HEAD"}, "source_commit_invalid"),
        ({"source_commit": "A" * 40}, "source_commit_invalid"),
        ({"clock": lambda: datetime(2026, 11, 1)}, "clock_invalid"),
    ],
)
def test_configuration_errors_use_fixed_codes(kwargs: dict, code: str) -> None:
    with pytest.raises(PrivateLearningReleaseError, match=f"^{code}$"):
        _evaluate(**kwargs)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda m: m.update(schema_version="openmed.eval.other.v1"),
        lambda m: m["release"].update(release_id="v9.9"),
        lambda m: m["release"].update(git_commit="main"),
        lambda m: m["policy"].update(max_evidence_age_seconds=0),
        lambda m: m["policy"].update(max_evidence_age_seconds=True),
        lambda m: m["items"][0].update(non_compensable="yes"),
        lambda m: m["items"][0].update(required_evidence_digest="sha256:abc"),
        lambda m: m["moved_work"][0].update(target_milestone="later"),
        lambda m: m.update(extra=1),
        lambda m: m.pop("moved_work"),
    ],
)
def test_structurally_invalid_manifest_raises_manifest_invalid(mutate) -> None:
    manifest = _manifest()
    mutate(manifest)

    with pytest.raises(PrivateLearningReleaseError) as excinfo:
        _evaluate(manifest=manifest)

    assert str(excinfo.value) in {"manifest_invalid", "release_unsupported"}


def test_sign_evidence_rejects_short_keys_and_unknown_fields() -> None:
    entry = _entry(_P0_ITEM)
    del entry["signature"]

    with pytest.raises(PrivateLearningReleaseError, match="^evidence_key_invalid$"):
        sign_private_learning_evidence(entry, b"short")
    with pytest.raises(PrivateLearningReleaseError, match="^evidence_invalid$"):
        sign_private_learning_evidence({**entry, "site_id": "x"}, _EVIDENCE_KEY)


def test_injected_clock_bounds_evidence_age() -> None:
    old = _NOW - timedelta(days=30)
    packet = _evaluate(clock=lambda: old + timedelta(hours=2))

    assert packet.to_dict()["release"]["evaluated_at"] == "2026-10-02T14:00:00Z"
    assert all(
        row["reason_codes"] == ["evidence_from_future"]
        for row in packet.to_dict()["items"]
    )
