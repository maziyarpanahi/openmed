"""Tests for the signed v4.0 OpenMedKit release gate.

All evidence is synthetic. Negative controls cover every release-blocking
reason code, not only the passing packet.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator

from openmed.eval import kit_qualification as kq
from openmed.eval import kit_release as kr
from tests.fixtures.kit_release import (
    CANDIDATE,
    IOS_COREML,
    OTHER_COMMIT,
    OTHER_TEST_ONLY_KEY,
    SIM_COREML,
    TEST_ONLY_KEY_ID,
    TEST_ONLY_SIGNING_KEY,
    entry,
    head_resolver,
    lane_record,
    promote_cell,
    qualification_report,
    rewrite,
    synthetic_matrix,
    write_evidence,
)


def build(
    root: Path,
    manifest: dict[str, Any],
    *,
    head: str = CANDIDATE,
    **kwargs: Any,
) -> kr.KitReleasePacket:
    return kr.evaluate_kit_release(
        manifest,
        repo_root=root,
        signing_key=TEST_ONLY_SIGNING_KEY,
        key_id=TEST_ONLY_KEY_ID,
        resolve_head=head_resolver(head),
        **kwargs,
    )


def verify(payload: dict[str, Any], **kwargs: Any) -> kr.KitReleaseVerification:
    return kr.verify_kit_release_packet(payload, TEST_ONLY_SIGNING_KEY, **kwargs)


def _json(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _read(root: Path, manifest: dict[str, Any], entry_id: str) -> Any:
    return json.loads((root / entry(manifest, entry_id)["path"]).read_text("utf-8"))


@pytest.fixture
def ready(tmp_path: Path) -> tuple[Path, dict[str, Any], kr.KitReleasePacket]:
    manifest = write_evidence(tmp_path)
    return tmp_path, manifest, build(tmp_path, manifest)


# --- passing packet ------------------------------------------------------------


def test_complete_manifest_yields_signed_packet_that_verifies_offline(
    ready: tuple[Path, dict[str, Any], kr.KitReleasePacket],
) -> None:
    _root, _manifest, packet = ready

    assert packet.decision == kr.KIT_RELEASE_READY
    assert packet.blocking_codes == ()
    assert packet.release["candidate_commit"] == CANDIDATE
    assert packet.repository_binding["verified"] is True
    assert packet.signature is not None
    assert packet.signature.key_id == TEST_ONLY_KEY_ID
    assert [row["kind"] for row in packet.evidence_kinds] == list(kr.EVIDENCE_KINDS)
    assert all(row["state"] == kr.KIND_PRESENT for row in packet.evidence_kinds)
    assert all(row["state"] == kr.EVIDENCE_VERIFIED for row in packet.evidence)
    assert {row["qualification"] for row in packet.supported_cells} == {"qualified"}
    assert {row["cell"] for row in packet.supported_cells} == {
        "python-linux-torch-cpu",
        "python-macos-torch-cpu",
        "python-windows-torch-cpu",
        "python-linux-onnx-int8-cpu",
        "swift-macos-mlx",
    }
    assert packet.limitations == kr.KIT_RELEASE_LIMITATIONS

    payload = json.loads(json.dumps(packet.to_dict()))
    result = verify(payload, expected_candidate=CANDIDATE, require_ready=True)
    assert result.valid and result.authentic
    assert result.reason_codes == ()
    restored = kr.KitReleasePacket.from_dict(payload)
    assert restored.verify(TEST_ONLY_SIGNING_KEY)
    assert restored.recompute_packet_digest() == packet.packet_digest
    wrong_key = kr.verify_kit_release_packet(payload, OTHER_TEST_ONLY_KEY)
    assert not wrong_key.valid
    assert wrong_key.reason_codes == ("signature_invalid",)


def test_manifest_and_packet_satisfy_bundled_strict_schemas(
    ready: tuple[Path, dict[str, Any], kr.KitReleasePacket],
) -> None:
    _root, manifest, packet = ready
    manifest_schema = kr.load_kit_release_manifest_schema()
    packet_schema = kr.load_kit_release_packet_schema()
    Draft202012Validator.check_schema(manifest_schema)
    Draft202012Validator.check_schema(packet_schema)
    Draft202012Validator(manifest_schema).validate(manifest)
    Draft202012Validator(packet_schema).validate(packet.to_dict())

    unsafe = copy.deepcopy(packet.to_dict())
    unsafe["evidence"][0]["path"] = "release-evidence/kit/openmed.whl"
    assert list(Draft202012Validator(packet_schema).iter_errors(unsafe))
    unsafe = copy.deepcopy(packet.to_dict())
    unsafe["evidence"][0]["summary"]["note"] = "Free text is not allowed here"
    assert list(Draft202012Validator(packet_schema).iter_errors(unsafe))
    bad_manifest = copy.deepcopy(manifest)
    bad_manifest["evidence"][0]["path"] = "../outside.whl"
    assert list(Draft202012Validator(manifest_schema).iter_errors(bad_manifest))


def test_packet_is_reproducible_for_the_same_inputs(
    ready: tuple[Path, dict[str, Any], kr.KitReleasePacket],
) -> None:
    root, manifest, packet = ready
    reordered = copy.deepcopy(manifest)
    reordered["evidence"].reverse()

    again = build(root, reordered)

    assert again.packet_digest == packet.packet_digest
    assert again.signature == packet.signature
    assert again.manifest_digest == packet.manifest_digest


def test_written_packet_round_trips_and_refuses_overwrite(
    ready: tuple[Path, dict[str, Any], kr.KitReleasePacket],
    tmp_path: Path,
) -> None:
    _root, _manifest, packet = ready
    output = tmp_path / "out" / "kit-release-packet.json"

    kr.write_kit_release_packet(packet, output)
    with pytest.raises(kr.KitReleaseError) as exists:
        kr.write_kit_release_packet(packet, output)

    assert exists.value.codes == ("output_exists",)
    assert verify(kr.load_kit_release_packet(output)).valid


# --- offline verification ------------------------------------------------------


def _resign(payload: dict[str, Any]) -> dict[str, Any]:
    """Re-sign an edited payload with the right key, as an insider could."""

    packet = kr.KitReleasePacket.from_dict(payload)
    return packet.sign(TEST_ONLY_SIGNING_KEY, key_id=TEST_ONLY_KEY_ID).to_dict()


def _swap_evidence(payload: dict[str, Any]) -> None:
    rows = payload["evidence"]
    rows[0], rows[1] = rows[1], rows[0]


def _drop_lane(payload: dict[str, Any]) -> None:
    rows = payload["evidence"]
    rows.remove(next(row for row in rows if row["kind"] == kr.TEST_LANE))


def _drop_kind(payload: dict[str, Any]) -> None:
    payload["evidence_kinds"] = [
        row for row in payload["evidence_kinds"] if row["kind"] != kr.THREAT_MODEL
    ]


def _other_candidate(payload: dict[str, Any]) -> None:
    payload["evidence"][0]["candidate_commit"] = OTHER_COMMIT


def _byte_size(payload: dict[str, Any]) -> None:
    payload["evidence"][0]["byte_size"] += 1


def _unsign(payload: dict[str, Any]) -> None:
    payload["signature"] = None


# Case -> (edit, re-sign: True with the right key, None to only re-hash,
# False to leave as edited, expected codes).
TAMPERING: dict[str, tuple[Callable[[dict[str, Any]], None], bool | None, set[str]]] = {
    "changed": (_byte_size, False, {"packet_digest_mismatch", "signature_invalid"}),
    "changed_and_rehashed": (_byte_size, None, {"signature_invalid"}),
    "missing_entry": (_drop_lane, True, {"evidence_count_mismatch:test_lane"}),
    "missing_kind": (_drop_kind, True, {"evidence_kind_missing:threat_model"}),
    "reordered": (_swap_evidence, True, {"evidence_order_invalid"}),
    "unsigned": (_unsign, False, {"signature_missing"}),
    "wrong_candidate": (
        _other_candidate,
        True,
        {"wrong_candidate:python-kit", "decision_inconsistent"},
    ),
}


@pytest.mark.parametrize("case", sorted(TAMPERING))
def test_tampered_packet_entries_fail_with_stable_codes(
    case: str,
    ready: tuple[Path, dict[str, Any], kr.KitReleasePacket],
) -> None:
    _root, _manifest, packet = ready
    mutate, resign, expected = TAMPERING[case]
    payload = copy.deepcopy(packet.to_dict())
    mutate(payload)
    if resign:
        payload = _resign(payload)
    elif resign is None:
        payload["packet_digest"] = kr.KitReleasePacket.from_dict(
            payload
        ).recompute_packet_digest()

    result = verify(payload)

    assert not result.valid
    assert expected <= set(result.reason_codes)
    if resign:
        # Structure is checked independently of the signature.
        assert result.authentic


def test_verifier_reports_schema_candidate_and_decision_codes(
    ready: tuple[Path, dict[str, Any], kr.KitReleasePacket],
) -> None:
    _root, _manifest, packet = ready
    payload = packet.to_dict()

    assert verify(payload, expected_candidate=OTHER_COMMIT).reason_codes == (
        "candidate_mismatch",
    )
    future = {**payload, "schema_version": "2.0.0"}
    assert verify(future).reason_codes == ("unknown_schema",)
    unhashable = {**payload, "schema_version": ["1.0.0"]}
    assert verify(unhashable).reason_codes == ("unknown_schema",)
    malformed = {key: value for key, value in payload.items() if key != "surfaces"}
    assert verify(malformed).reason_codes == ("packet_malformed",)
    assert verify([]).reason_codes == ("packet_malformed",)  # type: ignore[arg-type]

    flipped = copy.deepcopy(payload)
    flipped["decision"] = kr.KIT_RELEASE_NOT_READY
    result = verify(_resign(flipped))
    assert result.authentic
    assert result.reason_codes == ("decision_inconsistent",)

    with pytest.raises(kr.KitReleaseError) as short_key:
        kr.verify_kit_release_packet(payload, b"short")
    assert short_key.value.codes == ("signing_key_invalid",)


def test_not_ready_packet_verifies_but_fails_require_ready(tmp_path: Path) -> None:
    manifest = write_evidence(tmp_path)
    manifest["evidence"] = [
        row for row in manifest["evidence"] if row["kind"] != kr.THREAT_MODEL
    ]
    packet = build(tmp_path, manifest)

    assert packet.decision == kr.KIT_RELEASE_NOT_READY
    assert verify(packet.to_dict()).valid
    assert packet.verify(TEST_ONLY_SIGNING_KEY)
    assert verify(packet.to_dict(), require_ready=True).reason_codes == (
        "decision_not_ready",
    )
    flipped = copy.deepcopy(packet.to_dict())
    flipped["decision"] = kr.KIT_RELEASE_READY
    assert "decision_inconsistent" in verify(_resign(flipped)).reason_codes


# --- evidence presence and integrity ------------------------------------------


def test_pending_evidence_kinds_report_not_present(tmp_path: Path) -> None:
    manifest = write_evidence(tmp_path)
    pending = {
        kr.BUNDLE_MANIFEST,
        kr.PARITY_FIXTURE,
        kr.LICENSE_INVENTORY,
        kr.THREAT_MODEL,
    }
    manifest["evidence"] = [
        row for row in manifest["evidence"] if row["kind"] not in pending
    ]

    packet = build(tmp_path, manifest)

    kinds = {row["kind"]: row for row in packet.evidence_kinds}
    assert packet.decision == kr.KIT_RELEASE_NOT_READY
    for kind in pending:
        assert kinds[kind]["state"] == kr.KIND_NOT_PRESENT
        assert kinds[kind]["entry_count"] == 0
        assert f"evidence_not_present:{kind}" in packet.blocking_codes
    assert kinds[kr.SUPPORT_MATRIX]["state"] == kr.KIND_PRESENT
    assert verify(packet.to_dict()).valid


def test_empty_manifest_blocks_every_kind_and_cell_check(tmp_path: Path) -> None:
    manifest = write_evidence(tmp_path)
    manifest["evidence"] = []
    manifest["moved_items"][0]["narrowed_claims"] = ["react-native"]

    packet = build(tmp_path, manifest)

    assert packet.decision == kr.KIT_RELEASE_NOT_READY
    assert {f"evidence_not_present:{kind}" for kind in kr.EVIDENCE_KINDS} <= set(
        packet.blocking_codes
    )
    assert "supported_cells_unverifiable" in packet.blocking_codes
    assert "narrowed_claim_unverifiable:#3936" in packet.blocking_codes
    assert packet.supported_cells == ()


def _delete(root: Path, manifest: dict[str, Any]) -> None:
    (root / entry(manifest, "kit-threat-model")["path"]).unlink()


def _change(root: Path, manifest: dict[str, Any]) -> None:
    (root / entry(manifest, "swift-kit")["path"]).write_bytes(b"rebuilt artifact\n")


def _entry_commit(root: Path, manifest: dict[str, Any]) -> None:
    entry(manifest, "bundle-bert-int8")["candidate_commit"] = OTHER_COMMIT


def _lane_commit(root: Path, manifest: dict[str, Any]) -> None:
    rewrite(
        root,
        manifest,
        "lane-ci-swift-macos",
        _json(lane_record("ci-swift-macos", commit=OTHER_COMMIT)),
    )


def _expired(root: Path, manifest: dict[str, Any]) -> None:
    entry(manifest, "kit-license-inventory")["generated_at"] = "2026-10-01T00:00:00Z"


def _future(root: Path, manifest: dict[str, Any]) -> None:
    entry(manifest, "kit-license-inventory")["generated_at"] = "2026-11-09T00:00:00Z"


def _report_schema(root: Path, manifest: dict[str, Any]) -> None:
    report = _read(root, manifest, "device-qualification")
    report["schema"] = "openmed.kit_qualification.report.v9"
    rewrite(root, manifest, "device-qualification", _json(report))


def _report_edit(root: Path, manifest: dict[str, Any]) -> None:
    report = _read(root, manifest, "device-qualification")
    report["summary"]["cells_qualified"] += 1
    rewrite(root, manifest, "device-qualification", _json(report))


def _report_unreadable(root: Path, manifest: dict[str, Any]) -> None:
    rewrite(root, manifest, "device-qualification", b'{"schema": "x", "schema": 1}')


def _lane_schema(root: Path, manifest: dict[str, Any]) -> None:
    record = lane_record("ci-swift-macos")
    record["schema"] = "openmed.kit_release.test_lane.v2"
    rewrite(root, manifest, "lane-ci-swift-macos", _json(record))


def _lane_failed(root: Path, manifest: dict[str, Any]) -> None:
    record = lane_record("ci-swift-macos", state="failure", failed=2)
    rewrite(root, manifest, "lane-ci-swift-macos", _json(record))


def _lane_dropped(root: Path, manifest: dict[str, Any]) -> None:
    manifest["evidence"].remove(entry(manifest, "lane-ci-python-low-resource"))


def _matrix_edit(root: Path, manifest: dict[str, Any]) -> None:
    matrix = _read(root, manifest, "support-matrix")
    pin = next(p for p in matrix["runtime_pins"] if p["id"] == "mlx-swift")
    pin["version"] = "0.31.6"
    rewrite(root, manifest, "support-matrix", _json(matrix))


def _matrix_draft(root: Path, manifest: dict[str, Any]) -> None:
    matrix = _read(root, manifest, "support-matrix")
    matrix["state"] = "draft"
    rewrite(root, manifest, "support-matrix", _json(matrix))


def _matrix_invalid(root: Path, manifest: dict[str, Any]) -> None:
    matrix = _read(root, manifest, "support-matrix")
    matrix["cells"][0]["evidence"] = "pending"
    rewrite(root, manifest, "support-matrix", _json(matrix))


def _repeated(root: Path, manifest: dict[str, Any]) -> None:
    duplicate = dict(entry(manifest, "kit-threat-model"), id="kit-threat-model-copy")
    manifest["evidence"].append(duplicate)


def _swift_kit_missing(root: Path, manifest: dict[str, Any]) -> None:
    manifest["evidence"].remove(entry(manifest, "swift-kit"))


def _symlink(root: Path, manifest: dict[str, Any]) -> None:
    path = root / entry(manifest, "offset-parity-vectors")["path"]
    original = path.with_name("original.json")
    path.rename(original)
    path.symlink_to(original.name)


EVIDENCE_FAILURES: dict[
    str, tuple[Callable[[Path, dict[str, Any]], None], set[str]]
] = {
    "deleted": (_delete, {"evidence_missing:kit-threat-model"}),
    "changed": (_change, {"stale_digest:swift-kit"}),
    "entry_wrong_candidate": (_entry_commit, {"wrong_candidate:bundle-bert-int8"}),
    "lane_wrong_candidate": (
        _lane_commit,
        {
            "wrong_candidate:lane-ci-swift-macos",
            "lane_not_passing:ci-swift-macos:swift-macos-mlx",
        },
    ),
    "expired": (_expired, {"evidence_expired:kit-license-inventory"}),
    "future": (_future, {"evidence_from_future:kit-license-inventory"}),
    "report_unknown_schema": (_report_schema, {"unknown_schema:device-qualification"}),
    "report_edited": (
        _report_edit,
        {"report_digest_mismatch:device-qualification"},
    ),
    "report_unreadable": (
        _report_unreadable,
        {"evidence_unreadable:device-qualification"},
    ),
    "lane_unknown_schema": (_lane_schema, {"unknown_schema:lane-ci-swift-macos"}),
    "lane_failed": (
        _lane_failed,
        {
            "lane_failed:lane-ci-swift-macos",
            "lane_not_successful:lane-ci-swift-macos",
            "lane_not_passing:ci-swift-macos:swift-macos-mlx",
        },
    ),
    "lane_missing": (
        _lane_dropped,
        {"lane_result_missing:ci-python-low-resource:python-linux-onnx-int8-cpu"},
    ),
    "matrix_changed_after_qualification": (
        _matrix_edit,
        {"stale_matrix_digest:device-qualification"},
    ),
    "matrix_draft": (
        _matrix_draft,
        {
            "matrix_not_frozen:support-matrix",
            "stale_matrix_digest:device-qualification",
        },
    ),
    "matrix_invalid": (
        _matrix_invalid,
        {"matrix_invalid:support-matrix", "supported_cells_unverifiable"},
    ),
    "single_kind_repeated": (_repeated, {"evidence_kind_repeated:threat_model"}),
    "kit_artifact_missing": (_swift_kit_missing, {"kit_artifact_missing:swift"}),
    "symlink": (_symlink, {"evidence_missing:offset-parity-vectors"}),
}


@pytest.mark.parametrize("case", sorted(EVIDENCE_FAILURES))
def test_evidence_failures_block_with_stable_codes(case: str, tmp_path: Path) -> None:
    manifest = write_evidence(tmp_path)
    mutate, expected = EVIDENCE_FAILURES[case]
    mutate(tmp_path, manifest)

    packet = build(tmp_path, manifest)

    assert packet.decision == kr.KIT_RELEASE_NOT_READY
    assert expected <= set(packet.blocking_codes), packet.blocking_codes
    assert verify(packet.to_dict()).valid
    Draft202012Validator(kr.load_kit_release_packet_schema()).validate(packet.to_dict())


def test_checkout_must_be_at_the_candidate_commit(tmp_path: Path) -> None:
    manifest = write_evidence(tmp_path)

    mismatch = build(tmp_path, manifest, head=OTHER_COMMIT)
    unresolved = kr.evaluate_kit_release(
        manifest,
        repo_root=tmp_path,
        signing_key=TEST_ONLY_SIGNING_KEY,
    )

    assert mismatch.blocking_codes == ("checkout_commit_mismatch",)
    assert mismatch.repository_binding["verified"] is False
    assert unresolved.blocking_codes == ("checkout_unresolved",)
    assert unresolved.decision == kr.KIT_RELEASE_NOT_READY


# --- supported cells and real-device evidence ---------------------------------


def test_supported_cell_without_device_evidence_fails(tmp_path: Path) -> None:
    matrix = synthetic_matrix()
    report = qualification_report(
        matrix,
        unplanned=("python-windows-torch-cpu",),
        without_records=("python-macos-torch-cpu",),
    )
    manifest = write_evidence(tmp_path, matrix=matrix, report=report)

    packet = build(tmp_path, manifest)

    cells = {row["cell"]: row for row in packet.supported_cells}
    assert packet.decision == kr.KIT_RELEASE_NOT_READY
    assert cells["python-windows-torch-cpu"]["qualification"] == "unplanned"
    assert cells["python-macos-torch-cpu"]["qualification"] == "missing"
    assert {
        "supported_cell_without_evidence:python-windows-torch-cpu",
        "supported_cell_without_evidence:python-macos-torch-cpu",
    } <= set(packet.blocking_codes)
    assert cells["swift-macos-mlx"]["qualification"] == "qualified"


@pytest.mark.parametrize(
    ("execution", "code"),
    [
        (kq.SIMULATOR, "simulator_only_evidence"),
        (kq.FAKE_RUNTIME, "fake_runtime_only_evidence"),
    ],
)
def test_virtual_or_fake_runs_cannot_satisfy_a_real_device_cell(
    execution: str, code: str, tmp_path: Path
) -> None:
    matrix = synthetic_matrix(promote_cell(IOS_COREML, "ci-swift-ios-build"))
    report = qualification_report(matrix, executions={IOS_COREML: execution})
    manifest = write_evidence(tmp_path, matrix=matrix, report=report)

    packet = build(tmp_path, manifest)

    cell = next(row for row in packet.supported_cells if row["cell"] == IOS_COREML)
    assert packet.decision == kr.KIT_RELEASE_NOT_READY
    assert f"{code}:{IOS_COREML}" in packet.blocking_codes
    assert cell["qualification"] in {kq.SIMULATOR_ONLY, kq.FAKE_RUNTIME_ONLY}
    # Every other supported cell still qualifies on physical devices.
    others = [row for row in packet.supported_cells if row["cell"] != IOS_COREML]
    assert {row["qualification"] for row in others} == {"qualified"}


def test_forged_qualified_flag_over_simulator_runs_is_refused(tmp_path: Path) -> None:
    matrix = synthetic_matrix(promote_cell(IOS_COREML, "ci-swift-ios-build"))
    report = qualification_report(matrix, executions={IOS_COREML: kq.SIMULATOR})
    row = next(row for row in report["cells"] if row["cell"] == IOS_COREML)
    row["qualified"] = True
    row["status"] = kq.QUALIFIED
    for target in report["targets"]:
        if target["cell"] == IOS_COREML:
            target["status"] = kq.QUALIFIED
    report["report_digest"] = kq._hash(
        {key: value for key, value in report.items() if key != "report_digest"}
    )
    manifest = write_evidence(tmp_path, matrix=matrix, report=report)

    packet = build(tmp_path, manifest)

    assert f"simulator_only_evidence:{IOS_COREML}" in packet.blocking_codes


def test_supported_simulator_cell_is_never_real_device_evidence(
    tmp_path: Path,
) -> None:
    matrix = synthetic_matrix(promote_cell(SIM_COREML, "ci-swift-ios-build"))
    report = qualification_report(matrix, executions={SIM_COREML: kq.SIMULATOR})
    manifest = write_evidence(tmp_path, matrix=matrix, report=report)

    packet = build(tmp_path, manifest)

    assert f"simulator_only_evidence:{SIM_COREML}" in packet.blocking_codes


# --- moved P1 and P2 work -----------------------------------------------------


MOVED_FAILURES: dict[str, tuple[dict[str, Any], set[str]]] = {
    "no_rationale": ({"rationale": None}, {"moved_item_rationale_missing:#3936"}),
    "blank_rationale": ({"rationale": "  "}, {"moved_item_rationale_missing:#3936"}),
    "free_text_rationale": (
        {"rationale": "Waiting on vendor"},
        {"moved_item_rationale_unknown:#3936"},
    ),
    "no_destination": (
        {"destination": None},
        {"moved_item_destination_missing:#3936"},
    ),
    "same_release": (
        {"destination": "v4.0"},
        {"moved_item_destination_invalid:#3936"},
    ),
    "free_text_destination": (
        {"destination": "later"},
        {"moved_item_destination_invalid:#3936"},
    ),
    "p0": ({"priority": "P0"}, {"moved_item_p0:#3936"}),
    "claim_still_supported": (
        {"narrowed_claims": ["swift-macos-mlx", "swift"]},
        {
            "narrowed_claim_still_supported:swift-macos-mlx:#3936",
            "narrowed_claim_still_supported:swift:#3936",
        },
    ),
    "claim_unknown": (
        {"narrowed_claims": ["not-a-cell"]},
        {"narrowed_claim_unknown:#3936"},
    ),
}


@pytest.mark.parametrize("case", sorted(MOVED_FAILURES))
def test_moved_items_need_rationale_destination_and_narrowed_claims(
    case: str, tmp_path: Path
) -> None:
    manifest = write_evidence(tmp_path)
    changes, expected = MOVED_FAILURES[case]
    manifest["moved_items"][0].update(changes)

    packet = build(tmp_path, manifest)

    assert packet.decision == kr.KIT_RELEASE_NOT_READY
    assert set(packet.blocking_codes) == expected
    row = packet.moved_items[0]
    assert row["rationale"] in (None, *kr.MOVED_ITEM_RATIONALES)
    rendered = json.dumps(packet.to_dict())
    assert "Waiting on vendor" not in rendered
    assert '"later"' not in rendered
    assert "not-a-cell" not in rendered


def test_moved_items_with_rationale_and_destination_pass(
    ready: tuple[Path, dict[str, Any], kr.KitReleasePacket],
) -> None:
    root, manifest, _packet = ready
    manifest["moved_items"].append(
        {
            "issue": "#3945",
            "priority": "P2",
            "destination": "#4455",
            "rationale": "blocked_by_dependency",
            "narrowed_claims": [],
        }
    )

    packet = build(root, manifest)

    assert packet.decision == kr.KIT_RELEASE_READY
    assert [row["issue"] for row in packet.moved_items] == ["#3936", "#3945"]
    assert packet.moved_items[0]["narrowed_claims"] == ["react-native"]


# --- manifest structure ---------------------------------------------------------


CANARY = "SyntheticPatientName9824"


def _set(*path: Any, value: Any) -> Callable[[dict[str, Any]], None]:
    def mutate(manifest: dict[str, Any]) -> None:
        target: Any = manifest
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value

    return mutate


STRUCTURE_ERRORS: dict[str, tuple[Callable[[dict[str, Any]], None], str]] = {
    "unknown_schema": (_set("schema_version", value="9.0.0"), "unknown_schema"),
    "unhashable_schema": (_set("schema_version", value=[CANARY]), "unknown_schema"),
    "extra_field": (_set(CANARY, value=1), "manifest_fields_invalid"),
    "policy": (
        _set("compatibility_policy", value=CANARY),
        "compatibility_policy_unsupported",
    ),
    "version": (_set("release", "version", value=CANARY), "release_version_invalid"),
    "matrix_release": (
        _set("release", "matrix_release", value="v3.5"),
        "release_matrix_release_invalid",
    ),
    "commit": (
        _set("release", "candidate_commit", value=CANARY),
        "candidate_commit_invalid",
    ),
    "epoch": (
        _set("release", "source_date_epoch", value=1),
        "source_date_epoch_invalid",
    ),
    "age": (_set("policy", "max_evidence_age_seconds", value=0), "policy_invalid"),
    "parent_path": (
        _set("evidence", 0, "path", value="../" + CANARY),
        "evidence_path_invalid",
    ),
    "absolute_path": (
        _set("evidence", 0, "path", value="/Users/" + CANARY),
        "evidence_path_invalid",
    ),
    "kind": (_set("evidence", 0, "kind", value=CANARY), "evidence_kind_unknown"),
    "identifier_id": (
        _set("evidence", 0, "id", value="a1b2c3d4e5f6a7b8c9d0"),
        "evidence_id_invalid",
    ),
    "artifact_surface": (
        _set("evidence", 0, "surface", value="android"),
        "kit_artifact_surface_invalid",
    ),
    "artifact_format": (
        _set("evidence", 0, "format", value=[CANARY]),
        "kit_artifact_format_invalid",
    ),
    "digest": (_set("evidence", 0, "sha256", value=CANARY), "evidence_digest_invalid"),
    "timestamp": (
        _set("evidence", 0, "generated_at", value=CANARY),
        "evidence_generated_at_invalid",
    ),
    "moved_issue": (
        _set("moved_items", 0, "issue", value=CANARY),
        "moved_item_issue_invalid",
    ),
    "moved_priority": (
        _set("moved_items", 0, "priority", value="P3"),
        "moved_item_priority_invalid",
    ),
    "moved_claims": (
        _set("moved_items", 0, "narrowed_claims", value=[CANARY]),
        "moved_item_claims_invalid",
    ),
}


@pytest.mark.parametrize("case", sorted(STRUCTURE_ERRORS))
def test_manifest_structure_errors_are_value_free(case: str, tmp_path: Path) -> None:
    manifest = write_evidence(tmp_path)
    mutate, code = STRUCTURE_ERRORS[case]
    mutate(manifest)

    with pytest.raises(kr.KitReleaseError) as error:
        build(tmp_path, manifest)

    assert error.value.codes == (code,)
    assert CANARY not in str(error.value)


def test_duplicate_ids_and_keys_are_refused(tmp_path: Path) -> None:
    manifest = write_evidence(tmp_path)
    manifest["evidence"].append(dict(manifest["evidence"][0]))
    with pytest.raises(kr.KitReleaseError) as duplicate:
        kr.validate_kit_release_manifest(manifest)
    assert duplicate.value.codes == ("evidence_id_duplicate",)

    path = tmp_path / "manifest.json"
    path.write_text('{"' + CANARY + '": 1, "' + CANARY + '": 2}', encoding="utf-8")
    with pytest.raises(kr.KitReleaseError) as keys:
        kr.load_kit_release_manifest(path)
    assert keys.value.codes == ("manifest_unreadable",)
    assert CANARY not in str(keys.value)


# --- injected loaders -----------------------------------------------------------


def test_injected_loader_types_pending_evidence(tmp_path: Path) -> None:
    manifest = write_evidence(
        tmp_path,
        contents={kr.LICENSE_INVENTORY: _json({"decision": "fail", "packages": [1]})},
    )
    seen: list[kr.KitEvidenceSource] = []

    def license_loader(source: kr.KitEvidenceSource) -> kr.KitEvidenceResult:
        seen.append(source)
        report = source.read_json()
        failed = report["decision"] != "pass"
        return kr.KitEvidenceResult(
            blocking_codes=("license_gate_failed",) if failed else (),
            summary={"package_count": len(report["packages"])},
        )

    packet = build(tmp_path, manifest, loaders={kr.LICENSE_INVENTORY: license_loader})

    row = next(row for row in packet.evidence if row["kind"] == kr.LICENSE_INVENTORY)
    assert row["loader"] == kr.LOADER_TYPED
    assert row["summary"] == {"package_count": 1}
    assert "license_gate_failed:kit-license-inventory" in packet.blocking_codes
    assert seen[0].release["candidate_commit"] == CANDIDATE
    assert not hasattr(seen[0], "path")


def test_misbehaving_loaders_fail_closed_without_values(tmp_path: Path) -> None:
    manifest = write_evidence(tmp_path)

    def raising(source: kr.KitEvidenceSource) -> kr.KitEvidenceResult:
        raise RuntimeError(f"{CANARY} at /Users/{CANARY}/inventory.json")

    def leaking(source: kr.KitEvidenceSource) -> kr.KitEvidenceResult:
        return kr.KitEvidenceResult(
            blocking_codes=(f"Patient {CANARY}",),
            summary={"source_path": f"/Users/{CANARY}", "Bad Key": 1, "count": True},
        )

    def wrong_type(source: kr.KitEvidenceSource) -> Any:
        return {"blocking_codes": []}

    packet = build(
        tmp_path,
        manifest,
        loaders={
            kr.LICENSE_INVENTORY: raising,
            kr.THREAT_MODEL: leaking,
            kr.BUNDLE_MANIFEST: wrong_type,
        },
    )

    rendered = json.dumps(packet.to_dict())
    assert CANARY not in rendered
    assert "/Users/" not in rendered
    assert {
        "evidence_loader_failed:kit-license-inventory",
        "evidence_loader_failed:bundle-bert-int8",
        "loader_code_invalid:kit-threat-model",
        "loader_summary_invalid:kit-threat-model",
    } <= set(packet.blocking_codes)
    threat = next(row for row in packet.evidence if row["kind"] == kr.THREAT_MODEL)
    assert threat["summary"] == {"count": True}

    with pytest.raises(kr.KitReleaseError) as unknown:
        build(tmp_path, manifest, loaders={"model_card": raising})
    assert unknown.value.codes == ("loader_kind_unknown",)


def test_injected_documents_are_rechecked_before_cross_checks(tmp_path: Path) -> None:
    manifest = write_evidence(tmp_path)

    def bad_matrix(source: kr.KitEvidenceSource) -> kr.KitEvidenceResult:
        return kr.KitEvidenceResult(document={"cells": [{"status": "supported"}]})

    def bad_report(source: kr.KitEvidenceSource) -> kr.KitEvidenceResult:
        return kr.KitEvidenceResult(document={"schema": kq.REPORT_SCHEMA})

    packet = build(
        tmp_path,
        manifest,
        loaders={kr.SUPPORT_MATRIX: bad_matrix, kr.QUALIFICATION_REPORT: bad_report},
    )

    assert {
        "matrix_invalid:support-matrix",
        "report_invalid:device-qualification",
        "supported_cells_unverifiable",
    } <= set(packet.blocking_codes)
    assert packet.supported_cells == ()


def test_default_loaders_cover_every_kind_and_artifact_surfaces_exist() -> None:
    from openmed.eval.kit_support_matrix import EPIC_SURFACES

    assert set(kr.DEFAULT_EVIDENCE_LOADERS) == set(kr.EVIDENCE_KINDS)
    assert kr.REQUIRED_EVIDENCE_KINDS == kr.EVIDENCE_KINDS
    for surfaces in kr.KIT_ARTIFACT_FORMATS.values():
        assert surfaces <= set(EPIC_SURFACES)
    covered = set().union(*kr.KIT_ARTIFACT_FORMATS.values())
    assert covered == set(EPIC_SURFACES)


# --- command line -----------------------------------------------------------------


def test_cli_verify_mode_reports_codes_and_exit_status(
    ready: tuple[Path, dict[str, Any], kr.KitReleasePacket],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _root, _manifest, packet = ready
    key_file = tmp_path / "test-only-signing-key.txt"
    key_file.write_bytes(TEST_ONLY_SIGNING_KEY + b"\n")
    good = tmp_path / "packet.json"
    kr.write_kit_release_packet(packet, good)
    tampered = tmp_path / "tampered.json"
    payload = packet.to_dict()
    payload["evidence"].reverse()
    tampered.write_text(json.dumps(payload), encoding="utf-8")
    base = ["--signing-key-file", str(key_file), "--verify"]

    assert kr.main([*base, str(good), "--require-ready"]) == 0
    assert "VALID; decision=READY" in capsys.readouterr().out
    assert kr.main([*base, str(good), "--expected-candidate", OTHER_COMMIT]) == 1
    assert "candidate_mismatch" in capsys.readouterr().out
    assert kr.main([*base, str(tampered), "--json"]) == 1
    result = json.loads(capsys.readouterr().out)
    assert "evidence_order_invalid" in result["reason_codes"]
    assert kr.main([*base, str(good), "--expected-candidate", CANARY]) == 2
    assert kr.main([*base, str(tmp_path / "missing.json")]) == 2
    errors = capsys.readouterr().err
    assert "packet_unreadable" in errors
    assert str(tmp_path) not in errors


def test_cli_without_a_key_never_writes_a_packet(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    manifest = write_evidence(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    output = tmp_path / "packet.json"
    monkeypatch.delenv(kr.KIT_RELEASE_SIGNING_ENV, raising=False)
    arguments = ["--manifest", str(manifest_path), "--output", str(output)]

    assert kr.main([*arguments, "--repo-root", str(tmp_path)]) == 2
    assert "signing_key_missing" in capsys.readouterr().err
    short = tmp_path / "short-key.txt"
    short.write_bytes(b"short")
    assert kr.main([*arguments, "--signing-key-file", str(short)]) == 2
    assert not output.exists()

    monkeypatch.setenv(kr.KIT_RELEASE_SIGNING_ENV, TEST_ONLY_SIGNING_KEY.decode())
    with pytest.raises(SystemExit):
        kr.main(["--manifest", str(manifest_path)])


def test_packet_signing_requires_a_bounded_key_id(
    ready: tuple[Path, dict[str, Any], kr.KitReleasePacket],
) -> None:
    _root, _manifest, packet = ready
    with pytest.raises(kr.KitReleaseError) as key_id:
        packet.sign(TEST_ONLY_SIGNING_KEY, key_id=f"/Users/{CANARY}")
    assert key_id.value.codes == ("key_id_invalid",)
    with pytest.raises(kr.KitReleaseError) as short:
        packet.sign(b"short", key_id="kit-release")
    assert short.value.codes == ("signing_key_invalid",)
