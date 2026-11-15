"""Tests for the signed, fail-closed v4.0 secure-service release gate."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator

from openmed.eval import service_recovery_drill as drill
from openmed.eval.service_release import (
    EVIDENCE_KINDS,
    KIND_NOT_PRESENT,
    SERVICE_RELEASE_NOT_READY,
    SERVICE_RELEASE_READY,
    SERVICE_RELEASE_REASON_CODES,
    ServiceEvidenceResult,
    ServiceReleaseError,
    ServiceReleasePacket,
    build_failure_code_inventory,
    build_p0_status,
    evaluate_service_release,
    load_service_p0_checklist,
    load_service_recovery_drill_schema,
    load_service_release_manifest,
    load_service_release_manifest_schema,
    load_service_release_packet_schema,
    required_failure_codes,
    scan_for_sentinels,
    service_p0_checklist_digest,
    validate_service_p0_checklist,
    verify_service_release_packet,
)
from tests.fixtures import service_release as fx

V40_SERVICE_P0_ISSUES = (
    "#3544",
    "#3546",
    "#3547",
    "#3549",
    "#3550",
    "#3552",
    "#3574",
    "#3575",
    "#3603",
)


def _evaluate(root: Path, manifest: dict[str, Any], **overrides: Any):
    options: dict[str, Any] = {
        "repo_root": root,
        "signing_key": fx.TEST_ONLY_SIGNING_KEY,
        "key_id": fx.TEST_ONLY_KEY_ID,
        "loaders": fx.stand_in_loaders(),
        "resolve_head": fx.head_resolver(),
        "sentinels": [fx.PLANTED_SENTINEL],
    }
    options.update(overrides)
    packet = evaluate_service_release(manifest, **options)
    _assert_controlled(packet.blocking_codes)
    return packet


def _assert_controlled(codes: Any) -> None:
    # The TEST-ONLY stand-in loaders add one code of their own.
    roots = {*SERVICE_RELEASE_REASON_CODES, "stand_in_not_successful"}
    for code in codes:
        assert code.split(":", 1)[0] in roots, code


def _ready(tmp_path: Path) -> tuple[Path, dict[str, Any], ServiceReleasePacket]:
    root, manifest = fx.make_service_release(tmp_path)
    packet = _evaluate(root, manifest)
    assert packet.decision == SERVICE_RELEASE_READY, packet.blocking_codes
    return root, manifest, packet


# --- pass path ---------------------------------------------------------------


def test_complete_synthetic_evidence_produces_verifiable_ready_packet(
    tmp_path: Path,
) -> None:
    _root, manifest, packet = _ready(tmp_path)

    assert packet.blocking_codes == ()
    assert packet.release["candidate_commit"] == fx.CANDIDATE
    assert packet.release["target"] == "v4.0"
    assert [row["kind"] for row in packet.evidence_kinds] == list(EVIDENCE_KINDS)
    assert all(row["state"] == "present" for row in packet.evidence_kinds)
    assert [row["issue"] for row in packet.p0_items] == list(V40_SERVICE_P0_ISSUES)
    assert all(row["state"] == "complete" for row in packet.p0_items)
    assert packet.sentinel_scan["finding_count"] == 0
    assert packet.sentinel_scan["planted_sentinel_count"] == 1
    assert packet.p0_checklist["retargeted_from"] == "v3.4"
    verification = verify_service_release_packet(
        packet.to_dict(),
        fx.TEST_ONLY_SIGNING_KEY,
        expected_candidate=fx.CANDIDATE,
        expected_release=fx.VERSION,
        expected_p0_checklist_digest=service_p0_checklist_digest(
            load_service_p0_checklist()
        ),
        require_ready=True,
    )
    assert verification.valid and verification.authentic, verification
    assert packet.verify(fx.TEST_ONLY_SIGNING_KEY)
    Draft202012Validator(load_service_release_manifest_schema()).validate(manifest)


def test_manifest_packet_and_drill_satisfy_bundled_strict_schemas(
    tmp_path: Path,
) -> None:
    root, manifest = fx.make_service_release(tmp_path)
    schemas = (
        load_service_release_manifest_schema(),
        load_service_release_packet_schema(),
        load_service_recovery_drill_schema(),
    )
    for schema in schemas:
        Draft202012Validator.check_schema(schema)
    for loaders in (fx.stand_in_loaders(), {}):
        packet = _evaluate(root, manifest, loaders=loaders)
        Draft202012Validator(schemas[1]).validate(packet.to_dict())
    Draft202012Validator(schemas[2]).validate(fx.complete_drill_report())

    unsafe = packet.to_dict()
    unsafe["evidence"][0]["path"] = "evidence/security-profile.json"
    assert list(Draft202012Validator(schemas[1]).iter_errors(unsafe))
    unsafe_manifest = fx.clone(manifest)
    fx.entry(unsafe_manifest, "synthetic-request")["path"] = "evidence/real.json"
    assert list(Draft202012Validator(schemas[0]).iter_errors(unsafe_manifest))


def test_packet_is_deterministic_and_value_free(tmp_path: Path) -> None:
    root, manifest = fx.make_service_release(tmp_path)
    first = _evaluate(root, manifest)
    second = _evaluate(root, fx.clone(manifest))

    assert first.packet_digest == second.packet_digest
    assert first.signature == second.signature
    rendered = json.dumps(first.to_dict(), sort_keys=True)
    for forbidden in (
        str(tmp_path),
        "evidence/",
        "tests/fixtures/",
        fx.PLANTED_SENTINEL,
        fx.TEST_ONLY_SIGNING_KEY.decode(),
        "doc-synthetic-1",
    ):
        assert forbidden not in rendered
    assert scan_for_sentinels(rendered) == ()


# --- master defaults and missing evidence --------------------------------------


def test_master_defaults_block_kinds_whose_producers_are_open_work(
    tmp_path: Path,
) -> None:
    root, manifest = fx.make_service_release(tmp_path)
    packet = _evaluate(root, manifest, loaders={})

    assert packet.decision == SERVICE_RELEASE_NOT_READY
    for entry_id in ("security-profile", "no-egress", "migration-rehearsal"):
        assert f"typed_loader_unavailable:{entry_id}" in packet.blocking_codes
    assert "typed_loader_unavailable:protocol-map" in packet.blocking_codes
    blocked = {
        row["kind"] for row in packet.evidence_kinds if row["state"] != "present"
    }
    assert blocked == set(fx.STAND_IN_KINDS)
    p0 = {row["issue"]: row for row in packet.p0_items}
    assert p0["#3552"]["missing_evidence"] == ["security_profile_report"]
    assert p0["#3575"]["state"] == "complete"
    assert verify_service_release_packet(
        packet.to_dict(), fx.TEST_ONLY_SIGNING_KEY
    ).valid


@pytest.mark.parametrize("entry_id", sorted(fx.EVIDENCE_FILES))
def test_each_missing_kind_is_not_present_and_fails_release(
    tmp_path: Path, entry_id: str
) -> None:
    kind = fx.EVIDENCE_FILES[entry_id][0]
    root, manifest = fx.make_service_release(tmp_path, omit=(entry_id,))
    packet = _evaluate(root, manifest)

    assert packet.decision == SERVICE_RELEASE_NOT_READY
    assert f"evidence_not_present:{kind}" in packet.blocking_codes
    row = next(item for item in packet.evidence_kinds if item["kind"] == kind)
    assert row["state"] == KIND_NOT_PRESENT
    checklist = load_service_p0_checklist()
    for item in checklist["items"]:
        if kind in item["required_evidence"]:
            code = f"p0_evidence_missing:{item['issue']}:{kind}"
            assert code in packet.blocking_codes
    assert verify_service_release_packet(
        packet.to_dict(), fx.TEST_ONLY_SIGNING_KEY
    ).valid


def test_default_master_drill_report_blocks_with_not_available_steps(
    tmp_path: Path,
) -> None:
    report = drill.run_recovery_drill(
        candidate_commit=fx.CANDIDATE, generated_at=fx.GENERATED_AT
    )
    root, manifest = fx.make_service_release(
        tmp_path, documents={"recovery-drill": report}
    )
    packet = _evaluate(root, manifest)

    for code in (
        "drill_step_not_available",
        "drill_objective_not_measured",
        "drill_incomplete",
    ):
        assert f"{code}:recovery-drill" in packet.blocking_codes
    p0 = {row["issue"]: row for row in packet.p0_items}
    assert p0["#3574"]["missing_evidence"] == ["recovery_drill_report"]
    summary = next(row for row in packet.evidence if row["id"] == "recovery-drill")
    assert summary["summary"]["steps_not_available"] == 6


@pytest.mark.parametrize(
    ("mutate", "code"),
    [
        (
            lambda r: r.update(report_digest="sha256:" + "1" * 64),
            "drill_digest_mismatch",
        ),
        (lambda r: r.update(schema="openmed.other.v1"), "unknown_schema"),
        (lambda r: r.pop("summary"), "drill_report_invalid"),
    ],
)
def test_tampered_or_unknown_drill_reports_block(
    tmp_path: Path, mutate: Any, code: str
) -> None:
    report = fx.complete_drill_report()
    mutate(report)
    root, manifest = fx.make_service_release(
        tmp_path, documents={"recovery-drill": report}
    )
    packet = _evaluate(root, manifest)
    assert f"{code}:recovery-drill" in packet.blocking_codes


def test_failed_drill_objective_blocks(tmp_path: Path) -> None:
    def leaky(context: drill.DrillContext) -> drill.DrillStepOutcome:
        del context
        return drill.DrillStepOutcome(
            drill.STEP_FAILED,
            "failed_write_rolled_back",
            {"partial_writes_visible": 1},
        )

    steps = {**fx.stand_in_drill_steps(), (drill.STORAGE, "contain"): leaky}
    report = drill.run_recovery_drill(
        candidate_commit=fx.CANDIDATE, generated_at=fx.GENERATED_AT, steps=steps
    )
    root, manifest = fx.make_service_release(
        tmp_path, documents={"recovery-drill": report}
    )
    packet = _evaluate(root, manifest)
    assert "drill_step_failed:recovery-drill" in packet.blocking_codes
    assert "drill_objective_missed:recovery-drill" in packet.blocking_codes


def test_drill_bound_to_another_candidate_blocks(tmp_path: Path) -> None:
    report = fx.complete_drill_report(candidate=fx.OTHER_COMMIT)
    root, manifest = fx.make_service_release(
        tmp_path, documents={"recovery-drill": report}
    )
    packet = _evaluate(root, manifest)
    assert "wrong_candidate:recovery-drill" in packet.blocking_codes


# --- P0 checklist and status -------------------------------------------------


def test_committed_checklist_names_every_v40_service_p0() -> None:
    checklist = load_service_p0_checklist()

    assert checklist["release"] == "v4.0"
    assert checklist["retargeted_from"] == "v3.4"
    assert checklist["epic"] == "#3543"
    assert checklist["release_plan"] == "#4253"
    assert tuple(item["issue"] for item in checklist["items"]) == (
        V40_SERVICE_P0_ISSUES
    )
    assert all(item["required_evidence"] for item in checklist["items"])
    required = {
        kind for item in checklist["items"] for kind in item["required_evidence"]
    }
    assert required == set(EVIDENCE_KINDS)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda c: c.update(schema="other"),
        lambda c: c.update(release="v4"),
        lambda c: c["items"].append(dict(c["items"][0])),
        lambda c: c["items"][0].update(required_evidence=["unknown_kind"]),
        lambda c: c["items"][0].update(required_evidence=[]),
        lambda c: c["items"][0].update(issue="3544"),
        lambda c: c.update(protected_invariants=[]),
        lambda c: c.update(extra=True),
    ],
)
def test_invalid_checklists_are_rejected(mutate: Any) -> None:
    checklist = fx.clone(load_service_p0_checklist())
    mutate(checklist)
    with pytest.raises(ServiceReleaseError):
        validate_service_p0_checklist(checklist)


def test_open_p0_blocks_even_with_complete_evidence(tmp_path: Path) -> None:
    status = fx.all_closed_status()
    next(row for row in status["issues"] if row["issue"] == "#3552")["state"] = "open"
    root, manifest = fx.make_service_release(tmp_path, documents={"p0-status": status})
    packet = _evaluate(root, manifest)

    assert packet.blocking_codes == ("p0_open:#3552",)
    row = next(item for item in packet.p0_items if item["issue"] == "#3552")
    assert row["status"] == "open" and row["state"] == "incomplete"


def test_p0_missing_from_status_blocks(tmp_path: Path) -> None:
    status = fx.all_closed_status()
    status["issues"] = [row for row in status["issues"] if row["issue"] != "#3575"]
    root, manifest = fx.make_service_release(tmp_path, documents={"p0-status": status})
    packet = _evaluate(root, manifest)
    assert packet.blocking_codes == ("p0_status_missing:#3575",)


def test_unverifiable_status_blocks_every_p0(tmp_path: Path) -> None:
    status = fx.all_closed_status()
    status["candidate_commit"] = fx.OTHER_COMMIT
    root, manifest = fx.make_service_release(tmp_path, documents={"p0-status": status})
    packet = _evaluate(root, manifest)
    assert "wrong_candidate:p0-status" in packet.blocking_codes
    for issue in V40_SERVICE_P0_ISSUES:
        assert f"p0_status_unverifiable:{issue}" in packet.blocking_codes


@pytest.mark.parametrize(
    ("mutate", "code"),
    [
        (lambda s: s.update(release="v4.1"), "release_mismatch"),
        (lambda s: s["issues"].append(dict(s["issues"][0])), "p0_status_invalid"),
        (lambda s: s["issues"][0].update(state="done"), "p0_status_invalid"),
        (lambda s: s.update(schema="other"), "unknown_schema"),
    ],
)
def test_invalid_p0_status_blocks(tmp_path: Path, mutate: Any, code: str) -> None:
    status = fx.all_closed_status()
    mutate(status)
    root, manifest = fx.make_service_release(tmp_path, documents={"p0-status": status})
    packet = _evaluate(root, manifest)
    assert f"{code}:p0-status" in packet.blocking_codes


def test_build_p0_status_rejects_uncontrolled_values() -> None:
    with pytest.raises(ServiceReleaseError, match="p0_status_invalid"):
        build_p0_status({"#1": "merged"}, candidate_commit=fx.CANDIDATE)
    with pytest.raises(ServiceReleaseError, match="candidate_commit_invalid"):
        build_p0_status({"#1": "closed"}, candidate_commit="HEAD")


# --- moved items -------------------------------------------------------------


def _moved(**overrides: Any) -> dict[str, Any]:
    item = {
        "destination": "v4.1",
        "issue": "#3600",
        "narrowed_claims": ["diagnostic_bundle_available"],
        "priority": "P2",
        "rationale": "blocked_by_dependency",
        "residual_risk": "manual_diagnostics_only",
    }
    item.update(overrides)
    return item


def test_moved_p1_or_p2_with_rationale_does_not_block(tmp_path: Path) -> None:
    root, manifest = fx.make_service_release(tmp_path)
    manifest["moved_items"] = [_moved(), _moved(issue="#3602", priority="P1")]
    packet = _evaluate(root, manifest)

    assert packet.decision == SERVICE_RELEASE_READY, packet.blocking_codes
    assert [row["issue"] for row in packet.moved_items] == ["#3600", "#3602"]


@pytest.mark.parametrize(
    ("item", "code"),
    [
        (_moved(priority="P0"), "p0_moved:#3600"),
        (_moved(issue="#3552", priority="P1"), "p0_moved:#3552"),
        (_moved(rationale=None), "moved_item_rationale_missing:#3600"),
        (_moved(rationale="because"), "moved_item_rationale_missing:#3600"),
        (_moved(destination=None), "moved_item_destination_invalid:#3600"),
        (_moved(destination="v3.4"), "moved_item_destination_invalid:#3600"),
        (_moved(residual_risk=None), "moved_item_residual_risk_missing:#3600"),
        (
            _moved(narrowed_claims=["no_mandatory_egress"]),
            "moved_item_weakens_invariant:#3600",
        ),
    ],
)
def test_moved_item_failures_block(
    tmp_path: Path, item: dict[str, Any], code: str
) -> None:
    root, manifest = fx.make_service_release(tmp_path)
    manifest["moved_items"] = [item]
    packet = _evaluate(root, manifest)
    assert code in packet.blocking_codes
    assert packet.decision == SERVICE_RELEASE_NOT_READY


@pytest.mark.parametrize(
    ("item", "code"),
    [
        (_moved(destination="next quarter"), "moved_item_destination_invalid"),
        (_moved(residual_risk="Free text"), "moved_item_residual_risk_invalid"),
        (_moved(narrowed_claims=["a", "a"]), "moved_item_claims_invalid"),
        (_moved(issue="3600"), "moved_item_issue_invalid"),
        (_moved(priority="P3"), "moved_item_priority_invalid"),
    ],
)
def test_free_text_moved_items_are_rejected(
    tmp_path: Path, item: dict[str, Any], code: str
) -> None:
    root, manifest = fx.make_service_release(tmp_path)
    manifest["moved_items"] = [item]
    with pytest.raises(ServiceReleaseError, match=code):
        _evaluate(root, manifest)


# --- entry-level failure codes -----------------------------------------------


def test_stale_missing_and_mis_bound_entries_block(tmp_path: Path) -> None:
    root, manifest = fx.make_service_release(tmp_path)
    fx.entry(manifest, "protocol-map")["sha256"] = "sha256:" + "a" * 64
    (root / fx.entry(manifest, "no-egress")["path"]).unlink()
    fx.entry(manifest, "migration-rehearsal")["candidate_commit"] = fx.OTHER_COMMIT
    fx.entry(manifest, "security-profile")["generated_at"] = "2025-01-01T00:00:00Z"
    fx.entry(manifest, "threat-model")["generated_at"] = "2026-11-09T00:00:00Z"
    packet = _evaluate(root, manifest)

    for code in (
        "stale_digest:protocol-map",
        "evidence_missing:no-egress",
        "wrong_candidate:migration-rehearsal",
        "evidence_expired:security-profile",
        "evidence_from_future:threat-model",
    ):
        assert code in packet.blocking_codes
    assert verify_service_release_packet(
        packet.to_dict(), fx.TEST_ONLY_SIGNING_KEY
    ).valid


def test_symlinked_evidence_is_not_followed(tmp_path: Path) -> None:
    root, manifest = fx.make_service_release(tmp_path)
    path = root / fx.entry(manifest, "protocol-map")["path"]
    original = path.with_name("original.json")
    path.rename(original)
    path.symlink_to(original.name)
    packet = _evaluate(root, manifest)
    assert "evidence_missing:protocol-map" in packet.blocking_codes


def test_repeated_single_entry_kind_blocks(tmp_path: Path) -> None:
    root, manifest = fx.make_service_release(tmp_path)
    duplicate = dict(fx.entry(manifest, "protocol-map"), id="protocol-map-copy")
    manifest["evidence"].append(duplicate)
    packet = _evaluate(root, manifest)
    assert "evidence_kind_repeated:protocol_map" in packet.blocking_codes


def test_failure_code_inventory_must_cover_every_controlled_code(
    tmp_path: Path,
) -> None:
    inventory = build_failure_code_inventory(fx.CANDIDATE)
    inventory["codes"] = inventory["codes"][1:]
    root, manifest = fx.make_service_release(
        tmp_path, documents={"failure-codes": inventory}
    )
    packet = _evaluate(root, manifest)
    assert "failure_code_inventory_incomplete:failure-codes" in packet.blocking_codes

    inventory = build_failure_code_inventory(fx.CANDIDATE)
    inventory["codes"].append({"code": "Raw Value", "surface": "release_gate"})
    fx.rewrite(root, manifest, "failure-codes", inventory)
    packet = _evaluate(root, manifest)
    assert "failure_code_inventory_invalid:failure-codes" in packet.blocking_codes


def test_inventory_lists_gate_drill_and_service_error_codes() -> None:
    pairs = required_failure_codes()
    assert pairs == tuple(sorted(set(pairs)))
    surfaces = {pair.split(":", 1)[0] for pair in pairs}
    assert surfaces == {
        "recovery_drill",
        "release_gate",
        "release_verifier",
        "service_error",
    }
    assert "release_gate:p0_open" in pairs
    assert "recovery_drill:encrypted_backup_restore" in pairs
    assert "service_error:validation_error" in pairs
    assert list(SERVICE_RELEASE_REASON_CODES) == sorted(
        set(SERVICE_RELEASE_REASON_CODES)
    )


def test_failing_or_unsafe_loaders_fail_closed(tmp_path: Path) -> None:
    root, manifest = fx.make_service_release(tmp_path)

    def raising(source: Any) -> ServiceEvidenceResult:
        raise RuntimeError(f"secret detail {source.entry_id}")

    def unsafe(source: Any) -> ServiceEvidenceResult:
        del source
        return ServiceEvidenceResult(
            ("Contains Raw Value",), {"raw_note": "Free text with spaces"}
        )

    def untyped(source: Any) -> ServiceEvidenceResult:
        del source
        return ServiceEvidenceResult(loader="digest_only")

    loaders = {
        **fx.stand_in_loaders(),
        "security_profile_report": raising,
        "protocol_map": unsafe,
        "no_egress_attestation": untyped,
    }
    packet = _evaluate(root, manifest, loaders=loaders)

    assert "evidence_loader_failed:security-profile" in packet.blocking_codes
    assert "loader_code_invalid:protocol-map" in packet.blocking_codes
    assert "loader_summary_invalid:protocol-map" in packet.blocking_codes
    assert "typed_loader_unavailable:no-egress" in packet.blocking_codes
    assert "secret detail" not in json.dumps(packet.to_dict())


def test_stand_in_failure_state_blocks(tmp_path: Path) -> None:
    root, manifest = fx.make_service_release(
        tmp_path,
        documents={
            "no-egress": fx.stand_in_document("no_egress_attestation", state="failure")
        },
    )
    packet = _evaluate(root, manifest)
    assert "stand_in_not_successful:no-egress" in packet.blocking_codes


# --- repository binding and sentinels ------------------------------------------


@pytest.mark.parametrize(
    ("resolver", "code"),
    [
        (fx.head_resolver(fx.OTHER_COMMIT), "checkout_commit_mismatch"),
        (lambda _root: "not-a-commit", "checkout_unresolved"),
    ],
)
def test_checkout_must_be_the_exact_candidate(
    tmp_path: Path, resolver: Any, code: str
) -> None:
    root, manifest = fx.make_service_release(tmp_path)
    packet = _evaluate(root, manifest, resolve_head=resolver)
    assert code in packet.blocking_codes
    assert packet.repository_binding["verified"] is False


@pytest.mark.parametrize(
    ("content", "name"),
    [
        ({"note": fx.PLANTED_SENTINEL}, "planted_sentinel"),
        ({"contact": "person@example.org"}, "email_address"),
        ({"path": "/Users/someone/records.json"}, "absolute_user_path"),
        ({"pem": "-----BEGIN RSA PRIVATE KEY-----"}, "private_key_block"),
        ({"id": "123-45-6789"}, "us_ssn"),
    ],
)
def test_sentinel_scan_blocks_prohibited_values_without_echoing_them(
    tmp_path: Path, content: dict[str, Any], name: str
) -> None:
    root, manifest = fx.make_service_release(
        tmp_path, documents={"synthetic-request": content}
    )
    packet = _evaluate(root, manifest)

    assert f"sentinel_detected:{name}:synthetic-request" in packet.blocking_codes
    assert packet.sentinel_scan["finding_count"] == 1
    rendered = json.dumps(packet.to_dict())
    for value in content.values():
        assert value not in rendered


def test_sentinel_scanner_reports_classes_only() -> None:
    assert scan_for_sentinels("2026-11-08T00:00:00Z sha256:" + "a" * 64) == ()
    assert scan_for_sentinels(b"bearer abcdefghijklmnopqrstuvwxyz") == ("bearer_token",)
    assert scan_for_sentinels("x", sentinels=["x"]) == ("planted_sentinel",)


# --- structural refusals -----------------------------------------------------


@pytest.mark.parametrize(
    ("mutate", "code"),
    [
        (lambda m: m.update(schema_version="2.0.0"), "unknown_schema"),
        (lambda m: m.update(extra=1), "manifest_fields_invalid"),
        (lambda m: m.update(compatibility_policy="any"), "compatibility_policy"),
        (
            lambda m: m["release"].update(version="3.4.0", target="v3.4"),
            "release_target_unsupported",
        ),
        (lambda m: m["release"].update(target="v4.1"), "release_target_invalid"),
        (lambda m: m["release"].update(candidate_commit="HEAD"), "candidate_commit"),
        (lambda m: m["release"].update(source_date_epoch=1), "source_date_epoch"),
        (lambda m: m["policy"].update(max_evidence_age_seconds=0), "policy_invalid"),
        (lambda m: m["evidence"][0].update(path="../outside.json"), "evidence_path"),
        (lambda m: m["evidence"][0].update(path="/abs/file.json"), "evidence_path"),
        (lambda m: m["evidence"][0].update(kind="other"), "evidence_kind_unknown"),
        (lambda m: m["evidence"][0].update(id="Upper"), "evidence_id_invalid"),
        (lambda m: m["evidence"].append(dict(m["evidence"][0])), "evidence_id_dup"),
        (lambda m: m["evidence"][0].update(sha256="abc"), "evidence_digest"),
        (lambda m: m["evidence"][0].update(note="x"), "evidence_entry_invalid"),
        (
            lambda m: next(e for e in m["evidence"] if e["kind"] == "fixture").update(
                path="evidence/real-export.json"
            ),
            "fixture_path_not_synthetic",
        ),
    ],
)
def test_structurally_invalid_manifests_are_refused(
    tmp_path: Path, mutate: Any, code: str
) -> None:
    root, manifest = fx.make_service_release(tmp_path)
    mutate(manifest)
    with pytest.raises(ServiceReleaseError, match=code):
        _evaluate(root, manifest)


def test_unusable_keys_loaders_and_checklists_are_refused(tmp_path: Path) -> None:
    root, manifest = fx.make_service_release(tmp_path)
    with pytest.raises(ServiceReleaseError, match="signing_key_invalid"):
        _evaluate(root, manifest, signing_key=b"short")
    with pytest.raises(ServiceReleaseError, match="key_id_invalid"):
        _evaluate(root, manifest, key_id="Key With Spaces")
    with pytest.raises(ServiceReleaseError, match="loader_kind_unknown"):
        _evaluate(root, manifest, loaders={"other": fx.stand_in_loader("other")})
    with pytest.raises(ServiceReleaseError, match="sentinel_invalid"):
        _evaluate(root, manifest, sentinels=["short"])
    checklist = fx.clone(load_service_p0_checklist())
    checklist["release"] = "v4.1"
    with pytest.raises(ServiceReleaseError, match="p0_checklist_release_mismatch"):
        _evaluate(root, manifest, p0_checklist=checklist)


def test_manifest_loader_rejects_duplicate_keys(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text('{"schema_version": "1.0.0", "schema_version": "1.0.0"}')
    with pytest.raises(ServiceReleaseError, match="manifest_unreadable"):
        load_service_release_manifest(path)


# --- offline verification ----------------------------------------------------


def _resign(payload: dict[str, Any]) -> dict[str, Any]:
    packet = ServiceReleasePacket.from_dict(payload)
    return packet.sign(fx.TEST_ONLY_SIGNING_KEY, key_id=fx.TEST_ONLY_KEY_ID).to_dict()


def _codes(payload: Any, key: bytes = fx.TEST_ONLY_SIGNING_KEY, **kwargs: Any):
    result = verify_service_release_packet(payload, key, **kwargs)
    assert not result.valid
    for code in result.reason_codes:
        assert code.split(":", 1)[0] in SERVICE_RELEASE_REASON_CODES, code
    return set(result.reason_codes)


def test_verifier_rejects_changed_missing_reordered_unsigned_and_wrong_release(
    tmp_path: Path,
) -> None:
    _root, _manifest, packet = _ready(tmp_path)
    original = packet.to_dict()

    changed = fx.clone(original)
    changed["evidence"][0]["digest"] = "sha256:" + "b" * 64
    assert {"packet_digest_mismatch", "signature_invalid"} <= _codes(changed)

    missing = fx.clone(original)
    missing["evidence"] = missing["evidence"][1:]
    codes = _codes(missing)
    assert "evidence_count_mismatch:security_profile_report" in codes
    assert "packet_digest_mismatch" in codes
    assert "evidence_count_mismatch:security_profile_report" in _codes(_resign(missing))

    reordered = fx.clone(original)
    reordered["evidence"].reverse()
    assert "evidence_order_invalid" in _codes(_resign(reordered))

    unsigned = fx.clone(original)
    unsigned["signature"] = None
    assert "signature_missing" in _codes(unsigned)

    assert _codes(original, expected_release="4.0.1") == {"release_mismatch"}
    assert _codes(original, expected_candidate=fx.OTHER_COMMIT) == {
        "candidate_mismatch"
    }
    assert _codes(original, expected_p0_checklist_digest="sha256:" + "0" * 64) == {
        "p0_checklist_mismatch"
    }
    assert _codes(original, key=fx.OTHER_TEST_ONLY_KEY) == {"signature_invalid"}


def test_verifier_rejects_resigned_inconsistent_decisions(tmp_path: Path) -> None:
    root, manifest = fx.make_service_release(tmp_path, omit=("protocol-map",))
    blocked = _evaluate(root, manifest).to_dict()
    forged = fx.clone(blocked)
    forged["decision"] = SERVICE_RELEASE_READY
    forged["blocking_codes"] = []
    assert "decision_inconsistent" in _codes(_resign(forged))
    assert _codes(blocked, require_ready=True) == {"decision_not_ready"}

    wrong_candidate = fx.clone(blocked)
    wrong_candidate["evidence"][0]["candidate_commit"] = fx.OTHER_COMMIT
    assert "wrong_candidate:security-profile" in _codes(_resign(wrong_candidate))

    wrong_target = fx.clone(blocked)
    wrong_target["release"]["target"] = "v3.4"
    assert "release_target_unsupported" in _codes(_resign(wrong_target))


def test_verifier_rejects_unknown_and_malformed_packets(tmp_path: Path) -> None:
    _root, _manifest, packet = _ready(tmp_path)
    payload = packet.to_dict()
    assert _codes({**payload, "schema_version": "9.0.0"}) == {"unknown_schema"}
    assert _codes({**payload, "evidence": "rows"}) == {"packet_malformed"}
    assert _codes([]) == {"packet_malformed"}
    extra = {**payload, "notes": "free text"}
    assert _codes(extra) == {"packet_malformed"}
    with pytest.raises(ServiceReleaseError, match="signing_key_invalid"):
        verify_service_release_packet(payload, b"short")


def test_from_dict_round_trip_preserves_signature(tmp_path: Path) -> None:
    _root, _manifest, packet = _ready(tmp_path)
    restored = ServiceReleasePacket.from_dict(json.loads(json.dumps(packet.to_dict())))
    assert restored.verify(fx.TEST_ONLY_SIGNING_KEY)
    assert (
        replace(restored, decision="NOT_READY").verify(fx.TEST_ONLY_SIGNING_KEY)
        is False
    )
