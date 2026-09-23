"""Tests for the signed v3 Journey release gate."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from openmed.eval.journey_release import (
    JOURNEY_RELEASE_GATES,
    JOURNEY_RELEASE_NOT_READY,
    JOURNEY_RELEASE_READY,
    JOURNEY_RELEASE_STATES,
    JourneyReleaseError,
    JourneyReleasePacket,
    evaluate_journey_release,
    load_journey_release_manifest,
    load_journey_release_manifest_schema,
    load_journey_release_packet_schema,
    main,
    write_journey_release_packet,
)
from tests.fixtures.journey_release import (
    gate_report,
    make_release_repository,
)

SIGNING_KEY = "synthetic-release-signing-key-32-bytes-minimum"


def test_complete_tagged_evidence_produces_verifiable_ready_packet(
    tmp_path: Path,
) -> None:
    root, commit, manifest = make_release_repository(tmp_path)

    packet = evaluate_journey_release(
        manifest,
        repo_root=root,
        signing_key=SIGNING_KEY,
        key_id="v3-release-test",
    )

    assert packet.decision == JOURNEY_RELEASE_READY
    assert packet.verify(SIGNING_KEY)
    assert packet.release["git_commit"] == commit
    assert packet.repository_binding["verified"] is True
    assert tuple(item.gate for item in packet.gates) == JOURNEY_RELEASE_GATES
    assert all(item.passed for item in packet.gates)
    assert all(item["verified"] for item in packet.frozen_inputs)


def test_manifest_and_packet_satisfy_bundled_strict_schemas(tmp_path: Path) -> None:
    root, _commit, manifest = make_release_repository(tmp_path)
    packet = evaluate_journey_release(
        manifest,
        repo_root=root,
        signing_key=SIGNING_KEY,
    )
    manifest_schema = load_journey_release_manifest_schema()
    packet_schema = load_journey_release_packet_schema()

    Draft202012Validator.check_schema(manifest_schema)
    Draft202012Validator.check_schema(packet_schema)
    Draft202012Validator(manifest_schema).validate(manifest)
    Draft202012Validator(packet_schema).validate(packet.to_dict())

    unsafe_packet = copy.deepcopy(packet.to_dict())
    unsafe_packet["gates"][3]["metrics"]["source_text"] = "not-allowed"
    assert list(Draft202012Validator(packet_schema).iter_errors(unsafe_packet))


@pytest.mark.parametrize(
    ("gate", "metric"),
    [
        ("schema", "invalid_schema_count"),
        ("schema", "invalid_span_count"),
        ("schema", "invalid_evidence_count"),
        ("provenance", "broken_link_count"),
        ("provenance", "unhashed_artifact_count"),
        ("clinical_nlp", "unreviewed_high_risk_count"),
        ("clinical_nlp", "abstention_failure_count"),
        ("privacy", "critical_leakage_count"),
        ("privacy", "raw_value_finding_count"),
        ("interoperability", "conformance_failure_count"),
        ("interoperability", "roundtrip_loss_without_disclosure_count"),
        ("application", "failed_test_count"),
        ("application", "untyped_terminal_state_count"),
        ("application", "prohibited_public_claim_count"),
        ("performance", "slo_breach_count"),
        ("recovery", "non_idempotent_replay_count"),
        ("recovery", "migration_failure_count"),
        ("recovery", "recovery_failure_count"),
        ("security", "unresolved_critical_finding_count"),
        ("security", "unresolved_high_finding_count"),
        ("security", "unmitigated_threat_count"),
        ("licensing", "unchecked_asset_count"),
        ("licensing", "prohibited_distribution_count"),
    ],
)
def test_every_critical_failure_blocks_release(
    tmp_path: Path,
    gate: str,
    metric: str,
) -> None:
    root, _commit, manifest = make_release_repository(tmp_path)
    gate_report(manifest, gate)["metrics"][metric] = 1

    packet = evaluate_journey_release(
        manifest,
        repo_root=root,
        signing_key=SIGNING_KEY,
    )

    assert packet.decision == JOURNEY_RELEASE_NOT_READY
    result = next(item for item in packet.gates if item.gate == gate)
    assert result.passed is False
    assert f"nonzero:{metric}" in result.blocking_codes


@pytest.mark.parametrize("state", JOURNEY_RELEASE_STATES[1:])
def test_typed_non_success_states_are_preserved_and_block(
    tmp_path: Path,
    state: str,
) -> None:
    root, _commit, manifest = make_release_repository(tmp_path)
    gate_report(manifest, "application")["state"] = state

    packet = evaluate_journey_release(
        manifest,
        repo_root=root,
        signing_key=SIGNING_KEY,
    )

    result = next(item for item in packet.gates if item.gate == "application")
    assert packet.decision == JOURNEY_RELEASE_NOT_READY
    assert result.state == state
    assert f"state:{state}" in result.blocking_codes


def test_stale_or_future_artifacts_fail_closed(tmp_path: Path) -> None:
    root, _commit, manifest = make_release_repository(tmp_path)
    gate_report(manifest, "privacy")["generated_at"] = "2025-12-01T00:00:00Z"
    gate_report(manifest, "security")["generated_at"] = "2026-01-03T00:00:00Z"

    packet = evaluate_journey_release(
        manifest,
        repo_root=root,
        signing_key=SIGNING_KEY,
    )

    privacy = next(item for item in packet.gates if item.gate == "privacy")
    security = next(item for item in packet.gates if item.gate == "security")
    assert packet.decision == JOURNEY_RELEASE_NOT_READY
    assert "artifact_stale" in privacy.blocking_codes
    assert "artifact_from_future" in security.blocking_codes


def test_performance_requires_ordered_distribution_and_complete_context(
    tmp_path: Path,
) -> None:
    root, _commit, manifest = make_release_repository(tmp_path)
    metrics = gate_report(manifest, "performance")["metrics"]
    metrics["latency_p50_ms"] = 40.0
    metrics["latency_p95_ms"] = 20.0

    packet = evaluate_journey_release(
        manifest,
        repo_root=root,
        signing_key=SIGNING_KEY,
    )

    performance = next(item for item in packet.gates if item.gate == "performance")
    assert "latency_distribution_invalid" in performance.blocking_codes
    assert packet.environment["hardware_id"] == "synthetic_cpu_runner"
    assert performance.metrics["quantization"] == "int8"
    assert performance.metrics["concurrency"] == 4
    assert str(performance.metrics["dataset_digest"]).startswith("sha256:")


def test_frozen_input_tamper_blocks_provenance_without_copying_values(
    tmp_path: Path,
) -> None:
    root, _commit, manifest = make_release_repository(tmp_path)
    marker = "synthetic-sensitive-marker-never-in-packet"
    (root / "inputs" / "synthetic.json").write_text(marker, encoding="utf-8")

    packet = evaluate_journey_release(
        manifest,
        repo_root=root,
        signing_key=SIGNING_KEY,
    )
    rendered = json.dumps(packet.to_dict(), sort_keys=True)

    assert packet.decision == JOURNEY_RELEASE_NOT_READY
    provenance = next(item for item in packet.gates if item.gate == "provenance")
    assert "frozen_input_mismatch:golden_scenario" in provenance.blocking_codes
    assert marker not in rendered
    assert "inputs/synthetic.json" not in rendered


def test_tag_and_checkout_must_resolve_to_manifest_commit(tmp_path: Path) -> None:
    root, _commit, manifest = make_release_repository(tmp_path)
    manifest["release"]["git_commit"] = "f" * 40

    packet = evaluate_journey_release(
        manifest,
        repo_root=root,
        signing_key=SIGNING_KEY,
    )

    assert packet.decision == JOURNEY_RELEASE_NOT_READY
    assert packet.repository_binding["verified"] is False
    assert packet.repository_binding["blocking_codes"] == [
        "checkout_commit_mismatch",
        "tag_commit_mismatch",
    ]


def test_missing_tag_yields_signed_not_ready_packet(tmp_path: Path) -> None:
    root, _commit, manifest = make_release_repository(tmp_path)
    manifest["release"]["version"] = "3.0.1"
    manifest["release"]["git_tag"] = "v3.0.1"

    packet = evaluate_journey_release(
        manifest,
        repo_root=root,
        signing_key=SIGNING_KEY,
    )

    assert packet.decision == JOURNEY_RELEASE_NOT_READY
    assert packet.verify(SIGNING_KEY)
    assert "tag_unresolved" in packet.repository_binding["blocking_codes"]


def test_restricted_assets_are_user_supplied_and_never_waived(tmp_path: Path) -> None:
    root, _commit, manifest = make_release_repository(tmp_path)
    manifest["licenses"][1].update(
        {
            "license_id": "DUA-required",
            "redistributable": False,
            "distribution": "user_supplied",
            "use": "eval",
        }
    )
    allowed = evaluate_journey_release(
        manifest,
        repo_root=root,
        signing_key=SIGNING_KEY,
    )
    assert allowed.decision == JOURNEY_RELEASE_READY

    manifest["licenses"][1]["distribution"] = "bundled"
    blocked = evaluate_journey_release(
        manifest,
        repo_root=root,
        signing_key=SIGNING_KEY,
    )
    licensing = next(item for item in blocked.gates if item.gate == "licensing")
    assert blocked.decision == JOURNEY_RELEASE_NOT_READY
    assert "restricted_asset_bundled:synthetic_dataset" in licensing.blocking_codes


def test_critical_exception_cannot_override_a_failure(tmp_path: Path) -> None:
    root, _commit, manifest = make_release_repository(tmp_path)
    gate_report(manifest, "security")["metrics"][
        "unresolved_critical_finding_count"
    ] = 1
    manifest["exceptions"] = [
        {
            "code": "critical_security_waiver",
            "gate": "security",
            "severity": "critical",
            "disposition": "accepted",
            "expires_at": "2026-02-01T00:00:00Z",
        }
    ]

    packet = evaluate_journey_release(
        manifest,
        repo_root=root,
        signing_key=SIGNING_KEY,
    )

    assert packet.decision == JOURNEY_RELEASE_NOT_READY
    assert (
        "nonwaivable_exception:critical_security_waiver"
        in packet.reproduction["exception_blocking_codes"]
    )


def test_required_claims_and_limitations_are_non_removable(tmp_path: Path) -> None:
    root, _commit, manifest = make_release_repository(tmp_path)
    manifest["claims"].remove("typed_failure_states")
    manifest["limitations"].remove("not_clinically_validated")

    packet = evaluate_journey_release(
        manifest,
        repo_root=root,
        signing_key=SIGNING_KEY,
    )

    assert packet.decision == JOURNEY_RELEASE_NOT_READY
    assert set(packet.reproduction["policy_blocking_codes"]) == {
        "missing_claim:typed_failure_states",
        "missing_limitation:not_clinically_validated",
    }


def test_packet_round_trip_and_tamper_detection(tmp_path: Path) -> None:
    root, _commit, manifest = make_release_repository(tmp_path)
    packet = evaluate_journey_release(
        manifest,
        repo_root=root,
        signing_key=SIGNING_KEY,
    )

    restored = JourneyReleasePacket.from_dict(packet.to_dict())
    assert restored.to_dict() == packet.to_dict()
    assert restored.verify(SIGNING_KEY)

    tampered = copy.deepcopy(packet.to_dict())
    tampered["decision"] = JOURNEY_RELEASE_NOT_READY
    assert not JourneyReleasePacket.from_dict(tampered).verify(SIGNING_KEY)


def test_unknown_fields_and_missing_performance_context_are_rejected(
    tmp_path: Path,
) -> None:
    root, _commit, manifest = make_release_repository(tmp_path)
    gate_report(manifest, "privacy")["raw_text"] = "not-allowed"
    with pytest.raises(JourneyReleaseError, match="fields differ"):
        evaluate_journey_release(
            manifest,
            repo_root=root,
            signing_key=SIGNING_KEY,
        )

    del gate_report(manifest, "privacy")["raw_text"]
    del gate_report(manifest, "performance")["metrics"]["quantization"]
    with pytest.raises(JourneyReleaseError, match="fields differ"):
        evaluate_journey_release(
            manifest,
            repo_root=root,
            signing_key=SIGNING_KEY,
        )


def test_cli_writes_once_and_refuses_implicit_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, _commit, manifest = make_release_repository(tmp_path)
    manifest_path = root / "journey-release-manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    output = root / "journey-release-packet.json"
    monkeypatch.setenv("OPENMED_JOURNEY_RELEASE_KEY", SIGNING_KEY)

    assert (
        main(
            [
                "--manifest",
                str(manifest_path),
                "--repo-root",
                str(root),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert output.is_file()
    assert (
        main(
            [
                "--manifest",
                str(manifest_path),
                "--repo-root",
                str(root),
                "--output",
                str(output),
            ]
        )
        == 2
    )


def test_loader_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.json"
    path.write_text(
        '{"schema_version":"1.0.0","schema_version":"1.0.0"}', encoding="utf-8"
    )

    with pytest.raises(JourneyReleaseError, match="duplicate JSON key"):
        load_journey_release_manifest(path)


def test_writer_requires_explicit_overwrite(tmp_path: Path) -> None:
    root, _commit, manifest = make_release_repository(tmp_path)
    packet = evaluate_journey_release(
        manifest,
        repo_root=root,
        signing_key=SIGNING_KEY,
    )
    output = tmp_path / "packet.json"
    write_journey_release_packet(packet, output)

    with pytest.raises(JourneyReleaseError, match="refusing to overwrite"):
        write_journey_release_packet(packet, output)
    write_journey_release_packet(packet, output, overwrite=True)
    assert json.loads(output.read_text(encoding="utf-8"))["decision"] == "READY"


def test_writer_does_not_replace_a_destination_created_during_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _commit, manifest = make_release_repository(tmp_path)
    packet = evaluate_journey_release(manifest, repo_root=root, signing_key=SIGNING_KEY)
    output = tmp_path / "racing-packet.json"

    def create_racing_destination(
        _source: str | os.PathLike[str], destination: str | os.PathLike[str]
    ) -> None:
        Path(destination).write_text("other writer", encoding="utf-8")
        raise FileExistsError("destination appeared")

    monkeypatch.setattr(os, "link", create_racing_destination)
    with pytest.raises(JourneyReleaseError, match="refusing to overwrite"):
        write_journey_release_packet(packet, output)
    assert output.read_text(encoding="utf-8") == "other writer"
    assert not tuple(tmp_path.glob(".racing-packet.json.tmp-*"))
