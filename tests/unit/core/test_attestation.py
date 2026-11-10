"""Tests for PHI-free, template-driven data-residency attestations."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
from jsonschema import validate

from openmed import AttestationReport
from openmed.core.attestation import (
    generate_attestation,
    list_attestation_profiles,
    load_attestation_template,
)
from openmed.core.audit import AuditReport, AuditSpan, DetectorInfo, hash_text
from openmed.core.offline import (
    HF_OFFLINE_ENV_VARS,
    OFFLINE_ENV_VAR,
    offline_mode_assertion,
)
from openmed.core.policy import load_policy
from openmed.core.schemas import load_schema

_SOURCE_TEXT = "Patient Amara Test, national ID RW-123-456-789."
_OUTPUT_TEXT = "Patient [NAME], national ID [ID]."
_GENERATED_AT = datetime(2026, 7, 18, 12, 0, tzinfo=timezone.utc)
_EXPECTED_PROFILES = (
    "rwanda-law-058-2021",
    "ethiopia-proclamation-1321-2024",
    "kenya-dpa-2019",
    "egypt-pdpl-151-2020",
)


def _audit_report(*, policy: str = "strict_no_leak") -> AuditReport:
    return AuditReport(
        policy=policy,
        resolved_profile={
            "method": "mask",
            "model_name": "synthetic-local-model",
            "offline_assertion": {
                "asserted": True,
                "local_only": True,
                "source": "config",
                "network_guard_requested": True,
                "dependency_flags_enabled": True,
                "environment_flags": {
                    OFFLINE_ENV_VAR: False,
                    **{name: True for name in HF_OFFLINE_ENV_VARS},
                },
            },
        },
        detectors=[
            DetectorInfo(
                source="ml",
                model_id="synthetic-local-model",
                model_format="transformers",
                commit="0123456789abcdef",
            )
        ],
        safety_sweep={"enabled": True, "spans_added": 0},
        spans=[
            AuditSpan(
                start=8,
                end=18,
                label="NAME",
                canonical_label="PERSON",
                sources=["ml"],
                confidence=0.99,
                threshold=0.7,
                action="mask",
                surrogate=None,
                text_hash=hash_text("Amara Test"),
            )
        ],
        thresholds={"PERSON": 0.7},
        residual_risk={"projected_leakage": 0.01},
        openmed_version="1.9.0",
        manifest_hash=hash_text("synthetic manifest"),
        document_length=len(_SOURCE_TEXT),
        input_hash=hash_text(_SOURCE_TEXT),
        deidentified_text_hash=hash_text(_OUTPUT_TEXT),
    )


@pytest.fixture
def model_artifact(tmp_path: Path) -> Path:
    path = tmp_path / "model.bin"
    path.write_bytes(b"synthetic model weights")
    return path


@pytest.mark.parametrize("profile", _EXPECTED_PROFILES)
def test_every_bundled_african_profile_generates_schema_valid_json_and_markdown(
    profile: str,
    model_artifact: Path,
) -> None:
    report = _audit_report()

    attestation = report.attest(
        profile,
        model_artifacts={"pii-model": model_artifact},
        generated_at=_GENERATED_AT,
    )
    payload = attestation.to_dict()
    markdown = attestation.to_markdown()

    validate(payload, load_schema("attestation"))
    assert json.loads(attestation.to_json()) == payload
    assert payload["profile_id"] == profile
    assert payload["policy"]["name"] == "strict_no_leak"
    assert payload["policy"]["posture"] == "maximum_recall_no_leak"
    assert payload["execution"]["offline_assertion"]["asserted"] is True
    assert "Decision-support evidence only" in markdown
    assert payload["jurisdiction"]["residency_statement"] in markdown
    assert attestation.repro_hash_matches()
    assert attestation.integrity_hash_matches()


def test_bundled_profile_list_is_stable() -> None:
    assert list_attestation_profiles() == _EXPECTED_PROFILES


def test_policy_loader_records_exact_profile_and_fails_cleanly(
    model_artifact: Path,
) -> None:
    payload = generate_attestation(
        _audit_report(),
        _EXPECTED_PROFILES[0],
        model_artifacts={"pii-model": model_artifact},
        generated_at=_GENERATED_AT,
    ).to_dict()

    loaded = load_policy("strict_no_leak")
    assert payload["policy"] == {
        "name": loaded.name,
        "posture": loaded.posture,
        "schema_version": loaded.schema_version,
        "default_action": loaded.default_action,
    }

    with pytest.raises(ValueError, match="could not be loaded"):
        generate_attestation(
            _audit_report(policy="missing_africa_policy"),
            _EXPECTED_PROFILES[0],
            model_artifacts={"pii-model": model_artifact},
        )

    with pytest.raises(ValueError, match="requires policy 'strict_no_leak'"):
        generate_attestation(
            _audit_report(policy="gdpr_art9_health"),
            _EXPECTED_PROFILES[0],
            model_artifacts={"pii-model": model_artifact},
        )


def test_repro_hash_is_timestamp_independent_but_integrity_hash_is_not(
    model_artifact: Path,
) -> None:
    report = _audit_report()
    first = report.attest(
        _EXPECTED_PROFILES[0],
        model_artifacts={"pii-model": model_artifact},
        generated_at="2026-07-18T12:00:00Z",
    )
    second = report.attest(
        _EXPECTED_PROFILES[0],
        model_artifacts={"pii-model": model_artifact},
        generated_at="2026-07-18T13:00:00+00:00",
    )

    assert first.to_dict()["repro_hash"] == second.to_dict()["repro_hash"]
    assert first.to_dict()["integrity_hash"] != second.to_dict()["integrity_hash"]

    restored = AttestationReport.from_json(first.to_json())
    assert restored.repro_hash_matches()
    assert restored.integrity_hash_matches()
    tampered = restored.to_dict()
    tampered["run"]["span_count"] += 1
    assert not AttestationReport.from_dict(tampered).repro_hash_matches()
    assert not AttestationReport.from_dict(tampered).integrity_hash_matches()

    tampered_audit = _audit_report()
    tampered_audit.document_length += 1
    with pytest.raises(ValueError, match="Audit report repro hash does not match"):
        tampered_audit.attest(
            _EXPECTED_PROFILES[0],
            model_artifacts={"pii-model": model_artifact},
        )


def test_offline_assertion_requires_openmed_and_dependency_flags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(OFFLINE_ENV_VAR, raising=False)
    for name in HF_OFFLINE_ENV_VARS:
        monkeypatch.delenv(name, raising=False)

    assert offline_mode_assertion()["asserted"] is False

    for name in HF_OFFLINE_ENV_VARS:
        monkeypatch.setenv(name, "1")
    dependency_only = offline_mode_assertion()
    assert dependency_only["dependency_flags_enabled"] is True
    assert dependency_only["network_guard_requested"] is False
    assert dependency_only["asserted"] is False

    monkeypatch.setenv(OFFLINE_ENV_VAR, "true")
    complete = offline_mode_assertion()
    assert complete["source"] == "environment"
    assert complete["network_guard_requested"] is True
    assert complete["asserted"] is True


def test_attestation_never_substitutes_generation_environment_for_run_evidence(
    monkeypatch: pytest.MonkeyPatch,
    model_artifact: Path,
) -> None:
    monkeypatch.setenv(OFFLINE_ENV_VAR, "1")
    for name in HF_OFFLINE_ENV_VARS:
        monkeypatch.setenv(name, "1")
    report = _audit_report()
    report.resolved_profile.pop("offline_assertion")
    report.repro_hash = report.recompute_repro_hash()

    payload = report.attest(
        _EXPECTED_PROFILES[0],
        model_artifacts={"pii-model": model_artifact},
        generated_at=_GENERATED_AT,
    ).to_dict()

    assertion = payload["execution"]["offline_assertion"]
    assert assertion["asserted"] is False
    assert assertion["local_only"] is False
    assert assertion["source"] == "none"


def test_attestation_downgrades_inconsistent_offline_evidence(
    model_artifact: Path,
) -> None:
    report = _audit_report()
    report.resolved_profile["offline_assertion"] = {
        "asserted": True,
        "local_only": True,
        "source": "untrusted",
        "network_guard_requested": True,
        "dependency_flags_enabled": True,
        "environment_flags": {
            OFFLINE_ENV_VAR: False,
            **{name: False for name in HF_OFFLINE_ENV_VARS},
        },
    }
    report.repro_hash = report.recompute_repro_hash()

    payload = report.attest(
        _EXPECTED_PROFILES[0],
        model_artifacts={"pii-model": model_artifact},
        generated_at=_GENERATED_AT,
    ).to_dict()

    assertion = payload["execution"]["offline_assertion"]
    assert assertion["asserted"] is False
    assert assertion["network_guard_requested"] is False
    assert assertion["dependency_flags_enabled"] is False
    assert assertion["source"] == "none"
    validate(payload, load_schema("attestation"))


def test_template_controls_jurisdiction_rendering(
    tmp_path: Path,
    model_artifact: Path,
) -> None:
    template = load_attestation_template()
    template["profiles"][0]["attestation_title"] = "Controlled local title"
    template["profiles"][0]["residency_statement"] = (
        "Controlled jurisdiction wording from JSON."
    )
    path = tmp_path / "template.json"
    path.write_text(json.dumps(template), encoding="utf-8")

    attestation = generate_attestation(
        _audit_report(),
        _EXPECTED_PROFILES[0],
        model_artifacts={"pii-model": model_artifact},
        generated_at=_GENERATED_AT,
        template_path=path,
    )

    assert "# Controlled local title" in attestation.to_markdown()
    assert "Controlled jurisdiction wording from JSON." in attestation.to_markdown()
    assert (
        attestation.to_dict()["jurisdiction"]["residency_statement"]
        == "Controlled jurisdiction wording from JSON."
    )


def test_guide_covers_each_profile_policy_and_template_field() -> None:
    repository_root = Path(__file__).resolve().parents[3]
    guide = (
        repository_root / "docs/compliance/africa-data-residency-guide.md"
    ).read_text(encoding="utf-8")
    docs_template = json.loads(
        (
            repository_root
            / "docs/compliance/templates/africa-data-residency-attestation.json"
        ).read_text(encoding="utf-8")
    )
    runtime_template = load_attestation_template()

    assert docs_template == runtime_template
    assert "decision-support material, not legal advice" in guide
    for profile in runtime_template["profiles"]:
        loaded = load_policy(profile["policy_profile"])
        assert loaded.name == profile["policy_profile"]
        assert f"## {profile['jurisdiction']}" in guide
        assert f"Attestation profile: `{profile['id']}`" in guide
        assert f"Policy profile: `{loaded.name}`" in guide
        assert f"Template field: `{profile['guide_field']}`" in guide


def test_attestation_artifacts_do_not_leak_source_text_or_identifiers(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "model-with-sensitive-looking-test-bytes.bin"
    artifact.write_text(
        _SOURCE_TEXT + " amara.test@example.invalid +250 788 123 456",
        encoding="utf-8",
    )
    attestation = _audit_report().attest(
        _EXPECTED_PROFILES[0],
        model_artifacts={"pii-model": artifact},
        generated_at=_GENERATED_AT,
    )
    serialized = attestation.to_json() + attestation.to_markdown()

    for secret in (
        _SOURCE_TEXT,
        "Amara Test",
        "RW-123-456-789",
        "amara.test@example.invalid",
        "+250 788 123 456",
    ):
        assert secret not in serialized
    payload = attestation.to_dict()
    assert payload["run"]["document_length"] == len(_SOURCE_TEXT)
    assert payload["run"]["span_count"] == 1
    assert payload["run"]["input_hash"] == hash_text(_SOURCE_TEXT)


def test_missing_model_artifact_fails_without_network_access(tmp_path: Path) -> None:
    missing = tmp_path / "missing-model.bin"

    with pytest.raises(ValueError, match="does not exist"):
        _audit_report().attest(
            _EXPECTED_PROFILES[0],
            model_artifacts={"pii-model": missing},
        )


def test_directory_checksum_is_content_and_path_stable(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    weights = model_dir / "weights.bin"
    weights.write_bytes(b"weights-v1")

    first = _audit_report().attest(
        _EXPECTED_PROFILES[0],
        model_artifacts={"pii-model": model_dir},
        generated_at=_GENERATED_AT,
    )
    second = _audit_report().attest(
        _EXPECTED_PROFILES[0],
        model_artifacts={"pii-model": model_dir},
        generated_at=_GENERATED_AT,
    )
    assert first.to_dict()["models"] == second.to_dict()["models"]
    assert first.to_dict()["models"][0]["file_count"] == 2

    weights.write_bytes(b"weights-v2")
    changed = _audit_report().attest(
        _EXPECTED_PROFILES[0],
        model_artifacts={"pii-model": model_dir},
        generated_at=_GENERATED_AT,
    )
    assert (
        changed.to_dict()["models"][0]["checksum"]
        != first.to_dict()["models"][0]["checksum"]
    )
