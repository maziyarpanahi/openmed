"""CLI tests for structured QI discovery, assessment, and anonymization."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

import openmed.compliance as compliance_module
from openmed.cli import main_module
from openmed.compliance import (
    ExpertReviewEvidenceReport,
    create_expert_attestation,
)
from openmed.core.audit import stable_hash
from openmed.structured import read_table

ROWS = [
    {
        "patient_id": "patient-alpha",
        "full_name": "Alice Canary",
        "age": "30",
        "zip": "10001",
        "visit_date": "2024-01-01",
        "disease": "flu-canary",
    },
    {
        "patient_id": "patient-beta",
        "full_name": "Bob Canary",
        "age": "30",
        "zip": "10001",
        "visit_date": "2024-01-01",
        "disease": "cold-canary",
    },
    {
        "patient_id": "patient-gamma",
        "full_name": "Carol Canary",
        "age": "40",
        "zip": "20001",
        "visit_date": "2024-02-01",
        "disease": "flu-canary",
    },
    {
        "patient_id": "patient-delta",
        "full_name": "Dan Canary",
        "age": "40",
        "zip": "20001",
        "visit_date": "2024-02-01",
        "disease": "cold-canary",
    },
]


def _write_csv(path: Path, rows=ROWS) -> Path:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def _write_bom_csv(path: Path, rows=ROWS) -> Path:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def _policy_args() -> list[str]:
    return [
        "--qi",
        "age,zip,visit_date",
        "--sensitive",
        "disease",
        "--direct-id",
        "full_name",
        "--privacy-unit",
        "patient_id",
        "--k",
        "2",
        "--l",
        "2",
        "--t",
        "0",
    ]


def _anonymize_args(
    source: Path,
    release: Path,
    evidence: Path,
    *,
    markdown: Path | None = None,
    hierarchies: Path | None = None,
    overwrite: bool = False,
    policy_args: list[str] | None = None,
    privacy_unit_kind: str | None = "patient",
    assumptions_notes: Path | None = None,
) -> list[str]:
    args = [
        "risk",
        "anonymize",
        str(source),
        "--output",
        str(release),
        "--evidence",
        str(evidence),
        *(policy_args or _policy_args()),
    ]
    if markdown is not None:
        args.extend(("--evidence-markdown", str(markdown)))
    if hierarchies is not None:
        args.extend(("--hierarchies", str(hierarchies)))
    if privacy_unit_kind is not None:
        args.extend(("--privacy-unit-kind", privacy_unit_kind))
    if assumptions_notes is not None:
        args.extend(("--assumptions-notes", str(assumptions_notes)))
    args.extend(
        (
            "--release-model",
            "restricted",
            "--recipient-model",
            "named_researchers",
            "--auxiliary-data-model",
            "reasonably_available",
        )
    )
    if overwrite:
        args.append("--overwrite")
    return args


def test_risk_discover_writes_aggregate_manifest_without_path_or_values(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = _write_csv(tmp_path / "source-canary.csv")
    output = tmp_path / "discovery.json"

    code = main_module.main(
        [
            "risk",
            "discover",
            str(source),
            "--output",
            str(output),
            "--full-scan",
            "--privacy-unit",
            "patient_id",
            "--qi",
            "age,zip,visit_date",
            "--sensitive",
            "disease",
        ]
    )

    assert code == 0
    assert "Quasi-identifier discovery complete" in capsys.readouterr().out
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["sample"]["complete"] is True
    assert payload["sample"]["advisory"] is False
    assert payload["discovery"]["status"] == "candidates-found"
    serialized = json.dumps(payload, sort_keys=True)
    for canary in (
        "source-canary.csv",
        "patient-alpha",
        "Alice Canary",
        "10001",
        "flu-canary",
    ):
        assert canary not in serialized
    assert "key_fingerprints" not in serialized


def test_risk_discover_can_scan_safe_combination_candidates(
    tmp_path: Path,
) -> None:
    source = _write_csv(
        tmp_path / "factor-design.csv",
        [
            {
                "factor_a": f"a-{index // 10}",
                "factor_b": f"b-{index % 10}",
            }
            for index in range(100)
        ],
    )
    output = tmp_path / "discovery.json"

    code = main_module.main(
        [
            "risk",
            "discover",
            str(source),
            "--output",
            str(output),
            "--full-scan",
            "--max-set-size",
            "2",
            "--max-candidate-columns",
            "2",
            "--include-safe-candidates",
        ]
    )

    assert code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["search"]["candidate_scope"] == "all_reviewed_scalar_columns"
    assert any(
        candidate["columns"] == ["factor_a", "factor_b"]
        and candidate["min_equivalence_class_size"] == 1
        for candidate in payload["quasi_identifier_sets"]
    )


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("--privacy-unit", "not_a_column"),
        ("--qi", "not_a_column"),
        ("--sensitive", "not_a_column"),
        ("--role", "not_a_column=safe"),
    ],
)
def test_risk_discover_rejects_unknown_explicit_columns_as_usage_error(
    option: str,
    value: str,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = _write_csv(tmp_path / "source.csv")
    output = tmp_path / "discovery.json"

    code = main_module.main(
        [
            "risk",
            "discover",
            str(source),
            "--output",
            str(output),
            option,
            value,
        ]
    )

    assert code == 2
    error = capsys.readouterr().err
    assert "configuration does not match the input schema" in error
    assert "not_a_column" not in error
    assert not output.exists()


def test_risk_discover_rejects_duplicate_role_overrides_without_echo(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = _write_csv(tmp_path / "source.csv")
    output = tmp_path / "discovery.json"
    canary_column = "patient_id"

    code = main_module.main(
        [
            "risk",
            "discover",
            str(source),
            "--output",
            str(output),
            "--role",
            f"{canary_column}=direct-id",
            "--role",
            f"{canary_column}=safe",
        ]
    )

    assert code == 2
    error = capsys.readouterr().err
    assert "only one role override" in error
    assert canary_column not in error
    assert not output.exists()


@pytest.mark.parametrize(
    ("suffix", "payload"),
    [
        (".csv", "age,disease\n30,flu,sensitive-overflow-canary\n"),
        (
            ".jsonl",
            '{"age":{"nested":"sensitive-nested-canary"}}\n',
        ),
    ],
)
def test_risk_discover_rejects_malformed_rows_without_echoing_values(
    suffix: str,
    payload: str,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = tmp_path / f"malformed{suffix}"
    source.write_text(payload, encoding="utf-8")
    output = tmp_path / "discovery.json"

    code = main_module.main(
        [
            "risk",
            "discover",
            str(source),
            "--output",
            str(output),
            "--full-scan",
        ]
    )

    assert code == 1
    error = capsys.readouterr().err
    assert "Failed to discover structured quasi-identifiers" in error
    assert "sensitive-" not in error
    assert not output.exists()


def test_risk_assess_writes_safe_patient_level_report(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = _write_csv(tmp_path / "source.csv")
    output = tmp_path / "assessment.json"

    code = main_module.main(
        [
            "risk",
            "assess",
            str(source),
            "--output",
            str(output),
            *_policy_args(),
        ]
    )

    assert code == 0
    assert "Meets configured policy: PASS" in capsys.readouterr().out
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["privacy_unit_count"] == 4
    assert payload["k_anonymity"]["achieved_k"] == 2
    assert payload["meets_policy"] is True
    serialized = json.dumps(payload, sort_keys=True)
    assert "patient-alpha" not in serialized
    assert "Alice Canary" not in serialized
    assert "equivalence_classes" not in serialized


def test_risk_assess_can_publish_safe_html_dashboard(tmp_path: Path) -> None:
    source = _write_csv(tmp_path / "source.csv")
    output = tmp_path / "assessment.json"
    dashboard = tmp_path / "assessment.html"

    code = main_module.main(
        [
            "risk",
            "assess",
            str(source),
            "--output",
            str(output),
            "--dashboard",
            str(dashboard),
            *_policy_args(),
        ]
    )

    assert code == 0
    rendered = dashboard.read_text(encoding="utf-8")
    assert "<!doctype html>" in rendered.casefold()
    assert "patient-alpha" not in rendered
    assert "Alice Canary" not in rendered
    assert "10001" not in rendered


def test_risk_assess_repeatable_flags_support_literal_schema_names(
    tmp_path: Path,
) -> None:
    source = _write_csv(
        tmp_path / "literal-columns.csv",
        [
            {
                "患者 ID": "patient-a",
                "Patient Age": "40",
                "Région, cohort | `v1`": "Île-de-France",
                "study arm": "A",
            },
            {
                "患者 ID": "patient-b",
                "Patient Age": "40",
                "Région, cohort | `v1`": "Île-de-France",
                "study arm": "B",
            },
        ],
    )
    output = tmp_path / "assessment.json"
    dashboard = tmp_path / "assessment.html"

    code = main_module.main(
        [
            "risk",
            "assess",
            str(source),
            "--output",
            str(output),
            "--dashboard",
            str(dashboard),
            "--qi-column",
            "Patient Age",
            "--qi-column",
            "Région, cohort | `v1`",
            "--non-sensitive-column",
            "study arm",
            "--privacy-unit",
            "患者 ID",
            "--k",
            "2",
        ]
    )

    assert code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["meets_policy"] is True
    assert payload["quasi_identifiers"] == [
        "Patient Age",
        "Région, cohort | `v1`",
    ]
    assert "patient-a" not in json.dumps(payload, sort_keys=True)
    rendered = dashboard.read_text(encoding="utf-8")
    assert "Patient Age" in rendered
    assert "Région, cohort | `v1`" in rendered
    assert "patient-a" not in rendered
    assert "Île-de-France" not in rendered


def test_risk_assess_accepts_bom_prefixed_seven_column_csv(tmp_path: Path) -> None:
    rows = [
        {**row, "encounter_id": f"encounter-{index}"}
        for index, row in enumerate(ROWS, start=1)
    ]
    source = _write_bom_csv(tmp_path / "bom-source.csv", rows)
    output = tmp_path / "assessment.json"

    code = main_module.main(
        [
            "risk",
            "assess",
            str(source),
            "--output",
            str(output),
            "--qi",
            "age,zip,visit_date",
            "--sensitive",
            "disease",
            "--direct-id",
            "full_name,encounter_id",
            "--privacy-unit",
            "patient_id",
            "--k",
            "2",
            "--l",
            "2",
            "--t",
            "0",
        ]
    )

    assert code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["meets_policy"] is True
    assert payload["privacy_unit_count"] == len(ROWS)


def test_risk_discover_strips_bom_from_csv_schema(tmp_path: Path) -> None:
    source = _write_bom_csv(tmp_path / "bom-source.csv")
    output = tmp_path / "discovery.json"

    code = main_module.main(
        [
            "risk",
            "discover",
            str(source),
            "--output",
            str(output),
            "--full-scan",
        ]
    )

    assert code == 0
    serialized = output.read_text(encoding="utf-8")
    assert '"patient_id"' in serialized
    assert "\\ufeffpatient_id" not in serialized


def test_risk_population_assess_writes_aggregate_safe_evidence(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    sample = _write_csv(
        tmp_path / "sample.csv",
        [{"sample_id": "sample-canary", "age": "40", "region": "north"}],
    )
    reference = _write_csv(
        tmp_path / "reference.csv",
        [
            {"population_id": "population-a", "age": "40", "region": "north"},
            {"population_id": "population-b", "age": "40", "region": "north"},
        ],
    )
    output = tmp_path / "population-risk.json"

    code = main_module.main(
        [
            "risk",
            "population-assess",
            str(sample),
            str(reference),
            "--output",
            str(output),
            "--qi",
            "age,region",
            "--sample-privacy-unit",
            "sample_id",
            "--population-privacy-unit",
            "population_id",
            "--k-map",
            "2",
            "--max-delta-presence",
            "0.5",
        ]
    )

    assert code == 0
    assert "Achieved k-map: 2" in capsys.readouterr().out
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["meets_policy"] is True
    assert payload["achieved_k_map"] == 2
    serialized = json.dumps(payload, sort_keys=True)
    for canary in (
        "sample-canary",
        "population-a",
        "north",
        "sample_id",
        "population_id",
    ):
        assert canary not in serialized


def test_risk_population_assess_fails_closed_but_preserves_safe_evidence(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    sample = _write_csv(tmp_path / "sample.csv", [{"age": "canary-sample"}])
    reference = _write_csv(
        tmp_path / "reference.csv",
        [{"age": "canary-reference"}],
    )
    output = tmp_path / "population-risk.json"

    code = main_module.main(
        [
            "risk",
            "population-assess",
            str(sample),
            str(reference),
            "--output",
            str(output),
            "--qi",
            "age",
            "--k-map",
            "2",
            "--max-delta-presence",
            "0.5",
        ]
    )

    assert code == 1
    assert output.exists()
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["meets_policy"] is False
    assert payload["achieved_k_map"] == 0
    error = capsys.readouterr().err
    assert "does not meet" in error
    assert "canary-sample" not in error
    assert "canary-reference" not in error


def test_risk_population_assess_requires_explicit_policy_thresholds(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    sample = _write_csv(tmp_path / "sample.csv", [{"age": "40"}])
    reference = _write_csv(tmp_path / "reference.csv", [{"age": "40"}])
    output = tmp_path / "population-risk.json"

    with pytest.raises(SystemExit) as exc_info:
        main_module.main(
            [
                "risk",
                "population-assess",
                str(sample),
                str(reference),
                "--output",
                str(output),
                "--qi",
                "age",
            ]
        )

    assert exc_info.value.code == 2
    error = capsys.readouterr().err
    assert "--k-map" in error
    assert "--max-delta-presence" in error
    assert not output.exists()


def test_risk_assess_writes_evidence_but_returns_nonzero_when_policy_fails(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = _write_csv(tmp_path / "source.csv")
    output = tmp_path / "assessment.json"
    policy_args = _policy_args()
    policy_args[policy_args.index("--k") + 1] = "3"

    code = main_module.main(
        [
            "risk",
            "assess",
            str(source),
            "--output",
            str(output),
            *policy_args,
        ]
    )

    assert code == 1
    assert "does not meet" in capsys.readouterr().err
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["meets_policy"] is False
    assert payload["k_anonymity"]["achieved_k"] == 2
    assert "patient-alpha" not in json.dumps(payload, sort_keys=True)


def test_risk_anonymize_writes_validated_release_and_expert_evidence(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = _write_csv(tmp_path / "source.csv")
    release = tmp_path / "release.jsonl"
    evidence = tmp_path / "evidence.json"
    dashboard = tmp_path / "release-dashboard.html"

    code = main_module.main(
        [
            "risk",
            "anonymize",
            str(source),
            "--output",
            str(release),
            "--evidence",
            str(evidence),
            "--dashboard",
            str(dashboard),
            *_policy_args(),
            "--privacy-unit-kind",
            "patient",
            "--release-model",
            "restricted",
            "--recipient-model",
            "named_researchers",
            "--auxiliary-data-model",
            "reasonably_available",
        ]
    )

    assert code == 0
    output = capsys.readouterr().out
    assert "Meets configured policy: PASS" in output
    released = read_table(release)
    assert len(released) == 4
    assert all("patient_id" not in row for row in released)
    assert all("full_name" not in row for row in released)
    report = ExpertReviewEvidenceReport.from_json(evidence.read_text(encoding="utf-8"))
    assert report.verify() is True
    assert report.privacy_models.achieved_k == 2
    assert evidence.with_suffix(".md").exists()
    rendered_dashboard = dashboard.read_text(encoding="utf-8")
    assert "<!doctype html>" in rendered_dashboard.casefold()
    assert "patient-alpha" not in rendered_dashboard
    assert "Alice Canary" not in rendered_dashboard
    assert "10001" not in rendered_dashboard
    serialized = evidence.read_text(encoding="utf-8")
    assert "patient-alpha" not in serialized
    assert "Alice Canary" not in serialized
    assert '"records"' not in serialized

    verify_code = main_module.main(
        ["compliance", "expert-review-verify", str(evidence)]
    )
    assert verify_code == 0
    assert "verification: PASS" in capsys.readouterr().out


def test_risk_anonymize_accepts_bom_with_repeatable_policy_columns(
    tmp_path: Path,
) -> None:
    rows = [
        {**row, "encounter_id": f"encounter-{index}"}
        for index, row in enumerate(ROWS, start=1)
    ]
    source = _write_bom_csv(tmp_path / "bom-source.csv", rows)
    release = tmp_path / "release.jsonl"
    evidence = tmp_path / "evidence.json"
    policy_args = [
        "--qi-column",
        "age",
        "--qi-column",
        "zip",
        "--qi-column",
        "visit_date",
        "--sensitive-column",
        "disease",
        "--direct-id-column",
        "full_name",
        "--direct-id-column",
        "encounter_id",
        "--privacy-unit",
        "patient_id",
        "--k",
        "2",
        "--l",
        "2",
        "--t",
        "0",
    ]

    code = main_module.main(
        _anonymize_args(
            source,
            release,
            evidence,
            policy_args=policy_args,
        )
    )

    assert code == 0
    released = read_table(release)
    assert len(released) == len(ROWS)
    assert all("patient_id" not in row for row in released)
    assert all("encounter_id" not in row for row in released)
    assert all("full_name" not in row for row in released)


def test_other_release_context_requires_reviewed_assumptions_notes(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = _write_csv(tmp_path / "source.csv")
    release = tmp_path / "release.jsonl"
    evidence = tmp_path / "evidence.json"
    args = _anonymize_args(source, release, evidence)
    args[args.index("--release-model") + 1] = "other_documented"

    code = main_module.main(args)

    assert code == 2
    assert "--assumptions-notes" in capsys.readouterr().err
    assert not release.exists()
    assert not evidence.exists()


def test_reviewed_assumptions_notes_are_digest_bound_without_content_disclosure(
    tmp_path: Path,
) -> None:
    source = _write_csv(tmp_path / "source.csv")
    release = tmp_path / "release.jsonl"
    evidence = tmp_path / "evidence.json"
    notes = tmp_path / "reviewed-assumptions.md"
    notes.write_text(
        "Reviewed context detail assumptions-notes-canary",
        encoding="utf-8",
    )
    args = _anonymize_args(
        source,
        release,
        evidence,
        assumptions_notes=notes,
    )
    args[args.index("--release-model") + 1] = "other_documented"

    code = main_module.main(args)

    assert code == 0
    report = ExpertReviewEvidenceReport.from_json(evidence.read_text(encoding="utf-8"))
    assert report.verify()
    assert "assumptions-notes-canary" not in evidence.read_text(encoding="utf-8")


def test_expert_review_verify_rejects_duplicate_json_keys_without_echo(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = _write_csv(tmp_path / "source.csv")
    release = tmp_path / "release.jsonl"
    evidence = tmp_path / "evidence.json"
    assert (
        main_module.main(
            _anonymize_args(
                source,
                release,
                evidence,
            )
        )
        == 0
    )
    capsys.readouterr()
    rendered = evidence.read_text(encoding="utf-8")
    canary = "discarded-sensitive-canary"
    evidence.write_text(
        '{"title":"' + canary + '",' + rendered.lstrip()[1:],
        encoding="utf-8",
    )

    code = main_module.main(["compliance", "expert-review-verify", str(evidence)])

    assert code == 1
    error = capsys.readouterr().err
    assert "verification failed" in error
    assert canary not in error


def test_risk_gate_returns_ci_friendly_pass_and_policy_failure(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = _write_csv(tmp_path / "source.csv")
    release = tmp_path / "release.jsonl"
    evidence = tmp_path / "evidence.json"

    assert main_module.main(_anonymize_args(source, release, evidence)) == 0
    capsys.readouterr()

    assert main_module.main(["risk", "gate", str(evidence)]) == 0
    passed_output = capsys.readouterr().out
    assert "Technical policy: PASS" in passed_output
    assert "not an Expert Determination" in passed_output

    payload = json.loads(evidence.read_text(encoding="utf-8"))
    payload["privacy_models"]["k_anonymity"]["configured_k"] = 3
    payload["integrity_hash"] = stable_hash(
        {key: value for key, value in payload.items() if key != "integrity_hash"}
    )
    evidence.write_text(json.dumps(payload), encoding="utf-8")

    assert main_module.main(["risk", "gate", str(evidence)]) == 1
    failed_output = capsys.readouterr().out
    assert "Technical policy: FAIL" in failed_output
    assert "violates its configured technical policy" in failed_output


def test_expert_attestation_verify_reports_independent_facts_and_fails_mismatch(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    pytest.importorskip("cryptography")
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PrivateKey,
    )

    source = _write_csv(tmp_path / "source.csv")
    release = tmp_path / "release.jsonl"
    evidence_path = tmp_path / "evidence.json"
    assert main_module.main(_anonymize_args(source, release, evidence_path)) == 0
    capsys.readouterr()

    evidence = ExpertReviewEvidenceReport.from_json(
        evidence_path.read_text(encoding="utf-8")
    )
    private_key = Ed25519PrivateKey.generate()
    public_key_path = tmp_path / "expert-public.pem"
    public_key_path.write_bytes(
        private_key.public_key().public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )
    now = datetime.now(timezone.utc)
    supporting_digest = stable_hash({"kind": "population-risk-test"})
    attestation = create_expert_attestation(
        evidence,
        expert_identity="Dr. Taylor Example",
        qualifications="Independent statistical disclosure-control expert",
        scope_and_methodology=(
            "Reviewed the declared release context, technical evidence, "
            "reference-population evidence, and residual risk."
        ),
        conclusion="very_small_risk",
        issued_at=now - timedelta(hours=1),
        reassessment_at=now + timedelta(days=365),
        private_key=private_key,
        key_id="expert-key-2026",
        supporting_evidence_digests={"population_risk": supporting_digest},
    )
    attestation_path = tmp_path / "attestation.json"
    attestation_path.write_text(attestation.to_json() + "\n", encoding="utf-8")
    common_args = [
        "compliance",
        "expert-attestation-verify",
        str(attestation_path),
        "--evidence",
        str(evidence_path),
        "--public-key",
        str(public_key_path),
        "--key-id",
        "expert-key-2026",
        "--supporting-evidence",
        f"population_risk={supporting_digest}",
    ]

    assert main_module.main([*common_args, "--json"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["data"] == {
        "bindings_match": True,
        "conclusion": "very_small_risk",
        "cryptographically_valid": True,
        "evidence_integrity_valid": True,
        "fresh": True,
        "freshness_status": "current",
        "key_id_matches": True,
    }
    assert "approved" not in json.dumps(output, sort_keys=True)

    mismatched_args = list(common_args)
    mismatched_args[-1] = f"population_risk={stable_hash({'kind': 'wrong'})}"
    assert main_module.main(mismatched_args) == 1
    mismatched_output = capsys.readouterr().out
    assert "Evidence bindings: FAIL" in mismatched_output
    assert "not an automated Expert Determination" in mismatched_output


def test_risk_anonymize_fails_closed_on_lattice_exhaustion_without_echoing_data(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = _write_csv(tmp_path / "source.csv")
    release = tmp_path / "release.jsonl"
    evidence = tmp_path / "evidence.json"

    code = main_module.main(
        [
            "risk",
            "anonymize",
            str(source),
            "--output",
            str(release),
            "--evidence",
            str(evidence),
            *_policy_args(),
            "--privacy-unit-kind",
            "patient",
            "--max-lattice-nodes",
            "1",
            "--release-model",
            "restricted",
            "--recipient-model",
            "named_researchers",
            "--auxiliary-data-model",
            "reasonably_available",
        ]
    )

    assert code == 1
    error = capsys.readouterr().err
    assert "Failed to anonymize" in error
    assert "patient-alpha" not in error
    assert "Alice Canary" not in error
    assert not release.exists()
    assert not evidence.exists()


@pytest.mark.parametrize("suffix", [".csv", ".tsv"])
def test_risk_anonymize_validates_delimited_output_lexically(
    tmp_path: Path,
    suffix: str,
) -> None:
    source = _write_csv(tmp_path / "source.csv")
    release = tmp_path / f"release{suffix}"
    evidence = tmp_path / "evidence.json"

    code = main_module.main(
        [
            "risk",
            "anonymize",
            str(source),
            "--output",
            str(release),
            "--evidence",
            str(evidence),
            *_policy_args(),
            "--privacy-unit-kind",
            "patient",
            "--release-model",
            "restricted",
            "--recipient-model",
            "named_researchers",
            "--auxiliary-data-model",
            "reasonably_available",
        ]
    )

    assert code == 0
    assert len(read_table(release)) == 4
    assert ExpertReviewEvidenceReport.from_json(
        evidence.read_text(encoding="utf-8")
    ).verify()


def test_risk_anonymize_does_not_publish_release_if_evidence_build_fails(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _write_csv(tmp_path / "source.csv")
    release = tmp_path / "release.jsonl"
    evidence = tmp_path / "evidence.json"

    def fail_evidence(*args, **kwargs):
        raise ValueError("synthetic evidence failure")

    monkeypatch.setattr(
        compliance_module,
        "build_release_expert_review_evidence",
        fail_evidence,
    )
    code = main_module.main(
        [
            "risk",
            "anonymize",
            str(source),
            "--output",
            str(release),
            "--evidence",
            str(evidence),
            *_policy_args(),
            "--privacy-unit-kind",
            "patient",
            "--release-model",
            "restricted",
            "--recipient-model",
            "named_researchers",
            "--auxiliary-data-model",
            "reasonably_available",
        ]
    )

    assert code == 1
    assert "Failed to anonymize" in capsys.readouterr().err
    assert not release.exists()
    assert not evidence.exists()
    assert not evidence.with_suffix(".md").exists()
    assert not [path for path in tmp_path.iterdir() if path.name.startswith(".")]


def test_risk_anonymize_rejects_overlapping_output_paths(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = _write_csv(tmp_path / "source.csv")
    output = tmp_path / "release.jsonl"

    code = main_module.main(
        [
            "risk",
            "anonymize",
            str(source),
            "--output",
            str(output),
            "--evidence",
            str(output),
            *_policy_args(),
            "--privacy-unit-kind",
            "patient",
            "--release-model",
            "restricted",
            "--recipient-model",
            "named_researchers",
            "--auxiliary-data-model",
            "reasonably_available",
        ]
    )

    assert code == 2
    assert "must be distinct" in capsys.readouterr().err
    assert not output.exists()


@pytest.mark.parametrize("command", ["discover", "assess"])
def test_structured_commands_reject_symlink_aliases_with_overwrite(
    command: str,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = _write_csv(tmp_path / "source.csv")
    original = source.read_bytes()
    output = tmp_path / "source-alias.json"
    output.symlink_to(source)
    args = ["risk", command, str(source), "--output", str(output)]
    if command == "assess":
        args.extend(_policy_args())
    args.append("--overwrite")

    code = main_module.main(args)

    assert code == 2
    assert "distinct after resolving symlinks" in capsys.readouterr().err
    assert source.read_bytes() == original
    assert output.is_symlink()


@pytest.mark.parametrize(
    "alias_role",
    ["release", "evidence", "markdown", "hierarchies"],
)
def test_risk_anonymize_rejects_source_aliases_for_every_path_role(
    alias_role: str,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = _write_csv(tmp_path / "source.csv")
    original = source.read_bytes()
    release = tmp_path / "release.jsonl"
    evidence = tmp_path / "evidence.json"
    markdown = tmp_path / "evidence.md"
    hierarchies = tmp_path / "hierarchies.json"
    aliases = {
        "release": release,
        "evidence": evidence,
        "markdown": markdown,
        "hierarchies": hierarchies,
    }
    aliases[alias_role].symlink_to(source)

    code = main_module.main(
        _anonymize_args(
            source,
            release,
            evidence,
            markdown=markdown,
            hierarchies=hierarchies if alias_role == "hierarchies" else None,
            overwrite=True,
        )
    )

    assert code == 2
    assert "distinct after resolving symlinks" in capsys.readouterr().err
    assert source.read_bytes() == original
    assert aliases[alias_role].is_symlink()


def test_risk_anonymize_rejects_broken_symlink_alias_between_outputs(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = _write_csv(tmp_path / "source.csv")
    release = tmp_path / "release.jsonl"
    evidence = tmp_path / "evidence.json"
    markdown = tmp_path / "evidence.md"
    evidence.symlink_to(release)

    code = main_module.main(
        _anonymize_args(
            source,
            release,
            evidence,
            markdown=markdown,
            overwrite=True,
        )
    )

    assert code == 2
    assert "distinct after resolving symlinks" in capsys.readouterr().err
    assert evidence.is_symlink()
    assert not release.exists()
    assert not markdown.exists()


def test_risk_anonymize_rejects_hierarchy_alias_with_output(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = _write_csv(tmp_path / "source.csv")
    hierarchies = tmp_path / "hierarchies.json"
    hierarchies.write_text("{}", encoding="utf-8")
    release = tmp_path / "release.jsonl"
    release.symlink_to(hierarchies)
    evidence = tmp_path / "evidence.json"
    markdown = tmp_path / "evidence.md"

    code = main_module.main(
        _anonymize_args(
            source,
            release,
            evidence,
            markdown=markdown,
            hierarchies=hierarchies,
            overwrite=True,
        )
    )

    assert code == 2
    assert "distinct after resolving symlinks" in capsys.readouterr().err
    assert release.is_symlink()
    assert hierarchies.read_text(encoding="utf-8") == "{}"


@pytest.mark.parametrize("failed_target", ["evidence", "markdown", "release"])
def test_risk_anonymize_replacement_failure_leaves_no_partial_publication(
    failed_target: str,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _write_csv(tmp_path / "source.csv")
    release = tmp_path / "release.jsonl"
    evidence = tmp_path / "evidence.json"
    markdown = tmp_path / "evidence.md"
    targets = {
        "release": release,
        "evidence": evidence,
        "markdown": markdown,
    }
    real_replace = main_module.os.replace
    failed = False

    def fail_selected_replace(source_path, destination_path):
        nonlocal failed
        if not failed and Path(destination_path) == targets[failed_target]:
            failed = True
            raise OSError("synthetic publication failure")
        return real_replace(source_path, destination_path)

    monkeypatch.setattr(main_module.os, "replace", fail_selected_replace)

    code = main_module.main(
        _anonymize_args(
            source,
            release,
            evidence,
            markdown=markdown,
        )
    )

    assert code == 1
    assert failed is True
    assert "Failed to anonymize" in capsys.readouterr().err
    assert all(not path.exists() for path in targets.values())
    assert not [path for path in tmp_path.iterdir() if path.name.startswith(".")]


@pytest.mark.parametrize("failed_target", ["evidence", "markdown", "release"])
def test_risk_anonymize_replacement_failure_restores_all_prior_outputs(
    failed_target: str,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _write_csv(tmp_path / "source.csv")
    release = tmp_path / "release.jsonl"
    evidence = tmp_path / "evidence.json"
    markdown = tmp_path / "evidence.md"
    originals = {
        release: b'{"old":"release"}\n',
        evidence: b'{"old":"evidence"}\n',
        markdown: b"old evidence markdown\n",
    }
    for path, content in originals.items():
        path.write_bytes(content)
    targets = {
        "release": release,
        "evidence": evidence,
        "markdown": markdown,
    }
    real_replace = main_module.os.replace
    failed = False

    def fail_selected_replace(source_path, destination_path):
        nonlocal failed
        if not failed and Path(destination_path) == targets[failed_target]:
            failed = True
            raise OSError("synthetic publication failure")
        return real_replace(source_path, destination_path)

    monkeypatch.setattr(main_module.os, "replace", fail_selected_replace)

    code = main_module.main(
        _anonymize_args(
            source,
            release,
            evidence,
            markdown=markdown,
            overwrite=True,
        )
    )

    assert code == 1
    assert failed is True
    assert "Failed to anonymize" in capsys.readouterr().err
    assert {path: path.read_bytes() for path in originals} == originals
    assert not [path for path in tmp_path.iterdir() if path.name.startswith(".")]


@pytest.mark.parametrize("failed_backup", ["evidence", "markdown", "release"])
def test_risk_anonymize_backup_failure_restores_all_prior_outputs(
    failed_backup: str,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _write_csv(tmp_path / "source.csv")
    release = tmp_path / "release.jsonl"
    evidence = tmp_path / "evidence.json"
    markdown = tmp_path / "evidence.md"
    originals = {
        release: b'{"old":"release"}\n',
        evidence: b'{"old":"evidence"}\n',
        markdown: b"old evidence markdown\n",
    }
    for path, content in originals.items():
        path.write_bytes(content)
    targets = {
        "release": release,
        "evidence": evidence,
        "markdown": markdown,
    }
    real_replace = main_module.os.replace
    failed = False

    def fail_selected_backup(source_path, destination_path):
        nonlocal failed
        if not failed and Path(source_path) == targets[failed_backup]:
            failed = True
            raise OSError("synthetic backup failure")
        return real_replace(source_path, destination_path)

    monkeypatch.setattr(main_module.os, "replace", fail_selected_backup)

    code = main_module.main(
        _anonymize_args(
            source,
            release,
            evidence,
            markdown=markdown,
            overwrite=True,
        )
    )

    assert code == 1
    assert failed is True
    assert "Failed to anonymize" in capsys.readouterr().err
    assert {path: path.read_bytes() for path in originals} == originals
    assert not [path for path in tmp_path.iterdir() if path.name.startswith(".")]


def test_risk_anonymize_rolls_back_when_replace_raises_after_mutation(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _write_csv(tmp_path / "source.csv")
    release = tmp_path / "release.jsonl"
    evidence = tmp_path / "evidence.json"
    markdown = tmp_path / "evidence.md"
    originals = {
        release: b'{"old":"release"}\n',
        evidence: b'{"old":"evidence"}\n',
        markdown: b"old evidence markdown\n",
    }
    for path, content in originals.items():
        path.write_bytes(content)
    real_replace = main_module.os.replace
    failed = False

    def replace_then_fail(source_path, destination_path):
        nonlocal failed
        result = real_replace(source_path, destination_path)
        if not failed and Path(destination_path) == release:
            failed = True
            raise OSError("synthetic post-replacement failure")
        return result

    monkeypatch.setattr(main_module.os, "replace", replace_then_fail)

    code = main_module.main(
        _anonymize_args(
            source,
            release,
            evidence,
            markdown=markdown,
            overwrite=True,
        )
    )

    assert code == 1
    assert failed is True
    assert "Failed to anonymize" in capsys.readouterr().err
    assert {path: path.read_bytes() for path in originals} == originals
    assert not [path for path in tmp_path.iterdir() if path.name.startswith(".")]


def test_risk_anonymize_backup_cleanup_does_not_depend_on_path_unlink(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _write_csv(tmp_path / "source.csv")
    release = tmp_path / "release.jsonl"
    evidence = tmp_path / "evidence.json"
    markdown = tmp_path / "evidence.md"
    release.write_text('{"old":"release"}\n', encoding="utf-8")
    evidence.write_text('{"old":"evidence"}\n', encoding="utf-8")
    markdown.write_text("old evidence markdown\n", encoding="utf-8")

    def fail_path_unlink(self, *, missing_ok=False):
        del self, missing_ok
        raise OSError("synthetic Path.unlink failure")

    monkeypatch.setattr(Path, "unlink", fail_path_unlink)

    code = main_module.main(
        _anonymize_args(
            source,
            release,
            evidence,
            markdown=markdown,
            overwrite=True,
        )
    )

    assert code == 0
    assert "anonymization complete" in capsys.readouterr().out
    assert len(read_table(release)) == len(ROWS)
    assert ExpertReviewEvidenceReport.from_json(
        evidence.read_text(encoding="utf-8")
    ).verify()
    assert markdown.read_text(encoding="utf-8").startswith(
        "# De-identification Risk Analysis Evidence Bundle"
    )
    assert not [path for path in tmp_path.iterdir() if path.name.startswith(".")]


def test_risk_anonymize_reports_post_publication_backup_cleanup_failure(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _write_csv(tmp_path / "source.csv")
    release = tmp_path / "release.jsonl"
    evidence = tmp_path / "evidence.json"
    markdown = tmp_path / "evidence.md"
    release.write_text('{"old":"release"}\n', encoding="utf-8")
    evidence.write_text('{"old":"evidence"}\n', encoding="utf-8")
    markdown.write_text("old evidence markdown\n", encoding="utf-8")
    real_unlink = main_module.os.unlink
    cleanup_failed = False

    def fail_one_populated_backup(path):
        nonlocal cleanup_failed
        candidate = Path(path)
        if (
            not cleanup_failed
            and candidate.parent == tmp_path
            and candidate.name.startswith(".evidence.")
            and candidate.exists()
            and candidate.stat().st_size > 0
        ):
            cleanup_failed = True
            raise OSError("synthetic backup cleanup failure")
        return real_unlink(path)

    monkeypatch.setattr(main_module.os, "unlink", fail_one_populated_backup)

    code = main_module.main(
        _anonymize_args(
            source,
            release,
            evidence,
            markdown=markdown,
            overwrite=True,
        )
    )

    assert code == 1
    assert cleanup_failed is True
    error = capsys.readouterr().err
    assert "outputs were published" in error
    assert "backup cleanup failed" in error
    assert len(read_table(release)) == len(ROWS)
    assert ExpertReviewEvidenceReport.from_json(
        evidence.read_text(encoding="utf-8")
    ).verify()
    stale_backups = [
        path for path in tmp_path.iterdir() if path.name.startswith(".evidence.")
    ]
    assert len(stale_backups) == 1
    real_unlink(stale_backups[0])


def test_invalid_release_policy_is_usage_error(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = _write_csv(tmp_path / "source.csv")
    output = tmp_path / "assessment.json"

    code = main_module.main(
        [
            "risk",
            "assess",
            str(source),
            "--output",
            str(output),
            *_policy_args(),
            "--non-sensitive",
            "age",
        ]
    )

    assert code == 2
    error = capsys.readouterr().err
    assert "policy is invalid" in error
    assert "Cause (ValueError):" in error
    assert "quasi_identifiers cannot overlap non_sensitive_attributes" in error
    assert not output.exists()


@pytest.mark.parametrize("command", ["assess", "anonymize"])
def test_release_config_error_surfaces_safe_value_error_cause(
    command: str,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    rows = [
        {**row, "encounter_id": f"encounter-{index}"}
        for index, row in enumerate(ROWS, start=1)
    ]
    source = _write_csv(tmp_path / "source.csv", rows)
    output = tmp_path / "output.json"
    policy_args = [
        "--qi-column",
        "age,zip,visit_date",
        "--sensitive-column",
        "disease",
        "--direct-id-column",
        "full_name,encounter_id",
        "--privacy-unit",
        "patient_id",
        "--k",
        "2",
        "--l",
        "2",
        "--t",
        "0",
    ]
    if command == "assess":
        args = [
            "risk",
            "assess",
            str(source),
            "--output",
            str(output),
            *policy_args,
        ]
    else:
        args = _anonymize_args(
            source,
            tmp_path / "release.jsonl",
            tmp_path / "evidence.json",
            policy_args=policy_args,
        )
    args.append("--json")

    code = main_module.main(args)

    assert code == 2
    payload = json.loads(capsys.readouterr().out)
    message = payload["error"]["message"]
    assert payload["error"]["code"] == "invalid_release_config"
    assert "Cause (ValueError):" in message
    assert "Declared policy columns are absent from the table" in message
    assert "age,zip,visit_date" in message
    assert "full_name,encounter_id" in message
    for row in rows:
        for value in row.values():
            assert value not in message
    assert not output.exists()


@pytest.mark.parametrize("temporal_kind", ["timestamp_ns", "time64_ns"])
def test_risk_assess_rejects_submicrosecond_parquet_temporal_precision(
    temporal_kind: str,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    data_type = (
        pa.timestamp("ns") if temporal_kind == "timestamp_ns" else pa.time64("ns")
    )
    source = tmp_path / f"{temporal_kind}.parquet"
    output = tmp_path / "assessment.json"
    pq.write_table(
        pa.table({"event_time": pa.array([1, 2], type=data_type)}),
        source,
    )

    code = main_module.main(
        [
            "risk",
            "assess",
            str(source),
            "--output",
            str(output),
            "--qi",
            "event_time",
            "--k",
            "2",
        ]
    )

    assert code == 1
    assert "Failed to read" in capsys.readouterr().err
    assert not output.exists()


def test_unclassified_source_column_is_usage_error(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = _write_csv(
        tmp_path / "source.csv",
        [{**row, "unreviewed_column": "canary"} for row in ROWS],
    )
    output = tmp_path / "assessment.json"

    code = main_module.main(
        [
            "risk",
            "assess",
            str(source),
            "--output",
            str(output),
            *_policy_args(),
        ]
    )

    assert code == 2
    assert "does not match the input schema" in capsys.readouterr().err
    assert not output.exists()


@pytest.mark.parametrize(
    ("command", "source_suffix", "output_suffix"),
    [
        ("discover", ".txt", ".json"),
        ("assess", ".csv", ".txt"),
    ],
)
def test_invalid_structured_suffix_is_usage_error(
    command: str,
    source_suffix: str,
    output_suffix: str,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = _write_csv(tmp_path / f"source{source_suffix}")
    output = tmp_path / f"output{output_suffix}"
    args = ["risk", command, str(source), "--output", str(output)]
    if command == "assess":
        args.extend(_policy_args())

    code = main_module.main(args)

    assert code == 2
    assert "unsupported file suffix" in capsys.readouterr().err
    assert not output.exists()


def test_invalid_hierarchy_config_is_usage_error(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = _write_csv(tmp_path / "source.csv")
    release = tmp_path / "release.jsonl"
    evidence = tmp_path / "evidence.json"
    hierarchies = tmp_path / "hierarchies.json"
    hierarchies.write_text(
        json.dumps({"age": [{"loss": 0.75}, {"loss": 0.25}]}),
        encoding="utf-8",
    )

    code = main_module.main(
        _anonymize_args(
            source,
            release,
            evidence,
            hierarchies=hierarchies,
        )
    )

    assert code == 2
    assert "hierarchy configuration is invalid" in capsys.readouterr().err
    assert not release.exists()
    assert not evidence.exists()


@pytest.mark.parametrize(
    "levels",
    [
        [{"name": "collapsed", "loss": 0.0, "default": "*"}],
        [{"name": "mapped", "loss": 0.0, "values": {"30": "all"}}],
        [{"name": "not-exact", "loss": 0.5}],
        [
            {"name": "exact", "loss": 0.0},
            {"name": "collapsed", "loss": 0.0, "default": "*"},
        ],
    ],
)
def test_cli_rejects_noncanonical_hierarchy_identity_as_usage_error(
    levels: list[dict[str, object]],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = _write_csv(tmp_path / "source.csv")
    release = tmp_path / "release.jsonl"
    evidence = tmp_path / "evidence.json"
    hierarchies = tmp_path / "hierarchies.json"
    hierarchies.write_text(json.dumps({"age": levels}), encoding="utf-8")

    code = main_module.main(
        _anonymize_args(
            source,
            release,
            evidence,
            hierarchies=hierarchies,
        )
    )

    assert code == 2
    assert "hierarchy configuration is invalid" in capsys.readouterr().err
    assert not release.exists()
    assert not evidence.exists()


@pytest.mark.parametrize(
    "coarsening",
    [
        {
            "name": "mapped",
            "loss": 0.5,
            "values": {"30": "__OPENMED_INTERNAL_QI__:state:null"},
        },
        {
            "name": "defaulted",
            "loss": 0.5,
            "default": "__OPENMED_INTERNAL_QI__:state:missing",
        },
    ],
)
def test_cli_rejects_reserved_hierarchy_outputs_as_usage_error(
    coarsening: dict[str, object],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = _write_csv(tmp_path / "source.csv")
    release = tmp_path / "release.jsonl"
    evidence = tmp_path / "evidence.json"
    hierarchies = tmp_path / "hierarchies.json"
    hierarchies.write_text(
        json.dumps(
            {
                "age": [
                    {"name": "exact", "loss": 0.0},
                    coarsening,
                ]
            }
        ),
        encoding="utf-8",
    )

    code = main_module.main(
        _anonymize_args(
            source,
            release,
            evidence,
            hierarchies=hierarchies,
        )
    )

    assert code == 2
    assert "hierarchy configuration is invalid" in capsys.readouterr().err
    assert not release.exists()
    assert not evidence.exists()


def test_cli_rejects_hierarchy_split_after_merge_as_usage_error(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = _write_csv(tmp_path / "source.csv")
    release = tmp_path / "release.jsonl"
    evidence = tmp_path / "evidence.json"
    hierarchies = tmp_path / "hierarchies.json"
    hierarchies.write_text(
        json.dumps(
            {
                "age": [
                    {"name": "exact", "loss": 0.0},
                    {
                        "name": "merged",
                        "loss": 0.5,
                        "values": {"30": "all", "40": "all"},
                    },
                    {
                        "name": "invalid-split",
                        "loss": 0.75,
                        "values": {"30": "young", "40": "older"},
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    code = main_module.main(
        _anonymize_args(
            source,
            release,
            evidence,
            hierarchies=hierarchies,
        )
    )

    assert code == 2
    assert "hierarchy configuration is invalid" in capsys.readouterr().err
    assert not release.exists()
    assert not evidence.exists()


@pytest.mark.parametrize(
    "payload",
    [
        ('{"age":[{"values":{"30":"bucket","30":"sensitive-hierarchy-canary"}}]}'),
        '{"age":[{"loss":NaN}]}',
        '{"age":[{"loss":1e9999}]}',
    ],
)
def test_hierarchy_json_rejects_duplicate_keys_and_nonfinite_numbers(
    payload: str,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = _write_csv(tmp_path / "source.csv")
    release = tmp_path / "release.jsonl"
    evidence = tmp_path / "evidence.json"
    hierarchies = tmp_path / "hierarchies.json"
    hierarchies.write_text(payload, encoding="utf-8")

    code = main_module.main(
        _anonymize_args(
            source,
            release,
            evidence,
            hierarchies=hierarchies,
        )
    )

    assert code == 2
    error = capsys.readouterr().err
    assert "hierarchy configuration is invalid" in error
    assert "sensitive-hierarchy-canary" not in error
    assert not release.exists()
    assert not evidence.exists()
    assert not evidence.with_suffix(".md").exists()


def test_missing_structured_input_is_usage_error(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = tmp_path / "missing.csv"
    output = tmp_path / "assessment.json"

    code = main_module.main(
        [
            "risk",
            "assess",
            str(source),
            "--output",
            str(output),
            *_policy_args(),
        ]
    )

    assert code == 2
    assert "must be an existing file" in capsys.readouterr().err
    assert not output.exists()


@pytest.mark.parametrize(
    ("privacy_unit_args", "kind"),
    [
        (["--privacy-unit", "patient_id"], None),
        (["--privacy-unit", "patient_id"], "row"),
        ([], "patient"),
    ],
)
def test_inconsistent_privacy_unit_kind_is_usage_error(
    privacy_unit_args: list[str],
    kind: str | None,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = _write_csv(tmp_path / "source.csv")
    release = tmp_path / "release.jsonl"
    evidence = tmp_path / "evidence.json"
    policy_args = [
        "--qi",
        "age,zip,visit_date",
        "--sensitive",
        "disease",
        "--direct-id",
        "full_name,patient_id",
        "--k",
        "2",
        "--l",
        "2",
        "--t",
        "0",
        *privacy_unit_args,
    ]

    code = main_module.main(
        _anonymize_args(
            source,
            release,
            evidence,
            policy_args=policy_args,
            privacy_unit_kind=kind,
        )
    )

    assert code == 2
    assert "privacy-unit" in capsys.readouterr().err
    assert not release.exists()
    assert not evidence.exists()


def test_row_level_anonymization_derives_row_privacy_unit(
    tmp_path: Path,
) -> None:
    source = _write_csv(tmp_path / "source.csv")
    release = tmp_path / "release.jsonl"
    evidence = tmp_path / "evidence.json"
    policy_args = [
        "--qi",
        "age,zip,visit_date",
        "--sensitive",
        "disease",
        "--direct-id",
        "full_name,patient_id",
        "--k",
        "2",
        "--l",
        "2",
        "--t",
        "0",
    ]

    code = main_module.main(
        _anonymize_args(
            source,
            release,
            evidence,
            policy_args=policy_args,
            privacy_unit_kind=None,
        )
    )

    assert code == 0
    report = ExpertReviewEvidenceReport.from_json(evidence.read_text(encoding="utf-8"))
    assert report.assumptions.privacy_unit == "row"


def test_household_privacy_unit_kind_is_supported(tmp_path: Path) -> None:
    source = _write_csv(tmp_path / "source.csv")
    release = tmp_path / "release.jsonl"
    evidence = tmp_path / "evidence.json"

    code = main_module.main(
        _anonymize_args(
            source,
            release,
            evidence,
            privacy_unit_kind="household",
        )
    )

    assert code == 0
    report = ExpertReviewEvidenceReport.from_json(evidence.read_text(encoding="utf-8"))
    assert report.assumptions.privacy_unit == "household"


def test_non_sensitive_and_excluded_column_roles_are_wired_to_release_policy(
    tmp_path: Path,
) -> None:
    rows = [{**row, "site": "north", "review_note": "excluded-canary"} for row in ROWS]
    source = _write_csv(tmp_path / "source.csv", rows)
    release = tmp_path / "release.jsonl"
    evidence = tmp_path / "evidence.json"
    policy_args = [
        *_policy_args(),
        "--non-sensitive",
        "site",
        "--exclude",
        "review_note",
    ]

    code = main_module.main(
        _anonymize_args(
            source,
            release,
            evidence,
            policy_args=policy_args,
        )
    )

    assert code == 0
    released = read_table(release)
    assert all(row["site"] == "north" for row in released)
    assert all("review_note" not in row for row in released)


def test_csv_anonymization_rejects_non_injective_sensitive_value_encoding(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = tmp_path / "source.jsonl"
    rows = [
        {"patient_id": "patient-a", "age": 30, "disease": 1},
        {"patient_id": "patient-b", "age": 30, "disease": "1"},
        {"patient_id": "patient-c", "age": 30, "disease": 1},
        {"patient_id": "patient-d", "age": 30, "disease": "1"},
    ]
    source.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    release = tmp_path / "release.csv"
    evidence = tmp_path / "evidence.json"
    markdown = tmp_path / "evidence.md"
    policy_args = [
        "--qi",
        "age",
        "--sensitive",
        "disease",
        "--direct-id",
        "patient_id",
        "--k",
        "2",
    ]

    code = main_module.main(
        _anonymize_args(
            source,
            release,
            evidence,
            markdown=markdown,
            policy_args=policy_args,
            privacy_unit_kind=None,
        )
    )

    assert code == 1
    error = capsys.readouterr().err
    assert "Failed to anonymize" in error
    assert "patient-a" not in error
    assert not release.exists()
    assert not evidence.exists()
    assert not markdown.exists()
    assert not [path for path in tmp_path.iterdir() if path.name.startswith(".")]


def test_legacy_risk_json_is_aggregate_safe(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = _write_csv(tmp_path / "source.csv")

    code = main_module.main(["risk", "table", str(source), "--json"])

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    serialized = json.dumps(payload, sort_keys=True)
    assert payload["data"]["detail_level"] == "aggregate_phi_safe"
    assert "patient-alpha" not in serialized
    assert "Alice Canary" not in serialized
    assert "normalized_value" not in serialized
    assert "quasi_identifier_key" not in serialized
