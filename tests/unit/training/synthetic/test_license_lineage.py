"""Unit tests for synthetic license and lineage manifest validation."""

from __future__ import annotations

import json
import socket
from pathlib import Path

import pytest

from openmed.training.synthetic.license_lineage import (
    DEFAULT_LICENSE_LINEAGE_POLICY,
    DEFAULT_MAX_DEPENDENCIES,
    LICENSE_LINEAGE_FIELDS,
    LICENSE_LINEAGE_SCHEMA_VERSION,
    MAX_DEPENDENCIES,
    RESTRICTED_LICENSE_IDENTIFIERS,
    LicenseExpressionStatus,
    LicenseLineageFinding,
    LicenseLineageReasonCode,
    LicenseLineageReport,
    LicenseLineageSourceClass,
    LicenseLineageVerdict,
    RedistributionDecision,
    SyntheticDatasetDependency,
    SyntheticDatasetLicenseManifest,
    SyntheticGeneratorProvenance,
    SyntheticLicenseLineageError,
    SyntheticLicenseLineagePolicy,
    SyntheticModelProvenance,
    assert_redistribution_allowed,
    assess_license_expression,
    load_license_lineage_manifest,
    manifest_digest,
    validate_license_lineage,
)

DATASET_ID = "synthetic-discharge-summaries"
GENERATOR_ID = "openmed.synthetic.generator"
MODEL_ID = "openmed.qa.small"
DIGEST = "sha256:" + "a" * 64
MODEL_DIGEST = "sha256:" + "b" * 64
SENTINEL = "synthetic-private-sentinel-3099"
GOLDEN_MANIFEST_DIGEST = (
    "sha256:ca8d1131c4e50e6c2fd9abfd80cdba0df7c0dd8ec2adcb5b9c1aa89d1e6925b0"
)


def _dependency(
    dependency_id: str,
    license_expression: str | None = "MIT",
    *,
    source_class: LicenseLineageSourceClass | None = (
        LicenseLineageSourceClass.PUBLIC_DOMAIN_CORPUS
    ),
    redistribution: RedistributionDecision | None = RedistributionDecision.ALLOWED,
    digest: str | None = None,
) -> SyntheticDatasetDependency:
    return SyntheticDatasetDependency(
        dependency_id=dependency_id,
        source_class=source_class,
        license_expression=license_expression,
        redistribution=redistribution,
        digest=digest,
    )


def _manifest(**overrides: object) -> SyntheticDatasetLicenseManifest:
    payload: dict[str, object] = {
        "dataset_id": DATASET_ID,
        "source_class": LicenseLineageSourceClass.SYNTHETIC_GENERATED,
        "license_expression": "Apache-2.0",
        "generator": SyntheticGeneratorProvenance(
            generator_id=GENERATOR_ID,
            digest=DIGEST,
            version="1.0.0",
        ),
        "redistribution": RedistributionDecision.ALLOWED,
        "dependencies": (
            _dependency("corpus-alpha", "MIT"),
            _dependency("corpus-beta", "Apache-2.0"),
        ),
        "model_provenance": SyntheticModelProvenance(
            model_id=MODEL_ID,
            license_expression="MIT",
            digest=MODEL_DIGEST,
        ),
    }
    payload.update(overrides)
    return SyntheticDatasetLicenseManifest(**payload)  # type: ignore[arg-type]


def _payload() -> dict[str, object]:
    return {
        "schema_version": LICENSE_LINEAGE_SCHEMA_VERSION,
        "dataset_id": DATASET_ID,
        "source_class": "synthetic_generated",
        "license_expression": "Apache-2.0",
        "generator": {
            "generator_id": GENERATOR_ID,
            "digest": DIGEST,
            "version": "1.0.0",
        },
        "model_provenance": {
            "model_id": MODEL_ID,
            "license_expression": "MIT",
            "digest": MODEL_DIGEST,
        },
        "dependencies": [
            {
                "dependency_id": "corpus-alpha",
                "source_class": "public_domain_corpus",
                "license_expression": "MIT",
                "redistribution": "allowed",
            }
        ],
        "redistribution": "allowed",
    }


@pytest.mark.parametrize(
    ("expression", "status", "normalized", "permissive", "restricted"),
    [
        ("Apache-2.0", LicenseExpressionStatus.CANONICAL, "Apache-2.0", True, False),
        ("MIT", LicenseExpressionStatus.CANONICAL, "MIT", True, False),
        (" apache-2.0 ", LicenseExpressionStatus.NORMALIZED, "Apache-2.0", True, False),
        ("MIT", LicenseExpressionStatus.CANONICAL, "MIT", True, False),
        (
            "GPL-3.0-only",
            LicenseExpressionStatus.RESTRICTED,
            "GPL-3.0-only",
            False,
            True,
        ),
        (
            "gpl-3.0-only",
            LicenseExpressionStatus.RESTRICTED,
            "GPL-3.0-only",
            False,
            True,
        ),
        (
            "Proprietary-Internal-1.0",
            LicenseExpressionStatus.UNKNOWN,
            None,
            False,
            False,
        ),
        ("LicenseRef-Custom", LicenseExpressionStatus.UNKNOWN, None, False, False),
        ("MIT OR Apache-2.0", LicenseExpressionStatus.MALFORMED, None, False, False),
        ("MIT+", LicenseExpressionStatus.MALFORMED, None, False, False),
        ("", LicenseExpressionStatus.MALFORMED, None, False, False),
        (7, LicenseExpressionStatus.MALFORMED, None, False, False),
        (None, LicenseExpressionStatus.MALFORMED, None, False, False),
    ],
)
def test_assess_license_expression_classifies_candidates(
    expression: object,
    status: LicenseExpressionStatus,
    normalized: str | None,
    permissive: bool,
    restricted: bool,
) -> None:
    assessment = assess_license_expression(expression)
    assert assessment.status is status
    assert assessment.normalized == normalized
    assert assessment.is_permissive is permissive
    assert assessment.is_restricted is restricted
    assert assessment.schema_version == LICENSE_LINEAGE_SCHEMA_VERSION


def test_assess_license_expression_never_echoes_the_candidate() -> None:
    assessment = assess_license_expression(SENTINEL)
    assert assessment.status is LicenseExpressionStatus.UNKNOWN
    assert SENTINEL not in assessment.to_json()
    assert assessment.normalized is None


def test_assess_license_expression_serializes_deterministically() -> None:
    assessment = assess_license_expression("Apache-2.0")
    assert assessment.to_json() == (
        '{"schema_version":"openmed.training.synthetic_license_lineage.v1",'
        '"status":"canonical","normalized":"Apache-2.0","is_permissive":true,'
        '"is_restricted":false,"category":"license_expression_canonical"}'
    )
    assert assessment.to_json() == assess_license_expression("Apache-2.0").to_json()
    assert list(assessment.to_dict()) == [
        "schema_version",
        "status",
        "normalized",
        "is_permissive",
        "is_restricted",
        "category",
    ]


@pytest.mark.parametrize("identifier", sorted(RESTRICTED_LICENSE_IDENTIFIERS))
def test_restricted_allowlist_is_recognized(identifier: str) -> None:
    assessment = assess_license_expression(identifier)
    assert assessment.is_restricted is True
    assert assessment.is_permissive is False
    assert assessment.normalized == identifier


def test_long_expression_is_malformed() -> None:
    assessment = assess_license_expression("A" * 129)
    assert assessment.status is LicenseExpressionStatus.MALFORMED
    assert assessment.category == "license_expression_malformed_length"


def test_manifest_requires_a_lowercase_identifier_dataset_id() -> None:
    with pytest.raises(SyntheticLicenseLineageError):
        _manifest(dataset_id="Synthetic Dataset")
    with pytest.raises(SyntheticLicenseLineageError):
        _manifest(dataset_id="")


def test_manifest_rejects_a_foreign_schema_version() -> None:
    with pytest.raises(SyntheticLicenseLineageError):
        _manifest(schema_version="openmed.training.synthetic_license_lineage.v2")


def test_manifest_rejects_wrongly_typed_sections() -> None:
    with pytest.raises(SyntheticLicenseLineageError):
        _manifest(source_class="synthetic_generated")
    with pytest.raises(SyntheticLicenseLineageError):
        _manifest(redistribution="allowed")
    with pytest.raises(SyntheticLicenseLineageError):
        _manifest(generator={"generator_id": GENERATOR_ID})
    with pytest.raises(SyntheticLicenseLineageError):
        _manifest(model_provenance="openmed.qa.small")
    with pytest.raises(SyntheticLicenseLineageError):
        _manifest(dependencies=[_dependency("corpus-alpha")])


def test_manifest_orders_dependencies_by_identifier() -> None:
    manifest = _manifest(
        dependencies=(
            _dependency("corpus-zulu"),
            _dependency("corpus-alpha"),
        )
    )
    assert [item.dependency_id for item in manifest.dependencies] == [
        "corpus-alpha",
        "corpus-zulu",
    ]


def test_manifest_rejects_more_than_the_hard_dependency_limit() -> None:
    dependencies = tuple(
        _dependency(f"corpus-{index:04d}") for index in range(MAX_DEPENDENCIES + 1)
    )
    with pytest.raises(SyntheticLicenseLineageError):
        _manifest(dependencies=dependencies)


def test_manifest_round_trips_through_json() -> None:
    manifest = _manifest()
    assert SyntheticDatasetLicenseManifest.from_json(manifest.to_json()) == manifest
    assert (
        SyntheticDatasetLicenseManifest.from_json(
            manifest.to_json(indent=2).encode("utf-8")
        )
        == manifest
    )
    assert (
        manifest.to_dict()
        == SyntheticDatasetLicenseManifest.from_dict(manifest.to_dict()).to_dict()
    )


def test_manifest_from_dict_rejects_unknown_and_missing_keys() -> None:
    payload = _payload()
    payload["unexpected"] = True
    with pytest.raises(SyntheticLicenseLineageError):
        SyntheticDatasetLicenseManifest.from_dict(payload)

    payload = _payload()
    del payload["generator"]
    with pytest.raises(SyntheticLicenseLineageError):
        SyntheticDatasetLicenseManifest.from_dict(payload)


def test_manifest_from_dict_rejects_invalid_payloads() -> None:
    with pytest.raises(SyntheticLicenseLineageError):
        SyntheticDatasetLicenseManifest.from_json("not json")
    with pytest.raises(SyntheticLicenseLineageError):
        SyntheticDatasetLicenseManifest.from_dict("not a mapping")  # type: ignore[arg-type]
    payload = _payload()
    payload["dependencies"] = "corpus-alpha"
    with pytest.raises(SyntheticLicenseLineageError):
        SyntheticDatasetLicenseManifest.from_dict(payload)


def test_manifest_from_dict_keeps_unrecognized_enums_absent() -> None:
    payload = _payload()
    payload["source_class"] = "state_secret_corpus"
    manifest = SyntheticDatasetLicenseManifest.from_dict(payload)
    assert manifest.source_class is None
    report = validate_license_lineage(manifest)
    assert LicenseLineageReasonCode.SOURCE_CLASS_UNKNOWN in {
        finding.reason for finding in report.findings
    }


def test_permissive_manifest_is_cleared() -> None:
    report = validate_license_lineage(_manifest())
    assert report.verdict is LicenseLineageVerdict.CLEARED
    assert report.ok is True
    assert report.blocked is False
    assert report.findings == ()
    assert report.reason_codes == ()
    assert report.blocked_fields == ()
    assert report.dependency_count == 2
    assert report.license_expressions == ("Apache-2.0", "MIT")
    assert report.schema_version == LICENSE_LINEAGE_SCHEMA_VERSION
    assert_redistribution_allowed(report)


def test_mixed_permissive_dependencies_are_cleared() -> None:
    report = validate_license_lineage(
        _manifest(
            license_expression="MIT",
            dependencies=(
                _dependency("corpus-alpha", "Apache-2.0"),
                _dependency("corpus-beta", "BSD-3-Clause"),
                _dependency("corpus-gamma", "Zlib"),
            ),
            model_provenance=None,
        )
    )
    assert report.ok is True
    assert report.license_expressions == ("Apache-2.0", "BSD-3-Clause", "MIT", "Zlib")


def test_unknown_license_expression_blocks_the_manifest() -> None:
    report = validate_license_lineage(
        _manifest(license_expression="Proprietary-Internal-1.0")
    )
    assert report.blocked is True
    assert report.reason_codes == ("license_expression_unknown",)
    assert report.blocked_fields == ("license_expression",)
    assert "Proprietary-Internal-1.0" not in report.to_json()


def test_restricted_license_expression_blocks_the_manifest() -> None:
    report = validate_license_lineage(_manifest(license_expression="AGPL-3.0-only"))
    assert report.blocked is True
    assert report.reason_codes == ("license_expression_restricted",)
    assert report.blocked_fields == ("license_expression",)


def test_malformed_license_expression_blocks_the_manifest() -> None:
    report = validate_license_lineage(_manifest(license_expression="MIT OR Apache-2.0"))
    assert report.reason_codes == ("license_expression_malformed",)


def test_missing_license_expression_blocks_the_manifest() -> None:
    report = validate_license_lineage(_manifest(license_expression=None))
    assert report.reason_codes == ("license_expression_missing",)


def test_licenseref_expression_blocks_the_manifest() -> None:
    report = validate_license_lineage(_manifest(license_expression="LicenseRef-Custom"))
    assert report.reason_codes == ("license_expression_unknown",)


def test_unknown_source_class_blocks_the_manifest() -> None:
    report = validate_license_lineage(_manifest(source_class=None))
    assert report.reason_codes == ("source_class_unknown",)
    assert report.blocked_fields == ("source_class",)


def test_missing_generator_blocks_the_manifest() -> None:
    report = validate_license_lineage(_manifest(generator=None))
    assert report.reason_codes == ("generator_digest_missing",)
    assert report.blocked_fields == ("generator",)


def test_malformed_generator_digest_blocks_the_manifest() -> None:
    report = validate_license_lineage(
        _manifest(
            generator=SyntheticGeneratorProvenance(
                generator_id=GENERATOR_ID,
                digest="sha256:nothex",
            )
        )
    )
    assert report.reason_codes == ("generator_digest_malformed",)


def test_missing_generator_digest_alone_blocks_the_manifest() -> None:
    report = validate_license_lineage(
        _manifest(
            generator=SyntheticGeneratorProvenance(
                generator_id=GENERATOR_ID,
                digest=None,
            )
        )
    )
    assert report.reason_codes == ("generator_digest_missing",)


def test_model_provenance_uses_the_same_license_rules() -> None:
    report = validate_license_lineage(
        _manifest(
            model_provenance=SyntheticModelProvenance(
                model_id=MODEL_ID,
                license_expression="SSPL-1.0",
                digest=MODEL_DIGEST,
            )
        )
    )
    assert report.reason_codes == ("license_expression_restricted",)
    assert report.blocked_fields == ("model_provenance",)


def test_malformed_model_digest_blocks_the_manifest() -> None:
    report = validate_license_lineage(
        _manifest(
            model_provenance=SyntheticModelProvenance(
                model_id=MODEL_ID,
                license_expression="MIT",
                digest="not-a-digest",
            )
        )
    )
    assert report.reason_codes == ("model_digest_malformed",)


def test_model_provenance_can_be_required_by_policy() -> None:
    policy = SyntheticLicenseLineagePolicy(require_model_provenance=True)
    report = validate_license_lineage(_manifest(model_provenance=None), policy=policy)
    assert report.reason_codes == ("model_provenance_missing",)
    assert validate_license_lineage(_manifest(model_provenance=None)).ok is True


def test_dependency_without_lineage_blocks_the_manifest() -> None:
    report = validate_license_lineage(
        _manifest(
            dependencies=(
                _dependency("corpus-alpha", "MIT"),
                SyntheticDatasetDependency(dependency_id="corpus-gamma"),
            )
        )
    )
    assert report.blocked is True
    assert report.reason_codes == (
        "dependency_lineage_missing",
        "redistribution_unknown",
        "source_class_unknown",
    )
    assert report.blocked_fields == ("dependencies", "redistribution")
    findings = [
        finding
        for finding in report.findings
        if finding.dependency_id == "corpus-gamma"
    ]
    assert len(findings) == 3


def test_dependency_with_restricted_license_blocks_the_manifest() -> None:
    report = validate_license_lineage(
        _manifest(
            dependencies=(
                _dependency("corpus-alpha", "MIT"),
                _dependency("corpus-delta", "CC-BY-NC-4.0"),
            )
        )
    )
    assert report.reason_codes == ("license_expression_restricted",)
    assert report.findings[0].dependency_id == "corpus-delta"
    assert report.findings[0].field == "dependencies"


def test_dependency_without_a_redistribution_decision_blocks_the_manifest() -> None:
    report = validate_license_lineage(
        _manifest(
            dependencies=(_dependency("corpus-alpha", "MIT", redistribution=None),)
        )
    )
    assert report.reason_codes == (
        "dependency_lineage_missing",
        "redistribution_unknown",
    )


def test_dependency_digest_rules_follow_policy() -> None:
    dependency = _dependency("corpus-alpha", "MIT", digest="sha256:nothex")
    relaxed = validate_license_lineage(
        _manifest(dependencies=(dependency,)),
        policy=SyntheticLicenseLineagePolicy(require_dependency_digests=False),
    )
    assert relaxed.reason_codes == ("dependency_digest_malformed",)

    required = validate_license_lineage(
        _manifest(dependencies=(_dependency("corpus-alpha", "MIT"),)),
        policy=SyntheticLicenseLineagePolicy(require_dependency_digests=True),
    )
    assert required.reason_codes == ("dependency_digest_missing",)

    satisfied = validate_license_lineage(
        _manifest(dependencies=(_dependency("corpus-alpha", "MIT", digest=DIGEST),)),
        policy=SyntheticLicenseLineagePolicy(require_dependency_digests=True),
    )
    assert satisfied.ok is True


def test_dependency_limit_policy_is_enforced() -> None:
    dependencies = tuple(
        _dependency(f"corpus-{index:04d}")
        for index in range(DEFAULT_MAX_DEPENDENCIES + 1)
    )
    report = validate_license_lineage(
        _manifest(dependencies=dependencies),
        policy=SyntheticLicenseLineagePolicy(max_dependencies=DEFAULT_MAX_DEPENDENCIES),
    )
    assert report.reason_codes == ("dependency_limit_exceeded",)
    assert report.dependency_count == DEFAULT_MAX_DEPENDENCIES + 1

    exactly = validate_license_lineage(
        _manifest(dependencies=dependencies),
        policy=SyntheticLicenseLineagePolicy(
            max_dependencies=DEFAULT_MAX_DEPENDENCIES + 1
        ),
    )
    assert exactly.ok is True


def test_unknown_licenses_can_be_allowed_by_policy() -> None:
    policy = SyntheticLicenseLineagePolicy(allow_unknown_licenses=True)
    report = validate_license_lineage(
        _manifest(license_expression="Proprietary-Internal-1.0"),
        policy=policy,
    )
    assert report.ok is True
    assert validate_license_lineage(
        _manifest(license_expression="AGPL-3.0-only"), policy=policy
    ).reason_codes == ("license_expression_restricted",)


@pytest.mark.parametrize(
    ("decision", "reason"),
    [
        (RedistributionDecision.UNKNOWN, "redistribution_unknown"),
        (RedistributionDecision.RESTRICTED, "redistribution_restricted"),
        (RedistributionDecision.PROHIBITED, "redistribution_prohibited"),
    ],
)
def test_non_allowed_redistribution_blocks_the_manifest(
    decision: RedistributionDecision,
    reason: str,
) -> None:
    report = validate_license_lineage(_manifest(redistribution=decision))
    assert report.reason_codes == (reason,)
    assert report.blocked_fields == ("redistribution",)


def test_absent_redistribution_decision_blocks_the_manifest() -> None:
    report = validate_license_lineage(_manifest(redistribution=None))
    assert report.reason_codes == ("redistribution_unknown",)


def test_reordered_dependencies_produce_identical_output() -> None:
    first = _manifest()
    second = _manifest(
        dependencies=(
            _dependency("corpus-beta", "Apache-2.0"),
            _dependency("corpus-alpha", "MIT"),
        )
    )
    assert manifest_digest(first) == manifest_digest(second)
    assert validate_license_lineage(first).to_json() == (
        validate_license_lineage(second).to_json()
    )


def test_report_and_manifest_digests_are_stable_across_runs() -> None:
    first = validate_license_lineage(_manifest())
    second = validate_license_lineage(_manifest())
    assert first.to_json() == second.to_json()
    assert manifest_digest(_manifest()) == GOLDEN_MANIFEST_DIGEST
    assert manifest_digest(_manifest().to_dict()) == GOLDEN_MANIFEST_DIGEST
    assert manifest_digest(_manifest().to_json()) == GOLDEN_MANIFEST_DIGEST


def test_validation_is_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    def _blocked(*args: object, **kwargs: object) -> None:
        raise AssertionError("network access is not allowed")

    monkeypatch.setattr(socket, "socket", _blocked)
    monkeypatch.setattr(socket, "create_connection", _blocked)
    first = validate_license_lineage(_manifest())
    second = validate_license_lineage(_manifest())
    assert first.to_json() == second.to_json()


def test_report_round_trips_through_json() -> None:
    report = validate_license_lineage(
        _manifest(license_expression="Proprietary-Internal-1.0")
    )
    assert LicenseLineageReport.from_json(report.to_json()) == report
    assert (
        LicenseLineageReport.from_json(report.to_json(indent=2).encode("utf-8"))
        == report
    )
    assert report.to_json(indent=2).endswith("\n") is False
    assert list(report.to_dict()) == [
        "schema_version",
        "dataset_id",
        "verdict",
        "blocked",
        "dependency_count",
        "license_expressions",
        "reason_codes",
        "blocked_fields",
        "findings",
    ]


def test_report_from_dict_rejects_invalid_payloads() -> None:
    report = validate_license_lineage(_manifest())
    payload = report.to_dict()
    payload["extra"] = True
    with pytest.raises(SyntheticLicenseLineageError):
        LicenseLineageReport.from_dict(payload)

    payload = report.to_dict()
    payload["verdict"] = "maybe"
    with pytest.raises(SyntheticLicenseLineageError):
        LicenseLineageReport.from_dict(payload)

    payload = report.to_dict()
    payload["findings"] = "none"
    with pytest.raises(SyntheticLicenseLineageError):
        LicenseLineageReport.from_dict(payload)

    with pytest.raises(SyntheticLicenseLineageError):
        LicenseLineageReport.from_json("{")

    payload = report.to_dict()
    del payload["dataset_id"]
    with pytest.raises(SyntheticLicenseLineageError):
        LicenseLineageReport.from_dict(payload)


def test_report_invariants_are_enforced() -> None:
    finding = LicenseLineageFinding(
        field="license_expression",
        reason=LicenseLineageReasonCode.LICENSE_EXPRESSION_UNKNOWN,
    )
    with pytest.raises(SyntheticLicenseLineageError):
        LicenseLineageReport(
            verdict=LicenseLineageVerdict.CLEARED,
            findings=(finding,),
            dataset_id=DATASET_ID,
            dependency_count=1,
        )
    with pytest.raises(SyntheticLicenseLineageError):
        LicenseLineageReport(
            verdict=LicenseLineageVerdict.BLOCKED,
            findings=(),
            dataset_id=DATASET_ID,
            dependency_count=1,
        )
    unsorted = (
        LicenseLineageFinding(
            field="redistribution",
            reason=LicenseLineageReasonCode.REDISTRIBUTION_UNKNOWN,
        ),
        finding,
    )
    with pytest.raises(SyntheticLicenseLineageError):
        LicenseLineageReport(
            verdict=LicenseLineageVerdict.BLOCKED,
            findings=unsorted,
            dataset_id=DATASET_ID,
            dependency_count=1,
        )
    with pytest.raises(SyntheticLicenseLineageError):
        LicenseLineageReport(
            verdict=LicenseLineageVerdict.BLOCKED,
            findings=(finding,),
            dataset_id="Not An Identifier",
            dependency_count=1,
        )
    with pytest.raises(SyntheticLicenseLineageError):
        LicenseLineageReport(
            verdict=LicenseLineageVerdict.BLOCKED,
            findings=(finding,),
            dataset_id=DATASET_ID,
            dependency_count=1,
            license_expressions=("MIT", "Apache-2.0"),
        )
    with pytest.raises(SyntheticLicenseLineageError):
        LicenseLineageReport(
            verdict=LicenseLineageVerdict.BLOCKED,
            findings=(finding,),
            dataset_id=DATASET_ID,
            dependency_count=-1,
        )
    with pytest.raises(SyntheticLicenseLineageError):
        LicenseLineageReport(
            verdict=LicenseLineageVerdict.BLOCKED,
            findings=(finding, finding),
            dataset_id=DATASET_ID,
            dependency_count=1,
        )


def test_finding_rejects_unknown_fields_and_reasons() -> None:
    with pytest.raises(SyntheticLicenseLineageError):
        LicenseLineageFinding(
            field="unknown_field",
            reason=LicenseLineageReasonCode.LICENSE_EXPRESSION_UNKNOWN,
        )
    with pytest.raises(SyntheticLicenseLineageError):
        LicenseLineageFinding(
            field="license_expression",
            reason="license_expression_unknown",  # type: ignore[arg-type]
        )
    with pytest.raises(SyntheticLicenseLineageError):
        LicenseLineageFinding(
            field="license_expression",
            reason=LicenseLineageReasonCode.LICENSE_EXPRESSION_UNKNOWN,
            dependency_id="Not An Identifier",
        )
    finding = LicenseLineageFinding(
        field="dependencies",
        reason=LicenseLineageReasonCode.DEPENDENCY_LINEAGE_MISSING,
        dependency_id="corpus-alpha",
    )
    assert finding.sort_key == (
        LICENSE_LINEAGE_FIELDS.index("dependencies"),
        "corpus-alpha",
        "dependency_lineage_missing",
    )
    assert LicenseLineageFinding.from_dict(finding.to_dict()) == finding


def test_policy_validates_its_bounds() -> None:
    assert DEFAULT_LICENSE_LINEAGE_POLICY == SyntheticLicenseLineagePolicy()
    assert DEFAULT_LICENSE_LINEAGE_POLICY.max_dependencies == DEFAULT_MAX_DEPENDENCIES
    for overrides in (
        {"allow_unknown_licenses": "yes"},
        {"require_dependency_digests": 1},
        {"require_model_provenance": None},
        {"max_dependencies": "32"},
        {"max_dependencies": 0},
        {"max_dependencies": MAX_DEPENDENCIES + 1},
    ):
        with pytest.raises(SyntheticLicenseLineageError):
            SyntheticLicenseLineagePolicy(**overrides)  # type: ignore[arg-type]


def test_validate_rejects_unsupported_inputs() -> None:
    with pytest.raises(SyntheticLicenseLineageError):
        validate_license_lineage(42)
    with pytest.raises(SyntheticLicenseLineageError):
        validate_license_lineage(_manifest(), policy="permissive")  # type: ignore[arg-type]


def test_assert_redistribution_allowed_reports_counts_only() -> None:
    report = validate_license_lineage(
        _manifest(
            license_expression=SENTINEL, redistribution=RedistributionDecision.UNKNOWN
        )
    )
    with pytest.raises(SyntheticLicenseLineageError) as error:
        assert_redistribution_allowed(report)
    message = str(error.value)
    assert SENTINEL not in message
    assert DATASET_ID not in message
    assert "2 findings" in message
    assert "2 dependencies" in message

    with pytest.raises(SyntheticLicenseLineageError):
        assert_redistribution_allowed("report")  # type: ignore[arg-type]


def test_report_never_contains_raw_manifest_text() -> None:
    report = validate_license_lineage(
        _manifest(
            license_expression=SENTINEL,
            dependencies=(_dependency("corpus-alpha", SENTINEL),),
        )
    )
    payload = report.to_json()
    assert SENTINEL not in payload
    assert DATASET_ID in payload
    assert report.license_expressions == ("MIT",)


def test_manifest_file_helpers_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "lineage.json"
    _manifest().write_json(path)
    text = path.read_text(encoding="utf-8")
    assert text.endswith("\n")
    assert load_license_lineage_manifest(path) == _manifest()
    assert json.loads(text)["dataset_id"] == DATASET_ID

    report_path = tmp_path / "report.json"
    validate_license_lineage(_manifest()).write_json(report_path)
    assert report_path.read_text(encoding="utf-8").endswith("\n")

    with pytest.raises(SyntheticLicenseLineageError):
        load_license_lineage_manifest(tmp_path / "missing.json")
