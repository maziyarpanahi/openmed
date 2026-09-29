"""Table tests for the immutable, versioned privacy budget policy contract."""

from __future__ import annotations

import json
import re
import socket
from dataclasses import FrozenInstanceError

import pytest

from openmed.training import (
    DP_BUDGET_MAX_EPSILON,
    DP_BUDGET_POLICY_FIELDS,
    DP_BUDGET_POLICY_REASON_CODES,
    DP_BUDGET_POLICY_SCHEMA_VERSION,
    DPAccountant,
    DPBudgetPolicy,
    DPBudgetPolicyError,
    DPBudgetPolicyFinding,
    DPBudgetPolicyRejected,
    DPBudgetPolicyReport,
    DPBudgetScope,
    DPComposition,
    DPExhaustion,
    build_dp_budget_policy,
    fingerprint_dp_budget_policy,
    validate_dp_budget_policy,
)
from openmed.training import dp_budget_policy as policy_module

MINIMAL = {
    "policy_id": "round-policy",
    "max_epsilon": 8.0,
    "max_delta": 1e-5,
    "epsilon_scope": DPBudgetScope.TOTAL,
    "delta_scope": DPBudgetScope.TOTAL,
    "accountant": DPAccountant.RENYI,
    "composition": DPComposition.BASIC,
    "exhaustion": DPExhaustion.REQUIRE_REVIEW,
}

ADVANCED = {
    "policy_id": "advanced-policy",
    "max_epsilon": 4.0,
    "max_delta": 1e-6,
    "epsilon_scope": DPBudgetScope.PER_ROUND,
    "delta_scope": DPBudgetScope.TOTAL,
    "accountant": DPAccountant.GAUSSIAN,
    "composition": DPComposition.ADVANCED,
    "exhaustion": DPExhaustion.FAIL_CLOSED,
    "delta_prime": 1e-9,
    "min_rounds": 2,
    "max_rounds": 10,
}

BOUNDARY = {
    "policy_id": "boundary-policy",
    "max_epsilon": DP_BUDGET_MAX_EPSILON,
    "max_delta": 0.999999,
    "epsilon_scope": "total",
    "delta_scope": "per_round",
    "accountant": "zcdp",
    "composition": "basic",
    "exhaustion": "refuse_round",
    "max_rounds": None,
}

VALID_POLICIES = [MINIMAL, ADVANCED, BOUNDARY]

GOLDEN_DIGESTS = {
    "round-policy": "sha256:a468098fa4bf18ebed79244567593dd8f6d2eb634305e6a35caaaee8e4ec6df0",
    "advanced-policy": (
        "sha256:177d74e2c4f468e6f7730e8497fa97638eb3720b02cee6ae33282e1b13f6d120"
    ),
    "boundary-policy": (
        "sha256:b8b1fca54b67b0bacad01ff7af9f2dadf0e0084e18c6ff0be7d6ddc832acd31f"
    ),
}

SENTINELS = ("SITE-A", "cohort/1", "Participant@example.com", "123-45-6789")

DIGEST_PATTERN = re.compile(r"sha256:[0-9a-f]{64}\Z")

K = DPAccountant.RENYI
P = DPComposition.BASIC
U = DPExhaustion.FAIL_CLOSED
S = DPBudgetScope.TOTAL


def _payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = dict(MINIMAL)
    payload.update(overrides)
    return payload


def _report(**overrides: object) -> DPBudgetPolicyReport:
    return validate_dp_budget_policy(_payload(**overrides))


def _reasons(**overrides: object) -> set[str]:
    return set(_report(**overrides).reason_codes)


def _fields(**overrides: object) -> dict[str, str]:
    return {
        finding.field_name: finding.reason_code
        for finding in _report(**overrides).findings
    }


@pytest.mark.parametrize("fields", VALID_POLICIES)
def test_build_accepts_valid_policies(fields: dict[str, object]) -> None:
    policy = build_dp_budget_policy(**fields)
    assert isinstance(policy, DPBudgetPolicy)
    assert policy.schema_version == DP_BUDGET_POLICY_SCHEMA_VERSION


@pytest.mark.parametrize("fields", VALID_POLICIES)
def test_to_dict_uses_the_documented_field_order(fields: dict[str, object]) -> None:
    policy = build_dp_budget_policy(**fields)
    assert list(policy.to_dict()) == list(DP_BUDGET_POLICY_FIELDS)


@pytest.mark.parametrize("fields", VALID_POLICIES)
def test_to_json_is_a_sorted_dump_with_a_trailing_newline(
    fields: dict[str, object],
) -> None:
    policy = build_dp_budget_policy(**fields)
    expected = json.dumps(policy.to_dict(), indent=2, sort_keys=True) + "\n"
    assert policy.to_json(indent=2) == expected
    compact = json.dumps(policy.to_dict(), separators=(",", ":"), sort_keys=True) + "\n"
    assert policy.to_json() == compact


@pytest.mark.parametrize("fields", VALID_POLICIES)
def test_json_round_trip_returns_an_equal_policy(fields: dict[str, object]) -> None:
    policy = build_dp_budget_policy(**fields)
    assert DPBudgetPolicy.from_dict(json.loads(policy.to_json())) == policy


@pytest.mark.parametrize("fields", VALID_POLICIES)
def test_serialization_is_deterministic(fields: dict[str, object]) -> None:
    first = build_dp_budget_policy(**fields).to_json(indent=2)
    second = build_dp_budget_policy(**fields).to_json(indent=2)
    assert first == second


@pytest.mark.parametrize("fields", VALID_POLICIES)
def test_string_members_coerce_to_named_enums(fields: dict[str, object]) -> None:
    policy = build_dp_budget_policy(**fields)
    assert policy.epsilon_scope is DPBudgetScope(policy.epsilon_scope)
    assert policy.delta_scope is DPBudgetScope(policy.delta_scope)
    assert policy.accountant is DPAccountant(policy.accountant)
    assert policy.composition is DPComposition(policy.composition)
    assert policy.exhaustion is DPExhaustion(policy.exhaustion)


@pytest.mark.parametrize("field_name", DP_BUDGET_POLICY_FIELDS)
def test_policy_fields_are_immutable(field_name: str) -> None:
    policy = build_dp_budget_policy(**MINIMAL)
    with pytest.raises(FrozenInstanceError):
        setattr(policy, field_name, "replacement")


def test_policy_field_tuple_is_closed_and_unique() -> None:
    assert isinstance(DP_BUDGET_POLICY_FIELDS, tuple)
    assert len(DP_BUDGET_POLICY_FIELDS) == len(set(DP_BUDGET_POLICY_FIELDS))
    assert "exhaustion" in DP_BUDGET_POLICY_FIELDS


def test_validate_reports_an_unsupported_payload_type() -> None:
    for payload in ("not-a-mapping", ["policy_id"], 3, None):
        with pytest.raises(DPBudgetPolicyError, match="must be a mapping"):
            validate_dp_budget_policy(payload)  # type: ignore[arg-type]


@pytest.mark.parametrize("fields", VALID_POLICIES)
def test_validate_returns_a_valid_report(fields: dict[str, object]) -> None:
    report = validate_dp_budget_policy(fields)
    assert report.valid is True
    assert report.ok is True
    assert report.findings == ()
    assert report.reason_codes == ()
    assert report.schema_version == DP_BUDGET_POLICY_SCHEMA_VERSION
    assert report.policy_id == fields["policy_id"]


@pytest.mark.parametrize("fields", VALID_POLICIES)
def test_valid_report_digest_matches_the_policy(fields: dict[str, object]) -> None:
    report = validate_dp_budget_policy(fields)
    policy = build_dp_budget_policy(**fields)
    assert report.policy_digest == fingerprint_dp_budget_policy(policy)


@pytest.mark.parametrize("policy_id", sorted(GOLDEN_DIGESTS))
def test_golden_policy_digests_are_stable(policy_id: str) -> None:
    fields = next(item for item in VALID_POLICIES if item["policy_id"] == policy_id)
    policy = build_dp_budget_policy(**fields)
    assert fingerprint_dp_budget_policy(policy) == GOLDEN_DIGESTS[policy_id]
    assert DIGEST_PATTERN.fullmatch(fingerprint_dp_budget_policy(policy))


def test_fingerprint_rejects_anything_but_a_policy() -> None:
    for value in (MINIMAL, "round-policy", None, 3):
        with pytest.raises(
            DPBudgetPolicyError, match="requires a privacy budget policy"
        ):
            fingerprint_dp_budget_policy(value)  # type: ignore[arg-type]


def test_fingerprint_changes_when_a_bound_changes() -> None:
    policy = build_dp_budget_policy(**MINIMAL)
    other = build_dp_budget_policy(**{**MINIMAL, "max_epsilon": 7.0})
    assert fingerprint_dp_budget_policy(policy) != fingerprint_dp_budget_policy(other)


def test_report_round_trips_through_json() -> None:
    report = validate_dp_budget_policy(MINIMAL)
    assert DPBudgetPolicyReport.from_dict(json.loads(report.to_json())) == report
    assert report.to_json() == (
        json.dumps(report.to_dict(), separators=(",", ":"), sort_keys=True) + "\n"
    )


def test_rejected_report_round_trips_through_json() -> None:
    report = _report(policy_id="Bad Id")
    assert report.valid is False
    assert DPBudgetPolicyReport.from_dict(json.loads(report.to_json())) == report
    assert report.policy_id == ""
    assert report.policy_digest == ""


def test_report_from_dict_rejects_unknown_and_missing_fields() -> None:
    payload = validate_dp_budget_policy(MINIMAL).to_dict()
    for mutated in (
        {**payload, "extra": 1},
        {key: value for key, value in payload.items() if key != "valid"},
    ):
        with pytest.raises(DPBudgetPolicyError, match="invalid policy report fields"):
            DPBudgetPolicyReport.from_dict(mutated)


def test_report_from_dict_requires_a_findings_list() -> None:
    payload = _report(policy_id="Bad Id").to_dict()
    payload["findings"] = tuple(payload["findings"])
    with pytest.raises(DPBudgetPolicyError, match="invalid policy report findings"):
        DPBudgetPolicyReport.from_dict(payload)


def test_report_rejects_inconsistent_findings_and_digest() -> None:
    finding = DPBudgetPolicyFinding(reason_code="missing_field", field_name="max_delta")
    digest = fingerprint_dp_budget_policy(build_dp_budget_policy(**MINIMAL))
    with pytest.raises(DPBudgetPolicyError, match="cannot carry findings"):
        DPBudgetPolicyReport(
            schema_version=DP_BUDGET_POLICY_SCHEMA_VERSION,
            policy_id="round-policy",
            valid=True,
            findings=(finding,),
            policy_digest=digest,
        )
    with pytest.raises(DPBudgetPolicyError, match="needs a policy digest"):
        DPBudgetPolicyReport(
            schema_version=DP_BUDGET_POLICY_SCHEMA_VERSION,
            policy_id="round-policy",
            valid=True,
            findings=(),
            policy_digest="",
        )
    with pytest.raises(DPBudgetPolicyError, match="at least one finding"):
        DPBudgetPolicyReport(
            schema_version=DP_BUDGET_POLICY_SCHEMA_VERSION,
            policy_id="",
            valid=False,
            findings=(),
            policy_digest="",
        )
    with pytest.raises(DPBudgetPolicyError, match="cannot carry a digest"):
        DPBudgetPolicyReport(
            schema_version=DP_BUDGET_POLICY_SCHEMA_VERSION,
            policy_id="",
            valid=False,
            findings=(finding,),
            policy_digest=digest,
        )


def test_report_rejects_a_foreign_schema_version() -> None:
    with pytest.raises(DPBudgetPolicyError, match="unsupported report schema version"):
        DPBudgetPolicyReport(
            schema_version="openmed.training.dp_budget_policy.v0",
            policy_id="",
            valid=False,
            findings=(
                DPBudgetPolicyFinding(
                    reason_code="missing_field", field_name="max_delta"
                ),
            ),
            policy_digest="",
        )


def test_finding_rejects_unknown_reason_codes_and_fields() -> None:
    with pytest.raises(DPBudgetPolicyError, match="not a supported policy reason"):
        DPBudgetPolicyFinding(reason_code="looks_fine", field_name="max_delta")
    with pytest.raises(DPBudgetPolicyError, match="not a policy field"):
        DPBudgetPolicyFinding(reason_code="missing_field", field_name="site_id")
    with pytest.raises(DPBudgetPolicyError, match="integer or None"):
        DPBudgetPolicyFinding(
            reason_code="missing_field", field_name="max_delta", observed=1.5
        )


def test_finding_round_trips_through_dict() -> None:
    finding = DPBudgetPolicyFinding(
        reason_code="invalid_round_bounds", field_name="max_rounds", observed=0
    )
    assert DPBudgetPolicyFinding.from_dict(finding.to_dict()) == finding
    with pytest.raises(DPBudgetPolicyError, match="invalid policy finding fields"):
        DPBudgetPolicyFinding.from_dict(
            {"reason_code": "missing_field", "field_name": "max_delta"}
        )


def test_finding_from_dict_rejects_bad_values() -> None:
    with pytest.raises(DPBudgetPolicyError, match="invalid policy finding values"):
        DPBudgetPolicyFinding.from_dict(
            {"reason_code": 3, "field_name": "max_delta", "observed": None}
        )
    with pytest.raises(DPBudgetPolicyError, match="invalid policy finding observation"):
        DPBudgetPolicyFinding.from_dict(
            {"reason_code": "missing_field", "field_name": "max_delta", "observed": "1"}
        )


def test_unsupported_keys_are_reported_without_echoing_them() -> None:
    report = _report(**{"site_id": "SITE-A"})
    assert "unsupported_field" in report.reason_codes
    assert [
        finding.field_name
        for finding in report.findings
        if finding.reason_code == "unsupported_field"
    ] == [""]


@pytest.mark.parametrize(
    "field_name",
    [
        "policy_id",
        "max_epsilon",
        "max_delta",
        "epsilon_scope",
        "delta_scope",
        "accountant",
        "composition",
        "exhaustion",
    ],
)
def test_each_missing_required_field_is_reported(field_name: str) -> None:
    payload = {key: value for key, value in MINIMAL.items() if key != field_name}
    report = validate_dp_budget_policy(payload)
    assert report.valid is False
    assert "missing_field" in report.reason_codes
    assert field_name in {finding.field_name for finding in report.findings}


def test_builder_requires_every_safety_relevant_field() -> None:
    with pytest.raises(DPBudgetPolicyRejected) as failure:
        build_dp_budget_policy(policy_id="round-policy", max_epsilon=8.0)
    reasons = [finding.reason_code for finding in failure.value.report.findings]
    assert reasons == ["missing_field"] * 6
    assert failure.value.report.valid is False
    assert "round-policy" not in str(failure.value)


def test_builder_rejects_unsupported_keyword_fields() -> None:
    with pytest.raises(DPBudgetPolicyRejected) as failure:
        build_dp_budget_policy(**{**MINIMAL, "site_id": "SITE-A"})
    assert failure.value.report.reason_codes == ("unsupported_field",)


def test_builder_returns_a_policy_for_the_minimal_fields() -> None:
    policy = build_dp_budget_policy(**MINIMAL)
    assert policy.delta_prime == 0.0
    assert policy.min_rounds == 1
    assert policy.max_rounds is None


def test_policy_constructor_rejects_an_invalid_payload() -> None:
    with pytest.raises(DPBudgetPolicyRejected) as failure:
        DPBudgetPolicy(**{**MINIMAL, "max_delta": 1.5})
    assert "delta_out_of_range" in failure.value.report.reason_codes


@pytest.mark.parametrize(
    "field_name",
    [
        "policy_id",
        "max_epsilon",
        "max_delta",
        "delta_prime",
        "epsilon_scope",
        "delta_scope",
        "accountant",
        "composition",
        "exhaustion",
        "min_rounds",
        "max_rounds",
        "schema_version",
    ],
)
def test_to_dict_round_trip_needs_every_field(field_name: str) -> None:
    payload = build_dp_budget_policy(**MINIMAL).to_dict()
    del payload[field_name]
    with pytest.raises(
        DPBudgetPolicyError, match="invalid privacy budget policy fields"
    ):
        DPBudgetPolicy.from_dict(payload)


def test_from_dict_rejects_unknown_fields() -> None:
    payload = build_dp_budget_policy(**MINIMAL).to_dict()
    payload["site_id"] = "SITE-A"
    with pytest.raises(
        DPBudgetPolicyError, match="invalid privacy budget policy fields"
    ):
        DPBudgetPolicy.from_dict(payload)


@pytest.mark.parametrize(
    "policy_id",
    [
        "Round Policy",
        "round policy",
        "",
        "1round",
        "round__policy!",
        "a" * 65,
        "site-A",
    ],
)
def test_malformed_policy_identifiers_are_rejected(policy_id: str) -> None:
    assert "invalid_policy_id" in _reasons(policy_id=policy_id)


@pytest.mark.parametrize("policy_id", ["round-policy", "round_policy.2", "a" * 64])
def test_well_formed_policy_identifiers_are_accepted(policy_id: str) -> None:
    assert _report(policy_id=policy_id).valid is True


@pytest.mark.parametrize("policy_id", [3, None, ["round-policy"]])
def test_non_string_policy_identifiers_are_rejected(policy_id: object) -> None:
    assert _fields(policy_id=policy_id)["policy_id"] == "invalid_field_type"


@pytest.mark.parametrize(
    "schema_version",
    ["openmed.training.dp_budget_policy.v0", "", "v1", 1, None],
)
def test_incompatible_schema_versions_are_rejected(schema_version: object) -> None:
    reasons = _reasons(schema_version=schema_version)
    assert reasons & {"unsupported_schema_version", "invalid_field_type"}


@pytest.mark.parametrize("field_name", ["max_epsilon", "max_delta", "delta_prime"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_non_finite_numbers_are_rejected(field_name: str, value: float) -> None:
    assert "non_finite_value" in _reasons(**{field_name: value})


@pytest.mark.parametrize("field_name", ["max_epsilon", "max_delta", "delta_prime"])
@pytest.mark.parametrize("value", [True, False, "1.0", None, [1.0]])
def test_non_numeric_bounds_are_rejected(field_name: str, value: object) -> None:
    assert _fields(**{field_name: value})[field_name] == "invalid_field_type"


@pytest.mark.parametrize("value", [0, -1.0, DP_BUDGET_MAX_EPSILON * 2, 1e9])
def test_epsilon_bounds_are_enforced(value: float) -> None:
    assert "epsilon_out_of_range" in _reasons(max_epsilon=value)


@pytest.mark.parametrize("value", [DP_BUDGET_MAX_EPSILON, 1e-9, 0.5])
def test_epsilon_boundaries_are_accepted(value: float) -> None:
    assert _report(max_epsilon=value).valid is True


@pytest.mark.parametrize("value", [0, 1, 1.5, -1e-9])
def test_delta_bounds_are_enforced(value: float) -> None:
    assert "delta_out_of_range" in _reasons(max_delta=value)


@pytest.mark.parametrize("value", [1e-12, 0.5, 0.999999])
def test_delta_boundaries_are_accepted(value: float) -> None:
    assert _report(max_delta=value).valid is True


@pytest.mark.parametrize("scope_field", ["epsilon_scope", "delta_scope"])
@pytest.mark.parametrize(
    "value", ["per-round", "per round", "TOTAL", "", None, 2, True]
)
def test_ambiguous_unit_scopes_are_rejected(scope_field: str, value: object) -> None:
    assert _fields(**{scope_field: value})[scope_field] == "ambiguous_scope"


@pytest.mark.parametrize(
    "value", ["total", "per_round", DPBudgetScope.TOTAL, DPBudgetScope.PER_ROUND]
)
def test_declared_unit_scopes_are_accepted(value: object) -> None:
    assert _report(epsilon_scope=value, delta_scope=value).valid is True


@pytest.mark.parametrize("value", ["rdp", "Renyi", "", None, 3])
def test_unsupported_accountants_are_rejected(value: object) -> None:
    assert "unsupported_accountant" in _reasons(accountant=value)


@pytest.mark.parametrize(
    "value", ["basic", "renyi", "zcdp", "gaussian", DPAccountant.BASIC]
)
def test_declared_accountants_are_accepted(value: object) -> None:
    assert _report(accountant=value).valid is True


@pytest.mark.parametrize(
    "value", ["continue", "warn", "ignore", "permissive", "", None, True]
)
def test_permissive_exhaustion_values_are_rejected(value: object) -> None:
    assert _fields(exhaustion=value)["exhaustion"] == "unsafe_exhaustion"


@pytest.mark.parametrize(
    "value",
    ["fail_closed", "refuse_round", "require_review", DPExhaustion.REQUIRE_REVIEW],
)
def test_fail_closed_exhaustion_values_are_accepted(value: object) -> None:
    assert _report(exhaustion=value).valid is True


@pytest.mark.parametrize("value", [None, "RDP", "Basic", 1])
def test_unsupported_compositions_are_rejected(value: object) -> None:
    assert "unsupported_composition" in _reasons(composition=value)


def test_advanced_composition_requires_positive_slack() -> None:
    assert "unsupported_composition" in _reasons(
        composition="advanced", delta_prime=0.0
    )
    assert _report(composition="advanced", delta_prime=1e-9).valid is True


def test_basic_composition_rejects_a_non_zero_slack() -> None:
    assert "unsupported_composition" in _reasons(composition="basic", delta_prime=1e-9)


@pytest.mark.parametrize("delta_prime", [-1e-9, 1e-5, 0.5, 1.0])
def test_delta_prime_bounds_are_enforced(delta_prime: float) -> None:
    assert "delta_prime_out_of_range" in _reasons(delta_prime=delta_prime)


@pytest.mark.parametrize("delta_prime", [1e-12, 1e-6])
def test_delta_prime_boundaries_are_accepted(delta_prime: float) -> None:
    payload = {**ADVANCED, "delta_prime": delta_prime, "max_delta": 0.5}
    assert validate_dp_budget_policy(payload).valid is True


def test_basic_composition_accepts_a_zero_slack() -> None:
    assert _report(delta_prime=0.0).valid is True


@pytest.mark.parametrize("value", [0, -1, 1.5, True, "1", None])
def test_invalid_minimum_rounds_are_rejected(value: object) -> None:
    assert "invalid_round_bounds" in _reasons(min_rounds=value)


@pytest.mark.parametrize("value", [0, -3, 1.5, True, "4", [4]])
def test_invalid_maximum_rounds_are_rejected(value: object) -> None:
    assert "invalid_round_bounds" in _reasons(max_rounds=value)


def test_maximum_rounds_below_the_minimum_are_rejected() -> None:
    report = _report(min_rounds=5, max_rounds=4)
    assert "invalid_round_bounds" in report.reason_codes
    assert _report(min_rounds=5, max_rounds=5).valid is True
    assert _report(min_rounds=1, max_rounds=None).valid is True


def test_findings_are_sorted_and_unique() -> None:
    report = _report(
        policy_id="Bad Id",
        max_epsilon=float("inf"),
        max_delta=2.0,
        epsilon_scope="per-round",
        delta_scope=None,
        accountant="trust-me",
        composition="advanced",
        exhaustion="continue",
        delta_prime=-1.0,
        min_rounds=0,
        max_rounds=0,
        schema_version="v0",
    )
    observed = [
        (finding.field_name, finding.reason_code) for finding in report.findings
    ]
    assert observed == sorted(
        observed,
        key=lambda item: (
            DP_BUDGET_POLICY_FIELDS.index(item[0])
            if item[0] in DP_BUDGET_POLICY_FIELDS
            else len(DP_BUDGET_POLICY_FIELDS),
            item[1],
        ),
    )
    keys = [
        (finding.reason_code, finding.field_name, finding.observed)
        for finding in report.findings
    ]
    assert len(keys) == len(set(keys))
    assert len(report.reason_codes) == len(set(report.reason_codes))


def test_report_reasons_stay_inside_the_closed_vocabulary() -> None:
    report = _report(policy_id="Bad Id", exhaustion="continue", accountant="trust-me")
    assert set(report.reason_codes) <= DP_BUDGET_POLICY_REASON_CODES
    assert isinstance(DP_BUDGET_POLICY_REASON_CODES, frozenset)
    assert "unsafe_exhaustion" in DP_BUDGET_POLICY_REASON_CODES


@pytest.mark.parametrize("sentinel", SENTINELS)
def test_rejected_reports_never_echo_caller_identifiers(sentinel: str) -> None:
    report = _report(policy_id=sentinel, extra=1)
    assert report.valid is False
    assert sentinel not in report.to_json()
    assert sentinel not in json.dumps(report.to_dict())
    assert sentinel not in str(report)
    for finding in report.findings:
        assert sentinel not in str(finding)
    assert report.policy_id == ""


@pytest.mark.parametrize("sentinel", SENTINELS)
def test_rejections_never_echo_values_in_their_message(sentinel: str) -> None:
    with pytest.raises(DPBudgetPolicyError) as failure:
        build_dp_budget_policy(
            **{**MINIMAL, "policy_id": sentinel, "max_epsilon": float("nan")}
        )
    assert sentinel not in str(failure.value)
    assert "nan" not in str(failure.value)


@pytest.mark.parametrize("sentinel", SENTINELS)
def test_serialized_policies_never_carry_identifiers(sentinel: str) -> None:
    policy = build_dp_budget_policy(**MINIMAL)
    document = policy.to_json(indent=2)
    assert sentinel not in document
    for key in ("site", "cohort", "participant", "patient", "subject"):
        assert key not in policy.to_dict()
    assert "validation" in policy_module.__doc__ or "contract" in policy_module.__doc__


def test_validation_runs_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_socket(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("the policy contract opened a socket")

    monkeypatch.setattr(socket, "socket", fail_socket)
    monkeypatch.setattr(socket, "create_connection", fail_socket)
    policy = build_dp_budget_policy(**ADVANCED)
    report = validate_dp_budget_policy(policy.to_dict())
    assert report.valid is True
    assert report.policy_digest == fingerprint_dp_budget_policy(policy)
    assert validate_dp_budget_policy(_payload(max_delta=2.0)).valid is False


def test_public_names_are_registered_on_the_training_package() -> None:
    from openmed import training

    assert policy_module.__all__
    for name in policy_module.__all__:
        assert name in training.__all__, name
        assert getattr(policy_module, name) is not None


def test_module_exports_the_documented_surface() -> None:
    assert set(policy_module.__all__) == {
        "DP_BUDGET_MAX_EPSILON",
        "DP_BUDGET_POLICY_FIELDS",
        "DP_BUDGET_POLICY_REASON_CODES",
        "DP_BUDGET_POLICY_SCHEMA_VERSION",
        "DPAccountant",
        "DPBudgetPolicy",
        "DPBudgetPolicyError",
        "DPBudgetPolicyFinding",
        "DPBudgetPolicyRejected",
        "DPBudgetPolicyReport",
        "DPBudgetScope",
        "DPComposition",
        "DPExhaustion",
        "build_dp_budget_policy",
        "fingerprint_dp_budget_policy",
        "validate_dp_budget_policy",
    }


def test_accountant_and_composition_members_are_serialized_as_values() -> None:
    policy = build_dp_budget_policy(**MINIMAL)
    assert policy.to_dict()["accountant"] == "renyi"
    assert policy.to_dict()["composition"] == "basic"
    assert policy.to_dict()["exhaustion"] == "require_review"


def test_enum_aliases_keep_the_module_contract_stable() -> None:
    assert K is DPAccountant.RENYI
    assert P is DPComposition.BASIC
    assert U is DPExhaustion.FAIL_CLOSED
    assert S is DPBudgetScope.TOTAL
