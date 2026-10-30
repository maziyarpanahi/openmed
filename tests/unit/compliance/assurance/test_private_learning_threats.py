"""Tests for deployment-specific private-learning threat reports.

Every declaration in this module is synthetic. Sentinel strings stand in for
site names, hostnames, paths, and counts that must never reach a report, a
``repr``, or an exception message.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import re
import socket
from pathlib import Path
from typing import Any

import pytest

from openmed.compliance.assurance.private_learning_threats import (
    CONTROL_CATALOG,
    DECLARATION_SCHEMA,
    DECLARATION_SCHEMA_VERSION,
    MITIGATED_BY_DECLARED_CONTROL,
    NOT_APPLICABLE,
    PARTIALLY_MITIGATED,
    PRIVATE_LEARNING_THREAT_DISCLAIMER,
    PRIVATE_LEARNING_THREAT_REPORT_SCHEMA_VERSION,
    PRIVATE_LEARNING_THREAT_RULES_VERSION,
    REASON_CODES,
    SEVERITIES,
    THREAT_CATALOG,
    UNDECLARED,
    UNRESOLVED,
    PrivateLearningDeployment,
    PrivateLearningThreatError,
    PrivateLearningThreatReport,
    build_private_learning_threat_report,
)
from tests.fixtures.private_learning_forbidden import FORBIDDEN_FIELD_CASES

_ROOT = Path(__file__).resolve().parents[4]
_MODULE = _ROOT / "openmed" / "compliance" / "assurance" / "private_learning_threats.py"
_DOC = _ROOT / "docs" / "compliance" / "assurance" / "private-learning-threats.md"

_SENTINELS = (
    "SENTINEL-SITE-NAME-7f3a",
    "sentinel-host.invalid",
    "/sentinel/private/path",
    "4242-sentinel-patients",
)
_FORBIDDEN_CLAIM_TOKENS = frozenset(
    {"certified", "compliant", "safe", "guaranteed", "approved"}
)
_STATUSES = (
    MITIGATED_BY_DECLARED_CONTROL,
    PARTIALLY_MITIGATED,
    UNRESOLVED,
    NOT_APPLICABLE,
)

# A fully declared synthetic cross-silo deployment with every control declared.
_STRONG: dict[str, dict[str, str]] = {
    "topology": {"training_mode": "federated", "federation_scope": "cross_silo"},
    "trust": {
        "coordinator": "honest_but_curious",
        "clients": "honest_but_curious",
        "client_custody": "site_operated",
    },
    "secure_aggregation": {
        "protocol": "threshold_secret_sharing",
        "collusion_assumption": "below_threshold",
    },
    "aggregation": {
        "minimum_cohort": "enforced",
        "update_clipping": "enforced",
        "update_validation": "schema_allowlist",
        "robust_aggregation": "norm_bounding",
    },
    "differential_privacy": {
        "noise_location": "client_local",
        "privacy_unit": "record_level",
        "accountant": "basic_composition",
    },
    "cryptography": {
        "transport": "mutual_tls",
        "update_signing": "per_client_signatures",
        "replay_protection": "round_bound_nonces",
        "at_rest_encryption": "encrypted",
    },
    "network": {"topology": "vpn"},
    "data_flows": {
        "raw_records_leave_site": "never",
        "model_release": "internal_only",
        "metric_release": "aggregate_envelopes",
    },
    "retention": {
        "client_updates": "round_scoped",
        "checkpoints": "retained",
        "synthetic_data": "not_retained",
        "logs": "aggregate_only",
        "schedule": "enforced",
    },
    "synthetic_data": {
        "release": "external",
        "generator_privacy": "differentially_private",
    },
    "audits": {
        "membership_inference": "self_test",
        "synthetic_memorization": "release_gate",
    },
}


def _with(*, drop: tuple[str, ...] = (), **overrides: str) -> dict[str, Any]:
    """Return a copy of the strong declaration with dotted-path overrides.

    Keyword names use ``__`` in place of the dot, for example
    ``trust__coordinator="malicious"``.
    """
    declaration = copy.deepcopy(_STRONG)
    for key, value in overrides.items():
        section, field = key.split("__", 1)
        declaration[section][field] = value
    for path in drop:
        section, field = path.split(".", 1)
        del declaration[section][field]
    return declaration


def _finding(report: PrivateLearningThreatReport, threat: str) -> dict[str, Any]:
    matches = [
        item for item in report.to_dict()["findings"] if item["threat"] == threat
    ]
    assert len(matches) == 1
    return matches[0]


def _report(declaration: Any) -> PrivateLearningThreatReport:
    return build_private_learning_threat_report(declaration)


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z]+", text.lower()))


def _missing_references(root: Path, references: tuple[str, ...]) -> list[str]:
    return [ref for ref in references if not (root / ref).is_file()]


# -- catalog and determinism -------------------------------------------------


def test_threat_catalog_has_stable_ids_and_required_threat_classes() -> None:
    ids = [threat_id for threat_id, _ in THREAT_CATALOG]
    names = {name for _, name in THREAT_CATALOG}

    assert ids == sorted(ids)
    assert len(ids) == len(set(ids))
    assert all(re.fullmatch(r"PLT-[0-9]{3}", item) for item in ids)
    assert {
        "update_inversion",
        "poisoning_backdoor",
        "collusion",
        "membership_inference",
        "synthetic_memorization",
    } <= names


def test_strong_declaration_maps_every_threat_to_declared_controls() -> None:
    payload = _report(_STRONG).to_dict()

    assert payload["schema_version"] == PRIVATE_LEARNING_THREAT_REPORT_SCHEMA_VERSION
    assert payload["rules_version"] == PRIVATE_LEARNING_THREAT_RULES_VERSION
    assert [item["threat_id"] for item in payload["findings"]] == [
        threat_id for threat_id, _ in THREAT_CATALOG
    ]
    assert {item["status"] for item in payload["findings"]} == {
        MITIGATED_BY_DECLARED_CONTROL
    }
    assert payload["unresolved_threat_ids"] == []
    assert payload["status_counts"][MITIGATED_BY_DECLARED_CONTROL] == len(
        THREAT_CATALOG
    )


def test_report_is_byte_stable_and_digest_covers_the_payload() -> None:
    first = _report(_STRONG)
    second = _report(copy.deepcopy(_STRONG))
    reordered = {
        section: dict(reversed(list(fields.items())))
        for section, fields in reversed(list(_STRONG.items()))
    }
    third = _report(reordered)

    assert first.to_json() == second.to_json() == third.to_json()
    assert first.to_markdown() == second.to_markdown() == third.to_markdown()
    payload = first.to_dict()
    digest = payload.pop("report_digest")
    canonical = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    assert digest == "sha256:" + hashlib.sha256(canonical).hexdigest()
    assert first.report_digest == digest
    assert json.loads(first.to_json()) == first.to_dict()


def test_rules_v1_output_is_pinned_to_golden_digests() -> None:
    # A changed digest means the rule table or report schema changed: bump
    # the rules or schema version and update these pins in the same change.
    assert _report(_STRONG).report_digest == (
        "sha256:2c6180bc19afa043e7766d6f753f78975e66e61f91586fab3a2122fac10dfb4f"
    )
    assert _report({}).report_digest == (
        "sha256:a76f2208b93e53d8ed36d8c3da3a9e8eba513274ec7b306e8581f615dc63f034"
    )


def test_declaration_digest_is_bound_and_changes_with_declaration() -> None:
    strong = _report(_STRONG).to_dict()
    weaker = _report(_with(cryptography__transport="tls")).to_dict()

    assert re.fullmatch(r"sha256:[0-9a-f]{64}", strong["declaration_digest"])
    assert strong["declaration_digest"] != weaker["declaration_digest"]
    assert strong["report_digest"] != weaker["report_digest"]


def test_report_mutation_of_input_after_build_has_no_effect() -> None:
    declaration = copy.deepcopy(_STRONG)
    report = _report(declaration)
    before = report.to_json()
    declaration["trust"]["coordinator"] = "malicious"
    exported = report.to_dict()
    exported["findings"].clear()

    assert report.to_json() == before


def test_declaration_round_trips_and_explicit_undeclared_equals_omission() -> None:
    deployment = PrivateLearningDeployment.from_dict(_STRONG)
    assert PrivateLearningDeployment.from_dict(deployment.to_dict()) == deployment
    assert _report(deployment).to_json() == _report(_STRONG).to_json()

    omitted = _with(drop=("trust.coordinator",))
    explicit = _with(trust__coordinator=UNDECLARED)
    assert _report(omitted).to_json() == _report(explicit).to_json()


# -- update inversion --------------------------------------------------------


def test_malicious_coordinator_without_secure_aggregation_flags_inversion() -> None:
    report = _report(
        _with(
            trust__coordinator="malicious",
            secure_aggregation__protocol="none",
            differential_privacy__noise_location="none",
        )
    )
    finding = _finding(report, "update_inversion")

    assert finding["status"] == UNRESOLVED
    assert finding["severity"] == "critical"
    assert "individual_updates_exposed_to_coordinator" in finding["reason_codes"]
    assert finding["threat_id"] in report.to_dict()["unresolved_threat_ids"]


def test_secure_aggregation_partially_mitigates_malicious_coordinator() -> None:
    finding = _finding(
        _report(
            _with(
                trust__coordinator="malicious",
                differential_privacy__noise_location="none",
            )
        ),
        "update_inversion",
    )

    assert finding["status"] == PARTIALLY_MITIGATED
    assert "malicious_coordinator_can_isolate_updates" in finding["reason_codes"]


def test_honest_but_curious_coordinator_needs_cohort_with_secure_aggregation() -> None:
    mitigated = _finding(
        _report(_with(differential_privacy__noise_location="none")),
        "update_inversion",
    )
    small_cohort = _finding(
        _report(
            _with(
                differential_privacy__noise_location="none",
                aggregation__minimum_cohort="none",
            )
        ),
        "update_inversion",
    )

    assert mitigated["status"] == MITIGATED_BY_DECLARED_CONTROL
    assert small_cohort["status"] == PARTIALLY_MITIGATED
    assert "small_cohort_aggregate_exposure" in small_cohort["reason_codes"]


def test_central_noise_does_not_protect_against_curious_coordinator() -> None:
    finding = _finding(
        _report(
            _with(
                secure_aggregation__protocol="none",
                differential_privacy__noise_location="central",
            )
        ),
        "update_inversion",
    )

    assert finding["status"] == UNRESOLVED
    assert "central_noise_added_after_coordinator_access" in finding["reason_codes"]


def test_third_party_client_custody_caps_inversion_mitigation() -> None:
    finding = _finding(
        _report(_with(trust__client_custody="third_party_hosted")),
        "update_inversion",
    )

    assert finding["status"] == PARTIALLY_MITIGATED
    assert "client_host_observes_unprotected_updates" in finding["reason_codes"]


def test_centralized_training_makes_federated_threats_not_applicable() -> None:
    declaration = _with(
        topology__training_mode="centralized",
        topology__federation_scope="not_federated",
    )
    report = _report(declaration)

    for threat in (
        "update_inversion",
        "poisoning_backdoor",
        "collusion",
        "update_replay",
        "metric_disclosure",
        "transport_interception",
    ):
        finding = _finding(report, threat)
        assert finding["status"] == NOT_APPLICABLE, threat
        assert finding["severity"] == "none"
        assert finding["controls"] == []


# -- poisoning ---------------------------------------------------------------


def test_malicious_clients_without_defense_flag_poisoning() -> None:
    finding = _finding(
        _report(
            _with(
                trust__clients="malicious",
                aggregation__robust_aggregation="none",
            )
        ),
        "poisoning_backdoor",
    )

    assert finding["status"] == UNRESOLVED
    assert finding["severity"] == "high"
    assert "no_poisoning_defense_declared" in finding["reason_codes"]


def test_robust_aggregation_only_partially_mitigates_backdoors() -> None:
    finding = _finding(
        _report(
            _with(
                trust__clients="malicious",
                aggregation__robust_aggregation="byzantine_robust",
            )
        ),
        "poisoning_backdoor",
    )

    assert finding["status"] == PARTIALLY_MITIGATED
    assert (
        "robust_aggregation_does_not_exclude_targeted_backdoors"
        in finding["reason_codes"]
    )
    assert "secure_aggregation_limits_update_inspection" in finding["reason_codes"]
    robust = next(
        c for c in finding["controls"] if c["control"] == "robust_aggregation"
    )
    assert robust["declared"] is True
    assert robust["openmed_support"] == "deployment_provided"
    assert robust["references"] == []


def test_cross_device_without_signatures_flags_sybil_poisoning() -> None:
    finding = _finding(
        _report(
            _with(
                topology__federation_scope="cross_device",
                cryptography__update_signing="none",
            )
        ),
        "poisoning_backdoor",
    )

    assert finding["status"] == UNRESOLVED
    assert "sybil_client_enrollment" in finding["reason_codes"]


# -- collusion ---------------------------------------------------------------


def test_collusion_reaching_secure_aggregation_threshold_is_unresolved() -> None:
    finding = _finding(
        _report(
            _with(
                differential_privacy__noise_location="none",
                secure_aggregation__collusion_assumption="may_reach_threshold",
            )
        ),
        "collusion",
    )

    assert finding["status"] == UNRESOLVED
    assert finding["severity"] == "critical"
    assert "collusion_can_reach_secure_aggregation_threshold" in finding["reason_codes"]


def test_collusion_below_threshold_is_flagged_as_unverified_assumption() -> None:
    finding = _finding(
        _report(_with(differential_privacy__noise_location="none")),
        "collusion",
    )

    assert finding["status"] == PARTIALLY_MITIGATED
    assert (
        "secure_aggregation_threshold_assumption_unverified" in finding["reason_codes"]
    )


def test_malicious_cross_device_coordinator_can_enroll_sybils() -> None:
    finding = _finding(
        _report(
            _with(
                trust__coordinator="malicious",
                topology__federation_scope="cross_device",
                differential_privacy__noise_location="none",
            )
        ),
        "collusion",
    )

    assert finding["status"] == UNRESOLVED
    assert "coordinator_can_enroll_sybil_clients" in finding["reason_codes"]
    assert "coordinator_counts_toward_collusion_threshold" in finding["reason_codes"]


def test_collusion_assumption_is_required_when_secure_aggregation_is_declared() -> None:
    finding = _finding(
        _report(
            _with(
                differential_privacy__noise_location="none",
                drop=("secure_aggregation.collusion_assumption",),
            )
        ),
        "collusion",
    )

    assert finding["status"] == UNRESOLVED
    assert finding["undeclared_inputs"] == ["secure_aggregation.collusion_assumption"]


def test_client_local_noise_mitigates_collusion() -> None:
    finding = _finding(
        _report(_with(secure_aggregation__collusion_assumption="may_reach_threshold")),
        "collusion",
    )

    assert finding["status"] == MITIGATED_BY_DECLARED_CONTROL
    assert "client_side_noise_declared" in finding["reason_codes"]


# -- membership inference ----------------------------------------------------


def test_membership_inference_without_dp_or_audit_is_unresolved() -> None:
    finding = _finding(
        _report(
            _with(
                differential_privacy__noise_location="none",
                audits__membership_inference="none",
                data_flows__model_release="public",
            )
        ),
        "membership_inference",
    )

    assert finding["status"] == UNRESOLVED
    assert finding["severity"] == "critical"
    assert "no_membership_control_declared" in finding["reason_codes"]


def test_membership_audit_alone_is_partial_and_dp_mitigates() -> None:
    audit_only = _finding(
        _report(_with(differential_privacy__noise_location="none")),
        "membership_inference",
    )
    unaccounted = _finding(
        _report(_with(differential_privacy__accountant="none")),
        "membership_inference",
    )
    accounted = _finding(_report(_STRONG), "membership_inference")

    assert audit_only["status"] == PARTIALLY_MITIGATED
    assert "empirical_audit_only" in audit_only["reason_codes"]
    assert unaccounted["status"] == PARTIALLY_MITIGATED
    assert "noise_without_privacy_accounting" in unaccounted["reason_codes"]
    assert accounted["status"] == MITIGATED_BY_DECLARED_CONTROL
    assert "privacy_parameters_not_evaluated" in accounted["reason_codes"]


# -- synthetic memorization --------------------------------------------------


def test_synthetic_release_without_dp_flags_memorization() -> None:
    finding = _finding(
        _report(
            _with(
                synthetic_data__generator_privacy="none",
                audits__synthetic_memorization="none",
            )
        ),
        "synthetic_memorization",
    )

    assert finding["status"] == UNRESOLVED
    assert finding["severity"] == "high"
    assert "synthetic_release_without_privacy_or_audit" in finding["reason_codes"]


def test_memorization_gate_without_dp_is_partial() -> None:
    finding = _finding(
        _report(_with(synthetic_data__generator_privacy="none")),
        "synthetic_memorization",
    )

    assert finding["status"] == PARTIALLY_MITIGATED
    assert "memorization_gate_without_formal_privacy" in finding["reason_codes"]
    gate = next(c for c in finding["controls"] if c["control"] == "memorization_audit")
    assert "openmed/eval/synthetic_memorization.py" in gate["references"]


def test_no_synthetic_release_makes_memorization_not_applicable() -> None:
    finding = _finding(
        _report(
            _with(
                synthetic_data__release="none",
                synthetic_data__generator_privacy="none",
                audits__synthetic_memorization="none",
            )
        ),
        "synthetic_memorization",
    )

    assert finding["status"] == NOT_APPLICABLE
    assert finding["reason_codes"] == ["no_synthetic_release"]


# -- retention, transport, replay, metrics -----------------------------------


def test_content_bearing_logs_flag_retention_exposure() -> None:
    finding = _finding(
        _report(_with(retention__logs="content_bearing")), "retention_exposure"
    )

    assert finding["status"] == UNRESOLVED
    assert finding["severity"] == "critical"
    assert "content_bearing_logs_retained" in finding["reason_codes"]


def test_retained_client_updates_stay_partial_even_with_controls() -> None:
    finding = _finding(
        _report(_with(retention__client_updates="retained")), "retention_exposure"
    )
    uncontrolled = _finding(
        _report(
            _with(
                retention__client_updates="retained",
                retention__schedule="none",
                cryptography__at_rest_encryption="none",
            )
        ),
        "retention_exposure",
    )

    assert finding["status"] == PARTIALLY_MITIGATED
    assert "client_updates_retained_beyond_round" in finding["reason_codes"]
    assert uncontrolled["status"] == UNRESOLVED


def test_nothing_retained_makes_retention_not_applicable() -> None:
    finding = _finding(
        _report(
            _with(
                retention__checkpoints="not_retained",
                retention__logs="none",
                retention__schedule="none",
            )
        ),
        "retention_exposure",
    )

    assert finding["status"] == NOT_APPLICABLE


def test_unencrypted_internet_transport_is_unresolved() -> None:
    finding = _finding(
        _report(
            _with(cryptography__transport="none", network__topology="public_internet")
        ),
        "transport_interception",
    )
    air_gapped = _finding(
        _report(_with(cryptography__transport="none", network__topology="air_gapped")),
        "transport_interception",
    )

    assert finding["status"] == UNRESOLVED
    assert "unencrypted_transport" in finding["reason_codes"]
    assert air_gapped["status"] == PARTIALLY_MITIGATED


def test_centralized_raw_egress_keeps_transport_threat_applicable() -> None:
    finding = _finding(
        _report(
            _with(
                topology__training_mode="centralized",
                topology__federation_scope="not_federated",
                data_flows__raw_records_leave_site="raw",
                cryptography__transport="none",
                network__topology="public_internet",
            )
        ),
        "transport_interception",
    )

    assert finding["status"] == UNRESOLVED
    assert finding["severity"] == "critical"
    assert "raw_records_cross_site_boundary" in finding["reason_codes"]


def test_unbound_unsigned_updates_flag_replay() -> None:
    finding = _finding(
        _report(
            _with(
                cryptography__replay_protection="none",
                cryptography__update_signing="none",
            )
        ),
        "update_replay",
    )
    signed_only = _finding(
        _report(_with(cryptography__replay_protection="none")), "update_replay"
    )

    assert finding["status"] == UNRESOLVED
    assert "updates_not_bound_to_round" in finding["reason_codes"]
    assert signed_only["status"] == PARTIALLY_MITIGATED


def test_per_client_metrics_flag_metric_disclosure() -> None:
    per_client = _finding(
        _report(_with(data_flows__metric_release="per_client")), "metric_disclosure"
    )
    no_cohort = _finding(
        _report(_with(aggregation__minimum_cohort="none")), "metric_disclosure"
    )
    no_release = _finding(
        _report(_with(data_flows__metric_release="none")), "metric_disclosure"
    )

    assert per_client["status"] == UNRESOLVED
    assert "per_client_metrics_released" in per_client["reason_codes"]
    assert no_cohort["status"] == PARTIALLY_MITIGATED
    assert no_release["status"] == NOT_APPLICABLE


# -- undeclared inputs -------------------------------------------------------


def test_empty_declaration_yields_only_unresolved_findings() -> None:
    report = _report({})
    payload = report.to_dict()

    assert {item["status"] for item in payload["findings"]} == {UNRESOLVED}
    assert all(item["undeclared_inputs"] for item in payload["findings"])
    assert all(
        "undeclared_input" in item["reason_codes"] for item in payload["findings"]
    )
    assert payload["status_counts"][MITIGATED_BY_DECLARED_CONTROL] == 0
    assert len(payload["unresolved_threat_ids"]) == len(THREAT_CATALOG)


def test_each_dropped_field_never_improves_any_finding(subtests) -> None:
    rank = {
        NOT_APPLICABLE: 0,
        MITIGATED_BY_DECLARED_CONTROL: 1,
        PARTIALLY_MITIGATED: 2,
        UNRESOLVED: 3,
    }
    baseline = {
        item["threat_id"]: item["status"]
        for item in _report(_STRONG).to_dict()["findings"]
    }
    for section, fields in DECLARATION_SCHEMA.items():
        for field in fields:
            path = f"{section}.{field}"
            with subtests.test(path=path):
                payload = _report(_with(drop=(path,))).to_dict()
                for item in payload["findings"]:
                    before = baseline[item["threat_id"]]
                    if item["undeclared_inputs"]:
                        assert item["status"] == UNRESOLVED
                        assert path in item["undeclared_inputs"]
                    assert rank[item["status"]] >= rank[before]


def test_undeclared_dp_subfields_are_required_once_noise_is_declared() -> None:
    finding = _finding(
        _report(_with(drop=("differential_privacy.accountant",))),
        "membership_inference",
    )

    assert finding["status"] == UNRESOLVED
    assert finding["undeclared_inputs"] == ["differential_privacy.accountant"]


# -- closed declaration and value-free errors --------------------------------


def _assert_value_free(exc: PrivateLearningThreatError, *markers: str) -> None:
    text = f"{exc}|{exc!r}|{exc.args!r}"
    for marker in (*_SENTINELS, *markers):
        assert marker not in text
    assert exc.__cause__ is None
    assert exc.__suppress_context__ is True
    assert re.fullmatch(r"[a-z_.]+: [a-z_]+", str(exc))


def test_unknown_top_level_and_section_keys_are_rejected() -> None:
    top = copy.deepcopy(_STRONG)
    top[_SENTINELS[0]] = {"site": _SENTINELS[1]}
    nested = copy.deepcopy(_STRONG)
    nested["network"][_SENTINELS[1]] = _SENTINELS[2]

    with pytest.raises(PrivateLearningThreatError) as top_error:
        _report(top)
    with pytest.raises(PrivateLearningThreatError) as nested_error:
        _report(nested)

    assert str(top_error.value) == "declaration: unknown_field"
    assert str(nested_error.value) == "network: unknown_field"
    _assert_value_free(top_error.value)
    _assert_value_free(nested_error.value)


def test_forbidden_private_learning_fields_are_rejected_value_free(subtests) -> None:
    for case in FORBIDDEN_FIELD_CASES:
        for section in (None, "trust", "data_flows"):
            with subtests.test(field=case.field, section=section):
                declaration = copy.deepcopy(_STRONG)
                target = declaration if section is None else declaration[section]
                target[case.field] = copy.deepcopy(case.value)
                with pytest.raises(PrivateLearningThreatError) as error:
                    _report(declaration)
                markers = () if case.marker is None else (case.marker,)
                _assert_value_free(error.value, *markers)
                assert case.field not in str(error.value)


def test_invalid_values_types_and_shapes_are_rejected_value_free(subtests) -> None:
    cases = [
        ("value", _with(network__topology=_SENTINELS[1]), "network.topology"),
        ("int", _with(trust__coordinator=4242), "trust.coordinator"),
        ("bool", _with(trust__clients=True), "trust.clients"),
        ("section", {**_STRONG, "network": [_SENTINELS[2]]}, "network"),
        ("root", [_SENTINELS[0]], "declaration"),
        ("key", {**_STRONG, 4242: {}}, "declaration"),
        (
            "schema",
            {**_STRONG, "schema_version": _SENTINELS[3]},
            "schema_version",
        ),
    ]
    for name, declaration, field in cases:
        with subtests.test(case=name):
            with pytest.raises(PrivateLearningThreatError) as error:
                _report(declaration)
            assert str(error.value).split(":", 1)[0] == field
            _assert_value_free(error.value, "4242")


def test_inconsistent_topology_is_rejected() -> None:
    for mode, scope in (("federated", "not_federated"), ("centralized", "cross_silo")):
        with pytest.raises(PrivateLearningThreatError) as error:
            _report(
                _with(topology__training_mode=mode, topology__federation_scope=scope)
            )
        assert str(error.value) == "topology.federation_scope: inconsistent_value"


def test_declared_schema_version_is_accepted() -> None:
    declared = {"schema_version": DECLARATION_SCHEMA_VERSION, **_STRONG}

    assert _report(declared).to_json() == _report(_STRONG).to_json()


def test_direct_report_and_finding_construction_is_rejected() -> None:
    report = _report(_STRONG)

    with pytest.raises(PrivateLearningThreatError, match="report: builder_required"):
        PrivateLearningThreatReport(
            deployment=report.deployment,
            findings=report.findings,
        )


# -- value-free output -------------------------------------------------------


def test_reports_and_reprs_contain_no_sentinels_or_counts() -> None:
    report = _report(_with(trust__coordinator="malicious"))
    deployment = PrivateLearningDeployment.from_dict(_STRONG)
    rendered = "\n".join(
        (
            report.to_json(),
            report.to_markdown(),
            repr(report),
            str(report),
            repr(deployment),
            repr(report.findings),
        )
    )

    for sentinel in _SENTINELS:
        assert sentinel not in rendered
    assert report.report_digest in repr(report)
    assert "findings" not in repr(report)
    assert rendered.isascii()


def test_report_emits_disclaimer_and_never_claims_certification(subtests) -> None:
    declarations = {
        "strong": _STRONG,
        "empty": {},
        "weak": _with(
            trust__coordinator="malicious",
            secure_aggregation__protocol="none",
            differential_privacy__noise_location="none",
        ),
    }
    for name, declaration in declarations.items():
        with subtests.test(declaration=name):
            report = _report(declaration)
            payload = report.to_dict()
            assert payload["disclaimer"] == PRIVATE_LEARNING_THREAT_DISCLAIMER
            assert PRIVATE_LEARNING_THREAT_DISCLAIMER in report.to_markdown()
            for text in (report.to_json(), report.to_markdown()):
                assert not _tokens(text) & _FORBIDDEN_CLAIM_TOKENS


def test_every_emitted_code_is_from_the_closed_vocabularies() -> None:
    declarations = (
        _STRONG,
        {},
        _with(trust__coordinator="malicious", secure_aggregation__protocol="none"),
        _with(
            retention__logs="content_bearing", data_flows__metric_release="per_client"
        ),
    )
    for declaration in declarations:
        for item in _report(declaration).to_dict()["findings"]:
            assert item["status"] in _STATUSES
            assert item["severity"] in SEVERITIES
            assert item["reason_codes"]
            assert set(item["reason_codes"]) <= REASON_CODES
            assert item["reason_codes"] == sorted(set(item["reason_codes"]))
            for control in item["controls"]:
                assert control["control"] in CONTROL_CATALOG


# -- drift and module hygiene ------------------------------------------------


def test_every_control_reference_exists_in_the_repository() -> None:
    references = tuple(ref for refs in CONTROL_CATALOG.values() for ref in refs)

    assert references
    assert _missing_references(_ROOT, references) == []
    assert all(not ref.startswith(("/", "..")) for ref in references)
    assert any(ref.startswith("openmed/") for ref in references)


def test_reference_drift_check_detects_a_missing_path(tmp_path: Path) -> None:
    (tmp_path / "openmed").mkdir()
    (tmp_path / "openmed" / "present.py").write_text("", encoding="utf-8")

    assert _missing_references(
        tmp_path, ("openmed/present.py", "openmed/absent.py")
    ) == ["openmed/absent.py"]


def test_claim_and_sentinel_scanners_have_negative_controls() -> None:
    assert _tokens("deployment is CERTIFIED and safe") & _FORBIDDEN_CLAIM_TOKENS
    assert not _tokens("secure_aggregation safety_margin") & _FORBIDDEN_CLAIM_TOKENS
    planted = PrivateLearningThreatError("trust: " + _SENTINELS[0])
    with pytest.raises(AssertionError):
        _assert_value_free(planted)


def test_report_generation_performs_no_network_call(monkeypatch) -> None:
    def _blocked(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket, "socket", _blocked)
    monkeypatch.setattr(socket, "create_connection", _blocked)

    assert _report(_STRONG).to_json()


def test_module_is_stdlib_only_and_ascii() -> None:
    source = _MODULE.read_text(encoding="utf-8")
    allowed = {
        "__future__",
        "collections.abc",
        "dataclasses",
        "hashlib",
        "json",
        "types",
        "typing",
    }
    imported: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0
            imported.add(node.module or "")

    assert imported <= allowed
    assert source.isascii()


def test_documentation_covers_every_threat_and_control() -> None:
    text = _DOC.read_text(encoding="utf-8")

    assert text.isascii()
    assert PRIVATE_LEARNING_THREAT_DISCLAIMER in text.replace("\n> ", " ")
    for threat_id, threat in THREAT_CATALOG:
        assert f"`{threat_id}`" in text
        assert f"`{threat}`" in text
    for control in CONTROL_CATALOG:
        assert f"`{control}`" in text
    for section, fields in DECLARATION_SCHEMA.items():
        for field in fields:
            assert f"`{section}.{field}`" in text
