"""Deterministic private-learning threat reports for declared deployments.

Federated and synthetic-data deployments face different threats depending on
coordinator trust, client custody, secure aggregation, differential privacy,
cryptography, network topology, data flows, and artifact retention. This
module accepts a closed, enumerated deployment declaration and maps it through
a versioned rule table to findings with stable threat identifiers, controlled
status and reason codes, and references to the OpenMed modules and pages that
support each control.

The declaration admits only enumerated values. Site names, hostnames, paths,
endpoints, and counts have no field to occupy, and unknown keys are rejected
with fixed error codes that never echo the submitted key or value. Missing or
``undeclared`` inputs make the affected finding ``unresolved``; they never
count as a mitigation. A report records which controls a deployment declares.
It does not verify that a control is implemented or configured correctly, and
it is never a certification, audit opinion, or legal advice.

Report generation is local, deterministic, and performs no network call.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import InitVar, dataclass, field
from types import MappingProxyType
from typing import Any, Final

PRIVATE_LEARNING_THREAT_REPORT_SCHEMA_VERSION: Final = (
    "openmed.compliance.private_learning_threat_report.v1"
)
PRIVATE_LEARNING_THREAT_RULES_VERSION: Final = (
    "openmed.compliance.private_learning_threat_rules.v1"
)
DECLARATION_SCHEMA_VERSION: Final = "openmed.compliance.private_learning_deployment.v1"
PRIVATE_LEARNING_THREAT_DISCLAIMER: Final = (
    "This report maps declared deployment choices to private-learning threats "
    "with fixed rules. It is not a certification, audit opinion, or legal "
    "advice, and it does not verify that any declared control is implemented "
    "or configured correctly."
)

UNDECLARED: Final = "undeclared"

MITIGATED_BY_DECLARED_CONTROL: Final = "mitigated_by_declared_control"
PARTIALLY_MITIGATED: Final = "partially_mitigated"
UNRESOLVED: Final = "unresolved"
NOT_APPLICABLE: Final = "not_applicable"
_STATUSES: Final = (
    MITIGATED_BY_DECLARED_CONTROL,
    PARTIALLY_MITIGATED,
    UNRESOLVED,
    NOT_APPLICABLE,
)

SEVERITIES: Final = ("critical", "high", "medium", "low", "none")

_MODULE_REFERENCE: Final = "module_reference"
_DEPLOYMENT_PROVIDED: Final = "deployment_provided"


def _frozen_schema(
    raw: dict[str, dict[str, tuple[str, ...]]],
) -> Mapping[str, Mapping[str, tuple[str, ...]]]:
    return MappingProxyType(
        {section: MappingProxyType(dict(fields)) for section, fields in raw.items()}
    )


DECLARATION_SCHEMA: Final = _frozen_schema(
    {
        "topology": {
            "training_mode": ("centralized", "federated", "hybrid"),
            "federation_scope": ("not_federated", "cross_silo", "cross_device"),
        },
        "trust": {
            "coordinator": ("trusted", "honest_but_curious", "malicious"),
            "clients": ("trusted", "honest_but_curious", "malicious"),
            "client_custody": (
                "site_operated",
                "shared_infrastructure",
                "third_party_hosted",
            ),
        },
        "secure_aggregation": {
            "protocol": (
                "none",
                "pairwise_masking",
                "threshold_secret_sharing",
                "trusted_execution_environment",
            ),
            "collusion_assumption": (
                "no_collusion_assumed",
                "below_threshold",
                "may_reach_threshold",
            ),
        },
        "aggregation": {
            "minimum_cohort": ("none", "enforced"),
            "update_clipping": ("none", "enforced"),
            "update_validation": ("none", "schema_allowlist"),
            "robust_aggregation": ("none", "norm_bounding", "byzantine_robust"),
        },
        "differential_privacy": {
            "noise_location": ("none", "client_local", "distributed", "central"),
            "privacy_unit": ("record_level", "client_level"),
            "accountant": ("none", "basic_composition", "advanced_composition"),
        },
        "cryptography": {
            "transport": ("none", "tls", "mutual_tls"),
            "update_signing": ("none", "per_client_signatures"),
            "replay_protection": ("none", "round_bound_nonces"),
            "at_rest_encryption": ("none", "encrypted"),
        },
        "network": {
            "topology": ("air_gapped", "private_network", "vpn", "public_internet"),
        },
        "data_flows": {
            "raw_records_leave_site": ("never", "pseudonymized_only", "raw"),
            "model_release": ("internal_only", "external_partners", "public"),
            "metric_release": ("none", "aggregate_envelopes", "per_client"),
        },
        "retention": {
            "client_updates": ("not_retained", "round_scoped", "retained"),
            "checkpoints": ("not_retained", "retained"),
            "synthetic_data": ("not_retained", "retained"),
            "logs": ("none", "aggregate_only", "content_bearing"),
            "schedule": ("none", "enforced"),
        },
        "synthetic_data": {
            "release": ("none", "internal", "external"),
            "generator_privacy": ("none", "differentially_private"),
        },
        "audits": {
            "membership_inference": ("none", "self_test"),
            "synthetic_memorization": ("none", "release_gate"),
        },
    }
)

_PATHS: Final = tuple(
    f"{section}.{name}"
    for section, fields in DECLARATION_SCHEMA.items()
    for name in fields
)

# Repository-relative modules and pages that support each control. An empty
# tuple means OpenMed ships no module for the control: the deployment provides
# it, and the report says so instead of pointing at unrelated code.
CONTROL_CATALOG: Final[Mapping[str, tuple[str, ...]]] = MappingProxyType(
    {
        "aggregate_metric_envelopes": (
            "openmed/training/federated_metrics.py",
            "docs/training/federated-metrics.md",
        ),
        "aggregate_only_logging": ("docs/security/no-raw-phi-logging.md",),
        "at_rest_encryption": (),
        "differential_privacy": (
            "openmed/risk/differential_privacy.py",
            "docs/privacy/structured-risk-lab.md",
        ),
        "generator_privacy": (),
        "membership_audit": (
            "openmed/risk/membership_inference.py",
            "openmed/risk/membership.py",
            "docs/privacy/structured-risk-lab.md",
        ),
        "memorization_audit": (
            "openmed/eval/synthetic_memorization.py",
            "docs/evaluation/synthetic-memorization.md",
        ),
        "minimum_cohort": (
            "openmed/training/federated_status.py",
            "openmed/training/federated_metrics.py",
            "docs/training/federated-round-status.md",
        ),
        "network_isolation": (),
        "privacy_accounting": (
            "openmed/risk/differential_privacy.py",
            "openmed/risk/privacy_budget.py",
            "docs/security/privacy-budget.md",
        ),
        "replay_protection": (
            "openmed/training/federated_round.py",
            "openmed/training/federated_schedule.py",
            "docs/training/federated-round-lifecycle.md",
            "docs/training/federated-scheduling.md",
        ),
        "retention_schedule": (
            "openmed/risk/audit_retention.py",
            "openmed/risk/deletion_plan.py",
            "docs/security/audit-retention.md",
            "docs/security/deletion-plans.md",
        ),
        "robust_aggregation": (),
        "secure_aggregation": (),
        "transport_security": (),
        "update_clipping": (
            "openmed/training/federated_update_metadata.py",
            "docs/training/federated-update-metadata.md",
        ),
        "update_schema_validation": (
            "openmed/training/federated_update_metadata.py",
            "openmed/training/federated_schema_fingerprint.py",
            "docs/training/federated-update-metadata.md",
            "docs/training/federated-schema-fingerprints.md",
        ),
        "update_signing": (
            "openmed/core/key_custody.py",
            "docs/security/key-custody-metadata.md",
        ),
    }
)

# Declaration path and the values that count as declaring each control.
_CONTROL_DECLARED_BY: Final[Mapping[str, tuple[str, frozenset[str]]]] = (
    MappingProxyType(
        {
            "aggregate_metric_envelopes": (
                "data_flows.metric_release",
                frozenset({"aggregate_envelopes"}),
            ),
            "aggregate_only_logging": (
                "retention.logs",
                frozenset({"none", "aggregate_only"}),
            ),
            "at_rest_encryption": (
                "cryptography.at_rest_encryption",
                frozenset({"encrypted"}),
            ),
            "differential_privacy": (
                "differential_privacy.noise_location",
                frozenset({"client_local", "distributed", "central"}),
            ),
            "generator_privacy": (
                "synthetic_data.generator_privacy",
                frozenset({"differentially_private"}),
            ),
            "membership_audit": (
                "audits.membership_inference",
                frozenset({"self_test"}),
            ),
            "memorization_audit": (
                "audits.synthetic_memorization",
                frozenset({"release_gate"}),
            ),
            "minimum_cohort": ("aggregation.minimum_cohort", frozenset({"enforced"})),
            "network_isolation": (
                "network.topology",
                frozenset({"air_gapped", "private_network"}),
            ),
            "privacy_accounting": (
                "differential_privacy.accountant",
                frozenset({"basic_composition", "advanced_composition"}),
            ),
            "replay_protection": (
                "cryptography.replay_protection",
                frozenset({"round_bound_nonces"}),
            ),
            "retention_schedule": ("retention.schedule", frozenset({"enforced"})),
            "robust_aggregation": (
                "aggregation.robust_aggregation",
                frozenset({"norm_bounding", "byzantine_robust"}),
            ),
            "secure_aggregation": (
                "secure_aggregation.protocol",
                frozenset(
                    {
                        "pairwise_masking",
                        "threshold_secret_sharing",
                        "trusted_execution_environment",
                    }
                ),
            ),
            "transport_security": (
                "cryptography.transport",
                frozenset({"tls", "mutual_tls"}),
            ),
            "update_clipping": ("aggregation.update_clipping", frozenset({"enforced"})),
            "update_schema_validation": (
                "aggregation.update_validation",
                frozenset({"schema_allowlist"}),
            ),
            "update_signing": (
                "cryptography.update_signing",
                frozenset({"per_client_signatures"}),
            ),
        }
    )
)

REASON_CODES: Final = frozenset(
    {
        "aggregate_differencing_by_colluding_clients",
        "aggregate_envelopes_with_minimum_group",
        "aggregate_only_logs_with_retention_schedule",
        "authenticated_contributors_declared",
        "central_noise_added_after_coordinator_access",
        "checkpoints_retained",
        "client_custody_outside_site",
        "client_host_observes_unprotected_updates",
        "client_level_noise_declared",
        "client_side_noise_declared",
        "client_updates_retained_beyond_round",
        "collusion_can_reach_secure_aggregation_threshold",
        "collusion_excluded_by_assumption_only",
        "content_bearing_logs_retained",
        "coordinator_can_enroll_sybil_clients",
        "coordinator_counts_toward_collusion_threshold",
        "coordinator_trust_assumed_without_control",
        "distributed_noise_weakens_under_collusion",
        "empirical_audit_only",
        "encrypted_transport_declared",
        "formal_privacy_accounting_declared",
        "generator_privacy_and_memorization_gate_declared",
        "individual_updates_exposed_to_coordinator",
        "malicious_clients_assumed",
        "malicious_coordinator_can_isolate_updates",
        "memorization_gate_without_formal_privacy",
        "minimum_cohort_enforced",
        "minimum_cohort_not_enforced",
        "mutually_authenticated_transport_declared",
        "network_layer_encryption_only",
        "no_client_updates_shared",
        "no_cross_site_transfer",
        "no_external_update_contributors",
        "no_federated_metrics",
        "no_federated_participants",
        "no_membership_control_declared",
        "no_memorization_release_gate",
        "no_metric_release",
        "no_poisoning_defense_declared",
        "no_retained_artifacts",
        "no_synthetic_release",
        "noise_without_enforced_clipping",
        "noise_without_privacy_accounting",
        "per_client_metrics_released",
        "physical_isolation_without_transport_encryption",
        "privacy_parameters_not_evaluated",
        "pseudonymized_records_cross_site_boundary",
        "raw_records_cross_site_boundary",
        "replay_binding_without_signatures",
        "retained_artifacts_unencrypted",
        "retention_controls_declared",
        "retention_schedule_not_enforced",
        "robust_aggregation_does_not_exclude_targeted_backdoors",
        "round_bound_signed_updates",
        "secure_aggregation_declared",
        "secure_aggregation_limits_update_inspection",
        "secure_aggregation_threshold_assumption_unverified",
        "signatures_without_round_binding",
        "small_cohort_aggregate_exposure",
        "sybil_client_enrollment",
        "synthetic_data_retained",
        "synthetic_release_without_privacy_or_audit",
        "unauthenticated_update_injection",
        "undeclared_input",
        "unencrypted_transport",
        "updates_not_bound_to_round",
    }
)


class PrivateLearningThreatError(ValueError):
    """Raised with a fixed ``field: code`` message for an invalid declaration.

    Messages name only schema-defined sections or fields and never echo a
    submitted key or value.
    """


def _error(message: str) -> PrivateLearningThreatError:
    return PrivateLearningThreatError(message)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _digest(value: Any) -> str:
    encoded = _canonical_json(value).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


@dataclass(frozen=True, slots=True, repr=False)
class PrivateLearningDeployment:
    """A validated, closed private-learning deployment declaration.

    Build instances with :meth:`from_dict`. Every schema path is present; an
    omitted path holds :data:`UNDECLARED`.

    Attributes:
        values: ``(path, value)`` pairs in schema order, where ``path`` is a
            dotted ``section.field`` name from :data:`DECLARATION_SCHEMA`.
    """

    values: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        values = tuple(self.values)
        if tuple(path for path, _ in values) != _PATHS:
            raise _error("declaration: invalid_paths") from None
        for path, value in values:
            section, name = path.split(".", 1)
            if type(value) is not str or (
                value != UNDECLARED and value not in DECLARATION_SCHEMA[section][name]
            ):
                raise _error(f"{path}: invalid_value") from None
        object.__setattr__(self, "values", values)
        mapping = dict(values)
        mode = mapping["topology.training_mode"]
        scope = mapping["topology.federation_scope"]
        if (mode == "centralized" and scope not in (UNDECLARED, "not_federated")) or (
            mode in ("federated", "hybrid") and scope == "not_federated"
        ):
            raise _error("topology.federation_scope: inconsistent_value") from None

    @classmethod
    def from_dict(cls, declaration: Mapping[str, Any]) -> PrivateLearningDeployment:
        """Validate a nested ``{section: {field: value}}`` declaration.

        Args:
            declaration: Mapping of schema sections to mappings of enumerated
                string values. An optional ``schema_version`` key must equal
                :data:`DECLARATION_SCHEMA_VERSION`. Omitted sections and
                fields are recorded as :data:`UNDECLARED`.

        Returns:
            The validated, immutable deployment declaration.

        Raises:
            PrivateLearningThreatError: If the declaration is not a mapping,
                contains an unknown key, has an unsupported schema version, or
                holds a value outside the enumerated vocabulary.
        """
        if not isinstance(declaration, Mapping):
            raise _error("declaration: invalid_type") from None
        for key in declaration:
            if type(key) is not str or (
                key != "schema_version" and key not in DECLARATION_SCHEMA
            ):
                raise _error("declaration: unknown_field") from None
        if "schema_version" in declaration:
            version = declaration["schema_version"]
            if type(version) is not str or version != DECLARATION_SCHEMA_VERSION:
                raise _error("schema_version: unsupported_version") from None
        values: list[tuple[str, str]] = []
        for section, fields in DECLARATION_SCHEMA.items():
            raw = declaration.get(section, {})
            if not isinstance(raw, Mapping):
                raise _error(f"{section}: invalid_type") from None
            for key in raw:
                if type(key) is not str or key not in fields:
                    raise _error(f"{section}: unknown_field") from None
            for name, allowed in fields.items():
                value = raw.get(name, UNDECLARED)
                if type(value) is not str or (
                    value != UNDECLARED and value not in allowed
                ):
                    raise _error(f"{section}.{name}: invalid_value") from None
                values.append((f"{section}.{name}", value))
        return cls(values=tuple(values))

    def value(self, path: str) -> str:
        """Return the declared value for a dotted schema path.

        Args:
            path: Dotted ``section.field`` path from :data:`DECLARATION_SCHEMA`.

        Returns:
            The enumerated value, or :data:`UNDECLARED`.

        Raises:
            PrivateLearningThreatError: If ``path`` is not a schema path.
        """
        for candidate, value in self.values:
            if candidate == path:
                return value
        raise _error("path: unknown_field") from None

    def to_dict(self) -> dict[str, Any]:
        """Return the nested declaration, including ``undeclared`` fields."""
        nested: dict[str, Any] = {"schema_version": DECLARATION_SCHEMA_VERSION}
        for path, value in self.values:
            section, name = path.split(".", 1)
            nested.setdefault(section, {})[name] = value
        return nested

    @property
    def declaration_digest(self) -> str:
        """SHA-256 digest of the canonical nested declaration."""
        return _digest(self.to_dict())

    def __repr__(self) -> str:
        return (
            f"PrivateLearningDeployment(declaration_digest={self.declaration_digest!r})"
        )


@dataclass(frozen=True, slots=True)
class ThreatFinding:
    """One rule outcome for a single threat.

    Attributes:
        threat_id: Stable identifier such as ``PLT-001``.
        threat: Stable threat name from :data:`THREAT_CATALOG`.
        status: One of the four controlled status values.
        severity: One of :data:`SEVERITIES`; ``none`` when not applicable.
        reason_codes: Sorted, unique codes from :data:`REASON_CODES`.
        undeclared_inputs: Sorted schema paths that left the finding
            unresolved because they were not declared.
        controls: ``(control, declared)`` pairs for the controls that address
            this threat; empty when the threat is not applicable.
    """

    threat_id: str
    threat: str
    status: str
    severity: str
    reason_codes: tuple[str, ...]
    undeclared_inputs: tuple[str, ...]
    controls: tuple[tuple[str, bool], ...]

    def __post_init__(self) -> None:
        if (self.threat_id, self.threat) not in THREAT_CATALOG:
            raise _error("finding: unknown_threat") from None
        if self.status not in _STATUSES:
            raise _error("finding: invalid_status") from None
        if self.severity not in SEVERITIES:
            raise _error("finding: invalid_severity") from None
        reasons = tuple(sorted(set(self.reason_codes)))
        if not reasons or not set(reasons) <= REASON_CODES:
            raise _error("finding: invalid_reason_code") from None
        undeclared = tuple(sorted(set(self.undeclared_inputs)))
        if not set(undeclared) <= set(_PATHS):
            raise _error("finding: invalid_input_path") from None
        controls = tuple(self.controls)
        if any(
            name not in CONTROL_CATALOG or type(declared) is not bool
            for name, declared in controls
        ):
            raise _error("finding: invalid_control") from None
        object.__setattr__(self, "reason_codes", reasons)
        object.__setattr__(self, "undeclared_inputs", undeclared)
        object.__setattr__(self, "controls", controls)

    def to_dict(self) -> dict[str, Any]:
        """Return the stable finding schema with control references."""
        return {
            "controls": [
                {
                    "control": name,
                    "declared": declared,
                    "openmed_support": (
                        _MODULE_REFERENCE
                        if CONTROL_CATALOG[name]
                        else _DEPLOYMENT_PROVIDED
                    ),
                    "references": list(CONTROL_CATALOG[name]),
                }
                for name, declared in self.controls
            ],
            "reason_codes": list(self.reason_codes),
            "severity": self.severity,
            "status": self.status,
            "threat": self.threat,
            "threat_id": self.threat_id,
            "undeclared_inputs": list(self.undeclared_inputs),
        }


# -- rule table --------------------------------------------------------------

_Outcome = tuple[str, str, tuple[str, ...], tuple[str, ...]]
_UNDECLARED_SEVERITY: Final = "high"


class _Inputs:
    __slots__ = ("_values",)

    def __init__(self, deployment: PrivateLearningDeployment) -> None:
        self._values = dict(deployment.values)

    def value(self, path: str) -> str:
        return self._values[path]

    def missing(self, *paths: str) -> tuple[str, ...]:
        return tuple(path for path in paths if self._values[path] == UNDECLARED)

    def federated(self) -> bool:
        return self._values["topology.training_mode"] in ("federated", "hybrid")

    def dp_paths(self) -> tuple[str, ...]:
        noise = self._values["differential_privacy.noise_location"]
        if noise in (UNDECLARED, "none"):
            return ("differential_privacy.noise_location",)
        return (
            "differential_privacy.noise_location",
            "differential_privacy.privacy_unit",
            "differential_privacy.accountant",
        )

    def dp_state(self) -> str:
        if self._values["differential_privacy.noise_location"] == "none":
            return "none"
        if self._values["differential_privacy.accountant"] == "none":
            return "unaccounted"
        return "accounted"


def _unresolved_input(missing: tuple[str, ...]) -> _Outcome:
    return (UNRESOLVED, _UNDECLARED_SEVERITY, ("undeclared_input",), missing)


def _not_applicable(reason: str) -> _Outcome:
    return (NOT_APPLICABLE, "none", (reason,), ())


def _update_inversion(x: _Inputs) -> _Outcome:
    if missing := x.missing("topology.training_mode"):
        return _unresolved_input(missing)
    if not x.federated():
        return _not_applicable("no_client_updates_shared")
    if missing := x.missing(
        "trust.coordinator",
        "trust.client_custody",
        "secure_aggregation.protocol",
        "aggregation.minimum_cohort",
        "aggregation.update_clipping",
        *x.dp_paths(),
    ):
        return _unresolved_input(missing)
    coordinator = x.value("trust.coordinator")
    severity = {
        "malicious": "critical",
        "honest_but_curious": "high",
        "trusted": "medium",
    }[coordinator]
    secure = x.value("secure_aggregation.protocol") != "none"
    cohort = x.value("aggregation.minimum_cohort") == "enforced"
    clipped = x.value("aggregation.update_clipping") == "enforced"
    noise = x.value("differential_privacy.noise_location")
    dp = x.dp_state()
    local = noise == "client_local" and dp == "accounted"
    reasons: list[str] = []
    if dp == "unaccounted":
        reasons.append("noise_without_privacy_accounting")
    if local and not clipped:
        reasons.append("noise_without_enforced_clipping")
    if noise == "central" and coordinator != "trusted":
        reasons.append("central_noise_added_after_coordinator_access")
    if local and clipped:
        status = MITIGATED_BY_DECLARED_CONTROL
        reasons += ["client_side_noise_declared", "privacy_parameters_not_evaluated"]
    elif coordinator == "malicious":
        if secure:
            status = PARTIALLY_MITIGATED
            reasons.append("malicious_coordinator_can_isolate_updates")
        else:
            status = UNRESOLVED
            reasons.append("individual_updates_exposed_to_coordinator")
    elif coordinator == "honest_but_curious":
        if secure and cohort:
            status = MITIGATED_BY_DECLARED_CONTROL
            reasons += ["secure_aggregation_declared", "minimum_cohort_enforced"]
        elif secure:
            status = PARTIALLY_MITIGATED
            reasons.append("small_cohort_aggregate_exposure")
        else:
            status = UNRESOLVED
            reasons.append("individual_updates_exposed_to_coordinator")
    elif secure:
        status = MITIGATED_BY_DECLARED_CONTROL
        reasons.append("secure_aggregation_declared")
    else:
        status = PARTIALLY_MITIGATED
        reasons.append("coordinator_trust_assumed_without_control")
    if x.value("trust.client_custody") == "third_party_hosted":
        reasons.append("client_host_observes_unprotected_updates")
        if status == MITIGATED_BY_DECLARED_CONTROL:
            status = PARTIALLY_MITIGATED
    return (status, severity, tuple(reasons), ())


def _poisoning_backdoor(x: _Inputs) -> _Outcome:
    if missing := x.missing("topology.training_mode"):
        return _unresolved_input(missing)
    if not x.federated():
        return _not_applicable("no_external_update_contributors")
    if missing := x.missing(
        "topology.federation_scope",
        "trust.clients",
        "trust.client_custody",
        "aggregation.robust_aggregation",
        "aggregation.update_validation",
        "cryptography.update_signing",
    ):
        return _unresolved_input(missing)
    malicious = x.value("trust.clients") == "malicious"
    outside = x.value("trust.client_custody") == "third_party_hosted"
    robust = x.value("aggregation.robust_aggregation") != "none"
    validated = x.value("aggregation.update_validation") == "schema_allowlist"
    signed = x.value("cryptography.update_signing") == "per_client_signatures"
    reasons: list[str] = []
    if malicious:
        reasons.append("malicious_clients_assumed")
    if outside:
        reasons.append("client_custody_outside_site")
    if robust and x.value("secure_aggregation.protocol") not in (UNDECLARED, "none"):
        reasons.append("secure_aggregation_limits_update_inspection")
    if malicious or outside:
        severity = "high"
        if robust and validated:
            status = PARTIALLY_MITIGATED
            reasons.append("robust_aggregation_does_not_exclude_targeted_backdoors")
        else:
            status = UNRESOLVED
            reasons.append("no_poisoning_defense_declared")
    else:
        severity = "medium"
        if signed:
            status = MITIGATED_BY_DECLARED_CONTROL
            reasons.append("authenticated_contributors_declared")
        elif x.value("topology.federation_scope") == "cross_device":
            status = UNRESOLVED
            reasons.append("sybil_client_enrollment")
        else:
            status = PARTIALLY_MITIGATED
            reasons.append("unauthenticated_update_injection")
    return (status, severity, tuple(reasons), ())


def _collusion(x: _Inputs) -> _Outcome:
    if missing := x.missing("topology.training_mode"):
        return _unresolved_input(missing)
    if not x.federated():
        return _not_applicable("no_federated_participants")
    paths = [
        "topology.federation_scope",
        "trust.coordinator",
        "secure_aggregation.protocol",
        "aggregation.minimum_cohort",
        *x.dp_paths(),
    ]
    if x.value("secure_aggregation.protocol") not in (UNDECLARED, "none"):
        paths.append("secure_aggregation.collusion_assumption")
    if missing := x.missing(*paths):
        return _unresolved_input(missing)
    malicious = x.value("trust.coordinator") == "malicious"
    noise = x.value("differential_privacy.noise_location")
    dp = x.dp_state()
    client_level = (
        dp == "accounted"
        and x.value("differential_privacy.privacy_unit") == "client_level"
        and noise in ("distributed", "central")
    )
    reasons: list[str] = []
    if noise == "client_local" and dp == "accounted":
        status = MITIGATED_BY_DECLARED_CONTROL
        severity = "high"
        reasons += ["client_side_noise_declared", "privacy_parameters_not_evaluated"]
    elif x.value("secure_aggregation.protocol") != "none":
        assumption = x.value("secure_aggregation.collusion_assumption")
        if malicious:
            reasons.append("coordinator_counts_toward_collusion_threshold")
        if noise == "distributed":
            reasons.append("distributed_noise_weakens_under_collusion")
        if assumption == "may_reach_threshold":
            status, severity = UNRESOLVED, "critical"
            reasons.append("collusion_can_reach_secure_aggregation_threshold")
        elif malicious and x.value("topology.federation_scope") == "cross_device":
            status, severity = UNRESOLVED, "critical"
            reasons.append("coordinator_can_enroll_sybil_clients")
        elif assumption == "below_threshold":
            status, severity = PARTIALLY_MITIGATED, "high"
            reasons.append("secure_aggregation_threshold_assumption_unverified")
        else:
            status, severity = PARTIALLY_MITIGATED, "high"
            reasons.append("collusion_excluded_by_assumption_only")
    else:
        severity = "high"
        reasons.append("aggregate_differencing_by_colluding_clients")
        if x.value("aggregation.minimum_cohort") == "enforced":
            status = PARTIALLY_MITIGATED
        else:
            status = UNRESOLVED
            reasons.append("minimum_cohort_not_enforced")
    if status == UNRESOLVED and client_level:
        status = PARTIALLY_MITIGATED
        reasons.append("client_level_noise_declared")
    return (status, severity, tuple(reasons), ())


def _membership_inference(x: _Inputs) -> _Outcome:
    if missing := x.missing(
        "data_flows.model_release", "audits.membership_inference", *x.dp_paths()
    ):
        return _unresolved_input(missing)
    severity = {
        "internal_only": "medium",
        "external_partners": "high",
        "public": "critical",
    }[x.value("data_flows.model_release")]
    dp = x.dp_state()
    if dp == "accounted":
        status = MITIGATED_BY_DECLARED_CONTROL
        reasons = (
            "formal_privacy_accounting_declared",
            "privacy_parameters_not_evaluated",
        )
    elif dp == "unaccounted":
        status, reasons = PARTIALLY_MITIGATED, ("noise_without_privacy_accounting",)
    elif x.value("audits.membership_inference") == "self_test":
        status, reasons = PARTIALLY_MITIGATED, ("empirical_audit_only",)
    else:
        status, reasons = UNRESOLVED, ("no_membership_control_declared",)
    return (status, severity, reasons, ())


def _synthetic_memorization(x: _Inputs) -> _Outcome:
    if missing := x.missing("synthetic_data.release"):
        return _unresolved_input(missing)
    release = x.value("synthetic_data.release")
    if release == "none":
        return _not_applicable("no_synthetic_release")
    if missing := x.missing(
        "synthetic_data.generator_privacy", "audits.synthetic_memorization"
    ):
        return _unresolved_input(missing)
    severity = "high" if release == "external" else "medium"
    private = x.value("synthetic_data.generator_privacy") == "differentially_private"
    gated = x.value("audits.synthetic_memorization") == "release_gate"
    if private and gated:
        status = MITIGATED_BY_DECLARED_CONTROL
        reasons: tuple[str, ...] = (
            "generator_privacy_and_memorization_gate_declared",
            "privacy_parameters_not_evaluated",
        )
    elif private:
        status, reasons = PARTIALLY_MITIGATED, ("no_memorization_release_gate",)
    elif gated:
        status = PARTIALLY_MITIGATED
        reasons = ("memorization_gate_without_formal_privacy",)
    else:
        status = UNRESOLVED
        reasons = ("synthetic_release_without_privacy_or_audit",)
    return (status, severity, reasons, ())


def _retention_exposure(x: _Inputs) -> _Outcome:
    if missing := x.missing("topology.training_mode"):
        return _unresolved_input(missing)
    artifacts = ["retention.checkpoints", "retention.synthetic_data", "retention.logs"]
    if x.federated():
        artifacts.insert(0, "retention.client_updates")
    if missing := x.missing(*artifacts):
        return _unresolved_input(missing)
    updates = x.federated() and x.value("retention.client_updates") == "retained"
    checkpoints = x.value("retention.checkpoints") == "retained"
    synthetic = x.value("retention.synthetic_data") == "retained"
    logs = x.value("retention.logs")
    long_lived = updates or checkpoints or synthetic
    if not long_lived and logs == "none":
        return _not_applicable("no_retained_artifacts")
    if missing := x.missing("retention.schedule", "cryptography.at_rest_encryption"):
        return _unresolved_input(missing)
    scheduled = x.value("retention.schedule") == "enforced"
    encrypted = x.value("cryptography.at_rest_encryption") == "encrypted"
    reasons: list[str] = []
    if updates:
        reasons.append("client_updates_retained_beyond_round")
    if checkpoints:
        reasons.append("checkpoints_retained")
    if synthetic:
        reasons.append("synthetic_data_retained")
    if not scheduled:
        reasons.append("retention_schedule_not_enforced")
    if long_lived and not encrypted:
        reasons.append("retained_artifacts_unencrypted")
    if logs == "content_bearing":
        severity = "critical"
    elif updates:
        severity = "high"
    else:
        severity = "medium"
    if logs == "content_bearing":
        status = UNRESOLVED
        reasons.append("content_bearing_logs_retained")
    elif not long_lived:
        if scheduled:
            status = MITIGATED_BY_DECLARED_CONTROL
            reasons.append("aggregate_only_logs_with_retention_schedule")
        else:
            status = PARTIALLY_MITIGATED
    elif updates:
        status = PARTIALLY_MITIGATED if scheduled and encrypted else UNRESOLVED
    elif scheduled and encrypted:
        status = MITIGATED_BY_DECLARED_CONTROL
        reasons.append("retention_controls_declared")
    elif scheduled or encrypted:
        status = PARTIALLY_MITIGATED
    else:
        status = UNRESOLVED
    return (status, severity, tuple(reasons), ())


def _transport_interception(x: _Inputs) -> _Outcome:
    if missing := x.missing(
        "topology.training_mode", "data_flows.raw_records_leave_site"
    ):
        return _unresolved_input(missing)
    egress = x.value("data_flows.raw_records_leave_site")
    if not x.federated() and egress == "never":
        return _not_applicable("no_cross_site_transfer")
    if missing := x.missing("cryptography.transport", "network.topology"):
        return _unresolved_input(missing)
    reasons: list[str] = []
    if egress == "raw":
        reasons.append("raw_records_cross_site_boundary")
    elif egress == "pseudonymized_only":
        reasons.append("pseudonymized_records_cross_site_boundary")
    severity = "critical" if egress == "raw" else "high"
    transport = x.value("cryptography.transport")
    network = x.value("network.topology")
    if transport == "mutual_tls":
        status = MITIGATED_BY_DECLARED_CONTROL
        reasons.append("mutually_authenticated_transport_declared")
    elif transport == "tls":
        status = MITIGATED_BY_DECLARED_CONTROL
        reasons.append("encrypted_transport_declared")
    elif network == "air_gapped":
        status = PARTIALLY_MITIGATED
        reasons.append("physical_isolation_without_transport_encryption")
    elif network == "vpn":
        status = PARTIALLY_MITIGATED
        reasons.append("network_layer_encryption_only")
    else:
        status = UNRESOLVED
        reasons.append("unencrypted_transport")
    return (status, severity, tuple(reasons), ())


def _update_replay(x: _Inputs) -> _Outcome:
    if missing := x.missing("topology.training_mode"):
        return _unresolved_input(missing)
    if not x.federated():
        return _not_applicable("no_client_updates_shared")
    if missing := x.missing(
        "cryptography.replay_protection", "cryptography.update_signing"
    ):
        return _unresolved_input(missing)
    bound = x.value("cryptography.replay_protection") == "round_bound_nonces"
    signed = x.value("cryptography.update_signing") == "per_client_signatures"
    if bound and signed:
        status, reason = MITIGATED_BY_DECLARED_CONTROL, "round_bound_signed_updates"
    elif bound:
        status, reason = PARTIALLY_MITIGATED, "replay_binding_without_signatures"
    elif signed:
        status, reason = PARTIALLY_MITIGATED, "signatures_without_round_binding"
    else:
        status, reason = UNRESOLVED, "updates_not_bound_to_round"
    return (status, "medium", (reason,), ())


def _metric_disclosure(x: _Inputs) -> _Outcome:
    if missing := x.missing("topology.training_mode"):
        return _unresolved_input(missing)
    if not x.federated():
        return _not_applicable("no_federated_metrics")
    if missing := x.missing("data_flows.metric_release"):
        return _unresolved_input(missing)
    release = x.value("data_flows.metric_release")
    if release == "none":
        return _not_applicable("no_metric_release")
    if release == "per_client":
        return (UNRESOLVED, "high", ("per_client_metrics_released",), ())
    if missing := x.missing("aggregation.minimum_cohort"):
        return _unresolved_input(missing)
    if x.value("aggregation.minimum_cohort") == "enforced":
        return (
            MITIGATED_BY_DECLARED_CONTROL,
            "medium",
            ("aggregate_envelopes_with_minimum_group",),
            (),
        )
    return (PARTIALLY_MITIGATED, "medium", ("minimum_cohort_not_enforced",), ())


_RULES: Final[
    tuple[tuple[str, str, Callable[[_Inputs], _Outcome], tuple[str, ...]], ...]
] = (
    (
        "PLT-001",
        "update_inversion",
        _update_inversion,
        (
            "secure_aggregation",
            "differential_privacy",
            "privacy_accounting",
            "update_clipping",
            "minimum_cohort",
        ),
    ),
    (
        "PLT-002",
        "poisoning_backdoor",
        _poisoning_backdoor,
        ("update_schema_validation", "robust_aggregation", "update_signing"),
    ),
    (
        "PLT-003",
        "collusion",
        _collusion,
        (
            "secure_aggregation",
            "minimum_cohort",
            "differential_privacy",
            "privacy_accounting",
        ),
    ),
    (
        "PLT-004",
        "membership_inference",
        _membership_inference,
        ("differential_privacy", "privacy_accounting", "membership_audit"),
    ),
    (
        "PLT-005",
        "synthetic_memorization",
        _synthetic_memorization,
        ("generator_privacy", "memorization_audit"),
    ),
    (
        "PLT-006",
        "retention_exposure",
        _retention_exposure,
        ("retention_schedule", "at_rest_encryption", "aggregate_only_logging"),
    ),
    (
        "PLT-007",
        "transport_interception",
        _transport_interception,
        ("transport_security", "network_isolation"),
    ),
    (
        "PLT-008",
        "update_replay",
        _update_replay,
        ("replay_protection", "update_signing"),
    ),
    (
        "PLT-009",
        "metric_disclosure",
        _metric_disclosure,
        ("aggregate_metric_envelopes", "minimum_cohort"),
    ),
)

THREAT_CATALOG: Final = tuple((threat_id, threat) for threat_id, threat, _, _ in _RULES)


def _control_declared(deployment: PrivateLearningDeployment, control: str) -> bool:
    path, values = _CONTROL_DECLARED_BY[control]
    return deployment.value(path) in values


def _evaluate(deployment: PrivateLearningDeployment) -> tuple[ThreatFinding, ...]:
    inputs = _Inputs(deployment)
    findings: list[ThreatFinding] = []
    for threat_id, threat, rule, controls in _RULES:
        status, severity, reasons, undeclared = rule(inputs)
        mapped = (
            ()
            if status == NOT_APPLICABLE
            else tuple((name, _control_declared(deployment, name)) for name in controls)
        )
        findings.append(
            ThreatFinding(
                threat_id=threat_id,
                threat=threat,
                status=status,
                severity=severity,
                reason_codes=reasons,
                undeclared_inputs=undeclared,
                controls=mapped,
            )
        )
    return tuple(findings)


_REPORT_BUILDER_TOKEN: Final = object()


@dataclass(frozen=True, slots=True, repr=False)
class PrivateLearningThreatReport:
    """A content-addressed private-learning threat report.

    Build reports with :func:`build_private_learning_threat_report`; direct
    construction is rejected so findings always come from the rule table.

    Attributes:
        deployment: The validated declaration the report evaluates.
        findings: One finding per threat in :data:`THREAT_CATALOG` order.
        report_digest: SHA-256 digest of every report field except itself.
    """

    deployment: PrivateLearningDeployment
    findings: tuple[ThreatFinding, ...]
    report_digest: str = ""
    _builder_token: InitVar[object | None] = None

    def __post_init__(self, _builder_token: object | None) -> None:
        if _builder_token is not _REPORT_BUILDER_TOKEN:
            raise _error("report: builder_required") from None
        object.__setattr__(self, "findings", tuple(self.findings))
        calculated = _digest(self._payload(include_digest=False))
        if self.report_digest and self.report_digest != calculated:
            raise _error("report_digest: payload_mismatch") from None
        object.__setattr__(self, "report_digest", calculated)

    def _payload(self, *, include_digest: bool) -> dict[str, Any]:
        findings = [item.to_dict() for item in self.findings]
        payload: dict[str, Any] = {
            "declaration": self.deployment.to_dict(),
            "declaration_digest": self.deployment.declaration_digest,
            "declaration_schema_version": DECLARATION_SCHEMA_VERSION,
            "disclaimer": PRIVATE_LEARNING_THREAT_DISCLAIMER,
            "findings": findings,
            "rules_version": PRIVATE_LEARNING_THREAT_RULES_VERSION,
            "schema_version": PRIVATE_LEARNING_THREAT_REPORT_SCHEMA_VERSION,
            "status_counts": {
                status: sum(1 for item in self.findings if item.status == status)
                for status in _STATUSES
            },
            "unresolved_threat_ids": [
                item.threat_id for item in self.findings if item.status == UNRESOLVED
            ],
        }
        if include_digest:
            payload["report_digest"] = self.report_digest
        return payload

    def to_dict(self) -> dict[str, Any]:
        """Return the closed report schema as fresh built-in containers."""
        return self._payload(include_digest=True)

    def to_json(self) -> str:
        """Return canonical, sorted-key, ASCII-only JSON for the report."""
        return _canonical_json(self.to_dict())

    def to_markdown(self) -> str:
        """Render a deterministic operator review document."""
        lines = [
            "# Private-learning threat report",
            "",
            f"> {PRIVATE_LEARNING_THREAT_DISCLAIMER}",
            "",
            f"- Schema version: `{PRIVATE_LEARNING_THREAT_REPORT_SCHEMA_VERSION}`",
            f"- Rules version: `{PRIVATE_LEARNING_THREAT_RULES_VERSION}`",
            f"- Declaration digest: `{self.deployment.declaration_digest}`",
            f"- Report digest: `{self.report_digest}`",
            "",
            "## Findings",
            "",
            "| Threat ID | Threat | Status | Severity | Reason codes | "
            "Undeclared inputs |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for item in self.findings:
            reasons = ", ".join(f"`{code}`" for code in item.reason_codes)
            undeclared = (
                ", ".join(f"`{path}`" for path in item.undeclared_inputs) or "none"
            )
            lines.append(
                f"| `{item.threat_id}` | `{item.threat}` | `{item.status}` | "
                f"`{item.severity}` | {reasons} | {undeclared} |"
            )
        lines.extend(
            [
                "",
                "## Control mapping",
                "",
                "| Threat ID | Control | Declared | OpenMed support | References |",
                "| --- | --- | --- | --- | --- |",
            ]
        )
        for item in self.findings:
            for name, declared in item.controls:
                references = (
                    ", ".join(f"`{ref}`" for ref in CONTROL_CATALOG[name]) or "none"
                )
                support = (
                    _MODULE_REFERENCE if CONTROL_CATALOG[name] else _DEPLOYMENT_PROVIDED
                )
                lines.append(
                    f"| `{item.threat_id}` | `{name}` | "
                    f"`{'yes' if declared else 'no'}` | `{support}` | {references} |"
                )
        lines.append("")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"PrivateLearningThreatReport(report_digest={self.report_digest!r})"


def build_private_learning_threat_report(
    declaration: Mapping[str, Any] | PrivateLearningDeployment,
) -> PrivateLearningThreatReport:
    """Evaluate a deployment declaration against the versioned threat rules.

    Args:
        declaration: A validated :class:`PrivateLearningDeployment`, or a
            nested mapping accepted by :meth:`PrivateLearningDeployment.from_dict`.

    Returns:
        A deterministic report with one finding per catalog threat. Undeclared
        inputs yield ``unresolved`` findings, never mitigations.

    Raises:
        PrivateLearningThreatError: If the declaration violates the closed
            schema. The message is a fixed ``field: code`` pair.

    Example:
        >>> report = build_private_learning_threat_report(
        ...     {"topology": {"training_mode": "centralized"}}
        ... )
        >>> report.to_dict()["findings"][0]["status"]
        'not_applicable'
    """
    deployment = (
        declaration
        if isinstance(declaration, PrivateLearningDeployment)
        else PrivateLearningDeployment.from_dict(declaration)
    )
    return PrivateLearningThreatReport(
        deployment=deployment,
        findings=_evaluate(deployment),
        _builder_token=_REPORT_BUILDER_TOKEN,
    )


__all__ = [
    "CONTROL_CATALOG",
    "DECLARATION_SCHEMA",
    "DECLARATION_SCHEMA_VERSION",
    "MITIGATED_BY_DECLARED_CONTROL",
    "NOT_APPLICABLE",
    "PARTIALLY_MITIGATED",
    "PRIVATE_LEARNING_THREAT_DISCLAIMER",
    "PRIVATE_LEARNING_THREAT_REPORT_SCHEMA_VERSION",
    "PRIVATE_LEARNING_THREAT_RULES_VERSION",
    "REASON_CODES",
    "SEVERITIES",
    "THREAT_CATALOG",
    "UNDECLARED",
    "UNRESOLVED",
    "PrivateLearningDeployment",
    "PrivateLearningThreatError",
    "PrivateLearningThreatReport",
    "ThreatFinding",
    "build_private_learning_threat_report",
]
