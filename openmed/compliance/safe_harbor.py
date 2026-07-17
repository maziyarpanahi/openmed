"""PHI-safe HIPAA Safe Harbor attestation reports.

The generator in this module summarizes an existing de-identification audit
report. It deliberately emits only canonical category metadata, aggregate
counts, policy actions, and hashes. Source text, span offsets, context,
surrogates, and detector evidence are never copied into the attestation.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Final

from openmed.core.audit import stable_hash
from openmed.core.labels import (
    HIPAA_SAFE_HARBOR_CLASSES,
    LABEL_TO_HIPAA,
    normalize_label,
)
from openmed.core.policy import PolicyProfile, load_policy
from openmed.core.schemas.span import ACTION_VALUES

SAFE_HARBOR_POLICY: Final = "hipaa_safe_harbor"
SAFE_HARBOR_ATTESTATION_SCHEMA_VERSION: Final = 1
SAFE_HARBOR_ATTESTATION_NOTICE: Final = (
    "This report is evidence from one OpenMed run, not a certification or legal "
    "sign-off. Residual-risk items require qualified expert review."
)

# The regulatory order is meaningful and must not depend on frozenset ordering.
SAFE_HARBOR_CATEGORY_ORDER: Final[tuple[str, ...]] = (
    "NAME",
    "GEOGRAPHIC_SUBDIVISION",
    "DATE_ELEMENT",
    "TELEPHONE_NUMBER",
    "FAX_NUMBER",
    "EMAIL_ADDRESS",
    "SOCIAL_SECURITY_NUMBER",
    "MEDICAL_RECORD_NUMBER",
    "HEALTH_PLAN_BENEFICIARY_NUMBER",
    "ACCOUNT_NUMBER",
    "CERTIFICATE_LICENSE_NUMBER",
    "VEHICLE_IDENTIFIER",
    "DEVICE_IDENTIFIER",
    "URL",
    "IP_ADDRESS",
    "BIOMETRIC_IDENTIFIER",
    "FULL_FACE_PHOTO",
    "UNIQUE_IDENTIFIER",
)

_CATEGORY_NAMES: Final[Mapping[str, str]] = MappingProxyType(
    {
        "NAME": "Names",
        "GEOGRAPHIC_SUBDIVISION": "Geographic subdivisions",
        "DATE_ELEMENT": "Date elements and ages over 89",
        "TELEPHONE_NUMBER": "Telephone numbers",
        "FAX_NUMBER": "Fax numbers",
        "EMAIL_ADDRESS": "Email addresses",
        "SOCIAL_SECURITY_NUMBER": "Social Security numbers",
        "MEDICAL_RECORD_NUMBER": "Medical record numbers",
        "HEALTH_PLAN_BENEFICIARY_NUMBER": "Health plan beneficiary numbers",
        "ACCOUNT_NUMBER": "Account numbers",
        "CERTIFICATE_LICENSE_NUMBER": "Certificate and license numbers",
        "VEHICLE_IDENTIFIER": "Vehicle identifiers and serial numbers",
        "DEVICE_IDENTIFIER": "Device identifiers and serial numbers",
        "URL": "Web URLs",
        "IP_ADDRESS": "IP addresses",
        "BIOMETRIC_IDENTIFIER": "Biometric identifiers",
        "FULL_FACE_PHOTO": "Full-face photographs and comparable images",
        "UNIQUE_IDENTIFIER": "Other unique identifying numbers or codes",
    }
)

_labels_by_category: dict[str, tuple[str, ...]] = {
    category: tuple(
        sorted(
            label
            for label, mapped_category in LABEL_TO_HIPAA.items()
            if mapped_category == category
        )
    )
    for category in SAFE_HARBOR_CATEGORY_ORDER
}
SAFE_HARBOR_CATEGORY_LABELS: Final[Mapping[str, tuple[str, ...]]] = MappingProxyType(
    _labels_by_category
)

_SHA256_RE: Final = re.compile(r"^sha256:[0-9a-f]{64}$")
_EXPLICIT_CATEGORY_KEYS: Final[tuple[str, ...]] = (
    "hipaa_safe_harbor_class",
    "safe_harbor_class",
)


def _validate_category_table() -> None:
    categories = set(SAFE_HARBOR_CATEGORY_ORDER)
    if len(SAFE_HARBOR_CATEGORY_ORDER) != 18:
        raise RuntimeError("Safe Harbor attestation must enumerate 18 categories")
    if categories != set(HIPAA_SAFE_HARBOR_CLASSES):
        raise RuntimeError("Safe Harbor attestation categories differ from core labels")
    if categories != set(_CATEGORY_NAMES):
        raise RuntimeError("Safe Harbor category names are incomplete")


_validate_category_table()


@dataclass(frozen=True)
class SafeHarborCategoryAttestation:
    """Aggregate evidence for one of the 18 Safe Harbor categories."""

    ordinal: int
    category: str
    name: str
    mapped_labels: tuple[str, ...]
    policy_actions: Mapping[str, str]
    detection_count: int
    applied_action_counts: Mapping[str, int]
    residual_risk: bool
    residual_risk_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-compatible category record."""

        return {
            "ordinal": self.ordinal,
            "category": self.category,
            "name": self.name,
            "mapped_labels": list(self.mapped_labels),
            "policy_actions": dict(sorted(self.policy_actions.items())),
            "detection_count": self.detection_count,
            "applied_action_counts": dict(sorted(self.applied_action_counts.items())),
            "residual_risk": self.residual_risk,
            "residual_risk_reason": self.residual_risk_reason,
        }


@dataclass(frozen=True)
class SafeHarborAttestation:
    """A PHI-safe aggregate attestation for one de-identification run."""

    source_report_hash: str
    policy: str
    categories: tuple[SafeHarborCategoryAttestation, ...]
    schema_version: int = SAFE_HARBOR_ATTESTATION_SCHEMA_VERSION
    report_type: str = "hipaa_safe_harbor_attestation"
    notice: str = SAFE_HARBOR_ATTESTATION_NOTICE

    @property
    def total_detection_count(self) -> int:
        """Return the number of detections summarized across all categories."""

        return sum(category.detection_count for category in self.categories)

    @property
    def residual_risk_categories(self) -> tuple[str, ...]:
        """Return categories that require expert review."""

        return tuple(
            category.category for category in self.categories if category.residual_risk
        )

    @property
    def requires_expert_determination(self) -> bool:
        """Whether one or more category records carry residual risk."""

        return bool(self.residual_risk_categories)

    def _payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "report_type": self.report_type,
            "source_report_hash": self.source_report_hash,
            "policy": self.policy,
            "summary": {
                "category_count": len(self.categories),
                "detection_count": self.total_detection_count,
                "residual_risk_category_count": len(self.residual_risk_categories),
                "requires_expert_determination": (self.requires_expert_determination),
            },
            "categories": [category.to_dict() for category in self.categories],
            "notice": self.notice,
        }

    @property
    def attestation_hash(self) -> str:
        """Return the stable hash of the attestation payload."""

        return stable_hash(self._payload())

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-compatible attestation."""

        return {**self._payload(), "attestation_hash": self.attestation_hash}

    def to_json(self, *, indent: int | None = 2) -> str:
        """Serialize the attestation without exposing source span content."""

        return json.dumps(
            self.to_dict(),
            allow_nan=False,
            ensure_ascii=True,
            indent=indent,
            sort_keys=True,
        )


def generate_safe_harbor_attestation(
    audit_report: Any,
) -> SafeHarborAttestation:
    """Generate an attestation from an audit report or compatible mapping.

    The input must expose ``spans`` and may expose ``policy`` and
    ``repro_hash`` either as mapping keys or object attributes. Every span must
    contain a canonical/source label. Observed ``action`` values are counted;
    a missing action falls back to the selected policy's configured action.

    A span can override the canonical label-to-category fallback with a valid
    ``hipaa_safe_harbor_class`` or ``safe_harbor_class`` in its top-level
    fields, ``evidence``, or ``metadata``. ``regulatory_tags`` are also
    supported. This lets specialized detectors distinguish categories such as
    fax numbers while preserving the canonical ``PHONE`` label.
    """

    payload = _report_payload(audit_report)
    integrity_check = getattr(audit_report, "repro_hash_matches", None)
    if callable(integrity_check) and not integrity_check():
        raise ValueError("audit report reproducibility hash does not match")
    spans = _report_value(audit_report, "spans")
    if isinstance(spans, (str, bytes)) or not isinstance(spans, Iterable):
        raise TypeError("audit report spans must be an iterable of span records")

    policy_name = str(_report_value(audit_report, "policy", "") or "")
    if not policy_name:
        raise ValueError("audit report must identify its de-identification policy")
    profile = load_policy(policy_name)
    if profile.name != SAFE_HARBOR_POLICY:
        raise ValueError(
            "Safe Harbor attestations require the hipaa_safe_harbor policy"
        )

    source_report_hash = _source_report_hash(audit_report, payload)
    detection_counts: Counter[str] = Counter()
    action_counts: dict[str, Counter[str]] = {
        category: Counter() for category in SAFE_HARBOR_CATEGORY_ORDER
    }

    for span in spans:
        canonical_label = _canonical_span_label(span)
        category = _explicit_span_category(span) or LABEL_TO_HIPAA[canonical_label]
        action = str(_span_value(span, "action", "") or "")
        if not action:
            action = profile.action_for(canonical_label)
        if action not in ACTION_VALUES:
            raise ValueError(f"unsupported audit span action: {action!r}")
        detection_counts[category] += 1
        action_counts[category][action] += 1

    categories = tuple(
        _category_attestation(
            ordinal=ordinal,
            category=category,
            profile=profile,
            detection_count=detection_counts[category],
            action_counts=action_counts[category],
        )
        for ordinal, category in enumerate(SAFE_HARBOR_CATEGORY_ORDER, start=1)
    )
    return SafeHarborAttestation(
        source_report_hash=source_report_hash,
        policy=profile.name,
        categories=categories,
    )


def _category_attestation(
    *,
    ordinal: int,
    category: str,
    profile: PolicyProfile,
    detection_count: int,
    action_counts: Mapping[str, int],
) -> SafeHarborCategoryAttestation:
    labels = SAFE_HARBOR_CATEGORY_LABELS[category]
    policy_actions = {label: profile.action_for(label) for label in labels}
    reasons: list[str] = []
    if not labels:
        reasons.append(
            "No canonical OpenMed labels map to this category; qualified "
            "expert review is required."
        )
    if action_counts.get("keep", 0):
        reasons.append(
            "One or more detections used the keep action; qualified expert "
            "review is required."
        )
    return SafeHarborCategoryAttestation(
        ordinal=ordinal,
        category=category,
        name=_CATEGORY_NAMES[category],
        mapped_labels=labels,
        policy_actions=policy_actions,
        detection_count=detection_count,
        applied_action_counts=dict(action_counts),
        residual_risk=bool(reasons),
        residual_risk_reason=" ".join(reasons) or None,
    )


def _report_payload(report: Any) -> Mapping[str, Any]:
    if isinstance(report, Mapping):
        return report
    to_dict = getattr(report, "to_dict", None)
    if callable(to_dict):
        payload = to_dict()
        if isinstance(payload, Mapping):
            return payload
    raise TypeError("audit report must be a mapping or expose to_dict()")


def _report_value(report: Any, name: str, default: Any = None) -> Any:
    if isinstance(report, Mapping):
        return report.get(name, default)
    return getattr(report, name, default)


def _span_value(span: Any, name: str, default: Any = None) -> Any:
    if isinstance(span, Mapping):
        return span.get(name, default)
    return getattr(span, name, default)


def _canonical_span_label(span: Any) -> str:
    value = _span_value(span, "canonical_label") or _span_value(span, "label")
    if not value:
        raise ValueError("audit span must include canonical_label or label")
    try:
        return normalize_label(str(value))
    except (KeyError, ValueError) as exc:
        raise ValueError(f"unsupported audit span label: {value!r}") from exc


def _explicit_span_category(span: Any) -> str | None:
    for container in (
        span,
        _span_value(span, "evidence", {}),
        _span_value(span, "metadata", {}),
    ):
        if not isinstance(container, Mapping):
            continue
        for key in _EXPLICIT_CATEGORY_KEYS:
            if key not in container:
                continue
            category = str(container[key])
            if category not in HIPAA_SAFE_HARBOR_CLASSES:
                raise ValueError(f"unknown HIPAA Safe Harbor category: {category!r}")
            return category

    tags = _span_value(span, "regulatory_tags", ())
    if isinstance(tags, Sequence) and not isinstance(tags, (str, bytes)):
        for tag in tags:
            category = str(tag)
            if category in HIPAA_SAFE_HARBOR_CLASSES:
                return category
    return None


def _source_report_hash(report: Any, payload: Mapping[str, Any]) -> str:
    candidate = str(_report_value(report, "repro_hash", "") or "")
    if _SHA256_RE.fullmatch(candidate):
        return candidate
    return stable_hash(payload)


__all__ = [
    "SAFE_HARBOR_ATTESTATION_NOTICE",
    "SAFE_HARBOR_ATTESTATION_SCHEMA_VERSION",
    "SAFE_HARBOR_CATEGORY_LABELS",
    "SAFE_HARBOR_CATEGORY_ORDER",
    "SAFE_HARBOR_POLICY",
    "SafeHarborAttestation",
    "SafeHarborCategoryAttestation",
    "generate_safe_harbor_attestation",
]
