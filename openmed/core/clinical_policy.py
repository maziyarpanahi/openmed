"""Versioned category and role controls for clinical-preserving redaction."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Sequence

from .clinical_identifiers import personal_name_role
from .clinical_label_map import clinical_label
from .labels import CANONICAL_LABELS, normalize_label
from .policy import PolicyProfile, load_policy

CLINICAL_POLICY_VERSION = "clinical-preserve-v1"
CATEGORY_LABELS = MappingProxyType(
    {
        "person_name": frozenset(
            {"PERSON", "FIRST_NAME", "LAST_NAME", "MIDDLE_NAME", "PREFIX", "USERNAME"}
        ),
        "identifier": frozenset(
            {
                "ID_NUM",
                "SSN",
                "ACCOUNT_NUMBER",
                "PASSWORD",
                "PIN",
                "API_KEY",
                "CREDIT_CARD",
                "CVV",
                "IBAN",
                "BIC",
                "BITCOIN_ADDRESS",
                "ETHEREUM_ADDRESS",
                "LITECOIN_ADDRESS",
                "MASKED_NUMBER",
                "IP_ADDRESS",
                "MAC_ADDRESS",
                "USER_AGENT",
                "VIN",
                "VEHICLE_REGISTRATION",
                "IMEI",
            }
        ),
        "phone": frozenset({"PHONE"}),
        "email": frozenset({"EMAIL"}),
        "address": frozenset(
            {
                "LOCATION",
                "STREET_ADDRESS",
                "BUILDING_NUMBER",
                "ZIPCODE",
                "GPS_COORDINATES",
            }
        ),
        "date_of_birth": frozenset({"DATE_OF_BIRTH"}),
        "dates": frozenset({"DATE", "TIME"}),
        "url": frozenset({"URL"}),
        "organization": frozenset({"ORGANIZATION"}),
    }
)
DEFAULT_CATEGORIES = (
    "person_name",
    "identifier",
    "phone",
    "email",
    "address",
    "date_of_birth",
    "url",
)
REDACT_ROLES = frozenset({"patient", "clinician"})


@dataclass(frozen=True)
class ClinicalAction:
    """One resolved action with safe review metadata."""

    action: str
    role: str | None = None
    needs_review: bool = False
    reason: str | None = None


@dataclass(frozen=True)
class ClinicalPolicy:
    """An explicit policy shared by detection, arbitration and the final sweep."""

    profile: PolicyProfile
    categories: tuple[str, ...]
    roles: frozenset[str]
    keep_labels: tuple[str, ...]
    narrowed: bool

    def decide(
        self, label: str, *, text: str, start: int, end: int, role: str | None = None
    ) -> ClinicalAction:
        """Resolve an entity after the final sweep, retaining uncertain names.

        A role restriction never guesses that an unanchored name can be kept.
        Such a name is masked and flagged for review. Role selection is not a
        language/model qualification claim.
        """
        canonical = normalize_label(label)
        action = self.profile.action_for(canonical)
        if action == "keep" or canonical not in CATEGORY_LABELS["person_name"]:
            return ClinicalAction(action)
        resolved_role = role or personal_name_role(text, start, end)
        if resolved_role in REDACT_ROLES:
            return ClinicalAction(
                "mask" if resolved_role in self.roles else "keep", resolved_role
            )
        if self.roles != REDACT_ROLES:
            return ClinicalAction(
                "mask", needs_review=True, reason="person_role_uncertain"
            )
        return ClinicalAction("mask")

    def metadata(self) -> dict:
        """Return the effective policy without source strings or identifiers."""
        return {
            "profile": self.profile.name,
            "version": CLINICAL_POLICY_VERSION,
            "redact_categories": list(self.categories),
            "redact_roles": sorted(self.roles),
            "keep_labels": list(self.keep_labels),
            "policy_narrowed": self.narrowed,
            "safety_sweep_mandatory": True,
        }


def resolve_clinical_policy(
    *,
    redact_categories: Sequence[str] | None = None,
    redact_roles: Sequence[str] = ("patient", "clinician"),
    keep_labels: Sequence[str] = (),
) -> ClinicalPolicy:
    """Validate a clinical policy and reject contradictory label instructions.

    Args:
        redact_categories: Requested category names. Omission uses the complete
            clinical-preserve direct-identifier profile. A smaller selection is
            reported as narrowed and requires review before privacy promotion.
        redact_roles: Patient and/or clinician. Unknown roles are rejected.
        keep_labels: Canonical or known source labels to keep. A label selected
            for redaction cannot simultaneously be kept.

    Returns:
        A versioned profile and per-span role decision function.
    """
    categories = _unique_strings(
        DEFAULT_CATEGORIES if redact_categories is None else redact_categories,
        "redact_categories",
    )
    roles = frozenset(_unique_strings(redact_roles, "redact_roles"))
    if not categories or set(categories) - CATEGORY_LABELS.keys():
        raise ValueError(
            "redact_categories contains an unsupported category or is empty"
        )
    if not roles or not roles <= REDACT_ROLES:
        raise ValueError("redact_roles must select patient and/or clinician")
    kept = []
    for label in _unique_strings(keep_labels, "keep_labels"):
        canonical = "OTHER" if label.upper() == "OTHER" else clinical_label(label)
        if canonical not in CANONICAL_LABELS:
            raise ValueError("keep_labels contains an unknown label")
        kept.append(canonical)
    required = frozenset().union(
        *(CATEGORY_LABELS[category] for category in categories)
    )
    conflicts = required.intersection(kept)
    if conflicts:
        raise ValueError(
            "keep_labels conflicts with redact_categories: "
            + ", ".join(sorted(conflicts))
        )
    base = load_policy("clinical_preserve")
    profile = base.derive(
        actions={
            label: "mask" if label in required else "keep" for label in CANONICAL_LABELS
        },
        metadata={"profile_version": CLINICAL_POLICY_VERSION},
    )
    return ClinicalPolicy(
        profile,
        categories,
        roles,
        tuple(sorted(set(kept))),
        not set(DEFAULT_CATEGORIES) <= set(categories) or roles != REDACT_ROLES,
    )


def _unique_strings(value: Sequence[str], name: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise ValueError(f"{name} must be an array of non-empty strings")
    return tuple(dict.fromkeys(value))


__all__ = [
    "CLINICAL_POLICY_VERSION",
    "CATEGORY_LABELS",
    "DEFAULT_CATEGORIES",
    "ClinicalPolicy",
    "ClinicalAction",
    "resolve_clinical_policy",
]
