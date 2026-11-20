"""Versioned intended-use declarations for capability registries and receipts.

Use IntendedUse.from_dict to validate a declaration and digest() to bind it to
evidence. List-valued codes are sets and serialize in canonical sorted order.
Validation failures expose a stable reason code without the rejected value.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from functools import lru_cache
from importlib.resources import files
from typing import Any

from openmed.core.schemas import schema_fingerprint


class RegulatoryTier(StrEnum):
    """Operational tier for a capability's declared intended use."""

    NON_CLINICAL = "non_clinical"
    INFORMATIONAL = "informational"
    ASSISTIVE_REVIEW = "assistive_review"
    RESTRICTED = "restricted"

    @property
    def obligations(self) -> frozenset[str]:
        """Return the required safeguards for this tier."""
        return _OBLIGATIONS[self]


_OBLIGATIONS = {
    RegulatoryTier.NON_CLINICAL: frozenset(),
    RegulatoryTier.INFORMATIONAL: frozenset({"notice_required"}),
    RegulatoryTier.ASSISTIVE_REVIEW: frozenset(
        {"notice_required", "acknowledgement_required"}
    ),
    RegulatoryTier.RESTRICTED: frozenset(
        {
            "notice_required",
            "acknowledgement_required",
            "clinician_mode_required",
            "crisis_routing_required",
        }
    ),
}


@lru_cache(maxsize=1)
def _schema() -> dict[str, Any]:
    path = files("openmed.core.schemas.json").joinpath("intended_use.schema.json")
    return json.loads(path.read_text(encoding="utf-8"))


class IntendedUseError(ValueError):
    """A declaration failed validation for the given value-free reason code."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True, slots=True, kw_only=True)
class IntendedUse:
    """A capability's purpose, audience, exclusions and required oversight."""

    regulatory_tier: RegulatoryTier
    purpose: str
    intended_users: tuple[str, ...]
    excluded_uses: tuple[str, ...]
    input_modalities: tuple[str, ...]
    languages: tuple[str, ...]
    human_oversight: str
    autonomous_clinical_action: bool = False
    schema_version: str = "1.0.0"
    summary: str | None = None

    def __post_init__(self) -> None:
        fields = _schema()["properties"]
        if self.schema_version != fields["schema_version"]["const"]:
            raise IntendedUseError("unsupported_schema_version")
        if self.autonomous_clinical_action is not False:
            raise IntendedUseError("autonomous_clinical_action_forbidden")
        for name in ("regulatory_tier", "purpose", "human_oversight"):
            if getattr(self, name) not in fields[name]["enum"]:
                raise IntendedUseError(f"unknown_{name}")
        object.__setattr__(
            self, "regulatory_tier", RegulatoryTier(self.regulatory_tier)
        )
        for name in (
            "intended_users",
            "excluded_uses",
            "input_modalities",
            "languages",
        ):
            values = getattr(self, name)
            if (
                not isinstance(values, (list, tuple))
                or not values
                or any(value not in fields[name]["items"]["enum"] for value in values)
                or len(set(values)) != len(values)
            ):
                raise IntendedUseError(f"invalid_{name}")
            object.__setattr__(self, name, tuple(sorted(values)))
        if self.summary is not None:
            rule = fields["summary"]
            if (
                not isinstance(self.summary, str)
                or not self.summary.strip()
                or len(self.summary) > rule["maxLength"]
                or not re.fullmatch(rule["pattern"], self.summary)
            ):
                raise IntendedUseError("invalid_summary")
            if re.search(rule["not"]["pattern"], self.summary):
                raise IntendedUseError("prohibited_summary_claim")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> IntendedUse:
        """Parse a wire declaration without applying defaults to missing fields."""
        schema = _schema()
        if (
            not isinstance(data, Mapping)
            or set(data) - schema["properties"].keys()
            or not set(schema["required"]).issubset(data)
            or ("summary" in data and data["summary"] is None)
        ):
            raise IntendedUseError("invalid_declaration")
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible canonical declaration."""
        data = {name: getattr(self, name) for name in _schema()["required"]}
        data["regulatory_tier"] = self.regulatory_tier.value
        for name in (
            "intended_users",
            "excluded_uses",
            "input_modalities",
            "languages",
        ):
            data[name] = list(data[name])
        if self.summary is not None:
            data["summary"] = self.summary
        return data

    def digest(self) -> str:
        """Return the SHA-256 digest of the canonical declaration."""
        return schema_fingerprint(self.to_dict())
