"""Versioned English notices and value-free acknowledgements for reviewed use."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from types import MappingProxyType

from openmed.core.intended_use import RegulatoryTier, load_intended_use_registry
from openmed.core.no_phi_telemetry import record_use_error_if_enabled

OPENMED_CLINICAL_DISCLAIMER = (
    "OpenMed clinical outputs are assistive software outputs, not a medical "
    "device, diagnosis, or substitute for qualified clinical judgment."
)
MAPLE_MEDICAL_DISCLAIMER = (
    "Maple output is assistive and may be incomplete or incorrect. It must be "
    "reviewed by a qualified person and must not trigger a diagnosis, treatment, "
    "disclosure, or other clinical decision automatically."
)


@dataclass(frozen=True, slots=True)
class NoticeAcknowledgement:
    """Receipt binding an acknowledgement to a notice version and a UTC instant."""

    notice_id: str
    version_digest: str
    acknowledged_at: str


@dataclass(frozen=True, slots=True)
class OpenMedDisclaimer:
    """Published wording and tier obligations for one version of a notice."""

    notice_id: str
    version: str
    locale: str
    text: str
    tier: RegulatoryTier

    @property
    def version_digest(self) -> str:
        """Identify this notice's id, version, locale, wording and tier."""
        fields = [self.notice_id, self.version, self.locale, self.text, self.tier.value]
        wire = json.dumps(fields, ensure_ascii=False, separators=(",", ":"))
        return "sha256:" + hashlib.sha256(wire.encode("utf-8")).hexdigest()

    @property
    def requires_acknowledgement(self) -> bool:
        """Whether the tier requires an explicit acknowledgement."""
        return "acknowledgement_required" in self.tier.obligations

    def acknowledge(self, *, at: datetime) -> NoticeAcknowledgement | None:
        """Record an explicit acknowledgement with a caller-supplied aware instant."""
        if not self.requires_acknowledgement:
            return None
        if not isinstance(at, datetime) or at.utcoffset() is None:
            raise ValueError("invalid_acknowledgement_time")
        return NoticeAcknowledgement(
            self.notice_id, self.version_digest, at.astimezone(timezone.utc).isoformat()
        )

    def require_acknowledgement(
        self, receipt: NoticeAcknowledgement | None, *, capability: str = "notice"
    ) -> None:
        """Reject an absent or unbound receipt when this tier requires one."""
        if not self.requires_acknowledgement:
            return
        if receipt is None:
            record_use_error_if_enabled(
                "acknowledgement_skipped", capability, self.tier.value
            )
            raise ValueError("acknowledgement_required")
        if (
            not isinstance(receipt, NoticeAcknowledgement)
            or receipt.notice_id != self.notice_id
            or receipt.version_digest != self.version_digest
        ):
            raise ValueError("invalid_acknowledgement")
        try:
            if datetime.fromisoformat(receipt.acknowledged_at).utcoffset() is None:
                raise ValueError
        except (TypeError, ValueError):
            raise ValueError("invalid_acknowledgement") from None

    @classmethod
    def for_tier(cls, tier: RegulatoryTier | str) -> OpenMedDisclaimer:
        """Return the canonical notice for a regulatory tier."""
        try:
            tier = RegulatoryTier(tier)
        except ValueError:
            raise ValueError("unknown_regulatory_tier") from None
        return _TIERS[tier]

    @classmethod
    def for_capability(cls, capability: str) -> OpenMedDisclaimer:
        """Resolve a reviewed capability or entry-point alias to its notice."""
        registry = load_intended_use_registry()
        declaration = registry.resolve(capability)
        code = registry.entry_points.get(capability, capability)
        if code == "generate_text" and "maple" in capability.lower():
            return _MAPLE
        if code in {"ecg", "waveform", "visual_qa"}:
            from openmed.multimodal.notices import NoticeKind

            kind = (
                NoticeKind.VISUAL_DESCRIPTION
                if code == "visual_qa"
                else NoticeKind.MEASUREMENT
            )
            return multimodal_disclaimers()[kind]
        return cls.for_tier(declaration.regulatory_tier)


def _notice(name: str, text: str, tier: RegulatoryTier) -> OpenMedDisclaimer:
    return OpenMedDisclaimer(f"openmed.{name}", "1", "en", text, tier)


_TIERS = MappingProxyType(
    {
        RegulatoryTier.NON_CLINICAL: _notice(
            "non_clinical",
            "Review privacy processing results before release.",
            RegulatoryTier.NON_CLINICAL,
        ),
        RegulatoryTier.INFORMATIONAL: _notice(
            "informational",
            "For information only. Review the source before use.",
            RegulatoryTier.INFORMATIONAL,
        ),
        RegulatoryTier.ASSISTIVE_REVIEW: _notice(
            "clinical", OPENMED_CLINICAL_DISCLAIMER, RegulatoryTier.ASSISTIVE_REVIEW
        ),
        RegulatoryTier.RESTRICTED: _notice(
            "restricted",
            "Restricted to clinician review. A qualified clinician must independently verify the source before consequential use.",
            RegulatoryTier.RESTRICTED,
        ),
    }
)
_MAPLE = _notice("maple", MAPLE_MEDICAL_DISCLAIMER, RegulatoryTier.ASSISTIVE_REVIEW)


@lru_cache(maxsize=1)
def multimodal_disclaimers():
    """Expose the versioned identifiers and wording from the multimodal catalog."""
    from openmed.multimodal.notices import NOTICE_CATALOG, NoticeKind

    return MappingProxyType(
        {
            kind: OpenMedDisclaimer(
                notice.identifier,
                notice.identifier.rsplit(".v", 1)[1],
                "en",
                notice.text,
                RegulatoryTier.ASSISTIVE_REVIEW
                if kind == NoticeKind.DRAFT
                else RegulatoryTier.RESTRICTED,
            )
            for kind, notice in NOTICE_CATALOG.items()
        }
    )


@lru_cache(maxsize=1)
def notice_catalog():
    """Return all published notices indexed by their canonical identifiers."""
    notices = (*_TIERS.values(), _MAPLE, *multimodal_disclaimers().values())
    return MappingProxyType({notice.notice_id: notice for notice in notices})
