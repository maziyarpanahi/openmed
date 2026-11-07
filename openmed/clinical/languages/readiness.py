"""Evidence-based language readiness declarations, separate from locale routing."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import islice

_TOKEN = re.compile(r"[a-z][a-z0-9_]{0,63}")
_LANGUAGE = re.compile(r"[a-z]{2,3}(?:-[A-Z][a-z]{3})?(?:-[A-Z]{2})?")
_SCRIPT = re.compile(r"[A-Z][a-z]{1,31}")
_DIGEST = re.compile(r"[0-9a-f]{64}")
_VERSION = re.compile(r"[0-9]+\.[0-9]+\.[0-9]+")
_KINDS = frozenset({"lexicon", "fixture", "evaluation", "native_review", "device"})
_REVIEW = frozenset({"not_requested", "pending", "approved", "rejected"})


def _tokens(values: Iterable[str], pattern: re.Pattern[str]) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError("readiness codes must be a bounded collection")
    result = tuple(islice(values, 129))
    if len(result) > 128 or any(
        not isinstance(value, str) or pattern.fullmatch(value) is None
        for value in result
    ):
        raise ValueError("invalid readiness code collection")
    if len(set(result)) != len(result):
        raise ValueError("duplicate readiness code")
    return tuple(sorted(result))


@dataclass(frozen=True, slots=True)
class LanguageEvidence:
    """Opaque evidence reference bound to one task and optional model tier."""

    kind: str
    task: str
    digest: str
    model_tier: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, str) or self.kind not in _KINDS:
            raise ValueError("invalid readiness evidence kind")
        if not isinstance(self.task, str) or _TOKEN.fullmatch(self.task) is None:
            raise ValueError("invalid readiness task code")
        if not isinstance(self.digest, str) or _DIGEST.fullmatch(self.digest) is None:
            raise ValueError("invalid readiness evidence digest")
        if self.model_tier is not None and (
            not isinstance(self.model_tier, str)
            or _TOKEN.fullmatch(self.model_tier) is None
        ):
            raise ValueError("invalid readiness model tier")
        if self.kind == "device" and self.model_tier is None:
            raise ValueError("device evidence requires a model tier")
        if self.kind != "device" and self.model_tier is not None:
            raise ValueError("model tier is only valid for device evidence")


@dataclass(frozen=True, slots=True)
class ClinicalLanguageReadiness:
    """An immutable evidence inventory with conservative derived readiness."""

    language: str
    version: str
    scripts: tuple[str, ...]
    tasks: tuple[str, ...]
    model_tiers: tuple[str, ...] = ()
    reviewer_status: str = "not_requested"
    evidence: tuple[LanguageEvidence, ...] = ()
    limitations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.language, str):
            raise ValueError("invalid readiness language")
        parts = self.language.replace("_", "-").split("-")
        parts[0] = parts[0].lower()
        for i in range(1, len(parts)):
            parts[i] = parts[i].title() if len(parts[i]) == 4 else parts[i].upper()
        code = "-".join(parts)
        if _LANGUAGE.fullmatch(code) is None:
            raise ValueError("invalid readiness language")
        object.__setattr__(self, "language", code)
        if (
            not isinstance(self.version, str)
            or _VERSION.fullmatch(self.version) is None
        ):
            raise ValueError("invalid readiness version")
        for field, pattern in (
            ("scripts", _SCRIPT),
            ("tasks", _TOKEN),
            ("model_tiers", _TOKEN),
            ("limitations", _TOKEN),
        ):
            object.__setattr__(self, field, _tokens(getattr(self, field), pattern))
        if (
            not isinstance(self.reviewer_status, str)
            or self.reviewer_status not in _REVIEW
        ):
            raise ValueError("invalid readiness review status")
        if isinstance(self.evidence, (str, bytes)):
            raise ValueError("invalid readiness evidence collection")
        evidence = tuple(islice(self.evidence, 1025))
        if len(evidence) > 1024 or any(
            not isinstance(e, LanguageEvidence) for e in evidence
        ):
            raise ValueError("invalid readiness evidence collection")
        if len(set(evidence)) != len(evidence):
            raise ValueError("duplicate readiness evidence")
        for item in evidence:
            if item.task not in self.tasks:
                raise ValueError("evidence refers to an undeclared task")
            if item.model_tier is not None and item.model_tier not in self.model_tiers:
                raise ValueError("evidence refers to an undeclared model tier")
        object.__setattr__(
            self,
            "evidence",
            tuple(
                sorted(
                    evidence,
                    key=lambda e: (e.task, e.kind, e.model_tier or "", e.digest),
                )
            ),
        )

    def missing_evidence(self) -> tuple[str, ...]:
        """Return controlled reasons preventing full declared support."""
        missing = set()
        if not self.scripts:
            missing.add("scripts_missing")
        if not self.tasks:
            missing.add("tasks_missing")
        if self.reviewer_status != "approved":
            missing.add("native_review_not_approved")
        if not self.model_tiers:
            missing.add("model_tiers_missing")
        keys = {(e.task, e.kind, e.model_tier) for e in self.evidence}
        for task in self.tasks:
            for kind in ("lexicon", "fixture", "evaluation", "native_review"):
                if (task, kind, None) not in keys:
                    missing.add(kind + "_missing")
            for tier in self.model_tiers:
                if (task, "device", tier) not in keys:
                    missing.add("device_missing")
        if self.limitations:
            missing.add("declared_limitations")
        return tuple(sorted(missing))

    @property
    def state(self) -> str:
        """Compute experimental, preview or supported from explicit evidence.

        Supported is a declaration that required metadata exists, not a claim
        that the evidence itself proves clinical fitness or certification.
        """
        missing = set(self.missing_evidence())
        if not missing:
            return "supported"
        foundational = {
            "scripts_missing",
            "tasks_missing",
            "lexicon_missing",
            "fixture_missing",
            "evaluation_missing",
            "native_review_missing",
            "native_review_not_approved",
        }
        if missing & foundational:
            return "experimental"
        return "preview"

    def to_dict(self) -> dict[str, object]:
        """Export controlled metadata without reviewer names or clinical text."""
        return {
            "language": self.language,
            "version": self.version,
            "scripts": list(self.scripts),
            "tasks": list(self.tasks),
            "model_tiers": list(self.model_tiers),
            "reviewer_status": self.reviewer_status,
            "evidence": [
                {
                    "kind": e.kind,
                    "task": e.task,
                    "digest": e.digest,
                    "model_tier": e.model_tier,
                }
                for e in self.evidence
            ],
            "limitations": list(self.limitations),
            "state": self.state,
            "missing_evidence": list(self.missing_evidence()),
        }


class ClinicalLanguageReadinessRegistry:
    """A private explicit inventory; registration never promotes locale support."""

    def __init__(self, entries: Iterable[ClinicalLanguageReadiness] = ()) -> None:
        self._entries: dict[str, ClinicalLanguageReadiness] = {}
        for index, entry in enumerate(entries):
            if index >= 1000:
                raise ValueError("readiness registry limit exceeded")
            self.register(entry)

    def register(self, entry: ClinicalLanguageReadiness) -> None:
        """Register one immutable declaration; duplicate locales reject."""
        if not isinstance(entry, ClinicalLanguageReadiness):
            raise ValueError("invalid readiness registry entry")
        if entry.language in self._entries:
            raise ValueError("duplicate readiness registry language")
        if len(self._entries) >= 1000:
            raise ValueError("readiness registry limit exceeded")
        self._entries[entry.language] = entry

    def report(self) -> dict[str, object]:
        """Return stable metadata ordered by canonical language code."""
        return {
            "schema_version": "1.0.0",
            "languages": [
                self._entries[code].to_dict() for code in sorted(self._entries)
            ],
        }

    def to_json(self) -> str:
        """Serialize the bounded registry as deterministic JSON."""
        return json.dumps(self.report(), sort_keys=True, separators=(",", ":")) + "\n"
