"""Versioned, loss-aware exchange envelopes for cohort definitions.

The existing :mod:`openmed.structured.cohort.dsl` is the canonical cohort
language.  This module adds source-snapshot custody, criterion evidence
bindings, compatibility metadata, and explicit conversion losses without
introducing another query language or executable SQL surface.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib import resources
from typing import Any, Final

from openmed.clinical.journey_contracts import canonical_digest, canonical_json
from openmed.structured.store import StoreResult, StoreState

from .dsl import PHENOTYPE_SCHEMA_VERSION, PhenotypeDefinition

COHORT_EXCHANGE_SCHEMA_VERSION: Final = "1.0.0"
COHORT_EXCHANGE_COMPATIBILITY_POLICY: Final = "same_major"
COHORT_EXCHANGE_SCHEMA_NAME: Final = "cohort_definition_exchange"
COHORT_EXCHANGE_SCHEMA_PACKAGE: Final = "openmed.core.schemas.json"

_CONTROLLED_RE = re.compile(r"^[a-z][a-z0-9_.:/-]{0,127}$")
_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_OPAQUE_ID_RE = re.compile(r"^[a-z][a-z0-9_]{0,31}_[A-Za-z0-9_-]{16,128}$")


class CohortExchangeError(ValueError):
    """Base error for malformed cohort exchange contracts."""


class CohortExchangeConflictError(CohortExchangeError):
    """Raised when definition, snapshot, or evidence custody conflicts."""


class CohortExchangeUnsupportedError(CohortExchangeError):
    """Raised when an exchange version or capability is unsupported."""


@dataclass(frozen=True, slots=True)
class CohortSourceSnapshot:
    """Value-free reference to the data/vocabulary snapshot used by a cohort."""

    snapshot_id: str
    digest: str
    schema_version: str
    license_tags: tuple[str, ...] = ()
    bundled_vocabulary: bool = False

    def __post_init__(self) -> None:
        _opaque_id(self.snapshot_id, "source snapshot_id")
        _digest(self.digest, "source snapshot digest")
        _bounded_text(self.schema_version, "source snapshot schema_version", 64)
        tags = tuple(
            sorted({_controlled(item, "license tag") for item in self.license_tags})
        )
        object.__setattr__(self, "license_tags", tags)
        if type(self.bundled_vocabulary) is not bool:
            raise CohortExchangeError("bundled_vocabulary must be boolean")
        if self.bundled_vocabulary:
            raise CohortExchangeUnsupportedError(
                "cohort exchange cannot bundle vocabulary content"
            )

    def to_dict(self) -> dict[str, Any]:
        """Return the value-free snapshot reference."""

        return {
            "bundled_vocabulary": self.bundled_vocabulary,
            "digest": self.digest,
            "license_tags": list(self.license_tags),
            "schema_version": self.schema_version,
            "snapshot_id": self.snapshot_id,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CohortSourceSnapshot":
        """Build a strict source-snapshot reference."""

        data = _mapping(value, "source_snapshot")
        _exact_keys(
            data,
            {
                "bundled_vocabulary",
                "digest",
                "license_tags",
                "schema_version",
                "snapshot_id",
            },
            "source_snapshot",
        )
        bundled = data["bundled_vocabulary"]
        if type(bundled) is not bool:
            raise CohortExchangeError("bundled_vocabulary must be boolean")
        return cls(
            snapshot_id=_text(data["snapshot_id"], "source snapshot_id"),
            digest=_text(data["digest"], "source snapshot digest"),
            schema_version=_text(
                data["schema_version"], "source snapshot schema_version"
            ),
            license_tags=_text_sequence(data["license_tags"], "license_tags"),
            bundled_vocabulary=bundled,
        )


@dataclass(frozen=True, slots=True)
class CohortEvidenceBinding:
    """Criterion-to-concept-set custody retained across conversion."""

    criterion_id: str
    concept_set_id: str
    definition_digest: str

    def __post_init__(self) -> None:
        _controlled(self.criterion_id, "criterion_id")
        _controlled(self.concept_set_id, "concept_set_id")
        _digest(self.definition_digest, "definition_digest")

    def to_dict(self) -> dict[str, str]:
        """Return the stable evidence binding."""

        return {
            "concept_set_id": self.concept_set_id,
            "criterion_id": self.criterion_id,
            "definition_digest": self.definition_digest,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CohortEvidenceBinding":
        """Build a strict evidence binding."""

        data = _mapping(value, "evidence binding")
        _exact_keys(
            data,
            {"concept_set_id", "criterion_id", "definition_digest"},
            "evidence binding",
        )
        return cls(
            criterion_id=_text(data["criterion_id"], "criterion_id"),
            concept_set_id=_text(data["concept_set_id"], "concept_set_id"),
            definition_digest=_text(data["definition_digest"], "definition_digest"),
        )


@dataclass(frozen=True, slots=True)
class CohortConversionLoss:
    """Value-free record of an unsupported external conversion field."""

    path: str
    reason_code: str

    def __post_init__(self) -> None:
        _bounded_text(self.path, "loss path", 256)
        _controlled(self.reason_code, "loss reason_code")

    def to_dict(self) -> dict[str, str]:
        """Return the visible loss record."""

        return {"path": self.path, "reason_code": self.reason_code}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CohortConversionLoss":
        """Build a strict loss record."""

        data = _mapping(value, "conversion loss")
        _exact_keys(data, {"path", "reason_code"}, "conversion loss")
        return cls(
            path=_text(data["path"], "loss path"),
            reason_code=_text(data["reason_code"], "loss reason_code"),
        )


@dataclass(frozen=True, slots=True)
class CohortDefinitionExchange:
    """Canonical cohort definition with snapshot and evidence custody."""

    definition: PhenotypeDefinition
    source_snapshot: CohortSourceSnapshot
    evidence_mapping: tuple[CohortEvidenceBinding, ...]
    losses: tuple[CohortConversionLoss, ...] = ()
    schema_version: str = COHORT_EXCHANGE_SCHEMA_VERSION
    compatibility_policy: str = COHORT_EXCHANGE_COMPATIBILITY_POLICY

    def __post_init__(self) -> None:
        if self.schema_version != COHORT_EXCHANGE_SCHEMA_VERSION:
            raise CohortExchangeUnsupportedError(
                "cohort exchange schema version is unsupported"
            )
        if self.compatibility_policy != COHORT_EXCHANGE_COMPATIBILITY_POLICY:
            raise CohortExchangeUnsupportedError(
                "cohort exchange compatibility policy is unsupported"
            )
        if not isinstance(self.definition, PhenotypeDefinition):
            raise TypeError("definition must be PhenotypeDefinition")
        if not isinstance(self.source_snapshot, CohortSourceSnapshot):
            raise TypeError("source_snapshot must be CohortSourceSnapshot")
        bindings = tuple(
            sorted(self.evidence_mapping, key=lambda item: item.criterion_id)
        )
        criteria = {item.id: item.concept_set for item in self.definition.criteria()}
        if {item.criterion_id for item in bindings} != set(criteria):
            raise CohortExchangeConflictError(
                "evidence mapping must cover every criterion exactly once"
            )
        expected_digest = self.definition_digest
        for item in bindings:
            if criteria[item.criterion_id] != item.concept_set_id:
                raise CohortExchangeConflictError(
                    "evidence mapping concept set differs from definition"
                )
            if item.definition_digest != expected_digest:
                raise CohortExchangeConflictError(
                    "evidence mapping definition digest differs"
                )
        object.__setattr__(self, "evidence_mapping", bindings)
        object.__setattr__(
            self,
            "losses",
            tuple(sorted(self.losses, key=lambda item: (item.path, item.reason_code))),
        )

    @property
    def definition_digest(self) -> str:
        """Return the normalized digest of the canonical definition bytes."""

        return f"sha256:{self.definition.sha256}"

    @property
    def lossless(self) -> bool:
        """Return whether the exchange has no declared conversion loss."""

        return not self.losses

    @property
    def canonical_hash(self) -> str:
        """Return a digest of the complete exchange envelope."""

        return canonical_digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        """Return the strict persisted exchange envelope."""

        return {
            "compatibility_policy": self.compatibility_policy,
            "definition": self.definition.to_dict(),
            "definition_digest": self.definition_digest,
            "evidence_mapping": [item.to_dict() for item in self.evidence_mapping],
            "losses": [item.to_dict() for item in self.losses],
            "lossless": self.lossless,
            "schema_version": self.schema_version,
            "source_snapshot": self.source_snapshot.to_dict(),
        }

    def to_json(self) -> str:
        """Serialize to canonical, whitespace-free JSON."""

        return canonical_json(self.to_dict())

    def to_json_bytes(self) -> bytes:
        """Serialize to canonical UTF-8 bytes."""

        return self.to_json().encode("utf-8")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CohortDefinitionExchange":
        """Build a strict envelope and verify all derived fields."""

        data = _mapping(value, "cohort definition exchange")
        _exact_keys(
            data,
            {
                "compatibility_policy",
                "definition",
                "definition_digest",
                "evidence_mapping",
                "losses",
                "lossless",
                "schema_version",
                "source_snapshot",
            },
            "cohort definition exchange",
        )
        result = cls(
            definition=PhenotypeDefinition.from_dict(
                _mapping(data["definition"], "definition")
            ),
            source_snapshot=CohortSourceSnapshot.from_dict(
                _mapping(data["source_snapshot"], "source_snapshot")
            ),
            evidence_mapping=tuple(
                CohortEvidenceBinding.from_dict(_mapping(item, "evidence binding"))
                for item in _sequence(data["evidence_mapping"], "evidence_mapping")
            ),
            losses=tuple(
                CohortConversionLoss.from_dict(_mapping(item, "conversion loss"))
                for item in _sequence(data["losses"], "losses")
            ),
            schema_version=_text(data["schema_version"], "schema_version"),
            compatibility_policy=_text(
                data["compatibility_policy"], "compatibility_policy"
            ),
        )
        if data["definition_digest"] != result.definition_digest:
            raise CohortExchangeConflictError("definition digest differs")
        if data["lossless"] is not result.lossless:
            raise CohortExchangeConflictError("persisted lossless flag differs")
        return result

    @classmethod
    def from_json(cls, value: str | bytes | bytearray) -> "CohortDefinitionExchange":
        """Parse a cohort exchange JSON document."""

        try:
            payload = json.loads(value)
        except (TypeError, UnicodeDecodeError, json.JSONDecodeError):
            raise CohortExchangeError("cohort exchange is not valid JSON") from None
        return cls.from_dict(_mapping(payload, "cohort definition exchange"))


def export_cohort_definition(
    definition: PhenotypeDefinition,
    *,
    source_snapshot: CohortSourceSnapshot,
) -> StoreResult[CohortDefinitionExchange]:
    """Wrap a canonical phenotype in a reproducible exchange envelope."""

    try:
        if not isinstance(definition, PhenotypeDefinition):
            raise TypeError("definition must be PhenotypeDefinition")
        digest = f"sha256:{definition.sha256}"
        bindings = tuple(
            CohortEvidenceBinding(
                criterion_id=criterion.id,
                concept_set_id=criterion.concept_set,
                definition_digest=digest,
            )
            for criterion in definition.criteria()
        )
        return StoreResult.success(
            CohortDefinitionExchange(
                definition=definition,
                source_snapshot=source_snapshot,
                evidence_mapping=bindings,
            ),
            created=True,
        )
    except CohortExchangeUnsupportedError:
        return StoreResult.outcome(
            StoreState.UNSUPPORTED, "cohort_exchange_unsupported"
        )
    except CohortExchangeConflictError:
        return StoreResult.outcome(StoreState.CONFLICT, "cohort_exchange_conflict")
    except (CohortExchangeError, TypeError, ValueError):
        return StoreResult.outcome(StoreState.FAILURE, "cohort_exchange_invalid")


def import_cohort_definition(
    value: Mapping[str, Any] | str | bytes | bytearray,
    *,
    expected_snapshot: CohortSourceSnapshot | None = None,
    strict: bool = False,
) -> StoreResult[CohortDefinitionExchange]:
    """Import a cohort envelope with typed loss and snapshot handling."""

    try:
        result = (
            CohortDefinitionExchange.from_dict(value)
            if isinstance(value, Mapping)
            else CohortDefinitionExchange.from_json(value)
        )
        if (
            expected_snapshot is not None
            and result.source_snapshot != expected_snapshot
        ):
            raise CohortExchangeConflictError("source snapshot differs")
        if result.losses:
            state = StoreState.UNSUPPORTED if strict else StoreState.PARTIAL
            return StoreResult.outcome(
                state,
                "cohort_exchange_unsupported" if strict else "cohort_exchange_partial",
                value=result,
            )
        return StoreResult.success(result)
    except CohortExchangeUnsupportedError:
        return StoreResult.outcome(
            StoreState.UNSUPPORTED, "cohort_exchange_unsupported"
        )
    except CohortExchangeConflictError:
        return StoreResult.outcome(StoreState.CONFLICT, "cohort_exchange_conflict")
    except (CohortExchangeError, TypeError, ValueError):
        return StoreResult.outcome(StoreState.FAILURE, "cohort_exchange_invalid")


def load_cohort_exchange_schema() -> Mapping[str, Any]:
    """Load the bundled strict JSON Schema for exchange envelopes."""

    path = resources.files(COHORT_EXCHANGE_SCHEMA_PACKAGE).joinpath(
        f"{COHORT_EXCHANGE_SCHEMA_NAME}.schema.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != COHORT_EXCHANGE_SCHEMA_VERSION:
        raise RuntimeError("bundled cohort exchange schema is incompatible")
    return payload


def _mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CohortExchangeError(f"{field_name} must be an object")
    return value


def _sequence(value: Any, field_name: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise CohortExchangeError(f"{field_name} must be an array")
    return value


def _text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise CohortExchangeError(f"{field_name} must be non-empty text")
    return value


def _text_sequence(value: Any, field_name: str) -> tuple[str, ...]:
    return tuple(_text(item, field_name) for item in _sequence(value, field_name))


def _exact_keys(value: Mapping[str, Any], expected: set[str], field_name: str) -> None:
    if set(value) != expected:
        raise CohortExchangeError(f"{field_name} fields are incompatible")


def _bounded_text(value: str, field_name: str, maximum: int) -> str:
    if not isinstance(value, str) or not value or len(value) > maximum:
        raise CohortExchangeError(f"{field_name} must be bounded text")
    if any(ord(character) < 32 for character in value):
        raise CohortExchangeError(f"{field_name} contains a control character")
    return value


def _controlled(value: str, field_name: str) -> str:
    if not isinstance(value, str) or _CONTROLLED_RE.fullmatch(value) is None:
        raise CohortExchangeError(f"{field_name} must be controlled")
    return value


def _opaque_id(value: str, field_name: str) -> str:
    if not isinstance(value, str) or _OPAQUE_ID_RE.fullmatch(value) is None:
        raise CohortExchangeError(f"{field_name} must be opaque")
    return value


def _digest(value: str, field_name: str) -> str:
    if not isinstance(value, str) or _DIGEST_RE.fullmatch(value) is None:
        raise CohortExchangeError(f"{field_name} must be a digest")
    return value


__all__ = [
    "COHORT_EXCHANGE_COMPATIBILITY_POLICY",
    "COHORT_EXCHANGE_SCHEMA_NAME",
    "COHORT_EXCHANGE_SCHEMA_VERSION",
    "CohortConversionLoss",
    "CohortDefinitionExchange",
    "CohortEvidenceBinding",
    "CohortExchangeConflictError",
    "CohortExchangeError",
    "CohortExchangeUnsupportedError",
    "CohortSourceSnapshot",
    "export_cohort_definition",
    "import_cohort_definition",
    "load_cohort_exchange_schema",
    "PHENOTYPE_SCHEMA_VERSION",
]
