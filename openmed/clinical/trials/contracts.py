"""Versioned contracts for cached public clinical-trial study records."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import resources
from typing import Any, Final

from openmed.clinical.journey_contracts import canonical_digest, derived_opaque_id

TRIAL_SCHEMA_VERSION: Final = "1.0.0"
TRIAL_COMPATIBILITY_POLICY: Final = "same_major"
TRIAL_SCHEMA_PACKAGE: Final = "openmed.core.schemas.json"
TRIAL_SCHEMA_NAME: Final = "clinical_trial_study"

_NCT_ID_RE = re.compile(r"^NCT[0-9]{8}$")
_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_TIMESTAMP_RE = re.compile(
    r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}"
    r"(?:\.[0-9]+)?(?:Z|[+-][0-9]{2}:[0-9]{2})$"
)
_DATE_RE = re.compile(r"^[0-9]{4}-[0-9]{2}(?:-[0-9]{2})?$|^[0-9]{4}$")


class TrialContractError(ValueError):
    """Raised when a public study record violates its contract."""


class TrialSchemaDriftError(TrialContractError):
    """Raised when an upstream response no longer has the required shape."""


class TrialCacheCorruptionError(TrialContractError):
    """Raised when persisted cache content fails integrity validation."""


class TrialUnsupportedError(TrialContractError):
    """Raised when a persisted contract version is unsupported."""


class TrialSourceUnavailableError(TrialContractError):
    """Raised when an explicitly requested public source fetch fails."""


@dataclass(frozen=True, slots=True)
class TrialLocation:
    """One public study location."""

    facility: str | None = None
    city: str | None = None
    state: str | None = None
    country: str | None = None

    def __post_init__(self) -> None:
        for name in ("facility", "city", "state", "country"):
            value = getattr(self, name)
            if value is not None:
                _bounded_text(value, name, 512)
        if not any((self.facility, self.city, self.state, self.country)):
            raise TrialContractError("trial location must contain at least one field")

    def to_dict(self) -> dict[str, str | None]:
        """Return the canonical location."""

        return {
            "city": self.city,
            "country": self.country,
            "facility": self.facility,
            "state": self.state,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TrialLocation":
        """Parse a strict public location."""

        data = _mapping(value, "trial location")
        _exact_keys(data, {"city", "country", "facility", "state"}, "trial location")
        return cls(**{name: _optional_text(data[name], name) for name in data})


@dataclass(frozen=True, slots=True, order=True)
class TrialIntervention:
    """One named public study intervention."""

    intervention_type: str
    name: str

    def __post_init__(self) -> None:
        _bounded_text(self.intervention_type, "intervention_type", 128)
        _bounded_text(self.name, "intervention name", 1024)

    def to_dict(self) -> dict[str, str]:
        """Return the canonical intervention."""

        return {"intervention_type": self.intervention_type, "name": self.name}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TrialIntervention":
        """Parse a strict public intervention."""

        data = _mapping(value, "trial intervention")
        _exact_keys(data, {"intervention_type", "name"}, "trial intervention")
        return cls(
            intervention_type=_text(data["intervention_type"], "intervention_type"),
            name=_text(data["name"], "intervention name"),
        )


@dataclass(frozen=True, slots=True)
class TrialStudyRecord:
    """Content-addressed version of one public trial registry study."""

    study_id: str
    overall_status: str
    brief_title: str
    official_title: str | None
    last_update_date: str
    conditions: tuple[str, ...]
    interventions: tuple[TrialIntervention, ...]
    locations: tuple[TrialLocation, ...]
    eligibility_text: str
    retrieved_at: str
    source_digest: str
    schema_version: str = TRIAL_SCHEMA_VERSION
    compatibility_policy: str = TRIAL_COMPATIBILITY_POLICY

    def __post_init__(self) -> None:
        _contract_version(self.schema_version, self.compatibility_policy)
        if (
            not isinstance(self.study_id, str)
            or _NCT_ID_RE.fullmatch(self.study_id) is None
        ):
            raise TrialContractError("study_id must be a normalized NCT identifier")
        _bounded_text(self.overall_status, "overall_status", 128)
        _bounded_text(self.brief_title, "brief_title", 4096)
        if self.official_title is not None:
            _bounded_text(self.official_title, "official_title", 8192)
        if (
            not isinstance(self.last_update_date, str)
            or _DATE_RE.fullmatch(self.last_update_date) is None
        ):
            raise TrialContractError(
                "last_update_date must be an ISO partial or full date"
            )
        _timestamp(self.retrieved_at, "retrieved_at")
        _digest(self.source_digest, "source_digest")
        conditions = _sorted_texts(self.conditions, "conditions", 2048)
        interventions = tuple(sorted(self.interventions))
        locations = tuple(
            sorted(
                self.locations,
                key=lambda item: (
                    item.country or "",
                    item.state or "",
                    item.city or "",
                    item.facility or "",
                ),
            )
        )
        if any(not isinstance(item, TrialIntervention) for item in interventions):
            raise TypeError("interventions must contain TrialIntervention")
        if any(not isinstance(item, TrialLocation) for item in locations):
            raise TypeError("locations must contain TrialLocation")
        if len(interventions) != len(set(interventions)):
            raise TrialContractError("interventions must be unique")
        if len(locations) != len(set(locations)):
            raise TrialContractError("locations must be unique")
        _bounded_text(
            self.eligibility_text, "eligibility_text", 250_000, allow_empty=True
        )
        object.__setattr__(self, "conditions", conditions)
        object.__setattr__(self, "interventions", interventions)
        object.__setattr__(self, "locations", locations)

    @property
    def version_digest(self) -> str:
        """Return the semantic and source digest for this study version."""

        return canonical_digest(self._version_payload())

    @property
    def version_id(self) -> str:
        """Return a deterministic opaque version identifier."""

        return derived_opaque_id("trialversion", self.study_id, self.version_digest)

    def _version_payload(self) -> dict[str, Any]:
        return {
            "brief_title": self.brief_title,
            "conditions": list(self.conditions),
            "eligibility_text": self.eligibility_text,
            "interventions": [item.to_dict() for item in self.interventions],
            "last_update_date": self.last_update_date,
            "locations": [item.to_dict() for item in self.locations],
            "official_title": self.official_title,
            "overall_status": self.overall_status,
            "source_digest": self.source_digest,
            "study_id": self.study_id,
        }

    def to_dict(self) -> dict[str, Any]:
        """Return the complete, schema-valid study version."""

        return {
            "artifact_type": "clinical_trial_study_version",
            "compatibility_policy": self.compatibility_policy,
            **self._version_payload(),
            "retrieved_at": self.retrieved_at,
            "schema_version": self.schema_version,
            "version_digest": self.version_digest,
            "version_id": self.version_id,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TrialStudyRecord":
        """Parse and verify one strict study version."""

        data = _mapping(value, "trial study record")
        expected = {
            "artifact_type",
            "brief_title",
            "compatibility_policy",
            "conditions",
            "eligibility_text",
            "interventions",
            "last_update_date",
            "locations",
            "official_title",
            "overall_status",
            "retrieved_at",
            "schema_version",
            "source_digest",
            "study_id",
            "version_digest",
            "version_id",
        }
        _exact_keys(data, expected, "trial study record")
        if data["artifact_type"] != "clinical_trial_study_version":
            raise TrialContractError("unsupported trial artifact_type")
        record = cls(
            study_id=_text(data["study_id"], "study_id"),
            overall_status=_text(data["overall_status"], "overall_status"),
            brief_title=_text(data["brief_title"], "brief_title"),
            official_title=_optional_text(data["official_title"], "official_title"),
            last_update_date=_text(data["last_update_date"], "last_update_date"),
            conditions=_text_sequence(data["conditions"], "conditions"),
            interventions=tuple(
                TrialIntervention.from_dict(_mapping(item, "trial intervention"))
                for item in _sequence(data["interventions"], "interventions")
            ),
            locations=tuple(
                TrialLocation.from_dict(_mapping(item, "trial location"))
                for item in _sequence(data["locations"], "locations")
            ),
            eligibility_text=_text(
                data["eligibility_text"], "eligibility_text", allow_empty=True
            ),
            retrieved_at=_text(data["retrieved_at"], "retrieved_at"),
            source_digest=_text(data["source_digest"], "source_digest"),
            schema_version=_text(data["schema_version"], "schema_version"),
            compatibility_policy=_text(
                data["compatibility_policy"], "compatibility_policy"
            ),
        )
        if data["version_digest"] != record.version_digest:
            raise TrialCacheCorruptionError(
                "trial version digest does not match content"
            )
        if data["version_id"] != record.version_id:
            raise TrialCacheCorruptionError(
                "trial version identifier does not match content"
            )
        return record


def load_trial_study_schema() -> dict[str, Any]:
    """Load the bundled public study-version JSON Schema."""

    resource = resources.files(TRIAL_SCHEMA_PACKAGE).joinpath(
        f"{TRIAL_SCHEMA_NAME}.schema.json"
    )
    return json.loads(resource.read_text(encoding="utf-8"))


def _contract_version(schema_version: str, compatibility_policy: str) -> None:
    if compatibility_policy != TRIAL_COMPATIBILITY_POLICY:
        raise TrialUnsupportedError("unsupported trial compatibility policy")
    try:
        major = int(schema_version.split(".", 1)[0])
    except (AttributeError, ValueError) as exc:
        raise TrialUnsupportedError("invalid trial schema version") from exc
    if major != int(TRIAL_SCHEMA_VERSION.split(".", 1)[0]):
        raise TrialUnsupportedError("unsupported trial schema major version")


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise TrialContractError(f"{name} must be an object")
    return value


def _sequence(value: Any, name: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise TrialContractError(f"{name} must be an array")
    return value


def _exact_keys(data: Mapping[str, Any], expected: set[str], name: str) -> None:
    if set(data) != expected:
        raise TrialContractError(f"{name} fields do not match the contract")


def _text(value: Any, name: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        raise TrialContractError(f"{name} must be text")
    return value


def _optional_text(value: Any, name: str) -> str | None:
    return None if value is None else _text(value, name)


def _bounded_text(
    value: Any, name: str, limit: int, *, allow_empty: bool = False
) -> str:
    text = _text(value, name, allow_empty=allow_empty)
    if len(text.encode("utf-8")) > limit:
        raise TrialContractError(f"{name} exceeds the byte limit")
    return text


def _text_sequence(value: Any, name: str) -> tuple[str, ...]:
    return tuple(_text(item, name) for item in _sequence(value, name))


def _sorted_texts(value: Any, name: str, limit: int) -> tuple[str, ...]:
    texts = tuple(sorted(_bounded_text(item, name, limit) for item in value))
    if len(texts) != len(set(texts)):
        raise TrialContractError(f"{name} must be unique")
    return texts


def _digest(value: Any, name: str) -> str:
    if not isinstance(value, str) or _DIGEST_RE.fullmatch(value) is None:
        raise TrialContractError(f"{name} must be a normalized SHA-256 digest")
    return value


def _timestamp(value: Any, name: str) -> str:
    if not isinstance(value, str) or _TIMESTAMP_RE.fullmatch(value) is None:
        raise TrialContractError(f"{name} must be an ISO 8601 timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise TrialContractError(f"{name} must be an ISO 8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise TrialContractError(f"{name} must include a UTC offset")
    if parsed.astimezone(timezone.utc).utcoffset() is None:
        raise TrialContractError(f"{name} must include a UTC offset")
    return value
