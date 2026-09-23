"""Contracts for chart suspicions and population-level descriptive signals."""

from __future__ import annotations

import json
import math
import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from importlib import resources
from typing import Any, Final

from openmed.clinical.journey_contracts import canonical_digest, derived_opaque_id

DRUG_SAFETY_SCHEMA_VERSION: Final = "1.0.0"
DRUG_SAFETY_COMPATIBILITY_POLICY: Final = "same_major"
DRUG_SAFETY_METHOD_ID: Final = "report_level_prr_ror"
DRUG_SAFETY_METHOD_VERSION: Final = "1.0.0"
DRUG_SAFETY_FILTER_VERSION: Final = "1.0.0"
DRUG_SAFETY_SCHEMA_PACKAGE: Final = "openmed.core.schemas.json"
DRUG_SAFETY_SCHEMA_NAME: Final = "drug_safety_signal"
DRUG_SAFETY_ADVISORY: Final = (
    "This descriptive signal is hypothesis-generating only. It does not establish "
    "causality and must not direct diagnosis, treatment, prescribing, or outreach."
)
SUSPECTED_RELATION_ADVISORY: Final = (
    "This chart-level relation is a review-required suspicion only. It is not a "
    "population signal, causal claim, diagnosis, or treatment recommendation."
)

_CONTROLLED_RE = re.compile(r"^[a-z][a-z0-9_.:/-]{0,127}$")
_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_OPAQUE_ID_RE = re.compile(r"^[a-z][a-z0-9_]{0,31}_[A-Za-z0-9_-]{16,128}$")
_VERSION_RE = re.compile(r"^[1-9][0-9]*\.[0-9]+\.[0-9]+$")


class DrugSafetyContractError(ValueError):
    """Raised when a drug-safety artifact violates its contract."""


class DrugSafetyConflictError(DrugSafetyContractError):
    """Raised when immutable source rows or evidence conflict."""


class DrugSafetyUnsupportedError(DrugSafetyContractError):
    """Raised when a schema, adapter, or calculation is unsupported."""


class SafetySeriousness(str, Enum):
    """Normalized public event seriousness."""

    SERIOUS = "serious"
    NON_SERIOUS = "non_serious"
    UNKNOWN = "unknown"


class SignalState(str, Enum):
    """Explicit descriptive calculation outcomes."""

    COMPUTED = "computed"
    SUPPRESSED = "suppressed"
    INSUFFICIENT_DATA = "insufficient_data"


@dataclass(frozen=True, slots=True, order=True)
class NormalizedSafetyTerm:
    """Normalized drug or event identity from a public source."""

    kind: str
    code: str
    system: str = "open_text"

    def __post_init__(self) -> None:
        if self.kind not in {"drug", "event"}:
            raise DrugSafetyContractError("safety term kind must be drug or event")
        normalized = _normalize_term(self.code)
        if not normalized:
            raise DrugSafetyContractError("safety term code must be non-empty")
        _controlled(self.system, "term system")
        object.__setattr__(self, "code", normalized)

    def to_dict(self) -> dict[str, str]:
        """Return the normalized public term."""

        return {"code": self.code, "kind": self.kind, "system": self.system}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "NormalizedSafetyTerm":
        """Parse one strict normalized term."""

        data = _mapping(value, "safety term")
        _exact_keys(data, {"code", "kind", "system"}, "safety term")
        return cls(
            kind=_text(data["kind"], "term kind"),
            code=_text(data["code"], "term code"),
            system=_text(data["system"], "term system"),
        )


@dataclass(frozen=True, slots=True, order=True)
class ExposureWindow:
    """Inclusive day offsets from exposure start to event observation."""

    start_day: int
    end_day: int

    def __post_init__(self) -> None:
        for name in ("start_day", "end_day"):
            value = getattr(self, name)
            if type(value) is not int or not -36_500 <= value <= 36_500:
                raise DrugSafetyContractError(f"{name} is outside the supported range")
        if self.end_day < self.start_day:
            raise DrugSafetyContractError("exposure window end precedes start")

    def to_dict(self) -> dict[str, int]:
        """Return the exposure window."""

        return {"end_day": self.end_day, "start_day": self.start_day}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ExposureWindow":
        """Parse one strict exposure window."""

        data = _mapping(value, "exposure window")
        _exact_keys(data, {"end_day", "start_day"}, "exposure window")
        return cls(
            start_day=_integer(data["start_day"], "start_day"),
            end_day=_integer(data["end_day"], "end_day"),
        )


@dataclass(frozen=True, slots=True)
class DrugExposure:
    """One normalized drug and optional public exposure window."""

    drug: NormalizedSafetyTerm
    window: ExposureWindow | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.drug, NormalizedSafetyTerm) or self.drug.kind != "drug":
            raise DrugSafetyContractError("exposure requires a normalized drug")
        if self.window is not None and not isinstance(self.window, ExposureWindow):
            raise TypeError("window must be ExposureWindow")

    def to_dict(self) -> dict[str, Any]:
        """Return the normalized exposure."""

        return {
            "drug": self.drug.to_dict(),
            "window": self.window.to_dict() if self.window else None,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DrugExposure":
        """Parse one strict drug exposure."""

        data = _mapping(value, "drug exposure")
        _exact_keys(data, {"drug", "window"}, "drug exposure")
        window_value = data["window"]
        return cls(
            drug=NormalizedSafetyTerm.from_dict(
                _mapping(data["drug"], "exposure drug")
            ),
            window=(
                None
                if window_value is None
                else ExposureWindow.from_dict(_mapping(window_value, "exposure window"))
            ),
        )


@dataclass(frozen=True, slots=True, order=True)
class DrugSafetyEvidence:
    """Value-free source-row evidence for one imported public case."""

    row_digest: str
    source_digest: str

    def __post_init__(self) -> None:
        _digest(self.row_digest, "row_digest")
        _digest(self.source_digest, "source_digest")

    def to_dict(self) -> dict[str, str]:
        """Return value-free source evidence."""

        return {"row_digest": self.row_digest, "source_digest": self.source_digest}


@dataclass(frozen=True, slots=True)
class SafetyCase:
    """One de-duplicated public report projected into normalized terms."""

    case_id: str
    exposures: tuple[DrugExposure, ...]
    events: tuple[NormalizedSafetyTerm, ...]
    seriousness: SafetySeriousness
    evidence: tuple[DrugSafetyEvidence, ...]

    def __post_init__(self) -> None:
        _opaque_id(self.case_id, "case_id")
        exposures = tuple(self.exposures)
        events = tuple(self.events)
        evidence = tuple(self.evidence)
        if not exposures or any(
            not isinstance(item, DrugExposure) for item in exposures
        ):
            raise DrugSafetyContractError("safety case requires drug exposures")
        if not events or any(
            not isinstance(item, NormalizedSafetyTerm) or item.kind != "event"
            for item in events
        ):
            raise DrugSafetyContractError("safety case requires normalized events")
        if not evidence or any(
            not isinstance(item, DrugSafetyEvidence) for item in evidence
        ):
            raise DrugSafetyContractError("safety case requires source evidence")
        exposures = tuple(
            sorted(
                exposures,
                key=lambda item: (
                    item.drug,
                    item.window is None,
                    item.window.start_day if item.window else 0,
                    item.window.end_day if item.window else 0,
                ),
            )
        )
        events = tuple(sorted(events))
        evidence = tuple(sorted(evidence))
        if len(exposures) != len(set(exposures)) or len(events) != len(set(events)):
            raise DrugSafetyConflictError("case drugs and events must be unique")
        if len(evidence) != len(set(evidence)):
            raise DrugSafetyConflictError("case evidence must be unique")
        object.__setattr__(self, "seriousness", _seriousness(self.seriousness))
        object.__setattr__(self, "exposures", exposures)
        object.__setattr__(self, "events", events)
        object.__setattr__(self, "evidence", evidence)

    def to_dict(self) -> dict[str, Any]:
        """Return the normalized public case without source report identity."""

        return {
            "case_id": self.case_id,
            "events": [item.to_dict() for item in self.events],
            "evidence": [item.to_dict() for item in self.evidence],
            "exposures": [item.to_dict() for item in self.exposures],
            "seriousness": self.seriousness.value,
        }


@dataclass(frozen=True, slots=True)
class DrugSafetyDatasetManifest:
    """Pinned user-supplied public event dataset identity and terms."""

    dataset_id: str
    version: str
    source_digest: str
    license_id: str
    adapter_version: str

    def __post_init__(self) -> None:
        _controlled(self.dataset_id, "dataset_id")
        _bounded_text(self.version, "dataset version", 128)
        _digest(self.source_digest, "source_digest")
        _bounded_text(self.license_id, "license_id", 256)
        _bounded_text(self.adapter_version, "adapter_version", 128)

    @property
    def digest(self) -> str:
        """Return the exact manifest digest."""

        return canonical_digest(self.to_dict())

    def to_dict(self) -> dict[str, str]:
        """Return source, version, license, and adapter custody."""

        return {
            "adapter_version": self.adapter_version,
            "dataset_id": self.dataset_id,
            "license_id": self.license_id,
            "source_digest": self.source_digest,
            "version": self.version,
        }


@dataclass(frozen=True, slots=True)
class DrugSafetyDataset:
    """Imported normalized public cases with duplicate provenance."""

    manifest: DrugSafetyDatasetManifest
    cases: tuple[SafetyCase, ...]
    duplicate_row_count: int
    imported_row_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.manifest, DrugSafetyDatasetManifest):
            raise TypeError("manifest must be DrugSafetyDatasetManifest")
        cases = tuple(self.cases)
        if any(not isinstance(item, SafetyCase) for item in cases):
            raise TypeError("cases must contain SafetyCase")
        cases = tuple(sorted(cases, key=lambda item: item.case_id))
        if len({item.case_id for item in cases}) != len(cases):
            raise DrugSafetyConflictError("dataset case identifiers must be unique")
        for name in ("duplicate_row_count", "imported_row_count"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise DrugSafetyContractError(f"{name} must be non-negative")
        if self.duplicate_row_count > self.imported_row_count:
            raise DrugSafetyContractError("duplicate count exceeds imported rows")
        object.__setattr__(self, "cases", cases)

    @property
    def dataset_digest(self) -> str:
        """Return a normalized case and manifest digest."""

        return canonical_digest(
            {
                "cases": [item.to_dict() for item in self.cases],
                "duplicate_row_count": self.duplicate_row_count,
                "imported_row_count": self.imported_row_count,
                "manifest": self.manifest.to_dict(),
            }
        )


@dataclass(frozen=True, slots=True)
class SignalFilter:
    """Versioned, digest-bound population filter for one calculation."""

    seriousness: tuple[SafetySeriousness, ...] = ()
    minimum_exposure_start_day: int | None = None
    maximum_exposure_end_day: int | None = None
    version: str = DRUG_SAFETY_FILTER_VERSION

    def __post_init__(self) -> None:
        if self.version != DRUG_SAFETY_FILTER_VERSION:
            raise DrugSafetyUnsupportedError("unsupported signal filter version")
        seriousness = tuple(
            sorted(
                {_seriousness(item) for item in self.seriousness}, key=lambda x: x.value
            )
        )
        if self.minimum_exposure_start_day is not None:
            _bounded_day(self.minimum_exposure_start_day, "minimum_exposure_start_day")
        if self.maximum_exposure_end_day is not None:
            _bounded_day(self.maximum_exposure_end_day, "maximum_exposure_end_day")
        if (
            self.minimum_exposure_start_day is not None
            and self.maximum_exposure_end_day is not None
            and self.maximum_exposure_end_day < self.minimum_exposure_start_day
        ):
            raise DrugSafetyContractError("signal filter exposure bounds conflict")
        object.__setattr__(self, "seriousness", seriousness)

    @property
    def digest(self) -> str:
        """Return the exact population filter digest."""

        return canonical_digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        """Return the complete population filter."""

        return {
            "maximum_exposure_end_day": self.maximum_exposure_end_day,
            "minimum_exposure_start_day": self.minimum_exposure_start_day,
            "seriousness": [item.value for item in self.seriousness],
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SignalFilter":
        """Parse one strict population filter."""

        data = _mapping(value, "signal filter")
        _exact_keys(
            data,
            {
                "maximum_exposure_end_day",
                "minimum_exposure_start_day",
                "seriousness",
                "version",
            },
            "signal filter",
        )
        minimum = data["minimum_exposure_start_day"]
        maximum = data["maximum_exposure_end_day"]
        return cls(
            seriousness=tuple(
                _seriousness(item)
                for item in _sequence(data["seriousness"], "seriousness")
            ),
            minimum_exposure_start_day=(
                None
                if minimum is None
                else _integer(minimum, "minimum_exposure_start_day")
            ),
            maximum_exposure_end_day=(
                None
                if maximum is None
                else _integer(maximum, "maximum_exposure_end_day")
            ),
            version=_text(data["version"], "filter version"),
        )


@dataclass(frozen=True, slots=True)
class SignalPolicy:
    """Minimum-count and zero-cell suppression policy."""

    policy_id: str = "drug_safety_default"
    version: str = "1.0.0"
    minimum_pair_count: int = 3
    minimum_cell_count: int = 1

    def __post_init__(self) -> None:
        _controlled(self.policy_id, "policy_id")
        if _VERSION_RE.fullmatch(self.version) is None:
            raise DrugSafetyContractError("signal policy version must be semantic")
        for name in ("minimum_pair_count", "minimum_cell_count"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise DrugSafetyContractError(f"{name} must be positive")

    @property
    def digest(self) -> str:
        """Return the exact suppression policy digest."""

        return canonical_digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        """Return the complete signal policy."""

        return {
            "minimum_cell_count": self.minimum_cell_count,
            "minimum_pair_count": self.minimum_pair_count,
            "policy_id": self.policy_id,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SignalPolicy":
        """Parse one strict suppression policy."""

        data = _mapping(value, "signal policy")
        _exact_keys(
            data,
            {"minimum_cell_count", "minimum_pair_count", "policy_id", "version"},
            "signal policy",
        )
        return cls(
            policy_id=_text(data["policy_id"], "policy_id"),
            version=_text(data["version"], "policy version"),
            minimum_pair_count=_integer(
                data["minimum_pair_count"], "minimum_pair_count"
            ),
            minimum_cell_count=_integer(
                data["minimum_cell_count"], "minimum_cell_count"
            ),
        )


@dataclass(frozen=True, slots=True)
class SignalContingencyTable:
    """Four report-level cells for one drug-event pair."""

    drug_event: int
    drug_other_event: int
    other_drug_event: int
    other_drug_other_event: int

    def __post_init__(self) -> None:
        for name in (
            "drug_event",
            "drug_other_event",
            "other_drug_event",
            "other_drug_other_event",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise DrugSafetyContractError("contingency cells must be non-negative")

    @property
    def total(self) -> int:
        """Return the filtered report count."""

        return sum(self.to_dict().values())

    def to_dict(self) -> dict[str, int]:
        """Return cells using descriptive names."""

        return {
            "drug_event": self.drug_event,
            "drug_other_event": self.drug_other_event,
            "other_drug_event": self.other_drug_event,
            "other_drug_other_event": self.other_drug_other_event,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SignalContingencyTable":
        """Parse one strict report-level contingency table."""

        data = _mapping(value, "contingency table")
        expected = {
            "drug_event",
            "drug_other_event",
            "other_drug_event",
            "other_drug_other_event",
        }
        _exact_keys(data, expected, "contingency table")
        return cls(
            drug_event=_integer(data["drug_event"], "drug_event"),
            drug_other_event=_integer(data["drug_other_event"], "drug_other_event"),
            other_drug_event=_integer(data["other_drug_event"], "other_drug_event"),
            other_drug_other_event=_integer(
                data["other_drug_other_event"], "other_drug_other_event"
            ),
        )


@dataclass(frozen=True, slots=True)
class DescriptiveSignal:
    """Hypothesis-generating population signal with complete provenance."""

    signal_id: str
    drug: NormalizedSafetyTerm
    event: NormalizedSafetyTerm
    state: SignalState
    reason_codes: tuple[str, ...]
    table: SignalContingencyTable
    proportional_reporting_ratio: float | None
    reporting_odds_ratio: float | None
    dataset_id: str
    dataset_version: str
    license_id: str
    source_digest: str
    dataset_digest: str
    adapter_version: str
    filter: SignalFilter
    filter_digest: str
    policy: SignalPolicy
    policy_digest: str
    duplicate_row_count: int
    caveats: tuple[str, ...]
    method_id: str = DRUG_SAFETY_METHOD_ID
    method_version: str = DRUG_SAFETY_METHOD_VERSION
    schema_version: str = DRUG_SAFETY_SCHEMA_VERSION
    compatibility_policy: str = DRUG_SAFETY_COMPATIBILITY_POLICY

    def __post_init__(self) -> None:
        _contract(self.schema_version, self.compatibility_policy)
        if (
            self.method_id != DRUG_SAFETY_METHOD_ID
            or self.method_version != DRUG_SAFETY_METHOD_VERSION
        ):
            raise DrugSafetyUnsupportedError("unsupported signal calculation method")
        _opaque_id(self.signal_id, "signal_id")
        if not isinstance(self.drug, NormalizedSafetyTerm) or self.drug.kind != "drug":
            raise DrugSafetyContractError("signal drug is invalid")
        if (
            not isinstance(self.event, NormalizedSafetyTerm)
            or self.event.kind != "event"
        ):
            raise DrugSafetyContractError("signal event is invalid")
        object.__setattr__(self, "state", _signal_state(self.state))
        reasons = _controlled_values(self.reason_codes, "reason_codes")
        caveats = _controlled_values(self.caveats, "caveats")
        if not isinstance(self.table, SignalContingencyTable):
            raise TypeError("table must be SignalContingencyTable")
        for value, name in (
            (self.proportional_reporting_ratio, "proportional_reporting_ratio"),
            (self.reporting_odds_ratio, "reporting_odds_ratio"),
        ):
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) < 0
            ):
                raise DrugSafetyContractError(f"{name} must be finite and non-negative")
        if self.state is SignalState.COMPUTED and (
            self.proportional_reporting_ratio is None
            or self.reporting_odds_ratio is None
            or reasons
        ):
            raise DrugSafetyContractError(
                "computed signal requires both ratios and no reason"
            )
        if self.state is not SignalState.COMPUTED and (
            self.proportional_reporting_ratio is not None
            or self.reporting_odds_ratio is not None
            or not reasons
        ):
            raise DrugSafetyContractError(
                "non-computed signal requires reasons and no ratios"
            )
        _controlled(self.dataset_id, "dataset_id")
        _bounded_text(self.dataset_version, "dataset_version", 128)
        _bounded_text(self.license_id, "license_id", 256)
        for digest_value, name in (
            (self.source_digest, "source_digest"),
            (self.dataset_digest, "dataset_digest"),
            (self.filter_digest, "filter_digest"),
            (self.policy_digest, "policy_digest"),
        ):
            _digest(digest_value, name)
        _bounded_text(self.adapter_version, "adapter_version", 128)
        if (
            not isinstance(self.filter, SignalFilter)
            or self.filter.digest != self.filter_digest
        ):
            raise DrugSafetyContractError("signal filter custody differs")
        if (
            not isinstance(self.policy, SignalPolicy)
            or self.policy.digest != self.policy_digest
        ):
            raise DrugSafetyContractError("signal policy custody differs")
        expected_state, expected_reasons, expected_prr, expected_ror = (
            _calculate_table_outcome(self.table, self.policy)
        )
        if (
            self.state is not expected_state
            or reasons != tuple(sorted(expected_reasons))
            or self.proportional_reporting_ratio != expected_prr
            or self.reporting_odds_ratio != expected_ror
        ):
            raise DrugSafetyConflictError(
                "signal state or statistics differ from its table and policy"
            )
        if type(self.duplicate_row_count) is not int or self.duplicate_row_count < 0:
            raise DrugSafetyContractError("duplicate_row_count must be non-negative")
        if not caveats:
            raise DrugSafetyContractError("descriptive signal requires caveats")
        expected_signal_id = derived_opaque_id(
            "safetysignal",
            self.dataset_digest,
            self.drug.to_dict(),
            self.event.to_dict(),
            self.filter_digest,
            self.policy_digest,
            self.method_id,
            self.method_version,
        )
        if self.signal_id != expected_signal_id:
            raise DrugSafetyConflictError("signal identifier custody differs")
        object.__setattr__(self, "reason_codes", reasons)
        object.__setattr__(self, "caveats", caveats)

    @property
    def signal_digest(self) -> str:
        """Return the complete descriptive signal digest."""

        return canonical_digest(self._payload())

    def _payload(self) -> dict[str, Any]:
        return {
            "adapter_version": self.adapter_version,
            "advisory": DRUG_SAFETY_ADVISORY,
            "caveats": list(self.caveats),
            "compatibility_policy": self.compatibility_policy,
            "dataset_digest": self.dataset_digest,
            "dataset_id": self.dataset_id,
            "dataset_version": self.dataset_version,
            "drug": self.drug.to_dict(),
            "duplicate_row_count": self.duplicate_row_count,
            "event": self.event.to_dict(),
            "filter": self.filter.to_dict(),
            "filter_digest": self.filter_digest,
            "method_id": self.method_id,
            "method_version": self.method_version,
            "license_id": self.license_id,
            "policy": self.policy.to_dict(),
            "policy_digest": self.policy_digest,
            "proportional_reporting_ratio": self.proportional_reporting_ratio,
            "reason_codes": list(self.reason_codes),
            "reporting_odds_ratio": self.reporting_odds_ratio,
            "schema_version": self.schema_version,
            "signal_id": self.signal_id,
            "source_digest": self.source_digest,
            "state": self.state.value,
            "table": self.table.to_dict(),
        }

    def to_dict(self) -> dict[str, Any]:
        """Return the strict public signal artifact."""

        return {**self._payload(), "signal_digest": self.signal_digest}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DescriptiveSignal":
        """Parse and verify one strict descriptive signal artifact."""

        data = _mapping(value, "descriptive signal")
        expected = {
            "adapter_version",
            "advisory",
            "caveats",
            "compatibility_policy",
            "dataset_digest",
            "dataset_id",
            "dataset_version",
            "drug",
            "duplicate_row_count",
            "event",
            "filter",
            "filter_digest",
            "method_id",
            "method_version",
            "license_id",
            "policy",
            "policy_digest",
            "proportional_reporting_ratio",
            "reason_codes",
            "reporting_odds_ratio",
            "schema_version",
            "signal_digest",
            "signal_id",
            "source_digest",
            "state",
            "table",
        }
        _exact_keys(data, expected, "descriptive signal")
        if data["advisory"] != DRUG_SAFETY_ADVISORY:
            raise DrugSafetyContractError("drug-safety advisory differs")
        result = cls(
            signal_id=_text(data["signal_id"], "signal_id"),
            drug=NormalizedSafetyTerm.from_dict(_mapping(data["drug"], "signal drug")),
            event=NormalizedSafetyTerm.from_dict(
                _mapping(data["event"], "signal event")
            ),
            state=_signal_state(data["state"]),
            reason_codes=tuple(
                _text(item, "reason code")
                for item in _sequence(data["reason_codes"], "reason_codes")
            ),
            table=SignalContingencyTable.from_dict(
                _mapping(data["table"], "contingency table")
            ),
            proportional_reporting_ratio=_optional_number(
                data["proportional_reporting_ratio"],
                "proportional_reporting_ratio",
            ),
            reporting_odds_ratio=_optional_number(
                data["reporting_odds_ratio"], "reporting_odds_ratio"
            ),
            dataset_id=_text(data["dataset_id"], "dataset_id"),
            dataset_version=_text(data["dataset_version"], "dataset_version"),
            license_id=_text(data["license_id"], "license_id"),
            source_digest=_text(data["source_digest"], "source_digest"),
            dataset_digest=_text(data["dataset_digest"], "dataset_digest"),
            adapter_version=_text(data["adapter_version"], "adapter_version"),
            filter=SignalFilter.from_dict(_mapping(data["filter"], "signal filter")),
            filter_digest=_text(data["filter_digest"], "filter_digest"),
            method_id=_text(data["method_id"], "method_id"),
            method_version=_text(data["method_version"], "method_version"),
            policy=SignalPolicy.from_dict(_mapping(data["policy"], "signal policy")),
            policy_digest=_text(data["policy_digest"], "policy_digest"),
            duplicate_row_count=_integer(
                data["duplicate_row_count"], "duplicate_row_count"
            ),
            caveats=tuple(
                _text(item, "caveat") for item in _sequence(data["caveats"], "caveats")
            ),
            schema_version=_text(data["schema_version"], "schema_version"),
            compatibility_policy=_text(
                data["compatibility_policy"], "compatibility_policy"
            ),
        )
        if data["signal_digest"] != result.signal_digest:
            raise DrugSafetyConflictError("signal digest differs")
        return result


@dataclass(frozen=True, slots=True)
class SuspectedDrugEventRelation:
    """Chart-level suspicion, intentionally separate from population signals."""

    relation_id: str
    snapshot_id: str
    snapshot_digest: str
    drug_fact_ids: tuple[str, ...]
    event_fact_ids: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    reason_code: str
    review_required: bool = True
    schema_version: str = DRUG_SAFETY_SCHEMA_VERSION
    compatibility_policy: str = DRUG_SAFETY_COMPATIBILITY_POLICY

    def __post_init__(self) -> None:
        _contract(self.schema_version, self.compatibility_policy)
        for value, name in (
            (self.relation_id, "relation_id"),
            (self.snapshot_id, "snapshot_id"),
        ):
            _opaque_id(value, name)
        _digest(self.snapshot_digest, "snapshot_digest")
        object.__setattr__(
            self,
            "drug_fact_ids",
            _opaque_values(self.drug_fact_ids, "drug_fact_ids", 1),
        )
        object.__setattr__(
            self,
            "event_fact_ids",
            _opaque_values(self.event_fact_ids, "event_fact_ids", 1),
        )
        object.__setattr__(
            self, "evidence_ids", _opaque_values(self.evidence_ids, "evidence_ids", 1)
        )
        _controlled(self.reason_code, "reason_code")
        if self.review_required is not True:
            raise DrugSafetyContractError("suspected chart relation requires review")

    def to_dict(self) -> dict[str, Any]:
        """Return a value-free suspected relation without population metrics."""

        return {**self._payload(), "relation_digest": self.relation_digest}

    @property
    def relation_digest(self) -> str:
        """Return the exact value-free suspected-relation digest."""

        return canonical_digest(self._payload())

    def _payload(self) -> dict[str, Any]:
        return {
            "advisory": SUSPECTED_RELATION_ADVISORY,
            "artifact_type": "suspected_chart_drug_event_relation",
            "compatibility_policy": self.compatibility_policy,
            "drug_fact_ids": list(self.drug_fact_ids),
            "event_fact_ids": list(self.event_fact_ids),
            "evidence_ids": list(self.evidence_ids),
            "reason_code": self.reason_code,
            "relation_id": self.relation_id,
            "review_required": self.review_required,
            "schema_version": self.schema_version,
            "snapshot_digest": self.snapshot_digest,
            "snapshot_id": self.snapshot_id,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SuspectedDrugEventRelation":
        """Parse and verify one value-free suspected relation."""

        data = _mapping(value, "suspected drug-event relation")
        expected = {
            "advisory",
            "artifact_type",
            "compatibility_policy",
            "drug_fact_ids",
            "event_fact_ids",
            "evidence_ids",
            "reason_code",
            "relation_digest",
            "relation_id",
            "review_required",
            "schema_version",
            "snapshot_digest",
            "snapshot_id",
        }
        _exact_keys(data, expected, "suspected drug-event relation")
        if data["advisory"] != SUSPECTED_RELATION_ADVISORY:
            raise DrugSafetyContractError("suspected-relation advisory differs")
        if data["artifact_type"] != "suspected_chart_drug_event_relation":
            raise DrugSafetyContractError("suspected-relation type differs")
        relation = cls(
            relation_id=_text(data["relation_id"], "relation_id"),
            snapshot_id=_text(data["snapshot_id"], "snapshot_id"),
            snapshot_digest=_text(data["snapshot_digest"], "snapshot_digest"),
            drug_fact_ids=tuple(
                _text(item, "drug fact identifier")
                for item in _sequence(data["drug_fact_ids"], "drug_fact_ids")
            ),
            event_fact_ids=tuple(
                _text(item, "event fact identifier")
                for item in _sequence(data["event_fact_ids"], "event_fact_ids")
            ),
            evidence_ids=tuple(
                _text(item, "evidence identifier")
                for item in _sequence(data["evidence_ids"], "evidence_ids")
            ),
            reason_code=_text(data["reason_code"], "reason_code"),
            review_required=_boolean(data["review_required"], "review_required"),
            schema_version=_text(data["schema_version"], "schema_version"),
            compatibility_policy=_text(
                data["compatibility_policy"], "compatibility_policy"
            ),
        )
        if data["relation_digest"] != relation.relation_digest:
            raise DrugSafetyConflictError("suspected-relation digest differs")
        return relation


def load_drug_safety_signal_schema() -> dict[str, Any]:
    """Load the bundled descriptive signal JSON Schema."""

    resource = resources.files(DRUG_SAFETY_SCHEMA_PACKAGE).joinpath(
        f"{DRUG_SAFETY_SCHEMA_NAME}.schema.json"
    )
    return json.loads(resource.read_text(encoding="utf-8"))


def _normalize_term(value: str) -> str:
    if not isinstance(value, str):
        raise DrugSafetyContractError("safety term must be text")
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(re.findall(r"[^\W_]+", normalized, flags=re.UNICODE))


def _contract(schema_version: str, compatibility_policy: str) -> None:
    if compatibility_policy != DRUG_SAFETY_COMPATIBILITY_POLICY:
        raise DrugSafetyUnsupportedError("unsupported drug-safety compatibility policy")
    if (
        not isinstance(schema_version, str)
        or _VERSION_RE.fullmatch(schema_version) is None
        or schema_version.split(".", 1)[0]
        != DRUG_SAFETY_SCHEMA_VERSION.split(".", 1)[0]
    ):
        raise DrugSafetyUnsupportedError("unsupported drug-safety schema version")


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise DrugSafetyContractError(f"{name} must be an object")
    return value


def _exact_keys(data: Mapping[str, Any], expected: set[str], name: str) -> None:
    if set(data) != expected:
        raise DrugSafetyContractError(f"{name} fields do not match the contract")


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DrugSafetyContractError(f"{name} must be non-empty text")
    return value


def _bounded_text(value: Any, name: str, limit: int) -> str:
    text = _text(value, name)
    if len(text.encode("utf-8")) > limit:
        raise DrugSafetyContractError(f"{name} exceeds the byte limit")
    return text


def _controlled(value: Any, name: str) -> str:
    if not isinstance(value, str) or _CONTROLLED_RE.fullmatch(value) is None:
        raise DrugSafetyContractError(f"{name} must be controlled text")
    return value


def _controlled_values(values: Sequence[str], name: str) -> tuple[str, ...]:
    normalized = tuple(sorted({_controlled(item, name) for item in values}))
    return normalized


def _opaque_id(value: Any, name: str) -> str:
    if not isinstance(value, str) or _OPAQUE_ID_RE.fullmatch(value) is None:
        raise DrugSafetyContractError(f"{name} must be an opaque identifier")
    return value


def _opaque_values(
    values: Sequence[str], name: str, minimum: int = 0
) -> tuple[str, ...]:
    normalized = tuple(sorted(_opaque_id(item, name) for item in values))
    if len(normalized) < minimum or len(normalized) != len(set(normalized)):
        raise DrugSafetyContractError(f"{name} count or uniqueness is invalid")
    return normalized


def _digest(value: Any, name: str) -> str:
    if not isinstance(value, str) or _DIGEST_RE.fullmatch(value) is None:
        raise DrugSafetyContractError(f"{name} must be a normalized SHA-256 digest")
    return value


def _integer(value: Any, name: str) -> int:
    if type(value) is not int:
        raise DrugSafetyContractError(f"{name} must be an integer")
    return value


def _boolean(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise DrugSafetyContractError(f"{name} must be boolean")
    return value


def _optional_number(value: Any, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DrugSafetyContractError(f"{name} must be a number or null")
    result = float(value)
    if not math.isfinite(result):
        raise DrugSafetyContractError(f"{name} must be finite")
    return result


def _sequence(value: Any, name: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise DrugSafetyContractError(f"{name} must be an array")
    return value


def _bounded_day(value: Any, name: str) -> int:
    if type(value) is not int or not -36_500 <= value <= 36_500:
        raise DrugSafetyContractError(f"{name} is outside the supported range")
    return value


def _seriousness(value: Any) -> SafetySeriousness:
    try:
        return (
            value if isinstance(value, SafetySeriousness) else SafetySeriousness(value)
        )
    except (TypeError, ValueError):
        raise DrugSafetyContractError("seriousness is unsupported") from None


def _signal_state(value: Any) -> SignalState:
    try:
        return value if isinstance(value, SignalState) else SignalState(value)
    except (TypeError, ValueError):
        raise DrugSafetyContractError("signal state is unsupported") from None


def _calculate_table_outcome(
    table: SignalContingencyTable, policy: SignalPolicy
) -> tuple[SignalState, tuple[str, ...], float | None, float | None]:
    """Return the only valid state and ratios for a table-policy pair."""

    if table.total == 0:
        return (
            SignalState.INSUFFICIENT_DATA,
            ("filtered_population_empty",),
            None,
            None,
        )
    if table.drug_event + table.drug_other_event == 0:
        return (
            SignalState.INSUFFICIENT_DATA,
            ("drug_denominator_missing",),
            None,
            None,
        )
    if table.other_drug_event + table.other_drug_other_event == 0:
        return (
            SignalState.INSUFFICIENT_DATA,
            ("comparator_denominator_missing",),
            None,
            None,
        )
    reasons: list[str] = []
    if table.drug_event < policy.minimum_pair_count:
        reasons.append("pair_count_below_minimum")
    if table.other_drug_event == 0 or table.drug_other_event == 0:
        reasons.append("zero_cell_ratio_undefined")
    if any(value < policy.minimum_cell_count for value in table.to_dict().values()):
        reasons.append("cell_count_below_minimum")
    if reasons:
        return SignalState.SUPPRESSED, tuple(reasons), None, None
    drug_event_rate = table.drug_event / (table.drug_event + table.drug_other_event)
    other_event_rate = table.other_drug_event / (
        table.other_drug_event + table.other_drug_other_event
    )
    return (
        SignalState.COMPUTED,
        (),
        drug_event_rate / other_event_rate,
        (table.drug_event * table.other_drug_other_event)
        / (table.drug_other_event * table.other_drug_event),
    )
