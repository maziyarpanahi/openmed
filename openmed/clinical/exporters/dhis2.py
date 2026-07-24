"""Local-first DHIS2 aggregate and tracker payload export.

The exporter accepts already-shaped DHIS2 payloads, de-identifies the fields
that can carry free text, generalizes organisation-unit references against a
local hierarchy snapshot, and emits a PHI-free transformation manifest. It
does not contain an HTTP client and never contacts a DHIS2 instance.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Literal, cast

DateMode = Literal["shift", "coarsen", "none"]
PeriodGranularity = Literal["month", "year"]
TextRedactor = Callable[[str], Any]

DEFAULT_GENERALIZATION_LEVEL = 3
DEFAULT_SMALL_CELL_THRESHOLD = 5

_DATE_FIELDS = frozenset(
    {
        "completedAt",
        "completedDate",
        "createdAt",
        "createdAtClient",
        "dueDate",
        "enrolledAt",
        "enrollmentDate",
        "eventDate",
        "incidentDate",
        "occurredAt",
        "scheduledAt",
        "updatedAt",
        "updatedAtClient",
    }
)
_TEXT_FIELDS = frozenset({"comment", "completedBy", "storedBy"})
_PRECISE_GEOGRAPHY_FIELDS = frozenset({"geometry", "latitude", "longitude"})
_ISO_DATE_RE = re.compile(
    r"^(?P<date>\d{4}-\d{2}-\d{2})"
    r"(?P<suffix>T\d{2}:\d{2}(?::\d{2}(?:\.\d{1,9})?)?"
    r"(?:Z|[+-]\d{2}:?\d{2})?)?$"
)
_WEEKLY_PERIOD_RE = re.compile(r"^(\d{4})W(\d{1,2})$")
_UNSET = cast(int | None, object())


class DHIS2ExportError(ValueError):
    """Raised when a local snapshot or payload cannot be exported safely."""


@dataclass(frozen=True)
class DHIS2ExportConfig:
    """Privacy and geography policy for a DHIS2 export.

    Args:
        generalization_level: Highest permitted fine-grained organisation-unit
            level. DHIS2 levels start at 1; level 3 is conventionally district.
        small_cell_threshold: Suppress aggregate numeric values below this
            threshold. ``None`` or ``0`` disables suppression.
        date_mode: Shift event dates, coarsen them, or leave them unchanged.
        date_shift_days: Optional fixed non-zero shift. When omitted, a stable
            per-record offset is derived with the tabular date policy.
        date_shift_seed: Seed for deterministic per-record date shifts.
        keep_year: Preserve source years while shifting dates.
        period_granularity: Month or year precision used in coarsening mode.
        lang: OpenMed language hint passed to date and text de-identification.
        policy: OpenMed de-identification policy used by the default redactor.
    """

    generalization_level: int = DEFAULT_GENERALIZATION_LEVEL
    small_cell_threshold: int | None = DEFAULT_SMALL_CELL_THRESHOLD
    date_mode: DateMode = "shift"
    date_shift_days: int | None = None
    date_shift_seed: str = "openmed-dhis2-v1"
    keep_year: bool = True
    period_granularity: PeriodGranularity = "month"
    lang: str = "en"
    policy: str = "hipaa_safe_harbor"

    def __post_init__(self) -> None:
        if isinstance(self.generalization_level, bool) or not isinstance(
            self.generalization_level, int
        ):
            raise TypeError("generalization_level must be an integer")
        if self.generalization_level < 1:
            raise ValueError("generalization_level must be >= 1")
        threshold = self.small_cell_threshold
        if threshold is not None:
            if isinstance(threshold, bool) or not isinstance(threshold, int):
                raise TypeError("small_cell_threshold must be an integer or None")
            if threshold < 0:
                raise ValueError("small_cell_threshold must be >= 0")
        if self.date_mode not in {"shift", "coarsen", "none"}:
            raise ValueError("date_mode must be 'shift', 'coarsen', or 'none'")
        if self.date_shift_days is not None:
            if isinstance(self.date_shift_days, bool) or not isinstance(
                self.date_shift_days, int
            ):
                raise TypeError("date_shift_days must be an integer or None")
            if self.date_shift_days == 0:
                raise ValueError("date_shift_days must be non-zero")
        if self.date_mode != "shift" and self.date_shift_days is not None:
            raise ValueError("date_shift_days requires date_mode='shift'")
        if self.period_granularity not in {"month", "year"}:
            raise ValueError("period_granularity must be 'month' or 'year'")
        if not self.date_shift_seed:
            raise ValueError("date_shift_seed must not be empty")
        if not self.lang:
            raise ValueError("lang must not be empty")
        if not self.policy:
            raise ValueError("policy must not be empty")


@dataclass(frozen=True)
class DHIS2ExportResult:
    """De-identified DHIS2 endpoint payloads and their PHI-free manifest."""

    aggregate_payload: dict[str, Any]
    tracker_payload: dict[str, Any]
    manifest: dict[str, Any]

    @property
    def combined_payload(self) -> dict[str, Any]:
        """Return the three DHIS2 collections in one transport-neutral object."""

        return {
            "dataValueSets": self.aggregate_payload["dataValueSets"],
            "trackedEntities": self.tracker_payload["trackedEntities"],
            "events": self.tracker_payload["events"],
        }

    def aggregate_json(self) -> str:
        """Serialize the aggregate endpoint payload deterministically."""

        return _canonical_json(self.aggregate_payload)

    def tracker_json(self) -> str:
        """Serialize the tracker endpoint payload deterministically."""

        return _canonical_json(self.tracker_payload)

    def manifest_json(self) -> str:
        """Serialize the PHI-free manifest deterministically."""

        return _canonical_json(self.manifest)

    def to_json(self) -> str:
        """Serialize both endpoint payloads and the manifest deterministically."""

        return _canonical_json(
            {
                "aggregate": self.aggregate_payload,
                "manifest": self.manifest,
                "tracker": self.tracker_payload,
            }
        )


@dataclass(frozen=True)
class _OrgUnit:
    uid: str
    level: int
    parent_uid: str | None


class OrgUnitHierarchy:
    """Validated, in-memory DHIS2 organisation-unit hierarchy snapshot."""

    def __init__(self, units: Mapping[str, _OrgUnit]) -> None:
        self._units = dict(units)
        self._validate_links()

    @classmethod
    def from_snapshot(cls, snapshot: Any) -> OrgUnitHierarchy:
        """Load a hierarchy from local JSON data, a path, or a text stream."""

        payload = _load_json_source(snapshot, source_name="org-unit snapshot")
        if isinstance(payload, Mapping):
            raw_units = payload.get("organisationUnits")
        else:
            raw_units = payload
        if not _is_sequence(raw_units):
            raise DHIS2ExportError(
                "org-unit snapshot must contain an organisationUnits array"
            )

        units: dict[str, _OrgUnit] = {}
        for index, raw_unit in enumerate(raw_units):
            if not isinstance(raw_unit, Mapping):
                raise DHIS2ExportError(
                    f"organisationUnits[{index}] must be a JSON object"
                )
            uid = raw_unit.get("id", raw_unit.get("uid"))
            level = raw_unit.get("level")
            if not isinstance(uid, str) or not uid:
                raise DHIS2ExportError(
                    f"organisationUnits[{index}] must have a non-empty id"
                )
            if isinstance(level, bool) or not isinstance(level, int) or level < 1:
                raise DHIS2ExportError(
                    f"organisationUnits[{index}].level must be a positive integer"
                )
            parent_uid = _parent_uid(raw_unit.get("parent"), index=index)
            if uid in units:
                raise DHIS2ExportError("org-unit snapshot contains duplicate ids")
            units[uid] = _OrgUnit(uid=uid, level=level, parent_uid=parent_uid)

        if not units:
            raise DHIS2ExportError("org-unit snapshot must not be empty")
        return cls(units)

    def generalize(self, uid: str, *, target_level: int, path: str) -> str:
        """Return the target-level ancestor for one payload reference."""

        unit = self._lookup(uid, path=path)
        if unit.level <= target_level:
            return unit.uid

        seen: set[str] = set()
        current = unit
        while current.level > target_level:
            if current.uid in seen:
                raise DHIS2ExportError(
                    f"{path} cannot be generalized because the snapshot has a cycle"
                )
            seen.add(current.uid)
            if current.parent_uid is None:
                raise DHIS2ExportError(
                    f"{path} has no ancestor at generalization level {target_level}"
                )
            current = self._lookup(current.parent_uid, path=path)

        if current.level != target_level:
            raise DHIS2ExportError(
                f"{path} has no ancestor at generalization level {target_level}"
            )
        return current.uid

    def assert_permitted(self, uid: str, *, target_level: int, path: str) -> None:
        """Raise when a payload reference is finer than the configured level."""

        unit = self._lookup(uid, path=path)
        if unit.level > target_level:
            raise DHIS2ExportError(
                f"{path} remains below generalization level {target_level}"
            )

    def _lookup(self, uid: str, *, path: str) -> _OrgUnit:
        try:
            return self._units[uid]
        except KeyError as exc:
            raise DHIS2ExportError(
                f"{path} references an org unit absent from the local snapshot"
            ) from exc

    def _validate_links(self) -> None:
        for unit in self._units.values():
            if unit.parent_uid is None:
                continue
            parent = self._units.get(unit.parent_uid)
            if parent is None:
                raise DHIS2ExportError(
                    "org-unit snapshot contains a parent absent from the snapshot"
                )
            if parent.level >= unit.level:
                raise DHIS2ExportError(
                    "org-unit snapshot parent levels must be lower than child levels"
                )

        for unit in self._units.values():
            seen: set[str] = set()
            current = unit
            while current.parent_uid is not None:
                if current.uid in seen:
                    raise DHIS2ExportError("org-unit snapshot contains a cycle")
                seen.add(current.uid)
                current = self._units[current.parent_uid]


@dataclass
class _ManifestState:
    data_value_sets: int = 0
    aggregate_values_input: int = 0
    aggregate_values_output: int = 0
    suppressed_aggregate_values: int = 0
    tracked_entities: int = 0
    events: int = 0
    org_units_examined: int = 0
    org_units_generalized: int = 0
    precise_locations_removed: int = 0
    text_values_examined: int = 0
    text_values_redacted: int = 0
    dates_examined: int = 0
    dates_transformed: int = 0
    periods_examined: int = 0
    periods_coarsened: int = 0
    transformed_paths: set[str] | None = None

    def __post_init__(self) -> None:
        if self.transformed_paths is None:
            self.transformed_paths = set()

    def changed(self, path: str) -> None:
        if self.transformed_paths is None:
            self.transformed_paths = set()
        self.transformed_paths.add(path)

    def to_manifest(self, config: DHIS2ExportConfig) -> dict[str, Any]:
        """Build a deterministic manifest containing metadata and counts only."""

        if self.transformed_paths is None:
            self.transformed_paths = set()
        return {
            "schema_version": 1,
            "exporter": "dhis2",
            "generalization_level": config.generalization_level,
            "small_cell_threshold": config.small_cell_threshold,
            "date_policy": {
                "mode": config.date_mode,
                "fixed_shift_days": config.date_shift_days,
                "keep_year": config.keep_year,
                "period_granularity": config.period_granularity,
            },
            "counts": {
                "data_value_sets": self.data_value_sets,
                "aggregate_values_input": self.aggregate_values_input,
                "aggregate_values_output": self.aggregate_values_output,
                "suppressed_aggregate_values": (self.suppressed_aggregate_values),
                "tracked_entities": self.tracked_entities,
                "events": self.events,
                "org_units_examined": self.org_units_examined,
                "org_units_generalized": self.org_units_generalized,
                "precise_locations_removed": self.precise_locations_removed,
                "text_values_examined": self.text_values_examined,
                "text_values_redacted": self.text_values_redacted,
                "dates_examined": self.dates_examined,
                "dates_transformed": self.dates_transformed,
                "periods_examined": self.periods_examined,
                "periods_coarsened": self.periods_coarsened,
            },
            "transformed_paths": sorted(self.transformed_paths),
        }


class DHIS2Exporter:
    """Transform DHIS2 endpoint payloads entirely in local memory."""

    def __init__(
        self,
        org_unit_snapshot: Any,
        *,
        config: DHIS2ExportConfig | None = None,
        text_redactor: TextRedactor | None = None,
    ) -> None:
        self.config = config or DHIS2ExportConfig()
        self.hierarchy = OrgUnitHierarchy.from_snapshot(org_unit_snapshot)
        self._text_redactor = text_redactor or self._default_text_redactor

    def export(
        self,
        aggregate_payload: Any = None,
        tracker_payload: Any = None,
    ) -> DHIS2ExportResult:
        """Return de-identified aggregate/tracker payloads and a manifest."""

        aggregate = _normalize_aggregate_payload(aggregate_payload)
        tracker = _normalize_tracker_payload(tracker_payload)
        state = _ManifestState()

        transformed_aggregate = self._transform_aggregate(aggregate, state)
        transformed_tracker = self._transform_tracker(tracker, state)
        self._assert_org_units_permitted(transformed_aggregate, "aggregate")
        self._assert_org_units_permitted(transformed_tracker, "tracker")

        return DHIS2ExportResult(
            aggregate_payload=transformed_aggregate,
            tracker_payload=transformed_tracker,
            manifest=state.to_manifest(self.config),
        )

    def _transform_aggregate(
        self,
        payload: dict[str, Any],
        state: _ManifestState,
    ) -> dict[str, Any]:
        transformed_sets: list[dict[str, Any]] = []
        for set_index, raw_set in enumerate(payload["dataValueSets"]):
            path = f"aggregate.dataValueSets[{set_index}]"
            if not isinstance(raw_set, Mapping):
                raise DHIS2ExportError(f"{path} must be a JSON object")
            state.data_value_sets += 1
            shift = self._record_shift(raw_set, record_index=set_index)

            raw_values = raw_set.get("dataValues")
            if not _is_sequence(raw_values):
                raise DHIS2ExportError(f"{path}.dataValues must be an array")
            base = {key: value for key, value in raw_set.items() if key != "dataValues"}
            transformed_set = self._transform_mapping(
                base,
                path=path,
                state=state,
                record_shift=shift,
                aggregate_context=True,
            )

            transformed_values: list[dict[str, Any]] = []
            for value_index, raw_value in enumerate(raw_values):
                value_path = f"{path}.dataValues[{value_index}]"
                if not isinstance(raw_value, Mapping):
                    raise DHIS2ExportError(f"{value_path} must be a JSON object")
                state.aggregate_values_input += 1
                if self._should_suppress(raw_value.get("value")):
                    state.suppressed_aggregate_values += 1
                    state.changed(value_path)
                    continue
                transformed_values.append(
                    self._transform_mapping(
                        raw_value,
                        path=value_path,
                        state=state,
                        record_shift=shift,
                        aggregate_context=True,
                    )
                )

            transformed_values.sort(key=_data_value_sort_key)
            state.aggregate_values_output += len(transformed_values)
            transformed_set["dataValues"] = transformed_values
            transformed_sets.append(transformed_set)

        transformed_sets.sort(key=_data_value_set_sort_key)
        output = {
            key: value for key, value in payload.items() if key != "dataValueSets"
        }
        output["dataValueSets"] = transformed_sets
        return output

    def _transform_tracker(
        self,
        payload: dict[str, Any],
        state: _ManifestState,
    ) -> dict[str, Any]:
        entities: list[dict[str, Any]] = []
        for entity_index, raw_entity in enumerate(payload["trackedEntities"]):
            path = f"tracker.trackedEntities[{entity_index}]"
            if not isinstance(raw_entity, Mapping):
                raise DHIS2ExportError(f"{path} must be a JSON object")
            state.tracked_entities += 1
            entities.append(
                self._transform_mapping(
                    raw_entity,
                    path=path,
                    state=state,
                    record_shift=self._record_shift(
                        raw_entity,
                        record_index=entity_index,
                    ),
                    aggregate_context=False,
                )
            )

        events: list[dict[str, Any]] = []
        for event_index, raw_event in enumerate(payload["events"]):
            path = f"tracker.events[{event_index}]"
            if not isinstance(raw_event, Mapping):
                raise DHIS2ExportError(f"{path} must be a JSON object")
            state.events += 1
            events.append(
                self._transform_mapping(
                    raw_event,
                    path=path,
                    state=state,
                    record_shift=self._record_shift(
                        raw_event,
                        record_index=event_index,
                    ),
                    aggregate_context=False,
                )
            )

        entities.sort(key=_tracked_entity_sort_key)
        events.sort(key=_event_sort_key)
        output = {
            key: value
            for key, value in payload.items()
            if key not in {"trackedEntities", "events"}
        }
        output["trackedEntities"] = entities
        output["events"] = events
        return output

    def _transform_mapping(
        self,
        value: Mapping[str, Any],
        *,
        path: str,
        state: _ManifestState,
        record_shift: int | None,
        aggregate_context: bool,
        redact_value_record: bool = False,
    ) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for key in sorted(value):
            item = value[key]
            child_path = f"{path}.{key}"
            if key in _PRECISE_GEOGRAPHY_FIELDS:
                if item is not None:
                    state.precise_locations_removed += 1
                    state.changed(child_path)
                continue
            if key == "orgUnit":
                output[key] = self._generalize_org_unit(
                    item,
                    path=child_path,
                    state=state,
                )
            elif key in _TEXT_FIELDS:
                output[key] = self._redact_text_value(
                    item,
                    path=child_path,
                    state=state,
                )
            elif key == "value" and redact_value_record:
                output[key] = self._redact_text_value(
                    item,
                    path=child_path,
                    state=state,
                )
            elif key in _DATE_FIELDS:
                output[key] = self._transform_date(
                    item,
                    path=child_path,
                    state=state,
                    record_shift=record_shift,
                )
            elif key == "period" and aggregate_context:
                output[key] = self._transform_period(
                    item,
                    path=child_path,
                    state=state,
                )
            elif key == "attributes":
                output[key] = self._transform_special_list(
                    item,
                    path=child_path,
                    state=state,
                    record_shift=record_shift,
                    aggregate_context=aggregate_context,
                    redact_value_records=True,
                )
                output[key].sort(key=_attribute_sort_key)
            elif key == "notes":
                output[key] = self._transform_special_list(
                    item,
                    path=child_path,
                    state=state,
                    record_shift=record_shift,
                    aggregate_context=aggregate_context,
                    redact_value_records=True,
                )
            elif key == "dataValues":
                output[key] = self._transform_special_list(
                    item,
                    path=child_path,
                    state=state,
                    record_shift=record_shift,
                    aggregate_context=aggregate_context,
                    redact_value_records=not aggregate_context,
                )
                output[key].sort(key=_data_value_sort_key)
            else:
                output[key] = self._transform_nested(
                    item,
                    path=child_path,
                    state=state,
                    record_shift=record_shift,
                    aggregate_context=aggregate_context,
                )
        return output

    def _transform_special_list(
        self,
        value: Any,
        *,
        path: str,
        state: _ManifestState,
        record_shift: int | None,
        aggregate_context: bool,
        redact_value_records: bool,
    ) -> list[dict[str, Any]]:
        if not _is_sequence(value):
            raise DHIS2ExportError(f"{path} must be an array")
        output: list[dict[str, Any]] = []
        for index, item in enumerate(value):
            item_path = f"{path}[{index}]"
            if not isinstance(item, Mapping):
                raise DHIS2ExportError(f"{item_path} must be a JSON object")
            output.append(
                self._transform_mapping(
                    item,
                    path=item_path,
                    state=state,
                    record_shift=record_shift,
                    aggregate_context=aggregate_context,
                    redact_value_record=redact_value_records,
                )
            )
        return output

    def _transform_nested(
        self,
        value: Any,
        *,
        path: str,
        state: _ManifestState,
        record_shift: int | None,
        aggregate_context: bool,
    ) -> Any:
        if isinstance(value, Mapping):
            return self._transform_mapping(
                value,
                path=path,
                state=state,
                record_shift=record_shift,
                aggregate_context=aggregate_context,
            )
        if _is_sequence(value):
            return [
                self._transform_nested(
                    item,
                    path=f"{path}[{index}]",
                    state=state,
                    record_shift=record_shift,
                    aggregate_context=aggregate_context,
                )
                for index, item in enumerate(value)
            ]
        return value

    def _generalize_org_unit(
        self,
        value: Any,
        *,
        path: str,
        state: _ManifestState,
    ) -> str:
        if not isinstance(value, str) or not value:
            raise DHIS2ExportError(f"{path} must be a non-empty string")
        state.org_units_examined += 1
        generalized = self.hierarchy.generalize(
            value,
            target_level=self.config.generalization_level,
            path=path,
        )
        if generalized != value:
            state.org_units_generalized += 1
            state.changed(path)
        return generalized

    def _redact_text_value(
        self,
        value: Any,
        *,
        path: str,
        state: _ManifestState,
    ) -> Any:
        if value is None:
            return value
        if not isinstance(value, str):
            raise DHIS2ExportError(f"{path} must be a string or null")
        if not value:
            return value
        state.text_values_examined += 1
        try:
            redacted = _redacted_text(self._text_redactor(value))
        except Exception:
            raise DHIS2ExportError(
                f"text de-identification failed at {path}; raw value omitted"
            ) from None
        if redacted != value:
            state.text_values_redacted += 1
            state.changed(path)
        return redacted

    def _transform_date(
        self,
        value: Any,
        *,
        path: str,
        state: _ManifestState,
        record_shift: int | None,
    ) -> Any:
        if value is None:
            return value
        if not isinstance(value, str):
            raise DHIS2ExportError(f"{path} must be a string")
        state.dates_examined += 1
        if self.config.date_mode == "none":
            return value
        if self.config.date_mode == "shift":
            if record_shift is None:
                raise DHIS2ExportError("date shift policy did not resolve an offset")
            transformed = _shift_iso_date(
                value,
                shift_days=record_shift,
                keep_year=self.config.keep_year,
                lang=self.config.lang,
                path=path,
            )
        else:
            transformed = _coarsen_iso_date(
                value,
                granularity=self.config.period_granularity,
                path=path,
            )
        if transformed != value:
            state.dates_transformed += 1
            state.changed(path)
        return transformed

    def _transform_period(
        self,
        value: Any,
        *,
        path: str,
        state: _ManifestState,
    ) -> Any:
        if value is None:
            return value
        if not isinstance(value, str):
            raise DHIS2ExportError(f"{path} must be a string")
        state.periods_examined += 1
        if self.config.date_mode != "coarsen":
            return value
        transformed = _coarsen_period(
            value,
            granularity=self.config.period_granularity,
            path=path,
        )
        if transformed != value:
            state.periods_coarsened += 1
            state.changed(path)
        return transformed

    def _record_shift(
        self,
        record: Mapping[str, Any],
        *,
        record_index: int,
    ) -> int | None:
        if self.config.date_mode != "shift":
            return None
        from openmed.multimodal.tabular_csv import derive_date_shift_days

        return derive_date_shift_days(
            [_canonical_json(record)],
            record_index=record_index,
            fixed_days=self.config.date_shift_days,
            seed=self.config.date_shift_seed,
        )

    def _should_suppress(self, value: Any) -> bool:
        threshold = self.config.small_cell_threshold
        if threshold in {None, 0} or isinstance(value, bool):
            return False
        try:
            numeric = Decimal(str(value))
        except (InvalidOperation, ValueError):
            return False
        return numeric.is_finite() and Decimal(0) <= numeric < Decimal(threshold)

    def _default_text_redactor(self, value: str) -> Any:
        from openmed.core.pii import deidentify

        return deidentify(
            value,
            method="mask",
            lang=self.config.lang,
            policy=self.config.policy,
        )

    def _assert_org_units_permitted(self, value: Any, path: str) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                child_path = f"{path}.{key}"
                if key == "orgUnit":
                    if not isinstance(item, str):
                        raise DHIS2ExportError(f"{child_path} must be a string")
                    self.hierarchy.assert_permitted(
                        item,
                        target_level=self.config.generalization_level,
                        path=child_path,
                    )
                else:
                    self._assert_org_units_permitted(item, child_path)
        elif _is_sequence(value):
            for index, item in enumerate(value):
                self._assert_org_units_permitted(item, f"{path}[{index}]")


def export_dhis2(
    aggregate_payload: Any = None,
    tracker_payload: Any = None,
    org_unit_snapshot: Any = None,
    *,
    config: DHIS2ExportConfig | None = None,
    text_redactor: TextRedactor | None = None,
    generalization_level: int | None = None,
    small_cell_threshold: int | None = _UNSET,
    date_mode: DateMode | None = None,
    date_shift_days: int | None = None,
    period_granularity: PeriodGranularity | None = None,
) -> DHIS2ExportResult:
    """Export aggregate and tracker payloads without any network calls.

    The optional scalar policy arguments override the corresponding values in
    ``config``. Pass ``DHIS2ExportConfig(small_cell_threshold=None)`` to disable
    suppression; the convenience value ``0`` also disables it.

    Args:
        aggregate_payload: A data-value-set object, array, or envelope.
        tracker_payload: An envelope with trackedEntities and/or events arrays.
        org_unit_snapshot: Local ``organisationUnits`` JSON data, path, or stream.
        config: Optional complete exporter policy.
        text_redactor: Optional deterministic local de-identification callable.
        generalization_level: Optional level override.
        small_cell_threshold: Optional suppression-threshold override.
        date_mode: Optional date-policy override.
        date_shift_days: Optional fixed date-shift override.
        period_granularity: Optional coarsening-precision override.

    Returns:
        Endpoint-shaped payloads plus a PHI-free manifest.
    """
    if org_unit_snapshot is None:
        raise TypeError("org_unit_snapshot is required")
    resolved = config or DHIS2ExportConfig()
    updates: dict[str, Any] = {}
    if generalization_level is not None:
        updates["generalization_level"] = generalization_level
    if small_cell_threshold is not _UNSET:
        updates["small_cell_threshold"] = small_cell_threshold
    if date_mode is not None:
        updates["date_mode"] = date_mode
    if date_shift_days is not None:
        updates["date_shift_days"] = date_shift_days
    if period_granularity is not None:
        updates["period_granularity"] = period_granularity
    if updates:
        resolved = replace(resolved, **updates)

    return DHIS2Exporter(
        org_unit_snapshot,
        config=resolved,
        text_redactor=text_redactor,
    ).export(aggregate_payload, tracker_payload)


def _load_json_source(source: Any, *, source_name: str) -> Any:
    if isinstance(source, (Mapping, list, tuple)):
        return _json_clone(source, source_name=source_name)
    if hasattr(source, "read"):
        try:
            return json.load(source)
        except (TypeError, json.JSONDecodeError) as exc:
            raise DHIS2ExportError(f"{source_name} is not valid JSON") from exc
    if isinstance(source, Path):
        return _load_json_path(source, source_name=source_name)
    if isinstance(source, str):
        stripped = source.lstrip()
        if stripped.startswith(("{", "[")):
            try:
                return json.loads(source)
            except json.JSONDecodeError as exc:
                raise DHIS2ExportError(f"{source_name} is not valid JSON") from exc
        return _load_json_path(Path(source), source_name=source_name)
    raise TypeError(f"{source_name} must be JSON data, a local path, or a stream")


def _load_json_path(path: Path, *, source_name: str) -> Any:
    try:
        with path.open("r", encoding="utf-8") as stream:
            return json.load(stream)
    except OSError as exc:
        raise DHIS2ExportError(f"could not read local {source_name}") from exc
    except json.JSONDecodeError as exc:
        raise DHIS2ExportError(f"{source_name} is not valid JSON") from exc


def _json_clone(value: Any, *, source_name: str) -> Any:
    try:
        return json.loads(json.dumps(value, ensure_ascii=False))
    except (TypeError, ValueError) as exc:
        raise DHIS2ExportError(
            f"{source_name} must contain JSON-compatible data"
        ) from exc


def _normalize_aggregate_payload(payload: Any) -> dict[str, Any]:
    if payload is None:
        return {"dataValueSets": []}
    cloned = _json_clone(payload, source_name="aggregate payload")
    if isinstance(cloned, Mapping):
        if "dataValueSets" in cloned:
            if not _is_sequence(cloned["dataValueSets"]):
                raise DHIS2ExportError("aggregate.dataValueSets must be an array")
            return dict(cloned)
        if "dataValues" in cloned:
            return {"dataValueSets": [dict(cloned)]}
    if _is_sequence(cloned):
        return {"dataValueSets": list(cloned)}
    raise DHIS2ExportError(
        "aggregate payload must be a dataValueSet object, array, or envelope"
    )


def _normalize_tracker_payload(payload: Any) -> dict[str, Any]:
    if payload is None:
        return {"trackedEntities": [], "events": []}
    cloned = _json_clone(payload, source_name="tracker payload")
    if not isinstance(cloned, Mapping):
        raise DHIS2ExportError("tracker payload must be a JSON object")
    if not {"trackedEntities", "events"}.intersection(cloned):
        raise DHIS2ExportError("tracker payload must contain trackedEntities or events")
    output = dict(cloned)
    output.setdefault("trackedEntities", [])
    output.setdefault("events", [])
    if not _is_sequence(output["trackedEntities"]):
        raise DHIS2ExportError("tracker.trackedEntities must be an array")
    if not _is_sequence(output["events"]):
        raise DHIS2ExportError("tracker.events must be an array")
    return output


def _parent_uid(value: Any, *, index: int) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        if value:
            return value
    elif isinstance(value, Mapping):
        uid = value.get("id", value.get("uid"))
        if isinstance(uid, str) and uid:
            return uid
    raise DHIS2ExportError(
        f"organisationUnits[{index}].parent must contain a non-empty id"
    )


def _redacted_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        candidate = value.get("deidentified_text")
    else:
        candidate = getattr(value, "deidentified_text", None)
    if not isinstance(candidate, str):
        raise TypeError("text redactor must return text or deidentified_text")
    return candidate


def _shift_iso_date(
    value: str,
    *,
    shift_days: int,
    keep_year: bool,
    lang: str,
    path: str,
) -> str:
    match = _ISO_DATE_RE.fullmatch(value)
    if match is None:
        raise DHIS2ExportError(f"{path} must start with an ISO YYYY-MM-DD date")
    source_date = match.group("date")
    suffix = match.group("suffix") or ""
    try:
        date.fromisoformat(source_date)
        if suffix:
            datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise DHIS2ExportError(f"{path} contains an invalid ISO date") from exc

    from openmed.multimodal.tabular_csv import shift_quasi_identifier_date

    shifted = shift_quasi_identifier_date(
        source_date,
        shift_days=shift_days,
        keep_year=keep_year,
        lang=lang,
    )
    if shifted.startswith("["):
        raise DHIS2ExportError(f"{path} could not be shifted safely")
    return shifted + suffix


def _coarsen_iso_date(
    value: str,
    *,
    granularity: PeriodGranularity,
    path: str,
) -> str:
    match = _ISO_DATE_RE.fullmatch(value)
    if match is None:
        raise DHIS2ExportError(f"{path} must start with an ISO YYYY-MM-DD date")
    source_date = match.group("date")
    suffix = match.group("suffix") or ""
    try:
        parsed = date.fromisoformat(source_date)
        if suffix:
            datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise DHIS2ExportError(f"{path} contains an invalid ISO date") from exc
    if granularity == "year":
        coarsened = parsed.replace(month=1, day=1)
    else:
        coarsened = parsed.replace(day=1)
    if suffix.startswith("T"):
        timezone_match = re.search(r"(Z|[+-]\d{2}:?\d{2})$", suffix)
        timezone = timezone_match.group(1) if timezone_match else ""
        suffix = f"T00:00:00{timezone}"
    return coarsened.isoformat() + suffix


def _coarsen_period(
    value: str,
    *,
    granularity: PeriodGranularity,
    path: str,
) -> str:
    if not re.match(r"^\d{4}", value):
        raise DHIS2ExportError(f"{path} must start with a four-digit year")
    if granularity == "year":
        return value[:4]
    if re.fullmatch(r"\d{8}", value):
        try:
            date.fromisoformat(f"{value[:4]}-{value[4:6]}-{value[6:8]}")
        except ValueError as exc:
            raise DHIS2ExportError(f"{path} contains an invalid daily period") from exc
        return value[:6]
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        try:
            parsed = date.fromisoformat(value)
        except ValueError as exc:
            raise DHIS2ExportError(f"{path} contains an invalid daily period") from exc
        return f"{parsed.year:04d}{parsed.month:02d}"
    weekly = _WEEKLY_PERIOD_RE.fullmatch(value)
    if weekly is not None:
        try:
            monday = date.fromisocalendar(int(weekly.group(1)), int(weekly.group(2)), 1)
        except ValueError as exc:
            raise DHIS2ExportError(f"{path} contains an invalid weekly period") from exc
        return f"{monday.year:04d}{monday.month:02d}"
    if re.fullmatch(r"\d{6}", value):
        month = int(value[4:6])
        if not 1 <= month <= 12:
            raise DHIS2ExportError(f"{path} contains an invalid monthly period")
        return value
    if re.fullmatch(r"\d{4}(?:Q[1-4]|S[1-2])", value) or re.fullmatch(r"\d{4}", value):
        return value
    raise DHIS2ExportError(f"{path} uses an unsupported DHIS2 period format")


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sort_key(value: Mapping[str, Any], fields: Sequence[str]) -> tuple[str, ...]:
    return tuple(str(value.get(field, "")) for field in fields) + (
        _canonical_json(value),
    )


def _data_value_set_sort_key(value: Mapping[str, Any]) -> tuple[str, ...]:
    return _sort_key(value, ("dataSet", "period", "orgUnit", "completeDate"))


def _data_value_sort_key(value: Mapping[str, Any]) -> tuple[str, ...]:
    return _sort_key(
        value,
        ("dataElement", "categoryOptionCombo", "attributeOptionCombo"),
    )


def _tracked_entity_sort_key(value: Mapping[str, Any]) -> tuple[str, ...]:
    return _sort_key(value, ("trackedEntity", "trackedEntityType", "orgUnit"))


def _event_sort_key(value: Mapping[str, Any]) -> tuple[str, ...]:
    return _sort_key(
        value,
        ("event", "program", "programStage", "occurredAt", "orgUnit"),
    )


def _attribute_sort_key(value: Mapping[str, Any]) -> tuple[str, ...]:
    return _sort_key(value, ("attribute",))


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    )


__all__ = [
    "DEFAULT_GENERALIZATION_LEVEL",
    "DEFAULT_SMALL_CELL_THRESHOLD",
    "DHIS2ExportConfig",
    "DHIS2ExportError",
    "DHIS2ExportResult",
    "DHIS2Exporter",
    "OrgUnitHierarchy",
    "export_dhis2",
]
