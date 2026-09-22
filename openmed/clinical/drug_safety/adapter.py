"""Versioned adapter for optional user-supplied open event datasets."""

from __future__ import annotations

import csv
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from openmed.clinical.journey_contracts import (
    canonical_digest,
    derived_opaque_id,
    sha256_digest,
)

from .contracts import (
    DrugExposure,
    DrugSafetyConflictError,
    DrugSafetyContractError,
    DrugSafetyDataset,
    DrugSafetyDatasetManifest,
    DrugSafetyEvidence,
    ExposureWindow,
    NormalizedSafetyTerm,
    SafetyCase,
    SafetySeriousness,
)

DRUG_SAFETY_ADAPTER_VERSION: Final = "openmed.open-event/1.0.0"


@dataclass(frozen=True, slots=True)
class OpenEventColumnMap:
    """Explicit columns used by the generic open-event adapter."""

    report_id: str = "report_id"
    drug: str = "drug"
    event: str = "event"
    seriousness: str = "seriousness"
    exposure_start_day: str = "exposure_start_day"
    exposure_end_day: str = "exposure_end_day"

    def __post_init__(self) -> None:
        values = tuple(getattr(self, name) for name in self.__dataclass_fields__)
        if any(not isinstance(item, str) or not item.strip() for item in values):
            raise DrugSafetyContractError("column names must be non-empty text")
        if len(values) != len(set(values)):
            raise DrugSafetyContractError("open-event column names must be unique")

    def required(self) -> tuple[str, ...]:
        """Return the exact required source columns."""

        return tuple(getattr(self, name) for name in self.__dataclass_fields__)


@dataclass(slots=True)
class _CaseBuilder:
    seriousness: SafetySeriousness
    exposures: dict[tuple[str, int | None, int | None], DrugExposure]
    events: dict[str, NormalizedSafetyTerm]
    evidence: dict[str, DrugSafetyEvidence]


class OpenEventDatasetAdapter:
    """Normalize caller-supplied public rows without retaining source report IDs."""

    def __init__(self, columns: OpenEventColumnMap | None = None) -> None:
        self.columns = columns or OpenEventColumnMap()

    def import_rows(
        self,
        rows: Iterable[Mapping[str, Any]],
        *,
        dataset_id: str,
        version: str,
        source_digest: str,
        license_id: str,
    ) -> DrugSafetyDataset:
        """Import already-opened public rows through an explicit local boundary."""

        manifest = DrugSafetyDatasetManifest(
            dataset_id=dataset_id,
            version=version,
            source_digest=source_digest,
            license_id=license_id,
            adapter_version=DRUG_SAFETY_ADAPTER_VERSION,
        )
        builders: dict[str, _CaseBuilder] = {}
        seen_rows: set[str] = set()
        duplicate_count = 0
        imported_count = 0
        required = set(self.columns.required())
        for raw in rows:
            imported_count += 1
            if not isinstance(raw, Mapping) or any(
                not isinstance(key, str) for key in raw
            ):
                raise DrugSafetyContractError("open-event row must be an object")
            if not required.issubset(raw):
                raise DrugSafetyContractError(
                    "open-event row is missing required columns"
                )
            normalized = self._normalize_row(raw, source_digest=source_digest)
            row_digest = normalized["evidence"].row_digest
            if row_digest in seen_rows:
                duplicate_count += 1
                continue
            seen_rows.add(row_digest)
            source_report_id = normalized["source_report_id"]
            builder = builders.get(source_report_id)
            if builder is None:
                builder = _CaseBuilder(
                    seriousness=normalized["seriousness"],
                    exposures={},
                    events={},
                    evidence={},
                )
                builders[source_report_id] = builder
            elif builder.seriousness is not normalized["seriousness"]:
                if SafetySeriousness.UNKNOWN not in {
                    builder.seriousness,
                    normalized["seriousness"],
                }:
                    raise DrugSafetyConflictError(
                        "source report has conflicting seriousness"
                    )
                builder.seriousness = (
                    normalized["seriousness"]
                    if builder.seriousness is SafetySeriousness.UNKNOWN
                    else builder.seriousness
                )
            exposure = normalized["exposure"]
            window = exposure.window
            exposure_key = (
                exposure.drug.code,
                window.start_day if window else None,
                window.end_day if window else None,
            )
            builder.exposures[exposure_key] = exposure
            event = normalized["event"]
            builder.events[event.code] = event
            evidence = normalized["evidence"]
            builder.evidence[evidence.row_digest] = evidence
        cases = tuple(
            SafetyCase(
                case_id=derived_opaque_id(
                    "safetycase", manifest.digest, source_report_id
                ),
                exposures=tuple(builder.exposures.values()),
                events=tuple(builder.events.values()),
                seriousness=builder.seriousness,
                evidence=tuple(builder.evidence.values()),
            )
            for source_report_id, builder in sorted(builders.items())
        )
        return DrugSafetyDataset(
            manifest=manifest,
            cases=cases,
            duplicate_row_count=duplicate_count,
            imported_row_count=imported_count,
        )

    def _normalize_row(
        self, raw: Mapping[str, Any], *, source_digest: str
    ) -> dict[str, Any]:
        report_id = _source_text(raw[self.columns.report_id], "report_id")
        drug = NormalizedSafetyTerm(
            kind="drug", code=_source_text(raw[self.columns.drug], "drug")
        )
        event = NormalizedSafetyTerm(
            kind="event", code=_source_text(raw[self.columns.event], "event")
        )
        seriousness = _parse_seriousness(raw[self.columns.seriousness])
        start = _optional_integer(
            raw[self.columns.exposure_start_day], "exposure_start_day"
        )
        end = _optional_integer(raw[self.columns.exposure_end_day], "exposure_end_day")
        if (start is None) != (end is None):
            raise DrugSafetyContractError(
                "exposure start and end must both be supplied or both be empty"
            )
        if start is None:
            window = None
        else:
            assert end is not None
            window = ExposureWindow(start_day=start, end_day=end)
        row_payload = {
            "drug": drug.to_dict(),
            "event": event.to_dict(),
            "exposure_window": window.to_dict() if window else None,
            "report_key_digest": canonical_digest({"report_id": report_id}),
            "seriousness": seriousness.value,
        }
        return {
            "event": event,
            "evidence": DrugSafetyEvidence(
                row_digest=canonical_digest(row_payload),
                source_digest=source_digest,
            ),
            "exposure": DrugExposure(drug=drug, window=window),
            "seriousness": seriousness,
            "source_report_id": report_id,
        }


def import_open_event_csv(
    path: str | Path,
    *,
    dataset_id: str,
    version: str,
    source_digest: str,
    license_id: str,
    columns: OpenEventColumnMap | None = None,
) -> DrugSafetyDataset:
    """Import and digest-verify a caller-supplied local CSV dataset."""

    source = Path(path)
    try:
        content = source.read_bytes()
    except OSError as exc:
        raise DrugSafetyContractError("open-event CSV cannot be read") from exc
    if sha256_digest(content) != source_digest:
        raise DrugSafetyConflictError("open-event source digest does not match")
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise DrugSafetyContractError("open-event CSV must be UTF-8") from exc
    reader = csv.DictReader(text.splitlines())
    if reader.fieldnames is None:
        raise DrugSafetyContractError("open-event CSV is missing a header")
    if len(reader.fieldnames) != len(set(reader.fieldnames)):
        raise DrugSafetyContractError("open-event CSV header contains duplicates")
    active_columns = columns or OpenEventColumnMap()
    if not set(active_columns.required()).issubset(reader.fieldnames):
        raise DrugSafetyContractError("open-event CSV is missing required columns")
    return OpenEventDatasetAdapter(active_columns).import_rows(
        reader,
        dataset_id=dataset_id,
        version=version,
        source_digest=source_digest,
        license_id=license_id,
    )


def _source_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DrugSafetyContractError(f"{name} must be non-empty text")
    if len(value.encode("utf-8")) > 4096:
        raise DrugSafetyContractError(f"{name} exceeds the byte limit")
    return value.strip()


def _optional_integer(value: Any, name: str) -> int | None:
    if value is None or value == "":
        return None
    if type(value) is int:
        return value
    if not isinstance(value, str) or re.fullmatch(r"[+-]?[0-9]+", value.strip()) is None:
        raise DrugSafetyContractError(f"{name} must be an integer")
    try:
        return int(value.strip())
    except (TypeError, ValueError):
        raise DrugSafetyContractError(f"{name} must be an integer") from None


def _parse_seriousness(value: Any) -> SafetySeriousness:
    if value is None or value == "":
        return SafetySeriousness.UNKNOWN
    normalized = _source_text(value, "seriousness").casefold().replace("-", "_")
    aliases = {
        "yes": SafetySeriousness.SERIOUS,
        "true": SafetySeriousness.SERIOUS,
        "no": SafetySeriousness.NON_SERIOUS,
        "false": SafetySeriousness.NON_SERIOUS,
        "not_serious": SafetySeriousness.NON_SERIOUS,
    }
    if normalized in aliases:
        return aliases[normalized]
    try:
        return SafetySeriousness(normalized)
    except ValueError:
        raise DrugSafetyContractError("seriousness value is unsupported") from None
