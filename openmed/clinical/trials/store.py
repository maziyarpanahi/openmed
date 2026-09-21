"""Integrity-checked local cache for versioned public trial records."""

from __future__ import annotations

import json
import os
import tempfile
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

from openmed.clinical.journey_contracts import canonical_digest, canonical_json

from .contracts import (
    TRIAL_COMPATIBILITY_POLICY,
    TRIAL_SCHEMA_VERSION,
    TrialCacheCorruptionError,
    TrialContractError,
    TrialStudyRecord,
    TrialUnsupportedError,
)
from .source import TrialSourcePage

_CACHE_FILE: Final = "trial-study-cache.json"


@dataclass(frozen=True, slots=True)
class TrialQuery:
    """Offline metadata filters; no patient attributes are accepted."""

    statuses: tuple[str, ...] = ()
    conditions: tuple[str, ...] = ()
    countries: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in ("statuses", "conditions", "countries"):
            values = getattr(self, name)
            if any(not isinstance(item, str) or not item.strip() for item in values):
                raise TrialContractError(f"{name} must contain non-empty text")
            object.__setattr__(self, name, tuple(sorted(set(values))))


@dataclass(frozen=True, slots=True)
class TrialSyncReport:
    """Value-free counts and cursor custody for one applied source page."""

    created_versions: int
    unchanged_studies: int
    total_studies: int
    response_digest: str
    next_page_token: str | None


class LocalTrialStore:
    """Atomic append-only study-version cache with deterministic offline queries."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.path = self.root / _CACHE_FILE
        self._lock = threading.RLock()
        try:
            self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
            os.chmod(self.root, 0o700)
        except OSError as exc:
            raise TrialCacheCorruptionError(
                "trial cache cannot be initialized"
            ) from exc

    def apply_page(self, page: TrialSourcePage) -> TrialSyncReport:
        """Append only changed study versions from one fetched public page."""

        if not isinstance(page, TrialSourcePage):
            raise TypeError("page must be TrialSourcePage")
        with self._lock:
            records = list(self._read_records())
            by_study: dict[str, list[TrialStudyRecord]] = {}
            for record in records:
                by_study.setdefault(record.study_id, []).append(record)
            created = 0
            unchanged = 0
            for record in page.studies:
                history = by_study.setdefault(record.study_id, [])
                if any(item.version_id == record.version_id for item in history):
                    unchanged += 1
                    continue
                if history and _time_key(record.retrieved_at) < _time_key(
                    history[-1].retrieved_at
                ):
                    raise TrialContractError(
                        "trial retrieval time precedes cached history"
                    )
                history.append(record)
                records.append(record)
                created += 1
            records.sort(key=_record_key)
            if created:
                self._write_records(tuple(records))
            return TrialSyncReport(
                created_versions=created,
                unchanged_studies=unchanged,
                total_studies=len(page.studies),
                response_digest=page.response_digest,
                next_page_token=page.next_page_token,
            )

    def history(self, study_id: str) -> tuple[TrialStudyRecord, ...]:
        """Return verified study history in retrieval order."""

        return tuple(item for item in self._read_records() if item.study_id == study_id)

    def latest(self, study_id: str) -> TrialStudyRecord | None:
        """Return the latest verified version of one study."""

        history = self.history(study_id)
        return history[-1] if history else None

    def query(self, query: TrialQuery | None = None) -> tuple[TrialStudyRecord, ...]:
        """Query latest public study versions without network access."""

        selected = query or TrialQuery()
        latest: dict[str, TrialStudyRecord] = {}
        for record in self._read_records():
            latest[record.study_id] = record
        statuses = {item.casefold() for item in selected.statuses}
        conditions = {item.casefold() for item in selected.conditions}
        countries = {item.casefold() for item in selected.countries}
        output = []
        for record in latest.values():
            record_conditions = {item.casefold() for item in record.conditions}
            record_countries = {
                item.country.casefold()
                for item in record.locations
                if item.country is not None
            }
            if statuses and record.overall_status.casefold() not in statuses:
                continue
            if conditions and not conditions.intersection(record_conditions):
                continue
            if countries and not countries.intersection(record_countries):
                continue
            output.append(record)
        return tuple(sorted(output, key=lambda item: item.study_id))

    def _read_records(self) -> tuple[TrialStudyRecord, ...]:
        if not self.path.exists():
            return ()
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise TrialCacheCorruptionError("trial cache is unreadable") from exc
        if not isinstance(data, dict) or set(data) != {
            "cache_digest",
            "compatibility_policy",
            "records",
            "schema_version",
        }:
            raise TrialCacheCorruptionError("trial cache envelope is invalid")
        if data["compatibility_policy"] != TRIAL_COMPATIBILITY_POLICY:
            raise TrialUnsupportedError("unsupported trial cache compatibility policy")
        if not isinstance(data["schema_version"], str) or not data[
            "schema_version"
        ].startswith(f"{TRIAL_SCHEMA_VERSION.split('.', 1)[0]}."):
            raise TrialUnsupportedError("unsupported trial cache schema version")
        payload = {key: value for key, value in data.items() if key != "cache_digest"}
        if data["cache_digest"] != canonical_digest(payload):
            raise TrialCacheCorruptionError("trial cache digest does not match content")
        if not isinstance(data["records"], list):
            raise TrialCacheCorruptionError("trial cache records changed type")
        try:
            records = tuple(
                TrialStudyRecord.from_dict(item) for item in data["records"]
            )
        except TrialUnsupportedError:
            raise
        except (TypeError, TrialContractError) as exc:
            raise TrialCacheCorruptionError(
                "trial cache contains an invalid record"
            ) from exc
        expected = tuple(sorted(records, key=_record_key))
        if records != expected or len({item.version_id for item in records}) != len(
            records
        ):
            raise TrialCacheCorruptionError(
                "trial cache history order or uniqueness is invalid"
            )
        return records

    def _write_records(self, records: tuple[TrialStudyRecord, ...]) -> None:
        payload: dict[str, Any] = {
            "compatibility_policy": TRIAL_COMPATIBILITY_POLICY,
            "records": [item.to_dict() for item in records],
            "schema_version": TRIAL_SCHEMA_VERSION,
        }
        envelope = {"cache_digest": canonical_digest(payload), **payload}
        descriptor = -1
        temp_name = ""
        try:
            descriptor, temp_name = tempfile.mkstemp(
                prefix=".trial-cache-", dir=self.root
            )
            os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                descriptor = -1
                stream.write(canonical_json(envelope))
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp_name, self.path)
            os.chmod(self.path, 0o600)
        except OSError as exc:
            raise TrialCacheCorruptionError("trial cache cannot be written") from exc
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            if temp_name:
                Path(temp_name).unlink(missing_ok=True)


def _time_key(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def _record_key(record: TrialStudyRecord) -> tuple[str, datetime, str]:
    return (record.study_id, _time_key(record.retrieved_at), record.version_id)
