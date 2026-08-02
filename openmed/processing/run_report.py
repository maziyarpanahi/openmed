"""Operator-facing reports for distributed batch runs, free of record content.

A run manifest and a resume plan are internal structures: they are consumed by
the driver that owns them, and their fields are chosen for that job. A report is
the artifact an operator reads, files, and pastes into a ticket, so it is held to
a stricter standard than either of its sources.

Two fields illustrate why this layer re-derives rather than forwards. The
manifest records ``worker_id`` verbatim -- its validation bounds length and
rejects control characters, which does nothing to a host named after the ward it
serves -- so this module publishes only
:func:`~openmed.processing.resume.worker_ref`. The manifest also records
``output_path``; a relative path with no traversal is safe to *store*, but a
filename is chosen by whoever configured the run and is not worth publishing, so
no path reaches a report at all.

Every payload is validated against an allowlist before it is returned, and the
check covers mapping keys as well as values: a sentence lifted out of a record
reaches a payload just as easily as a key as it does as a value, and a check that
inspects only values reads as thorough while missing half the surface.
"""

from __future__ import annotations

import json
import re
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .resume import ResumePlan, worker_ref
from .run_manifest import (
    BatchRunManifest,
    ShardOutputValidation,
    ShardRecord,
    ShardStatus,
)

#: Schema version of the report document produced by this module.
RUN_REPORT_SCHEMA_VERSION = 1

#: Substituted for an exception type name that is not a valid identifier.
UNRECOGNIZED_ERROR_TYPE = "UnrecognizedError"

#: Every shard finished with a usable output.
RUN_STATE_COMPLETE = "complete"
#: The run stopped with shards that spent their attempt budget.
RUN_STATE_EXHAUSTED = "exhausted"
#: Work remains and no shard has exhausted its attempts.
RUN_STATE_IN_PROGRESS = "in_progress"

# A publishable token: an identifier, enum value, digest, timestamp or hashed
# reference. Deliberately excludes whitespace, so no prose can pass, and the
# markdown metacharacter "|", so no value can forge a table row.
_SAFE_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:+/-]{0,127}")

# Shard and plan fingerprints are bare hex; recorded output digests carry a
# "sha256:" prefix. They are not interchangeable.
_HEX_DIGEST = re.compile(r"[0-9a-f]{64}")
_PREFIXED_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")

_ERROR_TYPE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*")

# Names that must never appear in a report, whatever they happen to hold. The
# denylist makes the omissions structural rather than a property of the current
# builder: a later edit that forwards one of these fails immediately.
_FORBIDDEN_KEYS = frozenset(
    {
        "content",
        "document",
        "document_id",
        "document_ids",
        "documents",
        "message",
        "note",
        "notes",
        "output_path",
        "payload",
        "source",
        "text",
        "traceback",
        "worker_id",
    }
)


class RunReportError(ValueError):
    """Base error raised when a run report cannot be produced safely."""


class RunReportPrivacyError(RunReportError):
    """Raised when a payload would publish something a report may not carry."""


def assert_no_raw_text(payload: Any, *, where: str = "batch run report") -> None:
    """Assert ``payload`` carries only counters, timings, enums and digests.

    Mapping keys are checked with the same allowlist as values. Numbers,
    booleans and ``None`` are always allowed; any other type is rejected rather
    than coerced, because coercion is what turns an unexpected object into a
    string that reproduces its contents.

    Raised messages name the offending path and never quote the offending value.
    An error message is itself an operator-visible surface, and the CLI renders
    :class:`~openmed.cli._output.CliError` text verbatim, so echoing the value
    here would reintroduce the leak the check exists to prevent.
    """

    _walk(payload, where=where, path="<root>")


def _walk(value: Any, *, where: str, path: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise RunReportPrivacyError(
                    f"{where}: non-string mapping key at {path}"
                )
            if key in _FORBIDDEN_KEYS:
                raise RunReportPrivacyError(f"{where}: forbidden key {key!r} at {path}")
            if _SAFE_TOKEN.fullmatch(key) is None:
                raise RunReportPrivacyError(
                    f"{where}: mapping key is not a publishable token at {path}"
                )
            _walk(item, where=where, path=f"{path}.{key}")
        return

    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _walk(item, where=where, path=f"{path}[{index}]")
        return

    if value is None or isinstance(value, (bool, int, float)):
        return

    if isinstance(value, str):
        if _SAFE_TOKEN.fullmatch(value) is None:
            raise RunReportPrivacyError(
                f"{where}: string value is not a publishable token at {path}"
            )
        return

    raise RunReportPrivacyError(
        f"{where}: unsupported value type {type(value).__name__} at {path}"
    )


def _safe_error_type(value: str | None) -> str | None:
    """Return a renderable exception type name, or a constant when malformed.

    The manifest validates ``error_type`` on the way in, but validation raises
    rather than repairs, and a worker can report a class whose name is not an
    identifier. Anything unrecognised becomes a constant so that a malformed
    name cannot reach a rendered table.
    """

    if value is None:
        return None
    if not isinstance(value, str) or _ERROR_TYPE.fullmatch(value) is None:
        return UNRECOGNIZED_ERROR_TYPE
    return value


def _checked_digest(
    value: str | None,
    *,
    field_name: str,
    pattern: re.Pattern[str],
) -> str | None:
    """Return ``value`` when it has the exact digest shape, else fail closed.

    A digest is the one field whose shape is known precisely, which makes it the
    one place this layer can independently catch a value that was coerced into a
    string somewhere upstream: ``"None"`` is a plausible-looking token but not a
    plausible-looking digest.
    """

    if value is None:
        return None
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise RunReportPrivacyError(f"{field_name} is not a sha256 digest")
    return value


def _duration_seconds(record: ShardRecord) -> float | None:
    if record.started_at is None or record.completed_at is None:
        return None
    return record.completed_at - record.started_at


def _duration_summary(records: Sequence[ShardRecord]) -> dict[str, float | None]:
    measured = [
        duration
        for duration in (_duration_seconds(record) for record in records)
        if duration is not None
    ]
    if not measured:
        return {"measured_shards": 0, "total": None, "p50": None, "max": None}
    return {
        "measured_shards": len(measured),
        "total": float(sum(measured)),
        "p50": float(statistics.median(measured)),
        "max": float(max(measured)),
    }


def _shard_row(record: ShardRecord) -> dict[str, Any]:
    return {
        "shard_id": record.shard_id,
        "fingerprint": _checked_digest(
            record.fingerprint,
            field_name=f"shard {record.shard_id} fingerprint",
            pattern=_HEX_DIGEST,
        ),
        "document_count": record.document_count,
        "status": record.status.value,
        "attempts": record.attempts,
        "duration_seconds": _duration_seconds(record),
        "output_digest": _checked_digest(
            record.output_digest,
            field_name=f"shard {record.shard_id} output_digest",
            pattern=_PREFIXED_DIGEST,
        ),
        "output_bytes": record.output_bytes,
        "worker_ref": worker_ref(record.worker_id),
        "error_type": _safe_error_type(record.error_type),
    }


def _resume_block(plan: ResumePlan) -> dict[str, Any]:
    """Project a resume plan field by field rather than forwarding its payload.

    The plan's own payload is safe today, but it is maintained for a different
    consumer. Rebuilding the block here means a field added upstream reaches a
    report only when someone decides it should, instead of by default.
    """

    return {
        "schema_version": plan.schema_version,
        "fingerprint": _checked_digest(
            plan.fingerprint,
            field_name="resume plan fingerprint",
            pattern=_HEX_DIGEST,
        ),
        "is_complete": plan.is_complete,
        "is_exhausted": plan.is_exhausted,
        "queued": [
            {
                "shard_id": decision.shard_id,
                "reason": decision.reason.value,
                "attempts": decision.attempts,
            }
            for decision in plan.decisions
        ],
        "completed": list(plan.completed),
        "in_flight": list(plan.in_flight),
        "exhausted": list(plan.exhausted),
        "unmeasurable": list(plan.unmeasurable),
        "straggler_detection_enabled": plan.straggler_detection_enabled,
        "straggler_baseline_seconds": plan.straggler_baseline_seconds,
        "stragglers": [
            {
                "shard_id": candidate.shard_id,
                "worker_ref": candidate.worker_ref,
                "attempts": candidate.attempts,
                "elapsed_seconds": candidate.elapsed_seconds,
                "per_document_seconds": candidate.per_document_seconds,
                "baseline_per_document_seconds": (
                    candidate.baseline_per_document_seconds
                ),
                "threshold_per_document_seconds": (
                    candidate.threshold_per_document_seconds
                ),
                "elapsed_floor_seconds": candidate.elapsed_floor_seconds,
            }
            for candidate in plan.stragglers
        ],
    }


def _outputs_block(validation: ShardOutputValidation) -> dict[str, Any]:
    return {
        "all_valid": validation.all_valid,
        "valid": list(validation.valid),
        "missing": list(validation.missing),
        "mismatched": list(validation.mismatched),
    }


@dataclass(frozen=True)
class BatchRunReport:
    """Aggregate, publishable state of one distributed batch run."""

    run_id: str
    plan_fingerprint: str
    algorithm: str
    openmed_version: str
    generated_at: float
    created_at: float
    updated_at: float
    shard_count: int
    document_count: int
    run_state: str
    status_counts: Mapping[str, int]
    total_attempts: int
    duration_seconds: Mapping[str, Any]
    shards: tuple[Mapping[str, Any], ...]
    failures: tuple[Mapping[str, Any], ...]
    resume: Mapping[str, Any] | None = None
    outputs: Mapping[str, Any] | None = None
    schema_version: int = RUN_REPORT_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        """Return the publishable payload, validated before it is handed back.

        The guard runs here rather than at the call site so that no caller can
        obtain an unchecked payload: :meth:`to_json` and every renderer go
        through this method.
        """

        payload: dict[str, Any] = {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "plan_fingerprint": self.plan_fingerprint,
            "algorithm": self.algorithm,
            "openmed_version": self.openmed_version,
            "generated_at": self.generated_at,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "shard_count": self.shard_count,
            "document_count": self.document_count,
            "run_state": self.run_state,
            "status_counts": dict(self.status_counts),
            "total_attempts": self.total_attempts,
            "duration_seconds": dict(self.duration_seconds),
            "shards": [dict(row) for row in self.shards],
            "failures": [dict(row) for row in self.failures],
        }
        if self.resume is not None:
            payload["resume"] = dict(self.resume)
        if self.outputs is not None:
            payload["outputs"] = dict(self.outputs)

        assert_no_raw_text(payload)
        return payload

    def to_json(self, *, indent: int = 2) -> str:
        """Return the payload as byte-stable JSON."""

        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    def to_markdown(self) -> str:
        """Render a byte-stable Markdown summary of the run."""

        payload = self.to_dict()
        lines = [
            f"# Batch Run Report: {self.run_id}",
            "",
            "| Field | Value |",
            "| --- | --- |",
            f"| Run | `{self.run_id}` |",
            f"| State | `{self.run_state}` |",
            f"| Plan fingerprint | `{self.plan_fingerprint}` |",
            f"| Algorithm | `{self.algorithm}` |",
            f"| OpenMed | `{self.openmed_version}` |",
            f"| Shards | {self.shard_count} |",
            f"| Documents | {self.document_count} |",
            f"| Attempts | {self.total_attempts} |",
            f"| Generated at | `{self.generated_at}` |",
            f"| Schema | `{self.schema_version}` |",
            "",
            "## Shard Status",
            "",
            "| Status | Shards |",
            "| --- | ---: |",
        ]
        for status, count in sorted(payload["status_counts"].items()):
            lines.append(f"| `{status}` | {count} |")

        durations = payload["duration_seconds"]
        lines.extend(
            [
                "",
                "## Durations",
                "",
                "| Measure | Seconds |",
                "| --- | ---: |",
                f"| Measured shards | {durations['measured_shards']} |",
                f"| Total | {_number(durations['total'])} |",
                f"| Median | {_number(durations['p50'])} |",
                f"| Max | {_number(durations['max'])} |",
                "",
                "## Failures",
                "",
            ]
        )
        if not payload["failures"]:
            lines.append("None.")
        else:
            lines.extend(
                [
                    "| Shard | Error type | Attempts |",
                    "| ---: | --- | ---: |",
                ]
            )
            for failure in payload["failures"]:
                lines.append(
                    f"| {failure['shard_id']} | `{failure['error_type']}` | "
                    f"{failure['attempts']} |"
                )

        lines.extend(["", "## Stragglers", ""])
        resume = payload.get("resume")
        if resume is None:
            lines.append("Not evaluated.")
        elif not resume["straggler_detection_enabled"]:
            lines.append(
                "Not measured: too few completed shards to establish a baseline."
            )
        elif not resume["stragglers"]:
            lines.append("None lagging.")
        else:
            lines.extend(
                [
                    "| Shard | Worker | Elapsed | Per doc | Threshold |",
                    "| ---: | --- | ---: | ---: | ---: |",
                ]
            )
            for candidate in resume["stragglers"]:
                lines.append(
                    f"| {candidate['shard_id']} | `{candidate['worker_ref']}` | "
                    f"{_number(candidate['elapsed_seconds'])} | "
                    f"{_number(candidate['per_document_seconds'])} | "
                    f"{_number(candidate['threshold_per_document_seconds'])} |"
                )

        return "\n".join(lines) + "\n"


def _number(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.3f}"


def build_run_report(
    manifest: BatchRunManifest,
    *,
    generated_at: float,
    resume: ResumePlan | None = None,
    validation: ShardOutputValidation | None = None,
) -> BatchRunReport:
    """Reduce a run manifest, and optionally a resume plan, to a report.

    ``generated_at`` is required rather than defaulted to the current time so
    that a report rendered twice from the same state is byte-identical, which is
    what makes the rendered form diffable and testable.

    A zero-document shard that reports as completed is normal, not an anomaly:
    an empty shard publishes an empty output so that it settles instead of being
    re-queued forever, and it is counted here like any other completed shard.
    """

    records = manifest.shards
    counts = {status.value: 0 for status in ShardStatus}
    for record in records:
        counts[record.status.value] += 1

    if resume is not None and resume.is_exhausted:
        run_state = RUN_STATE_EXHAUSTED
    elif resume is not None:
        run_state = RUN_STATE_COMPLETE if resume.is_complete else RUN_STATE_IN_PROGRESS
    elif counts[ShardStatus.COMPLETED.value] == len(records):
        run_state = RUN_STATE_COMPLETE
    else:
        run_state = RUN_STATE_IN_PROGRESS

    failures = tuple(
        {
            "shard_id": record.shard_id,
            "error_type": _safe_error_type(record.error_type),
            "attempts": record.attempts,
        }
        for record in records
        if record.status is ShardStatus.FAILED
    )

    return BatchRunReport(
        run_id=manifest.run_id,
        plan_fingerprint=manifest.plan_fingerprint,
        algorithm=manifest.algorithm,
        openmed_version=manifest.openmed_version,
        generated_at=generated_at,
        created_at=manifest.created_at,
        updated_at=manifest.updated_at,
        shard_count=manifest.shard_count,
        document_count=manifest.document_count,
        run_state=run_state,
        status_counts=counts,
        total_attempts=sum(record.attempts for record in records),
        duration_seconds=_duration_summary(records),
        shards=tuple(_shard_row(record) for record in records),
        failures=failures,
        resume=None if resume is None else _resume_block(resume),
        outputs=None if validation is None else _outputs_block(validation),
    )


__all__ = [
    "RUN_REPORT_SCHEMA_VERSION",
    "RUN_STATE_COMPLETE",
    "RUN_STATE_EXHAUSTED",
    "RUN_STATE_IN_PROGRESS",
    "UNRECOGNIZED_ERROR_TYPE",
    "BatchRunReport",
    "RunReportError",
    "RunReportPrivacyError",
    "assert_no_raw_text",
    "build_run_report",
]
