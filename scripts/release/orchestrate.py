#!/usr/bin/env python3
"""Run and audit fail-closed model releases bound to signed gate reports.

Epics OM-047 and OM-748 require that every published artifact provably passed
the gate it claims, via an append-only run record linking the artifact to its
``RELEASABLE`` :class:`~openmed.eval.release_gates.GateReport` by hash. This
module delivers both the original release-manifest builder and the nightly
build, evaluation, gate, model-card, publish, smoke, and rollback chain. A run
outcome is reconstructable from committed state with no raw PHI.

Design notes
------------
* The per-family record binds the candidate artifact digest to its gate report
  hash. Rather than bend :func:`openmed.core.repro_hash.compute_reproducibility_hash`
  (which is shaped for training artifacts), the binding hash goes through
  :func:`openmed.core.repro_hash.compute_canonical_payload_hash` -- the generic
  helper carrying the same canonical-JSON + SHA-256 contract ``GateReport`` uses
  for its own ``repro_hash`` -- so a ledger row hashes consistently with the
  report it references.
* The referenced gate report hash is recomputed from the report's own contents
  before it is bound in. A report whose recorded ``repro_hash`` disagrees with
  that recomputation is rejected instead of being written to the ledger.
* Fail-closed: a family whose ``GateReport`` decision is not ``RELEASABLE`` is
  quarantined and gets no publish target. Quarantine is derived from the report,
  never asserted by the caller.
* No raw PHI: the ledger stores only identifiers, hashes, offsets, and enums.
  :func:`build_release_manifest` refuses to write a record containing a
  PHI-shaped value.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, NamedTuple, Sequence

from openmed.core.repro_hash import (
    compute_canonical_payload_hash,
    compute_file_digest,
    resolve_git_sha,
)
from openmed.eval.release_gates import (
    QUARANTINED,
    RELEASABLE,
    GateCheck,
    GateReport,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LEDGER = ROOT / "gates" / "release_runs.jsonl"
DEFAULT_QUEUE = ROOT / "gates" / "nightly_release_queue.json"
DEFAULT_REPORTS_DIR = ROOT / "gates" / "release_reports"
DEFAULT_MANIFEST = ROOT / "models.jsonl"
DEFAULT_REGISTRY_STATE = ROOT / "gates" / "registry_state.json"
DEFAULT_BASELINE = ROOT / "gates" / "baseline.json"

NIGHTLY_RECORD_TYPE = "nightly-release"
RUN_COMPLETE = "COMPLETE"
RUN_PARTIAL = "PARTIAL"
OUTCOME_PUBLISHED = "PUBLISHED"
OUTCOME_QUARANTINED = "QUARANTINED"
OUTCOME_ROLLED_BACK = "ROLLED_BACK"
OUTCOME_ROLLBACK_FAILED = "ROLLBACK_FAILED"
SMOKE_PASSED = "PASSED"
SMOKE_FAILED = "FAILED"
SMOKE_NOT_RUN = "NOT_RUN"

_WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday")
_SAFE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_SAFE_REPO_ID_RE = re.compile(r"^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$")
_SUPPORTED_NIGHTLY_FORMATS = frozenset({"onnx", "webgpu", "int8"})

#: Ledger fields that identity-bind the complete run outcome. The provenance
#: hash covers every semantic field, including run identity, timestamp, and
#: fail-closed quarantine state, so no row can be moved between runs or made to
#: look published without invalidating its hash.
_BINDING_FIELDS = (
    "run_id",
    "family",
    "tier",
    "format",
    "repo_id",
    "git_sha",
    "artifact_digest",
    "gate_report_hash",
    "decision",
    "quarantined",
    "pointer_target",
    "created_at",
)

# Nightly rows extend the original release-manifest record instead of changing
# its stable hash contract. Every final stage outcome is covered, including a
# smoke-triggered rollback, so an auditor cannot turn a quarantined row into a
# published one by editing only the extra fields.
_NIGHTLY_BINDING_FIELDS = (
    "record_type",
    "candidate_id",
    "run_status",
    *_BINDING_FIELDS,
    "gate_report_path",
    "outcome",
    "smoke_test",
    "rollback_target",
    "started_at",
    "completed_at",
    "failure_stage",
)

_SHA256_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_GIT_SHA_RE = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_REQUIRED_RECORD_FIELDS = frozenset((*_BINDING_FIELDS, "provenance_hash"))
_NIGHTLY_REQUIRED_RECORD_FIELDS = frozenset(
    (*_NIGHTLY_BINDING_FIELDS, "provenance_hash")
)

# Conservative PHI-shaped patterns. The builder controls its own inputs, so this
# is a defensive guard (and the subject of a no-raw-PHI test), not a scrubber.
_PHI_PATTERNS = (
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),  # US SSN
    re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),  # email
    re.compile(r"\b\d{10,}\b"),  # long digit runs (MRN / phone / account)
)

# Format-constrained hash/identifier fields. These are exactly the "hashes and
# identifiers" the ledger is meant to carry, and a valid SHA-256 or git SHA can
# legitimately end in a long digit run, so they are excluded from the PHI scan.
_HASH_FIELDS = frozenset(
    {"artifact_digest", "gate_report_hash", "provenance_hash", "git_sha"}
)


class ReleaseManifestError(RuntimeError):
    """Raised when a ledger record fails an integrity or safety invariant."""


class NightlyCandidate(NamedTuple):
    """One reviewed family/tier/format entry in the themed release queue."""

    candidate_id: str
    weekday: str
    theme: str
    source_model_id: str
    repo_id: str
    family: str
    tier: str
    param_count: int | None
    format: str
    fixture_path: str
    suite: str = "golden"
    device: str = "cpu"

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
        *,
        expected_theme: str | None = None,
    ) -> "NightlyCandidate":
        """Validate and load one JSON queue row."""

        def required(name: str) -> str:
            item = value.get(name)
            if not isinstance(item, str) or not item.strip():
                raise ReleaseManifestError(
                    f"nightly queue field {name!r} must be a non-empty string"
                )
            return item.strip()

        candidate_id = required("id")
        if not _SAFE_ID_RE.fullmatch(candidate_id):
            raise ReleaseManifestError(
                "nightly queue ids must use lowercase letters, digits, '.', '_', or '-'"
            )
        weekday = required("weekday").lower()
        if weekday not in _WEEKDAYS:
            raise ReleaseManifestError(
                f"nightly queue weekday must be one of {_WEEKDAYS}"
            )
        theme = required("theme")
        if expected_theme is not None and theme != expected_theme:
            raise ReleaseManifestError(
                f"nightly queue item {candidate_id!r} does not match its weekday theme"
            )
        source_model_id = required("source_model_id")
        repo_id = required("repo_id")
        if not _SAFE_REPO_ID_RE.fullmatch(source_model_id):
            raise ReleaseManifestError("nightly source_model_id is not a safe repo id")
        if not _SAFE_REPO_ID_RE.fullmatch(repo_id):
            raise ReleaseManifestError("nightly repo_id is not a safe repo id")
        format_name = required("format").lower().replace("_", "-")
        if format_name not in _SUPPORTED_NIGHTLY_FORMATS:
            raise ReleaseManifestError(
                f"nightly format {format_name!r} is not supported"
            )
        param_count = value.get("param_count")
        if param_count is not None and (
            not isinstance(param_count, int)
            or isinstance(param_count, bool)
            or param_count <= 0
        ):
            raise ReleaseManifestError(
                "nightly param_count must be a positive integer or null"
            )

        candidate = cls(
            candidate_id=candidate_id,
            weekday=weekday,
            theme=theme,
            source_model_id=source_model_id,
            repo_id=repo_id,
            family=required("family"),
            tier=required("tier"),
            param_count=param_count,
            format=format_name,
            fixture_path=required("fixture_path"),
            suite=str(value.get("suite") or "golden").strip(),
            device=str(value.get("device") or "cpu").strip(),
        )
        _assert_no_phi(candidate.to_dict())
        return candidate

    def to_dict(self) -> dict[str, Any]:
        """Return the PHI-free queue identity as a plain mapping."""

        return {
            "id": self.candidate_id,
            "weekday": self.weekday,
            "theme": self.theme,
            "source_model_id": self.source_model_id,
            "repo_id": self.repo_id,
            "family": self.family,
            "tier": self.tier,
            "param_count": self.param_count,
            "format": self.format,
            "fixture_path": self.fixture_path,
            "suite": self.suite,
            "device": self.device,
        }


class NightlyResult(NamedTuple):
    """Final, locally reconstructable outcome for one nightly candidate."""

    candidate: NightlyCandidate
    gate_report: GateReport
    gate_report_path: str
    artifact_digest: str | None
    outcome: str
    smoke_test: str
    pointer_target: str | None
    rollback_target: str | None
    started_at: str
    completed_at: str
    failure_stage: str | None = None


def _binding_hash(record: Mapping[str, Any]) -> str:
    """Return the ``sha256:`` provenance hash binding artifact to gate report."""

    fields = (
        _NIGHTLY_BINDING_FIELDS
        if record.get("record_type") == NIGHTLY_RECORD_TYPE
        else _BINDING_FIELDS
    )
    payload = {field: record.get(field) for field in fields}
    return compute_canonical_payload_hash(payload)


def _verified_report_hash(report: GateReport) -> str:
    """Return the gate report hash recomputed from the report's own contents.

    The stored ``repro_hash`` is never taken on trust: a report arriving from
    disk could carry a hash that no longer matches what it claims to attest.
    A recorded hash that disagrees with the recomputation is rejected rather
    than bound into the ledger.
    """

    verified = report.recompute_repro_hash()
    recorded = report.repro_hash
    if recorded and recorded != verified:
        raise ReleaseManifestError(
            f"gate report hash mismatch for family {report.family!r}: "
            f"recorded {recorded}, recomputed {verified}"
        )
    return verified


def _verified_artifact_digest(family: str, digest: str | None) -> str:
    """Return the candidate artifact digest, refusing an absent or odd one."""

    if not digest:
        raise ReleaseManifestError(
            f"no candidate artifact digest supplied for family {family!r}"
        )
    if not _SHA256_DIGEST_RE.match(digest):
        raise ReleaseManifestError(
            f"candidate artifact digest for family {family!r} is not a "
            f"sha256: digest: {digest}"
        )
    return digest


def _iter_record_strings(record: Mapping[str, Any]) -> Iterable[str]:
    def walk(key: str, value: Any) -> Iterable[str]:
        if (
            key in _HASH_FIELDS
            or key.endswith(("_hash", "_digest"))
            or key == "signature"
        ):
            return
        if isinstance(value, Mapping):
            for nested_key, nested_value in value.items():
                yield from walk(str(nested_key), nested_value)
        elif isinstance(value, Sequence) and not isinstance(
            value, (str, bytes, bytearray)
        ):
            for nested_value in value:
                yield from walk(key, nested_value)
        elif isinstance(value, str):
            yield value

    for key, value in record.items():
        yield from walk(str(key), value)


def _assert_no_phi(record: Mapping[str, Any]) -> None:
    """Reject a record whose values look like raw PHI before it is written."""

    for value in _iter_record_strings(record):
        for pattern in _PHI_PATTERNS:
            if pattern.search(value):
                raise ReleaseManifestError(
                    "refusing to write PHI-shaped value to the release ledger"
                )


def _assert_nonempty_string(record: Mapping[str, Any], field: str) -> None:
    value = record.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ReleaseManifestError(f"release ledger field {field!r} must be non-empty")


def _assert_record_invariants(record: Mapping[str, Any]) -> None:
    """Reject a ledger row that is incomplete, inconsistent, or malformed."""

    nightly = record.get("record_type") == NIGHTLY_RECORD_TYPE
    required_fields = (
        _NIGHTLY_REQUIRED_RECORD_FIELDS if nightly else _REQUIRED_RECORD_FIELDS
    )
    missing = required_fields - record.keys()
    if missing:
        raise ReleaseManifestError(
            f"release ledger record is missing required fields: {sorted(missing)}"
        )

    for field in (
        "run_id",
        "family",
        "tier",
        "format",
        "repo_id",
        "git_sha",
        "gate_report_hash",
        "decision",
        "created_at",
        "provenance_hash",
    ):
        _assert_nonempty_string(record, field)

    if not nightly:
        _assert_nonempty_string(record, "artifact_digest")

    if not _GIT_SHA_RE.fullmatch(str(record["git_sha"])):
        raise ReleaseManifestError("release ledger git_sha must be a 40- or 64-hex SHA")
    for field in ("gate_report_hash", "provenance_hash"):
        if not _SHA256_DIGEST_RE.fullmatch(str(record[field])):
            raise ReleaseManifestError(
                f"release ledger field {field!r} must be a sha256: digest"
            )
    artifact_digest = record.get("artifact_digest")
    if artifact_digest is not None and not _SHA256_DIGEST_RE.fullmatch(
        str(artifact_digest)
    ):
        raise ReleaseManifestError(
            "release ledger field 'artifact_digest' must be a sha256: digest or null"
        )

    decision = record["decision"]
    if decision not in {RELEASABLE, QUARANTINED}:
        raise ReleaseManifestError(
            f"release ledger decision must be {RELEASABLE!r} or {QUARANTINED!r}"
        )
    quarantined = record["quarantined"]
    if type(quarantined) is not bool:
        raise ReleaseManifestError("release ledger quarantined field must be boolean")

    if nightly:
        _assert_nightly_record_invariants(record)
        _assert_no_phi(record)
        return

    expected_quarantine = decision != RELEASABLE
    if quarantined is not expected_quarantine:
        raise ReleaseManifestError(
            "release ledger decision and quarantined state are inconsistent"
        )

    pointer_target = record["pointer_target"]
    if expected_quarantine:
        if pointer_target is not None:
            raise ReleaseManifestError(
                "quarantined release ledger records cannot have a publish target"
            )
    elif not isinstance(pointer_target, str) or not pointer_target.strip():
        raise ReleaseManifestError(
            "releasable release ledger records need a non-empty publish target"
        )

    _assert_no_phi(record)


def _assert_nightly_record_invariants(record: Mapping[str, Any]) -> None:
    """Validate final-outcome fields that exist only on nightly rows."""

    for field in (
        "candidate_id",
        "run_status",
        "gate_report_path",
        "outcome",
        "smoke_test",
        "started_at",
        "completed_at",
    ):
        _assert_nonempty_string(record, field)
    if not _SAFE_ID_RE.fullmatch(str(record["candidate_id"])):
        raise ReleaseManifestError("nightly candidate_id is malformed")
    if record["run_status"] not in {RUN_COMPLETE, RUN_PARTIAL}:
        raise ReleaseManifestError("nightly run_status is invalid")

    report_path = Path(str(record["gate_report_path"]))
    if report_path.is_absolute() or ".." in report_path.parts:
        raise ReleaseManifestError(
            "nightly gate_report_path must stay repository-relative"
        )

    outcome = record["outcome"]
    if outcome not in {
        OUTCOME_PUBLISHED,
        OUTCOME_QUARANTINED,
        OUTCOME_ROLLED_BACK,
        OUTCOME_ROLLBACK_FAILED,
    }:
        raise ReleaseManifestError("nightly outcome is invalid")
    smoke_test = record["smoke_test"]
    if smoke_test not in {SMOKE_PASSED, SMOKE_FAILED, SMOKE_NOT_RUN}:
        raise ReleaseManifestError("nightly smoke_test state is invalid")

    pointer_target = record.get("pointer_target")
    rollback_target = record.get("rollback_target")
    quarantined = bool(record["quarantined"])
    decision = record["decision"]
    artifact_digest = record.get("artifact_digest")

    if outcome == OUTCOME_PUBLISHED:
        if decision != RELEASABLE or quarantined or smoke_test != SMOKE_PASSED:
            raise ReleaseManifestError(
                "published nightly rows require a RELEASABLE gate and passing smoke"
            )
        if pointer_target != record["repo_id"] or rollback_target is not None:
            raise ReleaseManifestError(
                "published nightly rows must point at the published repository"
            )
    elif outcome == OUTCOME_ROLLED_BACK:
        if decision != RELEASABLE or not quarantined:
            raise ReleaseManifestError(
                "rolled-back nightly rows require a RELEASABLE pre-publish gate"
            )
        if smoke_test not in {SMOKE_PASSED, SMOKE_FAILED}:
            raise ReleaseManifestError(
                "rolled-back nightly rows require a completed smoke test"
            )
        if not isinstance(rollback_target, str) or not rollback_target:
            raise ReleaseManifestError(
                "rolled-back nightly rows need a rollback target"
            )
        if pointer_target != rollback_target:
            raise ReleaseManifestError(
                "rolled-back nightly pointer must equal its rollback target"
            )
    elif outcome == OUTCOME_ROLLBACK_FAILED:
        if (
            decision != RELEASABLE
            or not quarantined
            or smoke_test not in {SMOKE_PASSED, SMOKE_FAILED}
        ):
            raise ReleaseManifestError("rollback-failed nightly row is inconsistent")
        if pointer_target is not None or rollback_target is not None:
            raise ReleaseManifestError(
                "rollback-failed nightly rows cannot claim a safe pointer target"
            )
    else:
        if not quarantined or pointer_target is not None:
            raise ReleaseManifestError(
                "quarantined nightly rows cannot have a publish target"
            )
        if smoke_test != SMOKE_NOT_RUN or rollback_target is not None:
            raise ReleaseManifestError(
                "quarantined nightly rows cannot claim smoke or rollback completion"
            )

    if artifact_digest is None and not (
        outcome == OUTCOME_QUARANTINED and record.get("failure_stage") == "build"
    ):
        raise ReleaseManifestError(
            "nightly rows need an artifact digest unless the build itself failed"
        )


def _build_record(
    report: GateReport,
    *,
    run_id: str,
    git_sha: str,
    created_at: str,
    artifact_digest: str | None,
    pointer_target: str | None,
) -> dict[str, Any]:
    """Assemble one fail-closed run record for a single family's gate report."""

    if report.decision not in {RELEASABLE, QUARANTINED}:
        raise ReleaseManifestError(
            f"unknown gate decision for family {report.family!r}: {report.decision!r}"
        )
    releasable = report.decision == RELEASABLE
    # Fail-closed: a non-releasable family is quarantined and never gets a target.
    resolved_target = pointer_target if releasable else None

    record: dict[str, Any] = {
        "run_id": run_id,
        "family": report.family,
        "tier": report.tier,
        "format": report.format,
        "repo_id": report.repo_id,
        "git_sha": git_sha,
        "artifact_digest": _verified_artifact_digest(report.family, artifact_digest),
        "gate_report_hash": _verified_report_hash(report),
        "decision": report.decision,
        "quarantined": not releasable,
        "pointer_target": resolved_target,
        "created_at": created_at,
    }
    record["provenance_hash"] = _binding_hash(record)
    _assert_record_invariants(record)
    return record


def build_release_manifest(
    reports: Sequence[GateReport],
    *,
    run_id: str,
    created_at: str,
    artifact_digests: Mapping[str, str],
    git_sha: str | None = None,
    pointer_targets: Mapping[str, str] | None = None,
    ledger_path: str | Path = DEFAULT_LEDGER,
) -> list[dict[str, Any]]:
    """Append a per-family run record for ``reports`` and return the records.

    Each record binds the candidate artifact digest to its gate report hash via
    a provenance hash, so a row cannot silently be re-pointed at a different
    artifact or a different report. ``artifact_digests`` maps family to the
    ``sha256:`` digest of that family's candidate artifact and is required: a
    family with no digest is refused rather than recorded unbound. Families
    whose gate report is not ``RELEASABLE`` are quarantined with no publish
    target (fail-closed). Records are appended to ``ledger_path`` as JSON
    lines; the ledger is never truncated.

    ``created_at`` is injected (not read from the clock) so the same inputs
    produce a byte-identical record, which keeps the provenance hash and the
    reconstructed outcome deterministic and testable.
    """

    resolved_sha = git_sha or resolve_git_sha()
    digests = dict(artifact_digests)
    targets = dict(pointer_targets or {})
    ledger = Path(ledger_path)
    ledger.parent.mkdir(parents=True, exist_ok=True)

    if not reports:
        raise ReleaseManifestError("at least one gate report is required")
    families = [report.family for report in reports]
    duplicate_families = sorted(
        family for family in set(families) if families.count(family) > 1
    )
    if duplicate_families:
        raise ReleaseManifestError(
            f"duplicate gate-report families in run {run_id!r}: {duplicate_families}"
        )

    existing_rows = _load_ledger(ledger)
    for row in existing_rows:
        if not verify_record(row):
            raise ReleaseManifestError(
                "refusing to append to a release ledger containing an invalid row"
            )
    existing_keys = {(str(row["run_id"]), str(row["family"])) for row in existing_rows}
    collisions = sorted(
        family for family in families if (run_id, family) in existing_keys
    )
    if collisions:
        raise ReleaseManifestError(
            f"release ledger already contains run/family records: {collisions}"
        )

    records: list[dict[str, Any]] = []
    for report in reports:
        # Default the publish target to the report's own repo id; an explicit
        # override wins. Quarantine still strips it inside _build_record.
        target = targets.get(report.family, report.repo_id)
        records.append(
            _build_record(
                report,
                run_id=run_id,
                git_sha=resolved_sha,
                created_at=created_at,
                artifact_digest=digests.get(report.family),
                pointer_target=target,
            )
        )

    with ledger.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True, sort_keys=True))
            handle.write("\n")

    return records


def _load_ledger(ledger_path: str | Path) -> list[dict[str, Any]]:
    path = Path(ledger_path)
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = line.strip()
        if line:
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ReleaseManifestError(
                    f"release ledger line {line_number} must be a JSON object"
                )
            rows.append(row)
    return rows


def verify_record(record: Mapping[str, Any]) -> bool:
    """Return True when a record is structurally valid and hash-consistent."""

    try:
        _assert_record_invariants(record)
    except ReleaseManifestError:
        return False
    return _binding_hash(record) == record.get("provenance_hash")


def reconstruct_run(
    run_id: str,
    *,
    ledger_path: str | Path = DEFAULT_LEDGER,
) -> dict[str, dict[str, Any]]:
    """Reproduce the published-vs-quarantined outcome per family for ``run_id``.

    Reads only committed ledger state -- no live API call. Each row's provenance
    hash is re-verified against a recomputation; a mismatch raises
    :class:`ReleaseManifestError` rather than reporting a forged outcome.
    """

    outcome: dict[str, dict[str, Any]] = {}
    for row in _load_ledger(ledger_path):
        if not verify_record(row):
            raise ReleaseManifestError(
                f"provenance hash mismatch for family {row.get('family')!r} "
                f"in ledger run {row.get('run_id')!r}"
            )
        if row["run_id"] != run_id:
            continue
        key = str(row.get("candidate_id") or row["family"])
        if key in outcome:
            raise ReleaseManifestError(
                f"duplicate release key {key!r} in ledger run {run_id!r}"
            )
        final_outcome = row.get("outcome")
        published = (
            final_outcome == OUTCOME_PUBLISHED
            if row.get("record_type") == NIGHTLY_RECORD_TYPE
            else not row["quarantined"]
        )
        outcome[key] = {
            "published": published,
            "quarantined": bool(row["quarantined"]),
            "decision": row["decision"],
            "pointer_target": row["pointer_target"],
            "artifact_digest": row["artifact_digest"],
            "gate_report_hash": row["gate_report_hash"],
            "provenance_hash": row["provenance_hash"],
        }
        if row.get("record_type") == NIGHTLY_RECORD_TYPE:
            outcome[key].update(
                {
                    "family": row["family"],
                    "tier": row["tier"],
                    "format": row["format"],
                    "gate_report_path": row["gate_report_path"],
                    "outcome": row["outcome"],
                    "run_status": row["run_status"],
                    "smoke_test": row["smoke_test"],
                }
            )
    return outcome


def load_nightly_queue(
    path: str | Path = DEFAULT_QUEUE,
    *,
    weekday: str,
) -> list[NightlyCandidate]:
    """Load the reviewed candidates for one UTC weekday."""

    normalized_weekday = weekday.strip().lower()
    if normalized_weekday not in _WEEKDAYS:
        raise ReleaseManifestError(
            f"nightly weekday must be one of {_WEEKDAYS}, got {weekday!r}"
        )
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReleaseManifestError("could not load the nightly release queue") from exc
    if not isinstance(payload, Mapping) or payload.get("schema_version") != 1:
        raise ReleaseManifestError("nightly queue must use schema_version 1")
    themes = payload.get("weekly_themes")
    if not isinstance(themes, Mapping):
        raise ReleaseManifestError("nightly queue weekly_themes must be an object")
    normalized_themes = {
        str(day).lower(): str(theme)
        for day, theme in themes.items()
        if isinstance(day, str) and isinstance(theme, str) and theme
    }
    if set(normalized_themes) != set(_WEEKDAYS):
        raise ReleaseManifestError(
            "nightly queue must define exactly one theme for every weekday"
        )
    items = payload.get("candidates")
    if not isinstance(items, list) or not items:
        raise ReleaseManifestError("nightly queue candidates must be a non-empty array")

    candidates: list[NightlyCandidate] = []
    seen: set[str] = set()
    for item in items:
        if not isinstance(item, Mapping):
            raise ReleaseManifestError(
                "every nightly queue candidate must be an object"
            )
        candidate_weekday = str(item.get("weekday") or "").lower()
        expected_theme = normalized_themes.get(candidate_weekday)
        candidate = NightlyCandidate.from_mapping(
            item,
            expected_theme=expected_theme,
        )
        if candidate.candidate_id in seen:
            raise ReleaseManifestError(
                f"duplicate nightly candidate id: {candidate.candidate_id}"
            )
        seen.add(candidate.candidate_id)
        if candidate.weekday == normalized_weekday:
            candidates.append(candidate)
    return candidates


def _nightly_record(
    result: NightlyResult,
    *,
    run_id: str,
    run_status: str,
    git_sha: str,
) -> dict[str, Any]:
    candidate = result.candidate
    record: dict[str, Any] = {
        "record_type": NIGHTLY_RECORD_TYPE,
        "candidate_id": candidate.candidate_id,
        "run_id": run_id,
        "run_status": run_status,
        "family": candidate.family,
        "tier": candidate.tier,
        "format": candidate.format,
        "repo_id": candidate.repo_id,
        "git_sha": git_sha,
        "artifact_digest": result.artifact_digest,
        "gate_report_hash": _verified_report_hash(result.gate_report),
        "gate_report_path": result.gate_report_path,
        "decision": result.gate_report.decision,
        "outcome": result.outcome,
        "quarantined": result.outcome != OUTCOME_PUBLISHED,
        "smoke_test": result.smoke_test,
        "pointer_target": result.pointer_target,
        "rollback_target": result.rollback_target,
        "started_at": result.started_at,
        "completed_at": result.completed_at,
        "created_at": result.completed_at,
        "failure_stage": result.failure_stage,
    }
    record["provenance_hash"] = _binding_hash(record)
    _assert_record_invariants(record)
    return record


def append_nightly_records(
    results: Sequence[NightlyResult],
    *,
    run_id: str,
    git_sha: str,
    ledger_path: str | Path = DEFAULT_LEDGER,
) -> list[dict[str, Any]]:
    """Atomically append one integrity-bound final row per nightly candidate."""

    if not results:
        raise ReleaseManifestError("nightly release requires at least one candidate")
    if not _SAFE_ID_RE.fullmatch(run_id):
        raise ReleaseManifestError("nightly run_id is malformed")
    if not _GIT_SHA_RE.fullmatch(git_sha):
        raise ReleaseManifestError("nightly git_sha must be a 40- or 64-hex SHA")

    run_status = (
        RUN_COMPLETE
        if all(result.outcome == OUTCOME_PUBLISHED for result in results)
        else RUN_PARTIAL
    )
    records = [
        _nightly_record(
            result,
            run_id=run_id,
            run_status=run_status,
            git_sha=git_sha,
        )
        for result in results
    ]
    candidate_ids = [record["candidate_id"] for record in records]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ReleaseManifestError("duplicate candidate ids in nightly result set")

    ledger = Path(ledger_path)
    existing = _load_ledger(ledger)
    for row in existing:
        if not verify_record(row):
            raise ReleaseManifestError(
                "refusing to append to a release ledger containing an invalid row"
            )
    existing_keys = {
        (str(row.get("run_id")), str(row.get("candidate_id")))
        for row in existing
        if row.get("record_type") == NIGHTLY_RECORD_TYPE
    }
    collisions = sorted(
        candidate_id
        for candidate_id in candidate_ids
        if (run_id, candidate_id) in existing_keys
    )
    if collisions:
        raise ReleaseManifestError(
            f"nightly ledger already contains run/candidate records: {collisions}"
        )

    ledger.parent.mkdir(parents=True, exist_ok=True)
    with ledger.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True, sort_keys=True))
            handle.write("\n")
    return records


def audit_nightly_run(
    run_id: str,
    *,
    ledger_path: str | Path = DEFAULT_LEDGER,
    repository_root: str | Path = ROOT,
) -> dict[str, dict[str, Any]]:
    """Verify and reconstruct a nightly run using only committed local files."""

    root = Path(repository_root).resolve()
    rows = [
        row
        for row in _load_ledger(ledger_path)
        if row.get("run_id") == run_id and row.get("record_type") == NIGHTLY_RECORD_TYPE
    ]
    if not rows:
        raise ReleaseManifestError(f"nightly run is absent from ledger: {run_id}")

    expected_status = (
        RUN_COMPLETE
        if all(row.get("outcome") == OUTCOME_PUBLISHED for row in rows)
        else RUN_PARTIAL
    )
    for row in rows:
        if not verify_record(row):
            raise ReleaseManifestError("nightly ledger provenance verification failed")
        if row.get("run_status") != expected_status:
            raise ReleaseManifestError("nightly ledger run_status is inconsistent")
        report_path = (root / str(row["gate_report_path"])).resolve()
        try:
            report_path.relative_to(root)
        except ValueError as exc:
            raise ReleaseManifestError(
                "nightly gate report escaped the repository root"
            ) from exc
        try:
            report = _load_gate_report(report_path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise ReleaseManifestError("could not load a nightly gate report") from exc
        if _verified_report_hash(report) != row["gate_report_hash"]:
            raise ReleaseManifestError("nightly gate report hash does not match ledger")
        identity = (report.repo_id, report.family, report.tier, report.format)
        if identity != (
            row["repo_id"],
            row["family"],
            row["tier"],
            row["format"],
        ):
            raise ReleaseManifestError(
                "nightly gate report identity does not match ledger"
            )
        if row["outcome"] in {OUTCOME_PUBLISHED, OUTCOME_ROLLED_BACK}:
            if report.decision != RELEASABLE:
                raise ReleaseManifestError(
                    "nightly publish attempt lacks a RELEASABLE gate report"
                )
        if row["decision"] != report.decision:
            raise ReleaseManifestError("nightly gate decision does not match ledger")
    return reconstruct_run(run_id, ledger_path=ledger_path)


class ReleaseRuntime:
    """Concrete adapters for build, evaluation, publish, smoke, and pointers."""

    def __init__(
        self,
        *,
        root: str | Path = ROOT,
        output_root: str | Path | None = None,
        manifest_path: str | Path = DEFAULT_MANIFEST,
        registry_state_path: str | Path = DEFAULT_REGISTRY_STATE,
        baseline_path: str | Path = DEFAULT_BASELINE,
        reports_dir: str | Path = DEFAULT_REPORTS_DIR,
        signing_key: bytes | str | None = None,
        report_issues: bool = True,
    ) -> None:
        self.root = Path(root).resolve()
        self.output_root = Path(output_root or self.root / "release-artifacts")
        self.manifest_path = Path(manifest_path)
        self.registry_state_path = Path(registry_state_path)
        self.baseline_path = Path(baseline_path)
        self.reports_dir = Path(reports_dir)
        self.signing_key = signing_key or os.environ.get(
            "OPENMED_RELEASE_GATE_KEY",
            "openmed-release-gate-local-key",
        )
        self.report_issues = report_issues

    def artifact_dir(self, candidate: NightlyCandidate) -> Path:
        return self.output_root / candidate.candidate_id

    def build(self, candidate: NightlyCandidate) -> Path:
        """Convert one reviewed source model without exposing child output."""

        artifact_dir = self.artifact_dir(candidate)
        if artifact_dir.exists():
            raise ReleaseManifestError("nightly artifact directory already exists")
        command = [
            sys.executable,
            "-m",
            "openmed.onnx.convert",
            "--model",
            candidate.source_model_id,
            "--output",
            str(artifact_dir),
        ]
        if candidate.format == "int8":
            command.append("--no-webgpu")
        elif candidate.format == "onnx":
            command.append("--no-int8")
        completed = subprocess.run(
            command,
            cwd=self.root,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if completed.returncode != 0 or not artifact_dir.is_dir():
            raise ReleaseManifestError("nightly build stage failed")
        return artifact_dir

    def artifact_digest(self, artifact_dir: Path) -> str:
        from openmed.core.hf_publish import artifact_sha256

        return artifact_sha256(artifact_dir)

    def evaluate(
        self,
        candidate: NightlyCandidate,
        artifact_dir: Path,
        *,
        generated_at: str,
    ) -> Any:
        """Run the shared harness against the converted local artifact."""

        from openmed.core.config import OpenMedConfig
        from openmed.core.models import ModelLoader
        from openmed.core.pii import extract_pii
        from openmed.eval.harness import run_suite

        fixture_path = self._repository_path(candidate.fixture_path)
        config = OpenMedConfig(
            backend="onnx",
            device="cpu",
            hf_token=os.environ.get("HF_WRITE_TOKEN"),
            onnx_variant="int8" if candidate.format == "int8" else "auto",
            use_medical_tokenizer=False,
        )
        loader = ModelLoader(config)

        def runner(fixture: Any, model_name: str, device: str) -> Iterable[Any]:
            del device
            return extract_pii(
                fixture.text,
                model_name=model_name,
                lang=fixture.language,
                config=config,
                loader=loader,
            ).entities

        return run_suite(
            fixture_path,
            suite=candidate.suite,
            model_name=str(artifact_dir),
            device=candidate.device,
            runner=runner,
            generated_at=generated_at,
            metadata={
                "repo_id": candidate.repo_id,
                "family": candidate.family,
                "tier": candidate.tier,
                "param_count": candidate.param_count,
                "format": candidate.format,
            },
        )

    def gate(self, candidate: NightlyCandidate, benchmark_report: Any) -> GateReport:
        """Evaluate and sign the canonical release gate."""

        del candidate
        from openmed.eval.release_gates import ReleaseGate

        return ReleaseGate(
            baseline_path=self.baseline_path,
            signing_key=self.signing_key,
        ).evaluate(benchmark_report)

    def build_card(
        self,
        candidate: NightlyCandidate,
        artifact_dir: Path,
        report: GateReport,
        *,
        git_sha: str,
        released: str,
    ) -> None:
        """Generate the artifact-backed model card before any upload."""

        from openmed.core.hf_publish import build_manifest_row
        from openmed.eval.model_card_builder import build_model_card

        row = build_manifest_row(
            repo_id=candidate.repo_id,
            source_model_id=candidate.source_model_id,
            artifact_dir=artifact_dir,
            format_name=candidate.format,
            released=released,
            git_sha=git_sha,
        )
        card = build_model_card(row, report)
        card.write_markdown(artifact_dir / "README.md")
        card.write_datasheet(artifact_dir / "model-datasheet.json")

    def publish(
        self,
        candidate: NightlyCandidate,
        artifact_dir: Path,
        gate_report_path: Path,
        *,
        git_sha: str,
        released: str,
    ) -> Any:
        """Publish only after the card and signed gate report are complete."""

        from openmed.core.hf_publish import publish_artifact

        return publish_artifact(
            artifact_dir=artifact_dir,
            source_model_id=candidate.source_model_id,
            format_name=candidate.format,
            repo_id=candidate.repo_id,
            manifest_path=self.manifest_path,
            gate_report_path=gate_report_path,
            gate_signing_key=self.signing_key,
            skip_existing=False,
            released=released,
            git_sha=git_sha,
        )

    def promote(self, candidate: NightlyCandidate, report: GateReport) -> str:
        from openmed.core.registry_service import RegistryService

        service = RegistryService(
            manifest_path=self.manifest_path,
            state_path=self.registry_state_path,
        )
        service.promote(candidate.repo_id, gate_report=report)
        return str(service.pointers(candidate.family)["latest"])

    def smoke(self, candidate: NightlyCandidate) -> None:
        try:
            from scripts.release.smoke_test import run_fresh_venv_smoke
        except ImportError:  # pragma: no cover - direct script execution path
            from smoke_test import run_fresh_venv_smoke

        run_fresh_venv_smoke(
            candidate.repo_id,
            format_name=candidate.format,
            repository_root=self.root,
        )

    def mark_last_green(
        self,
        candidate: NightlyCandidate,
        report: GateReport,
    ) -> str:
        from openmed.core.registry_service import RegistryService

        service = RegistryService(
            manifest_path=self.manifest_path,
            state_path=self.registry_state_path,
        )
        service.flip_pointer(
            candidate.family,
            "last_green",
            candidate.repo_id,
            gate_report=report,
        )
        return str(service.pointers(candidate.family)["last_green"])

    def rollback(self, candidate: NightlyCandidate) -> str:
        from openmed.core.registry_service import RegistryService

        service = RegistryService(
            manifest_path=self.manifest_path,
            state_path=self.registry_state_path,
        )
        target = service.pointers(candidate.family).get("last_green")
        if not isinstance(target, str) or not target:
            raise ReleaseManifestError("nightly rollback target is missing")
        evidence = self._last_green_evidence(target)
        service.rollback(candidate.family, gate_report=evidence)
        restored = service.pointers(candidate.family).get("latest")
        if restored != target:
            raise ReleaseManifestError(
                "nightly rollback pointer did not reach last green"
            )
        return target

    def failure_report(
        self,
        candidate: NightlyCandidate,
        *,
        stage: str,
    ) -> GateReport:
        """Create a signed, PHI-free quarantine report for an aborted stage."""

        evidence_hash = compute_canonical_payload_hash(
            {
                "candidate_id": candidate.candidate_id,
                "family": candidate.family,
                "format": candidate.format,
                "stage": stage,
                "tier": candidate.tier,
            }
        )
        return GateReport(
            repo_id=candidate.repo_id,
            family=candidate.family,
            tier=candidate.tier,
            param_count=candidate.param_count,
            format=candidate.format,
            per_label_recall={},
            per_label_precision={},
            critical_leakage_count=1,
            residual_leakage_rate=1.0,
            quant_recall_delta=None,
            p50_ms=None,
            p95_ms=None,
            ram_mb=None,
            eval_set_hash=evidence_hash,
            leakage_fixture_hash=evidence_hash,
            decision=QUARANTINED,
            gate_results=(
                GateCheck(
                    gate="orchestrator_stage",
                    passed=False,
                    reason=f"{stage}_failed",
                    details={"candidate_hash": evidence_hash},
                ),
            ),
        ).sign(self.signing_key)

    def report_quarantine(
        self,
        candidate: NightlyCandidate,
        *,
        run_id: str,
        git_sha: str,
        stage: str,
        gate_report_hash: str,
    ) -> None:
        """Open or update the candidate's PHI-free quarantine issue."""

        repository = os.environ.get("GITHUB_REPOSITORY")
        if not self.report_issues or not repository:
            return
        title = f"Nightly release quarantine: {candidate.candidate_id}"
        body = "\n".join(
            (
                "## Nightly release quarantine",
                "",
                f"- Candidate: `{candidate.candidate_id}`",
                f"- Family/tier/format: `{candidate.family}/{candidate.tier}/{candidate.format}`",
                f"- Run: `{run_id}`",
                f"- Git SHA: `{git_sha}`",
                f"- Failure stage: `{stage}`",
                f"- Gate report hash: `{gate_report_hash}`",
                "",
                "The candidate was not left on the latest registry pointer.",
            )
        )
        try:
            lookup = subprocess.run(
                [
                    "gh",
                    "issue",
                    "list",
                    "--repo",
                    repository,
                    "--state",
                    "open",
                    "--search",
                    f"{title} in:title",
                    "--json",
                    "number,title",
                ],
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except OSError:
            return
        issue_number: str | None = None
        if lookup.returncode == 0:
            try:
                matches = json.loads(lookup.stdout)
            except json.JSONDecodeError:
                matches = []
            for match in matches if isinstance(matches, list) else []:
                if isinstance(match, Mapping) and match.get("title") == title:
                    issue_number = str(match.get("number"))
                    break
        command = (
            [
                "gh",
                "issue",
                "comment",
                issue_number,
                "--repo",
                repository,
                "--body",
                body,
            ]
            if issue_number
            else [
                "gh",
                "issue",
                "create",
                "--repo",
                repository,
                "--title",
                title,
                "--body",
                body,
            ]
        )
        try:
            subprocess.run(
                command,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except OSError:
            return

    def _repository_path(self, value: str | Path) -> Path:
        path = (self.root / value).resolve()
        try:
            path.relative_to(self.root)
        except ValueError as exc:
            raise ReleaseManifestError(
                "nightly path escaped the repository root"
            ) from exc
        if not path.is_file():
            raise ReleaseManifestError("nightly fixture path does not exist")
        return path

    def _last_green_evidence(self, repo_id: str) -> Mapping[str, Any] | GateReport:
        for path in sorted(self.reports_dir.rglob("*.json")):
            try:
                report = _load_gate_report(path)
            except (OSError, ValueError, json.JSONDecodeError):
                continue
            if report.repo_id == repo_id and report.decision == RELEASABLE:
                return report

        try:
            baseline = json.loads(self.baseline_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ReleaseManifestError("could not load last-green evidence") from exc
        entries = baseline.get("entries") if isinstance(baseline, Mapping) else None
        if isinstance(entries, Mapping):
            for entry in entries.values():
                if not isinstance(entry, Mapping) or entry.get("repo_id") != repo_id:
                    continue
                metadata = entry.get("metadata")
                report_hash = (
                    metadata.get("gate_report_hash")
                    if isinstance(metadata, Mapping)
                    else None
                )
                return {
                    "decision": RELEASABLE,
                    "repo_id": repo_id,
                    "family": entry.get("family"),
                    "tier": entry.get("tier"),
                    "format": entry.get("format"),
                    "repro_hash": report_hash
                    or compute_canonical_payload_hash(dict(entry)),
                }
        raise ReleaseManifestError("last-green pointer lacks committed gate evidence")


def _iso_timestamp(clock: Callable[[], datetime]) -> str:
    moment = clock()
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    else:
        moment = moment.astimezone(timezone.utc)
    return moment.isoformat()


def _gate_report_output(
    report: GateReport,
    *,
    candidate: NightlyCandidate,
    run_id: str,
    reports_dir: str | Path,
    repository_root: str | Path,
) -> tuple[Path, str]:
    _assert_no_phi(report.to_dict())
    output = Path(reports_dir) / run_id / f"{candidate.candidate_id}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report.to_json() + "\n", encoding="utf-8")
    root = Path(repository_root).resolve()
    resolved = output.resolve()
    try:
        relative = resolved.relative_to(root).as_posix()
    except ValueError as exc:
        raise ReleaseManifestError(
            "nightly gate reports must be stored inside the repository"
        ) from exc
    return output, relative


def _safe_stage(candidate: NightlyCandidate, stage: str) -> None:
    print(f"nightly candidate {candidate.candidate_id}: {stage}")


def orchestrate_nightly(
    candidates: Sequence[NightlyCandidate],
    *,
    run_id: str,
    git_sha: str,
    runtime: ReleaseRuntime,
    ledger_path: str | Path = DEFAULT_LEDGER,
    reports_dir: str | Path = DEFAULT_REPORTS_DIR,
    clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> list[NightlyResult]:
    """Run the full release chain per candidate and continue after quarantine."""

    if not candidates:
        raise ReleaseManifestError("nightly orchestration needs at least one candidate")
    results: list[NightlyResult] = []
    for candidate in candidates:
        started_at = _iso_timestamp(clock)
        artifact_dir: Path | None = None
        artifact_digest: str | None = None
        report: GateReport | None = None
        report_path: Path | None = None
        report_relative: str | None = None
        stage = "build"
        try:
            _safe_stage(candidate, stage)
            artifact_dir = runtime.build(candidate)
            artifact_digest = runtime.artifact_digest(artifact_dir)

            stage = "eval"
            _safe_stage(candidate, stage)
            benchmark = runtime.evaluate(
                candidate,
                artifact_dir,
                generated_at=started_at,
            )

            stage = "gate"
            _safe_stage(candidate, stage)
            report = runtime.gate(candidate, benchmark)
            report_path, report_relative = _gate_report_output(
                report,
                candidate=candidate,
                run_id=run_id,
                reports_dir=reports_dir,
                repository_root=runtime.root,
            )
        except Exception:
            report = report or runtime.failure_report(candidate, stage=stage)
            if report_relative is None:
                report_path, report_relative = _gate_report_output(
                    report,
                    candidate=candidate,
                    run_id=run_id,
                    reports_dir=reports_dir,
                    repository_root=runtime.root,
                )
            runtime.report_quarantine(
                candidate,
                run_id=run_id,
                git_sha=git_sha,
                stage=stage,
                gate_report_hash=_verified_report_hash(report),
            )
            _safe_stage(candidate, "quarantined")
            results.append(
                NightlyResult(
                    candidate=candidate,
                    gate_report=report,
                    gate_report_path=report_relative,
                    artifact_digest=artifact_digest,
                    outcome=OUTCOME_QUARANTINED,
                    smoke_test=SMOKE_NOT_RUN,
                    pointer_target=None,
                    rollback_target=None,
                    started_at=started_at,
                    completed_at=_iso_timestamp(clock),
                    failure_stage=stage,
                )
            )
            continue

        assert report is not None and report_path is not None and report_relative
        if report.decision != RELEASABLE:
            runtime.report_quarantine(
                candidate,
                run_id=run_id,
                git_sha=git_sha,
                stage="gate",
                gate_report_hash=_verified_report_hash(report),
            )
            _safe_stage(candidate, "quarantined")
            results.append(
                NightlyResult(
                    candidate=candidate,
                    gate_report=report,
                    gate_report_path=report_relative,
                    artifact_digest=artifact_digest,
                    outcome=OUTCOME_QUARANTINED,
                    smoke_test=SMOKE_NOT_RUN,
                    pointer_target=None,
                    rollback_target=None,
                    started_at=started_at,
                    completed_at=_iso_timestamp(clock),
                    failure_stage="gate",
                )
            )
            continue

        try:
            stage = "model-card"
            _safe_stage(candidate, stage)
            runtime.build_card(
                candidate,
                artifact_dir,
                report,
                git_sha=git_sha,
                released=started_at[:10],
            )
            stage = "publish"
            _safe_stage(candidate, stage)
            runtime.publish(
                candidate,
                artifact_dir,
                report_path,
                git_sha=git_sha,
                released=started_at[:10],
            )
            stage = "promote"
            _safe_stage(candidate, stage)
            runtime.promote(candidate, report)
        except Exception:
            runtime.report_quarantine(
                candidate,
                run_id=run_id,
                git_sha=git_sha,
                stage=stage,
                gate_report_hash=_verified_report_hash(report),
            )
            _safe_stage(candidate, "quarantined")
            results.append(
                NightlyResult(
                    candidate=candidate,
                    gate_report=report,
                    gate_report_path=report_relative,
                    artifact_digest=artifact_digest,
                    outcome=OUTCOME_QUARANTINED,
                    smoke_test=SMOKE_NOT_RUN,
                    pointer_target=None,
                    rollback_target=None,
                    started_at=started_at,
                    completed_at=_iso_timestamp(clock),
                    failure_stage=stage,
                )
            )
            continue

        smoke_status = SMOKE_NOT_RUN
        try:
            stage = "smoke"
            _safe_stage(candidate, stage)
            runtime.smoke(candidate)
            smoke_status = SMOKE_PASSED
            stage = "last-green"
            runtime.mark_last_green(candidate, report)
        except Exception:
            if stage == "smoke":
                smoke_status = SMOKE_FAILED
            try:
                rollback_target = runtime.rollback(candidate)
            except Exception:
                runtime.report_quarantine(
                    candidate,
                    run_id=run_id,
                    git_sha=git_sha,
                    stage="rollback",
                    gate_report_hash=_verified_report_hash(report),
                )
                _safe_stage(candidate, "rollback-failed")
                results.append(
                    NightlyResult(
                        candidate=candidate,
                        gate_report=report,
                        gate_report_path=report_relative,
                        artifact_digest=artifact_digest,
                        outcome=OUTCOME_ROLLBACK_FAILED,
                        smoke_test=smoke_status,
                        pointer_target=None,
                        rollback_target=None,
                        started_at=started_at,
                        completed_at=_iso_timestamp(clock),
                        failure_stage="rollback",
                    )
                )
                continue
            runtime.report_quarantine(
                candidate,
                run_id=run_id,
                git_sha=git_sha,
                stage=stage,
                gate_report_hash=_verified_report_hash(report),
            )
            _safe_stage(candidate, "rolled-back")
            results.append(
                NightlyResult(
                    candidate=candidate,
                    gate_report=report,
                    gate_report_path=report_relative,
                    artifact_digest=artifact_digest,
                    outcome=OUTCOME_ROLLED_BACK,
                    smoke_test=smoke_status,
                    pointer_target=rollback_target,
                    rollback_target=rollback_target,
                    started_at=started_at,
                    completed_at=_iso_timestamp(clock),
                    failure_stage=stage,
                )
            )
            continue

        _safe_stage(candidate, "published")
        results.append(
            NightlyResult(
                candidate=candidate,
                gate_report=report,
                gate_report_path=report_relative,
                artifact_digest=artifact_digest,
                outcome=OUTCOME_PUBLISHED,
                smoke_test=SMOKE_PASSED,
                pointer_target=candidate.repo_id,
                rollback_target=None,
                started_at=started_at,
                completed_at=_iso_timestamp(clock),
            )
        )

    append_nightly_records(
        results,
        run_id=run_id,
        git_sha=git_sha,
        ledger_path=ledger_path,
    )
    return results


def _load_gate_report(path: str | Path) -> GateReport:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return GateReport.from_dict(data)


def _parse_artifact(spec: str) -> tuple[str, str]:
    """Parse one ``FAMILY=PATH_OR_DIGEST`` argument into (family, digest)."""

    family, separator, value = spec.partition("=")
    if not separator or not family or not value:
        raise ReleaseManifestError(
            f"--artifact expects FAMILY=PATH_OR_DIGEST, got {spec!r}"
        )
    # A CI step that already has the digest can pass it straight through;
    # anything else is a path to the candidate artifact on disk.
    digest = value if _SHA256_DIGEST_RE.match(value) else compute_file_digest(value)
    return family, digest


def _nightly_run_main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="orchestrate.py run",
        description="Run the themed nightly model release queue.",
    )
    parser.add_argument("--queue", type=Path, default=DEFAULT_QUEUE)
    parser.add_argument("--weekday", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--git-sha", default=None)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--registry-state", type=Path, default=DEFAULT_REGISTRY_STATE)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--no-quarantine-issues", action="store_true")
    args = parser.parse_args(list(argv))

    try:
        candidates = load_nightly_queue(args.queue, weekday=args.weekday)
        if not candidates:
            print(f"nightly weekday {args.weekday.lower()}: no reviewed candidates")
            return 0
        git_sha = args.git_sha or resolve_git_sha()
        runtime = ReleaseRuntime(
            root=ROOT,
            output_root=args.output_root,
            manifest_path=args.manifest,
            registry_state_path=args.registry_state,
            baseline_path=args.baseline,
            reports_dir=args.reports_dir,
            report_issues=not args.no_quarantine_issues,
        )
        results = orchestrate_nightly(
            candidates,
            run_id=args.run_id,
            git_sha=git_sha,
            runtime=runtime,
            ledger_path=args.ledger,
            reports_dir=args.reports_dir,
        )
        audit_nightly_run(
            args.run_id,
            ledger_path=args.ledger,
            repository_root=ROOT,
        )
    except (OSError, ValueError, ReleaseManifestError):
        print("nightly release orchestration failed", file=sys.stderr)
        return 1

    published = sum(result.outcome == OUTCOME_PUBLISHED for result in results)
    quarantined = len(results) - published
    print(
        f"nightly run {args.run_id}: {published} published, {quarantined} quarantined"
    )
    return 0 if quarantined == 0 else 1


def _nightly_audit_main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="orchestrate.py audit",
        description="Audit one nightly run entirely from committed files.",
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--repository-root", type=Path, default=ROOT)
    args = parser.parse_args(list(argv))
    try:
        outcome = audit_nightly_run(
            args.run_id,
            ledger_path=args.ledger,
            repository_root=args.repository_root,
        )
    except ReleaseManifestError:
        print("nightly release audit failed", file=sys.stderr)
        return 1
    print(json.dumps(outcome, ensure_ascii=True, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    """Assemble and append a release run record from gate-report files."""

    resolved_argv = list(sys.argv[1:] if argv is None else argv)
    if resolved_argv and resolved_argv[0] == "run":
        return _nightly_run_main(resolved_argv[1:])
    if resolved_argv and resolved_argv[0] == "audit":
        return _nightly_audit_main(resolved_argv[1:])

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-id",
        required=True,
        help="Identifier grouping this batch of per-family records.",
    )
    parser.add_argument(
        "--gate-report",
        dest="gate_reports",
        action="append",
        default=[],
        type=Path,
        metavar="PATH",
        help="Path to a serialized GateReport JSON file (repeatable).",
    )
    parser.add_argument(
        "--artifact",
        dest="artifacts",
        action="append",
        default=[],
        metavar="FAMILY=PATH_OR_DIGEST",
        help=(
            "Candidate artifact for a family, as a path or a sha256: digest "
            "(repeatable). Every gate report's family needs one."
        ),
    )
    parser.add_argument(
        "--git-sha",
        help="Release git SHA; defaults to the resolved repository SHA.",
    )
    parser.add_argument(
        "--ledger",
        type=Path,
        default=DEFAULT_LEDGER,
        help=f"Append-only run ledger (default: {DEFAULT_LEDGER}).",
    )
    args = parser.parse_args(resolved_argv)

    if not args.gate_reports:
        parser.error("at least one --gate-report is required")

    try:
        reports = [_load_gate_report(path) for path in args.gate_reports]
        artifact_digests = dict(_parse_artifact(spec) for spec in args.artifacts)
        created_at = datetime.now(timezone.utc).isoformat()
        records = build_release_manifest(
            reports,
            run_id=args.run_id,
            created_at=created_at,
            artifact_digests=artifact_digests,
            git_sha=args.git_sha,
            ledger_path=args.ledger,
        )
    except (FileNotFoundError, json.JSONDecodeError, ReleaseManifestError) as exc:
        print(f"release manifest failed: {exc}", file=sys.stderr)
        return 1

    published = [r["family"] for r in records if not r["quarantined"]]
    quarantined = [r["family"] for r in records if r["quarantined"]]
    print(
        f"run {args.run_id}: {len(records)} record(s) -> "
        f"{len(published)} published, {len(quarantined)} quarantined"
    )
    for family in quarantined:
        print(f"  quarantined: {family}", file=sys.stderr)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
