#!/usr/bin/env python3
"""Build the release run-ledger that binds artifacts to their gate reports.

Epics OM-047 and OM-748 require that every published artifact provably passed
the gate it claims, via an append-only run record linking the artifact to its
``RELEASABLE`` :class:`~openmed.eval.release_gates.GateReport` by hash. This
module delivers the release-manifest / run-ledger builder (not the nightly
scheduler or the Hugging Face publish step): a run outcome is reconstructable
from committed state with no raw PHI.

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
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from openmed.core.repro_hash import (
    compute_canonical_payload_hash,
    compute_file_digest,
    resolve_git_sha,
)
from openmed.eval.release_gates import RELEASABLE, GateReport

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LEDGER = ROOT / "gates" / "release_runs.jsonl"

#: Ledger fields that identity-bind an artifact to its gate report. The
#: provenance hash covers exactly these -- at minimum the candidate artifact
#: digest and the verified gate report hash -- so the same artifact + gate
#: report + publish target always hashes identically, independent of run id or
#: wall clock.
_BINDING_FIELDS = (
    "family",
    "tier",
    "format",
    "repo_id",
    "git_sha",
    "artifact_digest",
    "gate_report_hash",
    "decision",
    "pointer_target",
)

_SHA256_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")

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


def _binding_hash(record: Mapping[str, Any]) -> str:
    """Return the ``sha256:`` provenance hash binding artifact to gate report."""

    payload = {field: record.get(field) for field in _BINDING_FIELDS}
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
    for key, value in record.items():
        if key in _HASH_FIELDS:
            continue
        if isinstance(value, str):
            yield value


def _assert_no_phi(record: Mapping[str, Any]) -> None:
    """Reject a record whose values look like raw PHI before it is written."""

    for value in _iter_record_strings(record):
        for pattern in _PHI_PATTERNS:
            if pattern.search(value):
                raise ReleaseManifestError(
                    "refusing to write PHI-shaped value to the release ledger"
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
    _assert_no_phi(record)
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
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def verify_record(record: Mapping[str, Any]) -> bool:
    """Return True if a record's provenance hash matches a recomputation."""

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
        if row.get("run_id") != run_id:
            continue
        if not verify_record(row):
            raise ReleaseManifestError(
                f"provenance hash mismatch for family {row.get('family')!r} "
                f"in run {run_id!r}"
            )
        outcome[str(row["family"])] = {
            "published": not row["quarantined"],
            "quarantined": bool(row["quarantined"]),
            "decision": row["decision"],
            "pointer_target": row["pointer_target"],
            "artifact_digest": row["artifact_digest"],
            "gate_report_hash": row["gate_report_hash"],
            "provenance_hash": row["provenance_hash"],
        }
    return outcome


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


def main(argv: list[str] | None = None) -> int:
    """Assemble and append a release run record from gate-report files."""

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
    args = parser.parse_args(argv)

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
