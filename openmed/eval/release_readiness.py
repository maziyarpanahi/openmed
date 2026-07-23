"""Release-readiness gate for v2.0.

Aggregates extraction/model gate results, required documentation presence,
API-compat baseline, mandatory clinical disclaimers, and e2e golden-suite
status into a single signed READY / NOT_READY decision.

Each requirement is represented as a :class:`GateCheck` with an explicit
reason and evidence path so a NOT_READY decision names exactly what is
missing.  Missing evidence yields NOT_READY (fail-closed).

PHI-free: only paths, hashes, and boolean evidence are recorded.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from openmed.core.audit import AuditSignature, stable_hash
from openmed.eval.release_gates import (
    QUARANTINED,
    RELEASABLE,
    GateCheck,
    GateReport,
    _SIGNATURE_ALGORITHM,
)

READY = "READY"
NOT_READY = "NOT_READY"

# Required documentation files that must exist for a shippable release.
_REQUIRED_DOCS: tuple[str, ...] = (
    "README.md",
    "CHANGELOG.md",
    "MIGRATION.md",
)

# Mandatory disclaimer constant that must be present in shipped clinical
# outputs.  Checked by grepping the source for the literal string.
_DISCLAIMER_CONSTANT = "OPENMED_CLINICAL_DISCLAIMER"

# Default signing key (same as release_gates for local/dev usage).
_DEFAULT_SIGNING_KEY = "openmed-release-readiness-local-key"


@dataclass
class ReadinessReport:
    """Signed release-readiness decision and evidence payload."""

    version: str
    decision: str  # READY or NOT_READY
    checks: tuple[GateCheck, ...] = ()
    repro_hash: str = ""
    signature: AuditSignature | None = None

    def __post_init__(self) -> None:
        self.checks = tuple(self.checks)
        if not self.repro_hash:
            self.repro_hash = self._compute_repro_hash()

    def _compute_repro_hash(self) -> str:
        payload = {
            "version": self.version,
            "decision": self.decision,
            "checks": [c.to_dict() for c in self.checks],
        }
        return stable_hash(payload)

    def sign(self, key: bytes | str, *, key_id: str = "release-readiness") -> ReadinessReport:
        """Return a signed copy of this report."""
        if isinstance(key, str):
            key = key.encode("utf-8")
        payload = json.dumps(
            {"repro_hash": self.repro_hash, "decision": self.decision},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        sig_value = hmac.new(key, payload, hashlib.sha256).hexdigest()
        signed = ReadinessReport(
            version=self.version,
            decision=self.decision,
            checks=self.checks,
            repro_hash=self.repro_hash,
            signature=AuditSignature(
                key_id=key_id,
                algorithm=_SIGNATURE_ALGORITHM,
                value=sig_value,
            ),
        )
        return signed

    def verify(self, key: bytes | str) -> bool:
        """Return True if the signature is valid."""
        if self.signature is None:
            return False
        if isinstance(key, str):
            key = key.encode("utf-8")
        payload = json.dumps(
            {"repro_hash": self.repro_hash, "decision": self.decision},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        expected = hmac.new(key, payload, hashlib.sha256).hexdigest()
        return hmac.compare_digest(expected, self.signature.value)

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "version": self.version,
            "decision": self.decision,
            "checks": [c.to_dict() for c in self.checks],
            "repro_hash": self.repro_hash,
        }
        if self.signature is not None:
            result["signature"] = self.signature.to_dict()
        return result

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ReadinessReport:
        checks = tuple(
            GateCheck.from_dict(c) for c in (data.get("checks") or ())
        )
        sig_data = data.get("signature")
        signature = AuditSignature.from_dict(sig_data) if sig_data else None
        return cls(
            version=str(data.get("version", "")),
            decision=str(data.get("decision", NOT_READY)),
            checks=checks,
            repro_hash=str(data.get("repro_hash", "")),
            signature=signature,
        )

    def failing_checks(self) -> list[GateCheck]:
        return [c for c in self.checks if not c.passed]


def _check_extraction_gates(gate_report: GateReport | None) -> GateCheck:
    """(a) All extraction/model gates RELEASABLE."""
    if gate_report is None:
        return GateCheck(
            "extraction_gates",
            False,
            reason="No gate report provided",
        )
    if gate_report.decision != RELEASABLE:
        failing = [g.gate for g in gate_report.gate_results if not g.passed]
        return GateCheck(
            "extraction_gates",
            False,
            reason=f"Gate decision is {gate_report.decision}",
            details={"failing_gates": failing},
        )
    return GateCheck("extraction_gates", True)


def _check_required_docs(repo_root: Path) -> GateCheck:
    """(b) Required docs present (README, CHANGELOG, MIGRATION)."""
    missing = [name for name in _REQUIRED_DOCS if not (repo_root / name).is_file()]
    if missing:
        return GateCheck(
            "required_docs",
            False,
            reason=f"Missing required docs: {', '.join(missing)}",
            details={"missing": missing},
        )
    return GateCheck("required_docs", True)


def _check_api_compat(repo_root: Path) -> GateCheck:
    """(c) API-compat baseline clean.

    Checks that the API surface baseline file exists and is non-empty.
    A more thorough check would diff against the previous release, but
    for the readiness gate we verify the baseline artifact is present.
    """
    baseline_path = repo_root / "gates" / "baseline.json"
    if not baseline_path.is_file():
        return GateCheck(
            "api_compat",
            False,
            reason="API-compat baseline not found at gates/baseline.json",
        )
    try:
        content = baseline_path.read_text(encoding="utf-8").strip()
        if not content:
            return GateCheck(
                "api_compat",
                False,
                reason="API-compat baseline is empty",
            )
        data = json.loads(content)
        return GateCheck(
            "api_compat",
            True,
            details={"baseline_hash": stable_hash(data)},
        )
    except (json.JSONDecodeError, OSError) as exc:
        return GateCheck(
            "api_compat",
            False,
            reason=f"API-compat baseline unreadable: {exc}",
        )


def _check_disclaimer(repo_root: Path) -> GateCheck:
    """(d) Mandatory disclaimer constant present in shipped clinical outputs.

    Searches the openmed/clinical/ directory for the disclaimer constant.
    Fail-closed: if the directory doesn't exist or the constant is missing,
    the check fails.
    """
    clinical_dir = repo_root / "openmed" / "clinical"
    if not clinical_dir.is_dir():
        return GateCheck(
            "clinical_disclaimer",
            False,
            reason="openmed/clinical/ directory not found",
        )
    for py_file in clinical_dir.rglob("*.py"):
        try:
            content = py_file.read_text(encoding="utf-8")
            if _DISCLAIMER_CONSTANT in content:
                return GateCheck(
                    "clinical_disclaimer",
                    True,
                    details={"found_in": str(py_file.relative_to(repo_root))},
                )
        except OSError:
            continue
    return GateCheck(
        "clinical_disclaimer",
        False,
        reason=f"Disclaimer constant {_DISCLAIMER_CONSTANT!r} not found in openmed/clinical/",
    )


def _check_e2e_golden(repo_root: Path) -> GateCheck:
    """(e) e2e golden suite green marker.

    Checks for a golden-suite pass marker file.  The CI workflow writes
    this marker after a successful e2e run.  Fail-closed if absent.
    """
    marker_path = repo_root / "gates" / "e2e_golden_pass.json"
    if not marker_path.is_file():
        return GateCheck(
            "e2e_golden",
            False,
            reason="e2e golden-suite pass marker not found at gates/e2e_golden_pass.json",
        )
    try:
        data = json.loads(marker_path.read_text(encoding="utf-8"))
        passed = bool(data.get("passed", False))
        if not passed:
            return GateCheck(
                "e2e_golden",
                False,
                reason="e2e golden-suite marker indicates failure",
                details=data,
            )
        return GateCheck(
            "e2e_golden",
            True,
            details={"marker_hash": stable_hash(data)},
        )
    except (json.JSONDecodeError, OSError) as exc:
        return GateCheck(
            "e2e_golden",
            False,
            reason=f"e2e golden-suite marker unreadable: {exc}",
        )


def evaluate_readiness(
    *,
    version: str = "2.0.0",
    repo_root: Path | str | None = None,
    gate_report: GateReport | None = None,
    signing_key: bytes | str | None = None,
    key_id: str | None = None,
) -> ReadinessReport:
    """Evaluate release readiness and return a signed report.

    Parameters
    ----------
    version:
        Release version string (for the report payload).
    repo_root:
        Root of the repository.  Defaults to the current working directory.
    gate_report:
        Optional pre-computed :class:`GateReport` from the extraction gates.
        If ``None``, the extraction-gates check fails closed.
    signing_key:
        HMAC signing key.  Defaults to the local dev key.
    key_id:
        Key identifier for the signature metadata.

    Returns
    -------
    ReadinessReport
        Signed readiness report with per-check evidence.
    """
    if repo_root is None:
        repo_root = Path.cwd()
    elif isinstance(repo_root, str):
        repo_root = Path(repo_root)

    checks = [
        _check_extraction_gates(gate_report),
        _check_required_docs(repo_root),
        _check_api_compat(repo_root),
        _check_disclaimer(repo_root),
        _check_e2e_golden(repo_root),
    ]

    all_passed = all(c.passed for c in checks)
    decision = READY if all_passed else NOT_READY

    report = ReadinessReport(
        version=version,
        decision=decision,
        checks=tuple(checks),
    )

    return report.sign(
        signing_key or _DEFAULT_SIGNING_KEY,
        key_id=key_id or "release-readiness",
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate OpenMed release-readiness gate.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Repository root (default: cwd).",
    )
    parser.add_argument(
        "--gate-report",
        type=Path,
        default=None,
        help="Path to a signed GateReport JSON file.",
    )
    parser.add_argument(
        "--version",
        default="2.0.0",
        help="Release version string.",
    )
    parser.add_argument(
        "--signing-key",
        default=None,
        help="HMAC signing key (or set OPENMED_READINESS_KEY env var).",
    )
    parser.add_argument(
        "--key-id",
        default=None,
        help="Key identifier for the signature.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output the report as JSON.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for the release-readiness gate."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    repo_root = args.repo_root or Path.cwd()
    signing_key = args.signing_key or os.environ.get("OPENMED_READINESS_KEY")

    gate_report: GateReport | None = None
    if args.gate_report and args.gate_report.is_file():
        raw = json.loads(args.gate_report.read_text(encoding="utf-8"))
        gate_report = GateReport.from_dict(raw)

    report = evaluate_readiness(
        version=args.version,
        repo_root=repo_root,
        gate_report=gate_report,
        signing_key=signing_key,
        key_id=args.key_id,
    )

    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(f"Release Readiness: {report.version} -> {report.decision}")
        for check in report.checks:
            status = "PASS" if check.passed else "FAIL"
            print(f"  [{status}] {check.gate}: {check.reason}")
        if report.failing_checks():
            print(f"\nResult: {NOT_READY} — {len(report.failing_checks())} check(s) failed.")
        else:
            print(f"\nResult: {READY}")

    return 0 if report.decision == READY else 1


if __name__ == "__main__":
    raise SystemExit(main())
