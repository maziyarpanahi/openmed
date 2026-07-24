"""Fail-closed release-readiness aggregation for OpenMed releases.

The gate combines signed model-gate evidence, required release documentation,
an API-surface compatibility report, the public clinical disclaimer, and a
workflow-produced golden-suite result into one signed ``READY`` or
``NOT_READY`` decision. Evidence contains only paths, hashes, counts, and
booleans.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import hmac
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from openmed.core.audit import AuditSignature, stable_hash
from openmed.eval.release_gates import (
    _SIGNATURE_ALGORITHM,
    RELEASABLE,
    GateCheck,
    GateReport,
)

READY = "READY"
NOT_READY = "NOT_READY"

_REQUIRED_DOCS: tuple[str, ...] = ("README.md", "CHANGELOG.md")
_DEFAULT_MIGRATION_GUIDE = Path("docs/migration/1.9-to-2.0.md")
_DEFAULT_API_COMPAT_REPORT = Path("gates/api_compat_report.json")
_DEFAULT_E2E_REPORT = Path("gates/e2e_golden_pass.json")
_DISCLAIMER_MODULE = Path("openmed/clinical/__init__.py")
_DISCLAIMER_CONSTANT = "OPENMED_CLINICAL_DISCLAIMER"
_DEFAULT_GATE_SIGNING_KEY = "openmed-release-gate-local-key"
_DEFAULT_READINESS_SIGNING_KEY = "openmed-release-readiness-local-key"


@dataclass
class ReadinessReport:
    """Signed release-readiness decision and PHI-free evidence payload."""

    version: str
    decision: str
    checks: tuple[GateCheck, ...] = ()
    repro_hash: str = ""
    signature: AuditSignature | None = None

    def __post_init__(self) -> None:
        self.checks = tuple(self.checks)
        if not self.repro_hash:
            self.repro_hash = self.recompute_repro_hash()

    def _payload(
        self,
        *,
        include_repro_hash: bool,
        include_signature: bool,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "version": self.version,
            "decision": self.decision,
            "checks": [check.to_dict() for check in self.checks],
        }
        if include_repro_hash:
            payload["repro_hash"] = self.repro_hash
        if include_signature:
            payload["signature"] = (
                self.signature.to_dict() if self.signature is not None else None
            )
        return payload

    def recompute_repro_hash(self) -> str:
        """Recompute the evidence hash without trusting the stored value."""

        return stable_hash(
            self._payload(include_repro_hash=False, include_signature=False)
        )

    def sign(
        self,
        key: bytes | str,
        *,
        key_id: str = "release-readiness",
    ) -> ReadinessReport:
        """Return a signed copy whose signature covers all report evidence."""

        signed = ReadinessReport(
            version=self.version,
            decision=self.decision,
            checks=self.checks,
        )
        message = _canonical_json(
            signed._payload(include_repro_hash=True, include_signature=False)
        ).encode("utf-8")
        signed.signature = AuditSignature(
            key_id=key_id,
            algorithm=_SIGNATURE_ALGORITHM,
            value=hmac.new(_key_bytes(key), message, hashlib.sha256).hexdigest(),
        )
        return signed

    def verify(self, key: bytes | str) -> bool:
        """Return whether both the evidence hash and signature are valid."""

        if self.recompute_repro_hash() != self.repro_hash:
            return False
        if self.signature is None or self.signature.algorithm != _SIGNATURE_ALGORITHM:
            return False
        message = _canonical_json(
            self._payload(include_repro_hash=True, include_signature=False)
        ).encode("utf-8")
        expected = hmac.new(_key_bytes(key), message, hashlib.sha256).hexdigest()
        return hmac.compare_digest(expected, self.signature.value)

    def to_dict(self) -> dict[str, Any]:
        """Return the stable JSON-compatible report payload."""

        return self._payload(include_repro_hash=True, include_signature=True)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ReadinessReport:
        """Restore a readiness report from its JSON-compatible payload."""

        checks = tuple(
            GateCheck.from_dict(check) for check in (data.get("checks") or ())
        )
        signature_data = data.get("signature")
        return cls(
            version=str(data.get("version", "")),
            decision=str(data.get("decision", NOT_READY)),
            checks=checks,
            repro_hash=str(data.get("repro_hash", "")),
            signature=(
                AuditSignature.from_dict(signature_data)
                if isinstance(signature_data, Mapping)
                else None
            ),
        )

    def failing_checks(self) -> list[GateCheck]:
        """Return failed component checks in report order."""

        return [check for check in self.checks if not check.passed]


def _check_extraction_gates(
    gate_report: GateReport | None,
    *,
    verification_key: bytes | str,
) -> GateCheck:
    if gate_report is None:
        return GateCheck(
            "extraction_gates",
            False,
            reason="No signed extraction/model gate report was provided",
        )
    if not gate_report.verify(verification_key):
        return GateCheck(
            "extraction_gates",
            False,
            reason="Extraction/model gate report signature or evidence hash is invalid",
        )
    if gate_report.decision != RELEASABLE:
        failing = [check.gate for check in gate_report.gate_results if not check.passed]
        return GateCheck(
            "extraction_gates",
            False,
            reason=f"Gate decision is {gate_report.decision}",
            details={"failing_gates": failing},
        )
    return GateCheck(
        "extraction_gates",
        True,
        details={
            "gate_report_hash": gate_report.repro_hash,
            "key_id": gate_report.signature.key_id,
        },
    )


def _check_required_docs(repo_root: Path, migration_guide: Path) -> GateCheck:
    required = [repo_root / name for name in _REQUIRED_DOCS]
    required.append(_resolve_path(repo_root, migration_guide))
    missing = [
        _display_path(path, repo_root) for path in required if not path.is_file()
    ]
    if missing:
        return GateCheck(
            "required_docs",
            False,
            reason=f"Missing required docs: {', '.join(missing)}",
            details={"missing": missing},
        )
    return GateCheck(
        "required_docs",
        True,
        details={
            "files": {
                _display_path(path, repo_root): _file_hash(path) for path in required
            }
        },
    )


def _check_api_compat(repo_root: Path, report_path: Path) -> GateCheck:
    path = _resolve_path(repo_root, report_path)
    if not path.is_file():
        return GateCheck(
            "api_compat",
            False,
            reason=f"API-compat report not found at {_display_path(path, repo_root)}",
        )
    try:
        data = _read_json_object(path)
        summary = data.get("summary")
        if data.get("schema_version") != 1 or not isinstance(summary, Mapping):
            raise ValueError("unsupported or missing API-compat report schema")
        breaking = int(summary.get("breaking", -1))
        if breaking != 0:
            return GateCheck(
                "api_compat",
                False,
                reason=f"API-compat report contains {breaking} breaking change(s)",
                details={
                    "report_hash": _file_hash(path),
                    "breaking": breaking,
                },
            )
        return GateCheck(
            "api_compat",
            True,
            details={
                "report_hash": _file_hash(path),
                "before_ref": str(data.get("before_ref", "")),
                "after_ref": str(data.get("after_ref", "")),
                "breaking": 0,
            },
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return GateCheck(
            "api_compat",
            False,
            reason=f"API-compat report is invalid: {exc}",
        )


def _check_disclaimer(repo_root: Path) -> GateCheck:
    path = repo_root / _DISCLAIMER_MODULE
    if not path.is_file():
        return GateCheck(
            "clinical_disclaimer",
            False,
            reason=f"{_DISCLAIMER_MODULE} was not found",
        )
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError) as exc:
        return GateCheck(
            "clinical_disclaimer",
            False,
            reason=f"Clinical disclaimer module is unreadable: {exc}",
        )

    value = _literal_assignment(tree, _DISCLAIMER_CONSTANT)
    if not isinstance(value, str) or not value.strip():
        return GateCheck(
            "clinical_disclaimer",
            False,
            reason=(
                f"Public constant {_DISCLAIMER_CONSTANT} is not assigned a "
                f"non-empty string in {_DISCLAIMER_MODULE}"
            ),
        )
    exports = _literal_assignment(tree, "__all__")
    if not isinstance(exports, (list, tuple)) or _DISCLAIMER_CONSTANT not in exports:
        return GateCheck(
            "clinical_disclaimer",
            False,
            reason=f"Public exports in {_DISCLAIMER_MODULE} omit {_DISCLAIMER_CONSTANT}",
        )
    return GateCheck(
        "clinical_disclaimer",
        True,
        details={
            "source_path": str(_DISCLAIMER_MODULE),
            "disclaimer_hash": stable_hash(value),
        },
    )


def _check_e2e_golden(repo_root: Path, report_path: Path) -> GateCheck:
    path = _resolve_path(repo_root, report_path)
    if not path.is_file():
        return GateCheck(
            "e2e_golden",
            False,
            reason=f"Golden-suite report not found at {_display_path(path, repo_root)}",
        )
    try:
        data = _read_json_object(path)
        passed = data.get("passed") is True
        suite = str(data.get("suite", "")).strip()
        if not passed or not suite:
            return GateCheck(
                "e2e_golden",
                False,
                reason="Golden-suite report does not record a named passing suite",
                details={"passed": passed, "suite": suite},
            )
        return GateCheck(
            "e2e_golden",
            True,
            details={
                "report_hash": _file_hash(path),
                "suite": suite,
            },
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return GateCheck(
            "e2e_golden",
            False,
            reason=f"Golden-suite report is invalid: {exc}",
        )


def evaluate_readiness(
    *,
    version: str = "2.0.0",
    repo_root: Path | str | None = None,
    gate_report: GateReport | None = None,
    gate_report_key: bytes | str | None = None,
    migration_guide: Path | str = _DEFAULT_MIGRATION_GUIDE,
    api_compat_report: Path | str = _DEFAULT_API_COMPAT_REPORT,
    e2e_report: Path | str = _DEFAULT_E2E_REPORT,
    signing_key: bytes | str | None = None,
    key_id: str | None = None,
) -> ReadinessReport:
    """Evaluate all release requirements and return a signed report.

    Args:
        version: Release version recorded in the report.
        repo_root: Repository root. Defaults to the current directory.
        gate_report: Signed extraction/model ``GateReport``.
        gate_report_key: Key used to verify ``gate_report``.
        migration_guide: Required migration-guide path.
        api_compat_report: Machine-readable API-surface diff path.
        e2e_report: Workflow-produced golden-suite result path.
        signing_key: Key used to sign the readiness report.
        key_id: Identifier recorded with the readiness signature.

    Returns:
        A signed, PHI-free readiness report.
    """

    root = Path(repo_root or Path.cwd())
    verification_key = (
        gate_report_key
        or os.environ.get("OPENMED_RELEASE_GATE_KEY")
        or _DEFAULT_GATE_SIGNING_KEY
    )
    readiness_key = (
        signing_key
        or os.environ.get("OPENMED_READINESS_KEY")
        or _DEFAULT_READINESS_SIGNING_KEY
    )
    checks = (
        _check_extraction_gates(
            gate_report,
            verification_key=verification_key,
        ),
        _check_required_docs(root, Path(migration_guide)),
        _check_api_compat(root, Path(api_compat_report)),
        _check_disclaimer(root),
        _check_e2e_golden(root, Path(e2e_report)),
    )
    report = ReadinessReport(
        version=version,
        decision=READY if all(check.passed for check in checks) else NOT_READY,
        checks=checks,
    )
    return report.sign(readiness_key, key_id=key_id or "release-readiness")


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the release-readiness command-line parser."""

    parser = argparse.ArgumentParser(
        description="Evaluate the fail-closed OpenMed release-readiness gate."
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--gate-report", type=Path)
    parser.add_argument("--gate-report-key")
    parser.add_argument(
        "--migration-guide",
        type=Path,
        default=_DEFAULT_MIGRATION_GUIDE,
    )
    parser.add_argument(
        "--api-compat-report",
        type=Path,
        default=_DEFAULT_API_COMPAT_REPORT,
    )
    parser.add_argument("--e2e-report", type=Path, default=_DEFAULT_E2E_REPORT)
    parser.add_argument("--version", default="2.0.0")
    parser.add_argument("--signing-key")
    parser.add_argument("--key-id", default="release-readiness")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the release-readiness CLI and fail closed on missing evidence."""

    parser = build_arg_parser()
    args = parser.parse_args(argv)
    gate_report: GateReport | None = None
    try:
        if args.gate_report is not None:
            gate_report = GateReport.from_dict(_read_json_object(args.gate_report))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"release-readiness: invalid gate report: {exc}", file=sys.stderr)
        return 2

    report = evaluate_readiness(
        version=args.version,
        repo_root=args.repo_root,
        gate_report=gate_report,
        gate_report_key=args.gate_report_key,
        migration_guide=args.migration_guide,
        api_compat_report=args.api_compat_report,
        e2e_report=args.e2e_report,
        signing_key=args.signing_key,
        key_id=args.key_id,
    )
    rendered = json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")

    if args.json:
        print(rendered, end="")
    else:
        print(f"Release Readiness: {report.version} -> {report.decision}")
        for check in report.checks:
            status = "PASS" if check.passed else "FAIL"
            print(f"  [{status}] {check.gate}: {check.reason}")
    return 0 if report.decision == READY else 1


def _literal_assignment(tree: ast.Module, name: str) -> Any:
    for statement in tree.body:
        value: ast.expr | None = None
        if isinstance(statement, ast.Assign):
            if any(
                isinstance(target, ast.Name) and target.id == name
                for target in statement.targets
            ):
                value = statement.value
        elif (
            isinstance(statement, ast.AnnAssign)
            and isinstance(statement.target, ast.Name)
            and statement.target.id == name
        ):
            value = statement.value
        if value is not None:
            try:
                return ast.literal_eval(value)
            except (ValueError, SyntaxError):
                return None
    return None


def _resolve_path(repo_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def _display_path(path: Path, repo_root: Path) -> str:
    try:
        return path.relative_to(repo_root).as_posix()
    except ValueError:
        return path.as_posix()


def _read_json_object(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        raise ValueError(f"{path} must contain a JSON object")
    return dict(data)


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_json(data: Any) -> str:
    return json.dumps(
        data,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _key_bytes(key: bytes | str) -> bytes:
    if isinstance(key, bytes):
        return key
    if isinstance(key, str):
        return key.encode("utf-8")
    raise TypeError("signing key must be bytes or str")


if __name__ == "__main__":
    raise SystemExit(main())
