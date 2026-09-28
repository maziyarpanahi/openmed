"""Fail-closed v3 Journey release gate and signed evidence packet.

The gate consumes only aggregate, code-like metadata.  It binds every input to
SHA-256, verifies the release tag and checkout, evaluates ten mandatory lanes,
and signs a deterministic packet that contains no clinical source values.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import math
import os
import re
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from importlib import resources
from pathlib import Path
from typing import Any, Final

from openmed.core.audit import AuditSignature, stable_hash

JOURNEY_RELEASE_SCHEMA_VERSION: Final = "1.0.0"
JOURNEY_RELEASE_COMPATIBILITY_POLICY: Final = "same_major"
JOURNEY_RELEASE_SIGNATURE_ALGORITHM: Final = "HMAC-SHA256"
JOURNEY_RELEASE_READY: Final = "READY"
JOURNEY_RELEASE_NOT_READY: Final = "NOT_READY"

JOURNEY_RELEASE_STATES: Final = (
    "success",
    "partial",
    "unknown",
    "conflict",
    "unsupported",
    "denied",
    "failure",
)
JOURNEY_RELEASE_GATES: Final = (
    "schema",
    "provenance",
    "clinical_nlp",
    "privacy",
    "interoperability",
    "application",
    "performance",
    "recovery",
    "security",
    "licensing",
)

_REPRODUCTION_COMMAND = (
    "python scripts/release/journey_release_gate.py --manifest "
    "journey-release-manifest.json --output journey-release-packet.json"
)
_SIGNING_ENV = "OPENMED_JOURNEY_RELEASE_KEY"
_SCHEMA_PACKAGE = "openmed.core.schemas.json"
_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_SEMVER_RE = re.compile(
    r"^(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)"
    r"(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?$"
)
_CODE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:+/@-]{0,127}$")
_PYTHON_VERSION_RE = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+$")

_MANIFEST_FIELDS = frozenset(
    {
        "claims",
        "compatibility_policy",
        "environment",
        "exceptions",
        "frozen_inputs",
        "gate_reports",
        "licenses",
        "limitations",
        "policy",
        "release",
        "schema_version",
    }
)
_RELEASE_FIELDS = frozenset(
    {"evaluated_at", "git_commit", "git_tag", "source_date_epoch", "version"}
)
_ENVIRONMENT_FIELDS = frozenset(
    {"architecture", "hardware_id", "os", "python_implementation", "python_version"}
)
_POLICY_FIELDS = frozenset({"max_artifact_age_seconds"})
_INPUT_FIELDS = frozenset({"id", "origin", "path", "sha256"})
_REPORT_FIELDS = frozenset(
    {
        "compatibility_policy",
        "gate",
        "generated_at",
        "metrics",
        "schema_version",
        "state",
    }
)
_LICENSE_FIELDS = frozenset(
    {
        "asset_id",
        "asset_type",
        "distribution",
        "license_id",
        "redistributable",
        "use",
    }
)
_EXCEPTION_FIELDS = frozenset({"code", "disposition", "expires_at", "gate", "severity"})

_INPUT_ORIGINS = frozenset({"aggregate", "configuration", "software", "synthetic"})
_ASSET_TYPES = frozenset({"dataset", "model", "software", "vocabulary"})
_ASSET_USES = frozenset({"build", "eval", "runtime", "train"})
_DISTRIBUTIONS = frozenset(
    {"bundled", "metadata_only", "out_of_process", "user_supplied"}
)
_SEVERITIES = frozenset({"low", "medium", "high", "critical"})
_DISPOSITIONS = frozenset({"accepted", "expired", "pending", "rejected"})

_REQUIRED_CLAIMS = frozenset(
    {
        "human_review_for_high_risk",
        "local_first",
        "offline_after_assets_downloaded",
        "synthetic_release_evidence",
        "typed_failure_states",
    }
)
_REQUIRED_LIMITATIONS = frozenset(
    {
        "no_autonomous_clinical_action",
        "not_a_medical_device",
        "not_clinically_validated",
        "requires_local_validation",
        "restricted_assets_user_supplied",
    }
)

_METRIC_FIELDS: Final[Mapping[str, frozenset[str]]] = {
    "schema": frozenset(
        {
            "invalid_evidence_count",
            "invalid_schema_count",
            "invalid_span_count",
            "persisted_schema_count",
            "public_schema_count",
        }
    ),
    "provenance": frozenset(
        {"artifact_count", "broken_link_count", "unhashed_artifact_count"}
    ),
    "clinical_nlp": frozenset(
        {
            "abstention_failure_count",
            "abstention_required_count",
            "evaluated_case_count",
            "unreviewed_high_risk_count",
        }
    ),
    "privacy": frozenset(
        {"critical_leakage_count", "evaluated_case_count", "raw_value_finding_count"}
    ),
    "interoperability": frozenset(
        {
            "conformance_failure_count",
            "evaluated_exchange_count",
            "roundtrip_loss_without_disclosure_count",
        }
    ),
    "application": frozenset(
        {
            "failed_test_count",
            "prohibited_public_claim_count",
            "test_count",
            "untyped_terminal_state_count",
        }
    ),
    "performance": frozenset(
        {
            "concurrency",
            "dataset_digest",
            "latency_p50_ms",
            "latency_p95_ms",
            "latency_p99_ms",
            "peak_memory_mib",
            "quantization",
            "sample_count",
            "slo_breach_count",
            "throughput_per_second",
            "warmup_count",
        }
    ),
    "recovery": frozenset(
        {
            "migration_failure_count",
            "non_idempotent_replay_count",
            "recovery_failure_count",
            "scenario_count",
        }
    ),
    "security": frozenset(
        {
            "finding_count",
            "threat_case_count",
            "unmitigated_threat_count",
            "unresolved_critical_finding_count",
            "unresolved_high_finding_count",
        }
    ),
    "licensing": frozenset(
        {
            "data_asset_count",
            "model_count",
            "prohibited_distribution_count",
            "unchecked_asset_count",
        }
    ),
}

_ZERO_BLOCKERS: Final[Mapping[str, tuple[str, ...]]] = {
    "schema": (
        "invalid_schema_count",
        "invalid_span_count",
        "invalid_evidence_count",
    ),
    "provenance": ("broken_link_count", "unhashed_artifact_count"),
    "clinical_nlp": ("unreviewed_high_risk_count", "abstention_failure_count"),
    "privacy": ("critical_leakage_count", "raw_value_finding_count"),
    "interoperability": (
        "conformance_failure_count",
        "roundtrip_loss_without_disclosure_count",
    ),
    "application": (
        "failed_test_count",
        "untyped_terminal_state_count",
        "prohibited_public_claim_count",
    ),
    "performance": ("slo_breach_count",),
    "recovery": (
        "non_idempotent_replay_count",
        "migration_failure_count",
        "recovery_failure_count",
    ),
    "security": (
        "unresolved_critical_finding_count",
        "unresolved_high_finding_count",
        "unmitigated_threat_count",
    ),
    "licensing": ("unchecked_asset_count", "prohibited_distribution_count"),
}

_POSITIVE_COUNTS: Final[Mapping[str, tuple[str, ...]]] = {
    "schema": ("public_schema_count", "persisted_schema_count"),
    "provenance": ("artifact_count",),
    "clinical_nlp": ("evaluated_case_count",),
    "privacy": ("evaluated_case_count",),
    "interoperability": ("evaluated_exchange_count",),
    "application": ("test_count",),
    "performance": ("concurrency", "sample_count"),
    "recovery": ("scenario_count",),
    "security": ("threat_case_count",),
    "licensing": ("model_count", "data_asset_count"),
}


class JourneyReleaseError(ValueError):
    """Raised when release evidence violates the strict input contract."""


@dataclass(frozen=True)
class JourneyGateResult:
    """Value-free decision for one mandatory release lane."""

    gate: str
    state: str
    passed: bool
    artifact_digest: str
    artifact_age_seconds: int
    metrics: Mapping[str, int | float | str]
    blocking_codes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return the deterministic JSON representation."""

        return {
            "artifact_age_seconds": self.artifact_age_seconds,
            "artifact_digest": self.artifact_digest,
            "blocking_codes": list(self.blocking_codes),
            "gate": self.gate,
            "metrics": dict(sorted(self.metrics.items())),
            "passed": self.passed,
            "state": self.state,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> JourneyGateResult:
        """Restore one gate result from a packet payload."""

        metrics = payload.get("metrics")
        if not isinstance(metrics, Mapping):
            raise JourneyReleaseError("packet gate metrics must be an object")
        return cls(
            gate=str(payload.get("gate", "")),
            state=str(payload.get("state", "failure")),
            passed=payload.get("passed") is True,
            artifact_digest=str(payload.get("artifact_digest", "")),
            artifact_age_seconds=_require_int(
                payload.get("artifact_age_seconds"),
                "artifact_age_seconds",
                minimum=0,
            ),
            metrics={str(key): value for key, value in metrics.items()},
            blocking_codes=tuple(
                _require_code(value, "blocking code")
                for value in _require_list(
                    payload.get("blocking_codes"), "blocking_codes"
                )
            ),
        )


@dataclass(frozen=True)
class JourneyReleasePacket:
    """Signed, reproducible and value-free Journey release decision."""

    release: Mapping[str, Any]
    environment: Mapping[str, str]
    decision: str
    repository_binding: Mapping[str, Any]
    frozen_inputs: tuple[Mapping[str, Any], ...]
    gates: tuple[JourneyGateResult, ...]
    license_summary: Mapping[str, Any]
    claims: tuple[str, ...]
    limitations: tuple[str, ...]
    exceptions: tuple[Mapping[str, str], ...]
    manifest_digest: str
    reproduction: Mapping[str, Any]
    schema_version: str = JOURNEY_RELEASE_SCHEMA_VERSION
    compatibility_policy: str = JOURNEY_RELEASE_COMPATIBILITY_POLICY
    packet_digest: str = ""
    signature: AuditSignature | None = None

    def _unsigned_payload(self) -> dict[str, Any]:
        return {
            "claims": list(self.claims),
            "compatibility_policy": self.compatibility_policy,
            "decision": self.decision,
            "environment": dict(sorted(self.environment.items())),
            "exceptions": [dict(item) for item in self.exceptions],
            "frozen_inputs": [dict(item) for item in self.frozen_inputs],
            "gates": [gate.to_dict() for gate in self.gates],
            "license_summary": dict(self.license_summary),
            "limitations": list(self.limitations),
            "manifest_digest": self.manifest_digest,
            "release": dict(self.release),
            "repository_binding": dict(self.repository_binding),
            "reproduction": dict(self.reproduction),
            "schema_version": self.schema_version,
        }

    def recompute_packet_digest(self) -> str:
        """Recompute the canonical digest without trusting the stored value."""

        return stable_hash(self._unsigned_payload())

    def sign(self, key: bytes | str, *, key_id: str) -> JourneyReleasePacket:
        """Return a copy signed over every packet field and its digest."""

        key_bytes = _key_bytes(key)
        packet_digest = self.recompute_packet_digest()
        signed_payload = {
            **self._unsigned_payload(),
            "packet_digest": packet_digest,
        }
        value = hmac.new(
            key_bytes,
            _canonical_json(signed_payload).encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        return replace(
            self,
            packet_digest=packet_digest,
            signature=AuditSignature(
                key_id=_require_code(key_id, "key_id"),
                algorithm=JOURNEY_RELEASE_SIGNATURE_ALGORITHM,
                value=value,
            ),
        )

    def verify(self, key: bytes | str) -> bool:
        """Return whether the packet digest and HMAC signature are intact."""

        if self.schema_version != JOURNEY_RELEASE_SCHEMA_VERSION:
            return False
        if self.compatibility_policy != JOURNEY_RELEASE_COMPATIBILITY_POLICY:
            return False
        if self.recompute_packet_digest() != self.packet_digest:
            return False
        if (
            self.signature is None
            or self.signature.algorithm != JOURNEY_RELEASE_SIGNATURE_ALGORITHM
        ):
            return False
        signed_payload = {
            **self._unsigned_payload(),
            "packet_digest": self.packet_digest,
        }
        expected = hmac.new(
            _key_bytes(key),
            _canonical_json(signed_payload).encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        return hmac.compare_digest(expected, self.signature.value)

    def to_dict(self) -> dict[str, Any]:
        """Return the complete deterministic JSON representation."""

        return {
            **self._unsigned_payload(),
            "packet_digest": self.packet_digest,
            "signature": self.signature.to_dict() if self.signature else None,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> JourneyReleasePacket:
        """Restore a packet for independent signature verification."""

        signature = payload.get("signature")
        gates = payload.get("gates")
        if not isinstance(gates, list):
            raise JourneyReleaseError("packet gates must be an array")
        return cls(
            release=_require_mapping(payload.get("release"), "release"),
            environment={
                str(key): str(value)
                for key, value in _require_mapping(
                    payload.get("environment"), "environment"
                ).items()
            },
            decision=str(payload.get("decision", JOURNEY_RELEASE_NOT_READY)),
            repository_binding=_require_mapping(
                payload.get("repository_binding"), "repository_binding"
            ),
            frozen_inputs=tuple(
                _require_mapping(item, "frozen input")
                for item in _require_list(payload.get("frozen_inputs"), "frozen_inputs")
            ),
            gates=tuple(
                JourneyGateResult.from_dict(_require_mapping(item, "gate"))
                for item in gates
            ),
            license_summary=_require_mapping(
                payload.get("license_summary"), "license_summary"
            ),
            claims=tuple(
                _require_code(item, "claim")
                for item in _require_list(payload.get("claims"), "claims")
            ),
            limitations=tuple(
                _require_code(item, "limitation")
                for item in _require_list(payload.get("limitations"), "limitations")
            ),
            exceptions=tuple(
                {
                    str(key): str(value)
                    for key, value in _require_mapping(item, "exception").items()
                }
                for item in _require_list(payload.get("exceptions"), "exceptions")
            ),
            manifest_digest=str(payload.get("manifest_digest", "")),
            reproduction=_require_mapping(payload.get("reproduction"), "reproduction"),
            schema_version=str(payload.get("schema_version", "")),
            compatibility_policy=str(payload.get("compatibility_policy", "")),
            packet_digest=str(payload.get("packet_digest", "")),
            signature=(
                AuditSignature.from_dict(signature)
                if isinstance(signature, Mapping)
                else None
            ),
        )


def evaluate_journey_release(
    manifest: Mapping[str, Any],
    *,
    repo_root: str | Path,
    signing_key: bytes | str,
    key_id: str = "journey-release",
) -> JourneyReleasePacket:
    """Evaluate and sign one v3 Journey release manifest.

    The repository must be checked out at the exact commit named by the
    manifest, and the named tag must resolve to that commit.  Inputs are read
    only from portable relative paths beneath ``repo_root``.
    """

    normalized = _validate_manifest(manifest)
    root = Path(repo_root).resolve()
    release = normalized["release"]
    evaluated_at = _parse_timestamp(release["evaluated_at"], "evaluated_at")
    max_age = normalized["policy"]["max_artifact_age_seconds"]

    repository_binding = _verify_repository_binding(root, release)
    frozen_inputs, input_blockers = _verify_frozen_inputs(
        root,
        normalized["frozen_inputs"],
    )
    gate_results = _evaluate_gate_reports(
        normalized["gate_reports"],
        evaluated_at=evaluated_at,
        max_age_seconds=max_age,
    )

    license_summary, license_blockers = _evaluate_license_catalog(
        normalized["licenses"]
    )
    gate_results = _merge_gate_blockers(
        gate_results,
        "provenance",
        input_blockers,
    )
    gate_results = _merge_gate_blockers(
        gate_results,
        "licensing",
        license_blockers,
    )
    gate_results = _cross_check_license_metrics(
        gate_results,
        license_summary,
    )

    claims = tuple(sorted(set(normalized["claims"])))
    limitations = tuple(sorted(set(normalized["limitations"])))
    policy_blockers = _claim_and_limitation_blockers(claims, limitations)
    exceptions, exception_blockers = _evaluate_exceptions(
        normalized["exceptions"],
        evaluated_at=evaluated_at,
    )

    binding_ok = repository_binding["verified"] is True
    gates_ok = all(item.passed for item in gate_results)
    decision = (
        JOURNEY_RELEASE_READY
        if binding_ok
        and gates_ok
        and not input_blockers
        and not policy_blockers
        and not exception_blockers
        else JOURNEY_RELEASE_NOT_READY
    )
    release_payload = {
        "evaluated_at": release["evaluated_at"],
        "git_commit": release["git_commit"],
        "git_tag": release["git_tag"],
        "source_date_epoch": release["source_date_epoch"],
        "version": release["version"],
    }
    reproduction = {
        "command": _REPRODUCTION_COMMAND,
        "input_count": len(frozen_inputs),
        "policy_blocking_codes": list(policy_blockers),
        "exception_blocking_codes": list(exception_blockers),
    }
    packet = JourneyReleasePacket(
        release=release_payload,
        environment=normalized["environment"],
        decision=decision,
        repository_binding=repository_binding,
        frozen_inputs=frozen_inputs,
        gates=gate_results,
        license_summary=license_summary,
        claims=claims,
        limitations=limitations,
        exceptions=exceptions,
        manifest_digest=stable_hash(normalized),
        reproduction=reproduction,
    )
    return packet.sign(signing_key, key_id=key_id)


def load_journey_release_manifest(path: str | Path) -> dict[str, Any]:
    """Load a manifest while rejecting duplicate JSON keys."""

    source = Path(path)
    try:
        payload = json.loads(
            source.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_non_finite,
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise JourneyReleaseError("Journey release manifest is unreadable") from exc
    if not isinstance(payload, Mapping):
        raise JourneyReleaseError("Journey release manifest must be an object")
    return _validate_manifest(payload)


def load_journey_release_manifest_schema() -> dict[str, Any]:
    """Load the bundled strict input-manifest JSON Schema."""

    return _load_bundled_schema("journey_release_manifest.schema.json")


def load_journey_release_packet_schema() -> dict[str, Any]:
    """Load the bundled signed-packet JSON Schema."""

    return _load_bundled_schema("journey_release_packet.schema.json")


def write_journey_release_packet(
    packet: JourneyReleasePacket,
    path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Atomically write a packet, refusing replacement unless requested."""

    output = Path(path)
    if output.exists() and not overwrite:
        raise JourneyReleaseError("refusing to overwrite existing output")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    if temporary.exists():
        raise JourneyReleaseError("temporary output already exists")
    rendered = (
        json.dumps(
            packet.to_dict(),
            allow_nan=False,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(rendered)
            handle.flush()
            os.fsync(handle.fileno())
        if overwrite:
            os.replace(temporary, output)
        else:
            try:
                os.link(temporary, output)
            except FileExistsError as exc:
                raise JourneyReleaseError(
                    "refusing to overwrite existing output"
                ) from exc
            temporary.unlink()
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return output


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the release-gate command-line parser."""

    parser = argparse.ArgumentParser(
        description="Create a signed, fail-closed v3 Journey release packet."
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--signing-key-file", type=Path)
    parser.add_argument("--key-id", default="journey-release")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the Journey release gate, returning nonzero unless ready."""

    args = build_arg_parser().parse_args(argv)
    try:
        key = _load_signing_key(args.signing_key_file)
        manifest = load_journey_release_manifest(args.manifest)
        packet = evaluate_journey_release(
            manifest,
            repo_root=args.repo_root,
            signing_key=key,
            key_id=args.key_id,
        )
        write_journey_release_packet(packet, args.output, overwrite=args.overwrite)
    except JourneyReleaseError as exc:
        print(f"journey-release: {exc}", file=sys.stderr)
        return 2
    except (OSError, subprocess.SubprocessError):
        print("journey-release: release evidence is unavailable", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(packet.to_dict(), indent=2, sort_keys=True))
    else:
        print(f"Journey Release: {packet.release['version']} -> {packet.decision}")
        for gate in packet.gates:
            status = "PASS" if gate.passed else "FAIL"
            blockers = ",".join(gate.blocking_codes) or "none"
            print(f"  [{status}] {gate.gate}: {gate.state}; blockers={blockers}")
        if not packet.repository_binding["verified"]:
            print("  [FAIL] repository_binding")
    return 0 if packet.decision == JOURNEY_RELEASE_READY else 1


def _validate_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    _require_exact_fields(manifest, _MANIFEST_FIELDS, "manifest")
    if manifest["schema_version"] != JOURNEY_RELEASE_SCHEMA_VERSION:
        raise JourneyReleaseError("unsupported Journey release schema_version")
    if manifest["compatibility_policy"] != JOURNEY_RELEASE_COMPATIBILITY_POLICY:
        raise JourneyReleaseError("unsupported Journey release compatibility_policy")

    release = _validate_release(_require_mapping(manifest["release"], "release"))
    environment = _validate_environment(
        _require_mapping(manifest["environment"], "environment")
    )
    policy = _validate_policy(_require_mapping(manifest["policy"], "policy"))
    frozen_inputs = _validate_frozen_inputs(
        _require_list(manifest["frozen_inputs"], "frozen_inputs")
    )
    gate_reports = _validate_gate_reports(
        _require_list(manifest["gate_reports"], "gate_reports")
    )
    licenses = _validate_licenses(_require_list(manifest["licenses"], "licenses"))
    claims = _validate_code_list(manifest["claims"], "claims")
    limitations = _validate_code_list(manifest["limitations"], "limitations")
    exceptions = _validate_exception_records(
        _require_list(manifest["exceptions"], "exceptions")
    )
    return {
        "claims": claims,
        "compatibility_policy": JOURNEY_RELEASE_COMPATIBILITY_POLICY,
        "environment": environment,
        "exceptions": exceptions,
        "frozen_inputs": frozen_inputs,
        "gate_reports": gate_reports,
        "licenses": licenses,
        "limitations": limitations,
        "policy": policy,
        "release": release,
        "schema_version": JOURNEY_RELEASE_SCHEMA_VERSION,
    }


def _validate_release(payload: Mapping[str, Any]) -> dict[str, Any]:
    _require_exact_fields(payload, _RELEASE_FIELDS, "release")
    version = str(payload["version"])
    if not _SEMVER_RE.fullmatch(version):
        raise JourneyReleaseError("release.version must be semantic version text")
    git_tag = _require_code(payload["git_tag"], "release.git_tag")
    if git_tag != f"v{version}":
        raise JourneyReleaseError("release.git_tag must equal v plus release.version")
    git_commit = str(payload["git_commit"])
    if not _COMMIT_RE.fullmatch(git_commit):
        raise JourneyReleaseError("release.git_commit must be a full lowercase SHA")
    evaluated_at = _canonical_timestamp(payload["evaluated_at"], "evaluated_at")
    source_date_epoch = _require_int(
        payload["source_date_epoch"], "source_date_epoch", minimum=0
    )
    if (
        int(_parse_timestamp(evaluated_at, "evaluated_at").timestamp())
        != source_date_epoch
    ):
        raise JourneyReleaseError(
            "source_date_epoch must equal the evaluated_at Unix timestamp"
        )
    return {
        "evaluated_at": evaluated_at,
        "git_commit": git_commit,
        "git_tag": git_tag,
        "source_date_epoch": source_date_epoch,
        "version": version,
    }


def _validate_environment(payload: Mapping[str, Any]) -> dict[str, str]:
    _require_exact_fields(payload, _ENVIRONMENT_FIELDS, "environment")
    environment = {
        key: _require_code(payload[key], f"environment.{key}")
        for key in sorted(_ENVIRONMENT_FIELDS)
    }
    if not _PYTHON_VERSION_RE.fullmatch(environment["python_version"]):
        raise JourneyReleaseError("environment.python_version must be x.y.z")
    return environment


def _validate_policy(payload: Mapping[str, Any]) -> dict[str, int]:
    _require_exact_fields(payload, _POLICY_FIELDS, "policy")
    return {
        "max_artifact_age_seconds": _require_int(
            payload["max_artifact_age_seconds"],
            "policy.max_artifact_age_seconds",
            minimum=1,
            maximum=31_536_000,
        )
    }


def _validate_frozen_inputs(items: list[Any]) -> list[dict[str, str]]:
    if not items:
        raise JourneyReleaseError("frozen_inputs must not be empty")
    inputs: list[dict[str, str]] = []
    ids: set[str] = set()
    for item in items:
        payload = _require_mapping(item, "frozen input")
        _require_exact_fields(payload, _INPUT_FIELDS, "frozen input")
        input_id = _require_code(payload["id"], "frozen input id")
        if input_id in ids:
            raise JourneyReleaseError("duplicate frozen input id")
        ids.add(input_id)
        origin = str(payload["origin"])
        if origin not in _INPUT_ORIGINS:
            raise JourneyReleaseError("unsupported frozen input origin")
        path = _require_portable_path(payload["path"], "frozen input path")
        digest = _require_digest(payload["sha256"], "frozen input sha256")
        inputs.append(
            {"id": input_id, "origin": origin, "path": path, "sha256": digest}
        )
    if not any(item["origin"] == "synthetic" for item in inputs):
        raise JourneyReleaseError("at least one frozen input must be synthetic")
    return sorted(inputs, key=lambda item: item["id"])


def _validate_gate_reports(items: list[Any]) -> list[dict[str, Any]]:
    if len(items) != len(JOURNEY_RELEASE_GATES):
        raise JourneyReleaseError("gate_reports must contain every mandatory gate once")
    reports: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in items:
        payload = _require_mapping(item, "gate report")
        _require_exact_fields(payload, _REPORT_FIELDS, "gate report")
        if payload["schema_version"] != JOURNEY_RELEASE_SCHEMA_VERSION:
            raise JourneyReleaseError("gate report schema_version is unsupported")
        if payload["compatibility_policy"] != JOURNEY_RELEASE_COMPATIBILITY_POLICY:
            raise JourneyReleaseError("gate report compatibility_policy is unsupported")
        gate = str(payload["gate"])
        if gate not in JOURNEY_RELEASE_GATES:
            raise JourneyReleaseError("unknown release gate")
        if gate in seen:
            raise JourneyReleaseError("duplicate release gate")
        seen.add(gate)
        state = str(payload["state"])
        if state not in JOURNEY_RELEASE_STATES:
            raise JourneyReleaseError("unknown release state")
        metrics = _validate_metrics(gate, payload["metrics"])
        reports.append(
            {
                "compatibility_policy": JOURNEY_RELEASE_COMPATIBILITY_POLICY,
                "gate": gate,
                "generated_at": _canonical_timestamp(
                    payload["generated_at"], f"{gate}.generated_at"
                ),
                "metrics": metrics,
                "schema_version": JOURNEY_RELEASE_SCHEMA_VERSION,
                "state": state,
            }
        )
    if seen != set(JOURNEY_RELEASE_GATES):
        missing = sorted(set(JOURNEY_RELEASE_GATES) - seen)
        raise JourneyReleaseError(f"missing mandatory release gates: {missing}")
    order = {name: index for index, name in enumerate(JOURNEY_RELEASE_GATES)}
    return sorted(reports, key=lambda item: order[item["gate"]])


def _validate_metrics(gate: str, raw: Any) -> dict[str, int | float | str]:
    metrics = _require_mapping(raw, f"{gate}.metrics")
    _require_exact_fields(metrics, _METRIC_FIELDS[gate], f"{gate}.metrics")
    result: dict[str, int | float | str] = {}
    for key, value in metrics.items():
        name = str(key)
        if gate == "performance" and name in {"dataset_digest", "quantization"}:
            result[name] = (
                _require_digest(value, "performance.dataset_digest")
                if name == "dataset_digest"
                else _require_code(value, "performance.quantization")
            )
        elif gate == "performance" and name in {
            "latency_p50_ms",
            "latency_p95_ms",
            "latency_p99_ms",
            "peak_memory_mib",
            "throughput_per_second",
        }:
            result[name] = _require_number(value, f"performance.{name}", minimum=0.0)
        else:
            result[name] = _require_int(value, f"{gate}.{name}", minimum=0)
    return dict(sorted(result.items()))


def _validate_licenses(items: list[Any]) -> list[dict[str, Any]]:
    if not items:
        raise JourneyReleaseError("licenses must not be empty")
    result: list[dict[str, Any]] = []
    ids: set[str] = set()
    for item in items:
        payload = _require_mapping(item, "license record")
        _require_exact_fields(payload, _LICENSE_FIELDS, "license record")
        asset_id = _require_code(payload["asset_id"], "license.asset_id")
        if asset_id in ids:
            raise JourneyReleaseError("duplicate license asset_id")
        ids.add(asset_id)
        asset_type = str(payload["asset_type"])
        use = str(payload["use"])
        distribution = str(payload["distribution"])
        if asset_type not in _ASSET_TYPES:
            raise JourneyReleaseError("unsupported asset_type")
        if use not in _ASSET_USES:
            raise JourneyReleaseError("unsupported license use")
        if distribution not in _DISTRIBUTIONS:
            raise JourneyReleaseError("unsupported distribution")
        redistributable = payload["redistributable"]
        if type(redistributable) is not bool:
            raise JourneyReleaseError("license.redistributable must be boolean")
        result.append(
            {
                "asset_id": asset_id,
                "asset_type": asset_type,
                "distribution": distribution,
                "license_id": _require_code(payload["license_id"], "license_id"),
                "redistributable": redistributable,
                "use": use,
            }
        )
    return sorted(result, key=lambda item: item["asset_id"])


def _validate_code_list(raw: Any, name: str) -> list[str]:
    values = [
        _require_code(value, name[:-1] if name.endswith("s") else name)
        for value in _require_list(raw, name)
    ]
    if len(values) != len(set(values)):
        raise JourneyReleaseError(f"{name} must not contain duplicates")
    return values


def _validate_exception_records(items: list[Any]) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    codes: set[str] = set()
    for item in items:
        payload = _require_mapping(item, "exception")
        _require_exact_fields(payload, _EXCEPTION_FIELDS, "exception")
        code = _require_code(payload["code"], "exception.code")
        if code in codes:
            raise JourneyReleaseError("duplicate exception code")
        codes.add(code)
        gate = str(payload["gate"])
        severity = str(payload["severity"])
        disposition = str(payload["disposition"])
        if gate not in JOURNEY_RELEASE_GATES:
            raise JourneyReleaseError("exception has unknown gate")
        if severity not in _SEVERITIES:
            raise JourneyReleaseError("exception has unknown severity")
        if disposition not in _DISPOSITIONS:
            raise JourneyReleaseError("exception has unknown disposition")
        records.append(
            {
                "code": code,
                "disposition": disposition,
                "expires_at": _canonical_timestamp(
                    payload["expires_at"], "exception.expires_at"
                ),
                "gate": gate,
                "severity": severity,
            }
        )
    return sorted(records, key=lambda item: item["code"])


def _verify_repository_binding(
    root: Path,
    release: Mapping[str, Any],
) -> dict[str, Any]:
    blockers: list[str] = []
    expected = str(release["git_commit"])
    tag = str(release["git_tag"])
    head = _git_output(root, "rev-parse", "HEAD")
    try:
        tag_commit = _git_output(root, "rev-list", "-n", "1", tag)
    except JourneyReleaseError:
        tag_commit = "0" * 40
        blockers.append("tag_unresolved")
    if head != expected:
        blockers.append("checkout_commit_mismatch")
    if tag_commit != expected:
        blockers.append("tag_commit_mismatch")
    return {
        "blocking_codes": sorted(blockers),
        "head_commit": head,
        "tag_commit": tag_commit,
        "verified": not blockers,
    }


def _git_output(root: Path, *arguments: str) -> str:
    if not (root / ".git").exists():
        raise JourneyReleaseError("repo_root is not a Git repository")
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=root,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30,
        )
    except subprocess.SubprocessError as exc:
        raise JourneyReleaseError("repository binding command failed") from exc
    value = completed.stdout.strip().lower()
    if not _COMMIT_RE.fullmatch(value):
        raise JourneyReleaseError("repository binding did not resolve a full commit")
    return value


def _verify_frozen_inputs(
    root: Path,
    inputs: Sequence[Mapping[str, str]],
) -> tuple[tuple[Mapping[str, Any], ...], tuple[str, ...]]:
    results: list[Mapping[str, Any]] = []
    blockers: list[str] = []
    for item in inputs:
        input_id = item["id"]
        try:
            path = _resolve_beneath(root, item["path"])
            if path.is_symlink() or not path.is_file():
                raise JourneyReleaseError("input is not a regular non-symlink file")
            digest = _file_digest(path)
            size = path.stat().st_size
            verified = hmac.compare_digest(digest, item["sha256"])
        except (OSError, JourneyReleaseError):
            digest = "sha256:" + ("0" * 64)
            size = 0
            verified = False
        if not verified:
            blockers.append(f"frozen_input_mismatch:{input_id}")
        results.append(
            {
                "byte_size": size,
                "digest": digest,
                "id": input_id,
                "origin": item["origin"],
                "verified": verified,
            }
        )
    return tuple(results), tuple(sorted(blockers))


def _evaluate_gate_reports(
    reports: Sequence[Mapping[str, Any]],
    *,
    evaluated_at: datetime,
    max_age_seconds: int,
) -> tuple[JourneyGateResult, ...]:
    results: list[JourneyGateResult] = []
    for report in reports:
        gate = str(report["gate"])
        state = str(report["state"])
        metrics = report["metrics"]
        generated_at = _parse_timestamp(report["generated_at"], f"{gate}.generated_at")
        age = int((evaluated_at - generated_at).total_seconds())
        blockers: list[str] = []
        if state != "success":
            blockers.append(f"state:{state}")
        if age < 0:
            blockers.append("artifact_from_future")
            age = 0
        elif age > max_age_seconds:
            blockers.append("artifact_stale")
        for metric in _ZERO_BLOCKERS[gate]:
            if metrics[metric] != 0:
                blockers.append(f"nonzero:{metric}")
        for metric in _POSITIVE_COUNTS[gate]:
            if metrics[metric] <= 0:
                blockers.append(f"empty:{metric}")
        if gate == "performance":
            blockers.extend(_performance_blockers(metrics))
        results.append(
            JourneyGateResult(
                gate=gate,
                state=state,
                passed=not blockers,
                artifact_digest=stable_hash(report),
                artifact_age_seconds=age,
                metrics=metrics,
                blocking_codes=tuple(sorted(set(blockers))),
            )
        )
    return tuple(results)


def _performance_blockers(metrics: Mapping[str, Any]) -> list[str]:
    blockers: list[str] = []
    p50 = float(metrics["latency_p50_ms"])
    p95 = float(metrics["latency_p95_ms"])
    p99 = float(metrics["latency_p99_ms"])
    if not (p50 <= p95 <= p99):
        blockers.append("latency_distribution_invalid")
    if float(metrics["throughput_per_second"]) <= 0:
        blockers.append("throughput_missing")
    if float(metrics["peak_memory_mib"]) <= 0:
        blockers.append("memory_missing")
    if int(metrics["warmup_count"]) >= int(metrics["sample_count"]):
        blockers.append("warmup_not_bounded")
    return blockers


def _evaluate_license_catalog(
    licenses: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], tuple[str, ...]]:
    blockers: list[str] = []
    counts = {asset_type: 0 for asset_type in sorted(_ASSET_TYPES)}
    for item in licenses:
        counts[str(item["asset_type"])] += 1
        if item["redistributable"] is False and item["distribution"] == "bundled":
            blockers.append(f"restricted_asset_bundled:{item['asset_id']}")
        if item["use"] == "train" and item["redistributable"] is False:
            blockers.append(f"restricted_training_asset:{item['asset_id']}")
    if counts["model"] == 0:
        blockers.append("model_license_inventory_empty")
    if counts["dataset"] == 0:
        blockers.append("dataset_license_inventory_empty")
    summary = {
        "asset_count": len(licenses),
        "asset_type_counts": counts,
        "catalog_digest": stable_hash(list(licenses)),
        "restricted_asset_count": sum(
            1 for item in licenses if item["redistributable"] is False
        ),
    }
    return summary, tuple(sorted(blockers))


def _cross_check_license_metrics(
    results: tuple[JourneyGateResult, ...],
    summary: Mapping[str, Any],
) -> tuple[JourneyGateResult, ...]:
    counts = summary["asset_type_counts"]
    licensing = next(item for item in results if item.gate == "licensing")
    blockers: list[str] = []
    if licensing.metrics["model_count"] != counts["model"]:
        blockers.append("model_license_count_mismatch")
    if licensing.metrics["data_asset_count"] != counts["dataset"]:
        blockers.append("dataset_license_count_mismatch")
    return _merge_gate_blockers(results, "licensing", blockers)


def _merge_gate_blockers(
    results: tuple[JourneyGateResult, ...],
    gate_name: str,
    blockers: Sequence[str],
) -> tuple[JourneyGateResult, ...]:
    if not blockers:
        return results
    merged: list[JourneyGateResult] = []
    for item in results:
        if item.gate != gate_name:
            merged.append(item)
            continue
        combined = tuple(sorted(set((*item.blocking_codes, *blockers))))
        merged.append(replace(item, passed=False, blocking_codes=combined))
    return tuple(merged)


def _claim_and_limitation_blockers(
    claims: Sequence[str],
    limitations: Sequence[str],
) -> tuple[str, ...]:
    blockers = [
        *(f"missing_claim:{value}" for value in sorted(_REQUIRED_CLAIMS - set(claims))),
        *(
            f"missing_limitation:{value}"
            for value in sorted(_REQUIRED_LIMITATIONS - set(limitations))
        ),
    ]
    return tuple(blockers)


def _evaluate_exceptions(
    exceptions: Sequence[Mapping[str, str]],
    *,
    evaluated_at: datetime,
) -> tuple[tuple[Mapping[str, str], ...], tuple[str, ...]]:
    blockers: list[str] = []
    results: list[Mapping[str, str]] = []
    for item in exceptions:
        code = item["code"]
        expires_at = _parse_timestamp(item["expires_at"], "exception.expires_at")
        if item["severity"] in {"high", "critical"}:
            blockers.append(f"nonwaivable_exception:{code}")
        if item["disposition"] != "accepted":
            blockers.append(f"unaccepted_exception:{code}")
        if expires_at <= evaluated_at:
            blockers.append(f"expired_exception:{code}")
        results.append(dict(item))
    return tuple(results), tuple(sorted(set(blockers)))


def _resolve_beneath(root: Path, relative: str) -> Path:
    unresolved = root / relative
    cursor = root
    for part in Path(relative).parts:
        cursor /= part
        if cursor.is_symlink():
            raise JourneyReleaseError("input path must not traverse a symlink")
    candidate = unresolved.resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise JourneyReleaseError("input path escapes repo_root") from exc
    return candidate


def _load_signing_key(path: Path | None) -> bytes:
    if path is not None:
        if path.is_symlink() or not path.is_file():
            raise JourneyReleaseError("signing key file must be a regular file")
        value = path.read_bytes().strip()
    else:
        value = os.environ.get(_SIGNING_ENV, "").encode("utf-8")
    if len(value) < 32:
        raise JourneyReleaseError(
            f"a signing key of at least 32 bytes is required via {_SIGNING_ENV} "
            "or --signing-key-file"
        )
    return value


def _require_exact_fields(
    payload: Mapping[str, Any],
    expected: frozenset[str],
    name: str,
) -> None:
    actual = set(payload)
    if actual != set(expected):
        raise JourneyReleaseError(f"{name} fields differ")


def _require_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise JourneyReleaseError(f"{name} must be an object")
    if not all(isinstance(key, str) for key in value):
        raise JourneyReleaseError(f"{name} keys must be strings")
    return dict(value)


def _require_list(value: Any, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise JourneyReleaseError(f"{name} must be an array")
    return list(value)


def _require_code(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _CODE_RE.fullmatch(value):
        raise JourneyReleaseError(f"{name} must be a bounded code")
    return value


def _require_digest(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise JourneyReleaseError(f"{name} must be a sha256 digest")
    return value


def _require_portable_path(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise JourneyReleaseError(f"{name} must be a portable relative path")
    path = Path(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise JourneyReleaseError(f"{name} must be a portable relative path")
    return path.as_posix()


def _require_int(
    value: Any,
    name: str,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise JourneyReleaseError(f"{name} must be an integer")
    if minimum is not None and value < minimum:
        raise JourneyReleaseError(f"{name} must be at least {minimum}")
    if maximum is not None and value > maximum:
        raise JourneyReleaseError(f"{name} must be at most {maximum}")
    return value


def _require_number(
    value: Any,
    name: str,
    *,
    minimum: float | None = None,
) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise JourneyReleaseError(f"{name} must be numeric")
    if not math.isfinite(float(value)):
        raise JourneyReleaseError(f"{name} must be finite")
    if minimum is not None and float(value) < minimum:
        raise JourneyReleaseError(f"{name} must be at least {minimum}")
    return value


def _canonical_timestamp(value: Any, name: str) -> str:
    parsed = _parse_timestamp(value, name)
    return parsed.isoformat().replace("+00:00", "Z")


def _parse_timestamp(value: Any, name: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise JourneyReleaseError(f"{name} must be an RFC 3339 UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise JourneyReleaseError(f"{name} must be an RFC 3339 UTC timestamp") from exc
    if parsed.tzinfo != timezone.utc or parsed.microsecond != 0:
        raise JourneyReleaseError(f"{name} must use UTC seconds without fractions")
    return parsed


def _file_digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _load_bundled_schema(name: str) -> dict[str, Any]:
    resource = resources.files(_SCHEMA_PACKAGE).joinpath(name)
    with resource.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise JourneyReleaseError(f"bundled schema is not an object: {name}")
    return payload


def _canonical_json(payload: Any) -> str:
    return json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _key_bytes(key: bytes | str) -> bytes:
    value = key if isinstance(key, bytes) else key.encode("utf-8")
    if len(value) < 32:
        raise JourneyReleaseError("signing key must contain at least 32 bytes")
    return value


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise JourneyReleaseError("duplicate JSON key")
        result[key] = value
    return result


def _reject_non_finite(value: str) -> None:
    raise JourneyReleaseError(f"non-finite JSON number is not allowed: {value}")


__all__ = [
    "JOURNEY_RELEASE_COMPATIBILITY_POLICY",
    "JOURNEY_RELEASE_GATES",
    "JOURNEY_RELEASE_NOT_READY",
    "JOURNEY_RELEASE_READY",
    "JOURNEY_RELEASE_SCHEMA_VERSION",
    "JOURNEY_RELEASE_STATES",
    "JourneyGateResult",
    "JourneyReleaseError",
    "JourneyReleasePacket",
    "evaluate_journey_release",
    "load_journey_release_manifest",
    "load_journey_release_manifest_schema",
    "load_journey_release_packet_schema",
    "main",
    "write_journey_release_packet",
]


if __name__ == "__main__":
    raise SystemExit(main())
