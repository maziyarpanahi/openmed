"""Offline exact-source candidate manifests and signed v3.1 gate decisions.

Inputs are local, caller-governed files. Their digests bind content, not its
origin or clinical validity. This module never builds, tags or publishes.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import re
import subprocess
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from openmed.eval.suites.agent_release import (
    DEFAULT_AGENT_RELEASE_GATES,
    NOT_READY,
    READY,
    AgentReleaseGateError,
    MetricEvidence,
    evaluate_agent_release_gates,
)

SOURCE_SCHEMA = "openmed.eval.agent_release_source.v1"
MANIFEST_SCHEMA = "openmed.eval.agent_candidate_manifest.v1"
PACKET_SCHEMA = "openmed.eval.agent_candidate_packet.v1"
_SHA = re.compile(r"[0-9a-f]{40}(?:[0-9a-f]{24})?")
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")
_METRICS = tuple(spec.metric for spec in DEFAULT_AGENT_RELEASE_GATES)
_METRIC_FIELDS = frozenset(
    {
        "schema_version",
        "metric",
        "value",
        "sample_size",
        "event_count",
        "ci_lower",
        "ci_upper",
        "evidence_digest",
        "slices",
        "limitations",
    }
)
_SLICE_FIELDS = frozenset(
    {"slice_ref", "value", "sample_size", "event_count", "ci_lower", "ci_upper"}
)
_GATE_REASONS = frozenset(
    {
        "missing_evidence",
        "required_statistical_basis_missing",
        "threshold_satisfied",
        "threshold_failed",
    }
)
_REASONS = frozenset(
    {
        "repository_unavailable",
        "dirty_tree",
        "sha_mismatch",
        "tag_unavailable",
        "tag_mismatch",
        "repository_changed",
        "wheel_missing",
        "sdist_missing",
        "tool_catalog_missing",
        "policy_missing",
        "evidence_missing",
        "evidence_invalid",
        "evidence_sha_mismatch",
        "evidence_duplicate_metric",
        "gate_missing_evidence",
        "gate_failed",
    }
)
_GATE_FIELDS = frozenset(
    {"metric", "reason_code", "sample_size", "event_count", "slice_count"}
)
_PACKET_FIELDS = frozenset(
    {
        "schema_version",
        "manifest_digest",
        "report_digest",
        "decision",
        "reason_codes",
        "gates",
        "packet_digest",
        "signature",
    }
)


class AgentCandidateError(ValueError):
    """A value-free candidate configuration or serialization error."""


def _canonical(payload: Any) -> str:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _digest(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _payload_digest(payload: Any) -> str:
    return _digest(_canonical(payload).encode("utf-8"))


def _key(key: bytes) -> bytes:
    if not isinstance(key, bytes) or len(key) < 32:
        raise AgentCandidateError("signing_key_invalid")
    return key


def _signature(payload: Mapping[str, Any], key: bytes) -> str:
    return (
        "sha256:"
        + hmac.new(
            _key(key), _canonical(payload).encode("utf-8"), hashlib.sha256
        ).hexdigest()
    )


def _valid_digest(value: Any) -> bool:
    return isinstance(value, str) and _DIGEST.fullmatch(value) is not None


def _validate_packet(payload: Mapping[str, Any]) -> None:
    if set(payload) != _PACKET_FIELDS or payload["schema_version"] != PACKET_SCHEMA:
        raise AgentCandidateError("packet_invalid")
    if any(
        not _valid_digest(payload[field])
        for field in ("manifest_digest", "report_digest", "packet_digest", "signature")
    ):
        raise AgentCandidateError("packet_invalid")
    reasons = payload["reason_codes"]
    gates = payload["gates"]
    if (
        not isinstance(reasons, list)
        or any(not isinstance(code, str) or code not in _REASONS for code in reasons)
        or reasons != sorted(set(reasons))
        or not isinstance(gates, list)
        or len(gates) != len(_METRICS)
    ):
        raise AgentCandidateError("packet_invalid")
    for metric, gate in zip(_METRICS, gates):
        if (
            not isinstance(gate, dict)
            or set(gate) != _GATE_FIELDS
            or gate["metric"] != metric
            or gate["reason_code"] not in _GATE_REASONS
            or any(
                type(gate[field]) is not int or gate[field] < 0
                for field in ("sample_size", "slice_count")
            )
            or (
                gate["event_count"] is not None
                and (
                    type(gate["event_count"]) is not int
                    or not 0 <= gate["event_count"] <= gate["sample_size"]
                )
            )
        ):
            raise AgentCandidateError("packet_invalid")
    expected = (
        READY
        if not reasons
        and all(gate["reason_code"] == "threshold_satisfied" for gate in gates)
        else NOT_READY
    )
    if payload["decision"] != expected:
        raise AgentCandidateError("packet_invalid")


@dataclass(frozen=True, slots=True)
class AgentCandidatePacket:
    """Immutable canonical decision packet with offline HMAC verification."""

    _json: str

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> AgentCandidatePacket:
        """Restore a closed packet schema without trusting its signature.

        Args:
            payload: Serialized packet fields.
        Returns:
            A packet whose authenticity must separately pass ``verify``.
        Raises:
            AgentCandidateError: The packet has an invalid public schema.
        """
        try:
            _validate_packet(payload)
            return cls(_canonical(payload))
        except (TypeError, ValueError, KeyError) as exc:
            raise AgentCandidateError("packet_invalid") from exc

    def to_dict(self) -> dict[str, Any]:
        """Return an independent copy of the content-free packet."""
        return json.loads(self._json)

    def to_json(self) -> str:
        """Return byte-stable canonical JSON."""
        return self._json

    @property
    def packet_digest(self) -> str:
        """Return the digest of all unsigned decision fields."""
        return self.to_dict()["packet_digest"]

    @property
    def decision(self) -> str:
        """Return READY or NOT_READY."""
        return self.to_dict()["decision"]

    def verify(self, signing_key: bytes) -> bool:
        """Verify the complete packet and its digest with an offline key.

        Args:
            signing_key: Injected HMAC key of at least 32 bytes.
        Returns:
            Whether the closed schema, digest and signature are intact.
        """
        try:
            payload = self.to_dict()
            _validate_packet(payload)
            signature = payload.pop("signature")
            expected_signature = _signature(payload, signing_key)
            packet_digest = payload.pop("packet_digest")
            return hmac.compare_digest(
                packet_digest, _payload_digest(payload)
            ) and hmac.compare_digest(signature, expected_signature)
        except (AgentCandidateError, TypeError, ValueError, KeyError):
            return False


@dataclass(frozen=True, slots=True)
class AgentReleaseCandidate:
    """Immutable canonical manifest and its signed, digest-bound decision."""

    _manifest_json: str
    packet: AgentCandidatePacket

    @property
    def manifest_digest(self) -> str:
        """Return the canonical candidate digest consumed by agent gates."""
        return _digest(self._manifest_json.encode("utf-8"))

    @property
    def manifest(self) -> dict[str, Any]:
        """Return an independent copy of the content-free candidate manifest."""
        return json.loads(self._manifest_json)

    def to_dict(self) -> dict[str, Any]:
        """Return the public manifest and packet without paths or inputs."""
        return {
            "manifest": self.manifest,
            "manifest_digest": self.manifest_digest,
            "packet": self.packet.to_dict(),
        }


def _git(root: Path, *arguments: str) -> str:
    try:
        return subprocess.run(
            ["git", *arguments],
            cwd=root,
            capture_output=True,
            check=True,
            timeout=30,
            text=True,
            encoding="utf-8",
            errors="replace",
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        raise AgentCandidateError("repository_unavailable") from exc


def _repository(root: Path, expected_sha: str, tag: str | None) -> dict[str, Any]:
    reasons: set[str] = set()
    head = None
    clean = "unknown"
    tag_sha = None
    try:
        head = _git(root, "rev-parse", "--verify", "HEAD^{commit}")
        if _SHA.fullmatch(head) is None:
            raise AgentCandidateError("repository_unavailable")
        clean = (
            "dirty"
            if _git(root, "status", "--porcelain", "--untracked-files=all")
            else "clean"
        )
        if clean == "dirty":
            reasons.add("dirty_tree")
        if head != expected_sha:
            reasons.add("sha_mismatch")
    except AgentCandidateError:
        head = None
        reasons.add("repository_unavailable")
    if tag is not None:
        try:
            # Require an actual ref name, never a Git revision expression.
            _git(root, "check-ref-format", f"refs/tags/{tag}")
            tag_sha = _git(
                root,
                "rev-parse",
                "--verify",
                "--end-of-options",
                f"refs/tags/{tag}^{{commit}}",
            )
            if _SHA.fullmatch(tag_sha) is None:
                raise AgentCandidateError("repository_unavailable")
            if tag_sha != expected_sha:
                reasons.add("tag_mismatch")
        except AgentCandidateError:
            tag_sha = None
            reasons.add("tag_unavailable")
    return {
        "git_sha": head,
        "expected_sha": expected_sha,
        "tree_status": clean,
        "tag_digest": _digest(tag.encode("utf-8")) if tag is not None else None,
        "tag_sha": tag_sha,
        "reason_codes": sorted(reasons),
    }


def _read_file(path: Path) -> bytes:
    if not path.is_file() or path.is_symlink():
        raise OSError("input_unavailable")
    return path.read_bytes()


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, value in pairs:
        if name in result:
            raise AgentCandidateError("evidence_invalid")
        result[name] = value
    return result


def _nonfinite(value: str) -> None:
    raise AgentCandidateError("evidence_invalid")


def _parse_evidence(data: bytes, expected_sha: str) -> tuple[MetricEvidence, ...]:
    payload = json.loads(data, object_pairs_hook=_pairs, parse_constant=_nonfinite)
    if (
        not isinstance(payload, dict)
        or set(payload) != {"schema_version", "source_sha", "metrics"}
        or payload["schema_version"] != SOURCE_SCHEMA
        or not isinstance(payload["source_sha"], str)
        or _SHA.fullmatch(payload["source_sha"]) is None
    ):
        raise AgentCandidateError("evidence_invalid")
    if payload["source_sha"] != expected_sha:
        raise AgentCandidateError("evidence_sha_mismatch")
    records = payload["metrics"]
    if not isinstance(records, list) or not records:
        raise AgentCandidateError("evidence_invalid")
    metrics = []
    for record in records:
        if (
            not isinstance(record, dict)
            or set(record) != _METRIC_FIELDS
            or record["metric"] not in _METRICS
            or not isinstance(record["slices"], list)
            or any(
                not isinstance(row, dict) or set(row) != _SLICE_FIELDS
                for row in record["slices"]
            )
        ):
            raise AgentCandidateError("evidence_invalid")
        metrics.append(MetricEvidence.from_dict(record))
    return tuple(metrics)


def run_agent_release_candidate(
    *,
    repo_root: str | Path,
    source_sha: str,
    wheel: str | Path,
    sdist: str | Path,
    tool_catalog: str | Path,
    policy: str | Path,
    evidence_files: Sequence[str | Path],
    signing_key: bytes,
    tag: str | None = None,
) -> AgentReleaseCandidate:
    """Build an exact-source manifest, evaluate gates and sign the decision.

    Args:
        repo_root: Local git checkout, including linked worktrees.
        source_sha: Required full lowercase source commit SHA.
        wheel: Caller-built wheel file (hashed as bytes, never built here).
        sdist: Caller-built source distribution file.
        tool_catalog: Caller-governed tool-catalog file.
        policy: Caller-governed workflow policy file.
        evidence_files: Local SOURCE_SCHEMA envelopes containing MetricEvidence.
        signing_key: Injected offline HMAC key of at least 32 bytes.
        tag: Optional existing tag that must resolve to source_sha.
    Returns:
        Canonical manifest and signed packet, including NOT_READY failures.
    Raises:
        AgentCandidateError: Source SHA or signing key configuration is invalid.
    """
    _key(signing_key)
    if not isinstance(source_sha, str) or _SHA.fullmatch(source_sha) is None:
        raise AgentCandidateError("source_sha_invalid")
    root = Path(repo_root)
    repository = _repository(root, source_sha, tag)
    reasons = set(repository["reason_codes"])
    artifacts: dict[str, str | None] = {}
    for name, path in (
        ("wheel", wheel),
        ("sdist", sdist),
        ("tool_catalog", tool_catalog),
        ("policy", policy),
    ):
        try:
            artifacts[name] = _digest(_read_file(Path(path)))
        except OSError:
            artifacts[name] = None
            reasons.add(name + "_missing")
    entries = []
    metrics: list[MetricEvidence] = []
    for path in evidence_files:
        entry: dict[str, Any] = {
            "file_digest": None,
            "status": "accepted",
            "metric_count": 0,
        }
        try:
            data = _read_file(Path(path))
            entry["file_digest"] = _digest(data)
            parsed = _parse_evidence(data, source_sha)
            entry["metric_count"] = len(parsed)
            metrics.extend(parsed)
        except OSError:
            entry["status"] = "evidence_missing"
        except AgentCandidateError as exc:
            entry["status"] = str(exc)
        except (AgentReleaseGateError, ValueError, TypeError, KeyError, RecursionError):
            entry["status"] = "evidence_invalid"
        if entry["status"] != "accepted":
            reasons.add(entry["status"])
        entries.append(entry)
    counts = Counter(item.metric for item in metrics)
    if any(count > 1 for count in counts.values()):
        reasons.add("evidence_duplicate_metric")
    # Reject every duplicated metric rather than picking a caller-order winner.
    metrics = [item for item in metrics if counts[item.metric] == 1]
    if not evidence_files:
        reasons.add("evidence_missing")
    after = _repository(root, source_sha, tag)
    if after != repository:
        reasons.add("repository_changed")
    reasons.update(after["reason_codes"])
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "repository": repository,
        "repository_after": after,
        "artifacts": artifacts,
        "evidence": sorted(entries, key=_canonical),
        "gate_policy_digest": _payload_digest(
            [spec.to_dict() for spec in DEFAULT_AGENT_RELEASE_GATES]
        ),
        "reason_codes": sorted(reasons),
    }
    manifest_json = _canonical(manifest)
    manifest_digest = _digest(manifest_json.encode("utf-8"))
    report = evaluate_agent_release_gates(metrics, candidate_digest=manifest_digest)
    if report.decision != READY:
        reasons.add("gate_failed")
    if any(row.reason_code == "missing_evidence" for row in report.gate_results):
        reasons.add("gate_missing_evidence")
    unsigned = {
        "schema_version": PACKET_SCHEMA,
        "manifest_digest": manifest_digest,
        "report_digest": report.report_digest,
        "decision": NOT_READY if reasons else READY,
        "reason_codes": sorted(reasons),
        "gates": [
            {
                "metric": row.metric,
                "reason_code": row.reason_code,
                "sample_size": row.sample_size,
                "event_count": row.event_count,
                "slice_count": len(row.slice_sizes),
            }
            for row in report.gate_results
        ],
    }
    signed = {**unsigned, "packet_digest": _payload_digest(unsigned)}
    packet = AgentCandidatePacket.from_dict(
        {**signed, "signature": _signature(signed, signing_key)}
    )
    return AgentReleaseCandidate(manifest_json, packet)


def add_candidate_arguments(parser: argparse.ArgumentParser) -> None:
    """Register the shared CLI and release-script candidate arguments."""
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--tag")
    for name in (
        "wheel",
        "sdist",
        "tool-catalog",
        "policy",
        "signing-key-file",
        "output",
    ):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--evidence", type=Path, action="append", default=[])


def candidate_from_arguments(args: argparse.Namespace) -> AgentReleaseCandidate:
    """Run and exclusively write a canonical bundle with value-free errors.

    Args:
        args: Parsed shared candidate command arguments.
    Returns:
        The evaluated and signed candidate.
    Raises:
        AgentCandidateError: Key, configuration or output is unavailable.
    """
    try:
        key = _read_file(args.signing_key_file)
    except OSError as exc:
        raise AgentCandidateError("signing_key_unavailable") from exc
    result = run_agent_release_candidate(
        repo_root=args.repo_root,
        source_sha=args.source_sha,
        tag=args.tag,
        wheel=args.wheel,
        sdist=args.sdist,
        tool_catalog=args.tool_catalog,
        policy=args.policy,
        evidence_files=args.evidence,
        signing_key=key,
    )
    try:
        # Do not overwrite artifacts or follow output symlinks.
        with args.output.open("x", encoding="utf-8") as handle:
            handle.write(_canonical(result.to_dict()) + "\n")
    except OSError as exc:
        raise AgentCandidateError("output_unavailable") from exc
    return result


def main(argv: Sequence[str] | None = None) -> int:
    """Run the offline release script; exit 0 for READY, 1 for NOT_READY."""
    parser = argparse.ArgumentParser(description=__doc__)
    add_candidate_arguments(parser)
    args = parser.parse_args(argv)
    try:
        result = candidate_from_arguments(args)
    except AgentCandidateError as exc:
        print(_canonical({"error_code": str(exc)}))
        return 2
    print(_canonical(result.packet.to_dict()))
    return 0 if result.packet.decision == READY else 1
