"""Deterministic, domain-separated commitments for agent run summaries.

An evidence bundle binds to an exact privacy-safe run summary by committing to
the canonical JSON bytes, so two systems can agree a summary changed without
copying the summary or trusting a filename. The commitment is a SHA-256 digest
over a fixed domain separator and the validated summary's canonical JSON; it
never hashes raw events or clinical content, and it performs no I/O, signing,
or key management.
"""

from __future__ import annotations

import hashlib
import hmac
import re
from enum import Enum

from .run_summary import RunSummary

RUN_COMMITMENT_SCHEMA_VERSION = "openmed.agent.run_commitment.v1"

_DOMAIN_SEPARATOR = "openmed.agent.run_commitment.v1"
_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


class RunCommitmentError(ValueError):
    """Value-free failure for invalid commitment input.

    Messages name only a stable field and error code, never submitted values.
    """


class CommitmentStatus(str, Enum):
    """Closed outcome of a commitment verification."""

    MATCH = "match"
    MISMATCH = "mismatch"


def commit_run_summary(summary: RunSummary) -> str:
    """Compute a domain-separated SHA-256 commitment for a run summary.

    The commitment is derived from the summary's deterministic canonical JSON,
    so identical summaries always produce identical commitments and any change
    to a validated metadata field changes the digest. Only validated
    :class:`~openmed.agent.run_summary.RunSummary` values are accepted; raw
    events, prompts, and clinical content never reach the hash function.

    Args:
        summary: A validated, metadata-only run summary.

    Returns:
        A ``sha256:<64 hex characters>`` commitment string.

    Raises:
        RunCommitmentError: If ``summary`` is not a :class:`RunSummary`.
    """
    if not isinstance(summary, RunSummary):
        raise RunCommitmentError("summary: invalid_summary")
    canonical = summary.to_json().encode("utf-8")
    digest = hashlib.sha256(
        _DOMAIN_SEPARATOR.encode("utf-8") + b"\x00" + canonical
    ).hexdigest()
    return f"sha256:{digest}"


def verify_run_commitment(summary: RunSummary, commitment: str) -> CommitmentStatus:
    """Verify a commitment against a run summary with constant-time comparison.

    The submitted commitment is never echoed on failure. A malformed commitment
    fails closed with a value-free error rather than being reported as a
    mismatch, so callers can distinguish an input problem from a genuine
    disagreement.

    Args:
        summary: A validated, metadata-only run summary.
        commitment: A ``sha256:<64 hex characters>`` commitment string.

    Returns:
        :attr:`CommitmentStatus.MATCH` when the commitment reproduces the
        summary, otherwise :attr:`CommitmentStatus.MISMATCH`.

    Raises:
        RunCommitmentError: If ``summary`` is not a :class:`RunSummary`, or if
            ``commitment`` is not a well-formed SHA-256 commitment.
    """
    if not isinstance(summary, RunSummary):
        raise RunCommitmentError("summary: invalid_summary")
    if type(commitment) is not str or _SHA256_RE.fullmatch(commitment) is None:
        raise RunCommitmentError("commitment: invalid_digest")
    expected = commit_run_summary(summary)
    if hmac.compare_digest(expected, commitment):
        return CommitmentStatus.MATCH
    return CommitmentStatus.MISMATCH


__all__ = [
    "RUN_COMMITMENT_SCHEMA_VERSION",
    "CommitmentStatus",
    "RunCommitmentError",
    "commit_run_summary",
    "verify_run_commitment",
]
