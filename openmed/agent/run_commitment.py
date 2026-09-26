"""Domain-separated commitments for validated agent run summaries.

Commitments bind evidence to canonical, privacy-safe ``RunSummary`` metadata.
This module performs no I/O, signing, key management, or raw-event hashing.
"""

from __future__ import annotations

import hashlib
import hmac
import re
from enum import Enum
from typing import Any, Final

from .run_summary import RunSummary

RUN_COMMITMENT_VERSION: Final = "openmed.agent.run_commitment.v1"

_DOMAIN_SEPARATOR: Final = RUN_COMMITMENT_VERSION.encode("ascii") + b"\x00"
_COMMITMENT_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_DUMMY_COMMITMENT: Final = "sha256:" + "0" * 64


class RunCommitmentError(ValueError):
    """Raised when commitment computation receives an invalid summary.

    Messages contain only stable field names and error codes, never submitted
    values.
    """


class RunCommitmentVerificationResult(str, Enum):
    """Closed, content-free result categories for commitment verification."""

    VERIFIED = "verified"
    MISMATCH = "mismatch"
    MALFORMED = "malformed_commitment"


def compute_run_summary_commitment(summary: RunSummary) -> str:
    """Return a versioned, domain-separated SHA-256 commitment.

    Only an exact, already validated :class:`RunSummary` is accepted. The hash
    input is the ASCII commitment domain, a NUL separator, and the summary's
    canonical UTF-8 JSON bytes.

    Args:
        summary: Validated privacy-safe run summary.

    Returns:
        A lowercase ``sha256:`` commitment.

    Raises:
        RunCommitmentError: If ``summary`` is not an exact ``RunSummary``.
    """

    if type(summary) is not RunSummary:
        raise RunCommitmentError("summary: invalid_type")

    canonical_json = summary.to_json().encode("utf-8")
    digest = hashlib.sha256(_DOMAIN_SEPARATOR + canonical_json).hexdigest()
    return f"sha256:{digest}"


def verify_run_summary_commitment(
    summary: RunSummary,
    commitment: Any,
) -> RunCommitmentVerificationResult:
    """Verify a commitment using a constant-time digest comparison.

    Malformed commitments are compared against a fixed dummy value before a
    categorical result is returned. Results never contain or echo the submitted
    commitment.

    Args:
        summary: Validated privacy-safe run summary.
        commitment: Candidate lowercase ``sha256:`` commitment.

    Returns:
        ``VERIFIED``, ``MISMATCH``, or ``MALFORMED``.

    Raises:
        RunCommitmentError: If ``summary`` is not an exact ``RunSummary``.
    """

    expected = compute_run_summary_commitment(summary)
    well_formed = (
        type(commitment) is str and _COMMITMENT_RE.fullmatch(commitment) is not None
    )
    candidate = commitment if well_formed else _DUMMY_COMMITMENT
    matched = hmac.compare_digest(expected, candidate)

    if not well_formed:
        return RunCommitmentVerificationResult.MALFORMED
    if matched:
        return RunCommitmentVerificationResult.VERIFIED
    return RunCommitmentVerificationResult.MISMATCH


__all__ = [
    "RUN_COMMITMENT_VERSION",
    "RunCommitmentError",
    "RunCommitmentVerificationResult",
    "compute_run_summary_commitment",
    "verify_run_summary_commitment",
]
