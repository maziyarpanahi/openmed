"""Deterministic, offline memorization audit for synthetic clinical fragments.

Local generation does not by itself guarantee privacy: a generator can
reproduce rare sequences that already exist in the data it was trained on.
This module audits generated candidates against caller-supplied protected
reference fingerprints and blocks a dataset release when a configured
memorization signal fires.

Three signals are evaluated per candidate/reference pair:

``exact``
    A normalized candidate fragment is byte-identical to a protected
    reference fragment (compared through SHA-256 fragment digests).
``fuzzy``
    A candidate fragment and a protected reference fragment share a token
    shingle set above the configured similarity floor.
``exposure``
    The candidate reproduces at least the configured ratio of the protected
    reference's character n-grams.

The audit is deterministic and performs no network call. Report payloads,
finding messages, and exceptions carry only identifiers, offsets, lengths,
scores, and SHA-256 digests -- never raw reference or candidate text.

Typical use::

    from openmed.eval.synthetic_memorization import (
        audit_synthetic_memorization,
        fingerprint_reference,
    )

    reference = fingerprint_reference("protected-001", protected_text)
    report = audit_synthetic_memorization(
        {"candidate-001": generated_text},
        (reference,),
    )
    if report.blocked:
        ...  # withhold the dataset release
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, Final

SYNTHETIC_MEMORIZATION_SCHEMA_VERSION: Final = "openmed.eval.synthetic_memorization.v1"

DEFAULT_MIN_FRAGMENT_CHARS: Final = 24
DEFAULT_FUZZY_SIMILARITY: Final = 0.72
DEFAULT_SHINGLE_SIZE: Final = 3
DEFAULT_EXPOSURE_NGRAM_SIZE: Final = 12
DEFAULT_EXPOSURE_MATCH_RATIO: Final = 0.4
DEFAULT_MAX_FINDINGS: Final = 64
DEFAULT_MAX_CANDIDATE_FRAGMENTS: Final = 256

MAX_TEXT_CHARS: Final = 200_000
MAX_REFERENCES: Final = 512
MAX_CANDIDATES: Final = 2_048
MAX_SHINGLE_SIZE: Final = 8
MAX_NGRAM_SIZE: Final = 64

_IDENTIFIER_RE: Final = re.compile(r"[a-z0-9][a-z0-9_.:-]{0,63}\Z")
_DIGEST_RE: Final = re.compile(r"sha256:[0-9a-f]{64}\Z")
_WHITESPACE_RE: Final = re.compile(r"\s+")
_FRAGMENT_BOUNDARY_RE: Final = re.compile(r"[.!?;\n\r\u2028\u2029]+")
_TOKEN_RE: Final = re.compile(r"[^\s]+")


class MemorizationSignal(str, Enum):
    """Category of memorization evidence collected for one candidate."""

    EXACT = "exact"
    FUZZY = "fuzzy"
    EXPOSURE = "exposure"


class MemorizationReasonCode(str, Enum):
    """Machine-readable reason attached to a single memorization finding."""

    EXACT_FRAGMENT_MATCH = "exact_fragment_match"
    FUZZY_FRAGMENT_MATCH = "fuzzy_fragment_match"
    EXPOSURE_NGRAM_MATCH = "exposure_ngram_match"


class MemorizationVerdict(str, Enum):
    """Release decision derived from the collected findings."""

    CLEAR = "clear"
    BLOCKED = "blocked"


_REASON_BY_SIGNAL: Final[dict[MemorizationSignal, MemorizationReasonCode]] = {
    MemorizationSignal.EXACT: MemorizationReasonCode.EXACT_FRAGMENT_MATCH,
    MemorizationSignal.FUZZY: MemorizationReasonCode.FUZZY_FRAGMENT_MATCH,
    MemorizationSignal.EXPOSURE: MemorizationReasonCode.EXPOSURE_NGRAM_MATCH,
}

_SIGNAL_ORDER: Final[tuple[MemorizationSignal, ...]] = (
    MemorizationSignal.EXACT,
    MemorizationSignal.EXPOSURE,
    MemorizationSignal.FUZZY,
)


class SyntheticMemorizationError(ValueError):
    """Raised when a memorization audit input or invariant is invalid."""


def normalize_text(text: str) -> str:
    """Return the whitespace-normalized, lower-cased audit representation.

    Args:
        text: Raw candidate or reference text.

    Returns:
        The normalized text used for every digest in this module.

    Raises:
        SyntheticMemorizationError: If ``text`` is not a string.
    """
    if not isinstance(text, str):
        raise SyntheticMemorizationError("text must be a string")
    return _WHITESPACE_RE.sub(" ", text.strip().lower())


def _digest(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _require_identifier(value: str, *, label: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER_RE.match(value):
        raise SyntheticMemorizationError(
            f"{label} must be a lowercase identifier of 1-64 characters"
        )
    return value


def _require_digest(value: str, *, label: str) -> str:
    if not isinstance(value, str) or not _DIGEST_RE.match(value):
        raise SyntheticMemorizationError(f"{label} must be a sha256 digest")
    return value


def _require_text(value: str, *, label: str) -> str:
    if not isinstance(value, str):
        raise SyntheticMemorizationError(f"{label} must be a string")
    if len(value) > MAX_TEXT_CHARS:
        raise SyntheticMemorizationError(
            f"{label} exceeds the {MAX_TEXT_CHARS}-character audit bound"
        )
    return value


def _token_shingles(tokens: Sequence[str], size: int) -> tuple[int, ...]:
    if size < 1:
        raise SyntheticMemorizationError("shingle size must be a positive integer")
    if not tokens:
        return ()
    if len(tokens) < size:
        window = " ".join(tokens)
        return (_shingle_hash(window),)
    hashes = {
        _shingle_hash(" ".join(tokens[index : index + size]))
        for index in range(len(tokens) - size + 1)
    }
    return tuple(sorted(hashes))


def _shingle_hash(value: str) -> int:
    digest = hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big")


def _char_ngram_digests(text: str, size: int) -> tuple[str, ...]:
    if size < 1:
        raise SyntheticMemorizationError("n-gram size must be a positive integer")
    if len(text) < size:
        return ()
    return tuple(
        sorted(
            {
                _digest(text[index : index + size])
                for index in range(len(text) - size + 1)
            }
        )
    )


@dataclass(frozen=True)
class SyntheticMemorizationPolicy:
    """Thresholds and bounds applied by :func:`audit_synthetic_memorization`.

    Attributes:
        min_fragment_chars: Minimum normalized fragment length audited for the
            exact and fuzzy signals.
        fuzzy_similarity: Minimum Jaccard similarity over token shingles for a
            fuzzy match.
        shingle_size: Token window size used to build fuzzy shingle sets.
        exposure_ngram_size: Character n-gram size used by the exposure signal.
        exposure_match_ratio: Minimum share of a reference's character n-grams
            a candidate must reproduce for an exposure match.
        blocking_signals: Signals that withhold a dataset release when present.
        max_findings: Maximum findings retained in the report.
        max_candidate_fragments: Maximum candidate fragments audited per
            candidate.
    """

    min_fragment_chars: int = DEFAULT_MIN_FRAGMENT_CHARS
    fuzzy_similarity: float = DEFAULT_FUZZY_SIMILARITY
    shingle_size: int = DEFAULT_SHINGLE_SIZE
    exposure_ngram_size: int = DEFAULT_EXPOSURE_NGRAM_SIZE
    exposure_match_ratio: float = DEFAULT_EXPOSURE_MATCH_RATIO
    blocking_signals: tuple[MemorizationSignal, ...] = _SIGNAL_ORDER
    max_findings: int = DEFAULT_MAX_FINDINGS
    max_candidate_fragments: int = DEFAULT_MAX_CANDIDATE_FRAGMENTS

    def __post_init__(self) -> None:
        if not isinstance(self.min_fragment_chars, int) or isinstance(
            self.min_fragment_chars, bool
        ):
            raise SyntheticMemorizationError("min_fragment_chars must be an integer")
        if not 1 <= self.min_fragment_chars <= MAX_TEXT_CHARS:
            raise SyntheticMemorizationError(
                f"min_fragment_chars must be between 1 and {MAX_TEXT_CHARS}"
            )
        if not isinstance(self.fuzzy_similarity, (int, float)) or isinstance(
            self.fuzzy_similarity, bool
        ):
            raise SyntheticMemorizationError("fuzzy_similarity must be a number")
        if not 0.0 < float(self.fuzzy_similarity) <= 1.0:
            raise SyntheticMemorizationError(
                "fuzzy_similarity must be greater than 0 and at most 1"
            )
        if not isinstance(self.shingle_size, int) or isinstance(
            self.shingle_size, bool
        ):
            raise SyntheticMemorizationError("shingle_size must be an integer")
        if not 1 <= self.shingle_size <= MAX_SHINGLE_SIZE:
            raise SyntheticMemorizationError(
                f"shingle_size must be between 1 and {MAX_SHINGLE_SIZE}"
            )
        if not isinstance(self.exposure_ngram_size, int) or isinstance(
            self.exposure_ngram_size, bool
        ):
            raise SyntheticMemorizationError("exposure_ngram_size must be an integer")
        if not 1 <= self.exposure_ngram_size <= MAX_NGRAM_SIZE:
            raise SyntheticMemorizationError(
                f"exposure_ngram_size must be between 1 and {MAX_NGRAM_SIZE}"
            )
        if not isinstance(self.exposure_match_ratio, (int, float)) or isinstance(
            self.exposure_match_ratio, bool
        ):
            raise SyntheticMemorizationError("exposure_match_ratio must be a number")
        if not 0.0 < float(self.exposure_match_ratio) <= 1.0:
            raise SyntheticMemorizationError(
                "exposure_match_ratio must be greater than 0 and at most 1"
            )
        signals = self.blocking_signals
        if not isinstance(signals, tuple) or not signals:
            raise SyntheticMemorizationError(
                "blocking_signals must be a non-empty tuple"
            )
        for signal in signals:
            if not isinstance(signal, MemorizationSignal):
                raise SyntheticMemorizationError(
                    "blocking_signals must contain MemorizationSignal values"
                )
        ordered = tuple(sorted(set(signals), key=lambda entry: entry.value))
        if ordered != signals:
            raise SyntheticMemorizationError(
                "blocking_signals must be sorted and free of duplicates"
            )
        if not isinstance(self.max_findings, int) or isinstance(
            self.max_findings, bool
        ):
            raise SyntheticMemorizationError("max_findings must be an integer")
        if not 1 <= self.max_findings <= 10_000:
            raise SyntheticMemorizationError("max_findings must be between 1 and 10000")
        if not isinstance(self.max_candidate_fragments, int) or isinstance(
            self.max_candidate_fragments, bool
        ):
            raise SyntheticMemorizationError(
                "max_candidate_fragments must be an integer"
            )
        if not 1 <= self.max_candidate_fragments <= 10_000:
            raise SyntheticMemorizationError(
                "max_candidate_fragments must be between 1 and 10000"
            )


@dataclass(frozen=True)
class ReferenceFragmentFingerprint:
    """Value-free fingerprint of one normalized protected reference fragment.

    Attributes:
        digest: SHA-256 digest of the normalized fragment text.
        token_count: Number of whitespace tokens in the normalized fragment.
        shingle_hashes: Sorted unique 64-bit hashes of the fragment's token
            shingles.
    """

    digest: str
    token_count: int
    shingle_hashes: tuple[int, ...]

    def __post_init__(self) -> None:
        _require_digest(self.digest, label="fragment digest")
        if not isinstance(self.token_count, int) or isinstance(self.token_count, bool):
            raise SyntheticMemorizationError("token_count must be an integer")
        if self.token_count < 1:
            raise SyntheticMemorizationError("token_count must be positive")
        if not isinstance(self.shingle_hashes, tuple):
            raise SyntheticMemorizationError("shingle_hashes must be a tuple")
        if not self.shingle_hashes:
            raise SyntheticMemorizationError("shingle_hashes must not be empty")
        for value in self.shingle_hashes:
            if not isinstance(value, int) or isinstance(value, bool):
                raise SyntheticMemorizationError("shingle hashes must be integers")
            if value < 0:
                raise SyntheticMemorizationError("shingle hashes must be unsigned")
        if tuple(sorted(set(self.shingle_hashes))) != self.shingle_hashes:
            raise SyntheticMemorizationError(
                "shingle_hashes must be sorted and free of duplicates"
            )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready mapping for this fragment fingerprint."""
        return {
            "digest": self.digest,
            "token_count": self.token_count,
            "shingle_hashes": list(self.shingle_hashes),
        }


@dataclass(frozen=True)
class ProtectedReferenceFingerprint:
    """Fingerprints of a protected reference held for memorization auditing.

    Attributes:
        reference_id: Caller-assigned identifier for the protected reference.
        digest: SHA-256 digest of the normalized reference text.
        char_length: Length of the normalized reference text.
        fragments: Fragment fingerprints sorted by digest.
        ngram_digests: Sorted unique character n-gram digests of the
            normalized reference text.
        ngram_size: Character n-gram size used to build ``ngram_digests``.
    """

    reference_id: str
    digest: str
    char_length: int
    fragments: tuple[ReferenceFragmentFingerprint, ...]
    ngram_digests: tuple[str, ...]
    ngram_size: int

    def __post_init__(self) -> None:
        _require_identifier(self.reference_id, label="reference_id")
        _require_digest(self.digest, label="reference digest")
        if not isinstance(self.char_length, int) or isinstance(self.char_length, bool):
            raise SyntheticMemorizationError("char_length must be an integer")
        if self.char_length < 1:
            raise SyntheticMemorizationError("char_length must be positive")
        if not isinstance(self.fragments, tuple) or not self.fragments:
            raise SyntheticMemorizationError("fragments must be a non-empty tuple")
        for fragment in self.fragments:
            if not isinstance(fragment, ReferenceFragmentFingerprint):
                raise SyntheticMemorizationError(
                    "fragments must contain ReferenceFragmentFingerprint values"
                )
        ordered = tuple(sorted(self.fragments, key=lambda entry: entry.digest))
        if ordered != self.fragments:
            raise SyntheticMemorizationError("fragments must be sorted by digest")
        digests = [fragment.digest for fragment in self.fragments]
        if len(set(digests)) != len(digests):
            raise SyntheticMemorizationError("fragment digests must be unique")
        if not isinstance(self.ngram_digests, tuple):
            raise SyntheticMemorizationError("ngram_digests must be a tuple")
        for value in self.ngram_digests:
            _require_digest(value, label="n-gram digest")
        if tuple(sorted(set(self.ngram_digests))) != self.ngram_digests:
            raise SyntheticMemorizationError(
                "ngram_digests must be sorted and free of duplicates"
            )
        if not isinstance(self.ngram_size, int) or isinstance(self.ngram_size, bool):
            raise SyntheticMemorizationError("ngram_size must be an integer")
        if not 1 <= self.ngram_size <= MAX_NGRAM_SIZE:
            raise SyntheticMemorizationError(
                f"ngram_size must be between 1 and {MAX_NGRAM_SIZE}"
            )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready mapping for this reference fingerprint."""
        return {
            "reference_id": self.reference_id,
            "digest": self.digest,
            "char_length": self.char_length,
            "ngram_size": self.ngram_size,
            "fragments": [fragment.to_dict() for fragment in self.fragments],
            "ngram_digests": list(self.ngram_digests),
        }


def fingerprint_reference(
    reference_id: str,
    text: str,
    *,
    policy: SyntheticMemorizationPolicy | None = None,
) -> ProtectedReferenceFingerprint:
    """Fingerprint a protected reference without retaining its text.

    Args:
        reference_id: Lowercase identifier for the protected reference.
        text: Raw protected reference text. It is normalized, digested, and
            discarded; only digests and aggregate counts are kept.
        policy: Thresholds controlling fragment and n-gram granularity.

    Returns:
        The value-free reference fingerprint.

    Raises:
        SyntheticMemorizationError: If the identifier is invalid, the text is
            not auditable, or it falls below the policy fragment floor.
    """
    resolved = policy or SyntheticMemorizationPolicy()
    _require_identifier(reference_id, label="reference_id")
    _require_text(text, label="reference text")
    normalized = normalize_text(text)
    if len(normalized) < resolved.min_fragment_chars:
        raise SyntheticMemorizationError(
            "reference text is below the minimum auditable fragment length"
        )
    fragments: dict[str, ReferenceFragmentFingerprint] = {}
    for fragment in _iter_fragments(normalized, resolved.min_fragment_chars):
        fragment_digest = _digest(fragment)
        if fragment_digest in fragments:
            continue
        tokens = _TOKEN_RE.findall(fragment)
        shingles = _token_shingles(tokens, resolved.shingle_size)
        if not shingles:
            continue
        fragments[fragment_digest] = ReferenceFragmentFingerprint(
            digest=fragment_digest,
            token_count=len(tokens),
            shingle_hashes=shingles,
        )
    if not fragments:
        raise SyntheticMemorizationError(
            "reference text has no auditable fragment shingles"
        )
    ordered = tuple(fragments[digest] for digest in sorted(fragments))
    return ProtectedReferenceFingerprint(
        reference_id=reference_id,
        digest=_digest(normalized),
        char_length=len(normalized),
        fragments=ordered,
        ngram_digests=_char_ngram_digests(normalized, resolved.exposure_ngram_size),
        ngram_size=resolved.exposure_ngram_size,
    )


def _iter_fragments(normalized: str, minimum: int) -> tuple[str, ...]:
    fragments: list[str] = []
    start = 0
    for match in _FRAGMENT_BOUNDARY_RE.finditer(normalized):
        candidate = normalized[start : match.start()].strip()
        if len(candidate) >= minimum:
            fragments.append(candidate)
        start = match.end()
    tail = normalized[start:].strip()
    if len(tail) >= minimum:
        fragments.append(tail)
    return tuple(fragments)


@dataclass(frozen=True)
class MemorizationFinding:
    """One memorization signal collected for a candidate/reference pair.

    Attributes:
        candidate_id: Identifier of the audited synthetic candidate.
        reference_id: Identifier of the protected reference that matched.
        signal: Signal that produced the finding.
        reason: Machine-readable reason code for the finding.
        fragment_digest: SHA-256 digest of the matched normalized fragment or
            of the matched candidate text for the exposure signal.
        candidate_offset: Offset of the matched candidate fragment in the
            normalized candidate text.
        candidate_length: Length of the matched candidate fragment.
        score: ``1.0`` for exact matches, the shingle similarity for fuzzy
            matches, and the matched n-gram share for exposure matches.
    """

    candidate_id: str
    reference_id: str
    signal: MemorizationSignal
    reason: MemorizationReasonCode
    fragment_digest: str
    candidate_offset: int
    candidate_length: int
    score: float

    def __post_init__(self) -> None:
        _require_identifier(self.candidate_id, label="candidate_id")
        _require_identifier(self.reference_id, label="reference_id")
        if not isinstance(self.signal, MemorizationSignal):
            raise SyntheticMemorizationError(
                "signal must be a MemorizationSignal value"
            )
        if not isinstance(self.reason, MemorizationReasonCode):
            raise SyntheticMemorizationError(
                "reason must be a MemorizationReasonCode value"
            )
        if _REASON_BY_SIGNAL[self.signal] is not self.reason:
            raise SyntheticMemorizationError(
                "reason must match the reported memorization signal"
            )
        _require_digest(self.fragment_digest, label="fragment_digest")
        for value, label in (
            (self.candidate_offset, "candidate_offset"),
            (self.candidate_length, "candidate_length"),
        ):
            if not isinstance(value, int) or isinstance(value, bool):
                raise SyntheticMemorizationError(f"{label} must be an integer")
            if value < 0:
                raise SyntheticMemorizationError(f"{label} must not be negative")
        if self.candidate_length < 1:
            raise SyntheticMemorizationError("candidate_length must be positive")
        if not isinstance(self.score, (int, float)) or isinstance(self.score, bool):
            raise SyntheticMemorizationError("score must be a number")
        if not 0.0 <= float(self.score) <= 1.0:
            raise SyntheticMemorizationError("score must be between 0 and 1")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready mapping for this finding."""
        return {
            "candidate_id": self.candidate_id,
            "reference_id": self.reference_id,
            "signal": self.signal.value,
            "reason": self.reason.value,
            "fragment_digest": self.fragment_digest,
            "candidate_offset": self.candidate_offset,
            "candidate_length": self.candidate_length,
            "score": float(self.score),
        }


@dataclass(frozen=True)
class SyntheticMemorizationReport:
    """Deterministic outcome of a synthetic memorization audit.

    Attributes:
        verdict: ``blocked`` when a blocking signal fired, else ``clear``.
        findings: Findings ordered by candidate, offset, signal, and reference.
        blocking_signals: Signals configured to withhold a release.
        candidate_count: Number of audited candidates.
        reference_count: Number of audited protected references.
        skipped_candidate_count: Candidates skipped for having no auditable
            text.
        truncated: Whether the finding list was capped by the policy.
        schema_version: Report schema version.
    """

    verdict: MemorizationVerdict
    findings: tuple[MemorizationFinding, ...]
    blocking_signals: tuple[MemorizationSignal, ...]
    candidate_count: int
    reference_count: int
    skipped_candidate_count: int
    truncated: bool
    schema_version: str = SYNTHETIC_MEMORIZATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.verdict, MemorizationVerdict):
            raise SyntheticMemorizationError(
                "verdict must be a MemorizationVerdict value"
            )
        if not isinstance(self.findings, tuple):
            raise SyntheticMemorizationError("findings must be a tuple")
        for finding in self.findings:
            if not isinstance(finding, MemorizationFinding):
                raise SyntheticMemorizationError(
                    "findings must contain MemorizationFinding values"
                )
        keys = [
            (
                finding.candidate_id,
                finding.candidate_offset,
                finding.signal.value,
                finding.reference_id,
                finding.fragment_digest,
            )
            for finding in self.findings
        ]
        if keys != sorted(keys):
            raise SyntheticMemorizationError(
                "findings must be deterministically ordered"
            )
        signals = self.blocking_signals
        if not isinstance(signals, tuple) or not signals:
            raise SyntheticMemorizationError(
                "blocking_signals must be a non-empty tuple"
            )
        for signal in signals:
            if not isinstance(signal, MemorizationSignal):
                raise SyntheticMemorizationError(
                    "blocking_signals must contain MemorizationSignal values"
                )
        expected = (
            MemorizationVerdict.BLOCKED
            if any(finding.signal in signals for finding in self.findings)
            else MemorizationVerdict.CLEAR
        )
        if self.verdict is not expected:
            raise SyntheticMemorizationError(
                "verdict must be derived from the blocking findings"
            )
        for value, label in (
            (self.candidate_count, "candidate_count"),
            (self.reference_count, "reference_count"),
            (self.skipped_candidate_count, "skipped_candidate_count"),
        ):
            if not isinstance(value, int) or isinstance(value, bool):
                raise SyntheticMemorizationError(f"{label} must be an integer")
            if value < 0:
                raise SyntheticMemorizationError(f"{label} must not be negative")
        if self.skipped_candidate_count > self.candidate_count:
            raise SyntheticMemorizationError(
                "skipped_candidate_count must not exceed candidate_count"
            )
        if not isinstance(self.truncated, bool):
            raise SyntheticMemorizationError("truncated must be a boolean")
        if not isinstance(self.schema_version, str) or not self.schema_version:
            raise SyntheticMemorizationError("schema_version must be a string")

    @property
    def blocked(self) -> bool:
        """Whether the audit withholds a dataset release."""
        return self.verdict is MemorizationVerdict.BLOCKED

    @property
    def ok(self) -> bool:
        """Whether the audit found no blocking memorization signal."""
        return not self.blocked

    @property
    def signals(self) -> tuple[MemorizationSignal, ...]:
        """Signals observed in the retained findings, in a stable order."""
        return tuple(
            signal
            for signal in _SIGNAL_ORDER
            if any(finding.signal is signal for finding in self.findings)
        )

    @property
    def signal_counts(self) -> dict[str, int]:
        """Finding counts keyed by signal value."""
        return {
            signal.value: sum(
                1 for finding in self.findings if finding.signal is signal
            )
            for signal in _SIGNAL_ORDER
        }

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready, value-free mapping for this report."""
        return {
            "schema_version": self.schema_version,
            "verdict": self.verdict.value,
            "blocked": self.blocked,
            "truncated": self.truncated,
            "candidate_count": self.candidate_count,
            "reference_count": self.reference_count,
            "skipped_candidate_count": self.skipped_candidate_count,
            "blocking_signals": [signal.value for signal in self.blocking_signals],
            "signals": [signal.value for signal in self.signals],
            "signal_counts": self.signal_counts,
            "findings": [finding.to_dict() for finding in self.findings],
        }

    def to_json(self) -> str:
        """Return the canonical JSON encoding of :meth:`to_dict`."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> SyntheticMemorizationReport:
        """Rebuild a report from a :meth:`to_dict` payload.

        Args:
            payload: Mapping previously produced by :meth:`to_dict`.

        Returns:
            The reconstructed report.

        Raises:
            SyntheticMemorizationError: If the payload is malformed.
        """
        if not isinstance(payload, Mapping):
            raise SyntheticMemorizationError("report payload must be a mapping")
        version = payload.get("schema_version")
        if version != SYNTHETIC_MEMORIZATION_SCHEMA_VERSION:
            raise SyntheticMemorizationError(
                "report payload has an unsupported schema_version"
            )
        try:
            verdict = MemorizationVerdict(payload["verdict"])
        except (KeyError, ValueError) as error:
            raise SyntheticMemorizationError("verdict must be a known value") from error
        try:
            blocking = tuple(
                MemorizationSignal(value) for value in payload["blocking_signals"]
            )
        except (KeyError, TypeError, ValueError) as error:
            raise SyntheticMemorizationError(
                "blocking_signals must be known signal values"
            ) from error
        raw_findings = payload.get("findings")
        if not isinstance(raw_findings, list):
            raise SyntheticMemorizationError("findings must be a list")
        findings = tuple(_finding_from_dict(entry) for entry in raw_findings)
        return cls(
            verdict=verdict,
            findings=findings,
            blocking_signals=blocking,
            candidate_count=_require_count(payload, "candidate_count"),
            reference_count=_require_count(payload, "reference_count"),
            skipped_candidate_count=_require_count(payload, "skipped_candidate_count"),
            truncated=_require_bool(payload, "truncated"),
            schema_version=version,
        )

    @classmethod
    def from_json(cls, payload: str) -> SyntheticMemorizationReport:
        """Rebuild a report from its canonical JSON encoding."""
        try:
            decoded = json.loads(payload)
        except (TypeError, ValueError) as error:
            raise SyntheticMemorizationError(
                "report payload is not valid JSON"
            ) from error
        if not isinstance(decoded, Mapping):
            raise SyntheticMemorizationError("report payload must be a JSON object")
        return cls.from_dict(decoded)


def _require_count(payload: Mapping[str, Any], key: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise SyntheticMemorizationError(f"{key} must be a non-negative integer")
    return value


def _require_bool(payload: Mapping[str, Any], key: str) -> bool:
    value = payload.get(key)
    if not isinstance(value, bool):
        raise SyntheticMemorizationError(f"{key} must be a boolean")
    return value


def _finding_from_dict(entry: Any) -> MemorizationFinding:
    if not isinstance(entry, Mapping):
        raise SyntheticMemorizationError("each finding must be a mapping")
    try:
        signal = MemorizationSignal(entry["signal"])
        reason = MemorizationReasonCode(entry["reason"])
    except (KeyError, ValueError) as error:
        raise SyntheticMemorizationError(
            "finding signal and reason must be known values"
        ) from error
    try:
        return MemorizationFinding(
            candidate_id=entry["candidate_id"],
            reference_id=entry["reference_id"],
            signal=signal,
            reason=reason,
            fragment_digest=entry["fragment_digest"],
            candidate_offset=entry["candidate_offset"],
            candidate_length=entry["candidate_length"],
            score=entry["score"],
        )
    except KeyError as error:
        raise SyntheticMemorizationError("finding payload is incomplete") from error


def assert_release_allowed(report: SyntheticMemorizationReport) -> None:
    """Raise when a report withholds the dataset release.

    The raised message exposes only counts and signal names.

    Args:
        report: Audit report to enforce.

    Raises:
        SyntheticMemorizationError: If the report is blocked.
    """
    if not isinstance(report, SyntheticMemorizationReport):
        raise SyntheticMemorizationError("report must be a SyntheticMemorizationReport")
    if report.blocked:
        raise SyntheticMemorizationError(
            "synthetic memorization audit blocked the dataset release: "
            f"{len(report.findings)} finding(s) across "
            f"{len(report.signals)} signal(s)"
        )


def audit_synthetic_memorization(
    candidates: Mapping[str, str] | Iterable[tuple[str, str]],
    references: Sequence[ProtectedReferenceFingerprint],
    *,
    policy: SyntheticMemorizationPolicy | None = None,
) -> SyntheticMemorizationReport:
    """Audit synthetic candidates against protected reference fingerprints.

    Args:
        candidates: Mapping or iterable of ``(candidate_id, text)`` pairs.
            Candidate identifiers must be lowercase identifiers.
        references: Protected reference fingerprints produced by
            :func:`fingerprint_reference`.
        policy: Thresholds and bounds; defaults to
            :class:`SyntheticMemorizationPolicy`.

    Returns:
        A deterministic, value-free audit report. Findings are ordered by
        candidate identifier, candidate offset, signal, and reference
        identifier, and are capped by ``policy.max_findings``.

    Raises:
        SyntheticMemorizationError: If inputs violate the audit bounds.
    """
    resolved = policy or SyntheticMemorizationPolicy()
    if not isinstance(resolved, SyntheticMemorizationPolicy):
        raise SyntheticMemorizationError("policy must be a SyntheticMemorizationPolicy")
    if not isinstance(references, Sequence) or isinstance(references, (str, bytes)):
        raise SyntheticMemorizationError("references must be a sequence")
    if len(references) > MAX_REFERENCES:
        raise SyntheticMemorizationError(
            f"references exceed the {MAX_REFERENCES}-entry audit bound"
        )
    for reference in references:
        if not isinstance(reference, ProtectedReferenceFingerprint):
            raise SyntheticMemorizationError(
                "references must contain ProtectedReferenceFingerprint values"
            )
    reference_ids = [reference.reference_id for reference in references]
    if len(set(reference_ids)) != len(reference_ids):
        raise SyntheticMemorizationError("reference identifiers must be unique")
    ordered_references = tuple(sorted(references, key=lambda entry: entry.reference_id))

    items = candidates.items() if isinstance(candidates, Mapping) else candidates
    normalized_candidates: list[tuple[str, str]] = []
    seen: set[str] = set()
    for candidate_id, text in items:
        _require_identifier(candidate_id, label="candidate_id")
        if candidate_id in seen:
            raise SyntheticMemorizationError("candidate identifiers must be unique")
        seen.add(candidate_id)
        _require_text(text, label="candidate text")
        if len(seen) > MAX_CANDIDATES:
            raise SyntheticMemorizationError(
                f"candidates exceed the {MAX_CANDIDATES}-entry audit bound"
            )
        normalized_candidates.append((candidate_id, normalize_text(text)))
    normalized_candidates.sort(key=lambda entry: entry[0])

    findings: list[MemorizationFinding] = []
    skipped = 0
    for candidate_id, normalized in normalized_candidates:
        if not normalized:
            skipped += 1
            continue
        fragment_matches = _candidate_fragments(
            candidate_id,
            normalized,
            references=ordered_references,
            policy=resolved,
        )
        findings.extend(fragment_matches)
        findings.extend(
            _exposure_findings(
                candidate_id,
                normalized,
                references=ordered_references,
                policy=resolved,
            )
        )

    findings.sort(
        key=lambda finding: (
            finding.candidate_id,
            finding.candidate_offset,
            finding.signal.value,
            finding.reference_id,
            finding.fragment_digest,
        )
    )
    truncated = len(findings) > resolved.max_findings
    retained = tuple(findings[: resolved.max_findings])
    return SyntheticMemorizationReport(
        verdict=(
            MemorizationVerdict.BLOCKED
            if any(finding.signal in resolved.blocking_signals for finding in retained)
            else MemorizationVerdict.CLEAR
        ),
        findings=retained,
        blocking_signals=resolved.blocking_signals,
        candidate_count=len(normalized_candidates),
        reference_count=len(ordered_references),
        skipped_candidate_count=skipped,
        truncated=truncated,
    )


def _candidate_fragments(
    candidate_id: str,
    normalized: str,
    *,
    references: Sequence[ProtectedReferenceFingerprint],
    policy: SyntheticMemorizationPolicy,
) -> tuple[MemorizationFinding, ...]:
    fragments: list[tuple[str, int, int]] = []
    start = 0
    for match in _FRAGMENT_BOUNDARY_RE.finditer(normalized):
        fragments.extend(_fragment_span(normalized, start, match.start(), policy))
        start = match.end()
    fragments.extend(_fragment_span(normalized, start, len(normalized), policy))
    if len(fragments) > policy.max_candidate_fragments:
        raise SyntheticMemorizationError(
            "candidate exposes more fragments than the audit bound"
        )
    candidates = tuple(fragments)
    if not candidates or not references:
        return ()

    exact_index: dict[str, str] = {}
    for reference in references:
        for fragment in reference.fragments:
            exact_index.setdefault(fragment.digest, reference.reference_id)

    findings: list[MemorizationFinding] = []
    for fragment_text, offset, length in candidates:
        fragment_digest = _digest(fragment_text)
        reference_id = exact_index.get(fragment_digest)
        if reference_id is not None:
            findings.append(
                MemorizationFinding(
                    candidate_id=candidate_id,
                    reference_id=reference_id,
                    signal=MemorizationSignal.EXACT,
                    reason=MemorizationReasonCode.EXACT_FRAGMENT_MATCH,
                    fragment_digest=fragment_digest,
                    candidate_offset=offset,
                    candidate_length=length,
                    score=1.0,
                )
            )
        shingles = _token_shingles(
            _TOKEN_RE.findall(fragment_text), policy.shingle_size
        )
        if not shingles:
            continue
        shingle_set = set(shingles)
        for reference in references:
            best = 0.0
            for protected in reference.fragments:
                score = _jaccard(shingle_set, set(protected.shingle_hashes))
                if score > best:
                    best = score
            if best >= float(policy.fuzzy_similarity):
                findings.append(
                    MemorizationFinding(
                        candidate_id=candidate_id,
                        reference_id=reference.reference_id,
                        signal=MemorizationSignal.FUZZY,
                        reason=MemorizationReasonCode.FUZZY_FRAGMENT_MATCH,
                        fragment_digest=fragment_digest,
                        candidate_offset=offset,
                        candidate_length=length,
                        score=best,
                    )
                )
    return tuple(findings)


def _fragment_span(
    normalized: str,
    start: int,
    end: int,
    policy: SyntheticMemorizationPolicy,
) -> list[tuple[str, int, int]]:
    raw = normalized[start:end]
    stripped = raw.strip()
    if len(stripped) < policy.min_fragment_chars:
        return []
    offset = start + (len(raw) - len(raw.lstrip()))
    return [(stripped, offset, len(stripped))]


def _jaccard(left: set[int], right: set[int]) -> float:
    if not left or not right:
        return 0.0
    union = left | right
    return len(left & right) / len(union)


def _exposure_findings(
    candidate_id: str,
    normalized: str,
    *,
    references: Sequence[ProtectedReferenceFingerprint],
    policy: SyntheticMemorizationPolicy,
) -> tuple[MemorizationFinding, ...]:
    ngrams = set(_char_ngram_digests(normalized, policy.exposure_ngram_size))
    if not ngrams:
        return ()
    findings: list[MemorizationFinding] = []
    for reference in references:
        if not reference.ngram_digests:
            continue
        protected = set(reference.ngram_digests)
        ratio = len(ngrams & protected) / len(protected)
        if ratio >= float(policy.exposure_match_ratio):
            findings.append(
                MemorizationFinding(
                    candidate_id=candidate_id,
                    reference_id=reference.reference_id,
                    signal=MemorizationSignal.EXPOSURE,
                    reason=MemorizationReasonCode.EXPOSURE_NGRAM_MATCH,
                    fragment_digest=_digest(normalized),
                    candidate_offset=0,
                    candidate_length=len(normalized),
                    score=ratio,
                )
            )
    return tuple(findings)
