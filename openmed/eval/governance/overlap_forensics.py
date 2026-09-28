"""Deterministic, privacy-safe overlap forensics for benchmark submissions.

The scanner works entirely in memory and emits ordinal references rather than
submitted or benchmark text. Reports bind their findings to an existing sealed
workflow manifest and associate them with a holdout commitment without
performing network requests.
"""

from __future__ import annotations

import json
import unicodedata
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Sequence, cast

from openmed.eval.governance.holdout_commitment import (
    HoldoutCommitment,
    verify_holdout_commitment,
)
from openmed.eval.workflows.sealed_manifest import SealedWorkflowManifest

OVERLAP_FORENSICS_SCHEMA_VERSION = "openmed.eval.overlap_forensics.v1"
OVERLAP_FORENSICS_POLICY_VERSION = "openmed.eval.overlap_forensics.policy.v1"

CHECK_EXACT = "exact_overlap"
CHECK_NORMALIZED = "normalized_overlap"
CHECK_FUZZY = "fuzzy_overlap"
CHECK_CANARY = "canary_signal"
CHECK_PUBLIC_VS_SHADOW = "public_vs_shadow"
OVERLAP_FORENSICS_CHECKS = (
    CHECK_EXACT,
    CHECK_NORMALIZED,
    CHECK_FUZZY,
    CHECK_CANARY,
    CHECK_PUBLIC_VS_SHADOW,
)

DEFAULT_FUZZY_THRESHOLD_BASIS_POINTS = 8_500
DEFAULT_SHADOW_MARGIN_BASIS_POINTS = 1_000

_SCOPE_ORDER = {"public": 0, "shadow": 1, "canary": 2}
_MANIFEST_KEYS = frozenset({"schema_version", "component_digests", "manifest_digest"})

INTERPRETATION_SIGNALS = (
    "Detected overlap or canary signals within the configured checks. "
    "These signals require review and do not by themselves establish "
    "contamination or hardcoding."
)
INTERPRETATION_NO_SIGNALS = (
    "No signals were detected within the configured checks and inputs. "
    "This limited result does not establish the absence of contamination "
    "or hardcoding."
)


class OverlapForensicsError(ValueError):
    """Raised when a forensic scan cannot be configured safely."""


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    characters = [
        character if unicodedata.category(character)[0] in {"L", "N"} else " "
        for character in normalized
    ]
    return " ".join("".join(characters).split())


def _trigrams(value: str) -> frozenset[str]:
    if len(value) < 3:
        return frozenset((value,))
    return frozenset(value[index : index + 3] for index in range(len(value) - 2))


def _similarity_basis_points(left: str, right: str) -> int:
    left_trigrams = _trigrams(left)
    right_trigrams = _trigrams(right)
    union_size = len(left_trigrams | right_trigrams)
    if not union_size:
        return 10_000
    return len(left_trigrams & right_trigrams) * 10_000 // union_size


def _validate_text_items(name: str, values: Sequence[str]) -> tuple[str, ...]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise OverlapForensicsError(f"{name}: invalid_sequence")
    normalized = tuple(values)
    if not normalized:
        raise OverlapForensicsError(f"{name}: empty")
    if not all(type(value) is str and value.strip() for value in normalized):
        raise OverlapForensicsError(f"{name}: invalid_item")
    return normalized


def _validate_threshold(name: str, value: int) -> int:
    if type(value) is not int or not 0 <= value <= 10_000:
        raise OverlapForensicsError(f"{name}: invalid_basis_points")
    return value


def _submission_manifest_digest(
    manifest: SealedWorkflowManifest | Mapping[str, Any],
) -> str:
    if isinstance(manifest, SealedWorkflowManifest):
        return manifest.manifest_digest
    if not isinstance(manifest, Mapping) or set(manifest) != _MANIFEST_KEYS:
        raise OverlapForensicsError("submission_manifest: invalid")
    try:
        parsed = SealedWorkflowManifest(
            component_digests=cast(
                Mapping[str, str], manifest.get("component_digests")
            ),
            manifest_digest=cast(str, manifest.get("manifest_digest")),
            schema_version=cast(str, manifest.get("schema_version")),
        )
    except (TypeError, ValueError):
        raise OverlapForensicsError("submission_manifest: invalid") from None
    return parsed.manifest_digest


def _holdout_commitment_digest(
    commitment: HoldoutCommitment | Mapping[str, Any],
) -> str:
    verification = verify_holdout_commitment(commitment)
    if not verification.valid:
        raise OverlapForensicsError("holdout_commitment: invalid")
    if isinstance(commitment, HoldoutCommitment):
        return commitment.commitment_digest
    return cast(str, commitment["commitment_digest"])


def _item_ref(scope: str, index: int) -> str:
    return f"{scope}:{index + 1:06d}"


@dataclass(frozen=True, slots=True)
class OverlapSignal:
    """Content-free evidence for one detected overlap or canary signal."""

    check: str
    submission_ref: str
    reference_scope: str
    reference_ref: str
    score_basis_points: int
    comparison_score_basis_points: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a stable JSON-compatible signal without source text."""
        return {
            "check": self.check,
            "comparison_score_basis_points": self.comparison_score_basis_points,
            "reference_ref": self.reference_ref,
            "reference_scope": self.reference_scope,
            "score_basis_points": self.score_basis_points,
            "submission_ref": self.submission_ref,
        }


@dataclass(frozen=True, slots=True)
class OverlapForensicsReport:
    """Versioned overlap report containing signals and explicit coverage."""

    submission_manifest_digest: str
    holdout_commitment_digest: str
    fuzzy_threshold_basis_points: int
    shadow_margin_basis_points: int
    coverage: Mapping[str, Any]
    signal_counts: Mapping[str, int]
    signals: tuple[OverlapSignal, ...]
    interpretation: str
    policy_version: str = OVERLAP_FORENSICS_POLICY_VERSION
    schema_version: str = OVERLAP_FORENSICS_SCHEMA_VERSION

    def __post_init__(self) -> None:
        frozen_coverage = dict(self.coverage)
        frozen_coverage["checks"] = tuple(frozen_coverage["checks"])
        frozen_coverage["comparison_counts"] = MappingProxyType(
            dict(frozen_coverage["comparison_counts"])
        )
        object.__setattr__(self, "coverage", MappingProxyType(frozen_coverage))
        object.__setattr__(
            self, "signal_counts", MappingProxyType(dict(self.signal_counts))
        )
        object.__setattr__(self, "signals", tuple(self.signals))

    @property
    def signals_detected(self) -> bool:
        """Return whether any configured check emitted a signal."""
        return bool(self.signals)

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical JSON-compatible forensic report."""
        coverage = dict(self.coverage)
        coverage["checks"] = list(coverage["checks"])
        coverage["comparison_counts"] = dict(coverage["comparison_counts"])
        return {
            "coverage": coverage,
            "fuzzy_threshold_basis_points": self.fuzzy_threshold_basis_points,
            "holdout_commitment_digest": self.holdout_commitment_digest,
            "interpretation": self.interpretation,
            "policy_version": self.policy_version,
            "schema_version": self.schema_version,
            "shadow_margin_basis_points": self.shadow_margin_basis_points,
            "signal_counts": dict(self.signal_counts),
            "signals": [signal.to_dict() for signal in self.signals],
            "signals_detected": self.signals_detected,
            "submission_manifest_digest": self.submission_manifest_digest,
        }

    def to_json(self) -> str:
        """Return deterministic compact JSON suitable for evidence storage."""
        return _canonical_json(self.to_dict())


def _overlap_signals(
    submissions: Sequence[str],
    normalized_submissions: Sequence[str],
    references: Sequence[str],
    normalized_references: Sequence[str],
    scope: str,
    fuzzy_threshold_basis_points: int,
) -> list[OverlapSignal]:
    signals: list[OverlapSignal] = []
    for submission_index, (submission, normalized_submission) in enumerate(
        zip(submissions, normalized_submissions)
    ):
        for reference_index, (reference, normalized_reference) in enumerate(
            zip(references, normalized_references)
        ):
            submission_ref = _item_ref("submission", submission_index)
            reference_ref = _item_ref(scope, reference_index)
            score = _similarity_basis_points(
                normalized_submission, normalized_reference
            )
            if submission == reference:
                signals.append(
                    OverlapSignal(
                        CHECK_EXACT,
                        submission_ref,
                        scope,
                        reference_ref,
                        10_000,
                    )
                )
                continue
            if normalized_submission == normalized_reference:
                signals.append(
                    OverlapSignal(
                        CHECK_NORMALIZED,
                        submission_ref,
                        scope,
                        reference_ref,
                        10_000,
                    )
                )
                continue
            if score >= fuzzy_threshold_basis_points:
                signals.append(
                    OverlapSignal(
                        CHECK_FUZZY,
                        submission_ref,
                        scope,
                        reference_ref,
                        score,
                    )
                )
    return signals


def _canary_signals(
    normalized_submissions: Sequence[str],
    normalized_canaries: Sequence[str],
) -> list[OverlapSignal]:
    signals: list[OverlapSignal] = []
    for submission_index, submission in enumerate(normalized_submissions):
        for canary_index, canary in enumerate(normalized_canaries):
            if canary in submission:
                signals.append(
                    OverlapSignal(
                        CHECK_CANARY,
                        _item_ref("submission", submission_index),
                        "canary",
                        _item_ref("canary", canary_index),
                        10_000,
                    )
                )
    return signals


def _best_similarity(submission: str, references: Sequence[str]) -> tuple[int, int]:
    scores = tuple(
        _similarity_basis_points(submission, reference) for reference in references
    )
    best_score = max(scores)
    return best_score, scores.index(best_score)


def _public_vs_shadow_signals(
    normalized_submissions: Sequence[str],
    normalized_public: Sequence[str],
    normalized_shadow: Sequence[str],
    fuzzy_threshold_basis_points: int,
    shadow_margin_basis_points: int,
) -> list[OverlapSignal]:
    signals: list[OverlapSignal] = []
    for submission_index, submission in enumerate(normalized_submissions):
        public_score, _ = _best_similarity(submission, normalized_public)
        shadow_score, shadow_index = _best_similarity(submission, normalized_shadow)
        if (
            shadow_score >= fuzzy_threshold_basis_points
            and shadow_score - public_score >= shadow_margin_basis_points
        ):
            signals.append(
                OverlapSignal(
                    CHECK_PUBLIC_VS_SHADOW,
                    _item_ref("submission", submission_index),
                    "shadow",
                    _item_ref("shadow", shadow_index),
                    shadow_score,
                    comparison_score_basis_points=public_score,
                )
            )
    return signals


def scan_overlap_forensics(
    submission_items: Sequence[str],
    *,
    public_items: Sequence[str],
    shadow_items: Sequence[str],
    canaries: Sequence[str],
    submission_manifest: SealedWorkflowManifest | Mapping[str, Any],
    holdout_commitment: HoldoutCommitment | Mapping[str, Any],
    fuzzy_threshold_basis_points: int = DEFAULT_FUZZY_THRESHOLD_BASIS_POINTS,
    shadow_margin_basis_points: int = DEFAULT_SHADOW_MARGIN_BASIS_POINTS,
) -> OverlapForensicsReport:
    """Scan benchmark submission text for five classes of forensic signal.

    Exact, normalized, fuzzy, canary, and public-versus-shadow checks run
    locally. Inputs are processed in memory; the returned report includes only
    ordinal references, counts, scores, policy versions, and upstream digests.

    Args:
        submission_items: Ordered textual artifacts from the sealed submission.
        public_items: Ordered public benchmark artifacts used for comparison.
        shadow_items: Ordered evaluator-held shadow artifacts.
        canaries: Ordered evaluator-held marker strings.
        submission_manifest: Valid sealed workflow manifest or parsed document.
        holdout_commitment: Valid holdout commitment or parsed document.
        fuzzy_threshold_basis_points: Inclusive trigram-Jaccard threshold.
        shadow_margin_basis_points: Minimum shadow-over-public score difference.

    Returns:
        A deterministic, content-free :class:`OverlapForensicsReport`.

    Raises:
        OverlapForensicsError: If configuration or upstream evidence is invalid.
    """
    submissions = _validate_text_items("submission_items", submission_items)
    public = _validate_text_items("public_items", public_items)
    shadow = _validate_text_items("shadow_items", shadow_items)
    canary_items = _validate_text_items("canaries", canaries)
    fuzzy_threshold = _validate_threshold(
        "fuzzy_threshold_basis_points", fuzzy_threshold_basis_points
    )
    shadow_margin = _validate_threshold(
        "shadow_margin_basis_points", shadow_margin_basis_points
    )
    manifest_digest = _submission_manifest_digest(submission_manifest)
    commitment_digest = _holdout_commitment_digest(holdout_commitment)

    normalized_submissions = tuple(_normalize_text(item) for item in submissions)
    normalized_public = tuple(_normalize_text(item) for item in public)
    normalized_shadow = tuple(_normalize_text(item) for item in shadow)
    normalized_canaries = tuple(_normalize_text(item) for item in canary_items)
    if not all(normalized_submissions):
        raise OverlapForensicsError("submission_items: empty_after_normalization")
    if not all(normalized_public):
        raise OverlapForensicsError("public_items: empty_after_normalization")
    if not all(normalized_shadow):
        raise OverlapForensicsError("shadow_items: empty_after_normalization")
    if not all(normalized_canaries):
        raise OverlapForensicsError("canaries: empty_after_normalization")

    signals = _overlap_signals(
        submissions,
        normalized_submissions,
        public,
        normalized_public,
        "public",
        fuzzy_threshold,
    )
    signals.extend(
        _overlap_signals(
            submissions,
            normalized_submissions,
            shadow,
            normalized_shadow,
            "shadow",
            fuzzy_threshold,
        )
    )
    signals.extend(_canary_signals(normalized_submissions, normalized_canaries))
    signals.extend(
        _public_vs_shadow_signals(
            normalized_submissions,
            normalized_public,
            normalized_shadow,
            fuzzy_threshold,
            shadow_margin,
        )
    )
    signals.sort(
        key=lambda signal: (
            OVERLAP_FORENSICS_CHECKS.index(signal.check),
            signal.submission_ref,
            _SCOPE_ORDER[signal.reference_scope],
            signal.reference_ref,
        )
    )

    reference_count = len(public) + len(shadow)
    pairwise_overlap_comparisons = len(submissions) * reference_count
    coverage = {
        "canary_items": len(canary_items),
        "checks": list(OVERLAP_FORENSICS_CHECKS),
        "comparison_counts": {
            CHECK_CANARY: len(submissions) * len(canary_items),
            CHECK_EXACT: pairwise_overlap_comparisons,
            CHECK_FUZZY: pairwise_overlap_comparisons,
            CHECK_NORMALIZED: pairwise_overlap_comparisons,
            CHECK_PUBLIC_VS_SHADOW: pairwise_overlap_comparisons,
        },
        "public_items": len(public),
        "shadow_items": len(shadow),
        "submission_items": len(submissions),
    }
    signal_counts = {
        check: sum(signal.check == check for signal in signals)
        for check in OVERLAP_FORENSICS_CHECKS
    }
    return OverlapForensicsReport(
        submission_manifest_digest=manifest_digest,
        holdout_commitment_digest=commitment_digest,
        fuzzy_threshold_basis_points=fuzzy_threshold,
        shadow_margin_basis_points=shadow_margin,
        coverage=coverage,
        signal_counts=signal_counts,
        signals=tuple(signals),
        interpretation=(
            INTERPRETATION_SIGNALS if signals else INTERPRETATION_NO_SIGNALS
        ),
    )


__all__ = [
    "CHECK_CANARY",
    "CHECK_EXACT",
    "CHECK_FUZZY",
    "CHECK_NORMALIZED",
    "CHECK_PUBLIC_VS_SHADOW",
    "DEFAULT_FUZZY_THRESHOLD_BASIS_POINTS",
    "DEFAULT_SHADOW_MARGIN_BASIS_POINTS",
    "INTERPRETATION_NO_SIGNALS",
    "INTERPRETATION_SIGNALS",
    "OVERLAP_FORENSICS_CHECKS",
    "OVERLAP_FORENSICS_POLICY_VERSION",
    "OVERLAP_FORENSICS_SCHEMA_VERSION",
    "OverlapForensicsError",
    "OverlapForensicsReport",
    "OverlapSignal",
    "scan_overlap_forensics",
]
