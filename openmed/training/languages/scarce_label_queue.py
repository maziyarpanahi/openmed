"""Deterministic metadata-only selection for budgeted native label review."""

from __future__ import annotations

import heapq
import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

from openmed.core.clinical_language import normalize_clinical_language

_CODE = re.compile(r"[A-Za-z][A-Za-z0-9_-]{0,63}\Z")
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
MAX_CANDIDATES = 10_000
MAX_BUDGET_CATEGORIES = 256
MAX_RANK_UPDATES = 100_000


@dataclass(frozen=True)
class ReviewCandidate:
    """Opaque candidate reference and bounded ranking evidence, without text.

    ``candidate_digest`` and ``diversity_digest`` identify local records and
    clusters without exporting source identifiers. Scores must be finite in
    [0, 1]. Scarcity is caller-declared; it is not inferred clinical importance.
    """

    candidate_digest: str
    language: str
    label: str
    scarcity: float
    disagreement: float
    diversity_digest: str
    error_signature_digest: str

    def __post_init__(self) -> None:
        for value in (
            self.candidate_digest,
            self.diversity_digest,
            self.error_signature_digest,
        ):
            if not isinstance(value, str) or not _DIGEST.fullmatch(value):
                raise ValueError("review references must be SHA-256 digests")
        language, _locale = normalize_clinical_language(self.language)
        if language == "auto":
            raise ValueError("review language must be explicit")
        object.__setattr__(self, "language", language)
        for value in (self.label,):
            if not isinstance(value, str) or not _CODE.fullmatch(value):
                raise ValueError("invalid review category code")
        for value in (self.scarcity, self.disagreement):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not 0 <= value <= 1
            ):
                raise ValueError("ranking evidence must be finite within [0, 1]")


@dataclass(frozen=True)
class ReviewSelection:
    """Selected opaque references and aggregate budget accounting."""

    selected: tuple[ReviewCandidate, ...]
    considered_count: int
    language_counts: Mapping[str, int]
    label_counts: Mapping[str, int]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.selected, Sequence)
            or len(self.selected) > MAX_CANDIDATES
            or not all(isinstance(row, ReviewCandidate) for row in self.selected)
            or isinstance(self.considered_count, bool)
            or not isinstance(self.considered_count, int)
            or not len(self.selected) <= self.considered_count <= MAX_CANDIDATES
        ):
            raise ValueError("invalid bounded review selection")
        if len({row.candidate_digest for row in self.selected}) != len(self.selected):
            raise ValueError("duplicate selected candidate")
        expected_languages = Counter(row.language for row in self.selected)
        expected_labels = Counter(row.label for row in self.selected)
        for actual, expected in (
            (self.language_counts, expected_languages),
            (self.label_counts, expected_labels),
        ):
            if (
                not isinstance(actual, Mapping)
                or len(actual) > MAX_BUDGET_CATEGORIES
                or any(
                    not isinstance(code, str)
                    or not _CODE.fullmatch(code)
                    or isinstance(count, bool)
                    or not isinstance(count, int)
                    or count < 0
                    for code, count in actual.items()
                )
                or dict(actual) != dict(expected)
            ):
                raise ValueError("review accounting must match selected candidates")
        object.__setattr__(self, "selected", tuple(self.selected))
        object.__setattr__(
            self, "language_counts", MappingProxyType(dict(self.language_counts))
        )
        object.__setattr__(
            self, "label_counts", MappingProxyType(dict(self.label_counts))
        )

    def to_dict(self) -> dict[str, object]:
        """Serialize references and controlled counts only, without examples."""
        return {
            "schema_version": 1,
            "selected_digests": [row.candidate_digest for row in self.selected],
            "considered_count": self.considered_count,
            "selected_count": len(self.selected),
            "language_counts": dict(sorted(self.language_counts.items())),
            "label_counts": dict(sorted(self.label_counts.items())),
        }


def select_native_review(
    candidates: Sequence[ReviewCandidate],
    *,
    language_budgets: Mapping[str, int],
    label_budgets: Mapping[str, int],
    total_budget: int,
) -> ReviewSelection:
    """Select scarce labels within simultaneous language, label and total caps.

    Scarcity ranks first, then model disagreement; equally ranked candidates
    prefer error signatures not yet selected, then diversity clusters and digest. Missing
    categories have zero budget. Selection order is independent of input order.
    A lazy heap avoids sorting the whole queue on each selection. The queue
    accepts at most 10,000 candidates and 256 categories per budget mapping. It
    is not an optimizer or automatic label approval.
    """
    _validate_budget(total_budget)
    if len(candidates) > MAX_CANDIDATES:
        raise ValueError("review candidate limit exceeded")
    if any(
        len(budgets) > MAX_BUDGET_CATEGORIES
        for budgets in (language_budgets, label_budgets)
    ):
        raise ValueError("review budget category limit exceeded")
    normalized_budgets = {}
    for language, count in language_budgets.items():
        code, _locale = normalize_clinical_language(language)
        if code == "auto" or code in normalized_budgets:
            raise ValueError("duplicate or implicit review language budget")
        normalized_budgets[code] = count
    language_budgets = normalized_budgets
    for budgets in (language_budgets, label_budgets):
        for code, count in budgets.items():
            if not isinstance(code, str) or not _CODE.fullmatch(code):
                raise ValueError("invalid budget category code")
            _validate_budget(count)
    rows = list(candidates)
    if not all(isinstance(row, ReviewCandidate) for row in rows):
        raise ValueError("expected validated review candidates")
    if len({row.candidate_digest for row in rows}) != len(rows):
        raise ValueError("duplicate review candidate digest")
    selected: list[ReviewCandidate] = []
    languages: Counter[str] = Counter()
    labels: Counter[str] = Counter()
    clusters: Counter[str] = Counter()
    signatures: Counter[str] = Counter()

    def rank(row: ReviewCandidate) -> tuple[float, float, int, int, str]:
        return (
            -row.scarcity,
            -row.disagreement,
            signatures[row.error_signature_digest],
            clusters[row.diversity_digest],
            row.candidate_digest,
        )

    pending = [(rank(row), row) for row in rows]
    heapq.heapify(pending)
    rank_updates = 0
    while pending and len(selected) < total_budget:
        previous_rank, row = heapq.heappop(pending)
        if languages[row.language] >= language_budgets.get(row.language, 0):
            continue
        if labels[row.label] >= label_budgets.get(row.label, 0):
            continue
        current_rank = rank(row)
        if current_rank != previous_rank:
            # Counts only increase: stale keys are conservative lower bounds.
            rank_updates += 1
            if rank_updates > MAX_RANK_UPDATES:
                raise ValueError("review ranking work limit exceeded")
            heapq.heappush(pending, (current_rank, row))
            continue
        selected.append(row)
        languages[row.language] += 1
        labels[row.label] += 1
        clusters[row.diversity_digest] += 1
        signatures[row.error_signature_digest] += 1
    return ReviewSelection(tuple(selected), len(candidates), languages, labels)


def _validate_budget(value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("review budgets must be non-negative integers")
