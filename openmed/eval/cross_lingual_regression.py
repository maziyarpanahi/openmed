"""Last-green multilingual transfer qualification from paired aggregate outcomes."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from openmed.core.clinical_language import normalize_clinical_language

_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_LABEL = re.compile(r"[A-Z][A-Z0-9_]{0,63}\Z")


@dataclass(frozen=True)
class PairedTransferCell:
    """Same-cohort outcomes for a candidate and per-language last-green baseline."""

    language: str
    label: str
    cohort_digest: str
    trials: int
    baseline_successes: int
    candidate_successes: int
    baseline_only: int
    candidate_only: int
    critical: bool = False

    def __post_init__(self) -> None:
        language, _locale = normalize_clinical_language(self.language)
        if (
            language == "auto"
            or not isinstance(self.label, str)
            or not _LABEL.fullmatch(self.label)
        ):
            raise ValueError("invalid controlled transfer cell category")
        if not isinstance(self.cohort_digest, str) or not _DIGEST.fullmatch(
            self.cohort_digest
        ):
            raise ValueError("transfer cohort reference must be a SHA-256 digest")
        values = (
            self.trials,
            self.baseline_successes,
            self.candidate_successes,
            self.baseline_only,
            self.candidate_only,
        )
        if any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 0 <= value <= 1_000_000
            for value in values
        ):
            raise ValueError("invalid paired transfer counts")
        common = self.baseline_successes - self.baseline_only
        if (
            common < 0
            or common != self.candidate_successes - self.candidate_only
            or common + self.baseline_only + self.candidate_only > self.trials
        ):
            raise ValueError("paired transfer counts are inconsistent")
        if not isinstance(self.critical, bool):
            raise ValueError("critical designation must be explicit boolean")
        object.__setattr__(self, "language", language)


def cross_lingual_regression_gate(
    cells: Sequence[PairedTransferCell],
    *,
    expected_baseline_cells: Mapping[tuple[str, str], tuple[str, bool]],
    baseline_checkpoint_digest: str,
    candidate_checkpoint_digest: str,
    alpha: float = 0.05,
    critical_drop_tolerance: float = 0.0,
    minimum_trials: int = 20,
) -> dict[str, object]:
    """Fail closed for missing baselines and any excessive critical point decline.

    Separate statistical classifications use exact paired-discordance binomial
    tails with Bonferroni correction across supplied cells. Critical safety policy
    remains independent of significance: small declines cannot hide in an
    inconclusive aggregate or be offset by target-language improvements.
    """
    for digest in (baseline_checkpoint_digest, candidate_checkpoint_digest):
        if not isinstance(digest, str) or not _DIGEST.fullmatch(digest):
            raise ValueError("checkpoint references must be SHA-256 digests")
    for value in (alpha, critical_drop_tolerance):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError("transfer policy must be finite")
    if not 0 < alpha < 1 or not 0 <= critical_drop_tolerance <= 1:
        raise ValueError("invalid transfer policy range")
    if (
        isinstance(minimum_trials, bool)
        or not isinstance(minimum_trials, int)
        or not 1 <= minimum_trials <= 1_000_000
    ):
        raise ValueError("invalid minimum transfer trial count")
    if (
        not isinstance(cells, Sequence)
        or len(cells) > 10_000
        or not all(isinstance(cell, PairedTransferCell) for cell in cells)
    ):
        raise ValueError("provide bounded validated transfer cells")
    if (
        not isinstance(expected_baseline_cells, Mapping)
        or not 1 <= len(expected_baseline_cells) <= 10_000
    ):
        raise ValueError("baseline policy must contain 1 to 10000 cells")
    expected = {}
    for key, policy in expected_baseline_cells.items():
        if (
            not isinstance(policy, tuple)
            or len(policy) != 2
            or not isinstance(policy[1], bool)
        ):
            raise ValueError(
                "baseline policy must bind cohort and critical designation"
            )
        cohort, critical = policy
        if not isinstance(key, tuple) or len(key) != 2:
            raise ValueError("expected baseline key must contain language and label")
        language, _locale = normalize_clinical_language(key[0])
        if (
            language == "auto"
            or not isinstance(key[1], str)
            or not _LABEL.fullmatch(key[1])
        ):
            raise ValueError("invalid baseline category")
        if not isinstance(cohort, str) or not _DIGEST.fullmatch(cohort):
            raise ValueError("baseline cohort must be a SHA-256 digest")
        normalized = language, key[1]
        if normalized in expected:
            raise ValueError("duplicate normalized baseline cell")
        expected[normalized] = cohort, critical
    observed = {(cell.language, cell.label): cell for cell in cells}
    if len(observed) != len(cells):
        raise ValueError("duplicate transfer cell")
    rows = []
    failures = []
    missing = sorted(set(expected) - set(observed))
    extra = sorted(set(observed) - set(expected))
    significance = alpha / max(1, len(cells))
    for key, cell in sorted(observed.items()):
        aligned = (
            key in expected
            and cell.cohort_digest == expected[key][0]
            and cell.critical == expected[key][1]
        )
        delta = (
            (cell.candidate_successes - cell.baseline_successes) / cell.trials
            if cell.trials
            else None
        )
        discordant = cell.baseline_only + cell.candidate_only
        minority = min(cell.baseline_only, cell.candidate_only)
        # Exact paired binomial two-sided test, calculated in log space to avoid
        # huge integer powers and underflow in individual probability terms.
        if minority == discordant // 2:
            p_value = 1.0
        elif discordant > 4096:
            p_value = None
        elif discordant:
            tail = math.fsum(
                math.exp(
                    math.lgamma(discordant + 1)
                    - math.lgamma(k + 1)
                    - math.lgamma(discordant - k + 1)
                    - discordant * math.log(2)
                )
                for k in range(minority + 1)
            )
            p_value = min(1.0, 2 * tail)
        else:
            p_value = 1.0
        classification = "inconclusive"
        if (
            aligned
            and cell.trials >= minimum_trials
            and delta is not None
            and p_value is not None
            and p_value <= significance
        ):
            classification = (
                "positive" if delta > 0 else "negative" if delta < 0 else "inconclusive"
            )
        critical_failure = (
            aligned
            and expected[key][1]
            and delta is not None
            and delta < -critical_drop_tolerance
        )
        if critical_failure:
            failures.append(key)
        if not aligned or cell.trials < minimum_trials:
            failures.append(key)
        rows.append(
            {
                "language": key[0],
                "label": key[1],
                "cohort_digest": cell.cohort_digest,
                "trials": cell.trials,
                "recall_delta": delta,
                "paired_p_value": p_value,
                "statistical_evidence": "computation_budget_exceeded"
                if p_value is None
                else "exact_paired_binomial",
                "classification": classification,
                "critical": cell.critical,
                "critical_regression": critical_failure,
                "baseline_aligned": aligned,
            }
        )
    allowed = bool(expected) and not missing and not extra and not failures
    return {
        "schema_version": 1,
        "baseline_checkpoint_digest": baseline_checkpoint_digest,
        "candidate_checkpoint_digest": candidate_checkpoint_digest,
        "release_allowed": allowed,
        "alpha": alpha,
        "familywise_cell_alpha": significance,
        "critical_drop_tolerance": critical_drop_tolerance,
        "missing_cells": [list(key) for key in missing],
        "unexpected_cells": [list(key) for key in extra],
        "failed_cells": [list(key) for key in sorted(set(failures))],
        "cells": rows,
    }


def require_cross_lingual_release(*args, **kwargs) -> dict[str, object]:
    """Evaluate and refuse promotion when multilingual qualification fails."""
    report = cross_lingual_regression_gate(*args, **kwargs)
    if not report["release_allowed"]:
        raise ValueError("cross-lingual release qualification failed")
    return report
