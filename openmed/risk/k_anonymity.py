"""Targeted k-anonymity analysis and minimal row suppression.

This module builds on :func:`openmed.risk.kanon_report` to add the policy
operations needed by tabular anonymization workflows: identify rows that do
not meet a declared ``k`` and propose the smallest row-suppression set that
removes those violations. It intentionally does not search a generalization
lattice; callers that need full-domain generalization can use
:func:`openmed.risk.enforce_kanon`.

Reports contain row offsets, sizes, and hashes only. Raw quasi-identifier
values are used in-process to form equivalence classes but are not returned.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from openmed.core.audit import stable_hash

from .kanon import kanon_report

__all__ = [
    "EquivalenceClass",
    "KAnonymityEngine",
    "KAnonymityReport",
    "SuppressionProposal",
    "analyze_k_anonymity",
    "apply_suppression",
    "propose_suppression",
]


@dataclass(frozen=True)
class EquivalenceClass:
    """Privacy-safe description of one quasi-identifier equivalence class."""

    class_hash: str
    size: int
    row_indices: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic, JSON-serializable representation."""
        return {
            "class_hash": self.class_hash,
            "size": int(self.size),
            "row_indices": list(self.row_indices),
        }


@dataclass(frozen=True)
class KAnonymityReport:
    """Equivalence-class analysis against a target k-anonymity policy."""

    record_count: int
    quasi_identifiers: tuple[str, ...]
    target_k: int
    achieved_k: int
    smallest_class_size: int
    equivalence_classes: tuple[EquivalenceClass, ...]
    violating_rows: tuple[int, ...]
    meets_target: bool

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic, JSON-serializable representation."""
        return {
            "record_count": int(self.record_count),
            "quasi_identifiers": list(self.quasi_identifiers),
            "target_k": int(self.target_k),
            "achieved_k": int(self.achieved_k),
            "smallest_class_size": int(self.smallest_class_size),
            "class_count": len(self.equivalence_classes),
            "equivalence_classes": [
                equivalence_class.to_dict()
                for equivalence_class in self.equivalence_classes
            ],
            "violating_rows": list(self.violating_rows),
            "meets_target": bool(self.meets_target),
        }


@dataclass(frozen=True)
class SuppressionProposal:
    """Minimal whole-row suppression proposal for a target k.

    ``feasible`` is false when row suppression would remove every record. In
    that case the proposal remains useful as a diagnostic, but :meth:`apply`
    refuses to emit an empty table as if it satisfied k-anonymity.
    """

    target_k: int
    source_record_count: int
    row_indices: tuple[int, ...]
    retained_count: int
    achieved_k_after_suppression: int
    feasible: bool

    @property
    def suppressed_count(self) -> int:
        """Return the number of rows selected for suppression."""
        return len(self.row_indices)

    @property
    def suppression_rate(self) -> float:
        """Return the proposed fraction of source rows to suppress."""
        if not self.source_record_count:
            return 0.0
        return self.suppressed_count / self.source_record_count

    def apply(self, records: Any) -> list[dict[str, Any]]:
        """Apply this proposal while preserving all fields in retained rows."""
        if not self.feasible:
            raise ValueError(
                "Target k cannot be reached by row suppression without removing "
                "every record."
            )
        return apply_suppression(records, self)

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic, JSON-serializable representation."""
        return {
            "strategy": "minimal_whole_row_suppression",
            "target_k": int(self.target_k),
            "source_record_count": int(self.source_record_count),
            "row_indices": list(self.row_indices),
            "suppressed_count": self.suppressed_count,
            "retained_count": int(self.retained_count),
            "suppression_rate": float(self.suppression_rate),
            "achieved_k_after_suppression": int(self.achieved_k_after_suppression),
            "feasible": bool(self.feasible),
        }


@dataclass(frozen=True, init=False)
class KAnonymityEngine:
    """Analyze and suppress tabular rows against a declared target k.

    Args:
        quasi_identifiers: Column names that form each equivalence-class key.
        target_k: Required minimum equivalence-class size.
    """

    quasi_identifiers: tuple[str, ...]
    target_k: int

    def __init__(
        self,
        quasi_identifiers: Sequence[str],
        target_k: int = 2,
    ) -> None:
        object.__setattr__(
            self,
            "quasi_identifiers",
            _validate_quasi_identifiers(quasi_identifiers),
        )
        object.__setattr__(self, "target_k", _validate_target_k(target_k))

    def analyze(self, records: Any) -> KAnonymityReport:
        """Compute equivalence classes and rows that violate ``target_k``."""
        rows = _materialize_rows(records)
        _validate_columns(rows, self.quasi_identifiers)
        return _analyze_rows(rows, self.quasi_identifiers, self.target_k)

    def propose_suppression(self, records: Any) -> SuppressionProposal:
        """Propose the minimal whole-row suppression set for ``target_k``.

        Every row in an undersized equivalence class must be suppressed: if
        any row in that class remained, its class could only stay the same size
        or shrink, and would still violate ``target_k``. Removing precisely all
        such classes is therefore the unique minimal whole-row suppression set.
        """
        rows = _materialize_rows(records)
        _validate_columns(rows, self.quasi_identifiers)
        before = _analyze_rows(rows, self.quasi_identifiers, self.target_k)
        suppressed = frozenset(before.violating_rows)
        retained = [row for index, row in enumerate(rows) if index not in suppressed]
        after = _analyze_rows(retained, self.quasi_identifiers, self.target_k)
        return SuppressionProposal(
            target_k=self.target_k,
            source_record_count=len(rows),
            row_indices=before.violating_rows,
            retained_count=len(retained),
            achieved_k_after_suppression=after.achieved_k,
            feasible=after.meets_target,
        )

    def suppress(self, records: Any) -> list[dict[str, Any]]:
        """Propose and apply minimal suppression in one operation."""
        proposal = self.propose_suppression(records)
        return proposal.apply(records)


def analyze_k_anonymity(
    records: Any,
    quasi_identifiers: Sequence[str],
    *,
    target_k: int = 2,
) -> KAnonymityReport:
    """Analyze equivalence classes against ``target_k`` fully in-process."""
    return KAnonymityEngine(quasi_identifiers, target_k).analyze(records)


def propose_suppression(
    records: Any,
    quasi_identifiers: Sequence[str],
    *,
    target_k: int = 2,
) -> SuppressionProposal:
    """Return the minimal whole-row suppression proposal for ``target_k``."""
    return KAnonymityEngine(quasi_identifiers, target_k).propose_suppression(records)


def apply_suppression(
    records: Any,
    proposal: SuppressionProposal | Sequence[int],
) -> list[dict[str, Any]]:
    """Return copies of rows not selected by a suppression proposal.

    Args:
        records: A sequence of row mappings or a DataFrame-like object.
        proposal: A :class:`SuppressionProposal` or positional row indices.

    Returns:
        Materialized copies of the retained rows in their original order.
    """
    rows = _materialize_rows(records)
    if isinstance(proposal, SuppressionProposal):
        if len(rows) != proposal.source_record_count:
            raise ValueError(
                "Suppression proposal record count does not match the supplied table."
            )
        if not proposal.feasible:
            raise ValueError(
                "Target k cannot be reached by row suppression without removing "
                "every record."
            )
        row_indices = proposal.row_indices
    else:
        row_indices = tuple(proposal)

    invalid = sorted(
        {
            index
            for index in row_indices
            if type(index) is not int or index < 0 or index >= len(rows)
        },
        key=str,
    )
    if invalid:
        raise ValueError(f"Suppression row indices are out of range: {invalid!r}")

    suppressed = frozenset(row_indices)
    return [dict(row) for index, row in enumerate(rows) if index not in suppressed]


def _analyze_rows(
    rows: Sequence[Mapping[str, Any]],
    quasi_identifiers: tuple[str, ...],
    target_k: int,
) -> KAnonymityReport:
    measurement = kanon_report(rows, quasi_identifiers=quasi_identifiers)
    classes = tuple(
        sorted(
            (
                EquivalenceClass(
                    class_hash=stable_hash(equivalence_class["key"]),
                    size=int(equivalence_class["size"]),
                    row_indices=tuple(
                        sorted(int(index) for index in equivalence_class["members"])
                    ),
                )
                for equivalence_class in measurement["equivalence_classes"]
            ),
            key=lambda equivalence_class: (
                equivalence_class.row_indices[0],
                equivalence_class.class_hash,
            ),
        )
    )
    violating_rows = tuple(
        sorted(
            index
            for equivalence_class in classes
            if equivalence_class.size < target_k
            for index in equivalence_class.row_indices
        )
    )
    achieved_k = int(measurement["k"])
    return KAnonymityReport(
        record_count=len(rows),
        quasi_identifiers=quasi_identifiers,
        target_k=target_k,
        achieved_k=achieved_k,
        smallest_class_size=achieved_k,
        equivalence_classes=classes,
        violating_rows=violating_rows,
        meets_target=bool(rows) and not violating_rows,
    )


def _validate_quasi_identifiers(
    quasi_identifiers: Sequence[str],
) -> tuple[str, ...]:
    if isinstance(quasi_identifiers, (str, bytes, bytearray)):
        raise TypeError("quasi_identifiers must be a sequence of column names")
    normalized: list[str] = []
    for field in quasi_identifiers:
        if not isinstance(field, str) or not field.strip():
            raise ValueError("quasi_identifiers must contain non-empty column names")
        normalized.append(field.strip())
    if not normalized:
        raise ValueError("At least one quasi-identifier must be declared")
    return tuple(sorted(dict.fromkeys(normalized)))


def _validate_target_k(target_k: int) -> int:
    if type(target_k) is not int or target_k < 1:
        raise ValueError("target_k must be an integer >= 1")
    return target_k


def _validate_columns(
    rows: Sequence[Mapping[str, Any]],
    quasi_identifiers: Sequence[str],
) -> None:
    if not rows:
        return
    available = {str(field) for row in rows for field in row}
    missing = sorted(set(quasi_identifiers) - available)
    if missing:
        raise ValueError(f"Unknown quasi-identifier columns: {missing!r}")


def _materialize_rows(records: Any) -> list[dict[str, Any]]:
    if records is None:
        return []

    to_dicts = getattr(records, "to_dicts", None)
    if callable(to_dicts):
        records = to_dicts()
    else:
        to_dict = getattr(records, "to_dict", None)
        if callable(to_dict) and not isinstance(records, Mapping):
            try:
                records = to_dict("records")
            except TypeError as error:
                raise TypeError(
                    "DataFrame-like records must support to_dict('records')."
                ) from error

    if isinstance(records, Mapping):
        container = next(
            (
                records[name]
                for name in ("records", "rows", "items")
                if name in records and _is_row_sequence(records[name])
            ),
            None,
        )
        records = container if container is not None else [records]

    if not _is_row_sequence(records):
        raise TypeError("records must be a sequence of row mappings")
    return [dict(row) for row in records]


def _is_row_sequence(value: Any) -> bool:
    return (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes, bytearray))
        and all(isinstance(row, Mapping) for row in value)
    )
