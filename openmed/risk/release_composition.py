"""Aggregate k-anonymity checks across successive, locally linkable releases."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any

from openmed.compliance.expert_review import CompositionEvidence
from openmed.core.audit import stable_hash
from openmed.risk.kanon import StreamingKanonState

_KEY = re.compile(r"(?:hmac-sha256:)?[0-9a-f]{64}")


@dataclass(frozen=True)
class ReleasedTable:
    """One release with its declared key and QI transformation.

    Supply the same keyed privacy-unit tokens across all tables. Use level zero
    for already generalized columns. For original columns supply the hierarchy
    and level used in the actual release.
    """

    rows: Sequence[Mapping[str, Any]] = field(repr=False)
    quasi_identifiers: Sequence[str]
    unit_column: str | None
    generalization: Mapping[str, int] | None = None
    hierarchies: Mapping[str, Sequence[Mapping[str, Any]]] | None = field(
        default=None, repr=False
    )


@dataclass(frozen=True)
class CompositionResult:
    """Aggregate risks and a receipt suitable for the expert-review evidence input."""

    evidence: CompositionEvidence
    reason: str | None
    target_k: int
    composed_k: int | None
    linked_units: int
    units_below_target: int | None
    single_release_k: tuple[int, ...]
    drop_pairs: tuple[tuple[int, int], ...]

    def to_dict(self) -> dict:
        """Serialize scores and one-based pairs of release indices."""
        return {
            "evidence": self.evidence.to_dict(),
            "reason": self.reason,
            "target_k": self.target_k,
            "composed_k": self.composed_k,
            "linked_units": self.linked_units,
            "units_below_target": self.units_below_target,
            "single_release_k": list(self.single_release_k),
            "drop_pairs": [list(pair) for pair in self.drop_pairs],
        }


def _classes(table: ReleasedTable) -> dict[str, tuple] | None:
    if not table.unit_column or table.unit_column in table.quasi_identifiers:
        return None
    classes = defaultdict(list)
    try:
        hierarchy = StreamingKanonState(
            table.quasi_identifiers,
            generalization=table.generalization,
            hierarchies=table.hierarchies,
            remove_direct_identifiers=False,
        )
        for row in table.rows:
            token = row.get(table.unit_column)
            if not isinstance(token, str) or not _KEY.fullmatch(token):
                return None
            if any(column not in row for column in hierarchy.quasi_identifiers):
                raise ValueError("missing QI")
            qis = {name: row[name] for name in hierarchy.quasi_identifiers}
            classes[token.removeprefix("hmac-sha256:")].append(
                hierarchy.class_key(hierarchy.generalize(qis))
            )
    except (KeyError, TypeError, ValueError):
        raise ValueError(
            "invalid released quasi-identifiers or generalization"
        ) from None
    return {unit: tuple(sorted(rows)) for unit, rows in classes.items()}


def _shrinks(groups: Counter, single_counts: Sequence[Counter]) -> bool:
    return any(
        count
        < min(census[cell] for census, cell in zip(single_counts, cells, strict=True))
        for cells, count in groups.items()
    )


def assess_release_composition(
    releases: Sequence[ReleasedTable], *, target_k: int
) -> CompositionResult:
    """Score the common cohort without exporting keys, cell values or class membership.

    Repeated rows form one order-independent multiset per unit per release.
    Single-release and pairwise k use the same common cohort as composed k.
    A drop pair shrinks at least one unit’s class below its size in both releases.
    Higher-order intersections can reduce k even when no pair alone reduces it.
    """
    if len(releases) < 2 or type(target_k) is not int or target_k < 1:
        raise ValueError("composition requires two releases and a positive k")
    tables = []
    reason = None
    for release in releases:
        classes = _classes(release)
        if classes is None:
            reason = "missing_linking_key"
            break
        tables.append(classes)
    units = (
        set.intersection(*(set(table) for table in tables))
        if tables and not reason
        else set()
    )
    individual = ()
    pairs = ()
    k = below = None
    if units:
        singles = [Counter(table[u] for u in units) for table in tables]
        individual = tuple(min(census.values()) for census in singles)
        groups = Counter(tuple(table[u] for table in tables) for u in units)
        k = min(groups.values())
        below = sum(n for n in groups.values() if n < target_k)
        pairs = tuple(
            (i + 1, j + 1)
            for i, j in combinations(range(len(tables)), 2)
            if _shrinks(
                Counter((tables[i][u], tables[j][u]) for u in units),
                (singles[i], singles[j]),
            )
        )
        increased = _shrinks(groups, singles)
        status = "increase_observed" if increased else "no_material_increase_observed"
    else:
        reason = reason or "no_common_units"
        status = "not_assessed"
    digest = stable_hash(
        {
            "kind": "release_composition",
            "release_count": len(releases),
            "reason": reason,
            "status": status,
            "target_k": target_k,
            "single_release_k": individual,
            "composed_k": k,
            "units_below_target": below,
            "linked_units": len(units),
            "drop_pairs": pairs,
        }
    )
    evidence = CompositionEvidence(
        release_count=len(releases),
        longitudinal_linkage_assessed=bool(units),
        prior_release_overlap_assessed=bool(units),
        risk_status=status,
        evidence_digest=digest,
    )
    return CompositionResult(
        evidence, reason, target_k, k, len(units), below, individual, pairs
    )
