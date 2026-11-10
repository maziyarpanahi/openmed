"""Deterministic ancestry-isolated partitions for explicitly synthetic records."""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_CODE = re.compile(r"[A-Za-z][A-Za-z0-9_-]{0,63}\Z")
MAX_SEARCH_STATES = 4096
MAX_SEARCH_GROUPS = 128
MAX_STRATA_REQUIREMENTS = 128


@dataclass(frozen=True)
class SyntheticSplitRecord:
    """Opaque synthetic artifact, trajectory, lineage and declared strata.

    ``lineage_digest`` identifies a shared derivation family, not merely a
    generator version. Every record is explicitly synthetic. No source text,
    patient identifier, or real-record matching is accepted by this planner.
    """

    artifact_digest: str
    trajectory_digest: str
    lineage_digest: str
    parent_digests: tuple[str, ...] = ()
    strata: tuple[str, ...] = ()
    synthetic: bool = True

    def __post_init__(self) -> None:
        if (
            not isinstance(self.parent_digests, Sequence)
            or not isinstance(self.strata, Sequence)
            or isinstance(self.parent_digests, (str, bytes))
            or isinstance(self.strata, (str, bytes))
        ):
            raise ValueError("ancestry and strata must be sequences of references")
        object.__setattr__(self, "parent_digests", tuple(self.parent_digests))
        object.__setattr__(self, "strata", tuple(self.strata))
        if len(self.parent_digests) > 10_000 or len(self.strata) > 256:
            raise ValueError("synthetic ancestry or strata limit exceeded")
        if self.synthetic is not True:
            raise ValueError("split records must be explicitly synthetic")
        for value in (
            self.artifact_digest,
            self.trajectory_digest,
            self.lineage_digest,
            *self.parent_digests,
        ):
            if not isinstance(value, str) or not _DIGEST.fullmatch(value):
                raise ValueError("synthetic references must be SHA-256 digests")
        if len(set(self.parent_digests)) != len(self.parent_digests):
            raise ValueError("duplicate ancestry reference")
        if len(set(self.strata)) != len(self.strata):
            raise ValueError("duplicate stratum")
        if any(
            not isinstance(code, str) or not _CODE.fullmatch(code)
            for code in self.strata
        ):
            raise ValueError("invalid synthetic stratum code")


@dataclass(frozen=True)
class SyntheticSplitPlan:
    """Partition assignments and unmet constraints without source payloads."""

    assignments: Mapping[str, str]
    group_count: int
    unmet_strata: tuple[tuple[str, str], ...]
    search_status: str = "complete"

    def __post_init__(self) -> None:
        if not isinstance(self.assignments, Mapping) or len(self.assignments) > 100_000:
            raise ValueError("split assignments must be a bounded mapping")
        assignments = dict(self.assignments)
        for digest, partition in assignments.items():
            if not isinstance(digest, str) or not _DIGEST.fullmatch(digest):
                raise ValueError("split artifact references must be SHA-256 digests")
            if not isinstance(partition, str) or not _CODE.fullmatch(partition):
                raise ValueError("invalid split partition code")
        if (
            isinstance(self.group_count, bool)
            or not isinstance(self.group_count, int)
            or self.group_count < 0
            or self.group_count > len(assignments)
            or (bool(assignments) != bool(self.group_count))
        ):
            raise ValueError("split group count is inconsistent with assignments")
        if (
            isinstance(self.unmet_strata, (str, bytes))
            or len(self.unmet_strata) > 65_536
        ):
            raise ValueError("unmet strata must be a bounded sequence")
        unmet = []
        for pair in self.unmet_strata:
            if (
                not isinstance(pair, Sequence)
                or isinstance(pair, (str, bytes))
                or len(pair) != 2
            ):
                raise ValueError(
                    "unmet strata must contain partition and stratum pairs"
                )
            if any(
                not isinstance(code, str) or not _CODE.fullmatch(code) for code in pair
            ):
                raise ValueError("invalid unmet stratum category code")
            unmet.append(tuple(pair))
        if len(set(unmet)) != len(unmet):
            raise ValueError("duplicate unmet stratum")
        if self.search_status not in {"complete", "unsatisfied", "budget_exhausted"}:
            raise ValueError("invalid split search status")
        if bool(unmet) != (self.search_status != "complete"):
            raise ValueError("search status inconsistent with unmet strata")
        if len(set(assignments.values())) > self.group_count:
            raise ValueError("partition count exceeds group count")
        object.__setattr__(self, "assignments", MappingProxyType(assignments))
        object.__setattr__(self, "unmet_strata", tuple(sorted(unmet)))

    def to_manifest(self) -> dict[str, object]:
        """Export only a complete plan; never relax ancestry for strata coverage."""
        if self.unmet_strata:
            raise ValueError("synthetic split strata constraints are unsatisfied")
        return {
            "schema_version": 1,
            "synthetic": True,
            "group_count": self.group_count,
            "assignments": dict(sorted(self.assignments.items())),
        }


def plan_synthetic_splits(
    records: Sequence[SyntheticSplitRecord],
    *,
    fractions: Mapping[str, float],
    seed: int,
    required_strata: Mapping[str, Sequence[str]] | None = None,
) -> SyntheticSplitPlan:
    """Plan stable partitions keeping shared trajectory and lineage together.

    Parent references must resolve in the input and may not form cycles. Group
    allocation uses seed-bound SHA-256 ordering. Required strata use scarcity-first bounded backtracking before ratio-based
    allocation; ratios are targets, not exact sizes.
    Unsatisfied strata are reported and prevent manifest export. This planner
    cannot prove clinical validity, privacy, or absence of external contamination.
    """
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if not fractions:
        raise ValueError("at least one partition is required")
    for code, fraction in fractions.items():
        if not isinstance(code, str) or not _CODE.fullmatch(code):
            raise ValueError("invalid partition code")
        if (
            isinstance(fraction, bool)
            or not isinstance(fraction, (int, float))
            or not math.isfinite(fraction)
            or fraction <= 0
        ):
            raise ValueError("partition fractions must be finite and positive")
    if not math.isclose(math.fsum(fractions.values()), 1.0, rel_tol=0, abs_tol=1e-12):
        raise ValueError("partition fractions must sum to one")
    if len(records) > 100_000:
        raise ValueError("synthetic split record limit exceeded")
    rows = list(records)
    if not all(isinstance(row, SyntheticSplitRecord) for row in rows):
        raise ValueError("expected validated synthetic split records")
    by_id = {row.artifact_digest: row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError("duplicate artifact digest")
    requirements = required_strata or {}
    needed: set[tuple[str, str]] = set()
    for split, strata in requirements.items():
        if split not in fractions:
            raise ValueError("stratum requirement references unknown partition")
        for stratum in strata:
            if not isinstance(stratum, str) or not _CODE.fullmatch(stratum):
                raise ValueError("invalid required stratum code")
            needed.add((split, stratum))
    # An explicit iterative topological traversal avoids recursion-depth limits.
    children: dict[str, list[str]] = {key: [] for key in by_id}
    indegree: dict[str, int] = {}
    for key, row in by_id.items():
        indegree[key] = len(row.parent_digests)
        for parent in row.parent_digests:
            if parent not in by_id:
                raise ValueError("unresolved ancestry reference")
            children[parent].append(key)
    ready = [key for key, count in indegree.items() if count == 0]
    visited = 0
    while ready:
        key = ready.pop()
        visited += 1
        for child in children[key]:
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(child)
    if visited != len(rows):
        raise ValueError("cyclic synthetic ancestry")
    representative = {key: key for key in by_id}

    def find(key: str) -> str:
        while representative[key] != key:
            representative[key] = representative[representative[key]]
            key = representative[key]
        return key

    def union(left: str, right: str) -> None:
        left, right = find(left), find(right)
        if left != right:
            representative[max(left, right)] = min(left, right)

    anchors: dict[tuple[str, str], str] = {}
    for row in sorted(rows, key=lambda item: item.artifact_digest):
        for field, digest in (
            ("trajectory", row.trajectory_digest),
            ("lineage", row.lineage_digest),
        ):
            key = field, digest
            if key in anchors:
                union(row.artifact_digest, anchors[key])
            else:
                anchors[key] = row.artifact_digest
        for parent in row.parent_digests:
            union(row.artifact_digest, parent)
    groups: dict[str, list[SyntheticSplitRecord]] = {}
    for row in rows:
        groups.setdefault(find(row.artifact_digest), []).append(row)

    def fingerprint(key: str) -> str:
        members = ",".join(sorted(row.artifact_digest for row in groups[key]))
        return hashlib.sha256(f"{seed}:{members}".encode("ascii")).hexdigest()

    pending = sorted(groups, key=fingerprint)
    group_strata = {
        key: {tag for row in group for tag in row.strata}
        for key, group in groups.items()
    }
    assigned: dict[str, str] = {}
    partition_counts: Counter[str] = Counter()
    if len(needed) > MAX_STRATA_REQUIREMENTS:
        raise ValueError("synthetic stratum requirement limit exceeded")
    eligible = {}
    truncated = False
    for pair in needed:
        keys = [key for key in pending if pair[1] in group_strata[key]]
        truncated = truncated or len(keys) > MAX_SEARCH_GROUPS
        eligible[pair] = keys[:MAX_SEARCH_GROUPS]
    states = 0
    exhausted = False

    def search(unmet: set[tuple[str, str]]) -> bool:
        nonlocal states, exhausted
        states += 1
        if states > MAX_SEARCH_STATES:
            exhausted = True
            return False
        if not unmet:
            return True
        pair = min(
            unmet,
            key=lambda item: (sum(key not in assigned for key in eligible[item]), item),
        )
        split, _tag = pair
        for key in eligible[pair]:
            if key in assigned:
                continue
            assigned[key] = split
            remaining = unmet - {(split, tag) for tag in group_strata[key]}
            if search(remaining):
                return True
            del assigned[key]
            if exhausted:
                return False
        return False

    found = not needed or search(needed)
    exhausted = exhausted or (not found and truncated)
    search_status = (
        "complete" if found else "budget_exhausted" if exhausted else "unsatisfied"
    )
    if found:
        needed.clear()
        for key, split in assigned.items():
            partition_counts[split] += len(groups[key])
    for key in pending:
        if key in assigned:
            continue
        split = min(
            fractions,
            key=lambda name: (partition_counts[name] / fractions[name], name),
        )
        assigned[key] = split
        partition_counts[split] += len(groups[key])
    for key, split in assigned.items():
        needed.difference_update((split, tag) for tag in group_strata[key])
    if not needed:
        search_status = "complete"
    assignments = {
        row.artifact_digest: assigned[key]
        for key, group in groups.items()
        for row in group
    }
    return SyntheticSplitPlan(
        dict(sorted(assignments.items())),
        len(groups),
        tuple(sorted(needed)),
        search_status,
    )
