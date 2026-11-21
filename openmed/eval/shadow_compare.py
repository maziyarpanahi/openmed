"""Value-free agreement counts for active and shadow detections on one document."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable
from functools import lru_cache
from importlib.resources import files

from jsonschema import Draft202012Validator

from openmed.core.audit import stable_hash
from openmed.core.labels import DIRECT_IDENTIFIER, policy_label_for
from openmed.core.schemas.span import OpenMedSpan


def _overlap_count(
    active: set[tuple[int, int]], candidate: set[tuple[int, int]]
) -> int:
    """Match the earliest finishing interval to its earliest finishing partner."""
    candidates = sorted(candidate, key=lambda interval: interval[::-1])
    matched = 0
    for start, end in sorted(active, key=lambda interval: interval[::-1]):
        for index, (other_start, other_end) in enumerate(candidates):
            if start < other_end and other_start < end:
                candidates.pop(index)
                matched += 1
                break
    return matched


def compare_shadow_spans(
    active: Iterable[OpenMedSpan], candidate: Iterable[OpenMedSpan]
) -> dict:
    """Compare span sets over the same document using half-open offsets.

    Duplicate spans count once. Exact same-label matches take precedence over
    one-to-one partial overlaps. A direct identifier is missed when it has no
    matched candidate. Policy totals aggregate the canonical-label comparisons.

    Args:
        active: Spans emitted by the active detector.
        candidate: Spans emitted by the shadow detector.

    Returns:
        Aggregate counts keyed by canonical and policy labels.

    Raises:
        ValueError: The inputs mix documents or contain an empty span.
    """
    documents = set()
    groups = []
    for spans in (active, candidate):
        by_label = defaultdict(set)
        for span in spans:
            documents.add(span.doc_id)
            if len(documents) > 1 or span.end <= span.start:
                raise ValueError("expected nonempty spans from one document")
            by_label[span.canonical_label].add((span.start, span.end))
        groups.append(by_label)

    active_groups, candidate_groups = groups
    canonical = {}
    policies = {}
    missed = 0
    for label in sorted(active_groups.keys() | candidate_groups.keys()):
        left, right = active_groups[label], candidate_groups[label]
        exact = left & right
        overlap = _overlap_count(left - exact, right - exact)
        counts = {
            "exact": len(exact),
            "overlap": overlap,
            "active_only": len(left) - len(exact) - overlap,
            "candidate_only": len(right) - len(exact) - overlap,
        }
        canonical[label] = counts
        policy = policy_label_for(label)
        totals = policies.setdefault(policy, dict.fromkeys(counts, 0))
        for key, count in counts.items():
            totals[key] += count
        if policy == DIRECT_IDENTIFIER:
            missed += counts["active_only"]
    return {
        "by_canonical_label": canonical,
        "by_policy_label": policies,
        "missed_direct_identifiers": missed,
    }


@lru_cache(maxsize=1)
def _aggregate_validator():
    source = files("openmed.core.schemas.json").joinpath(
        "shadow_comparison.schema.json"
    )
    return Draft202012Validator(json.loads(source.read_bytes()))


def shadow_comparison_digest(record: dict) -> str:
    """Validate the comparator wire record and bind its consistent totals."""
    if not _aggregate_validator().is_valid(record):
        raise ValueError("invalid_shadow_comparison")
    policies = {}
    for label, counts in record["by_canonical_label"].items():
        totals = policies.setdefault(policy_label_for(label), dict.fromkeys(counts, 0))
        for metric, count in counts.items():
            if type(count) is not int:
                raise ValueError("invalid_shadow_comparison")
            totals[metric] += count
    missed = policies.get(DIRECT_IDENTIFIER, {}).get("active_only", 0)
    if (
        policies != record["by_policy_label"]
        or type(record["missed_direct_identifiers"]) is not int
        or any(
            type(n) is not int
            for counts in record["by_policy_label"].values()
            for n in counts.values()
        )
        or missed != record["missed_direct_identifiers"]
    ):
        raise ValueError("invalid_shadow_comparison")
    return stable_hash(record)
