"""Bounded, aggregate-only membership probes for structured releases.

The probe is intentionally modest: it asks whether an attacker with a bounded
candidate population can match a released row to exactly one candidate using
only caller-declared quasi-identifiers. It is a self-test, not a calibrated
membership-inference guarantee. Keys and row values remain local and only
aggregate counts and digests are returned.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, time
from decimal import Decimal
from typing import Any

from openmed.core.audit import stable_hash

__all__ = [
    "MembershipSelfTestError",
    "MembershipSelfTestResult",
    "bounded_membership_inference_self_test",
    "membership_inference_self_test",
    "run_membership_inference_self_test",
]


class MembershipSelfTestError(ValueError):
    """Raised when a structured membership self-test cannot be configured."""


@dataclass(frozen=True)
class MembershipSelfTestResult:
    """Aggregate evidence from one bounded exact-QI membership self-test."""

    schema_version: int
    release_count: int
    candidate_count: int
    candidate_limit: int
    candidate_truncated: bool
    unique_match_count: int
    ambiguous_match_count: int
    no_match_count: int
    membership_inference_rate: float
    attacker_advantage: float
    max_inference_rate: float | None
    release_key_digest: str
    candidate_key_digest: str

    @property
    def meets_policy(self) -> bool:
        """Return whether the configured maximum inference rate is met."""

        return self.max_inference_rate is None or (
            self.membership_inference_rate <= self.max_inference_rate
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a report containing no row identifiers or cell values."""

        return {
            "schema_version": self.schema_version,
            "scope": "structured_quasi_identifier_membership_self_test",
            "release_count": self.release_count,
            "candidate_count": self.candidate_count,
            "candidate_limit": self.candidate_limit,
            "candidate_truncated": self.candidate_truncated,
            "unique_match_count": self.unique_match_count,
            "ambiguous_match_count": self.ambiguous_match_count,
            "no_match_count": self.no_match_count,
            "membership_inference_rate": self.membership_inference_rate,
            "attacker_advantage": self.attacker_advantage,
            "max_inference_rate": self.max_inference_rate,
            "release_key_digest": self.release_key_digest,
            "candidate_key_digest": self.candidate_key_digest,
            "meets_policy": self.meets_policy,
            "row_level_anonymization_claim": False,
        }


def membership_inference_self_test(
    released_records: Any,
    candidate_records: Any,
    *,
    quasi_identifiers: Sequence[str],
    max_candidates: int = 10_000,
    max_inference_rate: float | None = None,
) -> MembershipSelfTestResult:
    """Run a bounded exact-match membership self-test.

    ``candidate_records`` represents the locally supplied attacker candidate
    population. A released row is counted as a unique match when its declared
    quasi-identifier key occurs exactly once in the bounded candidate set. No
    source-side identifier is used, and no matched key is returned.

    Args:
        released_records: Rows in the intended or materialized release.
        candidate_records: Caller-supplied candidate population used only for
            this offline probe.
        quasi_identifiers: Explicit columns used to form the attack key.
        max_candidates: Hard cap on candidate rows processed. Extra rows are
            deterministically ignored and reported as truncated.
        max_inference_rate: Optional policy ceiling in ``[0, 1]``.

    Raises:
        MembershipSelfTestError: If the input or explicit configuration is
            invalid.
    """

    qis = _validated_columns(quasi_identifiers)
    limit = _positive_int(max_candidates, field_name="max_candidates")
    ceiling = _optional_rate(max_inference_rate)
    released = _materialize_rows(released_records, label="release")
    candidates_all = _materialize_rows(candidate_records, label="candidate")
    candidates = candidates_all[:limit]

    candidate_keys = [_key(row, qis) for row in candidates]
    release_keys = [_key(row, qis) for row in released]
    counts = Counter(candidate_keys)

    unique_matches = sum(1 for key in release_keys if counts.get(key) == 1)
    ambiguous_matches = sum(1 for key in release_keys if counts.get(key, 0) > 1)
    no_matches = len(release_keys) - unique_matches - ambiguous_matches
    rate = _rate(unique_matches, len(release_keys))

    return MembershipSelfTestResult(
        schema_version=1,
        release_count=len(release_keys),
        candidate_count=len(candidates),
        candidate_limit=limit,
        candidate_truncated=len(candidates_all) > len(candidates),
        unique_match_count=unique_matches,
        ambiguous_match_count=ambiguous_matches,
        no_match_count=no_matches,
        membership_inference_rate=rate,
        attacker_advantage=rate,
        max_inference_rate=ceiling,
        release_key_digest=_key_digest(release_keys, kind="release"),
        candidate_key_digest=_key_digest(candidate_keys, kind="candidate"),
    )


def bounded_membership_inference_self_test(
    released_records: Any,
    candidate_records: Any,
    *,
    quasi_identifiers: Sequence[str],
    max_candidates: int = 10_000,
    max_inference_rate: float | None = None,
) -> MembershipSelfTestResult:
    """Alias for :func:`membership_inference_self_test`."""

    return membership_inference_self_test(
        released_records,
        candidate_records,
        quasi_identifiers=quasi_identifiers,
        max_candidates=max_candidates,
        max_inference_rate=max_inference_rate,
    )


def run_membership_inference_self_test(
    released_records: Any,
    candidate_records: Any,
    *,
    quasi_identifiers: Sequence[str],
    max_candidates: int = 10_000,
    max_inference_rate: float | None = None,
) -> MembershipSelfTestResult:
    """Compatibility alias for the bounded structured self-test."""

    return membership_inference_self_test(
        released_records,
        candidate_records,
        quasi_identifiers=quasi_identifiers,
        max_candidates=max_candidates,
        max_inference_rate=max_inference_rate,
    )


def _validated_columns(value: Sequence[str]) -> tuple[str, ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise MembershipSelfTestError(
            "quasi_identifiers must be a non-empty sequence of column names"
        )
    columns: list[str] = []
    for column in value:
        if not isinstance(column, str) or not column or column in columns:
            if not isinstance(column, str) or not column:
                raise MembershipSelfTestError(
                    "quasi_identifiers must contain non-empty string names"
                )
            continue
        columns.append(column)
    if not columns:
        raise MembershipSelfTestError("quasi_identifiers must not be empty")
    return tuple(columns)


def _materialize_rows(data: Any, *, label: str) -> list[dict[str, Any]]:
    try:
        to_dicts = getattr(data, "to_dicts", None)
        if callable(to_dicts):
            data = to_dicts()
        else:
            to_dict = getattr(data, "to_dict", None)
            if callable(to_dict) and not isinstance(data, Mapping):
                data = to_dict("records")
        if isinstance(data, Mapping):
            rows: Any = [data]
        elif isinstance(data, Sequence) and not isinstance(
            data,
            (str, bytes, bytearray),
        ):
            rows = data
        else:
            raise TypeError
        if not all(isinstance(row, Mapping) for row in rows):
            raise TypeError
        materialized = [dict(row) for row in rows]
    except (AttributeError, TypeError, ValueError):
        raise MembershipSelfTestError(f"invalid {label} records") from None
    for row in materialized:
        if any(not isinstance(field, str) for field in row):
            raise MembershipSelfTestError(f"invalid {label} schema")
    return materialized


def _key(row: Mapping[str, Any], columns: Sequence[str]) -> str:
    return stable_hash(
        {
            "artifact": "openmed-structured-membership-key",
            "columns": list(columns),
            "values": [_canonical_value(row.get(column)) for column in columns],
        }
    )


def _key_digest(keys: Sequence[str], *, kind: str) -> str:
    return stable_hash(
        {
            "artifact": "openmed-structured-membership-key-set",
            "kind": kind,
            "keys": list(keys),
        }
    )


def _canonical_value(value: Any) -> Any:
    if value is None:
        return {"type": "null"}
    if type(value) is bool:
        return {"type": "bool", "value": value}
    if type(value) is int:
        return {"type": "int", "value": value}
    if type(value) is float:
        if not math.isfinite(value):
            raise MembershipSelfTestError("non-finite values are unsupported")
        return {"type": "float", "value": repr(value)}
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise MembershipSelfTestError("non-finite values are unsupported")
        return {"type": "decimal", "value": str(value)}
    if isinstance(value, datetime):
        return {"type": "datetime", "value": value.isoformat()}
    if isinstance(value, date):
        return {"type": "date", "value": value.isoformat()}
    if isinstance(value, time):
        return {"type": "time", "value": value.isoformat()}
    if isinstance(value, str):
        return {"type": "str", "value": value}
    if isinstance(value, Mapping):
        return {
            "type": "mapping",
            "value": {
                str(key): _canonical_value(item)
                for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            },
        }
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return {"type": "sequence", "value": [_canonical_value(item) for item in value]}
    raise MembershipSelfTestError("unsupported cell values are not allowed")


def _positive_int(value: Any, *, field_name: str) -> int:
    if type(value) is not int or value < 1:
        raise MembershipSelfTestError(f"{field_name} must be an integer >= 1")
    return value


def _optional_rate(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise MembershipSelfTestError("max_inference_rate must be between 0 and 1")
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        raise MembershipSelfTestError(
            "max_inference_rate must be between 0 and 1"
        ) from None
    if not math.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
        raise MembershipSelfTestError("max_inference_rate must be between 0 and 1")
    return parsed


def _rate(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0
