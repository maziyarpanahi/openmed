"""Shadow-style membership-inference checks for structured releases.

The self-test uses only in-memory row values and emits aggregate attack metrics
plus synthetic record identifiers and scores.  It deliberately has no model
or network dependency so a release review can run locally before publication.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any

DEFAULT_MEMBERSHIP_ADVANTAGE_BUDGET = 0.05
DEFAULT_RISKIEST_RECORD_COUNT = 10

_BASELINE = 0.5
_AUTO_ID_FIELDS = ("record_id", "synthetic_id", "row_id", "id")
_TEXT_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
_MISSING = object()


@dataclass(frozen=True)
class _FieldSpec:
    name: str
    numeric: bool
    minimum: float | None
    maximum: float | None
    reference_counts: Mapping[str, int]
    reference_count: int


@dataclass(frozen=True)
class MembershipInferenceResult:
    """Privacy-safe result of a structured-table membership self-test.

    ``advantage`` is the larger of the attacker's AUC and balanced-accuracy
    improvements over the 0.5 chance baseline.  The configured budget is
    applied to that non-negative value.  Per-record payloads intentionally
    contain only a caller-provided or generated synthetic identifier and a
    score; source cell values never appear in serialized output.
    """

    auc: float
    accuracy: float
    advantage: float
    auc_advantage: float
    accuracy_advantage: float
    decision_threshold: float
    advantage_budget: float
    passed: bool
    member_count: int
    heldout_count: int
    riskiest_records: tuple[dict[str, Any], ...]
    record_scores: tuple[dict[str, Any], ...]
    mode: str = "shadow_distribution"

    @property
    def attacker_auc(self) -> float:
        """Return the best-orientation AUC available to the attacker."""

        return max(self.auc, 1.0 - self.auc)

    @property
    def attacker_advantage(self) -> float:
        """Return the advantage under the best score orientation."""

        return self.advantage

    @property
    def per_record(self) -> tuple[dict[str, Any], ...]:
        """Return safe scores for every candidate, sorted by descending risk."""

        return self.record_scores

    @property
    def risk_records(self) -> tuple[dict[str, Any], ...]:
        """Return the highest-scoring safe record references."""

        return self.riskiest_records

    @property
    def meets_budget(self) -> bool:
        """Return whether the measured attack advantage is within budget."""

        return self.passed

    def to_dict(self) -> dict[str, Any]:
        """Serialize aggregate metrics and synthetic record risk references."""

        return {
            "schema_version": 1,
            "auc": float(self.auc),
            "attacker_auc": float(self.attacker_auc),
            "accuracy": float(self.accuracy),
            "advantage": float(self.advantage),
            "auc_advantage": float(self.auc_advantage),
            "accuracy_advantage": float(self.accuracy_advantage),
            "decision_threshold": float(self.decision_threshold),
            "advantage_budget": float(self.advantage_budget),
            "passed": bool(self.passed),
            "member_count": int(self.member_count),
            "heldout_count": int(self.heldout_count),
            "riskiest_records": [dict(item) for item in self.riskiest_records],
            "mode": self.mode,
        }

    def to_metric(self) -> dict[str, Any]:
        """Return the safe metric payload used by benchmark integrations."""

        return self.to_dict()

    def to_json(self) -> str:
        """Return deterministic JSON for an audit or release-gate artifact."""

        return json.dumps(self.to_dict(), sort_keys=True)


MembershipInferenceReport = MembershipInferenceResult


def membership_inference_self_test(
    released_table: Any,
    member_records: Any | None = None,
    heldout_records: Any | None = None,
    *,
    record_id_field: str | None = None,
    advantage_budget: float = DEFAULT_MEMBERSHIP_ADVANTAGE_BUDGET,
    top_k: int = DEFAULT_RISKIEST_RECORD_COUNT,
) -> MembershipInferenceResult:
    """Run a deterministic membership-inference self-test for a table release.

    The two-argument form treats ``released_table`` as the positive shadow
    member set and ``member_records`` as held-out non-members.  It is the
    compact form for a release-versus-held-out check::

        membership_inference_self_test(released_rows, heldout_rows)

    The three-argument form compares retained source members and held-out
    source records by their similarity to the released table::

        membership_inference_self_test(
            released_rows,
            member_records=source_members,
            heldout_records=source_heldout,
        )

    In both forms, the row identifier field is excluded from attack features.
    If no identifier field is present, deterministic synthetic identifiers are
    generated for the safe report.  ``advantage_budget`` is measured above
    chance and fails closed when the configured value is exceeded.

    Args:
        released_table: A non-empty row mapping sequence or DataFrame-like
            object exposing ``to_dict("records")``.
        member_records: Held-out records in the two-argument form, or source
            records included in the release in the three-argument form.
        heldout_records: Source records withheld from the release.  When
            omitted, the two-argument form is selected.
        record_id_field: Synthetic identifier column to expose in the report.
            It is excluded from the attack features.
        advantage_budget: Maximum allowed attacker advantage above chance.
        top_k: Number of highest-scoring records to expose in the report.

    Returns:
        A :class:`MembershipInferenceResult` containing only safe report data.

    Raises:
        TypeError: If table rows or numeric options have unsupported types.
        ValueError: If a table is empty, identifiers are ambiguous, or the
            advantage budget is outside ``[0, 0.5]``.
    """

    _validate_budget(advantage_budget)
    _validate_top_k(top_k)
    released_rows = _coerce_rows(released_table, "released_table")

    if heldout_records is None:
        if member_records is None:
            raise ValueError("heldout records are required")
        member_rows = released_rows
        heldout_rows = _coerce_rows(member_records, "heldout_records")
        mode = "shadow_distribution"
    else:
        if member_records is None:
            member_rows = released_rows
            mode = "shadow_distribution"
        else:
            member_rows = _coerce_rows(member_records, "member_records")
            mode = "release_proximity"
        heldout_rows = _coerce_rows(heldout_records, "heldout_records")

    id_field = _resolve_id_field(
        (*member_rows, *heldout_rows),
        record_id_field,
    )
    member_ids = _record_ids(member_rows, id_field, "member", record_id_field)
    heldout_ids = _record_ids(
        heldout_rows,
        id_field,
        "heldout",
        record_id_field,
    )
    if len(set(member_ids + heldout_ids)) != len(member_ids) + len(heldout_ids):
        raise ValueError("candidate record identifiers must be unique")

    if mode == "release_proximity":
        member_scores, heldout_scores = _release_proximity_scores(
            released_rows,
            member_rows,
            heldout_rows,
            id_field,
        )
        distribution_member, distribution_heldout = _shadow_distribution_scores(
            member_rows,
            heldout_rows,
            id_field,
        )
        member_scores = tuple(
            0.8 * release_score + 0.2 * distribution_score
            for release_score, distribution_score in zip(
                member_scores,
                distribution_member,
            )
        )
        heldout_scores = tuple(
            0.8 * release_score + 0.2 * distribution_score
            for release_score, distribution_score in zip(
                heldout_scores,
                distribution_heldout,
            )
        )
    else:
        member_scores, heldout_scores = _shadow_distribution_scores(
            member_rows,
            heldout_rows,
            id_field,
        )

    return _build_result(
        member_ids,
        member_scores,
        heldout_ids,
        heldout_scores,
        advantage_budget=advantage_budget,
        top_k=top_k,
        mode=mode,
    )


run_membership_inference_self_test = membership_inference_self_test


def _coerce_rows(data: Any, name: str) -> list[dict[str, Any]]:
    dataframe_records = _maybe_dataframe_records(data)
    if dataframe_records is not None:
        data = dataframe_records
    if isinstance(data, Mapping):
        raw_rows = [data]
    elif isinstance(data, Sequence) and not isinstance(data, (str, bytes, bytearray)):
        raw_rows = list(data)
    else:
        raise TypeError(f"{name} must be a row mapping or sequence of row mappings")
    if not raw_rows:
        raise ValueError(f"{name} must contain at least one row")
    if not all(isinstance(row, Mapping) for row in raw_rows):
        raise TypeError(f"every {name} row must be a mapping")

    rows: list[dict[str, Any]] = []
    for raw_row in raw_rows:
        row: dict[str, Any] = {}
        for key, value in raw_row.items():
            string_key = str(key)
            if string_key in row:
                raise ValueError(f"duplicate stringified column name: {string_key!r}")
            row[string_key] = value
        rows.append(row)
    return rows


def _maybe_dataframe_records(data: Any) -> list[Mapping[str, Any]] | None:
    if isinstance(data, Mapping):
        return None
    to_dict = getattr(data, "to_dict", None)
    if to_dict is None:
        return None
    try:
        records = to_dict("records")
    except TypeError:
        return None
    if isinstance(records, list) and all(isinstance(row, Mapping) for row in records):
        return records
    return None


def _resolve_id_field(
    rows: Sequence[Mapping[str, Any]],
    requested: str | None,
) -> str | None:
    if requested is not None:
        if not isinstance(requested, str) or not requested.strip():
            raise TypeError("record_id_field must be a non-empty string or None")
        return requested
    for field in _AUTO_ID_FIELDS:
        if rows and all(field in row for row in rows):
            return field
    return None


def _record_ids(
    rows: Sequence[Mapping[str, Any]],
    field: str | None,
    prefix: str,
    requested_field: str | None,
) -> list[str]:
    identifiers: list[str] = []
    for index, row in enumerate(rows, start=1):
        if field is not None and field not in row and requested_field is not None:
            raise ValueError(f"{field} must be present for every row")
        if field is not None and field in row:
            value = row[field]
            if value is None or not str(value).strip():
                if requested_field is not None:
                    raise ValueError(f"{field} must be populated for every row")
            else:
                identifiers.append(str(value))
                continue
        identifiers.append(f"{prefix}-{index:04d}")
    return identifiers


def _feature_fields(
    groups: Sequence[Sequence[Mapping[str, Any]]],
    id_field: str | None,
) -> tuple[str, ...]:
    names = {str(key) for group in groups for row in group for key in row}
    if id_field is not None:
        names.discard(id_field)
    return tuple(sorted(names))


def _shadow_distribution_scores(
    members: Sequence[Mapping[str, Any]],
    heldout: Sequence[Mapping[str, Any]],
    id_field: str | None,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    fields = _feature_fields((members, heldout), id_field)
    all_rows = (*members, *heldout)
    specs = _build_field_specs(fields, all_rows, reference_rows=members)
    raw_member_tokens = [_shadow_tokens(row, specs) for row in members]
    raw_heldout_tokens = [_shadow_tokens(row, specs) for row in heldout]
    global_counts = Counter(
        token
        for tokens in (*raw_member_tokens, *raw_heldout_tokens)
        for token in tokens
    )
    member_counts = Counter(
        token
        for tokens in raw_member_tokens
        for token in set(tokens)
        if global_counts[token] >= 2
    )
    heldout_counts = Counter(
        token
        for tokens in raw_heldout_tokens
        for token in set(tokens)
        if global_counts[token] >= 2
    )
    member_scores = tuple(
        _distribution_score(
            tokens,
            member_counts,
            heldout_counts,
            len(members),
            len(heldout),
        )
        for tokens in raw_member_tokens
    )
    heldout_scores = tuple(
        _distribution_score(
            tokens,
            member_counts,
            heldout_counts,
            len(members),
            len(heldout),
        )
        for tokens in raw_heldout_tokens
    )
    return member_scores, heldout_scores


def _build_field_specs(
    fields: Sequence[str],
    rows: Sequence[Mapping[str, Any]],
    *,
    reference_rows: Sequence[Mapping[str, Any]],
) -> tuple[_FieldSpec, ...]:
    specs: list[_FieldSpec] = []
    for field in fields:
        values = [row.get(field, _MISSING) for row in rows]
        numeric_values = [float(value) for value in values if _is_finite_number(value)]
        numeric = bool(numeric_values) and all(
            value is _MISSING or value is None or _is_finite_number(value)
            for value in values
        )
        minimum = min(numeric_values) if numeric else None
        maximum = max(numeric_values) if numeric else None
        reference_values = [row.get(field, _MISSING) for row in reference_rows]
        reference_counts = Counter(_value_key(value) for value in reference_values)
        specs.append(
            _FieldSpec(
                name=field,
                numeric=numeric,
                minimum=minimum,
                maximum=maximum,
                reference_counts=reference_counts,
                reference_count=len(reference_rows),
            )
        )
    return tuple(specs)


def _shadow_tokens(
    row: Mapping[str, Any], specs: Sequence[_FieldSpec]
) -> tuple[str, ...]:
    tokens: list[str] = []
    for spec in specs:
        value = row.get(spec.name, _MISSING)
        prefix = f"{spec.name}|"
        if value is _MISSING:
            tokens.append(prefix + "missing")
            continue
        if value is None:
            tokens.append(prefix + "null")
            continue
        if spec.numeric and _is_finite_number(value):
            tokens.append(prefix + "number|" + _numeric_bucket(value, spec))
            continue
        if isinstance(value, str):
            normalized = _normalise_text(value)
            tokens.append(prefix + "shape|" + _string_shape(normalized))
            tokens.append(prefix + "length|" + _length_bucket(normalized))
            for word in _TEXT_TOKEN_PATTERN.findall(normalized):
                if len(word) >= 2:
                    tokens.append(prefix + "word|" + word)
            tokens.append(prefix + "value|" + normalized)
            continue
        tokens.append(prefix + "value|" + _value_key(value))
    return tuple(tokens)


def _distribution_score(
    tokens: Sequence[str],
    member_counts: Mapping[str, int],
    heldout_counts: Mapping[str, int],
    member_count: int,
    heldout_count: int,
) -> float:
    if not tokens:
        return _BASELINE
    log_odds = 0.0
    for token in set(tokens):
        member_probability = (member_counts.get(token, 0) + 0.5) / (member_count + 1.0)
        heldout_probability = (heldout_counts.get(token, 0) + 0.5) / (
            heldout_count + 1.0
        )
        log_odds += math.log(member_probability / heldout_probability)
    return _sigmoid(log_odds / math.sqrt(max(1, len(set(tokens)))))


def _release_proximity_scores(
    released: Sequence[Mapping[str, Any]],
    members: Sequence[Mapping[str, Any]],
    heldout: Sequence[Mapping[str, Any]],
    id_field: str | None,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    fields = _feature_fields((released, members, heldout), id_field)
    all_rows = (*released, *members, *heldout)
    specs = _build_field_specs(fields, all_rows, reference_rows=released)
    member_scores = tuple(
        max(
            (_row_similarity(row, release_row, specs) for release_row in released),
            default=0.5,
        )
        for row in members
    )
    heldout_scores = tuple(
        max(
            (_row_similarity(row, release_row, specs) for release_row in released),
            default=0.5,
        )
        for row in heldout
    )
    return member_scores, heldout_scores


def _row_similarity(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    specs: Sequence[_FieldSpec],
) -> float:
    if not specs:
        return _BASELINE
    weighted_score = 0.0
    total_weight = 0.0
    for spec in specs:
        left_value = left.get(spec.name, _MISSING)
        right_value = right.get(spec.name, _MISSING)
        weight = _field_weight(spec, right_value)
        weighted_score += weight * _value_similarity(left_value, right_value, spec)
        total_weight += weight
    return weighted_score / total_weight if total_weight else _BASELINE


def _field_weight(spec: _FieldSpec, reference_value: Any) -> float:
    if spec.numeric or spec.reference_count == 0:
        return 1.0
    frequency = spec.reference_counts.get(_value_key(reference_value), 0)
    if frequency <= 0:
        return 1.0
    rarity = math.log((spec.reference_count + 1.0) / (frequency + 1.0))
    return 1.0 + min(3.0, max(0.0, rarity))


def _value_similarity(left: Any, right: Any, spec: _FieldSpec) -> float:
    if left is _MISSING or right is _MISSING:
        return 1.0 if left is right else 0.0
    if left is None or right is None:
        return 1.0 if left is right else 0.0
    if spec.numeric and _is_finite_number(left) and _is_finite_number(right):
        minimum = spec.minimum if spec.minimum is not None else 0.0
        maximum = spec.maximum if spec.maximum is not None else minimum
        scale = max(maximum - minimum, 1.0)
        return max(0.0, 1.0 - abs(float(left) - float(right)) / scale)
    left_key = _normalise_text(left) if isinstance(left, str) else _value_key(left)
    right_key = _normalise_text(right) if isinstance(right, str) else _value_key(right)
    if left_key == right_key:
        return 1.0
    if isinstance(left, str) and isinstance(right, str):
        left_words = set(_TEXT_TOKEN_PATTERN.findall(left_key))
        right_words = set(_TEXT_TOKEN_PATTERN.findall(right_key))
        union = left_words | right_words
        if union:
            return len(left_words & right_words) / len(union)
    return 0.0


def _numeric_bucket(value: Any, spec: _FieldSpec) -> str:
    if not _is_finite_number(value):
        return "missing"
    minimum = spec.minimum if spec.minimum is not None else float(value)
    maximum = spec.maximum if spec.maximum is not None else minimum
    if maximum <= minimum:
        return "0"
    position = (float(value) - minimum) / (maximum - minimum)
    return str(min(9, max(0, int(position * 10))))


def _normalise_text(value: Any) -> str:
    return " ".join(str(value).strip().casefold().split())


def _string_shape(value: str) -> str:
    shape = []
    for character in value:
        if character.isalpha():
            shape.append("a")
        elif character.isdigit():
            shape.append("0")
        elif character.isspace():
            shape.append("_")
        else:
            shape.append("-")
    return "".join(shape)[:32]


def _length_bucket(value: str) -> str:
    return str(min(8, len(value) // 4))


def _value_key(value: Any) -> str:
    if value is _MISSING:
        return "missing"
    if value is None:
        return "null"
    if isinstance(value, bool):
        return f"bool:{value}"
    if _is_finite_number(value):
        return f"number:{float(value):.12g}"
    if isinstance(value, str):
        return f"string:{_normalise_text(value)}"
    try:
        serialized = json.dumps(
            value, sort_keys=True, default=str, separators=(",", ":")
        )
    except (TypeError, ValueError):
        serialized = repr(value)
    return f"object:{serialized}"


def _is_finite_number(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, Real):
        return False
    try:
        return math.isfinite(float(value))
    except (OverflowError, TypeError, ValueError):
        return False


def _build_result(
    member_ids: Sequence[str],
    member_scores: Sequence[float],
    heldout_ids: Sequence[str],
    heldout_scores: Sequence[float],
    *,
    advantage_budget: float,
    top_k: int,
    mode: str,
) -> MembershipInferenceResult:
    labels = [1] * len(member_scores) + [0] * len(heldout_scores)
    scores = [float(score) for score in (*member_scores, *heldout_scores)]
    auc = _auc(member_scores, heldout_scores)
    threshold, accuracy = _best_threshold(scores, labels)
    auc_advantage = max(0.0, max(auc, 1.0 - auc) - _BASELINE)
    accuracy_advantage = max(0.0, accuracy - _BASELINE)
    advantage = max(auc_advantage, accuracy_advantage)
    all_records = [
        {"record_id": record_id, "score": _bounded_score(score)}
        for record_id, score in zip(
            (*member_ids, *heldout_ids),
            scores,
        )
    ]
    all_records.sort(key=lambda item: (-item["score"], item["record_id"]))
    record_scores = tuple(all_records)
    return MembershipInferenceResult(
        auc=float(auc),
        accuracy=float(accuracy),
        advantage=float(advantage),
        auc_advantage=float(auc_advantage),
        accuracy_advantage=float(accuracy_advantage),
        decision_threshold=float(threshold),
        advantage_budget=float(advantage_budget),
        passed=bool(advantage <= advantage_budget),
        member_count=len(member_scores),
        heldout_count=len(heldout_scores),
        riskiest_records=record_scores[:top_k],
        record_scores=record_scores,
        mode=mode,
    )


def _auc(member_scores: Sequence[float], heldout_scores: Sequence[float]) -> float:
    if not member_scores or not heldout_scores:
        return _BASELINE
    wins = 0.0
    for member_score in member_scores:
        for heldout_score in heldout_scores:
            if member_score > heldout_score:
                wins += 1.0
            elif member_score == heldout_score:
                wins += 0.5
    return wins / (len(member_scores) * len(heldout_scores))


def _best_threshold(
    scores: Sequence[float], labels: Sequence[int]
) -> tuple[float, float]:
    if not scores:
        return _BASELINE, _BASELINE
    unique_scores = sorted(set(_bounded_score(score) for score in scores))
    thresholds = [0.0, 1.0, *unique_scores]
    best_threshold = _BASELINE
    best_accuracy = -1.0
    for threshold in thresholds:
        predictions = [score >= threshold for score in scores]
        member_total = sum(label == 1 for label in labels)
        heldout_total = sum(label == 0 for label in labels)
        member_accuracy = (
            sum(
                prediction
                for prediction, label in zip(predictions, labels)
                if label == 1
            )
            / member_total
            if member_total
            else _BASELINE
        )
        heldout_accuracy = (
            sum(
                not prediction
                for prediction, label in zip(predictions, labels)
                if label == 0
            )
            / heldout_total
            if heldout_total
            else _BASELINE
        )
        accuracy = (member_accuracy + heldout_accuracy) / 2.0
        if accuracy > best_accuracy or (
            accuracy == best_accuracy
            and abs(threshold - _BASELINE) < abs(best_threshold - _BASELINE)
        ):
            best_threshold = threshold
            best_accuracy = accuracy
    return best_threshold, max(0.0, best_accuracy)


def _bounded_score(value: Any) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return _BASELINE
    if not math.isfinite(score):
        return _BASELINE
    return min(1.0, max(0.0, score))


def _sigmoid(value: float) -> float:
    if value >= 40.0:
        return 1.0
    if value <= -40.0:
        return 0.0
    return 1.0 / (1.0 + math.exp(-value))


def _validate_budget(value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("advantage_budget must be a finite number")
    numeric = float(value)
    if not math.isfinite(numeric) or not 0.0 <= numeric <= 0.5:
        raise ValueError("advantage_budget must be between 0 and 0.5")


def _validate_top_k(value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError("top_k must be a non-negative integer")
    if value < 0:
        raise ValueError("top_k must be a non-negative integer")
