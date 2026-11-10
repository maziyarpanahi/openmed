"""Declarative generalization function family for tabular quasi-identifiers.

Pure-Python leaf transforms that raise individual quasi-identifier values along
per-column-type generalization hierarchies. Each supported column type exposes
an ordered family of generalization levels: level ``0`` is the most specific
form the family offers and every higher level is guaranteed to be no more
specific than the one below it (monotone coarsening). A sibling lattice search
composes these families to reach a target k-anonymity; this module provides
only the leaf transforms plus their ordering and level metadata.

The families are:

* ``age``  -> 5/10/20-year bands with a single ``90+`` top band, then full
  suppression.
* ``zip``  -> prefix truncation, dropping trailing characters per level, then
  full suppression (also covers alphanumeric postcodes).
* ``date`` -> a stable per-subject day shift, then truncation to month, then to
  year, then full suppression.
* ``clinical_code`` -> exact code, caller-supplied parent chains, then full
  suppression. No terminology content is bundled.

Date shifting is routed through :func:`openmed.core.date_shift.stable_offset_for`
so a subject's offset is HMAC-derived, deterministic across calls, and never
stored. The module imports with no JVM, no bundled terminology, and no network
access.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, Final

from openmed.core.date_shift import DEFAULT_DATE_SHIFT_MAX_DAYS, stable_offset_for

HIERARCHY_SCHEMA_VERSION: Final = "1.1.0"

COLUMN_TYPE_AGE: Final = "age"
COLUMN_TYPE_ZIP: Final = "zip"
COLUMN_TYPE_DATE: Final = "date"
COLUMN_TYPE_CLINICAL_CODE: Final = "clinical_code"

#: Canonical token emitted when a value is fully suppressed (coarsest rung).
SUPPRESSED: Final = "*"

#: No banded age output is narrower than this many years.
AGE_MIN_BAND_WIDTH: Final = 5
#: Ages at or above this threshold collapse into the single top band.
AGE_TOP_THRESHOLD: Final = 90
AGE_TOP_BAND: Final = "90+"
#: Largest accepted age; values outside ``0..AGE_MAX`` are rejected.
AGE_MAX: Final = 150
#: Ordered band widths, finest first. Each is a multiple of the previous so the
#: bands nest and coarsening stays monotone.
AGE_BAND_WIDTHS: Final = (5, 10, 20)

#: Number of prefix-truncation rungs before guaranteed suppression. Dropping
#: this many trailing characters fully suppresses every supported postcode
#: length (5-digit ZIP and 7-character postcodes both included).
ZIP_MAX_TRUNCATION: Final = 7

_ISO_DATE_FORMAT: Final = "%Y-%m-%d"


class HierarchyError(ValueError):
    """Raised for an unknown column type, an out-of-range level, or a value
    that is not valid for the requested column type."""


# --------------------------------------------------------------------------- #
# Value coercion and leaf transforms                                          #
# --------------------------------------------------------------------------- #
def _coerce_age(value: object) -> int:
    """Return a validated integer age in ``0..AGE_MAX``."""
    if isinstance(value, bool):
        raise HierarchyError("age must be an integer, not a bool")
    if isinstance(value, int):
        age = value
    elif isinstance(value, str):
        text = value.strip()
        if not text.isdecimal():
            raise HierarchyError(f"age must be a non-negative integer, got {value!r}")
        age = int(text)
    else:
        raise HierarchyError(f"age must be int or str, got {type(value).__name__}")
    if age < 0 or age > AGE_MAX:
        raise HierarchyError(f"age {age} is outside the supported range 0..{AGE_MAX}")
    return age


def _age_band(value: object, *, width: int) -> str:
    """Bucket an age into a ``lower-upper`` band or the ``90+`` top band."""
    age = _coerce_age(value)
    if age >= AGE_TOP_THRESHOLD:
        return AGE_TOP_BAND
    lower = (age // width) * width
    upper = min(lower + width - 1, AGE_TOP_THRESHOLD - 1)
    return f"{lower}-{upper}"


def _coerce_zip(value: object) -> str:
    """Return a validated, whitespace-stripped alphanumeric ZIP/postcode."""
    if not isinstance(value, str):
        raise HierarchyError(f"zip/postcode must be a str, got {type(value).__name__}")
    text = value.strip()
    if not text:
        raise HierarchyError("zip/postcode must be non-empty")
    if not text.isalnum():
        raise HierarchyError(f"zip/postcode must be alphanumeric, got {value!r}")
    return text


def _zip_prefix(value: object, *, drop: int) -> str:
    """Keep the leading characters after dropping ``drop`` trailing ones."""
    text = _coerce_zip(value)
    keep = len(text) - drop
    if keep <= 0:
        return SUPPRESSED
    return text[:keep]


def _coerce_date(value: object) -> date:
    """Return a validated :class:`datetime.date` from a date or ISO string."""
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        text = value
        try:
            parsed = datetime.strptime(text, _ISO_DATE_FORMAT).date()
        except ValueError as exc:
            raise HierarchyError(
                f"date must be an ISO YYYY-MM-DD string, got {value!r}"
            ) from exc
        if parsed.isoformat() != text:
            raise HierarchyError(
                f"date must be an ISO YYYY-MM-DD string, got {value!r}"
            )
        return parsed
    raise HierarchyError(
        f"date must be a date or ISO string, got {type(value).__name__}"
    )


def _coerce_clinical_code(value: object) -> str:
    """Return a non-empty clinical code without interpreting its terminology."""
    if not isinstance(value, str):
        raise HierarchyError(f"clinical code must be a str, got {type(value).__name__}")
    if not value or value == SUPPRESSED:
        raise HierarchyError("clinical code must be non-empty and not reserved")
    return value


def _shift_date(
    value: object,
    *,
    patient_key: str | bytes | None,
    secret: str | bytes | None,
    max_days: int,
) -> str:
    """Shift a date by the subject's stable HMAC-derived day offset.

    The same ``patient_key`` and ``secret`` always yield the same offset, so
    interval deltas between a subject's dates are preserved while absolute dates
    are hidden. The offset is never returned or stored.
    """
    parsed = _coerce_date(value)
    if patient_key is None or secret is None:
        raise HierarchyError("date shifting requires both patient_key and secret")
    try:
        offset = stable_offset_for(patient_key, max_days=max_days, secret=secret)
    except (TypeError, ValueError) as exc:
        raise HierarchyError("date shifting parameters are invalid") from exc
    try:
        shifted = parsed + timedelta(days=offset)
    except OverflowError as exc:
        raise HierarchyError("date shift exceeds the supported date range") from exc
    return shifted.isoformat()


def _truncate_month(value: object) -> str:
    """Drop the day component, yielding ``YYYY-MM``."""
    parsed = _coerce_date(value)
    return f"{parsed.year:04d}-{parsed.month:02d}"


def _truncate_year(value: object) -> str:
    """Drop the day and month components, yielding ``YYYY``."""
    parsed = _coerce_date(value)
    return f"{parsed.year:04d}"


# A transform has a uniform keyword signature so the registry can call any rung
# the same way. ``patient_key``/``secret``/``max_days`` are consumed only by the
# date-shift rung and ignored elsewhere.
_Transform = Callable[..., str]


def _age_transform(width: int) -> _Transform:
    def transform(value: object, **_: object) -> str:
        return _age_band(value, width=width)

    return transform


def _zip_transform(drop: int) -> _Transform:
    def transform(value: object, **_: object) -> str:
        return _zip_prefix(value, drop=drop)

    return transform


def _date_shift_transform() -> _Transform:
    def transform(
        value: object,
        *,
        patient_key: str | bytes | None = None,
        secret: str | bytes | None = None,
        max_days: int = DEFAULT_DATE_SHIFT_MAX_DAYS,
        **_: object,
    ) -> str:
        return _shift_date(
            value, patient_key=patient_key, secret=secret, max_days=max_days
        )

    return transform


def _month_transform() -> _Transform:
    def transform(value: object, **_: object) -> str:
        return _truncate_month(value)

    return transform


def _year_transform() -> _Transform:
    def transform(value: object, **_: object) -> str:
        return _truncate_year(value)

    return transform


def _suppress_transform() -> _Transform:
    def transform(value: object, **_: object) -> str:
        return SUPPRESSED

    return transform


def _clinical_code_transform() -> _Transform:
    def transform(value: object, **_: object) -> str:
        return _coerce_clinical_code(value)

    return transform


# --------------------------------------------------------------------------- #
# Typed contract: levels and hierarchies                                      #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class GeneralizationLevel:
    """Immutable metadata for one rung of a column type's generalization family.

    ``level`` is the rung index (``0`` is finest, larger is coarser), ``key`` is
    a stable machine-readable identifier, and ``description`` is human-readable.
    """

    level: int
    key: str
    description: str


@dataclass(frozen=True)
class Hierarchy:
    """Ordered, versioned generalization family for a single column type.

    ``levels`` is index-aligned with the rung transforms: ``levels[i]`` describes
    the transform applied by ``apply(value, i)``. The ordering is the monotone
    contract a lattice search builds on -- applying a higher level never yields a
    more specific value than a lower one.
    """

    column_type: str
    version: str
    levels: tuple[GeneralizationLevel, ...]

    @property
    def max_level(self) -> int:
        """Highest valid level index (the fully suppressed rung)."""
        return len(self.levels) - 1

    def validate_level(self, level: int) -> None:
        """Raise :class:`HierarchyError` unless ``level`` is a valid rung index."""
        if isinstance(level, bool) or not isinstance(level, int):
            raise HierarchyError(
                f"level must be an integer, got {type(level).__name__}"
            )
        if level < 0 or level > self.max_level:
            raise HierarchyError(
                f"level {level} is out of range 0..{self.max_level} "
                f"for column_type {self.column_type!r}"
            )

    def describe_level(self, level: int) -> GeneralizationLevel:
        """Return the :class:`GeneralizationLevel` metadata for ``level``."""
        self.validate_level(level)
        return self.levels[level]

    def apply(
        self,
        value: object,
        level: int,
        *,
        patient_key: str | bytes | None = None,
        secret: str | bytes | None = None,
        max_days: int = DEFAULT_DATE_SHIFT_MAX_DAYS,
    ) -> str:
        """Generalize ``value`` at ``level``; see :func:`generalize_value`."""
        return generalize_value(
            self.column_type,
            value,
            level,
            patient_key=patient_key,
            secret=secret,
            max_days=max_days,
        )


def _build_age_hierarchy() -> tuple[Hierarchy, tuple[_Transform, ...]]:
    levels: list[GeneralizationLevel] = []
    transforms: list[_Transform] = []
    for index, width in enumerate(AGE_BAND_WIDTHS):
        levels.append(
            GeneralizationLevel(
                index,
                f"age:{width}y",
                f"{width}-year age bands with a {AGE_TOP_BAND} top band",
            )
        )
        transforms.append(_age_transform(width))
    top = len(AGE_BAND_WIDTHS)
    levels.append(GeneralizationLevel(top, "age:suppressed", "fully suppressed age"))
    transforms.append(_suppress_transform())
    hierarchy = Hierarchy(COLUMN_TYPE_AGE, HIERARCHY_SCHEMA_VERSION, tuple(levels))
    return hierarchy, tuple(transforms)


def _build_zip_hierarchy() -> tuple[Hierarchy, tuple[_Transform, ...]]:
    levels: list[GeneralizationLevel] = []
    transforms: list[_Transform] = []
    for drop in range(ZIP_MAX_TRUNCATION):
        if drop == 0:
            key, description = "zip:full", "full ZIP/postcode"
        else:
            plural = "s" if drop != 1 else ""
            key = f"zip:drop{drop}"
            description = f"drop {drop} trailing character{plural} (prefix)"
        levels.append(GeneralizationLevel(drop, key, description))
        transforms.append(_zip_transform(drop))
    levels.append(
        GeneralizationLevel(
            ZIP_MAX_TRUNCATION,
            "zip:suppressed",
            "fully suppressed ZIP/postcode",
        )
    )
    transforms.append(_suppress_transform())
    hierarchy = Hierarchy(COLUMN_TYPE_ZIP, HIERARCHY_SCHEMA_VERSION, tuple(levels))
    return hierarchy, tuple(transforms)


def _build_date_hierarchy() -> tuple[Hierarchy, tuple[_Transform, ...]]:
    levels = (
        GeneralizationLevel(
            0,
            "date:shift",
            "stable per-subject day shift (HMAC-derived, preserves deltas)",
        ),
        GeneralizationLevel(1, "date:month", "truncate to month (YYYY-MM)"),
        GeneralizationLevel(2, "date:year", "truncate to year (YYYY)"),
        GeneralizationLevel(3, "date:suppressed", "fully suppressed date"),
    )
    transforms = (
        _date_shift_transform(),
        _month_transform(),
        _year_transform(),
        _suppress_transform(),
    )
    hierarchy = Hierarchy(COLUMN_TYPE_DATE, HIERARCHY_SCHEMA_VERSION, levels)
    return hierarchy, transforms


def _build_clinical_code_hierarchy() -> tuple[Hierarchy, tuple[_Transform, ...]]:
    levels = (
        GeneralizationLevel(0, "clinical_code:exact", "exact clinical code"),
        GeneralizationLevel(
            1,
            "clinical_code:suppressed",
            "fully suppressed clinical code",
        ),
    )
    transforms = (_clinical_code_transform(), _suppress_transform())
    hierarchy = Hierarchy(
        COLUMN_TYPE_CLINICAL_CODE,
        HIERARCHY_SCHEMA_VERSION,
        levels,
    )
    return hierarchy, transforms


_REGISTRY: dict[str, Hierarchy] = {}
_TRANSFORMS: dict[str, tuple[_Transform, ...]] = {}
for _hierarchy, _transforms in (
    _build_age_hierarchy(),
    _build_zip_hierarchy(),
    _build_date_hierarchy(),
    _build_clinical_code_hierarchy(),
):
    _REGISTRY[_hierarchy.column_type] = _hierarchy
    _TRANSFORMS[_hierarchy.column_type] = _transforms

#: The column types this module can generalize.
SUPPORTED_COLUMN_TYPES: Final = frozenset(_REGISTRY)


# --------------------------------------------------------------------------- #
# Public API                                                                  #
# --------------------------------------------------------------------------- #
def get_hierarchy(column_type: str) -> Hierarchy:
    """Return the :class:`Hierarchy` for ``column_type``.

    Raises :class:`HierarchyError` for an unknown or non-string column type.
    """
    try:
        return _REGISTRY[column_type]
    except (KeyError, TypeError):
        supported = ", ".join(sorted(SUPPORTED_COLUMN_TYPES))
        raise HierarchyError(
            f"unknown column_type {column_type!r}; supported: {supported}"
        ) from None


def max_level(column_type: str) -> int:
    """Return the highest valid level index for ``column_type``."""
    return get_hierarchy(column_type).max_level


def describe_level(column_type: str, level: int) -> GeneralizationLevel:
    """Return the :class:`GeneralizationLevel` metadata for a rung."""
    return get_hierarchy(column_type).describe_level(level)


def generalize_value(
    column_type: str,
    value: object,
    level: int,
    *,
    patient_key: str | bytes | None = None,
    secret: str | bytes | None = None,
    max_days: int = DEFAULT_DATE_SHIFT_MAX_DAYS,
) -> str:
    """Generalize a single ``value`` for ``column_type`` at ``level``.

    ``level`` ``0`` is the most specific rung and each higher level is monotone
    coarser. ``patient_key``/``secret`` are required only by the ``date`` shift
    rung (level ``0``) and ignored by every other rung. The result is always a
    canonical string label suitable for equivalence-class grouping.

    Raises :class:`HierarchyError` for an unknown ``column_type``, an
    out-of-range ``level``, a value invalid for the column type, or a date-shift
    rung invoked without ``patient_key`` and ``secret``.
    """
    hierarchy = get_hierarchy(column_type)
    hierarchy.validate_level(level)
    transform = _TRANSFORMS[column_type][level]
    return transform(value, patient_key=patient_key, secret=secret, max_days=max_days)


# --------------------------------------------------------------------------- #
# Interop: emit enforce_kanon-compatible generalization specs                 #
# --------------------------------------------------------------------------- #
# ``openmed.risk.kanon.enforce_kanon`` accepts, per column, an ordered sequence
# of level specs (``Mapping[str, Any]``) finest-to-coarsest: index 0 is a
# canonical identity level (no ``values``/``default``) and each later level is a
# class-merging coarsening described by an explicit ``values`` map (source ->
# output) or a ``default`` catch-all. This module's declarative family is
# functional rather than value-enumerated, so a spec is materialized by mapping
# each observed value through the family's merging rungs.
#
# Only class-merging rungs participate in a k-anonymity merge lattice. Per
# column type these are the family level indices below; excluded rungs are:
#   * ``zip`` level 0 (full value) -- an identity rung, replaced by the
#     canonical identity level enforce_kanon prepends.
#   * ``date`` level 0 (per-subject HMAC shift) -- a separate privacy transform,
#     NOT a class-merging generalization, so it cannot take part in a k-anon
#     merge and is intentionally omitted from the emitted spec.
_MERGING_LEVEL_INDICES: Final = {
    COLUMN_TYPE_AGE: (0, 1, 2, 3),
    COLUMN_TYPE_ZIP: (1, 2, 3, 4, 5, 6, 7),
    COLUMN_TYPE_DATE: (1, 2, 3),
    COLUMN_TYPE_CLINICAL_CODE: (1,),
}


def to_enforce_kanon_hierarchy(
    column_type: str,
    values: Iterable[object] | None = None,
    *,
    clinical_code_parent_chains: Mapping[str, Sequence[str]] | None = None,
) -> list[dict[str, Any]]:
    """Return an ``enforce_kanon``-compatible level-spec sequence for a column.

    The result is an ordered list of level specs finest-to-coarsest, ready to be
    passed as one column of ``openmed.risk.kanon.enforce_kanon``'s
    ``hierarchies=`` mapping. Index 0 is a canonical identity level; each later
    entry is a class-merging rung of this module's declarative family, rendered
    as an explicit ``values`` map (or a ``default`` catch-all for suppression).

    Because ``enforce_kanon`` hierarchies are value-enumerated, the source domain
    must be materialized: pass the column's observed ``values``. For ``age`` the
    full ``0..AGE_MAX`` domain is enumerated when ``values`` is omitted; ``zip``
    and ``date`` require ``values`` (their domains are unbounded). The
    per-subject date-shift rung is excluded because it is not a class-merging
    generalization. ``clinical_code`` requires a caller-supplied mapping from
    each observed leaf code to its ordered immediate-parent-to-root chain; the
    mapping is data, so no ICD, SNOMED CT, or other terminology is bundled.

    Raises :class:`HierarchyError` for an unknown ``column_type`` or when
    ``values`` are required but not supplied.
    """
    hierarchy = get_hierarchy(column_type)
    if column_type == COLUMN_TYPE_CLINICAL_CODE:
        return _clinical_code_enforcement_hierarchy(
            values,
            parent_chains=clinical_code_parent_chains,
        )
    if values is None:
        if column_type == COLUMN_TYPE_AGE:
            domain: list[object] = list(range(AGE_MAX + 1))
        else:
            raise HierarchyError(
                f"column_type {column_type!r} requires observed values to "
                "materialize an enforce_kanon hierarchy"
            )
    else:
        domain = list(values)

    spec: list[dict[str, Any]] = [{"name": f"{column_type}:exact"}]
    merging = _MERGING_LEVEL_INDICES[column_type]

    # enforce_kanon's partition validator re-injects every level's outputs as
    # candidate source values at EVERY level (finer, same, and coarser) and, for
    # some field names, normalizes them before lookup. An output string that is
    # left un-mapped at some level falls through to an escaped literal, which can
    # look like a split of a class an adjacent level merged. Guard against that
    # by first collecting every output string any coarsening rung emits together
    # with a representative source value that produces it, then mapping that
    # string -- at every rung -- exactly as that representative coarsens. Each
    # re-injected output then behaves identically to a genuine occurrence of its
    # representative at all rungs, which is the monotonicity the validator
    # checks.
    output_reps: dict[str, object] = {}
    for index in merging:
        if index == hierarchy.max_level:
            continue
        for value in domain:
            output = generalize_value(column_type, value, index)
            output_reps.setdefault(output, value)

    for index in merging:
        level = hierarchy.levels[index]
        if index == hierarchy.max_level:
            # Fully suppressed rung: collapse every value into one class.
            spec.append({"name": level.key, "default": SUPPRESSED})
            continue
        value_map: dict[str, str] = {}
        for value in domain:
            value_map[str(value)] = generalize_value(column_type, value, index)
        # ``setdefault`` keeps observed-value mappings authoritative when an
        # output string coincides with an observed value (e.g. a short ZIP whose
        # literal spelling equals a longer ZIP's prefix).
        for output, representative in output_reps.items():
            value_map.setdefault(
                output, generalize_value(column_type, representative, index)
            )
        spec.append({"name": level.key, "values": value_map})
    return spec


def _clinical_code_enforcement_hierarchy(
    values: Iterable[object] | None,
    *,
    parent_chains: Mapping[str, Sequence[str]] | None,
) -> list[dict[str, Any]]:
    """Materialize caller-supplied clinical-code parent chains for enforcement."""
    if values is None:
        raise HierarchyError(
            "column_type 'clinical_code' requires observed values to materialize "
            "an enforce_kanon hierarchy"
        )
    if not isinstance(parent_chains, Mapping) or not parent_chains:
        raise HierarchyError(
            "column_type 'clinical_code' requires clinical code parent-chain data"
        )

    normalized: dict[str, tuple[str, ...]] = {}
    for code, chain in parent_chains.items():
        leaf = _coerce_clinical_code(code)
        if isinstance(chain, (str, bytes, bytearray)) or not isinstance(
            chain, Sequence
        ):
            raise HierarchyError(
                "each clinical code parent chain must be a sequence of codes"
            )
        parents = tuple(_coerce_clinical_code(parent) for parent in chain)
        if not parents:
            raise HierarchyError("each clinical code parent chain must be non-empty")
        if leaf in parents or len(set(parents)) != len(parents):
            raise HierarchyError(
                "clinical code parent chains must be acyclic and contain no duplicates"
            )
        normalized[leaf] = parents

    domain = [_coerce_clinical_code(value) for value in values]
    missing_count = sum(code not in normalized for code in domain)
    if missing_count:
        raise HierarchyError(
            f"clinical code hierarchy is missing {missing_count} observed value(s)"
        )

    max_depth = max(len(normalized[code]) for code in domain)

    def roll_up(code: str, depth: int) -> str:
        chain = normalized[code]
        return chain[min(depth - 1, len(chain) - 1)]

    _validate_clinical_code_partitions(normalized, max_depth=max_depth)

    output_representatives: dict[str, str] = {}
    for depth in range(1, max_depth + 1):
        for code in domain:
            output_representatives.setdefault(roll_up(code, depth), code)

    spec: list[dict[str, Any]] = [{"name": "clinical_code:exact"}]
    for depth in range(1, max_depth + 1):
        value_map = {code: roll_up(code, depth) for code in domain}
        for output, representative in output_representatives.items():
            value_map.setdefault(output, roll_up(representative, depth))
        spec.append(
            {
                "name": f"clinical_code:parent_{depth}",
                "values": value_map,
            }
        )
    spec.append({"name": "clinical_code:suppressed", "default": SUPPRESSED})
    return spec


def _validate_clinical_code_partitions(
    parent_chains: Mapping[str, Sequence[str]],
    *,
    max_depth: int,
) -> None:
    """Reject parent data that splits a class after an earlier merge."""

    def output(code: str, depth: int) -> str:
        chain = parent_chains[code]
        return chain[min(depth - 1, len(chain) - 1)]

    for depth in range(1, max_depth):
        groups: dict[str, list[str]] = {}
        for code in parent_chains:
            groups.setdefault(output(code, depth), []).append(code)
        for group in groups.values():
            if len({output(code, depth + 1) for code in group}) > 1:
                raise HierarchyError(
                    "clinical code parent chains split a class after an earlier merge"
                )


def build_enforcement_hierarchies(
    column_to_type: Mapping[str, str],
    records: Sequence[Mapping[str, Any]],
    *,
    clinical_code_hierarchies: Mapping[str, Mapping[str, Sequence[str]]] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Build an ``enforce_kanon`` ``hierarchies=`` mapping from a declarative plan.

    ``column_to_type`` maps each quasi-identifier column name to one of the
    supported column types (``age``/``zip``/``date``/``clinical_code``).
    Observed values are read from ``records`` so the emitted value maps key
    exactly on the values the engine will see. Clinical-code columns additionally
    require a ``column -> leaf -> parent chain`` mapping in
    ``clinical_code_hierarchies``. The returned mapping is ready to pass straight
    to ``openmed.risk.kanon.enforce_kanon(..., hierarchies=<result>)``.

    Raises :class:`HierarchyError` for an unknown column type.
    """
    if clinical_code_hierarchies is not None and not isinstance(
        clinical_code_hierarchies, Mapping
    ):
        raise HierarchyError("clinical_code_hierarchies must be a column mapping")
    code_hierarchies = clinical_code_hierarchies or {}
    unknown_code_columns = sorted(set(code_hierarchies) - set(column_to_type))
    if unknown_code_columns:
        raise HierarchyError(
            "clinical code hierarchies target undeclared columns: "
            f"{unknown_code_columns!r}"
        )

    result: dict[str, list[dict[str, Any]]] = {}
    for column, column_type in column_to_type.items():
        observed = [record[column] for record in records if column in record]
        parent_chains = code_hierarchies.get(column)
        if column_type != COLUMN_TYPE_CLINICAL_CODE and parent_chains is not None:
            raise HierarchyError(
                f"clinical code hierarchy was supplied for non-code column {column!r}"
            )
        result[column] = to_enforce_kanon_hierarchy(
            column_type,
            observed,
            clinical_code_parent_chains=parent_chains,
        )
    return result


__all__ = [
    "AGE_BAND_WIDTHS",
    "AGE_MAX",
    "AGE_MIN_BAND_WIDTH",
    "AGE_TOP_BAND",
    "AGE_TOP_THRESHOLD",
    "COLUMN_TYPE_AGE",
    "COLUMN_TYPE_CLINICAL_CODE",
    "COLUMN_TYPE_DATE",
    "COLUMN_TYPE_ZIP",
    "GeneralizationLevel",
    "HIERARCHY_SCHEMA_VERSION",
    "Hierarchy",
    "HierarchyError",
    "SUPPORTED_COLUMN_TYPES",
    "SUPPRESSED",
    "ZIP_MAX_TRUNCATION",
    "build_enforcement_hierarchies",
    "describe_level",
    "generalize_value",
    "get_hierarchy",
    "max_level",
    "to_enforce_kanon_hierarchy",
]
