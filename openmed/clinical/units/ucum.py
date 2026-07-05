"""Deterministic UCUM-subset parsing for clinical measurements."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, TypedDict

UnitParseStatus = Literal["ok", "unknown", "ambiguous"]
Dimension = tuple[tuple[str, int], ...]

MEASUREMENT_NORMALIZATION_ADVISORY = (
    "Measurement normalization is deterministic assistance for comparison and "
    "is not a substitute for the originating laboratory's own formal "
    "diagnostic flagging."
)


class UnitProvenance(TypedDict, total=False):
    """Trace how a unit string was interpreted."""

    input_unit: str
    normalized_unit: str
    source: str
    notes: list[str]


class ParsedUnit(TypedDict, total=False):
    """Public parse result for a UCUM-subset unit string."""

    status: UnitParseStatus
    unit: str | None
    canonical_unit: str | None
    dimension: dict[str, int]
    scale: float | None
    offset: float
    is_affine: bool
    reason: str
    advisory: str
    provenance: UnitProvenance


@dataclass(frozen=True)
class UnitExpression:
    """Parsed unit expression with canonical conversion coefficients."""

    normalized_unit: str
    dimension: Dimension
    scale: float
    offset: float
    is_affine: bool
    provenance: UnitProvenance

    def to_canonical(self, value: float) -> float:
        """Convert ``value`` from this unit to its canonical dimension unit."""

        return (value * self.scale) + self.offset

    def from_canonical(self, value: float) -> float:
        """Convert canonical ``value`` into this unit."""

        return (value - self.offset) / self.scale


@dataclass(frozen=True)
class UnitParse:
    """Internal parser result used by the conversion layer."""

    status: UnitParseStatus
    expression: UnitExpression | None
    reason: str
    provenance: UnitProvenance


@dataclass(frozen=True)
class _UnitAtom:
    code: str
    dimension: Dimension
    scale: float
    offset: float = 0.0
    is_affine: bool = False


def _dimension(**items: int) -> Dimension:
    return tuple(sorted((key, value) for key, value in items.items() if value))


def _dimension_mapping(dimension: Dimension) -> dict[str, int]:
    return dict(dimension)


def _dimension_from_mapping(items: Mapping[str, int]) -> Dimension:
    return tuple(sorted((key, value) for key, value in items.items() if value))


_DIMENSIONLESS = _dimension()
_MASS = _dimension(mass=1)
_AMOUNT = _dimension(amount=1)
_EQUIVALENT_AMOUNT = _dimension(equivalent_amount=1)
_VOLUME = _dimension(volume=1)
_LENGTH = _dimension(length=1)
_TIME = _dimension(time=1)
_PRESSURE = _dimension(pressure=1)
_TEMPERATURE = _dimension(temperature=1)
_ACTIVITY = _dimension(catalytic_activity=1)
_ENTITY = _dimension(entity=1)

_BASE_UNITS: Mapping[str, _UnitAtom] = {
    "1": _UnitAtom("1", _DIMENSIONLESS, 1.0),
    "%": _UnitAtom("%", _DIMENSIONLESS, 0.01),
    "g": _UnitAtom("g", _MASS, 1.0),
    "mol": _UnitAtom("mol", _AMOUNT, 1.0),
    "Eq": _UnitAtom("Eq", _EQUIVALENT_AMOUNT, 1.0),
    "L": _UnitAtom("L", _VOLUME, 1.0),
    "l": _UnitAtom("l", _VOLUME, 1.0),
    "m": _UnitAtom("m", _LENGTH, 1.0),
    "s": _UnitAtom("s", _TIME, 1.0),
    "min": _UnitAtom("min", _TIME, 60.0),
    "h": _UnitAtom("h", _TIME, 3600.0),
    "d": _UnitAtom("d", _TIME, 86400.0),
    "Pa": _UnitAtom("Pa", _PRESSURE, 1.0),
    "kPa": _UnitAtom("kPa", _PRESSURE, 1000.0),
    "mmHg": _UnitAtom("mmHg", _PRESSURE, 133.322387415),
    "K": _UnitAtom("K", _TEMPERATURE, 1.0),
    "Cel": _UnitAtom("Cel", _TEMPERATURE, 1.0, 273.15, True),
    "[degF]": _UnitAtom("[degF]", _TEMPERATURE, 5.0 / 9.0, 255.3722222222222, True),
    "U": _UnitAtom("U", _ACTIVITY, 1.0),
    "cell": _UnitAtom("cell", _ENTITY, 1.0),
    "cells": _UnitAtom("cells", _ENTITY, 1.0),
    "beat": _UnitAtom("beat", _ENTITY, 1.0),
    "breath": _UnitAtom("breath", _ENTITY, 1.0),
    "mcg": _UnitAtom("mcg", _MASS, 1e-6),
}

_PREFIXES: tuple[tuple[str, float], ...] = (
    ("da", 10.0),
    ("k", 1e3),
    ("d", 1e-1),
    ("c", 1e-2),
    ("m", 1e-3),
    ("u", 1e-6),
    ("n", 1e-9),
    ("p", 1e-12),
)
_PREFIXABLE_BASES = {"g", "mol", "Eq", "L", "l", "m", "s"}

_ALIASES: Mapping[str, tuple[str, str]] = {
    "percent": ("%", "clinical alias"),
    "mg/dl": ("mg/dL", "clinical alias"),
    "milligram/deciliter": ("mg/dL", "clinical alias"),
    "milligrams/deciliter": ("mg/dL", "clinical alias"),
    "g/dl": ("g/dL", "clinical alias"),
    "gram/deciliter": ("g/dL", "clinical alias"),
    "grams/deciliter": ("g/dL", "clinical alias"),
    "g/l": ("g/L", "clinical alias"),
    "mg/l": ("mg/L", "clinical alias"),
    "ug/ml": ("ug/mL", "clinical alias"),
    "mcg/ml": ("ug/mL", "clinical alias"),
    "ng/ml": ("ng/mL", "clinical alias"),
    "mmol/l": ("mmol/L", "clinical alias"),
    "umol/l": ("umol/L", "clinical alias"),
    "meq/l": ("mEq/L", "clinical alias"),
    "ueq/l": ("uEq/L", "clinical alias"),
    "u/l": ("U/L", "clinical alias"),
    "iu/l": ("U/L", "clinical alias"),
    "units/l": ("U/L", "clinical alias"),
    "mm hg": ("mmHg", "clinical alias"),
    "mmhg": ("mmHg", "clinical alias"),
    "c": ("Cel", "clinical alias"),
    "degc": ("Cel", "clinical alias"),
    "celsius": ("Cel", "clinical alias"),
    "f": ("[degF]", "clinical alias"),
    "degf": ("[degF]", "clinical alias"),
    "fahrenheit": ("[degF]", "clinical alias"),
    "/min": ("1/min", "clinical alias"),
    "per minute": ("1/min", "clinical alias"),
    "bpm": ("beat/min", "clinical alias"),
    "beats/min": ("beat/min", "clinical alias"),
    "beats per minute": ("beat/min", "clinical alias"),
    "breaths/min": ("breath/min", "clinical alias"),
    "breaths per minute": ("breath/min", "clinical alias"),
    "cells/ul": ("cell/uL", "clinical alias"),
    "cell/ul": ("cell/uL", "clinical alias"),
    "cells/mm3": ("cell/uL", "clinical alias"),
    "cell/mm3": ("cell/uL", "clinical alias"),
    "k/ul": ("10*3/uL", "clinical alias"),
    "k/mm3": ("10*3/uL", "clinical alias"),
    "x10^3/ul": ("10*3/uL", "clinical alias"),
    "10^3/ul": ("10*3/uL", "clinical alias"),
    "10*3/ul": ("10*3/uL", "clinical alias"),
    "10^9/l": ("10*9/L", "clinical alias"),
    "10*9/l": ("10*9/L", "clinical alias"),
}
_AMBIGUOUS_UNITS = {
    "u",
    "unit",
    "units",
    "international unit",
    "international units",
}

_EXPONENT_RE = re.compile(
    r"^(?P<symbol>10(?:\*|\^)-?\d+|[A-Za-z%\[\]]+?)(?P<exponent>-?\d+)?$"
)
_POWER_OF_TEN_RE = re.compile(r"^10(?:\*|\^)(?P<exponent>-?\d+)$")

_CANONICAL_UNITS: Mapping[Dimension, str] = {
    _DIMENSIONLESS: "1",
    _MASS: "g",
    _AMOUNT: "mol",
    _EQUIVALENT_AMOUNT: "Eq",
    _VOLUME: "L",
    _LENGTH: "m",
    _TIME: "s",
    _PRESSURE: "Pa",
    _TEMPERATURE: "K",
    _ACTIVITY: "U",
    _ENTITY: "count",
    _dimension(mass=1, volume=-1): "g/L",
    _dimension(amount=1, volume=-1): "mol/L",
    _dimension(equivalent_amount=1, volume=-1): "Eq/L",
    _dimension(catalytic_activity=1, volume=-1): "U/L",
    _dimension(entity=1, volume=-1): "count/L",
    _dimension(entity=1, time=-1): "count/s",
    _dimension(volume=1, time=-1): "L/s",
    _dimension(mass=1, length=-2): "g/m2",
}


def _canonical_unit(dimension: Dimension) -> str:
    return _CANONICAL_UNITS.get(dimension, _format_dimension(dimension))


def _format_dimension(dimension: Dimension) -> str:
    if not dimension:
        return "1"
    numerator: list[str] = []
    denominator: list[str] = []
    for name, exponent in dimension:
        target = numerator if exponent > 0 else denominator
        power = abs(exponent)
        target.append(name if power == 1 else f"{name}^{power}")
    if not denominator:
        return "*".join(numerator)
    return f"{'*'.join(numerator)}/{'*'.join(denominator)}"


def _alias_key(raw_unit: str) -> str:
    text = _clean_unit_text(raw_unit).casefold()
    text = re.sub(r"\s+per\s+", "/", text)
    text = re.sub(r"\s*/\s*", "/", text)
    text = re.sub(r"\s*\*\s*", "*", text)
    text = re.sub(r"\s*\^\s*", "^", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _clean_unit_text(raw_unit: str) -> str:
    text = raw_unit.strip()
    text = text.replace("\N{GREEK SMALL LETTER MU}", "u")
    text = text.replace("\N{MICRO SIGN}", "u")
    text = text.replace("\N{DEGREE SIGN}", "")
    text = text.replace("**", "^")
    text = text.replace(" x ", " ")
    text = text.replace(" X ", " ")
    text = text.replace(" per ", "/")
    text = re.sub(r"\s*/\s*", "/", text)
    text = re.sub(r"\s*\^\s*", "^", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _failure(
    raw_unit: str,
    status: UnitParseStatus,
    reason: str,
    *,
    normalized_unit: str | None = None,
    notes: list[str] | None = None,
) -> UnitParse:
    provenance: UnitProvenance = {
        "input_unit": raw_unit,
        "source": status,
        "notes": notes or [reason],
    }
    if normalized_unit is not None:
        provenance["normalized_unit"] = normalized_unit
    return UnitParse(
        status=status, expression=None, reason=reason, provenance=provenance
    )


def _resolve_alias(raw_unit: str) -> tuple[str, UnitProvenance] | UnitParse:
    key = _alias_key(raw_unit)
    if not key:
        return _failure(raw_unit, "unknown", "unit is empty")
    if key in _AMBIGUOUS_UNITS:
        return _failure(
            raw_unit,
            "ambiguous",
            "unit string is clinically ambiguous without local context",
            normalized_unit=key,
        )

    normalized = _clean_unit_text(raw_unit)
    notes = ["parsed as UCUM clinical subset"]
    source = "ucum_subset"
    if key in _ALIASES:
        normalized, note = _ALIASES[key]
        notes = [note, f"alias {raw_unit!r} resolved to {normalized!r}"]
        source = "alias_table"

    provenance: UnitProvenance = {
        "input_unit": raw_unit,
        "normalized_unit": normalized,
        "source": source,
        "notes": notes,
    }
    return normalized, provenance


def _unit_atom(symbol: str) -> _UnitAtom | None:
    if symbol in _BASE_UNITS:
        return _BASE_UNITS[symbol]

    for prefix, factor in _PREFIXES:
        if not symbol.startswith(prefix):
            continue
        base_symbol = symbol[len(prefix) :]
        if base_symbol not in _PREFIXABLE_BASES:
            continue
        base = _BASE_UNITS[base_symbol]
        return _UnitAtom(
            code=symbol,
            dimension=base.dimension,
            scale=base.scale * factor,
        )

    return None


def _split_product(part: str) -> list[str]:
    cleaned = part.strip().strip("()")
    if not cleaned:
        return []
    if cleaned == "1":
        return ["1"]
    if _POWER_OF_TEN_RE.fullmatch(cleaned):
        return [cleaned]
    return [
        token for token in re.split(r"(?:[.\s]+)", cleaned) if token and token != "1"
    ]


def _merge_dimension(
    target: dict[str, int],
    dimension: Dimension,
    exponent: int,
) -> None:
    for name, power in dimension:
        target[name] = target.get(name, 0) + (power * exponent)
        if target[name] == 0:
            del target[name]


def _parse_factor(
    token: str,
    exponent_sign: int,
) -> tuple[Dimension, float, bool, str | None]:
    if token == "1":
        return _DIMENSIONLESS, 1.0, False, None
    if power_match := _POWER_OF_TEN_RE.fullmatch(token):
        exponent = int(power_match.group("exponent")) * exponent_sign
        return _ENTITY, 10.0**exponent, False, None

    match = _EXPONENT_RE.fullmatch(token)
    if match is None:
        return _DIMENSIONLESS, 1.0, False, f"unsupported UCUM token {token!r}"

    symbol = match.group("symbol")
    atom = _unit_atom(symbol)
    if atom is None:
        return _DIMENSIONLESS, 1.0, False, f"unknown unit token {symbol!r}"

    suffix_exponent = int(match.group("exponent") or "1")
    exponent = suffix_exponent * exponent_sign
    if atom.is_affine and exponent != 1:
        return (
            _DIMENSIONLESS,
            1.0,
            False,
            "affine temperature units cannot be used as compound units",
        )

    dimension = _dimension_from_mapping(
        {name: power * exponent for name, power in atom.dimension}
    )
    return dimension, atom.scale**exponent, atom.is_affine, None


def _parse_unit_expression(raw_unit: object) -> UnitParse:
    if not isinstance(raw_unit, str):
        return _failure("", "unknown", "unit must be a string")

    resolved = _resolve_alias(raw_unit)
    if isinstance(resolved, UnitParse):
        return resolved

    normalized, provenance = resolved
    parts = normalized.split("/")
    if any(part == "" for part in parts):
        return _failure(
            raw_unit,
            "unknown",
            "unit expression has an empty numerator or denominator",
            normalized_unit=normalized,
            notes=provenance.get("notes"),
        )

    dimensions: dict[str, int] = {}
    scale = 1.0
    affine_count = 0
    token_count = 0

    for index, part in enumerate(parts):
        exponent_sign = 1 if index == 0 else -1
        for token in _split_product(part):
            token_count += 1
            dimension, factor, is_affine, error = _parse_factor(token, exponent_sign)
            if error is not None:
                return _failure(
                    raw_unit,
                    "unknown",
                    error,
                    normalized_unit=normalized,
                    notes=provenance.get("notes"),
                )
            if is_affine:
                affine_count += 1
            _merge_dimension(dimensions, dimension, 1)
            scale *= factor

    if token_count == 0:
        return _failure(
            raw_unit,
            "unknown",
            "unit expression did not contain a recognized unit",
            normalized_unit=normalized,
            notes=provenance.get("notes"),
        )
    if affine_count and token_count != 1:
        return _failure(
            raw_unit,
            "unknown",
            "affine temperature units cannot be used as compound units",
            normalized_unit=normalized,
            notes=provenance.get("notes"),
        )

    offset = 0.0
    if affine_count:
        atom = _unit_atom(parts[0])
        offset = atom.offset if atom is not None else 0.0

    dimension = _dimension_from_mapping(dimensions)
    expression = UnitExpression(
        normalized_unit=normalized,
        dimension=dimension,
        scale=scale,
        offset=offset,
        is_affine=bool(affine_count),
        provenance=provenance,
    )
    return UnitParse(
        status="ok", expression=expression, reason="", provenance=provenance
    )


def parse_unit(unit: object) -> ParsedUnit:
    """Parse a UCUM-subset unit string into a canonical dimension.

    The supported subset intentionally covers common clinical mass, amount,
    volume, pressure, temperature, time, count, and derived concentration
    expressions. Ambiguous strings such as ``"units"`` return ``"ambiguous"``
    instead of guessing.
    """

    parsed = _parse_unit_expression(unit)
    if parsed.expression is None:
        return {
            "status": parsed.status,
            "unit": None,
            "canonical_unit": None,
            "dimension": {},
            "scale": None,
            "offset": 0.0,
            "is_affine": False,
            "reason": parsed.reason,
            "advisory": MEASUREMENT_NORMALIZATION_ADVISORY,
            "provenance": parsed.provenance,
        }

    expression = parsed.expression
    return {
        "status": "ok",
        "unit": expression.normalized_unit,
        "canonical_unit": _canonical_unit(expression.dimension),
        "dimension": _dimension_mapping(expression.dimension),
        "scale": expression.scale,
        "offset": expression.offset,
        "is_affine": expression.is_affine,
        "reason": "",
        "advisory": MEASUREMENT_NORMALIZATION_ADVISORY,
        "provenance": expression.provenance,
    }


def is_dimensionless(dimension: Dimension) -> bool:
    """Return whether ``dimension`` has no base dimensions."""

    return not dimension


def canonical_unit_for_dimension(dimension: Dimension) -> str:
    """Return the display name for a canonical dimension unit."""

    return _canonical_unit(dimension)


def finite_float(value: object) -> float | None:
    """Return a finite float or ``None`` without accepting booleans."""

    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


__all__ = [
    "Dimension",
    "MEASUREMENT_NORMALIZATION_ADVISORY",
    "ParsedUnit",
    "UnitExpression",
    "UnitParse",
    "UnitParseStatus",
    "UnitProvenance",
    "canonical_unit_for_dimension",
    "finite_float",
    "is_dimensionless",
    "parse_unit",
]
