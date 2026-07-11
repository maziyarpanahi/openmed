"""Structured medication sig parsing (roadmap v1.9).

Medication sigs pack dose, dose-form, route, frequency, duration, and a
PRN condition into terse strings ("1 tab PO BID x7 days", "take 2 puffs q4h
PRN"). This module parses them into a structured :class:`Sig`, which is the
value layer beneath medication grounding, reconciliation, and FHIR Dosage
export.

Frequency and duration are *not* re-implemented here: the parser isolates the
frequency and duration candidates from the sig and delegates their
normalization to :func:`openmed.clinical.medication_sig.normalize_frequency`
and :func:`~openmed.clinical.medication_sig.normalize_duration`, the shipped
helpers that own that lexicon. This module adds only what those helpers
deliberately exclude: dose, dose-form, route, and the PRN condition. Parsing is
deterministic and offline; malformed or partial sigs parse what is present and
flag the missing components.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import TypedDict

from .medication_sig import normalize_duration, normalize_frequency

SIG_PARSER_ADVISORY = (
    "Medication sig parsing is deterministic support tooling and is not a "
    "substitute for clinician review; malformed sigs parse what is present and "
    "flag missing components rather than guessing."
)

# Route abbreviations / phrases -> controlled route. Longer phrases first.
_ROUTE_LEXICON: tuple[tuple[str, str], ...] = (
    ("by mouth", "oral"),
    ("per os", "oral"),
    ("po", "oral"),
    ("intravenous", "intravenous"),
    ("iv", "intravenous"),
    ("intramuscular", "intramuscular"),
    ("im", "intramuscular"),
    ("subcutaneous", "subcutaneous"),
    ("subcut", "subcutaneous"),
    ("subq", "subcutaneous"),
    ("sq", "subcutaneous"),
    ("sc", "subcutaneous"),
    ("sublingual", "sublingual"),
    ("sl", "sublingual"),
    ("per rectum", "rectal"),
    ("rectally", "rectal"),
    ("pr", "rectal"),
    ("topically", "topical"),
    ("topical", "topical"),
    ("inhaled", "inhaled"),
    ("nebulized", "inhaled"),
    ("inh", "inhaled"),
    ("intranasal", "nasal"),
    ("nasally", "nasal"),
    ("ophthalmic", "ophthalmic"),
)

# Dose-form words -> controlled form. The token doubles as the count unit.
_FORM_LEXICON: dict[str, str] = {
    "tab": "tablet",
    "tabs": "tablet",
    "tablet": "tablet",
    "tablets": "tablet",
    "cap": "capsule",
    "caps": "capsule",
    "capsule": "capsule",
    "capsules": "capsule",
    "puff": "puff",
    "puffs": "puff",
    "drop": "drop",
    "drops": "drop",
    "gtt": "drop",
    "spray": "spray",
    "sprays": "spray",
    "patch": "patch",
    "patches": "patch",
    "supp": "suppository",
    "suppository": "suppository",
    "lozenge": "lozenge",
}

# Measurement units that follow a dose amount (not a form).
_DOSE_UNITS: frozenset[str] = frozenset(
    {"mg", "mcg", "g", "kg", "ml", "l", "unit", "units", "meq", "iu", "%"}
)

_DOSE_RE = re.compile(
    r"(?<!\w)(\d+(?:\.\d+)?)\s*([A-Za-z%]+)",
)
_ROUTE_RES: tuple[tuple[re.Pattern[str], str], ...] = tuple(
    (re.compile(rf"(?<!\w){re.escape(cue)}(?!\w)", re.IGNORECASE), route)
    for cue, route in _ROUTE_LEXICON
)
# PRN condition: the reason following PRN / "as needed".
_PRN_CONDITION_RE = re.compile(
    r"(?:prn|as needed)(?:\s+for)?\s+([a-z][a-z ]*?)(?:$|[.,;])",
    re.IGNORECASE,
)
# Candidate duration phrase to hand to normalize_duration.
_DURATION_RE = re.compile(
    r"(?:x|for)?\s*\d+\s*(?:day|days|week|weeks|wk|wks|month|months|mo|hour|hours|hr|hrs)\b",
    re.IGNORECASE,
)
_FREQUENCY_RANGE_RE = re.compile(
    r"(?<!\w)q\s*(\d+(?:\.\d+)?)\s*[-–]\s*(\d+(?:\.\d+)?)\s*h(?:ours?|rs?)?(?!\w)",
    re.IGNORECASE,
)


class Sig(TypedDict):
    """A structured medication sig.

    ``missing`` lists the components (``dose`` / ``route`` / ``frequency``) that
    could not be parsed, so partial sigs are explicit rather than silently
    incomplete.
    """

    raw: str
    dose: float | None
    unit: str | None
    form: str | None
    route: str | None
    frequency_per_day: float | None
    frequency_period: object
    frequency_period_unit: str | None
    as_needed: bool
    condition: str | None
    duration_days: object
    missing: list[str]


class SpanSig(TypedDict):
    """A :class:`Sig` attached to the medication span it was parsed from."""

    span: tuple[int, int]
    sig: Sig


def _parse_dose(text: str) -> tuple[float | None, str | None, str | None]:
    """Return ``(dose, unit, form)`` from the first ``<number> <token>`` pair."""

    for match in _DOSE_RE.finditer(text):
        amount = float(match.group(1))
        token = match.group(2).lower()
        if token in _FORM_LEXICON:
            form = _FORM_LEXICON[token]
            return amount, form, form
        if token in _DOSE_UNITS:
            return amount, "ml" if token == "ml" else token, None
    return None, None, None


def _parse_route(text: str) -> str | None:
    best: tuple[int, str] | None = None
    for pattern, route in _ROUTE_RES:
        match = pattern.search(text)
        if match is not None and (best is None or match.start() < best[0]):
            best = (match.start(), route)
    return best[1] if best is not None else None


def _infer_route_from_form(form: str | None) -> str | None:
    if form == "puff":
        return "inhaled"
    return None


def _windows(text: str) -> list[str]:
    """Whitespace unigram/bigram/trigram windows, longest first, in order."""

    tokens = text.split()
    windows: list[str] = []
    for size in (3, 2, 1):
        for i in range(len(tokens) - size + 1):
            windows.append(" ".join(tokens[i : i + size]))
    return windows


def _range_frequency(text: str) -> tuple[float | None, object, str | None]:
    """Normalize q4-6h-style ranges to the shortest deterministic interval."""

    match = _FREQUENCY_RANGE_RE.search(text)
    if match is None:
        return None, None, None
    lower = float(match.group(1))
    upper = float(match.group(2))
    interval = min(lower, upper)
    if interval <= 0:
        return None, None, None
    result = normalize_frequency(f"q{interval:g}h")
    return (
        result["frequency_per_day"],
        result["period"],
        result["period_unit"],
    )


def _parse_frequency(text: str) -> tuple[float | None, object, str | None, bool]:
    """Delegate frequency normalization to the shipped helper.

    Isolates candidate windows and hands each to ``normalize_frequency`` so this
    module never re-implements the frequency lexicon.
    """

    per_day, period, period_unit = _range_frequency(text)
    as_needed = False
    for window in _windows(text):
        result = normalize_frequency(window)
        if not result["recognized"]:
            continue
        if result["as_needed"]:
            as_needed = True
        if per_day is None and result["frequency_per_day"] is not None:
            per_day = result["frequency_per_day"]
            period = result["period"]
            period_unit = result["period_unit"]
    return per_day, period, period_unit, as_needed


def _parse_duration(text: str) -> object:
    match = _DURATION_RE.search(text)
    if match is None:
        return None
    return normalize_duration(match.group(0))["days"]


def _parse_condition(text: str) -> str | None:
    match = _PRN_CONDITION_RE.search(text)
    if match is None:
        return None
    condition = match.group(1).strip()
    return condition or None


def parse_sig(text: str) -> Sig:
    """Parse a medication sig string into a structured :class:`Sig`."""

    raw = str(text)
    dose, unit, form = _parse_dose(raw)
    route = _parse_route(raw) or _infer_route_from_form(form)
    per_day, period, period_unit, as_needed = _parse_frequency(raw)
    condition = _parse_condition(raw)
    duration_days = _parse_duration(raw)

    missing: list[str] = []
    if dose is None:
        missing.append("dose")
    if route is None:
        missing.append("route")
    if per_day is None and not as_needed:
        missing.append("frequency")

    return Sig(
        raw=raw,
        dose=dose,
        unit=unit,
        form=form,
        route=route,
        frequency_per_day=per_day,
        frequency_period=period,
        frequency_period_unit=period_unit,
        as_needed=as_needed,
        condition=condition,
        duration_days=duration_days,
        missing=missing,
    )


def parse_sigs(
    text: str,
    spans: Iterable[Mapping[str, object]],
) -> list[SpanSig]:
    """Parse the sig covered by each medication span.

    Each span's covered substring is parsed independently; the returned
    ``SpanSig`` carries the span offset so results map back to the source.
    """

    results: list[SpanSig] = []
    for span in spans:
        try:
            start = int(span["start"])  # type: ignore[arg-type]
            end = int(span["end"])  # type: ignore[arg-type]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("spans require integer 'start' and 'end'") from exc
        results.append(SpanSig(span=(start, end), sig=parse_sig(text[start:end])))
    return results


__all__ = [
    "SIG_PARSER_ADVISORY",
    "Sig",
    "SpanSig",
    "parse_sig",
    "parse_sigs",
]
