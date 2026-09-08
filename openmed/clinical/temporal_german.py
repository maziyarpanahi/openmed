"""German temporal span parsing without translation or an implicit clock."""

from __future__ import annotations

import calendar
import re
from datetime import date, timedelta

_MONTHS = {
    name: index
    for index, name in enumerate(
        (
            "januar",
            "februar",
            "märz",
            "april",
            "mai",
            "juni",
            "juli",
            "august",
            "september",
            "oktober",
            "november",
            "dezember",
        ),
        1,
    )
}
_MONTHS["maerz"] = 3
_NUMBER = {
    "ein": 1,
    "eine": 1,
    "einem": 1,
    "einen": 1,
    "einer": 1,
    "zwei": 2,
    "drei": 3,
    "vier": 4,
    "fünf": 5,
    "fuenf": 5,
    "sechs": 6,
    "sieben": 7,
    "acht": 8,
    "neun": 9,
    "zehn": 10,
}
_NUMBER_RE = r"(?:\d{1,4}|" + "|".join(sorted(_NUMBER, key=len, reverse=True)) + ")"
_UNIT_RE = r"(?:Tage[n]?|Tag|Woche[n]?|Monat(?:e[n]?)?|Jahr(?:e[n]?)?)"
_MONTH_RE = "(?:" + "|".join(_MONTHS) + ")"
_NUMERIC = re.compile(
    r"(?P<day>\d{1,2})(?P<separator>[./])(?P<month>\d{1,2})(?P=separator)(?P<year>\d{2}|\d{4})"
)
_NAMED = re.compile(
    rf"(?P<day>\d{{1,2}})\.?\s+(?P<month>{_MONTH_RE})\s+(?P<year>\d{{4}})",
    re.IGNORECASE,
)
_RELATIVE = re.compile(
    rf"(?P<direction>vor|in|seit)\s+(?P<number>{_NUMBER_RE})\s+(?P<unit>{_UNIT_RE})",
    re.IGNORECASE,
)
_KEYWORDS = {
    "heute": 0,
    "gestern": -1,
    "vorgestern": -2,
    "morgen": 1,
    "übermorgen": 2,
    "uebermorgen": 2,
}

# A caller still decides whether an observed date refers to a clinical event,
# an identifier, or an unrelated clause. This regex performs no such inference.
GERMAN_TIMEX_RE = re.compile(
    rf"(?<![\w./-])(?:\d{{4}}-\d{{2}}-\d{{2}}|\d{{1,2}}[./]\d{{1,2}}[./](?:\d{{4}}|\d{{2}})|"
    rf"\d{{1,2}}\.?\s+{_MONTH_RE}\s+\d{{4}}|(?:vor|in|seit)\s+{_NUMBER_RE}\s+{_UNIT_RE}|"
    rf"{'|'.join(_KEYWORDS)})(?![\w/-]|\.\d)",
    re.IGNORECASE,
)


def _german_value(phrase: str, reference: date | None):
    """Return a normalized day or explicit ambiguity; never infer a date anchor."""
    text = " ".join(phrase.strip().split())
    lowered = text.casefold()
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        try:
            return date.fromisoformat(text).isoformat(), None, ("day",)
        except ValueError:
            return None, None, ("day", "invalid")
    numeric = _NUMERIC.fullmatch(text)
    named = _NAMED.fullmatch(text)
    if numeric or named:
        match = numeric or named
        if len(match["year"]) != 4:
            return None, None, ("day", "ambiguous")
        month = int(match["month"]) if numeric else _MONTHS[match["month"].casefold()]
        try:
            value = date(int(match["year"]), month, int(match["day"]))
        except ValueError:
            return None, None, ("day", "invalid")
        return value.isoformat(), None, ("day", "locale_dmy")
    if lowered in _KEYWORDS:
        if reference is None:
            return None, None, ("day", "unanchored")
        try:
            value = reference + timedelta(days=_KEYWORDS[lowered])
        except OverflowError:
            return None, reference.isoformat(), ("day", "invalid")
        return value.isoformat(), reference.isoformat(), ("day",)
    if match := _RELATIVE.fullmatch(text):
        number = match["number"].casefold()
        amount = int(number) if number.isdigit() else _NUMBER[number]
        unit = match["unit"].casefold()
        if reference is None:
            return None, None, ("day", "unanchored")
        direction = match["direction"].casefold()
        if direction in {"vor", "seit"}:
            amount = -amount
        try:
            if unit.startswith(("tag", "woche")):
                value = reference + timedelta(
                    days=amount * (7 if unit.startswith("woche") else 1)
                )
            else:
                months = amount * (12 if unit.startswith("jahr") else 1)
                offset = reference.year * 12 + reference.month - 1 + months
                year, month = divmod(offset, 12)
                month += 1
                value = date(
                    year, month, min(reference.day, calendar.monthrange(year, month)[1])
                )
        except (ValueError, OverflowError):
            return None, reference.isoformat(), ("day", "invalid")
        if direction == "seit":
            return (
                f"{value.isoformat()}/{reference.isoformat()}",
                reference.isoformat(),
                ("day", "interval"),
            )
        return value.isoformat(), reference.isoformat(), ("day",)
    return None, None, ("unknown", "unsupported")
