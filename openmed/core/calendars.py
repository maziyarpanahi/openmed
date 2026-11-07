"""Calendar converters keyed by one shared proleptic Gregorian day number.

Every converter maps a native ``(year, month, day)`` date to and from the
ordinal used by :meth:`datetime.date.toordinal`, where day 1 is 1 January of
year 1 in the proleptic Gregorian calendar. Shifting dates through that shared
day number keeps intervals identical whether a patient's dates are written in
the Gregorian calendar or in a native calendar.

Each converter declares the inclusive year range over which its arithmetic is
exact. Callers must treat dates outside that range as unsupported instead of
extrapolating. Converters are registered in one process-local registry so
other calendars, including table-driven ones, can be added without changing
the date-shift code that consumes them. The table-driven Umm al-Qura Hijri
(``hijri``) and Bikram Sambat (``bikram_sambat``) calendars come from
:mod:`openmed.core.calendar_tables` and are registered when this module is
imported.

The module is standard-library only. Errors never include the date values that
were passed in.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date
from threading import RLock
from typing import Final

from .errors import ConfigurationError, InputError

GREGORIAN: Final = "gregorian"
SOLAR_HIJRI: Final = "solar_hijri"
ETHIOPIAN: Final = "ethiopian"
THAI_BUDDHIST: Final = "thai_buddhist"
LUNAR_HIJRI: Final = "hijri"
BIKRAM_SAMBAT: Final = "bikram_sambat"

_GREGORIAN_MONTH_DAYS: Final = (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)


def _is_gregorian_leap(year: int) -> bool:
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


class CalendarConverter(ABC):
    """Convert one calendar's dates to and from a shared day number.

    Subclasses implement the unchecked arithmetic for a single calendar and
    declare the inclusive ``min_year``/``max_year`` range over which that
    arithmetic is exact. The public methods validate input against that range
    and the calendar's month lengths before converting.

    Attributes:
        code: Stable lowercase calendar code, for example ``"solar_hijri"``.
        min_year: First supported year in the calendar's own era, inclusive.
        max_year: Last supported year in the calendar's own era, inclusive.
    """

    code: str
    min_year: int
    max_year: int

    @abstractmethod
    def months_in_year(self, year: int) -> int:
        """Return the number of months in ``year``."""

    @abstractmethod
    def month_length(self, year: int, month: int) -> int:
        """Return the number of days in ``month`` of ``year``."""

    @abstractmethod
    def _day_number(self, year: int, month: int, day: int) -> int:
        """Return the day number of an already validated date."""

    @abstractmethod
    def _date_from_day_number(self, day_number: int) -> tuple[int, int, int]:
        """Return the date for a day number already inside the range."""

    def is_valid(self, year: int, month: int, day: int) -> bool:
        """Return whether a date exists in this calendar's supported range.

        Args:
            year: Year in the calendar's own era.
            month: One-based month number.
            day: One-based day of the month.

        Returns:
            ``True`` when the year is inside the supported range and the month
            and day exist in that year.
        """
        if not all(_is_int(value) for value in (year, month, day)):
            return False
        if not self.min_year <= year <= self.max_year:
            return False
        if not 1 <= month <= self.months_in_year(year):
            return False
        return 1 <= day <= self.month_length(year, month)

    def day_number_range(self) -> tuple[int, int]:
        """Return the first and last supported day numbers, inclusive."""
        first = self._day_number(self.min_year, 1, 1)
        last_month = self.months_in_year(self.max_year)
        last = self._day_number(
            self.max_year,
            last_month,
            self.month_length(self.max_year, last_month),
        )
        return first, last

    def to_day_number(self, year: int, month: int, day: int) -> int:
        """Convert a native date to the shared proleptic Gregorian ordinal.

        Args:
            year: Year in the calendar's own era.
            month: One-based month number.
            day: One-based day of the month.

        Returns:
            The day number used by :meth:`datetime.date.toordinal`.

        Raises:
            InputError: If the date is not valid in the supported range. The
                message names the calendar but never the date values.
        """
        if not self.is_valid(year, month, day):
            raise InputError(
                f"The date is not valid in the {self.code} calendar's supported "
                f"range ({self.min_year}-{self.max_year}). Pass a valid date "
                "inside that range.",
                details={"calendar": self.code, "reason": "invalid_date"},
            )
        return self._day_number(year, month, day)

    def from_day_number(self, day_number: int) -> tuple[int, int, int]:
        """Convert a shared day number to a native ``(year, month, day)``.

        Args:
            day_number: Proleptic Gregorian ordinal.

        Returns:
            The native ``(year, month, day)`` tuple.

        Raises:
            InputError: If the day number falls outside the supported range.
        """
        first, last = self.day_number_range()
        if not _is_int(day_number) or not first <= day_number <= last:
            raise InputError(
                f"The day number is outside the {self.code} calendar's supported "
                f"range ({self.min_year}-{self.max_year}).",
                details={"calendar": self.code, "reason": "out_of_range"},
            )
        return self._date_from_day_number(day_number)

    def shift(self, year: int, month: int, day: int, days: int) -> tuple[int, int, int]:
        """Return the native date ``days`` days after the given date.

        Args:
            year: Year in the calendar's own era.
            month: One-based month number.
            day: One-based day of the month.
            days: Signed number of days to add.

        Returns:
            The shifted native ``(year, month, day)`` tuple.

        Raises:
            InputError: If the source or shifted date is outside the range.
        """
        if not _is_int(days):
            raise InputError(
                "days must be an integer. Pass a whole number of days.",
                details={"argument": "days"},
            )
        return self.from_day_number(self.to_day_number(year, month, day) + days)


class GregorianCalendar(CalendarConverter):
    """Proleptic Gregorian calendar backed by :class:`datetime.date`."""

    code = GREGORIAN
    min_year = 1
    max_year = 9999

    def months_in_year(self, year: int) -> int:
        """Return twelve months."""
        return 12

    def month_length(self, year: int, month: int) -> int:
        """Return the Gregorian month length, including 29 February."""
        if month == 2 and _is_gregorian_leap(year):
            return 29
        return _GREGORIAN_MONTH_DAYS[month - 1]

    def _day_number(self, year: int, month: int, day: int) -> int:
        return date(year, month, day).toordinal()

    def _date_from_day_number(self, day_number: int) -> tuple[int, int, int]:
        value = date.fromordinal(day_number)
        return value.year, value.month, value.day


class ThaiBuddhistCalendar(CalendarConverter):
    """Thai solar calendar counted in the Buddhist era (B.E. = C.E. + 543).

    Months and month lengths are Gregorian. The supported range starts at
    B.E. 2484 (1941 C.E.), the first Thai year that began on 1 January;
    earlier years began on 1 April, so a fixed 543-year offset is not exact
    for their January to March dates.
    """

    code = THAI_BUDDHIST
    min_year = 2484
    max_year = 2665
    era_offset: Final = 543

    def months_in_year(self, year: int) -> int:
        """Return twelve months."""
        return 12

    def month_length(self, year: int, month: int) -> int:
        """Return the Gregorian length of the month in the matching C.E. year."""
        return _GREGORIAN.month_length(year - self.era_offset, month)

    def _day_number(self, year: int, month: int, day: int) -> int:
        return date(year - self.era_offset, month, day).toordinal()

    def _date_from_day_number(self, day_number: int) -> tuple[int, int, int]:
        value = date.fromordinal(day_number)
        return value.year + self.era_offset, value.month, value.day


class EthiopianCalendar(CalendarConverter):
    """Ethiopian calendar in the Amete Mihret (incarnation) era.

    The year has twelve 30-day months followed by Pagume, a thirteenth month
    of five days, or six days when ``year % 4 == 3``. The epoch, 1 Meskerem
    of year 1, is 29 August 8 C.E. in the Julian calendar (day number 2796).
    The arithmetic is exact for every year; the supported range is limited to
    years 1891-2114 (1898-2122 C.E.) so implausible years are not converted.
    """

    code = ETHIOPIAN
    min_year = 1891
    max_year = 2114
    _epoch: Final = 2796

    def months_in_year(self, year: int) -> int:
        """Return thirteen months, including Pagume."""
        return 13

    def month_length(self, year: int, month: int) -> int:
        """Return 30 days, or 5/6 days for Pagume depending on the leap year."""
        if month < 13:
            return 30
        return 6 if year % 4 == 3 else 5

    def _day_number(self, year: int, month: int, day: int) -> int:
        return self._epoch - 1 + 365 * (year - 1) + year // 4 + 30 * (month - 1) + day

    def _date_from_day_number(self, day_number: int) -> tuple[int, int, int]:
        year = (4 * (day_number - self._epoch) + 1463) // 1461
        month = (day_number - self._day_number(year, 1, 1)) // 30 + 1
        day = day_number + 1 - self._day_number(year, month, 1)
        return year, month, day


class SolarHijriCalendar(CalendarConverter):
    """Iranian Solar Hijri (Jalali) calendar using the 33-year arithmetic rule.

    The first six months have 31 days, the next five have 30 days, and Esfand
    has 29 days, or 30 in a leap year. A year is a leap year when
    ``year % 33`` is one of 1, 5, 9, 13, 17, 22, 26 or 30. Day numbers are
    anchored at 1 Farvardin 1404, which is 21 March 2025.

    The official calendar starts each year at the vernal equinox, so the
    arithmetic rule is only exact over a bounded span. The supported range,
    years 1277-1501 (1898-2123 C.E.), is the span where the rule agrees with
    astronomical reference computations of every year length.
    """

    code = SOLAR_HIJRI
    min_year = 1277
    max_year = 1501
    _leap_remainders: Final = frozenset({1, 5, 9, 13, 17, 22, 26, 30})
    _anchor_year: Final = 1404
    _anchor_day_number: Final = 739331  # date(2025, 3, 21).toordinal()
    _days_per_cycle: Final = 33 * 365 + 8

    def is_leap_year(self, year: int) -> bool:
        """Return whether Esfand has 30 days in ``year``."""
        return year % 33 in self._leap_remainders

    def months_in_year(self, year: int) -> int:
        """Return twelve months."""
        return 12

    def month_length(self, year: int, month: int) -> int:
        """Return 31, 30, or the leap-dependent length of Esfand."""
        if month <= 6:
            return 31
        if month <= 11:
            return 30
        return 30 if self.is_leap_year(year) else 29

    def _leap_years_before(self, year: int) -> int:
        remainder = year % 33
        partial = sum(1 for value in self._leap_remainders if value < remainder)
        return (year // 33) * 8 + partial

    def _year_start(self, year: int) -> int:
        return (
            self._anchor_day_number
            + 365 * (year - self._anchor_year)
            + self._leap_years_before(year)
            - self._leap_years_before(self._anchor_year)
        )

    @staticmethod
    def _days_before_month(month: int) -> int:
        if month <= 7:
            return 31 * (month - 1)
        return 30 * (month - 1) + 6

    def _day_number(self, year: int, month: int, day: int) -> int:
        return self._year_start(year) + self._days_before_month(month) + day - 1

    def _date_from_day_number(self, day_number: int) -> tuple[int, int, int]:
        offset = day_number - self._anchor_day_number
        year = self._anchor_year + (offset * 33) // self._days_per_cycle
        while self._year_start(year) > day_number:
            year -= 1
        while self._year_start(year + 1) <= day_number:
            year += 1
        day_of_year = day_number - self._year_start(year)
        if day_of_year < 186:
            month = day_of_year // 31 + 1
        else:
            month = (day_of_year - 6) // 30 + 1
        return year, month, day_of_year - self._days_before_month(month) + 1


_GREGORIAN: Final = GregorianCalendar()

_REGISTRY: dict[str, CalendarConverter] = {}
_REGISTRY_LOCK: Final = RLock()


def register_calendar(
    converter: CalendarConverter,
    *,
    replace: bool = False,
) -> CalendarConverter:
    """Register a calendar converter under its ``code``.

    Args:
        converter: Converter implementing :class:`CalendarConverter`.
        replace: Replace an existing converter with the same code when true.

    Returns:
        The registered converter.

    Raises:
        ConfigurationError: If ``converter`` is not a converter, declares an
            invalid code or range, or its code is already registered and
            replacement was not requested.
    """
    if not isinstance(converter, CalendarConverter):
        raise ConfigurationError(
            "converter must be a CalendarConverter. Subclass CalendarConverter "
            "before registering it.",
            details={"argument": "converter"},
        )
    code = getattr(converter, "code", None)
    if not is_calendar_code(code):
        raise ConfigurationError(
            "Calendar converters need a lowercase code such as 'solar_hijri'.",
            details={"argument": "code"},
        )
    min_year = getattr(converter, "min_year", None)
    max_year = getattr(converter, "max_year", None)
    if not (_is_int(min_year) and _is_int(max_year) and min_year <= max_year):
        raise ConfigurationError(
            "Calendar converters must declare an integer min_year <= max_year.",
            details={"calendar": code, "argument": "min_year"},
        )
    with _REGISTRY_LOCK:
        if code in _REGISTRY and not replace:
            raise ConfigurationError(
                f"Calendar {code!r} is already registered. Pass replace=True to "
                "replace it deliberately.",
                details={"calendar": code},
            )
        _REGISTRY[code] = converter
    return converter


def get_calendar(code: str | None) -> CalendarConverter | None:
    """Return the converter registered for ``code``, or ``None``."""
    if not isinstance(code, str):
        return None
    with _REGISTRY_LOCK:
        return _REGISTRY.get(code)


def calendar_codes() -> tuple[str, ...]:
    """Return a sorted snapshot of registered calendar codes."""
    with _REGISTRY_LOCK:
        return tuple(sorted(_REGISTRY))


def is_calendar_code(value: object) -> bool:
    """Return whether ``value`` is a well-formed lowercase calendar code."""
    return (
        isinstance(value, str)
        and bool(value)
        and value[0].isascii()
        and value[0].isalpha()
        and value.islower()
        and all(char.isascii() and (char.isalnum() or char == "_") for char in value)
    )


for _converter in (
    _GREGORIAN,
    SolarHijriCalendar(),
    EthiopianCalendar(),
    ThaiBuddhistCalendar(),
):
    register_calendar(_converter)

# The table-driven calendars register themselves on import. The import stays
# below the registry so calendar_tables can subclass CalendarConverter whichever
# of the two modules is imported first.
from . import calendar_tables as _calendar_tables  # noqa: E402,F401

__all__ = [
    "BIKRAM_SAMBAT",
    "ETHIOPIAN",
    "GREGORIAN",
    "LUNAR_HIJRI",
    "SOLAR_HIJRI",
    "THAI_BUDDHIST",
    "CalendarConverter",
    "EthiopianCalendar",
    "GregorianCalendar",
    "SolarHijriCalendar",
    "ThaiBuddhistCalendar",
    "calendar_codes",
    "get_calendar",
    "is_calendar_code",
    "register_calendar",
]
