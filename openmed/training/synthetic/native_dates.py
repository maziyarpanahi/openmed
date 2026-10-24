"""Native-digit and native-calendar date variants for synthetic locale PHI.

:class:`~openmed.training.synthetic.locale_phi.LocalePhiGenerator` renders every
date as an ISO Gregorian string in ASCII digits unless a caller opts in to a
profile from this module. A :class:`NativeDateProfile` fixes one language's
calendar, numeral system, numeric or month-name form, and separators, so
augmentation data can contain dates as clinicians write them: Solar Hijri dates
in Persian digits, Buddhist-era dates in Thai digits, Ethiopian dates, Lunar
Hijri dates with an era marker, or Bikram Sambat dates in Devanagari digits.

Dates are sampled directly in the target calendar from its month-length rules;
nothing here converts a date between calendars. Where a month's real length
depends on an astronomical observation or a published lookup table (the Solar
Hijri leap day, every Lunar Hijri month, every Bikram Sambat month), the
sampler draws only days that exist in that month in every year, so a rendered
date is valid without the table.

Span metadata built here is value-free. It names the calendar, numeral system,
form, profile, and any known deterministic-detection gap, never the rendered
text or its year, month, or day. A profile without a gap renders spans that the
language's existing deterministic date patterns return exactly; a profile with
a gap renders forms those patterns miss, which is what makes them useful as
augmentation and as detection stress data.
"""

from __future__ import annotations

import calendar as gregorian_calendar
import random
import string
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Final, Mapping

GREGORIAN_CALENDAR: Final = "gregorian"
THAI_BUDDHIST_CALENDAR: Final = "thai_buddhist"
SOLAR_HIJRI_CALENDAR: Final = "solar_hijri"
ETHIOPIAN_CALENDAR: Final = "ethiopian"
HIJRI_CALENDAR: Final = "hijri"
BIKRAM_SAMBAT_CALENDAR: Final = "bikram_sambat"

DATE_CALENDARS: Final[frozenset[str]] = frozenset(
    {
        GREGORIAN_CALENDAR,
        THAI_BUDDHIST_CALENDAR,
        SOLAR_HIJRI_CALENDAR,
        ETHIOPIAN_CALENDAR,
        HIJRI_CALENDAR,
        BIKRAM_SAMBAT_CALENDAR,
    }
)
"""Value-free calendar codes recorded in ``calendar`` span metadata."""

# Numeral systems use CLDR numbering-system identifiers. ``ethi`` is the
# additive Ethiopic system (no zero); the others are positional decimal digits.
_DECIMAL_DIGITS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "latn": "0123456789",
        "arab": "٠١٢٣٤٥٦٧٨٩",
        "arabext": "۰۱۲۳۴۵۶۷۸۹",
        "thai": "๐๑๒๓๔๕๖๗๘๙",
        "deva": "०१२३४५६७८९",
    }
)
_DIGIT_TABLES: Final[Mapping[str, dict[int, int]]] = MappingProxyType(
    {
        system: str.maketrans("0123456789", digits)
        for system, digits in _DECIMAL_DIGITS.items()
    }
)
_ETHIOPIC_NUMERALS: Final = "ethi"

NUMERAL_SYSTEMS: Final[frozenset[str]] = frozenset(
    {*_DECIMAL_DIGITS, _ETHIOPIC_NUMERALS}
)
"""CLDR numbering-system identifiers recorded in ``numeral_system`` metadata."""

NUMERIC_FORM: Final = "numeric"
MONTH_NAME_FORM: Final = "month_name"
DATE_FORMS: Final[frozenset[str]] = frozenset({NUMERIC_FORM, MONTH_NAME_FORM})
"""Rendering forms recorded in ``form`` span metadata."""

MONTH_NAME_NOT_MATCHED: Final = "month_name_not_matched"
"""No deterministic date pattern for the language knows these month names."""

ERA_MARKER_NOT_MATCHED: Final = "era_marker_not_matched"
"""Date detection returns the numeric date without the era marker that the gold
span keeps, either because no pattern covers the marker or because a shorter
numeric match outranks the one that does."""

GREGORIAN_FALLBACK_ONLY: Final = "gregorian_fallback_only"
"""Only the Gregorian numeric fallback can match, and its validator rejects a
native date whose numbers are not also a valid Gregorian date."""

DATE_DETECTION_GAPS: Final[frozenset[str]] = frozenset(
    {MONTH_NAME_NOT_MATCHED, ERA_MARKER_NOT_MATCHED, GREGORIAN_FALLBACK_ONLY}
)
"""Value-free codes for date forms the deterministic patterns do not match."""

DATE_FIELDS: Final[tuple[str, ...]] = ("date_of_birth", "date")
"""Locale PHI fields that a native date profile renders."""

# Sampled years, in each calendar's own era, for a date of birth and for a
# recent follow-up date. Both ranges map to roughly 1930-2005 and 2015-2025
# in the Gregorian calendar, so a follow-up never precedes the birth date.
_YEAR_RANGES: Final[Mapping[str, Mapping[str, tuple[int, int]]]] = MappingProxyType(
    {
        GREGORIAN_CALENDAR: {"date_of_birth": (1930, 2005), "date": (2015, 2025)},
        THAI_BUDDHIST_CALENDAR: {
            "date_of_birth": (2473, 2548),
            "date": (2558, 2568),
        },
        SOLAR_HIJRI_CALENDAR: {"date_of_birth": (1309, 1384), "date": (1394, 1404)},
        ETHIOPIAN_CALENDAR: {"date_of_birth": (1922, 1997), "date": (2007, 2017)},
        HIJRI_CALENDAR: {"date_of_birth": (1348, 1425), "date": (1436, 1446)},
        BIKRAM_SAMBAT_CALENDAR: {
            "date_of_birth": (1987, 2062),
            "date": (2072, 2082),
        },
    }
)

_THAI_BUDDHIST_ERA_OFFSET: Final = 543
_LOOKUP_MONTH_MIN_DAYS: Final = 29

_MONTH_NAMES: Final[Mapping[tuple[str, str], tuple[str, ...]]] = MappingProxyType(
    {
        (SOLAR_HIJRI_CALENDAR, "fa"): (
            "فروردین",
            "اردیبهشت",
            "خرداد",
            "تیر",
            "مرداد",
            "شهریور",
            "مهر",
            "آبان",
            "آذر",
            "دی",
            "بهمن",
            "اسفند",
        ),
        (THAI_BUDDHIST_CALENDAR, "th"): (
            "มกราคม",
            "กุมภาพันธ์",
            "มีนาคม",
            "เมษายน",
            "พฤษภาคม",
            "มิถุนายน",
            "กรกฎาคม",
            "สิงหาคม",
            "กันยายน",
            "ตุลาคม",
            "พฤศจิกายน",
            "ธันวาคม",
        ),
        (ETHIOPIAN_CALENDAR, "am"): (
            "መስከረም",
            "ጥቅምት",
            "ኅዳር",
            "ታኅሣሥ",
            "ጥር",
            "የካቲት",
            "መጋቢት",
            "ሚያዝያ",
            "ግንቦት",
            "ሰኔ",
            "ሐምሌ",
            "ነሐሴ",
            "ጳጉሜ",
        ),
        (HIJRI_CALENDAR, "ar"): (
            "محرم",
            "صفر",
            "ربيع الأول",
            "ربيع الآخر",
            "جمادى الأولى",
            "جمادى الآخرة",
            "رجب",
            "شعبان",
            "رمضان",
            "شوال",
            "ذو القعدة",
            "ذو الحجة",
        ),
        (HIJRI_CALENDAR, "ur"): (
            "محرم",
            "صفر",
            "ربیع الاول",
            "ربیع الثانی",
            "جمادی الاول",
            "جمادی الثانی",
            "رجب",
            "شعبان",
            "رمضان",
            "شوال",
            "ذوالقعدہ",
            "ذوالحجہ",
        ),
        (BIKRAM_SAMBAT_CALENDAR, "ne"): (
            "बैशाख",
            "जेठ",
            "असार",
            "साउन",
            "भदौ",
            "असोज",
            "कात्तिक",
            "मंसिर",
            "पुस",
            "माघ",
            "फागुन",
            "चैत",
        ),
    }
)

_ETHIOPIC_ONES: Final = "፩፪፫፬፭፮፯፰፱"
_ETHIOPIC_TENS: Final = "፲፳፴፵፶፷፸፹፺"
_ETHIOPIC_HUNDRED: Final = "፻"


@dataclass(frozen=True)
class NativeDateProfile:
    """One opt-in rendering of synthetic dates for one language.

    Attributes:
        name: Stable profile identifier recorded in span metadata.
        language: OpenMed language code whose examples use the profile.
        calendar: Calendar the date is sampled in, from :data:`DATE_CALENDARS`.
        numeral_system: CLDR numbering-system identifier for every number,
            from :data:`NUMERAL_SYSTEMS`.
        form: ``"numeric"`` or ``"month_name"``.
        template: ``str.format`` template with ``{day}``, ``{year}``, and
            ``{month}`` (numeric form) or ``{month_name}`` (month-name form).
            Component order, separators, and any era marker live in the
            template, and the whole rendered template is the gold span.
        zero_pad: Pad a numeric day and month to two digits. Ethiopic numerals
            have no zero and are never padded.
        detection_gap: ``None`` when the language's deterministic date
            patterns (``get_patterns_for_language(language, locale)`` run
            through ``safety_sweep``) return exactly the rendered span,
            otherwise a code from :data:`DATE_DETECTION_GAPS` naming what they
            miss. A gap is a known limitation of detection, not of the data.
    """

    name: str
    language: str
    calendar: str
    numeral_system: str
    form: str
    template: str
    zero_pad: bool = True
    detection_gap: str | None = None

    def __post_init__(self) -> None:
        if self.calendar not in DATE_CALENDARS:
            raise ValueError(f"unknown date calendar {self.calendar!r}")
        if self.numeral_system not in NUMERAL_SYSTEMS:
            raise ValueError(f"unknown numeral system {self.numeral_system!r}")
        if self.form not in DATE_FORMS:
            raise ValueError(f"unknown date form {self.form!r}")
        if (
            self.detection_gap is not None
            and self.detection_gap not in DATE_DETECTION_GAPS
        ):
            raise ValueError(f"unknown date detection gap {self.detection_gap!r}")
        month_field = "month_name" if self.form == MONTH_NAME_FORM else "month"
        fields = {
            field_name
            for _, field_name, _, _ in string.Formatter().parse(self.template)
            if field_name is not None
        }
        if fields != {"day", month_field, "year"}:
            raise ValueError(
                f"date profile {self.name!r} template must use exactly "
                f"{{day}}, {{{month_field}}} and {{year}}"
            )
        if (
            self.form == MONTH_NAME_FORM
            and (self.calendar, self.language) not in _MONTH_NAMES
        ):
            raise ValueError(
                f"no {self.calendar!r} month names for language {self.language!r}"
            )

    def span_metadata(self) -> dict[str, Any]:
        """Return value-free metadata for a date span rendered by this profile.

        Returns:
            Calendar, numeral system, form, profile name, and detection gap.
            The rendered text and its date components are never included.
        """

        return {
            "calendar": self.calendar,
            "date_profile": self.name,
            "detection_gap": self.detection_gap,
            "form": self.form,
            "numeral_system": self.numeral_system,
        }


_PROFILES: Final[tuple[NativeDateProfile, ...]] = (
    NativeDateProfile(
        name="fa_solar_hijri_numeric_arabext",
        language="fa",
        calendar=SOLAR_HIJRI_CALENDAR,
        numeral_system="arabext",
        form=NUMERIC_FORM,
        template="{year}/{month}/{day}",
    ),
    NativeDateProfile(
        name="fa_solar_hijri_numeric_latn",
        language="fa",
        calendar=SOLAR_HIJRI_CALENDAR,
        numeral_system="latn",
        form=NUMERIC_FORM,
        template="{year}/{month}/{day}",
    ),
    NativeDateProfile(
        name="fa_solar_hijri_month_name",
        language="fa",
        calendar=SOLAR_HIJRI_CALENDAR,
        numeral_system="arabext",
        form=MONTH_NAME_FORM,
        template="{day} {month_name} {year}",
        zero_pad=False,
    ),
    NativeDateProfile(
        name="th_thai_buddhist_numeric_thai",
        language="th",
        calendar=THAI_BUDDHIST_CALENDAR,
        numeral_system="thai",
        form=NUMERIC_FORM,
        template="{day}/{month}/{year}",
    ),
    NativeDateProfile(
        name="th_thai_buddhist_numeric_latn",
        language="th",
        calendar=THAI_BUDDHIST_CALENDAR,
        numeral_system="latn",
        form=NUMERIC_FORM,
        template="{day}/{month}/{year}",
    ),
    NativeDateProfile(
        name="th_thai_buddhist_month_name",
        language="th",
        calendar=THAI_BUDDHIST_CALENDAR,
        numeral_system="thai",
        form=MONTH_NAME_FORM,
        template="{day} {month_name} พ.ศ. {year}",
        zero_pad=False,
    ),
    NativeDateProfile(
        name="am_ethiopian_numeric_latn",
        language="am",
        calendar=ETHIOPIAN_CALENDAR,
        numeral_system="latn",
        form=NUMERIC_FORM,
        template="{day}/{month}/{year}",
    ),
    NativeDateProfile(
        name="am_ethiopian_numeric_ethi",
        language="am",
        calendar=ETHIOPIAN_CALENDAR,
        numeral_system="ethi",
        form=NUMERIC_FORM,
        template="{day}/{month}/{year}",
    ),
    NativeDateProfile(
        name="am_ethiopian_month_name",
        language="am",
        calendar=ETHIOPIAN_CALENDAR,
        numeral_system="latn",
        form=MONTH_NAME_FORM,
        template="{day} {month_name} {year}",
        zero_pad=False,
        detection_gap=MONTH_NAME_NOT_MATCHED,
    ),
    NativeDateProfile(
        name="ar_hijri_numeric_arab",
        language="ar",
        calendar=HIJRI_CALENDAR,
        numeral_system="arab",
        form=NUMERIC_FORM,
        template="{day}/{month}/{year} هـ",
        detection_gap=ERA_MARKER_NOT_MATCHED,
    ),
    NativeDateProfile(
        name="ar_hijri_month_name",
        language="ar",
        calendar=HIJRI_CALENDAR,
        numeral_system="arab",
        form=MONTH_NAME_FORM,
        template="{day} {month_name} {year} هـ",
        zero_pad=False,
        detection_gap=MONTH_NAME_NOT_MATCHED,
    ),
    NativeDateProfile(
        name="ar_gregorian_numeric_arab",
        language="ar",
        calendar=GREGORIAN_CALENDAR,
        numeral_system="arab",
        form=NUMERIC_FORM,
        template="{day}/{month}/{year}",
    ),
    NativeDateProfile(
        name="ur_hijri_numeric_arabext",
        language="ur",
        calendar=HIJRI_CALENDAR,
        numeral_system="arabext",
        form=NUMERIC_FORM,
        template="{day}/{month}/{year} ہجری",
        detection_gap=ERA_MARKER_NOT_MATCHED,
    ),
    NativeDateProfile(
        name="ur_hijri_month_name",
        language="ur",
        calendar=HIJRI_CALENDAR,
        numeral_system="arabext",
        form=MONTH_NAME_FORM,
        template="{day} {month_name} {year} ہجری",
        zero_pad=False,
        detection_gap=MONTH_NAME_NOT_MATCHED,
    ),
    NativeDateProfile(
        name="ur_gregorian_numeric_arabext",
        language="ur",
        calendar=GREGORIAN_CALENDAR,
        numeral_system="arabext",
        form=NUMERIC_FORM,
        template="{day}/{month}/{year}",
    ),
    NativeDateProfile(
        name="ne_bikram_sambat_numeric_deva",
        language="ne",
        calendar=BIKRAM_SAMBAT_CALENDAR,
        numeral_system="deva",
        form=NUMERIC_FORM,
        template="{year}/{month}/{day}",
        detection_gap=GREGORIAN_FALLBACK_ONLY,
    ),
    NativeDateProfile(
        name="ne_bikram_sambat_month_name",
        language="ne",
        calendar=BIKRAM_SAMBAT_CALENDAR,
        numeral_system="deva",
        form=MONTH_NAME_FORM,
        template="{day} {month_name} {year}",
        zero_pad=False,
        detection_gap=MONTH_NAME_NOT_MATCHED,
    ),
)

NATIVE_DATE_PROFILES: Final[Mapping[str, NativeDateProfile]] = MappingProxyType(
    {profile.name: profile for profile in _PROFILES}
)
"""Every built-in native date profile, keyed by profile name."""

DEFAULT_NATIVE_DATE_PROFILES: Final[Mapping[str, str]] = MappingProxyType(
    {
        "am": "am_ethiopian_numeric_latn",
        "ar": "ar_hijri_numeric_arab",
        "fa": "fa_solar_hijri_numeric_arabext",
        "ne": "ne_bikram_sambat_numeric_deva",
        "th": "th_thai_buddhist_numeric_thai",
        "ur": "ur_hijri_numeric_arabext",
    }
)
"""One native-calendar profile per language, for ``date_profiles=``.

``ne`` takes effect only on a generator route that renders Nepali notes.
"""


def native_date_profiles_for(language: str) -> tuple[NativeDateProfile, ...]:
    """Return the built-in profiles for ``language`` in registry order.

    Args:
        language: OpenMed language code.

    Returns:
        Matching profiles; empty when the language has none.
    """

    return tuple(profile for profile in _PROFILES if profile.language == language)


def resolve_native_date_profiles(
    profiles: Mapping[str, str | NativeDateProfile] | None,
) -> dict[str, NativeDateProfile]:
    """Validate a ``{language: profile}`` selection.

    Args:
        profiles: ``None`` for the default ISO Gregorian rendering, or a mapping
            from language code to a :data:`NATIVE_DATE_PROFILES` name or a
            :class:`NativeDateProfile`.

    Returns:
        A new mapping from language code to profile; empty for ``None``.

    Raises:
        TypeError: If ``profiles`` is not a mapping, or a value is neither a
            profile name nor a :class:`NativeDateProfile`.
        ValueError: If a name is unknown or a profile belongs to a different
            language than its key.
    """

    if profiles is None:
        return {}
    if not isinstance(profiles, Mapping):
        raise TypeError("date_profiles must map language codes to date profiles")
    resolved: dict[str, NativeDateProfile] = {}
    for language, selection in profiles.items():
        if isinstance(selection, NativeDateProfile):
            profile = selection
        elif isinstance(selection, str):
            try:
                profile = NATIVE_DATE_PROFILES[selection]
            except KeyError as exc:
                raise ValueError(
                    f"unknown native date profile {selection!r}; "
                    f"available={sorted(NATIVE_DATE_PROFILES)!r}"
                ) from exc
        else:
            raise TypeError(
                "date_profiles values must be profile names or NativeDateProfile"
            )
        if profile.language != language:
            raise ValueError(
                f"date profile {profile.name!r} renders {profile.language!r}, "
                f"not {language!r}"
            )
        resolved[language] = profile
    return resolved


def native_month_names(calendar: str, language: str) -> tuple[str, ...]:
    """Return the month names a month-name profile renders, in calendar order.

    Args:
        calendar: Calendar code from :data:`DATE_CALENDARS`.
        language: OpenMed language code.

    Returns:
        One spelling per month, starting with month 1.

    Raises:
        ValueError: If no month names are registered for the pair.
    """

    try:
        return _MONTH_NAMES[(calendar, language)]
    except KeyError as exc:
        raise ValueError(
            f"no {calendar!r} month names for language {language!r}"
        ) from exc


def months_in_year(calendar: str) -> int:
    """Return the number of months in a ``calendar`` year.

    Args:
        calendar: Calendar code from :data:`DATE_CALENDARS`.

    Returns:
        13 for the Ethiopian calendar (Pagume is the 13th month), else 12.

    Raises:
        ValueError: If ``calendar`` is unknown.
    """

    _require_calendar(calendar)
    return 13 if calendar == ETHIOPIAN_CALENDAR else 12


def sampled_month_length(calendar: str, year: int, month: int) -> int:
    """Return how many days of ``month`` the sampler may draw.

    Gregorian months are exact, and so are Thai Buddhist-era months because
    the B.E. year minus 543 is the Gregorian year. Ethiopian months 1-12 have
    30 days; Pagume has 6 days when ``year % 4 == 3`` and 5 otherwise. Solar
    Hijri months 1-6 have 31 days and months 7-11 have 30. Esfand's 30th day
    depends on the vernal equinox, every Lunar Hijri month on the moon, and
    every Bikram Sambat month on a published table, so those months return
    29, a length each of them always reaches.

    Args:
        calendar: Calendar code from :data:`DATE_CALENDARS`.
        year: Year in the calendar's own era.
        month: Month number, starting at 1.

    Returns:
        The number of days from 1 that are valid in every such month.

    Raises:
        ValueError: If the calendar is unknown, the month is out of range, or
            a Buddhist-era year precedes Gregorian year 1.
    """

    if not 1 <= month <= months_in_year(calendar):
        raise ValueError(f"month is outside the {calendar} calendar's year")
    if calendar == GREGORIAN_CALENDAR:
        return gregorian_calendar.monthrange(year, month)[1]
    if calendar == THAI_BUDDHIST_CALENDAR:
        gregorian_year = year - _THAI_BUDDHIST_ERA_OFFSET
        if gregorian_year < 1:
            raise ValueError("year is before the thai_buddhist calendar's range")
        return gregorian_calendar.monthrange(gregorian_year, month)[1]
    if calendar == ETHIOPIAN_CALENDAR:
        if month <= 12:
            return 30
        return 6 if year % 4 == 3 else 5
    if calendar == SOLAR_HIJRI_CALENDAR:
        if month <= 6:
            return 31
        if month <= 11:
            return 30
    return _LOOKUP_MONTH_MIN_DAYS


def sample_native_date(
    calendar: str,
    rng: random.Random,
    *,
    field: str = "date",
) -> tuple[int, int, int]:
    """Sample a valid ``(year, month, day)`` directly in ``calendar``.

    The year is drawn from the field's range in the calendar's own era, then
    the day is drawn uniformly over the days :func:`sampled_month_length`
    allows in that year.

    Args:
        calendar: Calendar code from :data:`DATE_CALENDARS`.
        rng: Seeded random source; the draw is deterministic for its state.
        field: ``"date_of_birth"`` for a past birth date, or ``"date"`` for a
            recent follow-up date.

    Returns:
        A ``(year, month, day)`` tuple in ``calendar``.

    Raises:
        ValueError: If the calendar or field is unknown.
    """

    _require_calendar(calendar)
    if field not in DATE_FIELDS:
        raise ValueError(f"unknown date field {field!r}; expected {DATE_FIELDS!r}")
    first_year, last_year = _YEAR_RANGES[calendar][field]
    year = rng.randint(first_year, last_year)
    lengths = [
        sampled_month_length(calendar, year, month)
        for month in range(1, months_in_year(calendar) + 1)
    ]
    ordinal = rng.randrange(sum(lengths))
    for month, length in enumerate(lengths, start=1):
        if ordinal < length:
            return year, month, ordinal + 1
        ordinal -= length
    raise AssertionError("ordinal exceeded the sampled year length")  # pragma: no cover


def ethiopic_numeral(value: int) -> str:
    """Render ``value`` in additive Ethiopic (Ge'ez) numerals.

    Args:
        value: Integer from 1 to 9999.

    Returns:
        The numeral, for example ``"፲፪"`` for 12 or ``"፳፻፲፮"`` for 2016.

    Raises:
        ValueError: If ``value`` is outside 1-9999.
    """

    if not 1 <= value <= 9999:
        raise ValueError("Ethiopic numerals are rendered for 1-9999 only")
    hundreds, rest = divmod(value, 100)
    prefix = ""
    if hundreds:
        multiplier = "" if hundreds == 1 else _ethiopic_below_hundred(hundreds)
        prefix = multiplier + _ETHIOPIC_HUNDRED
    return prefix + _ethiopic_below_hundred(rest)


def render_native_date(
    profile: NativeDateProfile, year: int, month: int, day: int
) -> str:
    """Render one date, given in the profile's calendar, with ``profile``.

    Args:
        profile: Rendering profile.
        year: Year in the profile calendar's own era.
        month: Month number, starting at 1.
        day: Day of the month, starting at 1.

    Returns:
        The rendered date. Every character belongs to the gold span.

    Raises:
        ValueError: If the date cannot exist in the profile's calendar. The
            message never contains the date.
    """

    if year < 1 or not 1 <= day <= _max_month_length(profile.calendar, year, month):
        raise ValueError(f"date is not valid in the {profile.calendar} calendar")
    pad = 2 if profile.zero_pad and profile.form == NUMERIC_FORM else 0
    fields = {
        "day": _render_number(day, profile.numeral_system, width=pad),
        "year": _render_number(year, profile.numeral_system, width=0),
    }
    if profile.form == MONTH_NAME_FORM:
        names = native_month_names(profile.calendar, profile.language)
        fields["month_name"] = names[month - 1]
    else:
        fields["month"] = _render_number(month, profile.numeral_system, width=pad)
    return profile.template.format(**fields)


def generate_native_date(
    profile: NativeDateProfile,
    rng: random.Random,
    *,
    field: str = "date",
) -> tuple[str, dict[str, Any]]:
    """Sample and render one date for a locale PHI field.

    Args:
        profile: Rendering profile.
        rng: Seeded random source.
        field: ``"date_of_birth"`` or ``"date"``.

    Returns:
        The rendered date and its value-free span metadata.
    """

    year, month, day = sample_native_date(profile.calendar, rng, field=field)
    return render_native_date(profile, year, month, day), profile.span_metadata()


def _require_calendar(calendar: str) -> None:
    if calendar not in DATE_CALENDARS:
        raise ValueError(f"unknown date calendar {calendar!r}")


def _max_month_length(calendar: str, year: int, month: int) -> int:
    """Return the longest a month can be, including table-dependent days."""

    length = sampled_month_length(calendar, year, month)
    if calendar == SOLAR_HIJRI_CALENDAR and month == 12:
        return 30
    if calendar == HIJRI_CALENDAR:
        return 30
    if calendar == BIKRAM_SAMBAT_CALENDAR:
        return 32
    return length


def _render_number(value: int, numeral_system: str, *, width: int) -> str:
    if numeral_system == _ETHIOPIC_NUMERALS:
        return ethiopic_numeral(value)
    return f"{value:0{width}d}".translate(_DIGIT_TABLES[numeral_system])


def _ethiopic_below_hundred(value: int) -> str:
    tens, ones = divmod(value, 10)
    return (_ETHIOPIC_TENS[tens - 1] if tens else "") + (
        _ETHIOPIC_ONES[ones - 1] if ones else ""
    )


__all__ = [
    "BIKRAM_SAMBAT_CALENDAR",
    "DATE_CALENDARS",
    "DATE_DETECTION_GAPS",
    "DATE_FIELDS",
    "DATE_FORMS",
    "DEFAULT_NATIVE_DATE_PROFILES",
    "ERA_MARKER_NOT_MATCHED",
    "ETHIOPIAN_CALENDAR",
    "GREGORIAN_CALENDAR",
    "GREGORIAN_FALLBACK_ONLY",
    "HIJRI_CALENDAR",
    "MONTH_NAME_FORM",
    "MONTH_NAME_NOT_MATCHED",
    "NATIVE_DATE_PROFILES",
    "NUMERAL_SYSTEMS",
    "NUMERIC_FORM",
    "SOLAR_HIJRI_CALENDAR",
    "THAI_BUDDHIST_CALENDAR",
    "NativeDateProfile",
    "ethiopic_numeral",
    "generate_native_date",
    "months_in_year",
    "native_date_profiles_for",
    "native_month_names",
    "render_native_date",
    "resolve_native_date_profiles",
    "sample_native_date",
    "sampled_month_length",
]
