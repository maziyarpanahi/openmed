"""Table-driven calendar converters loaded from verified month-length tables.

Lunar Hijri month starts follow a declared variant, such as the Umm al-Qura
calendar, and Bikram Sambat months last 29 to 32 days according to published
tables, so neither calendar can be computed with arithmetic. This module loads
a bounded month-length table from a versioned JSON file, verifies it offline,
and wraps it in a :class:`~openmed.core.calendars.CalendarConverter` so the
date shifter treats it like any arithmetic calendar. Dates outside the table's
years are unsupported and keep the date placeholder.

A table file is a JSON object with these keys:

- ``format`` (``"openmed-calendar-table"``) and ``format_version`` (``1``).
- ``calendar``: the registry code, for example ``"hijri"``.
- ``variant`` (optional): the declared variant, for example ``"umm_al_qura"``.
- ``version``: the table's own version string.
- ``license``: an SPDX identifier. Only permissive or public-domain licenses
  load unless the caller explicitly allows another license for a table it
  supplies itself.
- ``source``: a lineage record whose ``project`` names where the data came
  from; ``copyright``, ``license_text``, ``description`` and ``verification``
  are optional provenance fields.
- ``first_year`` and ``first_day``: the first year in the table and the ISO
  Gregorian date of its first day.
- ``month_lengths``: one list of month lengths per consecutive year.
- ``sha256``: the SHA-256 digest of the canonical JSON form of every other
  key (sorted keys, no insignificant whitespace, UTF-8).

The bundled Umm al-Qura and Bikram Sambat tables live in ``data/calendars``,
their digests are pinned below, and they are registered as the ``hijri`` and
``bikram_sambat`` calendars when :mod:`openmed.core.calendars` is imported.
Errors never include date values, table contents or file paths.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from bisect import bisect_right
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final, Mapping, Union

from .calendars import (
    BIKRAM_SAMBAT,
    LUNAR_HIJRI,
    CalendarConverter,
    is_calendar_code,
    register_calendar,
)
from .errors import ConfigurationError

CALENDAR_TABLE_FORMAT: Final = "openmed-calendar-table"
CALENDAR_TABLE_FORMAT_VERSION: Final = 1

PERMISSIVE_TABLE_LICENSES: Final = frozenset(
    {
        "0BSD",
        "Apache-2.0",
        "BSD-2-Clause",
        "BSD-3-Clause",
        "CC-BY-4.0",
        "CC0-1.0",
        "ISC",
        "MIT",
        "PDDL-1.0",
        "Unicode-3.0",
        "Unlicense",
        "Zlib",
    }
)
"""SPDX identifiers a table may declare without an explicit opt-in."""

_REQUIRED_KEYS: Final = frozenset(
    {
        "format",
        "format_version",
        "calendar",
        "version",
        "license",
        "source",
        "first_year",
        "first_day",
        "month_lengths",
        "sha256",
    }
)
_OPTIONAL_KEYS: Final = frozenset(
    {"variant", "description", "copyright", "license_text", "verification"}
)
_MAX_TABLE_BYTES: Final = 1024 * 1024
_MAX_YEARS: Final = 1000
_MAX_MONTHS_PER_YEAR: Final = 14
_MAX_MONTH_LENGTH: Final = 35
_SHA256_RE: Final = re.compile(r"^[0-9a-f]{64}$")

_DATA_DIR: Final = Path(__file__).with_name("data") / "calendars"

BUNDLED_CALENDAR_TABLES: Final[Mapping[str, tuple[str, str]]] = MappingProxyType(
    {
        LUNAR_HIJRI: (
            "hijri_umm_al_qura.json",
            "e6062ca4a759753f1d7ef6bbc0aa2593afc8cf53ea0cfd07d3cc79be91da0875",
        ),
        BIKRAM_SAMBAT: (
            "bikram_sambat.json",
            "694d412189f51e238b936da3ebe8ae353655b51110d89f37960dbf47b875a9fc",
        ),
    }
)
"""Bundled table file name and pinned SHA-256 digest per calendar code."""

TableSource = Union[str, "os.PathLike[str]", Mapping[str, Any]]


@dataclass(frozen=True)
class CalendarTable:
    """A verified month-length table and its license and lineage record.

    Attributes:
        calendar: Registry code the table is loaded under.
        variant: Declared calendar variant, for example ``"umm_al_qura"``.
        version: Version string of the table data.
        license: SPDX identifier declared by the table.
        source: Read-only lineage record; ``source["project"]`` names the
            upstream data.
        first_year: First year covered, in the calendar's own era.
        first_day_number: Shared day number of the first day of
            ``first_year`` (see :meth:`datetime.date.toordinal`).
        month_lengths: Month lengths for each consecutive year.
        sha256: Verified digest of the table's canonical JSON form.
    """

    calendar: str
    variant: str | None
    version: str
    license: str
    source: Mapping[str, Any]
    first_year: int
    first_day_number: int
    month_lengths: tuple[tuple[int, ...], ...]
    sha256: str

    @property
    def last_year(self) -> int:
        """Return the last year covered by the table."""
        return self.first_year + len(self.month_lengths) - 1


class TableCalendar(CalendarConverter):
    """Calendar converter backed by a bounded month-length table.

    The supported range is exactly the table's years; there is no
    extrapolation. Use :func:`load_calendar_table` to build one from a file
    and :func:`~openmed.core.calendars.register_calendar` to make it
    available to date shifting.

    Attributes:
        code: Registry code taken from the table.
        min_year: First year of the table.
        max_year: Last year of the table.
        variant: Declared variant taken from the table, if any.
        table: The verified :class:`CalendarTable`.
    """

    def __init__(self, table: CalendarTable) -> None:
        """Index a verified table for conversion.

        Args:
            table: Table returned by :func:`load_calendar_table` or
                :func:`verify_calendar_table`.
        """
        self.table = table
        self.code = table.calendar
        self.variant = table.variant
        self.min_year = table.first_year
        self.max_year = table.last_year
        month_starts: list[int] = []
        year_offsets: list[int] = []
        current = table.first_day_number
        for lengths in table.month_lengths:
            year_offsets.append(len(month_starts))
            for length in lengths:
                month_starts.append(current)
                current += length
        self._month_starts = tuple(month_starts)
        self._year_offsets = tuple(year_offsets)
        self._end = current

    def months_in_year(self, year: int) -> int:
        """Return the number of months the table lists for ``year``."""
        return len(self.table.month_lengths[year - self.min_year])

    def month_length(self, year: int, month: int) -> int:
        """Return the table's length of ``month`` in ``year``."""
        return self.table.month_lengths[year - self.min_year][month - 1]

    def _day_number(self, year: int, month: int, day: int) -> int:
        index = self._year_offsets[year - self.min_year] + month - 1
        return self._month_starts[index] + day - 1

    def _date_from_day_number(self, day_number: int) -> tuple[int, int, int]:
        index = bisect_right(self._month_starts, day_number) - 1
        year_index = bisect_right(self._year_offsets, index) - 1
        month = index - self._year_offsets[year_index] + 1
        day = day_number - self._month_starts[index] + 1
        return self.min_year + year_index, month, day


def calendar_table_digest(document: Mapping[str, Any]) -> str:
    """Return the SHA-256 digest a table file must declare.

    Args:
        document: Parsed table object. Its ``sha256`` key, if present, is
            ignored.

    Returns:
        The lowercase hexadecimal digest of the canonical JSON form.

    Raises:
        ConfigurationError: If the document cannot be serialized as JSON.
    """
    payload = {key: value for key, value in document.items() if key != "sha256"}
    try:
        canonical = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError):
        raise _table_error(
            "not_json", "The calendar table is not plain JSON data."
        ) from None
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def verify_calendar_table(
    document: Mapping[str, Any],
    *,
    expected_sha256: str | None = None,
    allow_non_permissive_license: bool = False,
) -> CalendarTable:
    """Validate a parsed table object and return its verified form.

    Args:
        document: Parsed table object.
        expected_sha256: Digest the caller pins for this table. When given,
            the table's declared and computed digests must both equal it.
        allow_non_permissive_license: Accept a declared license outside
            :data:`PERMISSIVE_TABLE_LICENSES`. Only for tables the caller
            supplies and is licensed to use; bundled tables never need it.

    Returns:
        The verified :class:`CalendarTable`.

    Raises:
        ConfigurationError: If the format, license, lineage, digest or month
            lengths are invalid. Messages name the failing field only.
    """
    if not isinstance(document, Mapping):
        raise _table_error("not_object", "A calendar table must be a JSON object.")
    keys = set(document)
    missing = sorted(_REQUIRED_KEYS - keys)
    if missing:
        raise _table_error(
            "missing_field",
            "The calendar table is missing required fields.",
            field=missing[0],
        )
    unknown = keys - _REQUIRED_KEYS - _OPTIONAL_KEYS
    if unknown:
        raise _table_error(
            "unknown_field",
            "The calendar table has fields this format does not define.",
        )
    if document["format"] != CALENDAR_TABLE_FORMAT or not _is_exact_int(
        document["format_version"], CALENDAR_TABLE_FORMAT_VERSION
    ):
        raise _table_error(
            "unsupported_format",
            "The calendar table format or format_version is not supported.",
            field="format",
        )

    declared = document["sha256"]
    if not isinstance(declared, str) or not _SHA256_RE.fullmatch(declared):
        raise _table_error(
            "invalid_digest",
            "The calendar table must declare a lowercase SHA-256 digest.",
            field="sha256",
        )
    if calendar_table_digest(document) != declared:
        raise _table_error(
            "digest_mismatch",
            "The calendar table does not match its declared SHA-256 digest.",
            field="sha256",
        )
    if expected_sha256 is not None and declared != expected_sha256:
        raise _table_error(
            "digest_mismatch",
            "The calendar table does not match the pinned SHA-256 digest.",
            field="sha256",
        )

    license_id = document["license"]
    if not isinstance(license_id, str) or not license_id.strip():
        raise _table_error(
            "missing_license",
            "The calendar table must declare an SPDX license identifier.",
            field="license",
        )
    if license_id not in PERMISSIVE_TABLE_LICENSES and not allow_non_permissive_license:
        raise _table_error(
            "non_permissive_license",
            "The calendar table's license is not on the permissive allowlist. "
            "Pass allow_non_permissive_license=True only for a table you supply "
            "and are licensed to use.",
            field="license",
        )

    source = document["source"]
    if (
        not isinstance(source, Mapping)
        or not isinstance(source.get("project"), str)
        or not source["project"].strip()
        or not all(isinstance(key, str) for key in source)
    ):
        raise _table_error(
            "missing_lineage",
            "The calendar table needs a source record naming its project.",
            field="source",
        )
    for name in ("description", "copyright", "license_text"):
        if name in document and not isinstance(document[name], str):
            raise _table_error(
                "invalid_field",
                "Calendar table text fields must be strings.",
                field=name,
            )
    if "verification" in document and not isinstance(document["verification"], list):
        raise _table_error(
            "invalid_field",
            "The calendar table's verification record must be a list.",
            field="verification",
        )

    code = document["calendar"]
    if not is_calendar_code(code):
        raise _table_error(
            "invalid_calendar",
            "The calendar table needs a lowercase calendar code.",
            field="calendar",
        )
    variant = document.get("variant")
    if variant is not None and not is_calendar_code(variant):
        raise _table_error(
            "invalid_variant",
            "A declared variant must be a lowercase code such as 'umm_al_qura'.",
            field="variant",
        )
    version = document["version"]
    if not isinstance(version, str) or not version.strip():
        raise _table_error(
            "invalid_version", "The calendar table needs a version.", field="version"
        )

    first_year = document["first_year"]
    if not _is_int(first_year) or first_year < 1:
        raise _table_error(
            "invalid_year",
            "first_year must be a positive integer.",
            field="first_year",
        )
    try:
        first_day = date.fromisoformat(document["first_day"])
    except (TypeError, ValueError):
        raise _table_error(
            "invalid_first_day",
            "first_day must be an ISO Gregorian date.",
            field="first_day",
        ) from None

    month_lengths = _validate_month_lengths(document["month_lengths"])
    total_days = sum(sum(lengths) for lengths in month_lengths)
    if first_day.toordinal() + total_days - 1 > date.max.toordinal():
        raise _table_error(
            "out_of_range",
            "The calendar table extends past the last supported Gregorian date.",
            field="month_lengths",
        )

    return CalendarTable(
        calendar=code,
        variant=variant,
        version=version,
        license=license_id,
        source=MappingProxyType(dict(source)),
        first_year=first_year,
        first_day_number=first_day.toordinal(),
        month_lengths=month_lengths,
        sha256=declared,
    )


def load_calendar_table(
    source: TableSource,
    *,
    expected_sha256: str | None = None,
    allow_non_permissive_license: bool = False,
) -> TableCalendar:
    """Load and verify a month-length table and return its converter.

    Use this for caller-supplied tables, such as another Hijri variant, then
    register the converter. Registering under ``"hijri"`` with
    ``replace=True`` makes era-marked and detected Hijri dates use it; a new
    code applies to spans whose ``calendar`` metadata names that code::

        converter = load_calendar_table("hijri_variant.json")
        register_calendar(converter, replace=True)

    Args:
        source: Path to a table file, or an already parsed table object.
        expected_sha256: Optional pinned digest the table must match.
        allow_non_permissive_license: Accept a non-permissive declared
            license for a caller-supplied table. Such tables must never be
            bundled with OpenMed.

    Returns:
        A :class:`TableCalendar` ready for
        :func:`~openmed.core.calendars.register_calendar`.

    Raises:
        ConfigurationError: If the file cannot be read or the table fails
            verification. Messages never include the path or table values.
    """
    if isinstance(source, Mapping):
        document: Any = source
    else:
        document = _read_table_file(source)
    table = verify_calendar_table(
        document,
        expected_sha256=expected_sha256,
        allow_non_permissive_license=allow_non_permissive_license,
    )
    return TableCalendar(table)


def bundled_calendar_table(code: str) -> TableCalendar:
    """Load one bundled table and verify it against its pinned digest.

    Args:
        code: ``"hijri"`` or ``"bikram_sambat"``.

    Returns:
        A fresh :class:`TableCalendar` for the bundled table.

    Raises:
        ConfigurationError: If no table is bundled for ``code`` or the
            bundled file fails verification.
    """
    entry = BUNDLED_CALENDAR_TABLES.get(code)
    if entry is None:
        raise ConfigurationError(
            "No calendar table is bundled for that code.",
            details={"argument": "code", "reason": "unknown_table"},
        )
    name, digest = entry
    converter = load_calendar_table(_DATA_DIR / name, expected_sha256=digest)
    if converter.code != code:
        raise _table_error(
            "invalid_calendar",
            "A bundled calendar table declares a different calendar code.",
            field="calendar",
        )
    return converter


def _read_table_file(path: Any) -> Any:
    try:
        resolved = Path(os.fspath(path))
        if resolved.stat().st_size > _MAX_TABLE_BYTES:
            raise _table_error(
                "too_large", "The calendar table file exceeds the size limit."
            )
        text = resolved.read_text(encoding="utf-8")
    except ConfigurationError:
        raise
    except (OSError, TypeError, ValueError):
        raise _table_error(
            "unreadable", "The calendar table file could not be read as UTF-8 text."
        ) from None
    try:
        return json.loads(text)
    except ValueError:
        raise _table_error(
            "not_json", "The calendar table file is not valid JSON."
        ) from None


def _validate_month_lengths(value: Any) -> tuple[tuple[int, ...], ...]:
    if not isinstance(value, list) or not 1 <= len(value) <= _MAX_YEARS:
        raise _table_error(
            "invalid_month_lengths",
            f"month_lengths must list between 1 and {_MAX_YEARS} years.",
            field="month_lengths",
        )
    years: list[tuple[int, ...]] = []
    for lengths in value:
        if (
            not isinstance(lengths, list)
            or not 1 <= len(lengths) <= _MAX_MONTHS_PER_YEAR
            or not all(
                _is_int(length) and 1 <= length <= _MAX_MONTH_LENGTH
                for length in lengths
            )
        ):
            raise _table_error(
                "invalid_month_lengths",
                f"Each year needs 1-{_MAX_MONTHS_PER_YEAR} month lengths of "
                f"1-{_MAX_MONTH_LENGTH} days.",
                field="month_lengths",
            )
        years.append(tuple(lengths))
    return tuple(years)


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_exact_int(value: object, expected: int) -> bool:
    return _is_int(value) and value == expected


def _table_error(
    reason: str,
    message: str,
    *,
    field: str | None = None,
) -> ConfigurationError:
    details: dict[str, str] = {"reason": reason}
    if field is not None:
        details["field"] = field
    return ConfigurationError(message, details=details)


for _code in BUNDLED_CALENDAR_TABLES:
    register_calendar(bundled_calendar_table(_code))


__all__ = [
    "BUNDLED_CALENDAR_TABLES",
    "CALENDAR_TABLE_FORMAT",
    "CALENDAR_TABLE_FORMAT_VERSION",
    "PERMISSIVE_TABLE_LICENSES",
    "CalendarTable",
    "TableCalendar",
    "bundled_calendar_table",
    "calendar_table_digest",
    "load_calendar_table",
    "verify_calendar_table",
]
