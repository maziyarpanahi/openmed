"""Property-based fuzz harness for the date-shift helpers.

``deidentify(method="shift_dates", keep_year=...)`` relies on ``_shift_date`` to
move clinical dates by a fixed offset while optionally pinning the year. That
helper parses many localized formats and does calendar arithmetic, which is a
classic source of edge-case defects (the Feb-29 -> non-leap-year crash fixed via
``_replace_year_safe`` being the motivating example).

Invariants asserted here for any generated date and offset:

* **No crash** — ``_shift_date`` never raises on arbitrary ``str`` input; on
  unparseable input it returns the ``[DATE_SHIFTED]`` placeholder.
* **Calendar validity** — when the helper returns a shifted date (not the
  placeholder), that output re-parses to a real calendar date. In particular the
  leap-day class of inputs (Feb 29) never yields an invalid date under
  ``keep_year``.
* **Interval semantics** — without ``keep_year``, shifting an ISO date by ``d``
  days lands exactly ``d`` days away; shifting by ``0`` is the identity.
* **Year preservation** — with ``keep_year=True``, the shifted year equals the
  original year (clamped Feb-28 for the Feb-29 leap-day edge).
* **``_replace_year_safe`` totality** — never raises for any datetime/year pair,
  and preserves the year it was asked for.

All dates are synthetic. No real PHI is used.
"""

from __future__ import annotations

import calendar
from datetime import date, datetime, timedelta

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from openmed.core.pii import _replace_year_safe, _shift_date

# Ensure the bounded/nightly Hypothesis profile is registered.
from . import conftest as _fuzz_conftest  # noqa: F401  (import for side effects)

pytestmark = pytest.mark.fuzz

_PLACEHOLDER = "[DATE_SHIFTED]"

# Keep years inside datetime's supported range with headroom for +/- offsets so
# that a *valid* shift never overflows into MINYEAR/MAXYEAR territory (which
# would legitimately produce the placeholder and is out of scope for the
# interval-semantics invariant).
_dates = st.dates(min_value=date(1900, 1, 1), max_value=date(2100, 12, 31))
_offsets = st.integers(min_value=-3650, max_value=3650)
_leap_years = st.sampled_from(
    [year for year in range(1900, 2101) if calendar.isleap(year)]
)


@given(d=_dates, shift=_offsets, keep_year=st.booleans())
@settings(deadline=1000)
def test_shift_date_iso_never_crashes_and_stays_valid(d, shift, keep_year):
    """ISO dates shift without crashing and always yield a valid calendar date."""
    iso = d.isoformat()  # YYYY-MM-DD
    out = _shift_date(iso, shift, keep_year=keep_year, lang="en")
    assert isinstance(out, str)
    if out == _PLACEHOLDER:
        return  # graceful degradation is acceptable
    # A returned date string must re-parse to a real calendar date.
    parsed = datetime.strptime(out, "%Y-%m-%d")
    assert 1 <= parsed.month <= 12
    assert 1 <= parsed.day <= 31


@given(d=_dates, shift=_offsets)
def test_shift_date_interval_semantics(d, shift):
    """Without ``keep_year``, an ISO shift lands exactly ``shift`` days away."""
    iso = d.isoformat()
    out = _shift_date(iso, shift, keep_year=False, lang="en")
    assume(out != _PLACEHOLDER)
    parsed = datetime.strptime(out, "%Y-%m-%d").date()
    assert (parsed - d).days == shift


@given(d=_dates)
def test_shift_date_zero_offset_is_identity(d):
    """A zero-day shift preserves the date exactly (ISO round-trip)."""
    iso = d.isoformat()
    out = _shift_date(iso, 0, keep_year=False, lang="en")
    assume(out != _PLACEHOLDER)
    assert out == iso


@given(d=_dates, shift=_offsets)
def test_shift_date_keep_year_preserves_year(d, shift):
    """With ``keep_year=True`` the exact shifted month/day is re-homed."""
    iso = d.isoformat()
    out = _shift_date(iso, shift, keep_year=True, lang="en")
    assume(out != _PLACEHOLDER)
    parsed = datetime.strptime(out, "%Y-%m-%d").date()
    shifted = datetime(d.year, d.month, d.day) + timedelta(days=shift)
    expected = _replace_year_safe(shifted, d.year).date()
    assert parsed == expected


@given(
    year=_leap_years,
    shift=_offsets,
    keep_year=st.booleans(),
)
def test_shift_date_leap_day_never_crashes(year, shift, keep_year):
    """Feb 29 in a leap year shifts without crashing under any keep_year mode.

    This is the motivating regression class: ``keep_year`` re-homing a shifted
    date onto a non-leap year must not raise (it clamps to Feb 28).
    """
    leap_day = date(year, 2, 29).isoformat()
    out = _shift_date(leap_day, shift, keep_year=keep_year, lang="en")
    assert isinstance(out, str)
    if out != _PLACEHOLDER:
        parsed = datetime.strptime(out, "%Y-%m-%d")
        if keep_year:
            assert parsed.year == year


@given(text=st.text(min_size=0, max_size=64), shift=_offsets)
def test_shift_date_arbitrary_text_never_crashes(text, shift):
    """``_shift_date`` never raises on arbitrary junk; returns a ``str`` always."""
    out = _shift_date(text, shift, keep_year=False, lang="en")
    assert isinstance(out, str)


@given(
    d=st.dates(min_value=date(1, 1, 1), max_value=date(9999, 12, 31)),
    year=st.integers(min_value=1, max_value=9999),
)
def test_replace_year_safe_is_total_and_preserves_year(d, year):
    """``_replace_year_safe`` never raises and returns the requested year.

    Covers the Feb-29 -> non-leap-year case directly: no ``ValueError`` escapes.
    """
    dt = datetime(d.year, d.month, d.day)
    out = _replace_year_safe(dt, year)
    assert isinstance(out, datetime)
    assert out.year == year
    assert out.month == d.month
    expected_day = min(d.day, calendar.monthrange(year, d.month)[1])
    assert out.day == expected_day
