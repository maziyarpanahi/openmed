"""Format-preserving helpers for surrogate generation.

When we replace a phone number, date, email, or ID with a fake surrogate,
we want the surrogate to *look like* the original — same digit groupings,
same separators, same overall shape. That keeps the deidentified text
useful for downstream tooling (regex matchers, length-sensitive UIs,
PDF templates) that may have been calibrated on the original format.

Each helper takes the original text and returns either:
  - a surrogate that mirrors the original's structure, or
  - a small "shape descriptor" that a generator can use to render a
    locale-appropriate value with the right shape.
"""

from __future__ import annotations

import random
import re
from datetime import datetime, timedelta
from typing import List, Optional


# ---------------------------------------------------------------------------
# Phone numbers
# ---------------------------------------------------------------------------

def extract_digit_groups(text: str) -> List[int]:
    """Return the lengths of each contiguous digit run in ``text``.

    >>> extract_digit_groups("+1 (415) 555-1234")
    [1, 3, 3, 4]
    >>> extract_digit_groups("+33 6 12 34 56 78")
    [2, 1, 2, 2, 2, 2]
    """
    return [len(m.group()) for m in re.finditer(r"\d+", text)]


def preserve_phone_format(original: str, *, rng: Optional[random.Random] = None) -> str:
    """Generate a fake phone number that mirrors ``original``'s structure.

    The non-digit characters (``+``, spaces, dashes, parentheses) are kept
    in place; each digit run is filled with new random digits. Useful when
    Faker's locale-specific ``phone_number()`` would change the digit
    grouping in ways that break downstream regexes.
    """
    rng = rng or random
    out = []
    for ch in original:
        if ch.isdigit():
            out.append(str(rng.randint(0, 9)))
        else:
            out.append(ch)
    return "".join(out)


# ---------------------------------------------------------------------------
# Dates
# ---------------------------------------------------------------------------

# Common date-format heuristics keyed off the locale. The first format
# wins when generating a surrogate, but ``preserve_date_format`` tries to
# match the original's separator and digit ordering before falling back.
_DATE_SEPARATORS = ("/", "-", ".", " ")


def preserve_date_format(
    original: str,
    *,
    day_first: bool = False,
    rng: Optional[random.Random] = None,
) -> str:
    """Generate a fake date that uses the same separator and ordering.

    Recognises the common ``dd/mm/yyyy``, ``mm/dd/yyyy``, ``yyyy-mm-dd``,
    and ``dd.mm.yyyy`` shapes. ``day_first`` controls fallback when the
    original is ambiguous (e.g. ``05/06/2020``).

    Returns a surrogate date in the same format, drawn uniformly from the
    last 100 years.
    """
    rng = rng or random

    days = rng.randint(0, 365 * 100)
    fake = datetime(1925, 1, 1) + timedelta(days=days)

    # Detect separator
    sep = "/"
    for candidate in _DATE_SEPARATORS:
        if candidate in original:
            sep = candidate
            break

    # Detect ordering: yyyy-first if a 4-digit run is at the start
    if re.match(r"^\d{4}", original):
        return f"{fake.year:04d}{sep}{fake.month:02d}{sep}{fake.day:02d}"
    if day_first:
        return f"{fake.day:02d}{sep}{fake.month:02d}{sep}{fake.year:04d}"
    return f"{fake.month:02d}{sep}{fake.day:02d}{sep}{fake.year:04d}"


# ---------------------------------------------------------------------------
# Emails
# ---------------------------------------------------------------------------

def preserve_email_pattern(original: str, fake_email: str) -> str:
    """Use ``fake_email``'s local part with ``original``'s domain.

    Keeps the domain stable (often a meaningful tenant marker like
    ``@hospital.org`` that downstream systems route on) while randomizing
    the local part. Falls back to ``fake_email`` verbatim if either side
    doesn't parse as ``local@domain``.
    """
    if "@" not in original or "@" not in fake_email:
        return fake_email
    fake_local = fake_email.split("@", 1)[0]
    original_domain = original.split("@", 1)[1]
    return f"{fake_local}@{original_domain}"


# ---------------------------------------------------------------------------
# Generic IDs (digit + separator preservation)
# ---------------------------------------------------------------------------

def preserve_id_pattern(original: str, *, rng: Optional[random.Random] = None) -> str:
    """Replace digits in ``original`` with random digits, keeping all other
    characters in place.

    Use for opaque IDs where format matters but no checksum applies (MRN,
    account numbers, sometimes ZIP codes). For checksum-bearing IDs (CPF,
    CNPJ, BSN, NIR, ...), use the dedicated Faker provider instead.
    """
    rng = rng or random
    out = []
    for ch in original:
        if ch.isdigit():
            out.append(str(rng.randint(0, 9)))
        elif ch.isalpha():
            out.append(rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ") if ch.isupper()
                       else rng.choice("abcdefghijklmnopqrstuvwxyz"))
        else:
            out.append(ch)
    return "".join(out)


__all__ = [
    "extract_digit_groups",
    "preserve_phone_format",
    "preserve_date_format",
    "preserve_email_pattern",
    "preserve_id_pattern",
]
