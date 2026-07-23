"""Deterministic TNM/AJCC cancer-staging descriptor extractor.

Pathology and oncology notes record staging as compact notations such as
``pT2 N1 M0`` or ``ypT3a``. This module parses those *stated* descriptors into a
structured :class:`TnmStage` -- a staging basis (``c``/``p``/``yc``/``yp``/``r``/
``a``), the T/N/M categories, and their subcategories -- following AJCC 8th
edition conventions.

It makes no staging decision of its own: categories are read verbatim from the
text, never derived. Staging-shaped tokens that are not valid AJCC categories
(e.g. ``T5``, ``N4``) are surfaced in ``unparsed`` with a reason rather than
coerced, and each parse carries a ``confidence`` flag. There is no stage-group
derivation, prognostic table, or treatment inference -- see
:data:`TNM_STAGING_ADVISORY`.
"""

from __future__ import annotations

import re
from typing import Literal, Optional, TypedDict

TnmBasis = Literal["c", "p", "yc", "yp", "r", "a"]

#: Full (unambiguous) staging bases. A bare ``y`` post-therapy prefix is
#: recognized but treated as ambiguous (unknown clinical/pathologic) rather than
#: coerced to one of these.
FULL_BASES: frozenset[str] = frozenset({"c", "p", "yc", "yp", "r", "a"})

TNM_STAGING_ADVISORY = (
    "TNM/AJCC staging extraction is deterministic assistive tooling that "
    "structures stage descriptors explicitly written in the note. It reads "
    "categories verbatim and performs no stage-group derivation, prognostic "
    "assessment, survival estimate, or treatment recommendation, and is not a "
    "substitute for a clinician's or pathologist's formal staging."
)


class UnparsedToken(TypedDict):
    """A staging-shaped token that was surfaced, with a reason, not coerced."""

    token: str
    reason: str


class TnmStage(TypedDict):
    """A structured TNM staging descriptor read from note text.

    ``basis`` is the staging prefix shared by the categories (``None`` when none
    is written). ``t``/``n``/``m`` are the normalized categories (e.g. ``"T2"``,
    ``"Tis"``, ``"TX"``, ``"N0"``, ``"M1"``) and the ``*_subcategory`` fields the
    suffix when present (e.g. ``"a"``, ``"mi"``, ``"i+"``). ``confidence`` is
    ``"low"`` whenever a token was surfaced as ``unparsed``, the prefixes across
    categories disagree, a bare ambiguous ``y`` prefix is used, or nothing was
    recognized; otherwise ``"high"``. ``unparsed`` lists staging-shaped tokens
    that were not used for the primary stage, each with a reason.
    """

    basis: Optional[str]
    t: Optional[str]
    t_subcategory: Optional[str]
    n: Optional[str]
    n_subcategory: Optional[str]
    m: Optional[str]
    m_subcategory: Optional[str]
    confidence: str
    unparsed: list[UnparsedToken]
    advisory: str


# --- token grammar -----------------------------------------------------------

# A staging token may be preceded by start-of-text, whitespace, common
# punctuation (including a hyphen, so ``pT2-N1-M0`` splits), or a digit (so
# contiguous runs like ``pT2N1M0`` split correctly), but never by a letter (so
# ``Free`` / ``point`` / ``stageN1`` do not match).
_BOUNDARY = r"(?<![^\s,;:()\[\]/=\d-])"

# Longest-first prefix alternation; a bare ``y`` is recognized but ambiguous.
_PREFIX = r"(?P<pre>yp|yc|y|c|p|r|a)?"

# The whole staging-shaped token is captured maximally: a T/N/M letter, then a
# value/subcategory ``body`` that runs up to the next separator or capitalized
# category letter, plus an optional parenthetical. Capturing the full token
# (rather than a prefix of it) means malformed tokens such as ``T10`` or ``N2d``
# are validated and surfaced as unparsed, never truncated into ``T1``/``N2``. The
# ``is`` in-situ value is matched case-insensitively so ``Tis`` and ``TIS`` both
# parse; T/N/M and the basis prefix stay case-sensitive to avoid prose matches.
_BODY = r"(?P<body>(?:(?i:is)|[Xx]|\d)[^\sTNM,;:()\[\]/=-]*)"
_CATEGORY_RE = re.compile(
    _BOUNDARY + _PREFIX + r"(?P<cat>[TNM])" + _BODY + r"(?:\((?P<paren>[^)]{1,12})\))?"
)

#: Values that are valid for each category (AJCC 8th edition).
_VALID_VALUES: dict[str, frozenset[str]] = {
    "T": frozenset({"is", "x", "0", "1", "2", "3", "4"}),
    "N": frozenset({"x", "0", "1", "2", "3"}),
    "M": frozenset({"x", "0", "1"}),
}

#: Recognized subcategory shapes per category (matched case-insensitively): T
#: allows ``a``-``d`` (optionally a nested digit, e.g. ``a1``) or ``mi``; N and M
#: allow ``a``-``c``. A parenthetical qualifier (``(i+)``, ``(DCIS)``) is handled
#: separately.
_SUBCATEGORY_RE: dict[str, re.Pattern[str]] = {
    "T": re.compile(r"mi|[a-d]\d?", re.IGNORECASE),
    "N": re.compile(r"[a-c]", re.IGNORECASE),
    "M": re.compile(r"[a-c]", re.IGNORECASE),
}


class _Category:
    """A validated primary/secondary category: its match plus normalized value."""

    __slots__ = ("match", "category", "subcategory")

    def __init__(self, match: re.Match[str], category: str, subcategory: Optional[str]):
        self.match = match
        self.category = category
        self.subcategory = subcategory

    @property
    def prefix(self) -> Optional[str]:
        return self.match.group("pre")


def _classify(
    match: re.Match[str],
) -> tuple[Optional[str], Optional[str], bool]:
    """Validate one matched token, returning ``(category, subcategory, valid)``.

    The token's value must be valid for its category and any trailing
    subcategory must be a recognized shape; otherwise the token is invalid and
    surfaced (never coerced). A parenthetical qualifier fills the subcategory
    when no letter subcategory is written.
    """
    cat = match.group("cat")
    body = match.group("body")
    lowered = body.lower()
    if lowered.startswith("is"):
        value, rest = "is", body[2:]
    else:
        value, rest = body[:1], body[1:]

    if value.lower() not in _VALID_VALUES[cat]:
        return None, None, False

    subcategory: Optional[str] = None
    if rest:
        if _SUBCATEGORY_RE[cat].fullmatch(rest) is None:
            return None, None, False
        subcategory = rest.lower()

    if value == "is":
        category = "Tis"
    elif value.lower() == "x":
        category = f"{cat}X"
    else:
        category = f"{cat}{value}"

    paren = match.group("paren")
    if paren:
        # Preserve both a letter subcategory and a parenthetical qualifier when
        # both are written (e.g. "N1a(sn)"), rather than dropping either.
        subcategory = f"{subcategory}({paren})" if subcategory else paren
    return category, subcategory, True


def _choose_primary(
    items: list[_Category],
) -> tuple[Optional[_Category], list[_Category]]:
    """Pick the primary category, preferring one carrying a basis prefix.

    A note occasionally mentions a bare category (e.g. a lab ``T4``) before the
    staged one; preferring a prefixed match makes the staged descriptor win. The
    remaining categories are returned as secondaries (surfaced, not used).
    """
    if not items:
        return None, []
    primary = next((item for item in items if item.prefix), items[0])
    secondaries = [item for item in items if item is not primary]
    return primary, secondaries


def parse_tnm(text: str) -> TnmStage:
    """Parse a single TNM/AJCC staging descriptor from ``text``.

    Args:
        text: Free-text pathology/oncology content containing a staging notation
            such as ``"pT2 N1 M0"`` or ``"ypT3a"``.

    Returns:
        A :class:`TnmStage`. The primary T/N/M categories (the first valid match
        of each, preferring a prefixed one) are normalized; ``basis`` is taken
        from the primary categories' shared prefix. Additional or invalid
        staging-shaped tokens are surfaced in ``unparsed`` with a reason and
        never coerced, and :data:`TNM_STAGING_ADVISORY` is always attached.
    """
    source = text or ""

    by_category: dict[str, list[_Category]] = {"T": [], "N": [], "M": []}
    invalid: list[re.Match[str]] = []
    for match in _CATEGORY_RE.finditer(source):
        category, subcategory, valid = _classify(match)
        if valid and category is not None:
            by_category[match.group("cat")].append(
                _Category(match, category, subcategory)
            )
        else:
            invalid.append(match)

    primary_t, secondary_t = _choose_primary(by_category["T"])
    primary_n, secondary_n = _choose_primary(by_category["N"])
    primary_m, secondary_m = _choose_primary(by_category["M"])

    t = primary_t.category if primary_t else None
    t_sub = primary_t.subcategory if primary_t else None
    n = primary_n.category if primary_n else None
    n_sub = primary_n.subcategory if primary_n else None
    m = primary_m.category if primary_m else None
    m_sub = primary_m.subcategory if primary_m else None

    primaries = [primary_t, primary_n, primary_m]
    raw_prefixes = [item.prefix for item in primaries if item and item.prefix]
    basis_prefixes = [pre for pre in raw_prefixes if pre in FULL_BASES]
    basis = basis_prefixes[0] if basis_prefixes else None
    mixed_basis = len(set(basis_prefixes)) > 1
    ambiguous_y = any(pre == "y" for pre in raw_prefixes)

    # Surface every token not used for the primary stage, ordered by position.
    surfaced: list[tuple[int, UnparsedToken]] = []
    for cat, secondaries in (
        ("T", secondary_t),
        ("N", secondary_n),
        ("M", secondary_m),
    ):
        for item in secondaries:
            surfaced.append(
                (
                    item.match.start(),
                    UnparsedToken(
                        token=item.match.group(0),
                        reason=f"additional {cat} category not used for the primary stage",
                    ),
                )
            )
    for match in invalid:
        surfaced.append(
            (
                match.start(),
                UnparsedToken(
                    token=match.group(0),
                    reason=f"unrecognized {match.group('cat')} category "
                    f"'{match.group(0)}'",
                ),
            )
        )
    surfaced.sort(key=lambda item: item[0])
    unparsed = [token for _, token in surfaced]

    nothing_recognized = primary_t is None and primary_n is None and primary_m is None
    confidence = (
        "low"
        if (unparsed or mixed_basis or ambiguous_y or nothing_recognized)
        else "high"
    )

    return TnmStage(
        basis=basis,
        t=t,
        t_subcategory=t_sub,
        n=n,
        n_subcategory=n_sub,
        m=m,
        m_subcategory=m_sub,
        confidence=confidence,
        unparsed=unparsed,
        advisory=TNM_STAGING_ADVISORY,
    )


__all__ = [
    "FULL_BASES",
    "TNM_STAGING_ADVISORY",
    "TnmBasis",
    "TnmStage",
    "UnparsedToken",
    "parse_tnm",
]
