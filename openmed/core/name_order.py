"""Culture-aware CJK name-order and honorific handling for PERSON spans.

East-Asian names (Japanese, Korean, Chinese) are written **family-name-first**
and are frequently followed by a trailing honorific or title — Japanese
``さん``/``様``, Korean ``씨``/``님``, Chinese ``先生``/``女士``.
The generic PERSON surrogate path is order- and honorific-agnostic, so a naive
replacement can drop or mangle the honorific ("田中さん" -> "Smith") or fold it
into the swapped name, which breaks both surrogate realism and span matching.

This module is a small, dependency-free helper that the surrogate-replacement
step consults **only** for ``ja``/``ko``/``zh`` PERSON spans:

- :func:`split_name` returns ``(family, given, honorific)`` using the
  family-name-first convention and a per-language honorific list.
- :func:`normalize_person_span` peels a trailing honorific off a span so the
  caller can swap the *name* while re-attaching the *honorific* verbatim.
- :data:`HONORIFICS` is the seed honorific set keyed by language; packs may
  register additions through :func:`register_honorific`.

Everything here operates on the span text alone. It never emits plaintext PHI,
performs no network calls, and carries no non-permissive data — the honorific
seed set is common-knowledge cultural vocabulary.

Scope note (OM-291): this is honorific + family/given separation only. Western
honorifics (``Mr``/``Dr``/``Mrs``) and full morphological name parsing are out
of scope and intentionally left to the generic path.
"""

from __future__ import annotations

from typing import Dict, Final, List, Mapping, Tuple

__all__ = [
    "CJK_LANGUAGES",
    "HONORIFICS",
    "honorifics_for",
    "normalize_person_span",
    "register_honorific",
    "split_name",
]


# Languages this helper is responsible for. Everything else is passed through
# unchanged by :func:`normalize_person_span` / :func:`split_name`.
CJK_LANGUAGES: Final[frozenset[str]] = frozenset({"ja", "ko", "zh"})


# Seed honorific suffixes per language, longest-first within each language so
# multi-character honorifics win over their prefixes when matching. Each entry
# is a trailing honorific that attaches directly after a PERSON span.
#
# These are common, non-restricted cultural terms:
#   ja  さん (san), 様/さま (sama), くん/君 (kun), ちゃん (chan), 氏 (shi),
#       先生 (sensei)
#   ko  씨 (ssi), 님 (nim), 군 (gun), 양 (yang), 선생님 (seonsaengnim)
#   zh  先生 (xiānsheng), 女士 (nǚshì), 小姐 (xiǎojiě), 太太 (tàitai),
#       老师 (lǎoshī)
_HONORIFIC_SEED: Final[Mapping[str, Tuple[str, ...]]] = {
    "ja": ("先生", "さん", "様", "さま", "ちゃん", "くん", "君", "氏"),
    "ko": ("선생님", "씨", "님", "군", "양"),
    "zh": ("先生", "女士", "小姐", "太太", "老师", "醫師", "医生"),
}


def _seed_honorifics() -> Dict[str, List[str]]:
    """Return a fresh, sorted (longest-first) copy of the seed honorifics."""

    return {
        lang: sorted(dict.fromkeys(values), key=len, reverse=True)
        for lang, values in _HONORIFIC_SEED.items()
    }


# Mutable, per-language honorific registry. Packs mutate this via
# :func:`register_honorific`; callers read it via :func:`honorifics_for`. It is
# seeded from the documented seed set above and always kept longest-first so the
# greedy suffix match in :func:`normalize_person_span` is unambiguous.
HONORIFICS: Dict[str, List[str]] = _seed_honorifics()


def register_honorific(lang: str, honorific: str) -> None:
    """Register an additional honorific for a CJK language pack.

    Args:
        lang: ISO 639-1 language code (``ja``, ``ko``, or ``zh``).
        honorific: Trailing honorific to recognize (e.g. a pack-specific
            regional variant). Unsupported languages and empty input are
            ignored; duplicate values are not added twice.

    The registry is kept sorted longest-first so multi-character additions
    take precedence over shorter prefixes during suffix matching.
    """

    honorific = (honorific or "").strip()
    if lang not in CJK_LANGUAGES or not honorific:
        return
    bucket = HONORIFICS.setdefault(lang, [])
    if honorific not in bucket:
        bucket.append(honorific)
        bucket.sort(key=len, reverse=True)


def honorifics_for(lang: str) -> Tuple[str, ...]:
    """Return the registered honorifics for ``lang`` (longest-first).

    Returns an empty tuple for languages with no registered honorifics, so
    non-CJK callers get a safe no-op.
    """

    return tuple(HONORIFICS.get(lang, ()))


def _split_trailing_honorific(text: str, lang: str) -> Tuple[str, str]:
    """Peel a single trailing honorific suffix off ``text``.

    Returns ``(core, suffix)`` where ``suffix`` is ``""`` when none of the
    language's registered honorifics match. When a match exists, ``suffix``
    preserves the exact separator and trailing whitespace from the input so
    surrogate replacement cannot concatenate a spaced honorific onto the new
    name. An honorific-only span returns an empty ``core`` and its full input as
    ``suffix`` so callers can use a safe fallback without dropping the title.
    """

    content_end = len(text.rstrip())
    stripped = text[:content_end]
    for honorific in honorifics_for(lang):
        if stripped.endswith(honorific):
            core = stripped[: -len(honorific)].rstrip()
            return core, text[len(core) :]
    return text, ""


def normalize_person_span(span_text: str, lang: str) -> Tuple[str, str]:
    """Separate a trailing honorific from a CJK PERSON span.

    This is the entry point the surrogate-replacement step uses: it hands back
    the bare ``name`` (to be swapped for a surrogate) and the honorific suffix
    (to be re-attached verbatim afterwards), so both the title and any separator
    whitespace survive replacement.

    Args:
        span_text: The detected PERSON span text.
        lang: ISO 639-1 language code. For non-CJK languages the span is
            returned unchanged with an empty honorific, so callers can apply
            this uniformly without a language guard.

    Returns:
        ``(name, honorific_suffix)``. The suffix includes separator/trailing
        whitespace exactly as written. It is ``""`` when no honorific is
        present (or ``lang`` is not a CJK language), in which case ``name`` is
        the original ``span_text`` unchanged.

    Example:
        >>> normalize_person_span("田中さん", "ja")
        ('田中', 'さん')
        >>> normalize_person_span("Smith", "en")
        ('Smith', '')
    """

    if lang not in CJK_LANGUAGES or not span_text:
        return span_text, ""
    return _split_trailing_honorific(span_text, lang)


def _split_family_given(name: str, lang: str) -> Tuple[str, str]:
    """Split a bare (honorific-free) CJK name family-name-first.

    Japanese names are typically space-separated (``田中 太郎``); Korean and
    Chinese names are written contiguously, where the leading character is the
    (usually single-character) family name and the remainder is the given name.

    Returns ``(family, given)``. ``given`` may be empty for a mononym or a
    single-character span.
    """

    parts = name.split()
    if len(parts) >= 2:
        # Whitespace-delimited (Japanese full names): first token is family.
        return parts[0], " ".join(parts[1:])

    single = parts[0] if parts else name
    if len(single) <= 1:
        return single, ""
    # Contiguous CJK (Korean/Chinese, and space-free Japanese): the leading
    # character is the family name, the remainder the given name.
    return single[0], single[1:]


def split_name(name: str, lang: str) -> Tuple[str, str, str]:
    """Split a CJK PERSON span into ``(family, given, honorific)``.

    Applies family-name-first ordering and peels a trailing honorific. For
    non-CJK languages the whole input is returned as ``family`` with empty
    ``given``/``honorific`` so the caller never has to special-case them.

    Args:
        name: The PERSON span text, optionally carrying a trailing honorific.
        lang: ISO 639-1 language code (``ja``, ``ko``, ``zh`` are handled;
            others pass through).

    Returns:
        ``(family, given, honorific)``. ``given`` and/or ``honorific`` may be
        empty strings.

    Example:
        >>> split_name("田中 太郎さん", "ja")
        ('田中', '太郎', 'さん')
        >>> split_name("김민준씨", "ko")
        ('김', '민준', '씨')
        >>> split_name("王伟先生", "zh")
        ('王', '伟', '先生')
    """

    if lang not in CJK_LANGUAGES or not name:
        return name, "", ""

    core, honorific_suffix = _split_trailing_honorific(name, lang)
    family, given = _split_family_given(core, lang)
    return family, given, honorific_suffix.strip()
