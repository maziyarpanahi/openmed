"""Offline structural validation for common BCP 47 locale tags."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Final

__all__ = [
    "MAX_LOCALE_TAG_LENGTH",
    "LocaleTagError",
    "normalize_locale_tag",
    "normalize_locale_tags",
]

MAX_LOCALE_TAG_LENGTH: Final[int] = 255
_LANGUAGE = re.compile(r"[a-zA-Z]{2,8}\Z")
_SCRIPT = re.compile(r"[a-zA-Z]{4}\Z")
_REGION = re.compile(r"(?:[a-zA-Z]{2}|[0-9]{3})\Z")
_VARIANT = re.compile(r"(?:[a-zA-Z0-9]{5,8}|[0-9][a-zA-Z0-9]{3})\Z")
_ALIAS_KEY = re.compile(r"[a-zA-Z0-9]+(?:[-_][a-zA-Z0-9]+)*\Z")


class LocaleTagError(ValueError):
    """Stable, value-free failure for locale or alias registry validation."""

    def __init__(self, category: str) -> None:
        self.category = category
        super().__init__(category)


def normalize_locale_tag(tag: str, *, aliases: Mapping[str, str] | None = None) -> str:
    """Normalize a common language[-Script][-REGION][-variant...] tag.

    Validation is structural, not IANA registry conformance. Extlangs,
    extensions, private-use and grandfathered forms are outside this subset,
    unless an explicitly supplied alias resolves them to a supported tag.
    Alias lookup is ASCII case-insensitive; caller mappings are not modified.
    There are no implicit aliases, registry downloads, or locale fallbacks.
    """
    resolved_aliases = _validate_aliases(aliases)
    return _normalize(tag, resolved_aliases)


def normalize_locale_tags(
    tags: Iterable[str], *, aliases: Mapping[str, str] | None = None
) -> tuple[str, ...]:
    """Normalize registry tags in input order and reject canonical duplicates.

    The returned tuple is built only after every entry passes. Case variants
    and explicit aliases that identify the same canonical tag are duplicates.
    A single string is not accepted as an iterable of registry entries.
    """
    if isinstance(tags, (str, bytes)):
        raise TypeError("tags must be an iterable of locale tags")
    resolved_aliases = _validate_aliases(aliases)
    normalized: list[str] = []
    seen: set[str] = set()
    for tag in tags:
        canonical = _normalize(tag, resolved_aliases)
        if canonical in seen:
            raise LocaleTagError("locale_tag_duplicate")
        seen.add(canonical)
        normalized.append(canonical)
    return tuple(normalized)


def _normalize(tag: str, aliases: Mapping[str, str]) -> str:
    if not isinstance(tag, str):
        raise TypeError("tag must be a string")
    if not tag or len(tag) > MAX_LOCALE_TAG_LENGTH or not tag.isascii():
        raise LocaleTagError("locale_tag_invalid")
    return aliases.get(tag.lower()) or _canonicalize(tag)


def _canonicalize(tag: str) -> str:
    if not tag or len(tag) > MAX_LOCALE_TAG_LENGTH:
        raise LocaleTagError("locale_tag_invalid")
    parts = tag.split("-")
    if not _LANGUAGE.fullmatch(parts[0]):
        raise LocaleTagError("locale_tag_invalid")
    result = [parts[0].lower()]
    index = 1
    if index < len(parts) and _SCRIPT.fullmatch(parts[index]):
        result.append(parts[index].title())
        index += 1
    if index < len(parts) and _REGION.fullmatch(parts[index]):
        result.append(parts[index].upper())
        index += 1
    seen_variants: set[str] = set()
    for part in parts[index:]:
        if not _VARIANT.fullmatch(part):
            raise LocaleTagError("locale_tag_invalid")
        variant = part.lower()
        if variant in seen_variants:
            raise LocaleTagError("locale_variant_duplicate")
        seen_variants.add(variant)
        result.append(variant)
    return "-".join(result)


def _validate_aliases(aliases: Mapping[str, str] | None) -> dict[str, str]:
    if aliases is None:
        return {}
    if not isinstance(aliases, Mapping):
        raise TypeError("aliases must be a mapping")
    normalized: dict[str, str] = {}
    for key, target in aliases.items():
        if (
            not isinstance(key, str)
            or len(key) > MAX_LOCALE_TAG_LENGTH
            or not _ALIAS_KEY.fullmatch(key)
            or not isinstance(target, str)
        ):
            raise LocaleTagError("locale_alias_invalid")
        folded = key.lower()
        if folded in normalized:
            raise LocaleTagError("locale_alias_duplicate")
        # Targets are canonical tags, never alias chains. This also prevents
        # cycles and makes results independent of mapping iteration order.
        try:
            canonical = _canonicalize(target)
        except LocaleTagError:
            pass
        else:
            normalized[folded] = canonical
            continue
        raise LocaleTagError("locale_alias_invalid")
    # A target may also be an alias key only if that alias is an identity.
    for target in normalized.values():
        alternate = normalized.get(target.lower())
        if alternate is not None and alternate != target:
            raise LocaleTagError("locale_alias_chain_unsupported")
    return normalized
