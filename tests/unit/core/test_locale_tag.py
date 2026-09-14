"""Offline structural and registry workflow tests for locale tags."""

from __future__ import annotations

import json
from types import MappingProxyType

import pytest

from openmed.core.locale_tag import (
    MAX_LOCALE_TAG_LENGTH,
    LocaleTagError,
    normalize_locale_tag,
    normalize_locale_tags,
)


@pytest.mark.parametrize(
    "tag,expected",
    [
        ("en", "en"),
        ("EN", "en"),
        ("eng", "eng"),
        ("ZH-hANT", "zh-Hant"),
        ("EN-us", "en-US"),
        ("es-419", "es-419"),
        ("SR-latn-rs", "sr-Latn-RS"),
        ("de-1901", "de-1901"),
        ("sl-ROZAJ-BISKE-1994", "sl-rozaj-biske-1994"),
        ("en-Latn-US-oxendict", "en-Latn-US-oxendict"),
        ("abcd", "abcd"),
        ("abcdefgh", "abcdefgh"),
    ],
)
def test_supported_structural_subset_and_casing(tag, expected) -> None:
    assert normalize_locale_tag(tag) == expected
    assert normalize_locale_tag(expected) == expected


@pytest.mark.parametrize(
    "tag",
    [
        "",
        "e",
        "abcdefghi",
        "12",
        "en_Us",
        " en",
        "en ",
        "en\n",
        "en--US",
        "en-",
        "-en",
        "en-USA",
        "en-12",
        "en-1234a5678",
        "en-Latn-Latn",
        "en-US-Latn",
        "en-a123",
        "en-Latn-ABCD",
        "en-u-ca-gregory",
        "x-private",
        "i-klingon",
        "zh-cmn-Hans",
        "ｅｎ",
        "en-١٢٣",
        "en-ÅBCD",
        "en-US/../../fixture",
        "\x00en",
    ],
)
def test_malformed_and_out_of_subset_tags_are_rejected(tag) -> None:
    with pytest.raises(LocaleTagError) as raised:
        normalize_locale_tag(tag)
    assert raised.value.category == "locale_tag_invalid"
    assert str(raised.value) == "locale_tag_invalid"
    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None


@pytest.mark.parametrize("tag", ["sl-rozaj-ROZAJ", "de-1901-1901"])
def test_repeated_variants_are_rejected(tag) -> None:
    with pytest.raises(LocaleTagError, match="^locale_variant_duplicate$"):
        normalize_locale_tag(tag)


def test_length_bound() -> None:
    assert MAX_LOCALE_TAG_LENGTH == 255
    with pytest.raises(LocaleTagError, match="^locale_tag_invalid$"):
        normalize_locale_tag("en-" + "a" * 253)


def test_unknown_but_structurally_valid_language_is_not_claimed_as_registered() -> None:
    assert normalize_locale_tag("ZZ-Latn-999") == "zz-Latn-999"
    assert normalize_locale_tag("iw") == "iw"  # No implicit language aliases.


def test_explicit_case_insensitive_aliases_are_not_mutated() -> None:
    aliases = MappingProxyType({"iw": "HE", "en_US": "en-us", "i-klingon": "tlh"})
    assert normalize_locale_tag("IW", aliases=aliases) == "he"
    assert normalize_locale_tag("EN_us", aliases=aliases) == "en-US"
    assert normalize_locale_tag("I-KLINGON", aliases=aliases) == "tlh"
    assert dict(aliases) == {"iw": "HE", "en_US": "en-us", "i-klingon": "tlh"}


def test_identity_alias_is_idempotent() -> None:
    aliases = {"en-us": "EN-US", "english": "en-us"}
    result = normalize_locale_tag("english", aliases=aliases)
    assert result == "en-US"
    assert normalize_locale_tag(result, aliases=aliases) == result


@pytest.mark.parametrize(
    "aliases,category",
    [
        ({"": "en"}, "locale_alias_invalid"),
        ({"bad key": "en"}, "locale_alias_invalid"),
        ({"é": "en"}, "locale_alias_invalid"),
        ({"a" * 256: "en"}, "locale_alias_invalid"),
        ({1: "en"}, "locale_alias_invalid"),
        ({"alias": None}, "locale_alias_invalid"),
        ({"alias": ""}, "locale_alias_invalid"),
        ({"alias": "a" * 256}, "locale_alias_invalid"),
        ({"alias": "en_US"}, "locale_alias_invalid"),
        ({"alias": "en\n"}, "locale_alias_invalid"),
        ({"alias": "en-rozaj-rozaj"}, "locale_alias_invalid"),
        ({"alias": "en", "ALIAS": "en"}, "locale_alias_duplicate"),
        ({"old": "en", "en": "fr"}, "locale_alias_chain_unsupported"),
        ({"en": "fr", "fr": "en"}, "locale_alias_chain_unsupported"),
    ],
)
def test_malformed_conflicting_and_chained_aliases(aliases, category) -> None:
    with pytest.raises(LocaleTagError) as raised:
        normalize_locale_tag("en", aliases=aliases)
    assert str(raised.value) == category
    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None


@pytest.mark.parametrize(
    "tags,aliases",
    [
        (["en-US", "EN-us"], None),
        (["iw", "he"], {"iw": "he"}),
        (["legacy", "old"], {"legacy": "en", "old": "en"}),
    ],
)
def test_duplicate_canonical_registry_entries(tags, aliases) -> None:
    with pytest.raises(LocaleTagError, match="^locale_tag_duplicate$"):
        normalize_locale_tags(tags, aliases=aliases)


def test_empty_and_generator_registries_preserve_order() -> None:
    assert normalize_locale_tags([]) == ()
    tags = (tag for tag in ["ZH-hant", "en-US", "es-419"])
    assert normalize_locale_tags(tags) == ("zh-Hant", "en-US", "es-419")


@pytest.mark.parametrize("tag", [None, b"en", True, 2, ["en"]])
def test_nonstring_tag_types(tag) -> None:
    with pytest.raises(TypeError, match="^tag must be a string$"):
        normalize_locale_tag(tag)


@pytest.mark.parametrize("tags", ["en", b"en"])
def test_registry_does_not_iterate_characters(tags) -> None:
    with pytest.raises(TypeError, match="^tags must be an iterable of locale tags$"):
        normalize_locale_tags(tags)


def test_aliases_must_be_mapping() -> None:
    with pytest.raises(TypeError, match="^aliases must be a mapping$"):
        normalize_locale_tag("en", aliases=[("old", "en")])


@pytest.mark.integration
def test_json_registry_round_trip_end_to_end(tmp_path) -> None:
    source = tmp_path / "synthetic-registry.json"
    output = tmp_path / "normalized-registry.json"
    source.write_text(
        json.dumps(
            {
                "locales": ["EN_us", "SR-latn-rs", "es-419"],
                "aliases": {"en_US": "en-US"},
            }
        ),
        encoding="utf-8",
    )
    raw = json.loads(source.read_text(encoding="utf-8"))
    result = normalize_locale_tags(raw["locales"], aliases=raw["aliases"])
    output.write_text(json.dumps({"locales": result}, sort_keys=True), encoding="utf-8")
    assert json.loads(output.read_text(encoding="utf-8")) == {
        "locales": ["en-US", "sr-Latn-RS", "es-419"]
    }
    assert normalize_locale_tags(result, aliases=raw["aliases"]) == result
