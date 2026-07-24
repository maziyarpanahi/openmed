"""Golden and live-registration tests for language-pack map adapters."""

from __future__ import annotations

import json
from pathlib import Path

from openmed.core.language_pack import LanguagePack, LanguagePackRegistry
from openmed.core.language_pack_catalog import (
    DEFAULT_MODEL_PLACEHOLDER_LANGUAGES,
    DEFAULT_PII_MODELS,
    LANG_TO_LOCALE,
    NATIONAL_ID_ONLY_LANGUAGES,
    NATIONAL_ID_PROVIDERS,
    SCRIPT_LANGUAGE_HINTS,
    SUPPORTED_LANGUAGES,
    LanguagePackAdapters,
    is_registered_segmenter,
)

SNAPSHOT_PATH = (
    Path(__file__).resolve().parents[2] / "fixtures" / "language_pack_adapters.json"
)


def _snapshot() -> dict[str, object]:
    return {
        "default_pii_models": dict(DEFAULT_PII_MODELS),
        "lang_to_locale": dict(LANG_TO_LOCALE),
        "national_id_only_languages": sorted(NATIONAL_ID_ONLY_LANGUAGES),
        "national_id_providers": {
            key: list(value) for key, value in NATIONAL_ID_PROVIDERS.items()
        },
        "script_language_hints": {
            key: list(value) for key, value in SCRIPT_LANGUAGE_HINTS.items()
        },
        "supported_languages": sorted(SUPPORTED_LANGUAGES),
    }


def test_registry_adapters_match_committed_snapshot_byte_for_byte() -> None:
    rendered = json.dumps(_snapshot(), indent=2, sort_keys=True) + "\n"
    assert rendered == SNAPSHOT_PATH.read_text(encoding="utf-8")


def test_public_adapter_runtime_types_remain_compatible() -> None:
    assert type(SUPPORTED_LANGUAGES) is set
    assert type(DEFAULT_PII_MODELS) is dict
    assert type(LANG_TO_LOCALE) is dict
    assert type(NATIONAL_ID_PROVIDERS) is dict
    assert type(SCRIPT_LANGUAGE_HINTS) is dict

    assert "en" in SUPPORTED_LANGUAGES
    assert DEFAULT_PII_MODELS.get("en")
    assert set(DEFAULT_PII_MODELS) == SUPPORTED_LANGUAGES
    assert SUPPORTED_LANGUAGES | {"xx"} == set(SUPPORTED_LANGUAGES) | {"xx"}


def test_one_registration_updates_every_downstream_adapter() -> None:
    registry = LanguagePackRegistry()
    adapters = LanguagePackAdapters(registry)
    pack = LanguagePack(
        code="xx",
        scripts=("Synthetic",),
        default_model="OpenMed/synthetic-pii",
        segmenter_id="unicode-sentence",
        recognizers=("synthetic-regex", "synthetic-model"),
        surrogate_locale="en_US",
        national_id_providers={"ssn": "en_US"},
    )

    registry.register(pack)

    assert adapters.supported_languages == {"xx"}
    assert adapters.default_pii_models == {"xx": "OpenMed/synthetic-pii"}
    assert adapters.lang_to_locale == {"xx": "en_US"}
    assert adapters.national_id_providers == {"xx": ("en_US", "ssn")}
    assert adapters.script_language_hints == {"Synthetic": ("xx",)}


def test_replacement_refreshes_existing_adapter_objects_in_place() -> None:
    registry = LanguagePackRegistry()
    adapters = LanguagePackAdapters(registry)
    models = adapters.default_pii_models
    locales = adapters.lang_to_locale
    registry.register(
        LanguagePack(
            code="xx",
            scripts=("Synthetic",),
            default_model="OpenMed/first",
            segmenter_id="pysbd",
            recognizers=("regex",),
            surrogate_locale="en_US",
        )
    )

    registry.register(
        LanguagePack(
            code="xx",
            scripts=("Synthetic",),
            default_model="OpenMed/replacement",
            segmenter_id="pysbd",
            recognizers=("regex",),
            surrogate_locale="en_GB",
        ),
        replace=True,
    )

    assert adapters.default_pii_models is models
    assert adapters.lang_to_locale is locales
    assert models["xx"] == "OpenMed/replacement"
    assert locales["xx"] == "en_GB"


def test_segmenter_resolver_rejects_undeclared_ids() -> None:
    assert is_registered_segmenter("jieba")
    assert is_registered_segmenter("pysbd")
    assert is_registered_segmenter("unicode-sentence")
    assert not is_registered_segmenter("no-such-segmenter")


def test_language_packs_keep_placeholder_models_explicit() -> None:
    assert DEFAULT_MODEL_PLACEHOLDER_LANGUAGES == {"ru", "zh"}
    assert DEFAULT_PII_MODELS["ru"] == "OpenMed/privacy-filter-multilingual"
    assert DEFAULT_PII_MODELS["zh"] == "OpenMed/privacy-filter-multilingual"
    assert SCRIPT_LANGUAGE_HINTS["Cyrillic"] == ("ru", "uk")
    assert SCRIPT_LANGUAGE_HINTS["Han"] == ("zh", "ja")
