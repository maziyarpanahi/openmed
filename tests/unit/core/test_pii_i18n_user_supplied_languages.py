"""Public-registration contract for the Indic and Urdu routing candidates.

These languages are registered on every public surface (display name, model
prefix, surrogate locale, REST/MCP enums) but deliberately claim no bundled
default PII model. A caller that supplies ``model_name`` must be accepted; a
caller that omits it must get an actionable error naming the missing model.
"""

from __future__ import annotations

import warnings

import pytest

from openmed.core.anonymizer.locales import resolve_locale
from openmed.core.language_pack_catalog import (
    DEFAULT_PII_MODELS as BUILTIN_DEFAULT_PII_MODELS,
)
from openmed.core.language_pack_catalog import LANG_TO_LOCALE
from openmed.core.model_registry import get_default_pii_model
from openmed.core.pii import _resolve_effective_pii_model
from openmed.core.pii_i18n import (
    DEFAULT_PII_MODELS,
    INDIC_NER_LANGUAGES,
    LANGUAGE_MODEL_PREFIX,
    LANGUAGE_NAMES,
    OPTIONAL_PII_MODEL,
    SUPPORTED_LANGUAGES,
    USER_SUPPLIED_MODEL_LANGUAGES,
    USER_SUPPLIED_PII_MODEL,
    get_patterns_for_language,
)
from openmed.core.pipeline import Pipeline

# The eleven Indic/Urdu routing candidates covered by the parent epic.
INDIC_AND_URDU_LANGUAGES = (
    "as",
    "bn",
    "gu",
    "kn",
    "ml",
    "mr",
    "ne",
    "or",
    "pa",
    "ta",
    "ur",
)

# Codes that ship no bundled weights at all (no pack, no Indic NER adapter).
# Derived exactly as ``openmed.core.pii`` derives it, so the split stays honest
# if a route later gains an adapter.
NO_WEIGHTS_LANGUAGES = tuple(
    sorted(USER_SUPPLIED_MODEL_LANGUAGES - INDIC_NER_LANGUAGES)
)
# User-supplied codes that DO have the optional Indic NER env-var path.
OPTIONAL_ADAPTER_LANGUAGES = tuple(
    sorted(USER_SUPPLIED_MODEL_LANGUAGES & INDIC_NER_LANGUAGES)
)

_DEFAULT_EN_MODEL = DEFAULT_PII_MODELS["en"]
_EXPLICIT_MODEL = "OpenMed/privacy-filter-multilingual"


@pytest.mark.parametrize("lang", INDIC_AND_URDU_LANGUAGES)
def test_every_code_is_accepted_when_the_caller_supplies_a_model(lang: str) -> None:
    assert _resolve_effective_pii_model(_EXPLICIT_MODEL, lang) == _EXPLICIT_MODEL


@pytest.mark.parametrize("lang", INDIC_AND_URDU_LANGUAGES)
def test_every_code_is_registered_on_the_public_display_surfaces(lang: str) -> None:
    assert LANGUAGE_NAMES[lang]
    assert LANGUAGE_MODEL_PREFIX[lang] == f"{LANGUAGE_NAMES[lang]}-"
    assert LANG_TO_LOCALE[lang]
    assert DEFAULT_PII_MODELS[lang]


@pytest.mark.parametrize("lang", INDIC_AND_URDU_LANGUAGES)
def test_every_code_resolves_deterministic_patterns(lang: str) -> None:
    assert get_patterns_for_language(lang)


@pytest.mark.parametrize("lang", INDIC_AND_URDU_LANGUAGES)
def test_pipeline_accepts_every_code_with_an_explicit_model(lang: str) -> None:
    pipeline = Pipeline(lang=lang, model_name=_EXPLICIT_MODEL)
    route = pipeline.stage2_language_script("MRN 4821")

    assert route.lang == lang
    assert route.model_name == _EXPLICIT_MODEL


@pytest.mark.parametrize("lang", sorted(USER_SUPPLIED_MODEL_LANGUAGES))
def test_user_supplied_languages_claim_no_bundled_default(lang: str) -> None:
    assert lang not in SUPPORTED_LANGUAGES
    assert lang not in BUILTIN_DEFAULT_PII_MODELS
    assert DEFAULT_PII_MODELS[lang] in {OPTIONAL_PII_MODEL, USER_SUPPLIED_PII_MODEL}


@pytest.mark.parametrize("lang", NO_WEIGHTS_LANGUAGES)
def test_languages_without_weights_resolve_to_no_default_model(lang: str) -> None:
    assert DEFAULT_PII_MODELS[lang] == USER_SUPPLIED_PII_MODEL
    assert get_default_pii_model(lang) is None


@pytest.mark.parametrize("lang", NO_WEIGHTS_LANGUAGES)
def test_missing_model_raises_an_actionable_error(lang: str) -> None:
    with pytest.raises(ValueError, match="pass an explicit model_name") as excinfo:
        _resolve_effective_pii_model(_DEFAULT_EN_MODEL, lang)

    message = str(excinfo.value)
    assert "no bundled OpenMed PII model" in message
    # The error enumerates exactly the codes that take this branch. Listing the
    # optional-adapter codes here would send the reader to a different error.
    for code in NO_WEIGHTS_LANGUAGES:
        assert code in message
    for code in OPTIONAL_ADAPTER_LANGUAGES:
        assert f" {code}," not in message and not message.endswith(f" {code}")


@pytest.mark.parametrize("lang", OPTIONAL_ADAPTER_LANGUAGES)
def test_optional_indic_routes_keep_their_env_var_guidance(
    lang: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENMED_INDIC_NER_MODEL", raising=False)

    with pytest.raises(ValueError, match="OPENMED_INDIC_NER_MODEL"):
        _resolve_effective_pii_model(_DEFAULT_EN_MODEL, lang)


def test_existing_hindi_and_telugu_defaults_are_unchanged() -> None:
    assert (
        DEFAULT_PII_MODELS["hi"]
        == "OpenMed/OpenMed-PII-Hindi-SuperClinical-Large-434M-v1"
    )
    assert (
        DEFAULT_PII_MODELS["te"]
        == "OpenMed/OpenMed-PII-Telugu-SuperClinical-Large-434M-v1"
    )
    assert {"hi", "te"} <= SUPPORTED_LANGUAGES
    assert {"hi", "te"}.isdisjoint(USER_SUPPLIED_MODEL_LANGUAGES)
    assert (
        _resolve_effective_pii_model(_DEFAULT_EN_MODEL, "hi")
        == DEFAULT_PII_MODELS["hi"]
    )
    assert (
        _resolve_effective_pii_model(_DEFAULT_EN_MODEL, "te")
        == DEFAULT_PII_MODELS["te"]
    )


def test_registering_these_codes_does_not_grow_the_model_backed_registry() -> None:
    # Model-backed language claims (README/docs/brand counts) key off
    # SUPPORTED_LANGUAGES. Registering a user-supplied route must never add to
    # it, whatever the maintainer's current pack count happens to be.
    assert USER_SUPPLIED_MODEL_LANGUAGES.isdisjoint(SUPPORTED_LANGUAGES)
    assert USER_SUPPLIED_MODEL_LANGUAGES.isdisjoint(BUILTIN_DEFAULT_PII_MODELS)
    # Every code this module exercises is registered somewhere public.
    assert set(INDIC_AND_URDU_LANGUAGES) <= (
        SUPPORTED_LANGUAGES | USER_SUPPLIED_MODEL_LANGUAGES
    )


@pytest.mark.parametrize("lang", sorted(USER_SUPPLIED_MODEL_LANGUAGES))
@pytest.mark.parametrize("sentinel", (OPTIONAL_PII_MODEL, USER_SUPPLIED_PII_MODEL))
def test_echoing_a_registry_placeholder_back_is_rejected(
    lang: str,
    sentinel: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A value copied from ``openmed_list_pii_languages`` must not be loadable.

    ``"user-supplied"`` passes ``validate_model_name`` and would otherwise be
    namespaced to ``OpenMed/user-supplied`` and fetched over the network.
    """
    monkeypatch.delenv("OPENMED_INDIC_NER_MODEL", raising=False)

    with pytest.raises(ValueError, match="model_name") as excinfo:
        _resolve_effective_pii_model(sentinel, lang)

    message = str(excinfo.value)
    # The caller gets the language's own guidance, not a download failure.
    assert "OpenMed/" not in message
    assert "Invalid characters" not in message


@pytest.mark.parametrize("sentinel", (OPTIONAL_PII_MODEL, USER_SUPPLIED_PII_MODEL))
def test_placeholder_is_rejected_even_for_a_model_backed_language(
    sentinel: str,
) -> None:
    with pytest.raises(ValueError, match="registry placeholder") as excinfo:
        _resolve_effective_pii_model(sentinel, "hi")

    assert "OpenMed/" not in str(excinfo.value)


@pytest.mark.parametrize("lang", NO_WEIGHTS_LANGUAGES)
def test_route_metadata_reports_no_available_default_model(lang: str) -> None:
    """Route metadata must not publish a sentinel as an available model."""
    pipeline = Pipeline(lang=lang, model_name=_EXPLICIT_MODEL)
    route = pipeline.stage2_language_script("MRN 4821")

    assert route.metadata["available_default_model"] is None
    assert route.metadata["available_default_model"] == get_default_pii_model(lang)


def test_nepali_uses_its_native_faker_locale_without_warning() -> None:
    from faker.config import AVAILABLE_LOCALES

    from openmed.core.anonymizer import locales as locales_module

    locales_module._warned.discard("ne")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        locale = resolve_locale("ne")

    assert locale == "ne_NP"
    assert locale in AVAILABLE_LOCALES
    assert not caught
    assert "ne" not in locales_module._APPROXIMATE_LOCALES
    # A native locale needs no conceptual Faker backend mapping.
    assert "ne_NP" not in locales_module.FAKER_BACKEND_LOCALE


def test_no_national_id_provider_is_invented_for_nepali() -> None:
    from openmed.core.language_pack_catalog import NATIONAL_ID_PROVIDERS

    assert "ne" not in NATIONAL_ID_PROVIDERS


def test_urdu_keeps_its_existing_cnic_provider_and_locale() -> None:
    from openmed.core.language_pack_catalog import (
        NATIONAL_ID_ONLY_CAPABILITIES,
        NATIONAL_ID_PROVIDERS,
    )

    assert NATIONAL_ID_PROVIDERS["ur"] == ("ur_PK", "cnic")
    assert NATIONAL_ID_ONLY_CAPABILITIES["ur"].locale == "ur_PK"
    assert LANG_TO_LOCALE["ur"] == "ur_PK"


def test_locale_coherence_report_covers_every_user_supplied_language() -> None:
    from openmed.core.anonymizer.locales import locale_coherence_report

    reported = {row["language"] for row in locale_coherence_report()}

    assert USER_SUPPLIED_MODEL_LANGUAGES <= reported
