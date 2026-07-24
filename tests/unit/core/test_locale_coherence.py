"""Per-language locale + national-ID surrogate coherence suite (OM-135).

A regression guard that keeps language packs honest as the count climbs from
12 toward 20+. It asserts that:

  (a) every ``SUPPORTED_LANGUAGES`` code maps to a real Faker locale (or a
      documented approximation),
  (b) generated national-ID surrogates pass the language's registered
      validator (round-trip surrogate fidelity),
  (c) only documented approximations emit the locale ``UserWarning`` — so a new
      pack mis-wired to a wrong locale fails loudly, and
  (d) ``locale_coherence_report()`` stays a faithful per-language summary the
      status/leaderboard work can reuse.

The contract these tests gate lives in
:mod:`openmed.core.anonymizer.locales`.
"""

import json
import warnings

import pytest
from faker.config import AVAILABLE_LOCALES

from openmed.core.anonymizer import Anonymizer
from openmed.core.anonymizer import locales as L
from openmed.core.anonymizer.locales import (
    CONCEPTUAL_LOCALE_LANGUAGES,
    FAKER_BACKEND_LOCALE,
    LANG_TO_LOCALE,
    NATIONAL_ID_PROVIDERS,
    locale_coherence_report,
    resolve_faker_backend_locale,
    resolve_locale,
)
from openmed.core.anonymizer.registry import _LOCALE_ID_METHODS
from openmed.core.labels import ID_NUM, normalize_label
from openmed.core.language_pack import get_language_pack
from openmed.core.pii_entity_merger import PII_PATTERNS
from openmed.core.pii_i18n import (
    DEFAULT_PII_MODELS,
    INDIC_NER_LANGUAGES,
    LANGUAGE_MODEL_PREFIX,
    LANGUAGE_MONTH_NAMES,
    LANGUAGE_NAMES,
    LANGUAGE_PII_PATTERNS,
    LOCALE_FAKE_DATA,
    NATIONAL_ID_ONLY_LANGUAGES,
    SUPPORTED_LANGUAGES,
    validate_aadhaar,
)

# Documented set of languages whose *default* Faker locale is an intentional
# approximation. Kept here, independent of the code under test, so that wiring a
# new pack to a wrong/approximate locale flips the assertions red until a human
# consciously updates this set.
DOCUMENTED_APPROXIMATE = {
    "af",
    "am",
    "as",
    "kn",
    "ml",
    "mr",
    "ms",
    "pa",
    "rw",
    "sr",
    "te",
    "ur",
    "xh",
}

# Languages that register a national-ID validator but have no Faker
# surrogate provider yet. Documented so a *new* such gap can't slip in silently.
KNOWN_PROVIDERLESS_VALIDATORS = set()

# Number of surrogates sampled per language for the round-trip check.
SAMPLE_SIZE = 40

REPORT_LANGUAGES = (
    SUPPORTED_LANGUAGES | NATIONAL_ID_ONLY_LANGUAGES | INDIC_NER_LANGUAGES
)

CONCEPTUAL_BACKENDS = {
    "fr_SN": "fr_FR",
    "fr_CI": "fr_FR",
    "fr_CM": "fr_FR",
    "pt_MZ": "pt_PT",
    "pt_AO": "pt_PT",
}


def _languages_with_national_id_validator():
    return {
        lang
        for lang in REPORT_LANGUAGES
        if any(
            p.validator is not None and normalize_label(p.entity_type) == ID_NUM
            for p in LANGUAGE_PII_PATTERNS.get(lang, [])
        )
    }


def _national_id_validators(lang):
    """Registered national-ID checksum validators for ``lang``.

    Prefer the language pack's own ``national_id`` validators; fall back to the
    shared base SSN/national-ID validators for languages (e.g. English) whose
    national ID is validated by the base pattern set.
    """
    lang_validators = [
        p.validator
        for p in LANGUAGE_PII_PATTERNS.get(lang, [])
        if p.validator is not None and normalize_label(p.entity_type) == ID_NUM
    ]
    if lang_validators:
        return lang_validators
    return [
        p.validator
        for p in PII_PATTERNS
        if p.validator is not None and p.entity_type in ("national_id", "ssn")
    ]


class TestLocaleResolution:
    @pytest.mark.parametrize("lang", sorted(REPORT_LANGUAGES))
    def test_every_supported_language_has_locale_entry(self, lang):
        assert lang in LANG_TO_LOCALE, f"{lang!r} missing LANG_TO_LOCALE entry"

    @pytest.mark.parametrize("lang", sorted(REPORT_LANGUAGES))
    def test_resolved_locale_exists_in_faker(self, lang):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            locale = resolve_locale(lang)
        assert locale, f"{lang!r} resolved to an empty locale"
        assert locale in AVAILABLE_LOCALES or lang in L._APPROXIMATE_LOCALES, (
            f"{lang!r} -> {locale!r} is not a real Faker locale and is not a "
            "documented approximation"
        )

    def test_swahili_uses_native_faker_locale_without_warning(self):
        L._warned.clear()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            locale = resolve_locale("sw")

        assert locale == "sw"
        assert locale in AVAILABLE_LOCALES
        assert not caught
        assert "sw" not in L._APPROXIMATE_LOCALES

    def test_urdu_pack_warns_once_and_uses_bundled_names(self):
        pack = get_language_pack("ur")

        assert pack is not None
        assert pack.scripts == ("Arabic",)
        assert "ur" in SUPPORTED_LANGUAGES
        assert DEFAULT_PII_MODELS["ur"] == "OpenMed/privacy-filter-multilingual"
        assert LANGUAGE_NAMES["ur"] == "Urdu"
        assert LANGUAGE_MODEL_PREFIX["ur"] == "Urdu-"
        assert LANGUAGE_MONTH_NAMES["ur"] == [
            "جنوری",
            "فروری",
            "مارچ",
            "اپریل",
            "مئی",
            "جون",
            "جولائی",
            "اگست",
            "ستمبر",
            "اکتوبر",
            "نومبر",
            "دسمبر",
        ]
        assert LANG_TO_LOCALE["ur"] == "ur_IN"
        assert NATIONAL_ID_PROVIDERS["ur"] == ("ur_IN", "aadhaar")
        assert FAKER_BACKEND_LOCALE["ur_IN"] == "en_IN"
        assert "ur" in L._APPROXIMATE_LOCALES

        L._warned.clear()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            assert resolve_locale("ur") == "ur_IN"
            assert resolve_locale("ur") == "ur_IN"
            anonymizer = Anonymizer(lang="ur", consistent=True, seed=694)
            name = anonymizer.surrogate("جناب سیّد علی خان صاحب", "PERSON")
            aadhaar = anonymizer.surrogate(
                "۲۴۶۷ ۷۸۳۲ ۵۴۸۴",
                "national_id",
            )

        user_warnings = [
            warning for warning in caught if issubclass(warning.category, UserWarning)
        ]
        assert len(user_warnings) == 1
        assert name in {
            f"{given} {family}"
            for given in L.URDU_GIVEN_NAMES
            for family in L.URDU_FAMILY_NAMES
        }
        assert name != "جناب سیّد علی خان صاحب"
        assert validate_aadhaar(aadhaar)

    @pytest.mark.parametrize("locale", sorted(CONCEPTUAL_BACKENDS))
    def test_conceptual_locale_resolves_to_installed_backend(self, locale):
        language = CONCEPTUAL_LOCALE_LANGUAGES[locale]

        assert resolve_locale(language, locale) == locale
        assert resolve_faker_backend_locale(locale) == CONCEPTUAL_BACKENDS[locale]
        assert CONCEPTUAL_BACKENDS[locale] in AVAILABLE_LOCALES


class TestNationalIdRoundTrip:
    def test_contract_matches_registry_dispatch(self):
        """Each provider's method must match the registry's locale dispatch and
        point at a real Faker locale."""
        for lang, (locale, method) in NATIONAL_ID_PROVIDERS.items():
            backend_locale = FAKER_BACKEND_LOCALE.get(locale, locale)
            assert backend_locale in AVAILABLE_LOCALES, (
                f"{lang!r} -> unknown Faker backend locale {backend_locale!r}"
            )
            assert _LOCALE_ID_METHODS.get(locale) == method, (
                f"{lang!r} provider {method!r} disagrees with registry dispatch "
                f"for {locale!r} ({_LOCALE_ID_METHODS.get(locale)!r})"
            )

    def test_providerless_validators_are_documented(self):
        """A national-ID validator with no surrogate provider is a known gap;
        a new one must be acknowledged here rather than slip in silently."""
        without_provider = _languages_with_national_id_validator() - set(
            NATIONAL_ID_PROVIDERS
        )
        assert without_provider == KNOWN_PROVIDERLESS_VALIDATORS, (
            "languages with a national_id validator but no NATIONAL_ID_PROVIDERS "
            f"entry changed: {sorted(without_provider)}"
        )

    @pytest.mark.parametrize("lang", sorted(NATIONAL_ID_PROVIDERS))
    def test_generated_national_ids_pass_registered_validator(self, lang):
        locale, _method = NATIONAL_ID_PROVIDERS[lang]
        validators = _national_id_validators(lang)
        assert validators, f"no national-ID validator registered for {lang!r}"
        for seed in range(SAMPLE_SIZE):
            anon = Anonymizer(lang=lang, consistent=True, seed=seed)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                surrogate = anon.surrogate("123456789", "national_id", locale=locale)
            assert any(v(surrogate) for v in validators), (
                f"{lang!r} surrogate {surrogate!r} (seed={seed}, locale={locale}) "
                f"failed every registered validator "
                f"{[v.__name__ for v in validators]}"
            )


class TestApproximateLocaleWarnings:
    @pytest.mark.parametrize("lang", sorted(REPORT_LANGUAGES))
    def test_only_documented_approximations_warn(self, lang):
        L._warned.clear()  # reset the one-time-warning cache for a clean read
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            resolve_locale(lang)
        warned = any(issubclass(w.category, UserWarning) for w in caught)
        assert warned == (lang in DOCUMENTED_APPROXIMATE), (
            f"{lang!r}: emitted approximate-locale warning={warned}, "
            f"expected {lang in DOCUMENTED_APPROXIMATE}"
        )

    def test_code_approximation_set_matches_documentation(self):
        assert set(L._APPROXIMATE_LOCALES) == DOCUMENTED_APPROXIMATE

    def test_isixhosa_approximation_warns_once_and_uses_zulu_backend(self):
        L._warned.clear()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            first = resolve_locale("xh")
            second = resolve_locale("xh")

        assert first == second == "xh_ZA"
        assert FAKER_BACKEND_LOCALE[first] == "zu_ZA"
        user_warnings = [
            warning for warning in caught if issubclass(warning.category, UserWarning)
        ]
        assert len(user_warnings) == 1
        assert "xh_ZA" in str(user_warnings[0].message)
        assert "zu_ZA" in str(user_warnings[0].message)

    def test_amharic_approximation_warns_once_and_uses_documented_backend(self):
        L._warned.clear()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            first = resolve_locale("am")
            second = resolve_locale("am")

        assert first == second == "am_ET"
        assert FAKER_BACKEND_LOCALE[first] == "en_KE"
        user_warnings = [
            warning for warning in caught if issubclass(warning.category, UserWarning)
        ]
        assert len(user_warnings) == 1
        assert "am_ET" in str(user_warnings[0].message)
        assert "en_KE" in str(user_warnings[0].message)


class TestLocaleCoherenceReport:
    def test_one_row_per_supported_language(self):
        rows = locale_coherence_report()
        default_rows = [row for row in rows if row["locale"] not in CONCEPTUAL_BACKENDS]

        assert len(default_rows) == len(REPORT_LANGUAGES)
        assert {r["language"] for r in default_rows} == REPORT_LANGUAGES

    def test_row_shape_and_values(self):
        rows = {
            r["language"]: r
            for r in locale_coherence_report()
            if r["locale"] not in CONCEPTUAL_BACKENDS
        }
        for lang, row in rows.items():
            assert set(row) == {
                "language",
                "locale",
                "approximate",
                "id_providers",
                "id_types",
                "id_locale",
            }
            assert row["locale"] == LANG_TO_LOCALE[lang]
            assert isinstance(row["approximate"], bool)
            assert row["approximate"] == (lang in DOCUMENTED_APPROXIMATE)
            assert isinstance(row["id_providers"], list)
            assert isinstance(row["id_types"], list)
            if lang in NATIONAL_ID_PROVIDERS:
                exp_locale, exp_method = NATIONAL_ID_PROVIDERS[lang]
                if lang not in {"hi", "te"}:
                    assert row["id_types"] == [exp_method]
                    assert row["id_providers"] == [exp_method]
                assert row["id_locale"] == exp_locale
            else:
                assert row["id_providers"] == []
                assert row["id_types"] == []
                assert row["id_locale"] is None

    def test_conceptual_rows_include_all_backends(self):
        rows = {
            row["locale"]: row
            for row in locale_coherence_report()
            if row["locale"] in CONCEPTUAL_BACKENDS
        }

        assert set(rows) == set(CONCEPTUAL_BACKENDS)
        for locale, backend in CONCEPTUAL_BACKENDS.items():
            row = rows[locale]
            assert set(row) == {
                "language",
                "locale",
                "approximate",
                "id_providers",
                "id_types",
                "id_locale",
            }
            assert row["language"] == CONCEPTUAL_LOCALE_LANGUAGES[locale]
            assert FAKER_BACKEND_LOCALE[locale] == backend
            assert row["approximate"] is False
            assert row["id_providers"] == []
            assert row["id_types"] == []
            assert row["id_locale"] is None

    def test_report_is_json_serializable(self):
        # The status/leaderboard work serializes this; keep it JSON-friendly.
        json.dumps(locale_coherence_report())


class TestAfricanFrenchPortugueseSurrogates:
    @pytest.mark.parametrize("locale", sorted(CONCEPTUAL_BACKENDS))
    def test_curated_name_location_address_and_phone_surrogates(self, locale):
        lang = CONCEPTUAL_LOCALE_LANGUAGES[locale]
        anonymizer = Anonymizer(lang=lang, locale=locale, consistent=True, seed=866)

        for label, key in (
            ("NAME", "NAME"),
            ("LOCATION", "LOCATION"),
            ("STREET_ADDRESS", "STREET_ADDRESS"),
            ("PHONE", "PHONE"),
        ):
            surrogate = anonymizer.surrogate(f"source-{key}", label)
            assert surrogate in LOCALE_FAKE_DATA[locale][key]

    def test_default_french_and_portuguese_outputs_are_unchanged(self):
        expected = {
            "fr": {
                "NAME": "Vincent Da Silva",
                "LOCATION": "Vaillant",
                "STREET_ADDRESS": "35, chemin Ruiz",
                "PHONE": "0558229502",
            },
            "pt": {
                "NAME": "Samuel Amorim",
                "LOCATION": "Vila Real",
                "STREET_ADDRESS": "R. de Amorim, 64",
                "PHONE": "(351) 929962295",
            },
        }
        originals = {
            "NAME": "Patient Exemple",
            "LOCATION": "Ville Exemple",
            "STREET_ADDRESS": "Adresse Exemple",
            "PHONE": "contact",
        }

        for lang, values in expected.items():
            anonymizer = Anonymizer(lang=lang, consistent=True, seed=866)
            assert {
                label: anonymizer.surrogate(originals[label], label) for label in values
            } == values

    @pytest.mark.parametrize(
        ("lang", "locale", "text", "model_entities", "sweep_values"),
        [
            (
                "fr",
                "fr_SN",
                "Patiente Aïssatou Ba, domiciliée au 9 rue de Rufisque, Dakar. "
                "Téléphone: +221 77 123 45 67. CNI: 1002199012345.",
                (
                    ("Aïssatou Ba", "NAME"),
                    ("9 rue de Rufisque, Dakar", "STREET_ADDRESS"),
                ),
                ("+221 77 123 45 67", "1002199012345"),
            ),
            (
                "pt",
                "pt_MZ",
                "Paciente Lúcia Matola, residente na 7 Avenida Samora Machel, "
                "Maputo. Telefone: +258 84 123 4567.",
                (
                    ("Lúcia Matola", "NAME"),
                    ("7 Avenida Samora Machel, Maputo", "STREET_ADDRESS"),
                ),
                ("+258 84 123 4567",),
            ),
        ],
    )
    def test_synthetic_clinical_fixtures_have_zero_leakage(
        self,
        lang,
        locale,
        text,
        model_entities,
        sweep_values,
    ):
        from openmed.core.pii import (
            _apply_safety_sweep_to_result,
            _build_deidentification_result,
        )
        from openmed.processing.outputs import EntityPrediction, PredictionResult

        entities = []
        for surface, label in model_entities:
            start = text.index(surface)
            entities.append(
                EntityPrediction(
                    text=surface,
                    label=label,
                    start=start,
                    end=start + len(surface),
                    confidence=0.99,
                )
            )
        prediction = PredictionResult(
            text=text,
            entities=entities,
            model_name="synthetic-fixture",
            timestamp="2026-07-18T00:00:00Z",
            metadata={"synthetic": True},
        )

        swept, added_count = _apply_safety_sweep_to_result(
            text,
            prediction,
            lang=lang,
            locale=locale,
        )
        result = _build_deidentification_result(
            text,
            swept,
            effective_method="replace",
            keep_year=False,
            date_shift_days=None,
            keep_mapping=False,
            lang=lang,
            consistent=True,
            seed=866,
            locale=locale,
            use_safety_sweep=True,
        )

        protected_values = [surface for surface, _label in model_entities]
        protected_values.extend(sweep_values)
        assert added_count == len(sweep_values)
        assert all(value not in result.deidentified_text for value in protected_values)
