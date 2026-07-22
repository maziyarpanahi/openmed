"""Tests for the locale national-ID provider registry."""

from __future__ import annotations

import pytest
from faker import Faker

from openmed.core.anonymizer.locales import FAKER_BACKEND_LOCALE
from openmed.core.anonymizer.providers import registry_ids
from openmed.core.anonymizer.providers.clinical_ids import (
    AfricanPhoneProvider,
    register_clinical_providers,
)
from openmed.core.anonymizer.providers.registry_ids import (
    ID_PROVIDER_REGISTRY,
    NationalIdSpec,
    clinical_faker_provider_classes,
    get_national_id,
    register_national_id,
)


def _always_valid(_text: str) -> bool:
    return True


EXPECTED_VALIDATOR_KEYS = (
    ("sw", "mpesa_tx_code"),
    ("tz", "nida_nin"),
    ("ug", "nin"),
    ("rw", "rwanda_id"),
    ("et", "fayda_fan"),
    ("fr", "nir"),
    ("de", "steuer_id"),
    ("it", "codice_fiscale"),
    ("es", "dni"),
    ("es", "nie"),
    ("nl", "bsn"),
    ("in", "aadhaar"),
    ("zh", "resident_id"),
    ("in", "pan"),
    ("in", "gstin"),
    ("in", "abha"),
    ("id", "nik"),
    ("ms", "mykad"),
    ("tl", "philsys_psn"),
    ("tl", "philhealth_pin"),
    ("da", "cpr"),
    ("th", "thai_national_id"),
    ("he", "teudat_zehut"),
    ("pl", "pesel"),
    ("lv", "personas_kods"),
    ("ko", "rrn"),
    ("hu", "taj"),
    ("cs", "rodne_cislo"),
    ("sk", "rodne_cislo"),
    ("pt", "cpf"),
    ("pt", "cnpj"),
    ("tr", "tckn"),
    ("vi", "cccd"),
    ("vi", "cmnd"),
    ("us", "npi"),
    ("ng", "nin"),
    ("ng", "bvn"),
    ("gh", "ghana_card_pin"),
    ("ke", "kenya_national_id"),
    ("ke", "maisha_namba"),
    ("za", "sa_id_number"),
    ("eg", "egyptian_national_id"),
    ("ma", "moroccan_cin"),
)


ROUND_TRIP_CASES = (
    ("tz", "nida_nin", "sw"),
    ("ug", "nin", "en_UG"),
    ("rw", "rwanda_id", "rw_RW"),
    ("et", "fayda_fan", "am_ET"),
    ("fr", "nir", "fr_FR"),
    ("de", "steuer_id", "de_DE"),
    ("it", "codice_fiscale", "it_IT"),
    ("es", "dni", "es_ES"),
    ("es", "nie", "es_ES"),
    ("nl", "bsn", "nl_NL"),
    ("in", "aadhaar", "en_IN"),
    ("zh", "resident_id", "zh_CN"),
    ("in", "pan", "en_IN"),
    ("in", "gstin", "en_IN"),
    ("in", "abha", "en_IN"),
    ("id", "nik", "id_ID"),
    ("ms", "mykad", "ms_MY"),
    ("tl", "philsys_psn", "fil_PH"),
    ("tl", "philhealth_pin", "fil_PH"),
    ("da", "cpr", "da_DK"),
    ("th", "thai_national_id", "th_TH"),
    ("he", "teudat_zehut", "he_IL"),
    ("pl", "pesel", "pl_PL"),
    ("lv", "personas_kods", "lv_LV"),
    ("ko", "rrn", "ko_KR"),
    ("hu", "taj", "hu_HU"),
    ("cs", "rodne_cislo", "cs_CZ"),
    ("sk", "rodne_cislo", "sk_SK"),
    ("pt", "cpf", "pt_BR"),
    ("pt", "cnpj", "pt_BR"),
    ("tr", "tckn", "tr_TR"),
    ("vi", "cccd", "vi_VN"),
    ("vi", "cmnd", "vi_VN"),
    ("us", "npi", "en_US"),
    ("ng", "nin", "en_NG"),
    ("ng", "bvn", "en_NG"),
    ("gh", "ghana_card_pin", "en_US"),
    ("ke", "kenya_national_id", "en_KE"),
    ("ke", "maisha_namba", "en_KE"),
    ("za", "sa_id_number", "zu_ZA"),
    ("eg", "egyptian_national_id", "ar_EG"),
    ("ma", "moroccan_cin", "ar_MA"),
)


class TestNationalIdRegistry:
    @pytest.mark.parametrize(
        ("alias", "id_type"),
        (
            ("tz", "nida_nin"),
            ("sw", "nida_nin"),
            ("en_TZ", "nida_nin"),
            ("ug", "nin"),
            ("en_UG", "nin"),
            ("rw", "rwanda_id"),
            ("rw_RW", "rwanda_id"),
            ("et", "fayda_fan"),
            ("am", "fayda_fan"),
            ("en_ET", "fayda_fan"),
        ),
    )
    def test_east_african_aliases_resolve(self, alias, id_type):
        spec = get_national_id(alias, id_type)
        assert spec is not None
        assert spec.faker_provider is not None

    def test_acceptance_lookups_return_validating_specs(self):
        for lang, id_type, locale in (
            ("it", "codice_fiscale", "it_IT"),
            ("in", "aadhaar", "en_IN"),
        ):
            spec = get_national_id(lang, id_type)
            assert spec is not None

            faker = Faker(locale)
            register_clinical_providers(faker)
            surrogate = getattr(faker, spec.faker_method)()

            assert spec.validate(surrogate), (
                f"{lang!r}/{id_type!r} generated invalid surrogate {surrogate!r}"
            )

    @pytest.mark.parametrize(("lang", "id_type"), EXPECTED_VALIDATOR_KEYS)
    def test_pre_existing_validators_are_reachable(self, lang, id_type):
        spec = get_national_id(lang, id_type)
        assert spec is not None
        assert spec.validate is not None
        assert spec.faker_method

    @pytest.mark.parametrize(("lang", "id_type", "locale"), ROUND_TRIP_CASES)
    def test_registered_surrogates_pass_registered_validator(
        self,
        lang,
        id_type,
        locale,
    ):
        spec = get_national_id(lang, id_type)
        assert spec is not None

        faker = Faker(FAKER_BACKEND_LOCALE.get(locale, locale))
        register_clinical_providers(faker)
        faker.seed_instance(42)
        surrogate = getattr(faker, spec.faker_method)()

        assert spec.validate(surrogate), (
            f"{lang!r}/{id_type!r} generated invalid surrogate {surrogate!r}"
        )

    def test_lookup_normalizes_case_hyphens_and_locale(self):
        assert get_national_id("IT-it", "Codice Fiscale") == get_national_id(
            "it_IT",
            "codice_fiscale",
        )
        assert get_national_id("EN-in", "Aadhaar") == get_national_id(
            "en_IN",
            "aadhaar",
        )

    def test_south_african_aliases_resolve_matching_specs(self):
        aliases = ("za", "en_ZA", "af", "af_ZA", "zu", "zu_ZA", "xh", "xh_ZA")
        specs = [get_national_id(alias, "sa_id_number") for alias in aliases]

        assert all(spec is not None for spec in specs)
        assert len({spec.validate for spec in specs if spec is not None}) == 1
        assert {spec.faker_method for spec in specs if spec is not None} == {
            "south_african_id"
        }

    @pytest.mark.parametrize(
        ("aliases", "id_type", "faker_method"),
        (
            (
                ("eg", "ar", "ar_EG"),
                "egyptian_national_id",
                "egyptian_national_id",
            ),
            (("ma", "ar_MA", "fr_MA"), "moroccan_cin", "moroccan_cin"),
        ),
    )
    def test_egypt_morocco_aliases_resolve_working_specs(
        self,
        aliases,
        id_type,
        faker_method,
    ):
        specs = [get_national_id(alias, id_type) for alias in aliases]
        resolved_specs = [spec for spec in specs if spec is not None]
        assert len(resolved_specs) == len(aliases)
        assert {spec.validate for spec in resolved_specs} == {
            resolved_specs[0].validate
        }
        assert {spec.faker_method for spec in resolved_specs} == {faker_method}

        faker = Faker("ar_EG")
        register_clinical_providers(faker)
        faker.seed_instance(842)
        surrogate = getattr(faker, faker_method)()
        assert resolved_specs[0].validate(surrogate)

    @pytest.mark.parametrize("id_type", ("nin", "bvn"))
    def test_nigerian_aliases_resolve_working_faker_methods(self, id_type):
        specs = [
            get_national_id(alias, id_type)
            for alias in ("ng", "en_NG", "ha", "ig", "yo")
        ]
        assert all(spec is not None for spec in specs)
        assert {spec.validate for spec in specs} == {specs[0].validate}

        faker = Faker("en_NG")
        register_clinical_providers(faker)
        faker.seed_instance(840)
        surrogate = getattr(faker, specs[0].faker_method)()
        assert specs[0].validate(surrogate)

    @pytest.mark.parametrize(
        ("aliases", "id_type", "faker_method"),
        (
            (("gh", "en_GH"), "ghana_card_pin", "ghana_card_pin"),
            (("ke", "en_KE", "sw"), "kenya_national_id", "kenya_national_id"),
            (("ke", "en_KE", "sw"), "maisha_namba", "kenya_maisha_namba"),
        ),
    )
    def test_ghana_kenya_aliases_resolve_working_specs(
        self,
        aliases,
        id_type,
        faker_method,
    ):
        specs = [get_national_id(alias, id_type) for alias in aliases]
        assert all(spec is not None for spec in specs)
        assert {spec.validate for spec in specs} == {specs[0].validate}
        assert {spec.faker_method for spec in specs} == {faker_method}

        faker = Faker("sw" if "sw" in aliases else "en_US")
        register_clinical_providers(faker)
        faker.seed_instance(841)
        surrogate = getattr(faker, faker_method)()
        assert specs[0].validate(surrogate)

    def test_unknown_lookup_returns_none(self):
        assert get_national_id("zz", "unknown") is None

    @pytest.mark.parametrize("alias", ("ke", "tz", "sw", "en_ke", "en_tz"))
    def test_mpesa_transaction_code_aliases_generate_valid_surrogates(self, alias):
        spec = get_national_id(alias, "mpesa_tx_code")
        assert spec is not None

        faker = Faker("en_US")
        register_clinical_providers(faker)
        faker.seed_instance(859)
        surrogate = getattr(faker, spec.faker_method)("TB17CVOCY9")

        assert surrogate.startswith("T")
        assert surrogate != "TB17CVOCY9"
        assert spec.validate(surrogate)

    def test_duplicate_registration_rejects_key_with_clear_error(self):
        key = ("zz", "dummy")
        ID_PROVIDER_REGISTRY.pop(key, None)
        spec = NationalIdSpec("zz", "dummy", _always_valid, "word")

        try:
            register_national_id(spec)
            with pytest.raises(ValueError, match="already registered.*zz.*dummy"):
                register_national_id(spec)
        finally:
            ID_PROVIDER_REGISTRY.pop(key, None)

    def test_module_docstring_documents_language_pack_steps(self):
        doc = registry_ids.__doc__
        assert doc is not None
        for expected in (
            "Implement or import a deterministic validator",
            "Ensure Faker can generate matching surrogates",
            "Register a NationalIdSpec",
            "Register every needed alias explicitly",
            "Add unit tests",
        ):
            assert expected in doc


class TestAfricanPhoneProviderRegistry:
    def test_provider_is_registered_once(self):
        provider_classes = clinical_faker_provider_classes()

        assert provider_classes.count(AfricanPhoneProvider) == 1

    @pytest.mark.parametrize(
        ("original", "preserved_digits"),
        [
            ("+251 91 234 5678", "25191"),
            ("00250 78 123 4567", "0025078"),
            ("0752 876 543", "0752"),
        ],
    )
    def test_registered_provider_preserves_country_and_operator_prefix(
        self,
        original,
        preserved_digits,
    ):
        faker = Faker("en_US")
        register_clinical_providers(faker)
        faker.seed_instance(858)

        surrogate = faker.african_phone(original)

        assert surrogate is not None
        assert "".join(char for char in surrogate if char.isdigit()).startswith(
            preserved_digits
        )
        assert surrogate != original
        assert "".join(char for char in surrogate if not char.isdigit()) == "".join(
            char for char in original if not char.isdigit()
        )

    def test_seeded_provider_is_deterministic(self):
        first = Faker("en_US")
        second = Faker("en_US")
        register_clinical_providers(first)
        register_clinical_providers(second)
        first.seed_instance(858)
        second.seed_instance(858)

        assert first.african_phone("+233 24 123 4567") == second.african_phone(
            "+233 24 123 4567"
        )
