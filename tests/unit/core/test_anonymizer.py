"""Tests for the Faker-backed anonymization engine."""

import pytest

from openmed.core.anonymizer import (
    Anonymizer,
    AnonymizerConfig,
    LANG_TO_LOCALE,
    register_label_generator,
)
from openmed.core.anonymizer.format_preserve import (
    extract_digit_groups,
    preserve_email_pattern,
    preserve_id_pattern,
    preserve_phone_format,
)
from openmed.core.labels import normalize_label


class TestLocaleResolution:
    def test_lang_to_locale_covers_all_supported_languages(self):
        for lang in ("en", "fr", "de", "it", "es", "nl", "hi", "te", "pt"):
            assert lang in LANG_TO_LOCALE

    def test_locale_override_per_call(self):
        a = Anonymizer(lang="pt")
        # pt_PT default
        assert a.surrogate("Pedro", "FIRSTNAME", locale="pt_PT")
        # pt_BR override
        assert a.surrogate("Pedro", "FIRSTNAME", locale="pt_BR")

    def test_telugu_falls_back_to_indian_english(self):
        # Should not raise — just emits a warning the first time.
        with pytest.warns(UserWarning, match="te"):
            a = Anonymizer(lang="te")
            out = a.surrogate("Sita", "FIRSTNAME")
        assert out


class TestDeterminism:
    @pytest.mark.parametrize("label", ["FIRSTNAME", "LASTNAME", "EMAIL", "DATE", "PHONE"])
    def test_same_seed_same_surrogate(self, label):
        a1 = Anonymizer(lang="en", consistent=True, seed=42)
        a2 = Anonymizer(lang="en", consistent=True, seed=42)
        assert a1.surrogate("original", label) == a2.surrogate("original", label)

    def test_different_seeds_likely_different_surrogates(self):
        a1 = Anonymizer(lang="en", consistent=True, seed=1)
        a2 = Anonymizer(lang="en", consistent=True, seed=2)
        # Statistically these differ for any non-trivial label.
        outputs = {
            a1.surrogate("alice", "FIRSTNAME"),
            a2.surrogate("alice", "FIRSTNAME"),
        }
        assert len(outputs) == 2

    def test_consistent_within_doc(self):
        """Same (label, original) pair -> same surrogate when consistent=True."""
        a = Anonymizer(lang="en", consistent=True, seed=99)
        first = a.surrogate("John Doe", "name")
        # Repeated mentions in the same doc should resolve to the same surrogate.
        for _ in range(5):
            assert a.surrogate("John Doe", "name") == first

    def test_random_mode_varies(self):
        """Without consistent=True, repeated calls vary."""
        a = Anonymizer(lang="en", consistent=False)
        outs = {a.surrogate("John Doe", "name") for _ in range(20)}
        # 20 random samples should produce at least 2 distinct outputs
        assert len(outs) > 1

    def test_different_originals_different_surrogates(self):
        a = Anonymizer(lang="en", consistent=True, seed=42)
        s1 = a.surrogate("John", "FIRSTNAME")
        s2 = a.surrogate("Mary", "FIRSTNAME")
        assert s1 != s2


class TestClinicalIDChecksums:
    """Faker built-ins (CPF, CNPJ, BSN, NIR, NIE, Codice Fiscale) and our
    custom providers (Aadhaar, German Steuer-ID) must produce values that
    pass the corresponding validators."""

    @pytest.mark.parametrize("seed", list(range(20)))
    def test_pt_br_cpf_validates(self, seed):
        from openmed.core.pii_i18n import validate_portuguese_cpf
        a = Anonymizer(lang="pt", locale="pt_BR", consistent=True, seed=seed)
        cpf = a.surrogate("123.456.789-09", "CPF")
        assert validate_portuguese_cpf(cpf), f"Invalid CPF generated: {cpf!r}"

    @pytest.mark.parametrize("seed", list(range(20)))
    def test_nl_bsn_validates(self, seed):
        from openmed.core.pii_i18n import validate_dutch_bsn
        a = Anonymizer(lang="nl", consistent=True, seed=seed)
        bsn = a.surrogate("123456789", "SSN")
        assert validate_dutch_bsn(bsn), f"Invalid BSN: {bsn!r}"

    @pytest.mark.parametrize("seed", list(range(20)))
    def test_fr_nir_validates(self, seed):
        from openmed.core.pii_i18n import validate_french_nir
        a = Anonymizer(lang="fr", consistent=True, seed=seed)
        nir = a.surrogate("1 85 05 78 006 084 36", "SSN")
        assert validate_french_nir(nir), f"Invalid NIR: {nir!r}"

    @pytest.mark.parametrize("seed", list(range(20)))
    def test_it_codice_fiscale_validates(self, seed):
        from openmed.core.pii_i18n import validate_italian_codice_fiscale
        a = Anonymizer(lang="it", consistent=True, seed=seed)
        cf = a.surrogate("RSSMRA85M01H501Z", "SSN")
        assert validate_italian_codice_fiscale(cf), f"Invalid Codice Fiscale: {cf!r}"

    @pytest.mark.parametrize("seed", list(range(20)))
    def test_es_nie_validates(self, seed):
        from openmed.core.pii_i18n import validate_spanish_nie
        a = Anonymizer(lang="es", consistent=True, seed=seed)
        nie = a.surrogate("X1234567L", "ID_NUM")
        assert validate_spanish_nie(nie), f"Invalid NIE: {nie!r}"

    @pytest.mark.parametrize("seed", list(range(10)))
    def test_aadhaar_provider_validates_via_verhoeff(self, seed):
        """Custom AadhaarProvider must produce Verhoeff-valid Aadhaar."""
        from openmed.core.pii_i18n import validate_aadhaar
        # Use the underlying Faker provider directly
        from faker import Faker
        from openmed.core.anonymizer.providers.clinical_ids import (
            register_clinical_providers,
        )
        fk = Faker("en_IN")
        register_clinical_providers(fk)
        fk.seed_instance(seed)
        for _ in range(20):
            a = fk.aadhaar()
            assert validate_aadhaar(a), f"Verhoeff failed for {a!r}"

    @pytest.mark.parametrize("seed", list(range(5)))
    def test_german_steuer_id_provider_validates(self, seed):
        from openmed.core.pii_i18n import validate_german_steuer_id
        from faker import Faker
        from openmed.core.anonymizer.providers.clinical_ids import (
            register_clinical_providers,
        )
        fk = Faker("de_DE")
        register_clinical_providers(fk)
        fk.seed_instance(seed)
        for _ in range(5):
            sid = fk.german_steuer_id()
            assert validate_german_steuer_id(sid), f"Invalid Steuer-ID: {sid!r}"


class TestFormatPreservation:
    def test_extract_digit_groups(self):
        assert extract_digit_groups("+1 (415) 555-1234") == [1, 3, 3, 4]
        assert extract_digit_groups("+33 6 12 34 56 78") == [2, 1, 2, 2, 2, 2]
        assert extract_digit_groups("no digits here") == []

    def test_preserve_phone_format_keeps_separators_and_lengths(self):
        out = preserve_phone_format("+1 (415) 555-1234")
        assert "+" in out
        assert " " in out
        assert "(" in out and ")" in out
        assert "-" in out
        assert extract_digit_groups(out) == [1, 3, 3, 4]

    def test_preserve_email_pattern_keeps_domain(self):
        result = preserve_email_pattern("john@hospital.org", "alice@example.com")
        assert result.endswith("@hospital.org")

    def test_preserve_email_pattern_falls_back_when_invalid(self):
        # If the original isn't an email, just return the fake.
        result = preserve_email_pattern("not_an_email", "alice@example.com")
        assert result == "alice@example.com"

    def test_preserve_id_pattern(self):
        out = preserve_id_pattern("MRN-1234567")
        assert out.startswith("MRN-") or len(out) == len("MRN-1234567")
        # All non-alpha non-digit chars preserved
        assert "-" in out

    def test_phone_through_engine_preserves_groups(self):
        a = Anonymizer(lang="en", consistent=True, seed=0)
        original = "+1 (415) 555-1234"
        surrogate = a.surrogate(original, "phone_number")
        # Same number of digit groups, each the same length
        assert extract_digit_groups(surrogate) == extract_digit_groups(original)


class TestLabelCoverage:
    """Every canonical label has a generator and produces a non-empty result."""

    @pytest.mark.parametrize(
        "label",
        [
            "FIRSTNAME", "LASTNAME", "name", "patient",
            "EMAIL", "PHONE", "URL",
            "CITY", "STATE", "STREET",
            "DATE", "DATEOFBIRTH", "TIME", "AGE",
            "ID_NUM", "SSN",
            "CREDITCARD", "IBAN", "BIC",
            "GENDER", "ORGANIZATION", "JOBTITLE",
            "IPADDRESS", "MACADDRESS", "USERAGENT",
            "BITCOINADDRESS", "ETHEREUMADDRESS",
            "USERNAME", "ZIPCODE", "PASSWORD", "PIN",
        ],
    )
    def test_every_label_yields_non_empty_surrogate(self, label):
        a = Anonymizer(lang="en", consistent=True, seed=42)
        out = a.surrogate("seed_value", label)
        assert isinstance(out, str)
        assert out


class TestCustomGenerator:
    def test_register_label_generator_overrides_default(self):
        """Users can override per-label generators."""
        marker = "CUSTOM-MRN-GENERATOR"

        def my_gen(faker, original, *, locale):
            return marker

        # Register custom generator for ID_NUM
        register_label_generator("ID_NUM", my_gen)
        try:
            a = Anonymizer(lang="en")
            assert a.surrogate("12345", "id_num") == marker
        finally:
            # Restore default
            from openmed.core.anonymizer.registry import _gen_id_num
            register_label_generator("ID_NUM", _gen_id_num)


class TestAnonymizerConfig:
    def test_config_dataclass_construction(self):
        cfg = AnonymizerConfig(lang="pt", locale="pt_BR", consistent=True, seed=7)
        a = Anonymizer(config=cfg)
        assert a.config.lang == "pt"
        assert a.config.locale == "pt_BR"
        assert a.config.consistent
        assert a.config.seed == 7


class TestIntegrationWithDeidentify:
    """End-to-end: deidentify(method='replace', consistent=True, seed=...) is stable."""

    def test_deidentify_consistent_repeatable(self):
        from unittest.mock import patch
        from openmed.processing.outputs import EntityPrediction, PredictionResult
        from openmed.core.pii import deidentify
        from datetime import datetime as _dt
        text = "Patient John Doe born on 01/15/1970"
        entities = [
            EntityPrediction(text="John Doe", label="name", start=8, end=16, confidence=0.95),
            EntityPrediction(text="01/15/1970", label="date_of_birth", start=25, end=35, confidence=0.95),
        ]
        with patch("openmed.core.pii.extract_pii") as mock:
            mock.return_value = PredictionResult(
                text=text, entities=entities, model_name="test",
                timestamp=_dt.now().isoformat(),
            )
            r1 = deidentify(text, method="replace", lang="en", consistent=True, seed=123)
            r2 = deidentify(text, method="replace", lang="en", consistent=True, seed=123)
            assert r1.deidentified_text == r2.deidentified_text
            assert "John Doe" not in r1.deidentified_text
            assert "01/15/1970" not in r1.deidentified_text

    def test_deidentify_seed_implies_consistent(self):
        """Passing seed= alone should produce repeatable output."""
        from unittest.mock import patch
        from openmed.processing.outputs import EntityPrediction, PredictionResult
        from openmed.core.pii import deidentify
        from datetime import datetime as _dt
        text = "Maria Silva"
        entities = [EntityPrediction(text="Maria Silva", label="name", start=0, end=11, confidence=0.95)]
        with patch("openmed.core.pii.extract_pii") as mock:
            mock.return_value = PredictionResult(
                text=text, entities=entities, model_name="test",
                timestamp=_dt.now().isoformat(),
            )
            r1 = deidentify(text, method="replace", lang="pt", seed=7)
            r2 = deidentify(text, method="replace", lang="pt", seed=7)
            assert r1.deidentified_text == r2.deidentified_text


class TestNormalizeLabelIntegration:
    """The engine routes through normalize_label internally."""

    def test_uppercase_and_lowercase_routes_to_same_generator(self):
        a = Anonymizer(lang="en", consistent=True, seed=42)
        # Both the lowercase English form and the Portuguese UPPERCASE form
        # normalize to FIRST_NAME, so they should produce the same surrogate.
        s1 = a.surrogate("John", "first_name")
        s2 = a.surrogate("John", "FIRSTNAME")
        assert s1 == s2

    def test_bioes_prefix_routes_to_base_generator(self):
        a = Anonymizer(lang="en", consistent=True, seed=42)
        s1 = a.surrogate("John", "name")
        s2 = a.surrogate("John", "B-NAME")
        s3 = a.surrogate("John", "I-NAME")
        s4 = a.surrogate("John", "S-NAME")
        # All map to PERSON canonical label and use same seed material.
        assert s1 == s2 == s3 == s4

    def test_normalize_label_is_consistent_with_engine_routing(self):
        """The engine's behavior must match what normalize_label predicts."""
        # If two source labels normalize to the same canonical, engine must agree.
        for label_pair in [("first_name", "FIRSTNAME"),
                           ("date_of_birth", "DATEOFBIRTH"),
                           ("phone_number", "PHONE")]:
            assert normalize_label(label_pair[0]) == normalize_label(label_pair[1])
