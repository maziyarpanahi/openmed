"""Tests for the canonical PII label taxonomy."""

import pytest

from openmed.core.labels import (
    AGE,
    CANONICAL_LABELS,
    CREDIT_CARD,
    DATE,
    DATE_OF_BIRTH,
    EMAIL,
    FIRST_NAME,
    GENDER,
    ID_NUM,
    IP_ADDRESS,
    LAST_NAME,
    LOCATION,
    OCCUPATION,
    ORGANIZATION,
    OTHER,
    PERSON,
    PHONE,
    SSN,
    STREET_ADDRESS,
    URL,
    USERNAME,
    ZIPCODE,
    normalize_label,
)


class TestEnglishLabels:
    """Lowercase snake_case forms emitted by the English/multilingual models."""

    @pytest.mark.parametrize(
        "label,expected",
        [
            ("first_name", FIRST_NAME),
            ("last_name", LAST_NAME),
            ("name", PERSON),
            ("patient", PERSON),
            ("doctor", PERSON),
            ("email", EMAIL),
            ("phone_number", PHONE),
            ("phone", PHONE),
            ("city", LOCATION),
            ("state", LOCATION),
            ("country", LOCATION),
            ("street_address", STREET_ADDRESS),
            ("date", DATE),
            ("date_of_birth", DATE_OF_BIRTH),
            ("dob", DATE_OF_BIRTH),
            ("ssn", SSN),
            ("medical_record_number", ID_NUM),
            ("id_num", ID_NUM),
            ("national_id", ID_NUM),
            ("age", AGE),
            ("username", USERNAME),
            ("url_personal", URL),
            ("zipcode", ZIPCODE),
            ("zip", ZIPCODE),
            ("postal_code", ZIPCODE),
            ("credit_debit_card", CREDIT_CARD),
        ],
    )
    def test_english_lowercase(self, label, expected):
        assert normalize_label(label) == expected


class TestPortugueseUppercaseLabels:
    """All 52 Portuguese UPPERCASE labels from the registry."""

    @pytest.mark.parametrize(
        "label,expected",
        [
            ("ACCOUNTNAME", "ACCOUNT_NUMBER"),
            ("AGE", AGE),
            ("AMOUNT", "AMOUNT"),
            ("BANKACCOUNT", "ACCOUNT_NUMBER"),
            ("BIC", "BIC"),
            ("BITCOINADDRESS", "BITCOIN_ADDRESS"),
            ("BUILDINGNUMBER", "BUILDING_NUMBER"),
            ("CITY", LOCATION),
            ("COUNTY", LOCATION),
            ("CREDITCARD", CREDIT_CARD),
            ("CREDITCARDISSUER", "CREDIT_CARD_ISSUER"),
            ("CURRENCY", "CURRENCY"),
            ("CURRENCYCODE", "CURRENCY"),
            ("CURRENCYNAME", "CURRENCY"),
            ("CURRENCYSYMBOL", "CURRENCY"),
            ("CVV", "CVV"),
            ("DATE", DATE),
            ("DATEOFBIRTH", DATE_OF_BIRTH),
            ("EMAIL", EMAIL),
            ("ETHEREUMADDRESS", "ETHEREUM_ADDRESS"),
            ("EYECOLOR", "EYE_COLOR"),
            ("FIRSTNAME", FIRST_NAME),
            ("GENDER", GENDER),
            ("GPSCOORDINATES", "GPS_COORDINATES"),
            ("HEIGHT", "HEIGHT"),
            ("IBAN", "IBAN"),
            ("IMEI", "IMEI"),
            ("IPADDRESS", IP_ADDRESS),
            ("JOBDEPARTMENT", "JOB_DEPARTMENT"),
            ("JOBTITLE", "JOB_TITLE"),
            ("LASTNAME", LAST_NAME),
            ("LITECOINADDRESS", "LITECOIN_ADDRESS"),
            ("MACADDRESS", "MAC_ADDRESS"),
            ("MASKEDNUMBER", "MASKED_NUMBER"),
            ("MIDDLENAME", "MIDDLE_NAME"),
            ("OCCUPATION", OCCUPATION),
            ("ORDINALDIRECTION", "ORDINAL_DIRECTION"),
            ("ORGANIZATION", ORGANIZATION),
            ("PASSWORD", "PASSWORD"),
            ("PHONE", PHONE),
            ("PIN", "PIN"),
            ("PREFIX", "PREFIX"),
            ("SECONDARYADDRESS", STREET_ADDRESS),
            ("SEX", GENDER),
            ("SSN", SSN),
            ("STATE", LOCATION),
            ("STREET", STREET_ADDRESS),
            ("TIME", "TIME"),
            ("URL", URL),
            ("USERAGENT", "USER_AGENT"),
            ("USERNAME", USERNAME),
            ("VIN", "VIN"),
            ("VRM", "VEHICLE_REGISTRATION"),
            ("ZIPCODE", ZIPCODE),
        ],
    )
    def test_portuguese_uppercase(self, label, expected):
        assert normalize_label(label, lang="pt") == expected


class TestBIOESPrefixes:
    """Privacy-filter family emits BIOES-tagged labels."""

    @pytest.mark.parametrize(
        "label,expected",
        [
            ("B-NAME", PERSON),
            ("I-NAME", PERSON),
            ("E-NAME", PERSON),
            ("S-NAME", PERSON),
            ("B-EMAIL", EMAIL),
            ("I-EMAIL", EMAIL),
            ("B-PHONE", PHONE),
            ("S-PHONE", PHONE),
            ("B-DATE", DATE),
            ("E-LOCATION", LOCATION),
            ("S-FIRSTNAME", FIRST_NAME),
            ("B-CPF", ID_NUM),
            ("I-CREDITCARD", CREDIT_CARD),
        ],
    )
    def test_bioes_prefix_stripped(self, label, expected):
        assert normalize_label(label) == expected


class TestEdgeCases:
    def test_empty_label_returns_other(self):
        assert normalize_label("") == OTHER

    def test_whitespace_label_returns_other(self):
        assert normalize_label("   ") == OTHER

    def test_unknown_label_returns_other(self):
        assert normalize_label("totally_made_up_label") == OTHER

    def test_mixed_case_normalized(self):
        assert normalize_label("First-Name") == FIRST_NAME
        assert normalize_label("first name") == FIRST_NAME
        assert normalize_label("FIRST_NAME") == FIRST_NAME

    def test_canonical_label_round_trip(self):
        """Every canonical label normalizes back to itself."""
        for canonical in CANONICAL_LABELS:
            assert normalize_label(canonical) == canonical, (
                f"Canonical label {canonical!r} did not round-trip"
            )

    def test_canonical_labels_are_screaming_snake(self):
        for label in CANONICAL_LABELS:
            assert label == label.upper(), f"{label} is not uppercase"
            for ch in label:
                assert ch.isalnum() or ch == "_", f"{label} has non-snake char {ch!r}"

    def test_lang_parameter_accepted(self):
        """The lang hint is accepted but currently doesn't change output."""
        for lang in ("en", "fr", "de", "it", "es", "nl", "hi", "te", "pt"):
            assert normalize_label("first_name", lang=lang) == FIRST_NAME


class TestRegistryCoverage:
    """The 52 Portuguese registry labels all map to canonical names."""

    PORTUGUESE_REGISTRY_LABELS = [
        "ACCOUNTNAME", "AGE", "AMOUNT", "BANKACCOUNT", "BIC", "BITCOINADDRESS",
        "BUILDINGNUMBER", "CITY", "COUNTY", "CREDITCARD", "CREDITCARDISSUER",
        "CURRENCY", "CURRENCYCODE", "CURRENCYNAME", "CURRENCYSYMBOL", "CVV",
        "DATE", "DATEOFBIRTH", "EMAIL", "ETHEREUMADDRESS", "EYECOLOR",
        "FIRSTNAME", "GENDER", "GPSCOORDINATES", "HEIGHT", "IBAN", "IMEI",
        "IPADDRESS", "JOBDEPARTMENT", "JOBTITLE", "LASTNAME", "LITECOINADDRESS",
        "MACADDRESS", "MASKEDNUMBER", "MIDDLENAME", "OCCUPATION",
        "ORDINALDIRECTION", "ORGANIZATION", "PASSWORD", "PHONE", "PIN",
        "PREFIX", "SECONDARYADDRESS", "SEX", "SSN", "STATE", "STREET",
        "TIME", "URL", "USERAGENT", "USERNAME", "VIN", "VRM", "ZIPCODE",
    ]

    def test_all_portuguese_labels_have_canonical_mapping(self):
        unmapped = []
        for label in self.PORTUGUESE_REGISTRY_LABELS:
            canonical = normalize_label(label, lang="pt")
            if canonical == OTHER:
                unmapped.append(label)
        assert not unmapped, f"Portuguese labels not mapped: {unmapped}"

    def test_all_portuguese_labels_map_to_known_canonical(self):
        for label in self.PORTUGUESE_REGISTRY_LABELS:
            canonical = normalize_label(label, lang="pt")
            assert canonical in CANONICAL_LABELS, (
                f"{label} -> {canonical} which is not in CANONICAL_LABELS"
            )
