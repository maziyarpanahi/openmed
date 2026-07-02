"""Tests for the canonical PII label taxonomy."""

import pytest

from openmed.core.labels import (
    ACCOUNT_NUMBER,
    AGE,
    AMOUNT,
    ANTIBIOTIC,
    API_KEY,
    BIC,
    BITCOIN_ADDRESS,
    BODY_SITE,
    BUILDING_NUMBER,
    CANONICAL_LABELS,
    CLINICAL_CONCEPT,
    CLINICAL_SIGNIFICANCE,
    CONDITION,
    CREDIT_CARD,
    CREDIT_CARD_ISSUER,
    CURRENCY,
    CVV,
    DATE,
    DATE_OF_BIRTH,
    DIET_TYPE,
    EMAIL,
    ETHEREUM_ADDRESS,
    EYE_COLOR,
    FEEDING_ROUTE,
    FIRST_NAME,
    GENDER,
    GENE_SYMBOL,
    GPS_COORDINATES,
    HEIGHT,
    HIPAA_SAFE_HARBOR_CLASSES,
    IBAN,
    ID_NUM,
    ID_SUBTYPE_MRN,
    ID_SUBTYPE_NATIONAL_ID,
    ID_SUBTYPE_NPI,
    ID_SUBTYPES,
    IMEI,
    IP_ADDRESS,
    JOB_DEPARTMENT,
    JOB_TITLE,
    LAB_TEST,
    LAST_NAME,
    LITECOIN_ADDRESS,
    LOCATION,
    MAC_ADDRESS,
    MASKED_NUMBER,
    MEDICATION,
    MICROORGANISM,
    MIDDLE_NAME,
    NUTRITION_TARGET,
    NUTRITIONAL_STATUS,
    OCCUPATION,
    ORDINAL_DIRECTION,
    ORGANIZATION,
    OTHER,
    PASSWORD,
    PERSON,
    PHONE,
    PIN,
    PREFIX,
    PROCEDURE,
    PROTEIN_CHANGE,
    SSN,
    STREET_ADDRESS,
    SUSCEPTIBILITY,
    TIME,
    URL,
    USER_AGENT,
    USERNAME,
    VARIANT_DESCRIPTOR,
    VEHICLE_REGISTRATION,
    VIN,
    ZIPCODE,
    ZYGOSITY,
    hipaa_class_for,
    id_subtype_for,
    normalize_label,
    policy_label_for,
    risk_level_for,
    system_hints_for,
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
            ("nhs_number", ID_NUM),
            ("id_num", ID_NUM),
            ("national_id", ID_NUM),
            ("teudat_zehut", ID_NUM),
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

    def test_top_level_label_import_stability(self):
        from openmed import CANONICAL_LABELS as top_level_labels
        from openmed import normalize_label as top_level_normalize

        assert top_level_labels is CANONICAL_LABELS
        assert top_level_normalize("id_num") == ID_NUM

    def test_policy_accessors_import(self):
        assert policy_label_for("ID_NUM") == "DIRECT_IDENTIFIER"
        assert risk_level_for("ID_NUM") == "high"
        assert system_hints_for("ID_NUM") == ()
        assert hipaa_class_for("ID_NUM") == "UNIQUE_IDENTIFIER"


class TestIdentifierSubtypes:
    @pytest.mark.parametrize(
        "label,expected_subtype",
        [
            ("medical_record_number", ID_SUBTYPE_MRN),
            ("mrn", ID_SUBTYPE_MRN),
            ("npi", ID_SUBTYPE_NPI),
            ("nhs_number", ID_SUBTYPE_NATIONAL_ID),
            ("national_id", ID_SUBTYPE_NATIONAL_ID),
            ("nationalid", ID_SUBTYPE_NATIONAL_ID),
            ("cpf", ID_SUBTYPE_NATIONAL_ID),
            ("cnpj", ID_SUBTYPE_NATIONAL_ID),
            ("nir", ID_SUBTYPE_NATIONAL_ID),
            ("steuerid", ID_SUBTYPE_NATIONAL_ID),
            ("codicefiscale", ID_SUBTYPE_NATIONAL_ID),
            ("dni", ID_SUBTYPE_NATIONAL_ID),
            ("nie", ID_SUBTYPE_NATIONAL_ID),
            ("bsn", ID_SUBTYPE_NATIONAL_ID),
            ("aadhaar", ID_SUBTYPE_NATIONAL_ID),
            ("teudat_zehut", ID_SUBTYPE_NATIONAL_ID),
            ("tz", ID_SUBTYPE_NATIONAL_ID),
            ("B-CPF", ID_SUBTYPE_NATIONAL_ID),
        ],
    )
    def test_id_aliases_keep_canonical_id_num_with_subtype(
        self,
        label,
        expected_subtype,
    ):
        assert normalize_label(label) == ID_NUM
        assert id_subtype_for(label) == expected_subtype
        assert expected_subtype in ID_SUBTYPES

    def test_flat_canonical_id_num_consumers_are_unchanged(self):
        assert normalize_label("mrn") == ID_NUM
        assert normalize_label("npi") == ID_NUM
        assert policy_label_for("mrn") == "DIRECT_IDENTIFIER"
        assert risk_level_for("npi") == "high"

    def test_non_id_num_labels_have_no_id_subtype(self):
        assert id_subtype_for("ssn") is None
        assert id_subtype_for("email") is None
        assert id_subtype_for("id_num") is None
        assert normalize_label("ssn") == SSN

    def test_clinical_id_provider_exposes_regex_subtype_mapping(self):
        from openmed.core.anonymizer.providers import clinical_ids

        assert clinical_ids.id_subtype_for_entity_type("medical_record_number") == (
            ID_SUBTYPE_MRN
        )
        assert clinical_ids.id_subtype_for_entity_type("npi") == ID_SUBTYPE_NPI
        assert clinical_ids.id_subtype_for_entity_type("aadhaar") == (
            ID_SUBTYPE_NATIONAL_ID
        )


class TestRegistryCoverage:
    """The 52 Portuguese registry labels all map to canonical names."""

    PORTUGUESE_REGISTRY_LABELS = [
        "ACCOUNTNAME",
        "AGE",
        "AMOUNT",
        "BANKACCOUNT",
        "BIC",
        "BITCOINADDRESS",
        "BUILDINGNUMBER",
        "CITY",
        "COUNTY",
        "CREDITCARD",
        "CREDITCARDISSUER",
        "CURRENCY",
        "CURRENCYCODE",
        "CURRENCYNAME",
        "CURRENCYSYMBOL",
        "CVV",
        "DATE",
        "DATEOFBIRTH",
        "EMAIL",
        "ETHEREUMADDRESS",
        "EYECOLOR",
        "FIRSTNAME",
        "GENDER",
        "GPSCOORDINATES",
        "HEIGHT",
        "IBAN",
        "IMEI",
        "IPADDRESS",
        "JOBDEPARTMENT",
        "JOBTITLE",
        "LASTNAME",
        "LITECOINADDRESS",
        "MACADDRESS",
        "MASKEDNUMBER",
        "MIDDLENAME",
        "OCCUPATION",
        "ORDINALDIRECTION",
        "ORGANIZATION",
        "PASSWORD",
        "PHONE",
        "PIN",
        "PREFIX",
        "SECONDARYADDRESS",
        "SEX",
        "SSN",
        "STATE",
        "STREET",
        "TIME",
        "URL",
        "USERAGENT",
        "USERNAME",
        "VIN",
        "VRM",
        "ZIPCODE",
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


class TestClinicalConceptLabels:
    """Clinical-concept canonical labels added for grounding (issue #266)."""

    NEW_LABELS = (CONDITION, MEDICATION, LAB_TEST, PROCEDURE, BODY_SITE)

    def test_new_labels_in_canonical_set(self):
        for label in self.NEW_LABELS:
            assert label in CANONICAL_LABELS

    def test_new_labels_round_trip(self):
        for label in self.NEW_LABELS:
            assert normalize_label(label) == label

    @pytest.mark.parametrize(
        "alias,expected",
        [
            ("disease", CONDITION),
            ("diagnosis", CONDITION),
            ("finding", CONDITION),
            ("drug", MEDICATION),
            ("medication", MEDICATION),
            ("chemical", MEDICATION),
            ("test", LAB_TEST),
            ("measurement", LAB_TEST),
            ("analyte", LAB_TEST),
            ("surgery", PROCEDURE),
            ("procedure", PROCEDURE),
            ("operation", PROCEDURE),
            ("anatomy", BODY_SITE),
            ("organ", BODY_SITE),
            ("body site", BODY_SITE),
        ],
    )
    def test_clinical_aliases_resolve(self, alias, expected):
        assert normalize_label(alias) == expected

    def test_new_labels_are_clinical_concepts(self):
        for label in self.NEW_LABELS:
            assert policy_label_for(label) == CLINICAL_CONCEPT
            assert system_hints_for(label)  # non-empty grounding hints

    def test_new_labels_have_hipaa_class(self):
        for label in self.NEW_LABELS:
            assert hipaa_class_for(label) in HIPAA_SAFE_HARBOR_CLASSES


class TestClinicalLabelsAreAdditive:
    """The clinical additions must not disturb the existing PII taxonomy."""

    ORIGINAL_LABELS = frozenset(
        {
            PERSON,
            FIRST_NAME,
            LAST_NAME,
            MIDDLE_NAME,
            PREFIX,
            USERNAME,
            EMAIL,
            PHONE,
            URL,
            LOCATION,
            STREET_ADDRESS,
            BUILDING_NUMBER,
            ZIPCODE,
            GPS_COORDINATES,
            ORDINAL_DIRECTION,
            DATE,
            DATE_OF_BIRTH,
            TIME,
            AGE,
            ID_NUM,
            SSN,
            ACCOUNT_NUMBER,
            PASSWORD,
            PIN,
            API_KEY,
            CREDIT_CARD,
            CREDIT_CARD_ISSUER,
            CVV,
            IBAN,
            BIC,
            AMOUNT,
            CURRENCY,
            BITCOIN_ADDRESS,
            ETHEREUM_ADDRESS,
            LITECOIN_ADDRESS,
            MASKED_NUMBER,
            GENDER,
            EYE_COLOR,
            HEIGHT,
            ORGANIZATION,
            JOB_TITLE,
            JOB_DEPARTMENT,
            OCCUPATION,
            IP_ADDRESS,
            MAC_ADDRESS,
            USER_AGENT,
            VIN,
            VEHICLE_REGISTRATION,
            IMEI,
            MICROORGANISM,
            ANTIBIOTIC,
            SUSCEPTIBILITY,
            OTHER,
        }
    )

    NEW_LABELS = frozenset(
        {
            CONDITION,
            MEDICATION,
            LAB_TEST,
            PROCEDURE,
            BODY_SITE,
            DIET_TYPE,
            NUTRITION_TARGET,
            FEEDING_ROUTE,
            NUTRITIONAL_STATUS,
            GENE_SYMBOL,
            VARIANT_DESCRIPTOR,
            PROTEIN_CHANGE,
            ZYGOSITY,
            CLINICAL_SIGNIFICANCE,
        }
    )

    # Full pre-#266 alias map from master. These resolutions must stay
    # byte-identical while clinical aliases are added.
    PRESERVED_ALIASES = {
        "name": PERSON,
        "person": PERSON,
        "patient": PERSON,
        "doctor": PERSON,
        "fullname": PERSON,
        "firstname": FIRST_NAME,
        "givenname": FIRST_NAME,
        "lastname": LAST_NAME,
        "surname": LAST_NAME,
        "familyname": LAST_NAME,
        "middlename": MIDDLE_NAME,
        "prefix": PREFIX,
        "title": PREFIX,
        "username": USERNAME,
        "userhandle": USERNAME,
        "email": EMAIL,
        "emailaddress": EMAIL,
        "phone": PHONE,
        "phonenumber": PHONE,
        "telephone": PHONE,
        "fax": PHONE,
        "url": URL,
        "urlpersonal": URL,
        "website": URL,
        "personalurl": URL,
        "location": LOCATION,
        "city": LOCATION,
        "state": LOCATION,
        "country": LOCATION,
        "county": LOCATION,
        "region": LOCATION,
        "place": LOCATION,
        "address": STREET_ADDRESS,
        "street": STREET_ADDRESS,
        "streetaddress": STREET_ADDRESS,
        "secondaryaddress": STREET_ADDRESS,
        "buildingnumber": BUILDING_NUMBER,
        "zipcode": ZIPCODE,
        "zip": ZIPCODE,
        "postcode": ZIPCODE,
        "postalcode": ZIPCODE,
        "gpscoordinates": GPS_COORDINATES,
        "gps": GPS_COORDINATES,
        "ordinaldirection": ORDINAL_DIRECTION,
        "date": DATE,
        "dateofbirth": DATE_OF_BIRTH,
        "dob": DATE_OF_BIRTH,
        "birthdate": DATE_OF_BIRTH,
        "time": TIME,
        "age": AGE,
        "idnum": ID_NUM,
        "id": ID_NUM,
        "identifier": ID_NUM,
        "medicalrecordnumber": ID_NUM,
        "mrn": ID_NUM,
        "nationalid": ID_NUM,
        "cpf": ID_NUM,
        "cnpj": ID_NUM,
        "nir": ID_NUM,
        "steuerid": ID_NUM,
        "codicefiscale": ID_NUM,
        "dni": ID_NUM,
        "nie": ID_NUM,
        "bsn": ID_NUM,
        "aadhaar": ID_NUM,
        "npi": ID_NUM,
        "ssn": SSN,
        "socialsecuritynumber": SSN,
        "accountnumber": ACCOUNT_NUMBER,
        "accountname": ACCOUNT_NUMBER,
        "bankaccount": ACCOUNT_NUMBER,
        "password": PASSWORD,
        "pin": PIN,
        "apikey": API_KEY,
        "creditcard": CREDIT_CARD,
        "creditdebitcard": CREDIT_CARD,
        "creditcardnumber": CREDIT_CARD,
        "creditcardissuer": CREDIT_CARD_ISSUER,
        "cvv": CVV,
        "iban": IBAN,
        "bic": BIC,
        "swift": BIC,
        "amount": AMOUNT,
        "currency": CURRENCY,
        "currencycode": CURRENCY,
        "currencyname": CURRENCY,
        "currencysymbol": CURRENCY,
        "bitcoinaddress": BITCOIN_ADDRESS,
        "ethereumaddress": ETHEREUM_ADDRESS,
        "litecoinaddress": LITECOIN_ADDRESS,
        "maskednumber": MASKED_NUMBER,
        "gender": GENDER,
        "sex": GENDER,
        "eyecolor": EYE_COLOR,
        "height": HEIGHT,
        "organization": ORGANIZATION,
        "company": ORGANIZATION,
        "employer": ORGANIZATION,
        "jobtitle": JOB_TITLE,
        "jobdepartment": JOB_DEPARTMENT,
        "department": JOB_DEPARTMENT,
        "occupation": OCCUPATION,
        "profession": OCCUPATION,
        "ipaddress": IP_ADDRESS,
        "ip": IP_ADDRESS,
        "macaddress": MAC_ADDRESS,
        "useragent": USER_AGENT,
        "vin": VIN,
        "vrm": VEHICLE_REGISTRATION,
        "licenseplate": VEHICLE_REGISTRATION,
        "imei": IMEI,
        "microorganism": MICROORGANISM,
        "microbe": MICROORGANISM,
        "organism": MICROORGANISM,
        "pathogen": MICROORGANISM,
        "antibiotic": ANTIBIOTIC,
        "antimicrobial": ANTIBIOTIC,
        "susceptibility": SUSCEPTIBILITY,
        "susceptibilityresult": SUSCEPTIBILITY,
    }

    def test_original_labels_still_present(self):
        assert CANONICAL_LABELS - self.NEW_LABELS == self.ORIGINAL_LABELS

    def test_existing_aliases_unchanged(self):
        for alias, expected in self.PRESERVED_ALIASES.items():
            assert normalize_label(alias) == expected

    def test_id_num_mapping_unchanged(self):
        assert normalize_label("ID_NUM") == ID_NUM
        assert normalize_label("mrn") == ID_NUM

    def test_unknown_labels_still_fall_back_to_other(self):
        assert normalize_label("totally_made_up_label") == OTHER
