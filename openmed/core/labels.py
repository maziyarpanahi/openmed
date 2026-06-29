"""Canonical PII/PHI label taxonomy.

Different OpenMed PII model families use different label-naming conventions:

- English / multilingual SuperClinical models use lowercase ``snake_case``
  (``first_name``, ``date_of_birth``).
- Portuguese models expose 52 ``UPPERCASE`` labels (``FIRSTNAME``,
  ``DATEOFBIRTH``).
- The privacy-filter family emits BIOES-tagged labels (``B-NAME``,
  ``I-EMAIL``, ``E-ADDRESS``, ``S-PHONE``).

This module provides a single ``CANONICAL_LABELS`` taxonomy in
``UPPER_SNAKE_CASE`` and a ``normalize_label`` helper that maps any of the
above input forms to its canonical name. Downstream code (anonymization,
mapping tables, configuration) should key off canonical labels only.
"""

from __future__ import annotations

import re
from typing import Final, FrozenSet, Mapping, cast

# ---------------------------------------------------------------------------
# Canonical taxonomy
# ---------------------------------------------------------------------------

#: People-related entities
PERSON: Final = "PERSON"
FIRST_NAME: Final = "FIRST_NAME"
LAST_NAME: Final = "LAST_NAME"
MIDDLE_NAME: Final = "MIDDLE_NAME"
PREFIX: Final = "PREFIX"
USERNAME: Final = "USERNAME"

#: Contact
EMAIL: Final = "EMAIL"
PHONE: Final = "PHONE"
URL: Final = "URL"

#: Location
LOCATION: Final = "LOCATION"
STREET_ADDRESS: Final = "STREET_ADDRESS"
BUILDING_NUMBER: Final = "BUILDING_NUMBER"
ZIPCODE: Final = "ZIPCODE"
GPS_COORDINATES: Final = "GPS_COORDINATES"
ORDINAL_DIRECTION: Final = "ORDINAL_DIRECTION"

#: Time
DATE: Final = "DATE"
DATE_OF_BIRTH: Final = "DATE_OF_BIRTH"
TIME: Final = "TIME"
AGE: Final = "AGE"

#: Identifiers
ID_NUM: Final = "ID_NUM"
SSN: Final = "SSN"
ACCOUNT_NUMBER: Final = "ACCOUNT_NUMBER"
PASSWORD: Final = "PASSWORD"
PIN: Final = "PIN"
API_KEY: Final = "API_KEY"

#: Financial
CREDIT_CARD: Final = "CREDIT_CARD"
CREDIT_CARD_ISSUER: Final = "CREDIT_CARD_ISSUER"
CVV: Final = "CVV"
IBAN: Final = "IBAN"
BIC: Final = "BIC"
AMOUNT: Final = "AMOUNT"
CURRENCY: Final = "CURRENCY"
BITCOIN_ADDRESS: Final = "BITCOIN_ADDRESS"
ETHEREUM_ADDRESS: Final = "ETHEREUM_ADDRESS"
LITECOIN_ADDRESS: Final = "LITECOIN_ADDRESS"
MASKED_NUMBER: Final = "MASKED_NUMBER"

#: Demographics
GENDER: Final = "GENDER"
EYE_COLOR: Final = "EYE_COLOR"
HEIGHT: Final = "HEIGHT"

#: Work
ORGANIZATION: Final = "ORGANIZATION"
JOB_TITLE: Final = "JOB_TITLE"
JOB_DEPARTMENT: Final = "JOB_DEPARTMENT"
OCCUPATION: Final = "OCCUPATION"

#: Tech
IP_ADDRESS: Final = "IP_ADDRESS"
MAC_ADDRESS: Final = "MAC_ADDRESS"
USER_AGENT: Final = "USER_AGENT"
VIN: Final = "VIN"
VEHICLE_REGISTRATION: Final = "VEHICLE_REGISTRATION"
IMEI: Final = "IMEI"

#: Microbiology
MICROORGANISM: Final = "MICROORGANISM"
ANTIBIOTIC: Final = "ANTIBIOTIC"
SUSCEPTIBILITY: Final = "SUSCEPTIBILITY"

#: Clinical concepts (grounding targets for RxNorm/ICD-10-CM/LOINC/SNOMED/HPO)
CONDITION: Final = "CONDITION"
MEDICATION: Final = "MEDICATION"
LAB_TEST: Final = "LAB_TEST"
PROCEDURE: Final = "PROCEDURE"
BODY_SITE: Final = "BODY_SITE"

#: Nutrition and diet-order concepts (issue #951)
DIET_TYPE: Final = "DIET_TYPE"
NUTRITION_TARGET: Final = "NUTRITION_TARGET"
FEEDING_ROUTE: Final = "FEEDING_ROUTE"
NUTRITIONAL_STATUS: Final = "NUTRITIONAL_STATUS"

#: Catch-all
OTHER: Final = "OTHER"


# Optional sub-type metadata for labels that still normalize to ID_NUM.
ID_SUBTYPE_MRN: Final = "mrn"
ID_SUBTYPE_NPI: Final = "npi"
ID_SUBTYPE_NATIONAL_ID: Final = "national_id"
ID_SUBTYPE_SSN_ADJACENT: Final = "ssn_adjacent"
ID_SUBTYPES: Final[FrozenSet[str]] = frozenset(
    {
        ID_SUBTYPE_MRN,
        ID_SUBTYPE_NPI,
        ID_SUBTYPE_NATIONAL_ID,
        ID_SUBTYPE_SSN_ADJACENT,
    }
)


CANONICAL_LABELS: Final[FrozenSet[str]] = frozenset(
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
        CONDITION,
        MEDICATION,
        LAB_TEST,
        PROCEDURE,
        BODY_SITE,
        DIET_TYPE,
        NUTRITION_TARGET,
        FEEDING_ROUTE,
        NUTRITIONAL_STATUS,
        OTHER,
    }
)


# ---------------------------------------------------------------------------
# Policy metadata
# ---------------------------------------------------------------------------

DIRECT_IDENTIFIER: Final = "DIRECT_IDENTIFIER"
QUASI_IDENTIFIER: Final = "QUASI_IDENTIFIER"
CLINICAL_CONCEPT: Final = "CLINICAL_CONCEPT"
POLICY_LABELS: Final[FrozenSet[str]] = frozenset(
    {
        DIRECT_IDENTIFIER,
        QUASI_IDENTIFIER,
        CLINICAL_CONCEPT,
    }
)

RISK_LOW: Final = "low"
RISK_MEDIUM: Final = "medium"
RISK_HIGH: Final = "high"
RISK_LEVELS: Final[tuple[str, ...]] = (RISK_LOW, RISK_MEDIUM, RISK_HIGH)

RXNORM: Final = "RxNorm"
LOINC: Final = "LOINC"
ICD_10_CM: Final = "ICD-10-CM"
HPO: Final = "HPO"
SNOMED: Final = "SNOMED"
CLINICAL_SYSTEM_HINTS: Final[tuple[str, ...]] = (
    SNOMED,
    ICD_10_CM,
    HPO,
    RXNORM,
    LOINC,
)

HIPAA_NAME: Final = "NAME"
HIPAA_GEOGRAPHIC_SUBDIVISION: Final = "GEOGRAPHIC_SUBDIVISION"
HIPAA_DATE_ELEMENT: Final = "DATE_ELEMENT"
HIPAA_TELEPHONE_NUMBER: Final = "TELEPHONE_NUMBER"
HIPAA_FAX_NUMBER: Final = "FAX_NUMBER"
HIPAA_EMAIL_ADDRESS: Final = "EMAIL_ADDRESS"
HIPAA_SOCIAL_SECURITY_NUMBER: Final = "SOCIAL_SECURITY_NUMBER"
HIPAA_MEDICAL_RECORD_NUMBER: Final = "MEDICAL_RECORD_NUMBER"
HIPAA_HEALTH_PLAN_BENEFICIARY_NUMBER: Final = "HEALTH_PLAN_BENEFICIARY_NUMBER"
HIPAA_ACCOUNT_NUMBER: Final = "ACCOUNT_NUMBER"
HIPAA_CERTIFICATE_LICENSE_NUMBER: Final = "CERTIFICATE_LICENSE_NUMBER"
HIPAA_VEHICLE_IDENTIFIER: Final = "VEHICLE_IDENTIFIER"
HIPAA_DEVICE_IDENTIFIER: Final = "DEVICE_IDENTIFIER"
HIPAA_URL: Final = "URL"
HIPAA_IP_ADDRESS: Final = "IP_ADDRESS"
HIPAA_BIOMETRIC_IDENTIFIER: Final = "BIOMETRIC_IDENTIFIER"
HIPAA_FULL_FACE_PHOTO: Final = "FULL_FACE_PHOTO"
HIPAA_UNIQUE_IDENTIFIER: Final = "UNIQUE_IDENTIFIER"

HIPAA_SAFE_HARBOR_CLASSES: Final[FrozenSet[str]] = frozenset(
    {
        HIPAA_NAME,
        HIPAA_GEOGRAPHIC_SUBDIVISION,
        HIPAA_DATE_ELEMENT,
        HIPAA_TELEPHONE_NUMBER,
        HIPAA_FAX_NUMBER,
        HIPAA_EMAIL_ADDRESS,
        HIPAA_SOCIAL_SECURITY_NUMBER,
        HIPAA_MEDICAL_RECORD_NUMBER,
        HIPAA_HEALTH_PLAN_BENEFICIARY_NUMBER,
        HIPAA_ACCOUNT_NUMBER,
        HIPAA_CERTIFICATE_LICENSE_NUMBER,
        HIPAA_VEHICLE_IDENTIFIER,
        HIPAA_DEVICE_IDENTIFIER,
        HIPAA_URL,
        HIPAA_IP_ADDRESS,
        HIPAA_BIOMETRIC_IDENTIFIER,
        HIPAA_FULL_FACE_PHOTO,
        HIPAA_UNIQUE_IDENTIFIER,
    }
)

_NO_SYSTEM_HINTS: Final[tuple[str, ...]] = ()


def _label_metadata(
    policy_label: str,
    risk_level: str,
    system_hints: tuple[str, ...] = _NO_SYSTEM_HINTS,
) -> Mapping[str, object]:
    return {
        "policy_label": policy_label,
        "risk_level": risk_level,
        "system_hints": system_hints,
    }


LABEL_METADATA: Final[Mapping[str, Mapping[str, object]]] = {
    # People
    PERSON: _label_metadata(DIRECT_IDENTIFIER, RISK_HIGH),
    FIRST_NAME: _label_metadata(DIRECT_IDENTIFIER, RISK_HIGH),
    LAST_NAME: _label_metadata(DIRECT_IDENTIFIER, RISK_HIGH),
    MIDDLE_NAME: _label_metadata(DIRECT_IDENTIFIER, RISK_HIGH),
    PREFIX: _label_metadata(DIRECT_IDENTIFIER, RISK_HIGH),
    USERNAME: _label_metadata(DIRECT_IDENTIFIER, RISK_HIGH),
    # Contact
    EMAIL: _label_metadata(DIRECT_IDENTIFIER, RISK_HIGH),
    PHONE: _label_metadata(DIRECT_IDENTIFIER, RISK_HIGH),
    URL: _label_metadata(DIRECT_IDENTIFIER, RISK_HIGH),
    # Location
    LOCATION: _label_metadata(QUASI_IDENTIFIER, RISK_MEDIUM),
    STREET_ADDRESS: _label_metadata(DIRECT_IDENTIFIER, RISK_HIGH),
    BUILDING_NUMBER: _label_metadata(DIRECT_IDENTIFIER, RISK_HIGH),
    ZIPCODE: _label_metadata(QUASI_IDENTIFIER, RISK_MEDIUM),
    GPS_COORDINATES: _label_metadata(DIRECT_IDENTIFIER, RISK_HIGH),
    ORDINAL_DIRECTION: _label_metadata(QUASI_IDENTIFIER, RISK_MEDIUM),
    # Time
    DATE: _label_metadata(QUASI_IDENTIFIER, RISK_MEDIUM),
    DATE_OF_BIRTH: _label_metadata(DIRECT_IDENTIFIER, RISK_HIGH),
    TIME: _label_metadata(QUASI_IDENTIFIER, RISK_MEDIUM),
    AGE: _label_metadata(QUASI_IDENTIFIER, RISK_MEDIUM),
    # Identifiers
    ID_NUM: _label_metadata(DIRECT_IDENTIFIER, RISK_HIGH),
    SSN: _label_metadata(DIRECT_IDENTIFIER, RISK_HIGH),
    ACCOUNT_NUMBER: _label_metadata(DIRECT_IDENTIFIER, RISK_HIGH),
    PASSWORD: _label_metadata(DIRECT_IDENTIFIER, RISK_HIGH),
    PIN: _label_metadata(DIRECT_IDENTIFIER, RISK_HIGH),
    API_KEY: _label_metadata(DIRECT_IDENTIFIER, RISK_HIGH),
    # Financial
    CREDIT_CARD: _label_metadata(DIRECT_IDENTIFIER, RISK_HIGH),
    CREDIT_CARD_ISSUER: _label_metadata(QUASI_IDENTIFIER, RISK_MEDIUM),
    CVV: _label_metadata(DIRECT_IDENTIFIER, RISK_HIGH),
    IBAN: _label_metadata(DIRECT_IDENTIFIER, RISK_HIGH),
    BIC: _label_metadata(DIRECT_IDENTIFIER, RISK_HIGH),
    AMOUNT: _label_metadata(QUASI_IDENTIFIER, RISK_MEDIUM),
    CURRENCY: _label_metadata(QUASI_IDENTIFIER, RISK_MEDIUM),
    BITCOIN_ADDRESS: _label_metadata(DIRECT_IDENTIFIER, RISK_HIGH),
    ETHEREUM_ADDRESS: _label_metadata(DIRECT_IDENTIFIER, RISK_HIGH),
    LITECOIN_ADDRESS: _label_metadata(DIRECT_IDENTIFIER, RISK_HIGH),
    MASKED_NUMBER: _label_metadata(DIRECT_IDENTIFIER, RISK_HIGH),
    # Demographics
    GENDER: _label_metadata(QUASI_IDENTIFIER, RISK_MEDIUM),
    EYE_COLOR: _label_metadata(QUASI_IDENTIFIER, RISK_MEDIUM),
    HEIGHT: _label_metadata(QUASI_IDENTIFIER, RISK_MEDIUM),
    # Work
    ORGANIZATION: _label_metadata(QUASI_IDENTIFIER, RISK_MEDIUM),
    JOB_TITLE: _label_metadata(QUASI_IDENTIFIER, RISK_MEDIUM),
    JOB_DEPARTMENT: _label_metadata(QUASI_IDENTIFIER, RISK_MEDIUM),
    OCCUPATION: _label_metadata(QUASI_IDENTIFIER, RISK_MEDIUM),
    # Tech
    IP_ADDRESS: _label_metadata(DIRECT_IDENTIFIER, RISK_HIGH),
    MAC_ADDRESS: _label_metadata(DIRECT_IDENTIFIER, RISK_HIGH),
    USER_AGENT: _label_metadata(DIRECT_IDENTIFIER, RISK_HIGH),
    VIN: _label_metadata(DIRECT_IDENTIFIER, RISK_HIGH),
    VEHICLE_REGISTRATION: _label_metadata(DIRECT_IDENTIFIER, RISK_HIGH),
    IMEI: _label_metadata(DIRECT_IDENTIFIER, RISK_HIGH),
    # Microbiology
    MICROORGANISM: _label_metadata(CLINICAL_CONCEPT, RISK_LOW, (SNOMED, LOINC)),
    ANTIBIOTIC: _label_metadata(CLINICAL_CONCEPT, RISK_LOW, (RXNORM, SNOMED)),
    SUSCEPTIBILITY: _label_metadata(CLINICAL_CONCEPT, RISK_LOW, (LOINC, SNOMED)),
    # Clinical concepts
    CONDITION: _label_metadata(CLINICAL_CONCEPT, RISK_LOW, (ICD_10_CM, SNOMED)),
    MEDICATION: _label_metadata(CLINICAL_CONCEPT, RISK_LOW, (RXNORM, SNOMED)),
    LAB_TEST: _label_metadata(CLINICAL_CONCEPT, RISK_LOW, (LOINC, SNOMED)),
    PROCEDURE: _label_metadata(CLINICAL_CONCEPT, RISK_LOW, (SNOMED,)),
    BODY_SITE: _label_metadata(CLINICAL_CONCEPT, RISK_LOW, (SNOMED,)),
    # Nutrition and diet-order concepts (issue #951)
    DIET_TYPE: _label_metadata(CLINICAL_CONCEPT, RISK_LOW, (SNOMED,)),
    NUTRITION_TARGET: _label_metadata(CLINICAL_CONCEPT, RISK_LOW, (SNOMED,)),
    FEEDING_ROUTE: _label_metadata(CLINICAL_CONCEPT, RISK_LOW, (SNOMED,)),
    NUTRITIONAL_STATUS: _label_metadata(CLINICAL_CONCEPT, RISK_LOW, (SNOMED,)),
    # Catch-all
    OTHER: _label_metadata(CLINICAL_CONCEPT, RISK_LOW, CLINICAL_SYSTEM_HINTS),
}

LABEL_TO_HIPAA: Final[Mapping[str, str]] = {
    # People
    PERSON: HIPAA_NAME,
    FIRST_NAME: HIPAA_NAME,
    LAST_NAME: HIPAA_NAME,
    MIDDLE_NAME: HIPAA_NAME,
    PREFIX: HIPAA_NAME,
    USERNAME: HIPAA_UNIQUE_IDENTIFIER,
    # Contact
    EMAIL: HIPAA_EMAIL_ADDRESS,
    PHONE: HIPAA_TELEPHONE_NUMBER,
    URL: HIPAA_URL,
    # Location
    LOCATION: HIPAA_GEOGRAPHIC_SUBDIVISION,
    STREET_ADDRESS: HIPAA_GEOGRAPHIC_SUBDIVISION,
    BUILDING_NUMBER: HIPAA_GEOGRAPHIC_SUBDIVISION,
    ZIPCODE: HIPAA_GEOGRAPHIC_SUBDIVISION,
    GPS_COORDINATES: HIPAA_GEOGRAPHIC_SUBDIVISION,
    ORDINAL_DIRECTION: HIPAA_GEOGRAPHIC_SUBDIVISION,
    # Time
    DATE: HIPAA_DATE_ELEMENT,
    DATE_OF_BIRTH: HIPAA_DATE_ELEMENT,
    TIME: HIPAA_DATE_ELEMENT,
    AGE: HIPAA_DATE_ELEMENT,
    # Identifiers
    ID_NUM: HIPAA_UNIQUE_IDENTIFIER,
    SSN: HIPAA_SOCIAL_SECURITY_NUMBER,
    ACCOUNT_NUMBER: HIPAA_ACCOUNT_NUMBER,
    PASSWORD: HIPAA_UNIQUE_IDENTIFIER,
    PIN: HIPAA_UNIQUE_IDENTIFIER,
    API_KEY: HIPAA_UNIQUE_IDENTIFIER,
    # Financial
    CREDIT_CARD: HIPAA_ACCOUNT_NUMBER,
    CREDIT_CARD_ISSUER: HIPAA_UNIQUE_IDENTIFIER,
    CVV: HIPAA_ACCOUNT_NUMBER,
    IBAN: HIPAA_ACCOUNT_NUMBER,
    BIC: HIPAA_ACCOUNT_NUMBER,
    AMOUNT: HIPAA_UNIQUE_IDENTIFIER,
    CURRENCY: HIPAA_UNIQUE_IDENTIFIER,
    BITCOIN_ADDRESS: HIPAA_ACCOUNT_NUMBER,
    ETHEREUM_ADDRESS: HIPAA_ACCOUNT_NUMBER,
    LITECOIN_ADDRESS: HIPAA_ACCOUNT_NUMBER,
    MASKED_NUMBER: HIPAA_ACCOUNT_NUMBER,
    # Demographics
    GENDER: HIPAA_UNIQUE_IDENTIFIER,
    EYE_COLOR: HIPAA_UNIQUE_IDENTIFIER,
    HEIGHT: HIPAA_UNIQUE_IDENTIFIER,
    # Work
    ORGANIZATION: HIPAA_UNIQUE_IDENTIFIER,
    JOB_TITLE: HIPAA_UNIQUE_IDENTIFIER,
    JOB_DEPARTMENT: HIPAA_UNIQUE_IDENTIFIER,
    OCCUPATION: HIPAA_UNIQUE_IDENTIFIER,
    # Tech
    IP_ADDRESS: HIPAA_IP_ADDRESS,
    MAC_ADDRESS: HIPAA_DEVICE_IDENTIFIER,
    USER_AGENT: HIPAA_UNIQUE_IDENTIFIER,
    VIN: HIPAA_VEHICLE_IDENTIFIER,
    VEHICLE_REGISTRATION: HIPAA_VEHICLE_IDENTIFIER,
    IMEI: HIPAA_DEVICE_IDENTIFIER,
    # Microbiology
    MICROORGANISM: HIPAA_UNIQUE_IDENTIFIER,
    ANTIBIOTIC: HIPAA_UNIQUE_IDENTIFIER,
    SUSCEPTIBILITY: HIPAA_UNIQUE_IDENTIFIER,
    # Clinical concepts
    CONDITION: HIPAA_UNIQUE_IDENTIFIER,
    MEDICATION: HIPAA_UNIQUE_IDENTIFIER,
    LAB_TEST: HIPAA_UNIQUE_IDENTIFIER,
    PROCEDURE: HIPAA_UNIQUE_IDENTIFIER,
    BODY_SITE: HIPAA_UNIQUE_IDENTIFIER,
    # Nutritional and Diet concepts
    DIET_TYPE: HIPAA_UNIQUE_IDENTIFIER,
    NUTRITION_TARGET: HIPAA_UNIQUE_IDENTIFIER,
    FEEDING_ROUTE: HIPAA_UNIQUE_IDENTIFIER,
    NUTRITIONAL_STATUS: HIPAA_UNIQUE_IDENTIFIER,
    # Catch-all
    OTHER: HIPAA_UNIQUE_IDENTIFIER,
}


def _validate_label_metadata() -> None:
    metadata_labels = set(LABEL_METADATA)
    hipaa_labels = set(LABEL_TO_HIPAA)
    if metadata_labels != CANONICAL_LABELS:
        missing = sorted(CANONICAL_LABELS - metadata_labels)
        extra = sorted(metadata_labels - CANONICAL_LABELS)
        raise RuntimeError(
            "LABEL_METADATA must cover CANONICAL_LABELS exactly; "
            f"missing={missing}, extra={extra}"
        )
    if hipaa_labels != CANONICAL_LABELS:
        missing = sorted(CANONICAL_LABELS - hipaa_labels)
        extra = sorted(hipaa_labels - CANONICAL_LABELS)
        raise RuntimeError(
            "LABEL_TO_HIPAA must cover CANONICAL_LABELS exactly; "
            f"missing={missing}, extra={extra}"
        )
    for label, metadata in LABEL_METADATA.items():
        policy_label = metadata["policy_label"]
        risk_level = metadata["risk_level"]
        system_hints = metadata["system_hints"]
        if policy_label not in POLICY_LABELS:
            raise RuntimeError(f"{label} has invalid policy_label {policy_label!r}")
        if risk_level not in RISK_LEVELS:
            raise RuntimeError(f"{label} has invalid risk_level {risk_level!r}")
        if not isinstance(system_hints, tuple):
            raise RuntimeError(f"{label} system_hints must be a tuple")
        if policy_label == CLINICAL_CONCEPT and not system_hints:
            raise RuntimeError(f"{label} clinical concepts require system_hints")
        if policy_label != CLINICAL_CONCEPT and system_hints:
            raise RuntimeError(f"{label} identifier labels must not have system_hints")
    for label, hipaa_class in LABEL_TO_HIPAA.items():
        if hipaa_class not in HIPAA_SAFE_HARBOR_CLASSES:
            raise RuntimeError(f"{label} has invalid HIPAA class {hipaa_class!r}")


# ---------------------------------------------------------------------------
# Alias map
# ---------------------------------------------------------------------------

# Inputs are lowercased + non-alphanumerics stripped before lookup. So
# ``first_name``, ``FIRSTNAME``, ``First-Name`` all reduce to ``firstname``.
_ALIAS_MAP: Final[Mapping[str, str]] = {
    # People
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
    # Contact
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
    # Location
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
    # Time
    "date": DATE,
    "dateofbirth": DATE_OF_BIRTH,
    "dob": DATE_OF_BIRTH,
    "birthdate": DATE_OF_BIRTH,
    "time": TIME,
    "age": AGE,
    # Identifiers
    "idnum": ID_NUM,
    "id": ID_NUM,
    "identifier": ID_NUM,
    "medicalrecordnumber": ID_NUM,
    "mrn": ID_NUM,
    "nhsnumber": ID_NUM,
    "nhs": ID_NUM,
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
    "teudatzehut": ID_NUM,
    "tz": ID_NUM,
    "npi": ID_NUM,
    "ssn": SSN,
    "socialsecuritynumber": SSN,
    "accountnumber": ACCOUNT_NUMBER,
    "accountname": ACCOUNT_NUMBER,
    "bankaccount": ACCOUNT_NUMBER,
    "password": PASSWORD,
    "pin": PIN,
    "apikey": API_KEY,
    # Financial
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
    # Demographics
    "gender": GENDER,
    "sex": GENDER,
    "eyecolor": EYE_COLOR,
    "height": HEIGHT,
    # Work
    "organization": ORGANIZATION,
    "company": ORGANIZATION,
    "employer": ORGANIZATION,
    "jobtitle": JOB_TITLE,
    "jobdepartment": JOB_DEPARTMENT,
    "department": JOB_DEPARTMENT,
    "occupation": OCCUPATION,
    "profession": OCCUPATION,
    # Tech
    "ipaddress": IP_ADDRESS,
    "ip": IP_ADDRESS,
    "macaddress": MAC_ADDRESS,
    "useragent": USER_AGENT,
    "vin": VIN,
    "vrm": VEHICLE_REGISTRATION,
    "licenseplate": VEHICLE_REGISTRATION,
    "imei": IMEI,
    # Microbiology
    "microorganism": MICROORGANISM,
    "microbe": MICROORGANISM,
    "organism": MICROORGANISM,
    "pathogen": MICROORGANISM,
    "antibiotic": ANTIBIOTIC,
    "antimicrobial": ANTIBIOTIC,
    "susceptibility": SUSCEPTIBILITY,
    "susceptibilityresult": SUSCEPTIBILITY,
    # Clinical concepts
    "condition": CONDITION,
    "disease": CONDITION,
    "diagnosis": CONDITION,
    "finding": CONDITION,
    "problem": CONDITION,
    "disorder": CONDITION,
    "syndrome": CONDITION,
    "medication": MEDICATION,
    "drug": MEDICATION,
    "chemical": MEDICATION,
    "substance": MEDICATION,
    "labtest": LAB_TEST,
    "test": LAB_TEST,
    "lab": LAB_TEST,
    "measurement": LAB_TEST,
    "analyte": LAB_TEST,
    "procedure": PROCEDURE,
    "surgery": PROCEDURE,
    "operation": PROCEDURE,
    "intervention": PROCEDURE,
    "bodysite": BODY_SITE,
    "bodypart": BODY_SITE,
    "anatomy": BODY_SITE,
    "anatomical": BODY_SITE,
    "organ": BODY_SITE,
    # Nutrition and Diet concepts
    "diettype": DIET_TYPE,
    "diet": DIET_TYPE,
    "nutritiontarget": NUTRITION_TARGET,
    "feedingroute": FEEDING_ROUTE,
    "nutritionalstatus": NUTRITIONAL_STATUS,
}


ID_ALIAS_SUBTYPES: Final[Mapping[str, str]] = {
    "medicalrecordnumber": ID_SUBTYPE_MRN,
    "mrn": ID_SUBTYPE_MRN,
    "npi": ID_SUBTYPE_NPI,
    "nhsnumber": ID_SUBTYPE_NATIONAL_ID,
    "nhs": ID_SUBTYPE_NATIONAL_ID,
    "nationalid": ID_SUBTYPE_NATIONAL_ID,
    "cpf": ID_SUBTYPE_NATIONAL_ID,
    "cnpj": ID_SUBTYPE_NATIONAL_ID,
    "nir": ID_SUBTYPE_NATIONAL_ID,
    "steuerid": ID_SUBTYPE_NATIONAL_ID,
    "codicefiscale": ID_SUBTYPE_NATIONAL_ID,
    "dni": ID_SUBTYPE_NATIONAL_ID,
    "nie": ID_SUBTYPE_NATIONAL_ID,
    "bsn": ID_SUBTYPE_NATIONAL_ID,
    "aadhaar": ID_SUBTYPE_NATIONAL_ID,
    "teudatzehut": ID_SUBTYPE_NATIONAL_ID,
    "tz": ID_SUBTYPE_NATIONAL_ID,
}


_BIOES_PREFIX_RE: Final = re.compile(r"^[BIES]-")


def _strip_bioes_prefix(label: str) -> str:
    """Strip an optional BIOES-style prefix from a label.

    ``B-NAME`` -> ``NAME``; ``I-DATE`` -> ``DATE``; ``S-EMAIL`` -> ``EMAIL``.
    Labels without a prefix are returned unchanged.
    """
    return _BIOES_PREFIX_RE.sub("", label, count=1)


def _key(label: str) -> str:
    """Lowercase, strip non-alphanumerics, drop BIOES prefix."""
    stripped = _strip_bioes_prefix(label.strip())
    return re.sub(r"[^a-z0-9]", "", stripped.lower())


def normalize_label(label: str, lang: str = "en") -> str:
    """Normalize an entity label to the canonical taxonomy.

    Accepts any of:
      - English lowercase ``snake_case`` (``first_name``)
      - Portuguese ``UPPERCASE`` no-separator (``FIRSTNAME``)
      - BIOES-tagged forms (``B-NAME``, ``I-EMAIL``)
      - Mixed case with arbitrary separators (``First-Name``, ``First Name``)

    Unknown labels fall through to ``OTHER`` rather than raising — callers
    that need strict checking should compare against ``CANONICAL_LABELS``
    explicitly.

    Args:
        label: Source label as emitted by a model or registered in a config.
        lang: ISO 639-1 language hint (currently unused but reserved for
            language-conditional disambiguation, e.g. mapping ambiguous
            tokens differently per locale).

    Returns:
        A canonical label in ``UPPER_SNAKE_CASE``.
    """
    if not label:
        return OTHER
    key = _key(label)
    if not key:
        return OTHER
    canonical = _ALIAS_MAP.get(key)
    if canonical is not None:
        return canonical
    # If the input already matches a canonical label after stripping
    # separators (e.g. ``ID_NUM`` -> key ``idnum`` -> aliased; but
    # ``CREDIT_CARD`` -> ``creditcard`` -> aliased), the alias map covers
    # it. The ``upper`` fallback handles any future canonical label not
    # yet in the alias map.
    upper = re.sub(r"[^A-Z0-9_]", "", label.upper().replace("-", "_").replace(" ", "_"))
    if upper in CANONICAL_LABELS:
        return upper
    return OTHER


def id_subtype_for(label: str, lang: str = "en") -> str | None:
    """Return optional ID_NUM subtype metadata for a source label.

    The canonical taxonomy remains flat: all values returned by this helper
    still normalize to ``ID_NUM``. Callers that only need canonical labels
    should continue to use :func:`normalize_label`.
    """
    if normalize_label(label, lang=lang) != ID_NUM:
        return None
    return ID_ALIAS_SUBTYPES.get(_key(label))


def _metadata_for(label: str, lang: str = "en") -> Mapping[str, object]:
    return LABEL_METADATA[normalize_label(label, lang=lang)]


def policy_label_for(label: str, lang: str = "en") -> str:
    """Return the policy class for a label after canonical normalization."""
    return cast(str, _metadata_for(label, lang=lang)["policy_label"])


def risk_level_for(label: str, lang: str = "en") -> str:
    """Return the residual-risk level for a label after canonical normalization."""
    return cast(str, _metadata_for(label, lang=lang)["risk_level"])


def system_hints_for(label: str, lang: str = "en") -> tuple[str, ...]:
    """Return candidate clinical coding systems for a normalized label."""
    return cast(tuple[str, ...], _metadata_for(label, lang=lang)["system_hints"])


def hipaa_class_for(label: str, lang: str = "en") -> str:
    """Return the outbound HIPAA Safe Harbor class for a normalized label."""
    return LABEL_TO_HIPAA[normalize_label(label, lang=lang)]


_validate_label_metadata()


__all__ = [
    "CANONICAL_LABELS",
    "normalize_label",
    "id_subtype_for",
    "ID_ALIAS_SUBTYPES",
    "ID_SUBTYPES",
    "ID_SUBTYPE_MRN",
    "ID_SUBTYPE_NPI",
    "ID_SUBTYPE_NATIONAL_ID",
    "ID_SUBTYPE_SSN_ADJACENT",
    "LABEL_METADATA",
    "LABEL_TO_HIPAA",
    "POLICY_LABELS",
    "DIRECT_IDENTIFIER",
    "QUASI_IDENTIFIER",
    "CLINICAL_CONCEPT",
    "RISK_LEVELS",
    "RISK_LOW",
    "RISK_MEDIUM",
    "RISK_HIGH",
    "CLINICAL_SYSTEM_HINTS",
    "HIPAA_SAFE_HARBOR_CLASSES",
    "policy_label_for",
    "risk_level_for",
    "system_hints_for",
    "hipaa_class_for",
    # canonical label constants
    "PERSON",
    "FIRST_NAME",
    "LAST_NAME",
    "MIDDLE_NAME",
    "PREFIX",
    "USERNAME",
    "EMAIL",
    "PHONE",
    "URL",
    "LOCATION",
    "STREET_ADDRESS",
    "BUILDING_NUMBER",
    "ZIPCODE",
    "GPS_COORDINATES",
    "ORDINAL_DIRECTION",
    "DATE",
    "DATE_OF_BIRTH",
    "TIME",
    "AGE",
    "ID_NUM",
    "SSN",
    "ACCOUNT_NUMBER",
    "PASSWORD",
    "PIN",
    "API_KEY",
    "CREDIT_CARD",
    "CREDIT_CARD_ISSUER",
    "CVV",
    "IBAN",
    "BIC",
    "AMOUNT",
    "CURRENCY",
    "BITCOIN_ADDRESS",
    "ETHEREUM_ADDRESS",
    "LITECOIN_ADDRESS",
    "MASKED_NUMBER",
    "GENDER",
    "EYE_COLOR",
    "HEIGHT",
    "ORGANIZATION",
    "JOB_TITLE",
    "JOB_DEPARTMENT",
    "OCCUPATION",
    "IP_ADDRESS",
    "MAC_ADDRESS",
    "USER_AGENT",
    "VIN",
    "VEHICLE_REGISTRATION",
    "IMEI",
    "MICROORGANISM",
    "ANTIBIOTIC",
    "SUSCEPTIBILITY",
    "CONDITION",
    "MEDICATION",
    "LAB_TEST",
    "PROCEDURE",
    "BODY_SITE",
    "DIET_TYPE",
    "NUTRITION_TARGET",
    "FEEDING_ROUTE",
    "NUTRITIONAL_STATUS",
    "OTHER",
]
