"""Canonical PII/PHI label taxonomy.

Different OpenMed PII model families use different label-naming conventions:

- English / multilingual SuperClinical models use lowercase ``snake_case``
  (``first_name``, ``date_of_birth``).
- Portuguese models expose 52 ``UPPERCASE`` labels (``FIRSTNAME``,
  ``DATEOFBIRTH``).
- The OpenAI privacy-filter family emits BIOES-tagged labels (``B-NAME``,
  ``I-EMAIL``, ``E-ADDRESS``, ``S-PHONE``).

This module provides a single ``CANONICAL_LABELS`` taxonomy in
``UPPER_SNAKE_CASE`` and a ``normalize_label`` helper that maps any of the
above input forms to its canonical name. Downstream code (anonymization,
mapping tables, configuration) should key off canonical labels only.
"""

from __future__ import annotations

import re
from typing import Final, FrozenSet, Mapping


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

#: Catch-all
OTHER: Final = "OTHER"


CANONICAL_LABELS: Final[FrozenSet[str]] = frozenset({
    PERSON, FIRST_NAME, LAST_NAME, MIDDLE_NAME, PREFIX, USERNAME,
    EMAIL, PHONE, URL,
    LOCATION, STREET_ADDRESS, BUILDING_NUMBER, ZIPCODE, GPS_COORDINATES,
    ORDINAL_DIRECTION,
    DATE, DATE_OF_BIRTH, TIME, AGE,
    ID_NUM, SSN, ACCOUNT_NUMBER, PASSWORD, PIN, API_KEY,
    CREDIT_CARD, CREDIT_CARD_ISSUER, CVV, IBAN, BIC, AMOUNT, CURRENCY,
    BITCOIN_ADDRESS, ETHEREUM_ADDRESS, LITECOIN_ADDRESS, MASKED_NUMBER,
    GENDER, EYE_COLOR, HEIGHT,
    ORGANIZATION, JOB_TITLE, JOB_DEPARTMENT, OCCUPATION,
    IP_ADDRESS, MAC_ADDRESS, USER_AGENT, VIN, VEHICLE_REGISTRATION, IMEI,
    OTHER,
})


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


__all__ = [
    "CANONICAL_LABELS",
    "normalize_label",
    # canonical label constants
    "PERSON", "FIRST_NAME", "LAST_NAME", "MIDDLE_NAME", "PREFIX", "USERNAME",
    "EMAIL", "PHONE", "URL",
    "LOCATION", "STREET_ADDRESS", "BUILDING_NUMBER", "ZIPCODE",
    "GPS_COORDINATES", "ORDINAL_DIRECTION",
    "DATE", "DATE_OF_BIRTH", "TIME", "AGE",
    "ID_NUM", "SSN", "ACCOUNT_NUMBER", "PASSWORD", "PIN", "API_KEY",
    "CREDIT_CARD", "CREDIT_CARD_ISSUER", "CVV", "IBAN", "BIC",
    "AMOUNT", "CURRENCY",
    "BITCOIN_ADDRESS", "ETHEREUM_ADDRESS", "LITECOIN_ADDRESS",
    "MASKED_NUMBER",
    "GENDER", "EYE_COLOR", "HEIGHT",
    "ORGANIZATION", "JOB_TITLE", "JOB_DEPARTMENT", "OCCUPATION",
    "IP_ADDRESS", "MAC_ADDRESS", "USER_AGENT", "VIN",
    "VEHICLE_REGISTRATION", "IMEI",
    "OTHER",
]
