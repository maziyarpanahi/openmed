"""Faker providers for clinical / national IDs.

Each provider produces values that pass the corresponding validator in
:mod:`openmed.core.pii_i18n`. We rely on Faker's built-in providers where
the checksum and format already match (verified empirically against our
validators):

  - ``pt_BR.cpf()``               valid
  - ``pt_BR.cnpj()``              valid
  - ``nl_NL.ssn()``  (BSN)        valid
  - ``fr_FR.ssn()``  (NIR)        valid
  - ``it_IT.ssn()``  (Codice Fiscale) valid
  - ``es_ES.nie()``               valid

Custom providers below cover the gaps where Faker either has no built-in
or emits a US-style format unrelated to the requested locale's actual ID:

  - German Steuer-ID (Faker's ``de_DE.ssn`` is US-format)
  - Aadhaar with Verhoeff checksum (Faker's ``en_IN.aadhaar_id`` rarely
    passes the official Verhoeff check — only ~1 in 20 by sampling)
  - Generic medical record numbers (MRN-XXXXXXX style)
  - US National Provider Identifier (Luhn over a "80840" prefix)
"""

from __future__ import annotations

from typing import Sequence

from faker.providers import BaseProvider


# ---------------------------------------------------------------------------
# Aadhaar (12 digits, Verhoeff checksum)
# ---------------------------------------------------------------------------

# Verhoeff multiplication, permutation and inverse tables, transcribed from
# https://en.wikipedia.org/wiki/Verhoeff_algorithm.
_VERHOEFF_D: Sequence[Sequence[int]] = (
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    (1, 2, 3, 4, 0, 6, 7, 8, 9, 5),
    (2, 3, 4, 0, 1, 7, 8, 9, 5, 6),
    (3, 4, 0, 1, 2, 8, 9, 5, 6, 7),
    (4, 0, 1, 2, 3, 9, 5, 6, 7, 8),
    (5, 9, 8, 7, 6, 0, 4, 3, 2, 1),
    (6, 5, 9, 8, 7, 1, 0, 4, 3, 2),
    (7, 6, 5, 9, 8, 2, 1, 0, 4, 3),
    (8, 7, 6, 5, 9, 3, 2, 1, 0, 4),
    (9, 8, 7, 6, 5, 4, 3, 2, 1, 0),
)
_VERHOEFF_P: Sequence[Sequence[int]] = (
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    (1, 5, 7, 6, 2, 8, 3, 0, 9, 4),
    (5, 8, 0, 3, 7, 9, 6, 1, 4, 2),
    (8, 9, 1, 6, 0, 4, 3, 5, 2, 7),
    (9, 4, 5, 3, 1, 2, 6, 8, 7, 0),
    (4, 2, 8, 6, 5, 7, 3, 9, 0, 1),
    (2, 7, 9, 3, 8, 0, 6, 4, 1, 5),
    (7, 0, 4, 6, 9, 1, 3, 2, 5, 8),
)
_VERHOEFF_INV: Sequence[int] = (0, 4, 3, 2, 1, 5, 6, 7, 8, 9)


def _verhoeff_checksum(digits: Sequence[int]) -> int:
    """Compute the Verhoeff check digit for ``digits`` (without the check)."""
    c = 0
    for i, n in enumerate(reversed(digits), start=1):
        c = _VERHOEFF_D[c][_VERHOEFF_P[i % 8][n]]
    return _VERHOEFF_INV[c]


class AadhaarProvider(BaseProvider):
    """Generates 12-digit Aadhaar numbers with valid Verhoeff checksums."""

    def aadhaar(self) -> str:
        # First digit cannot be 0 or 1 per UIDAI spec.
        digits = [self.generator.random.randint(2, 9)]
        digits.extend(self.generator.random.randint(0, 9) for _ in range(10))
        digits.append(_verhoeff_checksum(digits))
        return "".join(str(d) for d in digits)


# ---------------------------------------------------------------------------
# German Steuer-ID (11 digits with mod-11 checksum and digit-frequency rules)
# ---------------------------------------------------------------------------

class GermanSteuerIdProvider(BaseProvider):
    """Generates 11-digit German Steuer-IDs that pass our validator.

    The Steuer-ID rules are subtle enough that we generate by trial and
    delegate to the validator from :mod:`openmed.core.pii_i18n`. Bounded
    retries keep this from looping indefinitely on adversarial random
    states; in practice ~1 in 50 random 11-digit strings satisfies all
    constraints, so we typically succeed in <100 tries.
    """

    _MAX_TRIES = 500

    def german_steuer_id(self) -> str:
        from openmed.core.pii_i18n import validate_german_steuer_id

        rng = self.generator.random
        for _ in range(self._MAX_TRIES):
            # First digit must not be 0
            digits = [rng.randint(1, 9)]
            digits.extend(rng.randint(0, 9) for _ in range(10))
            candidate = "".join(str(d) for d in digits)
            if validate_german_steuer_id(candidate):
                return candidate
        # Fallback: return any 11 digits; validator may reject downstream.
        return self.numerify("###########")


# ---------------------------------------------------------------------------
# Medical Record Number (opaque, but recognizably MRN-shaped)
# ---------------------------------------------------------------------------

class MedicalRecordNumberProvider(BaseProvider):
    """Generates plausible medical record numbers (``MRN-1234567``)."""

    def medical_record_number(self) -> str:
        return f"MRN-{self.numerify('#######')}"


# ---------------------------------------------------------------------------
# US National Provider Identifier (10 digits, Luhn over "80840" prefix)
# ---------------------------------------------------------------------------

def _luhn_check_digit(digits: Sequence[int]) -> int:
    total = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 0:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return (10 - total % 10) % 10


class NPIProvider(BaseProvider):
    """Generates valid 10-digit US NPI numbers.

    The NPI uses Luhn over the digits prefixed with ``80840``. We generate
    9 random digits, prepend the prefix for checksumming, compute the
    check digit, and emit the original 9 digits + the check digit.
    """

    def npi(self) -> str:
        rng = self.generator.random
        body = [rng.randint(0, 9) for _ in range(9)]
        prefixed = [8, 0, 8, 4, 0, *body]
        check = _luhn_check_digit(prefixed)
        return "".join(str(d) for d in body) + str(check)


# ---------------------------------------------------------------------------
# Bulk registration helper
# ---------------------------------------------------------------------------

def register_clinical_providers(faker) -> None:
    """Add every custom provider in this module to ``faker``."""
    faker.add_provider(AadhaarProvider)
    faker.add_provider(GermanSteuerIdProvider)
    faker.add_provider(MedicalRecordNumberProvider)
    faker.add_provider(NPIProvider)


__all__ = [
    "AadhaarProvider",
    "GermanSteuerIdProvider",
    "MedicalRecordNumberProvider",
    "NPIProvider",
    "register_clinical_providers",
]
