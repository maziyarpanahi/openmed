"""Custom Faker providers for clinical / national IDs that Faker doesn't
cover natively or covers with the wrong format.

These providers integrate with Faker's standard ``add_provider`` mechanism
and produce values that pass the existing checksum validators in
:mod:`openmed.core.pii_i18n`.
"""

from .clinical_ids import (
    AadhaarProvider,
    GermanSteuerIdProvider,
    MedicalRecordNumberProvider,
    NPIProvider,
    register_clinical_providers,
)

__all__ = [
    "AadhaarProvider",
    "GermanSteuerIdProvider",
    "MedicalRecordNumberProvider",
    "NPIProvider",
    "register_clinical_providers",
]
