"""Custom Faker providers for clinical / national IDs that Faker doesn't
cover natively or covers with the wrong format.

These providers integrate with Faker's standard ``add_provider`` mechanism
and produce values that pass the existing checksum validators in
:mod:`openmed.core.pii_i18n`.
"""

from .clinical_ids import (
    AadhaarProvider,
    GermanSteuerIdProvider,
    IndonesianNIKProvider,
    IsraeliTeudatZehutProvider,
    KoreanRRNProvider,
    LatvianPersonasKodsProvider,
    MedicalRecordNumberProvider,
    NPIProvider,
    PolishPeselProvider,
    SpanishDNIProvider,
    generate_indonesian_nik,
    generate_korean_rrn,
    generate_latvian_personas_kods,
    generate_pesel,
    generate_teudat_zehut,
    register_clinical_providers,
)
from .registry_ids import (
    ID_PROVIDER_REGISTRY,
    NationalIdSpec,
    get_national_id,
    register_national_id,
)

__all__ = [
    "AadhaarProvider",
    "GermanSteuerIdProvider",
    "ID_PROVIDER_REGISTRY",
    "IndonesianNIKProvider",
    "IsraeliTeudatZehutProvider",
    "KoreanRRNProvider",
    "LatvianPersonasKodsProvider",
    "MedicalRecordNumberProvider",
    "NPIProvider",
    "NationalIdSpec",
    "PolishPeselProvider",
    "SpanishDNIProvider",
    "generate_indonesian_nik",
    "generate_teudat_zehut",
    "generate_korean_rrn",
    "generate_latvian_personas_kods",
    "generate_pesel",
    "get_national_id",
    "register_clinical_providers",
    "register_national_id",
]
