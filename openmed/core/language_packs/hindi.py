"""Hindi reference language pack."""

from ..language_pack import LanguagePack

HINDI_LANGUAGE_PACK = LanguagePack(
    code="hi",
    scripts=("Devanagari",),
    default_model="OpenMed/OpenMed-PII-Hindi-SuperClinical-Large-434M-v1",
    segmenter_id="pysbd",
    recognizers=("builtin-patterns", "model"),
    surrogate_locale="hi_IN",
    national_id_providers={"aadhaar": "hi_IN"},
    policy_overrides={"profile": "strict_no_leak"},
)
"""Complete Devanagari-script declaration registered by the built-in catalog."""

__all__ = ["HINDI_LANGUAGE_PACK"]
