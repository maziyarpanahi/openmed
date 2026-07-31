"""Telugu reference language pack."""

from ..language_pack import LanguagePack

TELUGU_LANGUAGE_PACK = LanguagePack(
    code="te",
    scripts=("Telugu",),
    default_model="OpenMed/OpenMed-PII-Telugu-SuperClinical-Large-434M-v1",
    segmenter_id="pysbd",
    recognizers=("builtin-patterns", "model"),
    # Faker has no native Telugu locale. The existing locale contract uses the
    # installed en_IN backend for generic values while script-aware providers
    # keep Telugu names in-script.
    surrogate_locale="en_IN",
    surrogate_locale_approximation=(
        "Faker has no native Telugu locale; generic fields use en_IN"
    ),
    national_id_providers={"aadhaar": "en_IN"},
    policy_overrides={"profile": "strict_no_leak"},
)
"""Complete Telugu-script declaration with an explicit locale approximation."""

__all__ = ["TELUGU_LANGUAGE_PACK"]
