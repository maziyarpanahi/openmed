"""Chinese reference language pack."""

from ..language_pack import LanguagePack

CHINESE_LANGUAGE_PACK = LanguagePack(
    code="zh",
    scripts=("Han",),
    default_model="OpenMed/OpenMed-PII-Chinese-BigMed-Large-560M-v1",
    segmenter_id="jieba",
    recognizers=("builtin-patterns", "model"),
    surrogate_locale="zh_CN",
    national_id_providers={"chinese_resident_id": "zh_CN"},
    policy_overrides={"profile": "strict_no_leak"},
)
"""Complete Han-script declaration registered by the built-in catalog."""

__all__ = ["CHINESE_LANGUAGE_PACK"]
