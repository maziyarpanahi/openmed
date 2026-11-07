"""Built-in reference language-pack declarations.

Definitions in this package are intentionally inert: importing an individual
module creates an immutable :class:`~openmed.core.language_pack.LanguagePack`
value but does not mutate the process-local registry. The built-in catalog is
the single bootstrap that registers these declarations.
"""

from .chinese import CHINESE_LANGUAGE_PACK
from .hindi import HINDI_LANGUAGE_PACK
from .telugu import TELUGU_LANGUAGE_PACK

REFERENCE_LANGUAGE_PACKS = (
    CHINESE_LANGUAGE_PACK,
    HINDI_LANGUAGE_PACK,
    TELUGU_LANGUAGE_PACK,
)
"""Cross-script packs that demonstrate the public onboarding contract."""

__all__ = [
    "CHINESE_LANGUAGE_PACK",
    "HINDI_LANGUAGE_PACK",
    "REFERENCE_LANGUAGE_PACKS",
    "TELUGU_LANGUAGE_PACK",
]
