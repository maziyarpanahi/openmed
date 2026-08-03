"""Optional out-of-process interoperability bridges.

Permissive-only adapters may run in-process; GPL or source-available tools must
be reached strictly through subprocess bridges to preserve invariant I2.
"""

from .snomed_terminology_bridge import (
    DEFAULT_SCTID_PATTERN,
    SNOMED_SYSTEM_URI,
    SNOMED_TERMINOLOGY_SYSTEM,
    SNOMEDTerminologyBridge,
    SnomedTerminologyBridge,
    SNOMEDTerminologyBridgeError,
    SNOMEDTerminologyConfig,
    SNOMEDTerminologyConfigurationError,
    SNOMEDTerminologyServerError,
)

__all__ = [
    "DEFAULT_SCTID_PATTERN",
    "SNOMED_SYSTEM_URI",
    "SNOMED_TERMINOLOGY_SYSTEM",
    "SNOMEDTerminologyBridge",
    "SNOMEDTerminologyBridgeError",
    "SNOMEDTerminologyConfig",
    "SNOMEDTerminologyConfigurationError",
    "SNOMEDTerminologyServerError",
    "SnomedTerminologyBridge",
]
