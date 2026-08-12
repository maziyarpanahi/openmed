"""Subprocess bridge package for interoperability integrations.

Permissive-only adapters may run in-process; GPL or source-available tools must
be reached strictly through subprocess bridges to preserve invariant I2.
"""

from .arx import (
    ARX_PROTOCOL_SCHEMA_VERSION,
    DEFAULT_ARX_TIMEOUT_SECONDS,
    ArxBridge,
    ArxBridgeError,
    ArxNotAvailableError,
    ArxProtocolError,
    ArxResult,
)
from .icd10cn import (
    ICD10CNBridge,
    ICD10CNMapping,
    load_icd10cn_crosswalk,
    map_icd10_to_icd10cn,
    map_icd10cn_code,
)

__all__ = [
    "ARX_PROTOCOL_SCHEMA_VERSION",
    "DEFAULT_ARX_TIMEOUT_SECONDS",
    "ArxBridge",
    "ArxBridgeError",
    "ArxNotAvailableError",
    "ArxProtocolError",
    "ArxResult",
    "ICD10CNBridge",
    "ICD10CNMapping",
    "load_icd10cn_crosswalk",
    "map_icd10_to_icd10cn",
    "map_icd10cn_code",
]
