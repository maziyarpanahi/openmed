"""Optional out-of-process interoperability bridges.

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
from .cohort_definition import (
    COHORT_SERVICE_COMPATIBILITY_POLICY,
    COHORT_SERVICE_PROTOCOL_VERSION,
    DEFAULT_COHORT_SERVICE_TIMEOUT_SECONDS,
    MAX_COHORT_SERVICE_RESPONSE_BYTES,
    CohortDefinitionServiceBridge,
    CohortServiceBridgeError,
    CohortServiceConversion,
    CohortServiceProtocolError,
    CohortServiceUnavailableError,
)
from .icd10cn import (
    ICD10CNBridge,
    ICD10CNMapping,
    load_icd10cn_crosswalk,
    map_icd10_to_icd10cn,
    map_icd10cn_code,
)
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
    "ARX_PROTOCOL_SCHEMA_VERSION",
    "COHORT_SERVICE_COMPATIBILITY_POLICY",
    "COHORT_SERVICE_PROTOCOL_VERSION",
    "DEFAULT_COHORT_SERVICE_TIMEOUT_SECONDS",
    "MAX_COHORT_SERVICE_RESPONSE_BYTES",
    "DEFAULT_ARX_TIMEOUT_SECONDS",
    "ArxBridge",
    "ArxBridgeError",
    "ArxNotAvailableError",
    "ArxProtocolError",
    "ArxResult",
    "CohortDefinitionServiceBridge",
    "CohortServiceBridgeError",
    "CohortServiceConversion",
    "CohortServiceProtocolError",
    "CohortServiceUnavailableError",
    "ICD10CNBridge",
    "ICD10CNMapping",
    "load_icd10cn_crosswalk",
    "map_icd10_to_icd10cn",
    "map_icd10cn_code",
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
