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

_ANNOTATION_TOOL_EXPORTS = frozenset(
    {
        "export_fact_correction_rows",
        "export_registry_label_rows",
        "import_fact_correction_rows",
        "import_registry_label_rows",
    }
)
_PIPELINE_MIGRATION_EXPORTS = frozenset(
    {
        "MAX_PIPELINE_DESCRIPTION_BYTES",
        "MAX_PIPELINE_JSON_DEPTH",
        "MAX_PIPELINE_JSON_NODES",
        "MAX_PIPELINE_STAGES",
        "MAX_PIPELINE_STRING_CHARS",
        "PIPELINE_MIGRATION_COMPATIBILITY",
        "PIPELINE_MIGRATION_SCHEMA_VERSION",
        "PipelineMigrationError",
        "PipelineMigrationReport",
        "PipelineMigrationState",
        "PipelineStageDisposition",
        "PipelineStageMigration",
        "load_pipeline_migration_schema",
        "scan_pipeline_json",
        "scan_pipeline_mapping",
    }
)


def __getattr__(name: str) -> object:
    """Load new in-process bridges lazily to keep package imports acyclic."""

    if name in _ANNOTATION_TOOL_EXPORTS:
        from . import annotation_tools

        value = getattr(annotation_tools, name)
        globals()[name] = value
        return value
    if name in _PIPELINE_MIGRATION_EXPORTS:
        from . import pipeline_migration

        value = getattr(pipeline_migration, name)
        globals()[name] = value
        return value
    raise AttributeError(name)


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
    "MAX_PIPELINE_DESCRIPTION_BYTES",
    "MAX_PIPELINE_JSON_DEPTH",
    "MAX_PIPELINE_JSON_NODES",
    "MAX_PIPELINE_STAGES",
    "MAX_PIPELINE_STRING_CHARS",
    "PIPELINE_MIGRATION_COMPATIBILITY",
    "PIPELINE_MIGRATION_SCHEMA_VERSION",
    "PipelineMigrationError",
    "PipelineMigrationReport",
    "PipelineMigrationState",
    "PipelineStageDisposition",
    "PipelineStageMigration",
    "load_icd10cn_crosswalk",
    "load_pipeline_migration_schema",
    "map_icd10_to_icd10cn",
    "map_icd10cn_code",
    "export_fact_correction_rows",
    "export_registry_label_rows",
    "import_fact_correction_rows",
    "import_registry_label_rows",
    "scan_pipeline_json",
    "scan_pipeline_mapping",
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
