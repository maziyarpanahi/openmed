"""FHIR interoperability helpers.

The package keeps the bulk-data gateway separate from the older single-resource
FHIR operation module while retaining a small, importable public surface.
"""

from .bulk import (
    BULK_DATA_VERSION,
    DEFAULT_MAX_BUFFERED_RESOURCES,
    SUPPORTED_R4_RESOURCE_TYPES,
    BulkDataGateway,
    BulkExportSummary,
    BulkGatewayConfig,
    BulkJobCancelled,
    BulkRejection,
    FHIRBulkConfig,
    FHIRBulkGateway,
    FHIRBulkJobReport,
    FHIRNDJSONLineError,
    NDJSONFileSummary,
    NDJSONLineError,
    RejectionRecord,
    deidentify_export,
    deidentify_ndjson,
    deidentify_ndjson_async,
    deidentify_ndjson_stream,
    iter_ndjson,
)

__all__ = [
    "BULK_DATA_VERSION",
    "DEFAULT_MAX_BUFFERED_RESOURCES",
    "SUPPORTED_R4_RESOURCE_TYPES",
    "BulkDataGateway",
    "BulkExportSummary",
    "BulkGatewayConfig",
    "FHIRBulkConfig",
    "FHIRBulkGateway",
    "BulkJobCancelled",
    "BulkRejection",
    "FHIRBulkJobReport",
    "FHIRNDJSONLineError",
    "NDJSONFileSummary",
    "NDJSONLineError",
    "RejectionRecord",
    "deidentify_export",
    "deidentify_ndjson",
    "deidentify_ndjson_async",
    "deidentify_ndjson_stream",
    "iter_ndjson",
]
