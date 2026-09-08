"""Backward-compatible import path for the FHIR Bulk Data gateway.

The canonical implementation lives in :mod:`openmed.interop.fhir.bulk`.
Existing integrations can continue importing ``openmed.interop.fhir_bulk``.
"""

from .fhir.bulk import (
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
    "FHIRBulkGateway",
    "BulkExportSummary",
    "BulkGatewayConfig",
    "FHIRBulkConfig",
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
