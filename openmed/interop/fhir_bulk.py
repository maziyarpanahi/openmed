"""Backward-compatible import path for the FHIR Bulk Data gateway.

The canonical implementation lives in :mod:`openmed.interop.fhir.bulk`.
The compatibility path warns from 4.3 and is scheduled for removal in 5.0.
"""

from openmed.utils import deprecation as _deprecation

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

_deprecation.deprecated_module(
    __name__,
    since="4.3",
    remove_in="5.0",
    replacement="openmed.interop.fhir.bulk",
    warn_on_import=True,
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
