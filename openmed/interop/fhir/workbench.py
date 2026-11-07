"""Compatibility import for the clinical FHIR exchange workbench."""

from __future__ import annotations

from openmed.clinical.exporters.fhir.exchange import (
    FHIRClinicalExchangeWorkbench,
    FHIRExchange,
    FHIRExchangeError,
    FHIRExchangeWorkbench,
    FHIRValidationError,
    build_clinical_document,
    build_fhir_document,
    build_ipa_example,
    build_ipa_patient_access,
    build_ips_patient_summary,
    build_patient_summary,
    deidentify_bundle,
    deidentify_fhir,
    export_bundle,
    export_fhir,
    import_bundle,
    import_fhir,
    validate_exchange,
)

__all__ = [
    "FHIRClinicalExchangeWorkbench",
    "FHIRExchange",
    "FHIRExchangeError",
    "FHIRExchangeWorkbench",
    "FHIRValidationError",
    "build_clinical_document",
    "build_fhir_document",
    "build_ipa_example",
    "build_ipa_patient_access",
    "build_ips_patient_summary",
    "build_patient_summary",
    "deidentify_fhir",
    "deidentify_bundle",
    "export_bundle",
    "export_fhir",
    "import_bundle",
    "import_fhir",
    "validate_exchange",
]
