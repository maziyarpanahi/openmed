"""Backward-compatible import path for the FHIR Bulk Data gateway.

The canonical implementation lives in :mod:`openmed.interop.fhir.bulk`.
Existing integrations can continue importing ``openmed.interop.fhir_bulk``.
"""

from .fhir.bulk import *  # noqa: F401,F403
from .fhir.bulk import __all__
