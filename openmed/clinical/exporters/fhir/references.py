"""Deterministic FHIR Bundle reference helpers.

Use :func:`deterministic_fullurl` when an exporter needs to pre-compute the
``urn:uuid`` fullUrl that :func:`openmed.clinical.exporters.fhir.to_bundle`
will assign to a resource at a known position. The helper shares the Bundle
assembler's namespace and seed format, so exporter-provided references created
with it survive Bundle assembly unchanged.
"""

from __future__ import annotations

import uuid

__all__ = ["deterministic_fullurl"]

_BUNDLE_NAMESPACE = uuid.uuid5(
    uuid.NAMESPACE_URL,
    "https://openmed.ai/fhir/bundle",
)


def deterministic_fullurl(doc_id: str, index: int) -> str:
    """Return the deterministic Bundle fullUrl for ``doc_id`` and ``index``."""

    name = f"{doc_id}:{index}"
    return f"urn:uuid:{uuid.uuid5(_BUNDLE_NAMESPACE, name)}"
