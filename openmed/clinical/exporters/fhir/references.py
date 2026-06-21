from __future__ import annotations

import uuid

__all__ = ["deterministic_fullurl"]

_BUNDLE_NAMESPACE = uuid.uuid5(
    uuid.NAMESPACE_URL,
    "https://openmed.ai/fhir/bundle",
)


def deterministic_fullurl(doc_id: str, index: int) -> str:
    """Return the deterministic Bundle fullUrl for a resource."""

    name = f"{doc_id}:{index}"
    return f"urn:uuid:{uuid.uuid5(_BUNDLE_NAMESPACE, name)}"