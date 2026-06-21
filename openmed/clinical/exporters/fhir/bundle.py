"""Assemble exported FHIR resources into a deterministic R4 transaction Bundle.

Per-resource exporters emit standalone resources (``Condition``,
``Observation``, ``DiagnosticReport``, ...). A FHIR server, however, ingests a
single transaction *Bundle*, and the resources inside it must cross-reference
one another through the Bundle (``Condition.subject`` -> Patient,
``DiagnosticReport.result`` -> Observation, ``Observation.encounter`` ->
Encounter).

:func:`to_bundle` wraps a document's resources into a valid R4 Bundle:

* every resource is given a stable ``urn:uuid`` ``fullUrl`` (seeded by
  ``doc_id`` + resource index, so the output is byte-stable across runs);
* every literal reference (``"ResourceType/id"``) that targets a resource
  present in the Bundle is rewritten to point at that resource's ``fullUrl``,
  so there are no dangling internal references;
* for ``transaction``/``batch`` bundles, each entry carries the ``request``
  block (``method``/``url``) a server needs to create the resource.

The assembler is purely mechanical: it never synthesises resources (a Patient
removed by de-identification stays absent) and does not validate profiles.
"""

from __future__ import annotations

import uuid
from typing import Any, Mapping, Sequence

__all__ = ["to_bundle"]

# Fixed namespace so the generated ``urn:uuid`` values are reproducible across
# runs and machines (uuid5 is a deterministic hash of namespace + name).
_BUNDLE_NAMESPACE = uuid.uuid5(
    uuid.NAMESPACE_URL,
    "https://openmed.ai/fhir/bundle",
)

# Bundle types whose entries must carry a ``request`` block.
_REQUEST_BUNDLE_TYPES = frozenset({"transaction", "batch"})


def to_bundle(
    resources: Sequence[Mapping[str, Any]],
    *,
    doc_id: str = "openmed-document",
    bundle_type: str = "transaction",
) -> dict[str, Any]:
    """Assemble ``resources`` into a single R4 Bundle.

    Parameters
    ----------
    resources:
        The standalone FHIR resources to wrap, in the order they should appear
        in the Bundle. Each must be a mapping carrying a ``resourceType``.
    doc_id:
        Stable identifier for the source document. Together with the resource
        index it seeds the deterministic ``urn:uuid`` ``fullUrl`` of each
        entry, so the same input always produces byte-identical output.
    bundle_type:
        The Bundle ``type`` (defaults to ``"transaction"``). For
        ``transaction``/``batch`` bundles each entry also gets a ``request``
        block.

    Returns
    -------
    dict
        A ``resourceType=Bundle`` mapping with one entry per resource.

    Raises
    ------
    ValueError
        If a resource lacks a ``resourceType``, or if two resources share the
        same ``resourceType`` and ``id``. Duplicate ids would silently
        overwrite one another in the internal reference map, corrupting
        cross-references, so they are rejected explicitly. Resources without
        an ``id`` remain valid (they are simply unreferenceable).
    """

    entries: list[dict[str, Any]] = []
    urns: list[str] = []
    for index, resource in enumerate(resources):
        if not isinstance(resource, Mapping) or "resourceType" not in resource:
            raise ValueError(
                f"resource at index {index} is not a FHIR resource "
                "(missing 'resourceType')"
            )
        urns.append(_stable_urn(doc_id, index))

    # Map ``"ResourceType/id"`` -> ``fullUrl`` so references can be resolved
    # against the resources that are actually present in this Bundle.
    reference_map: dict[str, str] = {}
    for urn, resource in zip(urns, resources):
        resource_id = resource.get("id")
        if resource_id is not None:
            ref_key = f"{resource['resourceType']}/{resource_id}"
            if ref_key in reference_map:
                raise ValueError(
                    f"duplicate resource id {ref_key!r}: two resources share "
                    "the same resourceType and id, which would silently "
                    "corrupt internal cross-references"
                )
            reference_map[ref_key] = urn

    emit_request = bundle_type in _REQUEST_BUNDLE_TYPES
    for urn, resource in zip(urns, resources):
        rewritten = _rewrite_references(resource, reference_map)
        entry: dict[str, Any] = {"fullUrl": urn, "resource": rewritten}
        if emit_request:
            entry["request"] = {
                "method": "POST",
                "url": resource["resourceType"],
            }
        entries.append(entry)

    return {"resourceType": "Bundle", "type": bundle_type, "entry": entries}


def _stable_urn(doc_id: str, index: int) -> str:
    """Return a reproducible ``urn:uuid`` for ``doc_id`` at ``index``."""

    name = f"{doc_id}:{index}"
    return f"urn:uuid:{uuid.uuid5(_BUNDLE_NAMESPACE, name)}"


def _rewrite_references(node: Any, reference_map: Mapping[str, str]) -> Any:
    """Deep-copy ``node``, rewriting in-Bundle references to their ``fullUrl``.

    Any ``{"reference": "ResourceType/id"}`` whose target is present in
    ``reference_map`` is repointed at the corresponding ``urn:uuid``.
    References to resources absent from the Bundle (e.g. a de-identified
    Patient) are left untouched. The input is never mutated.
    """

    if isinstance(node, Mapping):
        result: dict[str, Any] = {}
        for key, value in node.items():
            if key == "reference" and isinstance(value, str) and value in reference_map:
                result[key] = reference_map[value]
            else:
                result[key] = _rewrite_references(value, reference_map)
        return result
    if isinstance(node, (list, tuple)):
        return [_rewrite_references(item, reference_map) for item in node]
    return node
