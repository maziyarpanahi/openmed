"""FHIR ``$de-identify`` operation logic over the OpenMed privacy pipeline.

FHIR servers expose a ``$de-identify`` operation that accepts a resource or a
Bundle and returns a de-identified copy, driven by a ``Parameters`` input. This
module implements that operation's *logic* (not an HTTP endpoint): it walks a
resource's string-typed elements and its ``text`` narrative (XHTML), runs each
free-text value through the existing :func:`openmed.core.pii.deidentify`
pipeline, and returns the transformed resource.

Three entry points are exposed:

* :func:`de_identify_resource` de-identifies a single resource.
* :func:`de_identify_bundle` applies the operation across every Bundle entry
  while preserving Bundle structure and references.
* :func:`de_identify` accepts and returns a FHIR ``Parameters`` resource, the
  ``$de-identify`` input/output envelope, and reports the modified element
  paths as an :class:`~openmed.clinical.exporters.fhir.OperationOutcome`.

Design constraints:

* Only free-text string content is transformed. Coded elements (``Coding``,
  ``CodeableConcept.coding``), identifiers, references, systems, and temporal
  values are never invented or altered.
* The input is never mutated; every function returns a deep copy.
* No HTTP server, no model loading at import time. The privacy pipeline is
  resolved lazily so importing this module stays cheap and side-effect free.
* The narrative is parsed with the Python standard-library HTML parser, so no
  third-party XML/HTML dependency is introduced.
"""

from __future__ import annotations

import copy
import html
from html.parser import HTMLParser
from typing import Any, Callable, Optional

from ..clinical.exporters.fhir import OperationOutcomeIssue, to_operation_outcome

__all__ = [
    "de_identify_resource",
    "de_identify_bundle",
    "de_identify",
]

# Default operation parameters, matching the ``$de-identify`` contract.
_DEFAULT_POLICY = "hipaa_safe_harbor"
_DEFAULT_METHOD = "replace"

# Keys whose entire subtree is structural/coded and must never be walked for
# free text. ``coding`` holds coded values, ``identifier`` holds identifiers
# (never invented or altered), and ``meta`` holds server/profile metadata.
_SKIP_CONTAINERS = frozenset({"coding", "identifier", "meta"})

# Primitive keys whose string value is a code, system, reference, or temporal
# value rather than human-readable free text. These are left untouched.
_SKIP_KEYS = frozenset(
    {
        # Structural / identity
        "resourceType",
        "id",
        "fullUrl",
        "reference",
        "type",
        "system",
        "code",
        "version",
        "url",
        "uri",
        "profile",
        # Coded primitives
        "status",
        "use",
        "gender",
        "unit",
        "comparator",
        "contentType",
        "language",
        # ``value[x]`` non-string primitives that surface as JSON strings
        "valueCode",
        "valueUri",
        "valueUrl",
        "valueCanonical",
        "valueId",
        "valueOid",
        "valueUuid",
        "valueDateTime",
        "valueDate",
        "valueInstant",
        "valueTime",
        "valueBase64Binary",
        # Temporal
        "date",
        "dateTime",
        "instant",
        "birthDate",
        "deceasedDateTime",
        "start",
        "end",
        "issued",
        "authoredOn",
        "recorded",
        "created",
        "effectiveDateTime",
        "effectiveInstant",
        "time",
        "when",
        "timestamp",
    }
)

# A callable that turns one free-text string into its de-identified form.
_TextDeidentifier = Callable[[str], str]

# A callable matching ``deidentify(text, *, method=..., policy=...)`` that
# returns an object exposing ``deidentified_text``.
Deidentifier = Callable[..., Any]


def de_identify_resource(
    resource: Any,
    *,
    policy: str = _DEFAULT_POLICY,
    method: str = _DEFAULT_METHOD,
    deidentifier: Optional[Deidentifier] = None,
) -> dict[str, Any]:
    """Return a de-identified copy of a single FHIR resource.

    Walks the resource's string-typed elements and its ``text`` narrative,
    de-identifying free-text content via the privacy pipeline. Coded elements,
    identifiers, references, and temporal values are left unchanged.

    Args:
        resource: A FHIR resource mapping carrying ``resourceType``.
        policy: Privacy policy profile passed to the pipeline.
        method: De-identification method (``mask``/``remove``/``replace``/...).
        deidentifier: Optional override for the privacy pipeline callable,
            mainly for testing. Defaults to :func:`openmed.core.pii.deidentify`.

    Returns:
        A de-identified deep copy of ``resource``.

    Raises:
        TypeError: If ``resource`` is not a mapping.
        ValueError: If ``resource`` lacks a ``resourceType``.
    """

    transformed, _ = _de_identify_resource(
        resource,
        policy=policy,
        method=method,
        deidentifier=deidentifier,
    )
    return transformed


def de_identify_bundle(
    bundle: Any,
    *,
    policy: str = _DEFAULT_POLICY,
    method: str = _DEFAULT_METHOD,
    deidentifier: Optional[Deidentifier] = None,
) -> dict[str, Any]:
    """Return a de-identified copy of a FHIR Bundle.

    Applies :func:`de_identify_resource` to every ``entry.resource`` while
    preserving Bundle ``type``, entry order, ``fullUrl`` values, ``request``
    blocks, and references.

    Args:
        bundle: A FHIR ``Bundle`` resource mapping.
        policy: Privacy policy profile passed to the pipeline.
        method: De-identification method.
        deidentifier: Optional override for the privacy pipeline callable.

    Returns:
        A de-identified deep copy of ``bundle``.

    Raises:
        TypeError: If ``bundle`` is not a mapping.
        ValueError: If ``bundle`` is not a ``Bundle`` resource.
    """

    transformed, _ = _de_identify_bundle(
        bundle,
        policy=policy,
        method=method,
        deidentifier=deidentifier,
    )
    return transformed


def de_identify(
    parameters: Any,
    *,
    deidentifier: Optional[Deidentifier] = None,
) -> dict[str, Any]:
    """Run ``$de-identify`` over a FHIR ``Parameters`` envelope.

    Accepts a ``Parameters`` resource carrying a ``resource`` or ``bundle``
    input part plus optional ``policy`` and ``method`` parameters, de-identifies
    the target, and returns a ``Parameters`` resource carrying the
    de-identified target, the round-tripped ``policy``/``method``, and an
    ``OperationOutcome`` listing every modified element path.

    Args:
        parameters: A FHIR ``Parameters`` resource mapping.
        deidentifier: Optional override for the privacy pipeline callable.

    Returns:
        A ``Parameters`` resource with the de-identified target and outcome.

    Raises:
        TypeError: If ``parameters`` is not a mapping.
        ValueError: If ``parameters`` is not a ``Parameters`` resource or
            carries neither a ``resource`` nor a ``bundle`` input.
    """

    if not isinstance(parameters, dict):
        raise TypeError("parameters must be a FHIR Parameters mapping")
    if parameters.get("resourceType") != "Parameters":
        raise ValueError("input resourceType must be 'Parameters'")

    policy = _read_value(parameters, "policy") or _DEFAULT_POLICY
    method = _read_value(parameters, "method") or _DEFAULT_METHOD

    bundle = _read_resource(parameters, "bundle")
    resource = _read_resource(parameters, "resource")

    if bundle is not None:
        transformed, changes = _de_identify_bundle(
            bundle,
            policy=policy,
            method=method,
            deidentifier=deidentifier,
        )
        target_name = "bundle"
    elif resource is not None:
        transformed, changes = _de_identify_resource(
            resource,
            policy=policy,
            method=method,
            deidentifier=deidentifier,
        )
        target_name = "resource"
    else:
        raise ValueError(
            "Parameters must carry a 'resource' or 'bundle' input parameter"
        )

    outcome = _outcome_from_changes(changes)
    return {
        "resourceType": "Parameters",
        "parameter": [
            {"name": target_name, "resource": transformed},
            {"name": "policy", "valueString": policy},
            {"name": "method", "valueCode": method},
            {"name": "outcome", "resource": outcome},
        ],
    }


# --- internal helpers -------------------------------------------------------


def _de_identify_resource(
    resource: Any,
    *,
    policy: str,
    method: str,
    deidentifier: Optional[Deidentifier],
) -> tuple[dict[str, Any], list[str]]:
    """De-identify a resource, returning the copy and its modified paths."""

    if not isinstance(resource, dict):
        raise TypeError("resource must be a FHIR resource mapping")
    resource_type = resource.get("resourceType")
    if not resource_type:
        raise ValueError("resource is missing 'resourceType'")

    deid = _bind_text_deidentifier(deidentifier, policy=policy, method=method)
    transformed = copy.deepcopy(resource)
    changes: list[str] = []
    _walk(transformed, str(resource_type), changes, deid)
    return transformed, changes


def _de_identify_bundle(
    bundle: Any,
    *,
    policy: str,
    method: str,
    deidentifier: Optional[Deidentifier],
) -> tuple[dict[str, Any], list[str]]:
    """De-identify every Bundle entry, preserving structure and references."""

    if not isinstance(bundle, dict):
        raise TypeError("bundle must be a FHIR Bundle mapping")
    if bundle.get("resourceType") != "Bundle":
        raise ValueError("bundle resourceType must be 'Bundle'")

    deid = _bind_text_deidentifier(deidentifier, policy=policy, method=method)
    transformed = copy.deepcopy(bundle)
    changes: list[str] = []

    entries = transformed.get("entry")
    if isinstance(entries, list):
        for index, entry in enumerate(entries):
            if not isinstance(entry, dict):
                continue
            entry_resource = entry.get("resource")
            if not isinstance(entry_resource, dict):
                continue
            resource_type = entry_resource.get("resourceType")
            if not resource_type:
                continue
            path = f"Bundle.entry[{index}].resource"
            _walk(entry_resource, path, changes, deid)

    return transformed, changes


def _bind_text_deidentifier(
    deidentifier: Optional[Deidentifier],
    *,
    policy: str,
    method: str,
) -> _TextDeidentifier:
    """Bind the pipeline callable + policy/method into a ``str -> str`` func."""

    if deidentifier is None:
        from ..core.pii import deidentify as deidentifier  # lazy, avoids cycles

    def deid(text: str) -> str:
        result = deidentifier(text, method=method, policy=policy)
        return result.deidentified_text

    return deid


def _walk(node: Any, path: str, changes: list[str], deid: _TextDeidentifier) -> None:
    """Recursively de-identify free-text strings in ``node`` (mutates in place).

    ``path`` is the FHIRPath-style location of ``node``; modified leaf paths are
    appended to ``changes``.
    """

    if isinstance(node, dict):
        skip_text = isinstance(node.get("coding"), list)
        for key, value in list(node.items()):
            if key in _SKIP_CONTAINERS:
                continue
            child_path = f"{path}.{key}"
            if key == "text" and isinstance(value, dict) and "div" in value:
                new_div, changed = _deidentify_narrative(value["div"], deid)
                if changed:
                    value["div"] = new_div
                    changes.append(f"{child_path}.div")
                continue
            if isinstance(value, str):
                if key in _SKIP_KEYS or (key == "text" and skip_text):
                    continue
                new_value = _deidentify_string(value, deid)
                if new_value is not None and new_value != value:
                    node[key] = new_value
                    changes.append(child_path)
            elif isinstance(value, (dict, list)):
                _walk(value, child_path, changes, deid)
    elif isinstance(node, list):
        for index, item in enumerate(node):
            child_path = f"{path}[{index}]"
            if isinstance(item, str):
                new_value = _deidentify_string(item, deid)
                if new_value is not None and new_value != item:
                    node[index] = new_value
                    changes.append(child_path)
            elif isinstance(item, (dict, list)):
                _walk(item, child_path, changes, deid)


def _deidentify_string(value: str, deid: _TextDeidentifier) -> Optional[str]:
    """De-identify a single string; ``None`` means 'left unchanged'."""

    if not value.strip():
        return None
    return deid(value)


def _deidentify_narrative(div: Any, deid: _TextDeidentifier) -> tuple[Any, bool]:
    """De-identify the visible text of an XHTML ``div``, preserving markup.

    Returns ``(new_div, changed)``. On any parse failure, falls back to
    de-identifying the whole ``div`` string so PHI is never left behind.
    """

    if not isinstance(div, str) or not div.strip():
        return div, False

    try:
        redactor = _NarrativeRedactor(deid)
        redactor.feed(div)
        redactor.close()
    except Exception:
        new_div = deid(div)
        return new_div, new_div != div

    if not redactor.changed:
        return div, False
    return redactor.result(), True


class _NarrativeRedactor(HTMLParser):
    """Rebuild XHTML, de-identifying only the text nodes between tags."""

    def __init__(self, deid: _TextDeidentifier) -> None:
        # ``convert_charrefs=False`` so entity/char refs are preserved verbatim
        # instead of being merged into (and re-escaped within) text data.
        super().__init__(convert_charrefs=False)
        self._deid = deid
        self._parts: list[str] = []
        self.changed = False

    def result(self) -> str:
        return "".join(self._parts)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, Optional[str]]]) -> None:
        self._parts.append(self._format_starttag(tag, attrs, self_closing=False))

    def handle_startendtag(
        self, tag: str, attrs: list[tuple[str, Optional[str]]]
    ) -> None:
        self._parts.append(self._format_starttag(tag, attrs, self_closing=True))

    def handle_endtag(self, tag: str) -> None:
        self._parts.append(f"</{tag}>")

    def handle_data(self, data: str) -> None:
        if data.strip():
            redacted = self._deid(data)
            if redacted != data:
                self.changed = True
            self._parts.append(html.escape(redacted, quote=False))
        else:
            self._parts.append(data)

    def handle_entityref(self, name: str) -> None:
        self._parts.append(f"&{name};")

    def handle_charref(self, name: str) -> None:
        self._parts.append(f"&#{name};")

    def handle_comment(self, data: str) -> None:
        self._parts.append(f"<!--{data}-->")

    def handle_decl(self, decl: str) -> None:
        self._parts.append(f"<!{decl}>")

    @staticmethod
    def _format_starttag(
        tag: str,
        attrs: list[tuple[str, Optional[str]]],
        *,
        self_closing: bool,
    ) -> str:
        pieces = [tag]
        for name, value in attrs:
            if value is None:
                pieces.append(name)
            else:
                pieces.append(f'{name}="{html.escape(value, quote=True)}"')
        inner = " ".join(pieces)
        return f"<{inner}/>" if self_closing else f"<{inner}>"


def _outcome_from_changes(changes: list[str]) -> dict[str, Any]:
    """Build an ``OperationOutcome`` summarising the modified element paths."""

    issues = [
        OperationOutcomeIssue(
            severity="information",
            code="informational",
            diagnostics="De-identified element.",
            expression=path,
        )
        for path in changes
    ]
    return to_operation_outcome(issues)


def _read_resource(parameters: dict[str, Any], name: str) -> Optional[dict[str, Any]]:
    """Return the ``resource`` of the named ``Parameters.parameter`` entry."""

    param = _find_parameter(parameters, name)
    if param is None:
        return None
    resource = param.get("resource")
    return resource if isinstance(resource, dict) else None


def _read_value(parameters: dict[str, Any], name: str) -> Optional[str]:
    """Return the ``valueString``/``valueCode`` of the named parameter."""

    param = _find_parameter(parameters, name)
    if param is None:
        return None
    for key in ("valueString", "valueCode", "valueUri"):
        value = param.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _find_parameter(parameters: dict[str, Any], name: str) -> Optional[dict[str, Any]]:
    """Find the first ``Parameters.parameter`` entry with the given name."""

    for param in parameters.get("parameter") or []:
        if isinstance(param, dict) and param.get("name") == name:
            return param
    return None
