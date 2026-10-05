"""Offline FHIR input screening before untrusted resources reach agent context.

This is a lexical guard, not FHIR schema validation or clinical authorization.
Only declared coded paths are exempt; unknown fields are scanned. Attachments
are never fetched. The copied value is for dispatch; evidence is content-free.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import html
import re
from collections.abc import Mapping
from typing import Any

from openmed.agent.security.injection_guard import (
    GuardedInput,
    InjectionFinding,
    InjectionGuard,
)
from openmed.interop.fhir_server import extract_narrative_text

MAX_FHIR_ATTACHMENT_BYTES = 65_536
MAX_FHIR_TEXT_CHARS = 131_072
MAX_FHIR_INPUT_NODES = 10_000
MAX_FHIR_INPUT_DEPTH = 64
_MAX_ENCODED_CHARS = 4 * ((MAX_FHIR_ATTACHMENT_BYTES + 2) // 3)
_MARKER = "[OPENMED_QUARANTINED_FHIR_INPUT]"
_TEXT_TYPES = {"text/plain", "text/html", "text/markdown", "application/xhtml+xml"}

# Paths are relative to a declared resource root, not exemptions for key names
# at arbitrary depths. Concept.text and unexpected children remain untrusted.
_CONCEPT_PATHS = {
    "Observation": {"code", "category[]", "interpretation[]", "component[].code"},
    "Condition": {"code", "category[]", "clinicalStatus", "verificationStatus"},
    "DiagnosticReport": {"code", "category[]", "conclusionCode[]"},
    "DocumentReference": {"type", "category[]"},
    "Procedure": {"code", "category[]"},
    "MedicationRequest": {"medicationCodeableConcept"},
    "AllergyIntolerance": {"code", "clinicalStatus", "verificationStatus"},
}
_CODE_PATHS = {
    "Patient": {"gender", "contact[].gender"},
    "Observation": {"status"},
    "DiagnosticReport": {"status"},
    "DocumentReference": {"status", "docStatus"},
    "Procedure": {"status"},
    "MedicationRequest": {"status", "intent"},
    "AllergyIntolerance": {"type", "category[]", "criticality"},
    "Bundle": {"type", "entry[].request.method"},
    "Subscription": {"status", "channel.type"},
    "SubscriptionStatus": {"status", "type"},
    "Binary": set(),
    "Condition": set(),
}
_RESOURCE_TYPES = set(_CONCEPT_PATHS) | set(_CODE_PATHS)
# Unknown property names become opaque references so an attacker-controlled JSON
# key cannot place source payload or PHI in an exception or audit finding.
_PATH_ELEMENTS = set(
    """
resourceType text div status code coding system version display userSelected
category interpretation component clinicalStatus verificationStatus conclusionCode
content format attachment type docStatus presentedForm data contentType url title
size hash creation language note extension valueString valueAttachment valueCoding
valueCodeableConcept valueCode valueQuantity valueIdentifier identifier use value
name given family gender contact medicationCodeableConcept intent criticality
entry resource contained request method channel notificationEvent focus part
parameter response payload patient subject encounter description conclusion
meta id fullUrl reference profile timestamp issued effectiveDateTime period start
end birthDate address line city state postalCode country telecom questionnaire
answer item action result
""".split()
)
_CODING_LEAVES = {"system", "version", "code", "display", "userSelected"}
_CODING_PATHS = {"DocumentReference": {"content[].format"}}


def _safe_element(key: str) -> str:
    if key in _PATH_ELEMENTS:
        return key
    return (
        "element_"
        + hashlib.sha256(key.encode("utf-8", errors="replace")).hexdigest()[:16]
    )


def _coded_path(resource_type: str, path: str) -> bool:
    normalized = re.sub(r"\[\d+\]", "[]", path)
    if normalized in _CODE_PATHS.get(resource_type, set()):
        return True
    for concept in _CONCEPT_PATHS.get(resource_type, set()):
        if normalized in {f"{concept}.coding[].{leaf}" for leaf in _CODING_LEAVES}:
            return True
    for coding in _CODING_PATHS.get(resource_type, set()):
        if normalized in {f"{coding}.{leaf}" for leaf in _CODING_LEAVES}:
            return True
    return False


def screen_fhir_input(
    value: Mapping[str, Any], *, guard: InjectionGuard | None = None
) -> GuardedInput:
    """Return a screened copy and path/offset findings without strict-mode raises.

    Narrative offsets address the original XHTML string. Attachment offsets
    address its decoded UTF-8 text in characters; quarantine offsets cover the
    encoded value. Unsupported, remote, oversized and malformed attachments
    are replaced with empty objects before any agent dispatch, in either mode.
    Nested resources in Bundles and Subscription notifications are screened.
    """
    if not isinstance(value, Mapping):
        raise TypeError("FHIR input must be an object")
    selected_guard = guard if guard is not None else InjectionGuard()
    findings: list[InjectionFinding] = []
    budget = [MAX_FHIR_INPUT_NODES]

    def quarantine(path: str, code: str, length: int = 1) -> None:
        findings.append(InjectionFinding(code, 0, max(1, length), "high", path))

    def scan_text(text: str, path: str, narrative: bool = False) -> str:
        if len(text) > MAX_FHIR_TEXT_CHARS:
            quarantine(path, "fhir_text_oversized", len(text))
            return _MARKER
        scan = selected_guard.scan(text)
        for item in scan.findings:
            findings.append(
                InjectionFinding(
                    item.pattern_id, item.start, item.end, item.severity, path
                )
            )
        if not narrative:
            return scan.quarantined_text
        try:
            visible = extract_narrative_text(text)
        except ValueError:
            quarantine(path, "fhir_narrative_invalid", len(text))
            return _MARKER
        visible_scan = selected_guard.scan(visible.text)
        for item in visible_scan.findings:
            spans = visible.source_spans[item.start : item.end]
            findings.append(
                InjectionFinding(
                    item.pattern_id,
                    min(s[0] for s in spans),
                    max(s[1] for s in spans),
                    item.severity,
                    path,
                )
            )
        # Rebuild from visible text only. No hidden bodies, attributes, external
        # references, or raw markup ever become an alternate agent input.
        safe = _MARKER if scan.flagged else visible_scan.quarantined_text
        return (
            '<div xmlns="http://www.w3.org/1999/xhtml">'
            + html.escape(safe, quote=False)
            + "</div>"
        )

    def attachment(
        node: Mapping[str, Any],
        path: str,
        resource_type: str,
        relative: str,
        depth: int,
    ) -> dict[str, Any]:
        encoded = node.get("data")
        if not isinstance(encoded, str):
            quarantine(path + ".data", "attachment_unavailable")
            return {}
        if len(encoded) > _MAX_ENCODED_CHARS:
            quarantine(path + ".data", "attachment_oversized", len(encoded))
            return {}
        decoded_size = (len(encoded) // 4) * 3 - (
            len(encoded) - len(encoded.rstrip("="))
        )
        if decoded_size > MAX_FHIR_ATTACHMENT_BYTES:
            quarantine(path + ".data", "attachment_oversized", len(encoded))
            return {}
        content_type = node.get("contentType")
        if not isinstance(content_type, str):
            quarantine(path + ".data", "attachment_non_text", len(encoded))
            return {}
        media_type, *parameters = content_type.lower().split(";")
        if media_type.strip() not in _TEXT_TYPES:
            quarantine(path + ".data", "attachment_non_text", len(encoded))
            return {}
        if any(
            parameter.strip()
            not in {"charset=utf-8", 'charset="utf-8"', "charset=us-ascii"}
            for parameter in parameters
        ):
            quarantine(path + ".data", "attachment_undecodable", len(encoded))
            return {}
        try:
            decoded_bytes = base64.b64decode(encoded, validate=True)
            if len(decoded_bytes) > MAX_FHIR_ATTACHMENT_BYTES:
                quarantine(path + ".data", "attachment_oversized", len(encoded))
                return {}
            decoded = decoded_bytes.decode("utf-8")
        except (ValueError, UnicodeDecodeError, binascii.Error):
            quarantine(path + ".data", "attachment_undecodable", len(encoded))
            return {}
        safe = scan_text(
            decoded,
            path + ".data",
            narrative=media_type.strip() in {"text/html", "application/xhtml+xml"},
        )
        # Remote references and stale size/hash metadata are never forwarded.
        result = {}
        for key, child in node.items():
            if not isinstance(key, str):
                quarantine(path, "fhir_input_invalid")
                continue
            if key in {"data", "url", "hash", "size"}:
                continue
            if budget[0] < 0:
                quarantine(path, "fhir_input_limit")
                break
            result[key] = walk(
                child,
                path + "." + _safe_element(key),
                resource_type,
                relative + "." + key,
                depth + 1,
            )
        result["data"] = base64.b64encode(safe.encode("utf-8")).decode("ascii")
        return result

    def walk(
        node: Any, path: str, resource_type: str, relative: str, depth: int
    ) -> Any:
        budget[0] -= 1
        if depth > MAX_FHIR_INPUT_DEPTH or budget[0] < 0:
            quarantine(path, "fhir_input_limit")
            return _MARKER
        if isinstance(node, str):
            if _coded_path(resource_type, relative):
                return node
            return scan_text(
                node,
                path,
                narrative=relative.endswith(".text.div") or relative == "text.div",
            )
        if isinstance(node, Mapping):
            declared_type = node.get("resourceType")
            if declared_type is not None:
                resource_type = (
                    declared_type
                    if isinstance(declared_type, str)
                    and declared_type in _RESOURCE_TYPES
                    else "Resource"
                )
                relative = ""
            if (
                "data" in node
                or "contentType" in node
                or relative.endswith("attachment")
                or relative.endswith("valueAttachment")
                or re.fullmatch(r"presentedForm\[\d+\]", relative)
            ):
                return attachment(node, path, resource_type, relative, depth)
            result = {}
            for key, child in node.items():
                if not isinstance(key, str):
                    quarantine(path, "fhir_input_invalid")
                    continue
                if budget[0] < 0:
                    quarantine(path, "fhir_input_limit")
                    break
                result[key] = walk(
                    child,
                    path + "." + _safe_element(key),
                    resource_type,
                    (relative + "." if relative else "") + key,
                    depth + 1,
                )
            return result
        if isinstance(node, (list, tuple)):
            result_list = []
            for index, child in enumerate(node):
                if budget[0] < 0:
                    quarantine(path, "fhir_input_limit")
                    break
                result_list.append(
                    walk(
                        child,
                        f"{path}[{index}]",
                        resource_type,
                        f"{relative}[{index}]",
                        depth + 1,
                    )
                )
            return result_list
        if node is None or type(node) in {bool, int, float}:
            return node
        quarantine(path, "fhir_input_invalid")
        return _MARKER

    declared = value.get("resourceType")
    root = (
        declared
        if isinstance(declared, str) and declared in _RESOURCE_TYPES
        else "Resource"
    )
    safe_value = walk(value, root, root, "", 0)
    return GuardedInput(
        safe_value,
        tuple(
            sorted(
                set(findings),
                key=lambda item: (
                    item.element_path or "",
                    item.start,
                    item.end,
                    item.pattern_id,
                ),
            )
        ),
    )
