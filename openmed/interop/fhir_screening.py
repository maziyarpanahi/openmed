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
from dataclasses import dataclass, field
from html.parser import HTMLParser
from typing import Any

from openmed.agent.security.injection_guard import (
    GuardedInput,
    InjectionFinding,
    InjectionGuard,
)


@dataclass(frozen=True)
class NarrativeText:
    """Visible text with one original XHTML span per decoded character."""

    text: str = field(repr=False)
    source_spans: tuple[tuple[int, int], ...] = field(repr=False)


def extract_narrative_text(div: str) -> NarrativeText:
    """Remove XHTML markup and decode entities with original character offsets.

    Script, style and template bodies are excluded. No URL is resolved and no
    source content is included in diagnostics or representations.
    """
    if not isinstance(div, str):
        raise TypeError("FHIR narrative must be text")
    parser = _NarrativeTextParser(div)
    try:
        parser.feed(div)
        parser.close()
    except Exception:
        raise ValueError("FHIR narrative parsing failed") from None
    return NarrativeText("".join(parser.parts), tuple(parser.spans))


class _NarrativeTextParser(HTMLParser):
    _BLOCK_TAGS = {
        "div",
        "p",
        "li",
        "ul",
        "ol",
        "table",
        "tr",
        "td",
        "th",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "blockquote",
        "pre",
    }

    def __init__(self, source: str) -> None:
        super().__init__(convert_charrefs=False)
        self.source = source
        self.line_starts = [0] + [
            index + 1 for index, char in enumerate(source) if char == "\n"
        ]
        self.parts: list[str] = []
        self.spans: list[tuple[int, int]] = []
        self.hidden: list[str] = []

    def _offset(self) -> int:
        line, column = self.getpos()
        return self.line_starts[line - 1] + column

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "template"}:
            self.hidden.append(tag)
        elif tag == "br" and not self.hidden:
            start = self._offset()
            self.parts.append(" ")
            self.spans.append((start, start + len(self.get_starttag_text())))
        elif tag in self._BLOCK_TAGS and not self.hidden:
            self._block_boundary(len(self.get_starttag_text()))

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "br":
            self.handle_starttag(tag, attrs)

    def handle_endtag(self, tag: str) -> None:
        if self.hidden and self.hidden[-1] == tag:
            self.hidden.pop()
        elif tag in self._BLOCK_TAGS and not self.hidden:
            start = self._offset()
            end = self.source.find(">", start)
            self._block_boundary(end - start + 1 if end >= start else 1)

    def _block_boundary(self, length: int) -> None:
        if self.parts and self.parts[-1] and not self.parts[-1][-1].isspace():
            start = self._offset()
            self.parts.append(" ")
            self.spans.append((start, start + length))

    def handle_data(self, data: str) -> None:
        if not self.hidden:
            start = self._offset()
            self.parts.append(data)
            self.spans.extend((start + i, start + i + 1) for i in range(len(data)))

    def _entity(self, token: str) -> None:
        if self.hidden:
            return
        start = self._offset()
        if self.source[start + len(token) : start + len(token) + 1] == ";":
            token += ";"
        decoded = html.unescape(token)
        self.parts.append(decoded)
        self.spans.extend((start, start + len(token)) for _ in decoded)

    def handle_entityref(self, name: str) -> None:
        self._entity("&" + name)

    def handle_charref(self, name: str) -> None:
        self._entity("&#" + name)


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


def _resource_boundary(resource_type: str, relative: str, depth: int) -> bool:
    # A declared type can reset coded paths only at a structural resource slot.
    # An unexpected object cannot impersonate a resource inside a coded element.
    return (
        depth == 0
        or re.fullmatch(r"contained\[\d+\]", relative) is not None
        or (
            resource_type == "Bundle"
            and re.fullmatch(r"entry\[\d+\]\.resource", relative) is not None
        )
        or (
            resource_type == "SubscriptionStatus"
            and re.fullmatch(r"notificationEvent\[\d+\]\.focus\.resource", relative)
            is not None
        )
    )


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
                relative + "." + _safe_element(key),
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
            if declared_type is not None and _resource_boundary(
                resource_type, relative, depth
            ):
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
                    (relative + "." if relative else "") + _safe_element(key),
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
