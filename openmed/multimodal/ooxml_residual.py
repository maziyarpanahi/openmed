"""Fail-closed OOXML package verification using only the standard library.

Coverage is an internal set of XML child-index addresses taken from the actual
writer's runs/cells, never a blanket exemption for a document or worksheet.
Unknown parts and opaque assets are refused rather than qualified by a detector.
"""

from __future__ import annotations

import hashlib
import io
import os
import posixpath
import re
import tempfile
import zipfile
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO
from xml.etree import ElementTree as ET

Coverage = Mapping[str, set[tuple[int, ...]]]
_W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
_A = "http://schemas.openxmlformats.org/drawingml/2006/main"
_S = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
_R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
_TECHNICAL = re.compile(
    r"(?:\[Content_Types\]\.xml|(?:.*/)?_rels/[^/]*\.rels|"
    r"word/(?:document|styles|stylesWithEffects|settings|fontTable|numbering|"
    r"webSettings|header\d+|footer\d+)\.xml|"
    r"(?:word|xl|ppt)/theme/theme\d+\.xml|"
    r"xl/(?:workbook|styles|sharedStrings|calcChain|worksheets/sheet\d+)\.xml|"
    r"ppt/(?:presentation|presProps|viewProps|tableStyles|slides/slide\d+|"
    r"notesSlides/notesSlide\d+|slideMasters/slideMaster\d+|"
    r"slideLayouts/slideLayout\d+|notesMasters/notesMaster\d+)\.xml)"
)
_PROPERTIES = {
    "docProps/core.xml": "{http://schemas.openxmlformats.org/package/2006/metadata/core-properties}coreProperties",
    "docProps/app.xml": "{http://schemas.openxmlformats.org/officeDocument/2006/extended-properties}Properties",
    "docProps/custom.xml": "{http://schemas.openxmlformats.org/officeDocument/2006/custom-properties}Properties",
}


@dataclass(frozen=True)
class OoxmlResidualFinding:
    """Controlled category, occurrence count, and content digest only."""

    category: str
    count: int
    digest: str

    def to_dict(self) -> dict[str, str | int]:
        """Return value-free evidence suitable for diagnostics."""
        return {"category": self.category, "count": self.count, "digest": self.digest}


class OoxmlResidualError(ValueError):
    """Typed refusal with no source text, member names, or filesystem paths."""

    def __init__(self, findings: tuple[OoxmlResidualFinding, ...]) -> None:
        self.findings = findings
        super().__init__(
            "OOXML residual verification refused: "
            + ", ".join(f"{item.category}={item.count}" for item in findings)
        )


def _finding(category: str, data: bytes, count: int = 1) -> OoxmlResidualFinding:
    return OoxmlResidualFinding(category, count, hashlib.sha256(data).hexdigest())


def _parts(data: bytes) -> dict[str, bytes]:
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            if (
                not infos
                or len(infos) > 4096
                or sum(info.file_size for info in infos) > 128 * 1024 * 1024
                or len(names) != len(set(names))
                or archive.comment
                or any(info.comment or info.extra for info in infos)
                or any(
                    name.startswith("/") or ".." in name.split("/") for name in names
                )
            ):
                raise ValueError
            return {info.filename: archive.read(info) for info in infos}
    except (OSError, ValueError, RuntimeError, zipfile.BadZipFile):
        raise OoxmlResidualError((_finding("invalid_package", data),)) from None


def _xml(data: bytes) -> ET.Element:
    try:
        # Refuse declarations even in UTF-16; never expand caller-defined entities.
        declarations = data.replace(b"\x00", b"").upper()
        if b"<!DOCTYPE" in declarations or b"<!ENTITY" in declarations:
            raise ValueError
        if b"<!--" in declarations or re.search(rb"<\?(?!XML(?:\s|\?))", declarations):
            raise OoxmlResidualError((_finding("uncovered_text", data),))
        parser = ET.XMLParser(
            target=ET.TreeBuilder(insert_comments=True, insert_pis=True)
        )
        return ET.fromstring(data, parser=parser)
    except OoxmlResidualError:
        raise
    except (ET.ParseError, ValueError):
        raise OoxmlResidualError((_finding("invalid_xml", data),)) from None


def _walk(root: ET.Element, address: tuple[int, ...] = ()):
    yield root, address
    for index, child in enumerate(root):
        yield from _walk(child, (*address, index))


def _category(name: str) -> str | None:
    lowered = name.lower()
    for fragment, category in (
        ("comments", "comments_notes"),
        ("footnotes", "comments_notes"),
        ("endnotes", "comments_notes"),
        ("notesslides/", "comments_notes"),
        ("header", "headers_footers"),
        ("footer", "headers_footers"),
        ("pivot", "pivot_caches"),
        ("externallink", "external_links"),
        ("embedding", "embedded_objects"),
        ("customxml/", "custom_xml"),
    ):
        if fragment in lowered:
            return category
    if name in _PROPERTIES:
        return {
            "docProps/core.xml": "core_properties",
            "docProps/app.xml": "app_properties",
            "docProps/custom.xml": "custom_properties",
        }[name]
    return None


def inspect_ooxml(
    data: bytes, *, covered_text: Coverage | None = None
) -> tuple[OoxmlResidualFinding, ...]:
    """Inspect every package member and return safe residual evidence.

    Args:
        data: OOXML ZIP bytes, held in memory.
        covered_text: Internal writer coverage by part and XML child address.
            Without coverage, all ordinary text is unverified as well.

    Returns:
        Findings containing only controlled categories, counts and digests.
        Malformed packages also return findings, never parser or path details.
    """
    coverage = covered_text or {}
    findings: list[OoxmlResidualFinding] = []
    try:
        parts = _parts(data)
        for name, payload in parts.items():
            category = _category(name)
            if name.endswith("/"):
                findings.append(_finding("uncovered_part", payload))
                continue
            if not (name.endswith(".xml") or name.endswith(".rels")):
                findings.append(_finding(category or "embedded_objects", payload))
                continue
            root = _xml(payload)
            if not _TECHNICAL.fullmatch(name) and name not in _PROPERTIES:
                findings.append(_finding(category or "uncovered_part", payload))
                continue
            counts: dict[str, int] = defaultdict(int)
            if name in _PROPERTIES and (
                len(root) or root.attrib or (root.text or "").strip()
            ):
                counts[category] += 1
            for element, address in _walk(root):
                tag = element.tag
                if not isinstance(tag, str):
                    counts["uncovered_text"] += 1
                    continue
                local = tag.rsplit("}", 1)[-1]
                if tag.startswith("{" + _W + "}") and (
                    local in {"ins", "del", "moveFrom", "moveTo", "delText"}
                    or local.endswith("Change")
                    or "author" in {key.rsplit("}", 1)[-1] for key in element.attrib}
                ):
                    counts["tracked_changes"] += 1
                attr_names = {key.rsplit("}", 1)[-1] for key in element.attrib}
                if tag.startswith("{" + _W + "}") and local in {
                    "fldSimple",
                    "docVar",
                    "dataBinding",
                    "bookmarkStart",
                }:
                    counts["uncovered_text"] += 1
                if "author" in attr_names and not tag.startswith("{" + _W + "}"):
                    counts[category or "uncovered_text"] += 1
                if "tooltip" in attr_names:
                    counts["uncovered_text"] += 1
                if tag == f"{{{_S}}}headerFooter" and len(element):
                    counts["headers_footers"] += 1
                if tag == f"{{{_S}}}definedName":
                    counts["defined_names"] += 1
                if (
                    tag == f"{{{_S}}}sheet"
                    and element.get("state", "visible") != "visible"
                ):
                    counts["hidden_sheets"] += 1
                if local == "Relationship":
                    target = element.get("Target", "")
                    if element.get("TargetMode") == "External" or re.match(
                        r"[A-Za-z][A-Za-z0-9+.-]*:", target
                    ):
                        counts["external_links"] += 1
                    else:
                        base = name.split("/_rels/", 1)[0] if "/_rels/" in name else ""
                        resolved = posixpath.normpath(
                            posixpath.join(base, target)
                        ).lstrip("/")
                        if resolved not in parts:
                            counts["invalid_package"] += 1
                    if set(element.attrib) - {"Id", "Type", "Target", "TargetMode"}:
                        counts["uncovered_text"] += 1
                if local in {"docPr", "cNvPr"} and any(
                    element.get(key) for key in ("descr", "title", "name")
                ):
                    counts["alt_text"] += 1
                technical_text = tag == f"{{{_A}}}tableStyleId" and re.fullmatch(
                    r"\{[0-9A-Fa-f]{8}(?:-[0-9A-Fa-f]{4}){3}-[0-9A-Fa-f]{12}\}",
                    element.text or "",
                )
                if (
                    (element.text or "").strip()
                    and not technical_text
                    and address not in coverage.get(name, set())
                ):
                    counts[category or "uncovered_text"] += 1
                if (element.tail or "").strip():
                    counts[category or "uncovered_text"] += 1
            findings.extend(
                _finding(key, payload, count) for key, count in sorted(counts.items())
            )
    except OoxmlResidualError as exc:
        findings.extend(exc.findings)
    return tuple(findings)


def _serialize(root: ET.Element, original: bytes) -> bytes:
    namespaces: dict[str, str] = {}
    for _, (prefix, uri) in ET.iterparse(io.BytesIO(original), events=("start-ns",)):
        namespaces[prefix] = uri
    used = {
        value[1:].split("}", 1)[0]
        for element in root.iter()
        for value in (element.tag, *element.attrib)
        if isinstance(value, str) and value.startswith("{")
    }
    for prefix, uri in namespaces.items():
        if re.fullmatch(r"ns\d+", prefix):
            continue
        ET.register_namespace(prefix, uri)
        # Preserve declarations referenced in markup-compatibility attribute values.
        if prefix and uri not in used:
            root.set("xmlns:" + prefix, uri)
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def _sanitize(data: bytes) -> bytes:
    parts = _parts(data)
    removed = {
        name
        for name in parts
        if name.startswith("customXml/")
        or name.startswith("docProps/thumbnail.")
        or "/printerSettings/" in name
    }
    for name in removed:
        del parts[name]
    for name, payload in list(parts.items()):
        if not (name.endswith(".xml") or name.endswith(".rels")):
            continue
        root = _xml(payload)
        changed = False
        if name in _PROPERTIES:
            # Rebuild metadata roots too: custom tags and unused namespace
            # declarations must not carry caller-supplied identifiers forward.
            root = ET.Element(_PROPERTIES[name])
            changed = True
        else:
            for parent, _ in list(_walk(root)):
                for child in list(parent):
                    if (
                        isinstance(child.tag, str)
                        and child.tag.rsplit("}", 1)[-1] == "printerSettings"
                    ):
                        parent.remove(child)
                        changed = True
            for element, _ in _walk(root):
                if not isinstance(element.tag, str):
                    continue
                local = element.tag.rsplit("}", 1)[-1]
                if (
                    local in {"printerSettings", "pageSetup"}
                    and f"{{{_R}}}id" in element.attrib
                ):
                    del element.attrib[f"{{{_R}}}id"]
                    changed = True
                if local in {"docPr", "cNvPr"}:
                    changed |= any(
                        key in element.attrib for key in ("descr", "title", "name")
                    )
                    element.attrib.pop("descr", None)
                    element.attrib.pop("title", None)
                    if "name" in element.attrib:
                        element.set("name", "")
                # Master/layout placeholder text is never extracted by the writer.
                if (
                    name.startswith(
                        ("ppt/slideMasters/", "ppt/slideLayouts/", "ppt/notesMasters/")
                    )
                    and element.tag == f"{{{_A}}}t"
                ):
                    changed |= element.text is not None
                    element.text = None
            if name.endswith(".rels") or name == "[Content_Types].xml":
                base = name.split("/_rels/", 1)[0] if "/_rels/" in name else ""
                for child in list(root):
                    target = child.get("PartName", "").lstrip("/")
                    if name.endswith(".rels"):
                        target = posixpath.normpath(
                            posixpath.join(base, child.get("Target", ""))
                        )
                    if target in removed:
                        root.remove(child)
                        changed = True
        if changed:
            parts[name] = (
                ET.tostring(root, encoding="utf-8", xml_declaration=True)
                if name in _PROPERTIES
                else _serialize(root, payload)
            )
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, payload in parts.items():
            archive.writestr(name, payload)
    return output.getvalue()


def verify_ooxml(data: bytes, *, covered_text: Coverage | None = None) -> None:
    """Refuse residual content with controlled categories before publication."""
    findings = inspect_ooxml(data, covered_text=covered_text)
    if findings:
        raise OoxmlResidualError(findings)


def _read_source(source: str | Path | BinaryIO) -> bytes:
    if hasattr(source, "read"):
        source.seek(0)
        data = source.read()
        source.seek(0)
        return data
    return Path(source).read_bytes()


def _check_source(source: str | Path | BinaryIO, coverage: Coverage) -> bytes:
    # Check the original ZIP too: optional writers can silently drop unknown parts.
    sanitized = _sanitize(_read_source(source))
    verify_ooxml(sanitized, covered_text=coverage)
    return sanitized


def _run_coverage(
    parts: Mapping[str, Iterable[Any]],
) -> dict[str, set[tuple[int, ...]]]:
    coverage: dict[str, set[tuple[int, ...]]] = defaultdict(set)
    for name, runs in parts.items():
        for run in runs:
            for element in run:
                if element.tag not in {f"{{{_W}}}t", f"{{{_A}}}t"}:
                    continue
                address = []
                child = element
                while child.getparent() is not None:
                    parent = child.getparent()
                    address.append(parent.index(child))
                    child = parent
                coverage[name.lstrip("/")].add(tuple(reversed(address)))
    return dict(coverage)


def _xlsx_coverage(data: bytes) -> dict[str, set[tuple[int, ...]]]:
    try:
        return _xlsx_coverage_impl(data)
    except (KeyError, ValueError, IndexError) as exc:
        if isinstance(exc, OoxmlResidualError):
            raise
        raise OoxmlResidualError((_finding("invalid_package", data),)) from None


def _xlsx_coverage_impl(data: bytes) -> dict[str, set[tuple[int, ...]]]:
    parts = _parts(data)
    workbook = _xml(parts["xl/workbook.xml"])
    ids = {element.get(f"{{{_R}}}id") for element in workbook.iter(f"{{{_S}}}sheet")}
    relations = _xml(parts["xl/_rels/workbook.xml.rels"])
    sheets = {
        posixpath.normpath(posixpath.join("xl", element.get("Target", ""))).lstrip("/")
        for element in relations
        if element.get("Id") in ids
    }
    coverage: dict[str, set[tuple[int, ...]]] = defaultdict(set)
    shared_ids: set[int] = set()
    for name in sheets:
        root = _xml(parts[name])
        for element, address in _walk(root):
            if (
                element.tag != f"{{{_S}}}c"
                or len(address) != 3
                or root[address[0]].tag != f"{{{_S}}}sheetData"
                or root[address[0]][address[1]].tag != f"{{{_S}}}row"
            ):
                continue
            for child, suffix in _walk(element):
                if (len(suffix) == 1 and child.tag in {f"{{{_S}}}v", f"{{{_S}}}f"}) or (
                    child.tag == f"{{{_S}}}t"
                    and len(suffix) in {2, 3}
                    and element[suffix[0]].tag == f"{{{_S}}}is"
                    and (
                        len(suffix) == 2
                        or element[suffix[0]][suffix[1]].tag == f"{{{_S}}}r"
                    )
                ):
                    coverage[name].add((*address, *suffix))
                    if element.get("t") == "s" and child.tag == f"{{{_S}}}v":
                        shared_ids.add(int(child.text or "0"))
    if "xl/sharedStrings.xml" in parts:
        root = _xml(parts["xl/sharedStrings.xml"])
        for index, item in enumerate(root):
            if index in shared_ids:
                for element, address in _walk(item, (index,)):
                    if element.tag == f"{{{_S}}}t" and (
                        len(address) == 2
                        or (len(address) == 3 and item[address[1]].tag == f"{{{_S}}}r")
                    ):
                        coverage["xl/sharedStrings.xml"].add(address)
    return dict(coverage)


def _publish(
    data: bytes, destination: str | Path | BinaryIO, coverage: Coverage
) -> None:
    safe_data = _sanitize(data)
    verify_ooxml(safe_data, covered_text=coverage)
    if hasattr(destination, "write"):
        # A reused stream must represent exactly the verified ZIP, with no
        # unverified prefix or old trailing payload. Refuse non-seekable sinks.
        try:
            if not destination.seekable() or not destination.writable():
                raise ValueError
            position = destination.tell()
            destination.seek(position)
            if not callable(destination.truncate):
                raise ValueError
        except (AttributeError, OSError, ValueError):
            raise OoxmlResidualError(
                (_finding("invalid_destination", safe_data),)
            ) from None
        destination.seek(0)
        destination.write(safe_data)
        destination.truncate(len(safe_data))
        return
    target = Path(destination)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=target.parent, prefix=".ooxml-", delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(safe_data)
        os.replace(temporary, target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
