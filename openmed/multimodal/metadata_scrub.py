"""Metadata scrubber for image and document PHI containers.

The module stays import-light: Pillow, piexif, pikepdf, and python-docx are
resolved only when a format-specific scrub or verification path needs them.
Reports intentionally avoid raw metadata values so they can be embedded in
privacy audit artifacts without leaking PHI.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import os
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence
from xml.etree import ElementTree

from .exceptions import MissingDependencyError, UnsupportedDocumentError

_METADATA_INSTALL_HINT = 'Install with: pip install "openmed[multimodal]".'

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
_PDF_EXTENSIONS = {".pdf"}
_DOCX_EXTENSIONS = {".docx"}

_IMAGE_DEPS = (("PIL", "Pillow"), ("piexif", "piexif"))
_PDF_DEPS = (("pikepdf", "pikepdf"),)
_DOCX_DEPS = (("docx", "python-docx"),)

_TECHNICAL_ALLOWLIST = frozenset({"image.icc_profile"})

_DOCX_PARTS = {
    "docProps/core.xml": "docx.core",
    "docProps/app.xml": "docx.app",
    "docProps/custom.xml": "docx.custom",
}

_DOCX_NAMESPACES = {
    "cp": "http://schemas.openxmlformats.org/package/2006/metadata/core-properties",
    "dc": "http://purl.org/dc/elements/1.1/",
    "dcterms": "http://purl.org/dc/terms/",
    "dcmitype": "http://purl.org/dc/dcmitype/",
    "xsi": "http://www.w3.org/2001/XMLSchema-instance",
    "ep": "http://schemas.openxmlformats.org/officeDocument/2006/extended-properties",
    "vt": "http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes",
    "custom": "http://schemas.openxmlformats.org/officeDocument/2006/custom-properties",
}

for _prefix, _uri in _DOCX_NAMESPACES.items():
    ElementTree.register_namespace(_prefix, _uri)


@dataclass(frozen=True)
class MetadataFinding:
    """A scrubbed, preserved, or residual metadata key without its raw value."""

    container: str
    key: str
    status: str
    allowed: bool = False
    value_sha256: str | None = None
    value_length: int | None = None

    @property
    def identifier(self) -> str:
        """Return the normalized allowlist identifier for this key."""

        return _metadata_identifier(self.container, self.key)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable audit-safe representation."""

        payload: dict[str, Any] = {
            "container": self.container,
            "key": self.key,
            "identifier": self.identifier,
            "status": self.status,
            "allowed": self.allowed,
        }
        if self.value_sha256 is not None:
            payload["value_sha256"] = self.value_sha256
        if self.value_length is not None:
            payload["value_length"] = self.value_length
        return payload


@dataclass(frozen=True)
class ResidualMetadataReport:
    """Structured verification report for metadata that remains after scrubbing."""

    file_type: str
    checked_containers: tuple[str, ...]
    residual_metadata: tuple[MetadataFinding, ...] = ()
    preserved_metadata: tuple[MetadataFinding, ...] = ()

    @property
    def residual_count(self) -> int:
        """Number of disallowed metadata keys still present."""

        return len(self.residual_metadata)

    @property
    def clean(self) -> bool:
        """Whether no disallowed metadata keys remain."""

        return self.residual_count == 0

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable audit-safe report."""

        return {
            "type": "metadata_residual_report",
            "file_type": self.file_type,
            "checked_containers": list(self.checked_containers),
            "clean": self.clean,
            "residual_count": self.residual_count,
            "residual_metadata": [
                finding.to_dict() for finding in self.residual_metadata
            ],
            "preserved_metadata": [
                finding.to_dict() for finding in self.preserved_metadata
            ],
        }


@dataclass(frozen=True)
class MetadataScrubResult:
    """Result returned by :func:`scrub_metadata`."""

    source_path: Path
    output_path: Path
    file_type: str
    removed_metadata: tuple[MetadataFinding, ...]
    residual_report: ResidualMetadataReport

    @property
    def clean(self) -> bool:
        """Whether verification found no disallowed metadata."""

        return self.residual_report.clean

    def to_audit_report(self) -> dict[str, Any]:
        """Return an audit-safe dictionary suitable for downstream reports."""

        return {
            "type": "metadata_scrub",
            "source_path_sha256": _hash_value(str(self.source_path)),
            "output_path_sha256": _hash_value(str(self.output_path)),
            "source_suffix": self.source_path.suffix.lower(),
            "output_suffix": self.output_path.suffix.lower(),
            "file_type": self.file_type,
            "clean": self.clean,
            "removed_count": len(self.removed_metadata),
            "removed_metadata": [
                finding.to_dict() for finding in self.removed_metadata
            ],
            "residual_report": self.residual_report.to_dict(),
        }


class MetadataScrubError(ValueError):
    """Raised when verification finds residual disallowed metadata."""

    def __init__(self, report: ResidualMetadataReport) -> None:
        super().__init__(
            "Disallowed metadata remained after scrubbing: "
            f"{report.residual_count} residual key(s)."
        )
        self.report = report


@dataclass(frozen=True)
class _MetadataEntry:
    container: str
    key: str
    value: Any


def scrub_metadata(
    path: str | Path,
    *,
    output_path: str | Path | None = None,
    allowlist: Iterable[str] | None = None,
    verify: bool = True,
) -> MetadataScrubResult:
    """Remove embedded metadata from a supported image, PDF, or DOCX file.

    The default behavior scrubs in place. Pass ``output_path`` to write a
    separate scrubbed copy. ``allowlist`` accepts normalized identifiers such as
    ``"image.icc_profile"`` for benign technical metadata that should survive.

    Args:
        path: Source file path.
        output_path: Optional destination. Defaults to in-place replacement.
        allowlist: Metadata identifiers to preserve.
        verify: Re-read the output and raise if disallowed metadata remains.

    Returns:
        A structured result with removed keys and the residual verifier report.
    """

    source = Path(path)
    file_type = _detect_file_type(source)
    normalized_allowlist = _normalize_allowlist(allowlist)
    _ensure_dependencies(_dependencies_for(file_type))

    before = _read_metadata_entries(source, file_type)
    scrubbed_path = _scrub_by_type(
        source,
        file_type=file_type,
        output_path=Path(output_path) if output_path is not None else None,
        allowlist=normalized_allowlist,
    )
    residual_report = verify_metadata(scrubbed_path, allowlist=normalized_allowlist)

    removed_metadata = _removed_findings(before, residual_report, normalized_allowlist)
    if verify and not residual_report.clean:
        raise MetadataScrubError(residual_report)

    return MetadataScrubResult(
        source_path=source,
        output_path=scrubbed_path,
        file_type=file_type,
        removed_metadata=tuple(removed_metadata),
        residual_report=residual_report,
    )


def verify_metadata(
    path: str | Path,
    *,
    allowlist: Iterable[str] | None = None,
) -> ResidualMetadataReport:
    """Return residual disallowed metadata for a supported file."""

    source = Path(path)
    file_type = _detect_file_type(source)
    normalized_allowlist = _normalize_allowlist(allowlist)
    _ensure_dependencies(_dependencies_for(file_type))

    entries = _read_metadata_entries(source, file_type)
    residual: list[MetadataFinding] = []
    preserved: list[MetadataFinding] = []
    for entry in entries:
        allowed = _is_allowed(entry.container, entry.key, normalized_allowlist)
        status = "preserved" if allowed else "residual"
        finding = _finding_from_entry(entry, status=status, allowed=allowed)
        if allowed:
            preserved.append(finding)
        else:
            residual.append(finding)

    return ResidualMetadataReport(
        file_type=file_type,
        checked_containers=_containers_for(file_type),
        residual_metadata=tuple(residual),
        preserved_metadata=tuple(preserved),
    )


def assert_metadata_clean(
    path: str | Path,
    *,
    allowlist: Iterable[str] | None = None,
) -> ResidualMetadataReport:
    """Verify ``path`` and raise if disallowed metadata remains."""

    report = verify_metadata(path, allowlist=allowlist)
    if not report.clean:
        raise MetadataScrubError(report)
    return report


def _detect_file_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in _IMAGE_EXTENSIONS:
        return "image"
    if suffix in _PDF_EXTENSIONS:
        return "pdf"
    if suffix in _DOCX_EXTENSIONS:
        return "docx"
    raise UnsupportedDocumentError(
        f"No metadata scrubber registered for extension {suffix or '(none)'!r}."
    )


def _dependencies_for(file_type: str) -> Sequence[tuple[str, str]]:
    if file_type == "image":
        return _IMAGE_DEPS
    if file_type == "pdf":
        return _PDF_DEPS
    if file_type == "docx":
        return _DOCX_DEPS
    raise UnsupportedDocumentError(f"No metadata scrubber for file type {file_type!r}.")


def _containers_for(file_type: str) -> tuple[str, ...]:
    if file_type == "image":
        return ("image.exif", "image.xmp", "image.iptc", "image")
    if file_type == "pdf":
        return ("pdf.info", "pdf.xmp")
    if file_type == "docx":
        return ("docx.core", "docx.app", "docx.custom")
    return ()


def _missing_dependencies(requirements: Sequence[tuple[str, str]]) -> list[str]:
    return [
        distribution
        for module_name, distribution in requirements
        if importlib.util.find_spec(module_name) is None
    ]


def _ensure_dependencies(requirements: Sequence[tuple[str, str]]) -> None:
    missing = _missing_dependencies(requirements)
    if missing:
        raise MissingDependencyError(
            dependency=", ".join(missing),
            instruction=_METADATA_INSTALL_HINT,
        )


def _import_module(module_name: str) -> Any:
    return importlib.import_module(module_name)


def _normalize_allowlist(allowlist: Iterable[str] | None) -> frozenset[str]:
    if allowlist is None:
        return frozenset()
    return frozenset(_normalize_identifier(item) for item in allowlist)


def _normalize_identifier(identifier: str) -> str:
    normalized = identifier.strip().lower().replace(":", ".").replace("/", ".")
    normalized = normalized.replace(" ", "_")
    while ".." in normalized:
        normalized = normalized.replace("..", ".")
    return normalized.strip(".")


def _metadata_identifier(container: str, key: str) -> str:
    return _normalize_identifier(f"{container}.{key}")


def _is_allowed(container: str, key: str, allowlist: frozenset[str]) -> bool:
    identifier = _metadata_identifier(container, key)
    key_only = _normalize_identifier(key)
    container_only = _normalize_identifier(container)
    return (
        identifier in allowlist
        or key_only in allowlist
        or f"{container_only}.*" in allowlist
        or identifier in _TECHNICAL_ALLOWLIST.intersection(allowlist)
    )


def _hash_value(value: Any) -> str:
    if isinstance(value, bytes):
        payload = value
    else:
        payload = str(value).encode("utf-8", errors="replace")
    return hashlib.sha256(payload).hexdigest()


def _value_length(value: Any) -> int | None:
    if isinstance(value, bytes):
        return len(value)
    if isinstance(value, str):
        return len(value)
    return None


def _finding_from_entry(
    entry: _MetadataEntry,
    *,
    status: str,
    allowed: bool,
) -> MetadataFinding:
    return MetadataFinding(
        container=entry.container,
        key=entry.key,
        status=status,
        allowed=allowed,
        value_sha256=_hash_value(entry.value),
        value_length=_value_length(entry.value),
    )


def _removed_findings(
    before: Sequence[_MetadataEntry],
    report: ResidualMetadataReport,
    allowlist: frozenset[str],
) -> list[MetadataFinding]:
    residual_ids = {
        (finding.container, finding.key) for finding in report.residual_metadata
    }
    removed: list[MetadataFinding] = []
    for entry in before:
        if _is_allowed(entry.container, entry.key, allowlist):
            continue
        if (entry.container, entry.key) in residual_ids:
            continue
        removed.append(_finding_from_entry(entry, status="removed", allowed=False))
    return removed


def _read_metadata_entries(path: Path, file_type: str) -> list[_MetadataEntry]:
    if file_type == "image":
        return _read_image_metadata(path)
    if file_type == "pdf":
        return _read_pdf_metadata(path)
    if file_type == "docx":
        return _read_docx_metadata(path)
    raise UnsupportedDocumentError(f"No metadata reader for file type {file_type!r}.")


def _scrub_by_type(
    path: Path,
    *,
    file_type: str,
    output_path: Path | None,
    allowlist: frozenset[str],
) -> Path:
    if file_type == "image":
        return _scrub_image(path, output_path=output_path, allowlist=allowlist)
    if file_type == "pdf":
        return _scrub_pdf(path, output_path=output_path, allowlist=allowlist)
    if file_type == "docx":
        return _scrub_docx(path, output_path=output_path, allowlist=allowlist)
    raise UnsupportedDocumentError(f"No metadata scrubber for file type {file_type!r}.")


def _write_safely(
    source: Path,
    output_path: Path | None,
    writer: Callable[[Path], None],
) -> Path:
    target = output_path or source
    target.parent.mkdir(parents=True, exist_ok=True)
    if target != source:
        writer(target)
        return target

    handle = tempfile.NamedTemporaryFile(
        delete=False,
        dir=str(source.parent),
        prefix=f".{source.name}.",
        suffix=source.suffix,
    )
    temp_path = Path(handle.name)
    handle.close()
    try:
        writer(temp_path)
        os.replace(temp_path, source)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise
    return source


def _read_image_metadata(path: Path) -> list[_MetadataEntry]:
    image_module = _import_module("PIL.Image")
    piexif = _import_module("piexif")
    entries: list[_MetadataEntry] = []
    with image_module.open(path) as image:
        info = dict(image.info)

    exif_bytes = info.get("exif")
    if exif_bytes:
        entries.extend(_read_exif_entries(exif_bytes, piexif))

    for info_key, container, key in (
        ("xmp", "image.xmp", "packet"),
        ("XML:com.adobe.xmp", "image.xmp", "packet"),
        ("iptc", "image.iptc", "packet"),
        ("photoshop", "image.iptc", "photoshop"),
    ):
        value = info.get(info_key)
        if _has_value(value):
            entries.append(_MetadataEntry(container=container, key=key, value=value))

    if _has_value(info.get("icc_profile")):
        entries.append(
            _MetadataEntry(
                container="image",
                key="icc_profile",
                value=info["icc_profile"],
            )
        )

    return entries


def _read_exif_entries(exif_bytes: bytes, piexif: Any) -> list[_MetadataEntry]:
    try:
        exif_data = piexif.load(exif_bytes)
    except Exception:
        return [_MetadataEntry(container="image.exif", key="raw", value=exif_bytes)]

    entries: list[_MetadataEntry] = []
    for ifd_name in ("0th", "Exif", "GPS", "Interop", "1st"):
        tags = exif_data.get(ifd_name, {})
        if not isinstance(tags, Mapping):
            continue
        for tag_id, value in tags.items():
            if not _has_value(value):
                continue
            tag_name = _exif_tag_name(piexif, ifd_name, tag_id)
            entries.append(
                _MetadataEntry(
                    container=f"image.exif.{ifd_name.lower()}",
                    key=tag_name,
                    value=value,
                )
            )

    thumbnail = exif_data.get("thumbnail")
    if _has_value(thumbnail):
        entries.append(
            _MetadataEntry(
                container="image.exif",
                key="thumbnail",
                value=thumbnail,
            )
        )
    return entries


def _exif_tag_name(piexif: Any, ifd_name: str, tag_id: int) -> str:
    tag_info = piexif.TAGS.get(ifd_name, {}).get(tag_id)
    if tag_info is None:
        return str(tag_id)
    return str(tag_info.get("name", tag_id))


def _scrub_image(
    path: Path,
    *,
    output_path: Path | None,
    allowlist: frozenset[str],
) -> Path:
    image_module = _import_module("PIL.Image")

    def write_image(target: Path) -> None:
        with image_module.open(path) as image:
            image.load()
            image_format = image.format or _image_format_from_suffix(path)
            output = image
            if image_format == "JPEG" and image.mode not in {"L", "RGB", "CMYK"}:
                output = image.convert("RGB")

            save_kwargs: dict[str, Any] = {}
            icc_profile = image.info.get("icc_profile")
            if _is_allowed("image", "icc_profile", allowlist) and icc_profile:
                save_kwargs["icc_profile"] = icc_profile

            output.save(target, format=image_format, **save_kwargs)

    return _write_safely(path, output_path, write_image)


def _image_format_from_suffix(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        return "JPEG"
    if suffix == ".png":
        return "PNG"
    if suffix in {".tif", ".tiff"}:
        return "TIFF"
    if suffix == ".webp":
        return "WEBP"
    raise UnsupportedDocumentError(
        f"No image metadata scrubber registered for extension {suffix!r}."
    )


def _read_pdf_metadata(path: Path) -> list[_MetadataEntry]:
    pikepdf = _import_module("pikepdf")
    entries: list[_MetadataEntry] = []
    with pikepdf.open(path) as pdf:
        for key, value in pdf.docinfo.items():
            if _has_value(value):
                entries.append(
                    _MetadataEntry(
                        container="pdf.info",
                        key=_clean_pdf_key(key),
                        value=value,
                    )
                )

        metadata_stream = pdf.Root.get("/Metadata")
        if metadata_stream is not None:
            raw_xmp = bytes(metadata_stream.read_bytes())
            entries.extend(_read_xml_entries(raw_xmp, container="pdf.xmp"))
    return entries


def _scrub_pdf(
    path: Path,
    *,
    output_path: Path | None,
    allowlist: frozenset[str],
) -> Path:
    pikepdf = _import_module("pikepdf")

    def write_pdf(target: Path) -> None:
        with pikepdf.open(path) as pdf:
            preserved_docinfo: dict[str, Any] = {}
            for key, value in list(pdf.docinfo.items()):
                clean_key = _clean_pdf_key(key)
                if _is_allowed("pdf.info", clean_key, allowlist):
                    preserved_docinfo[str(key)] = value
                del pdf.docinfo[key]

            for key, value in preserved_docinfo.items():
                pdf.docinfo[key] = value

            if pdf.Root.get("/Metadata") is not None and not _is_allowed(
                "pdf.xmp", "packet", allowlist
            ):
                del pdf.Root["/Metadata"]

            pdf.save(target)

    return _write_safely(path, output_path, write_pdf)


def _clean_pdf_key(key: Any) -> str:
    return str(key).lstrip("/")


def _read_xml_entries(xml_bytes: bytes, *, container: str) -> list[_MetadataEntry]:
    if not xml_bytes:
        return []
    try:
        root = ElementTree.fromstring(xml_bytes)
    except ElementTree.ParseError:
        return [_MetadataEntry(container=container, key="packet", value=xml_bytes)]

    entries: list[_MetadataEntry] = []
    for element in root.iter():
        text = (element.text or "").strip()
        if text:
            entries.append(
                _MetadataEntry(
                    container=container,
                    key=_local_name(element.tag),
                    value=text,
                )
            )
        for attr_name, attr_value in element.attrib.items():
            if attr_value:
                entries.append(
                    _MetadataEntry(
                        container=container,
                        key=f"{_local_name(element.tag)}.{_local_name(attr_name)}",
                        value=attr_value,
                    )
                )

    if not entries:
        entries.append(
            _MetadataEntry(container=container, key="packet", value=xml_bytes)
        )
    return entries


def _read_docx_metadata(path: Path) -> list[_MetadataEntry]:
    _import_module("docx")
    entries: list[_MetadataEntry] = []
    with zipfile.ZipFile(path) as archive:
        for member, container in _DOCX_PARTS.items():
            try:
                xml_bytes = archive.read(member)
            except KeyError:
                continue
            entries.extend(_read_docx_part_entries(xml_bytes, container=container))
    return entries


def _read_docx_part_entries(
    xml_bytes: bytes,
    *,
    container: str,
) -> list[_MetadataEntry]:
    try:
        root = ElementTree.fromstring(xml_bytes)
    except ElementTree.ParseError:
        return [_MetadataEntry(container=container, key="xml", value=xml_bytes)]

    if container == "docx.custom":
        return _read_docx_custom_entries(root)

    entries: list[_MetadataEntry] = []
    for element in root.iter():
        if element is root:
            continue
        text = _element_text(element)
        if text:
            entries.append(
                _MetadataEntry(
                    container=container,
                    key=_local_name(element.tag),
                    value=text,
                )
            )
    return entries


def _read_docx_custom_entries(root: ElementTree.Element) -> list[_MetadataEntry]:
    entries: list[_MetadataEntry] = []
    for prop in list(root):
        prop_name = prop.attrib.get("name", _local_name(prop.tag))
        text = _element_text(prop)
        if text:
            entries.append(
                _MetadataEntry(
                    container="docx.custom",
                    key=prop_name,
                    value=text,
                )
            )
    return entries


def _scrub_docx(
    path: Path,
    *,
    output_path: Path | None,
    allowlist: frozenset[str],
) -> Path:
    _import_module("docx")

    def write_docx(target: Path) -> None:
        with (
            zipfile.ZipFile(path) as source,
            zipfile.ZipFile(
                target,
                "w",
                compression=zipfile.ZIP_DEFLATED,
            ) as scrubbed,
        ):
            for info in source.infolist():
                data = source.read(info.filename)
                container = _DOCX_PARTS.get(info.filename)
                if container is not None:
                    data = _scrub_docx_part(
                        data, container=container, allowlist=allowlist
                    )
                scrubbed.writestr(info, data)

    return _write_safely(path, output_path, write_docx)


def _scrub_docx_part(
    xml_bytes: bytes,
    *,
    container: str,
    allowlist: frozenset[str],
) -> bytes:
    try:
        root = ElementTree.fromstring(xml_bytes)
    except ElementTree.ParseError:
        return b""

    if container == "docx.custom":
        _scrub_docx_custom(root, allowlist=allowlist)
    else:
        _scrub_docx_standard(root, container=container, allowlist=allowlist)

    return ElementTree.tostring(root, encoding="utf-8", xml_declaration=True)


def _scrub_docx_standard(
    root: ElementTree.Element,
    *,
    container: str,
    allowlist: frozenset[str],
) -> None:
    for element in list(root):
        key = _local_name(element.tag)
        if _is_allowed(container, key, allowlist):
            continue
        element.clear()


def _scrub_docx_custom(
    root: ElementTree.Element,
    *,
    allowlist: frozenset[str],
) -> None:
    for prop in list(root):
        prop_name = prop.attrib.get("name", _local_name(prop.tag))
        if _is_allowed("docx.custom", prop_name, allowlist):
            continue
        root.remove(prop)


def _element_text(element: ElementTree.Element) -> str:
    return " ".join(text.strip() for text in element.itertext() if text.strip())


def _local_name(tag: Any) -> str:
    value = str(tag)
    if "}" in value:
        return value.rsplit("}", 1)[1]
    return value


def _has_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, (bytes, str)):
        return value
    if isinstance(value, Mapping):
        return bool(value)
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        try:
            iterator = iter(value)
        except TypeError:
            return True
        return any(_has_value(item) for item in iterator)
    return True
