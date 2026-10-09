"""Offline, custody-bound NAACCR XML files from governed registry cases."""

from __future__ import annotations

import hashlib
import hmac
import math
import os
import re
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass, field, replace
from datetime import date
from types import MappingProxyType
from typing import Any

from openmed.clinical.journey_contracts import (
    ClinicalFact,
    canonical_digest,
    canonical_json,
)

from .contracts import (
    RegistryCase,
    RegistryCaseState,
    RegistryDefinitionVersion,
    RegistryExportAuthorization,
    RegistryExportEnvelope,
    RegistryFieldState,
)
from .workflow import build_registry_export

__all__ = [
    "NAACCR_NAMESPACE",
    "NAACCRProjectionError",
    "NAACCRDictionaryError",
    "NAACCRValueError",
    "NAACCRItemDefinition",
    "NAACCRDictionary",
    "NAACCRFieldMapping",
    "NAACCRLoss",
    "NAACCRExportReport",
    "parse_naaccr_dictionary",
    "write_naaccr_xml",
]

NAACCR_NAMESPACE = "http://naaccr.org/naaccrxml"
_ID = re.compile(r"[A-Za-z][A-Za-z0-9]{0,31}\Z")
_FIELD = re.compile(r"[a-z][a-z0-9_.:/-]{0,127}\Z")
_TYPES = frozenset(
    {"digits", "alpha", "alphanumeric", "numeric", "date", "dateTime", "text"}
)
_PARENTS = frozenset({"NaaccrData", "Patient", "Tumor"})
_ERRORS = frozenset(
    {
        "input_invalid",
        "dictionary_invalid",
        "dictionary_unsupported",
        "mapping_invalid",
        "custody_invalid",
        "authorization_refused",
        "length",
        "data_type",
        "xml_character",
        "resolver_failed",
        "input_limit",
        "patient_key_collision",
        "output_refused",
    }
)
_LOSSES = frozenset(
    {
        "case_not_export_ready",
        "case_custody_mismatch",
        "unmapped",
        "unknown",
        "conflict",
        "missing_required",
        "not_applicable",
        "unsupported",
        "fact_missing",
        "fact_custody_mismatch",
        "fact_state_refused",
        "value_conflict",
    }
)


class NAACCRProjectionError(ValueError):
    """Fixed diagnostic with controlled code and optional input indices."""

    def __init__(
        self, code: str, case_index: int | None = None, field_index: int | None = None
    ):
        self.code = code if type(code) is str and code in _ERRORS else "input_invalid"
        self.case_index = (
            case_index if type(case_index) is int and case_index >= 0 else None
        )
        self.field_index = (
            field_index if type(field_index) is int and field_index >= 0 else None
        )
        super().__init__("NAACCR projection refused.")

    def to_dict(self) -> dict[str, Any]:
        """Return a source-free diagnostic without paths, values or identifiers."""
        return {
            "code": self.code,
            "case_index": self.case_index,
            "field_index": self.field_index,
        }


class NAACCRDictionaryError(NAACCRProjectionError):
    """Invalid or unsupported caller-supplied dictionary."""


class NAACCRValueError(NAACCRProjectionError):
    """A resolved value violates the declared item constraint."""


def _safe_code(code: str, cls=NAACCRProjectionError, case=None, index=None):
    return cls(code, case, index)


def _uri(value: Any) -> str:
    if (
        type(value) is not str
        or len(value) > 512
        or re.fullmatch(
            r"(?:https?://[A-Za-z0-9.-]+(?:/[A-Za-z0-9._/-]+)?|urn:[A-Za-z0-9:._-]+)",
            value,
        )
        is None
    ):
        raise _safe_code("dictionary_invalid", NAACCRDictionaryError)
    return value


@dataclass(frozen=True)
class NAACCRItemDefinition:
    """One caller-declared item; no standard items or value tables are bundled."""

    naaccr_id: str
    naaccr_num: int
    length: int
    parent: str
    data_type: str = "text"
    record_types: tuple[str, ...] = ("A", "M", "C", "I")
    unlimited_text: bool = False

    def __post_init__(self):
        if (
            type(self.naaccr_id) is not str
            or _ID.fullmatch(self.naaccr_id) is None
            or type(self.naaccr_num) is not int
            or self.naaccr_num < 1
            or type(self.length) is not int
            or not 1 <= self.length <= 1_048_576
            or type(self.parent) is not str
            or self.parent not in _PARENTS
            or type(self.data_type) is not str
            or self.data_type not in _TYPES
            or type(self.record_types) is not tuple
            or not self.record_types
            or any(
                type(v) is not str or v not in ("A", "M", "C", "I")
                for v in self.record_types
            )
            or len(set(self.record_types)) != len(self.record_types)
            or type(self.unlimited_text) is not bool
            or (self.unlimited_text and self.data_type != "text")
        ):
            raise _safe_code("dictionary_invalid", NAACCRDictionaryError)
        if (self.data_type == "date" and self.length != 8) or (
            self.data_type == "dateTime" and self.length != 25
        ):
            raise _safe_code("dictionary_invalid", NAACCRDictionaryError)


@dataclass(frozen=True)
class NAACCRDictionary:
    """Immutable projection constraints parsed from a caller's local XML bytes."""

    uri: str = field(repr=False)
    specification_version: str
    items: Mapping[str, NAACCRItemDefinition] = field(repr=False)
    source_digest: str

    def __post_init__(self):
        object.__setattr__(self, "uri", _uri(self.uri))
        if type(
            self.specification_version
        ) is not str or self.specification_version not in tuple(
            f"1.{v}" for v in range(9)
        ):
            raise _safe_code("dictionary_unsupported", NAACCRDictionaryError)
        if (
            not isinstance(self.items, Mapping)
            or not 1 <= len(self.items) <= 8192
            or type(self.source_digest) is not str
            or re.fullmatch(r"sha256:[0-9a-f]{64}", self.source_digest) is None
        ):
            raise _safe_code("dictionary_invalid", NAACCRDictionaryError)
        items: dict[str, NAACCRItemDefinition] = {}
        nums = set()
        for key, item in self.items.items():
            if type(item) is not NAACCRItemDefinition:
                raise _safe_code("dictionary_invalid", NAACCRDictionaryError)
            item = replace(item)
            if key != item.naaccr_id or item.naaccr_num in nums:
                raise _safe_code("dictionary_invalid", NAACCRDictionaryError)
            if (
                item.data_type == "dateTime" and self.specification_version != "1.8"
            ) or (item.unlimited_text and self.specification_version >= "1.6"):
                raise _safe_code("dictionary_unsupported", NAACCRDictionaryError)
            items[key] = item
            nums.add(item.naaccr_num)
        object.__setattr__(self, "items", MappingProxyType(items))


def parse_naaccr_dictionary(value: bytes | str) -> NAACCRDictionary:
    """Parse local dictionary bytes without retrieving URIs or external entities.

    Args:
        value: UTF-8 XML bytes or text supplied by the caller, at most 4 MiB.

    Returns:
        Immutable item constraints and the exact input-byte digest.

    Raises:
        NAACCRDictionaryError: For malformed, oversized or unsupported XML.
    """
    try:
        if type(value) is str:
            value = value.encode("utf-8")
        if type(value) is not bytes or not value or len(value) > 4_194_304:
            raise _safe_code("dictionary_invalid", NAACCRDictionaryError)
        if b"\0" in value or re.search(rb"<!\s*(?:DOCTYPE|ENTITY)", value, re.I):
            raise _safe_code("dictionary_invalid", NAACCRDictionaryError)
        decoded = value.decode("utf-8")
        declaration = re.match(r"\ufeff?<\?xml\s+([^?]*)\?>", decoded)
        if declaration:
            encoding = re.search(r"encoding\s*=\s*(['\"])(.*?)\1", declaration[1])
            if encoding and encoding[2].lower() not in ("utf-8", "utf8"):
                raise _safe_code("dictionary_unsupported", NAACCRDictionaryError)
        root = ET.fromstring(decoded)
        if root.tag != f"{{{NAACCR_NAMESPACE}}}NaaccrDictionary":
            raise _safe_code("dictionary_invalid", NAACCRDictionaryError)
        pending = [(root, 0)]
        count = 0
        while pending:
            node, depth = pending.pop()
            count += 1
            if depth > 32 or count > 65_536:
                raise _safe_code("dictionary_invalid", NAACCRDictionaryError)
            pending.extend((child, depth + 1) for child in node)
        version = root.attrib.get("specificationVersion", "")
        containers = root.findall(f"{{{NAACCR_NAMESPACE}}}ItemDefs")
        if len(containers) != 1:
            raise _safe_code("dictionary_invalid", NAACCRDictionaryError)
        items: dict[str, NAACCRItemDefinition] = {}
        for node in containers[0]:
            if node.tag != f"{{{NAACCR_NAMESPACE}}}ItemDef" or len(items) >= 8192:
                raise _safe_code("dictionary_invalid", NAACCRDictionaryError)
            attrs = node.attrib
            if version in ("1.6", "1.7", "1.8") and "allowUnlimitedText" in attrs:
                raise _safe_code("dictionary_unsupported", NAACCRDictionaryError)
            if attrs.get("allowUnlimitedText", "false") not in ("true", "false"):
                raise _safe_code("dictionary_invalid", NAACCRDictionaryError)
            for key in ("naaccrNum", "length"):
                if re.fullmatch(r"[0-9]{1,8}", attrs.get(key, "")) is None:
                    raise _safe_code("dictionary_invalid", NAACCRDictionaryError)
            item = NAACCRItemDefinition(
                attrs.get("naaccrId", ""),
                int(attrs["naaccrNum"]),
                int(attrs["length"]),
                attrs.get("parentXmlElement", ""),
                attrs.get("dataType", "text"),
                tuple(attrs.get("recordTypes", "A,M,C,I").split(",")),
                attrs.get("allowUnlimitedText") == "true",
            )
            if item.naaccr_id in items:
                raise _safe_code("dictionary_invalid", NAACCRDictionaryError)
            items[item.naaccr_id] = item
        return NAACCRDictionary(
            root.attrib.get("dictionaryUri", ""),
            version,
            items,
            "sha256:" + hashlib.sha256(value).hexdigest(),
        )
    except Exception as exc:
        code = (
            exc.code if isinstance(exc, NAACCRDictionaryError) else "dictionary_invalid"
        )
        raise NAACCRDictionaryError(code) from None


@dataclass(frozen=True)
class NAACCRFieldMapping:
    """Bind a registry field to an item and a static path inside each Journey value."""

    field_id: str
    naaccr_id: str
    value_path: tuple[str | int, ...] = ()

    def __post_init__(self):
        if (
            type(self.field_id) is not str
            or _FIELD.fullmatch(self.field_id) is None
            or type(self.naaccr_id) is not str
            or _ID.fullmatch(self.naaccr_id) is None
            or type(self.value_path) is not tuple
            or len(self.value_path) > 8
        ):
            raise _safe_code("mapping_invalid")
        for part in self.value_path:
            if type(part) is int and 0 <= part <= 4096:
                continue
            if type(part) is str and re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", part):
                continue
            raise _safe_code("mapping_invalid")


@dataclass(frozen=True)
class NAACCRLoss:
    """Controlled refusal associated with input case/field indices only."""

    case_index: int
    field_index: int | None
    code: str

    def __post_init__(self):
        if (
            type(self.case_index) is not int
            or self.case_index < 0
            or type(self.code) is not str
            or self.code not in _LOSSES
        ):
            raise _safe_code("input_invalid")
        if self.field_index is not None and (
            type(self.field_index) is not int or self.field_index < 0
        ):
            raise _safe_code("input_invalid")

    def to_dict(self) -> dict[str, Any]:
        """Return indices and fixed loss code, without dictionary or fact values."""
        return {
            "case_index": self.case_index,
            "field_index": self.field_index,
            "code": self.code,
        }


@dataclass(frozen=True)
class NAACCRExportReport:
    """Value-free receipt; clinical values are available only in the protected file."""

    file_written: bool
    xml_digest: str | None
    bytes_written: int
    case_count: int
    patient_count: int
    tumor_count: int
    item_count: int
    losses: tuple[NAACCRLoss, ...]
    dictionary_digest: str
    export_manifest_digest: str
    mapping_digest: str

    def to_dict(self) -> dict[str, Any]:
        """Return counts, content digests and controlled losses, never XML or paths."""
        return {
            "file_written": self.file_written,
            "xml_digest": self.xml_digest,
            "bytes_written": self.bytes_written,
            "case_count": self.case_count,
            "patient_count": self.patient_count,
            "tumor_count": self.tumor_count,
            "item_count": self.item_count,
            "losses": [v.to_dict() for v in self.losses],
            "loss_counts": dict(sorted(Counter(v.code for v in self.losses).items())),
            "dictionary_digest": self.dictionary_digest,
            "export_manifest_digest": self.export_manifest_digest,
            "mapping_digest": self.mapping_digest,
            "edits_validated": False,
            "submitted": False,
        }


def _valid_xml(value: str) -> bool:
    return all(
        ch in "\t\n\r"
        or 0x20 <= ord(ch) <= 0xD7FF
        or 0xE000 <= ord(ch) <= 0xFFFD
        or 0x10000 <= ord(ch) <= 0x10FFFF
        for ch in value
    )


def _calendar(value: str, compact: bool) -> bool:
    pattern = (
        r"(?:18|19|20)[0-9]{2}(?:[0-9]{2}(?:[0-9]{2})?)?"
        if compact
        else r"[0-9]{4}(?:-[0-9]{2}(?:-[0-9]{2})?)?"
    )
    if re.fullmatch(pattern, value) is None:
        return False
    parts = [int(value[:4])]
    if compact:
        parts.extend(int(value[i : i + 2]) for i in range(4, len(value), 2))
    else:
        parts.extend(int(v) for v in value.split("-")[1:])
    try:
        date(*parts, *([1] * (3 - len(parts))))
        return True
    except ValueError:
        return False


def _date_time(value: str) -> bool:
    if "T" not in value:
        return _calendar(value, False)
    day, time = value.split("T", 1)
    return (
        len(day) == 10
        and _calendar(day, False)
        and re.fullmatch(
            r"(?:[01][0-9]|2[0-3]):[0-5][0-9]:(?:[0-5][0-9]|60)(?:Z|[+-](?:(?:0[0-9]|1[0-3]):[0-5][0-9]|14:00))",
            time,
        )
        is not None
    )


def _value(
    value: Any,
    item: NAACCRItemDefinition,
    case: int | None = None,
    index: int | None = None,
) -> str:
    if type(value) is str:
        result = value
    elif type(value) is int:
        if not -(10**64) < value < 10**64:
            raise _safe_code("data_type", NAACCRValueError, case, index)
        result = str(value)
    elif item.data_type == "numeric" and type(value) is float and math.isfinite(value):
        result = str(value)
    else:
        raise _safe_code("data_type", NAACCRValueError, case, index)
    if not _valid_xml(result):
        raise _safe_code("xml_character", NAACCRValueError, case, index)
    if not result.strip() or len(result) > (
        1_048_576 if item.unlimited_text else item.length
    ):
        raise _safe_code("length", NAACCRValueError, case, index)
    checks = {
        "digits": lambda: (
            re.fullmatch(r"[0-9]+", result) is not None and len(result) == item.length
        ),
        "alpha": lambda: (
            re.fullmatch(r"[A-Z]+", result) is not None and len(result) == item.length
        ),
        "alphanumeric": lambda: (
            re.fullmatch(r"[A-Z0-9]+", result) is not None
            and len(result) == item.length
        ),
        "numeric": lambda: re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", result) is not None,
        "date": lambda: _calendar(result, True),
        "dateTime": lambda: _date_time(result),
        "text": lambda: True,
    }
    if not checks[item.data_type]():
        raise _safe_code("data_type", NAACCRValueError, case, index)
    return result


def _patient_key(subject: str, secret: bytes, item: NAACCRItemDefinition) -> str:
    alphabet = {
        "digits": "0123456789",
        "numeric": "0123456789",
        "alpha": "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
        "alphanumeric": "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
        "text": "0123456789abcdef",
    }.get(item.data_type)
    if alphabet is None or item.unlimited_text:
        raise _safe_code("mapping_invalid")
    width = (
        item.length
        if item.data_type in ("digits", "alpha", "alphanumeric")
        else min(item.length, 64)
    )
    if width > 128:
        raise _safe_code("mapping_invalid")
    digest = hmac.new(
        secret, b"openmed.naaccr.patient\0" + subject.encode(), hashlib.sha256
    ).digest()
    number = int.from_bytes(digest, "big")
    chars = []
    for _ in range(width):
        number, remainder = divmod(number, len(alphabet))
        chars.append(alphabet[remainder])
    return _value("".join(reversed(chars)), item)


def _selected(value: Any, path: tuple[str | int, ...]) -> Any:
    for part in path:
        if type(part) is str and isinstance(value, Mapping) and part in value:
            value = value[part]
        elif (
            type(part) is int and isinstance(value, (list, tuple)) and part < len(value)
        ):
            value = value[part]
        else:
            raise _safe_code("data_type", NAACCRValueError)
    return value


def _write(path: Any, payload: bytes) -> None:
    fd = None
    owned = None
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        owned = os.fstat(fd)
        with os.fdopen(fd, "wb") as stream:
            if stream.write(payload) != len(payload):
                raise OSError()
            stream.flush()
            os.fsync(stream.fileno())
        fd = None
    except Exception:
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass
        if owned is not None:
            try:
                current = os.lstat(path)
                if (current.st_dev, current.st_ino) == (owned.st_dev, owned.st_ino):
                    os.unlink(path)
            except OSError:
                pass
        raise _safe_code("output_refused") from None


def write_naaccr_xml(
    cases: Sequence[RegistryCase],
    definition_version: RegistryDefinitionVersion,
    *,
    authorization: RegistryExportAuthorization,
    export_envelope: RegistryExportEnvelope,
    dictionary: NAACCRDictionary,
    field_map: Sequence[NAACCRFieldMapping],
    resolver: Callable[[RegistryCase, str, str], ClinicalFact | None],
    output_path: str | os.PathLike[str],
    patient_key_item: str,
    patient_key_secret: bytes,
    record_type: str = "I",
) -> NAACCRExportReport:
    """Write one protected XML file after validating registry and Journey custody.

    Args:
        cases: Existing registry cases, at most 128, from the bound export envelope.
        definition_version: Existing immutable registry definition and workflow.
        authorization: Existing explicit local export authorization.
        export_envelope: Existing privacy-safe manifest for the exact case versions.
        dictionary: Caller-supplied constraints from parse_naaccr_dictionary.
        field_map: Static field/item bindings with optional Journey value paths.
        resolver: Trusted local lookup accepting case, field id and opaque fact id.
        output_path: New caller-chosen file; no overwrite, stdout XML or submission.
        patient_key_item: A declared Patient item reserved for pseudonymous keys.
        patient_key_secret: Caller-owned HMAC key, at least 32 bytes; never stored.
        record_type: Explicit NAACCR record type A, M, C or I.

    Returns:
        Counts, digests and controlled losses. No file is created for a refused batch.

    Raises:
        NAACCRProjectionError: For invalid configuration, custody or file operations.
        NAACCRValueError: When a resolved value violates type, length or XML rules.
    """
    try:
        return _project(
            cases,
            definition_version,
            authorization,
            export_envelope,
            dictionary,
            field_map,
            resolver,
            output_path,
            patient_key_item,
            patient_key_secret,
            record_type,
        )
    except Exception as exc:
        if isinstance(exc, NAACCRProjectionError):
            cls = (
                NAACCRValueError
                if isinstance(exc, NAACCRValueError)
                else NAACCRProjectionError
            )
            raise cls(exc.code, exc.case_index, exc.field_index) from None
        raise _safe_code("input_invalid") from None


def _project(
    cases,
    definition,
    authorization,
    envelope,
    dictionary,
    mappings,
    resolver,
    path,
    key_item,
    secret,
    record_type,
):
    if (
        isinstance(cases, (str, bytes))
        or not 1 <= len(cases) <= 128
        or any(type(c) is not RegistryCase for c in cases)
        or type(definition) is not RegistryDefinitionVersion
        or type(authorization) is not RegistryExportAuthorization
        or type(envelope) is not RegistryExportEnvelope
        or type(dictionary) is not NAACCRDictionary
        or not callable(resolver)
        or type(secret) is not bytes
        or not 32 <= len(secret) <= 4096
        or record_type not in ("A", "M", "C", "I")
    ):
        raise _safe_code("input_invalid")
    dictionary = replace(dictionary)
    definition = RegistryDefinitionVersion.from_dict(definition.to_dict())
    authorization = replace(authorization)
    envelope = RegistryExportEnvelope.from_dict(envelope.to_dict())
    if len(canonical_json([c.to_dict() for c in cases]).encode()) > 4_194_304:
        raise _safe_code("input_limit")
    cases = tuple(RegistryCase.from_dict(c.to_dict()) for c in cases)
    if any(len(c.fields) > 256 for c in cases) or len(
        {c.case_id for c in cases}
    ) != len(cases):
        raise _safe_code("input_limit")
    rules = {r.field_id: r for r in definition.definition.fields}
    workflow = definition.definition.workflow
    if (
        not authorization.export_approved
        or authorization.definition_version_id != definition.version_id
        or authorization.privacy_policy_digest != workflow.privacy_policy_digest
        or authorization.export_policy_digest != workflow.export_policy_digest
    ):
        raise _safe_code("authorization_refused")
    if (
        set(envelope.case_digests) != {c.case_id for c in cases}
        or envelope.authorization_digest != authorization.digest
        or envelope.definition_digest != definition.definition_digest
        or envelope.definition_version_id != definition.version_id
        or envelope.privacy_policy_digest != workflow.privacy_policy_digest
        or envelope.export_policy_digest != workflow.export_policy_digest
    ):
        raise _safe_code("custody_invalid")
    if (
        isinstance(mappings, (str, bytes))
        or len(mappings) > 512
        or any(type(m) is not NAACCRFieldMapping for m in mappings)
    ):
        raise _safe_code("mapping_invalid")
    mappings = tuple(replace(m) for m in mappings)
    by_field = {m.field_id: m for m in mappings}
    if (
        len(by_field) != len(mappings)
        or len({m.naaccr_id for m in mappings}) != len(mappings)
        or any(
            m.field_id not in rules or m.naaccr_id not in dictionary.items
            for m in mappings
        )
        or key_item not in dictionary.items
        or dictionary.items[key_item].parent != "Patient"
        or any(m.naaccr_id == key_item for m in mappings)
        or any(
            record_type not in dictionary.items[m.naaccr_id].record_types
            for m in mappings
        )
        or record_type not in dictionary.items[key_item].record_types
    ):
        raise _safe_code("mapping_invalid")
    mapping_digest = canonical_digest(
        {
            "mappings": [vars(m) for m in sorted(mappings, key=lambda m: m.field_id)],
            "patient_key_item": key_item,
            "record_type": record_type,
        }
    )
    losses = []
    for i, case in enumerate(cases):
        if case.state is not RegistryCaseState.EXPORT_READY:
            losses.append(NAACCRLoss(i, None, "case_not_export_ready"))
        elif envelope.case_digests[case.case_id] != case.case_digest:
            losses.append(NAACCRLoss(i, None, "case_custody_mismatch"))
    if losses:
        return NAACCRExportReport(
            False,
            None,
            0,
            0,
            0,
            0,
            0,
            tuple(losses),
            dictionary.source_digest,
            envelope.manifest_digest,
            mapping_digest,
        )
    expected = build_registry_export(
        definition, cases, authorization=authorization, created_at=envelope.created_at
    )
    if (
        expected.value is None
        or expected.value.manifest_digest != envelope.manifest_digest
    ):
        raise _safe_code("custody_invalid")
    keys = {
        c.subject_id: _patient_key(c.subject_id, secret, dictionary.items[key_item])
        for c in cases
    }
    if len(set(keys.values())) != len(keys):
        raise _safe_code("patient_key_collision")
    groups = defaultdict(list)
    calls = 0
    projected_bytes = 0
    with open(os.devnull, "w") as silence:
        for ci, case in enumerate(cases):
            for fi, field_result in enumerate(case.fields):
                if field_result.state not in (
                    RegistryFieldState.PRESENT,
                    RegistryFieldState.CORRECTED,
                ):
                    losses.append(NAACCRLoss(ci, fi, field_result.state.value))
                    continue
                mapping = by_field.get(field_result.field_id)
                if mapping is None:
                    losses.append(NAACCRLoss(ci, fi, "unmapped"))
                    continue
                item = dictionary.items[mapping.naaccr_id]
                rule = rules[field_result.field_id]
                evidence = field_result.evidence
                values = []
                resolved_evidence = set()
                reason = None
                for fact_id, value_digest, derivation in zip(
                    evidence.fact_ids,
                    evidence.value_digests,
                    evidence.derivation_digests,
                ):
                    calls += 1
                    if calls > 4096:
                        raise _safe_code("input_limit")
                    try:
                        with redirect_stdout(silence), redirect_stderr(silence):
                            fact = resolver(case, field_result.field_id, fact_id)
                    except Exception:
                        raise _safe_code("resolver_failed", case=ci, index=fi) from None
                    if fact is None:
                        reason = "fact_missing"
                        break
                    if type(fact) is not ClinicalFact:
                        reason = "fact_custody_mismatch"
                        break
                    if len(canonical_json(fact.to_dict()).encode()) > 1_048_576:
                        raise _safe_code("input_limit")
                    fact = ClinicalFact.from_dict(fact.to_dict())
                    if (
                        fact.fact_id != fact_id
                        or fact.subject_id != case.subject_id
                        or fact.fact_type != rule.fact_type
                        or canonical_digest(fact.value) != value_digest
                        or fact.derivation_hash != derivation
                        or not set(fact.evidence_ids) <= set(evidence.evidence_ids)
                    ):
                        reason = "fact_custody_mismatch"
                        break
                    if fact.status not in rule.allowed_statuses or fact.status in (
                        *rule.unknown_statuses,
                        *rule.conflict_statuses,
                    ):
                        reason = "fact_state_refused"
                        break
                    resolved_evidence.update(fact.evidence_ids)
                    try:
                        selected_value = _value(
                            _selected(fact.value, mapping.value_path), item, ci, fi
                        )
                        projected_bytes += len(selected_value.encode("utf-8"))
                        if projected_bytes > 4_194_304:
                            raise _safe_code("input_limit")
                        values.append(selected_value)
                    except NAACCRValueError as exc:
                        raise NAACCRValueError(exc.code, ci, fi) from None
                if reason is None and resolved_evidence != set(evidence.evidence_ids):
                    reason = "fact_custody_mismatch"
                if reason is not None:
                    losses.append(NAACCRLoss(ci, fi, reason))
                elif not values or len(set(values)) != 1:
                    losses.append(NAACCRLoss(ci, fi, "value_conflict"))
                else:
                    scope = (
                        "root"
                        if item.parent == "NaaccrData"
                        else case.subject_id
                        if item.parent == "Patient"
                        else case.case_id
                    )
                    groups[(item.parent, scope, item.naaccr_id)].append(
                        (ci, fi, values[0])
                    )
    accepted = {}
    for binding, instances in groups.items():
        if len({v for _, _, v in instances}) != 1:
            losses.extend(
                NAACCRLoss(ci, fi, "value_conflict") for ci, fi, _ in instances
            )
        else:
            accepted[binding] = instances[0][2]
    root = ET.Element(
        "NaaccrData",
        {
            "xmlns": NAACCR_NAMESPACE,
            "baseDictionaryUri": dictionary.uri,
            "recordType": record_type,
            "specificationVersion": dictionary.specification_version,
        },
    )
    items_written = 0

    def add_items(parent, parent_name, scope):
        nonlocal items_written
        for (parent_kind, binding_scope, item_id), value in sorted(accepted.items()):
            if parent_kind != parent_name or binding_scope != scope:
                continue
            node = ET.SubElement(parent, "Item", {"naaccrId": item_id})
            node.text = value
            items_written += 1

    add_items(root, "NaaccrData", "root")
    tumors = 0
    for subject, key in sorted(keys.items(), key=lambda pair: pair[1]):
        patient = ET.SubElement(root, "Patient")
        ET.SubElement(patient, "Item", {"naaccrId": key_item}).text = key
        items_written += 1
        add_items(patient, "Patient", subject)
        for case in sorted(
            (c for c in cases if c.subject_id == subject), key=lambda c: c.case_id
        ):
            if any(
                parent == "Tumor" and scope == case.case_id
                for parent, scope, _ in accepted
            ):
                tumor = ET.SubElement(patient, "Tumor")
                add_items(tumor, "Tumor", case.case_id)
                tumors += 1
    # XML parsers normalize literal carriage returns; character references retain
    # the exact reviewed value when a consumer reads the file.
    payload = ET.tostring(root, encoding="utf-8", xml_declaration=True).replace(
        b"\r", b"&#13;"
    )
    if len(payload) > 4_194_304:
        raise _safe_code("input_limit")
    _write(path, payload)
    return NAACCRExportReport(
        True,
        "sha256:" + hashlib.sha256(payload).hexdigest(),
        len(payload),
        len(cases),
        len(keys),
        tumors,
        items_written,
        tuple(
            sorted(
                losses,
                key=lambda v: (
                    v.case_index,
                    -1 if v.field_index is None else v.field_index,
                    v.code,
                ),
            )
        ),
        dictionary.source_digest,
        envelope.manifest_digest,
        mapping_digest,
    )
