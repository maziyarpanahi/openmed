"""Eval-only loader for credentialed i2b2 de-identification corpora.

The i2b2 2006 Track 1B and i2b2/UTHealth 2014 de-identification corpora
require approved local access under the i2b2/DBMI data-use agreement. This
module never downloads or vendors those records; it only parses XML files from
an explicit credentialed directory outside the repository tree.
"""

from __future__ import annotations

import hashlib
import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Iterable, Mapping

from openmed.core.labels import (
    AGE,
    CANONICAL_LABELS,
    DATE,
    EMAIL,
    ID_NUM,
    IP_ADDRESS,
    LOCATION,
    OCCUPATION,
    ORGANIZATION,
    PERSON,
    PHONE,
    SSN,
    STREET_ADDRESS,
    URL,
    USERNAME,
    VEHICLE_REGISTRATION,
    ZIPCODE,
    normalize_label,
)
from openmed.eval.datasets.dua_stubs import DUACredentialRequired
from openmed.eval.harness import BenchmarkFixture
from openmed.eval.metrics import EvalSpan

I2B2 = "i2b2"
SUPPORTED_I2B2_YEARS: tuple[int, ...] = (2006, 2014)
I2B2_DUA_NAME = "i2b2/DBMI DUA"
I2B2_PATH_ENV = "OPENMED_I2B2_PATH"
I2B2_YEAR_ENV = "OPENMED_I2B2_YEAR"

I2B2_PHI_TAGS: tuple[str, ...] = (
    "AGE",
    "DATE",
    "PROFESSION",
    "NAME/PATIENT",
    "NAME/DOCTOR",
    "NAME/USERNAME",
    "LOCATION/HOSPITAL",
    "LOCATION/ORGANIZATION",
    "LOCATION/ROOM",
    "LOCATION/DEPARTMENT",
    "LOCATION/STREET",
    "LOCATION/CITY",
    "LOCATION/STATE",
    "LOCATION/COUNTRY",
    "LOCATION/ZIP",
    "LOCATION/LOCATION_OTHER",
    "CONTACT/PHONE",
    "CONTACT/FAX",
    "CONTACT/EMAIL",
    "CONTACT/URL",
    "CONTACT/IPADDRESS",
    "ID/SSN",
    "ID/MEDICALRECORD",
    "ID/HEALTHPLAN",
    "ID/ACCOUNT",
    "ID/LICENSE",
    "ID/VEHICLE",
    "ID/DEVICE",
    "ID/BIOID",
    "ID/IDNUM",
)

I2B2_PHI_TAG_TO_CANONICAL: Mapping[str, str] = {
    "AGE": AGE,
    "DATE": DATE,
    "PROFESSION": OCCUPATION,
    "NAME/PATIENT": PERSON,
    "NAME/DOCTOR": PERSON,
    "NAME/USERNAME": USERNAME,
    "LOCATION/HOSPITAL": ORGANIZATION,
    "LOCATION/ORGANIZATION": ORGANIZATION,
    "LOCATION/ROOM": LOCATION,
    "LOCATION/DEPARTMENT": ORGANIZATION,
    "LOCATION/STREET": STREET_ADDRESS,
    "LOCATION/CITY": LOCATION,
    "LOCATION/STATE": LOCATION,
    "LOCATION/COUNTRY": LOCATION,
    "LOCATION/ZIP": ZIPCODE,
    "LOCATION/LOCATION_OTHER": LOCATION,
    "CONTACT/PHONE": PHONE,
    "CONTACT/FAX": PHONE,
    "CONTACT/EMAIL": EMAIL,
    "CONTACT/URL": URL,
    "CONTACT/IPADDRESS": IP_ADDRESS,
    "ID/SSN": SSN,
    "ID/MEDICALRECORD": ID_NUM,
    "ID/HEALTHPLAN": ID_NUM,
    "ID/ACCOUNT": ID_NUM,
    "ID/LICENSE": ID_NUM,
    "ID/VEHICLE": VEHICLE_REGISTRATION,
    "ID/DEVICE": ID_NUM,
    "ID/BIOID": ID_NUM,
    "ID/IDNUM": ID_NUM,
}

I2B2_PHI_TAG_ALIASES: Mapping[str, str] = {
    "PATIENT": "NAME/PATIENT",
    "DOCTOR": "NAME/DOCTOR",
    "USERNAME": "NAME/USERNAME",
    "HOSPITAL": "LOCATION/HOSPITAL",
    "ORGANIZATION": "LOCATION/ORGANIZATION",
    "ROOM": "LOCATION/ROOM",
    "DEPARTMENT": "LOCATION/DEPARTMENT",
    "STREET": "LOCATION/STREET",
    "CITY": "LOCATION/CITY",
    "STATE": "LOCATION/STATE",
    "COUNTRY": "LOCATION/COUNTRY",
    "ZIP": "LOCATION/ZIP",
    "LOCATION": "LOCATION/LOCATION_OTHER",
    "LOCATION_OTHER": "LOCATION/LOCATION_OTHER",
    "PHONE": "CONTACT/PHONE",
    "FAX": "CONTACT/FAX",
    "EMAIL": "CONTACT/EMAIL",
    "URL": "CONTACT/URL",
    "IP_ADDRESS": "CONTACT/IPADDRESS",
    "IPADDRESS": "CONTACT/IPADDRESS",
    "SSN": "ID/SSN",
    "SOCIAL_SECURITY_NUMBER": "ID/SSN",
    "MEDICAL_RECORD": "ID/MEDICALRECORD",
    "MEDICAL_RECORD_NUMBER": "ID/MEDICALRECORD",
    "MEDICALRECORD": "ID/MEDICALRECORD",
    "MRN": "ID/MEDICALRECORD",
    "HEALTH_PLAN": "ID/HEALTHPLAN",
    "HEALTH_PLAN_NUMBER": "ID/HEALTHPLAN",
    "HEALTHPLAN": "ID/HEALTHPLAN",
    "ACCOUNT_NUMBER": "ID/ACCOUNT",
    "ACCOUNT": "ID/ACCOUNT",
    "LICENSE_NUMBER": "ID/LICENSE",
    "LICENSE": "ID/LICENSE",
    "VEHICLE_ID": "ID/VEHICLE",
    "VEHICLE": "ID/VEHICLE",
    "DEVICE_ID": "ID/DEVICE",
    "DEVICE": "ID/DEVICE",
    "BIOID": "ID/BIOID",
    "BIOMETRIC_ID": "ID/BIOID",
    "IDNUM": "ID/IDNUM",
    "ID": "ID/IDNUM",
}

I2B2_SUITE_METADATA: Mapping[str, Any] = {
    "access": (
        "requires an approved local i2b2/DBMI DUA credentialed directory; "
        f"pass path=... or set {I2B2_PATH_ENV}"
    ),
    "dua": I2B2_DUA_NAME,
    "label_mapping": dict(sorted(I2B2_PHI_TAG_TO_CANONICAL.items())),
    "redistribution": "not vendored; eval-only local credentialed directory",
    "suite": I2B2,
    "supported_years": SUPPORTED_I2B2_YEARS,
}

_CATEGORY_TAGS = {"CONTACT", "ID", "LOCATION", "NAME"}
_DIRECT_TAGS = {"AGE", "DATE", "PROFESSION"}
_REPO_ROOT = Path(__file__).resolve().parents[3]


class I2B2CredentialRequired(DUACredentialRequired):
    """Raised when i2b2 loading lacks approved local DUA access."""


def load_i2b2_deid(
    path: str | Path | None = None,
    year: int | str | None = None,
) -> list[BenchmarkFixture]:
    """Load i2b2 de-identification XML files from a credentialed directory.

    Args:
        path: Approved local directory containing i2b2 XML files. If omitted,
            ``OPENMED_I2B2_PATH`` is used.
        year: Supported corpus year, currently ``2006`` or ``2014``.

    Returns:
        Benchmark fixtures with canonical-label gold spans.

    Raises:
        I2B2CredentialRequired: If no approved local path is configured, the
            path is empty, or it points inside this repository.
        ValueError: If XML spans are malformed or contain unknown PHI tags.
    """
    parsed_year = _parse_year(year or os.environ.get(I2B2_YEAR_ENV, 2014))
    root = _credentialed_directory(path)
    xml_files = tuple(_iter_xml_files(root))
    if not xml_files:
        raise I2B2CredentialRequired(
            f"{I2B2_DUA_NAME} credentialed directory is empty or contains no "
            f"i2b2 XML files: {root}"
        )

    fixtures = [
        _fixture_from_xml(xml_path, root=root, year=parsed_year)
        for xml_path in xml_files
    ]
    _validate_unique_fixture_ids(fixtures)
    return fixtures


def i2b2_suite_metadata() -> dict[str, Any]:
    """Return i2b2 benchmark suite metadata without reading local data."""
    return dict(I2B2_SUITE_METADATA)


def map_i2b2_phi_tag(label: str) -> str:
    """Map an i2b2 PHI tag or ``CATEGORY/TYPE`` pair to a canonical label."""
    source_tag = _canonical_source_tag(label)
    canonical = I2B2_PHI_TAG_TO_CANONICAL.get(source_tag)
    if canonical is None:
        allowed = ", ".join(I2B2_PHI_TAGS)
        raise ValueError(f"unknown i2b2 PHI tag {label!r}; expected one of: {allowed}")
    normalized = normalize_label(canonical)
    if normalized not in CANONICAL_LABELS:
        raise RuntimeError(
            f"i2b2 mapping for {source_tag!r} is not canonical: {canonical!r}"
        )
    return normalized


def _fixture_from_xml(path: Path, *, root: Path, year: int) -> BenchmarkFixture:
    try:
        document = ET.parse(path)
    except ET.ParseError as exc:
        raise ValueError(f"failed to parse i2b2 XML {path.name}: {exc}") from exc

    text_node = _first_child(document.getroot(), "TEXT")
    tags_node = _first_child(document.getroot(), "TAGS")
    if text_node is None:
        raise ValueError(f"i2b2 XML {path.name} is missing a TEXT element")
    if tags_node is None:
        raise ValueError(f"i2b2 XML {path.name} is missing a TAGS element")

    text = "".join(text_node.itertext())
    source_hash = _source_hash(path, root)
    spans = tuple(
        _span_from_element(element, text=text, source_file=path.name)
        for element in tags_node
        if isinstance(element.tag, str)
    )
    return BenchmarkFixture(
        fixture_id=f"i2b2-{year}-{source_hash}",
        text=text,
        gold_spans=spans,
        language="en",
        metadata={
            "dua": I2B2_DUA_NAME,
            "redistribution": "not vendored; loaded from credentialed path",
            "source_path_hash": source_hash,
            "suite": I2B2,
            "year": year,
        },
    )


def _span_from_element(
    element: ET.Element,
    *,
    text: str,
    source_file: str,
) -> EvalSpan:
    attrs = _attributes(element)
    start = _required_int(attrs, "start", source_file=source_file)
    end = _required_int(attrs, "end", source_file=source_file)
    if start < 0 or end < start or end > len(text):
        raise ValueError(
            f"invalid i2b2 span offsets {start}:{end} in {source_file} "
            f"for text length {len(text)}"
        )

    category = _source_category(element)
    source_type = _normalize_token(str(attrs.get("type", "")))
    source_tag = _source_tag(category, source_type)
    canonical_label = map_i2b2_phi_tag(source_tag)
    canonical_source_tag = _canonical_source_tag(source_tag)
    return EvalSpan(
        start=start,
        end=end,
        label=canonical_label,
        text=text[start:end],
        language="en",
        metadata={
            "canonical_label": canonical_label,
            "i2b2_category": category,
            "i2b2_tag": canonical_source_tag,
            "i2b2_type": source_type,
            "span_id": str(attrs.get("id", "")),
        },
    )


def _source_tag(category: str, source_type: str) -> str:
    if category in _CATEGORY_TAGS and source_type:
        return f"{category}/{source_type}"
    if category in _DIRECT_TAGS:
        return category
    if category == "PHI" and source_type:
        return source_type
    if source_type and category not in I2B2_PHI_TAG_TO_CANONICAL:
        return source_type
    return category


def _credentialed_directory(path: str | Path | None) -> Path:
    raw_path = path or os.environ.get(I2B2_PATH_ENV)
    if raw_path is None or str(raw_path).strip() == "":
        raise I2B2CredentialRequired(
            f"{I2B2_DUA_NAME} credentialed local path is required; pass path=... "
            f"or set {I2B2_PATH_ENV}. No i2b2 data is bundled."
        )

    candidate = Path(raw_path).expanduser().resolve(strict=False)
    if _is_relative_to(candidate, _REPO_ROOT):
        raise I2B2CredentialRequired(
            f"{I2B2_DUA_NAME} data must be kept outside the repository tree; "
            f"refusing to read {candidate}"
        )
    if not candidate.exists():
        raise I2B2CredentialRequired(
            f"{I2B2_DUA_NAME} credentialed path does not exist: {candidate}"
        )
    if not candidate.is_dir():
        raise I2B2CredentialRequired(
            f"{I2B2_DUA_NAME} credentialed path must be a directory: {candidate}"
        )
    return candidate


def _iter_xml_files(root: Path) -> Iterable[Path]:
    return (
        path
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.suffix.lower() == ".xml"
    )


def _parse_year(year: int | str) -> int:
    try:
        parsed = int(year)
    except (TypeError, ValueError):
        raise ValueError(f"unsupported i2b2 de-identification year: {year!r}") from None
    if parsed not in SUPPORTED_I2B2_YEARS:
        allowed = ", ".join(str(item) for item in SUPPORTED_I2B2_YEARS)
        raise ValueError(
            f"unsupported i2b2 de-identification year {parsed}; use {allowed}"
        )
    return parsed


def _source_hash(path: Path, root: Path) -> str:
    relative = path.relative_to(root).as_posix()
    return hashlib.sha256(relative.encode("utf-8")).hexdigest()[:16]


def _canonical_source_tag(label: str) -> str:
    if "/" in label:
        category, source_type = (
            _normalize_token(part) for part in label.split("/", maxsplit=1)
        )
        normalized = f"{category}/{source_type}"
        if normalized in I2B2_PHI_TAG_TO_CANONICAL:
            return normalized
        aliased = I2B2_PHI_TAG_ALIASES.get(normalized)
        if aliased is not None:
            return aliased
        type_alias = I2B2_PHI_TAG_ALIASES.get(source_type)
        if type_alias is not None:
            return type_alias
        return normalized
    normalized = _normalize_token(label)
    return I2B2_PHI_TAG_ALIASES.get(normalized, normalized)


def _source_category(element: ET.Element) -> str:
    return _normalize_token(_local_name(element.tag))


def _attributes(element: ET.Element) -> dict[str, Any]:
    return {
        _normalize_token(_local_name(key)).lower(): value
        for key, value in element.attrib.items()
    }


def _local_name(name: str) -> str:
    return name.rsplit("}", maxsplit=1)[-1]


def _normalize_token(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9]+", "_", value.strip()).strip("_").upper()
    return token


def _first_child(root: ET.Element, name: str) -> ET.Element | None:
    for element in root.iter():
        if _local_name(element.tag).upper() == name:
            return element
    return None


def _required_int(
    attrs: Mapping[str, Any],
    key: str,
    *,
    source_file: str,
) -> int:
    try:
        return int(attrs[key])
    except (KeyError, TypeError, ValueError):
        raise ValueError(
            f"i2b2 tag in {source_file} missing integer {key!r}: {attrs!r}"
        ) from None


def _validate_unique_fixture_ids(fixtures: Iterable[BenchmarkFixture]) -> None:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for fixture in fixtures:
        if fixture.fixture_id in seen:
            duplicates.add(fixture.fixture_id)
        seen.add(fixture.fixture_id)
    if duplicates:
        joined = ", ".join(sorted(duplicates))
        raise ValueError(f"duplicate i2b2 benchmark fixture id(s): {joined}")


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


_missing_mappings = sorted(set(I2B2_PHI_TAGS) - set(I2B2_PHI_TAG_TO_CANONICAL))
_extra_mappings = sorted(set(I2B2_PHI_TAG_TO_CANONICAL) - set(I2B2_PHI_TAGS))
_invalid_mappings = {
    tag: canonical
    for tag, canonical in I2B2_PHI_TAG_TO_CANONICAL.items()
    if normalize_label(canonical) not in CANONICAL_LABELS
}
if _missing_mappings or _extra_mappings or _invalid_mappings:
    raise RuntimeError(
        "i2b2 PHI mapping must cover the committed tag table exactly; "
        f"missing={_missing_mappings}, extra={_extra_mappings}, "
        f"invalid={_invalid_mappings}"
    )


__all__ = [
    "I2B2",
    "I2B2CredentialRequired",
    "I2B2_DUA_NAME",
    "I2B2_PATH_ENV",
    "I2B2_PHI_TAGS",
    "I2B2_PHI_TAG_ALIASES",
    "I2B2_PHI_TAG_TO_CANONICAL",
    "I2B2_SUITE_METADATA",
    "I2B2_YEAR_ENV",
    "SUPPORTED_I2B2_YEARS",
    "i2b2_suite_metadata",
    "load_i2b2_deid",
    "map_i2b2_phi_tag",
]
