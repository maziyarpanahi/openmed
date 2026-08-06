"""Eval-only loaders for credentialed i2b2 and clinical relation corpora.

The i2b2 2006 Track 1B and i2b2/UTHealth 2014 de-identification corpora
require approved local access under the i2b2/DBMI data-use agreement. This
module never downloads or vendors those records; it only parses XML files from
an explicit credentialed directory outside the repository tree.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from openmed.core.labels import (
    AGE,
    CANONICAL_LABELS,
    CONDITION,
    DATE,
    DOSAGE,
    DURATION,
    EMAIL,
    FORM,
    FREQUENCY,
    GENE_SYMBOL,
    ID_NUM,
    INDICATION,
    IP_ADDRESS,
    LOCATION,
    MEDICATION,
    MICROORGANISM,
    OCCUPATION,
    ORGANIZATION,
    OTHER,
    PERSON,
    PHONE,
    ROUTE,
    SSN,
    STREET_ADDRESS,
    STRENGTH,
    URL,
    USERNAME,
    VARIANT_DESCRIPTOR,
    VEHICLE_REGISTRATION,
    ZIPCODE,
    normalize_label,
)
from openmed.eval.datasets.drugprot import (
    DrugProtEntity,
    DrugProtRelation,
    DrugProtRelationFixture,
)
from openmed.eval.datasets.dua_stubs import DUACredentialRequired
from openmed.eval.datasets.licenses import license_for
from openmed.eval.harness import BenchmarkFixture
from openmed.eval.metrics import EvalSpan

I2B2 = "i2b2"
SUPPORTED_I2B2_YEARS: tuple[int, ...] = (2006, 2014)
I2B2_DUA_NAME = "i2b2/DBMI DUA"
I2B2_PATH_ENV = "OPENMED_I2B2_PATH"
I2B2_YEAR_ENV = "OPENMED_I2B2_YEAR"

BIORED = "biored"
N2C2_2018 = "n2c2-2018"
N2C2_2022 = "n2c2-2022"
DUA_RELATION_CORPORA: tuple[str, ...] = (BIORED, N2C2_2018, N2C2_2022)
BIORED_PATH_ENV = "OPENMED_BIORED_PATH"
N2C2_2018_PATH_ENV = "OPENMED_N2C2_2018_PATH"
N2C2_2022_PATH_ENV = "OPENMED_N2C2_2022_PATH"

DUA_RELATION_PATH_ENVS: Mapping[str, str] = {
    BIORED: BIORED_PATH_ENV,
    N2C2_2018: N2C2_2018_PATH_ENV,
    N2C2_2022: N2C2_2022_PATH_ENV,
}
DUA_RELATION_NAMES: Mapping[str, str] = {
    BIORED: "BioRED data-use terms",
    N2C2_2018: "n2c2/DBMI 2018 DUA",
    N2C2_2022: "n2c2/DBMI 2022 and SHAC DUA",
}

BIORED_ENTITY_TO_CANONICAL: Mapping[str, str] = {
    "cellline": OTHER,
    "chemical": MEDICATION,
    "chemicalentity": MEDICATION,
    "disease": CONDITION,
    "diseaseorphenotypicfeature": CONDITION,
    "gene": GENE_SYMBOL,
    "geneorgeneproduct": GENE_SYMBOL,
    "organism": MICROORGANISM,
    "organismtaxon": MICROORGANISM,
    "sequencevariant": VARIANT_DESCRIPTOR,
    "variant": VARIANT_DESCRIPTOR,
}
BIORED_RELATION_TO_CANONICAL: Mapping[str, str] = {
    "association": "ASSOCIATED_WITH",
    "bind": "BINDS",
    "comparison": "COMPARED_WITH",
    "conversion": "CONVERTS_TO",
    "cotreatment": "CO_TREATMENT",
    "druginteraction": "DRUG_INTERACTION",
    "negativecorrelation": "NEGATIVELY_CORRELATED_WITH",
    "positivecorrelation": "POSITIVELY_CORRELATED_WITH",
}

N2C2_2018_ENTITY_TO_CANONICAL: Mapping[str, str] = {
    "ade": CONDITION,
    "dosage": DOSAGE,
    "drug": MEDICATION,
    "duration": DURATION,
    "form": FORM,
    "frequency": FREQUENCY,
    "reason": INDICATION,
    "route": ROUTE,
    "strength": STRENGTH,
}
N2C2_2018_RELATION_TO_CANONICAL: Mapping[str, str] = {
    "adedrug": "DRUG_TO_ADE",
    "dosagedrug": "DRUG_TO_DOSE",
    "durationdrug": "DRUG_TO_DURATION",
    "formdrug": "DRUG_TO_FORM",
    "frequencydrug": "DRUG_TO_FREQUENCY",
    "reasondrug": "DRUG_TO_INDICATION",
    "routedrug": "DRUG_TO_ROUTE",
    "strengthdrug": "DRUG_TO_STRENGTH",
}

N2C2_2022_ENTITY_TO_CANONICAL: Mapping[str, str] = {
    "alcohol": OTHER,
    "amount": DOSAGE,
    "drug": OTHER,
    "duration": DURATION,
    "employment": OCCUPATION,
    "frequency": FREQUENCY,
    "history": OTHER,
    "livingstatus": OTHER,
    "status": OTHER,
    "statusemploy": OTHER,
    "statustime": OTHER,
    "tobacco": OTHER,
    "type": OTHER,
    "typeemploy": OTHER,
    "typeliving": OTHER,
}
N2C2_2022_RELATION_TO_CANONICAL: Mapping[str, str] = {
    "amount": "HAS_AMOUNT",
    "duration": "HAS_DURATION",
    "enddate": "HAS_END_DATE",
    "frequency": "HAS_FREQUENCY",
    "history": "HAS_HISTORY",
    "startdate": "HAS_START_DATE",
    "status": "HAS_STATUS",
    "statusemploy": "HAS_STATUS",
    "statustime": "HAS_STATUS",
    "temporality": "HAS_TEMPORALITY",
    "type": "HAS_TYPE",
    "typeemploy": "HAS_TYPE",
    "typeliving": "HAS_TYPE",
}

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


def map_biored_entity_label(label: str) -> str:
    """Map a BioRED entity type onto the OpenMed canonical taxonomy."""

    return _mapped_dua_label(
        label,
        BIORED_ENTITY_TO_CANONICAL,
        corpus=BIORED,
        value_kind="entity label",
    )


def map_dua_relation_type(corpus: str, relation_type: str) -> str:
    """Map a source DUA-corpus relation type onto the canonical taxonomy."""

    resolved = _normalize_dua_relation_corpus(corpus)
    relation_maps = {
        BIORED: BIORED_RELATION_TO_CANONICAL,
        N2C2_2018: N2C2_2018_RELATION_TO_CANONICAL,
        N2C2_2022: N2C2_2022_RELATION_TO_CANONICAL,
    }
    key = _mapping_key(relation_type)
    canonical = relation_maps[resolved].get(key)
    if canonical is None:
        allowed = ", ".join(sorted(relation_maps[resolved].values()))
        raise ValueError(
            f"unknown {resolved} relation type {relation_type!r}; "
            f"canonical values are: {allowed}"
        )
    return canonical


def dua_relation_suite_metadata(corpus: str) -> dict[str, Any]:
    """Return row-free metadata for a credentialed relation corpus."""

    resolved = _normalize_dua_relation_corpus(corpus)
    entity_maps: Mapping[str, Mapping[str, str]] = {
        BIORED: BIORED_ENTITY_TO_CANONICAL,
        N2C2_2018: N2C2_2018_ENTITY_TO_CANONICAL,
        N2C2_2022: N2C2_2022_ENTITY_TO_CANONICAL,
    }
    relation_maps: Mapping[str, Mapping[str, str]] = {
        BIORED: BIORED_RELATION_TO_CANONICAL,
        N2C2_2018: N2C2_2018_RELATION_TO_CANONICAL,
        N2C2_2022: N2C2_2022_RELATION_TO_CANONICAL,
    }
    return {
        "access": (
            f"credentialed local path only; pass path=... or set "
            f"{DUA_RELATION_PATH_ENVS[resolved]}"
        ),
        "cache_corpus_rows": False,
        "cadence": "human-run",
        "daily_blocking": False,
        "dataset": resolved,
        "dua": DUA_RELATION_NAMES[resolved],
        "entity_label_mapping": dict(sorted(entity_maps[resolved].items())),
        "eval_only": True,
        "gate_tier": "promotion",
        "license": license_for(resolved).to_dict(),
        "network_fetch": False,
        "promotion_blocking": True,
        "redistribution": "never; read-only from user-supplied credentialed path",
        "relation_type_mapping": dict(sorted(relation_maps[resolved].items())),
        "suite": resolved,
        "task": "relation",
    }


def load_dua_relation_fixtures(
    corpus: str,
    path: str | Path | None = None,
) -> list[DrugProtRelationFixture]:
    """Load one credentialed DUA relation corpus without fetching or caching.

    BioRED accepts BioC JSON, JSONL, or XML. The n2c2 loaders accept paired
    BRAT ``.txt`` and ``.ann`` files. Source rows remain in memory only.

    Args:
        corpus: ``biored``, ``n2c2-2018``, or ``n2c2-2022``.
        path: Authorized local file or directory outside the repository.

    Returns:
        DrugProt-compatible relation fixtures with canonical argument spans.

    Raises:
        DUACredentialRequired: If an authorized external path is unavailable.
        ValueError: If source annotations are malformed or contain unknown labels.
    """

    resolved = _normalize_dua_relation_corpus(corpus)
    source = _credentialed_relation_source(resolved, path)
    if resolved == BIORED:
        fixtures = _load_biored_relation_fixtures(source)
    else:
        fixtures = _load_n2c2_relation_fixtures(source, corpus=resolved)
    if not fixtures:
        raise DUACredentialRequired(
            f"{DUA_RELATION_NAMES[resolved]} path contains no supported relation "
            "fixtures; no corpus rows were loaded"
        )
    _validate_unique_relation_fixture_ids(fixtures, corpus=resolved)
    return fixtures


def load_biored_relation_fixtures(
    path: str | Path | None = None,
) -> list[DrugProtRelationFixture]:
    """Load credentialed BioRED BioC relation fixtures."""

    return load_dua_relation_fixtures(BIORED, path)


def load_n2c2_2018_relation_fixtures(
    path: str | Path | None = None,
) -> list[DrugProtRelationFixture]:
    """Load credentialed n2c2 2018 ADE/medication BRAT fixtures."""

    return load_dua_relation_fixtures(N2C2_2018, path)


def load_n2c2_2022_relation_fixtures(
    path: str | Path | None = None,
) -> list[DrugProtRelationFixture]:
    """Load credentialed n2c2 2022 SDOH BRAT event fixtures."""

    return load_dua_relation_fixtures(N2C2_2022, path)


def _credentialed_relation_source(
    corpus: str,
    path: str | Path | None,
) -> Path:
    path_env = DUA_RELATION_PATH_ENVS[corpus]
    raw_path = path or os.environ.get(path_env)
    if raw_path is None or not str(raw_path).strip():
        raise DUACredentialRequired(
            f"{DUA_RELATION_NAMES[corpus]} credentialed local path is required; "
            f"pass path=... or set {path_env}. No corpus rows were loaded."
        )
    candidate = Path(raw_path).expanduser().resolve(strict=False)
    if _is_relative_to(candidate, _REPO_ROOT):
        raise DUACredentialRequired(
            f"{DUA_RELATION_NAMES[corpus]} data must stay outside the repository "
            f"tree; refusing to read {candidate}. No corpus rows were loaded."
        )
    if not candidate.exists():
        raise DUACredentialRequired(
            f"{DUA_RELATION_NAMES[corpus]} credentialed path does not exist: "
            f"{candidate}. No corpus rows were loaded."
        )
    if not candidate.is_file() and not candidate.is_dir():
        raise DUACredentialRequired(
            f"{DUA_RELATION_NAMES[corpus]} path must be a file or directory: "
            f"{candidate}. No corpus rows were loaded."
        )
    return candidate


def _load_biored_relation_fixtures(root: Path) -> list[DrugProtRelationFixture]:
    sources = _biored_source_files(root)
    fixtures: list[DrugProtRelationFixture] = []
    for source in sources:
        _refuse_relation_source(source, BIORED)
        for document in _biored_documents(source):
            fixtures.append(
                _biored_fixture_from_mapping(document, source=source, root=root)
            )
    return fixtures


def _biored_source_files(root: Path) -> tuple[Path, ...]:
    supported = {".json", ".jsonl", ".xml"}
    if root.is_file():
        files = (root,) if root.suffix.casefold() in supported else ()
    else:
        files = tuple(
            path
            for path in sorted(root.rglob("*"))
            if path.is_file() and path.suffix.casefold() in supported
        )
    if not files:
        raise DUACredentialRequired(
            "BioRED credentialed path contains no BioC JSON, JSONL, or XML "
            "files; no corpus rows were loaded"
        )
    return files


def _biored_documents(path: Path) -> list[Mapping[str, Any]]:
    if path.suffix.casefold() == ".xml":
        return _bioc_xml_documents(path)
    if path.suffix.casefold() == ".jsonl":
        documents: list[Mapping[str, Any]] = []
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            payload = json.loads(line)
            rows = _bioc_json_documents(payload)
            if not rows:
                raise ValueError(
                    f"BioRED JSONL row {line_number} contains no BioC document"
                )
            documents.extend(rows)
        return documents
    return _bioc_json_documents(json.loads(path.read_text(encoding="utf-8")))


def _bioc_json_documents(payload: Any) -> list[Mapping[str, Any]]:
    if isinstance(payload, list):
        documents: list[Mapping[str, Any]] = []
        for item in payload:
            documents.extend(_bioc_json_documents(item))
        return documents
    if not isinstance(payload, Mapping):
        raise ValueError("BioRED BioC JSON must be a mapping or list")
    nested = payload.get("collection")
    if isinstance(nested, Mapping):
        return _bioc_json_documents(nested)
    rows = payload.get("documents")
    if isinstance(rows, list):
        return [_require_mapping(row, "BioRED document") for row in rows]
    if payload.get("id") is not None and isinstance(payload.get("passages"), list):
        return [payload]
    raise ValueError("BioRED BioC JSON contains no documents")


def _bioc_xml_documents(path: Path) -> list[Mapping[str, Any]]:
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as exc:
        raise ValueError(f"failed to parse BioRED BioC XML {path.name}: {exc}") from exc
    documents = [
        _bioc_xml_document(element)
        for element in root.iter()
        if _local_name(element.tag).casefold() == "document"
    ]
    if not documents:
        raise ValueError(f"BioRED BioC XML {path.name} contains no documents")
    return documents


def _bioc_xml_document(element: ET.Element) -> Mapping[str, Any]:
    passages: list[dict[str, Any]] = []
    document_relations: list[dict[str, Any]] = []
    for child in element:
        local_name = _local_name(child.tag).casefold()
        if local_name == "passage":
            passages.append(_bioc_xml_passage(child))
        elif local_name == "relation":
            document_relations.append(_bioc_xml_relation(child))
    return {
        "id": _bioc_child_text(element, "id"),
        "passages": passages,
        "relations": document_relations,
    }


def _bioc_xml_passage(element: ET.Element) -> dict[str, Any]:
    annotations: list[dict[str, Any]] = []
    relations: list[dict[str, Any]] = []
    for child in element:
        local_name = _local_name(child.tag).casefold()
        if local_name == "annotation":
            annotations.append(_bioc_xml_annotation(child))
        elif local_name == "relation":
            relations.append(_bioc_xml_relation(child))
    return {
        "offset": _parse_int(_bioc_child_text(element, "offset"), "BioC offset"),
        "text": _bioc_child_text(element, "text"),
        "annotations": annotations,
        "relations": relations,
    }


def _bioc_xml_annotation(element: ET.Element) -> dict[str, Any]:
    return {
        "id": str(element.attrib.get("id") or ""),
        "infons": _bioc_xml_infons(element),
        "locations": [
            {
                "offset": _parse_int(child.attrib.get("offset"), "BioC offset"),
                "length": _parse_int(child.attrib.get("length"), "BioC length"),
            }
            for child in element
            if _local_name(child.tag).casefold() == "location"
        ],
        "text": _bioc_child_text(element, "text"),
    }


def _bioc_xml_relation(element: ET.Element) -> dict[str, Any]:
    return {
        "id": str(element.attrib.get("id") or ""),
        "infons": _bioc_xml_infons(element),
        "nodes": [
            {
                "refid": str(child.attrib.get("refid") or ""),
                "role": str(child.attrib.get("role") or ""),
            }
            for child in element
            if _local_name(child.tag).casefold() == "node"
        ],
    }


def _bioc_xml_infons(element: ET.Element) -> dict[str, str]:
    return {
        str(child.attrib.get("key") or ""): str(child.text or "")
        for child in element
        if _local_name(child.tag).casefold() == "infon"
    }


def _bioc_child_text(element: ET.Element, name: str) -> str:
    for child in element:
        if _local_name(child.tag).casefold() == name.casefold():
            return str(child.text or "")
    return ""


def _biored_fixture_from_mapping(
    document: Mapping[str, Any],
    *,
    source: Path,
    root: Path,
) -> DrugProtRelationFixture:
    document_id = str(document.get("id") or "").strip()
    if not document_id:
        raise ValueError(f"BioRED document in {source.name} is missing an id")
    passages_value = document.get("passages")
    if not isinstance(passages_value, list) or not passages_value:
        raise ValueError(f"BioRED document {document_id!r} contains no passages")
    passages = [_require_mapping(row, "BioRED passage") for row in passages_value]
    text = _bioc_document_text(passages)
    fixture_id = _dua_fixture_id(BIORED, source, root, document_id)
    entities_by_id: dict[str, DrugProtEntity] = {}
    for annotation in _bioc_annotation_rows(document, passages):
        entity = _biored_entity_from_mapping(
            annotation,
            fixture_id=fixture_id,
            text=text,
        )
        if entity.entity_id in entities_by_id:
            raise ValueError(f"duplicate BioRED entity id: {entity.entity_id}")
        entities_by_id[entity.entity_id] = entity

    relations = tuple(
        _biored_relation_from_mapping(
            row,
            fixture_id=fixture_id,
            entities_by_id=entities_by_id,
            relation_index=index,
        )
        for index, row in enumerate(_bioc_relation_rows(document, passages), start=1)
    )
    metadata = {
        **dua_relation_suite_metadata(BIORED),
        "source_path_hash": _source_path_hash(source, root),
    }
    return DrugProtRelationFixture(
        fixture_id=fixture_id,
        text=text,
        entities=tuple(
            sorted(
                entities_by_id.values(),
                key=lambda entity: (entity.start, entity.end, entity.entity_id),
            )
        ),
        relations=relations,
        metadata=metadata,
    )


def _bioc_document_text(passages: Sequence[Mapping[str, Any]]) -> str:
    positioned: list[tuple[int, str]] = []
    for passage in passages:
        offset = _parse_int(passage.get("offset", 0), "BioC passage offset")
        passage_text = str(passage.get("text") or "")
        if offset < 0:
            raise ValueError("BioC passage offset must be non-negative")
        positioned.append((offset, passage_text))
    text_length = max(offset + len(value) for offset, value in positioned)
    characters = [" "] * text_length
    for offset, value in positioned:
        for index, character in enumerate(value, start=offset):
            current = characters[index]
            if current != " " and current != character:
                raise ValueError("overlapping BioC passages contain conflicting text")
            characters[index] = character
    return "".join(characters)


def _bioc_annotation_rows(
    document: Mapping[str, Any],
    passages: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    values: list[Any] = list(document.get("annotations") or [])
    for passage in passages:
        values.extend(passage.get("annotations") or [])
    return [_require_mapping(value, "BioRED annotation") for value in values]


def _bioc_relation_rows(
    document: Mapping[str, Any],
    passages: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    values: list[Any] = list(document.get("relations") or [])
    for passage in passages:
        values.extend(passage.get("relations") or [])
    return [_require_mapping(value, "BioRED relation") for value in values]


def _biored_entity_from_mapping(
    annotation: Mapping[str, Any],
    *,
    fixture_id: str,
    text: str,
) -> DrugProtEntity:
    entity_id = str(annotation.get("id") or "").strip()
    if not entity_id:
        raise ValueError("BioRED annotation id is required")
    infons = _mapping_or_empty(annotation.get("infons"))
    source_label = str(
        infons.get("type") or infons.get("entity_type") or annotation.get("type") or ""
    )
    canonical_label = map_biored_entity_label(source_label)
    raw_locations = annotation.get("locations") or annotation.get("location") or []
    if isinstance(raw_locations, Mapping):
        raw_locations = [raw_locations]
    if not isinstance(raw_locations, list) or not raw_locations:
        raise ValueError(f"BioRED entity {entity_id!r} has no location")
    locations = [_require_mapping(row, "BioRED location") for row in raw_locations]
    starts = [_parse_int(row.get("offset"), "BioC entity offset") for row in locations]
    ends = [
        start + _parse_int(row.get("length"), "BioC entity length")
        for start, row in zip(starts, locations, strict=True)
    ]
    start = min(starts)
    end = max(ends)
    if start < 0 or end < start or end > len(text):
        raise ValueError(
            f"invalid BioRED span offsets {start}:{end} for text length {len(text)}"
        )
    span_text = text[start:end]
    annotated_text = str(annotation.get("text") or "")
    if len(locations) == 1 and annotated_text and annotated_text != span_text:
        raise ValueError(f"BioRED span text mismatch for entity {entity_id!r}")
    return DrugProtEntity(
        pmid=fixture_id,
        entity_id=entity_id,
        source_label=source_label,
        start=start,
        end=end,
        text=span_text,
        canonical_label=canonical_label,
    )


def _biored_relation_from_mapping(
    relation: Mapping[str, Any],
    *,
    fixture_id: str,
    entities_by_id: Mapping[str, DrugProtEntity],
    relation_index: int,
) -> DrugProtRelation:
    infons = _mapping_or_empty(relation.get("infons"))
    source_type = str(
        infons.get("type") or infons.get("relation") or relation.get("type") or ""
    )
    relation_type = map_dua_relation_type(BIORED, source_type)
    nodes_value = relation.get("nodes") or relation.get("arguments") or []
    if not isinstance(nodes_value, list) or len(nodes_value) != 2:
        raise ValueError("BioRED relation must contain exactly two nodes")
    node_rows = [_require_mapping(node, "BioRED relation node") for node in nodes_value]
    node_ids = [
        str(row.get("refid") or row.get("id") or "").strip() for row in node_rows
    ]
    if any(not node_id for node_id in node_ids):
        raise ValueError("BioRED relation nodes require an entity reference")
    role_indexes = {
        "arg1": 0,
        "argument1": 0,
        "entity1": 0,
        "head": 0,
        "source": 0,
        "subject": 0,
        "arg2": 1,
        "argument2": 1,
        "entity2": 1,
        "object": 1,
        "tail": 1,
        "target": 1,
    }
    ordered_node_ids: list[str | None] = [None, None]
    for node_id, row in zip(node_ids, node_rows, strict=True):
        role_index = role_indexes.get(_mapping_key(str(row.get("role") or "")))
        if role_index is None or ordered_node_ids[role_index] is not None:
            ordered_node_ids = [None, None]
            break
        ordered_node_ids[role_index] = node_id
    if all(node_id is not None for node_id in ordered_node_ids):
        node_ids = [node_id for node_id in ordered_node_ids if node_id is not None]
    try:
        arg1, arg2 = (entities_by_id[node_id] for node_id in node_ids)
    except KeyError as exc:
        raise ValueError(
            f"BioRED relation references unknown entity {exc.args[0]!r}"
        ) from exc
    relation_id = str(relation.get("id") or f"R{relation_index}")
    return DrugProtRelation(
        pmid=fixture_id,
        relation_type=relation_type,
        arg1_id=node_ids[0],
        arg2_id=node_ids[1],
        arg1=arg1,
        arg2=arg2,
        scope="document",
        relation_id=relation_id,
        metadata={
            "canonical_relation_type": relation_type,
            "source_relation_type": source_type,
        },
    )


def _load_n2c2_relation_fixtures(
    root: Path,
    *,
    corpus: str,
) -> list[DrugProtRelationFixture]:
    pairs = _brat_document_pairs(root)
    fixtures = [
        _n2c2_fixture_from_brat(
            text_path,
            annotation_path,
            root=root,
            corpus=corpus,
        )
        for text_path, annotation_path in pairs
    ]
    return fixtures


def _brat_document_pairs(root: Path) -> tuple[tuple[Path, Path], ...]:
    if root.is_file():
        if root.suffix.casefold() == ".ann":
            annotation_files = (root,)
        elif root.suffix.casefold() == ".txt":
            annotation_files = (root.with_suffix(".ann"),)
        else:
            annotation_files = ()
    else:
        annotation_files = tuple(
            path for path in sorted(root.rglob("*.ann")) if path.is_file()
        )
    pairs: list[tuple[Path, Path]] = []
    for annotation_path in annotation_files:
        text_path = annotation_path.with_suffix(".txt")
        if not annotation_path.exists() or not text_path.exists():
            raise ValueError(
                "n2c2 BRAT input requires paired .ann and .txt files: "
                f"{annotation_path.name}"
            )
        pairs.append((text_path, annotation_path))
    if not pairs:
        raise DUACredentialRequired(
            "n2c2 credentialed path contains no paired BRAT .txt/.ann files; "
            "no corpus rows were loaded"
        )
    return tuple(pairs)


def _n2c2_fixture_from_brat(
    text_path: Path,
    annotation_path: Path,
    *,
    root: Path,
    corpus: str,
) -> DrugProtRelationFixture:
    _refuse_relation_source(text_path, corpus)
    _refuse_relation_source(annotation_path, corpus)
    text = _read_relation_text(text_path)
    fixture_id = _dua_fixture_id(corpus, text_path, root, text_path.stem)
    lines = annotation_path.read_text(encoding="utf-8").splitlines()
    entities_by_id: dict[str, DrugProtEntity] = {}
    for line in lines:
        if not line.startswith("T"):
            continue
        entity = _brat_entity_from_line(
            line,
            fixture_id=fixture_id,
            text=text,
            corpus=corpus,
        )
        if entity.entity_id in entities_by_id:
            raise ValueError(f"duplicate {corpus} entity id: {entity.entity_id}")
        entities_by_id[entity.entity_id] = entity

    relations: list[DrugProtRelation] = []
    for line in lines:
        if line.startswith("R"):
            relations.append(
                _brat_relation_from_line(
                    line,
                    fixture_id=fixture_id,
                    entities_by_id=entities_by_id,
                    corpus=corpus,
                )
            )
        elif line.startswith("E"):
            if corpus != N2C2_2022:
                raise ValueError(f"{corpus} does not support BRAT event annotations")
            relations.extend(
                _brat_event_relations_from_line(
                    line,
                    fixture_id=fixture_id,
                    entities_by_id=entities_by_id,
                )
            )
    metadata = {
        **dua_relation_suite_metadata(corpus),
        "source_path_hash": _source_path_hash(text_path, root),
    }
    return DrugProtRelationFixture(
        fixture_id=fixture_id,
        text=text,
        entities=tuple(
            sorted(
                entities_by_id.values(),
                key=lambda entity: (entity.start, entity.end, entity.entity_id),
            )
        ),
        relations=tuple(relations),
        metadata=metadata,
    )


def _brat_entity_from_line(
    line: str,
    *,
    fixture_id: str,
    text: str,
    corpus: str,
) -> DrugProtEntity:
    columns = line.split("\t")
    if len(columns) < 3:
        raise ValueError(f"malformed {corpus} BRAT entity line")
    entity_id = columns[0]
    label_and_offsets = columns[1].split(maxsplit=1)
    if len(label_and_offsets) != 2:
        raise ValueError(f"malformed {corpus} BRAT entity offsets")
    source_label, offset_spec = label_and_offsets
    spans = []
    for segment in offset_spec.split(";"):
        values = segment.split()
        if len(values) != 2:
            raise ValueError(f"malformed {corpus} discontinuous BRAT span")
        spans.append(
            (
                _parse_int(values[0], "BRAT span start"),
                _parse_int(values[1], "BRAT span end"),
            )
        )
    start = min(value[0] for value in spans)
    end = max(value[1] for value in spans)
    if start < 0 or end < start or end > len(text):
        raise ValueError(
            f"invalid {corpus} span offsets {start}:{end} for text length {len(text)}"
        )
    span_text = text[start:end]
    if len(spans) == 1 and columns[2] != span_text:
        raise ValueError(f"{corpus} span text mismatch for entity {entity_id!r}")
    canonical_label = _map_n2c2_entity_label(corpus, source_label)
    return DrugProtEntity(
        pmid=fixture_id,
        entity_id=entity_id,
        source_label=source_label,
        start=start,
        end=end,
        text=span_text,
        canonical_label=canonical_label,
    )


def _brat_relation_from_line(
    line: str,
    *,
    fixture_id: str,
    entities_by_id: Mapping[str, DrugProtEntity],
    corpus: str,
) -> DrugProtRelation:
    columns = line.split("\t")
    if len(columns) != 2:
        raise ValueError(f"malformed {corpus} BRAT relation line")
    relation_id = columns[0]
    values = columns[1].split()
    if len(values) != 3:
        raise ValueError(f"malformed {corpus} BRAT relation arguments")
    source_type = values[0]
    argument_ids = [_brat_argument_id(value) for value in values[1:]]
    relation_type = map_dua_relation_type(corpus, source_type)
    return _dua_relation(
        fixture_id=fixture_id,
        relation_id=relation_id,
        relation_type=relation_type,
        source_type=source_type,
        argument_ids=argument_ids,
        entities_by_id=entities_by_id,
        reorder_medication=corpus == N2C2_2018,
    )


def _brat_event_relations_from_line(
    line: str,
    *,
    fixture_id: str,
    entities_by_id: Mapping[str, DrugProtEntity],
) -> list[DrugProtRelation]:
    columns = line.split("\t")
    if len(columns) != 2:
        raise ValueError("malformed n2c2-2022 BRAT event line")
    event_id = columns[0]
    values = columns[1].split()
    if len(values) < 2:
        raise ValueError("n2c2-2022 BRAT event requires trigger and argument")
    event_label, separator, trigger_id = values[0].partition(":")
    if separator != ":" or not event_label or not trigger_id:
        raise ValueError("malformed n2c2-2022 BRAT event trigger")
    relations: list[DrugProtRelation] = []
    for index, value in enumerate(values[1:], start=1):
        role, separator, argument_id = value.partition(":")
        if separator != ":" or not role or not argument_id:
            raise ValueError("malformed n2c2-2022 BRAT event argument")
        relation_type = map_dua_relation_type(N2C2_2022, role)
        relations.append(
            _dua_relation(
                fixture_id=fixture_id,
                relation_id=f"{event_id}-{index}",
                relation_type=relation_type,
                source_type=f"{event_label}:{role}",
                argument_ids=[trigger_id, argument_id],
                entities_by_id=entities_by_id,
                reorder_medication=False,
            )
        )
    return relations


def _dua_relation(
    *,
    fixture_id: str,
    relation_id: str,
    relation_type: str,
    source_type: str,
    argument_ids: list[str],
    entities_by_id: Mapping[str, DrugProtEntity],
    reorder_medication: bool,
) -> DrugProtRelation:
    try:
        arguments = [entities_by_id[argument_id] for argument_id in argument_ids]
    except KeyError as exc:
        raise ValueError(
            f"DUA relation references unknown entity {exc.args[0]!r}"
        ) from exc
    if reorder_medication:
        medication_indexes = [
            index
            for index, entity in enumerate(arguments)
            if entity.canonical_label == MEDICATION
        ]
        if len(medication_indexes) != 1:
            raise ValueError(
                "n2c2-2018 relation must connect exactly one medication entity"
            )
        medication_index = medication_indexes[0]
        if medication_index == 1:
            arguments.reverse()
            argument_ids.reverse()
    return DrugProtRelation(
        pmid=fixture_id,
        relation_type=relation_type,
        arg1_id=argument_ids[0],
        arg2_id=argument_ids[1],
        arg1=arguments[0],
        arg2=arguments[1],
        scope="document",
        relation_id=relation_id,
        metadata={
            "canonical_relation_type": relation_type,
            "source_relation_type": source_type,
        },
    )


def _brat_argument_id(value: str) -> str:
    _, separator, argument_id = value.partition(":")
    if separator != ":" or not argument_id:
        raise ValueError(f"malformed BRAT relation argument: {value!r}")
    return argument_id


def _read_relation_text(path: Path) -> str:
    """Read BRAT text without normalizing newlines used by span offsets."""

    with path.open("r", encoding="utf-8", newline="") as handle:
        return handle.read()


def _map_n2c2_entity_label(corpus: str, label: str) -> str:
    mappings = {
        N2C2_2018: N2C2_2018_ENTITY_TO_CANONICAL,
        N2C2_2022: N2C2_2022_ENTITY_TO_CANONICAL,
    }
    return _mapped_dua_label(
        label,
        mappings[corpus],
        corpus=corpus,
        value_kind="entity label",
    )


def _mapped_dua_label(
    label: str,
    mapping: Mapping[str, str],
    *,
    corpus: str,
    value_kind: str,
) -> str:
    canonical = mapping.get(_mapping_key(label))
    if canonical is None:
        allowed = ", ".join(sorted(mapping))
        raise ValueError(
            f"unknown {corpus} {value_kind} {label!r}; expected one of: {allowed}"
        )
    if canonical not in CANONICAL_LABELS:
        raise RuntimeError(f"{corpus} mapping is not canonical: {canonical!r}")
    return canonical


def _normalize_dua_relation_corpus(corpus: str) -> str:
    key = str(corpus).strip().casefold().replace("_", "-")
    aliases = {
        "biored": BIORED,
        "n2c2-2018": N2C2_2018,
        "n2c2-2022": N2C2_2022,
    }
    try:
        return aliases[key]
    except KeyError as exc:
        allowed = ", ".join(DUA_RELATION_CORPORA)
        raise ValueError(
            f"unknown DUA relation corpus {corpus!r}; expected one of: {allowed}"
        ) from exc


def _mapping_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().casefold())


def _mapping_or_empty(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _require_mapping(value: Any, description: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{description} must be a mapping")
    return value


def _dua_fixture_id(corpus: str, source: Path, root: Path, record_id: str) -> str:
    relative_source = _relative_source_path(source, root)
    digest = hashlib.sha256(
        f"{corpus}:{relative_source}:{record_id}".encode("utf-8")
    ).hexdigest()[:16]
    return f"{corpus}-{digest}"


def _source_path_hash(source: Path, root: Path) -> str:
    return hashlib.sha256(
        _relative_source_path(source, root).encode("utf-8")
    ).hexdigest()


def _relative_source_path(source: Path, root: Path) -> str:
    base = root if root.is_dir() else root.parent
    try:
        return source.relative_to(base).as_posix()
    except ValueError:
        return source.name


def _refuse_relation_source(source: Path, corpus: str) -> None:
    resolved = source.resolve(strict=False)
    if _is_relative_to(resolved, _REPO_ROOT):
        raise DUACredentialRequired(
            f"{DUA_RELATION_NAMES[corpus]} data must stay outside the repository "
            f"tree; refusing to read {resolved}. No corpus rows were loaded."
        )


def _validate_unique_relation_fixture_ids(
    fixtures: Iterable[DrugProtRelationFixture],
    *,
    corpus: str,
) -> None:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for fixture in fixtures:
        if fixture.fixture_id in seen:
            duplicates.add(fixture.fixture_id)
        seen.add(fixture.fixture_id)
    if duplicates:
        raise ValueError(
            f"duplicate {corpus} relation fixture ids: " + ", ".join(sorted(duplicates))
        )


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


def _parse_int(value: Any, field_name: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{field_name} must be an integer: {value!r}") from None


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

_dua_entity_mappings = {
    BIORED: BIORED_ENTITY_TO_CANONICAL,
    N2C2_2018: N2C2_2018_ENTITY_TO_CANONICAL,
    N2C2_2022: N2C2_2022_ENTITY_TO_CANONICAL,
}
_invalid_dua_entity_mappings = {
    corpus: {
        source_label: canonical
        for source_label, canonical in mapping.items()
        if canonical not in CANONICAL_LABELS
    }
    for corpus, mapping in _dua_entity_mappings.items()
}
_invalid_dua_entity_mappings = {
    corpus: mapping
    for corpus, mapping in _invalid_dua_entity_mappings.items()
    if mapping
}
if _invalid_dua_entity_mappings:
    raise RuntimeError(
        "DUA relation entity mappings must use canonical labels: "
        f"{_invalid_dua_entity_mappings}"
    )


__all__ = [
    "BIORED",
    "BIORED_ENTITY_TO_CANONICAL",
    "BIORED_PATH_ENV",
    "BIORED_RELATION_TO_CANONICAL",
    "DUA_RELATION_CORPORA",
    "DUA_RELATION_NAMES",
    "DUA_RELATION_PATH_ENVS",
    "I2B2",
    "I2B2CredentialRequired",
    "I2B2_DUA_NAME",
    "I2B2_PATH_ENV",
    "I2B2_PHI_TAGS",
    "I2B2_PHI_TAG_ALIASES",
    "I2B2_PHI_TAG_TO_CANONICAL",
    "I2B2_SUITE_METADATA",
    "I2B2_YEAR_ENV",
    "N2C2_2018",
    "N2C2_2018_ENTITY_TO_CANONICAL",
    "N2C2_2018_PATH_ENV",
    "N2C2_2018_RELATION_TO_CANONICAL",
    "N2C2_2022",
    "N2C2_2022_ENTITY_TO_CANONICAL",
    "N2C2_2022_PATH_ENV",
    "N2C2_2022_RELATION_TO_CANONICAL",
    "SUPPORTED_I2B2_YEARS",
    "dua_relation_suite_metadata",
    "i2b2_suite_metadata",
    "load_biored_relation_fixtures",
    "load_dua_relation_fixtures",
    "load_i2b2_deid",
    "load_n2c2_2018_relation_fixtures",
    "load_n2c2_2022_relation_fixtures",
    "map_biored_entity_label",
    "map_dua_relation_type",
    "map_i2b2_phi_tag",
]
