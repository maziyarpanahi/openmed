"""Eval-only loader for the n2c2 2018 Track 1 de-identification corpus.

The n2c2 2018 Track 1 corpus is distributed as paired text and BRAT
standoff annotation files under the DBMI data-use agreement.  OpenMed never
vendors or downloads those records; this module reads only a user-supplied
credentialed directory outside the repository tree.
"""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

from openmed.core.labels import (
    CANONICAL_LABELS,
    normalize_label,
)
from openmed.eval.datasets.dua_stubs import DUACredentialRequired
from openmed.eval.datasets.i2b2 import I2B2_PHI_TAG_TO_CANONICAL
from openmed.eval.harness import BenchmarkFixture
from openmed.eval.metrics import EvalSpan

N2C2 = "n2c2"
N2C2_2018 = "n2c2-2018"
N2C2_DUA_NAME = "n2c2/DBMI DUA"
N2C2_PATH_ENV = "OPENMED_N2C2_PATH"
N2C2_TRACK = "track_1"
N2C2_YEAR = 2018

# n2c2 2018 uses the flat PHI category names below, rather than the nested
# CATEGORY/TYPE names used by the i2b2 XML release.  The source categories are
# retained in fixture metadata while their scoring labels are canonicalized.
N2C2_PHI_TAGS: tuple[str, ...] = (
    "AGE",
    "DATE",
    "NAME",
    "PROFESSION",
    "LOCATION",
    "ORGANIZATION",
    "PHONE",
    "FAX",
    "EMAIL",
    "URL",
    "IPADDRESS",
    "MEDICALRECORD",
    "HEALTHPLAN",
    "ACCOUNT",
    "LICENSE",
    "VEHICLE",
    "DEVICE",
    "BIOID",
    "IDNUM",
)

N2C2_PHI_TAG_TO_CANONICAL: Mapping[str, str] = {
    "AGE": I2B2_PHI_TAG_TO_CANONICAL["AGE"],
    "DATE": I2B2_PHI_TAG_TO_CANONICAL["DATE"],
    "NAME": I2B2_PHI_TAG_TO_CANONICAL["NAME/PATIENT"],
    "PROFESSION": I2B2_PHI_TAG_TO_CANONICAL["PROFESSION"],
    "LOCATION": I2B2_PHI_TAG_TO_CANONICAL["LOCATION/LOCATION_OTHER"],
    "ORGANIZATION": I2B2_PHI_TAG_TO_CANONICAL["LOCATION/ORGANIZATION"],
    "PHONE": I2B2_PHI_TAG_TO_CANONICAL["CONTACT/PHONE"],
    "FAX": I2B2_PHI_TAG_TO_CANONICAL["CONTACT/FAX"],
    "EMAIL": I2B2_PHI_TAG_TO_CANONICAL["CONTACT/EMAIL"],
    "URL": I2B2_PHI_TAG_TO_CANONICAL["CONTACT/URL"],
    "IPADDRESS": I2B2_PHI_TAG_TO_CANONICAL["CONTACT/IPADDRESS"],
    "MEDICALRECORD": I2B2_PHI_TAG_TO_CANONICAL["ID/MEDICALRECORD"],
    "HEALTHPLAN": I2B2_PHI_TAG_TO_CANONICAL["ID/HEALTHPLAN"],
    "ACCOUNT": I2B2_PHI_TAG_TO_CANONICAL["ID/ACCOUNT"],
    "LICENSE": I2B2_PHI_TAG_TO_CANONICAL["ID/LICENSE"],
    "VEHICLE": I2B2_PHI_TAG_TO_CANONICAL["ID/VEHICLE"],
    "DEVICE": I2B2_PHI_TAG_TO_CANONICAL["ID/DEVICE"],
    "BIOID": I2B2_PHI_TAG_TO_CANONICAL["ID/BIOID"],
    "IDNUM": I2B2_PHI_TAG_TO_CANONICAL["ID/IDNUM"],
}

# These flat source labels are the n2c2-specific part of the lineage shared
# with i2b2.  The complete source category table is also recorded in suite
# metadata so reports preserve the distinction between source and canonical
# labels.
N2C2_SPECIFIC_PHI_TAGS: tuple[str, ...] = (
    "NAME",
    "LOCATION",
    "ORGANIZATION",
    "PHONE",
    "FAX",
    "EMAIL",
    "URL",
    "IPADDRESS",
    "MEDICALRECORD",
    "HEALTHPLAN",
    "ACCOUNT",
    "LICENSE",
    "VEHICLE",
    "DEVICE",
    "BIOID",
    "IDNUM",
)

N2C2_SUITE_METADATA: Mapping[str, Any] = {
    "access": (
        "requires an approved local n2c2/DBMI DUA credentialed directory; "
        f"pass path=... or set {N2C2_PATH_ENV}"
    ),
    "annotation_format": "BRAT standoff (.txt + .ann)",
    "dua": N2C2_DUA_NAME,
    "label_mapping": dict(sorted(N2C2_PHI_TAG_TO_CANONICAL.items())),
    "n2c2_categories": N2C2_PHI_TAGS,
    "n2c2_specific_categories": N2C2_SPECIFIC_PHI_TAGS,
    "redistribution": "not vendored; eval-only local credentialed directory",
    "suite": N2C2,
    "track": N2C2_TRACK,
    "year": N2C2_YEAR,
}

_REPO_ROOT = Path(__file__).resolve().parents[3]
_BRAT_TEXT_SUFFIX = ".txt"
_BRAT_ANNOTATION_SUFFIX = ".ann"


class N2C2CredentialRequired(DUACredentialRequired):
    """Raised when n2c2 loading lacks approved local DUA access."""


def load_n2c2_2018_deid(
    path: str | Path | None = None,
) -> list[BenchmarkFixture]:
    """Load n2c2 2018 Track 1 BRAT documents from a credentialed directory.

    Args:
        path: Approved local directory containing paired ``.txt`` and ``.ann``
            files. If omitted, ``OPENMED_N2C2_PATH`` is used.

    Returns:
        Benchmark fixtures with canonical-label gold spans.

    Raises:
        N2C2CredentialRequired: If no approved local path is configured, the
            path is empty, it does not contain paired documents, or it points
            inside this repository.
        ValueError: If a standoff record is malformed or uses an unknown PHI
            category.
    """

    source_root = _credentialed_path(path)
    pairs = tuple(_iter_document_pairs(source_root))
    if not pairs:
        raise N2C2CredentialRequired(
            f"{N2C2_DUA_NAME} credentialed path contains no paired n2c2 "
            f"{_BRAT_TEXT_SUFFIX}/{_BRAT_ANNOTATION_SUFFIX} documents: "
            f"{source_root}"
        )

    fixtures = [
        _fixture_from_pair(text_path, ann_path, root=source_root)
        for text_path, ann_path in pairs
    ]
    _validate_unique_fixture_ids(fixtures)
    return fixtures


def n2c2_suite_metadata() -> dict[str, Any]:
    """Return n2c2 suite metadata without reading local corpus data."""

    return dict(N2C2_SUITE_METADATA)


def map_n2c2_phi_tag(label: str) -> str:
    """Map an n2c2 PHI category onto an OpenMed canonical label."""

    source_tag = _canonical_source_tag(label)
    canonical = N2C2_PHI_TAG_TO_CANONICAL.get(source_tag)
    if canonical is None:
        allowed = ", ".join(N2C2_PHI_TAGS)
        raise ValueError(f"unknown n2c2 PHI tag {label!r}; expected one of: {allowed}")
    normalized = normalize_label(canonical)
    if normalized not in CANONICAL_LABELS:
        raise RuntimeError(
            f"n2c2 mapping for {source_tag!r} is not canonical: {canonical!r}"
        )
    return normalized


def _fixture_from_pair(
    text_path: Path,
    ann_path: Path,
    *,
    root: Path,
) -> BenchmarkFixture:
    _refuse_repository_path(text_path)
    _refuse_repository_path(ann_path)
    text = _read_exact(text_path)
    annotations = _read_exact(ann_path)
    source_hash = _source_hash(text_path, root)
    spans = _parse_annotations(
        annotations,
        text=text,
        source_file=ann_path.name,
    )
    return BenchmarkFixture(
        fixture_id=f"{N2C2_2018}-{source_hash}",
        text=text,
        gold_spans=spans,
        language="en",
        metadata={
            "annotation_format": "brat",
            "dua": N2C2_DUA_NAME,
            "n2c2_categories": N2C2_PHI_TAGS,
            "n2c2_specific_categories": N2C2_SPECIFIC_PHI_TAGS,
            "redistribution": "not vendored; loaded from credentialed path",
            "source_path_hash": source_hash,
            "suite": N2C2,
            "track": N2C2_TRACK,
            "year": N2C2_YEAR,
        },
    )


def _parse_annotations(
    annotations: str,
    *,
    text: str,
    source_file: str,
) -> tuple[EvalSpan, ...]:
    issues: list[str] = []
    spans: list[EvalSpan] = []
    annotation_ids: set[str] = set()
    seen_spans: set[tuple[int, int, str]] = set()

    for line_number, line in enumerate(annotations.splitlines(), start=1):
        if not line.strip():
            continue
        fields = line.split("\t", 2)
        annotation_id = fields[0].strip() if fields else ""
        location = f"{source_file}:{line_number}"
        if len(fields) != 3:
            issues.append(f"{location}: expected ID, label offsets, and text")
            continue
        if not annotation_id.startswith("T") or not annotation_id[1:].isdigit():
            issues.append(f"{location}: unsupported annotation ID {annotation_id!r}")
            continue
        if annotation_id in annotation_ids:
            issues.append(f"{location}: duplicate annotation ID {annotation_id}")
            continue
        annotation_ids.add(annotation_id)

        descriptor = fields[1].strip()
        if ";" in descriptor:
            issues.append(f"{location}: discontinuous spans are not supported")
            continue
        parts = descriptor.split()
        if len(parts) != 3:
            issues.append(f"{location}: expected 'LABEL START END' in the second field")
            continue
        label, raw_start, raw_end = parts
        try:
            start, end = int(raw_start), int(raw_end)
        except ValueError:
            issues.append(f"{location}: start and end offsets must be integers")
            continue
        if start < 0 or end > len(text) or start >= end:
            issues.append(
                f"{location}: offsets {start}:{end} must satisfy "
                f"0 <= start < end <= {len(text)}"
            )
            continue
        expected_surface = _brat_surface(text[start:end])
        if fields[2] != expected_surface:
            issues.append(
                f"{location}: annotated text does not match offsets {start}:{end}"
            )
            continue

        try:
            canonical_label = map_n2c2_phi_tag(label)
        except ValueError as exc:
            issues.append(f"{location}: {exc}")
            continue
        identity = (start, end, canonical_label)
        if identity in seen_spans:
            issues.append(
                f"{location}: duplicate span {start}:{end} and label {canonical_label}"
            )
            continue
        seen_spans.add(identity)
        source_tag = _canonical_source_tag(label)
        spans.append(
            EvalSpan(
                start=start,
                end=end,
                label=canonical_label,
                text=text[start:end],
                language="en",
                metadata={
                    "annotation_format": "brat",
                    "canonical_label": canonical_label,
                    "n2c2_category": source_tag,
                    "n2c2_tag": source_tag,
                    "span_id": annotation_id,
                },
            )
        )

    if issues:
        details = "\n".join(f"- {issue}" for issue in issues)
        raise ValueError(f"Invalid n2c2 BRAT standoff data:\n{details}")
    return tuple(sorted(spans, key=lambda span: (span.start, span.end, span.label)))


def _credentialed_path(path: str | Path | None) -> Path:
    raw_path = path if path is not None else os.environ.get(N2C2_PATH_ENV)
    if raw_path is None or str(raw_path).strip() == "":
        raise N2C2CredentialRequired(
            f"{N2C2_DUA_NAME} credentialed local path is required; pass path=... "
            f"or set {N2C2_PATH_ENV}. No n2c2 data is bundled."
        )

    candidate = Path(raw_path).expanduser().resolve(strict=False)
    if _is_relative_to(candidate, _REPO_ROOT):
        raise N2C2CredentialRequired(
            f"{N2C2_DUA_NAME} data must be kept outside the repository tree; "
            f"refusing to read {candidate}"
        )
    if not candidate.exists():
        raise N2C2CredentialRequired(
            f"{N2C2_DUA_NAME} credentialed path does not exist: {candidate}"
        )
    if not candidate.is_dir() and candidate.suffix.lower() != _BRAT_TEXT_SUFFIX:
        raise N2C2CredentialRequired(
            f"{N2C2_DUA_NAME} credentialed path must be a directory or .txt file: "
            f"{candidate}"
        )
    return candidate


def _iter_document_pairs(root: Path) -> Iterable[tuple[Path, Path]]:
    if root.is_file():
        text_path = root
        ann_path = text_path.with_suffix(_BRAT_ANNOTATION_SUFFIX)
        if not ann_path.is_file():
            return ()
        return ((text_path, ann_path),)

    pairs: list[tuple[Path, Path]] = []
    for ann_path in sorted(root.rglob(f"*{_BRAT_ANNOTATION_SUFFIX}")):
        if not ann_path.is_file():
            continue
        text_path = ann_path.with_suffix(_BRAT_TEXT_SUFFIX)
        if text_path.is_file():
            pairs.append((text_path, ann_path))
    return tuple(pairs)


def _source_hash(path: Path, root: Path) -> str:
    relative = path.name if root.is_file() else path.relative_to(root).as_posix()
    return hashlib.sha256(relative.encode("utf-8")).hexdigest()[:16]


def _read_exact(path: Path) -> str:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return handle.read()


def _canonical_source_tag(label: str) -> str:
    normalized = _normalize_token(label)
    aliases = {
        "IP_ADDRESS": "IPADDRESS",
        "IPADDRESS": "IPADDRESS",
        "MEDICAL_RECORD": "MEDICALRECORD",
        "MEDICAL_RECORD_NUMBER": "MEDICALRECORD",
        "MEDICALRECORD": "MEDICALRECORD",
        "MRN": "MEDICALRECORD",
        "HEALTH_PLAN": "HEALTHPLAN",
        "HEALTH_PLAN_NUMBER": "HEALTHPLAN",
        "HEALTHPLAN": "HEALTHPLAN",
        "ACCOUNT_NUMBER": "ACCOUNT",
        "LICENSE_NUMBER": "LICENSE",
        "VEHICLE_ID": "VEHICLE",
        "DEVICE_ID": "DEVICE",
        "BIOMETRIC_ID": "BIOID",
        "ID": "IDNUM",
        "ID_NUMBER": "IDNUM",
        "PATIENT": "NAME",
        "DOCTOR": "NAME",
        "PERSON": "NAME",
        "NAME_PATIENT": "NAME",
        "NAME_DOCTOR": "NAME",
        "LOCATION_OTHER": "LOCATION",
        "HOSPITAL": "ORGANIZATION",
    }
    return aliases.get(normalized, normalized)


def _normalize_token(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", value.strip()).strip("_").upper()


def _brat_surface(value: str) -> str:
    return value.replace("\r", " ").replace("\n", " ")


def _validate_unique_fixture_ids(fixtures: Iterable[BenchmarkFixture]) -> None:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for fixture in fixtures:
        if fixture.fixture_id in seen:
            duplicates.add(fixture.fixture_id)
        seen.add(fixture.fixture_id)
    if duplicates:
        joined = ", ".join(sorted(duplicates))
        raise ValueError(f"duplicate n2c2 benchmark fixture id(s): {joined}")


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _refuse_repository_path(path: Path) -> None:
    resolved = path.resolve(strict=False)
    if _is_relative_to(resolved, _REPO_ROOT):
        raise N2C2CredentialRequired(
            f"{N2C2_DUA_NAME} data must be kept outside the repository tree; "
            f"refusing to read {resolved}"
        )


_missing_mappings = sorted(set(N2C2_PHI_TAGS) - set(N2C2_PHI_TAG_TO_CANONICAL))
_extra_mappings = sorted(set(N2C2_PHI_TAG_TO_CANONICAL) - set(N2C2_PHI_TAGS))
_invalid_mappings = {
    tag: canonical
    for tag, canonical in N2C2_PHI_TAG_TO_CANONICAL.items()
    if normalize_label(canonical) not in CANONICAL_LABELS
}
if _missing_mappings or _extra_mappings or _invalid_mappings:
    raise RuntimeError(
        "n2c2 PHI mapping must cover the source tag table exactly; "
        f"missing={_missing_mappings}, extra={_extra_mappings}, "
        f"invalid={_invalid_mappings}"
    )


__all__ = [
    "N2C2",
    "N2C2_2018",
    "N2C2_DUA_NAME",
    "N2C2_PATH_ENV",
    "N2C2_PHI_TAGS",
    "N2C2_PHI_TAG_TO_CANONICAL",
    "N2C2_SPECIFIC_PHI_TAGS",
    "N2C2_SUITE_METADATA",
    "N2C2_TRACK",
    "N2C2_YEAR",
    "N2C2CredentialRequired",
    "load_n2c2_2018_deid",
    "map_n2c2_phi_tag",
    "n2c2_suite_metadata",
]
