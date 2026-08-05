"""Deterministic synthetic section and document-type training data.

This module writes local-only JSONL records for the DocType/Section family.
The note text is generated from repository-owned templates, section boundaries
come from the canonical clinical section detector, and the sidecar manifest
contains hashes and provenance rather than copied evaluation text.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any, Final

from openmed.clinical.lexicons import load_section_headers
from openmed.clinical.sections import classify_document, detect_sections
from openmed.training.corpus import (
    CorpusManifestError,
    jsonl_records_hash,
    normalize_passage_text,
    validate_corpus_record,
    write_jsonl_records,
)

SECTION_RECORD_SCHEMA_VERSION: Final = "openmed.training.synthetic_section.v1"
SECTION_MANIFEST_SCHEMA_VERSION: Final = (
    "openmed.training.synthetic_section_manifest.v1"
)
SYNTHETIC_SECTION_SOURCE: Final = "openmed.synthetic.section_labels"
SYNTHETIC_SECTION_LICENSE: Final = "Apache-2.0"
DEFAULT_SECTION_DATASET_PATH: Final = Path("section_labels.jsonl")
DEFAULT_SECTION_EVAL_FIXTURE: Final = (
    Path(__file__).resolve().parents[2]
    / "eval"
    / "fixtures"
    / "section_multilingual.jsonl"
)

_TOKEN_RE = re.compile(
    r"\w+(?:['\N{RIGHT SINGLE QUOTATION MARK}-]\w+)*|[^\w\s]", re.UNICODE
)
_DOC_TYPE_RESOURCE = "data/doctype_signatures.json"
_DOC_TYPE_TITLES: Final[dict[str, str]] = {
    "discharge_summary": "DISCHARGE SUMMARY",
    "progress_note": "PROGRESS NOTE",
    "radiology_report": "RADIOLOGY REPORT",
    "pathology_report": "PATHOLOGY REPORT",
    "operative_note": "OPERATIVE NOTE",
    "consult_note": "CONSULTATION NOTE",
}
_SECTION_CONTENT: Final[dict[str, tuple[str, ...]]] = {
    "allergies": (
        "Synthetic allergy status is documented for label training.",
        "The example records no known allergy in a fictional note.",
    ),
    "assessment": (
        "Synthetic assessment language is included for classification practice.",
        "This fictional assessment remains an assistive training example.",
    ),
    "assessment_and_plan": (
        "Synthetic review is stable with a routine demonstration follow-up.",
        "The fictional plan records monitoring language without a clinical action.",
    ),
    "chief_complaint": (
        "A fictional visit reason is included as a section-boundary example.",
        "Synthetic concern wording exercises the chief-complaint header.",
    ),
    "family_history": (
        "Synthetic family context is included without a real person or event.",
        "The fictional family history contains generic demonstration text.",
    ),
    "findings": (
        "Synthetic findings describe a generic observation for training.",
        "The fictional report contains a non-diagnostic demonstration finding.",
    ),
    "history": (
        "Synthetic history text provides a short generic context window.",
        "This fictional history section is present only for boundary coverage.",
    ),
    "history_of_present_illness": (
        "Synthetic symptom wording supports section-boundary training.",
        "A fictional presenting concern is described in generic terms.",
    ),
    "impression": (
        "Synthetic impression language is intentionally non-clinical advice.",
        "The fictional impression is a label-training placeholder.",
    ),
    "medications": (
        "Synthetic medication entries use generic demonstration names only.",
        "The fictional medication list contains no prescription or patient data.",
    ),
    "past_medical_history": (
        "Synthetic prior-history wording is included for section coverage.",
        "The fictional past history contains generic, non-identifying content.",
    ),
    "plan": (
        "Synthetic plan wording indicates routine follow-up for the example.",
        "The fictional plan is a non-prescriptive placeholder for training.",
    ),
    "problem_list": (
        "Synthetic problem entries are generic labels without patient details.",
        "The fictional problem list exercises a list-bearing section header.",
    ),
    "review_of_systems": (
        "Synthetic review text covers a generic system statement.",
        "The fictional review of systems is included for boundary coverage.",
    ),
    "social_history": (
        "Synthetic social context uses a generic fictional living arrangement.",
        "The example has no real social or geographic details.",
    ),
}
_PREFIXES: Final[tuple[str, ...]] = (
    "Generated training example {index} from seed {seed}.",
    "Offline synthetic note sample {index}, generation seed {seed}.",
)
_SECTION_HEADERS = load_section_headers()


def _load_document_types() -> tuple[str, ...]:
    resource = resources.files("openmed.clinical").joinpath(_DOC_TYPE_RESOURCE)
    payload = json.loads(resource.read_text(encoding="utf-8"))
    raw_types = payload.get("document_types") if isinstance(payload, Mapping) else None
    if not isinstance(raw_types, list):
        raise RuntimeError("synthetic document-type signatures are malformed")
    labels = tuple(
        item.get("type")
        for item in raw_types
        if isinstance(item, Mapping) and isinstance(item.get("type"), str)
    )
    if not labels or any(label not in _DOC_TYPE_TITLES for label in labels):
        raise RuntimeError("synthetic document-type signatures are incomplete")
    return labels


SECTION_LABELS: Final[tuple[str, ...]] = tuple(sorted(_SECTION_HEADERS))
CANONICAL_SECTION_LABELS: Final[frozenset[str]] = frozenset(SECTION_LABELS)
DOCUMENT_TYPES: Final[tuple[str, ...]] = _load_document_types()
CANONICAL_DOCUMENT_TYPES: Final[frozenset[str]] = frozenset(DOCUMENT_TYPES)


class SectionDatasetLeakageError(ValueError):
    """Raised when generated text overlaps a held-out synthetic fixture."""


@dataclass(frozen=True)
class SectionDatasetBuildResult:
    """Paths, hashes, and in-memory rows returned by a dataset build."""

    dataset_path: Path
    manifest_path: Path
    dataset_hash: str
    manifest_hash: str
    records: tuple[Mapping[str, Any], ...]
    leakage_count: int

    @property
    def path(self) -> Path:
        """Return the generated JSONL path."""

        return self.dataset_path

    @property
    def record_count(self) -> int:
        """Return the number of generated records."""

        return len(self.records)

    def __fspath__(self) -> str:
        """Allow the result to be passed to path-oriented APIs."""

        return str(self.dataset_path)


def build_section_dataset(
    seed: int,
    n: int,
    output_path: str | Path = DEFAULT_SECTION_DATASET_PATH,
    *,
    eval_fixture_path: str | Path = DEFAULT_SECTION_EVAL_FIXTURE,
    section_labels: Sequence[str] | None = None,
) -> SectionDatasetBuildResult:
    """Build and write a deterministic synthetic section-label dataset.

    Args:
        seed: Local pseudo-random seed. No global random state is changed.
        n: Number of JSONL records to emit. Zero emits an empty valid dataset.
        output_path: Destination JSONL path. A ``.manifest.json`` sidecar is
            written beside it.
        eval_fixture_path: Synthetic held-out section fixture used by the
            leakage guard.
        section_labels: Optional restricted canonical label subset. The
            default covers every label in the packaged section lexicon.

    Returns:
        A result containing output paths, hashes, and the generated records.

    Raises:
        SectionDatasetLeakageError: If generated text overlaps held-out text.
        ValueError: If seed, count, or requested labels are invalid.
    """

    _validate_seed(seed)
    _validate_count(n)
    selected_labels = _normalize_section_labels(section_labels)
    output = Path(output_path)
    fixture = Path(eval_fixture_path)
    rng = random.Random(seed)
    records = tuple(
        _build_record(
            seed=seed,
            index=index,
            rng=rng,
            section_labels=selected_labels,
        )
        for index in range(n)
    )
    for index, record in enumerate(records):
        validate_section_record(record, context=f"generated record {index}")

    leakage_count = assert_no_eval_overlap(records, eval_fixture_path=fixture)
    write_jsonl_records(records, output)
    dataset_hash = jsonl_records_hash(records)
    manifest = _build_manifest(
        seed=seed,
        record_count=n,
        dataset_hash=dataset_hash,
        leakage_count=leakage_count,
        eval_fixture_path=fixture,
        section_labels=selected_labels,
    )
    manifest_hash = _payload_hash(manifest)
    manifest["manifest_hash"] = manifest_hash
    manifest_path = output.with_suffix(".manifest.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(_canonical_json(manifest) + "\n", encoding="utf-8")
    return SectionDatasetBuildResult(
        dataset_path=output,
        manifest_path=manifest_path,
        dataset_hash=dataset_hash,
        manifest_hash=manifest_hash,
        records=records,
        leakage_count=leakage_count,
    )


def validate_section_labels(labels: Iterable[str]) -> tuple[str, ...]:
    """Validate and return unique canonical section labels in input order."""

    if isinstance(labels, (str, bytes)):
        raise TypeError("section labels must be an iterable of strings")
    normalized = tuple(labels)
    if any(not isinstance(label, str) or not label.strip() for label in normalized):
        raise ValueError("section labels must be non-empty strings")
    unknown = sorted(set(normalized).difference(CANONICAL_SECTION_LABELS))
    if unknown:
        raise ValueError("unsupported section label(s): " + ", ".join(unknown))
    if len(set(normalized)) != len(normalized):
        raise ValueError("section labels must not contain duplicates")
    if not normalized:
        raise ValueError("section labels must not be empty")
    return normalized


def validate_section_record(
    record: Mapping[str, Any], *, context: str = "section record"
) -> None:
    """Validate a generated record's corpus, BIO, and provenance contract."""

    try:
        validate_corpus_record(record, context=context)
    except CorpusManifestError:
        raise
    schema_version = record.get("schema_version")
    if schema_version != SECTION_RECORD_SCHEMA_VERSION:
        raise ValueError(
            f"{context} schema_version must be {SECTION_RECORD_SCHEMA_VERSION!r}"
        )
    if record.get("source") != SYNTHETIC_SECTION_SOURCE:
        raise ValueError(f"{context} source must identify the synthetic builder")
    if record.get("synthetic") is not True:
        raise ValueError(f"{context} must declare synthetic=true")
    if record.get("contains_real_phi") is not False:
        raise ValueError(f"{context} must declare contains_real_phi=false")
    if record.get("restricted_data") is not False:
        raise ValueError(f"{context} must declare restricted_data=false")

    doc_type = record.get("doc_type")
    if doc_type not in CANONICAL_DOCUMENT_TYPES:
        raise ValueError(f"{context} has unsupported document type {doc_type!r}")
    tokens = _sequence_field(record, "tokens", context)
    tags = _sequence_field(record, "section_tags", context)
    labels = _sequence_field(record, "labels", context)
    if tuple(tags) != tuple(labels):
        raise ValueError(f"{context} labels and section_tags must match")
    if len(tokens) != len(tags):
        raise ValueError(f"{context} tokens and section_tags must have equal length")
    if any(not isinstance(token, str) or not token for token in tokens):
        raise ValueError(f"{context} tokens must be non-empty strings")
    _validate_bio_tags(tags, context=context)

    offsets = _sequence_field(record, "token_offsets", context)
    if len(offsets) != len(tokens):
        raise ValueError(f"{context} token_offsets must match tokens")
    text = record["text"]
    for index, offset in enumerate(offsets):
        if (
            not isinstance(offset, Sequence)
            or isinstance(offset, (str, bytes))
            or len(offset) != 2
            or not all(
                isinstance(value, int) and not isinstance(value, bool)
                for value in offset
            )
        ):
            raise ValueError(f"{context} token_offsets[{index}] is invalid")
        start, end = offset
        if not 0 <= start < end <= len(text) or text[start:end] != tokens[index]:
            raise ValueError(f"{context} token_offsets[{index}] does not match text")

    sections = _sequence_field(record, "sections", context)
    previous_end = 0
    for index, section in enumerate(sections):
        if not isinstance(section, Mapping):
            raise ValueError(f"{context} sections[{index}] must be a mapping")
        label = section.get("label")
        validate_section_labels((label,))
        start = section.get("start")
        end = section.get("end")
        if (
            not isinstance(start, int)
            or isinstance(start, bool)
            or not isinstance(end, int)
            or isinstance(end, bool)
            or not 0 <= start < end <= len(text)
            or start < previous_end
        ):
            raise ValueError(f"{context} sections[{index}] has invalid bounds")
        previous_end = end


def assert_no_eval_overlap(
    records: Iterable[Mapping[str, Any]],
    *,
    eval_fixture_path: str | Path = DEFAULT_SECTION_EVAL_FIXTURE,
    eval_texts: Iterable[str] | None = None,
) -> int:
    """Assert that generated text has no normalized overlap with eval text.

    Exact normalized matches are rejected, as are containment matches for
    substantial fixture text. Errors identify record and fixture indices only;
    they never include clinical text.
    """

    fixture_values = (
        tuple(eval_texts)
        if eval_texts is not None
        else _load_eval_fixture_texts(Path(eval_fixture_path))
    )
    normalized_fixtures = tuple(
        normalize_passage_text(text) for text in fixture_values if str(text).strip()
    )
    overlaps: list[tuple[int, int]] = []
    for record_index, record in enumerate(records):
        text = record.get("text")
        if not isinstance(text, str):
            raise ValueError(f"record {record_index} text must be a string")
        normalized = normalize_passage_text(text)
        for fixture_index, fixture_text in enumerate(normalized_fixtures):
            if (
                normalized == fixture_text
                or (len(fixture_text) >= 16 and fixture_text in normalized)
                or (len(normalized) >= 16 and normalized in fixture_text)
            ):
                overlaps.append((record_index, fixture_index))
    if overlaps:
        details = ", ".join(
            f"record {record_index}/fixture {fixture_index}"
            for record_index, fixture_index in overlaps
        )
        raise SectionDatasetLeakageError(
            "synthetic section dataset overlaps held-out eval text at " + details
        )
    return 0


def load_section_dataset(path: str | Path) -> tuple[Mapping[str, Any], ...]:
    """Load and validate section JSONL records from a local path."""

    rows: list[Mapping[str, Any]] = []
    for line_number, line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path} line {line_number} is not valid JSON") from exc
        if not isinstance(record, Mapping):
            raise ValueError(f"{path} line {line_number} must be an object")
        validate_section_record(record, context=f"{path} line {line_number}")
        rows.append(dict(record))
    return tuple(rows)


def load_section_manifest(path: str | Path) -> Mapping[str, Any]:
    """Load a generated sidecar manifest without loading dataset text."""

    manifest_path = Path(path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("section dataset manifest must be an object")
    if payload.get("schema_version") != SECTION_MANIFEST_SCHEMA_VERSION:
        raise ValueError("section dataset manifest has an unsupported schema version")
    leakage_check = payload.get("leakage_check")
    if (
        not isinstance(leakage_check, Mapping)
        or leakage_check.get("overlap_count") != 0
    ):
        raise SectionDatasetLeakageError("section dataset manifest reports leakage")
    return dict(payload)


def _build_record(
    *,
    seed: int,
    index: int,
    rng: random.Random,
    section_labels: Sequence[str],
) -> dict[str, Any]:
    doc_type = DOCUMENT_TYPES[index % len(DOCUMENT_TYPES)]
    ordered_labels = list(section_labels)
    rng.shuffle(ordered_labels)
    lines = [_PREFIXES[rng.randrange(len(_PREFIXES))].format(seed=seed, index=index)]
    lines.append(_DOC_TYPE_TITLES[doc_type])
    for label in ordered_labels:
        headers = _SECTION_HEADERS[label]
        header = headers[rng.randrange(len(headers))]
        content = _SECTION_CONTENT[label][rng.randrange(len(_SECTION_CONTENT[label]))]
        lines.append(f"{header}: {content}")
    text = "\n".join(lines)
    detected = tuple(detect_sections(text, language="en"))
    section_spans = tuple(span for span in detected if span["label"] != "unsectioned")
    observed_labels = tuple(span["label"] for span in section_spans)
    if set(observed_labels) != set(section_labels):
        raise RuntimeError("synthetic note did not cover its requested section labels")
    if any(label not in CANONICAL_SECTION_LABELS for label in observed_labels):
        raise ValueError("synthetic section detector emitted an invalid label")
    if classify_document(text)["type"] != doc_type:
        raise RuntimeError("synthetic note did not match its document-type label")

    tokens_with_offsets = tuple(
        (match.group(0), match.start(), match.end())
        for match in _TOKEN_RE.finditer(text)
    )
    tags = _bio_tags(tokens_with_offsets, section_spans)
    record_id = f"synthetic-section-{seed}-{index:04d}"
    return {
        "contains_real_phi": False,
        "doc_type": doc_type,
        "id": record_id,
        "labels": list(tags),
        "license": SYNTHETIC_SECTION_LICENSE,
        "metadata": {
            "contains_real_phi": False,
            "language": "en",
            "restricted_data": False,
            "synthetic": True,
            "synthetic_source": SYNTHETIC_SECTION_SOURCE,
        },
        "record_id": record_id,
        "restricted_data": False,
        "schema_version": SECTION_RECORD_SCHEMA_VERSION,
        "sections": [
            {
                "end": int(span["end"]),
                "header": str(span.get("header", "")),
                "label": str(span["label"]),
                "start": int(span["start"]),
            }
            for span in section_spans
        ],
        "section_tags": list(tags),
        "source": SYNTHETIC_SECTION_SOURCE,
        "synthetic": True,
        "text": text,
        "token_offsets": [[start, end] for _, start, end in tokens_with_offsets],
        "tokens": [token for token, _, _ in tokens_with_offsets],
    }


def _bio_tags(
    tokens: Sequence[tuple[str, int, int]], sections: Sequence[Mapping[str, Any]]
) -> tuple[str, ...]:
    tags: list[str] = []
    previous_section_index: int | None = None
    for _, token_start, token_end in tokens:
        section_index = next(
            (
                index
                for index, section in enumerate(sections)
                if token_start < int(section["end"])
                and token_end > int(section["start"])
            ),
            None,
        )
        if section_index is None:
            tags.append("O")
            previous_section_index = None
            continue
        label = str(sections[section_index]["label"])
        prefix = "I" if section_index == previous_section_index else "B"
        tags.append(f"{prefix}-{label}")
        previous_section_index = section_index
    return tuple(tags)


def _validate_bio_tags(tags: Sequence[Any], *, context: str) -> None:
    active_label: str | None = None
    for index, raw_tag in enumerate(tags):
        if not isinstance(raw_tag, str):
            raise ValueError(f"{context} tag {index} must be a string")
        if raw_tag == "O":
            active_label = None
            continue
        prefix, separator, label = raw_tag.partition("-")
        if separator != "-" or prefix not in {"B", "I"}:
            raise ValueError(f"{context} tag {index} is not BIO-formatted")
        validate_section_labels((label,))
        if prefix == "I" and active_label != label:
            raise ValueError(f"{context} tag {index} has no matching BIO span")
        active_label = label


def _normalize_section_labels(
    section_labels: Sequence[str] | None,
) -> tuple[str, ...]:
    return validate_section_labels(
        SECTION_LABELS if section_labels is None else section_labels
    )


def _sequence_field(
    record: Mapping[str, Any], field_name: str, context: str
) -> Sequence[Any]:
    value = record.get(field_name)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{context} {field_name} must be a sequence")
    return value


def _build_manifest(
    *,
    seed: int,
    record_count: int,
    dataset_hash: str,
    leakage_count: int,
    eval_fixture_path: Path,
    section_labels: Sequence[str],
) -> dict[str, Any]:
    return {
        "canonical_document_types": list(DOCUMENT_TYPES),
        "canonical_section_labels": list(sorted(section_labels)),
        "dataset_hash": dataset_hash,
        "dataset_source": SYNTHETIC_SECTION_SOURCE,
        "license": SYNTHETIC_SECTION_LICENSE,
        "record_count": record_count,
        "record_schema_version": SECTION_RECORD_SCHEMA_VERSION,
        "restricted_data": False,
        "schema_version": SECTION_MANIFEST_SCHEMA_VERSION,
        "seed": seed,
        "synthetic": True,
        "contains_real_phi": False,
        "eval_fixture": {
            "path": _stable_fixture_name(eval_fixture_path),
            "sha256": _file_hash(eval_fixture_path),
        },
        "leakage_check": {
            "fixture": _stable_fixture_name(eval_fixture_path),
            "overlap_count": leakage_count,
        },
    }


def _load_eval_fixture_texts(path: Path) -> tuple[str, ...]:
    if not path.is_file():
        raise FileNotFoundError(f"synthetic section eval fixture not found: {path}")
    texts: list[str] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, Mapping):
            raise ValueError(f"eval fixture line {line_number} must be an object")
        if payload.get("kind") == "meta":
            continue
        if payload.get("synthetic") is not True:
            raise SectionDatasetLeakageError(
                f"eval fixture line {line_number} is not marked synthetic"
            )
        text = payload.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"eval fixture line {line_number} requires text")
        texts.append(text)
    return tuple(texts)


def _validate_seed(seed: int) -> None:
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed must be an integer")


def _validate_count(n: int) -> None:
    if not isinstance(n, int) or isinstance(n, bool) or n < 0:
        raise ValueError("n must be a non-negative integer")


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(
        dict(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _payload_hash(payload: Mapping[str, Any]) -> str:
    return (
        f"sha256:{hashlib.sha256(_canonical_json(payload).encode('utf-8')).hexdigest()}"
    )


def _file_hash(path: Path) -> str:
    return f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"


def _stable_fixture_name(path: Path) -> str:
    try:
        return (
            path.resolve().relative_to(Path(__file__).resolve().parents[2]).as_posix()
        )
    except ValueError:
        return path.name


__all__ = [
    "CANONICAL_DOCUMENT_TYPES",
    "CANONICAL_SECTION_LABELS",
    "DEFAULT_SECTION_DATASET_PATH",
    "DEFAULT_SECTION_EVAL_FIXTURE",
    "DOCUMENT_TYPES",
    "SECTION_LABELS",
    "SECTION_MANIFEST_SCHEMA_VERSION",
    "SECTION_RECORD_SCHEMA_VERSION",
    "SYNTHETIC_SECTION_LICENSE",
    "SYNTHETIC_SECTION_SOURCE",
    "SectionDatasetBuildResult",
    "SectionDatasetLeakageError",
    "assert_no_eval_overlap",
    "build_section_dataset",
    "load_section_dataset",
    "load_section_manifest",
    "validate_section_labels",
    "validate_section_record",
]
