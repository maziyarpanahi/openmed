"""Hash-only DAPT corpus assembly utilities.

The assembler is intentionally local-first: callers provide public-source
exports from PubMed/PMC, arXiv q-bio, or another compatible source object.
It emits a deterministic JSONL manifest containing hashes and counts, not
passage text.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Protocol, Sequence

DAPT_CORPUS_SCHEMA_VERSION = "openmed.training.dapt_corpus.v1"
DAPT_CORPUS_MANIFEST_PATH = Path(__file__).with_name("configs") / "dapt_corpus.jsonl"
MIMIC_III_DUA_NAME = "PhysioNet MIMIC-III credentialed DUA"

PUBLIC_DAPT_SOURCES = frozenset({"pubmed", "pmc", "arxiv-q-bio"})
GATED_DAPT_SOURCES = frozenset(
    {
        "i2b2",
        "made",
        "mednli",
        "mimic",
        "mimic-iii",
        "mimic-iv",
        "n2c2",
        "shac",
        "thyme",
    }
)
RAW_TEXT_MANIFEST_FIELDS = frozenset(
    {"abstract", "body", "content", "note_text", "passage", "raw_text", "text"}
)
_SHA256_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")


class CorpusManifestError(ValueError):
    """Raised when a corpus passage or manifest row violates the contract."""


class GatedCorpusAccessError(PermissionError):
    """Raised when a DUA-gated source is requested without credentialed data."""


@dataclass(frozen=True)
class Passage:
    """A raw passage supplied by a source adapter before manifest hashing."""

    source: str
    source_id: str
    text: str
    license: str
    provenance: str | None = None
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_non_empty(self.source, "source")
        _require_non_empty(self.source_id, "source_id")
        _require_non_empty(self.text, "text")
        _require_non_empty(self.license, "license")


class PassageSource(Protocol):
    """Minimal pluggable source interface for DAPT passage assembly."""

    name: str
    license: str

    def iter_passages(self) -> Iterable[Passage]:
        """Yield raw passages from this source."""


@dataclass(frozen=True)
class RecordPassageSource:
    """Passage source backed by already-loaded public records."""

    name: str
    license: str
    records: Sequence[Mapping[str, Any]]
    text_field: str = "text"
    id_field: str = "id"
    provenance_field: str | None = None
    metadata_fields: tuple[str, ...] = ()

    def iter_passages(self) -> Iterator[Passage]:
        """Yield passages from records using configured field names."""

        for index, record in enumerate(self.records):
            text = _record_string(record, self.text_field, self.name, index)
            source_id = _record_optional_string(record, self.id_field)
            if source_id is None:
                source_id = f"{self.name}:{index}"
            provenance = (
                _record_optional_string(record, self.provenance_field)
                if self.provenance_field is not None
                else None
            )
            metadata = {
                field_name: str(record[field_name])
                for field_name in self.metadata_fields
                if field_name in record and record[field_name] is not None
            }
            yield Passage(
                source=self.name,
                source_id=source_id,
                text=text,
                license=self.license,
                provenance=provenance,
                metadata=metadata,
            )


@dataclass(frozen=True)
class JsonlPassageSource:
    """Passage source backed by a local JSONL export."""

    path: str | Path
    name: str
    license: str
    text_field: str = "text"
    id_field: str = "id"
    provenance_field: str | None = None
    metadata_fields: tuple[str, ...] = ()

    def iter_passages(self) -> Iterator[Passage]:
        """Yield passages from a JSONL export without network access."""

        source_path = Path(self.path)
        if not source_path.is_file():
            raise FileNotFoundError(f"Passage source not found: {source_path}")
        with source_path.open("r", encoding="utf-8") as handle:
            for index, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    record = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    raise CorpusManifestError(
                        f"{source_path} line {index} is not valid JSON"
                    ) from exc
                if not isinstance(record, Mapping):
                    raise CorpusManifestError(
                        f"{source_path} line {index} must be a JSON object"
                    )
                yield from RecordPassageSource(
                    name=self.name,
                    license=self.license,
                    records=(record,),
                    text_field=self.text_field,
                    id_field=self.id_field,
                    provenance_field=self.provenance_field,
                    metadata_fields=self.metadata_fields,
                ).iter_passages()


@dataclass(frozen=True)
class MimicIIIDuaSource:
    """DUA-gated MIMIC-III adapter that only reads user-supplied local exports."""

    credentialed_path: str | Path | None = None
    name: str = "mimic-iii"
    license: str = MIMIC_III_DUA_NAME
    manifest_filename: str = "passages.jsonl"

    def iter_passages(self) -> Iterator[Passage]:
        """Yield MIMIC-III passages only when a credentialed path is supplied."""

        if self.credentialed_path is None:
            raise GatedCorpusAccessError(
                "MIMIC-III is DUA-gated; pass credentialed_path only after "
                "the caller has local PhysioNet access."
            )
        source_path = Path(self.credentialed_path)
        if source_path.is_dir():
            source_path = source_path / self.manifest_filename
        if not source_path.is_file():
            raise GatedCorpusAccessError(
                "MIMIC-III credentialed_path must point to a local JSONL file "
                f"or a directory containing {self.manifest_filename!r}."
            )
        yield from JsonlPassageSource(
            path=source_path,
            name=self.name,
            license=self.license,
            text_field="text",
            id_field="note_id",
            metadata_fields=("subject_id_hash", "hadm_id_hash"),
        ).iter_passages()


@dataclass(frozen=True)
class DaptCorpusAssemblyResult:
    """Summary returned after writing a DAPT corpus manifest."""

    manifest_path: Path
    corpus_manifest_hash: str
    input_count: int
    passage_count: int
    duplicate_count: int
    dedup_rate: float
    rows: tuple[Mapping[str, Any], ...]


def pubmed_abstract_source(
    records: Iterable[Mapping[str, Any]], *, license: str
) -> RecordPassageSource:
    """Build a PubMed abstract source from caller-supplied records."""

    return RecordPassageSource(
        name="pubmed",
        license=license,
        records=tuple(records),
        text_field="abstract",
        id_field="pmid",
        provenance_field="url",
    )


def pmc_abstract_source(
    records: Iterable[Mapping[str, Any]], *, license: str
) -> RecordPassageSource:
    """Build a PubMed Central abstract source from caller-supplied records."""

    return RecordPassageSource(
        name="pmc",
        license=license,
        records=tuple(records),
        text_field="abstract",
        id_field="pmcid",
        provenance_field="url",
    )


def arxiv_qbio_source(
    records: Iterable[Mapping[str, Any]], *, license: str
) -> RecordPassageSource:
    """Build an arXiv q-bio passage source from caller-supplied records."""

    return RecordPassageSource(
        name="arxiv-q-bio",
        license=license,
        records=tuple(records),
        text_field="abstract",
        id_field="arxiv_id",
        provenance_field="url",
        metadata_fields=("category",),
    )


def assemble_dapt_corpus(
    sources: Iterable[PassageSource],
    output_path: str | Path = DAPT_CORPUS_MANIFEST_PATH,
) -> DaptCorpusAssemblyResult:
    """Assemble, deduplicate, and write a hash-only DAPT manifest.

    Deduplication uses a normalized-text hash, while the manifest row keeps the
    SHA-256 hash of the original passage text supplied by the source.
    """

    output = Path(output_path)
    by_normalized_hash: dict[str, dict[str, Any]] = {}
    input_count = 0

    for source in sources:
        for passage in source.iter_passages():
            input_count += 1
            row = manifest_row_for_passage(passage)
            normalized_hash = row["normalized_sha256"]
            existing = by_normalized_hash.get(normalized_hash)
            if existing is None or _row_sort_key(row) < _row_sort_key(existing):
                by_normalized_hash[normalized_hash] = row

    rows = tuple(sorted(by_normalized_hash.values(), key=_row_sort_key))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
            handle.write("\n")

    duplicate_count = input_count - len(rows)
    return DaptCorpusAssemblyResult(
        manifest_path=output,
        corpus_manifest_hash=corpus_manifest_hash(rows),
        input_count=input_count,
        passage_count=len(rows),
        duplicate_count=duplicate_count,
        dedup_rate=(duplicate_count / input_count if input_count else 0.0),
        rows=rows,
    )


def manifest_row_for_passage(passage: Passage) -> dict[str, Any]:
    """Return the hash-only JSON-serializable manifest row for one passage."""

    normalized_text = normalize_passage_text(passage.text)
    if not normalized_text:
        raise CorpusManifestError("passage text must contain at least one token")

    row: dict[str, Any] = {
        "license": passage.license,
        "normalized_sha256": _sha256(normalized_text),
        "schema_version": DAPT_CORPUS_SCHEMA_VERSION,
        "sha256": _sha256(passage.text),
        "source": passage.source,
        "source_id": passage.source_id,
        "token_count": token_count(passage.text),
    }
    if passage.provenance is not None:
        row["provenance"] = passage.provenance
    if passage.metadata:
        row["metadata"] = {
            str(key): str(value) for key, value in sorted(passage.metadata.items())
        }
    _validate_manifest_row(row)
    return row


def load_corpus_manifest(
    path: str | Path = DAPT_CORPUS_MANIFEST_PATH,
) -> tuple[Mapping[str, Any], ...]:
    """Load and validate a DAPT corpus manifest JSONL file."""

    manifest_path = Path(path)
    rows: list[Mapping[str, Any]] = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise CorpusManifestError(
                    f"{manifest_path} line {line_number} is not valid JSON"
                ) from exc
            if not isinstance(row, Mapping):
                raise CorpusManifestError(
                    f"{manifest_path} line {line_number} must be a JSON object"
                )
            _validate_manifest_row(row, context=f"{manifest_path} line {line_number}")
            rows.append(dict(row))
    return tuple(rows)


def corpus_manifest_hash(
    manifest: str | Path | Iterable[Mapping[str, Any]] = DAPT_CORPUS_MANIFEST_PATH,
) -> str:
    """Return the deterministic hash for a DAPT corpus manifest."""

    rows = (
        load_corpus_manifest(manifest)
        if isinstance(manifest, (str, Path))
        else manifest
    )
    canonical_rows = tuple(sorted((dict(row) for row in rows), key=_row_sort_key))
    encoded = json.dumps(
        canonical_rows,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def assert_manifest_has_no_raw_text(
    path: str | Path = DAPT_CORPUS_MANIFEST_PATH,
) -> None:
    """Assert that a committed manifest contains only metadata and hashes."""

    for row in load_corpus_manifest(path):
        raw_fields = RAW_TEXT_MANIFEST_FIELDS.intersection(row)
        if raw_fields:
            fields = ", ".join(sorted(raw_fields))
            raise CorpusManifestError(
                f"manifest row {row['source']}:{row['source_id']} contains raw "
                f"text field(s): {fields}"
            )


def normalize_passage_text(text: str) -> str:
    """Normalize passage text for duplicate detection."""

    normalized = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(normalized.split())


def token_count(text: str) -> int:
    """Return a deterministic whitespace token count for a passage."""

    return len(text.split())


def _validate_manifest_row(
    row: Mapping[str, Any], context: str = "manifest row"
) -> None:
    required_fields = {
        "license",
        "normalized_sha256",
        "schema_version",
        "sha256",
        "source",
        "source_id",
        "token_count",
    }
    missing = sorted(required_fields - set(row))
    if missing:
        raise CorpusManifestError(
            f"{context} missing required field(s): {', '.join(missing)}"
        )
    if row["schema_version"] != DAPT_CORPUS_SCHEMA_VERSION:
        raise CorpusManifestError(
            f"{context} schema_version must be {DAPT_CORPUS_SCHEMA_VERSION!r}"
        )
    for field_name in ("license", "source", "source_id"):
        value = row[field_name]
        if not isinstance(value, str) or not value:
            raise CorpusManifestError(f"{context} {field_name} must be a string")
    for field_name in ("sha256", "normalized_sha256"):
        value = row[field_name]
        if not isinstance(value, str) or not _SHA256_PATTERN.fullmatch(value):
            raise CorpusManifestError(
                f"{context} {field_name} must be sha256:<64 lower hex>"
            )
    token_value = row["token_count"]
    if not isinstance(token_value, int) or isinstance(token_value, bool):
        raise CorpusManifestError(f"{context} token_count must be an integer")
    if token_value <= 0:
        raise CorpusManifestError(f"{context} token_count must be positive")
    raw_fields = RAW_TEXT_MANIFEST_FIELDS.intersection(row)
    if raw_fields:
        fields = ", ".join(sorted(raw_fields))
        raise CorpusManifestError(f"{context} contains raw text field(s): {fields}")


def _row_sort_key(row: Mapping[str, Any]) -> tuple[str, str, str]:
    return (str(row["source"]), str(row["source_id"]), str(row["sha256"]))


def _record_string(
    record: Mapping[str, Any], field_name: str, source: str, index: int
) -> str:
    value = record.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise CorpusManifestError(
            f"{source} record {index} field {field_name!r} must be a non-empty string"
        )
    return value


def _record_optional_string(
    record: Mapping[str, Any], field_name: str | None
) -> str | None:
    if field_name is None:
        return None
    value = record.get(field_name)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise CorpusManifestError(f"field {field_name!r} must be a string when present")
    return value


def _require_non_empty(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise CorpusManifestError(f"{field_name} must be a non-empty string")


def _sha256(text: str) -> str:
    return f"sha256:{hashlib.sha256(text.encode('utf-8')).hexdigest()}"
