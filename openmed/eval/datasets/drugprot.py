"""DrugProt BioCreative VII public relation-extraction corpus loader."""

from __future__ import annotations

import csv
import hashlib
import io
import os
import shutil
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence
from urllib.request import Request, urlopen

from openmed.core.labels import CANONICAL_LABELS, OTHER
from openmed.eval.datasets.licenses import license_for
from openmed.eval.harness import BenchmarkFixture
from openmed.eval.metrics import EvalSpan

DRUGPROT = "drugprot"
DRUGPROT_ZENODO_RECORD = "4955411"
DRUGPROT_DOI = "10.5281/zenodo.4955411"
DRUGPROT_SOURCE_URL = f"https://zenodo.org/records/{DRUGPROT_ZENODO_RECORD}"
DRUGPROT_ARCHIVE_NAME = "drugprot-gs.zip"
DRUGPROT_ARCHIVE_MD5 = "0c11a875b9066a19571157ece0df6f63"
DRUGPROT_DOWNLOAD_URL = (
    f"{DRUGPROT_SOURCE_URL}/files/{DRUGPROT_ARCHIVE_NAME}?download=1"
)
DEFAULT_SPLIT = "training"

DRUGPROT_ENTITY_TO_CANONICAL: Mapping[str, str] = {
    "CHEMICAL": OTHER,
    "GENE": OTHER,
    "GENE-N": OTHER,
    "GENE-Y": OTHER,
}

DRUGPROT_RELATION_TYPES: tuple[str, ...] = (
    "ACTIVATOR",
    "AGONIST",
    "AGONIST-ACTIVATOR",
    "AGONIST-INHIBITOR",
    "ANTAGONIST",
    "DIRECT-REGULATOR",
    "INDIRECT-DOWNREGULATOR",
    "INDIRECT-UPREGULATOR",
    "INHIBITOR",
    "PART-OF",
    "PRODUCT-OF",
    "SUBSTRATE",
    "SUBSTRATE_PRODUCT-OF",
)

Downloader = Callable[[str, Path], Path | None]


@dataclass(frozen=True)
class DrugProtEntity:
    """One DrugProt entity mention."""

    pmid: str
    entity_id: str
    source_label: str
    start: int
    end: int
    text: str
    canonical_label: str

    @property
    def entity_group(self) -> str:
        """Return DrugProt's coarse entity family."""
        normalized = self.source_label.strip().upper()
        if normalized.startswith("GENE"):
            return "GENE"
        return normalized

    def to_eval_span(self, document_text: str) -> EvalSpan:
        """Convert the entity mention to the eval harness span schema."""
        span_text = document_text[self.start : self.end] or self.text
        return EvalSpan(
            start=self.start,
            end=self.end,
            label=self.canonical_label,
            text=span_text,
            language="en",
            metadata={
                "canonical_label": self.canonical_label,
                "drugprot_label": self.source_label,
                "entity_group": self.entity_group,
                "entity_id": self.entity_id,
                "source_label": self.source_label,
                "source_pmid": self.pmid,
            },
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready entity representation."""
        return {
            "canonical_label": self.canonical_label,
            "end": self.end,
            "entity_group": self.entity_group,
            "entity_id": self.entity_id,
            "pmid": self.pmid,
            "source_label": self.source_label,
            "start": self.start,
            "text": self.text,
        }


@dataclass(frozen=True)
class DrugProtRelation:
    """One typed DrugProt relation between two entity mentions."""

    pmid: str
    relation_type: str
    arg1_id: str
    arg2_id: str
    arg1: DrugProtEntity
    arg2: DrugProtEntity

    def to_tuple(self) -> tuple[str, str, str]:
        """Return the compact relation tuple used by tests and adapters."""
        return (self.relation_type, self.arg1_id, self.arg2_id)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready relation representation."""
        return {
            "arg1": self.arg1.to_dict(),
            "arg1_id": self.arg1_id,
            "arg2": self.arg2.to_dict(),
            "arg2_id": self.arg2_id,
            "pmid": self.pmid,
            "relation_type": self.relation_type,
        }


@dataclass(frozen=True)
class DrugProtRecord:
    """One DrugProt abstract with entities and relations."""

    pmid: str
    title: str
    abstract: str
    text: str
    entities: tuple[DrugProtEntity, ...]
    relations: tuple[DrugProtRelation, ...]
    split: str = DEFAULT_SPLIT

    def to_benchmark_fixture(self) -> BenchmarkFixture:
        """Expose the NER view as an eval harness fixture."""
        return BenchmarkFixture(
            fixture_id=self.pmid,
            text=self.text,
            gold_spans=tuple(
                entity.to_eval_span(self.text) for entity in self.entities
            ),
            language="en",
            metadata={
                "dataset": DRUGPROT,
                "doi": DRUGPROT_DOI,
                "license": license_for(DRUGPROT).to_dict(),
                "relation_count": len(self.relations),
                "source_pmid": self.pmid,
                "split": self.split,
                "task": "ner",
            },
        )

    def to_relation_fixture(self) -> "DrugProtRelationFixture":
        """Expose the relation-extraction view for clinical relation harnesses."""
        return DrugProtRelationFixture(
            fixture_id=self.pmid,
            text=self.text,
            entities=self.entities,
            relations=self.relations,
            metadata={
                "dataset": DRUGPROT,
                "doi": DRUGPROT_DOI,
                "license": license_for(DRUGPROT).to_dict(),
                "source_pmid": self.pmid,
                "split": self.split,
                "task": "relation",
            },
        )


@dataclass(frozen=True)
class DrugProtRelationFixture:
    """One relation-extraction fixture preserving entity references."""

    fixture_id: str
    text: str
    entities: tuple[DrugProtEntity, ...]
    relations: tuple[DrugProtRelation, ...]
    language: str = "en"
    metadata: Mapping[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready fixture representation."""
        return {
            "entities": [entity.to_dict() for entity in self.entities],
            "fixture_id": self.fixture_id,
            "language": self.language,
            "metadata": dict(self.metadata or {}),
            "relations": [relation.to_dict() for relation in self.relations],
            "text": self.text,
        }


DrugProtSuiteFixture = BenchmarkFixture | DrugProtRelationFixture


@dataclass(frozen=True)
class DrugProtCorpus:
    """Loaded DrugProt corpus split."""

    records: tuple[DrugProtRecord, ...]
    split: str
    source_path: str

    def to_ner_fixtures(self) -> list[BenchmarkFixture]:
        """Return NER benchmark fixtures."""
        return [record.to_benchmark_fixture() for record in self.records]

    def to_relation_fixtures(self) -> list[DrugProtRelationFixture]:
        """Return relation-extraction fixtures."""
        return [record.to_relation_fixture() for record in self.records]


def map_drugprot_entity_label(label: str) -> str:
    """Map a DrugProt entity label onto OpenMed's canonical taxonomy."""
    normalized = label.strip().upper()
    if normalized.startswith("GENE-"):
        normalized = (
            "GENE" if normalized not in DRUGPROT_ENTITY_TO_CANONICAL else normalized
        )
    canonical = DRUGPROT_ENTITY_TO_CANONICAL.get(normalized)
    if canonical is None:
        allowed = ", ".join(sorted(DRUGPROT_ENTITY_TO_CANONICAL))
        raise ValueError(f"unknown DrugProt entity label {label!r}; expected {allowed}")
    if canonical not in CANONICAL_LABELS:
        raise RuntimeError(f"DrugProt label maps to non-canonical label {canonical!r}")
    return canonical


def drugprot_suite_metadata(*, task: str = "ner") -> dict[str, Any]:
    """Return source, license, and label metadata for DrugProt."""
    normalized_task = _normalize_task(task)
    dataset_license = license_for(DRUGPROT)
    return {
        "archive": DRUGPROT_ARCHIVE_NAME,
        "archive_md5": DRUGPROT_ARCHIVE_MD5,
        "dataset": DRUGPROT,
        "doi": DRUGPROT_DOI,
        "download_url": DRUGPROT_DOWNLOAD_URL,
        "entity_label_mapping": dict(sorted(DRUGPROT_ENTITY_TO_CANONICAL.items())),
        "license": dataset_license.license_id,
        "license_metadata": dataset_license.to_dict(),
        "redistribution": "not vendored; downloaded into local cache on demand",
        "relation_types": DRUGPROT_RELATION_TYPES,
        "source_url": DRUGPROT_SOURCE_URL,
        "suite": DRUGPROT,
        "task": normalized_task,
    }


def load_drugprot_corpus(
    path: str | Path | None = None,
    *,
    cache_dir: str | Path | None = None,
    downloader: Downloader | None = None,
    split: str = DEFAULT_SPLIT,
) -> DrugProtCorpus:
    """Load DrugProt TSV files from a directory, zip archive, or local cache.

    When ``path`` is omitted, the public Zenodo archive is downloaded into the
    OpenMed cache on demand. No corpus rows are stored in the repository.
    """
    source_path = _resolve_source(path, cache_dir=cache_dir, downloader=downloader)
    abstracts = _read_tsv_rows(source_path, _abstract_names(split))
    entities = _read_tsv_rows(source_path, _entity_names(split))
    relations = _read_tsv_rows(source_path, _relation_names(split))
    return corpus_from_rows(
        abstracts,
        entities,
        relations,
        split=split,
        source_path=str(source_path),
    )


def load_drugprot_ner_fixtures(
    path: str | Path | None = None,
    *,
    cache_dir: str | Path | None = None,
    downloader: Downloader | None = None,
    split: str = DEFAULT_SPLIT,
) -> list[BenchmarkFixture]:
    """Load DrugProt's NER view as benchmark fixtures."""
    return load_drugprot_corpus(
        path,
        cache_dir=cache_dir,
        downloader=downloader,
        split=split,
    ).to_ner_fixtures()


def load_drugprot_relation_fixtures(
    path: str | Path | None = None,
    *,
    cache_dir: str | Path | None = None,
    downloader: Downloader | None = None,
    split: str = DEFAULT_SPLIT,
) -> list[DrugProtRelationFixture]:
    """Load DrugProt's relation-extraction view."""
    return load_drugprot_corpus(
        path,
        cache_dir=cache_dir,
        downloader=downloader,
        split=split,
    ).to_relation_fixtures()


def load_drugprot_fixtures(
    path: str | Path | None = None,
    *,
    task: str = "ner",
    cache_dir: str | Path | None = None,
    downloader: Downloader | None = None,
    split: str = DEFAULT_SPLIT,
) -> list[DrugProtSuiteFixture]:
    """Load the requested DrugProt task view."""
    normalized_task = _normalize_task(task)
    corpus = load_drugprot_corpus(
        path,
        cache_dir=cache_dir,
        downloader=downloader,
        split=split,
    )
    if normalized_task == "ner":
        return corpus.to_ner_fixtures()
    return corpus.to_relation_fixtures()


def corpus_from_rows(
    abstract_rows: Iterable[Sequence[str]],
    entity_rows: Iterable[Sequence[str]],
    relation_rows: Iterable[Sequence[str]],
    *,
    split: str = DEFAULT_SPLIT,
    source_path: str = "<memory>",
) -> DrugProtCorpus:
    """Build a DrugProt corpus from parsed TSV rows."""
    abstract_by_pmid = _parse_abstracts(abstract_rows)
    entities_by_pmid = _parse_entities(entity_rows)

    records_by_pmid: dict[str, DrugProtRecord] = {}
    entity_lookup_by_pmid: dict[str, dict[str, DrugProtEntity]] = {}
    relation_rows_by_pmid: dict[str, list[Sequence[str]]] = defaultdict(list)
    for row in relation_rows:
        if len(row) < 4:
            raise ValueError(f"DrugProt relation row must have 4 columns: {row!r}")
        relation_rows_by_pmid[str(row[0])].append(row)

    for pmid, (title, abstract) in abstract_by_pmid.items():
        text = _document_text(title, abstract)
        entities = tuple(
            sorted(
                (_validate_entity(entity, text) for entity in entities_by_pmid[pmid]),
                key=lambda entity: (entity.start, entity.end, entity.entity_id),
            )
        )
        entity_lookup = {entity.entity_id: entity for entity in entities}
        entity_lookup_by_pmid[pmid] = entity_lookup
        records_by_pmid[pmid] = DrugProtRecord(
            pmid=pmid,
            title=title,
            abstract=abstract,
            text=text,
            entities=entities,
            relations=(),
            split=split,
        )

    records: list[DrugProtRecord] = []
    for pmid, record in records_by_pmid.items():
        relation_rows_for_record = relation_rows_by_pmid.get(pmid, [])
        relations = tuple(
            _parse_relation(row, entity_lookup_by_pmid[pmid])
            for row in relation_rows_for_record
        )
        records.append(
            DrugProtRecord(
                pmid=record.pmid,
                title=record.title,
                abstract=record.abstract,
                text=record.text,
                entities=record.entities,
                relations=relations,
                split=record.split,
            )
        )

    return DrugProtCorpus(
        records=tuple(records),
        split=split,
        source_path=source_path,
    )


def _parse_abstracts(
    rows: Iterable[Sequence[str]],
) -> dict[str, tuple[str, str]]:
    abstracts: dict[str, tuple[str, str]] = {}
    for row in rows:
        if len(row) < 3:
            raise ValueError(f"DrugProt abstract row must have 3 columns: {row!r}")
        pmid = str(row[0])
        if pmid in abstracts:
            raise ValueError(f"duplicate DrugProt abstract PMID: {pmid}")
        abstracts[pmid] = (str(row[1]), str(row[2]))
    return abstracts


def _parse_entities(
    rows: Iterable[Sequence[str]],
) -> dict[str, list[DrugProtEntity]]:
    entities_by_pmid: dict[str, list[DrugProtEntity]] = defaultdict(list)
    for row in rows:
        if len(row) < 6:
            raise ValueError(f"DrugProt entity row must have 6 columns: {row!r}")
        pmid = str(row[0])
        source_label = str(row[2])
        entities_by_pmid[pmid].append(
            DrugProtEntity(
                pmid=pmid,
                entity_id=str(row[1]),
                source_label=source_label,
                start=_parse_int(row[3], "entity start"),
                end=_parse_int(row[4], "entity end"),
                text=str(row[5]),
                canonical_label=map_drugprot_entity_label(source_label),
            )
        )
    return entities_by_pmid


def _parse_relation(
    row: Sequence[str],
    entities_by_id: Mapping[str, DrugProtEntity],
) -> DrugProtRelation:
    pmid = str(row[0])
    relation_type = str(row[1]).strip().upper()
    if relation_type not in DRUGPROT_RELATION_TYPES:
        allowed = ", ".join(DRUGPROT_RELATION_TYPES)
        raise ValueError(
            f"unknown DrugProt relation type {row[1]!r}; expected one of: {allowed}"
        )
    arg1_id = _parse_relation_arg(str(row[2]), "Arg1")
    arg2_id = _parse_relation_arg(str(row[3]), "Arg2")
    try:
        arg1 = entities_by_id[arg1_id]
        arg2 = entities_by_id[arg2_id]
    except KeyError as exc:
        raise ValueError(
            f"DrugProt relation references unknown entity {exc.args[0]!r}"
        ) from exc
    return DrugProtRelation(
        pmid=pmid,
        relation_type=relation_type,
        arg1_id=arg1_id,
        arg2_id=arg2_id,
        arg1=arg1,
        arg2=arg2,
    )


def _validate_entity(entity: DrugProtEntity, document_text: str) -> DrugProtEntity:
    if entity.start < 0 or entity.end < entity.start or entity.end > len(document_text):
        raise ValueError(
            "invalid DrugProt span offsets "
            f"{entity.start}:{entity.end} for text length {len(document_text)}"
        )
    span_text = document_text[entity.start : entity.end]
    if span_text != entity.text:
        raise ValueError(
            "DrugProt span text mismatch for "
            f"{entity.entity_id}: offsets {entity.start}:{entity.end} select "
            f"{span_text!r}, TSV text is {entity.text!r}"
        )
    return entity


def _parse_relation_arg(value: str, expected_name: str) -> str:
    name, separator, entity_id = value.partition(":")
    if separator != ":" or name != expected_name or not entity_id:
        raise ValueError(
            f"DrugProt relation argument must look like {expected_name}:T1: {value!r}"
        )
    return entity_id


def _parse_int(value: object, field_name: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        raise ValueError(
            f"DrugProt {field_name} must be an integer: {value!r}"
        ) from None


def _document_text(title: str, abstract: str) -> str:
    return " ".join(part for part in (title.strip(), abstract.strip()) if part)


def _normalize_task(task: str) -> str:
    normalized = task.strip().lower()
    if normalized not in {"ner", "relation"}:
        raise ValueError("DrugProt task must be 'ner' or 'relation'")
    return normalized


def _resolve_source(
    path: str | Path | None,
    *,
    cache_dir: str | Path | None,
    downloader: Downloader | None,
) -> Path:
    if path is not None:
        source_path = Path(path).expanduser()
        if not source_path.exists():
            raise FileNotFoundError(f"DrugProt source path not found: {source_path}")
        return source_path
    return _ensure_drugprot_archive(cache_dir=cache_dir, downloader=downloader)


def _ensure_drugprot_archive(
    *,
    cache_dir: str | Path | None,
    downloader: Downloader | None,
) -> Path:
    cache_root = Path(cache_dir).expanduser() if cache_dir else _default_cache_root()
    archive_path = cache_root / DRUGPROT_ARCHIVE_NAME
    if archive_path.exists() and _md5(archive_path) == DRUGPROT_ARCHIVE_MD5:
        return archive_path

    cache_root.mkdir(parents=True, exist_ok=True)
    if downloader is None:
        _download_url(DRUGPROT_DOWNLOAD_URL, archive_path)
    else:
        downloaded = downloader(DRUGPROT_DOWNLOAD_URL, archive_path)
        if downloaded is not None:
            archive_path = Path(downloaded).expanduser()

    if not archive_path.exists():
        raise RuntimeError(f"DrugProt download did not create {archive_path}")
    actual_md5 = _md5(archive_path)
    if actual_md5 != DRUGPROT_ARCHIVE_MD5:
        raise RuntimeError(
            "DrugProt archive checksum mismatch: "
            f"expected {DRUGPROT_ARCHIVE_MD5}, got {actual_md5}"
        )
    return archive_path


def _default_cache_root() -> Path:
    env_cache = os.getenv("OPENMED_CACHE_DIR")
    if env_cache:
        return Path(env_cache).expanduser() / "datasets" / DRUGPROT
    try:
        from openmed.core.config import get_config

        configured = get_config().cache_dir
    except Exception:
        configured = None
    return Path(configured or "~/.cache/openmed").expanduser() / "datasets" / DRUGPROT


def _download_url(url: str, target: Path) -> None:
    temporary = target.with_suffix(target.suffix + ".tmp")
    request = Request(url, headers={"User-Agent": "OpenMed dataset loader"})
    with urlopen(request, timeout=60) as response:  # nosec: fixed public Zenodo URL
        with temporary.open("wb") as handle:
            shutil.copyfileobj(response, handle)
    temporary.replace(target)


def _md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_tsv_rows(source_path: Path, names: Sequence[str]) -> list[tuple[str, ...]]:
    if source_path.is_dir():
        path = _find_file(source_path, names)
        text = path.read_text(encoding="utf-8")
    elif zipfile.is_zipfile(source_path):
        text = _read_zip_text(source_path, names)
    else:
        raise ValueError(
            "DrugProt source must be a directory containing TSV files or a zip archive"
        )
    return [
        tuple(row)
        for row in csv.reader(io.StringIO(text), delimiter="\t")
        if row and any(cell.strip() for cell in row)
    ]


def _find_file(root: Path, names: Sequence[str]) -> Path:
    for name in names:
        direct = root / name
        if direct.exists():
            return direct
    wanted = set(names)
    for path in sorted(root.rglob("*.tsv")):
        if path.name in wanted:
            return path
    expected = ", ".join(names)
    raise FileNotFoundError(f"DrugProt TSV file not found under {root}: {expected}")


def _read_zip_text(archive_path: Path, names: Sequence[str]) -> str:
    with zipfile.ZipFile(archive_path) as archive:
        member = _find_zip_member(archive, names)
        with archive.open(member) as handle:
            return handle.read().decode("utf-8")


def _find_zip_member(archive: zipfile.ZipFile, names: Sequence[str]) -> str:
    wanted = set(names)
    for member in archive.namelist():
        if Path(member).name in wanted:
            return member
    expected = ", ".join(names)
    raise FileNotFoundError(f"DrugProt TSV file not found in archive: {expected}")


def _abstract_names(split: str) -> tuple[str, ...]:
    return (
        f"drugprot_{split}_abstracts.tsv",
        f"drugprot_{split}_abstracs.tsv",
    )


def _entity_names(split: str) -> tuple[str, ...]:
    return (f"drugprot_{split}_entities.tsv",)


def _relation_names(split: str) -> tuple[str, ...]:
    return (f"drugprot_{split}_relations.tsv",)


__all__ = [
    "DEFAULT_SPLIT",
    "DRUGPROT",
    "DRUGPROT_ARCHIVE_MD5",
    "DRUGPROT_ARCHIVE_NAME",
    "DRUGPROT_DOI",
    "DRUGPROT_DOWNLOAD_URL",
    "DRUGPROT_ENTITY_TO_CANONICAL",
    "DRUGPROT_RELATION_TYPES",
    "DRUGPROT_SOURCE_URL",
    "DRUGPROT_ZENODO_RECORD",
    "DrugProtCorpus",
    "DrugProtEntity",
    "DrugProtRecord",
    "DrugProtRelation",
    "DrugProtRelationFixture",
    "DrugProtSuiteFixture",
    "corpus_from_rows",
    "drugprot_suite_metadata",
    "load_drugprot_corpus",
    "load_drugprot_fixtures",
    "load_drugprot_ner_fixtures",
    "load_drugprot_relation_fixtures",
    "map_drugprot_entity_label",
]
