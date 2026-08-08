"""Public biomedical dataset adapters for evaluation and training prep."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Mapping

from openmed.core.labels import CANONICAL_LABELS, normalize_label
from openmed.eval.harness import BenchmarkFixture
from openmed.eval.metrics import EvalSpan

from .licenses import PUBLIC_DATASET_LICENSES, DatasetLicense, license_for

PUBLIC_DATASETS: tuple[str, ...] = (
    "shield",
    "drugprot",
    "medmentions",
    "ncbi_disease",
    "bc5cdr",
    "jnlpba",
    "species_800",
    "bc2gm",
)

PUBLIC_LABEL_MAPS: Mapping[str, Mapping[str, str]] = {
    "shield": {
        "age": "AGE",
        "date": "DATE",
        "doctor": "PERSON",
        "hospital": "ORGANIZATION",
        "id": "ID_NUM",
        "location": "LOCATION",
        "patient": "PERSON",
        "phone": "PHONE",
        "web": "URL",
    },
    "drugprot": {
        "chemical": "OTHER",
        "gene": "OTHER",
        "gene-n": "OTHER",
        "gene-y": "OTHER",
        "protein": "OTHER",
        "relation": "OTHER",
    },
    "medmentions": {
        "concept": "OTHER",
        "mention": "OTHER",
        "disease": "OTHER",
        "drug": "OTHER",
    },
    "ncbi_disease": {
        "disease": "OTHER",
        "specific_disease": "OTHER",
        "modifier": "OTHER",
        "composite_mention": "OTHER",
    },
    "bc5cdr": {
        "chemical": "OTHER",
        "disease": "OTHER",
        "relation": "OTHER",
    },
    "jnlpba": {
        "cell_line": "OTHER",
        "cell_type": "OTHER",
        "dna": "OTHER",
        "protein": "OTHER",
        "rna": "OTHER",
    },
    "species_800": {
        "organism": "MICROORGANISM",
        "species": "MICROORGANISM",
        "taxon": "MICROORGANISM",
    },
    "bc2gm": {
        "gene": "OTHER",
        "gene_mention": "OTHER",
        "protein": "OTHER",
    },
}

CLINICAL_MODEL_FAMILIES: tuple[str, ...] = (
    "doctype",
    "section",
    "relex_med",
    "relex_ade",
    "link",
)
CLINICAL_LINK_VOCABULARIES: tuple[str, ...] = (
    "rxnorm",
    "icd10cm",
    "loinc",
    "hpo",
    "mesh",
)


@dataclass(frozen=True)
class ClinicalDatasetBinding:
    """Metadata-only dataset policy for one clinical model family.

    The binding describes permitted uses without carrying paths, credentials,
    corpus rows, or controlled-vocabulary content. Optional bindings represent
    evaluations that can run only when the caller already has licensed access.
    """

    dataset: str
    uses: tuple[str, ...]
    access: str
    required: bool = True
    user_key_gated: bool = False

    def __post_init__(self) -> None:
        if not self.dataset:
            raise ValueError("clinical dataset binding requires a dataset id")
        if not self.uses or set(self.uses) - {"train", "eval"}:
            raise ValueError("clinical dataset uses must contain train and/or eval")
        if len(set(self.uses)) != len(self.uses):
            raise ValueError("clinical dataset uses must not contain duplicates")
        if self.access not in {
            "public",
            "synthetic",
            "dua-eval-only",
            "user-supplied",
        }:
            raise ValueError("unknown clinical dataset access policy")
        if self.access == "dua-eval-only" and self.uses != ("eval",):
            raise ValueError("DUA datasets are eval-only")
        if self.user_key_gated and self.access != "user-supplied":
            raise ValueError("user-key gating requires user-supplied access")


CLINICAL_FAMILY_DATASET_BINDINGS: Mapping[str, tuple[ClinicalDatasetBinding, ...]] = (
    MappingProxyType(
        {
            "doctype": (
                ClinicalDatasetBinding(
                    "synthetic_section_doctype",
                    ("train", "eval"),
                    "synthetic",
                ),
            ),
            "section": (
                ClinicalDatasetBinding(
                    "synthetic_section_doctype",
                    ("train", "eval"),
                    "synthetic",
                ),
            ),
            "relex_med": (
                ClinicalDatasetBinding("drugprot", ("train", "eval"), "public"),
                ClinicalDatasetBinding(
                    "n2c2", ("eval",), "dua-eval-only", required=False
                ),
                ClinicalDatasetBinding(
                    "made", ("eval",), "dua-eval-only", required=False
                ),
            ),
            "relex_ade": (
                ClinicalDatasetBinding("drugprot", ("train", "eval"), "public"),
                ClinicalDatasetBinding(
                    "n2c2", ("eval",), "dua-eval-only", required=False
                ),
                ClinicalDatasetBinding(
                    "made", ("eval",), "dua-eval-only", required=False
                ),
            ),
            "link": (
                ClinicalDatasetBinding(
                    "redistributable_vocabulary",
                    ("train",),
                    "user-supplied",
                ),
                ClinicalDatasetBinding("medmentions", ("eval",), "public"),
                ClinicalDatasetBinding(
                    "umls",
                    ("eval",),
                    "user-supplied",
                    required=False,
                    user_key_gated=True,
                ),
                ClinicalDatasetBinding(
                    "snomed",
                    ("eval",),
                    "user-supplied",
                    required=False,
                    user_key_gated=True,
                ),
            ),
        }
    )
)

_CLINICAL_EVIDENCE_FIELDS = frozenset(
    {
        "corpus_bundled",
        "license_id",
        "manifest_hash",
        "metrics",
        "redistributable",
        "restricted_vocabulary_bundled",
        "user_key_gated",
        "uses",
        "vocab",
    }
)
_SHA256_PREFIX = "sha256:"

_CONTROLLED_METADATA_KEYS = {
    "cui",
    "concept_id",
    "concept_ids",
    "mesh",
    "mesh_id",
    "ontology",
    "semantic_type_id",
    "semantic_type_ids",
    "snomed",
    "umls",
    "umls_id",
}

_GATED_CONTENT_MARKERS = (
    "CEGS",
    "UMLS",
    "SNOMED",
    "CPT",
    "i2b2",
    "n2c2",
    "SHAC",
    "THYME",
    "MedNLI",
    "MADE",
    "MIMIC",
    "N-GRID",
    "BHC",
)
_UNAMBIGUOUS_GATED_MARKERS = frozenset(
    {
        "CEGS",
        "UMLS",
        "SNOMED",
        "CPT",
        "i2b2",
        "n2c2",
        "MedNLI",
        "N-GRID",
        "BHC",
    }
)
_STRUCTURED_GATED_MARKERS = frozenset(_GATED_CONTENT_MARKERS) - (
    _UNAMBIGUOUS_GATED_MARKERS
)
_STRUCTURED_MARKER_RE = re.compile(
    r'"(?:corpus|dataset|label|source)"\s*:\s*"(?P<marker>[^"]+)"',
    re.IGNORECASE,
)
_DATA_EXTENSIONS = {".csv", ".json", ".jsonl", ".ndjson", ".tsv", ".txt"}


class DatasetUnavailable(FileNotFoundError):
    """Raised when a dataset source was requested but is absent."""


@dataclass(frozen=True)
class PublicDatasetSpan:
    start: int
    end: int
    label: str
    text: str = ""
    source_label: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_eval_span(self, *, language: str = "en") -> EvalSpan:
        return EvalSpan(
            start=self.start,
            end=self.end,
            label=self.label,
            text=self.text,
            language=language,
            metadata={
                **dict(self.metadata),
                "source_label": self.source_label or self.label,
            },
        )


@dataclass(frozen=True)
class PublicDatasetRecord:
    record_id: str
    dataset: str
    text: str
    spans: tuple[PublicDatasetSpan, ...]
    split: str = "unspecified"
    language: str = "en"
    metadata: Mapping[str, Any] = field(default_factory=dict)
    license: DatasetLicense | None = None

    def to_benchmark_fixture(self) -> BenchmarkFixture:
        return BenchmarkFixture(
            fixture_id=self.record_id,
            text=self.text,
            gold_spans=tuple(
                span.to_eval_span(language=self.language) for span in self.spans
            ),
            language=self.language,
            metadata={
                **dict(self.metadata),
                "dataset": self.dataset,
                "license": self.license.to_dict() if self.license else None,
                "split": self.split,
            },
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "id": self.record_id,
            "language": self.language,
            "license": self.license.to_dict() if self.license else None,
            "metadata": dict(self.metadata),
            "spans": [
                {
                    "end": span.end,
                    "label": span.label,
                    "metadata": dict(span.metadata),
                    "source_label": span.source_label,
                    "start": span.start,
                    "text": span.text,
                }
                for span in self.spans
            ],
            "split": self.split,
            "text": self.text,
        }


@dataclass(frozen=True)
class DatasetLoadResult:
    dataset: str
    records: tuple[PublicDatasetRecord, ...]
    skipped: bool = False
    reason: str = ""
    license: DatasetLicense | None = None

    def to_benchmark_fixtures(self) -> list[BenchmarkFixture]:
        return [record.to_benchmark_fixture() for record in self.records]


@dataclass(frozen=True)
class PublicDatasetAdapter:
    dataset: str
    label_map: Mapping[str, str]
    license: DatasetLicense

    def load(self, path: str | Path | None = None) -> DatasetLoadResult:
        if path is None:
            return DatasetLoadResult(
                dataset=self.dataset,
                records=(),
                skipped=True,
                reason="dataset path not provided",
                license=self.license,
            )

        source_path = Path(path)
        if not source_path.exists():
            return DatasetLoadResult(
                dataset=self.dataset,
                records=(),
                skipped=True,
                reason=f"dataset path not found: {source_path}",
                license=self.license,
            )

        rows = _load_rows(source_path)
        records = tuple(self._record_from_mapping(row) for row in rows)
        return DatasetLoadResult(
            dataset=self.dataset,
            records=records,
            skipped=False,
            license=self.license,
        )

    def _record_from_mapping(self, row: Mapping[str, Any]) -> PublicDatasetRecord:
        text = _record_text(row)
        language = str(row.get("language") or row.get("lang") or "en")
        spans = tuple(
            self._span_from_mapping(span, text=text, language=language)
            for span in row.get("spans")
            or row.get("entities")
            or row.get("annotations")
            or []
        )
        return PublicDatasetRecord(
            record_id=str(
                row.get("id") or row.get("record_id") or row.get("pmid") or "record"
            ),
            dataset=self.dataset,
            text=text,
            spans=spans,
            split=str(row.get("split") or "unspecified"),
            language=language,
            metadata=_clean_metadata(row.get("metadata") or {}),
            license=self.license,
        )

    def _span_from_mapping(
        self,
        span: Mapping[str, Any],
        *,
        text: str,
        language: str,
    ) -> PublicDatasetSpan:
        start = _int_field(span, "start", "span_start", "begin")
        end = _int_field(span, "end", "span_end", "offset_end")
        source_label = str(
            span.get("label")
            or span.get("source_label")
            or span.get("entity_type")
            or span.get("type")
            or "OTHER"
        )
        canonical = map_public_label(self.dataset, source_label, language=language)
        span_text = str(span.get("text") or text[start:end])
        return PublicDatasetSpan(
            start=start,
            end=end,
            label=canonical,
            text=span_text,
            source_label=source_label,
            metadata=_clean_metadata(span.get("metadata") or {}),
        )


def adapter_for(dataset: str) -> PublicDatasetAdapter:
    if dataset not in PUBLIC_DATASETS:
        raise ValueError(f"unknown public dataset: {dataset}")
    return PublicDatasetAdapter(
        dataset=dataset,
        label_map=PUBLIC_LABEL_MAPS[dataset],
        license=license_for(dataset),
    )


def load_public_dataset(
    dataset: str, path: str | Path | None = None
) -> DatasetLoadResult:
    return adapter_for(dataset).load(path)


def map_public_label(dataset: str, label: str, *, language: str = "en") -> str:
    label_map = PUBLIC_LABEL_MAPS.get(dataset)
    if label_map is None:
        raise ValueError(f"unknown public dataset: {dataset}")
    mapped = label_map.get(label.lower(), label_map.get(label.upper(), label))
    canonical = normalize_label(mapped, language)
    if canonical not in CANONICAL_LABELS:
        return "OTHER"
    return canonical


def clinical_family_dataset_bindings(
    family: str,
) -> tuple[ClinicalDatasetBinding, ...]:
    """Return the immutable dataset policy for a clinical model family.

    Args:
        family: Canonical clinical family identifier.

    Returns:
        Ordered dataset bindings for required and optional sources.

    Raises:
        ValueError: If ``family`` is unknown.
    """

    try:
        return CLINICAL_FAMILY_DATASET_BINDINGS[family]
    except KeyError as exc:
        raise ValueError(f"unknown clinical model family: {family}") from exc


def validate_clinical_family_dataset_evidence(
    family: str,
    evidence: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Validate aggregate-only dataset evidence for a completed training run.

    Required training/evaluation sources must be present with a stable manifest
    hash. DUA sources can appear only as optional evaluation evidence, and every
    source must explicitly state that neither corpus rows nor restricted
    vocabulary content is bundled.

    Args:
        family: Canonical clinical family identifier.
        evidence: Per-dataset aggregate metrics and provenance declarations.

    Returns:
        Canonical aggregate-only evidence ordered by dataset identifier.

    Raises:
        TypeError: If the evidence shape is not mapping-based.
        ValueError: If required evidence is absent or violates dataset policy.
    """

    if not isinstance(evidence, Mapping):
        raise TypeError("clinical dataset evidence must be a mapping")

    bindings = clinical_family_dataset_bindings(family)
    policy = {binding.dataset: binding for binding in bindings}
    unknown = sorted(set(evidence) - set(policy))
    if unknown:
        raise ValueError("unexpected clinical dataset evidence: " + ", ".join(unknown))
    missing = sorted(
        binding.dataset
        for binding in bindings
        if binding.required and binding.dataset not in evidence
    )
    if missing:
        raise ValueError(
            "missing required clinical dataset evidence: " + ", ".join(missing)
        )

    validated: dict[str, dict[str, Any]] = {}
    for dataset in sorted(evidence):
        raw = evidence[dataset]
        if not isinstance(raw, Mapping):
            raise TypeError(f"dataset evidence for {dataset} must be a mapping")
        extra = sorted(set(raw) - _CLINICAL_EVIDENCE_FIELDS)
        if extra:
            raise ValueError(
                f"dataset evidence for {dataset} contains unsupported fields: "
                + ", ".join(extra)
            )

        binding = policy[dataset]
        uses = _evidence_uses(raw.get("uses"), dataset=dataset)
        if uses != binding.uses:
            raise ValueError(
                f"dataset evidence for {dataset} must declare uses {binding.uses!r}"
            )
        manifest_hash = str(raw.get("manifest_hash") or "")
        if not _valid_sha256(manifest_hash):
            raise ValueError(
                f"dataset evidence for {dataset} requires a sha256 manifest_hash"
            )
        if raw.get("corpus_bundled") is not False:
            raise ValueError(f"dataset evidence for {dataset} must not bundle rows")
        if raw.get("restricted_vocabulary_bundled") is not False:
            raise ValueError(
                f"dataset evidence for {dataset} must not bundle restricted vocabulary"
            )
        if binding.user_key_gated and raw.get("user_key_gated") is not True:
            raise ValueError(
                f"dataset evidence for {dataset} must declare user_key_gated=true"
            )

        metrics = _evidence_metrics(raw.get("metrics", {}), dataset=dataset)
        if dataset == "medmentions" and "top1_accuracy" not in metrics:
            raise ValueError("MedMentions evidence requires top1_accuracy")
        if dataset == "redistributable_vocabulary":
            if raw.get("redistributable") is not True:
                raise ValueError(
                    "concept-linking training vocabulary must be redistributable"
                )
            vocab = raw.get("vocab")
            if not isinstance(vocab, str) or not vocab.strip():
                raise ValueError("concept-linking evidence requires a vocabulary id")
            if vocab.strip().casefold() not in CLINICAL_LINK_VOCABULARIES:
                raise ValueError(
                    "concept-linking training vocabulary must be a supported "
                    "redistributable vocabulary"
                )

        item = {
            "access": binding.access,
            "corpus_bundled": False,
            "manifest_hash": manifest_hash,
            "metrics": metrics,
            "restricted_vocabulary_bundled": False,
            "uses": list(uses),
        }
        for key in ("license_id", "redistributable", "user_key_gated", "vocab"):
            if key in raw:
                item[key] = raw[key]
        validated[dataset] = item
    return validated


def assert_no_gated_content_committed(root: str | Path) -> None:
    """Fail if committed dataset payload files contain gated-code markers."""

    root_path = Path(root)
    offenders: list[str] = []
    for path in _iter_payload_files(root_path):
        text = path.read_text(encoding="utf-8", errors="ignore")
        offenders.extend(f"{path}: {marker}" for marker in _gated_markers(path, text))
    if offenders:
        raise AssertionError(
            "gated dataset content must not be committed: " + ", ".join(offenders)
        )


def _load_rows(path: Path) -> list[Mapping[str, Any]]:
    if path.is_dir():
        rows: list[Mapping[str, Any]] = []
        for child in sorted(path.iterdir()):
            if child.suffix.lower() in {".json", ".jsonl", ".ndjson"}:
                rows.extend(_load_rows(child))
        return rows

    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, Mapping):
        rows = (
            raw.get("records")
            or raw.get("documents")
            or raw.get("fixtures")
            or raw.get("examples")
            or []
        )
    else:
        rows = raw
    if not isinstance(rows, list):
        raise ValueError("public dataset payload must contain a list of records")
    return [row for row in rows if isinstance(row, Mapping)]


def _record_text(row: Mapping[str, Any]) -> str:
    if isinstance(row.get("text"), str):
        return str(row["text"])
    parts = [
        str(row[key])
        for key in ("title", "abstract", "note_text")
        if isinstance(row.get(key), str)
    ]
    return " ".join(part for part in parts if part).strip()


def _int_field(data: Mapping[str, Any], *keys: str) -> int:
    for key in keys:
        value = data.get(key)
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.isdigit():
            return int(value)
    raise ValueError(f"span missing integer field from {keys!r}")


def _clean_metadata(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        str(key): _plain_metadata(item)
        for key, item in value.items()
        if str(key).lower() not in _CONTROLLED_METADATA_KEYS
    }


def _plain_metadata(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _clean_metadata(value)
    if isinstance(value, (list, tuple)):
        return [_plain_metadata(item) for item in value]
    return value


def _evidence_uses(value: Any, *, dataset: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"dataset evidence for {dataset} requires uses")
    uses = tuple(str(item) for item in value)
    if len(set(uses)) != len(uses) or set(uses) - {"train", "eval"}:
        raise ValueError(f"dataset evidence for {dataset} has invalid uses")
    return uses


def _evidence_metrics(value: Any, *, dataset: str) -> dict[str, float]:
    if not isinstance(value, Mapping):
        raise TypeError(f"dataset evidence metrics for {dataset} must be a mapping")
    metrics: dict[str, float] = {}
    for key, raw in value.items():
        if (
            not isinstance(key, str)
            or not key
            or isinstance(raw, bool)
            or not isinstance(raw, (int, float))
            or not math.isfinite(float(raw))
        ):
            raise ValueError(
                f"dataset evidence metrics for {dataset} must be finite numbers"
            )
        metrics[key] = float(raw)
    return metrics


def _valid_sha256(value: str) -> bool:
    if not value.startswith(_SHA256_PREFIX):
        return False
    digest = value.removeprefix(_SHA256_PREFIX)
    return len(digest) == 64 and all(
        character in "0123456789abcdef" for character in digest
    )


def _gated_markers(path: Path, text: str) -> tuple[str, ...]:
    folded = text.casefold()
    found = {
        marker for marker in _UNAMBIGUOUS_GATED_MARKERS if marker.casefold() in folded
    }
    structured_values = {
        match.group("marker").casefold()
        for match in _STRUCTURED_MARKER_RE.finditer(text)
    }
    for marker in _STRUCTURED_GATED_MARKERS:
        if marker in text or marker.casefold() in structured_values:
            found.add(marker)
        elif path.suffix.casefold() in {".csv", ".tsv"} and re.search(
            rf"(?i)(?:^|[,\t])\s*{re.escape(marker)}\s*(?:[,\t]|$)", text
        ):
            found.add(marker)
    return tuple(sorted(found))


def _iter_payload_files(root: Path) -> Iterable[Path]:
    if root.is_file():
        if root.suffix.lower() in _DATA_EXTENSIONS:
            yield root
        return
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in _DATA_EXTENSIONS:
            yield path


__all__ = [
    "CLINICAL_FAMILY_DATASET_BINDINGS",
    "CLINICAL_LINK_VOCABULARIES",
    "CLINICAL_MODEL_FAMILIES",
    "ClinicalDatasetBinding",
    "DatasetLoadResult",
    "DatasetUnavailable",
    "PUBLIC_DATASETS",
    "PUBLIC_DATASET_LICENSES",
    "PUBLIC_LABEL_MAPS",
    "PublicDatasetAdapter",
    "PublicDatasetRecord",
    "PublicDatasetSpan",
    "adapter_for",
    "assert_no_gated_content_committed",
    "clinical_family_dataset_bindings",
    "load_public_dataset",
    "map_public_label",
    "validate_clinical_family_dataset_evidence",
]
