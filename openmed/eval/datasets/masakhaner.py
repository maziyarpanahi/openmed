"""Offline-only MasakhaNER benchmark loader for African-language NER.

MasakhaNER corpus rows are never bundled or downloaded by this module. Callers
must supply either a local CoNLL file/tree or an already-populated local cache
and must explicitly accept the applicable dataset-card license.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from openmed.core.labels import DATE, HIPAA_NAME, LOCATION, ORGANIZATION, PERSON
from openmed.eval.data_provenance import DatasetProvenance, build_dataset_provenance
from openmed.eval.harness import (
    BenchmarkFixture,
    ModelRunner,
    run_masakhaner_scorecard,
)
from openmed.eval.metrics import EvalSpan
from openmed.eval.report import BenchmarkReport

MASAKHANER = "masakhaner"
MASAKHANER_VERSION_1 = "1.0"
MASAKHANER_VERSION_2 = "2.0"
DEFAULT_VERSION = MASAKHANER_VERSION_2
DEFAULT_SPLIT = "test"
MASAKHANER_2_LICENSE_ID = "CC-BY-NC-4.0"
MASAKHANER_1_LICENSE_ID = "per-card: CC 4.0 Non-Commercial"
MASAKHANER_2_REPOSITORY = "masakhane/masakhaner2"
MASAKHANER_1_REPOSITORY = "masakhane/masakhaner"
MASAKHANER_2_LICENSE_SOURCE_URL = (
    "https://github.com/masakhane-io/masakhane-ner#license-information"
)
MASAKHANER_2_LICENSE_NOTICE = (
    "The upstream repository and dataset-card prose declare the corpus "
    "non-commercial, while the current Hugging Face card metadata tags it "
    "AFL-3.0. OpenMed applies the more restrictive CC-BY-NC-4.0 gate; users "
    "must verify the upstream terms for their use case."
)
MASAKHANER_ORG_HANDLING = (
    "ORG spans map to OpenMed ORGANIZATION for exact NER scoring. This preserves "
    "the source annotation and does not imply that every organization mention is PHI."
)

MASAKHANER_LABEL_MAPPING: Mapping[str, str] = {
    "DATE": DATE,
    "LOC": LOCATION,
    "ORG": ORGANIZATION,
    "PER": HIPAA_NAME,
}
MASAKHANER_CANONICAL_LABEL_MAPPING: Mapping[str, str] = {
    **MASAKHANER_LABEL_MAPPING,
    "PER": PERSON,
}

_V2_LANGUAGE_NAMES: Mapping[str, str] = {
    "bam": "Bambara",
    "bbj": "Ghomala",
    "ewe": "Ewe",
    "fon": "Fon",
    "hau": "Hausa",
    "ibo": "Igbo",
    "kin": "Kinyarwanda",
    "lug": "Luganda",
    "luo": "Dholuo",
    "mos": "Mossi",
    "nya": "Chichewa",
    "pcm": "Nigerian Pidgin",
    "sna": "chiShona",
    "swa": "Kiswahili",
    "tsn": "Setswana",
    "twi": "Twi",
    "wol": "Wolof",
    "xho": "isiXhosa",
    "yor": "Yoruba",
    "zul": "isiZulu",
}
_V1_LANGUAGE_NAMES: Mapping[str, str] = {
    "amh": "Amharic",
    "hau": "Hausa",
    "ibo": "Igbo",
    "kin": "Kinyarwanda",
    "lug": "Luganda",
    "luo": "Dholuo",
    "pcm": "Nigerian Pidgin",
    "swa": "Kiswahili",
    "wol": "Wolof",
    "yor": "Yoruba",
}

MASAKHANER_LANGUAGES: tuple[str, ...] = tuple(_V2_LANGUAGE_NAMES)
MASAKHANER_1_LANGUAGES: tuple[str, ...] = tuple(_V1_LANGUAGE_NAMES)
_CONLL_SUFFIXES = (".bio", ".conll", ".iob", ".txt", ".tsv")


class MasakhaNerLicenseRequired(PermissionError):
    """Raised when corpus loading is attempted without license acceptance."""


class MasakhaNerCorpusRequired(FileNotFoundError):
    """Raised when no user-supplied local MasakhaNER source can be resolved."""


@dataclass(frozen=True)
class MasakhaNerSource:
    """Metadata for one MasakhaNER language configuration."""

    language: str
    display_name: str
    version: str
    repository: str
    license_id: str
    license_notice: str = ""
    license_source_url: str = ""
    label_mapping: Mapping[str, str] = field(
        default_factory=lambda: dict(MASAKHANER_LABEL_MAPPING)
    )

    @property
    def source_url(self) -> str:
        """Return the canonical dataset-card URL."""

        return f"https://huggingface.co/datasets/{self.repository}"

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-ready source metadata without corpus rows."""

        return {
            "display_name": self.display_name,
            "label_mapping": dict(sorted(self.label_mapping.items())),
            "language": self.language,
            "license_id": self.license_id,
            "license_notice": self.license_notice,
            "license_source_url": self.license_source_url,
            "organization_handling": MASAKHANER_ORG_HANDLING,
            "repository": self.repository,
            "source_url": self.source_url,
            "version": self.version,
        }


@dataclass(frozen=True)
class MasakhaNerSpan:
    """One MasakhaNER mention with exact reconstructed-text offsets."""

    start: int
    end: int
    source_label: str
    canonical_label: str
    text: str

    def to_eval_span(self, *, language: str) -> EvalSpan:
        """Convert this source mention to the shared eval span schema."""

        return EvalSpan(
            start=self.start,
            end=self.end,
            label=self.canonical_label,
            text=self.text,
            language=language,
            metadata={
                "canonical_label": self.canonical_label,
                "policy_label": MASAKHANER_LABEL_MAPPING[self.source_label],
                "source_label": self.source_label,
            },
        )


@dataclass(frozen=True)
class MasakhaNerRecord:
    """One reconstructed MasakhaNER CoNLL sentence."""

    record_id: str
    language: str
    text: str
    spans: tuple[MasakhaNerSpan, ...]
    split: str = DEFAULT_SPLIT
    version: str = DEFAULT_VERSION
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_benchmark_fixture(self) -> BenchmarkFixture:
        """Expose the sentence through the standard eval harness schema."""

        return BenchmarkFixture(
            fixture_id=self.record_id,
            text=self.text,
            gold_spans=tuple(
                span.to_eval_span(language=self.language) for span in self.spans
            ),
            language=self.language,
            metadata={
                **dict(self.metadata),
                "benchmark": MASAKHANER,
                "dataset": MASAKHANER,
                "language": self.language,
                "split": self.split,
                "suite": MASAKHANER,
                "task": "ner",
                "version": self.version,
            },
        )


@dataclass(frozen=True)
class MasakhaNerCorpus:
    """Loaded MasakhaNER split plus license and content provenance."""

    language: str
    records: tuple[MasakhaNerRecord, ...]
    split: str
    version: str
    source_path: str
    source_kind: str
    provenance: DatasetProvenance

    @property
    def fixture_count(self) -> int:
        """Return the number of parsed sentences."""

        return len(self.records)

    def to_benchmark_fixtures(self) -> list[BenchmarkFixture]:
        """Return standard harness fixtures with source provenance attached."""

        provenance = self.provenance.to_dict()
        fixtures: list[BenchmarkFixture] = []
        for record in self.records:
            fixture = record.to_benchmark_fixture()
            fixtures.append(
                BenchmarkFixture(
                    fixture_id=fixture.fixture_id,
                    text=fixture.text,
                    gold_spans=fixture.gold_spans,
                    language=fixture.language,
                    metadata={
                        **dict(fixture.metadata),
                        "content_hash": self.provenance.content_hash,
                        "license_id": self.provenance.license_id,
                        "provenance": provenance,
                        "source_hash": self.provenance.source_hash,
                        "source_kind": self.source_kind,
                    },
                )
            )
        return fixtures


def _build_sources(
    names: Mapping[str, str],
    *,
    version: str,
    repository: str,
    license_id: str,
    license_notice: str = "",
    license_source_url: str = "",
) -> dict[str, MasakhaNerSource]:
    return {
        language: MasakhaNerSource(
            language=language,
            display_name=display_name,
            version=version,
            repository=repository,
            license_id=license_id,
            license_notice=license_notice,
            license_source_url=license_source_url,
        )
        for language, display_name in names.items()
    }


MASAKHANER_SOURCES: Mapping[str, MasakhaNerSource] = _build_sources(
    _V2_LANGUAGE_NAMES,
    version=MASAKHANER_VERSION_2,
    repository=MASAKHANER_2_REPOSITORY,
    license_id=MASAKHANER_2_LICENSE_ID,
    license_notice=MASAKHANER_2_LICENSE_NOTICE,
    license_source_url=MASAKHANER_2_LICENSE_SOURCE_URL,
)
MASAKHANER_1_SOURCES: Mapping[str, MasakhaNerSource] = _build_sources(
    _V1_LANGUAGE_NAMES,
    version=MASAKHANER_VERSION_1,
    repository=MASAKHANER_1_REPOSITORY,
    license_id=MASAKHANER_1_LICENSE_ID,
)


def masakhaner_source_for(
    language: str,
    *,
    version: str = DEFAULT_VERSION,
) -> MasakhaNerSource:
    """Return source metadata for a language and corpus version."""

    normalized_version = _normalize_version(version)
    sources = (
        MASAKHANER_SOURCES
        if normalized_version == MASAKHANER_VERSION_2
        else MASAKHANER_1_SOURCES
    )
    key = str(language).strip().lower()
    try:
        return sources[key]
    except KeyError as exc:
        allowed = ", ".join(sources)
        raise ValueError(
            f"unknown MasakhaNER {normalized_version} language {language!r}: {allowed}"
        ) from exc


def map_masakhaner_label(label: str) -> str:
    """Map a MasakhaNER PER/ORG/LOC/DATE label to OpenMed taxonomy."""

    source_label = str(label).strip().upper()
    try:
        return MASAKHANER_CANONICAL_LABEL_MAPPING[source_label]
    except KeyError as exc:
        allowed = ", ".join(sorted(MASAKHANER_CANONICAL_LABEL_MAPPING))
        raise ValueError(
            f"unknown MasakhaNER entity label {label!r}: {allowed}"
        ) from exc


def masakhaner_suite_metadata(
    *,
    languages: Sequence[str] | None = None,
    split: str = DEFAULT_SPLIT,
    version: str = DEFAULT_VERSION,
) -> dict[str, Any]:
    """Return corpus, license, and language metadata without loading rows."""

    normalized_version = _normalize_version(version)
    selected_languages = _normalize_languages(
        (
            _languages_for_version(normalized_version)
            if languages is None
            else languages
        ),
        version=normalized_version,
    )
    if not selected_languages:
        raise ValueError("MasakhaNER metadata requires at least one language")
    sources = [
        masakhaner_source_for(language, version=normalized_version)
        for language in selected_languages
    ]
    license_ids = sorted({source.license_id for source in sources})
    return {
        "access": (
            "explicit license acceptance and a user-supplied local path or "
            "pre-populated cache are required"
        ),
        "canonical_label_mapping": dict(
            sorted(MASAKHANER_CANONICAL_LABEL_MAPPING.items())
        ),
        "label_mapping": dict(sorted(MASAKHANER_LABEL_MAPPING.items())),
        "languages": [source.language for source in sources],
        "license_id": license_ids[0] if len(license_ids) == 1 else license_ids,
        "license_notice": next(
            (source.license_notice for source in sources if source.license_notice),
            "",
        ),
        "organization_handling": MASAKHANER_ORG_HANDLING,
        "redistribution": "user-supplied only; never bundled or redistributed",
        "sources": {source.language: source.to_dict() for source in sources},
        "split": _normalize_split(split),
        "suite": MASAKHANER,
        "task": "ner",
        "version": normalized_version,
    }


def load_masakhaner_corpus(
    language: str,
    path: str | Path | None = None,
    *,
    cache_dir: str | Path | None = None,
    split: str = DEFAULT_SPLIT,
    version: str = DEFAULT_VERSION,
    accept_license: bool = False,
) -> MasakhaNerCorpus:
    """Load one user-supplied MasakhaNER CoNLL split without network access.

    Args:
        language: MasakhaNER three-letter language configuration.
        path: Explicit local CoNLL file or corpus directory.
        cache_dir: Explicit pre-populated local cache directory. The loader
            only reads original CoNLL files already present there.
        split: ``train``, ``validation``/``dev``, or ``test``.
        version: MasakhaNER ``2.0`` (default) or ``1.0``.
        accept_license: Must be true to acknowledge the dataset-card license.

    Raises:
        MasakhaNerLicenseRequired: If ``accept_license`` is false.
        MasakhaNerCorpusRequired: If no readable local split can be resolved.
    """

    normalized_version = _normalize_version(version)
    normalized_split = _normalize_split(split)
    source = masakhaner_source_for(language, version=normalized_version)
    _require_license_acceptance(source, accepted=accept_license)
    source_path, source_kind = _resolve_source_path(
        language=source.language,
        path=path,
        cache_dir=cache_dir,
        split=normalized_split,
        version=normalized_version,
    )
    provenance = build_dataset_provenance(
        dataset_id=MASAKHANER,
        license_id=source.license_id,
        source=source.source_url,
        content_path=source_path,
        version=normalized_version,
        split=normalized_split,
        languages=(source.language,),
    )
    records = records_from_masakhaner_conll(
        source_path.read_text(encoding="utf-8"),
        language=source.language,
        split=normalized_split,
        version=normalized_version,
    )
    if not records:
        raise ValueError(f"MasakhaNER source contains no CoNLL records: {source_path}")
    return MasakhaNerCorpus(
        language=source.language,
        records=records,
        split=normalized_split,
        version=normalized_version,
        source_path=str(source_path),
        source_kind=source_kind,
        provenance=provenance,
    )


def load_masakhaner_fixtures(
    paths: Mapping[str, str | Path] | str | Path | None = None,
    *,
    cache_dir: str | Path | None = None,
    languages: Sequence[str] | None = None,
    split: str = DEFAULT_SPLIT,
    version: str = DEFAULT_VERSION,
    accept_license: bool = False,
) -> list[BenchmarkFixture]:
    """Load selected language splits as standard eval harness fixtures."""

    normalized_version = _normalize_version(version)
    selected_languages = _normalize_languages(
        (
            _languages_for_version(normalized_version)
            if languages is None
            else languages
        ),
        version=normalized_version,
    )
    if not selected_languages:
        raise ValueError("MasakhaNER loading requires at least one language")
    normalized_paths: dict[str, str | Path] | None = None
    if isinstance(paths, Mapping):
        normalized_paths = {}
        for language, language_path in paths.items():
            normalized_language = masakhaner_source_for(
                language,
                version=normalized_version,
            ).language
            if normalized_language in normalized_paths:
                raise ValueError(
                    "duplicate MasakhaNER path after language normalization: "
                    f"{normalized_language}"
                )
            normalized_paths[normalized_language] = language_path
    elif paths is not None and Path(paths).expanduser().is_file():
        if len(selected_languages) != 1:
            raise ValueError(
                "a single MasakhaNER CoNLL file requires exactly one language"
            )

    fixtures: list[BenchmarkFixture] = []
    for language in selected_languages:
        language_path = (
            normalized_paths.get(language) if normalized_paths is not None else paths
        )
        corpus = load_masakhaner_corpus(
            language,
            language_path,
            cache_dir=cache_dir,
            split=split,
            version=normalized_version,
            accept_license=accept_license,
        )
        fixtures.extend(corpus.to_benchmark_fixtures())
    _validate_unique_fixture_ids(fixtures)
    return fixtures


def records_from_masakhaner_conll(
    content: str,
    *,
    language: str,
    split: str = DEFAULT_SPLIT,
    version: str = DEFAULT_VERSION,
) -> tuple[MasakhaNerRecord, ...]:
    """Parse MasakhaNER CoNLL content into exact-offset sentence records."""

    source = masakhaner_source_for(language, version=version)
    normalized_split = _normalize_split(split)
    sentences: list[tuple[list[str], list[str]]] = []
    tokens: list[str] = []
    tags: list[str] = []
    for line_number, line in enumerate(content.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            if tokens:
                sentences.append((tokens, tags))
                tokens, tags = [], []
            continue
        if stripped.startswith("#"):
            continue
        parts = stripped.split()
        if len(parts) < 2:
            raise ValueError(
                f"invalid MasakhaNER CoNLL row at line {line_number}: "
                "expected token and BIO tag"
            )
        tokens.append(parts[0])
        tags.append(parts[-1])
    if tokens:
        sentences.append((tokens, tags))

    records: list[MasakhaNerRecord] = []
    for index, (sentence_tokens, sentence_tags) in enumerate(sentences, start=1):
        text, offsets = _reconstruct_text(sentence_tokens)
        spans = _spans_from_bio(
            text,
            offsets,
            sentence_tags,
            record_number=index,
        )
        records.append(
            MasakhaNerRecord(
                record_id=(
                    f"masakhaner-{source.version}-{source.language}-"
                    f"{normalized_split}-{index:06d}"
                ),
                language=source.language,
                text=text,
                spans=spans,
                split=normalized_split,
                version=source.version,
                metadata={
                    "display_name": source.display_name,
                    "license_id": source.license_id,
                    "organization_handling": MASAKHANER_ORG_HANDLING,
                    "source_format": "conll-bio",
                    "source_url": source.source_url,
                },
            )
        )
    return tuple(records)


def run_masakhaner_benchmark(
    fixtures: Sequence[BenchmarkFixture],
    *,
    model_name: str,
    device: str = "cpu",
    runner: ModelRunner | None = None,
    languages: Sequence[str] | None = None,
    checkpoint_path: str | Path | None = None,
) -> BenchmarkReport:
    """Run MasakhaNER as an inference-only per-language NER scorecard."""

    return run_masakhaner_scorecard(
        fixtures,
        model_name=model_name,
        device=device,
        runner=runner,
        languages=languages,
        checkpoint_path=checkpoint_path,
    )


def _require_license_acceptance(
    source: MasakhaNerSource,
    *,
    accepted: bool,
) -> None:
    if accepted:
        return
    if source.version == MASAKHANER_VERSION_2:
        detail = "CC-BY-NC-4.0 restriction"
    else:
        detail = (
            "per-card license for MasakhaNER 1.0 (declared as CC 4.0 Non-Commercial)"
        )
    raise MasakhaNerLicenseRequired(
        f"MasakhaNER {source.version} loading requires explicit acceptance of "
        f"the {detail}; review {source.source_url} and pass accept_license=True. "
        "OpenMed does not provide commercial-use legal guidance."
        + (
            f" License provenance note: {source.license_notice}"
            if source.license_notice
            else ""
        )
    )


def _resolve_source_path(
    *,
    language: str,
    path: str | Path | None,
    cache_dir: str | Path | None,
    split: str,
    version: str,
) -> tuple[Path, str]:
    if path is not None and cache_dir is not None:
        raise ValueError("pass either path or cache_dir, not both")
    if path is None and cache_dir is None:
        raise MasakhaNerCorpusRequired(
            f"MasakhaNER {version} {language} requires a user-supplied local "
            "path or pre-populated cache; network download is disabled"
        )
    root = Path(path if path is not None else cache_dir).expanduser()
    source_kind = "local-path" if path is not None else "pre-populated-hf-cache"
    if not root.exists():
        raise MasakhaNerCorpusRequired(
            f"MasakhaNER {source_kind} does not exist: {root}"
        )
    if root.is_file():
        if root.suffix.lower() not in _CONLL_SUFFIXES:
            raise MasakhaNerCorpusRequired(
                f"MasakhaNER source must be a CoNLL text file, got: {root}"
            )
        return root.resolve(), source_kind

    split_names = _split_file_names(split)
    version_dir = "MasakhaNER2.0" if version == MASAKHANER_VERSION_2 else "data"
    candidates = (
        [root / language / file_name for file_name in split_names]
        + [
            root / version_dir / "data" / language / file_name
            for file_name in split_names
        ]
        + [root / "data" / language / file_name for file_name in split_names]
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve(), source_kind

    allowed_stems = {Path(file_name).stem for file_name in split_names}
    recursive_matches = [
        child
        for child in sorted(root.rglob("*"))
        if child.is_file()
        and child.suffix.lower() in _CONLL_SUFFIXES
        and child.stem in allowed_stems
        and (language in child.parts or root.name == language)
    ]
    if len(recursive_matches) == 1:
        return recursive_matches[0].resolve(), source_kind
    if len(recursive_matches) > 1:
        matches = ", ".join(str(item) for item in recursive_matches[:5])
        raise MasakhaNerCorpusRequired(
            f"multiple MasakhaNER {language} {split} files found; pass one "
            f"explicit path: {matches}"
        )
    raise MasakhaNerCorpusRequired(
        f"no pre-existing CoNLL split found for MasakhaNER {version} "
        f"{language}/{split} under {root}; network download is disabled"
    )


def _split_file_names(split: str) -> tuple[str, ...]:
    stems = ("dev", "validation") if split == "validation" else (split,)
    return tuple(f"{stem}{suffix}" for stem in stems for suffix in _CONLL_SUFFIXES)


def _reconstruct_text(tokens: Sequence[str]) -> tuple[str, tuple[tuple[int, int], ...]]:
    parts: list[str] = []
    offsets: list[tuple[int, int]] = []
    cursor = 0
    for token in tokens:
        if not token:
            raise ValueError("MasakhaNER CoNLL tokens must be non-empty")
        if parts:
            cursor += 1
        start = cursor
        parts.append(token)
        cursor += len(token)
        offsets.append((start, cursor))
    return " ".join(parts), tuple(offsets)


def _spans_from_bio(
    text: str,
    offsets: Sequence[tuple[int, int]],
    tags: Sequence[str],
    *,
    record_number: int,
) -> tuple[MasakhaNerSpan, ...]:
    if len(offsets) != len(tags):
        raise ValueError("MasakhaNER token/tag lengths do not match")
    spans: list[MasakhaNerSpan] = []
    active_label: str | None = None
    active_start = 0
    active_end = 0

    def flush() -> None:
        nonlocal active_label
        if active_label is None:
            return
        spans.append(
            MasakhaNerSpan(
                start=active_start,
                end=active_end,
                source_label=active_label,
                canonical_label=map_masakhaner_label(active_label),
                text=text[active_start:active_end],
            )
        )
        active_label = None

    for token_index, (tag, (start, end)) in enumerate(zip(tags, offsets, strict=True)):
        normalized = tag.strip().upper()
        if normalized == "O":
            flush()
            continue
        if "-" not in normalized:
            raise ValueError(
                f"invalid BIO tag {tag!r} in record {record_number}, "
                f"token {token_index + 1}"
            )
        prefix, source_label = normalized.split("-", 1)
        map_masakhaner_label(source_label)
        if prefix == "B":
            flush()
            active_label = source_label
            active_start = start
            active_end = end
        elif prefix == "I":
            if active_label != source_label:
                raise ValueError(
                    f"invalid BIO continuation {tag!r} in record {record_number}, "
                    f"token {token_index + 1}"
                )
            active_end = end
        else:
            raise ValueError(
                f"unsupported BIO prefix {prefix!r} in record {record_number}, "
                f"token {token_index + 1}"
            )
    flush()
    return tuple(spans)


def _normalize_version(version: str) -> str:
    normalized = str(version).strip().lower().removeprefix("v")
    if normalized in {"1", "1.0"}:
        return MASAKHANER_VERSION_1
    if normalized in {"2", "2.0"}:
        return MASAKHANER_VERSION_2
    raise ValueError("MasakhaNER version must be '1.0' or '2.0'")


def _languages_for_version(version: str) -> tuple[str, ...]:
    return (
        MASAKHANER_LANGUAGES
        if version == MASAKHANER_VERSION_2
        else MASAKHANER_1_LANGUAGES
    )


def _normalize_languages(
    languages: Sequence[str] | str,
    *,
    version: str,
) -> tuple[str, ...]:
    values = (languages,) if isinstance(languages, str) else languages
    return tuple(
        dict.fromkeys(
            masakhaner_source_for(language, version=version).language
            for language in values
        )
    )


def _normalize_split(split: str) -> str:
    normalized = str(split).strip().lower()
    if normalized == "dev":
        return "validation"
    if normalized not in {"train", "validation", "test"}:
        raise ValueError("MasakhaNER split must be train, validation/dev, or test")
    return normalized


def _validate_unique_fixture_ids(fixtures: Sequence[BenchmarkFixture]) -> None:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for fixture in fixtures:
        if fixture.fixture_id in seen:
            duplicates.add(fixture.fixture_id)
        seen.add(fixture.fixture_id)
    if duplicates:
        raise ValueError(
            f"duplicate MasakhaNER fixture ids: {', '.join(sorted(duplicates))}"
        )


__all__ = [
    "DEFAULT_SPLIT",
    "DEFAULT_VERSION",
    "MASAKHANER",
    "MASAKHANER_1_LANGUAGES",
    "MASAKHANER_1_LICENSE_ID",
    "MASAKHANER_1_SOURCES",
    "MASAKHANER_2_LICENSE_ID",
    "MASAKHANER_2_LICENSE_NOTICE",
    "MASAKHANER_2_LICENSE_SOURCE_URL",
    "MASAKHANER_CANONICAL_LABEL_MAPPING",
    "MASAKHANER_LABEL_MAPPING",
    "MASAKHANER_LANGUAGES",
    "MASAKHANER_ORG_HANDLING",
    "MASAKHANER_SOURCES",
    "MASAKHANER_VERSION_1",
    "MASAKHANER_VERSION_2",
    "MasakhaNerCorpus",
    "MasakhaNerCorpusRequired",
    "MasakhaNerLicenseRequired",
    "MasakhaNerRecord",
    "MasakhaNerSource",
    "MasakhaNerSpan",
    "load_masakhaner_corpus",
    "load_masakhaner_fixtures",
    "map_masakhaner_label",
    "masakhaner_source_for",
    "masakhaner_suite_metadata",
    "records_from_masakhaner_conll",
    "run_masakhaner_benchmark",
]
