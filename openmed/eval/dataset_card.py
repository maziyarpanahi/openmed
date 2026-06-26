"""Dataset cards for registered evaluation suites."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from openmed.eval.datasets.licenses import DatasetLicense, license_for
from openmed.eval.golden import GOLDEN_CATEGORIES, load_golden_fixtures
from openmed.eval.suites import DRUGPROT, GOLDEN, SHIELD, suite_metadata

DATASET_CARD_SUITES: tuple[str, ...] = (GOLDEN, SHIELD, DRUGPROT)
_EXTERNAL_SUITES = {SHIELD, DRUGPROT}


@dataclass(frozen=True)
class DatasetCard:
    """Provenance, license, and fixture summary for one eval dataset."""

    dataset: str
    source_url: str
    license_id: str
    redistribution: str
    record_count: int
    labels: tuple[str, ...] = field(default_factory=tuple)
    languages: tuple[str, ...] = field(default_factory=tuple)
    splits: tuple[str, ...] = field(default_factory=tuple)
    provenance: str = ""
    notes: str = ""
    schema_version: str = "openmed.eval.dataset_card.v1"

    def __post_init__(self) -> None:
        object.__setattr__(self, "labels", _stable_unique(self.labels))
        object.__setattr__(self, "languages", _stable_unique(self.languages))
        object.__setattr__(self, "splits", _stable_unique(self.splits))
        if self.record_count < 0:
            raise ValueError("record_count must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-ready representation with no row text."""

        return {
            "dataset": self.dataset,
            "labels": list(self.labels),
            "languages": list(self.languages),
            "license_id": self.license_id,
            "notes": self.notes,
            "provenance": self.provenance,
            "record_count": self.record_count,
            "redistribution": self.redistribution,
            "schema_version": self.schema_version,
            "source_url": self.source_url,
            "splits": list(self.splits),
        }

    def to_json(self) -> str:
        """Return byte-stable JSON for identical card contents."""

        return (
            json.dumps(
                self.to_dict(),
                ensure_ascii=True,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )

    def to_markdown(self) -> str:
        """Render a deterministic Markdown dataset card."""

        rows = [
            ("Dataset", self.dataset),
            ("Source URL", self.source_url),
            ("License", self.license_id),
            ("Redistribution", self.redistribution),
            ("Record count", str(self.record_count)),
            ("Labels", _markdown_list(self.labels)),
            ("Languages", _markdown_list(self.languages)),
            ("Splits", _markdown_list(self.splits)),
            ("Provenance", self.provenance),
            ("Notes", self.notes),
        ]
        lines = [
            f"# Dataset Card: {self.dataset}",
            "",
            "| Field | Value |",
            "| --- | --- |",
        ]
        lines.extend(f"| {field} | {_markdown_cell(value)} |" for field, value in rows)
        return "\n".join(lines) + "\n"


def build_dataset_card(suite: str, **loader_kwargs: Any) -> DatasetCard:
    """Build a dataset card for a concrete registered eval suite.

    External corpora are summarized from metadata by default so this function
    does not download rows. Pass suite loader keyword arguments, such as a
    DrugProt ``path`` or SHIELD ``rows_loader``, to include fixture counts from
    an explicitly supplied source.
    """

    if suite == GOLDEN:
        return _golden_card(**loader_kwargs)
    if suite == SHIELD:
        return _shield_card(**loader_kwargs)
    if suite == DRUGPROT:
        return _drugprot_card(**loader_kwargs)
    allowed = ", ".join(DATASET_CARD_SUITES)
    raise ValueError(
        f"unknown dataset card suite {suite!r}; expected one of: {allowed}"
    )


def build_all_dataset_cards(
    suites: Sequence[str] | None = None,
    *,
    suite_kwargs: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[DatasetCard, ...]:
    """Build dataset cards for all concrete registered eval suites."""

    selected = DATASET_CARD_SUITES if suites is None else tuple(suites)
    kwargs_by_suite = suite_kwargs or {}
    return tuple(
        build_dataset_card(suite, **dict(kwargs_by_suite.get(suite, {})))
        for suite in selected
    )


def render_dataset_card_markdown(card: DatasetCard) -> str:
    """Render a dataset card as deterministic Markdown."""

    return card.to_markdown()


def _golden_card(**loader_kwargs: Any) -> DatasetCard:
    fixtures = load_golden_fixtures(**loader_kwargs)
    license_metadata = license_for(GOLDEN)
    labels = {span.label for fixture in fixtures for span in fixture.gold_spans}
    languages = {fixture.language for fixture in fixtures}
    return _card_from_license(
        license_metadata,
        record_count=len(fixtures),
        labels=labels,
        languages=languages,
        splits=GOLDEN_CATEGORIES,
        provenance="committed synthetic golden fixtures",
    )


def _shield_card(**loader_kwargs: Any) -> DatasetCard:
    metadata = suite_metadata(SHIELD, use_sample=loader_kwargs.get("use_sample", True))
    fixtures = _load_external_fixtures(SHIELD, loader_kwargs)
    labels = _labels_from_mapping(metadata.get("label_mapping"))
    languages = _fixture_languages(fixtures, fallback=("en",))
    splits = _fixture_splits(fixtures, fallback=(str(metadata["split"]),))
    return _card_from_license(
        license_for(SHIELD),
        record_count=len(fixtures),
        labels=labels,
        languages=languages,
        splits=splits,
        provenance=str(metadata["source_url"]),
        notes=str(metadata["annotation"]),
    )


def _drugprot_card(**loader_kwargs: Any) -> DatasetCard:
    metadata = suite_metadata(DRUGPROT, task=loader_kwargs.get("task", "ner"))
    fixtures = _load_external_fixtures(DRUGPROT, loader_kwargs)
    labels = _labels_from_mapping(metadata.get("entity_label_mapping"))
    languages = _fixture_languages(fixtures, fallback=("en",))
    splits = _fixture_splits(
        fixtures,
        fallback=(str(loader_kwargs.get("split", "training")),),
    )
    return _card_from_license(
        license_for(DRUGPROT),
        record_count=len(fixtures),
        labels=labels,
        languages=languages,
        splits=splits,
        provenance=str(metadata["doi"]),
        notes=str(metadata["redistribution"]),
    )


def _card_from_license(
    license_metadata: DatasetLicense,
    *,
    record_count: int,
    labels: Iterable[str],
    languages: Iterable[str],
    splits: Iterable[str],
    provenance: str,
    notes: str = "",
) -> DatasetCard:
    return DatasetCard(
        dataset=license_metadata.dataset,
        source_url=license_metadata.source_url,
        license_id=license_metadata.license_id,
        redistribution=license_metadata.redistribution,
        record_count=record_count,
        labels=tuple(labels),
        languages=tuple(languages),
        splits=tuple(splits),
        provenance=provenance,
        notes=notes or license_metadata.notes,
    )


def _load_external_fixtures(suite: str, loader_kwargs: Mapping[str, Any]) -> list[Any]:
    if not loader_kwargs:
        return []
    if suite in _EXTERNAL_SUITES and not _has_explicit_source(loader_kwargs):
        return []

    from openmed.eval.suites import load_suite_fixtures

    return list(load_suite_fixtures(suite, **dict(loader_kwargs)))


def _has_explicit_source(loader_kwargs: Mapping[str, Any]) -> bool:
    explicit_keys = {"path", "rows_loader", "downloader"}
    return any(loader_kwargs.get(key) is not None for key in explicit_keys)


def _labels_from_mapping(value: Any) -> tuple[str, ...]:
    if not isinstance(value, Mapping):
        return ()
    return _stable_unique(str(label) for label in value.values() if str(label))


def _fixture_languages(
    fixtures: Sequence[Any],
    *,
    fallback: Iterable[str],
) -> tuple[str, ...]:
    languages = {
        str(getattr(fixture, "language", ""))
        for fixture in fixtures
        if str(getattr(fixture, "language", ""))
    }
    return _stable_unique(languages or fallback)


def _fixture_splits(
    fixtures: Sequence[Any],
    *,
    fallback: Iterable[str],
) -> tuple[str, ...]:
    splits = set()
    for fixture in fixtures:
        metadata = getattr(fixture, "metadata", None)
        if isinstance(metadata, Mapping) and metadata.get("split") is not None:
            splits.add(str(metadata["split"]))
    return _stable_unique(splits or fallback)


def _stable_unique(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(
        sorted({str(value) for value in values if value is not None and str(value)})
    )


def _markdown_list(values: Sequence[str]) -> str:
    if not values:
        return ""
    return ", ".join(f"`{value}`" for value in values)


def _markdown_cell(value: str) -> str:
    cleaned = value.replace("\n", " ").replace("|", "\\|").strip()
    return cleaned or "not declared"


__all__ = [
    "DATASET_CARD_SUITES",
    "DatasetCard",
    "build_all_dataset_cards",
    "build_dataset_card",
    "render_dataset_card_markdown",
]
