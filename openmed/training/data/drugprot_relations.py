"""Build joint relation-head training examples from public DrugProt records."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from openmed.clinical.relations import SpanReference
from openmed.eval.datasets.drugprot import (
    DEFAULT_SPLIT,
    DRUGPROT,
    DRUGPROT_DOI,
    DRUGPROT_RELATION_TYPES,
    DRUGPROT_SOURCE_URL,
    Downloader,
    DrugProtCorpus,
    DrugProtEntity,
    DrugProtRecord,
    DrugProtRelation,
    load_drugprot_corpus,
)
from openmed.eval.datasets.licenses import license_for

DRUGPROT_TO_RELATION_LABEL: Mapping[str, str] = MappingProxyType(
    {
        "ACTIVATOR": "activator",
        "AGONIST": "agonist",
        "AGONIST-ACTIVATOR": "agonist_activator",
        "AGONIST-INHIBITOR": "agonist_inhibitor",
        "ANTAGONIST": "antagonist",
        "DIRECT-REGULATOR": "direct_regulator",
        "INDIRECT-DOWNREGULATOR": "indirect_downregulator",
        "INDIRECT-UPREGULATOR": "indirect_upregulator",
        "INHIBITOR": "inhibitor",
        "PART-OF": "part_of",
        "PRODUCT-OF": "product_of",
        "SUBSTRATE": "substrate",
        "SUBSTRATE_PRODUCT-OF": "substrate_product_of",
    }
)
DRUGPROT_RELATION_LABELS: tuple[str, ...] = tuple(
    DRUGPROT_TO_RELATION_LABEL[source_label] for source_label in DRUGPROT_RELATION_TYPES
)


@dataclass(frozen=True)
class DrugProtRelationExample:
    """One positive or hard-negative example for the joint relation head.

    ``head_span``, ``tail_span``, and ``relation_type`` are the training view.
    The ``head``, ``tail``, and ``type`` aliases expose the section 5.4
    ``Relation(head, type, tail)`` shape used by the clinical relation API.
    ``relation_type`` is ``None`` for a no-relation negative, matching the
    joint-head span-pair convention.
    """

    text: str
    head_span: SpanReference
    tail_span: SpanReference
    relation_type: str | None
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not self.text:
            raise ValueError("DrugProt relation example text must be non-empty")
        if self.head_span.offset_key() == self.tail_span.offset_key():
            raise ValueError("DrugProt relation endpoints must differ")
        for endpoint_name, span in (
            ("head", self.head_span),
            ("tail", self.tail_span),
        ):
            if not 0 <= span.start < span.end <= len(self.text):
                raise ValueError(f"DrugProt {endpoint_name} span offsets are invalid")
            if self.text[span.start : span.end] != span.text:
                raise ValueError(f"DrugProt {endpoint_name} span text does not match")
        if (
            self.relation_type is not None
            and self.relation_type not in DRUGPROT_RELATION_LABELS
        ):
            raise ValueError(
                f"unsupported canonical DrugProt relation label: {self.relation_type!r}"
            )
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @property
    def head(self) -> SpanReference:
        """Return the relation-shape head endpoint."""
        return self.head_span

    @property
    def tail(self) -> SpanReference:
        """Return the relation-shape tail endpoint."""
        return self.tail_span

    @property
    def type(self) -> str | None:
        """Return the relation-shape type, or ``None`` for a negative."""
        return self.relation_type

    @property
    def is_negative(self) -> bool:
        """Return whether this is a sampled no-relation pair."""
        return self.relation_type is None

    def to_relation_dict(self) -> dict[str, Any]:
        """Return the clinical ``Relation(head, type, tail)`` shape."""
        return {
            "head": self.head_span.to_dict(),
            "type": self.relation_type,
            "tail": self.tail_span.to_dict(),
        }

    def to_dict(self) -> dict[str, Any]:
        """Return the complete JSON-ready joint training example."""
        return {
            "text": self.text,
            "head_span": self.head_span.to_dict(),
            "tail_span": self.tail_span.to_dict(),
            "relation_type": self.relation_type,
            "metadata": dict(self.metadata),
        }


def map_drugprot_relation_type(relation_type: str) -> str:
    """Map a DrugProt source relation onto the joint head's label space.

    Args:
        relation_type: DrugProt relation label, case-insensitively.

    Returns:
        The canonical lowercase snake-case relation label.

    Raises:
        ValueError: If ``relation_type`` is not part of the DrugProt schema.
    """
    normalized = relation_type.strip().upper()
    try:
        return DRUGPROT_TO_RELATION_LABEL[normalized]
    except KeyError:
        allowed = ", ".join(DRUGPROT_RELATION_TYPES)
        raise ValueError(
            f"unknown DrugProt relation type {relation_type!r}; expected one of: "
            f"{allowed}"
        ) from None


def build_drugprot_relation_examples(
    corpus: DrugProtCorpus,
    *,
    negatives_per_positive: int = 1,
    seed: int = 0,
) -> tuple[DrugProtRelationExample, ...]:
    """Build positives and deterministic hard negatives from a loaded corpus.

    Hard negatives are chemical-gene pairs from the same abstract whose entity
    IDs do not participate together in any gold relation. This keeps negatives
    type-compatible and co-occurring while excluding mislabeled reverse edges.

    Args:
        corpus: Corpus returned by the public DrugProt dataset adapter.
        negatives_per_positive: Maximum sampled negatives per positive relation
            in each abstract.
        seed: Stable seed used to rank candidate negative pairs.

    Returns:
        Deterministically ordered joint relation examples.

    Raises:
        TypeError: If ``negatives_per_positive`` is not an integer.
        ValueError: If ``negatives_per_positive`` is negative.
    """
    if isinstance(negatives_per_positive, bool) or not isinstance(
        negatives_per_positive, int
    ):
        raise TypeError("negatives_per_positive must be an integer")
    if negatives_per_positive < 0:
        raise ValueError("negatives_per_positive must be non-negative")

    examples: list[DrugProtRelationExample] = []
    for record in corpus.records:
        positives = [
            _positive_example(record, relation) for relation in record.relations
        ]
        examples.extend(positives)
        max_negatives = len(positives) * negatives_per_positive
        examples.extend(
            _negative_examples(record, max_negatives=max_negatives, seed=seed)
        )
    return tuple(examples)


def load_drugprot_relation_examples(
    path: str | Path | None = None,
    *,
    cache_dir: str | Path | None = None,
    downloader: Downloader | None = None,
    split: str = DEFAULT_SPLIT,
    negatives_per_positive: int = 1,
    seed: int = 0,
) -> tuple[DrugProtRelationExample, ...]:
    """Load DrugProt through its public adapter and build relation examples.

    This function delegates all source resolution, download, cache, checksum,
    parsing, and span validation to the existing DrugProt adapter.

    Args:
        path: Local DrugProt directory or archive accepted by the adapter.
        cache_dir: Optional adapter cache directory when ``path`` is omitted.
        downloader: Optional adapter downloader override.
        split: DrugProt corpus split.
        negatives_per_positive: Maximum sampled negatives per positive relation.
        seed: Stable negative-sampling seed.

    Returns:
        Joint positive and hard-negative relation examples.
    """
    corpus = load_drugprot_corpus(
        path,
        cache_dir=cache_dir,
        downloader=downloader,
        split=split,
    )
    return build_drugprot_relation_examples(
        corpus,
        negatives_per_positive=negatives_per_positive,
        seed=seed,
    )


def _positive_example(
    record: DrugProtRecord,
    relation: DrugProtRelation,
) -> DrugProtRelationExample:
    return _example(
        record,
        head=relation.arg1,
        tail=relation.arg2,
        relation_type=map_drugprot_relation_type(relation.relation_type),
        source_relation_type=relation.relation_type,
    )


def _negative_examples(
    record: DrugProtRecord,
    *,
    max_negatives: int,
    seed: int,
) -> tuple[DrugProtRelationExample, ...]:
    if max_negatives == 0:
        return ()

    related_pairs = {
        frozenset((relation.arg1_id, relation.arg2_id)) for relation in record.relations
    }
    chemicals = tuple(
        entity for entity in record.entities if entity.entity_group == "CHEMICAL"
    )
    genes = tuple(entity for entity in record.entities if entity.entity_group == "GENE")
    candidates = [
        (head, tail)
        for head in chemicals
        for tail in genes
        if frozenset((head.entity_id, tail.entity_id)) not in related_pairs
    ]
    ranked = sorted(
        candidates,
        key=lambda pair: (
            _negative_rank(seed, record.pmid, pair[0], pair[1]),
            _pair_key(pair),
        ),
    )
    selected = sorted(ranked[:max_negatives], key=_pair_key)
    return tuple(
        _example(
            record,
            head=head,
            tail=tail,
            relation_type=None,
            source_relation_type=None,
        )
        for head, tail in selected
    )


def _example(
    record: DrugProtRecord,
    *,
    head: DrugProtEntity,
    tail: DrugProtEntity,
    relation_type: str | None,
    source_relation_type: str | None,
) -> DrugProtRelationExample:
    dataset_license = license_for(DRUGPROT)
    return DrugProtRelationExample(
        text=record.text,
        head_span=_span_reference(head),
        tail_span=_span_reference(tail),
        relation_type=relation_type,
        metadata={
            "dataset": DRUGPROT,
            "doi": DRUGPROT_DOI,
            "head_entity_id": head.entity_id,
            "head_source_label": head.source_label,
            "is_negative": relation_type is None,
            "license": dataset_license.to_dict(),
            "public": True,
            "source": "Zenodo",
            "source_pmid": record.pmid,
            "source_relation_type": source_relation_type,
            "source_url": DRUGPROT_SOURCE_URL,
            "split": record.split,
            "tail_entity_id": tail.entity_id,
            "tail_source_label": tail.source_label,
        },
    )


def _span_reference(entity: DrugProtEntity) -> SpanReference:
    return SpanReference(
        text=entity.text,
        label=entity.entity_group.lower(),
        start=entity.start,
        end=entity.end,
        score=1.0,
    )


def _negative_rank(
    seed: int,
    pmid: str,
    head: DrugProtEntity,
    tail: DrugProtEntity,
) -> str:
    payload = f"{seed}\0{pmid}\0{head.entity_id}\0{tail.entity_id}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _pair_key(
    pair: tuple[DrugProtEntity, DrugProtEntity],
) -> tuple[int, int, int, int, str, str]:
    head, tail = pair
    return (
        head.start,
        head.end,
        tail.start,
        tail.end,
        head.entity_id,
        tail.entity_id,
    )


__all__ = [
    "DRUGPROT_RELATION_LABELS",
    "DRUGPROT_TO_RELATION_LABEL",
    "DrugProtRelationExample",
    "build_drugprot_relation_examples",
    "load_drugprot_relation_examples",
    "map_drugprot_relation_type",
]
