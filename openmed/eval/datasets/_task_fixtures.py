"""Small task-view records used by gated evaluation adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from openmed.eval.metrics import EvalSpan
from openmed.eval.relation_metrics import EvalRelation


@dataclass(frozen=True)
class RelationTaskFixture:
    """A document with typed entity spans and directed gold relations."""

    fixture_id: str
    text: str
    entities: Mapping[str, EvalSpan]
    gold_relations: tuple[EvalRelation, ...]
    language: str = "en"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def gold_spans(self) -> tuple[EvalSpan, ...]:
        """Return relation arguments in deterministic source order."""

        return tuple(
            span
            for _, span in sorted(
                self.entities.items(),
                key=lambda item: (item[1].start, item[1].end, item[0]),
            )
        )

    @property
    def relations(self) -> tuple[EvalRelation, ...]:
        """Expose the same gold relations under the legacy adapter name."""

        return self.gold_relations

    @property
    def gold_tlinks(self) -> tuple[EvalRelation, ...]:
        """Expose temporal relations under the THYME terminology."""

        return self.gold_relations

    @property
    def task(self) -> str:
        """Return the broad task name from fixture metadata."""

        return str(self.metadata.get("task", ""))

    @property
    def view(self) -> str:
        """Return the detailed task view from fixture metadata."""

        return self.task_view

    @property
    def task_view(self) -> str:
        """Return the stable task-view identifier in fixture metadata."""

        return str(self.metadata.get("task_view") or self.metadata.get("task", ""))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready fixture representation."""

        return {
            "entities": [
                {"id": entity_id, **_span_to_dict(span)}
                for entity_id, span in sorted(self.entities.items())
            ],
            "fixture_id": self.fixture_id,
            "gold_relations": [relation.to_dict() for relation in self.gold_relations],
            "language": self.language,
            "metadata": dict(self.metadata),
            "text": self.text,
        }


@dataclass(frozen=True)
class SentencePairFixture:
    """One sentence-pair classification record for NLI evaluation."""

    fixture_id: str
    premise: str
    hypothesis: str
    label: str
    language: str = "en"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def sentence1(self) -> str:
        """Return the premise under the common MedNLI field name."""

        return self.premise

    @property
    def sentence2(self) -> str:
        """Return the hypothesis under the common MedNLI field name."""

        return self.hypothesis

    @property
    def gold_label(self) -> str:
        """Return the normalized gold NLI label."""

        return self.label

    @property
    def task_view(self) -> str:
        """Return the stable task-view identifier in fixture metadata."""

        return str(self.metadata.get("task_view") or self.metadata.get("task", ""))

    @property
    def task(self) -> str:
        """Return the broad task name from fixture metadata."""

        return str(self.metadata.get("task", ""))

    @property
    def view(self) -> str:
        """Return the detailed task view from fixture metadata."""

        return self.task_view

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready sentence-pair representation."""

        return {
            "fixture_id": self.fixture_id,
            "gold_label": self.label,
            "hypothesis": self.hypothesis,
            "language": self.language,
            "metadata": dict(self.metadata),
            "premise": self.premise,
        }


@dataclass(frozen=True)
class DocumentSummaryFixture:
    """One source document and reference summary for summarization eval."""

    fixture_id: str
    document: str
    summary: str
    language: str = "en"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def source_text(self) -> str:
        """Return the source document under the generic harness name."""

        return self.document

    @property
    def source(self) -> str:
        """Return the source document under pair-dataset terminology."""

        return self.document

    @property
    def text(self) -> str:
        """Return the source document under generic fixture terminology."""

        return self.document

    @property
    def reference_summary(self) -> str:
        """Return the reference summary under the explicit metric name."""

        return self.summary

    @property
    def target(self) -> str:
        """Return the reference summary under seq2seq terminology."""

        return self.summary

    @property
    def task_view(self) -> str:
        """Return the stable task-view identifier in fixture metadata."""

        return str(self.metadata.get("task_view") or self.metadata.get("task", ""))

    @property
    def task(self) -> str:
        """Return the broad task name from fixture metadata."""

        return str(self.metadata.get("task", ""))

    @property
    def view(self) -> str:
        """Return the detailed task view from fixture metadata."""

        return self.task_view

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready document-summary representation."""

        return {
            "document": self.document,
            "fixture_id": self.fixture_id,
            "language": self.language,
            "metadata": dict(self.metadata),
            "summary": self.summary,
        }


def _span_to_dict(span: EvalSpan) -> dict[str, Any]:
    return {
        "end": span.end,
        "label": span.label,
        "metadata": dict(span.metadata),
        "start": span.start,
        "text": span.text,
    }


__all__ = [
    "DocumentSummaryFixture",
    "RelationTaskFixture",
    "SentencePairFixture",
]
