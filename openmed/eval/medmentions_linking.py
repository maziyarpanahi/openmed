"""User-supplied MedMentions st21pv top-1 linking benchmark."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openmed.clinical.grounding import Candidate, GroundedSpan
from openmed.eval.datasets.licenses import license_for
from openmed.eval.report import BenchmarkReport

__all__ = [
    "MEDMENTIONS_PATH_TO_TARGET",
    "MEDMENTIONS_ST21PV",
    "MEDMENTIONS_TOP1_FLOOR",
    "MEDMENTIONS_TOP1_TARGET",
    "MedMentionsLinkingCase",
    "evaluate_medmentions_st21pv",
]

MEDMENTIONS_ST21PV = "medmentions_st21pv_linking"
MEDMENTIONS_TOP1_FLOOR = 0.55
MEDMENTIONS_TOP1_TARGET = 0.70
MEDMENTIONS_PATH_TO_TARGET = (
    "Union sparse lexical candidates with a local SapBERT-class dense index, "
    "then apply the context-aware reranker and calibrate on a held-out split."
)

MedMentionsProvider = Callable[[str, int], Sequence[Candidate] | GroundedSpan]


@dataclass(frozen=True)
class MedMentionsLinkingCase:
    """One caller-supplied st21pv mention and its expected UMLS CUI.

    Args:
        mention: Source mention sent only to the caller-provided linker.
        expected_cui: Gold UMLS concept unique identifier.
    """

    mention: str
    expected_cui: str


def evaluate_medmentions_st21pv(
    path: str | Path,
    *,
    provider: MedMentionsProvider,
    top_k: int = 5,
) -> BenchmarkReport:
    """Score top-1/top-k linking without bundling MedMentions or UMLS data.

    ``path`` must be a caller-created JSONL projection with ``mention`` (or
    ``text``) and ``cui`` (or ``expected_cui``). No row content is copied into
    the report. The provider is responsible for using a caller-licensed local
    UMLS index; this function performs no download or credential lookup.

    Args:
        path: Local caller-created st21pv JSONL projection.
        provider: Local callback returning ranked UMLS candidates.
        top_k: Candidate depth used for the secondary top-k metric.

    Returns:
        Aggregate, text-free benchmark metrics and license provenance.

    Raises:
        ValueError: If ``top_k`` is invalid or the projection is empty.
    """

    if top_k < 1:
        raise ValueError("top_k must be positive")
    cases = _load_cases(path)
    if not cases:
        raise ValueError("MedMentions st21pv projection contains no cases")

    top1_hits = 0
    topk_hits = 0
    abstentions = 0
    for case in cases:
        output = provider(case.mention, top_k)
        candidates = output.candidates if isinstance(output, GroundedSpan) else output
        codes = [
            candidate.code
            for candidate in candidates
            if candidate.system.casefold() == "umls"
        ][:top_k]
        if not codes:
            abstentions += 1
        top1_hits += int(codes[:1] == [case.expected_cui])
        topk_hits += int(case.expected_cui in codes)

    top1 = top1_hits / len(cases)
    return BenchmarkReport(
        suite=MEDMENTIONS_ST21PV,
        model_name="caller-supplied-grounder",
        device="local",
        fixture_count=len(cases),
        metrics={
            "top1_accuracy": top1,
            f"top{top_k}_accuracy": topk_hits / len(cases),
            "abstention_rate": abstentions / len(cases),
            "floor": MEDMENTIONS_TOP1_FLOOR,
            "passed": top1 >= MEDMENTIONS_TOP1_FLOOR,
            "target": MEDMENTIONS_TOP1_TARGET,
        },
        metadata={
            "dataset": "MedMentions st21pv",
            "license": license_for("medmentions").to_dict(),
            "corpus_bundled": False,
            "restricted_vocabulary_bundled": False,
            "path_to_target": MEDMENTIONS_PATH_TO_TARGET,
        },
    )


def _load_cases(path: str | Path) -> tuple[MedMentionsLinkingCase, ...]:
    source = Path(path).expanduser()
    cases: list[MedMentionsLinkingCase] = []
    with source.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, Mapping):
                raise ValueError(f"{source}:{line_number} must contain an object")
            mention = str(payload.get("mention") or payload.get("text") or "")
            cui = str(payload.get("expected_cui") or payload.get("cui") or "")
            if not mention or not cui:
                raise ValueError(
                    f"{source}:{line_number} requires mention/text and cui"
                )
            cases.append(MedMentionsLinkingCase(mention, cui))
    return tuple(cases)
