"""Bounded sentence selection over caller-reviewed, offset-only facts.

No facts, importance classes or review approvals are inferred from source text.
UTF-8 bytes conservatively charge the existing token allowance; this is not a
measured tokenizer or clinical quality claim. Downstream safety gates still run.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from openmed.clinical.summarize import _SENTENCE_BOUNDARY
from openmed.clinical.summarize_backends import (
    MAX_INPUT_BYTES,
    MAX_OUTPUT_BYTES,
    LocalSummarizerError,
)
from openmed.clinical.summary_length_budget import (
    SUMMARY_EVIDENCE_CLASS_NAMES,
    SummaryLengthBudget,
)
from openmed.clinical.summary_omission_budget import (
    ImportanceClassPolicy,
    SummaryEvidenceCoverage,
    evaluate_summary_omission_budget,
)
from openmed.eval.summary_coverage import compute_summary_fact_coverage

MAX_SELECTION_STATES = 50_000
MAX_SELECTION_FACTS = 64


@dataclass(frozen=True)
class ExtractiveFact:
    """One reviewed fact's opaque identity, importance class and source offsets.

    Args:
        evidence_id: Opaque SHA-256 fact identity.
        importance_class_id: Opaque identity of its reviewed omission policy.
        start: Inclusive Unicode-scalar offset into the de-identified source.
        end: Exclusive Unicode-scalar offset into that source.
        length_class: Existing length-allocation class charged by this fact.
    """

    evidence_id: str
    importance_class_id: str
    start: int
    end: int
    length_class: str = "key_findings"

    def __post_init__(self) -> None:
        SummaryEvidenceCoverage(self.evidence_id, self.importance_class_id, False)
        if self.length_class not in SUMMARY_EVIDENCE_CLASS_NAMES:
            raise LocalSummarizerError("invalid extractive evidence class")
        if (
            type(self.start) is not int
            or type(self.end) is not int
            or not 0 <= self.start < self.end
        ):
            raise LocalSummarizerError("invalid extractive evidence offsets")


@dataclass(frozen=True)
class ExtractiveSelection:
    """Protected extract plus deterministic, value-free selection diagnostics."""

    status: str
    summary: str = field(repr=False)
    _audit: str = field(repr=False)

    def to_dict(self) -> dict:
        """Return defensive counts, offsets, opaque references and gate results."""
        import json

        return {"status": self.status, **json.loads(self._audit)}


class ExtractiveSelectionError(LocalSummarizerError):
    """Carry an explicit safe selection failure through the guarded pipeline."""

    def __init__(self, result: ExtractiveSelection) -> None:
        self.result = result
        super().__init__("extractive selection refused")


def select_extractive_sentences(
    text: str,
    *,
    evidence: tuple[ExtractiveFact, ...],
    importance_classes: tuple[ImportanceClassPolicy, ...],
    length_budget: SummaryLengthBudget,
) -> ExtractiveSelection:
    """Select the best whole-sentence extract satisfying omission and length gates.

    Args:
        text: Already de-identified source, never raw input to the public pipeline.
        evidence: At most 64 unique reviewed facts with offsets into ``text``.
            Identical facts and duplicate spans count once; conflicting identities
            fail closed. A fact must fit entirely inside one source sentence.
        importance_classes: Existing independent, non-compensable omission rules.
        length_budget: Existing global and class-specific generation allowances.

    Returns:
        ``selected``, ``empty_evidence``, ``invalid_evidence``,
        ``insufficient_budget`` or ``selection_limit_exceeded``. Failed results
        contain no partial summary or citations. Selection maximizes unique fact
        count, then severity-weighted coverage, then minimizes charged bytes,
        then prefers earlier source sentence indices. An exact bounded dynamic
        program proves infeasibility before returning insufficient budget; a
        resource limit is reported separately. Coverage metrics are unchanged.
    """
    import json

    def result(status, selected=(), facts=()):
        summary = " ".join(text[a:b] for a, b in selected)
        represented = {
            f.evidence_id
            for f in facts
            if any(a <= f.start < f.end <= b for a, b in selected)
        }
        citations = []
        output_start = 0
        for a, b in selected:
            citations.append(
                {
                    "source_start": a,
                    "source_end": b,
                    "output_start": output_start,
                    "output_end": output_start + b - a,
                    "evidence_ids": [
                        f.evidence_id for f in facts if a <= f.start < f.end <= b
                    ],
                }
            )
            output_start += b - a + 1
        coverage = compute_summary_fact_coverage(
            [{"fact_id": f.evidence_id} for f in facts],
            [{"source_fact_id": ref} for ref in sorted(represented)],
        )
        audit = {
            "citations": citations,
            "charged_tokens": len(summary.encode("utf-8")),
            "coverage": coverage.to_dict(),
        }
        if facts:
            audit["omission_budget"] = evaluate_summary_omission_budget(
                tuple(
                    SummaryEvidenceCoverage(
                        f.evidence_id,
                        f.importance_class_id,
                        f.evidence_id in represented,
                    )
                    for f in facts
                ),
                importance_classes,
            ).to_dict()
        return ExtractiveSelection(status, summary, json.dumps(audit, sort_keys=True))

    if (
        not isinstance(text, str)
        or len(text.encode("utf-8")) > MAX_INPUT_BYTES
        or type(evidence) is not tuple
        or len(evidence) > MAX_SELECTION_FACTS * 4
        or any(type(f) is not ExtractiveFact for f in evidence)
        or type(importance_classes) is not tuple
        or not 0 < len(importance_classes) <= MAX_SELECTION_FACTS
        or any(type(p) is not ImportanceClassPolicy for p in importance_classes)
        or type(length_budget) is not SummaryLengthBudget
    ):
        return result("invalid_evidence")
    policies = sorted(importance_classes, key=lambda p: p.class_id)
    class_indices = {p.class_id: i for i, p in enumerate(policies)}
    if len(class_indices) != len(policies):
        return result("invalid_evidence")
    unique = {}
    spans = {}
    for f in sorted(evidence, key=lambda f: (f.start, f.end, f.evidence_id)):
        if f.end > len(text) or f.importance_class_id not in class_indices:
            return result("invalid_evidence")
        if f.length_class not in {a.evidence_class for a in length_budget.allocations}:
            return result("invalid_evidence")
        if f.evidence_id in unique and unique[f.evidence_id] != f:
            return result("invalid_evidence")
        unique[f.evidence_id] = f
        prior = spans.get((f.start, f.end))
        if prior is not None:
            if (prior.importance_class_id, prior.length_class) != (
                f.importance_class_id,
                f.length_class,
            ):
                return result("invalid_evidence")
            # Alternate identifiers for the same span cannot inflate coverage.
            continue
        spans[f.start, f.end] = f
    facts = tuple(spans.values())
    if not facts:
        return result("empty_evidence")
    if len(facts) > MAX_SELECTION_FACTS:
        return result("selection_limit_exceeded")

    sentences = []
    start = 0
    for boundary in (*_SENTENCE_BOUNDARY.finditer(text), None):
        end = boundary.start() if boundary is not None else len(text)
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        if start < end:
            sentences.append((start, end))
        if boundary is not None:
            start = boundary.end()
    if any(sum(a <= f.start < f.end <= b for a, b in sentences) != 1 for f in facts):
        return result("invalid_evidence")
    length_classes = sorted({f.length_class for f in facts})
    caps = tuple(length_budget.budget_for(c) for c in length_classes)
    candidates = []
    for a, b in sentences:
        counts = [0] * len(policies)
        for f in facts:
            if a <= f.start < f.end <= b:
                counts[class_indices[f.importance_class_id]] += 1
        if any(counts):
            charged_classes = {
                f.length_class for f in facts if a <= f.start < f.end <= b
            }
            candidates.append(
                (
                    (a, b),
                    tuple(counts),
                    len(text[a:b].encode("utf-8")),
                    tuple(c in charged_classes for c in length_classes),
                )
            )
    totals = tuple(
        sum(f.importance_class_id == p.class_id for f in facts) for p in policies
    )
    cap = min(length_budget.max_tokens, MAX_OUTPUT_BYTES)
    # States retain exact class counts and charge, so no greedy early choice can
    # wrongly turn a feasible bounded extract into an insufficient-budget refusal.
    states = {(0, (0,) * len(length_classes), (0,) * len(policies)): ()}
    for i, (_, counts, cost, charged_classes) in enumerate(candidates):
        next_states = dict(states)
        for (used, class_used, covered), indices in states.items():
            charge = used + cost + bool(indices)
            if charge > cap:
                continue
            class_charge = tuple(
                u + (cost + bool(indices)) * applies
                for u, applies in zip(class_used, charged_classes)
            )
            if any(u > cap for u, cap in zip(class_charge, caps)):
                continue
            key = (charge, class_charge, tuple(a + b for a, b in zip(covered, counts)))
            chosen = indices + (i,)
            if key not in next_states or chosen < next_states[key]:
                next_states[key] = chosen
            if len(next_states) > MAX_SELECTION_STATES:
                return result("selection_limit_exceeded", facts=facts)
        remaining = tuple(
            sum(c[1][j] for c in candidates[i + 1 :]) for j in range(len(policies))
        )
        states = {
            key: indices
            for key, indices in next_states.items()
            if all(
                count + rest >= total - p.omission_limit
                for count, rest, total, p in zip(key[2], remaining, totals, policies)
            )
        }
    feasible = [
        (used, covered, indices)
        for (used, _, covered), indices in states.items()
        if indices
        and all(
            total - count <= p.omission_limit
            for total, count, p in zip(totals, covered, policies)
        )
    ]
    if not feasible:
        return result("insufficient_budget", facts=facts)
    _, _, indices = min(
        feasible,
        key=lambda s: (
            -sum(s[1]),
            -sum(n * p.severity_weight for n, p in zip(s[1], policies)),
            s[0],
            s[2],
        ),
    )
    return result("selected", tuple(candidates[i][0] for i in indices), facts)
