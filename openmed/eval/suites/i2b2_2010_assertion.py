"""Score the local clinical assertion layer over user-supplied gold problems."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict

from openmed import __version__
from openmed.clinical.context import assert_context
from openmed.clinical.lexicons import get_clinical_cue_lexicon
from openmed.core.audit import stable_hash
from openmed.eval.aggregate_report import _rate
from openmed.eval.datasets.i2b2_2010 import I2B2_2010_ASSERTION_CLASSES as CLASSES

MAPPING_VERSION = 1
PRECEDENCE = (
    ("experiencer", ("family", "other"), "associated_with_someone_else"),
    ("temporality", ("hypothetical",), "hypothetical"),
    ("negation", ("negated",), "absent"),
    ("uncertainty", ("uncertain",), "possible"),
)
AXES = {
    "experiencer": ("family", "other", "patient"),
    "temporality": ("recent", "historical", "hypothetical"),
    "negation": ("affirmed", "negated"),
    "uncertainty": ("certain", "uncertain"),
}


def assertion_class(axes: dict) -> str:
    """Map four validated ConText axes to an i2b2 assertion class."""
    if any(axes.get(key) not in allowed for key, allowed in AXES.items()):
        raise ValueError("invalid_assertion_axes")
    for axis, values, label in PRECEDENCE:
        if axes[axis] in values:
            return label
    return "present"


def score_i2b2_2010(
    fixtures, *, classifier=assert_context, minimum_gold_spans: int = 5
) -> dict:
    """Evaluate in memory using only gold problem spans and native line boundaries."""
    if not fixtures or type(minimum_gold_spans) is not int or minimum_gold_spans < 1:
        raise ValueError("invalid_assertion_evaluation")
    counts = Counter()
    digests = []
    for fixture in fixtures:
        problems = [
            s
            for s in fixture.gold_spans
            if s.label == "PROBLEM" and "assertion" in s.metadata
        ]
        spans = [{"start": s.start, "end": s.end, "label": "PROBLEM"} for s in problems]
        gold = [s.metadata["assertion"] for s in problems]
        bounds = [
            {"start": start, "end": end}
            for start, end in fixture.metadata["sentence_offsets"]
            if start < end
        ]
        digests.append(stable_hash([fixture.text, spans, gold, bounds]))
        results = (
            classifier(fixture.text, spans, sentences=bounds, language="en")
            if spans
            else []
        )
        for expected, span, axes in zip(gold, spans, results, strict=True):
            if expected not in CLASSES or any(
                axes.get(k) != span[k] for k in ("start", "end")
            ):
                raise ValueError("assertion_span_mismatch")
            counts[expected, assertion_class(axes)] += 1
    total = sum(counts.values())
    by_class = {}
    confusion = {}
    f1s = []
    for label in CLASSES:
        tp = counts[label, label]
        support = sum(counts[label, pred] for pred in CLASSES)
        predicted = sum(counts[actual, label] for actual in CLASSES)
        fn, fp = support - tp, predicted - tp
        f1 = 2 * tp / (support + predicted) if support + predicted else 0.0
        f1s.append(f1)
        cell = {"system_emits_class": label != "conditional"}
        if support < minimum_gold_spans:
            cell["status"] = "suppressed"
        else:
            cell.update(
                status="reported",
                tp=tp,
                fp=fp,
                fn=fn,
                f1=f1,
                precision=_rate(tp, predicted),
                recall=_rate(tp, support),
            )
            confusion[label] = {
                pred: counts[label, pred]
                if counts[label, pred] >= minimum_gold_spans
                else None
                for pred in CLASSES
            }
        by_class[label] = cell
    return {
        "suite": "i2b2-2010-assertion",
        "year": 2010,
        "corpus": "i2b2-2010",
        "mapping_version": MAPPING_VERSION,
        "openmed_version": __version__,
        "cue_lexicon_digest": stable_hash(asdict(get_clinical_cue_lexicon("en"))),
        "fixture_count": len(fixtures),
        "dataset_digest": stable_hash(sorted(digests)),
        "minimum_gold_spans": minimum_gold_spans,
        "by_class": by_class,
        "confusion": confusion,
        "metrics": {
            "micro_f1": sum(counts[c, c] for c in CLASSES) / total
            if total >= minimum_gold_spans
            else None,
            "macro_f1": sum(f1s) / len(CLASSES)
            if total >= minimum_gold_spans
            else None,
        },
        "scope": "Assistive annotation on one corpus. Clinical validation requires separate evidence.",
    }
