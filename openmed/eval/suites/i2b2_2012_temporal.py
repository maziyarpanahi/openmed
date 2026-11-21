"""Aggregate-only evaluation of temporal extraction over local gold spans."""

from __future__ import annotations

from collections import Counter

from openmed.clinical.context import assert_context
from openmed.clinical.relations.temporal import (
    decode_tlink_candidates,
    extract_tlink_candidates,
)
from openmed.clinical.temporal_normalizer import normalize_temporal
from openmed.core.audit import stable_hash
from openmed.eval.aggregate_report import _rate
from openmed.eval.datasets.i2b2_2012 import (
    FOLD_VERSION,
    PREDICTION_FOLD,
    TemporalDocument,
)
from openmed.eval.metrics import (
    compute_temporal_awareness_f1,
    compute_temporal_closure_consistency,
)


def score_i2b2_2012(
    documents: tuple[TemporalDocument, ...], *, minimum_gold_spans: int = 5
) -> dict:
    """Score gold span normalization, assertion polarity and temporal links locally."""
    if not documents or type(minimum_gold_spans) is not int or minimum_gold_spans < 1:
        raise ValueError("invalid_temporal_evaluation")
    counts = Counter()
    digests = []
    for doc in documents:
        digests.append(
            stable_hash([doc.text, doc.events, doc.timexes, doc.sectimes, doc.links])
        )
        reference = next(
            (s["val"] for s in doc.sectimes if s["type"] == "ADMISSION"), None
        )
        for gold in doc.timexes:
            prediction = normalize_temporal(doc.text, [gold], reference)
            counts["timex"] += 1
            if prediction:
                counts["type"] += prediction[0].timex_type == gold["type"]
                counts["value"] += prediction[0].value == gold["val"]

        def span(row, label):
            return {
                "start": row["start"],
                "end": row["end"],
                "label": label,
                "text": doc.text[row["start"] : row["end"]],
                "metadata": {"source_type": row["type"]},
            }

        events = [span(e, "EVENT") for e in doc.events]
        for gold, result in zip(
            doc.events, assert_context(doc.text, events), strict=True
        ):
            counts["event"] += 1
            counts["polarity"] += (result["negation"] == "negated") == (
                gold["polarity"] in ("NEG", "NEGATIVE")
            )
        times = {
            (t["start"], t["end"]): span(t, "TIMEX")
            for t in (*doc.timexes, *doc.sectimes)
        }
        candidates = extract_tlink_candidates(doc.text, [*events, *times.values()])
        graph = decode_tlink_candidates(candidates)
        # Compare by role and span within this document. A section-time alias
        # and a TIMEX over the same interval identify the same endpoint.
        keys = {
            r["id"]: f"{role}:{r['start']}:{r['end']}"
            for role, rows in (
                ("EVENT", doc.events),
                ("TIMEX", (*doc.timexes, *doc.sectimes)),
            )
            for r in rows
        }
        nodes = {n.node_id: f"{n.label}:{n.start}:{n.end}" for n in graph.nodes}
        predicted = [
            (PREDICTION_FOLD[r], nodes[a], nodes[b]) for r, a, b in graph.edge_keys()
        ]
        gold_links = [(r, keys[a], keys[b]) for r, a, b in doc.links]
        score = compute_temporal_awareness_f1(gold_links, predicted)
        consistency = compute_temporal_closure_consistency(predicted)
        counts.update(
            gold=score.gold_reduced_relations,
            pred=score.predicted_reduced_relations,
            pm=score.precision_matches,
            rm=score.recall_matches,
            violations=sum(len(v) for v in consistency.violations.values()),
        )

    def cell(key, support):
        if counts[support] < minimum_gold_spans:
            return {"status": "suppressed"}
        return {
            "status": "reported",
            "gold_count": counts[support],
            **_rate(counts[key], counts[support]),
        }

    cells = {
        name: cell(key, support)
        for name, key, support in (
            ("timex_type_accuracy", "type", "timex"),
            ("timex_value_accuracy", "value", "timex"),
            ("event_polarity_agreement", "polarity", "event"),
        )
    }
    awareness = None
    if counts["gold"] >= minimum_gold_spans:
        precision = counts["pm"] / counts["pred"] if counts["pred"] else 0
        recall = counts["rm"] / counts["gold"]
        awareness = (
            2 * precision * recall / (precision + recall) if precision + recall else 0
        )
        cells["tlinks"] = {
            "status": "reported",
            "gold_count": counts["gold"],
            "awareness_f1": awareness,
            "closure_violations": counts["violations"],
        }
    else:
        cells["tlinks"] = {"status": "suppressed"}
    return {
        "suite": "i2b2-2012-temporal",
        "corpus": "i2b2-2012",
        "year": 2012,
        "fixture_count": len(documents),
        "dataset_digest": stable_hash(sorted(digests)),
        "minimum_gold_spans": minimum_gold_spans,
        "fold_version": FOLD_VERSION,
        "cells": cells,
        "metrics": {"awareness_f1": awareness},
    }
