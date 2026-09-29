"""Offline, conservative literal-alignment summary benchmark.

This protocol never equates literal span checks with semantic adjudication.
Missing adjudication remains a failed release gate, including for extractive
output. Reports contain counts, scalar metrics and digests only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

from openmed.clinical.summarize import summarize_deidentified
from openmed.clinical.summary_claim_segments import segment_summary_claims
from openmed.core.offline import network_blocked_if_offline
from openmed.core.pii import DeidentificationResult, PIIEntity
from openmed.eval.clinical_fixtures import generate_fixture
from openmed.eval.report import BenchmarkReport
from openmed.eval.summary_gate import evaluate_summary_gate


def digest(value):
    """Hash canonical JSON without retaining evaluation values."""
    return (
        "sha256:"
        + hashlib.sha256(
            json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            ).encode()
        ).hexdigest()
    )


def run_summary_benchmark(model: str, *, seeds, thresholds) -> BenchmarkReport:
    """Run actual local generation on seeded documents with gold-span masking.

    Gold masking isolates summarization; this is not a de-identification model
    benchmark. Fact retention requires the exact assertion-bearing sentence.
    Paraphrases are unresolved rather than guessed to be supported.
    """
    if model not in {"extractive", "mlx"} or not seeds or len(seeds) > 100:
        raise ValueError("invalid summary benchmark configuration")
    outcomes = []
    source_hashes = []
    output_hashes = []
    generated_count = 0
    with network_blocked_if_offline(local_only=True):
        for seed in seeds:
            fixture = generate_fixture("discharge_summary", seed=seed)
            source_hashes.append(fixture.text_hash)
            pii = [
                PIIEntity(
                    text=s.text,
                    label=s.label,
                    start=s.start,
                    end=s.end,
                    confidence=1.0,
                    redacted_text="[" + s.label + "]",
                )
                for s in fixture.gold_spans
                if s.label in {"ID_NUM", "DATE"}
            ]
            masked = fixture.text
            for entity in sorted(pii, key=lambda item: item.start, reverse=True):
                masked = (
                    masked[: entity.start] + entity.redacted_text + masked[entity.end :]
                )
            masked = " ".join(masked.split())
            value = DeidentificationResult(
                fixture.text, masked, pii, "mask", datetime(2026, 1, 1)
            )
            try:
                summary = summarize_deidentified(value, model=model).summary
            except Exception:
                outcomes.append(
                    [
                        {
                            "gate": "summary_generation",
                            "passed": False,
                            "reason": "local_generation_unavailable",
                            "details": {},
                        }
                    ]
                )
                continue
            generated_count += 1
            output_hashes.append(digest(summary))
            sources = segment_summary_claims(masked).segments
            targets = segment_summary_claims(summary).segments
            evidence = [
                {"evidence_id": "e" + str(i), "start": s.start, "end": s.end}
                for i, s in enumerate(sources)
            ]
            claims, relations = [], []
            for i, target in enumerate(targets):
                matching = [
                    j
                    for j, source in enumerate(sources)
                    if target.text == source.text and not target.review_required
                ]
                ids = ["e" + str(j) for j in matching] if len(matching) == 1 else []
                claim_id = "c" + str(i)
                claims.append(
                    {
                        "claim_id": claim_id,
                        "claim_class": "finding",
                        "evidence_ids": ids,
                        "citations": [{"evidence_id": e} for e in ids],
                    }
                )
                relations.extend(
                    {
                        "evidence_id": e,
                        "claim_id": claim_id,
                        "relation": "supported",
                        "approved": True,
                    }
                    for e in ids
                )
            gold, retained = [], []
            for span in fixture.gold_spans:
                if span.label not in {"CONDITION", "MEDICATION", "CARE_PLAN"}:
                    continue
                fact = (
                    "medication" if span.label == "MEDICATION" else "problem",
                    span.text,
                    span.assertion,
                    span.temporality,
                    span.experiencer,
                )
                gold.append(fact)
                containing = [s.text for s in sources if span.text in s.text]
                if len(containing) == 1 and any(
                    t.text == containing[0] for t in targets
                ):
                    retained.append(fact)
            checks = evaluate_summary_gate(
                deidentified=value,
                summary=summary,
                source_facts=gold,
                summary_facts=retained,
                source_evidence=evidence,
                claims=claims,
                support_evidence=relations,
                thresholds=thresholds,
            )
            outcomes.append([check.to_dict() for check in checks])
    metrics = {
        "generated_count": generated_count,
        "failed_fixture_count": sum(
            not all(c["passed"] for c in row) for row in outcomes
        ),
        "checks": outcomes,
    }
    provenance = {
        "aggregate_only": True,
        "synthetic_only": True,
        "clinical_validation": False,
        "protocol": "gold-masked-literal-assertion-sentence-v1",
        "fixture_digest": digest(source_hashes),
        "output_digest": digest(output_hashes),
        "threshold_digest": digest(thresholds),
        "adjudication_available": False,
        "model_revision": "361db5da5e74ff6fcdd852d478e1f266ce11013a"
        if model == "mlx"
        else None,
    }
    report = BenchmarkReport(
        "summaries",
        model,
        "mlx" if model == "mlx" else "cpu",
        len(seeds),
        metrics,
        metadata=provenance,
    )
    return BenchmarkReport(
        report.suite,
        report.model_name,
        report.device,
        report.fixture_count,
        report.metrics,
        metadata={**provenance, "report_digest": digest(report.to_dict())},
    )


def verify_summary_report(report: dict, *, thresholds=None) -> bool:
    """Verify a committed report and reject failed or incomplete release evidence."""
    try:
        payload = {**report, "metadata": dict(report["metadata"])}
        recorded = payload["metadata"].pop("report_digest")
        if recorded != digest(payload) or report["suite"] != "summaries":
            return False
        if thresholds is not None and report["metadata"]["threshold_digest"] != digest(
            thresholds
        ):
            return False
        count = report["fixture_count"]
        if (
            report["metrics"]["generated_count"] != count
            or report["metrics"]["failed_fixture_count"] != 0
        ):
            return False
        checks = report["metrics"]["checks"]
        expected = {
            "summary_" + name
            for name in (
                "clinical_fact_recall",
                "fact_coverage",
                "unsupported_claim_rate",
                "citation_support",
                "leakage",
            )
        }
        return (
            type(count) is int
            and count > 0
            and len(checks) == count
            and all(
                {c["gate"] for c in row} == expected
                and len(row) == 5
                and all(c["passed"] is True for c in row)
                for row in checks
            )
        )
    except (KeyError, TypeError, ValueError):
        return False


def main(argv=None) -> int:
    """Generate actual local evidence or verify committed reports fail closed."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("extractive", "mlx"), default="extractive")
    parser.add_argument("--baseline", type=Path, default=Path("gates/baseline.json"))
    parser.add_argument(
        "--fixtures",
        type=Path,
        default=Path("tests/fixtures/eval/summaries/seeds.json"),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify", type=Path, nargs="+")
    args = parser.parse_args(argv)
    try:
        if args.verify:
            thresholds = json.loads(args.baseline.read_text())["summary"]
            results = [
                verify_summary_report(json.loads(p.read_text()), thresholds=thresholds)
                for p in args.verify
            ]
            print(json.dumps({"report_count": len(results), "passed": all(results)}))
            return 0 if all(results) else 1
        if args.output is None:
            parser.error("--output is required for generation")
        config = json.loads(args.fixtures.read_text())
        report = run_summary_benchmark(
            args.model,
            seeds=config["seeds"],
            thresholds=json.loads(args.baseline.read_text())["summary"],
        )
        report.write_json(args.output)
        print(
            json.dumps(
                {
                    "fixture_count": report.fixture_count,
                    "generated_count": report.metrics["generated_count"],
                    "passed": verify_summary_report(report.to_dict()),
                }
            )
        )
        return 0
    except Exception:
        print('{"passed":false,"reason":"summary_evaluation_failed"}')
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
