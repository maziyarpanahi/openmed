"""Offline Chinese clinical NER scorecard with a synthetic PHI leakage gate."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from openmed.core.quality_gates import validate_entity_spans
from openmed.eval.datasets.cmeee import load_cmeee
from openmed.eval.harness import BenchmarkFixture, ModelRunner
from openmed.eval.metrics import EvalSpan, compute_exact_span_f1, normalize_eval_spans
from openmed.eval.report import BenchmarkReport

CHINESE_CLINICAL_NER = "chinese-clinical-ner"
DEFAULT_SYNTHETIC_CMEEE_PATH = (
    Path(__file__).resolve().parents[1] / "fixtures" / "cmeee_zh_synthetic.jsonl"
)

RedactionRunner = Callable[[BenchmarkFixture, tuple[EvalSpan, ...]], str]


@dataclass(frozen=True)
class PhiTokenLeakageFinding:
    """PHI-safe evidence that one synthetic identifier survived redaction."""

    fixture_id: str
    start: int
    end: int
    label: str
    token_hash: str

    def to_dict(self) -> dict[str, int | str]:
        """Return JSON-ready evidence without the identifier surface."""

        return {
            "end": self.end,
            "fixture_id": self.fixture_id,
            "label": self.label,
            "start": self.start,
            "token_hash": self.token_hash,
        }


class ChineseClinicalNerLeakageError(RuntimeError):
    """Raised when a Chinese clinical NER run leaves synthetic PHI intact."""

    def __init__(self, report: BenchmarkReport) -> None:
        leakage = report.metrics["phi_token_leakage"]
        super().__init__(
            "Chinese clinical NER leakage gate failed: "
            f"{leakage['leaked_tokens']} of {leakage['total_tokens']} "
            "synthetic PHI tokens survived"
        )
        self.report = report


def load_chinese_clinical_ner_fixtures(
    path: str | Path | None = None,
) -> list[BenchmarkFixture]:
    """Load explicit CMeEE data or the bundled synthetic offline fixture.

    Passing ``path`` retains the CMeEE loader license boundary: real corpus
    records must live outside the repository. Omitting it loads only the tiny
    synthetic fixture shipped for deterministic CI smoke coverage.
    """

    synthetic = path is None
    source_path = DEFAULT_SYNTHETIC_CMEEE_PATH if synthetic else Path(path)
    result = load_cmeee(
        source_path,
        split="synthetic" if synthetic else "test",
        allow_repo_path=synthetic,
    )
    fixtures = result.to_benchmark_fixtures()
    if synthetic and not fixtures:
        raise ValueError("bundled synthetic CMeEE fixture must not be empty")
    return fixtures


def chinese_clinical_ner_metadata() -> dict[str, Any]:
    """Return the suite license, model, and redistribution disclaimers."""

    return {
        "data_boundary": (
            "CMeEE, CBLUE, and eHealth records are user-supplied local inputs; "
            "OpenMed bundles only synthetic smoke records."
        ),
        "language": "zh",
        "model_notice": (
            "The bundled Chinese default is the multilingual privacy fallback, "
            "not a dedicated Chinese clinical NER checkpoint."
        ),
        "redistribution": "no licensed corpus records or model weights are bundled",
        "suite": CHINESE_CLINICAL_NER,
        "task": "clinical_ner_with_phi_leakage_gate",
    }


def run_chinese_clinical_ner_suite(
    fixtures: Sequence[BenchmarkFixture],
    *,
    model_name: str,
    runner: ModelRunner,
    redactor: RedactionRunner,
    device: str = "cpu",
    generated_at: str | None = None,
    min_per_label_recall: float | None = None,
    fail_on_leakage: bool = True,
) -> BenchmarkReport:
    """Score canonical labels and fail when a synthetic PHI token survives.

    The report never retains identifier text. Leakage findings contain only
    fixture IDs, offsets, canonical labels, and SHA-256 token hashes.
    """

    if not fixtures:
        raise ValueError("Chinese clinical NER suite requires at least one fixture")

    predictions: dict[str, tuple[EvalSpan, ...]] = {}
    redacted_outputs: dict[str, str] = {}
    for fixture in fixtures:
        if fixture.fixture_id in predictions:
            raise ValueError(f"duplicate fixture id: {fixture.fixture_id!r}")
        predicted = tuple(
            normalize_eval_spans(
                runner(fixture, model_name, device),
                default_language=fixture.language,
                default_device=device,
                source_text=fixture.text,
            )
        )
        validate_entity_spans(
            [span.to_entity() for span in predicted],
            fixture.text,
        )
        redacted = redactor(fixture, predicted)
        if not isinstance(redacted, str):
            raise TypeError("Chinese clinical NER redactor must return text")
        predictions[fixture.fixture_id] = predicted
        redacted_outputs[fixture.fixture_id] = redacted

    gold, predicted, source_text = _corpus_coordinates(fixtures, predictions)
    overall = compute_exact_span_f1(gold, predicted, source_text=source_text)
    labels = sorted({span.label for span in [*gold, *predicted]})
    per_label = {
        label: compute_exact_span_f1(
            [span for span in gold if span.label == label],
            [span for span in predicted if span.label == label],
            source_text=source_text,
        ).to_dict()
        for label in labels
    }
    leakage = _phi_token_leakage(fixtures, redacted_outputs)

    failures: list[dict[str, Any]] = []
    if leakage["total_tokens"] == 0:
        failures.append({"reason": "no_synthetic_phi_tokens"})
    if leakage["leaked_tokens"]:
        failures.append(
            {
                "leaked_tokens": leakage["leaked_tokens"],
                "reason": "phi_token_leakage",
                "threshold": 0.0,
            }
        )
    if min_per_label_recall is not None:
        for label, metrics in per_label.items():
            if float(metrics["recall"]) < min_per_label_recall:
                failures.append(
                    {
                        "label": label,
                        "reason": "per_label_recall_below_threshold",
                        "recall": metrics["recall"],
                        "threshold": min_per_label_recall,
                    }
                )

    report = BenchmarkReport(
        suite=CHINESE_CLINICAL_NER,
        model_name=model_name,
        device=device,
        fixture_count=len(fixtures),
        generated_at=generated_at,
        metadata={
            **chinese_clinical_ner_metadata(),
            "fixture_ids": [fixture.fixture_id for fixture in fixtures],
        },
        metrics={
            "exact_span_f1": overall.to_dict(),
            "gate": {
                "failures": failures,
                "max_phi_token_leakage_rate": 0.0,
                "min_per_label_recall": min_per_label_recall,
                "passed": not failures,
            },
            "per_label": per_label,
            "phi_token_leakage": leakage,
        },
    )
    if fail_on_leakage and leakage["leaked_tokens"]:
        raise ChineseClinicalNerLeakageError(report)
    return report


def run_synthetic_chinese_clinical_ner_smoke() -> BenchmarkReport:
    """Run the bundled fixture with deterministic offline oracle adapters."""

    return run_chinese_clinical_ner_suite(
        load_chinese_clinical_ner_fixtures(),
        model_name="synthetic-oracle",
        runner=_identity_runner,
        redactor=_mask_synthetic_phi,
        min_per_label_recall=1.0,
    )


def _corpus_coordinates(
    fixtures: Sequence[BenchmarkFixture],
    predictions: Mapping[str, tuple[EvalSpan, ...]],
) -> tuple[list[EvalSpan], list[EvalSpan], str]:
    gold: list[EvalSpan] = []
    predicted: list[EvalSpan] = []
    texts: list[str] = []
    offset = 0
    for fixture in fixtures:
        texts.append(fixture.text)
        gold.extend(_shift_spans(fixture.gold_spans, offset))
        predicted.extend(_shift_spans(predictions[fixture.fixture_id], offset))
        offset += len(fixture.text) + 1
    return gold, predicted, "\n".join(texts)


def _shift_spans(spans: Iterable[EvalSpan], offset: int) -> list[EvalSpan]:
    return [
        replace(span, start=span.start + offset, end=span.end + offset)
        for span in spans
    ]


def _phi_token_leakage(
    fixtures: Sequence[BenchmarkFixture],
    redacted_outputs: Mapping[str, str],
) -> dict[str, Any]:
    findings: list[PhiTokenLeakageFinding] = []
    total_tokens = 0
    for fixture in fixtures:
        redacted = redacted_outputs[fixture.fixture_id].casefold()
        for span in _synthetic_phi_spans(fixture):
            total_tokens += 1
            surface = fixture.text[span.start : span.end]
            if surface and surface.casefold() in redacted:
                findings.append(
                    PhiTokenLeakageFinding(
                        fixture_id=fixture.fixture_id,
                        start=span.start,
                        end=span.end,
                        label=span.label,
                        token_hash=(
                            "sha256:"
                            + hashlib.sha256(surface.encode("utf-8")).hexdigest()
                        ),
                    )
                )
    leaked_tokens = len(findings)
    return {
        "findings": [finding.to_dict() for finding in findings],
        "leaked_tokens": leaked_tokens,
        "rate": leaked_tokens / total_tokens if total_tokens else 0.0,
        "total_tokens": total_tokens,
    }


def _synthetic_phi_spans(fixture: BenchmarkFixture) -> tuple[EvalSpan, ...]:
    raw = fixture.metadata.get("phi_spans") or ()
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ValueError(f"fixture {fixture.fixture_id!r} has invalid phi_spans")
    spans = tuple(
        normalize_eval_spans(
            raw,
            default_language=fixture.language,
            source_text=fixture.text,
        )
    )
    validate_entity_spans([span.to_entity() for span in spans], fixture.text)
    return spans


def _identity_runner(
    fixture: BenchmarkFixture,
    model_name: str,
    device: str,
) -> tuple[EvalSpan, ...]:
    _ = (model_name, device)
    return fixture.gold_spans


def _mask_synthetic_phi(
    fixture: BenchmarkFixture,
    predicted: tuple[EvalSpan, ...],
) -> str:
    _ = predicted
    text = fixture.text
    for span in sorted(
        _synthetic_phi_spans(fixture), key=lambda item: item.start, reverse=True
    ):
        text = f"{text[: span.start]}[{span.label}]{text[span.end :]}"
    return text


__all__ = [
    "CHINESE_CLINICAL_NER",
    "DEFAULT_SYNTHETIC_CMEEE_PATH",
    "ChineseClinicalNerLeakageError",
    "PhiTokenLeakageFinding",
    "RedactionRunner",
    "chinese_clinical_ner_metadata",
    "load_chinese_clinical_ner_fixtures",
    "run_chinese_clinical_ner_suite",
    "run_synthetic_chinese_clinical_ner_smoke",
]
