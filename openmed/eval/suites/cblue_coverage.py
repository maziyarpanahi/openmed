"""Offline CBLUE task-shape coverage scorecard with a provenance gate.

The suite scores every supported CBLUE task shape and fails closed when a
fixture cannot prove where it came from. Provenance evidence is raw-text-free:
findings carry fixture ids, reason codes, metadata key names, and hashes, never
mention text or a resolved corpus path.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from openmed.core.quality_gates import validate_entity_spans
from openmed.eval.datasets.cblue import (
    CBLUE_LANGUAGE,
    CBLUE_SCRIPT,
    CBLUE_TASKS,
    CHIP_CDN,
    cblue_suite_metadata,
    cblue_task_shape,
    configured_cblue_task_path,
    load_cblue_task_fixtures,
    synthetic_cblue_fixture_path,
)
from openmed.eval.harness import BenchmarkFixture, ModelRunner
from openmed.eval.metrics import EvalSpan, compute_exact_span_f1, normalize_eval_spans
from openmed.eval.report import BenchmarkReport

CBLUE_TASK_COVERAGE = "cblue-task-coverage"
SYNTHETIC_SOURCE = "synthetic-smoke"
USER_SUPPLIED_SOURCE = "user-supplied"

#: Returns the standard terms a model predicts for one normalization fixture.
NormalizationRunner = Callable[[BenchmarkFixture], Sequence[str]]

_PAYLOAD_SUFFIXES = (".bio", ".conll", ".csv", ".iob", ".json", ".jsonl", ".tsv")
_REQUIRED_LICENSE_KEYS = ("dataset", "license_id", "redistribution", "source_url")


@dataclass(frozen=True)
class CblueProvenanceFinding:
    """Raw-text-free evidence that one fixture failed a provenance check."""

    task: str
    fixture_id: str
    reason: str
    evidence: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-ready evidence without benchmark content."""

        return {
            "evidence": dict(self.evidence),
            "fixture_id": self.fixture_id,
            "reason": self.reason,
            "task": self.task,
        }


@dataclass(frozen=True)
class CblueTaskCoverage:
    """Deterministic per-task coverage aggregates."""

    task: str
    shape: str
    fixture_count: int
    span_count: int
    label_counts: Mapping[str, int]
    exact_span_f1: Mapping[str, Any]
    normalization_accuracy: float | None = None
    normalized_term_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic, raw-text-free mapping for this task."""

        payload: dict[str, Any] = {
            "exact_span_f1": dict(self.exact_span_f1),
            "fixture_count": self.fixture_count,
            "label_counts": {
                label: int(self.label_counts[label])
                for label in sorted(self.label_counts)
            },
            "shape": self.shape,
            "span_count": self.span_count,
            "task": self.task,
        }
        if self.normalization_accuracy is not None:
            payload["normalization_accuracy"] = self.normalization_accuracy
            payload["normalized_term_count"] = self.normalized_term_count
        return payload


class CblueProvenanceError(RuntimeError):
    """Raised when a CBLUE task fixture fails the provenance gate."""

    def __init__(self, report: BenchmarkReport) -> None:
        failures = report.metrics["gate"]["failures"]
        reasons = ", ".join(sorted({str(failure["reason"]) for failure in failures}))
        super().__init__(
            f"CBLUE task-coverage provenance gate failed: {len(failures)} "
            f"finding(s) [{reasons}]"
        )
        self.report = report


def load_cblue_task_coverage_fixtures(
    paths: Mapping[str, str | Path] | None = None,
    *,
    tasks: Sequence[str] = CBLUE_TASKS,
) -> list[BenchmarkFixture]:
    """Load configured CBLUE data or the bundled synthetic offline fixtures.

    Path resolution matches the rest of the loader: an explicit entry in
    *paths* wins, then the task's own environment variable, then a per-task
    child of ``OPENMED_CBLUE_PATH``. Only when none of those is set does the
    suite fall back to the tiny synthetic fixture shipped for deterministic CI
    smoke coverage. Real corpus records must live outside the repository.

    Every fixture is tagged with ``source_kind`` so the report can state which
    of the two it actually scored rather than inferring it from configuration.
    """

    fixtures: list[BenchmarkFixture] = []
    for task in tasks:
        shape = cblue_task_shape(task)
        supplied = None if paths is None else paths.get(shape.task)
        resolved = configured_cblue_task_path(shape.task, supplied)
        synthetic = resolved is None
        source = synthetic_cblue_fixture_path(shape.task) if synthetic else resolved
        loaded = load_cblue_task_fixtures(
            shape.task,
            source,
            split="synthetic" if synthetic else "test",
            allow_repo_path=synthetic,
        )
        if synthetic and not loaded:
            raise ValueError(
                f"bundled synthetic {shape.display_name} fixture must not be empty"
            )
        source_kind = SYNTHETIC_SOURCE if synthetic else USER_SUPPLIED_SOURCE
        fixtures.extend(
            replace(
                fixture,
                metadata={**dict(fixture.metadata), "source_kind": source_kind},
            )
            for fixture in loaded
        )
    return fixtures


def cblue_task_coverage_metadata(
    paths: Mapping[str, str | Path] | None = None,
    *,
    tasks: Sequence[str] = CBLUE_TASKS,
) -> dict[str, Any]:
    """Return the suite license, task, and redistribution disclaimers."""

    return {
        **cblue_suite_metadata(paths, tasks=tasks),
        "model_notice": (
            "Task coverage measures the loader and scoring path. Chinese "
            "checkpoint selection and its evidence are reported by the "
            "chinese-clinical-ner suite."
        ),
        "suite": CBLUE_TASK_COVERAGE,
        "task": "cblue_task_coverage_with_provenance_gate",
    }


def run_cblue_task_coverage(
    fixtures: Sequence[BenchmarkFixture],
    *,
    model_name: str,
    runner: ModelRunner,
    normalizer: NormalizationRunner | None = None,
    device: str = "cpu",
    generated_at: str | None = None,
    fail_on_provenance: bool = True,
) -> BenchmarkReport:
    """Score every CBLUE task shape and gate on fixture provenance.

    The report never retains benchmark text. Provenance findings contain only
    fixture ids, reason codes, metadata key names, and hashes.
    """

    if not fixtures:
        raise ValueError("CBLUE task coverage requires at least one fixture")

    seen: set[str] = set()
    by_task: dict[str, list[BenchmarkFixture]] = {}
    findings: list[CblueProvenanceFinding] = []
    for fixture in fixtures:
        if fixture.fixture_id in seen:
            raise ValueError(f"duplicate fixture id: {fixture.fixture_id!r}")
        seen.add(fixture.fixture_id)
        task = str(fixture.metadata.get("cblue_task") or "")
        findings.extend(_provenance_findings(fixture, task))
        by_task.setdefault(task, []).append(fixture)

    coverage: list[CblueTaskCoverage] = []
    for task in sorted(by_task):
        # An unrecognized task has already been recorded as an ``unknown_task``
        # finding. Scoring it would raise from the task-shape lookup, which
        # would defeat ``fail_on_provenance=False``.
        if task not in CBLUE_TASKS:
            continue
        coverage.append(
            _task_coverage(
                task,
                by_task[task],
                model_name=model_name,
                runner=runner,
                normalizer=normalizer,
                device=device,
            )
        )

    failures = [finding.to_dict() for finding in findings]
    for task in CBLUE_TASKS:
        if not by_task.get(task):
            failures.append(
                CblueProvenanceFinding(
                    task=task,
                    fixture_id="",
                    reason="no_task_fixtures",
                    evidence={"expected_tasks": list(CBLUE_TASKS)},
                ).to_dict()
            )

    report = BenchmarkReport(
        suite=CBLUE_TASK_COVERAGE,
        model_name=model_name,
        device=device,
        fixture_count=len(fixtures),
        generated_at=generated_at,
        metadata={
            **_scored_metadata(by_task),
            "fixture_ids": [fixture.fixture_id for fixture in fixtures],
        },
        metrics={
            "gate": {
                "failures": failures,
                "passed": not failures,
                "required_tasks": list(CBLUE_TASKS),
            },
            "tasks": {item.task: item.to_dict() for item in coverage},
        },
    )
    if fail_on_provenance and failures:
        raise CblueProvenanceError(report)
    return report


def run_synthetic_cblue_task_coverage_smoke() -> BenchmarkReport:
    """Run the bundled fixtures with deterministic offline oracle adapters."""

    return run_cblue_task_coverage(
        load_cblue_task_coverage_fixtures(),
        model_name="synthetic-oracle",
        runner=_identity_runner,
        normalizer=_gold_normalizer,
    )


def _scored_metadata(
    by_task: Mapping[str, Sequence[BenchmarkFixture]],
) -> dict[str, Any]:
    """Return suite metadata whose availability describes what was scored.

    The static metadata reports how the loader was *configured*. A report is
    evidence about a run, so each task's availability is rewritten from the
    fixtures that actually reached the scorer; otherwise a run over licensed
    data could be labelled ``skipped`` and a synthetic run ``configured``.
    """

    metadata = cblue_task_coverage_metadata()
    observed: dict[str, Any] = {}
    for task, payload in metadata["tasks"].items():
        entry = dict(payload)
        scored = list(by_task.get(task, ()))
        path_env = dict(entry["availability"])["path_env"]
        kinds = sorted(
            {str(f.metadata.get("source_kind") or "unknown") for f in scored}
        )
        if not scored:
            entry["availability"] = {
                "configured": False,
                "path_env": path_env,
                "reason": "no fixtures were scored for this task",
                "source_kind": "none",
                "status": "missing",
            }
        else:
            source_kind = kinds[0] if len(kinds) == 1 else "mixed"
            user_supplied = source_kind == USER_SUPPLIED_SOURCE
            entry["availability"] = {
                "configured": user_supplied,
                "path_env": path_env,
                "reason": (
                    ""
                    if user_supplied
                    else f"scored the bundled {source_kind} fixture, not {path_env}"
                ),
                "source_kind": source_kind,
                "status": "configured" if user_supplied else "synthetic",
            }
        observed[task] = entry
    metadata["tasks"] = observed
    return metadata


def _task_coverage(
    task: str,
    fixtures: Sequence[BenchmarkFixture],
    *,
    model_name: str,
    runner: ModelRunner,
    normalizer: NormalizationRunner | None,
    device: str,
) -> CblueTaskCoverage:
    gold: list[EvalSpan] = []
    predicted: list[EvalSpan] = []
    texts: list[str] = []
    labels: Counter[str] = Counter()
    offset = 0
    matched_normalizations = 0
    normalized_terms = 0

    for fixture in fixtures:
        spans = tuple(
            normalize_eval_spans(
                runner(fixture, model_name, device),
                default_language=fixture.language,
                default_device=device,
                source_text=fixture.text,
            )
        )
        validate_entity_spans([span.to_entity() for span in spans], fixture.text)
        texts.append(fixture.text)
        gold.extend(_shift_spans(fixture.gold_spans, offset))
        predicted.extend(_shift_spans(spans, offset))
        labels.update(span.label for span in fixture.gold_spans)
        offset += len(fixture.text) + 1

        if normalizer is not None and task == CHIP_CDN:
            expected = _gold_normalized_terms(fixture)
            normalized_terms += len(expected)
            actual = {str(term).strip() for term in normalizer(fixture)}
            if actual == expected:
                matched_normalizations += 1

    scores = compute_exact_span_f1(gold, predicted, source_text="\n".join(texts))
    accuracy: float | None = None
    if normalizer is not None and task == CHIP_CDN:
        accuracy = round(matched_normalizations / len(fixtures), 6)

    return CblueTaskCoverage(
        task=task,
        shape=cblue_task_shape(task).shape,
        fixture_count=len(fixtures),
        span_count=len(gold),
        label_counts=dict(labels),
        exact_span_f1=scores.to_dict(),
        normalization_accuracy=accuracy,
        normalized_term_count=normalized_terms,
    )


def _provenance_findings(
    fixture: BenchmarkFixture,
    task: str,
) -> list[CblueProvenanceFinding]:
    metadata = dict(fixture.metadata)
    findings: list[CblueProvenanceFinding] = []

    def record(reason: str, evidence: Mapping[str, Any]) -> None:
        findings.append(
            CblueProvenanceFinding(
                task=task,
                fixture_id=fixture.fixture_id,
                reason=reason,
                evidence=dict(evidence),
            )
        )

    if task not in CBLUE_TASKS:
        record("unknown_task", {"expected_tasks": list(CBLUE_TASKS)})
        return findings

    dataset_license = metadata.get("license")
    if not isinstance(dataset_license, Mapping):
        record("missing_license_block", {"metadata_key": "license"})
    else:
        missing = [
            key for key in _REQUIRED_LICENSE_KEYS if not dataset_license.get(key)
        ]
        if missing:
            record("incomplete_license_block", {"missing_keys": sorted(missing)})
        elif str(dataset_license.get("redistribution")) != "user-supplied":
            record(
                "unexpected_redistribution",
                {"redistribution": str(dataset_license.get("redistribution"))},
            )

    source_path_hash = str(metadata.get("source_path_hash") or "")
    if not source_path_hash.startswith("sha256:"):
        record("missing_source_path_hash", {"metadata_key": "source_path_hash"})

    if str(metadata.get("script")) != CBLUE_SCRIPT:
        record("unexpected_script", {"script": str(metadata.get("script"))})
    if fixture.language != CBLUE_LANGUAGE:
        record("unexpected_language", {"language": fixture.language})

    if metadata.get("unmapped_labels"):
        record(
            "unmapped_source_label",
            {
                "source_labels": sorted(
                    str(item) for item in metadata["unmapped_labels"]
                )
            },
        )

    leaked_paths = sorted(
        key
        for key, value in metadata.items()
        if isinstance(value, str) and value.lower().endswith(_PAYLOAD_SUFFIXES)
    )
    if leaked_paths:
        record("raw_source_path_in_metadata", {"metadata_keys": leaked_paths})

    if task == CHIP_CDN and not _gold_normalized_terms(fixture):
        record("missing_normalized_terms", {"metadata_key": "normalized_terms"})

    return findings


def _gold_normalized_terms(fixture: BenchmarkFixture) -> set[str]:
    raw = fixture.metadata.get("normalized_terms") or ()
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
        return set()
    return {str(term).strip() for term in raw if str(term).strip()}


def _shift_spans(spans: Iterable[EvalSpan], offset: int) -> list[EvalSpan]:
    return [
        replace(span, start=span.start + offset, end=span.end + offset)
        for span in spans
    ]


def _identity_runner(
    fixture: BenchmarkFixture,
    model_name: str,
    device: str,
) -> tuple[EvalSpan, ...]:
    _ = (model_name, device)
    return fixture.gold_spans


def _gold_normalizer(fixture: BenchmarkFixture) -> tuple[str, ...]:
    return tuple(sorted(_gold_normalized_terms(fixture)))


__all__ = [
    "CBLUE_TASK_COVERAGE",
    "CblueProvenanceError",
    "CblueProvenanceFinding",
    "CblueTaskCoverage",
    "NormalizationRunner",
    "cblue_task_coverage_metadata",
    "load_cblue_task_coverage_fixtures",
    "run_cblue_task_coverage",
    "run_synthetic_cblue_task_coverage_smoke",
]
