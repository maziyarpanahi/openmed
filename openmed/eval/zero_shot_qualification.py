"""Qualification gates for exported GLiNER span and GLiClass artifacts.

This module extends the family-neutral harness in
:mod:`openmed.eval.export_qualification` to label-conditioned profiles. The
reference runner (the PyTorch reference, itself checked against the MLX
runtime) and every candidate runtime receive identical feeds built by the
profile contract in :mod:`openmed.ner.families.zero_shot_export`, once in
the suite's candidate-label order and once in reversed order, so an export
that froze the label count or label positions of its tracing example fails.

GLiNER span gates: logit parity on valid span and label cells, decoded span
parity (labels, code-point offsets, overlap resolution, thresholds),
direct-identifier recall, and critical leakage. GLiClass gates: logit
parity, label-ranking parity, and thresholded-decision parity. Reports reuse
:class:`~openmed.eval.export_qualification.ExportQualificationReport`, so
:func:`~openmed.eval.export_qualification.verify_qualification_report` and
:func:`~openmed.eval.export_qualification.check_qualified_request` work
unchanged. Reports carry digests, counts, and aggregate metrics only: never
fixture text, candidate label text, token ids, logits, or local paths.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Mapping, Sequence

from openmed.core.labels import normalize_label
from openmed.eval.export_qualification import (
    DEFAULT_QUALIFICATION_SUITE,
    EXPORT_QUALIFICATION_FILENAME,
    ExportQualificationError,
    ExportQualificationReport,
    ExportReason,
    LogitRunner,
    QualificationPolicy,
    RuntimeSupport,
    VariantCandidate,
    VariantQualification,
    _artifacts_size,
    _count_by_label,
    _finite_or_none,
    _latency_summary,
    _peak_rss_bytes,
    _relative_artifacts,
    _span_parity,
    artifact_file_digests,
    converter_environment,
    critical_leaks,
    direct_identifier_labels,
    recall_by_label,
    sha256_file,
    tokenizer_identity,
)
from openmed.eval.quant_delta import evaluate_onnx_logit_parity
from openmed.ner.families.zero_shot_export import (
    DEFAULT_GLINER_LABEL_PROMPTS,
    DEFAULT_THRESHOLD,
    TASK_ZERO_SHOT_NER,
    ZERO_SHOT_OFFSET_CONTRACTS,
    ZeroShotBounds,
    ZeroShotEncoding,
    ZeroShotModelSpec,
    ZeroShotProfile,
    ZeroShotPromptSpec,
    build_io_contract,
    decode_gliclass,
    decode_gliner_spans,
    encode_gliclass,
    encode_gliner_span,
    label_schema_identity,
)

DEFAULT_CLASSIFICATION_SUITE = (
    Path(__file__).resolve().parent / "fixtures" / "zero_shot_classification.jsonl"
)
LABEL_ORDERS = ("suite", "reversed")


@dataclass(frozen=True)
class ZeroShotFixture:
    """One synthetic qualification request, held in memory only.

    Attributes:
        fixture_id: Stable fixture id.
        text: Synthetic source text.
        labels: Candidate labels in suite order.
        canonical: GLiNER only: candidate label to canonical OpenMed label.
        gold_labels: GLiClass only: expected labels (evidence only).
        benchmark: GLiNER only: the harness fixture with gold spans.
    """

    fixture_id: str
    text: str
    labels: tuple[str, ...]
    canonical: Mapping[str, str] = field(default_factory=dict)
    gold_labels: tuple[str, ...] = ()
    benchmark: Any = None


@dataclass(frozen=True)
class _EncodedRequest:
    key: str
    order: str
    fixture: ZeroShotFixture
    labels: tuple[str, ...]
    encoding: ZeroShotEncoding


@dataclass(frozen=True)
class ZeroShotVariantQualification(VariantQualification):
    """:class:`VariantQualification` with the GLiClass ranking gates."""

    classification_parity: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-ready variant evidence."""
        payload = super().to_dict()
        payload["classification_parity"] = dict(self.classification_parity)
        return payload


@dataclass(frozen=True)
class ZeroShotQualificationReport(ExportQualificationReport):
    """:class:`ExportQualificationReport` for a label-conditioned profile."""

    profile: Mapping[str, Any] = field(default_factory=dict)

    def _payload(self) -> dict[str, Any]:
        payload = super()._payload()
        payload["offset_contract"] = dict(ZERO_SHOT_OFFSET_CONTRACTS[self.task])
        payload["profile"] = dict(self.profile)
        return payload


def load_zero_shot_fixtures(
    profile: ZeroShotProfile,
    path: str | Path | None = None,
    *,
    label_prompts: Mapping[str, str] | None = None,
) -> tuple[list[ZeroShotFixture], dict[str, Any]]:
    """Load a synthetic suite for *profile* and its value-free identity.

    GLiNER span suites use the harness fixture format (gold spans with
    canonical labels; the packaged synthetic PHI suite by default). Each
    canonical label becomes the natural-language candidate label from
    *label_prompts* and every fixture asks for all of them, in canonical
    label order. GLiClass suites are JSON or JSONL rows with ``id``,
    ``text``, ``candidate_labels``, and optional ``gold_labels``.

    Raises:
        ExportQualificationError: ``missing_evidence`` for an empty suite or
            a gold label without a candidate-label prompt.
    """

    if profile.task == TASK_ZERO_SHOT_NER:
        return _load_span_fixtures(path, label_prompts)
    return _load_classification_fixtures(path)


def encode_zero_shot_fixtures(
    fixtures: Sequence[ZeroShotFixture],
    *,
    profile: ZeroShotProfile,
    spec: ZeroShotModelSpec,
    tokenizer: Any,
    prompt: ZeroShotPromptSpec,
    bounds: ZeroShotBounds,
) -> tuple[_EncodedRequest, ...]:
    """Encode every fixture in suite and reversed label order.

    Raises:
        ZeroShotRequestError: When a fixture exceeds the declared bounds;
            qualify with a suite that fits them.
    """

    requests: list[_EncodedRequest] = []
    for fixture in fixtures:
        for order in LABEL_ORDERS:
            labels = fixture.labels if order == "suite" else fixture.labels[::-1]
            if order == "reversed" and len(labels) < 2:
                continue
            if profile.task == TASK_ZERO_SHOT_NER:
                encoding = encode_gliner_span(
                    tokenizer,
                    fixture.text,
                    labels,
                    prompt=prompt,
                    bounds=bounds,
                    graph_span_width=int(spec.span_width or 1),
                )
            else:
                encoding = encode_gliclass(
                    tokenizer, fixture.text, labels, prompt=prompt, bounds=bounds
                )
            requests.append(
                _EncodedRequest(
                    key=f"{fixture.fixture_id}#{order}",
                    order=order,
                    fixture=fixture,
                    labels=tuple(labels),
                    encoding=encoding,
                )
            )
    return tuple(requests)


def collect_zero_shot_outputs(
    runner: LogitRunner,
    requests: Sequence[_EncodedRequest],
    *,
    profile: ZeroShotProfile,
    graph_span_width: int | None,
) -> tuple[dict[str, Any], list[float]]:
    """Run *runner* on every request and validate the output shapes.

    Returns:
        ``(logits_by_request, latencies_ms)`` with float32 numpy logits.

    Raises:
        ExportQualificationError: ``inference_failed`` when the runner raises
            (its message is suppressed because runtime errors may echo
            inputs) or ``io_contract_mismatch`` for a wrong output shape.
    """

    import numpy as np

    outputs: dict[str, Any] = {}
    latencies: list[float] = []
    for request in requests:
        started = time.perf_counter()
        try:
            output = runner(request.encoding.feed)
        except ExportQualificationError:
            raise
        except Exception:
            raise ExportQualificationError(ExportReason.INFERENCE_FAILED) from None
        latencies.append((time.perf_counter() - started) * 1000.0)
        array = np.asarray(output, dtype=np.float32)
        encoding = request.encoding
        if profile.task == TASK_ZERO_SHOT_NER:
            expected: tuple[int, ...] = (
                1,
                len(encoding.word_offsets),
                int(graph_span_width or 1),
                encoding.num_labels,
            )
        else:
            expected = (1, encoding.num_labels)
        if array.shape != expected:
            raise ExportQualificationError(ExportReason.IO_CONTRACT_MISMATCH)
        outputs[request.key] = array
    return outputs, latencies


def qualify_zero_shot_variant(
    *,
    candidate: VariantCandidate,
    profile: ZeroShotProfile,
    support: RuntimeSupport,
    requests: Sequence[_EncodedRequest],
    reference_logits: Mapping[str, Any],
    graph_span_width: int | None,
    threshold: float = DEFAULT_THRESHOLD,
    artifact_root: str | Path | None = None,
    direct_identifier_recall_floor: float | None = None,
) -> ZeroShotVariantQualification:
    """Run every gate for one candidate against precomputed reference logits.

    Unsupported precision, quantization, or provider combinations are rejected
    before the candidate runs. Inference failures and wrong shapes are
    recorded as failing reasons so one broken variant never hides the
    evidence for its siblings.

    Raises:
        ExportQualificationError: For combinations *support* does not declare.
    """

    support.validate(
        precision=candidate.precision,
        quantization=candidate.quantization,
        provider=candidate.provider,
    )
    policy = QualificationPolicy.for_variant(
        precision=candidate.precision,
        quantization=candidate.quantization,
        format_name=candidate.name,
        direct_identifier_recall_floor=direct_identifier_recall_floor,
    )
    base: dict[str, Any] = {
        "name": candidate.name,
        "runtime": support.runtime,
        "precision": candidate.precision,
        "quantization": candidate.quantization,
        "provider": candidate.provider,
        "target_provider": candidate.target_provider,
        "policy": policy,
        "buckets_evaluated": tuple(
            sorted({request.encoding.token_count for request in requests})
        ),
        "artifacts": _relative_artifacts(candidate.artifact_paths, artifact_root),
        "size_bytes": _artifacts_size(candidate.artifact_paths),
        "evidence": dict(candidate.evidence),
    }
    not_applicable = {"applicable": False}
    rss_before = _peak_rss_bytes()
    try:
        logits, latencies = collect_zero_shot_outputs(
            candidate.runner,
            requests,
            profile=profile,
            graph_span_width=graph_span_width,
        )
    except ExportQualificationError as exc:
        return ZeroShotVariantQualification(
            passed=False,
            reasons=(exc.reason_code,),
            logit_parity={},
            span_parity={},
            direct_identifier_recall={},
            critical_leakage={},
            quantization_delta=None,
            **base,
        )
    rss_after = _peak_rss_bytes()

    reasons: list[str] = []
    reference_rows, candidate_rows = _valid_rows(
        requests, reference_logits, logits, profile=profile
    )
    parity = evaluate_onnx_logit_parity(
        [reference_rows],
        [candidate_rows],
        atol=policy.logit_atol,
        rtol=policy.logit_rtol,
    )
    logit_payload = {
        "within_tolerance": parity.logits_within_tolerance,
        "max_abs_diff": _finite_or_none(parity.max_abs_diff),
        "max_rel_diff": _finite_or_none(parity.max_rel_diff),
        "argmax_mismatches": parity.token_mismatches,
        "rows_evaluated": parity.tokens_evaluated,
        "label_orders": sorted({request.order for request in requests}),
        "gated": policy.gate_logits,
    }
    if policy.gate_logits and not parity.logits_within_tolerance:
        reasons.append(ExportReason.LOGIT_PARITY_FAILED.value)

    if profile.task != TASK_ZERO_SHOT_NER:
        classification = _classification_parity(
            requests, reference_logits, logits, threshold=threshold
        )
        if policy.gate_spans and not classification["ranking_identical"]:
            reasons.append(ExportReason.LABEL_ORDER_PARITY_FAILED.value)
        if policy.gate_spans and not classification["decisions_identical"]:
            reasons.append(ExportReason.DECISION_PARITY_FAILED.value)
        return ZeroShotVariantQualification(
            passed=not reasons,
            reasons=tuple(dict.fromkeys(reasons)),
            logit_parity=logit_payload,
            span_parity=not_applicable,
            direct_identifier_recall=not_applicable,
            critical_leakage=not_applicable,
            quantization_delta=None,
            classification_parity=classification,
            latency_ms=_latency_summary(latencies),
            process_peak_rss_bytes=max(rss_before or 0, rss_after or 0) or None,
            **base,
        )

    reference_spans = _decoded_spans(requests, reference_logits, threshold=threshold)
    candidate_spans = _decoded_spans(requests, logits, threshold=threshold)
    span_payload = _span_parity(
        reference_spans, candidate_spans, tolerance=policy.span_tolerance
    )
    span_payload["gated"] = policy.gate_spans
    span_payload["threshold"] = threshold
    span_payload["offsets_valid"] = _offsets_valid(requests, candidate_spans)
    if policy.gate_spans and not (
        span_payload["identical"] and span_payload["offsets_valid"]
    ):
        reasons.append(ExportReason.SPAN_PARITY_FAILED.value)

    fixtures = [request.fixture.benchmark for request in requests]
    canonical_reference = _canonical_spans(requests, reference_spans)
    canonical_candidate = _canonical_spans(requests, candidate_spans)
    recall_fixtures = _keyed_benchmarks(requests)
    labels = direct_identifier_labels(fixtures)
    reference_recall = recall_by_label(
        recall_fixtures, canonical_reference, labels=labels
    )
    candidate_recall = recall_by_label(
        recall_fixtures, canonical_candidate, labels=labels
    )
    deltas = {
        label: max(reference_recall[label] - candidate_recall[label], 0.0)
        for label in labels
    }
    max_delta = max(deltas.values()) if deltas else None
    floor_failed = direct_identifier_recall_floor is not None and any(
        value < direct_identifier_recall_floor for value in candidate_recall.values()
    )
    delta_failed = max_delta is None or max_delta > policy.recall_delta_limit + 1e-12
    recall_payload = {
        "labels": list(labels),
        "reference": reference_recall,
        "candidate": candidate_recall,
        "per_label_delta": deltas,
        "max_delta": max_delta,
        "limit": policy.recall_delta_limit,
        "floor": direct_identifier_recall_floor,
        "passed": not (delta_failed or floor_failed),
    }
    if delta_failed or floor_failed:
        reasons.append(
            ExportReason.MISSING_EVIDENCE.value
            if max_delta is None
            else ExportReason.DIRECT_IDENTIFIER_RECALL_FAILED.value
        )

    reference_leaks = critical_leaks(recall_fixtures, canonical_reference)
    candidate_leaks = critical_leaks(recall_fixtures, canonical_candidate)
    new_leaks = candidate_leaks - reference_leaks
    leak_payload = {
        "reference_leaked_spans": len(reference_leaks),
        "candidate_leaked_spans": len(candidate_leaks),
        "new_leaks": len(new_leaks),
        "new_leaks_by_label": _count_by_label(new_leaks),
        "limit": policy.max_new_critical_leaks,
        "passed": len(new_leaks) <= policy.max_new_critical_leaks,
    }
    if not leak_payload["passed"]:
        reasons.append(ExportReason.CRITICAL_LEAKAGE_DETECTED.value)

    return ZeroShotVariantQualification(
        passed=not reasons,
        reasons=tuple(dict.fromkeys(reasons)),
        logit_parity=logit_payload,
        span_parity=span_payload,
        direct_identifier_recall=recall_payload,
        critical_leakage=leak_payload,
        quantization_delta=None,
        latency_ms=_latency_summary(latencies),
        process_peak_rss_bytes=max(rss_before or 0, rss_after or 0) or None,
        **base,
    )


def qualify_zero_shot_export(
    *,
    profile: ZeroShotProfile,
    spec: ZeroShotModelSpec,
    prompt: ZeroShotPromptSpec,
    bounds: ZeroShotBounds,
    tokenizer: Any,
    reference_runner: LogitRunner,
    candidates: Sequence[VariantCandidate],
    source: Mapping[str, Any],
    artifact_root: str | Path,
    runtime: str = "onnx",
    artifact_paths: Iterable[str | Path] = (),
    fixtures_path: str | Path | None = None,
    label_prompts: Mapping[str, str] | None = None,
    threshold: float = DEFAULT_THRESHOLD,
    exclude_artifact_names: Iterable[str] = (),
    direct_identifier_recall_floor: float | None = None,
    environment: Mapping[str, Any] | None = None,
) -> ZeroShotQualificationReport:
    """Qualify every candidate of one zero-shot export against its reference.

    Args:
        profile: The export profile.
        spec: Validated architecture facts.
        prompt: Prompt contract of the artifact.
        bounds: Declared bounds of the artifact.
        tokenizer: The artifact's ``tokenizers.Tokenizer``.
        reference_runner: PyTorch (or MLX) reference logits runner.
        candidates: Exported variants to qualify.
        source: Output of
            :func:`~openmed.eval.export_qualification.source_identity`.
        artifact_root: Directory every artifact path is reported relative to.
        runtime: Runtime the candidates run on; only ``"onnx"`` is declared.
        artifact_paths: Extra files (tokenizer, config) to digest.
        fixtures_path: Synthetic suite; defaults to the packaged suite.
        label_prompts: GLiNER only: canonical label to candidate label.
        threshold: Decoder threshold recorded in the artifact contract.
        exclude_artifact_names: Metadata files excluded from digests.
        direct_identifier_recall_floor: Optional absolute recall floor.
        environment: Converter environment; defaults to the installed one.

    Returns:
        The complete report. Call
        :meth:`~openmed.eval.export_qualification.ExportQualificationReport.raise_for_failure`
        to fail closed.

    Raises:
        ExportQualificationError: For runtime, support, suite, or bound
            violations detected before inference.
    """

    support = profile.support_for(runtime)
    fixtures, suite = load_zero_shot_fixtures(
        profile, fixtures_path, label_prompts=label_prompts
    )
    requests = encode_zero_shot_fixtures(
        fixtures,
        profile=profile,
        spec=spec,
        tokenizer=tokenizer,
        prompt=prompt,
        bounds=bounds,
    )
    graph_width = spec.span_width if profile.task == TASK_ZERO_SHOT_NER else None
    reference_logits, _ = collect_zero_shot_outputs(
        reference_runner, requests, profile=profile, graph_span_width=graph_width
    )
    variants = tuple(
        qualify_zero_shot_variant(
            candidate=candidate,
            profile=profile,
            support=support,
            requests=requests,
            reference_logits=reference_logits,
            graph_span_width=graph_width,
            threshold=threshold,
            artifact_root=artifact_root,
            direct_identifier_recall_floor=direct_identifier_recall_floor,
        )
        for candidate in candidates
    )
    all_paths = [
        *artifact_paths,
        *(path for candidate in candidates for path in candidate.artifact_paths),
    ]
    digests = artifact_file_digests(
        all_paths,
        root=artifact_root,
        exclude_names={EXPORT_QUALIFICATION_FILENAME, *exclude_artifact_names},
    )
    return ZeroShotQualificationReport(
        family=profile.family,
        adapter_version=profile.adapter_version,
        runtime=support.runtime,
        support=support,
        source=dict(source),
        tokenizer=zero_shot_tokenizer_identity(tokenizer),
        label_map=label_schema_identity(profile, prompt, bounds),
        io_contract=build_io_contract(profile, bounds, graph_span_width=graph_width),
        environment=dict(environment or converter_environment()),
        suite=suite,
        variants=variants,
        artifacts=digests,
        task=profile.task,
        profile={
            "name": profile.name,
            "contract_version": profile.to_dict()["contract_version"],
            "decoder": profile.decoder_contract(threshold),
            "prompt": prompt.to_dict(),
            "label_orders": list(LABEL_ORDERS),
        },
    )


def zero_shot_tokenizer_identity(tokenizer: Any) -> dict[str, Any]:
    """Return the identity digest of a ``tokenizers.Tokenizer``.

    The digest covers the serialized tokenizer (vocabulary, normalizer,
    pre-tokenizer, model, post-processor, added tokens) so the runtime can
    recompute it from the bundled ``tokenizer.json``.

    Raises:
        ExportQualificationError: ``missing_offset_mapping`` when the object
            cannot serialize itself.
    """

    backend = getattr(tokenizer, "backend_tokenizer", tokenizer)
    identity = tokenizer_identity(
        SimpleNamespace(backend_tokenizer=backend, special_tokens_map={})
    )
    identity["kind"] = "tokenizers_json"
    return identity


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _load_span_fixtures(
    path: str | Path | None,
    label_prompts: Mapping[str, str] | None,
) -> tuple[list[ZeroShotFixture], dict[str, Any]]:
    from openmed.eval.harness import load_fixtures

    suite_path = Path(path) if path is not None else DEFAULT_QUALIFICATION_SUITE
    benchmarks = load_fixtures(suite_path)
    if not benchmarks:
        raise ExportQualificationError(ExportReason.MISSING_EVIDENCE)
    prompts = dict(label_prompts or DEFAULT_GLINER_LABEL_PROMPTS)
    canonical_labels = sorted(
        {
            normalize_label(span.label, fixture.language)
            for fixture in benchmarks
            for span in fixture.gold_spans
        }
    )
    if any(label not in prompts for label in canonical_labels):
        raise ExportQualificationError(ExportReason.MISSING_EVIDENCE)
    candidates = tuple(dict.fromkeys(prompts[label] for label in canonical_labels))
    canonical = {prompts[label]: label for label in canonical_labels}
    fixtures = [
        ZeroShotFixture(
            fixture_id=str(item.fixture_id),
            text=item.text,
            labels=candidates,
            canonical=canonical,
            benchmark=item,
        )
        for item in benchmarks
    ]
    identity = {
        "default": path is None
        or suite_path.resolve() == DEFAULT_QUALIFICATION_SUITE.resolve(),
        "sha256": sha256_file(suite_path),
        "fixture_count": len(benchmarks),
        "gold_span_count": sum(len(item.gold_spans) for item in benchmarks),
        "labels": canonical_labels,
        "candidate_label_count": len(candidates),
        "candidate_labels_sha256": _digest(list(candidates)),
        "languages": sorted({item.language for item in benchmarks}),
        "synthetic": True,
    }
    return fixtures, identity


def _load_classification_fixtures(
    path: str | Path | None,
) -> tuple[list[ZeroShotFixture], dict[str, Any]]:
    suite_path = Path(path) if path is not None else DEFAULT_CLASSIFICATION_SUITE
    try:
        raw = suite_path.read_text(encoding="utf-8")
    except OSError:
        raise ExportQualificationError(ExportReason.MISSING_EVIDENCE) from None
    stripped = raw.strip()
    try:
        if stripped.startswith("["):
            rows = json.loads(stripped)
        else:
            rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    except ValueError:
        raise ExportQualificationError(ExportReason.MISSING_EVIDENCE) from None
    fixtures: list[ZeroShotFixture] = []
    for index, row in enumerate(rows):
        labels = row.get("candidate_labels") if isinstance(row, Mapping) else None
        text = row.get("text") if isinstance(row, Mapping) else None
        if not isinstance(text, str) or not isinstance(labels, list) or not labels:
            raise ExportQualificationError(ExportReason.MISSING_EVIDENCE)
        fixtures.append(
            ZeroShotFixture(
                fixture_id=str(row.get("id") or f"fixture-{index}"),
                text=text,
                labels=tuple(str(label) for label in labels),
                gold_labels=tuple(str(item) for item in row.get("gold_labels") or ()),
            )
        )
    if not fixtures:
        raise ExportQualificationError(ExportReason.MISSING_EVIDENCE)
    label_sets = sorted({fixture.labels for fixture in fixtures})
    identity = {
        "default": path is None
        or suite_path.resolve() == DEFAULT_CLASSIFICATION_SUITE.resolve(),
        "sha256": sha256_file(suite_path),
        "fixture_count": len(fixtures),
        "candidate_label_sets": len(label_sets),
        "max_candidate_labels": max(len(fixture.labels) for fixture in fixtures),
        "candidate_labels_sha256": _digest([list(item) for item in label_sets]),
        "synthetic": True,
    }
    return fixtures, identity


def _valid_rows(
    requests: Sequence[_EncodedRequest],
    reference: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    profile: ZeroShotProfile,
) -> tuple[list[list[float]], list[list[float]]]:
    import numpy as np

    reference_rows: list[list[float]] = []
    candidate_rows: list[list[float]] = []
    for request in requests:
        left = reference[request.key]
        right = candidate[request.key]
        if profile.task == TASK_ZERO_SHOT_NER:
            mask = np.asarray(request.encoding.feed["span_mask"], dtype=bool)[0]
            left_rows = left[0].reshape(-1, left.shape[-1])[mask]
            right_rows = right[0].reshape(-1, right.shape[-1])[mask]
        else:
            left_rows, right_rows = left, right
        reference_rows.extend(left_rows.tolist())
        candidate_rows.extend(right_rows.tolist())
    return reference_rows, candidate_rows


def _decoded_spans(
    requests: Sequence[_EncodedRequest],
    logits: Mapping[str, Any],
    *,
    threshold: float,
) -> dict[str, list[dict[str, Any]]]:
    decoded: dict[str, list[dict[str, Any]]] = {}
    for request in requests:
        result = decode_gliner_spans(
            logits[request.key],
            word_offsets=request.encoding.word_offsets,
            labels=request.labels,
            threshold=threshold,
            flat_ner=True,
            span_width=request.encoding.span_width,
        )
        decoded[request.key] = [
            {"label": span.label, "start": span.start, "end": span.end}
            for span in result.spans
        ]
    return decoded


def _offsets_valid(
    requests: Sequence[_EncodedRequest],
    spans: Mapping[str, Sequence[Mapping[str, Any]]],
) -> bool:
    for request in requests:
        starts = {start for start, _ in request.encoding.word_offsets}
        ends = {end for _, end in request.encoding.word_offsets}
        limit = len(request.fixture.text)
        for span in spans.get(request.key, ()):
            start, end = int(span["start"]), int(span["end"])
            if not (0 <= start < end <= limit) or start not in starts:
                return False
            if end not in ends:
                return False
    return True


def _keyed_benchmarks(requests: Sequence[_EncodedRequest]) -> list[Any]:
    """Return one harness fixture per request, keyed by the request key."""
    from dataclasses import replace

    return [
        replace(request.fixture.benchmark, fixture_id=request.key)
        for request in requests
    ]


def _canonical_spans(
    requests: Sequence[_EncodedRequest],
    spans: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, list[dict[str, Any]]]:
    canonical: dict[str, list[dict[str, Any]]] = {}
    for request in requests:
        mapping = request.fixture.canonical
        canonical[request.key] = [
            {
                "label": mapping.get(str(span["label"]), str(span["label"])),
                "start": span["start"],
                "end": span["end"],
            }
            for span in spans.get(request.key, ())
        ]
    return canonical


def _classification_parity(
    requests: Sequence[_EncodedRequest],
    reference: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    threshold: float,
) -> dict[str, Any]:
    ranking_mismatches = 0
    decision_mismatches = 0
    top1_mismatches = 0
    accepted_reference = 0
    accepted_candidate = 0
    for request in requests:
        left = decode_gliclass(
            reference[request.key], labels=request.labels, threshold=threshold
        )
        right = decode_gliclass(
            candidate[request.key], labels=request.labels, threshold=threshold
        )
        left_rank = [item.label_index for item in left.scores]
        right_rank = [item.label_index for item in right.scores]
        left_accept = {item.label_index for item in left.accepted}
        right_accept = {item.label_index for item in right.accepted}
        accepted_reference += len(left_accept)
        accepted_candidate += len(right_accept)
        ranking_mismatches += int(left_rank != right_rank)
        top1_mismatches += int(left_rank[:1] != right_rank[:1])
        decision_mismatches += int(left_accept != right_accept)
    return {
        "requests_compared": len(requests),
        "ranking_identical": ranking_mismatches == 0,
        "ranking_mismatches": ranking_mismatches,
        "top1_mismatches": top1_mismatches,
        "decisions_identical": decision_mismatches == 0,
        "decision_mismatches": decision_mismatches,
        "reference_accepted": accepted_reference,
        "candidate_accepted": accepted_candidate,
        "threshold": threshold,
        "label_orders": sorted({request.order for request in requests}),
    }


def _digest(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


__all__ = [
    "DEFAULT_CLASSIFICATION_SUITE",
    "LABEL_ORDERS",
    "ZeroShotFixture",
    "ZeroShotQualificationReport",
    "ZeroShotVariantQualification",
    "collect_zero_shot_outputs",
    "encode_zero_shot_fixtures",
    "load_zero_shot_fixtures",
    "qualify_zero_shot_export",
    "qualify_zero_shot_variant",
    "zero_shot_tokenizer_identity",
]
