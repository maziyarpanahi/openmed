"""Offline Chinese clinical NER scorecard with a synthetic PHI leakage gate."""

from __future__ import annotations

import hashlib
import math
import os
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence

from openmed.core.labels import CANONICAL_LABELS
from openmed.core.quality_gates import (
    validate_entity_spans,
    validate_entity_spans_strict,
)
from openmed.eval.datasets.cmeee import load_cmeee
from openmed.eval.harness import BenchmarkFixture, ModelRunner
from openmed.eval.metrics import (
    EvalSpan,
    compute_exact_span_f1,
    compute_latency_summary,
    normalize_eval_spans,
)
from openmed.eval.report import BenchmarkReport

CHINESE_CLINICAL_NER = "chinese-clinical-ner"
MULTILINGUAL_FALLBACK_MODEL = "OpenMed/privacy-filter-multilingual"
DEFAULT_SYNTHETIC_CMEEE_PATH = (
    Path(__file__).resolve().parents[1] / "fixtures" / "cmeee_zh_synthetic.jsonl"
)

#: Environment variable pointing at a user-provisioned local checkpoint
#: directory. It mirrors ``OPENMED_CMEEE_PATH``: OpenMed never downloads,
#: vendors, or discovers Chinese clinical NER weights.
CHINESE_CLINICAL_NER_MODEL_DIR_ENV = "OPENMED_ZH_CLINICAL_NER_MODEL_DIR"

#: A local checkpoint directory must carry a config plus at least one
#: tokenizer artifact and one weight artifact. The weight suffixes cover the
#: PyTorch, TensorFlow, Flax, ONNX, GGUF, TFLite, and MLX artifacts a user may
#: legitimately have provisioned, so a valid checkpoint is never rejected for
#: shipping in a non-PyTorch format.
CHINESE_LOCAL_MODEL_CONFIG_FILE = "config.json"
CHINESE_LOCAL_MODEL_TOKENIZER_FILES: tuple[str, ...] = (
    "sentencepiece.bpe.model",
    "spiece.model",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.txt",
)
CHINESE_LOCAL_MODEL_WEIGHT_SUFFIXES: tuple[str, ...] = (
    ".bin",
    ".ckpt",
    ".gguf",
    ".h5",
    ".msgpack",
    ".npz",
    ".onnx",
    ".pt",
    ".pth",
    ".safetensors",
    ".tflite",
)

RedactionRunner = Callable[[BenchmarkFixture, tuple[EvalSpan, ...]], str]
Clock = Callable[[], float]


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


class ChineseClinicalNerAssetUnavailable(RuntimeError):
    """Raised when no usable local Chinese clinical NER checkpoint is configured.

    The message always names :data:`CHINESE_CLINICAL_NER_MODEL_DIR_ENV` so that
    an optional test can skip with actionable configuration guidance instead of
    failing an offline default run.
    """


class ChineseClinicalNerContractError(RuntimeError):
    """Raised when adapter predictions break the span or label contract.

    Evidence is deliberately surface-free: it carries offsets, canonical
    labels, and problem codes, never the predicted span text.
    """

    def __init__(self, findings: Sequence[Mapping[str, Any]]) -> None:
        super().__init__(
            "Chinese clinical NER conformance contract failed: "
            f"{len(findings)} prediction(s) broke the offset or label contract"
        )
        self.findings = tuple(dict(finding) for finding in findings)


@dataclass(frozen=True)
class ChineseLocalModelAsset:
    """A user-provisioned local checkpoint directory accepted for conformance.

    Serialized evidence is path-free and operator-name-free: a SHA-256
    fingerprint of the resolved path, the whitelisted tokenizer file names,
    and the whitelisted weight-artifact suffixes with a count. The directory
    basename is deliberately excluded because it is operator-controlled and
    can carry a username (``/home/alice`` would otherwise publish ``alice``).
    The absolute path stays on the object for the caller's own loader and is
    never written into a report.
    """

    path: Path
    config_file: str
    tokenizer_files: tuple[str, ...]
    weight_file_suffixes: tuple[str, ...]
    weight_file_count: int
    fingerprint: str

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-ready asset evidence without the local path."""

        return {
            "config_file": self.config_file,
            "fingerprint": self.fingerprint,
            "path_env": CHINESE_CLINICAL_NER_MODEL_DIR_ENV,
            "tokenizer_files": list(self.tokenizer_files),
            "weight_file_count": self.weight_file_count,
            "weight_file_suffixes": list(self.weight_file_suffixes),
        }


class ChineseClinicalNerAdapter(Protocol):  # pragma: no cover - interface only
    """Opt-in adapter contract for a locally provisioned Chinese NER model.

    Implementations wrap weights the user provisioned outside this repository.
    The conformance runner only ever calls these two methods, so it can be
    exercised offline with a deterministic stub.
    """

    model_name: str

    def predict_spans(self, text: str, *, language: str) -> Sequence[Any]: ...

    def redact(self, text: str, *, spans: Sequence[Any]) -> str: ...


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
    """Return the suite license, model, and redistribution disclaimers.

    The model notice is model-card evidence, so it names the routed default
    rather than a fixed claim: the ``zh`` language pack now resolves to a
    dedicated Chinese PII checkpoint, and callers evaluating a different local
    model must record that substitution in their own model card.
    """

    default_model = _zh_default_model()
    return {
        "data_boundary": (
            "CMeEE, CBLUE, and eHealth records are user-supplied local inputs; "
            "OpenMed bundles only synthetic smoke records."
        ),
        "language": "zh",
        "model_evidence": {
            "dedicated_zh_model": default_model != MULTILINGUAL_FALLBACK_MODEL,
            "routed_default_model": default_model,
            "weights_bundled": False,
        },
        "model_notice": (
            f"The routed Chinese default is {default_model}. It is a PII "
            "checkpoint scored here for entity coverage, not a CMeEE-trained "
            "clinical NER checkpoint; OpenMed bundles no weights, and any "
            "dedicated local model substituted for it must be recorded as "
            "model-card evidence with its own license and provenance."
        ),
        "redistribution": "no licensed corpus records or model weights are bundled",
        "suite": CHINESE_CLINICAL_NER,
        "task": "clinical_ner_with_phi_leakage_gate",
    }


def _zh_default_model() -> str:
    """Return the model the ``zh`` language pack currently routes to."""

    from openmed.core.language_pack_catalog import (
        DEFAULT_MODEL_PLACEHOLDER_LANGUAGES,
        DEFAULT_PII_MODELS,
    )

    if "zh" in DEFAULT_MODEL_PLACEHOLDER_LANGUAGES:
        return MULTILINGUAL_FALLBACK_MODEL
    return str(DEFAULT_PII_MODELS.get("zh") or MULTILINGUAL_FALLBACK_MODEL)


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
    if min_per_label_recall is not None and not 0.0 <= min_per_label_recall <= 1.0:
        raise ValueError("min_per_label_recall must be between 0 and 1")

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


def resolve_chinese_local_model_asset(
    path: str | Path | None = None,
) -> ChineseLocalModelAsset:
    """Resolve a user-provisioned local checkpoint directory.

    Args:
        path: Explicit checkpoint directory. When omitted the directory is read
            from :data:`CHINESE_CLINICAL_NER_MODEL_DIR_ENV`.

    Returns:
        The validated :class:`ChineseLocalModelAsset`.

    Raises:
        ChineseClinicalNerAssetUnavailable: When nothing is configured, the
            directory is unreadable, or it is missing the config, tokenizer,
            or weight artifacts. Every message names the environment variable
            to set, and unreadable directories surface here rather than as a
            bare ``OSError`` so optional tests can still skip cleanly.
    """

    raw = (
        path if path is not None else os.environ.get(CHINESE_CLINICAL_NER_MODEL_DIR_ENV)
    )
    if raw is None or not str(raw).strip() or str(raw).strip() in {"", "."}:
        raise ChineseClinicalNerAssetUnavailable(
            f"{CHINESE_CLINICAL_NER_MODEL_DIR_ENV} is not set: point it at a "
            "local directory holding a Chinese clinical NER checkpoint you are "
            "licensed to use. OpenMed never downloads or bundles these weights."
        )

    try:
        directory = Path(str(raw)).expanduser()
        if not directory.is_dir():
            raise ChineseClinicalNerAssetUnavailable(
                f"{CHINESE_CLINICAL_NER_MODEL_DIR_ENV} is configured but does "
                "not point at an existing directory; provision the checkpoint "
                "locally and re-run."
            )

        resolved = directory.resolve()
        if not (resolved / CHINESE_LOCAL_MODEL_CONFIG_FILE).is_file():
            raise ChineseClinicalNerAssetUnavailable(
                f"{CHINESE_CLINICAL_NER_MODEL_DIR_ENV} is configured but the "
                f"directory has no {CHINESE_LOCAL_MODEL_CONFIG_FILE}."
            )

        names = sorted(item.name for item in resolved.iterdir() if item.is_file())
    except OSError as exc:
        raise ChineseClinicalNerAssetUnavailable(
            f"{CHINESE_CLINICAL_NER_MODEL_DIR_ENV} is configured but the "
            f"directory could not be read ({exc.__class__.__name__}); check "
            "the path and its permissions."
        ) from exc

    tokenizer_files = tuple(
        name for name in names if name in CHINESE_LOCAL_MODEL_TOKENIZER_FILES
    )
    if not tokenizer_files:
        expected = ", ".join(CHINESE_LOCAL_MODEL_TOKENIZER_FILES)
        raise ChineseClinicalNerAssetUnavailable(
            f"{CHINESE_CLINICAL_NER_MODEL_DIR_ENV} is configured but the "
            f"directory has no tokenizer artifact (expected one of: {expected})."
        )

    weight_suffixes = tuple(
        sorted(
            {
                Path(name).suffix
                for name in names
                if Path(name).suffix in CHINESE_LOCAL_MODEL_WEIGHT_SUFFIXES
            }
        )
    )
    weight_count = sum(
        1 for name in names if Path(name).suffix in CHINESE_LOCAL_MODEL_WEIGHT_SUFFIXES
    )
    if not weight_suffixes:
        expected = ", ".join(CHINESE_LOCAL_MODEL_WEIGHT_SUFFIXES)
        raise ChineseClinicalNerAssetUnavailable(
            f"{CHINESE_CLINICAL_NER_MODEL_DIR_ENV} is configured but the "
            f"directory has no model weights (expected one of: {expected})."
        )

    return ChineseLocalModelAsset(
        path=resolved,
        config_file=CHINESE_LOCAL_MODEL_CONFIG_FILE,
        tokenizer_files=tokenizer_files,
        weight_file_suffixes=weight_suffixes,
        weight_file_count=weight_count,
        fingerprint=(
            "sha256:" + hashlib.sha256(resolved.as_posix().encode("utf-8")).hexdigest()
        ),
    )


def chinese_local_model_asset_skip_reason(
    path: str | Path | None = None,
) -> str | None:
    """Return an actionable skip reason, or ``None`` when an asset is usable.

    Optional model tests call this instead of resolving directly so a missing
    or unreadable checkpoint skips with configuration guidance rather than
    erroring.
    """

    try:
        resolve_chinese_local_model_asset(path)
    except ChineseClinicalNerAssetUnavailable as exc:
        return str(exc)
    return None


def run_chinese_clinical_ner_conformance(
    adapter: ChineseClinicalNerAdapter,
    *,
    fixtures: Sequence[BenchmarkFixture] | None = None,
    asset: ChineseLocalModelAsset | None = None,
    device: str = "cpu",
    generated_at: str | None = None,
    min_per_label_recall: float | None = None,
    fail_on_leakage: bool = True,
    clock: Clock | None = None,
) -> BenchmarkReport:
    """Score an opt-in local-model adapter against the shared suite contract.

    Every predicted span is enforced, not merely warned about: a span whose
    offsets do not slice its own text out of the fixture, a span outside the
    fixture bounds, an inverted or empty span, or a label outside
    ``CANONICAL_LABELS`` raises :class:`ChineseClinicalNerContractError`.
    Overlapping and nested spans are explicitly allowed, because nested
    entities are normal in CMeEE-shaped Chinese clinical data.

    On top of the pre-existing suite (which supplies ``exact_span_f1``,
    ``per_label``, and ``phi_token_leakage``), this wrapper adds per-fixture
    latency percentiles and PHI-free local-asset evidence.

    Scope limits, stated plainly:

    * This proves the conformance *contract* and the skip semantics, not the
      measured quality of any particular checkpoint. Quality numbers for a
      real checkpoint only exist for the caller's own opt-in run against
      their own licensed corpus.
    * The PHI-leakage gate judges de-identification behaviour. A clinical
      *entity* checkpoint (a CMeEE-trained model emitting ``bod``/``dis``/
      ``sym`` and no PHI types) will legitimately show leakage on the bundled
      synthetic PHI record, because it is not a de-identifier. Pass
      ``fail_on_leakage=False`` for such a model and read the leakage block as
      evidence rather than as a verdict.

    Args:
        adapter: Wrapper around locally provisioned weights.
        fixtures: Benchmark fixtures; defaults to the bundled synthetic set.
        asset: Resolved local checkpoint evidence to record in the report.
        device: Device label recorded on the report.
        generated_at: Optional timestamp override.
        min_per_label_recall: Optional per-label recall floor.
        fail_on_leakage: Raise when a synthetic identifier survives redaction.
        clock: Monotonic clock injected for deterministic latency in tests.

    Returns:
        A :class:`BenchmarkReport` whose metrics add a ``latency`` block and
        whose metadata adds ``local_model_asset``.

    Raises:
        ChineseClinicalNerContractError: When a prediction breaks the span or
            canonical-label contract.
        ChineseClinicalNerLeakageError: When ``fail_on_leakage`` is set and a
            synthetic identifier survived redaction.
        ValueError: When the adapter exposes no usable ``model_name``.
    """

    model_name = str(getattr(adapter, "model_name", "") or "").strip()
    if not model_name:
        raise ValueError("Chinese clinical NER adapter must expose a model_name")

    resolved_fixtures = (
        list(fixtures) if fixtures is not None else load_chinese_clinical_ner_fixtures()
    )
    now = clock or time.perf_counter
    latencies_ms: list[float] = []
    contract_findings: list[dict[str, Any]] = []

    def runner(
        fixture: BenchmarkFixture,
        model_name: str,
        run_device: str,
    ) -> Sequence[Any]:
        _ = (model_name, run_device)
        started = now()
        predicted = adapter.predict_spans(fixture.text, language=fixture.language)
        elapsed_ms = (now() - started) * 1000.0
        if not math.isfinite(elapsed_ms) or elapsed_ms < 0.0:
            elapsed_ms = 0.0
        latencies_ms.append(elapsed_ms)
        contract_findings.extend(_contract_findings(fixture, predicted))
        return predicted

    def redactor(
        fixture: BenchmarkFixture,
        predicted: tuple[EvalSpan, ...],
    ) -> str:
        return adapter.redact(fixture.text, spans=predicted)

    report = run_chinese_clinical_ner_suite(
        resolved_fixtures,
        model_name=model_name,
        runner=runner,
        redactor=redactor,
        device=device,
        generated_at=generated_at,
        min_per_label_recall=min_per_label_recall,
        fail_on_leakage=False,
    )

    if contract_findings:
        raise ChineseClinicalNerContractError(contract_findings)

    metadata = {
        **dict(report.metadata),
        "conformance": (
            "opt-in local-asset conformance: proves the adapter contract and "
            "skip semantics, not the measured quality of any checkpoint"
        ),
        "local_model_asset": asset.to_dict() if asset is not None else None,
    }
    metrics = {
        **dict(report.metrics),
        "latency": compute_latency_summary(latencies_ms).to_dict(),
    }
    conformance_report = BenchmarkReport(
        suite=report.suite,
        model_name=report.model_name,
        device=report.device,
        fixture_count=report.fixture_count,
        generated_at=report.generated_at,
        metadata=metadata,
        metrics=metrics,
    )
    if fail_on_leakage and metrics["phi_token_leakage"]["leaked_tokens"]:
        raise ChineseClinicalNerLeakageError(conformance_report)
    return conformance_report


def _contract_findings(
    fixture: BenchmarkFixture,
    predicted: Sequence[Any],
) -> list[dict[str, Any]]:
    """Return surface-free findings for predictions that break the contract.

    The contract is that every predicted span is well formed against the
    fixture text -- in bounds, non-inverted, non-empty, carrying a canonical
    label, and (when the adapter supplies its own surface) exactly aligned to
    the characters it claims. It is deliberately *not* "the prediction equals
    gold": a correctly formed but wrong span is a quality failure that
    ``exact_span_f1`` already reports, not a contract breach.

    Findings never carry the predicted or source surface, only offsets,
    canonical labels, and problem codes.
    """

    findings: list[dict[str, Any]] = []
    text_length = len(fixture.text)

    normalized = tuple(
        normalize_eval_spans(
            predicted,
            default_language=fixture.language,
            source_text=fixture.text,
        )
    )
    strict = validate_entity_spans_strict(
        [span.to_entity() for span in normalized],
        fixture.text,
    )
    invalid_indices = {issue.index for issue in strict.offending_spans}
    problem_codes = {
        issue.index: tuple(sorted(issue.problems)) for issue in strict.offending_spans
    }

    for index, span in enumerate(normalized):
        problems: list[str] = []
        if index in invalid_indices:
            problems.extend(problem_codes.get(index, ()))
        if span.start < 0 or span.end > text_length:
            problems.append("span_out_of_bounds")
        if span.start >= span.end:
            problems.append("span_not_positive_length")
        if span.label not in CANONICAL_LABELS:
            problems.append("label_not_canonical")

        raw = predicted[index] if index < len(predicted) else None
        raw_text = _raw_span_text(raw)
        if (
            raw_text
            and 0 <= span.start < span.end <= text_length
            and raw_text != fixture.text[span.start : span.end]
        ):
            problems.append("span_text_offset_mismatch")

        if problems:
            findings.append(
                {
                    "fixture_id": fixture.fixture_id,
                    "start": span.start,
                    "end": span.end,
                    "label": span.label,
                    "problems": sorted(set(problems)),
                }
            )
    return findings


def _raw_span_text(raw: Any) -> str | None:
    """Return an adapter-supplied span surface when one is present."""

    if raw is None:
        return None
    if isinstance(raw, Mapping):
        value = raw.get("text")
    else:
        value = getattr(raw, "text", None)
    return value if isinstance(value, str) and value else None


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
    "CHINESE_CLINICAL_NER_MODEL_DIR_ENV",
    "CHINESE_LOCAL_MODEL_CONFIG_FILE",
    "CHINESE_LOCAL_MODEL_TOKENIZER_FILES",
    "CHINESE_LOCAL_MODEL_WEIGHT_SUFFIXES",
    "DEFAULT_SYNTHETIC_CMEEE_PATH",
    "MULTILINGUAL_FALLBACK_MODEL",
    "ChineseClinicalNerAdapter",
    "ChineseClinicalNerAssetUnavailable",
    "ChineseClinicalNerContractError",
    "ChineseClinicalNerLeakageError",
    "ChineseLocalModelAsset",
    "PhiTokenLeakageFinding",
    "RedactionRunner",
    "chinese_clinical_ner_metadata",
    "chinese_local_model_asset_skip_reason",
    "load_chinese_clinical_ner_fixtures",
    "resolve_chinese_local_model_asset",
    "run_chinese_clinical_ner_conformance",
    "run_chinese_clinical_ner_suite",
    "run_synthetic_chinese_clinical_ner_smoke",
]
