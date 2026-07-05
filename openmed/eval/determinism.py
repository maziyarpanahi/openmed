"""Determinism regression harness for the public de-identification API.

OM-822 -- Deterministic-output guarantees with a same-input-same-spans
regression harness.

This module runs :func:`openmed.deidentify`, :func:`openmed.extract_pii`, and
:func:`openmed.analyze_text` over a fixed, fully synthetic corpus with a fixed
seed multiple times and asserts that the detected spans, labels, and applied
replacements are byte-identical across runs (and, where the caller opts in,
across fresh interpreter processes).

Design notes
------------
* **Offline and model-free.** Real model inference is bit-identical only within
  a fixed hardware/accelerator/library stack, so the harness injects a
  deterministic in-process detector (a fake :class:`ModelLoader`) through the
  public ``loader=`` seam. This exercises the *entire* real pipeline --
  segmentation, span normalization, smart merging, the deterministic safety
  sweep, redaction, surrogate generation, and the quality gates -- which is
  precisely where dict/set ordering and unseeded randomness would leak. The
  harness therefore proves determinism of the *pipeline*, not of the model
  weights (see ``Out of scope`` in the task spec).
* **PHI-free diagnostics.** Every record the harness emits or hashes contains
  only offsets, canonical labels, confidences, and salted hashes of applied
  replacements. Raw entity text and raw surrogate values never enter a report,
  a golden file, or a divergence diagnostic.
* **Single request-scoped seed.** Stochastic steps (surrogate replacement, date
  shifting) are pinned from one seed / patient-key + secret per corpus item, so
  the whole run is reproducible from the corpus definition alone.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from openmed.core.repro_hash import canonicalize_span_records, compute_span_set_hash

__all__ = [
    "DETERMINISM_CORPUS_VERSION",
    "CorpusItem",
    "DeterministicDetector",
    "DeterministicLoader",
    "DivergenceError",
    "ItemDeterminismReport",
    "SyntheticSpan",
    "build_corpus_signature",
    "default_corpus",
    "run_api_once",
    "run_determinism_check",
]

DETERMINISM_CORPUS_VERSION = "openmed.determinism_corpus.v1"

# Public API entrypoints exercised by the harness.
_API_NAMES = ("extract_pii", "deidentify", "analyze_text")


class DivergenceError(AssertionError):
    """Raised when a determinism run diverges between iterations.

    The message carries only offsets, labels, and hashes -- never raw text --
    so that a failing CI log cannot leak synthetic (or, in a misconfigured
    deployment, real) identifiers.
    """


@dataclass(frozen=True)
class SyntheticSpan:
    """A single synthetic gold span within a corpus item.

    Attributes:
        word: The exact synthetic substring the deterministic detector emits.
        label: The entity label the detector assigns (canonical PII label).
        start: Character start offset of ``word`` in the item text.
        end: Character end offset of ``word`` in the item text.
        confidence: Fixed confidence the detector reports.
    """

    word: str
    label: str
    start: int
    end: int
    confidence: float = 0.97


@dataclass(frozen=True)
class CorpusItem:
    """A fixed synthetic document and the de-identification knobs to apply.

    Attributes:
        item_id: Stable identifier used in reports (no PHI).
        text: The synthetic document text (contains only invented identifiers).
        language: ISO 639-1 language code passed to the API.
        spans: Synthetic gold spans the deterministic detector emits.
        method: De-identification method for :func:`openmed.deidentify`.
        seed: Optional seed pinning surrogate replacement.
        patient_key: Optional stable key pinning date-shift offsets.
        date_shift_secret: Optional HMAC secret for patient-keyed date shifting.
    """

    item_id: str
    text: str
    language: str
    spans: tuple[SyntheticSpan, ...]
    method: str = "mask"
    seed: int | None = None
    patient_key: str | None = None
    date_shift_secret: str | None = None

    def prediction_rows(self) -> list[dict[str, Any]]:
        """Return raw token-classification rows for the deterministic detector."""
        return [
            {
                "word": span.word,
                "entity_group": span.label,
                "score": span.confidence,
                "start": span.start,
                "end": span.end,
            }
            for span in self.spans
        ]

    def validate(self) -> None:
        """Assert that declared spans match the synthetic text exactly."""
        for span in self.spans:
            actual = self.text[span.start : span.end]
            if actual != span.word:
                raise ValueError(
                    f"corpus item {self.item_id!r} span {span.label} "
                    f"[{span.start}:{span.end}] does not match declared text"
                )


class DeterministicDetector:
    """A deterministic token-classification callable used as a fake pipeline.

    It returns fixed rows for the exact text it was built for. Any dict/set
    ordering in downstream merging is deliberately left to the real pipeline so
    the harness can catch ordering leaks there.
    """

    def __init__(self, rows_by_text: Mapping[str, Sequence[Mapping[str, Any]]]):
        self._rows_by_text = {key: list(value) for key, value in rows_by_text.items()}
        # ``analyze_text`` reads ``tokenizer.model_max_length``; expose a stub.
        self.tokenizer = _StubTokenizer()

    def __call__(self, text: Any, **_: Any) -> Any:
        if isinstance(text, list):
            return [self._rows_for(chunk) for chunk in text]
        return self._rows_for(text)

    def _rows_for(self, chunk: str) -> list[dict[str, Any]]:
        rows = self._rows_by_text.get(chunk)
        if rows is not None:
            return [dict(row) for row in rows]
        # The API may segment the item into sentences/chunks. Re-project any
        # declared row whose word appears in this chunk, recomputing offsets so
        # the emitted spans stay valid for the chunk the pipeline passed in.
        projected: list[dict[str, Any]] = []
        for source_rows in self._rows_by_text.values():
            for row in source_rows:
                word = row["word"]
                index = chunk.find(word)
                if index != -1 and chunk[index : index + len(word)] == word:
                    reprojected = dict(row)
                    reprojected["start"] = index
                    reprojected["end"] = index + len(word)
                    projected.append(reprojected)
        return projected


class DeterministicLoader:
    """A fake ``ModelLoader`` that drives the real pipeline with fixed spans.

    Passed through the public ``loader=`` argument of the API functions, so no
    private seam is touched and the whole downstream pipeline runs unchanged.
    """

    def __init__(self, item: CorpusItem):
        item.validate()
        self.config = None
        self._detector = DeterministicDetector({item.text: item.prediction_rows()})

    def create_pipeline(self, model_name: str, **_: Any) -> DeterministicDetector:
        return self._detector

    def get_max_sequence_length(
        self, model_name: str, tokenizer: Any | None = None
    ) -> int:
        return 512


class _StubTokenizer:
    model_max_length = 512


@dataclass(frozen=True)
class ItemDeterminismReport:
    """PHI-free per-item determinism outcome.

    Attributes:
        item_id: Corpus item identifier.
        api_hashes: Mapping of API name -> span-set / output content hash.
        iterations: Number of in-process iterations that agreed.
    """

    item_id: str
    api_hashes: dict[str, str]
    iterations: int


@dataclass
class _AggregatedReport:
    """Internal collector for a full determinism run."""

    corpus_signature: str
    items: list[ItemDeterminismReport] = field(default_factory=list)

    def to_golden_mapping(self) -> dict[str, Any]:
        """Return the PHI-free golden payload for a checked-in golden file."""
        return {
            "schema_version": DETERMINISM_CORPUS_VERSION,
            "corpus_signature": self.corpus_signature,
            "items": {
                report.item_id: dict(sorted(report.api_hashes.items()))
                for report in sorted(self.items, key=lambda item: item.item_id)
            },
        }


def default_corpus() -> tuple[CorpusItem, ...]:
    """Return the fixed, fully synthetic determinism corpus.

    Every identifier below is invented for testing. No corpus item contains real
    protected health information. The corpus intentionally covers the three
    replacement families that carry stochastic steps so that seeding regressions
    are caught: ``mask`` (deterministic by construction), ``replace`` (seeded
    surrogates), and ``shift_dates`` (patient-keyed HMAC offsets).
    """

    items: list[CorpusItem] = []

    # 1. Masking -- deterministic by construction, multiple labels, ordering.
    text_mask = "Patient Casey Example emailed casey.example@example.org today."
    items.append(
        CorpusItem(
            item_id="mask_name_email",
            text=text_mask,
            language="en",
            spans=(
                SyntheticSpan("Casey Example", "NAME", 8, 21),
                SyntheticSpan(
                    "casey.example@example.org",
                    "EMAIL",
                    30,
                    55,
                ),
            ),
            method="mask",
        )
    )

    # 2. Seeded replacement -- surrogates must be stable across runs and process
    #    boundaries for a fixed seed.
    text_replace = "Contact Jordan Rivera at the Example Clinic front desk."
    items.append(
        CorpusItem(
            item_id="replace_seeded_name",
            text=text_replace,
            language="en",
            spans=(SyntheticSpan("Jordan Rivera", "NAME", 8, 21),),
            method="replace",
            seed=20220822,
        )
    )

    # 3. Patient-keyed date shifting -- HMAC offset must be reproducible.
    text_dates = "Admitted 01/15/1970 and discharged 01/22/1970 for observation."
    items.append(
        CorpusItem(
            item_id="shift_dates_patient_keyed",
            text=text_dates,
            language="en",
            spans=(
                SyntheticSpan("01/15/1970", "DATE", 9, 19),
                SyntheticSpan("01/22/1970", "DATE", 35, 45),
            ),
            method="shift_dates",
            patient_key="synthetic-patient-0001",
            date_shift_secret="synthetic-date-shift-secret",
        )
    )

    # 4. Multiple same-label mentions -- exercises merge tie-breaks / ordering.
    text_multi = "Nurses Ada Stone and Ada Stone reviewed the synthetic chart."
    items.append(
        CorpusItem(
            item_id="mask_repeated_name",
            text=text_multi,
            language="en",
            spans=(
                SyntheticSpan("Ada Stone", "NAME", 7, 16),
                SyntheticSpan("Ada Stone", "NAME", 21, 30),
            ),
            method="mask",
        )
    )

    return tuple(items)


def build_corpus_signature(corpus: Sequence[CorpusItem]) -> str:
    """Return a PHI-free signature over the corpus definition.

    The signature binds item ids, offsets, labels, and knobs (not raw text) so a
    golden file cannot silently drift from the corpus it was generated for.
    """

    signature_spans: list[dict[str, Any]] = []
    for item in corpus:
        for span in item.spans:
            signature_spans.append(
                {
                    "start": span.start,
                    "end": span.end,
                    "label": span.label,
                    "action": f"{item.item_id}:{item.method}",
                }
            )
    return compute_span_set_hash(
        canonicalize_span_records(signature_spans),
        method="corpus",
        text_length=sum(len(item.text) for item in corpus),
    )


def run_api_once(item: CorpusItem, api_name: str) -> dict[str, Any]:
    """Run one API over one corpus item and return a PHI-free result view.

    Returns a mapping with the content hash and the canonical (offsets/labels
    only) span records so callers can diff divergences precisely.
    """

    import openmed

    loader = DeterministicLoader(item)

    if api_name == "extract_pii":
        result = openmed.extract_pii(
            item.text,
            loader=loader,
            lang=item.language,
        )
        spans = list(result.entities)
        content_hash = compute_span_set_hash(
            spans, text_length=len(item.text), method="extract_pii"
        )
    elif api_name == "analyze_text":
        result = openmed.analyze_text(
            item.text,
            model_name="determinism-fixture-model",
            loader=loader,
        )
        spans = list(result.entities)
        content_hash = compute_span_set_hash(
            spans, text_length=len(item.text), method="analyze_text"
        )
    elif api_name == "deidentify":
        kwargs: dict[str, Any] = {
            "method": item.method,
            "loader": loader,
            "lang": item.language,
        }
        if item.seed is not None:
            kwargs["seed"] = item.seed
        if item.patient_key is not None:
            kwargs["patient_key"] = item.patient_key
        if item.date_shift_secret is not None:
            kwargs["date_shift_secret"] = item.date_shift_secret
        result = openmed.deidentify(item.text, **kwargs)
        spans = list(result.pii_entities)
        # The span records already fold in each applied replacement via a salted
        # ``replacement_digest``. Additionally bind a salted digest of the full
        # de-identified output so drift *outside* detected spans (for example a
        # stray whitespace or ordering change in redaction) is also caught. The
        # raw output text is never stored -- only its salted hash.
        output_digest = _output_digest(result.deidentified_text)
        content_hash = compute_span_set_hash(
            [
                *spans,
                {
                    "start": -1,
                    "end": -1,
                    "label": "__deidentified_output__",
                    "surrogate": output_digest,
                },
            ],
            text_length=len(item.text),
            method=f"deidentify:{item.method}",
        )
    else:  # pragma: no cover - guarded by _API_NAMES
        raise ValueError(f"unknown api_name: {api_name!r}")

    return {
        "hash": content_hash,
        "spans": canonicalize_span_records(spans),
    }


def _output_digest(text: str) -> str:
    """Return a salted hex digest of de-identified output (never the raw text)."""
    import hashlib

    return hashlib.sha256(
        f"openmed.determinism.output.v1:{text}".encode("utf-8")
    ).hexdigest()


def _diff_records(
    first: Sequence[Mapping[str, Any]],
    second: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Return PHI-free per-span differences between two canonical span lists."""
    diffs: list[dict[str, Any]] = []
    length = max(len(first), len(second))
    for index in range(length):
        left = first[index] if index < len(first) else None
        right = second[index] if index < len(second) else None
        if left != right:
            diffs.append({"index": index, "run_a": left, "run_b": right})
    return diffs


def run_determinism_check(
    corpus: Sequence[CorpusItem] | None = None,
    *,
    iterations: int = 5,
    api_names: Sequence[str] = _API_NAMES,
    runner: Callable[[CorpusItem, str], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run the determinism harness and return a PHI-free golden mapping.

    For every corpus item and every API, the harness runs ``iterations`` times in
    the current process and asserts that the canonical span records (and thus the
    content hash) are byte-identical across all iterations. On divergence it
    raises :class:`DivergenceError` carrying offsets/labels/hashes only.

    Args:
        corpus: Corpus items to run. Defaults to :func:`default_corpus`.
        iterations: Number of in-process repeats per (item, api). Must be >= 2 to
            detect run-to-run drift.
        api_names: Which public APIs to exercise.
        runner: Optional override of :func:`run_api_once`, used by tests to inject
            a deliberately nondeterministic variant.

    Returns:
        A PHI-free golden mapping (schema version, corpus signature, per-item
        content hashes) suitable for writing to or comparing against a golden
        file.

    Raises:
        DivergenceError: If any (item, api) diverges across iterations.
        ValueError: If ``iterations`` < 2.
    """

    if iterations < 2:
        raise ValueError("iterations must be >= 2 to detect run-to-run drift")

    active_corpus = tuple(corpus) if corpus is not None else default_corpus()
    run_once = runner or run_api_once
    report = _AggregatedReport(corpus_signature=build_corpus_signature(active_corpus))

    for item in active_corpus:
        api_hashes: dict[str, str] = {}
        for api_name in api_names:
            baseline = dict(run_once(item, api_name))
            for iteration in range(1, iterations):
                current = dict(run_once(item, api_name))
                if current["hash"] != baseline["hash"]:
                    raise DivergenceError(
                        _format_divergence(
                            item_id=item.item_id,
                            api_name=api_name,
                            iteration=iteration,
                            baseline=baseline,
                            current=current,
                        )
                    )
            api_hashes[api_name] = baseline["hash"]
        report.items.append(
            ItemDeterminismReport(
                item_id=item.item_id,
                api_hashes=api_hashes,
                iterations=iterations,
            )
        )

    return report.to_golden_mapping()


def _format_divergence(
    *,
    item_id: str,
    api_name: str,
    iteration: int,
    baseline: Mapping[str, Any],
    current: Mapping[str, Any],
) -> str:
    """Return a PHI-free multi-line divergence diagnostic."""
    diffs = _diff_records(
        baseline.get("spans", []),
        current.get("spans", []),
    )
    lines = [
        "determinism divergence detected (offsets/labels/hashes only)",
        f"  item_id={item_id!r} api={api_name!r} iteration={iteration}",
        f"  hash_run_0={baseline.get('hash')}",
        f"  hash_run_{iteration}={current.get('hash')}",
    ]
    for diff in diffs:
        lines.append(
            f"  span[{diff['index']}] run_a={diff['run_a']} run_b={diff['run_b']}"
        )
    return "\n".join(lines)


def _main(argv: Sequence[str] | None = None) -> int:
    """Emit the PHI-free determinism golden mapping as JSON on stdout.

    Used by the cross-process determinism test: a fresh interpreter runs this
    entrypoint and the parent asserts the emitted hashes match the in-process
    run. ``PYTHONHASHSEED`` differences between processes therefore cannot hide
    dict/set ordering leaks.
    """

    import argparse
    import json

    parser = argparse.ArgumentParser(description="OpenMed determinism harness")
    parser.add_argument(
        "--iterations",
        type=int,
        default=2,
        help="in-process repeats per (item, api) before emitting hashes",
    )
    args = parser.parse_args(argv)
    mapping = run_determinism_check(iterations=args.iterations)
    print(json.dumps(mapping, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised via subprocess
    raise SystemExit(_main())
