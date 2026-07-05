"""Property-based contract tests for de-identification pipeline stage boundaries.

This suite pins the **structural contract of each stage boundary** in the ten-stage
:class:`openmed.core.pipeline.Pipeline` (``normalize → language_script →
doc_type_section → deterministic_detectors → fast_pii_model → clinical_phi_model →
span_arbitration → policy_actions → safety_sweep → emit``). Rather than checking a
single end-to-end example, each test uses `Hypothesis <https://hypothesis.readthedocs.io/>`_
to exercise a wide space of synthetic inputs and assert the invariants that must hold at
the seam between stages:

* **Span integrity** — every emitted span satisfies
  ``0 <= start <= end <= len(text)`` and ``text[start:end]`` is a valid code-point slice
  (offsets are code-point indices, never byte indices).
* **Non-overlap after merge** — :func:`openmed.core.arbitration.arbitrate` (balanced
  mode) and the final :attr:`PipelineResult.spans` are pairwise non-overlapping and
  sorted.
* **Canonical labels** — every emitted span's ``canonical_label`` is a member of
  :data:`openmed.core.labels.CANONICAL_LABELS`.
* **Type stability** — each stage's input/output types match its declared contract
  (:class:`NormalizedDocument`, :class:`LanguageRoute`, ``tuple[OpenMedSpan, ...]``,
  :class:`PipelineResult`, :class:`DeidentificationResult`).
* **Leakage-first** — structured span records (``OpenMedSpan.to_dict`` and the audit
  payloads) expose only offsets and ``hmac-sha256:`` hashes; no raw PHI surface text
  appears in structured outputs beyond the synthetic gold string the test itself supplied.

Relationship to the OM-083 fuzz harness
----------------------------------------
The OM-083 property-based *fuzz* harness drives the top-level ``deidentify`` entry point
with adversarial/random text to hunt for crashes and robustness failures across the path
as a whole. This suite is *complementary*: it asserts the **per-stage structural
contracts** at each boundary — the shape and invariants of the hand-off between adjacent
stages — rather than end-to-end robustness. The two together give both "it never crashes"
(fuzz) and "each seam keeps its promise" (contract) coverage without duplication.

Everything here runs fully offline: the model-backed stages
(``stage5_fast_pii_model`` / ``stage6_clinical_phi_model``) are driven by deterministic
in-process fakes, so no network access or model download is required. Runtime is bounded
via capped ``max_examples`` and an explicit Hypothesis deadline.
"""

from __future__ import annotations

from datetime import datetime
from typing import Sequence

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from openmed.core.arbitration import arbitrate
from openmed.core.decoding import coerce_token_classification_spans
from openmed.core.decoding.spans import TokenClassificationSpan
from openmed.core.labels import CANONICAL_LABELS, PERSON, normalize_label
from openmed.core.pii import DeidentificationResult
from openmed.core.pipeline import (
    LanguageRoute,
    NormalizedDocument,
    Pipeline,
    PipelineContext,
    PipelineResult,
    PipelineStageResult,
)
from openmed.core.schemas.span import ACTION_VALUES, OpenMedSpan, hmac_text_hash
from openmed.processing.outputs import EntityPrediction, PredictionResult

# Bounded runtime: keep the property sweep cheap enough for CI while still
# exploring a meaningful space. A shared profile keeps every test consistent.
CONTRACT_SETTINGS = settings(
    max_examples=40,
    deadline=1000,
    suppress_health_check=[HealthCheck.too_slow],
)

pytestmark = pytest.mark.contract

_HMAC_SECRET = "om-798-contract-secret"

# A canonical label always present so span construction never depends on frozenset
# iteration order. ``PERSON`` is a member of ``CANONICAL_LABELS`` by construction.
_SORTED_CANONICAL_LABELS = sorted(CANONICAL_LABELS)


# ---------------------------------------------------------------------------
# Hypothesis strategies (synthetic-only)
# ---------------------------------------------------------------------------

# Printable, de-identification-relevant text. Includes ASCII plus a few
# accented / non-Latin code points so offset invariants are exercised against
# multi-byte characters (offsets must remain code-point indices, not byte
# indices).
_TEXT_ALPHABET = st.characters(
    whitelist_categories=("Lu", "Ll", "Nd", "Pc", "Pd", "Po", "Zs"),
    min_codepoint=0x20,
    max_codepoint=0x2FF,
)

synthetic_text = st.text(alphabet=_TEXT_ALPHABET, min_size=1, max_size=120)


@st.composite
def _span_within(draw: st.DrawFn, text: str) -> tuple[int, int, str]:
    """Draw a (start, end, canonical_label) triple whose offsets lie in ``text``."""

    length = len(text)
    start = draw(st.integers(min_value=0, max_value=max(length - 1, 0)))
    end = draw(st.integers(min_value=start + 1, max_value=length))
    label = draw(st.sampled_from(_SORTED_CANONICAL_LABELS))
    return start, end, label


@st.composite
def _openmed_spans(draw: st.DrawFn) -> tuple[str, tuple[OpenMedSpan, ...]]:
    """Draw synthetic text plus a bag of well-formed ``OpenMedSpan`` records.

    The spans may overlap arbitrarily — that is exactly the pre-arbitration state
    the merge stage is contracted to resolve.
    """

    text = draw(st.text(alphabet=_TEXT_ALPHABET, min_size=4, max_size=80))
    count = draw(st.integers(min_value=0, max_value=6))
    spans: list[OpenMedSpan] = []
    for index in range(count):
        start, end, label = draw(_span_within(text))
        surface = text[start:end]
        score = draw(st.floats(min_value=0.0, max_value=1.0))
        spans.append(
            OpenMedSpan(
                doc_id="doc-contract",
                start=start,
                end=end,
                text_hash=hmac_text_hash(surface, _HMAC_SECRET),
                entity_type=label,
                canonical_label=label,
                score=score,
                detector=draw(
                    st.sampled_from(
                        ["model:fake", "rules:regex", "safety_sweep", "plugin:x"]
                    )
                ),
                metadata={"draw_index": index},
            )
        )
    return text, tuple(spans)


@st.composite
def _raw_predictions(draw: st.DrawFn) -> tuple[str, list[dict]]:
    """Draw synthetic text plus raw backend prediction dicts for the decode seam."""

    text = draw(st.text(alphabet=_TEXT_ALPHABET, min_size=4, max_size=80))
    count = draw(st.integers(min_value=0, max_value=6))
    bioes = draw(st.sampled_from(["", "B-", "I-", "E-", "S-"]))
    predictions: list[dict] = []
    for _ in range(count):
        start, end, label = draw(_span_within(text))
        predictions.append(
            {
                "start": start,
                "end": end,
                "entity_group": f"{bioes}{label}",
                "score": draw(st.floats(min_value=0.0, max_value=1.0)),
                "word": text[start:end],
            }
        )
    return text, predictions


# ---------------------------------------------------------------------------
# Shared invariant assertions
# ---------------------------------------------------------------------------


def _assert_span_integrity(spans: Sequence[OpenMedSpan], text: str) -> None:
    """Every span indexes a valid code-point slice with a canonical label."""

    for span in spans:
        assert isinstance(span, OpenMedSpan)
        assert isinstance(span.start, int) and isinstance(span.end, int)
        assert 0 <= span.start <= span.end <= len(text), (
            span.start,
            span.end,
            len(text),
        )
        # A valid code-point slice never raises and round-trips by length.
        surface = text[span.start : span.end]
        assert len(surface) == span.end - span.start
        assert span.canonical_label in CANONICAL_LABELS
        assert span.action in ACTION_VALUES
        if span.score is not None:
            assert 0.0 <= span.score <= 1.0


def _assert_sorted_non_overlapping(spans: Sequence[OpenMedSpan]) -> None:
    """Spans are sorted by (start, end) and pairwise non-overlapping."""

    previous_end: int | None = None
    last_key: tuple[int, int] | None = None
    for span in spans:
        key = (span.start, span.end)
        if last_key is not None:
            assert key >= last_key, ("not sorted", last_key, key)
        last_key = key
        if previous_end is not None:
            # Half-open [start, end) intervals: no overlap means start >= prev end.
            assert span.start >= previous_end, ("overlap", previous_end, span.start)
        previous_end = span.end


def _assert_no_raw_phi_in_structured_output(
    spans: Sequence[OpenMedSpan], text: str
) -> None:
    """Leakage-first: structured span payloads never carry raw surface text.

    ``OpenMedSpan`` deliberately has no ``text`` field — only offsets and an
    ``hmac-sha256:`` hash. This asserts the contract holds for the serialized
    payload and the PHI-safe audit dict of each span.
    """

    for span in spans:
        payload = span.to_dict()
        assert "text" not in payload
        assert payload["text_hash"].startswith("hmac-sha256:")
        surface = text[span.start : span.end]
        if len(surface) >= 3:
            # A non-trivial surface string must not be recoverable from the
            # structured record. Guard against accidental plaintext leakage.
            serialized = span.to_json()
            assert surface not in serialized


# ---------------------------------------------------------------------------
# Stage 1 — normalize: offset-map round-trip contract
# ---------------------------------------------------------------------------


@CONTRACT_SETTINGS
@given(text=synthetic_text)
def test_stage1_normalize_returns_document_with_valid_offset_map(text: str) -> None:
    """Stage 1 output type is stable and every offset maps back into bounds."""

    document = Pipeline().stage1_normalize(text)

    assert isinstance(document, NormalizedDocument)
    assert document.original_text == text
    assert isinstance(document.normalized_text, str)

    normalized = document.normalized_text
    offset_map = document.offset_map

    # Every normalized index maps back to a valid original code-point index.
    for index in range(len(normalized)):
        original_index = offset_map.normalized_index_to_original(index)
        assert 0 <= original_index < len(text), (index, original_index, len(text))

    # Every original index either drops out (None) or maps into the normalized
    # text bounds — never out of range.
    for index in range(len(text)):
        mapped = offset_map.original_index_to_normalized(index)
        if mapped is not None:
            assert 0 <= mapped <= len(normalized), (index, mapped, len(normalized))


@CONTRACT_SETTINGS
@given(text=synthetic_text)
def test_stage1_normalized_span_round_trip_stays_in_bounds(text: str) -> None:
    """A normalized span maps back to an in-bounds original span slice."""

    document = Pipeline().stage1_normalize(text)
    normalized = document.normalized_text
    if not normalized:
        return

    # Take the whole normalized text as one span and round-trip it.
    original_start, original_end = (
        document.offset_map.normalized_span_to_original_offsets(0, len(normalized))
    )
    assert 0 <= original_start <= original_end <= len(text)
    # The mapped original slice is itself a valid code-point slice.
    assert len(text[original_start:original_end]) == original_end - original_start


# ---------------------------------------------------------------------------
# Raw decode seam — coerce_token_classification_spans
# ---------------------------------------------------------------------------


@CONTRACT_SETTINGS
@given(payload=_raw_predictions())
def test_decode_seam_emits_sorted_in_bounds_bioes_stripped_spans(
    payload: tuple[str, list[dict]],
) -> None:
    """Raw backend predictions coerce to sorted, in-bounds, BIOES-stripped spans."""

    text, predictions = payload
    spans = coerce_token_classification_spans(
        predictions, text, confidence_threshold=0.0
    )

    assert isinstance(spans, list)

    previous_key: tuple[int, int, str, str] | None = None
    for span in spans:
        assert isinstance(span, TokenClassificationSpan)
        # In-bounds, non-empty, code-point-indexed offsets.
        assert 0 <= span.start < span.end <= len(text)
        assert len(text[span.start : span.end]) == span.end - span.start
        # BIOES prefixes are stripped by the decode seam.
        for prefix in ("B-", "I-", "E-", "S-"):
            assert not span.label.startswith(prefix)
        # Byte offsets, when present, are consistent with a byte encoding.
        if span.byte_start is not None and span.byte_end is not None:
            assert 0 <= span.byte_start <= span.byte_end
        key = (span.start, span.end, span.label, span.id)
        if previous_key is not None:
            assert key >= previous_key, ("decode output not sorted", previous_key, key)
        previous_key = key


# ---------------------------------------------------------------------------
# Stages 2-4 — deterministic, offline detector stages
# ---------------------------------------------------------------------------


def _offline_context(pipeline: Pipeline, text: str) -> tuple[str, PipelineContext]:
    """Run the deterministic stages 1-3 and build a PipelineContext (no model)."""

    document = pipeline.stage1_normalize(text)
    route = pipeline.stage2_language_script(document.normalized_text)
    assert isinstance(route, LanguageRoute)
    section_metadata = pipeline.stage3_doc_type_section(document.normalized_text)
    assert isinstance(section_metadata, dict)
    context = PipelineContext(
        doc_id="doc-contract",
        original_text=document.original_text,
        normalized_text=document.normalized_text,
        offset_map=document.offset_map,
        route=route,
        section_metadata=section_metadata,
    )
    return document.normalized_text, context


@CONTRACT_SETTINGS
@given(text=synthetic_text)
def test_stage4_deterministic_detectors_emit_well_formed_spans(text: str) -> None:
    """Stage 4 emits canonical, in-bounds, PHI-safe spans with no model loaded."""

    pipeline = Pipeline()
    normalized_text, context = _offline_context(pipeline, text)

    spans = pipeline.stage4_deterministic_detectors(normalized_text, context)

    assert isinstance(spans, tuple)
    _assert_span_integrity(spans, normalized_text)
    _assert_no_raw_phi_in_structured_output(spans, normalized_text)


@CONTRACT_SETTINGS
@given(text=synthetic_text)
def test_stage4_spans_survive_arbitration_as_non_overlapping(text: str) -> None:
    """Stage 4 → Stage 7 hand-off: arbitration de-overlaps deterministic spans."""

    pipeline = Pipeline()
    normalized_text, context = _offline_context(pipeline, text)

    detected = pipeline.stage4_deterministic_detectors(normalized_text, context)
    arbitrated = pipeline.stage7_arbitration(detected, context)

    assert isinstance(arbitrated, tuple)
    _assert_span_integrity(arbitrated, normalized_text)
    _assert_sorted_non_overlapping(arbitrated)


# ---------------------------------------------------------------------------
# Merge stage — arbitrate() overlap-resolution contract
# ---------------------------------------------------------------------------


@CONTRACT_SETTINGS
@given(payload=_openmed_spans())
def test_arbitrate_balanced_is_sorted_and_non_overlapping(
    payload: tuple[str, tuple[OpenMedSpan, ...]],
) -> None:
    """Balanced arbitration returns sorted, pairwise non-overlapping spans."""

    text, spans = payload
    resolved = arbitrate(spans)

    assert isinstance(resolved, tuple)
    _assert_span_integrity(resolved, text)
    _assert_sorted_non_overlapping(resolved)


@CONTRACT_SETTINGS
@given(payload=_openmed_spans())
def test_arbitrate_only_returns_input_spans(
    payload: tuple[str, tuple[OpenMedSpan, ...]],
) -> None:
    """Arbitration never fabricates a span; every winner came from the input set."""

    text, spans = payload
    resolved = arbitrate(spans)

    input_keys = {
        (s.doc_id, s.start, s.end, s.canonical_label, s.detector) for s in spans
    }
    for span in resolved:
        key = (span.doc_id, span.start, span.end, span.canonical_label, span.detector)
        assert key in input_keys, ("fabricated span", key)


@CONTRACT_SETTINGS
@given(payload=_openmed_spans())
def test_arbitrate_preserves_leakage_free_structured_output(
    payload: tuple[str, tuple[OpenMedSpan, ...]],
) -> None:
    """Merge stage never reintroduces raw PHI text into structured span records."""

    text, spans = payload
    resolved = arbitrate(spans)
    _assert_no_raw_phi_in_structured_output(resolved, text)


# ---------------------------------------------------------------------------
# Whole-pipeline contract — Pipeline.run() with an offline model detector
# ---------------------------------------------------------------------------


def _fake_model_detector_factory(gold_surface: str, gold_label: str):
    """Build an offline model detector that emits one span for ``gold_surface``.

    The detector only fires when the (already de-identified/normalized) text
    literally contains the synthetic gold surface, so downstream stages get a
    deterministic, offline entity to arbitrate and redact.
    """

    def model_detector(text: str, **kwargs) -> PredictionResult:
        entities: list[EntityPrediction] = []
        start = text.find(gold_surface)
        if start != -1:
            entities.append(
                EntityPrediction(
                    text=gold_surface,
                    label=gold_label,
                    start=start,
                    end=start + len(gold_surface),
                    confidence=0.97,
                )
            )
        return PredictionResult(
            text=text,
            entities=entities,
            model_name=kwargs.get("model_name", "fake-model"),
            timestamp=datetime.now().isoformat(),
        )

    return model_detector


# Synthetic gold surfaces: clearly-not-real names embedded in carrier sentences.
_GOLD_SURFACES = ["Jane Roe", "John Q Public", "Alex Sample", "Pat Doe"]
_CARRIERS = [
    "Patient {name} was seen in clinic today.",
    "Referral note for {name} regarding follow-up.",
    "Discharge summary: {name} is stable.",
]


@CONTRACT_SETTINGS
@given(
    gold_surface=st.sampled_from(_GOLD_SURFACES),
    carrier=st.sampled_from(_CARRIERS),
)
def test_full_pipeline_result_types_and_span_contracts(
    gold_surface: str, carrier: str
) -> None:
    """End-to-end run keeps stage types stable and span contracts intact."""

    text = carrier.format(name=gold_surface)
    detector = _fake_model_detector_factory(gold_surface, "NAME")

    result = Pipeline(model_detector=detector, use_safety_sweep=False).run(
        text, method="mask"
    )

    # Type-stability contract at the emit boundary.
    assert isinstance(result, PipelineResult)
    assert isinstance(result.spans, tuple)
    assert isinstance(result.redacted_text, str)
    assert isinstance(result.deidentification_result, DeidentificationResult)
    for stage_result in result.stage_results:
        assert isinstance(stage_result, PipelineStageResult)
        assert isinstance(stage_result.spans, tuple)

    # Span integrity + non-overlap on the final, arbitrated span set.
    _assert_span_integrity(result.spans, text)
    _assert_sorted_non_overlapping(result.spans)
    _assert_no_raw_phi_in_structured_output(result.spans, text)


@CONTRACT_SETTINGS
@given(
    gold_surface=st.sampled_from(_GOLD_SURFACES),
    carrier=st.sampled_from(_CARRIERS),
)
def test_full_pipeline_redacts_every_acted_span_from_output_text(
    gold_surface: str, carrier: str
) -> None:
    """Emit contract: an acted span's surface never survives in redacted text."""

    text = carrier.format(name=gold_surface)
    detector = _fake_model_detector_factory(gold_surface, "NAME")

    result = Pipeline(model_detector=detector, use_safety_sweep=False).run(
        text, method="mask"
    )

    # The synthetic name is detected and redacted out of the emitted text.
    assert gold_surface not in result.redacted_text
    assert gold_surface not in result.deidentification_result.deidentified_text

    # Detected entities carry canonical labels; NAME normalizes to PERSON.
    assert any(span.canonical_label == PERSON for span in result.spans)
    for entity in result.deidentification_result.pii_entities:
        assert normalize_label(entity.label) in CANONICAL_LABELS


@CONTRACT_SETTINGS
@given(
    gold_surface=st.sampled_from(_GOLD_SURFACES),
    carrier=st.sampled_from(_CARRIERS),
)
def test_full_pipeline_stage_boundaries_are_ordered_and_typed(
    gold_surface: str, carrier: str
) -> None:
    """Every declared stage boundary is present, ordered, and carries typed spans."""

    text = carrier.format(name=gold_surface)
    detector = _fake_model_detector_factory(gold_surface, "NAME")

    result = Pipeline(model_detector=detector, use_safety_sweep=False).run(
        text, method="mask"
    )

    emitted_names = [sr.name for sr in result.stage_results]
    # Stage order is a subsequence of the declared STAGE_NAMES contract.
    declared = list(Pipeline.stage_names)
    positions = [declared.index(name) for name in emitted_names]
    assert positions == sorted(positions), ("stages out of order", emitted_names)

    # Every span the pipeline threaded through a stage boundary is well-formed.
    for stage_result in result.stage_results:
        _assert_span_integrity(stage_result.spans, result.normalized_text)


if __name__ == "__main__":  # pragma: no cover - manual invocation helper
    raise SystemExit(pytest.main([__file__, "-q"]))
