"""Property-based fuzz harness for ``deidentify`` / ``extract_pii`` / ``reidentify``.

The de-identification path handles untrusted, arbitrary text: mixed scripts,
zero-width joiners, astral (multi-code-unit) characters, and very long inputs.
This module drives the *real* pipeline (input normalization, span remapping,
smart merging, overlap resolution, right-to-left redaction) with
Hypothesis-generated text and synthetic detector output, asserting the
invariants that must hold for any input:

* **No crash** — ``deidentify`` / ``extract_pii`` never raise on valid ``str``.
* **Span integrity** — every returned span is in ``[0, len(result_text)]`` with
  ``start < end``, references a real code-point slice, and spans do not overlap.
* **Code-point offsets** — offsets index Python ``str`` code points, so
  ``result_text[start:end]`` is always well-defined even across astral chars.
* **No raw-identifier leakage** — for planted synthetic identifiers that the
  (mocked) detector reports, the redacted output never contains the raw
  identifier substring verbatim. This is the leakage-first invariant.
* **Idempotence** — masking already-masked output is a fixed point.
* **Determinism** — with a fixed seed and identical input, ``deidentify`` yields
  byte-identical output across runs.
* **Re-identification inverse** — with a kept mapping, ``reidentify`` restores
  the canonical original document exactly.
* **Long and streaming inputs** — multi-kilobyte documents and identifiers split
  across source chunks use the production offset-remap and carry-buffer paths.

All identifiers are synthetic, generated locally. No real PHI is ever used. The
model call (``openmed.analyze_text``) is mocked so the harness is offline,
deterministic, and fast — the code under test is the pure-Python privacy logic.

Note on normalization: ``deidentify`` operates on ``text.strip()`` after folding
adversarial unicode (zero-width chars, combining marks, confusables). The
detector stub therefore locates sentinels by substring search in whatever
inference text the pipeline passes it, and invariants are asserted against the
result's own ``original_text`` / ``text`` (the canonical text spans index into),
not the raw generator input.
"""

from __future__ import annotations

import hashlib
from typing import Sequence
from unittest.mock import patch

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from openmed.core.pii import _shift_date, deidentify, extract_pii, reidentify
from openmed.core.pipeline import Pipeline
from openmed.core.streaming import StreamingDeidentifier
from openmed.processing.outputs import EntityPrediction, PredictionResult

# Ensure the bounded/nightly Hypothesis profile is registered even if this module
# is collected before the package conftest side effects run.
from . import conftest as _fuzz_conftest  # noqa: F401  (import for side effects)

pytestmark = pytest.mark.fuzz

_MODEL = "fixture-pii-model"

# Unique, easy-to-spot sentinel identifiers. Each is a *synthetic* value that a
# leak check can search for unambiguously: if any of these survive verbatim in
# redacted output the leakage invariant has been violated. They are pure ASCII
# so unicode folding never alters them, and they never collide with a
# ``[LABEL]`` mask placeholder.
_PLANTED_IDENTIFIERS: tuple[tuple[str, str], ...] = (
    ("NAME", "Zzyxx Qwerty"),
    ("EMAIL", "zzq-sentinel@fuzz.invalid"),
    ("PHONE", "5550100999"),
    ("ID", "MRN-ZZ-90210-777"),
    ("SSN", "900-55-4242"),
)

# Filler that exercises the offset-remap and adversarial-unicode handling. These
# are interleaved with planted identifiers so the real spans land at non-trivial
# code-point offsets and the folding/remap logic is stressed. None of these
# fragments contains an ASCII sentinel substring.
_FILLER_ALPHABET = st.sampled_from(
    [
        "Patient",
        "seen",
        "record",
        "\n",
        " ",
        "\U0001f9ec",  # DNA emoji (astral: 2 UTF-16 units, 1 code point)
        "é",  # e + combining acute accent (combining mark folded in inference)
        "‍",  # zero-width joiner
        "​",  # zero-width space
        "﻿",  # BOM / zero-width no-break space
        "а",  # Cyrillic small a (confusable, folded in inference)
        "北京",  # CJK
        "مستشفى",  # RTL Arabic
        ".",
    ]
)

_LONG_FILLER = "Patient record 🧬 é ‍ 北京 مستشفى. "
_STREAM_MAX_BUFFER = 96
_DATE_IDENTIFIER = ("DATE", "2020-02-29")


def _safe_text_ref(value: str) -> str:
    """Return a diagnostic fingerprint without reproducing the source text."""
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]
    return f"length={len(value)} sha256={digest}"


def _safe_identifier_ref(label: str, value: str) -> str:
    """Describe a planted identifier without exposing its raw value."""
    return f"label={label} {_safe_text_ref(value)}"


class PlantedDocument:
    """Synthetic fuzz input whose failure representation never includes raw text."""

    __slots__ = ("planted", "text")

    def __init__(self, text: str, planted: tuple[tuple[str, str], ...]) -> None:
        self.text = text
        self.planted = planted

    def __repr__(self) -> str:
        identifiers = ", ".join(
            _safe_identifier_ref(label, value) for label, value in self.planted
        )
        return f"PlantedDocument({_safe_text_ref(self.text)}, planted=[{identifiers}])"


class ChunkBoundaryDocument:
    """Streaming fuzz input with a PHI-safe Hypothesis failure representation."""

    __slots__ = ("chunks", "planted", "text")

    def __init__(
        self,
        text: str,
        chunks: tuple[str, str],
        planted: tuple[str, str],
    ) -> None:
        self.text = text
        self.chunks = chunks
        self.planted = planted

    def __repr__(self) -> str:
        label, value = self.planted
        return (
            f"ChunkBoundaryDocument({_safe_text_ref(self.text)}, "
            f"chunk_lengths=({len(self.chunks[0])}, {len(self.chunks[1])}), "
            f"planted={_safe_identifier_ref(label, value)})"
        )


@st.composite
def planted_documents(draw):
    """Generate a synthetic document plus the sentinels planted into it.

    Returns a redacted-representation ``PlantedDocument`` whose ``planted``
    field contains ``(label, value)`` sentinel identifiers embedded in its text.
    The text has
    no leading/trailing whitespace so ``text.strip() == text`` (keeps the
    canonical text stable for exact-match assertions), and the surrounding
    filler is adversarial unicode that the pipeline folds/remaps.
    """
    n_ids = draw(st.integers(min_value=0, max_value=len(_PLANTED_IDENTIFIERS)))
    chosen = draw(
        st.lists(
            st.sampled_from(_PLANTED_IDENTIFIERS),
            min_size=n_ids,
            max_size=n_ids,
            unique_by=lambda pair: pair[1],
        )
    )

    parts: list[str] = ["Patient"]  # non-whitespace anchor
    for label, value in chosen:
        gap = "".join(draw(st.lists(_FILLER_ALPHABET, min_size=1, max_size=3)))
        parts.append(gap)
        parts.append(value)
    parts.append("end")  # non-whitespace trailing anchor

    text = "".join(parts)
    return PlantedDocument(text=text, planted=tuple(chosen))


@st.composite
def long_planted_documents(draw):
    """Generate a 4-KiB-plus synthetic document with one planted identifier."""
    planted = draw(st.sampled_from(_PLANTED_IDENTIFIERS))
    repeats = draw(st.integers(min_value=128, max_value=256))
    insertion = draw(st.integers(min_value=1, max_value=repeats - 1))
    parts = [_LONG_FILLER] * repeats
    parts.insert(insertion, f"{planted[1]} ")
    text = "".join(parts) + "end"
    return PlantedDocument(text=text, planted=(planted,))


@st.composite
def repeated_identifier_documents(draw):
    """Generate two occurrences of one identifier for consistency properties."""
    planted = draw(st.sampled_from(_PLANTED_IDENTIFIERS))
    gap = draw(st.sampled_from((" and ", "\n", " 🧬 é ‍ 北京 ")))
    text = f"Patient {planted[1]}{gap}{planted[1]} end"
    return PlantedDocument(text=text, planted=(planted,))


@st.composite
def chunk_boundary_documents(draw):
    """Generate chunks whose boundary falls strictly inside an identifier."""
    planted = draw(st.sampled_from(_PLANTED_IDENTIFIERS))
    split = draw(st.integers(min_value=1, max_value=len(planted[1]) - 1))
    prefix_repeats = draw(st.integers(min_value=8, max_value=16))
    suffix_repeats = draw(st.integers(min_value=8, max_value=16))
    prefix = "Synthetic clinical note without identifiers. " * prefix_repeats
    prefix += "Contact "
    suffix = " completed. " + "No identifier in this sentence. " * suffix_repeats
    suffix += "end"
    text = f"{prefix}{planted[1]}{suffix}"
    chunks = (f"{prefix}{planted[1][:split]}", f"{planted[1][split:]}{suffix}")
    return ChunkBoundaryDocument(text=text, chunks=chunks, planted=planted)


def _find_spans(query_text: str, planted: Sequence[tuple[str, str]]):
    """Locate each planted sentinel in ``query_text`` and return spans.

    Substring search makes the stub robust to the pipeline's internal
    normalization (strip/fold): wherever the sentinel actually lands in the
    inference text, we report a correctly-offset span for every occurrence.
    Sentinels absent from ``query_text`` (e.g. placeholder re-scans) yield
    nothing.
    """
    entities: list[EntityPrediction] = []
    for label, value in planted:
        cursor = 0
        while True:
            start = query_text.find(value, cursor)
            if start == -1:
                break
            end = start + len(value)
            entities.append(
                EntityPrediction(
                    text=value,
                    label=label,
                    confidence=0.99,
                    start=start,
                    end=end,
                )
            )
            cursor = end
    # Detector output must be non-overlapping and sorted; sentinel values are
    # non-substrings of one another, so sorting by start suffices.
    entities.sort(key=lambda e: e.start)
    return entities


def _make_analyze_stub(planted: Sequence[tuple[str, str]]):
    """Return a stub that mimics ``openmed.analyze_text`` by locating sentinels."""

    def _stub(query_text, *args, **kwargs):
        return PredictionResult(
            text=query_text,
            entities=_find_spans(query_text, planted),
            model_name=_MODEL,
            timestamp="2026-01-01T00:00:00",
        )

    return _stub


def _assert_span_invariants(entities, text: str) -> None:
    """Assert bounds, ordering, non-overlap, and code-point validity."""
    n = len(text)
    prev_end = -1
    for ent in entities:
        start, end = ent.start, ent.end
        if start is None or end is None:
            raise AssertionError("retained entity is missing character offsets")
        if not 0 <= start < end <= n:
            raise AssertionError(f"span [{start}:{end}] out of bounds for n={n}")
        # ``text[start:end]`` is well-defined for valid code-point indices; this
        # also guards against surrogate/astral index confusion.
        if text[start:end] != ent.text:
            raise AssertionError(
                "span text mismatch at "
                f"[{start}:{end}]: source {_safe_text_ref(text[start:end])}; "
                f"entity {_safe_text_ref(ent.text)}"
            )
        if start < prev_end:
            raise AssertionError(
                f"overlapping spans: prev_end={prev_end} start={start}"
            )
        prev_end = end


def _assert_planted_spans_covered(
    entities, text: str, planted: Sequence[tuple[str, str]]
) -> None:
    """Assert every occurrence of every planted value is covered by a span."""
    for label, value in planted:
        cursor = 0
        occurrences = 0
        while True:
            start = text.find(value, cursor)
            if start == -1:
                break
            occurrences += 1
            end = start + len(value)
            if not any(
                entity.start is not None
                and entity.end is not None
                and entity.start <= start
                and entity.end >= end
                for entity in entities
            ):
                raise AssertionError(
                    f"{_safe_identifier_ref(label, value)} at [{start}:{end}] "
                    "was not covered by a retained span"
                )
            cursor = end
        if occurrences == 0:
            raise AssertionError(
                f"{_safe_identifier_ref(label, value)} was absent from source text"
            )


def _assert_identifier_absent(
    output: str,
    label: str,
    value: str,
    *,
    context: str,
) -> None:
    """Fail with a fingerprint rather than echoing a leaked identifier."""
    if value in output:
        raise AssertionError(
            f"{_safe_identifier_ref(label, value)} survived in {context}"
        )


def _assert_text_equal(actual: str, expected: str, *, context: str) -> None:
    """Compare sensitive-shaped text while keeping diagnostics content-free."""
    if actual != expected:
        raise AssertionError(
            f"{context}: actual {_safe_text_ref(actual)}; "
            f"expected {_safe_text_ref(expected)}"
        )


def test_planted_case_failure_representations_do_not_expose_values() -> None:
    """Hypothesis counterexamples for planted cases contain hashes, never values."""
    case = PlantedDocument("prefix Zzyxx Qwerty suffix", (_PLANTED_IDENTIFIERS[0],))
    rendered = repr(case)

    assert "sha256=" in rendered
    for _, value in case.planted:
        assert value not in rendered


@pytest.mark.parametrize(
    ("label", "value"),
    _PLANTED_IDENTIFIERS,
    ids=[label.lower() for label, _ in _PLANTED_IDENTIFIERS],
)
def test_every_sentinel_is_deterministically_covered_and_removed(label, value):
    """The bounded suite exercises every sentinel, independent of random draws."""
    text = f"Patient {value} end"
    planted = ((label, value),)
    with patch("openmed.analyze_text", _make_analyze_stub(planted)):
        result = deidentify(
            text,
            method="mask",
            model_name=_MODEL,
            confidence_threshold=0.5,
            use_safety_sweep=False,
        )

    _assert_span_invariants(result.pii_entities, result.original_text)
    _assert_planted_spans_covered(result.pii_entities, result.original_text, planted)
    _assert_identifier_absent(
        result.deidentified_text,
        label,
        value,
        context="masked output",
    )


def test_replace_heterogeneous_semantic_union_never_reuses_raw_components():
    """A greedy typed surrogate cannot preserve another label's raw value."""
    planted = (_PLANTED_IDENTIFIERS[1], _PLANTED_IDENTIFIERS[0])
    text = "PatientPatientzzq-sentinel@fuzz.invalidPatientZzyxx Qwertyend"
    with patch("openmed.analyze_text", _make_analyze_stub(planted)):
        result = deidentify(
            text,
            method="replace",
            model_name=_MODEL,
            confidence_threshold=0.5,
            consistent=True,
            seed=1234,
            use_safety_sweep=False,
        )

    assert len(result.pii_entities) == 1
    semantic_merge = result.pii_entities[0].metadata.get("semantic_merge", {})
    assert semantic_merge.get("mixed_label_union") is True
    for label, value in planted:
        _assert_identifier_absent(
            result.deidentified_text,
            label,
            value,
            context="heterogeneous replacement output",
        )


@settings(deadline=1000)
@given(doc=planted_documents())
def test_deidentify_mask_invariants(doc):
    """Masking arbitrary planted documents preserves every structural invariant
    and never leaks a planted identifier verbatim."""
    text, planted = doc.text, doc.planted
    with patch("openmed.analyze_text", _make_analyze_stub(planted)):
        result = deidentify(
            text,
            method="mask",
            model_name=_MODEL,
            confidence_threshold=0.5,
            use_safety_sweep=False,  # detected set == planted set for the leak check
        )

    # Never crashes; documented shape; canonical text is the stripped input.
    _assert_text_equal(result.original_text, text, context="canonical source mismatch")
    assert isinstance(result.deidentified_text, str)

    _assert_span_invariants(result.pii_entities, result.original_text)
    _assert_planted_spans_covered(
        result.pii_entities,
        result.original_text,
        planted,
    )

    # Leakage-first invariant: no planted identifier survives verbatim.
    for label, value in planted:
        _assert_identifier_absent(
            result.deidentified_text,
            label,
            value,
            context="masked output",
        )


@given(doc=planted_documents())
def test_extract_pii_span_invariants(doc):
    """``extract_pii`` returns only in-bounds, non-overlapping, code-point spans
    that slice back to their recorded text."""
    text, planted = doc.text, doc.planted
    with patch("openmed.analyze_text", _make_analyze_stub(planted)):
        result = extract_pii(
            text,
            model_name=_MODEL,
            confidence_threshold=0.5,
        )
    _assert_span_invariants(result.entities, result.text)
    _assert_planted_spans_covered(result.entities, result.text, planted)


@given(doc=planted_documents())
def test_deidentify_mask_idempotent(doc):
    """Model-detection masking is a fixed point on already-masked output.

    A second de-identification pass over placeholder text (sentinels gone, so
    the model detector finds nothing) must leave the text unchanged. The
    deterministic regex layers (smart merging, safety sweep) are disabled here
    so the invariant isolates *model-detection* idempotence: those layers can
    legitimately act on placeholder-adjacent text (e.g. ``Patient.Patient``
    matching a URL pattern), which is a property of the detector configuration,
    not a leak. The leakage invariant is covered by
    ``test_deidentify_mask_invariants``; here we assert the second pass neither
    churns the masked text nor reintroduces any planted identifier.
    """
    text, planted = doc.text, doc.planted
    stub = _make_analyze_stub(planted)
    with patch("openmed.analyze_text", stub):
        first = deidentify(
            text,
            method="mask",
            model_name=_MODEL,
            confidence_threshold=0.5,
            use_safety_sweep=False,
            use_smart_merging=False,
        )
    masked = first.deidentified_text
    with patch("openmed.analyze_text", stub):
        second = deidentify(
            masked,
            method="mask",
            model_name=_MODEL,
            confidence_threshold=0.5,
            use_safety_sweep=False,
            use_smart_merging=False,
        )
    _assert_text_equal(
        second.deidentified_text,
        masked,
        context="second mask pass changed output",
    )
    for label, value in planted:
        _assert_identifier_absent(
            second.deidentified_text,
            label,
            value,
            context="second-pass masked output",
        )


@given(doc=planted_documents())
def test_deidentify_deterministic_fixed_seed(doc):
    """A fixed seed yields byte-identical redacted output across runs."""
    text, planted = doc.text, doc.planted
    assume(planted)
    stub = _make_analyze_stub(planted)
    outputs = []
    for _ in range(2):
        with patch("openmed.analyze_text", stub):
            result = deidentify(
                text,
                method="replace",  # surrogate generation must be reproducible
                model_name=_MODEL,
                confidence_threshold=0.5,
                consistent=True,
                seed=1234,
                use_safety_sweep=False,
            )
        outputs.append(result.deidentified_text)
    _assert_text_equal(outputs[0], outputs[1], context="seeded replacement drifted")
    for label, value in planted:
        _assert_identifier_absent(
            outputs[0],
            label,
            value,
            context="seeded replacement output",
        )


@given(doc=planted_documents())
def test_deidentify_keep_mapping_reidentify_inverse(doc):
    """With ``keep_mapping``, ``reidentify`` restores every planted identifier.

    ``reidentify`` must be a left inverse of the redaction mapping: applying it
    to the masked text brings back the original identifiers exactly.
    """
    text, planted = doc.text, doc.planted
    assume(planted)  # round-trip is only meaningful when something was redacted
    with patch("openmed.analyze_text", _make_analyze_stub(planted)):
        result = deidentify(
            text,
            method="mask",
            model_name=_MODEL,
            confidence_threshold=0.5,
            keep_mapping=True,
            use_safety_sweep=False,
        )
    assert result.mapping is not None
    _assert_planted_spans_covered(
        result.pii_entities,
        result.original_text,
        planted,
    )
    mapped_values = tuple(result.mapping.values())
    for label, value in planted:
        _assert_identifier_absent(
            result.deidentified_text,
            label,
            value,
            context="reversibly masked output",
        )
        if not any(value in mapped_value for mapped_value in mapped_values):
            raise AssertionError(
                f"{_safe_identifier_ref(label, value)} was not covered by "
                "the reversible mapping"
            )
    restored = reidentify(result.deidentified_text, result.mapping)
    _assert_text_equal(restored, result.original_text, context="reidentify mismatch")
    _assert_text_equal(result.original_text, text, context="canonical source mismatch")


@given(
    doc=repeated_identifier_documents(),
    seed=st.integers(min_value=0, max_value=2**32 - 1),
)
def test_repeated_identifier_surrogate_and_mapping_consistency(doc, seed):
    """Repeated values reuse one surrogate and remain exactly reversible."""
    text, planted = doc.text, doc.planted
    label, value = planted[0]
    stub = _make_analyze_stub(planted)

    with patch("openmed.analyze_text", stub):
        replaced = deidentify(
            text,
            method="replace",
            model_name=_MODEL,
            confidence_threshold=0.5,
            consistent=True,
            seed=seed,
            use_smart_merging=False,
            use_safety_sweep=False,
        )

    matching = [
        entity for entity in replaced.pii_entities if entity.original_text == value
    ]
    if len(matching) != 2:
        raise AssertionError(
            f"{_safe_identifier_ref(label, value)} retained {len(matching)} of 2 spans"
        )
    surrogates = [entity.redacted_text for entity in matching]
    if any(not surrogate for surrogate in surrogates):
        raise AssertionError(
            f"{_safe_identifier_ref(label, value)} produced an empty surrogate"
        )
    if surrogates[0] != surrogates[1]:
        raise AssertionError(
            f"{_safe_identifier_ref(label, value)} produced inconsistent surrogates"
        )
    _assert_identifier_absent(
        replaced.deidentified_text,
        label,
        value,
        context="replacement output",
    )

    with patch("openmed.analyze_text", stub):
        masked = deidentify(
            text,
            method="mask",
            model_name=_MODEL,
            confidence_threshold=0.5,
            keep_mapping=True,
            use_smart_merging=False,
            use_safety_sweep=False,
        )
    if masked.mapping is None:
        raise AssertionError("reversible mask did not return a mapping")
    _assert_planted_spans_covered(masked.pii_entities, masked.original_text, planted)
    mapped_occurrences = sum(
        value in mapped_value for mapped_value in masked.mapping.values()
    )
    if mapped_occurrences != 2:
        raise AssertionError(
            f"{_safe_identifier_ref(label, value)} mapped {mapped_occurrences} of 2 "
            "occurrences"
        )
    restored = reidentify(masked.deidentified_text, masked.mapping)
    _assert_text_equal(restored, text, context="repeated-value reidentify mismatch")


@given(
    shift=st.one_of(
        st.integers(min_value=-3650, max_value=-1),
        st.integers(min_value=1, max_value=3650),
    ),
    keep_year=st.booleans(),
)
def test_repeated_dates_share_one_pipeline_shift(shift, keep_year):
    """The full pipeline applies one exact shift to every repeated date value."""
    label, value = _DATE_IDENTIFIER
    planted = (_DATE_IDENTIFIER,)
    text = f"Dates {value} and {value} end"
    with patch("openmed.analyze_text", _make_analyze_stub(planted)):
        result = deidentify(
            text,
            method="shift_dates",
            model_name=_MODEL,
            confidence_threshold=0.5,
            date_shift_days=shift,
            keep_year=keep_year,
            use_smart_merging=False,
            use_safety_sweep=False,
        )

    matching = [
        entity for entity in result.pii_entities if entity.original_text == value
    ]
    if len(matching) != 2:
        raise AssertionError(
            f"{_safe_identifier_ref(label, value)} retained {len(matching)} of 2 spans"
        )
    redactions = [entity.redacted_text or "" for entity in matching]
    _assert_text_equal(
        redactions[0],
        redactions[1],
        context="repeated date received inconsistent shifts",
    )
    expected = _shift_date(
        value,
        shift,
        keep_year=keep_year,
        lang="en",
        require_dateutil=True,
    )
    _assert_text_equal(
        redactions[0],
        expected,
        context="pipeline date shift diverged from helper",
    )


@settings(deadline=1000)
@given(doc=long_planted_documents())
def test_deidentify_long_input_preserves_offsets_and_never_leaks(doc):
    """The real one-shot path handles multi-kilobyte normalized input exactly."""
    text, planted = doc.text, doc.planted
    assert len(text) >= 4096
    with patch("openmed.analyze_text", _make_analyze_stub(planted)):
        result = deidentify(
            text,
            method="mask",
            model_name=_MODEL,
            confidence_threshold=0.5,
            use_safety_sweep=False,
        )

    _assert_text_equal(result.original_text, text, context="long source mismatch")
    _assert_span_invariants(result.pii_entities, result.original_text)
    _assert_planted_spans_covered(
        result.pii_entities,
        result.original_text,
        planted,
    )
    for label, value in planted:
        _assert_identifier_absent(
            result.deidentified_text,
            label,
            value,
            context="long masked output",
        )


@given(case=chunk_boundary_documents())
def test_streaming_chunk_boundary_matches_single_pass(case):
    """A boundary inside an identifier is buffered and redacted as one span."""
    text, chunks, planted = case.text, case.chunks, case.planted
    detector = _make_analyze_stub([planted])
    single = Pipeline(
        model_detector=detector,
        confidence_threshold=0.5,
        use_smart_merging=False,
        use_safety_sweep=False,
    ).run(text, method="mask")
    streamer = StreamingDeidentifier(
        pipeline=Pipeline(
            model_detector=detector,
            confidence_threshold=0.5,
            use_smart_merging=False,
            use_safety_sweep=False,
        ),
        method="mask",
        max_buffer=_STREAM_MAX_BUFFER,
    )

    events = []
    for chunk in chunks:
        events.extend(streamer.feed(chunk))
    events.extend(streamer.flush())

    redacted = "".join(event.redacted_text for event in events)
    assert redacted == single.redacted_text
    _assert_identifier_absent(
        redacted,
        planted[0],
        planted[1],
        context="streaming masked output",
    )
    assert streamer.max_observed_buffer <= _STREAM_MAX_BUFFER
    identifier_start = text.index(planted[1])
    assert (identifier_start, identifier_start + len(planted[1])) in {
        (span.start, span.end) for span in streamer.spans
    }


@given(text=st.text(min_size=0, max_size=400))
def test_deidentify_never_crashes_on_arbitrary_text(text):
    """The de-identification path never raises on any valid ``str``.

    The detector reports nothing, so this fuzzes input handling, preprocessing,
    and the empty-entity code path against arbitrary unicode (astral chars,
    control chars, BOMs, combining marks). Invariants are asserted against the
    result's own canonical text, not the raw input.
    """
    stub = _make_analyze_stub([])
    with patch("openmed.analyze_text", stub):
        result = deidentify(
            text,
            method="mask",
            model_name=_MODEL,
            confidence_threshold=0.5,
        )
    assert isinstance(result.original_text, str)
    assert isinstance(result.deidentified_text, str)
    _assert_span_invariants(result.pii_entities, result.original_text)
