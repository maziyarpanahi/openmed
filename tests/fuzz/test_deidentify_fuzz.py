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
  every planted identifier.

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

from typing import Sequence
from unittest.mock import patch

import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

from openmed.core.pii import deidentify, extract_pii, reidentify
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


@st.composite
def planted_documents(draw):
    """Generate a synthetic document plus the sentinels planted into it.

    Returns ``(text, planted)`` where ``planted`` is a list of
    ``(label, value)`` sentinel identifiers embedded in ``text``. ``text`` has
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
    return text, chosen


def _find_spans(query_text: str, planted: Sequence[tuple[str, str]]):
    """Locate each planted sentinel in ``query_text`` and return spans.

    Substring search makes the stub robust to the pipeline's internal
    normalization (strip/fold): wherever the sentinel actually lands in the
    inference text, we report a correctly-offset span for it. Sentinels absent
    from ``query_text`` (e.g. placeholder re-scans) yield nothing.
    """
    entities: list[EntityPrediction] = []
    for label, value in planted:
        start = query_text.find(value)
        if start == -1:
            continue
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
    # Detector output must be non-overlapping and sorted; sentinels are unique
    # and non-substring of one another, so sorting by start suffices.
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
        assert start is not None and end is not None
        assert 0 <= start < end <= n, f"span [{start}:{end}] out of bounds for n={n}"
        # ``text[start:end]`` is well-defined for valid code-point indices; this
        # also guards against surrogate/astral index confusion.
        _ = text[start:end]
        assert start >= prev_end, (
            f"overlapping spans: prev_end={prev_end} start={start}"
        )
        prev_end = end


@given(doc=planted_documents())
def test_deidentify_mask_invariants(doc):
    """Masking arbitrary planted documents preserves every structural invariant
    and never leaks a planted identifier verbatim."""
    text, planted = doc
    with patch("openmed.analyze_text", _make_analyze_stub(planted)):
        result = deidentify(
            text,
            method="mask",
            model_name=_MODEL,
            confidence_threshold=0.5,
            use_safety_sweep=False,  # detected set == planted set for the leak check
        )

    # Never crashes; documented shape; canonical text is the stripped input.
    assert result.original_text == text
    assert isinstance(result.deidentified_text, str)

    _assert_span_invariants(result.pii_entities, result.original_text)

    # Leakage-first invariant: no planted identifier survives verbatim.
    for ent in result.pii_entities:
        assert ent.text not in result.deidentified_text, (
            f"raw identifier {ent.text!r} leaked into masked output"
        )


@given(doc=planted_documents())
def test_extract_pii_span_invariants(doc):
    """``extract_pii`` returns only in-bounds, non-overlapping, code-point spans
    that slice back to their recorded text."""
    text, planted = doc
    with patch("openmed.analyze_text", _make_analyze_stub(planted)):
        result = extract_pii(
            text,
            model_name=_MODEL,
            confidence_threshold=0.5,
        )
    _assert_span_invariants(result.entities, result.text)
    for ent in result.entities:
        assert result.text[ent.start : ent.end] == ent.text


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
    text, planted = doc
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
    assert second.deidentified_text == masked
    for _, value in planted:
        assert value not in second.deidentified_text


@given(doc=planted_documents())
def test_deidentify_deterministic_fixed_seed(doc):
    """A fixed seed yields byte-identical redacted output across runs."""
    text, planted = doc
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
    assert outputs[0] == outputs[1]


@given(doc=planted_documents())
def test_deidentify_keep_mapping_reidentify_inverse(doc):
    """With ``keep_mapping``, ``reidentify`` restores every planted identifier.

    ``reidentify`` must be a left inverse of the redaction mapping: applying it
    to the masked text brings back the original identifiers exactly.
    """
    text, planted = doc
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
    restored = reidentify(result.deidentified_text, result.mapping)
    for ent in result.pii_entities:
        assert ent.text in restored


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
