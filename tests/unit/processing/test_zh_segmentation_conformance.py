"""Shared conformance suite for the pluggable Chinese segmentation backends.

The suite has two layers. Layer A drives the harness with deterministic stub
segmenters and runs everywhere with no optional dependency installed. Layer B
enrolls really installed backends and their user-supplied domain models; it is
marked ``integration`` and skips with an explicit reason when an asset is
absent. No model weights or external corpora are downloaded or committed.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from openmed.processing import zh_segmentation
from openmed.processing.tokenization import SpanToken
from openmed.processing.zh_segmentation import (
    DEFAULT_MEDICAL_USER_DICTIONARY,
    SEGMENTATION_BOUNDARY_F1_FLOOR,
    SEGMENTATION_CONFORMANCE_CHECKS,
    PkusegSegmenter,
    SegmentationAlignmentError,
    SegmentationConformanceCase,
    SegmentationConformanceReport,
    create_chinese_segmenter,
    run_segmenter_conformance,
)

FIXTURE_PATH = (
    Path(__file__).parents[2]
    / "fixtures"
    / "processing"
    / "zh_segmentation_conformance.json"
)
HANLP_MODEL_ENV_VAR = "OPENMED_HANLP_MODEL_PATH"

STUB_TEXT = "患者王芳因心房颤动入院"
STUB_WORDS = ("患者", "王芳", "因", "心房颤动", "入院")
NORMALIZED_TEXT = "患者ＣＴ检查"
NORMALIZED_WORDS = ("患者", "ＣＴ", "检查")


# --- Synthetic corpus ------------------------------------------------------


def _document() -> dict:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def _case_words(document: dict) -> list[tuple[str, ...]]:
    names = document["names"]
    conditions = document["conditions"]
    templates = document["templates"]
    words: list[tuple[str, ...]] = []
    for index in range(len(names) * len(conditions)):
        name = names[index // len(conditions)]
        condition = conditions[index % len(conditions)]
        words.append(
            tuple(
                word.format(name=name, condition=condition)
                for word in templates[index % len(templates)]
            )
        )
    return words


def _conformance_cases() -> list[SegmentationConformanceCase]:
    document = _document()
    dictionary_terms = set(document["dictionary_terms"])
    cases = []
    for words in _case_words(document):
        cases.append(
            SegmentationConformanceCase(
                text="".join(words),
                gold_words=words,
                required_terms=tuple(
                    dict.fromkeys(word for word in words if word in dictionary_terms)
                ),
            )
        )
    return cases


def _reference_vocabulary() -> set[str]:
    """Every word the reference segmenter is allowed to know about."""

    document = _document()
    vocabulary = set(document["dictionary_terms"])
    for template in document["templates"]:
        vocabulary.update(word for word in template if not word.startswith("{"))
    return vocabulary


def _stub_case(required_terms: tuple[str, ...] = ()) -> SegmentationConformanceCase:
    return SegmentationConformanceCase(STUB_TEXT, STUB_WORDS, required_terms)


# --- Stub segmenters -------------------------------------------------------


class ReferenceDictionarySegmenter:
    """Deterministic longest-match segmenter used to exercise the harness."""

    def __init__(self, vocabulary: set[str]) -> None:
        self._vocabulary = set(vocabulary)
        self._longest = max(len(term) for term in self._vocabulary)

    def segment(self, text: str) -> list[SpanToken]:
        tokens: list[SpanToken] = []
        cursor = 0
        while cursor < len(text):
            if text[cursor].isspace():
                cursor += 1
                continue
            width = min(self._longest, len(text) - cursor)
            while width > 1 and text[cursor : cursor + width] not in self._vocabulary:
                width -= 1
            tokens.append(
                SpanToken(text[cursor : cursor + width], cursor, cursor + width)
            )
            cursor += width
        return tokens


class _FixedTokenSegmenter:
    """Returns a hard-coded token list so one invariant breaks in isolation."""

    def __init__(self, tokens: list) -> None:
        self._tokens = tokens

    def segment(self, text: str) -> list:
        return list(self._tokens)


class _RaisingSegmenter:
    def segment(self, text: str) -> list[SpanToken]:
        raise RuntimeError("backend exploded")


class _UnalignableSegmenter:
    """Fails the way ``_align_words_to_text`` fails on an unlocatable word."""

    def segment(self, text: str) -> list[SpanToken]:
        raise SegmentationAlignmentError(
            "Backend token 'CT' could not be aligned after offset 0"
        )


class _NoneReturningSegmenter:
    def segment(self, text: str):
        return None


class _GeneratorSegmenter:
    """A lazy tokenizer shape: yields tokens instead of returning a list."""

    def segment(self, text: str):
        return (
            token
            for token in ReferenceDictionarySegmenter(_reference_vocabulary()).segment(
                text
            )
        )


class _BadOffsetTypeSegmenter:
    def segment(self, text: str) -> list[SpanToken]:
        return [SpanToken("患者", None, 2)]


def _wrong_offset_tokens() -> list[SpanToken]:
    # Same length as the source span, so only the text/offset pairing is wrong.
    return [
        SpanToken("患者", 0, 2),
        SpanToken("李雷", 2, 4),
        SpanToken("因", 4, 5),
        SpanToken("心房颤动", 5, 9),
        SpanToken("入院", 9, 11),
    ]


def _overlapping_tokens() -> list[SpanToken]:
    return [
        SpanToken("患者", 0, 2),
        SpanToken("者王芳", 1, 4),
        SpanToken("因", 4, 5),
        SpanToken("心房颤动", 5, 9),
        SpanToken("入院", 9, 11),
    ]


def _non_monotonic_tokens() -> list[SpanToken]:
    return [
        SpanToken("患者", 0, 2),
        SpanToken("因", 4, 5),
        SpanToken("心房颤动", 5, 9),
        SpanToken("入院", 9, 11),
        SpanToken("王芳", 2, 4),
    ]


def _dropped_span_tokens() -> list[SpanToken]:
    return [
        SpanToken("患者", 0, 2),
        SpanToken("因", 4, 5),
        SpanToken("心房颤动", 5, 9),
        SpanToken("入院", 9, 11),
    ]


def _split_dictionary_term_tokens() -> list[SpanToken]:
    return [
        SpanToken("患者", 0, 2),
        SpanToken("王芳", 2, 4),
        SpanToken("因", 4, 5),
        SpanToken("心房", 5, 7),
        SpanToken("颤动", 7, 9),
        SpanToken("入院", 9, 11),
    ]


def _non_span_token_tokens() -> list:
    return [
        SpanToken("患者", 0, 2),
        ("王芳", 2, 4),
        SpanToken("因", 4, 5),
        SpanToken("心房颤动", 5, 9),
        SpanToken("入院", 9, 11),
    ]


def _run_stub(tokens: list, required_terms: tuple[str, ...] = ()):
    return run_segmenter_conformance(
        _FixedTokenSegmenter(tokens),
        [_stub_case(required_terms)],
        backend="stub",
    )


# --- Layer A: fixture integrity -------------------------------------------


def test_conformance_corpus_is_synthetic_and_reproducible():
    document = _document()
    metadata = document["metadata"]
    texts = ["".join(words) for words in _case_words(document)]
    digest = hashlib.sha256("\n".join(texts).encode("utf-8")).hexdigest()

    assert metadata["synthetic"] is True
    assert metadata["generated_only"] is True
    assert metadata["generator"] == "cartesian-template"
    assert metadata["case_count"] == len(texts) == 200
    assert metadata["char_count"] == sum(len(text) for text in texts)
    assert metadata["sha256"] == f"sha256:{digest}"


def test_every_conformance_name_resolves_from_the_package_namespace():
    # A name in the submodule's __all__ but missing from the package is a
    # documented import path that does not resolve. Pin all of them together
    # so one cannot drift away from the rest.
    import openmed.processing as processing

    exported = {
        "SEGMENTATION_BOUNDARY_F1_FLOOR",
        "SEGMENTATION_CONFORMANCE_CHECKS",
        "SegmentationAlignmentError",
        "SegmentationConformanceCase",
        "SegmentationConformanceIssue",
        "SegmentationConformanceReport",
        "run_segmenter_conformance",
    }

    assert exported <= set(zh_segmentation.__all__)
    assert exported <= set(processing.__all__)
    for name in exported:
        assert getattr(processing, name) is getattr(zh_segmentation, name)


def test_fixture_dictionary_terms_match_the_shipped_medical_dictionary():
    # Without this the fixture is a hand copy that can silently drift from the
    # dictionary the backends actually load.
    shipped = {
        line.split()[0]
        for line in DEFAULT_MEDICAL_USER_DICTIONARY.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip() and not line.startswith("#")
    }

    assert set(_document()["dictionary_terms"]) == shipped


def test_conformance_cases_carry_gold_cuts_and_dictionary_requirements():
    cases = _conformance_cases()

    assert len(cases) == 200
    assert all("".join(case.gold_words) == case.text for case in cases)
    assert all(case.required_terms for case in cases)
    assert {term for case in cases for term in case.required_terms} >= {
        "王芳",
        "心房颤动",
        "心电图",
        "转氨酶",
        "磁共振成像",
    }


@pytest.mark.parametrize(
    ("text", "gold_words"),
    [
        # Too short: the cut stops before the end of the source.
        ("患者王芳", ("患者",)),
        # Right length, wrong content -- a length-only check would accept this.
        ("患者王芳", ("患者", "李雷")),
        ("患者", ("XX",)),
    ],
)
def test_gold_words_that_do_not_reconstruct_the_case_are_rejected(text, gold_words):
    broken = SegmentationConformanceCase(text, gold_words)

    with pytest.raises(ValueError, match="reconstruct"):
        run_segmenter_conformance(_RaisingSegmenter(), [broken], backend="stub")


def test_an_empty_case_list_is_rejected_instead_of_reporting_a_vacuous_pass():
    with pytest.raises(ValueError, match="at least one case"):
        run_segmenter_conformance(_RaisingSegmenter(), [], backend="stub")


@pytest.mark.parametrize(
    "segmenter_factory",
    [_NoneReturningSegmenter, _BadOffsetTypeSegmenter],
    ids=["returns-none", "non-integer-offsets"],
)
def test_malformed_backend_output_is_reported_rather_than_crashing(segmenter_factory):
    report = run_segmenter_conformance(
        segmenter_factory(),
        [_stub_case()],
        backend="stub",
    )

    assert report.ok is False
    assert report.checks_triggered == {"protocol"}
    assert "TypeError" in report.issues[0].detail


def test_a_lazy_generator_backend_is_accepted():
    cases = _conformance_cases()

    report = run_segmenter_conformance(
        _GeneratorSegmenter(),
        cases,
        backend="generator",
    )

    assert report.ok is True
    assert report.boundary_f1 == pytest.approx(1.0)
    assert report.token_count > 0


def test_report_metadata_is_read_only():
    report = run_segmenter_conformance(
        ReferenceDictionarySegmenter(_reference_vocabulary()),
        [_stub_case()],
        backend="stub",
        metadata={"corpus": "x"},
    )

    with pytest.raises(TypeError):
        report.metadata["corpus"] = "mutated"
    assert report.metadata["corpus"] == "x"


# --- Layer A: the harness accepts a conforming backend ---------------------


def test_reference_segmenter_passes_the_shared_suite_and_records_evidence():
    cases = _conformance_cases()
    segmenter = ReferenceDictionarySegmenter(_reference_vocabulary())

    report = run_segmenter_conformance(
        segmenter,
        cases,
        backend="reference",
        metadata={"source": FIXTURE_PATH.name},
    )

    assert isinstance(report, SegmentationConformanceReport)
    assert report.ok is True
    assert report.issues == ()
    assert report.case_count == 200
    assert report.boundary_f1 == pytest.approx(1.0)
    assert report.dictionary_hit_rate == pytest.approx(1.0)
    assert report.chars_per_second > 0.0

    evidence = report.to_evidence()
    assert evidence["backend"] == "reference"
    assert evidence["issue_count"] == 0
    assert evidence["checks_triggered"] == []
    assert evidence["metadata"] == {"source": FIXTURE_PATH.name}
    assert json.loads(json.dumps(evidence)) == evidence


# --- Layer A: every declared check has a proving stub ----------------------


def test_wrong_offsets_are_reported_as_an_offset_defect():
    report = _run_stub(_wrong_offset_tokens())

    assert report.ok is False
    assert report.checks_triggered == {"offsets"}
    assert report.issues[0].case_index == 0
    assert "李雷" in report.issues[0].detail


def test_normalized_token_text_is_reported_and_never_absorbed():
    # A backend that emits a half-width normalization of a full-width source
    # span keeps the span length identical, so only an exact text comparison
    # can catch it.
    report = run_segmenter_conformance(
        _FixedTokenSegmenter(
            [
                SpanToken("患者", 0, 2),
                SpanToken("CT", 2, 4),
                SpanToken("检查", 4, 6),
            ]
        ),
        [SegmentationConformanceCase(NORMALIZED_TEXT, NORMALIZED_WORDS)],
        backend="stub",
    )

    assert report.ok is False
    assert report.checks_triggered == {"offsets"}
    assert "CT" in report.issues[0].detail


def test_overlapping_tokens_are_reported_as_an_overlap_defect():
    report = _run_stub(_overlapping_tokens())

    assert report.ok is False
    assert report.checks_triggered == {"overlap"}
    assert "inside the span" in report.issues[0].detail


def test_non_monotonic_tokens_are_reported_as_an_ordering_defect():
    report = _run_stub(_non_monotonic_tokens())

    assert report.ok is False
    # This particular stub also leaves a gap where the moved token used to sit,
    # so it reports coverage too. Ordering and coverage are not coupled in
    # general: a repeated-substring backwards jump reports ordering alone.
    assert report.checks_triggered == {"ordering", "coverage"}
    ordering = [issue for issue in report.issues if issue.check == "ordering"]
    assert len(ordering) == 1
    assert "after a token starting at" in ordering[0].detail


def test_dropped_source_span_is_reported_as_a_coverage_defect():
    report = _run_stub(_dropped_span_tokens())

    assert report.ok is False
    assert report.checks_triggered == {"coverage"}
    assert "uncovered" in report.issues[0].detail


def test_split_dictionary_term_is_reported_as_a_dictionary_defect():
    report = _run_stub(_split_dictionary_term_tokens(), ("王芳", "心房颤动"))

    assert report.ok is False
    assert report.checks_triggered == {"dictionary"}
    assert report.dictionary_hit_rate == pytest.approx(0.5)
    assert "心房颤动" in report.issues[0].detail


def test_non_spantoken_output_is_reported_as_a_protocol_defect():
    report = _run_stub(_non_span_token_tokens())

    assert report.ok is False
    # This stub's rejected token also leaves its own span uncovered. The two
    # are not coupled in general: a rejected token whose span another token
    # already covers reports protocol alone.
    assert report.checks_triggered == {"protocol", "coverage"}
    protocol = [issue for issue in report.issues if issue.check == "protocol"]
    assert "not SpanToken" in protocol[0].detail


@pytest.mark.parametrize(
    ("tokens", "expected"),
    [
        # A backwards jump over a repeated substring leaves no gap.
        (
            [SpanToken("A", 0, 1), SpanToken("B", 1, 2), SpanToken("A", 0, 1)],
            {"ordering"},
        ),
        # A rejected token whose span is already covered leaves no gap.
        ([SpanToken("AB", 0, 2), ("A", 0, 1)], {"protocol"}),
    ],
    ids=["ordering-without-coverage", "protocol-without-coverage"],
)
def test_ordering_and_protocol_are_not_inherently_coupled_to_coverage(tokens, expected):
    report = run_segmenter_conformance(
        _FixedTokenSegmenter(tokens),
        [SegmentationConformanceCase("AB", ("A", "B"))],
        backend="stub",
    )

    assert report.checks_triggered == expected


def test_backend_exception_is_reported_as_a_protocol_defect():
    report = run_segmenter_conformance(
        _RaisingSegmenter(),
        [_stub_case()],
        backend="stub",
    )

    assert report.ok is False
    assert report.checks_triggered == {"protocol"}
    assert "RuntimeError: backend exploded" in report.issues[0].detail


def test_unalignable_backend_word_is_reported_as_an_alignment_defect(monkeypatch):
    real_import = importlib.import_module
    fake_module = SimpleNamespace(
        pkuseg=lambda **kwargs: SimpleNamespace(cut=lambda text: ["患者", "CT", "检查"])
    )
    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda name: fake_module if name == "pkuseg" else real_import(name),
    )
    segmenter = PkusegSegmenter(model_name="medicine")

    report = run_segmenter_conformance(
        segmenter,
        [SegmentationConformanceCase(NORMALIZED_TEXT, NORMALIZED_WORDS)],
        backend="pkuseg",
    )

    assert report.ok is False
    assert report.checks_triggered == {"alignment"}
    assert "could not be aligned" in report.issues[0].detail

    with pytest.raises(SegmentationAlignmentError):
        segmenter.segment(NORMALIZED_TEXT)
    # Callers written against the previous contract keep working.
    with pytest.raises(ValueError):
        segmenter.segment(NORMALIZED_TEXT)


def test_alignable_backend_words_never_silently_drop_source_text(monkeypatch):
    # A backend that reports only some of the words still yields exact offsets:
    # the alignment back-fill covers the remainder rather than losing it.
    text = "患者王芳因心房颤动入院"
    real_import = importlib.import_module
    fake_module = SimpleNamespace(
        pkuseg=lambda **kwargs: SimpleNamespace(cut=lambda _text: ["心房颤动"])
    )
    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda name: fake_module if name == "pkuseg" else real_import(name),
    )

    report = run_segmenter_conformance(
        PkusegSegmenter(model_name="medicine"),
        [SegmentationConformanceCase(text, STUB_WORDS, ("心房颤动",))],
        backend="pkuseg",
    )

    assert report.ok is True
    assert report.boundary_f1 < 1.0


def test_conformance_suite_runs_end_to_end_against_a_fake_installed_backend(
    monkeypatch,
):
    # Exercises the same code path Layer B uses -- adapter construction, word
    # alignment and the full 200-case corpus -- without any optional
    # dependency, so the opt-in tests are never dead code.
    reference = ReferenceDictionarySegmenter(_reference_vocabulary())
    real_import = importlib.import_module
    fake_module = SimpleNamespace(
        pkuseg=lambda **kwargs: SimpleNamespace(
            cut=lambda text: [token.text for token in reference.segment(text)]
        )
    )
    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda name: fake_module if name == "pkuseg" else real_import(name),
    )
    cases = _conformance_cases()
    segmenter = PkusegSegmenter(model_name="medicine")

    report = run_segmenter_conformance(segmenter, cases, backend="pkuseg")

    assert report.ok is True
    assert report.case_count == 200
    assert report.dictionary_hit_rate == pytest.approx(1.0)
    assert report.boundary_f1 == pytest.approx(1.0)
    # Realignment must reproduce the reference offsets exactly.
    assert segmenter.segment(cases[0].text) == reference.segment(cases[0].text)


def _blob_spans(text: str, terms: set[str]) -> list[tuple[int, int]]:
    longest = max(len(term) for term in terms)
    spans: list[tuple[int, int]] = []
    index = 0
    start: int | None = None
    while index < len(text):
        hit = None
        for width in range(min(longest, len(text) - index), 1, -1):
            if text[index : index + width] in terms:
                hit = text[index : index + width]
                break
        if hit:
            if start is not None:
                spans.append((start, index))
                start = None
            spans.append((index, index + len(hit)))
            index += len(hit)
        else:
            if start is None:
                start = index
            index += 1
    if start is not None:
        spans.append((start, len(text)))
    return spans


class _BlobSegmenter:
    """Emits dictionary terms exactly; merges every other run into one token."""

    def __init__(self, terms: set[str]) -> None:
        self._terms = terms

    def segment(self, text: str) -> list[SpanToken]:
        return [
            SpanToken(text[start:end], start, end)
            for start, end in _blob_spans(text, self._terms)
        ]


class _CharacterSegmenter(_BlobSegmenter):
    """Same, but splits every non-dictionary run into single characters."""

    def segment(self, text: str) -> list[SpanToken]:
        tokens: list[SpanToken] = []
        for start, end in _blob_spans(text, self._terms):
            if text[start:end] in self._terms:
                tokens.append(SpanToken(text[start:end], start, end))
            else:
                tokens.extend(
                    SpanToken(text[offset], offset, offset + 1)
                    for offset in range(start, end)
                )
        return tokens


def test_boundary_f1_floor_catches_a_character_level_segmenter():
    # Documents exactly what the shipped floor does and does not catch. Both
    # stubs satisfy the span protocol and emit every dictionary term, so the
    # conformance checks alone pass them; only the floor separates them.
    terms = set(_document()["dictionary_terms"])
    cases = _conformance_cases()

    blob = run_segmenter_conformance(_BlobSegmenter(terms), cases, backend="blob")
    characters = run_segmenter_conformance(
        _CharacterSegmenter(terms), cases, backend="characters"
    )

    assert blob.ok is True
    assert characters.ok is True

    # A character-level segmenter is rejected by the floor.
    assert characters.boundary_f1 < SEGMENTATION_BOUNDARY_F1_FLOOR
    # Known scope limit: merging non-dictionary runs still clears the floor.
    # Raising the floor past this point would exceed the shipped default
    # backend's own gate, so this stays out of scope rather than being hidden.
    assert blob.boundary_f1 >= SEGMENTATION_BOUNDARY_F1_FLOOR


def test_every_declared_conformance_check_has_a_proving_stub():
    triggered: set[str] = set()
    triggered |= _run_stub(_wrong_offset_tokens()).checks_triggered
    triggered |= _run_stub(_overlapping_tokens()).checks_triggered
    triggered |= _run_stub(_non_monotonic_tokens()).checks_triggered
    triggered |= _run_stub(_dropped_span_tokens()).checks_triggered
    triggered |= _run_stub(
        _split_dictionary_term_tokens(), ("王芳", "心房颤动")
    ).checks_triggered
    triggered |= _run_stub(_non_span_token_tokens()).checks_triggered
    triggered |= run_segmenter_conformance(
        _RaisingSegmenter(), [_stub_case()], backend="stub"
    ).checks_triggered
    # Driven through the real except branch rather than seeded by hand, so
    # deleting that branch turns this red instead of leaving it green.
    triggered |= run_segmenter_conformance(
        _UnalignableSegmenter(), [_stub_case()], backend="stub"
    ).checks_triggered

    assert triggered == set(SEGMENTATION_CONFORMANCE_CHECKS)


# --- Layer B: opt-in tests for really installed backends and models --------


def _skip_without_pkuseg_medicine_model() -> None:
    pkuseg = pytest.importorskip(
        "pkuseg",
        reason="requires the optional openmed[zh-pkuseg] extra (MIT)",
    )
    config = getattr(pkuseg, "config", None)
    home = getattr(config, "pkuseg_home", None)
    if not home or not (Path(home).expanduser() / "medicine").is_dir():
        pytest.skip(
            "requires a user-supplied pkuseg 'medicine' domain model; "
            "OpenMed never downloads model files"
        )


def _hanlp_model_path() -> Path:
    pytest.importorskip(
        "hanlp",
        reason="requires the optional openmed[zh-hanlp] extra (Apache-2.0)",
    )
    raw = os.environ.get(HANLP_MODEL_ENV_VAR)
    if not raw:
        pytest.skip(
            f"requires a user-supplied HanLP model path in {HANLP_MODEL_ENV_VAR}; "
            "OpenMed never downloads model weights"
        )
    path = Path(raw).expanduser()
    if not path.exists():
        pytest.skip(f"{HANLP_MODEL_ENV_VAR} points at a missing path: {path}")
    return path


def _installed_backend(name: str):
    if name == "jieba":
        pytest.importorskip(
            "jieba",
            reason="requires the core jieba dependency to be installed",
        )
        return create_chinese_segmenter("jieba")
    if name == "pkuseg":
        _skip_without_pkuseg_medicine_model()
        return create_chinese_segmenter("pkuseg", pkuseg_domain="medicine")
    return create_chinese_segmenter("hanlp", hanlp_model=_hanlp_model_path())


@pytest.mark.integration
@pytest.mark.parametrize("backend", ["jieba", "pkuseg", "hanlp"])
def test_installed_backend_passes_shared_conformance_suite(backend, record_property):
    segmenter = _installed_backend(backend)
    cases = _conformance_cases()

    report = run_segmenter_conformance(
        segmenter,
        cases,
        backend=backend,
        metadata={"corpus": FIXTURE_PATH.name},
    )
    record_property(
        f"zh-segmentation-conformance::{backend}",
        json.dumps(report.to_evidence(), ensure_ascii=False),
    )

    assert report.ok, [
        (issue.case_index, issue.check, issue.detail) for issue in report.issues[:5]
    ]
    assert report.dictionary_hit_rate == pytest.approx(1.0)
    # Deterministic over a checksum-pinned corpus for a given backend build,
    # so it is gated at the same floor the default backend already meets.
    assert report.boundary_f1 >= SEGMENTATION_BOUNDARY_F1_FLOOR
    # Throughput is the only hardware-dependent number, so it is recorded only.
    assert report.chars_per_second > 0.0


@pytest.mark.integration
def test_installed_backends_agree_on_dictionary_override_spans(record_property):
    available = []
    for backend in ("jieba", "pkuseg", "hanlp"):
        try:
            available.append((backend, _installed_backend(backend)))
        except pytest.skip.Exception:
            continue
    if len(available) < 2:
        pytest.skip(
            "requires at least two installed Chinese segmentation backends "
            "with their user-supplied domain models"
        )

    cases = _conformance_cases()
    spans: dict[str, list[tuple[int, str, tuple[int, int] | None]]] = {}
    for backend, segmenter in available:
        observed = []
        for index, case in enumerate(cases):
            emitted = {
                token.text: (token.start, token.end)
                for token in segmenter.segment(case.text)
            }
            for term in case.required_terms:
                # A backend that never emitted the term records None so the
                # comparison reports the disagreement instead of raising.
                observed.append((index, term, emitted.get(term)))
        spans[backend] = observed

    record_property(
        "zh-segmentation-conformance::agreement",
        json.dumps(sorted(spans), ensure_ascii=False),
    )
    reference = spans[available[0][0]]
    for backend, _segmenter in available[1:]:
        assert spans[backend] == reference
