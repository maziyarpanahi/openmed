"""Contract tests for the portable GLiNER span and GLiClass export profiles.

These run everywhere: they need numpy only and use a pure-Python character
tokenizer instead of ``tokenizers``.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from openmed.eval.export_qualification import (
    ExportIOContract,
    ExportQualificationError,
    ExportReason,
    check_qualified_request,
)
from openmed.mlx.models import normalize_model_config
from openmed.ner.families import zero_shot_export as zs
from tests.fixtures.zero_shot_export_stub import (
    GLICLASS_MARKERS,
    GLINER_MARKERS,
    CharTokenizer,
    gliclass_config,
    gliner_config,
)


def _spec(config: dict) -> zs.ZeroShotModelSpec:
    normalized = normalize_model_config(dict(config))
    return zs.validate_zero_shot_config(
        zs.resolve_zero_shot_profile(normalized), normalized
    )


def _gliner(**bounds_kwargs):
    config = normalize_model_config(gliner_config())
    spec = zs.validate_zero_shot_config(zs.GLINER_SPAN_PROFILE, config)
    tokenizer = CharTokenizer(GLINER_MARKERS)
    prompt = zs.resolve_prompt_spec(zs.GLINER_SPAN_PROFILE, config, spec, tokenizer)
    bounds = zs.default_bounds(spec, **bounds_kwargs)
    return spec, tokenizer, prompt, bounds


def _gliclass(**config_overrides):
    config = normalize_model_config(gliclass_config(**config_overrides))
    spec = zs.validate_zero_shot_config(zs.GLICLASS_PROFILE, config)
    tokenizer = CharTokenizer(GLICLASS_MARKERS)
    prompt = zs.resolve_prompt_spec(zs.GLICLASS_PROFILE, config, spec, tokenizer)
    return spec, tokenizer, prompt, zs.default_bounds(spec)


def _reason(callable_, *args, **kwargs) -> ExportReason:
    with pytest.raises(ExportQualificationError) as error:
        callable_(*args, **kwargs)
    return error.value.reason


# ---------------------------------------------------------------------------
# Profiles, variants, and refusals
# ---------------------------------------------------------------------------


def test_profiles_resolve_from_mlx_family_and_task() -> None:
    assert zs.resolve_zero_shot_profile(gliner_config()) is zs.GLINER_SPAN_PROFILE
    assert zs.resolve_zero_shot_profile(gliclass_config()) is zs.GLICLASS_PROFILE
    manifest = {"family": "gliner-uni-encoder-span", "task": "zero-shot-ner"}
    assert zs.resolve_zero_shot_profile({}, manifest) is zs.GLINER_SPAN_PROFILE


@pytest.mark.parametrize(
    ("config", "reason"),
    [
        (
            {"_mlx_family": "gliner-uni-encoder-token-relex"},
            ExportReason.UNSUPPORTED_TASK,
        ),
        (
            {
                "_mlx_family": "gliner-uni-encoder-span",
                "_mlx_task": "zero-shot-relation-extraction",
            },
            ExportReason.UNSUPPORTED_TASK,
        ),
        (
            {
                "_mlx_family": "gliclass-uni-encoder",
                "_mlx_task": "zero-shot-ner",
            },
            ExportReason.UNSUPPORTED_TASK,
        ),
        ({"_mlx_family": "modernbert"}, ExportReason.UNSUPPORTED_FAMILY),
        (
            {"_mlx_family": "gliner-uni-encoder-span", "auto_map": {"AutoModel": "x"}},
            ExportReason.UNSUPPORTED_REMOTE_CODE,
        ),
        (
            {"_mlx_family": "gliclass-uni-encoder", "trust_remote_code": True},
            ExportReason.UNSUPPORTED_REMOTE_CODE,
        ),
        (
            {
                "_mlx_family": "gliner-uni-encoder-span",
                "_mlx_quantization": {"bits": 8},
            },
            ExportReason.UNSUPPORTED_QUANTIZATION,
        ),
    ],
)
def test_unsupported_profiles_are_refused(config: dict, reason: ExportReason) -> None:
    assert _reason(zs.resolve_zero_shot_profile, config) is reason


@pytest.mark.parametrize(
    "overrides",
    [
        {"span_mode": "token_level"},
        {"num_rnn_layers": 2},
        {"conv_kernel_size": 3},
        {"hidden_act": "relu"},
        {"max_width": 64},
        {"class_token_index": None},
        {"model_type": "bert", "_mlx_model_type": "bert"},
    ],
)
def test_gliner_variants_outside_the_profile_are_refused(overrides: dict) -> None:
    assert (
        _reason(_spec, gliner_config(**overrides))
        is ExportReason.UNSUPPORTED_MODEL_VARIANT
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {"scorer_type": "weighted-dot"},
        {"architecture_type": "bi-encoder"},
        {"extract_text_features": True},
        {"use_lstm": True},
        {"squeeze_layers": True},
        {"pooling_strategy": "last"},
        {"text_token_index": None},
    ],
)
def test_gliclass_variants_outside_the_profile_are_refused(overrides: dict) -> None:
    assert (
        _reason(_spec, gliclass_config(**overrides))
        is ExportReason.UNSUPPORTED_MODEL_VARIANT
    )


def test_explicit_first_token_pooling_has_no_reference() -> None:
    zs.reject_raw_first_token_pooling({"class_token_pooling": "average"})
    zs.reject_raw_first_token_pooling({})
    assert (
        _reason(zs.reject_raw_first_token_pooling, {"class_token_pooling": "first"})
        is ExportReason.UNSUPPORTED_MODEL_VARIANT
    )


def test_prompt_markers_must_match_the_tokenizer() -> None:
    config = normalize_model_config(gliner_config(class_token_index=9))
    spec = zs.validate_zero_shot_config(zs.GLINER_SPAN_PROFILE, config)
    tokenizer = CharTokenizer(GLINER_MARKERS)
    assert (
        _reason(zs.resolve_prompt_spec, zs.GLINER_SPAN_PROFILE, config, spec, tokenizer)
        is ExportReason.UNSUPPORTED_MODEL_VARIANT
    )
    missing = CharTokenizer(("<<OTHER>>",))
    config = normalize_model_config(gliner_config())
    spec = zs.validate_zero_shot_config(zs.GLINER_SPAN_PROFILE, config)
    assert (
        _reason(zs.resolve_prompt_spec, zs.GLINER_SPAN_PROFILE, config, spec, missing)
        is ExportReason.UNSUPPORTED_MODEL_VARIANT
    )


def test_only_onnx_runtime_cpu_float32_is_declared() -> None:
    for profile in (zs.GLINER_SPAN_PROFILE, zs.GLICLASS_PROFILE):
        assert (
            _reason(profile.support_for, "coreml") is ExportReason.UNSUPPORTED_RUNTIME
        )
        assert (
            _reason(profile.support_for, "tflite") is ExportReason.UNSUPPORTED_RUNTIME
        )
        support = profile.support_for("onnx")
        assert support is zs.ZERO_SHOT_ONNX_SUPPORT
    support = zs.ZERO_SHOT_ONNX_SUPPORT
    support.validate(
        precision="float32",
        quantization="none",
        provider="CPUExecutionProvider",
        shape_mode="dynamic",
    )
    for kwargs, reason in (
        ({"precision": "float16"}, ExportReason.UNSUPPORTED_PRECISION),
        ({"quantization": "int8"}, ExportReason.UNSUPPORTED_QUANTIZATION),
        ({"provider": "CoreMLExecutionProvider"}, ExportReason.UNSUPPORTED_PROVIDER),
        ({"provider": "WebGpuExecutionProvider"}, ExportReason.UNSUPPORTED_PROVIDER),
        ({"shape_mode": "static"}, ExportReason.UNSUPPORTED_SEQUENCE_SHAPE),
    ):
        assert _reason(support.validate, **kwargs) is reason


@pytest.mark.parametrize(
    ("config", "runtime", "reason"),
    [
        (gliner_config(), "coreml", ExportReason.UNSUPPORTED_RUNTIME),
        (gliclass_config(), "coreml", ExportReason.UNSUPPORTED_RUNTIME),
        (gliner_config(), "onnx", ExportReason.UNSUPPORTED_TASK),
        (gliclass_config(), "onnx", ExportReason.UNSUPPORTED_TASK),
        (
            {
                "model_type": "deberta-v2",
                "_mlx_family": "gliner-uni-encoder-token-relex",
            },
            "coreml",
            ExportReason.UNSUPPORTED_TASK,
        ),
        ({"model_type": "gliner"}, "onnx", ExportReason.UNSUPPORTED_TASK),
    ],
)
def test_token_classification_converters_refuse_zero_shot_checkpoints(
    config: dict, runtime: str, reason: ExportReason
) -> None:
    with pytest.raises(ExportQualificationError) as error:
        zs.reject_zero_shot_token_classification(config, runtime=runtime)
    assert error.value.reason is reason
    assert "gliner-uni" not in str(error.value)


def test_token_classification_checkpoints_pass_the_guard() -> None:
    zs.reject_zero_shot_token_classification(
        {"model_type": "deberta-v2", "id2label": {"0": "O"}}, runtime="coreml"
    )
    zs.reject_zero_shot_token_classification({"model_type": "bert"}, runtime="onnx")


def test_coreml_converter_refuses_gliner_before_the_allowlist() -> None:
    from openmed.coreml.convert import resolve_supported_model_type

    assert (
        _reason(resolve_supported_model_type, gliner_config())
        is ExportReason.UNSUPPORTED_RUNTIME
    )
    assert resolve_supported_model_type({"model_type": "deberta-v2"}) == "deberta-v2"


# ---------------------------------------------------------------------------
# Bounds
# ---------------------------------------------------------------------------


def test_default_bounds_follow_the_checkpoint() -> None:
    spec, _, _, bounds = _gliner()
    assert bounds.max_labels == zs.DEFAULT_MAX_LABELS
    assert bounds.max_sequence_length == 128  # max_position_embeddings
    assert bounds.max_words == 96  # max_len
    assert bounds.max_span_width == 4  # max_width
    assert zs.ZeroShotBounds.from_dict(bounds.to_dict()) == bounds
    assert bounds.to_dict()["overflow"] == "reject"
    cls_spec, _, _, cls_bounds = _gliclass()
    assert cls_bounds.max_words is None and cls_bounds.max_span_width is None
    assert spec.span_width == 4 and cls_spec.span_width is None


@pytest.mark.parametrize(
    ("kwargs", "reason", "limit"),
    [
        ({"max_labels": 65}, ExportReason.UNSUPPORTED_LABEL_COUNT, 64),
        ({"max_labels": 0}, ExportReason.UNSUPPORTED_LABEL_COUNT, 64),
        ({"max_sequence_length": 129}, ExportReason.UNSUPPORTED_SEQUENCE_LENGTH, 128),
        ({"max_words": 97}, ExportReason.UNSUPPORTED_WORD_COUNT, 96),
        ({"max_span_width": 5}, ExportReason.UNSUPPORTED_SPAN_WIDTH, 4),
    ],
)
def test_export_bounds_beyond_the_model_are_refused(
    kwargs: dict, reason: ExportReason, limit: int
) -> None:
    spec, _, _, _ = _gliner()
    with pytest.raises(zs.ZeroShotRequestError) as error:
        zs.default_bounds(spec, **kwargs)
    assert error.value.reason is reason
    assert error.value.limit == limit


def test_gliclass_bounds_reject_span_options() -> None:
    spec, _, _, _ = _gliclass()
    assert (
        _reason(zs.default_bounds, spec, max_span_width=2)
        is ExportReason.IO_CONTRACT_MISMATCH
    )


def test_bounds_payload_must_be_bounded() -> None:
    assert (
        _reason(zs.ZeroShotBounds.from_dict, {"max_labels": 4})
        is ExportReason.IO_CONTRACT_MISMATCH
    )
    assert (
        _reason(
            zs.ZeroShotBounds,
            max_labels=4,
            max_sequence_length=0,
        )
        is ExportReason.IO_CONTRACT_MISMATCH
    )


# ---------------------------------------------------------------------------
# Labels and prompt encoding
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("labels", "reason", "dimension"),
    [
        ([], ExportReason.UNSUPPORTED_LABEL_COUNT, "labels"),
        (["x"] * 33, ExportReason.UNSUPPORTED_LABEL_COUNT, "labels"),
        ("person", ExportReason.UNSUPPORTED_LABEL, "labels"),
        (["person", " "], ExportReason.UNSUPPORTED_LABEL, "label"),
        (["person", 3], ExportReason.UNSUPPORTED_LABEL, "label"),
        (["person", "person"], ExportReason.UNSUPPORTED_LABEL, "labels"),
        (["a" * 129], ExportReason.UNSUPPORTED_LABEL, "label_chars"),
        (["per\x00son"], ExportReason.UNSUPPORTED_LABEL, "label"),
        (["person<<ENT>>"], ExportReason.UNSUPPORTED_CONTROL_MARKER, "label"),
        (["<<SEP>>"], ExportReason.UNSUPPORTED_CONTROL_MARKER, "label"),
    ],
)
def test_label_contract_rejects_invalid_candidate_labels(
    labels, reason: ExportReason, dimension: str
) -> None:
    _, _, prompt, bounds = _gliner()
    with pytest.raises(zs.ZeroShotRequestError) as error:
        zs.check_labels(labels, bounds=bounds, prompt=prompt)
    assert error.value.reason is reason
    assert error.value.dimension == dimension
    refusal = zs.ZeroShotUnsupported.from_error(error.value)
    assert refusal.status == "unsupported" and refusal.ok is False
    assert set(refusal.to_dict()) == {"status", "reason", "dimension", "limit"}
    assert "person" not in json.dumps(refusal.to_dict())


def test_labels_keep_caller_order() -> None:
    _, _, prompt, bounds = _gliner()
    labels = ("zeta", "alpha", "mid")
    assert zs.check_labels(list(labels), bounds=bounds, prompt=prompt) == labels


def test_word_splitter_matches_the_mlx_runtime_and_counts_code_points() -> None:
    from openmed.mlx.inference import _split_words_with_offsets

    text = "café ZOË-HALL saw 🙂 naïve dr. o'neil 44-17"
    words, offsets = _split_words_with_offsets(text)
    split = zs.split_words(text)
    assert [word for word, _, _ in split] == words
    assert [(start, end) for _, start, end in split] == offsets
    for word, start, end in split:
        assert text[start:end] == word
    zoe = next(item for item in split if item[0].startswith("ZOË"))
    assert zoe[1:] == (5, 13)  # code points, not UTF-8 bytes


def test_gliner_encoding_builds_the_documented_tensors() -> None:
    spec, tokenizer, prompt, bounds = _gliner()
    text = "seen ZORVAK QUILLAN today"
    encoding = zs.encode_gliner_span(
        tokenizer,
        text,
        ("person", "identifier"),
        prompt=prompt,
        bounds=bounds,
        graph_span_width=4,
    )
    feed = encoding.feed
    assert list(feed) == list(zs.GLINER_SPAN_PROFILE.input_names)
    ids = feed["input_ids"][0].tolist()
    assert ids.count(prompt.marker_token_id) == 2
    assert feed["input_ids"].dtype == np.int64
    assert feed["span_mask"].dtype == bool
    assert feed["text_lengths"].tolist() == [[4]]
    words_mask = feed["words_mask"][0].tolist()
    assert sorted(value for value in words_mask if value) == [1, 2, 3, 4]
    # The first token of each word carries its 1-based index.
    first_tokens = [ids[index] for index, value in enumerate(words_mask) if value]
    assert [chr(token - 100) for token in first_tokens] == ["s", "Z", "Q", "t"]
    assert feed["span_idx"].shape == (1, 16, 2)
    spans = feed["span_idx"][0].tolist()
    assert spans[:4] == [[0, 0], [0, 1], [0, 2], [0, 3]]
    mask = feed["span_mask"][0].tolist()
    # Word 3 (last) only has a width-1 span inside the text.
    assert mask[12:16] == [True, False, False, False]
    assert encoding.word_offsets == ((0, 4), (5, 11), (12, 19), (20, 25))
    assert encoding.token_count == len(ids)

    narrow = zs.encode_gliner_span(
        tokenizer,
        text,
        ("person",),
        prompt=prompt,
        bounds=bounds,
        graph_span_width=4,
        span_width=2,
    )
    assert narrow.feed["span_mask"][0].tolist()[:4] == [True, True, False, False]


@pytest.mark.parametrize(
    ("text", "kwargs", "reason", "limit"),
    [
        ("word " * 97, {}, ExportReason.UNSUPPORTED_WORD_COUNT, 96),
        ("x" * 200, {}, ExportReason.UNSUPPORTED_SEQUENCE_LENGTH, 128),
        ("short text", {"span_width": 5}, ExportReason.UNSUPPORTED_SPAN_WIDTH, 4),
        ("short text", {"span_width": 0}, ExportReason.UNSUPPORTED_SPAN_WIDTH, 4),
    ],
)
def test_gliner_requests_outside_the_bounds_are_refused(
    text: str, kwargs: dict, reason: ExportReason, limit: int
) -> None:
    _, tokenizer, prompt, bounds = _gliner()
    with pytest.raises(zs.ZeroShotRequestError) as error:
        zs.encode_gliner_span(
            tokenizer,
            text,
            ("person",),
            prompt=prompt,
            bounds=bounds,
            graph_span_width=4,
            **kwargs,
        )
    assert error.value.reason is reason
    assert error.value.limit == limit


def test_gliner_refuses_words_the_tokenizer_drops() -> None:
    _, _, prompt, bounds = _gliner()
    dropping = CharTokenizer(GLINER_MARKERS, drop="§")
    with pytest.raises(zs.ZeroShotRequestError) as error:
        zs.encode_gliner_span(
            dropping,
            "see § here",
            ("person",),
            prompt=prompt,
            bounds=bounds,
            graph_span_width=4,
        )
    assert error.value.reason is ExportReason.MISSING_OFFSET_MAPPING


def test_gliner_text_cannot_inject_prompt_markers() -> None:
    _, tokenizer, prompt, bounds = _gliner()
    encoding = zs.encode_gliner_span(
        tokenizer,
        "text with <<ENT>> inside",
        ("person",),
        prompt=prompt,
        bounds=bounds,
        graph_span_width=4,
    )
    # The word splitter breaks the marker into punctuation and a word.
    assert encoding.feed["input_ids"][0].tolist().count(prompt.marker_token_id) == 1


def test_gliclass_encoding_layout_and_refusals() -> None:
    spec, tokenizer, prompt, bounds = _gliclass()
    encoding = zs.encode_gliclass(
        tokenizer, "chest pain", ("cardiology", "billing"), prompt=prompt, bounds=bounds
    )
    ids = encoding.feed["input_ids"][0].tolist()
    assert list(encoding.feed) == ["input_ids", "attention_mask"]
    assert ids[:2] == [2, prompt.marker_token_id]
    assert ids.count(prompt.marker_token_id) == 2
    assert ids.count(prompt.separator_token_id) == 1
    assert ids.index(prompt.separator_token_id) < ids.index(100 + ord("h"))

    _, tokenizer_last, prompt_last, _ = _gliclass(
        _mlx_prompt_spec={
            "label_token": "<<LABEL>>",
            "separator_token": "<<SEP>>",
            "prompt_first": False,
        }
    )
    last = zs.encode_gliclass(
        tokenizer_last, "chest", ("cardiology",), prompt=prompt_last, bounds=bounds
    )
    assert last.feed["input_ids"][0].tolist()[1] == 100 + ord("c")

    for text, instruction, reason in (
        ("text <<LABEL>> injected", None, ExportReason.UNSUPPORTED_CONTROL_MARKER),
        ("ok", "use <<SEP>> here", ExportReason.UNSUPPORTED_CONTROL_MARKER),
        ("x" * 200, None, ExportReason.UNSUPPORTED_SEQUENCE_LENGTH),
    ):
        with pytest.raises(zs.ZeroShotRequestError) as error:
            zs.encode_gliclass(
                tokenizer,
                text,
                ("cardiology",),
                prompt=prompt,
                bounds=bounds,
                instruction=instruction,
            )
        assert error.value.reason is reason


def test_memory_budget_is_enforced_before_inference() -> None:
    spec, _, _, bounds = _gliner()
    small = zs.estimate_activation_bytes(
        spec, sequence_length=32, num_labels=2, num_words=8, span_width=4
    )
    large = zs.estimate_activation_bytes(
        spec, sequence_length=128, num_labels=32, num_words=96, span_width=4
    )
    assert 0 < small < large
    assert (
        zs.check_memory_budget(
            spec,
            budget_bytes=None,
            sequence_length=32,
            num_labels=2,
            num_words=8,
        )
        > 0
    )
    with pytest.raises(zs.ZeroShotRequestError) as error:
        zs.check_memory_budget(
            spec,
            budget_bytes=small,
            sequence_length=128,
            num_labels=32,
            num_words=96,
        )
    assert error.value.reason is ExportReason.UNSUPPORTED_MEMORY_BUDGET
    assert error.value.limit == small


# ---------------------------------------------------------------------------
# Decoders
# ---------------------------------------------------------------------------


def _span_logits(cells: dict, *, words: int = 4, width: int = 3, labels: int = 2):
    logits = np.full((1, words, width, labels), -9.0, dtype=np.float32)
    for (start, step, label), value in cells.items():
        logits[0, start, step, label] = value
    return logits


OFFSETS = ((0, 4), (5, 9), (10, 14), (15, 19))


def test_gliner_decoder_threshold_overlap_and_ordering() -> None:
    logits = _span_logits(
        {
            (1, 1, 0): 3.0,  # words 1-2 label 0
            (1, 0, 0): 2.0,  # word 1 label 0, overlaps
            (2, 0, 1): 1.0,  # word 2 label 1, overlaps
            (3, 0, 1): 0.0,  # exactly 0.5: kept (score >= threshold)
            (0, 0, 1): -0.1,  # below threshold
        }
    )
    flat = zs.decode_gliner_spans(
        logits, word_offsets=OFFSETS, labels=("person", "identifier")
    )
    assert [(s.label, s.start, s.end) for s in flat.spans] == [
        ("person", 5, 14),
        ("identifier", 15, 19),
    ]
    assert flat.spans[1].score == pytest.approx(0.5)
    nested = zs.decode_gliner_spans(
        logits, word_offsets=OFFSETS, labels=("person", "identifier"), flat_ner=False
    )
    assert [(s.label_index, s.start, s.end) for s in nested.spans] == [
        (0, 5, 9),
        (0, 5, 14),
        (1, 10, 14),
        (1, 15, 19),
    ]
    narrow = zs.decode_gliner_spans(
        logits, word_offsets=OFFSETS, labels=("person", "identifier"), span_width=1
    )
    assert [(s.start, s.end) for s in narrow.spans] == [(5, 9), (10, 14), (15, 19)]


def test_gliner_decoder_breaks_ties_deterministically() -> None:
    tied = _span_logits({(1, 0, 1): 2.0, (1, 0, 0): 2.0, (0, 1, 0): 2.0})
    result = zs.decode_gliner_spans(
        tied, word_offsets=OFFSETS, labels=("person", "identifier")
    )
    # Equal scores: earlier start wins (words 0-1), then caller label order.
    assert [(s.label_index, s.start, s.end) for s in result.spans] == [(0, 0, 9)]
    same_span = _span_logits({(2, 0, 1): 2.0, (2, 0, 0): 2.0})
    result = zs.decode_gliner_spans(
        same_span, word_offsets=OFFSETS, labels=("first", "second")
    )
    assert [(s.label, s.start) for s in result.spans] == [("first", 10)]


def test_gliner_decoder_ignores_spans_past_the_text_and_checks_shapes() -> None:
    logits = _span_logits({(3, 2, 0): 5.0})  # ends past the last word
    assert (
        zs.decode_gliner_spans(logits, word_offsets=OFFSETS, labels=("a", "b")).spans
        == ()
    )
    assert (
        zs.decode_gliner_spans(
            np.zeros((1, 0, 3, 1)), word_offsets=(), labels=("a",)
        ).spans
        == ()
    )
    for bad, labels in (
        (np.zeros((2, 4, 3, 2)), ("a", "b")),
        (np.zeros((1, 3, 3, 2)), ("a", "b")),
        (np.zeros((1, 4, 3, 1)), ("a", "b")),
    ):
        assert (
            _reason(zs.decode_gliner_spans, bad, word_offsets=OFFSETS, labels=labels)
            is ExportReason.IO_CONTRACT_MISMATCH
        )


def test_gliclass_decoder_ranks_with_label_order_ties() -> None:
    result = zs.decode_gliclass(
        [[0.0, 2.0, 2.0, -3.0]], labels=("a", "b", "c", "d"), threshold=0.5
    )
    assert [item.label for item in result.scores] == ["b", "c", "a", "d"]
    assert [item.label for item in result.accepted] == ["b", "c", "a"]
    assert result.ok and result.scores[2].score == pytest.approx(0.5)
    assert (
        _reason(zs.decode_gliclass, [[0.0]], labels=("a", "b"))
        is ExportReason.IO_CONTRACT_MISMATCH
    )


# ---------------------------------------------------------------------------
# Harness integration
# ---------------------------------------------------------------------------


def test_io_contract_works_with_the_harness_request_check() -> None:
    _, _, prompt, bounds = _gliner()
    contract = zs.build_io_contract(zs.GLINER_SPAN_PROFILE, bounds, graph_span_width=4)
    payload = contract.to_dict()
    assert [item["name"] for item in payload["inputs"]] == list(
        zs.GLINER_SPAN_PROFILE.input_names
    )
    assert payload["outputs"][0]["shape"] == [1, "words", 4, "labels"]
    assert payload["bounds"]["max_labels"] == 32
    rebuilt = ExportIOContract.from_dict(payload)
    assert rebuilt.max_sequence_length == 128
    assert rebuilt.bucket_for(100) == 100
    report = {
        "format": "openmed-export-qualification",
        "schema_version": 1,
        "runtime": "onnx",
        "family": "gliner-uni-encoder-span",
        "source": {"revision": "sha256:0"},
        "tokenizer": {"sha256": "1"},
        "label_map": zs.label_schema_identity(zs.GLINER_SPAN_PROFILE, prompt, bounds),
        "io_contract": payload,
        "offset_contract": {},
        "environment": {},
        "suite": {},
        "artifacts": {"model.onnx": "2"},
        "variants": [
            {
                "name": "onnx-fp32",
                "precision": "float32",
                "provider": "CPUExecutionProvider",
                "passed": True,
                "artifacts": ["model.onnx"],
            }
        ],
    }
    check_qualified_request(
        report, runtime="onnx", artifact="model.onnx", sequence_length=128
    )
    with pytest.raises(ExportQualificationError) as error:
        check_qualified_request(
            report, runtime="onnx", artifact="model.onnx", sequence_length=129
        )
    assert error.value.reason is ExportReason.UNSUPPORTED_SEQUENCE_LENGTH


def test_label_schema_identity_records_no_label_text() -> None:
    _, _, prompt, bounds = _gliner()
    identity = zs.label_schema_identity(zs.GLINER_SPAN_PROFILE, prompt, bounds)
    assert identity["kind"] == "caller_supplied_labels"
    assert identity["ordering"] == "caller_order"
    assert len(identity["sha256"]) == 64
    assert "person" not in json.dumps(identity)


def test_typed_refusal_round_trips_to_an_error() -> None:
    refusal = zs.ZeroShotUnsupported(
        reason=ExportReason.UNSUPPORTED_LABEL_COUNT, dimension="labels", limit=32
    )
    assert refusal.reason_code == "unsupported_label_count"
    with pytest.raises(zs.ZeroShotRequestError) as error:
        refusal.raise_for_status()
    assert error.value.reason is ExportReason.UNSUPPORTED_LABEL_COUNT
    assert error.value.limit == 32
