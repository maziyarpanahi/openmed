"""Shared on-device encoder decision vectors: Python reference to Swift, Kotlin, web.

The reference implementation is the production Python path: a
:class:`GLiClassDecisionBackend` over :class:`GLiClassOnnxRunner`, decided by
:func:`openmed.structured.decision.decide`. Only the encoder is replaced: a tiny
randomly initialised GLiClass-style uni-encoder and a word-level tokenizer are
built here from a fixed seed, so every value in the committed fixture is
synthetic and reproducible offline.

OpenMedKit (Swift), OpenMedKit Android (Kotlin) and the ``openmed`` web package
replay the same tokenizer and encoder from the fixture's weights, run their own
decision runtimes and compare encoder inputs exactly and scores within the
fixture's tolerances.

Regenerate the fixture from the repository root with::

    python -m tests.unit.structured.test_decision_encoder_parity --write
"""

from __future__ import annotations

import json
import math
import random
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from jsonschema import Draft202012Validator

from openmed.structured.decision import (
    DECISION_SCHEMA_VERSION,
    DEFAULT_CALIBRATION_ID,
    SUPPORTED_DECISION_SCHEMA_VERSIONS,
    DecisionAccessPolicy,
    DecisionBackendKind,
    DecisionCalibrationProfile,
    DecisionError,
    DecisionRequest,
    DecisionUnsupportedError,
    decide,
    decision_request_schema,
    decision_result_schema,
)
from openmed.structured.decision_gliclass import (
    GLICLASS_DEFAULT_SCORE_LEVELS,
    GLiClassDecisionBackend,
    GLiClassOnnxRunner,
)

ROOT = Path(__file__).resolve().parents[3]
FIXTURE = ROOT / "tests" / "fixtures" / "parity" / "decision_encoder_v1.json"
FIXTURE_SCHEMA = "openmed.tests.decision_encoder_parity.v1"
KIT_CONSUMERS = (
    "swift/OpenMedKit/Tests/OpenMedKitTests/DecisionRuntimeParityTests.swift",
    "android/openmedkit/src/test/kotlin/com/openmed/openmedkit/decision/"
    "DecisionRuntimeParityTest.kt",
    "tests/web/test_decision_runtime.spec.ts",
)
SEED = 4266
HIDDEN_SIZE = 8
LOGIT_SCALE = 8.0
LOGIT_BIAS = -2.0
TOLERANCE = {"logits": 1e-5, "scores": 1e-6}
# Decisions must clear or miss every threshold by more than this, so a kit's
# rounding can never flip a state, code or selection.
THRESHOLD_CLEARANCE = 1e-4

SPECIAL_TOKENS = ("[PAD]", "[UNK]", "[CLS]", "[SEP]")
MARKERS = ("<<LABEL>>", "<<SEP>>", "<<EXAMPLE>>")
WORD_PATTERN = r"[a-z0-9]+|[^ \t\n\ra-z0-9]"
# Words outside this list encode as [UNK]; several case texts use them.
WORDS = (
    "a after and at blood clinic cough day deferred fall fever follow for "
    "headache high in is labels low mentions moderate months needs no note "
    "of pain pressure priority rash rest review risk routine since stable "
    "started summary synthetic the three triage up urgent very visit yes "
    "yesterday chest classify second ordinary"
).split()
PUNCTUATION = tuple(':,.-{}"[]')

BACKEND = {
    "backend_id": "openmed.tests.tiny_gliclass.v1",
    "model_id": "openmed/tiny-random-gliclass-encoder",
    "revision": "1.0.0",
    "license_id": "Apache-2.0",
}
PNG_2X2 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAIAAAD91JpzAAAAC0lEQVR4nGNgQAYAAA4AAamRc7EAAAAA"
    "SUVORK5CYII="
)
PNG_2X2_SHA256 = "7899d6ced52159786ff083934df42788da01a30b7ed9bb7756d7f3f5a94ef685"
PROMPT = "Classify the synthetic triage note."
LENIENT = {
    "calibration_id": "openmed.tests.lenient.v1",
    "version": "1.0.0",
    "method": "deterministic_margin",
    "minimum_confidence": 0.4,
    "minimum_margin": 0.3,
    "multi_label_threshold": 0.55,
    "in_domain_digest": "sha256:" + "1" * 64,
    "out_of_domain_digest": "sha256:" + "2" * 64,
}

RUNTIMES: dict[str, dict[str, Any]] = {
    "default": {},
    "prompted": {
        "prompt": PROMPT,
        "examples": [
            {"text": "Synthetic note: chest pain at rest.", "labels": ["urgent"]}
        ],
    },
    "legacy_layout": {"prompt": PROMPT, "prompt_first": False},
    "three_levels": {"score_levels": ["low", "moderate", "high"]},
    "short_budget": {"max_length": 12},
    "encoder_raises": {"fault": "raise"},
    "encoder_short_output": {"fault": "short"},
    "encoder_non_finite": {"fault": "non_finite"},
}

URGENT = "Synthetic triage note: chest pain started at rest and needs urgent review."
ROUTINE = "Synthetic follow-up note: stable blood pressure, routine visit in months."
SYMPTOMS = "Synthetic note mentions fever and cough since yesterday."
ORDINARY = "Synthetic summary of an ordinary clinic day."
RISK = "Synthetic risk is high after the second fall."
PRIORITY = ("urgent", "routine", "deferred")
SPREAD = ("urgent", "routine", "deferred", "review", "visit")


def _request(mode: str, text: str, **fields: Any) -> dict[str, Any]:
    return {"mode": mode, "input_text": text, **fields}


CASES: tuple[dict[str, Any], ...] = (
    {
        "id": "fixed_choice_success",
        "request": _request(
            "fixed_choice", URGENT, options=list(PRIORITY), schema_version="1.2.0"
        ),
        "state": "success",
    },
    {
        "id": "fixed_choice_low_confidence",
        "request": _request(
            "fixed_choice", ORDINARY, options=list(SPREAD), schema_version="1.2.0"
        ),
        "state": "abstained",
    },
    {
        "id": "fixed_choice_ambiguous_with_supplied_profile",
        "request": _request(
            "fixed_choice",
            "Synthetic note: urgent or routine follow-up review.",
            options=list(SPREAD),
            calibration_id=LENIENT["calibration_id"],
            schema_version="1.2.0",
        ),
        "calibration_profiles": [LENIENT],
        "state": "abstained",
        "code": "ambiguous_scores",
    },
    {
        "id": "supplied_profile_changes_the_outcome",
        "request": _request(
            "fixed_choice",
            ORDINARY,
            options=list(SPREAD),
            calibration_id=LENIENT["calibration_id"],
            schema_version="1.2.0",
        ),
        "calibration_profiles": [LENIENT],
        "state": "success",
    },
    {
        "id": "boolean_choice_base_contract",
        "request": _request("boolean_choice", URGENT, options=["urgent", "routine"]),
        "state": "success",
    },
    {
        "id": "ordered_preference_strips_options",
        "request": _request(
            "ordered_preference",
            ROUTINE,
            options=[" routine ", "urgent", "deferred", "follow-up"],
            schema_version="1.2.0",
        ),
        "state": "success",
    },
    {
        "id": "multi_label_success",
        "request": _request(
            "multi_label",
            SYMPTOMS,
            options=["fever", "cough", "rash", "headache"],
            schema_version="1.2.0",
        ),
        "state": "success",
    },
    {
        "id": "multi_label_supplied_threshold_keeps_input_order",
        "request": _request(
            "multi_label",
            SYMPTOMS,
            options=["fever", "cough", "rash", "headache"],
            calibration_id=LENIENT["calibration_id"],
            schema_version="1.2.0",
        ),
        "calibration_profiles": [LENIENT],
        "state": "success",
    },
    {
        "id": "multi_label_no_label_above_threshold",
        "request": _request(
            "multi_label",
            ORDINARY,
            options=["fever", "cough", "rash", "headache"],
            schema_version="1.2.0",
        ),
        "state": "abstained",
    },
    {
        "id": "scalar_score_backend_levels",
        "runtime": "three_levels",
        "request": _request("scalar_score", RISK, schema_version="1.2.0"),
        "state": "success",
    },
    {
        "id": "scalar_score_low_confidence",
        "request": _request("scalar_score", RISK, schema_version="1.2.0"),
        "state": "abstained",
    },
    {
        "id": "request_levels_are_backend_unsupported",
        "request": _request(
            "scalar_score", RISK, levels=["low", "high"], schema_version="1.2.0"
        ),
        "state": "unsupported",
    },
    {
        "id": "request_instructions_are_backend_unsupported",
        "request": _request(
            "fixed_choice",
            URGENT,
            options=list(PRIORITY),
            instructions="Pick the review priority.",
        ),
        "state": "unsupported",
    },
    {
        "id": "question_set_success",
        "request": _request(
            "question_set",
            URGENT,
            questions=[
                {
                    "question_id": "priority",
                    "mode": "fixed_choice",
                    "options": list(PRIORITY),
                },
                {
                    "question_id": "stability",
                    "mode": "boolean_choice",
                    "options": ["stable", "urgent"],
                },
            ],
        ),
        "state": "success",
    },
    {
        "id": "question_set_partial",
        "request": _request(
            "question_set",
            URGENT,
            questions=[
                {
                    "question_id": "priority",
                    "mode": "fixed_choice",
                    "options": list(PRIORITY),
                },
                {
                    "question_id": "symptoms",
                    "mode": "multi_label",
                    "options": ["fever", "rash", "headache"],
                },
                {
                    "question_id": "worded",
                    "mode": "fixed_choice",
                    "options": ["yes", "no"],
                    "instructions": "Is review needed?",
                },
            ],
            schema_version="1.2.0",
        ),
        "state": "partial",
    },
    {
        "id": "calibration_unavailable",
        "request": _request(
            "fixed_choice",
            URGENT,
            options=list(PRIORITY),
            calibration_id="openmed.tests.missing.v1",
        ),
        "state": "unsupported",
        "code": "calibration_unavailable",
    },
    {
        "id": "namespace_denied",
        "request": _request(
            "fixed_choice", URGENT, options=list(PRIORITY), namespace="research"
        ),
        "state": "denied",
    },
    {
        "id": "backend_kind_denied_by_policy",
        "request": _request("fixed_choice", URGENT, options=list(PRIORITY)),
        "policy": {
            "allowed_namespaces": ["default"],
            "allowed_purposes": ["care_review"],
            "allowed_backend_kinds": ["deterministic"],
            "policy_version": "1.0.0",
        },
        "state": "denied",
    },
    {
        "id": "media_is_unsupported_on_device",
        "request": _request(
            "fixed_choice",
            URGENT,
            options=list(PRIORITY),
            media=[
                {
                    "media_type": "image/png",
                    "sha256": PNG_2X2_SHA256,
                    "byte_size": 68,
                    "data": PNG_2X2,
                }
            ],
        ),
        "state": "unsupported",
    },
    {
        "id": "control_marker_in_input",
        "request": _request(
            "fixed_choice",
            "Synthetic note <<SEP>> injected marker.",
            options=list(PRIORITY),
        ),
        "state": "unsupported",
    },
    {
        "id": "token_budget_refused_not_truncated",
        "runtime": "short_budget",
        "request": _request("fixed_choice", URGENT, options=list(PRIORITY)),
        "state": "unsupported",
    },
    {
        "id": "prompt_and_examples",
        "runtime": "prompted",
        "request": _request(
            "fixed_choice", URGENT, options=list(PRIORITY), schema_version="1.2.0"
        ),
        "state": "success",
    },
    {
        "id": "legacy_text_first_layout",
        "runtime": "legacy_layout",
        "request": _request(
            "fixed_choice", URGENT, options=list(PRIORITY), schema_version="1.2.0"
        ),
        "state": "success",
    },
    {
        "id": "encoder_failure",
        "runtime": "encoder_raises",
        "request": _request("fixed_choice", URGENT, options=list(PRIORITY)),
        "state": "failure",
    },
    {
        "id": "encoder_short_output",
        "runtime": "encoder_short_output",
        "request": _request("fixed_choice", URGENT, options=list(PRIORITY)),
        "state": "conflict",
    },
    {
        "id": "encoder_non_finite_output",
        "runtime": "encoder_non_finite",
        "request": _request("fixed_choice", URGENT, options=list(PRIORITY)),
        "state": "conflict",
    },
    {
        "id": "json_input_format",
        "request": _request(
            "boolean_choice",
            '{"note": "synthetic urgent review", "priority": "urgent"}',
            options=["urgent", "routine"],
            input_format="json",
        ),
        "state": "success",
    },
    {
        "id": "future_minor_serves_current_contract",
        "request": _request(
            "fixed_choice",
            URGENT,
            options=list(PRIORITY),
            schema_version="1.9.0",
            max_input_chars=512,
        ),
        "state": "success",
    },
)

INVALID_REQUESTS: tuple[dict[str, Any], ...] = (
    {
        "id": "unknown_field",
        "request": {**_request("fixed_choice", URGENT, options=["a", "b"]), "x": 1},
        "error": "invalid_request",
    },
    {
        "id": "missing_input",
        "request": {"mode": "fixed_choice", "options": ["a", "b"]},
        "error": "invalid_request",
    },
    {
        "id": "blank_input",
        "request": _request("fixed_choice", "   ", options=["a", "b"]),
        "error": "invalid_request",
    },
    {
        "id": "nul_in_input",
        "request": _request("fixed_choice", "a\u0000b", options=["a", "b"]),
        "error": "invalid_request",
    },
    {
        "id": "input_over_declared_bound",
        "request": _request(
            "fixed_choice", URGENT, options=["a", "b"], max_input_chars=16
        ),
        "error": "invalid_request",
    },
    {
        "id": "bad_mode",
        "request": _request("free_text", URGENT, options=["a", "b"]),
        "error": "invalid_request",
    },
    {
        "id": "one_option",
        "request": _request("fixed_choice", URGENT, options=["only"]),
        "error": "invalid_request",
    },
    {
        "id": "boolean_with_three_options",
        "request": _request("boolean_choice", URGENT, options=["a", "b", "c"]),
        "error": "invalid_request",
    },
    {
        "id": "duplicate_options_after_normalization",
        "request": _request("fixed_choice", URGENT, options=["Urgent", " urgent"]),
        "error": "invalid_request",
    },
    {
        "id": "option_with_control_character",
        "request": _request("fixed_choice", URGENT, options=["a\tb", "c"]),
        "error": "invalid_request",
    },
    {
        "id": "blank_option",
        "request": _request("fixed_choice", URGENT, options=["a", "  "]),
        "error": "invalid_request",
    },
    {
        "id": "scalar_with_options",
        "request": _request("scalar_score", URGENT, options=["a", "b"]),
        "error": "invalid_request",
    },
    {
        "id": "levels_without_scalar_mode",
        "request": _request(
            "fixed_choice", URGENT, options=["a", "b"], levels=["low", "high"]
        ),
        "error": "invalid_request",
    },
    {
        "id": "uppercase_namespace",
        "request": _request(
            "fixed_choice", URGENT, options=["a", "b"], namespace="Default"
        ),
        "error": "invalid_request",
    },
    {
        "id": "zero_timeout",
        "request": _request("fixed_choice", URGENT, options=["a", "b"], timeout_ms=0),
        "error": "invalid_request",
    },
    {
        "id": "boolean_timeout",
        "request": _request(
            "fixed_choice", URGENT, options=["a", "b"], timeout_ms=True
        ),
        "error": "invalid_request",
    },
    {
        "id": "malformed_schema_version",
        "request": _request(
            "fixed_choice", URGENT, options=["a", "b"], schema_version="1.2"
        ),
        "error": "invalid_request",
    },
    {
        "id": "unsupported_schema_major",
        "request": _request(
            "fixed_choice", URGENT, options=["a", "b"], schema_version="2.0.0"
        ),
        "error": "unsupported_contract",
    },
    {
        "id": "question_fields_on_base_contract",
        "request": _request(
            "fixed_choice",
            URGENT,
            options=["a", "b"],
            question_id="priority",
            schema_version="1.0.0",
        ),
        "error": "unsupported_contract",
    },
    {
        "id": "media_on_question_contract",
        "request": _request(
            "fixed_choice",
            URGENT,
            options=["a", "b"],
            schema_version="1.1.0",
            media=[
                {
                    "media_type": "image/png",
                    "sha256": PNG_2X2_SHA256,
                    "byte_size": 68,
                    "data": PNG_2X2,
                }
            ],
        ),
        "error": "unsupported_contract",
    },
    {
        "id": "media_with_unsupported_type",
        "request": _request(
            "fixed_choice",
            URGENT,
            options=["a", "b"],
            media=[
                {
                    "media_type": "image/gif",
                    "sha256": PNG_2X2_SHA256,
                    "byte_size": 68,
                    "data": PNG_2X2,
                }
            ],
        ),
        "error": "invalid_request",
    },
    {
        "id": "other_compatibility_policy",
        "request": _request(
            "fixed_choice", URGENT, options=["a", "b"], compatibility_policy="strict"
        ),
        "error": "unsupported_contract",
    },
    {
        "id": "question_set_with_top_level_options",
        "request": _request(
            "question_set",
            URGENT,
            options=["a", "b"],
            questions=[
                {"question_id": "q", "mode": "boolean_choice", "options": ["a", "b"]}
            ],
        ),
        "error": "invalid_request",
    },
    {
        "id": "question_set_duplicate_identifiers",
        "request": _request(
            "question_set",
            URGENT,
            questions=[
                {"question_id": "q", "mode": "boolean_choice", "options": ["a", "b"]},
                {"question_id": "q", "mode": "boolean_choice", "options": ["c", "d"]},
            ],
        ),
        "error": "invalid_request",
    },
    {
        "id": "nested_question_set",
        "request": _request(
            "question_set",
            URGENT,
            questions=[{"question_id": "q", "mode": "question_set"}],
        ),
        "error": "invalid_request",
    },
    {
        "id": "question_with_unknown_field",
        "request": _request(
            "question_set",
            URGENT,
            questions=[
                {
                    "question_id": "q",
                    "mode": "boolean_choice",
                    "options": ["a", "b"],
                    "weight": 1,
                }
            ],
        ),
        "error": "invalid_request",
    },
    {
        "id": "json_input_that_is_not_a_container",
        "request": _request(
            "boolean_choice", "42", options=["a", "b"], input_format="json"
        ),
        "error": "invalid_request",
    },
    {
        "id": "misaligned_option_descriptions",
        "request": _request(
            "fixed_choice",
            URGENT,
            options=["a", "b"],
            option_descriptions=["only one"],
        ),
        "error": "invalid_request",
    },
)


class _WordLevelTokenizer:
    """Tiny tokenizer: control markers, ASCII lowercase, then a word pattern."""

    def __init__(self, spec: dict[str, Any]) -> None:
        self._ids = {token: index for index, token in enumerate(spec["vocab"])}
        self._markers = tuple(spec["markers"])
        self._pattern = re.compile(spec["word_pattern"])
        self._split = re.compile(
            "(" + "|".join(re.escape(item) for item in self._markers) + ")"
        )
        self._unk = self._ids[spec["unk_token"]]
        self._cls = self._ids[spec["cls_token"]]
        self._sep = self._ids[spec["sep_token"]]

    def no_truncation(self) -> None:
        """The tokenizer never truncates."""

    def no_padding(self) -> None:
        """The tokenizer never pads."""

    def token_to_id(self, token: str) -> int | None:
        return self._ids.get(token)

    def encode(self, text: str) -> SimpleNamespace:
        ids = [self._cls]
        for piece in self._split.split(text):
            if piece in self._markers:
                ids.append(self._ids[piece])
                continue
            lowered = "".join(
                chr(ord(char) + 32) if "A" <= char <= "Z" else char for char in piece
            )
            for word in self._pattern.findall(lowered):
                ids.append(self._ids.get(word, self._unk))
        ids.append(self._sep)
        return SimpleNamespace(ids=ids, attention_mask=[1] * len(ids))


def tiny_encoder_logits(
    spec: dict[str, Any], input_ids: list[int], attention_mask: list[int]
) -> list[float]:
    """Return one logit per ``<<LABEL>>`` slot, in float64.

    Every hidden state is ``tanh(E[id] @ W + b)``. A label is the mean hidden
    state of its ``<<LABEL>>`` token and the words up to the next label or
    separator; the context is the mean of every other unmasked, non-ignored
    token. The logit is ``scale * dot(label, context) + bias``. Sums run left
    to right so each kit can reproduce the same arithmetic.
    """

    embeddings = spec["embeddings"]
    projection = spec["projection"]
    bias = spec["projection_bias"]
    size = spec["hidden_size"]
    label_id = spec["label_token_id"]
    separator_id = spec["separator_token_id"]
    ignored = set(spec["ignored_token_ids"])
    count = len(input_ids)
    hidden: list[list[float]] = []
    for token in input_ids:
        row = embeddings[token]
        values = []
        for column in range(size):
            total = 0.0
            for inner in range(size):
                total += row[inner] * projection[inner][column]
            values.append(math.tanh(total + bias[column]))
        hidden.append(values)

    spans: list[range] = []
    for position in range(count):
        if attention_mask[position] != 1 or input_ids[position] != label_id:
            continue
        end = position + 1
        while end < count and input_ids[end] not in (label_id, separator_id):
            end += 1
        spans.append(range(position, end))
    in_label = {position for span in spans for position in span}
    context = [
        position
        for position in range(count)
        if attention_mask[position] == 1
        and position not in in_label
        and input_ids[position] not in ignored
    ]

    def mean(positions: list[int]) -> list[float]:
        totals = [0.0] * size
        for position in positions:
            for column in range(size):
                totals[column] += hidden[position][column]
        if not positions:
            return totals
        return [value / len(positions) for value in totals]

    text = mean(context)
    logits = []
    for span in spans:
        label = mean([position for position in span if attention_mask[position] == 1])
        dot = 0.0
        for column in range(size):
            dot += label[column] * text[column]
        logits.append(spec["logit_scale"] * dot + spec["logit_bias"])
    return logits


class _TinyEncoderSession:
    """ONNX Runtime-shaped session that runs the tiny encoder in float32 out."""

    def __init__(self, spec: dict[str, Any], fault: str | None) -> None:
        self._spec = spec
        self._fault = fault
        self.calls: list[dict[str, Any]] = []

    def get_inputs(self) -> list[SimpleNamespace]:
        return [
            SimpleNamespace(name="input_ids"),
            SimpleNamespace(name="attention_mask"),
        ]

    def get_outputs(self) -> list[SimpleNamespace]:
        return [SimpleNamespace(name="logits")]

    def run(self, names: list[str], feeds: dict[str, Any]) -> list[Any]:
        import numpy as np

        input_ids = [int(item) for item in feeds["input_ids"][0]]
        attention_mask = [int(item) for item in feeds["attention_mask"][0]]
        call: dict[str, Any] = {"input_ids": input_ids, "logits": None}
        self.calls.append(call)
        if self._fault == "raise":
            raise RuntimeError("synthetic encoder fault")
        logits = tiny_encoder_logits(self._spec, input_ids, attention_mask)
        if self._fault == "short":
            logits = logits[:-1]
        if self._fault == "non_finite":
            logits[0] = math.nan
        output = np.asarray([logits], dtype=np.float32)
        if self._fault != "non_finite":
            call["logits"] = [float(value) for value in output[0]]
        return [output]


def _uniform(rng: random.Random) -> float:
    return round(rng.random() * 2.0 - 1.0, 6)


def _tokenizer_spec() -> dict[str, Any]:
    vocab = [*SPECIAL_TOKENS, *MARKERS, *sorted(set(WORDS)), *PUNCTUATION]
    if len(vocab) != len(set(vocab)):
        raise AssertionError("tiny vocabulary entries must be unique")
    return {
        "kind": "openmed.tests.word_level.v1",
        "vocab": vocab,
        "unk_token": "[UNK]",
        "cls_token": "[CLS]",
        "sep_token": "[SEP]",
        "markers": list(MARKERS),
        "lowercase": "ascii",
        "word_pattern": WORD_PATTERN,
    }


def _encoder_spec(vocab: list[str]) -> dict[str, Any]:
    rng = random.Random(SEED)
    ids = {token: index for index, token in enumerate(vocab)}
    return {
        "kind": "openmed.tests.tiny_gliclass_encoder.v1",
        "seed": SEED,
        "hidden_size": HIDDEN_SIZE,
        "embeddings": [[_uniform(rng) for _ in range(HIDDEN_SIZE)] for _ in vocab],
        "projection": [
            [_uniform(rng) for _ in range(HIDDEN_SIZE)] for _ in range(HIDDEN_SIZE)
        ],
        "projection_bias": [round(_uniform(rng) * 0.1, 6) for _ in range(HIDDEN_SIZE)],
        "logit_scale": LOGIT_SCALE,
        "logit_bias": LOGIT_BIAS,
        "label_token_id": ids["<<LABEL>>"],
        "separator_token_id": ids["<<SEP>>"],
        "ignored_token_ids": [
            ids[token] for token in (*SPECIAL_TOKENS, "<<SEP>>", "<<EXAMPLE>>")
        ],
        "input_names": ["input_ids", "attention_mask"],
        "output_name": "logits",
        "output_dtype": "float32",
    }


def _runtime(name: str) -> dict[str, Any]:
    settings = RUNTIMES[name]
    return {
        "prompt": settings.get("prompt"),
        "examples": settings.get("examples", []),
        "score_levels": settings.get(
            "score_levels", list(GLICLASS_DEFAULT_SCORE_LEVELS)
        ),
        "prompt_first": settings.get("prompt_first", True),
        "max_length": settings.get("max_length", 2048),
        "fault": settings.get("fault"),
    }


def _backend(
    runtime: dict[str, Any], tokenizer: dict[str, Any], encoder: dict[str, Any]
) -> tuple[GLiClassDecisionBackend, _TinyEncoderSession]:
    session = _TinyEncoderSession(encoder, runtime["fault"])
    runner = GLiClassOnnxRunner(
        session,
        _WordLevelTokenizer(tokenizer),
        prompt_first=runtime["prompt_first"],
        max_length=runtime["max_length"],
    )
    backend = GLiClassDecisionBackend(
        runner,
        prompt=runtime["prompt"],
        examples=runtime["examples"],
        score_levels=runtime["score_levels"],
        **BACKEND,
    )
    return backend, session


def _policy(payload: dict[str, Any] | None) -> DecisionAccessPolicy | None:
    if payload is None:
        return None
    return DecisionAccessPolicy(
        allowed_namespaces=frozenset(payload["allowed_namespaces"]),
        allowed_purposes=frozenset(payload["allowed_purposes"]),
        allowed_backend_kinds=frozenset(
            DecisionBackendKind(item) for item in payload["allowed_backend_kinds"]
        ),
        policy_version=payload["policy_version"],
    )


def _profiles(
    payload: list[dict[str, Any]] | None,
) -> dict[str, DecisionCalibrationProfile] | None:
    if payload is None:
        return None
    return {
        item["calibration_id"]: DecisionCalibrationProfile(**item) for item in payload
    }


def _request_error(payload: dict[str, Any]) -> str | None:
    try:
        DecisionRequest.from_dict(payload)
    except DecisionUnsupportedError:
        return "unsupported_contract"
    except (DecisionError, TypeError, ValueError):
        return "invalid_request"
    return None


def fixture() -> dict[str, Any]:
    """Build the shared parity fixture from the Python reference path."""

    tokenizer = _tokenizer_spec()
    encoder = _encoder_spec(tokenizer["vocab"])
    runtimes = {name: _runtime(name) for name in RUNTIMES}
    cases = []
    for case in CASES:
        runtime_name = case.get("runtime", "default")
        backend, session = _backend(runtimes[runtime_name], tokenizer, encoder)
        result = decide(
            case["request"],
            backend=backend,
            policy=_policy(case.get("policy")),
            calibration_profiles=_profiles(case.get("calibration_profiles")),
        ).to_dict()
        cases.append(
            {
                "id": case["id"],
                "runtime": runtime_name,
                "request": case["request"],
                "calibration_profiles": case.get("calibration_profiles"),
                "policy": case.get("policy"),
                "expected": {"encoder_calls": session.calls, "result": result},
            }
        )
    invalid = [
        {
            "id": case["id"],
            "request": case["request"],
            "error": _request_error(case["request"]),
        }
        for case in INVALID_REQUESTS
    ]
    return {
        "schema_version": FIXTURE_SCHEMA,
        "synthetic": True,
        "decision_schema_version": DECISION_SCHEMA_VERSION,
        "reference": (
            "openmed.structured.decision_gliclass.GLiClassOnnxRunner + "
            "openmed.structured.decision.decide"
        ),
        "tolerance": dict(TOLERANCE),
        "tokenizer": tokenizer,
        "encoder": encoder,
        "backend": {**BACKEND, "runtime": GLiClassOnnxRunner.runtime},
        "runtimes": runtimes,
        "cases": cases,
        "invalid_requests": invalid,
    }


def _committed() -> dict[str, Any]:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def _render(payload: dict[str, Any]) -> str:
    """Render two-space JSON with scalar lists kept on one line."""

    def render(value: Any, depth: int) -> str:
        pad = "  " * depth
        if isinstance(value, dict) and value:
            items = [
                f"{pad}  {json.dumps(key)}: {render(item, depth + 1)}"
                for key, item in value.items()
            ]
            return "{\n" + ",\n".join(items) + "\n" + pad + "}"
        if isinstance(value, list) and any(
            isinstance(item, (dict, list)) for item in value
        ):
            items = [f"{pad}  {render(item, depth + 1)}" for item in value]
            return "[\n" + ",\n".join(items) + "\n" + pad + "]"
        return json.dumps(value, ensure_ascii=True)

    return render(payload, 0) + "\n"


def _assert_same(actual: Any, expected: Any, path: str = "$") -> None:
    """Compare JSON trees exactly, allowing last-bit libm differences in floats."""

    if isinstance(expected, float) and not isinstance(actual, bool):
        assert isinstance(actual, (int, float)), path
        assert math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12), path
    elif isinstance(expected, dict):
        assert isinstance(actual, dict) and list(actual) == list(expected), path
        for key, value in expected.items():
            _assert_same(actual[key], value, f"{path}.{key}")
    elif isinstance(expected, list):
        assert isinstance(actual, list) and len(actual) == len(expected), path
        for index, (left, right) in enumerate(zip(actual, expected)):
            _assert_same(left, right, f"{path}[{index}]")
    else:
        assert type(actual) is type(expected) and actual == expected, path


def test_committed_vectors_match_the_python_reference() -> None:
    pytest.importorskip("numpy")
    rebuilt = fixture()
    _assert_same(rebuilt, _committed())
    # The committed text is the canonical rendering of the committed vectors.
    assert FIXTURE.read_text(encoding="utf-8") == _render(_committed())


def test_every_expected_result_validates_against_the_current_schema() -> None:
    result_schema = Draft202012Validator(decision_result_schema())
    request_schema = Draft202012Validator(decision_request_schema())
    corpus = _committed()

    assert corpus["decision_schema_version"] == DECISION_SCHEMA_VERSION == "1.2.0"
    for case in corpus["cases"]:
        declared = case["request"].get("schema_version", "1.0.0")
        if declared in SUPPORTED_DECISION_SCHEMA_VERSIONS:
            request_schema.validate(case["request"])
        else:
            # A newer minor of the same major is served under the newest
            # supported contract, exactly as the Python runtime does.
            assert not request_schema.is_valid(case["request"])
            assert declared.split(".")[0] == "1"
        result = case["expected"]["result"]
        result_schema.validate(result)
        assert result["autonomous_action"] is False
        assert result["review"]["required"] is True
        assert result["backend"]["kind"] == "encoder"
        assert result["backend"]["local"] is True


def test_vectors_cover_every_state_mode_and_refusal() -> None:
    corpus = _committed()
    results = {case["id"]: case["expected"]["result"] for case in corpus["cases"]}
    expected = {case["id"]: case for case in CASES}

    for case_id, result in results.items():
        assert result["state"] == expected[case_id]["state"], case_id
        if "code" in expected[case_id]:
            assert result["code"] == expected[case_id]["code"], case_id
    assert {result["state"] for result in results.values()} == {
        "success",
        "abstained",
        "partial",
        "unsupported",
        "denied",
        "failure",
        "conflict",
    }
    assert {result["mode"] for result in results.values()} == {
        "fixed_choice",
        "boolean_choice",
        "ordered_preference",
        "multi_label",
        "scalar_score",
        "question_set",
    }
    assert {result["schema_version"] for result in results.values()} == {
        "1.0.0",
        "1.1.0",
        "1.2.0",
    }
    codes = {result["code"] for result in results.values()}
    assert {
        "low_confidence",
        "ambiguous_scores",
        "no_label_above_threshold",
        "calibration_unavailable",
        "namespace_denied",
        "backend_denied",
        "backend_unsupported",
        "backend_failure",
        "backend_conflict",
        "question_set_partial",
    } <= codes
    assert results["media_is_unsupported_on_device"]["code"] == "backend_unsupported"
    assert corpus["cases"][0]["calibration_profiles"] is None
    for case in corpus["invalid_requests"]:
        assert case["error"] == next(
            item["error"] for item in INVALID_REQUESTS if item["id"] == case["id"]
        ), case["id"]


def test_selections_clear_every_threshold_with_margin() -> None:
    corpus = _committed()
    for case in corpus["cases"]:
        result = case["expected"]["result"]
        profile = result["calibration"]
        for answer in [result, *result.get("questions", [])]:
            if answer["mode"] == "question_set" or answer["state"] not in {
                "success",
                "abstained",
            }:
                continue
            confidence = answer["confidence"]
            assert abs(confidence - profile["minimum_confidence"]) > (
                THRESHOLD_CLEARANCE
            ), case["id"]
            if answer["margin"] is not None and answer["mode"] != "multi_label":
                assert abs(answer["margin"] - profile["minimum_margin"]) > (
                    THRESHOLD_CLEARANCE
                ), case["id"]
            scores = [item["score"] for item in answer["option_scores"]]
            if answer["mode"] == "multi_label":
                for score in scores:
                    assert abs(score - profile["multi_label_threshold"]) > (
                        THRESHOLD_CLEARANCE
                    ), case["id"]
            ordered = sorted(scores, reverse=True)
            for first, second in zip(ordered, ordered[1:]):
                assert first - second > THRESHOLD_CLEARANCE, case["id"]


def test_vectors_carry_no_source_text_outside_requests() -> None:
    corpus = _committed()
    for case in corpus["cases"]:
        text = case["request"]["input_text"]
        rendered = json.dumps(case["expected"]["result"])
        assert text not in rendered
        assert json.dumps(text)[1:-1] not in rendered
        if "instructions" in case["request"]:
            assert case["request"]["instructions"] not in rendered
    assert corpus["synthetic"] is True
    assert corpus["backend"]["license_id"] == "Apache-2.0"
    assert DEFAULT_CALIBRATION_ID in json.dumps(corpus)


def test_tiny_encoder_scores_the_label_slots_the_runner_counts() -> None:
    pytest.importorskip("numpy")
    corpus = _committed()
    label_id = corpus["encoder"]["label_token_id"]
    for case in corpus["cases"]:
        for call in case["expected"]["encoder_calls"]:
            if call["logits"] is None:
                continue
            slots = call["input_ids"].count(label_id)
            fault = corpus["runtimes"][case["runtime"]]["fault"]
            assert len(call["logits"]) == (slots - 1 if fault == "short" else slots)


def test_every_kit_consumes_the_shared_vectors() -> None:
    for relative in KIT_CONSUMERS:
        source = (ROOT / relative).read_text(encoding="utf-8")
        assert FIXTURE.name in source, relative


if __name__ == "__main__":  # pragma: no cover - fixture maintenance entry point
    if sys.argv[1:] != ["--write"]:
        raise SystemExit("usage: python -m " + __spec__.name + " --write")
    FIXTURE.write_text(_render(fixture()), encoding="utf-8")
