"""Synthetic HL7 v2 message generation, truth manifest and leakage harness."""

from __future__ import annotations

import hashlib
import json
import re

import pytest

from openmed.core.labels import CANONICAL_LABELS
from openmed.interop.hl7v2 import (
    DEFAULT_FIELD_MAP,
    HL7FieldRule,
    HL7V2Encoding,
    parse_hl7v2,
)
from openmed.training.synthetic.hl7v2_messages import (
    HL7V2_SURROGATE_PROVIDERS,
    HL7V2_SYNTHETIC_MESSAGE_TYPES,
    HL7V2_SYNTHETIC_SCHEMA_VERSION,
    MAX_HL7V2_MESSAGES,
    MAX_HL7V2_REPETITIONS,
    HL7V2LeakageError,
    HL7V2LeakageReport,
    HL7V2MessageGenerator,
    HL7V2SynthesisConfig,
    HL7V2SynthesisError,
    evaluate_hl7v2_leakage,
    generate_hl7v2_messages,
)

ENCODINGS = (
    HL7V2Encoding(),
    HL7V2Encoding(
        field="#", component="$", repetition="*", escape="?", subcomponent="%"
    ),
    HL7V2Encoding(
        field="!", component="@", repetition=";", escape="/", subcomponent=":"
    ),
)

# Positions master redacted only after #3884 was filed. They stay generated so
# the harness can prove it would expose a regression.
ISSUE_POSITIONS = ("PID-6", "PID-9", "PID-20", "PID-21", "MRG-1", "MRG-3", "MRG-7")

# PHI positions the default map does not cover on master. Fixing one shrinks
# the observed set, which stays a subset, so improvements never fail.
KNOWN_UNMAPPED_RESIDUALS = frozenset(
    {
        "NK1-6",
        "OBR-7",
        "OBR-16",
        "OBX-5",
        "OBX-14",
        "ORC-12",
        "PID-4",
        "PID-18",
        "PV1-7",
        "PV1-44",
        "ZPD-2",
        "ZPD-3",
        "ZPD-4",
    }
)


def mask_all(text, **_kwargs):
    return "[MASKED]"


def keep_all(text, **_kwargs):
    return text


def leaf_at(message, truth):
    parsed = parse_hl7v2(message.text)
    encoding = parsed.encoding
    value = parsed.segments[truth.segment_index].get_field(truth.field)
    repetition = value.split(encoding.repetition)[truth.repetition - 1]
    component = repetition.split(encoding.component)[truth.component - 1]
    return component.split(encoding.subcomponent)[truth.subcomponent - 1]


def batch(count=12, *, seed=7, **settings):
    return generate_hl7v2_messages(
        count, seed=seed, config=HL7V2SynthesisConfig(**settings)
    )


@pytest.mark.parametrize("encoding", ENCODINGS, ids=["default", "alt-1", "alt-2"])
def test_messages_parse_and_every_truth_offset_slices_its_value(encoding):
    messages = batch(18, encoding=encoding, escape_probability=0.5)

    assert {message.message_type for message in messages} == set(
        HL7V2_SYNTHETIC_MESSAGE_TYPES
    )
    for message in messages:
        parsed = parse_hl7v2(message.text)
        assert parsed.encoding == encoding
        assert parsed.serialize() == message.text
        assert parsed.segment_names()[0] == "MSH"
        assert message.truth
        for truth in message.truth:
            assert message.text[truth.start : truth.end] == truth.raw
            assert parsed.segments[truth.segment_index].name == truth.segment
            leaf = leaf_at(message, truth)
            if truth.whole_leaf:
                assert leaf == truth.raw
            else:
                assert truth.raw in leaf


def test_escaped_and_plain_truth_values_use_raw_offsets():
    encoding = ENCODINGS[1]
    messages = batch(12, encoding=encoding, escape_probability=1.0)
    truths = [truth for message in messages for truth in message.truth]

    escaped = [truth for truth in truths if truth.raw != truth.value]
    plain = [truth for truth in truths if truth.raw == truth.value]
    assert escaped and plain
    for truth in escaped:
        assert encoding.escape in truth.raw
        assert any(
            char in truth.value
            for char in (
                encoding.field,
                encoding.component,
                encoding.repetition,
                encoding.escape,
                encoding.subcomponent,
            )
        )
        assert truth.to_dict()["raw"] == truth.raw
    for message in messages:
        for truth in message.truth:
            assert message.text[truth.start : truth.end] == truth.raw


def test_repetitions_components_and_subcomponents_are_one_based():
    messages = batch(12, max_repetitions=3)
    truths = [truth for message in messages for truth in message.truth]

    assert min(truth.repetition for truth in truths) == 1
    assert max(truth.repetition for truth in truths) == 3
    assert any(truth.subcomponent > 1 for truth in truths)
    assert any(truth.component > 1 for truth in truths)
    assert all(
        truth.repetition >= 1 and truth.component >= 1 and truth.subcomponent >= 1
        for truth in truths
    )
    assert all(truth.repetition <= 3 for truth in truths)


def test_timestamps_vary_precision_and_offset():
    messages = batch(24, timezone_offsets=("", "-0500", "+0530"))
    dates = [
        truth.value
        for message in messages
        for truth in message.truth
        if truth.label in {"DATE", "DATE_OF_BIRTH"} and truth.whole_leaf
    ]
    stems = {value.split("+")[0].split("-")[0] for value in dates}

    assert {4, 6, 8, 12, 14} <= {len(stem) for stem in stems}
    assert any("." in stem for stem in stems)
    assert any(value.endswith(("-0500", "+0530")) for value in dates)
    assert any(len(value) == len(value.split("+")[0].split("-")[0]) for value in dates)


def test_free_text_z_segments_and_phi_outside_default_map():
    messages = batch(18)
    names = {
        name
        for message in messages
        for name in parse_hl7v2(message.text).segment_names()
    }
    truths = [truth for message in messages for truth in message.truth]

    assert {"OBX", "NTE", "ZPD", "MRG", "PV1", "ORC", "OBR"} <= names
    assert any(not truth.whole_leaf for truth in truths if truth.segment == "OBX")
    assert any(not truth.whole_leaf for truth in truths if truth.segment == "NTE")
    outside = {f"{t.segment}-{t.field}" for t in truths if not t.in_default_map}
    inside = {f"{t.segment}-{t.field}" for t in truths if t.in_default_map}
    assert {"ZPD-2", "PID-18", "PV1-44"} <= outside
    assert {"PID-5", "PID-6", "PID-9", "PID-20", "PID-21", "MRG-1"} <= inside
    for truth in truths:
        assert truth.in_default_map == _applies(truth, messages)


def _applies(truth, messages):
    rule = DEFAULT_FIELD_MAP.get((truth.segment, truth.field))
    if rule is None:
        return False
    if not rule.allowed_type_values or rule.type_field is None:
        return True
    for message in messages:
        if truth in message.truth:
            segment = parse_hl7v2(message.text).segments[truth.segment_index]
            value = segment.get_field(rule.type_field) or ""
            return value.split(segment.encoding.component)[0] in (
                rule.allowed_type_values
            )
    raise AssertionError("truth value not found")


def test_identical_seeds_produce_identical_bytes_and_manifests():
    first = batch(10, seed=41, escape_probability=0.5)
    second = batch(10, seed=41, escape_probability=0.5)
    other = batch(10, seed=42, escape_probability=0.5)

    def digest(messages):
        hasher = hashlib.sha256()
        for message in messages:
            hasher.update(message.to_bytes())
            hasher.update(
                json.dumps(message.manifest(), sort_keys=True).encode("utf-8")
            )
        return hasher.hexdigest()

    assert [m.to_bytes() for m in first] == [m.to_bytes() for m in second]
    assert [m.manifest() for m in first] == [m.manifest() for m in second]
    assert [m.manifest_sha256 for m in first] == [m.manifest_sha256 for m in second]
    assert digest(first) == digest(second)
    assert digest(first) != digest(other)
    assert first[0].text_sha256 == hashlib.sha256(first[0].to_bytes()).hexdigest()


def test_generator_and_batch_function_agree():
    config = HL7V2SynthesisConfig(message_types=("ORM",))
    generator = HL7V2MessageGenerator(config, seed=3)

    generated = (generator.generate(), generator.generate())
    assert generated == generate_hl7v2_messages(2, seed=3, config=config)
    assert generator.generate("ADT").message_type == "ADT"


def test_manifest_is_versioned_synthetic_and_values_come_from_providers():
    messages = batch(15, escape_probability=0.5)
    for message in messages:
        manifest = message.manifest()
        assert manifest["schema_version"] == HL7V2_SYNTHETIC_SCHEMA_VERSION
        assert manifest["synthetic"] is True
        assert manifest["contains_real_phi"] is False
        assert manifest["text_sha256"] == message.text_sha256
        assert manifest["encoding"] == message.encoding.field + (
            message.encoding.as_msh_2()
        )
        assert len(manifest["truth"]) == len(message.truth)
        for truth in message.truth:
            assert truth.label in CANONICAL_LABELS
            assert truth.provider in HL7V2_SURROGATE_PROVIDERS
            assert truth.value


def _config_error(**settings):
    with pytest.raises(HL7V2SynthesisError) as excinfo:
        HL7V2SynthesisConfig(**settings)
    return excinfo.value


@pytest.mark.parametrize(
    ("settings", "code"),
    [
        ({"message_types": ("ADT", "SIU")}, "unknown_message_type"),
        ({"message_types": ()}, "unknown_message_type"),
        ({"adt_events": ("A99",)}, "unknown_trigger_event"),
        ({"max_repetitions": 0}, "repetitions_out_of_bounds"),
        ({"max_repetitions": MAX_HL7V2_REPETITIONS + 1}, "repetitions_out_of_bounds"),
        ({"max_repetitions": float("inf")}, "repetitions_out_of_bounds"),
        ({"max_repetitions": True}, "repetitions_out_of_bounds"),
        ({"max_obx_segments": 10_000}, "segments_out_of_bounds"),
        ({"max_nte_segments": -1}, "segments_out_of_bounds"),
        ({"escape_probability": 1.5}, "invalid_probability"),
        ({"timestamp_precisions": ("hour",)}, "invalid_timestamp_precision"),
        ({"timezone_offsets": ("+2500",)}, "invalid_timezone_offset"),
        (
            {"encoding": HL7V2Encoding(component="|")},
            "invalid_encoding",
        ),
        (
            {"encoding": HL7V2Encoding(subcomponent="A")},
            "invalid_encoding",
        ),
        (
            {"encoding": HL7V2Encoding(repetition="\r")},
            "invalid_encoding",
        ),
    ],
)
def test_invalid_settings_fail_with_typed_errors(settings, code):
    error = _config_error(**settings)

    assert isinstance(error, ValueError)
    assert error.code == code


@pytest.mark.parametrize("count", [0, -1, MAX_HL7V2_MESSAGES + 1, 2.5, True])
def test_unbounded_message_counts_fail_with_typed_errors(count):
    with pytest.raises(HL7V2SynthesisError) as excinfo:
        generate_hl7v2_messages(count)
    assert excinfo.value.code == "count_out_of_bounds"


def test_invalid_seed_and_message_type_fail_with_typed_errors():
    with pytest.raises(HL7V2SynthesisError) as excinfo:
        HL7V2MessageGenerator(seed="seven")
    assert excinfo.value.code == "invalid_seed"

    with pytest.raises(HL7V2SynthesisError) as excinfo:
        HL7V2MessageGenerator(seed=1).generate("MDM")
    assert excinfo.value.code == "unknown_message_type"
    assert "MDM" not in str(excinfo.value)


def _all_truth_values(messages):
    return {truth.value for message in messages for truth in message.truth}


def test_harness_reports_residual_positions_without_echoing_values():
    messages = batch(9, escape_probability=0.5)
    noop_map = {("ZZZ", 1): HL7FieldRule("clear")}

    report = evaluate_hl7v2_leakage(
        messages, field_map=noop_map, deidentifier=keep_all, date_shift_days=11
    )

    assert isinstance(report, HL7V2LeakageReport)
    assert report.message_count == 9
    assert report.truth_count == sum(len(message.truth) for message in messages)
    assert report.residual_count == report.truth_count
    assert {position.label for position in report.positions} <= CANONICAL_LABELS
    rendered = (report.to_json(), repr(report), str(report), str(report.to_dict()))
    for value in _all_truth_values(messages):
        if len(value) >= 5:
            assert all(value not in text for text in rendered)
    payload = json.loads(report.to_json())
    assert set(payload["positions"][0]) == {
        "segment",
        "field",
        "repetition",
        "component",
        "subcomponent",
        "label",
        "in_default_map",
        "truth_count",
        "residual_count",
    }


def test_harness_errors_do_not_echo_values():
    messages = batch(2)
    sentinel = "SentinelLeakValue"

    def explode(text, **_kwargs):
        raise RuntimeError(f"{sentinel} {text}")

    with pytest.raises(HL7V2LeakageError) as excinfo:
        evaluate_hl7v2_leakage(messages, deidentifier=explode, date_shift_days=3)

    error = excinfo.value
    assert error.code == "redaction_failed"
    assert error.message_index in {0, 1}
    for text in (str(error), repr(error), str(error.args)):
        assert sentinel not in text
        for value in _all_truth_values(messages):
            if len(value) >= 5:
                assert value not in text


def test_harness_rejects_non_generated_inputs():
    with pytest.raises(HL7V2LeakageError) as excinfo:
        evaluate_hl7v2_leakage(["MSH|^~\\&|A"])
    assert excinfo.value.code == "invalid_message"


def test_harness_detects_residuals_from_weakened_field_map():
    messages = batch(12, adt_events=("A40",), message_types=("ADT",))
    weakened = {
        key: rule
        for key, rule in DEFAULT_FIELD_MAP.items()
        if f"{key[0]}-{key[1]}" not in ISSUE_POSITIONS
    }

    weak = evaluate_hl7v2_leakage(
        messages, field_map=weakened, deidentifier=mask_all, date_shift_days=9
    )
    full = evaluate_hl7v2_leakage(messages, deidentifier=mask_all, date_shift_days=9)

    assert set(ISSUE_POSITIONS) <= set(weak.residual_fields)
    assert not set(ISSUE_POSITIONS) & set(full.residual_fields)
    assert weak.residual_count > full.residual_count


def test_default_map_residuals_on_master_are_known_findings():
    messages = batch(24, escape_probability=0.5)

    report = evaluate_hl7v2_leakage(messages, deidentifier=mask_all, date_shift_days=29)

    unmapped = {p.field_key for p in report.residual_positions if not p.in_default_map}
    mapped = [p for p in report.residual_positions if p.in_default_map]
    assert unmapped <= KNOWN_UNMAPPED_RESIDUALS
    # Mapped fields leak only when a surrogate happens to equal its source,
    # never through hash, date-shift or free-text rules.
    for position in mapped:
        rule = DEFAULT_FIELD_MAP[(position.segment, position.field)]
        assert rule.action == "surrogate"
        assert position.residual_count < position.truth_count
    assert sum(p.residual_count for p in mapped) <= report.truth_count // 50


def test_free_text_residuals_follow_the_injected_deidentifier():
    messages = batch(9, message_types=("ORU",), max_obx_segments=4)

    def glue_suffix(text, **_kwargs):
        return re.sub(r"(\w+)", r"\1zz", text)

    kept = evaluate_hl7v2_leakage(messages, deidentifier=keep_all, date_shift_days=5)
    glued = evaluate_hl7v2_leakage(
        messages, deidentifier=glue_suffix, date_shift_days=5
    )
    masked = evaluate_hl7v2_leakage(messages, deidentifier=mask_all, date_shift_days=5)

    def free_text(report):
        return {
            p.field_key
            for p in report.residual_positions
            if p.in_default_map and p.field_key in {"OBX-5", "NTE-3"}
        }

    assert free_text(kept) == {"OBX-5", "NTE-3"}
    # Values glued to other characters are no longer whole tokens.
    assert not free_text(glued)
    assert not free_text(masked)


@pytest.mark.parametrize("offset", ("+1401", "-1459", "+0000\n"))
def test_timezone_offsets_reject_out_of_range_or_trailing_data(offset):
    assert _config_error(timezone_offsets=(offset,)).code == "invalid_timezone_offset"


def test_harness_bounds_input_consumption_before_materializing_batch():
    def oversized():
        for _ in range(MAX_HL7V2_MESSAGES + 1):
            yield None
        raise AssertionError("consumed beyond the batch limit")

    with pytest.raises(HL7V2LeakageError, match="count_out_of_bounds"):
        evaluate_hl7v2_leakage(oversized())
