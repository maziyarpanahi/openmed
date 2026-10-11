"""Synthetic parity, privacy and fail-closed tests for clinical CLI commands."""

from __future__ import annotations

import json
import math
import os
from copy import deepcopy
from datetime import date
from pathlib import Path
from unittest.mock import Mock

import pytest

from openmed.cli import clinical
from openmed.cli.main import main
from openmed.clinical.context import assert_context
from openmed.clinical.relations import extract_relations
from openmed.clinical.sdoh import extract_sdoh
from openmed.clinical.temporal_normalizer import normalize_temporal
from openmed.clinical.timeline import build_timeline
from openmed.processing.outputs import EntityPrediction, PredictionResult


def _entity(text, surface, label, **extra):
    start = text.index(surface)
    return {
        "text": surface,
        "label": label,
        "start": start,
        "end": start + len(surface),
        "confidence": 0.9,
        **extra,
    }


def _files(tmp_path, text, entities):
    note = tmp_path / "PRIVATE_NOTE.txt"
    spans = tmp_path / "PRIVATE_SPANS.json"
    note.write_text(text, encoding="utf-8")
    payload = {
        "ok": True,
        "command": "analyze",
        "data": {"text": text, "entities": entities},
    }
    spans.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return note, spans, payload


def _run(capsys, name, note, spans, *options):
    code = main(
        [
            "clinical",
            name,
            "--note",
            str(note),
            "--spans",
            str(spans),
            "--json",
            *options,
        ]
    )
    captured = capsys.readouterr()
    assert captured.err == ""
    assert str(note) not in captured.out
    assert str(spans) not in captured.out
    return code, json.loads(captured.out)


def _api_spans(entities):
    return [
        {
            "start": e["start"],
            "end": e["end"],
            "text": e["text"],
            "label": e["label"],
            "score": e["confidence"],
        }
        for e in entities
    ]


def test_sdoh_matches_python_without_trigger_occupation_or_extent(tmp_path, capsys):
    text = "Synthetic Person is employed as a teacher and lives alone. Smokes 2 packs daily."
    entities = [_entity(text, "Synthetic Person", "PERSON")]
    note, spans, _ = _files(tmp_path, text, entities)
    expected = extract_sdoh(text, _api_spans(entities))
    code, envelope = _run(capsys, "sdoh", note, spans)
    assert code == 0 and envelope["command"] == "clinical sdoh"
    records = envelope["data"]["findings"]
    assert [
        (r["span"], r["category"], r["status"], r["temporality"], r["score"])
        for r in records
    ] == [
        (list(f.span), f.category, f.status, f.temporality, f.score) for f in expected
    ]
    serialized = json.dumps(envelope)
    for protected in (
        text,
        "Synthetic Person",
        "teacher",
        "2 packs",
        "Smokes",
        "lives alone",
    ):
        assert protected not in serialized
    assert all(r["review_required"] for r in records)
    assert len(envelope["data"]["source_digest"]) == 64


def test_sdoh_sections_use_same_python_scope(tmp_path, capsys):
    text = "Social history: unemployed. Assessment: unemployed."
    note, spans, _ = _files(tmp_path, text, [])
    boundary = text.index("Assessment")
    sections = [
        {"start": 0, "end": boundary, "label": "social_history"},
        {"start": boundary, "end": len(text), "label": "assessment"},
    ]
    section_file = tmp_path / "sections.json"
    section_file.write_text(json.dumps(sections))
    code, envelope = _run(capsys, "sdoh", note, spans, "--sections", str(section_file))
    expected = extract_sdoh(text, [], sections=sections)
    assert code == 0
    assert [r["span"] for r in envelope["data"]["findings"]] == [
        list(f.span) for f in expected
    ]
    assert len(expected) == 1


@pytest.mark.parametrize(
    "language,text,mentions",
    [
        (
            "en",
            "Metformin 500 mg orally twice daily.",
            [
                ("Metformin", "MEDICATION"),
                ("500 mg", "DOSAGE"),
                ("twice daily", "FREQUENCY"),
            ],
        ),
        (
            "zh",
            "肺炎使用阿莫西林治疗。",
            [("肺炎", "CONDITION"), ("阿莫西林", "MEDICATION")],
        ),
        (
            "hi",
            "निमोनिया का इलाज एमोक्सिसिलिन से किया गया।",
            [("निमोनिया", "CONDITION"), ("एमोक्सिसिलिन", "MEDICATION")],
        ),
    ],
)
def test_relations_match_native_python_labels_and_offsets(
    tmp_path, capsys, language, text, mentions
):
    entities = [_entity(text, surface, label) for surface, label in mentions]
    note, spans, _ = _files(tmp_path, text, entities)
    expected = extract_relations(
        text, _api_spans(entities), language=None if language == "en" else language
    )
    code, envelope = _run(capsys, "relations", note, spans, "--language", language)
    assert code == 0
    records = envelope["data"]["relations"]
    assert expected
    assert [
        (
            r["head"]["start"],
            r["head"]["end"],
            r["head"]["label"],
            r["tail"]["start"],
            r["tail"]["end"],
            r["tail"]["label"],
            r["relation_type"],
            r["score"],
        )
        for r in records
    ] == [
        (
            r.head.start,
            r.head.end,
            r.head.label,
            r.tail.start,
            r.tail.end,
            r.tail.label,
            r.relation_type,
            r.score,
        )
        for r in expected
    ]
    output = json.dumps(envelope, ensure_ascii=False)
    for surface, _ in mentions:
        assert surface not in output


@pytest.mark.parametrize("reference", [None, "2026-06-15", "2026-06-15T12:30:00Z"])
def test_timeline_matches_bucket_api_with_explicit_date_link(
    tmp_path, capsys, reference
):
    text = "History of asthma. Fever began 3 days ago. Plan for surgery."
    entities = [
        _entity(text, "asthma", "CONDITION"),
        _entity(text, "Fever", "SYMPTOM"),
        _entity(text, "3 days ago", "DATE"),
        _entity(text, "surgery", "PROCEDURE"),
    ]
    time_span = [entities[2]["start"], entities[2]["end"]]
    entities[1]["metadata"] = {"clinical_time_span": time_span}
    note, spans, _ = _files(tmp_path, text, entities)
    options = () if reference is None else ("--reference-time", reference)
    code, envelope = _run(capsys, "timeline", note, spans, *options)
    assert code == 0
    result = envelope["data"]
    tagged = assert_context(text, _api_spans(entities), language="en")
    times = normalize_temporal(
        text, [entities[2]], None if reference is None else date(2026, 6, 15)
    )
    if reference:
        for index in (1, 2):
            tagged[index]["normalized_time"] = times[0].value
    for span in tagged:
        span["certainty"] = span["uncertainty"]
    expected = build_timeline(tagged)
    labels = {(e["start"], e["end"]): e["label"] for e in entities}
    assert [
        (r["start"], r["end"], r["label"], r["assertion"]) for r in result["events"]
    ] == [
        (e.start, e.end, labels[(e.start, e.end)], e.assertion.to_dict())
        for e in expected.events
    ]
    assert result["lanes"] == expected.to_dict()["lanes"]
    assert result["temporal"][0]["state"] == (
        "unanchored" if reference is None else "normalized"
    )
    fever = next(r for r in result["events"] if r["start"] == entities[1]["start"])
    assert fever["time_state"] == ("unanchored" if reference is None else "anchored")
    output = json.dumps(envelope)
    for protected in (
        text,
        "asthma",
        "Fever",
        "surgery",
        "3 days ago",
        "2026-06-12",
        "2026-06-15",
    ):
        assert protected not in output


def test_timeline_does_not_infer_nearby_event_date(tmp_path, capsys):
    text = "Fever started yesterday."
    entities = [_entity(text, "Fever", "SYMPTOM"), _entity(text, "yesterday", "DATE")]
    note, spans, _ = _files(tmp_path, text, entities)
    code, envelope = _run(
        capsys, "timeline", note, spans, "--reference-time", "2026-06-15"
    )
    assert code == 0
    assert (
        next(r for r in envelope["data"]["events"] if r["label"] == "SYMPTOM")[
            "time_state"
        ]
        == "unanchored"
    )
    assert (
        next(r for r in envelope["data"]["events"] if r["label"] == "DATE")[
            "time_state"
        ]
        == "anchored"
    )


def test_timeline_preserves_supplied_context_and_coincident_labels(tmp_path, capsys):
    text = "Fever."
    entities = [
        _entity(
            text,
            "Fever",
            "SYMPTOM",
            metadata={
                "clinical_context": {
                    "temporality": "hypothetical",
                    "uncertainty": "uncertain",
                    "negation": "negated",
                    "experiencer": "family",
                }
            },
        ),
        _entity(text, "Fever", "CONDITION"),
    ]
    note, spans, _ = _files(tmp_path, text, entities)
    code, envelope = _run(capsys, "timeline", note, spans)
    assert code == 0
    records = envelope["data"]["events"]
    assert [r["label"] for r in records] == ["CONDITION", "SYMPTOM"]
    assert records[1]["assertion"] == {
        "temporality": "hypothetical",
        "certainty": "uncertain",
        "negation": "negated",
        "experiencer": "family",
    }


def test_german_temporal_rules_are_native_and_unanchored(tmp_path, capsys):
    text = "Fieber seit drei Tagen."
    entities = [
        _entity(text, "Fieber", "SYMPTOM"),
        _entity(text, "seit drei Tagen", "DATE"),
    ]
    note, spans, _ = _files(tmp_path, text, entities)
    code, envelope = _run(capsys, "timeline", note, spans, "--language", "de")
    assert code == 0
    assert envelope["data"]["temporal"][0]["state"] == "unanchored"


def test_actual_analyze_formatter_result_is_accepted(tmp_path, capsys):
    text = "No tobacco use."
    prediction = PredictionResult(
        text=text,
        entities=[
            EntityPrediction(
                text="tobacco", label="OTHER", confidence=0.75, start=3, end=10
            )
        ],
        model_name="PRIVATE_MODEL",
        timestamp="2026-06-15T00:00:00Z",
        metadata={"secret": "PRIVATE_METADATA"},
    )
    note, spans, _ = _files(tmp_path, text, [])
    spans.write_text(
        json.dumps({"ok": True, "command": "analyze", "data": prediction.to_dict()})
    )
    code, envelope = _run(capsys, "sdoh", note, spans)
    assert code == 0
    assert envelope["data"]["input_span_count"] == 1
    assert "PRIVATE_" not in json.dumps(envelope)


@pytest.mark.parametrize("name", ["sdoh", "relations", "timeline"])
@pytest.mark.parametrize(
    "change,expected",
    [
        (lambda p: p.update(ok=False), "clinical_spans_invalid"),
        (lambda p: p.update(command="PRIVATE_COMMAND"), "clinical_spans_invalid"),
        (
            lambda p: p["data"].update(text="PRIVATE_OTHER_NOTE"),
            "clinical_source_mismatch",
        ),
        (lambda p: p["data"].update(entities={}), "clinical_spans_invalid"),
        (lambda p: p["data"]["entities"][0].update(start=-1), "clinical_span_invalid"),
        (lambda p: p["data"]["entities"][0].update(end=9999), "clinical_span_invalid"),
        (
            lambda p: p["data"]["entities"][0].update(start=True),
            "clinical_span_invalid",
        ),
        (lambda p: p["data"]["entities"][0].update(end="5"), "clinical_span_invalid"),
        (
            lambda p: p["data"]["entities"][0].update(text="PRIVATE_SURFACE"),
            "clinical_source_mismatch",
        ),
        (
            lambda p: p["data"]["entities"][0].update(label="PRIVATE_LABEL"),
            "clinical_label_unsupported",
        ),
        (
            lambda p: p["data"]["entities"][0].update(label="PERSON张三"),
            "clinical_label_unsupported",
        ),
        (
            lambda p: p["data"]["entities"][0].update(label="P/E/R/S/O/N"),
            "clinical_label_unsupported",
        ),
        (
            lambda p: p["data"]["entities"][0].update(confidence=True),
            "clinical_span_invalid",
        ),
        (
            lambda p: p["data"]["entities"][0].update(confidence=2),
            "clinical_span_invalid",
        ),
        (
            lambda p: p["data"]["entities"][0].update(confidence=10**400),
            "clinical_span_invalid",
        ),
        (lambda p: p["data"]["entities"][0].update(score=0.3), "clinical_span_invalid"),
        (
            lambda p: p["data"]["entities"][0].update(
                metadata={"clinical_context": {"temporality": "PRIVATE_AXIS"}}
            ),
            "clinical_span_invalid",
        ),
        (
            lambda p: p["data"]["entities"][0].update(
                metadata={"clinical_time_span": [0, 5]}
            ),
            "clinical_time_link_invalid",
        ),
        (
            lambda p: p["data"].update(entities=p["data"]["entities"] * 513),
            "clinical_spans_too_many",
        ),
    ],
)
def test_malformed_spans_fail_before_any_processing(
    tmp_path, capsys, monkeypatch, name, change, expected
):
    note, spans, payload = _files(
        tmp_path, "Fever.", [_entity("Fever.", "Fever", "SYMPTOM")]
    )
    change(payload)
    spans.write_text(json.dumps(payload))
    processing = Mock(side_effect=AssertionError("must not process"))
    monkeypatch.setattr(clinical, "_sdoh", processing)
    monkeypatch.setattr(clinical, "_relations", processing)
    monkeypatch.setattr(clinical, "_timeline", processing)
    code, envelope = _run(capsys, name, note, spans)
    assert code == 2 and envelope["error"]["code"] == expected
    assert "PRIVATE_" not in json.dumps(envelope)
    processing.assert_not_called()


@pytest.mark.parametrize(
    "raw",
    [
        b'{"entities":[],"entities":[]}',
        b'{"entities":[],"x":NaN}',
        b'{"entities":[],"x":Infinity}',
        b'{"entities":[],"x":1e999}',
        b"{",
        b"[" * 1000 + b"]" * 1000,
        b"\xff",
        b'{"entities":[],"x":' + b"[" * 40 + b"0" + b"]" * 40 + b"}",
    ],
)
def test_invalid_json_and_encoding_have_controlled_codes(tmp_path, capsys, raw):
    note, spans, _ = _files(tmp_path, "", [])
    spans.write_bytes(raw)
    code, envelope = _run(capsys, "sdoh", note, spans)
    assert code == 2
    assert envelope["error"]["code"] in {
        "clinical_json_invalid",
        "clinical_input_encoding",
    }


@pytest.mark.parametrize(
    "reference",
    [
        "PRIVATE_REFERENCE",
        "2026-02-30",
        "20260615",
        "2026-W01-1",
        "2026-06-15T12:00:00",
        "٢٠٢٦-٠٦-١٥",
        "2026-06-15T12:00:00+99:00",
        "2026-06-15T12:00:00+00:60",
    ],
)
def test_bad_reference_rejected_before_reading_or_processing(
    tmp_path, capsys, monkeypatch, reference
):
    reader = Mock(side_effect=AssertionError("must not read"))
    monkeypatch.setattr(clinical, "_read", reader)
    code, envelope = _run(
        capsys,
        "timeline",
        tmp_path / "absent",
        tmp_path / "spans",
        "--reference-time",
        reference,
    )
    assert code == 2 and envelope["error"]["code"] == "clinical_reference_invalid"
    reader.assert_not_called()


@pytest.mark.parametrize(
    "name,language,options",
    [
        ("sdoh", "de", ()),
        ("timeline", "zh", ()),
        ("relations", "fr", ()),
        ("sdoh", "PRIVATE_LANGUAGE", ()),
        ("relations", "hi", ("--sections", "PRIVATE_SECTIONS")),
    ],
)
def test_unsupported_language_or_options_never_read_inputs(
    tmp_path, capsys, monkeypatch, name, language, options
):
    reader = Mock(side_effect=AssertionError("must not read"))
    monkeypatch.setattr(clinical, "_read", reader)
    code, envelope = _run(
        capsys,
        name,
        tmp_path / "absent",
        tmp_path / "spans",
        "--language",
        language,
        *options,
    )
    assert code == 2
    assert envelope["error"]["code"] in {
        "clinical_language_unsupported",
        "clinical_options_unsupported",
    }
    reader.assert_not_called()


@pytest.mark.parametrize(
    "options",
    [
        ["--private-option", "PRIVATE_VALUE"],
        ["--reference-time", "PRIVATE_VALUE"],
        ["--spans"],
        [],
    ],
)
def test_parser_errors_never_echo_private_arguments(capsys, options):
    code = main(["clinical", "sdoh", "--json", *options])
    captured = capsys.readouterr()
    assert captured.err == ""
    assert code == 2
    assert json.loads(captured.out)["error"]["code"] == "clinical_arguments_invalid"
    assert "PRIVATE" not in captured.out


def test_non_json_errors_and_global_config_path_are_private(tmp_path, capsys):
    code = main(
        [
            "--config-path",
            str(tmp_path / "PRIVATE_CONFIG"),
            "clinical",
            "PRIVATE_SUBCOMMAND",
        ]
    )
    captured = capsys.readouterr()
    assert code == 2 and captured.out == ""
    assert "PRIVATE" not in captured.err


@pytest.mark.parametrize(
    "target", ["missing", "directory", "symlink", "fifo", "too_large", "bad_utf8"]
)
def test_bounded_reader_rejects_unsafe_note_inputs(tmp_path, capsys, target):
    note, spans, _ = _files(tmp_path, "", [])
    note.unlink()
    if target == "directory":
        note.mkdir()
    elif target == "symlink":
        note.symlink_to(spans)
    elif target == "fifo":
        os.mkfifo(note)
    elif target == "too_large":
        note.write_bytes(b"x" * (clinical._NOTE_BYTES + 1))
    elif target == "bad_utf8":
        note.write_bytes(b"\xff")
    code, envelope = _run(capsys, "sdoh", note, spans)
    assert code == 2
    assert envelope["error"]["code"] in {
        "clinical_input_unavailable",
        "clinical_input_not_regular",
        "clinical_input_too_large",
        "clinical_input_encoding",
    }


@pytest.mark.parametrize(
    "section",
    [
        [{"start": 1, "end": 6, "label": "social_history"}],
        [{"start": 0, "end": 5, "label": "social_history"}],
        [{"start": 0, "end": 6, "label": "PRIVATE_SECTION"}],
        [{"start": False, "end": 6, "label": "social_history"}],
    ],
)
def test_invalid_sections_fail_before_processing(
    tmp_path, capsys, monkeypatch, section
):
    note, spans, _ = _files(tmp_path, "Fever.", [])
    section_file = tmp_path / "PRIVATE_SECTIONS.json"
    section_file.write_text(json.dumps(section))
    processing = Mock()
    monkeypatch.setattr(clinical, "_sdoh", processing)
    code, envelope = _run(capsys, "sdoh", note, spans, "--sections", str(section_file))
    assert code == 2 and envelope["error"]["code"] == "clinical_sections_invalid"
    processing.assert_not_called()


def test_output_is_private_new_file_and_refuses_overwrite(tmp_path, capsys):
    note, spans, _ = _files(tmp_path, "No tobacco use.", [])
    output = tmp_path / "PRIVATE_OUTPUT.json"
    code, envelope = _run(capsys, "sdoh", note, spans, "--output", str(output))
    assert code == 0 and envelope["data"]["output_written"] is True
    written = output.read_bytes()
    assert output.stat().st_mode & 0o777 == 0o600
    saved = json.loads(written)
    assert saved["command"] == "clinical sdoh" and saved["ok"]
    assert saved["data"]["findings"] == envelope["data"]["findings"]
    code, envelope = _run(capsys, "sdoh", note, spans, "--output", str(output))
    assert code == 1 and envelope["error"]["code"] == "clinical_output_unavailable"
    assert output.read_bytes() == written


def test_output_symlink_is_not_followed(tmp_path, capsys):
    note, spans, _ = _files(tmp_path, "", [])
    output = tmp_path / "output.json"
    output.symlink_to(note)
    code, envelope = _run(capsys, "sdoh", note, spans, "--output", str(output))
    assert code == 1 and envelope["error"]["code"] == "clinical_output_unavailable"
    assert note.read_bytes() == b""


def test_partial_output_is_removed_and_exception_text_is_hidden(
    tmp_path, capsys, monkeypatch
):
    note, spans, _ = _files(tmp_path, "", [])
    output = tmp_path / "output.json"

    def broken_dump(payload, stream, **kwargs):
        stream.write('{"partial":')
        raise OSError("PRIVATE_FAILURE")

    monkeypatch.setattr(clinical.json, "dump", broken_dump)
    code, envelope = _run(capsys, "sdoh", note, spans, "--output", str(output))
    assert code == 1 and envelope["error"]["code"] == "clinical_output_unavailable"
    assert not output.exists() and "PRIVATE_FAILURE" not in json.dumps(envelope)


def test_processor_failures_are_fixed_and_private(tmp_path, capsys, monkeypatch):
    note, spans, _ = _files(tmp_path, "Synthetic Person", [])
    monkeypatch.setattr(
        clinical,
        "_sdoh",
        Mock(side_effect=RuntimeError("Synthetic Person PRIVATE_FAILURE")),
    )
    code, envelope = _run(capsys, "sdoh", note, spans)
    assert code == 1
    assert envelope["error"] == {
        "code": "clinical_processing_failed",
        "message": "Clinical processing failed.",
    }


def test_processor_console_output_is_discarded(tmp_path, capsys, monkeypatch):
    import sys

    note, spans, _ = _files(tmp_path, "Synthetic Person", [])

    def noisy_processor(*args):
        print("Synthetic Person PRIVATE_DIAGNOSTIC")
        print("Synthetic Person PRIVATE_DIAGNOSTIC", file=sys.stderr)
        raise RuntimeError("PRIVATE_DIAGNOSTIC")

    monkeypatch.setattr(clinical, "_sdoh", noisy_processor)
    code, envelope = _run(capsys, "sdoh", note, spans)
    assert code == 1
    assert envelope["error"]["code"] == "clinical_processing_failed"
    assert "PRIVATE" not in json.dumps(envelope)


@pytest.mark.parametrize("name", ["sdoh", "relations", "timeline"])
def test_empty_note_and_spans_are_valid_offline(tmp_path, capsys, name):
    note, spans, _ = _files(tmp_path, "", [])
    code, envelope = _run(capsys, name, note, spans)
    assert code == 0 and envelope["data"]["input_span_count"] == 0


@pytest.mark.parametrize("boundary", ["read", "decode", "json", "write", "processor"])
def test_clinical_refusals_discard_private_exception_context(
    tmp_path, monkeypatch, boundary
):
    from openmed.cli._output import CliError
    from openmed.cli.main import build_parser

    if boundary == "read":
        call = lambda: clinical._read(str(tmp_path / "PRIVATE_SOURCE"), 16)
    elif boundary == "decode":
        call = lambda: clinical._decode(b"\xffPRIVATE_SOURCE")
    elif boundary == "json":
        call = lambda: clinical._load_json(b'{"PRIVATE_SOURCE":')
    elif boundary == "write":

        def fail(*args, **kwargs):
            raise OSError("PRIVATE_OUTPUT_DETAIL")

        monkeypatch.setattr(clinical.json, "dump", fail)
        call = lambda: clinical._write_new(str(tmp_path / "output.json"), {})
    else:
        note, spans, _ = _files(tmp_path, "", [])
        monkeypatch.setattr(
            clinical,
            "_sdoh",
            Mock(side_effect=RuntimeError("PRIVATE_PROCESSOR_DETAIL")),
        )
        args = build_parser().parse_args(
            ["clinical", "sdoh", "--note", str(note), "--spans", str(spans), "--json"]
        )
        call = lambda: clinical._handle_clinical(args)
    with pytest.raises(CliError) as caught:
        call()
    assert caught.value.__context__ is None
    assert caught.value.__cause__ is None


class _PrivateClinicalCode(str):
    def __new__(cls):
        return super().__new__(cls, "PRIVATE_SOURCE_VALUE")

    def __eq__(self, other):
        return other == "tobacco"

    def __hash__(self):
        return hash("tobacco")


def test_clinical_projection_rejects_private_scalar_alias(
    tmp_path, capsys, monkeypatch
):
    from types import SimpleNamespace

    import openmed.clinical.sdoh as sdoh

    note, spans, _ = _files(tmp_path, "smokes", [])
    finding = SimpleNamespace(
        span=(0, 6),
        category=_PrivateClinicalCode(),
        status="current",
        temporality="recent",
        score=0.9,
        extent=None,
    )
    monkeypatch.setattr(sdoh, "extract_sdoh", lambda *args, **kwargs: [finding])
    code, envelope = _run(capsys, "sdoh", note, spans)
    assert code == 1 and envelope["error"]["code"] == "clinical_processing_failed"
    assert "PRIVATE" not in json.dumps(envelope)


def test_partial_writer_cleanup_preserves_replaced_file(tmp_path, monkeypatch):
    from openmed.cli._output import CliError

    target = tmp_path / "output.json"

    def replace_target(payload, stream, **kwargs):
        target.unlink()
        target.write_text("owner replacement")
        raise OSError("PRIVATE_DETAIL")

    monkeypatch.setattr(clinical.json, "dump", replace_target)
    with pytest.raises(CliError):
        clinical._write_new(str(target), {})
    assert target.read_text() == "owner replacement"


def test_custom_extractor_oversized_output_refused_before_file_creation(
    tmp_path, capsys, monkeypatch
):
    from types import SimpleNamespace

    import openmed.clinical.sdoh as sdoh

    note, spans, _ = _files(tmp_path, "smokes", [])
    finding = SimpleNamespace(
        span=(0, 6),
        category="tobacco",
        status="current",
        temporality="recent",
        score=0.9,
        extent=None,
    )
    monkeypatch.setattr(
        sdoh, "extract_sdoh", lambda *args, **kwargs: [finding] * 10_000
    )
    target = tmp_path / "output.json"
    code, envelope = _run(capsys, "sdoh", note, spans, "--output", str(target))
    assert code == 1 and envelope["error"]["code"] == "clinical_processing_failed"
    assert not target.exists()
