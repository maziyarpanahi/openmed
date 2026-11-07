"""Offline command/library composition with wholly synthetic runtime adapters."""

import json
from dataclasses import replace

import pytest

from openmed.cli.main import build_parser, main
from openmed.cli.nli_qualification import handle_nli_qualification
from openmed.clinical.brief import BriefRefusal, build_clinical_brief
from openmed.clinical.nli_qualification import bind_qualified_nli, qualify_local_nli
from tests.unit.clinical.test_brief import fixture_context
from tests.unit.clinical.test_nli_qualification import (
    FakeLoader,
    declared_inputs,
    inputs,  # noqa: F401 - shared synthetic pytest fixture
)

pytestmark = pytest.mark.integration


def test_command_and_library_emit_identical_synthetic_receipt(inputs, tmp_path, capsys):
    paths = {}
    for name in ("development", "evaluation", "label_mapping"):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(inputs[name]))
        paths[name] = path
    policy_path = tmp_path / "policy.json"
    policy_path.write_text('{"min_per_class":1}')
    args = build_parser().parse_args(
        [
            "nli-qualify",
            "--artifact",
            str(inputs["model_path"]),
            "--label-mapping",
            str(paths["label_mapping"]),
            "--development",
            str(paths["development"]),
            "--evaluation",
            str(paths["evaluation"]),
            "--policy",
            str(policy_path),
            "--json",
        ]
    )
    assert handle_nli_qualification(args, loader=FakeLoader()) == 1
    envelope = json.loads(capsys.readouterr().out)
    assert (
        envelope["data"] == qualify_local_nli(**inputs, loader=FakeLoader()).to_dict()
    )
    assert envelope["data"]["status"] == "synthetic_only"


def test_command_missing_evaluation_is_unavailable(inputs, tmp_path, capsys):
    mapping = tmp_path / "mapping.json"
    mapping.write_text(json.dumps(inputs["label_mapping"]))
    assert (
        main(
            [
                "nli-qualify",
                "--artifact",
                str(inputs["model_path"]),
                "--label-mapping",
                str(mapping),
                "--json",
            ]
        )
        == 1
    )
    payload = json.loads(capsys.readouterr().out)["data"]
    assert payload["reasons"] == ["development_unavailable", "evaluation_unavailable"]
    assert payload["reports"] == {}


def test_command_corrupt_input_has_value_free_error(inputs, tmp_path, capsys):
    mapping = tmp_path / "private-SECRET.json"
    mapping.write_text("SECRET not JSON 555-0101")
    assert (
        main(
            [
                "nli-qualify",
                "--artifact",
                str(inputs["model_path"]),
                "--label-mapping",
                str(mapping),
                "--json",
            ]
        )
        == 1
    )
    output = capsys.readouterr()
    assert json.loads(output.out)["error"]["code"] == "nli_qualification_failed"
    assert "SECRET" not in output.out + output.err
    assert "555-0101" not in output.out + output.err


def test_receipt_callback_composes_with_existing_brief_and_refuses_drift(inputs):
    # Caller provenance admission mechanics are exercised using fake data;
    # this is not a clinical model qualification or a released checkpoint.
    options = declared_inputs(inputs)
    receipt = qualify_local_nli(**options, loader=FakeLoader())
    backend = bind_qualified_nli(receipt, **options, loader=FakeLoader())
    result, context = fixture_context()
    context = replace(context, nli_predict=backend, thresholds=backend.thresholds)
    brief = build_clinical_brief(result, model="extractive", context=context)
    assert brief.refusal_reason is None
    assert brief.envelope["requires_human_review"]
    (inputs["model_path"] / "tokenizer.json").write_text("changed tokenizer")
    refused = build_clinical_brief(result, model="extractive", context=context)
    assert refused.refusal_reason is not None
    assert refused.summary == ""
    assert refused.refusal_reason is BriefRefusal.STAGE_FAILED
