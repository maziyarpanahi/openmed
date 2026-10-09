"""Synthetic offline notice propagation across Python vision and CLI surfaces."""

import json
from pathlib import Path

from openmed.cli.main import main
from openmed.mlx.vlm import OpenMedMLXVisionLanguageModel, VisionLanguageGeneration
from openmed.multimodal.notices import NOTICE_RESULT_TYPES

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures/multimodal/notices_v1.json"


def test_injected_offline_vision_generation_retains_notice(monkeypatch):
    case = json.loads(FIXTURE.read_text())[-1]
    generation = VisionLanguageGeneration.from_dict(case["result"])
    # Fixed result injection tests rendering only, never provider qualification.
    monkeypatch.setattr(
        OpenMedMLXVisionLanguageModel,
        "generate_with_metadata",
        lambda *a, **kw: generation,
    )
    model = object.__new__(OpenMedMLXVisionLanguageModel)
    rendered = model.generate("Describe synthetic geometric shapes")
    assert generation.notice.identifier in rendered
    assert generation.notice.text in rendered
    assert generation.text in rendered
    assert json.loads(generation.to_json())["notice"] == case["result"]["notice"]


def test_cli_text_and_json_keep_exact_catalog_notice(tmp_path, capsys):
    for case in json.loads(FIXTURE.read_text())[:-1]:
        path = tmp_path / "synthetic.json"
        path.write_text(json.dumps(case["result"]))
        command = ["multimodal-notice", "--kind", case["kind"], "--input", str(path)]
        assert main(command) == 0
        text = capsys.readouterr().out
        assert case["result"]["notice"]["identifier"] in text
        assert case["result"]["notice"]["text"] in text
        assert main([*command, "--json"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["ok"] is True
        assert payload["data"] == case["result"]
        assert (
            NOTICE_RESULT_TYPES[case["kind"]].from_dict(payload["data"]).notice.text
            in text
        )


def test_cli_refuses_invalid_notice_without_private_values(tmp_path, capsys):
    path = tmp_path / "synthetic-private-record.json"
    path.write_text('{"notice":"Synthetic Ada synthetic-mrn-98765"}')
    command = [
        "multimodal-notice",
        "--kind",
        "measurement_for_review",
        "--input",
        str(path),
        "--json",
    ]
    assert main(command) == 2
    output = capsys.readouterr().out
    assert "Ada" not in output
    assert "98765" not in output
    assert str(path) not in output
    assert json.loads(output)["error"]["code"] == "invalid_multimodal_notice"
