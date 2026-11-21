"""Offline intended-use queries through the public CLI."""

import json

from openmed.cli.main import main
from openmed.core.intended_use import load_intended_use_registry


def test_list_contains_every_capability(capsys):
    assert main(["intended-use", "list", "--json"]) == 0
    envelope = json.loads(capsys.readouterr().out)
    assert envelope["ok"] and envelope["command"] == "intended-use list"
    rows = envelope["data"]["capabilities"]
    expected = load_intended_use_registry().declarations
    assert [row["code"] for row in rows] == sorted(expected)
    for row in rows:
        tier = expected[row["code"]].regulatory_tier
        assert row["tier"] == tier.value
        assert set(row["obligations"]) == tier.obligations


def test_filter_and_show(capsys):
    assert main(["intended-use", "list", "--tier", "restricted", "--json"]) == 0
    rows = json.loads(capsys.readouterr().out)["data"]["capabilities"]
    assert "ecg" in {row["code"] for row in rows}
    assert {row["tier"] for row in rows} == {"restricted"}
    assert main(["intended-use", "show", "ecg"]) == 0
    output = capsys.readouterr().out
    for term in (
        "Display the notice.",
        "Record acknowledgement.",
        "Enable clinician mode.",
        "Provide crisis routing.",
        "treatment_selection",
    ):
        assert term in output
    assert main(["intended-use", "show", "openmed.deidentify", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)["data"]
    assert data["code"] == "deidentify"
    assert (
        data["declaration"]
        == load_intended_use_registry().resolve("deidentify").to_dict()
    )


def test_unknown_capability(capsys):
    assert main(["intended-use", "show", "unknown-sensitive-input", "--json"]) == 2
    result = capsys.readouterr()
    assert "unknown-sensitive-input" not in result.out + result.err
    envelope = json.loads(result.out)
    assert envelope["ok"] is False
    assert envelope["error"]["code"] == "unknown_capability"
