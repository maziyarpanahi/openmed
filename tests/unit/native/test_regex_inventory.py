"""Offline construct inventory tests on synthetic and registered patterns."""

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from openmed.core.pii_entity_merger import PIIPattern
from scripts.native.regex_inventory import (
    classify_pattern,
    inventory,
    main,
    registered_patterns,
)


def test_three_synthetic_classes_and_value_free_summary():
    banks = {
        "xx": [
            PIIPattern(r"SYNTHETIC_SENTINEL\d+", "synthetic", context_words=["SECRET"]),
            PIIPattern(r"(?<=x)y", "synthetic"),
            PIIPattern(r"(x)\1", "synthetic"),
        ],
        "yy": [],
    }
    report = inventory(banks)
    assert report == {
        "total": 3,
        "constructs": {"plain": 1, "lookaround": 1, "backreference": 1},
        "languages": {
            "xx": {"total": 3, "plain": 1, "lookaround": 1, "backreference": 1},
            "yy": {"total": 0, "plain": 0, "lookaround": 0, "backreference": 0},
        },
        "patterns": [
            {"id": "xx:0", "constructs": ["plain"]},
            {"id": "xx:1", "constructs": ["lookaround"]},
            {"id": "xx:2", "constructs": ["backreference"]},
        ],
    }
    encoded = json.dumps(report)
    assert "SYNTHETIC_SENTINEL" not in encoded
    assert "SECRET" not in encoded
    assert "synthetic" not in encoded


@pytest.mark.parametrize(
    ("regex", "flags", "expected"),
    [
        (r"x(?=y)", 0, ["lookaround"]),
        (r"x(?!y)", 0, ["lookaround"]),
        (r"(?<!x)y", 0, ["lookaround"]),
        (r"(?P<part>x)(?P=part)", 0, ["backreference"]),
        (r"(x)?(?(1)y|z)", 0, ["backreference"]),
        (r"(?:(x)\1|y)+", 0, ["backreference"]),
        (r"(?:(?=x)x)+?", 0, ["lookaround"]),
        (r"(?:(?=x)x)++", 0, ["lookaround"]),
        (r"(?>x(?=y))", 0, ["lookaround"]),
        (r"(x)?(?(1)(?=y)y|z)", 0, ["backreference", "lookaround"]),
        (r"(?=(x)\1)", 0, ["backreference", "lookaround"]),
        (r"\(\?=x\)\\1", 0, ["plain"]),
        (r"[(?=!<)\1]+", 0, ["plain"]),
        (r"\123", 0, ["plain"]),
        (r"(?# (?= ignored)x", 0, ["plain"]),
        ("x # (?= ignored \\1\n y", re.VERBOSE, ["plain"]),
        (r"(?:x|y)+", 0, ["plain"]),
    ],
)
def test_nested_constructs_and_literal_negative_controls(regex, flags, expected):
    assert classify_pattern(PIIPattern(regex, "synthetic", flags=flags)) == expected


def test_table_and_json_cli_are_consistent(monkeypatch, capsys):
    banks = {"xx": [PIIPattern(r"(?=x)x", "synthetic")]}
    monkeypatch.setattr(
        "scripts.native.regex_inventory.registered_patterns", lambda: banks
    )
    assert main([]) == 0
    assert capsys.readouterr().out.splitlines() == [
        "language       total   plain  lookaround  backreference",
        "xx                 1       0           1              0",
    ]
    assert main(["--json"]) == 0
    assert json.loads(capsys.readouterr().out) == inventory(banks)


def test_invalid_registered_pattern_does_not_echo_source(monkeypatch, capsys):
    monkeypatch.setattr(
        "scripts.native.regex_inventory.registered_patterns",
        lambda: {"xx": [PIIPattern("SYNTHETIC_SENTINEL(", "synthetic")]},
    )
    assert main(["--json"]) == 1
    assert capsys.readouterr() == ("", "registered_pattern_invalid\n")


def test_registered_routes_and_direct_script_run_offline(monkeypatch):
    import socket

    def no_network(*args, **kwargs):
        pytest.fail("inventory attempted network access")

    monkeypatch.setattr(socket, "socket", no_network)
    banks = registered_patterns()
    report = inventory(banks)
    assert {"en", "fr", "en_gb", "ar_eg"} <= banks.keys()
    assert report["total"] == sum(map(len, banks.values()))
    assert len({row["id"] for row in report["patterns"]}) == report["total"]
    assert report["constructs"]["lookaround"] > 0
    assert report["constructs"]["backreference"] > 0
    root = Path(__file__).resolve().parents[3]
    result = subprocess.run(
        [sys.executable, str(root / "scripts/native/regex_inventory.py"), "--json"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stderr == ""
    assert json.loads(result.stdout) == report
