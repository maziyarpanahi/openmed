"""Versioned rule precedence and aggregate-only assertion reports."""

import json
import socket
from dataclasses import replace
from pathlib import Path

from openmed.cli.main import main
from openmed.eval.datasets.i2b2_2010 import load_i2b2_2010
from openmed.eval.suites.i2b2_2010_assertion import (
    MAPPING_VERSION,
    PRECEDENCE,
    assertion_class,
    score_i2b2_2010,
)
from tests.fixtures.eval.i2b2_2010_synthetic import write_i2b2_2010_corpus


def test_precedence_v1():
    assert MAPPING_VERSION == 1
    assert PRECEDENCE == (
        ("experiencer", ("family", "other"), "associated_with_someone_else"),
        ("temporality", ("hypothetical",), "hypothetical"),
        ("negation", ("negated",), "absent"),
        ("uncertainty", ("uncertain",), "possible"),
    )
    axes = dict(
        experiencer="other",
        temporality="hypothetical",
        negation="negated",
        uncertainty="uncertain",
    )
    assert assertion_class(axes) == "associated_with_someone_else"
    assert assertion_class({**axes, "experiencer": "patient"}) == "hypothetical"
    assert (
        assertion_class(
            dict(
                experiencer="patient",
                temporality="historical",
                negation="affirmed",
                uncertainty="certain",
            )
        )
        == "present"
    )


def test_counts_denies_and_suppression(tmp_path):
    fixtures = load_i2b2_2010(write_i2b2_2010_corpus(tmp_path))
    result = score_i2b2_2010(fixtures, minimum_gold_spans=1)
    assert {k: (v["tp"], v["fp"], v["fn"]) for k, v in result["by_class"].items()} == {
        "present": (4, 1, 0),
        "absent": (1, 0, 0),
        "possible": (1, 0, 0),
        "conditional": (0, 0, 1),
        "hypothetical": (1, 0, 0),
        "associated_with_someone_else": (1, 0, 0),
    }
    assert result["by_class"]["conditional"]["system_emits_class"] is False
    assert result["confusion"]["conditional"]["present"] == 1
    assert result["metrics"]["micro_f1"] == 8 / 9
    text = "Denies rash."
    f = fixtures[0]
    span = replace(
        f.gold_spans[0],
        start=7,
        end=11,
        label="PROBLEM",
        metadata={"assertion": "absent"},
    )
    f = replace(
        f,
        text=text,
        entities={"p": span},
        metadata={"sentence_offsets": [[0, len(text)]]},
    )
    assert score_i2b2_2010([f], minimum_gold_spans=1)["by_class"]["absent"]["tp"] == 1


def test_cli_preflight_and_offline_output(tmp_path, monkeypatch):
    from openmed.eval.datasets import i2b2_2010

    corpus = write_i2b2_2010_corpus(tmp_path / "private-input")
    monkeypatch.setenv("OPENMED_I2B2_2010_PATH", str(corpus))

    def forbidden(*args, **kwargs):
        raise AssertionError("unexpected I/O")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    args = ["eval", "credentialed", "--suite", "i2b2-2010-assertion", "--output"]
    with monkeypatch.context() as patch:
        patch.setattr(i2b2_2010, "load_i2b2_2010", forbidden)
        assert main([*args, str(Path.cwd() / "unused.json")]) == 2
    target = tmp_path / "report.json"
    assert main([*args, str(target)]) == 0
    raw = target.read_text()
    assert all(
        v not in raw
        for v in ("private-input", "chest pain", "note-001", '"start"', '"end"')
    )
    result = json.loads(raw)
    assert result["mapping_version"] == 1
    assert result["cue_lexicon_digest"].startswith("sha256:")
    assert all(v["status"] == "suppressed" for v in result["by_class"].values())
