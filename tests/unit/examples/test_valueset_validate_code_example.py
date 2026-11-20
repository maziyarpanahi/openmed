"""Tests for the offline ValueSet validate-code example."""

from __future__ import annotations

import json
import socket

import pytest

from examples import valueset_validate_code as example

EXAMPLE_SYSTEM_PREFIX = "urn:openmed:example:"


def _by_case(records: list[dict]) -> dict[str, dict]:
    return {record["case"]: record for record in records}


def test_example_prints_one_json_line_per_case(capsys) -> None:
    records = example.main()

    lines = capsys.readouterr().out.strip().splitlines()
    assert [json.loads(line) for line in lines] == records
    assert [record["case"] for record in records] == [
        "member",
        "non_member",
        "excluded",
        "codeable_concept_annotate",
        "codeable_concept_drop",
        "codeable_concept_downgrade",
    ]


def test_member_is_valid_and_non_member_is_not(capsys) -> None:
    cases = _by_case(example.main())
    capsys.readouterr()

    assert cases["member"]["valid"] is True
    assert cases["non_member"]["valid"] is False
    assert "not a member" in cases["non_member"]["message"]


def test_excluded_code_is_reported_as_a_non_member(capsys) -> None:
    cases = _by_case(example.main())
    capsys.readouterr()

    assert cases["excluded"]["valid"] is False
    assert "explicitly excluded" in cases["excluded"]["message"]


def test_each_policy_treats_the_non_member_coding_differently(capsys) -> None:
    cases = _by_case(example.main())
    capsys.readouterr()

    annotate = cases["codeable_concept_annotate"]
    assert annotate["codes"] == ["soon", "unlisted"]
    assert annotate["annotated_codes"] == ["unlisted"]
    assert annotate["severities"] == ["error"]

    drop = cases["codeable_concept_drop"]
    assert drop["codes"] == ["soon"]
    assert drop["annotated_codes"] == []
    assert drop["severities"] == ["error"]

    downgrade = cases["codeable_concept_downgrade"]
    assert downgrade["codes"] == ["soon", "unlisted"]
    assert downgrade["annotated_codes"] == ["unlisted"]
    assert downgrade["severities"] == ["warning"]

    assert all(case["valid"] is False for case in (annotate, drop, downgrade))


def test_example_uses_only_the_invented_example_system() -> None:
    systems = {include["system"] for include in example.VALUESET["compose"]["include"]}
    systems |= {exclude["system"] for exclude in example.VALUESET["compose"]["exclude"]}
    systems |= {coding["system"] for coding in example.CONCEPT["coding"]}
    systems.add(example.SYSTEM)

    assert systems
    assert all(system.startswith(EXAMPLE_SYSTEM_PREFIX) for system in systems)


def test_example_makes_no_network_connection(monkeypatch, capsys) -> None:
    def refuse(*args, **kwargs):
        raise AssertionError("the example must not open a network connection")

    monkeypatch.setattr(socket, "socket", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)

    records = example.main()
    capsys.readouterr()

    assert len(records) == 6


@pytest.mark.parametrize("policy", ["annotate", "drop", "downgrade"])
def test_documented_policies_are_the_ones_the_library_accepts(policy) -> None:
    from openmed.clinical.exporters.valueset import validate_codeable_concept

    result = validate_codeable_concept(example.CONCEPT, example.VALUESET, policy=policy)

    assert result.valid is False
