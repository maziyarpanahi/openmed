"""Typos in budget maps must not silently become unlimited requests."""

import json
from types import MappingProxyType

import pytest

from openmed.core.budget import RequestBudget, coerce_budget
from openmed.core.errors import InputError


def test_unknown_fields_raise_instead_of_being_ignored(subtests):
    for payload in [
        {"max_wall_tme": 0.5},
        {"max_input_char": 10},
        {"timeout": 0.5},
        {"max_wall_time": 1, "unused": 10},
        {"max_input_chars": 10, "unused": None},
        {42: 1},
        {None: 1},
    ]:
        with subtests.test(payload=payload):
            with pytest.raises(InputError):
                coerce_budget(payload)


def test_unknown_field_error_does_not_echo_keys_or_values():
    with pytest.raises(InputError) as raised:
        coerce_budget({"SYNTHETIC_UNKNOWN_FIELD": "SYNTHETIC_UNKNOWN_VALUE"})
    rendered = str(raised.value) + json.dumps(raised.value.to_dict())
    assert "SYNTHETIC_UNKNOWN" not in rendered
    assert "max_wall_time" in rendered
    assert "max_input_chars" in rendered


def test_explicit_unlimited_forms_are_preserved(subtests):
    for payload in [
        None,
        {},
        {"max_wall_time": None},
        {"max_input_chars": None},
        {"max_wall_time": None, "max_input_chars": None},
    ]:
        with subtests.test(payload=payload):
            assert coerce_budget(payload) is None


def test_valid_read_only_mapping_is_not_modified():
    source = {"max_wall_time": 0.5, "max_input_chars": 10}
    result = coerce_budget(MappingProxyType(source))
    assert result == RequestBudget(max_wall_time=0.5, max_input_chars=10)
    assert source == {"max_wall_time": 0.5, "max_input_chars": 10}


def test_existing_budget_identity_is_preserved():
    budget = RequestBudget(max_input_chars=10)
    assert coerce_budget(budget) is budget


def test_known_value_validation_still_rejects_invalid_values(subtests):
    for value in [0, -1, True, "bad", float("nan")]:
        with subtests.test(value=value):
            with pytest.raises(InputError):
                coerce_budget({"max_wall_time": value})
