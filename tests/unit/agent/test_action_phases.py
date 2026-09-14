"""Behavior contracts for agent action execution phases."""

from __future__ import annotations

import json
import subprocess
import sys
import traceback

import pytest

from openmed.agent.action_phases import (
    ACTION_PHASE_TRANSITIONS,
    ActionPhase,
    ActionPhaseError,
    is_resumable_phase,
    is_terminal_phase,
    validate_action_transition,
)

# Independent expected behavior, rather than deriving cases from the model.
ORDINARY_EDGES = (
    ("queued", "preflight"),
    ("queued", "aborted"),
    ("preflight", "ready"),
    ("preflight", "waiting-review"),
    ("preflight", "aborted"),
    ("ready", "running"),
    ("ready", "aborted"),
    ("running", "waiting-review"),
    ("running", "completed"),
    ("running", "aborted"),
    ("waiting-review", "aborted"),
)
REVIEWED_EDGE = ("waiting-review", "preflight")


def test_phase_vocabulary_is_closed_and_json_safe():
    expected = [
        "queued",
        "preflight",
        "ready",
        "running",
        "waiting-review",
        "completed",
        "aborted",
    ]
    assert [phase.value for phase in ActionPhase] == expected
    assert json.loads(json.dumps(list(ActionPhase))) == expected


@pytest.mark.parametrize(("current", "target"), ORDINARY_EDGES)
def test_every_ordinary_transition(current, target):
    assert validate_action_transition(current, target) is None
    assert validate_action_transition(ActionPhase(current), ActionPhase(target)) is None


@pytest.mark.parametrize("current", list(ActionPhase))
@pytest.mark.parametrize("target", list(ActionPhase))
@pytest.mark.parametrize("reviewed", [False, True])
def test_complete_transition_matrix(current, target, reviewed):
    edge = (current.value, target.value)
    if edge in ORDINARY_EDGES or (edge == REVIEWED_EDGE and reviewed):
        assert validate_action_transition(current, target, reviewed=reviewed) is None
    else:
        expected_code = (
            "review_required" if edge == REVIEWED_EDGE else "invalid_transition"
        )
        with pytest.raises(ActionPhaseError) as caught:
            validate_action_transition(current, target, reviewed=reviewed)
        assert caught.value.code == expected_code


@pytest.mark.parametrize("entry", ["preflight", "running"])
def test_reviewed_resume_reenters_preflight_before_execution(entry):
    validate_action_transition(entry, "waiting-review")
    with pytest.raises(ActionPhaseError, match="review_required"):
        validate_action_transition("waiting-review", "preflight")
    validate_action_transition("waiting-review", "preflight", reviewed=True)
    with pytest.raises(ActionPhaseError, match="invalid_transition"):
        validate_action_transition("preflight", "running", reviewed=True)
    validate_action_transition("preflight", "ready")
    validate_action_transition("ready", "running")
    validate_action_transition("running", "completed")


@pytest.mark.parametrize("reviewed", [False, True])
@pytest.mark.parametrize("target", ["ready", "running", "completed"])
def test_review_cannot_jump_to_execution_or_completion(target, reviewed):
    with pytest.raises(ActionPhaseError, match="invalid_transition"):
        validate_action_transition("waiting-review", target, reviewed=reviewed)


@pytest.mark.parametrize("terminal", ["completed", "aborted"])
@pytest.mark.parametrize("target", list(ActionPhase))
@pytest.mark.parametrize("reviewed", [False, True])
def test_terminal_phases_cannot_be_revived_or_reentered(terminal, target, reviewed):
    with pytest.raises(ActionPhaseError, match="invalid_transition"):
        validate_action_transition(terminal, target, reviewed=reviewed)


@pytest.mark.parametrize("phase", list(ActionPhase))
def test_terminal_and_resumable_are_distinct(phase):
    for value in (phase, phase.value):
        assert is_terminal_phase(value) is (phase.value in {"completed", "aborted"})
        assert is_resumable_phase(value) is (phase.value == "waiting-review")


class StringSubclass(str):
    """Reject noncanonical string types, as in neighboring agent contracts."""


class Untrusted:
    """Input whose hooks must never be evaluated."""

    def __str__(self):
        raise AssertionError("untrusted string conversion")

    def __repr__(self):
        raise AssertionError("untrusted representation")

    def __hash__(self):
        raise AssertionError("untrusted hashing")

    def __eq__(self, other):
        raise AssertionError("untrusted comparison")

    def __bool__(self):
        raise AssertionError("untrusted truthiness")


INVALID_PHASES = [
    "unknown",
    "READY",
    " ready",
    "ready\n",
    "waiting_review",
    "",
    None,
    True,
    1,
    [],
    {},
    b"ready",
    StringSubclass("ready"),
    Untrusted(),
]


@pytest.mark.parametrize("value", INVALID_PHASES, ids=range(len(INVALID_PHASES)))
@pytest.mark.parametrize("field", ["current", "target"])
def test_unknown_phases_fail_closed_in_both_positions(value, field):
    values = {"current": "queued", "target": "preflight"}
    values[field] = value
    with pytest.raises(ActionPhaseError) as caught:
        validate_action_transition(**values)
    assert caught.value.code == "unknown_phase"
    assert caught.value.field_name == field


@pytest.mark.parametrize("value", INVALID_PHASES, ids=range(len(INVALID_PHASES)))
@pytest.mark.parametrize("predicate", [is_terminal_phase, is_resumable_phase])
def test_phase_predicates_reject_unknown_inputs(predicate, value):
    with pytest.raises(ActionPhaseError) as caught:
        predicate(value)
    assert caught.value.code == "unknown_phase"
    assert caught.value.field_name == "phase"


@pytest.mark.parametrize(
    "value", [None, 0, 1, "true", "false", [], {}, Untrusted()], ids=range(8)
)
def test_review_declaration_requires_an_exact_boolean(value):
    with pytest.raises(ActionPhaseError) as caught:
        validate_action_transition("waiting-review", "preflight", reviewed=value)
    assert caught.value.code == "invalid_reviewed"
    assert caught.value.field_name == "reviewed"


@pytest.mark.parametrize("field", ["current", "target", "reviewed"])
def test_invalid_input_errors_are_deterministic_and_payload_free(field):
    sentinel = "Synthetic patient /private/chart bearer-token-sentinel"
    values = {"current": "queued", "target": "preflight", "reviewed": False}
    values[field] = sentinel
    errors = []
    for _ in range(2):
        with pytest.raises(ActionPhaseError) as caught:
            validate_action_transition(**values)
        error = caught.value
        errors.append((error.code, error.field_name, str(error)))
        assert sentinel not in "".join(
            traceback.format_exception(caught.type, error, caught.tb)
        )
        assert sentinel not in repr(vars(error))
        assert error.__cause__ is None
        assert error.__context__ is None
    assert errors[0] == errors[1]


def test_transition_table_exposes_review_requirement_and_is_read_only():
    assert set(ACTION_PHASE_TRANSITIONS) == set(ActionPhase)
    for phase in (ActionPhase.COMPLETED, ActionPhase.ABORTED):
        assert not ACTION_PHASE_TRANSITIONS[phase]
    assert ACTION_PHASE_TRANSITIONS[ActionPhase.WAITING_REVIEW] == {
        ActionPhase.PREFLIGHT: True,
        ActionPhase.ABORTED: False,
    }
    with pytest.raises(TypeError):
        ACTION_PHASE_TRANSITIONS[ActionPhase.COMPLETED] = {}
    with pytest.raises(TypeError):
        ACTION_PHASE_TRANSITIONS[ActionPhase.WAITING_REVIEW][ActionPhase.READY] = False


def test_phase_contract_is_available_from_public_agent_api():
    import openmed.agent as agent

    for name in (
        "ACTION_PHASE_TRANSITIONS",
        "ActionPhase",
        "ActionPhaseError",
        "is_resumable_phase",
        "is_terminal_phase",
        "validate_action_transition",
    ):
        assert getattr(agent, name) is globals()[name]
        assert name in agent.__all__


def test_validation_is_import_light_and_has_no_execution_side_effects():
    script = """
import sys
import openmed.agent as agent

def reject_effects(event, args):
    if event in {"open", "os.system", "subprocess.Popen"} or event.startswith("socket."):
        raise AssertionError(event)

sys.addaudithook(reject_effects)
for current in agent.ActionPhase:
    for target in agent.ActionPhase:
        for reviewed in (False, True):
            try:
                agent.validate_action_transition(current, target, reviewed=reviewed)
            except agent.ActionPhaseError:
                pass
assert not {"torch", "transformers", "openmed.core.review_workflow"} & sys.modules.keys()
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
