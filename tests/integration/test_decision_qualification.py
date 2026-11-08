"""Offline command/library composition for decision-backend qualification."""

import json
import os
import stat

import pytest

from openmed.cli.decision_qualification import handle_decision_qualification
from openmed.cli.main import build_parser, main
from openmed.eval.decision_qualification import (
    DecisionQualificationPolicy,
    qualify_decision_backend,
)
from openmed.structured import decision as decision_module
from openmed.structured.decision import (
    DecisionCalibrationProfile,
    DecisionRequest,
    DecisionState,
    DeterministicDecisionBackend,
    decide,
)
from tests.unit.eval import test_decision_qualification as fixtures

pytestmark = pytest.mark.integration

ENCODER = "tests.unit.eval.test_decision_qualification:synthetic_encoder_backend"
SPY = "tests.unit.eval.test_decision_qualification:spy_encoder_backend"


def command(
    *extra, development=None, evaluation=None, backend=("--backend", "deterministic")
):
    return [
        "decision-qualify",
        *backend,
        "--development",
        str(development or fixtures.DEVELOPMENT),
        "--evaluation",
        str(evaluation or fixtures.EVALUATION),
        "--calibration-id",
        "site.triage",
        *extra,
        "--json",
    ]


def test_command_and_library_emit_identical_deterministic_report(capsys):
    args = build_parser().parse_args(command("--min-examples-per-option", "2"))

    assert handle_decision_qualification(args) == 0
    envelope = json.loads(capsys.readouterr().out)
    expected = qualify_decision_backend(
        DeterministicDecisionBackend(),
        development=fixtures.DEVELOPMENT,
        evaluation=fixtures.EVALUATION,
        calibration_id="site.triage",
        policy=DecisionQualificationPolicy(min_examples_per_option=2),
    ).to_dict()
    assert envelope["command"] == "decision-qualify"
    assert envelope["data"] == expected
    assert envelope["data"]["status"] == "qualified"


def test_factory_encoder_writes_private_profile_usable_by_decide(tmp_path, capsys):
    output = tmp_path / "profiles.json"

    status = main(
        command(
            "--min-examples-per-option",
            "2",
            "--profile-output",
            str(output),
            backend=("--backend-factory", ENCODER),
        )
    )

    assert status == 0
    data = json.loads(capsys.readouterr().out)["data"]
    assert data["backend"]["kind"] == "encoder"
    if os.name != "nt":
        assert stat.S_IMODE(output.stat().st_mode) == 0o600
    written = json.loads(output.read_text("utf-8"))
    assert written == data["profiles"]
    profiles = {
        key: DecisionCalibrationProfile(**value) for key, value in written.items()
    }
    result = decide(
        DecisionRequest(
            mode="fixed_choice",
            input_text="Synthetic review priority is clearly urgent.",
            options=("urgent", "routine"),
            calibration_id="site.triage.fixed_choice",
        ),
        backend=fixtures.synthetic_encoder_backend(),
        calibration_profiles=profiles,
    )
    assert result.state is DecisionState.SUCCESS
    assert result.choice == "urgent"


def test_default_support_minimum_withholds_profile(tmp_path, capsys):
    output = tmp_path / "profiles.json"

    status = main(command("--profile-output", str(output)))

    assert status == 1
    data = json.loads(capsys.readouterr().out)["data"]
    assert data["status"] == "insufficient"
    assert data["policy"]["min_examples_per_option"] == 20
    assert data["profiles"] == {}
    assert not output.exists()


def test_human_output_states_review_and_withheld_profile(tmp_path, capsys):
    output = tmp_path / "profiles.json"
    arguments = command("--profile-output", str(output))
    arguments.remove("--json")

    assert main(arguments) == 1
    captured = capsys.readouterr()
    assert captured.out.strip() == (
        "Decision qualification: insufficient; human review required. Profile withheld."
    )


def test_evaluation_is_scored_only_after_fit_through_the_command(capsys):
    fixtures.SCORED.clear()
    dev_inputs = {
        fixtures.text_digest(row["input_text"])
        for row in fixtures.records(fixtures.DEVELOPMENT)
    }
    eval_inputs = {
        fixtures.text_digest(row["input_text"])
        for row in fixtures.records(fixtures.EVALUATION)
    }

    assert (
        main(
            command(
                "--min-examples-per-option", "2", backend=("--backend-factory", SPY)
            )
        )
        == 0
    )
    capsys.readouterr()

    assert set(fixtures.SCORED[: len(dev_inputs)]) == dev_inputs
    assert set(fixtures.SCORED[len(dev_inputs) :]) == eval_inputs
    fixtures.SCORED.clear()


def test_command_never_registers_the_selected_backend_as_default(capsys):
    default_backend = decision_module.DETERMINISTIC_DECISION_BACKEND
    default_profile = decision_module.DEFAULT_CALIBRATION_PROFILE

    for _ in range(2):
        assert (
            main(
                command(
                    "--min-examples-per-option",
                    "2",
                    backend=("--backend-factory", ENCODER),
                )
            )
            == 0
        )
    capsys.readouterr()

    assert decision_module.DETERMINISTIC_DECISION_BACKEND is default_backend
    assert decision_module.DEFAULT_CALIBRATION_PROFILE is default_profile
    result = decide(
        DecisionRequest(
            mode="fixed_choice",
            input_text="Synthetic review priority is clearly urgent.",
            options=("urgent", "routine"),
            calibration_id="site.triage.fixed_choice",
        )
    )
    assert result.backend["kind"] == "deterministic"
    assert result.code == "calibration_unavailable"


@pytest.mark.parametrize(
    ("factory", "code"),
    [
        ("not a factory", "backend_factory_invalid"),
        ("tests.unit.eval.test_decision_qualification", "backend_factory_invalid"),
        ("tests.unit.eval.test_decision_qualification:records", "backend_unavailable"),
        (
            "tests.unit.eval.test_decision_qualification:text_digest",
            "backend_unavailable",
        ),
        ("tests.unit.eval.missing_module_marker:factory", "backend_unavailable"),
        ("json:JSONDecoder", "backend_factory_invalid"),
    ],
)
def test_invalid_backend_factories_fail_with_controlled_codes(factory, code, capsys):
    assert main(command(backend=("--backend-factory", factory))) == 1
    error = json.loads(capsys.readouterr().out)["error"]
    assert error["code"] == "decision_qualification_failed"
    assert error["message"] == f"Decision qualification failed: {code}."


def test_corrupt_input_has_value_free_error(tmp_path, capsys):
    corrupt = tmp_path / "private-SECRET.jsonl"
    corrupt.write_text("SECRET not JSON 555-0101\n", "utf-8")

    assert main(command(development=corrupt)) == 1
    output = capsys.readouterr()
    error = json.loads(output.out)["error"]
    assert error["message"] == "Decision qualification failed: invalid_record."
    assert "SECRET" not in output.out + output.err
    assert "555-0101" not in output.out + output.err


def test_existing_profile_output_is_never_overwritten(tmp_path, capsys):
    output = tmp_path / "profiles.json"
    output.write_text("caller-owned", "utf-8")

    status = main(
        command(
            "--min-examples-per-option",
            "2",
            "--profile-output",
            str(output),
        )
    )

    assert status == 1
    error = json.loads(capsys.readouterr().out)["error"]
    assert error["message"] == "Decision qualification failed: profile_output_exists."
    assert output.read_text("utf-8") == "caller-owned"


def test_backend_selection_is_required():
    arguments = command()
    arguments[1:3] = []
    with pytest.raises(SystemExit) as raised:
        build_parser().parse_args(arguments)
    assert raised.value.code == 2
