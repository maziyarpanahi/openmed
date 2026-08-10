"""Focused tests for the deterministic CLI help-surface drift checker."""

from __future__ import annotations

import json

import pytest

from openmed.cli.help_drift import (
    EXIT_ADDED,
    EXIT_CHANGED,
    EXIT_CLEAN,
    EXIT_INVALID,
    EXIT_MIXED,
    EXIT_REMOVED,
    DriftCategory,
    HelpDriftError,
    compare_help_surfaces,
    main,
    normalize_help_records,
)


def _surface(*, required: bool = True, include_json: bool = False) -> list[dict]:
    options = [
        {
            "flags": ["--input", "-i"],
            "required": required,
            "default": "discarded-value",
            "help": "Synthetic input description.",
            "choices": ["discarded-choice"],
        },
        {
            "flags": ["--output", "-o"],
            "action": "store",
            "default": "another-discarded-value",
        },
    ]
    if include_json:
        options.append(
            {
                "flags": ["--json"],
                "action": "store_true",
                "default": False,
                "help": "Synthetic output toggle.",
            }
        )
    return [{"command": ["reports", "inspect"], "options": options}]


def test_normalization_is_order_independent_and_value_free() -> None:
    first = normalize_help_records(_surface())
    second = normalize_help_records(
        [
            {
                "options": list(reversed(_surface()[0]["options"])),
                "command": "reports inspect",
                "description": "Different discarded description.",
            }
        ]
    )

    assert first == second
    assert first.to_dict() == {
        "schema_version": "openmed.cli.help_drift.v1",
        "commands": [
            {
                "command": ["reports", "inspect"],
                "options": [
                    {
                        "flags": ["--input", "-i"],
                        "required": True,
                        "arity": "one",
                        "repeatable": False,
                    },
                    {
                        "flags": ["--output", "-o"],
                        "required": False,
                        "arity": "one",
                        "repeatable": False,
                    },
                ],
            }
        ],
    }
    assert first.digest == second.digest
    assert "discarded-value" not in first.to_json()
    assert "discarded-choice" not in first.to_json()


@pytest.mark.parametrize(
    ("candidate", "category", "exit_code", "field"),
    [
        (_surface(include_json=True), DriftCategory.ADDED, EXIT_ADDED, "added"),
        (_surface(required=False), DriftCategory.CHANGED, EXIT_CHANGED, "changed"),
        (
            [{"command": ["reports", "inspect"], "options": []}],
            DriftCategory.REMOVED,
            EXIT_REMOVED,
            "removed",
        ),
    ],
)
def test_option_drift_uses_deterministic_exit_categories(
    candidate: list[dict],
    category: DriftCategory,
    exit_code: int,
    field: str,
) -> None:
    report = compare_help_surfaces(_surface(), candidate)

    assert report.category is category
    assert report.exit_category is category
    assert report.exit_code == exit_code
    assert len(getattr(report, field)) == (1 if field != "removed" else 2)
    assert report.to_dict()["category"] == category.value


def test_mixed_option_and_command_drift_is_sorted_and_value_free() -> None:
    candidate = [
        {"command": ["new"], "options": [{"flags": ["--added"]}]},
        {
            "command": ["reports", "inspect"],
            "options": [
                {"flags": ["--output", "-o"], "nargs": "?"},
            ],
        },
    ]

    report = compare_help_surfaces(_surface(), candidate)

    assert report.category is DriftCategory.MIXED
    assert report.exit_code == EXIT_MIXED
    assert report.added_commands == (("new",),)
    assert [change.option for change in report.removed] == ["--input"]
    assert [change.option for change in report.added] == ["--added"]
    assert [change.option for change in report.changed] == ["--output"]
    assert "discarded-value" not in json.dumps(report.to_dict())


def test_clean_surfaces_have_zero_exit_code() -> None:
    report = compare_help_surfaces(_surface(), _surface())

    assert report.is_clean
    assert report.category is DriftCategory.CLEAN
    assert report.exit_code == EXIT_CLEAN
    assert report.added == ()
    assert report.removed == ()
    assert report.changed == ()


def test_invalid_shape_does_not_echo_input_values() -> None:
    with pytest.raises(HelpDriftError) as exc_info:
        normalize_help_records(
            [
                {
                    "command": "reports inspect",
                    "options": [
                        {
                            "flags": "not-an-option",
                            "default": "synthetic-sensitive-placeholder",
                        }
                    ],
                }
            ]
        )

    assert "synthetic-sensitive-placeholder" not in str(exc_info.value)


def test_json_cli_is_local_and_returns_category_code(tmp_path, capsys) -> None:
    baseline_path = tmp_path / "baseline.json"
    candidate_path = tmp_path / "candidate.json"
    baseline_path.write_text(json.dumps(_surface()), encoding="utf-8")
    candidate_path.write_text(json.dumps(_surface(include_json=True)), encoding="utf-8")

    exit_code = main([str(baseline_path), str(candidate_path)])
    output = json.loads(capsys.readouterr().out)

    assert exit_code == EXIT_ADDED
    assert output["exit_category"] == "added"
    assert output["added"][0]["option"] == "--json"
    assert "discarded-value" not in json.dumps(output)


def test_json_cli_reports_invalid_input_without_raw_error(tmp_path, capsys) -> None:
    baseline_path = tmp_path / "baseline.json"
    candidate_path = tmp_path / "candidate.json"
    baseline_path.write_text("[]", encoding="utf-8")
    candidate_path.write_text(
        json.dumps([{"command": "reports", "options": [{"flags": "bad flag"}]}]),
        encoding="utf-8",
    )

    exit_code = main([str(baseline_path), str(candidate_path)])
    captured = capsys.readouterr()

    assert exit_code == EXIT_INVALID
    assert captured.out == ""
    assert captured.err.strip() == "help surface input is invalid"
