"""Coherence checks for the argparse command help surface."""

from __future__ import annotations

import argparse

import pytest

from openmed.cli.main import build_parser


def _command_paths(
    parser: argparse.ArgumentParser,
    prefix: tuple[str, ...] = (),
) -> list[tuple[tuple[str, ...], str]]:
    paths: list[tuple[tuple[str, ...], str]] = []

    for action in parser._actions:
        if not isinstance(action, argparse._SubParsersAction):
            continue

        help_by_name = {choice.dest: choice.help for choice in action._choices_actions}
        for name, child_parser in action.choices.items():
            path = (*prefix, name)
            paths.append((path, help_by_name.get(name, "")))
            paths.extend(_command_paths(child_parser, path))

    return paths


def test_every_subcommand_has_help_text() -> None:
    command_paths = _command_paths(build_parser())

    assert command_paths
    missing = [" ".join(path) for path, help_text in command_paths if not help_text]
    assert not missing, f"Subcommands without help text: {missing}"


@pytest.mark.parametrize(
    "command_path",
    [path for path, _ in _command_paths(build_parser())],
    ids=lambda path: " ".join(path),
)
def test_every_command_path_prints_help(
    command_path: tuple[str, ...],
    capsys: pytest.CaptureFixture[str],
) -> None:
    parser = build_parser()

    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args([*command_path, "--help"])

    assert exc_info.value.code == 0
    output = capsys.readouterr().out
    assert output.strip()
    assert output.startswith(f"usage: openmed {' '.join(command_path)}")


def test_top_level_help_lists_every_subcommand() -> None:
    parser = build_parser()
    help_text = parser.format_help()
    normalized_help = " ".join(help_text.split())

    top_level_commands = [
        (path[0], description)
        for path, description in _command_paths(parser)
        if len(path) == 1
    ]
    for name, description in top_level_commands:
        assert name in normalized_help
        assert " ".join(description.split()) in normalized_help
