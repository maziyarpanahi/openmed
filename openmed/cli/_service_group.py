"""Shared ``openmed service`` command group.

Several modules contribute subcommands (``preflight``, ``diagnose``,
``verify-bundle``) to one ``openmed service`` group. Argparse refuses a second
parser with the same name, so every contributor registers through
:func:`service_commands`, which creates the group once.
"""

from __future__ import annotations

import argparse

SERVICE_COMMAND = "service"
_COMMANDS_ATTRIBUTE = "_openmed_service_commands"


def service_commands(
    subparsers: argparse._SubParsersAction,
) -> argparse._SubParsersAction:
    """Return the ``openmed service`` subcommand collection, creating it once.

    Args:
        subparsers: The root subparser collection.

    Returns:
        The subparser collection that ``openmed service`` subcommands join.
    """
    parser = subparsers.choices.get(SERVICE_COMMAND)
    if parser is None:
        parser = subparsers.add_parser(
            SERVICE_COMMAND,
            help="Check service security and collect diagnostics without PHI.",
        )
        commands = parser.add_subparsers(dest="service_command", required=True)
        setattr(parser, _COMMANDS_ATTRIBUTE, commands)
    return getattr(parser, _COMMANDS_ATTRIBUTE)


__all__ = ["SERVICE_COMMAND", "service_commands"]
