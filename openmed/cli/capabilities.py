"""``openmed capabilities``: print the canonical capability descriptor."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from typing import Any

from ._output import emit


def add_capabilities_command(subparsers: argparse._SubParsersAction) -> None:
    """Register ``openmed capabilities``.

    Args:
        subparsers: The root ``openmed`` subparser action.
    """
    parser = subparsers.add_parser(
        "capabilities",
        help="Print the value-free capability descriptor and its digest.",
        description=(
            "Print which optional backends and surfaces are installed, the "
            "canonical schema versions and the policy profiles, with the "
            "descriptor digest. Reports only; always exits 0."
        ),
    )
    parser.set_defaults(handler=run_from_args)


def run_from_args(args: argparse.Namespace) -> int:
    """Emit the capability descriptor; a missing extra is reported, not failed.

    Args:
        args: Parsed arguments; ``--json`` selects the result envelope.

    Returns:
        Always ``0``.
    """
    from openmed.core.capabilities import capability_descriptor

    descriptor = capability_descriptor()
    return emit(args, descriptor, human=render_capabilities(descriptor))


def render_capabilities(descriptor: Mapping[str, Any]) -> str:
    """Render a capability descriptor as stable, human-readable text.

    Surfaces, backends, schema versions and policy profiles are listed in the
    descriptor's sorted order, so the output is the same for the same install.

    Args:
        descriptor: A descriptor from
            :func:`openmed.core.capabilities.capability_descriptor`.

    Returns:
        The rendered text, without a trailing newline.
    """

    def state(available: Any) -> str:
        return ("available" if available else "missing").ljust(9)

    lines = [
        f"OpenMed capabilities ({descriptor['schema']} "
        f"v{descriptor['schema_version']})",
        f"Digest: {descriptor['digest']}",
        "",
        "Surfaces:",
    ]
    lines.extend(
        f"  {state(item['available'])}  {item['name']}"
        for item in descriptor["surfaces"]
    )
    lines.extend(["", "Backends:"])
    width = max((len(str(item["name"])) for item in descriptor["backends"]), default=0)
    lines.extend(
        f"  {state(item['available'])}  {str(item['name']).ljust(width)}  "
        f"openmed[{item['extra']}]"
        for item in descriptor["backends"]
    )
    lines.extend(["", "Schema versions:"])
    width = max(
        (len(str(item["name"])) for item in descriptor["schema_versions"]), default=0
    )
    lines.extend(
        f"  {str(item['name']).ljust(width)}  {item['version']}"
        for item in descriptor["schema_versions"]
    )
    lines.extend(["", "Policy profiles:"])
    lines.extend(f"  {name}" for name in descriptor["policy_profiles"])
    served = descriptor["served_models"]
    lines.extend(["", "Served models:" if served else "Served models: none"])
    lines.extend(f"  {name}" for name in served)
    return "\n".join(lines)


__all__ = ["add_capabilities_command", "render_capabilities", "run_from_args"]
