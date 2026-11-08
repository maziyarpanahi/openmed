#!/usr/bin/env python3
"""In-boundary entry point for ``openmed airgap rehearse`` (internal).

``openmed airgap rehearse`` wraps this script with the no-egress gate's
``boundary_command`` so it runs in a Linux user + network namespace or a
container with ``--network none``. Inside, it:

1. loads ``scripts/security/no_egress_gate.py`` (stdlib only) and runs the
   gate's boundary inspection and every positive and negative network control
   before ``openmed`` is imported, so no in-process guard can mask a hole;
2. runs the lifecycle phases with
   :func:`openmed.cli.airgap_rehearsal.run_rehearsal_inside`;
3. runs the negative controls again, proving the boundary held throughout;
4. prints one value-free result line behind the gate's result marker.

It never prints paths, payloads, or exception text on stdout.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parents[2]
GATE_PATH = ROOT / "scripts" / "security" / "no_egress_gate.py"


def load_gate() -> ModuleType:
    """Load the no-egress gate; it puts this checkout first on ``sys.path``."""

    name = "openmed_airgap_no_egress_gate"
    spec = importlib.util.spec_from_file_location(name, GATE_PATH)
    if spec is None or spec.loader is None:
        raise SystemExit(1)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def build_parser() -> argparse.ArgumentParser:
    """Return the internal command-line parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--boundary-kind", choices=("netns", "container"), required=True
    )
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--fault-point", required=True)
    parser.add_argument("--hook", dest="hooks", action="append", default=[])
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the rehearsal inside the boundary and print one result line."""

    args = build_parser().parse_args(argv)
    gate = load_gate()
    result = gate.collect_boundary_evidence(args.boundary_kind)

    from openmed.cli.airgap_rehearsal import (  # noqa: E402 - after the probes
        parse_hook_specs,
        run_rehearsal_inside,
    )

    result["rehearsal"] = run_rehearsal_inside(
        args.bundle,
        fault_point=args.fault_point,
        hooks=parse_hook_specs(args.hooks),
        verbose=args.verbose,
    )
    result["negative_controls_after"] = gate.run_probe_set(
        gate.negative_probes(), gate.classify_negative
    )
    sys.stdout.write(gate.RESULT_MARKER + json.dumps(result, sort_keys=True) + "\n")
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
