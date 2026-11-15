"""Run from the source checkout: python -m examples.multimodal.ambient_reference."""

import json

from openmed.multimodal.reference_workflow import run_reference_workflow
from tests.fixtures.multimodal.reference_workflow import reference_fixture


def main():
    """Print only content-free receipts from fresh synthetic fixtures."""
    inputs, _ = reference_fixture()
    report = run_reference_workflow(inputs)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["reached_review"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
