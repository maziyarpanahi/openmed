#!/usr/bin/env python3
"""Run the fail-closed v3 Journey release gate from a source checkout."""

from openmed.eval.journey_release import main

if __name__ == "__main__":
    raise SystemExit(main())
