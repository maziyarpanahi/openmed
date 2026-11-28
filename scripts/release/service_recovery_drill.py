#!/usr/bin/env python3
"""Run the reference v4.0 service recovery drill offline.

The library lives in ``openmed.eval.service_recovery_drill``. Usage::

    python scripts/release/service_recovery_drill.py --output drill.json

The drill injects one storage, process, network and corrupted-state failure
into synthetic local state in a private temporary directory, runs the
recovery steps available on this checkout and records the others as
``not_available`` with their owning issue. The report holds codes, counts and
a self-digest only. Exit status is ``0`` only for a ``complete`` report.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Pin imports to this checkout, as the release gate script does.
_root_string = str(ROOT)
if _root_string in sys.path:
    sys.path.remove(_root_string)
sys.path.insert(0, _root_string)

from openmed.eval.service_recovery_drill import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
