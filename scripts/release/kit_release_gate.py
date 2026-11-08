#!/usr/bin/env python3
"""Build or verify the signed v4.0 OpenMedKit release packet offline.

The library lives in ``openmed.eval.kit_release``. Usage::

    python scripts/release/kit_release_gate.py \\
        --manifest kit-release-manifest.json --output kit-release-packet.json
    python scripts/release/kit_release_gate.py \\
        --verify kit-release-packet.json --expected-candidate <commit> \\
        --require-ready

Supply the signing key through ``OPENMED_KIT_RELEASE_KEY`` or
``--signing-key-file``. Output carries reason codes, counts and controlled
identifiers only; paths, file contents and key material are never printed.
The script makes no network call and does not tag, publish or promote.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# A direct ``python scripts/release/...`` invocation otherwise puts only
# ``scripts/release`` first on ``sys.path``; pin imports to this checkout.
_root_string = str(ROOT)
if _root_string in sys.path:
    sys.path.remove(_root_string)
sys.path.insert(0, _root_string)

from openmed.eval.kit_release import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
