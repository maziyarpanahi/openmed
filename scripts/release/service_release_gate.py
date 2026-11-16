#!/usr/bin/env python3
"""Build or verify the signed v4.0 secure-service release packet offline.

The library lives in ``openmed.eval.service_release``. Usage::

    python scripts/release/service_release_gate.py \\
        --write-failure-codes failure-codes.json --candidate <commit>
    python scripts/release/service_release_gate.py \\
        --manifest service-release-manifest.json \\
        --output service-release-packet.json
    python scripts/release/service_release_gate.py \\
        --verify service-release-packet.json --expected-candidate <commit> \\
        --expected-release 4.0.0 --require-ready

Supply the signing key through ``OPENMED_SERVICE_RELEASE_KEY`` or
``--signing-key-file``. Output carries reason codes, counts and issue
references only; paths, file contents and key material are never printed.
The script makes no network call and does not tag, publish or deploy.
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

from openmed.eval.service_release import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
