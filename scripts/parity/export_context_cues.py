#!/usr/bin/env python3
"""Export the English ConText cue table for the clinical assertion vectors.

Writes ``tests/fixtures/parity/context_cues_en_v1.json`` from the live English
``ClinicalCueLexicon`` and the experiencer family/other cue lists, then
records the file's SHA-256 as ``cue_table_sha256`` in
``tests/fixtures/parity/clinical_assertion_v1.json``. ``--check`` writes
nothing and exits non-zero when either file is stale. The expected assertion
axes in the vectors are never rewritten here; they are reviewed by hand.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from openmed.clinical import experiencer  # noqa: E402
from openmed.clinical.lexicons.context_cues import (  # noqa: E402
    ClinicalCueLexicon,
    get_clinical_cue_lexicon,
)

PARITY_DIR = ROOT / "tests" / "fixtures" / "parity"
CUE_TABLE = PARITY_DIR / "context_cues_en_v1.json"
VECTORS = PARITY_DIR / "clinical_assertion_v1.json"


def build_cue_table() -> dict[str, Any]:
    """Return the English cue table as sorted, JSON-ready data."""

    lexicon = get_clinical_cue_lexicon("en")
    entries: dict[str, Any] = {}
    for item in fields(ClinicalCueLexicon):
        value = getattr(lexicon, item.name)
        entries[item.name] = sorted(value) if isinstance(value, tuple) else value
    return {
        "experiencer": {
            "family_cues": sorted(experiencer._FAMILY_CUES),
            "other_cues": sorted(experiencer._OTHER_CUES),
        },
        "experiencer_source": "openmed.clinical.experiencer",
        "language": "en",
        "lexicon": entries,
        "lexicon_source": (
            "openmed.clinical.lexicons.context_cues.ENGLISH_CONTEXT_LEXICON"
        ),
        "schema": "openmed.tests.context_cues",
        "version": 1,
    }


def render_json(document: Any) -> str:
    """Return the canonical text of a parity fixture."""

    return json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def cue_table_text() -> str:
    """Return the canonical cue table text."""

    return render_json(build_cue_table())


def cue_table_digest(text: str) -> str:
    """Return the SHA-256 hex digest of cue table text as UTF-8 bytes."""

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def vectors_text(cue_text: str) -> str:
    """Return the vector file text with ``cue_table_sha256`` refreshed."""

    document = json.loads(VECTORS.read_text(encoding="utf-8"))
    document["cue_table_sha256"] = cue_table_digest(cue_text)
    return render_json(document)


def main(argv: Sequence[str] | None = None) -> int:
    """Write or check the cue table and the vector digest."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    cue_text = cue_table_text()
    vector_text = vectors_text(cue_text)
    if args.check:
        stale = [
            path.name
            for path, text in ((CUE_TABLE, cue_text), (VECTORS, vector_text))
            if not path.is_file() or path.read_text(encoding="utf-8") != text
        ]
        if stale:
            print("stale: " + ", ".join(stale), file=sys.stderr)
            return 1
        print("context cue table is current")
        return 0
    CUE_TABLE.write_text(cue_text, encoding="utf-8")
    VECTORS.write_text(vector_text, encoding="utf-8")
    print(f"wrote {CUE_TABLE.name} and refreshed {VECTORS.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
