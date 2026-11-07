#!/usr/bin/env python3
"""Regenerate the synthetic CBLUE-shaped smoke fixtures.

Data provenance: every surface form below is composed from a closed, invented
vocabulary of enumerator morphemes (jia/yi/bing, used the way English uses
A/B/C) joined to generic shape words. No real clinical Chinese text is
reproduced and no CBLUE record is read, copied, or paraphrased. Character
offsets and per-character BIO tags are computed from the composed segments
rather than written by hand.

The generator is deterministic by construction: it draws no random numbers and
reads no clock, environment, or filesystem state, so there is no seed to pin
and repeated runs emit identical bytes. ``--check`` verifies that the
committed fixtures still match a fresh generation byte for byte, which is what
makes the provenance claim reproducible rather than merely asserted.

Usage::

    python scripts/benchmarks/generate_cblue_synthetic_fixtures.py
    python scripts/benchmarks/generate_cblue_synthetic_fixtures.py --check
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from openmed.eval.datasets.cblue import (
    CHIP_CDN,
    IMCS_V2_NER,
    synthetic_cblue_fixture_path,
)

DEFAULT_FIXTURE_DIR = synthetic_cblue_fixture_path(CHIP_CDN).parent

# Invented closed vocabulary. Enumerator morphemes only; none of these
# compounds is a real diagnosis, drug, examination, or procedure name.
ENUMERATORS = ("甲", "乙", "丙")
CORES = ("热", "痛", "肿")
ZONES = ("甲区", "乙区", "丙区")

_SYNTHETIC_METADATA = {"synthetic": True, "contains_real_phi": False}


def build_chip_cdn_rows() -> list[dict[str, Any]]:
    """Return CHIP-CDN rows: a raw mention plus ``##``-joined standard terms."""

    rows: list[dict[str, Any]] = []
    for index, (enumerator, core, zone) in enumerate(zip(ENUMERATORS, CORES, ZONES)):
        mention = f"{zone}{enumerator}型{core}症"
        standard_terms = [f"{enumerator}型{core}症", f"{zone}病变"]
        rows.append(
            {
                "id": f"cblue-chip-cdn-synthetic-{index + 1}",
                "text": mention,
                "normalized_result": "##".join(standard_terms),
                "language": "zh",
                "split": "synthetic",
                "metadata": dict(_SYNTHETIC_METADATA),
            }
        )
    return rows


def _dialogue_segments(index: int) -> list[tuple[str, str | None]]:
    """Return ``(surface, entity_type)`` segments for one dialogue turn."""

    enumerator = ENUMERATORS[index]
    core = CORES[index]
    zone = ZONES[index]
    segments: list[tuple[str, str | None]] = [
        ("服用", None),
        (f"{enumerator}司林", "Drug"),
        ("后", None),
        (f"{enumerator}型{core}症", "Symptom"),
        ("好转，复查", None),
        (f"{zone}影像检查", "Medical_Examination"),
        ("，安排", None),
        (f"{zone}切除术", "Operation"),
        ("。", None),
    ]
    if index == 0:
        # Cover the fifth source category exactly once.
        segments.insert(1, (f"{enumerator}类抗{core}药", "Drug_Category"))
        segments.insert(2, ("中的", None))
    return segments


def build_imcs_v2_ner_rows() -> list[dict[str, Any]]:
    """Return IMCS-V2-NER rows with computed per-character BIO tags."""

    rows: list[dict[str, Any]] = []
    for index in range(len(ENUMERATORS)):
        characters: list[str] = []
        tags: list[str] = []
        for surface, entity_type in _dialogue_segments(index):
            for position, character in enumerate(surface):
                characters.append(character)
                if entity_type is None:
                    tags.append("O")
                else:
                    prefix = "B" if position == 0 else "I"
                    tags.append(f"{prefix}-{entity_type}")
        if len(characters) != len(tags):  # pragma: no cover - construction invariant
            raise AssertionError("character and tag sequences must be parallel")
        rows.append(
            {
                "id": f"cblue-imcs-v2-ner-synthetic-{index + 1}",
                "sentence": characters,
                "BIO_label": tags,
                "language": "zh",
                "split": "synthetic",
                "metadata": dict(_SYNTHETIC_METADATA),
            }
        )
    return rows


def render_rows(rows: Sequence[dict[str, Any]]) -> str:
    """Serialize rows as canonical JSONL with stable key ordering."""

    return "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
        for row in rows
    )


def synthetic_fixture_payloads() -> dict[str, str]:
    """Return the exact file contents keyed by fixture filename."""

    return {
        synthetic_cblue_fixture_path(CHIP_CDN).name: render_rows(build_chip_cdn_rows()),
        synthetic_cblue_fixture_path(IMCS_V2_NER).name: render_rows(
            build_imcs_v2_ner_rows()
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_FIXTURE_DIR)
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify the committed fixtures match a fresh generation",
    )
    args = parser.parse_args(argv)

    drifted: list[str] = []
    for name, payload in sorted(synthetic_fixture_payloads().items()):
        path = args.output_dir / name
        if args.check:
            current = path.read_text(encoding="utf-8") if path.exists() else ""
            if current != payload:
                drifted.append(str(path))
            continue
        path.write_text(payload, encoding="utf-8")
        print(path)

    if drifted:
        for path_text in drifted:
            print(f"drifted: {path_text}")
        return 1
    if args.check:
        print("committed CBLUE synthetic fixtures match the generator")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
