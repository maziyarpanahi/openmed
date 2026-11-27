#!/usr/bin/env python3
"""Inventory language/locale PII routes without printing regexes or matches.

Run ``python scripts/native/regex_inventory.py [--json]`` from a source checkout.
Counts are per route (including its universal base), not unique regexes across
routes. Construct counts overlap when a pattern uses both kinds. ``plain`` means
neither lookaround nor backreferences, not proven Rust regex compatibility.
Standalone helper regexes and token-tag-gated code-mixed banks are out of scope.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from re import _parser

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from openmed.core.pii_entity_merger import PIIPattern
from openmed.core.pii_i18n import (
    INDIC_NER_LANGUAGES,
    LANGUAGE_PII_PATTERNS,
    LOCALE_PII_PATTERNS,
    NATIONAL_ID_ONLY_LANGUAGES,
    SUPPORTED_LANGUAGES,
    USER_SUPPLIED_MODEL_LANGUAGES,
    get_patterns_for_language,
)

CONSTRUCTS = ("plain", "lookaround", "backreference")


def classify_pattern(pattern: PIIPattern) -> list[str]:
    """Return construct codes, respecting escapes, comments and regex flags."""
    found = set()

    # shortcut: CPython's private parser, revisit if supported Python changes it.
    def visit(nodes):
        for op, argument in nodes:
            if op in {_parser.ASSERT, _parser.ASSERT_NOT}:
                found.add("lookaround")
                visit(argument[1])
            elif op == _parser.GROUPREF:
                found.add("backreference")
            elif op == _parser.GROUPREF_EXISTS:
                found.add("backreference")
                visit(argument[1])
                if argument[2] is not None:
                    visit(argument[2])
            elif op == _parser.SUBPATTERN:
                visit(argument[3])
            elif op == _parser.BRANCH:
                for branch in argument[1]:
                    visit(branch)
            elif op in {
                _parser.MAX_REPEAT,
                _parser.MIN_REPEAT,
                _parser.POSSESSIVE_REPEAT,
            }:
                visit(argument[2])
            elif op == _parser.ATOMIC_GROUP:
                visit(argument)

    visit(_parser.parse(pattern.pattern, pattern.flags))
    return sorted(found) if found else ["plain"]


def registered_patterns() -> dict[str, list[PIIPattern]]:
    """Load offline language and locale routes, including universal patterns."""
    languages = (
        SUPPORTED_LANGUAGES
        | NATIONAL_ID_ONLY_LANGUAGES
        | INDIC_NER_LANGUAGES
        | USER_SUPPLIED_MODEL_LANGUAGES
        | LANGUAGE_PII_PATTERNS.keys()
        | {key.split("_", 1)[0] for key in LOCALE_PII_PATTERNS}
    )
    return {
        route: get_patterns_for_language(route)
        for route in sorted(languages | LOCALE_PII_PATTERNS.keys())
    }


def inventory(banks: Mapping[str, Sequence[PIIPattern]]) -> dict:
    """Return only route/pattern identifiers and per-construct counts."""
    rows = []
    languages = {}
    totals = Counter(dict.fromkeys(CONSTRUCTS, 0))
    for language, patterns in sorted(banks.items()):
        counts = Counter(dict.fromkeys(CONSTRUCTS, 0))
        for index, pattern in enumerate(patterns):
            constructs = classify_pattern(pattern)
            counts.update(constructs)
            rows.append({"id": f"{language}:{index}", "constructs": constructs})
        languages[language] = {"total": len(patterns), **counts}
        totals.update(counts)
    return {
        "total": len(rows),
        "constructs": dict(totals),
        "languages": languages,
        "patterns": rows,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Print a value-free table by default, or the full summary with --json."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Print JSON instead.")
    args = parser.parse_args(argv)
    try:
        report = inventory(registered_patterns())
    except re.error:
        print("registered_pattern_invalid", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(report, sort_keys=True))
    else:
        print(
            f"{'language':<12} {'total':>7} {'plain':>7} {'lookaround':>11} {'backreference':>14}"
        )
        for language, counts in report["languages"].items():
            print(
                f"{language:<12} {counts['total']:>7} {counts['plain']:>7} "
                f"{counts['lookaround']:>11} {counts['backreference']:>14}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
