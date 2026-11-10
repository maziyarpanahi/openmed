#!/usr/bin/env python3
"""Lint README.md and docs/**/*.md for regulatory and certification claims.

Flagged phrases would present OpenMed as FDA-cleared, FDA-approved, CE-marked,
clinically validated, or certified for clinical use. A phrase is ignored when
its clause carries an allowlisted negation, such as "not a medical device",
"does not certify", or "not FDA-cleared". Clauses split on punctuation
``. ; : , ! ? | ( )`` and on the words and, but, while, whereas, and although,
so "not a medical device, but CE-marked" and "not a medical device and
CE-marked" both still fail. Exact (relative path, stripped line) pairs in
``ALLOWED_LINES`` skip negations that wrap onto another line.

Each violation prints as ``relative/path.md:LINE: phrase: text``, with the text
cut to 120 characters. The script exits 1 when any violation is found and 0
otherwise. It reads local files only and makes no network calls.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PHRASES = {
    "FDA-cleared": r"\bFDA[\s-]cleared\b",
    "FDA-approved": r"\bFDA[\s-]approved\b",
    "CE-marked": r"\bCE[\s-]marked\b",
    "clinically validated": r"\bclinically\s+validated\b",
    "certified for clinical use": r"\bcertified\s+for\s+clinical\s+use\b",
}
NEGATION = re.compile(
    r"\b(?:not\s+(?:an?\s+)?(?:medical\s+device|FDA[\s-](?:cleared|approved)"
    r"|CE[\s-]marked|clinically\s+validated|certified)"
    r"|does\s+not\s+(?:certify|make)|no\s+FDA\s+clearance)\b",
    re.IGNORECASE,
)
CLAUSE_BREAK = re.compile(
    r"[.;:,!?|()]|\b(?:and|but|while|whereas|although)\b", re.IGNORECASE
)
ALLOWED_LINES: tuple[tuple[str, str], ...] = (
    # Negated on the previous line: "not a released clinical SLM or ...".
    (
        "docs/clinical/summarization.md",
        "clinically validated summarizer. Summaries require qualified clinical review;",
    ),
    # Negated on the previous line: "not normal physiological ranges, ...".
    (
        "docs/multimodal/physiological-waveforms.md",
        "physiological ranges, clinically validated thresholds or universal device",
    ),
)


def find_violations(root: Path) -> list[str]:
    """Return a report line for each flagged phrase in the scanned Markdown.

    Args:
        root: Directory that holds ``README.md`` and ``docs/``.

    Returns:
        Lines formatted as ``relative/path.md:LINE: phrase: text``.
    """

    paths = sorted(p for p in (root / "docs").rglob("*.md") if p.is_file())
    if (root / "README.md").is_file():
        paths.append(root / "README.md")
    found: list[str] = []
    for path in paths:
        rel = path.relative_to(root).as_posix()
        lines = path.read_text(encoding="utf-8").splitlines()
        for number, line in enumerate(lines, start=1):
            text = line.strip()
            if (rel, text) in ALLOWED_LINES:
                continue
            clauses = [c for c in CLAUSE_BREAK.split(text) if not NEGATION.search(c)]
            for phrase, pattern in PHRASES.items():
                if any(re.search(pattern, c, re.IGNORECASE) for c in clauses):
                    found.append(f"{rel}:{number}: {phrase}: {text[:120]}")
    return found


def main(argv: list[str] | None = None) -> int:
    """Run the lint, print each violation, and return the exit status.

    Args:
        argv: Command-line arguments, or ``None`` to read ``sys.argv``.

    Returns:
        ``0`` when the scanned docs are clean, ``1`` when any violation exists.
    """

    parser = argparse.ArgumentParser(description="Lint docs for regulatory claims.")
    parser.add_argument("--root", type=Path, default=REPO_ROOT, help="repo root")
    args = parser.parse_args(argv)
    violations = find_violations(args.root.resolve())
    for violation in violations:
        print(violation)
    if violations:
        return 1
    print("Regulatory claim lint OK: no violations in README.md or docs/**/*.md.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
