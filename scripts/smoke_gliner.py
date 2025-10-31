"""Quick smoke test runner for GLiNER zero-shot models.

This script requires the optional ``gliner`` dependency group. Execute via:

    python scripts/smoke_gliner.py --limit 2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional, Sequence

from openmed.ner import (
    Entity,
    NerRequest,
    TokenClassificationResult,
    infer,
    is_gliner_available,
    load_index,
    to_token_classification,
)
from openmed.ner.families import ModelFamily


DEFAULT_SAMPLE_TEXTS: Dict[str, str] = {
    "biomedical": "Imatinib inhibits BCR-ABL in chronic myeloid leukemia patients.",
    "clinical": "The patient received 5mg of warfarin daily after knee surgery.",
    "genomic": "BRCA1 variants were screened alongside KRAS mutations.",
    "finance": "Acme Corp reported record Q3 earnings after the merger announcement.",
    "legal": "The Supreme Court upheld the lower court's patent ruling.",
    "news": "NASA launched a new satellite to monitor climate change this week.",
    "ecommerce": "Customers praised the new smartwatch for its battery life.",
    "cybersecurity": "CVE-2024-12345 affects routers running outdated firmware.",
    "chemistry": "Sodium chloride readily dissolves in water forming ions.",
    "organism": "Escherichia coli colonies were observed in the culture sample.",
    "education": "The curriculum includes project-based modules on data science.",
    "social": "@openmed shared a thread about clinical NLP breakthroughs.",
    "public_health": "Vaccination drives reduced measles cases in the region by 40%.",
    "generic": "OpenMed partners announced global expansion plans in 2025.",
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run GLiNER smoke tests.")
    parser.add_argument(
        "--index",
        type=Path,
        help="Optional path to models/index.json (defaults to packaged index).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=3,
        help="Limit number of models to test (default 3).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.4,
        help="Confidence threshold to apply (default 0.4).",
    )
    parser.add_argument(
        "--adapter",
        action="store_true",
        help="Run token classification adapter after inference.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if not is_gliner_available():
        parser.error("GLiNER dependencies unavailable. Install with `pip install .[gliner]`.")

    try:
        index = load_index(args.index)
    except FileNotFoundError as exc:
        sys.stderr.write(str(exc) + "\n")
        return 1

    gliner_models = [
        record for record in index.models if record.family == ModelFamily.GLINER.value
    ]
    if not gliner_models:
        sys.stderr.write("No GLiNER models found in the index.\n")
        return 1

    selected = gliner_models[: max(1, args.limit)]

    for record in selected:
        domain = record.domains[0] if record.domains else "generic"
        sample_text = DEFAULT_SAMPLE_TEXTS.get(domain, DEFAULT_SAMPLE_TEXTS["generic"])
        request = NerRequest(
            model_id=record.id,
            text=sample_text,
            threshold=args.threshold,
            domain=domain,
        )
        sys.stdout.write(f"\n>>> Model: {record.id} (domain={domain})\n")
        try:
            response = infer(request, index=index)
        except Exception as exc:  # pragma: no cover - defensive
            sys.stderr.write(f"Inference failed for {record.id}: {exc}\n")
            continue

        if not response.entities:
            sys.stdout.write("No entities detected.\n")
            continue

        for entity in response.entities:
            _print_entity(entity)

        if args.adapter:
            adapter_result = to_token_classification(response.entities, request.text)
            _print_adapter(adapter_result)

    return 0


def _print_entity(entity: Entity) -> None:
    span = f"{entity.start}-{entity.end}"
    group = f" group={entity.group}" if entity.group else ""
    sys.stdout.write(
        f"- {entity.label}: '{entity.text}' [{span}] score={entity.score:.3f}{group}\n"
    )


def _print_adapter(result: TokenClassificationResult) -> None:
    sys.stdout.write("  Token labels (" + result.scheme + "):\n")
    for idx, annotation in enumerate(result.tokens):
        sys.stdout.write(
            f"    [{idx:02d}] {annotation.token!r} -> {annotation.label}\n"
        )


if __name__ == "__main__":  # pragma: no cover - manual invocation
    raise SystemExit(main())
