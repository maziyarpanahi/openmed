"""Ground a synthetic mention offline and build a FHIR CodeableConcept."""

from __future__ import annotations

import hashlib
from pathlib import Path

from openmed import ground
from openmed.clinical.exporters import to_codeable_concept
from openmed.clinical.grounding import VocabLoader, VocabSource


def main() -> None:
    """Run the deterministic synthetic grounding example."""

    root = Path(__file__).resolve().parents[1]
    fixture = root / "openmed/eval/golden/fixtures/grounding_vocab_synthetic.jsonl"
    cache_dir = root / ".openmed-grounding-example"
    digest = hashlib.sha256(fixture.read_bytes()).hexdigest()
    loader = VocabLoader(
        cache_dir=cache_dir,
        local_only=True,
        registry={
            "icd10cm": VocabSource(
                system="icd10cm",
                path=fixture,
                sha256=digest,
                version="synthetic-fixture-1",
            )
        },
    )
    result = ground(
        "type 2 diabetes",
        systems=["icd10cm"],
        loader=loader,
        offline=True,
    )[0]
    print(result.to_dict())
    print(to_codeable_concept(result))


if __name__ == "__main__":
    main()
