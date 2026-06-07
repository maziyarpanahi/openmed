#!/usr/bin/env python3
"""Batch PII extraction and de-identification with OpenMed."""

from __future__ import annotations

from openmed import BatchProcessor


TEXTS = [
    "Patient John Doe, DOB 01/15/1970, phone (555) 123-4567.",
    "Jane Roe emailed jane.roe@example.org from Boston.",
    "MRN 4471882 belongs to Maria Garcia at 1200 Pine Street.",
]


def print_entities() -> None:
    processor = BatchProcessor(
        operation="extract_pii",
        model_name="pii_detection",
        batch_size=8,
        confidence_threshold=0.5,
        use_smart_merging=True,
    )
    result = processor.process_texts(TEXTS, ids=["note-1", "note-2", "note-3"])

    print("PII extraction")
    print("=" * 40)
    for item in result.items:
        if not item.success:
            print(f"{item.id}: failed: {item.error}")
            continue

        print(item.id)
        for entity in item.result.entities:
            print(f"  {entity.label:18s} {entity.text!r}")


def print_deidentified() -> None:
    processor = BatchProcessor(
        operation="deidentify",
        model_name="pii_detection",
        batch_size=8,
        method="replace",
        consistent=True,
        seed=42,
        confidence_threshold=0.7,
    )
    result = processor.process_texts(TEXTS, ids=["note-1", "note-2", "note-3"])

    print()
    print("De-identification")
    print("=" * 40)
    for item in result.items:
        if not item.success:
            print(f"{item.id}: failed: {item.error}")
            continue
        print(f"{item.id}: {item.result.deidentified_text}")


def main() -> None:
    print_entities()
    print_deidentified()


if __name__ == "__main__":
    main()
