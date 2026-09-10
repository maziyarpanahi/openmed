"""Runnable Dagster definitions using a fully synthetic in-memory dataset."""

from dagster import Definitions

from openmed.integrations.dagster_assets import redacted_dataset

SYNTHETIC_SOURCE_DATASET = [
    {
        "record_id": "synthetic-001",
        "note": "Synthetic subject John Doe called 555-0101.",
    },
    {
        "record_id": "synthetic-002",
        "note": "Synthetic subject Jane Roe emailed jane@example.test.",
    },
]


defs = Definitions(
    assets=[redacted_dataset],
    resources={"source_dataset": SYNTHETIC_SOURCE_DATASET},
)
