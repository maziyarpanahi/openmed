"""License metadata for dataset adapter declarations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class DatasetLicense:
    dataset: str
    license_id: str
    source_url: str
    redistribution: str
    notes: str = ""

    def to_dict(self) -> dict[str, str]:
        return {
            "dataset": self.dataset,
            "license_id": self.license_id,
            "source_url": self.source_url,
            "redistribution": self.redistribution,
            "notes": self.notes,
        }


PUBLIC_DATASET_LICENSES: Mapping[str, DatasetLicense] = {
    "shield": DatasetLicense(
        dataset="shield",
        license_id="access-controlled-full; public-sample",
        source_url="https://huggingface.co/datasets/tds-research-tech/shield-sample",
        redistribution="reference-only",
        notes="Public sample can be loaded by reference; full dataset requires access approval.",
    ),
    "drugprot": DatasetLicense(
        dataset="drugprot",
        license_id="CC-BY-4.0",
        source_url="https://zenodo.org/records/4955411",
        redistribution="reference-only",
        notes="Adapter stores no corpus rows and expects a local user-provided path.",
    ),
    "medmentions": DatasetLicense(
        dataset="medmentions",
        license_id="CC0-1.0",
        source_url="https://github.com/chanzuckerberg/MedMentions",
        redistribution="reference-only",
        notes="Adapter strips controlled-vocabulary identifiers from loaded metadata.",
    ),
    "ncbi_disease": DatasetLicense(
        dataset="ncbi_disease",
        license_id="public-research-corpus",
        source_url="https://www.ncbi.nlm.nih.gov/research/bionlp/Data/disease/",
        redistribution="reference-only",
        notes="Adapter stores no corpus rows and expects a local user-provided path.",
    ),
    "bc5cdr": DatasetLicense(
        dataset="bc5cdr",
        license_id="public-domain-us-government-work",
        source_url="https://pubmed.ncbi.nlm.nih.gov/27161011/",
        redistribution="reference-only",
        notes="Adapter stores no corpus rows and expects a local user-provided path.",
    ),
}


def license_for(dataset: str) -> DatasetLicense:
    try:
        return PUBLIC_DATASET_LICENSES[dataset]
    except KeyError as exc:
        raise ValueError(f"unknown public dataset: {dataset}") from exc


__all__ = ["DatasetLicense", "PUBLIC_DATASET_LICENSES", "license_for"]
