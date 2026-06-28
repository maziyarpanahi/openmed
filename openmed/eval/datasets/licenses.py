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
    "golden": DatasetLicense(
        dataset="golden",
        license_id="Apache-2.0",
        source_url="openmed/eval/golden/fixtures",
        redistribution="committed synthetic fixtures only",
        notes="Synthetic-only fixtures; no real PHI and no DUA content.",
    ),
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
        redistribution="download-on-demand",
        notes=(
            "Zenodo DOI 10.5281/zenodo.4955411; adapter caches the public "
            "archive locally and stores no corpus rows in the repository."
        ),
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
        license_id="CC0-1.0",
        source_url="https://www.ncbi.nlm.nih.gov/research/bionlp/Data/disease/",
        redistribution="download-on-demand",
        notes=(
            "BigBIO metadata lists CC0-1.0; adapter loads public rows by "
            "reference and stores no corpus rows in the repository."
        ),
    ),
    "bc5cdr": DatasetLicense(
        dataset="bc5cdr",
        license_id="Public-Domain-Mark-1.0",
        source_url="https://huggingface.co/datasets/bigbio/bc5cdr",
        redistribution="download-on-demand",
        notes=(
            "BigBIO metadata lists PUBLIC_DOMAIN_MARK_1p0; adapter loads "
            "public rows by reference and stores no corpus rows in the "
            "repository."
        ),
    ),
    "jnlpba": DatasetLicense(
        dataset="jnlpba",
        license_id="CC-BY-3.0",
        source_url="https://huggingface.co/datasets/bigbio/jnlpba",
        redistribution="download-on-demand",
        notes=(
            "BigBIO metadata lists CC_BY_3p0; adapter loads public rows by "
            "reference and stores no corpus rows in the repository."
        ),
    ),
    "species_800": DatasetLicense(
        dataset="species_800",
        license_id="Medline-restrictions",
        source_url="https://huggingface.co/datasets/spyysalo/species_800",
        redistribution="download-on-demand",
        notes=(
            "Hugging Face dataset card lists license as unknown and notes "
            "Medline restrictions; adapter loads by reference and stores no "
            "corpus rows in the repository."
        ),
    ),
    "bc2gm": DatasetLicense(
        dataset="bc2gm",
        license_id="unknown-source-license",
        source_url="https://huggingface.co/datasets/bigbio/blurb",
        redistribution="download-on-demand",
        notes=(
            "BigBIO BLURB metadata lists license as other; source corpus "
            "licensing is not normalized to SPDX. Adapter loads public rows "
            "by reference and stores no corpus rows in the repository."
        ),
    ),
}


def license_for(dataset: str) -> DatasetLicense:
    try:
        return PUBLIC_DATASET_LICENSES[dataset]
    except KeyError as exc:
        raise ValueError(f"unknown public dataset: {dataset}") from exc


__all__ = ["DatasetLicense", "PUBLIC_DATASET_LICENSES", "license_for"]
