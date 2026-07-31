"""License metadata and gates for datasets and user-supplied terminology."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from openmed.core.terminology_licenses import (
    TERMINOLOGY_REDISTRIBUTION_PERMITTED,
    TERMINOLOGY_REDISTRIBUTION_RESTRICTED,
    TERMINOLOGY_REDISTRIBUTION_VALUES,
    RestrictedTerminologyLocationError,
    TerminologyLicense,
    validate_terminology_source_path,
)


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


@dataclass(frozen=True)
class EncoderLicense:
    """License and provenance policy for an optional encoder backbone."""

    family: str
    license_id: str
    source_url: str
    redistribution: str
    notes: str = ""

    def to_dict(self) -> dict[str, str]:
        return {
            "family": self.family,
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
    "naamapadam": DatasetLicense(
        dataset="naamapadam",
        license_id="CC0-1.0",
        source_url="https://huggingface.co/datasets/ai4bharat/naamapadam",
        redistribution="reference-only",
        notes=(
            "AI4Bharat publishes the dataset packaging under CC0-1.0. OpenMed "
            "loads user-supplied copies by reference, commits original synthetic "
            "fixtures only, and redistributes no Naamapadam corpus rows."
        ),
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
    "masakhaner": DatasetLicense(
        dataset="masakhaner",
        license_id="CC-BY-NC-4.0",
        source_url="https://huggingface.co/datasets/masakhane/masakhaner2",
        redistribution="user-supplied only",
        notes=(
            "MasakhaNER 2.0 corpus data is non-commercial and is never bundled, "
            "downloaded, mirrored, or redistributed by OpenMed. Loading requires "
            "explicit license acceptance. The upstream repository and dataset-card "
            "prose declare non-commercial terms, while the current card metadata "
            "tags AFL-3.0; callers must verify the upstream terms for their use case."
        ),
    ),
    "cblue": DatasetLicense(
        dataset="cblue",
        license_id="CBLUE-access-controlled",
        source_url="https://tianchi.aliyun.com/dataset/95414",
        redistribution="user-supplied",
        notes=(
            "CBLUE access and usage terms apply. OpenMed never downloads, "
            "caches, or redistributes the benchmark corpus."
        ),
    ),
    "cmeee": DatasetLicense(
        dataset="cmeee",
        license_id="CBLUE-access-controlled",
        source_url="https://tianchi.aliyun.com/dataset/95414",
        redistribution="user-supplied",
        notes=(
            "CMeEE is the CBLUE clinical NER task. Supply an authorized local "
            "copy through OPENMED_CMEEE_PATH."
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


PERMISSIVE_ENCODER_LICENSES: Mapping[str, EncoderLicense] = MappingProxyType(
    {
        "muril": EncoderLicense(
            family="MuRIL",
            license_id="Apache-2.0",
            source_url="https://huggingface.co/google/muril-base-cased",
            redistribution="user-supplied-reference-only",
            notes=(
                "OpenMed bundles no weights. MuRIL supports 17 Indian languages "
                "and transliterated text, including Hinglish/code-mixed paths."
            ),
        ),
        "indicbert": EncoderLicense(
            family="IndicBERT",
            license_id="MIT",
            source_url="https://huggingface.co/ai4bharat/indic-bert",
            redistribution="user-supplied-reference-only",
            notes="OpenMed bundles no weights; users resolve an approved repo or local path.",
        ),
    }
)


def license_for(dataset: str) -> DatasetLicense:
    try:
        return PUBLIC_DATASET_LICENSES[dataset]
    except KeyError as exc:
        raise ValueError(f"unknown dataset license: {dataset}") from exc


def encoder_license_for(family: str) -> EncoderLicense:
    """Return permissive license metadata for a supported Indic encoder."""

    if not isinstance(family, str):
        raise TypeError("encoder family must be a string")
    normalized = family.strip().casefold().replace("-", "").replace("_", "")
    try:
        return PERMISSIVE_ENCODER_LICENSES[normalized]
    except KeyError as exc:
        raise ValueError(f"unknown permissive encoder family: {family}") from exc


__all__ = [
    "DatasetLicense",
    "EncoderLicense",
    "PERMISSIVE_ENCODER_LICENSES",
    "PUBLIC_DATASET_LICENSES",
    "RestrictedTerminologyLocationError",
    "TERMINOLOGY_REDISTRIBUTION_PERMITTED",
    "TERMINOLOGY_REDISTRIBUTION_RESTRICTED",
    "TERMINOLOGY_REDISTRIBUTION_VALUES",
    "TerminologyLicense",
    "encoder_license_for",
    "license_for",
    "validate_terminology_source_path",
]
