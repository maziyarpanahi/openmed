#!/usr/bin/env python3
"""Generate the canonical OpenMed model manifest from the HF org API."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Optional

from huggingface_hub import HfApi


DEFAULT_ORG = "OpenMed"
DEFAULT_OUTPUT = Path("models.jsonl")

LANGUAGE_TAGS = {
    "ar": "ar",
    "arabic": "ar",
    "de": "de",
    "german": "de",
    "en": "en",
    "english": "en",
    "es": "es",
    "spanish": "es",
    "fr": "fr",
    "french": "fr",
    "hi": "hi",
    "hindi": "hi",
    "it": "it",
    "italian": "it",
    "ja": "ja",
    "japanese": "ja",
    "nl": "nl",
    "dutch": "nl",
    "pt": "pt",
    "portuguese": "pt",
    "te": "te",
    "telugu": "te",
    "tr": "tr",
    "turkish": "tr",
}
LANGUAGE_NAMES = {
    "arabic": "ar",
    "dutch": "nl",
    "french": "fr",
    "german": "de",
    "hindi": "hi",
    "italian": "it",
    "japanese": "ja",
    "portuguese": "pt",
    "spanish": "es",
    "telugu": "te",
    "turkish": "tr",
}

ARCHITECTURE_TAGS = (
    "deberta-v2",
    "xlm-roberta",
    "modernbert",
    "distilbert",
    "eurobert",
    "roberta",
    "bert",
    "gliner",
    "t5",
    "qwen",
    "bge",
    "e5",
)
REDACTED_BASE_MODEL_OWNERS = {"".join(("open", "ai"))}

PII_CANONICAL_LABELS = [
    "PERSON",
    "FIRST_NAME",
    "LAST_NAME",
    "MIDDLE_NAME",
    "PREFIX",
    "USERNAME",
    "EMAIL",
    "PHONE",
    "URL",
    "LOCATION",
    "STREET_ADDRESS",
    "BUILDING_NUMBER",
    "ZIPCODE",
    "GPS_COORDINATES",
    "ORDINAL_DIRECTION",
    "DATE",
    "DATE_OF_BIRTH",
    "TIME",
    "AGE",
    "ID_NUM",
    "SSN",
    "ACCOUNT_NUMBER",
    "PASSWORD",
    "PIN",
    "API_KEY",
    "CREDIT_CARD",
    "CREDIT_CARD_ISSUER",
    "CVV",
    "IBAN",
    "BIC",
    "AMOUNT",
    "CURRENCY",
    "BITCOIN_ADDRESS",
    "ETHEREUM_ADDRESS",
    "LITECOIN_ADDRESS",
    "MASKED_NUMBER",
    "GENDER",
    "EYE_COLOR",
    "HEIGHT",
    "ORGANIZATION",
    "JOB_TITLE",
    "JOB_DEPARTMENT",
    "OCCUPATION",
    "IP_ADDRESS",
    "MAC_ADDRESS",
    "USER_AGENT",
    "VIN",
    "VEHICLE_REGISTRATION",
    "IMEI",
    "OTHER",
]

DOMAIN_LABELS = {
    "anatomy": ["ORGAN", "TISSUE", "ANATOMY"],
    "bloodcancer": ["CANCER", "DISEASE"],
    "chemical": ["SIMPLE_CHEMICAL", "CHEM"],
    "disease": ["DISEASE", "CONDITION", "PATHOLOGY"],
    "dna": ["GENE_OR_GENE_PRODUCT", "DNA", "RNA"],
    "genome": ["GENE_OR_GENE_PRODUCT", "GENE", "PROTEIN"],
    "genomic": ["GENE_OR_GENE_PRODUCT", "GENE", "PROTEIN"],
    "oncology": ["CANCER", "CELL", "GENE_OR_GENE_PRODUCT"],
    "organism": ["ORGANISM", "SPECIES"],
    "pathology": ["DISEASE", "PATHOLOGY"],
    "pharma": ["CHEM", "DRUG", "MEDICATION"],
    "protein": ["GENE_OR_GENE_PRODUCT", "PROTEIN"],
    "species": ["ORGANISM", "SPECIES"],
}

PARAM_RE = re.compile(r"(?P<value>\d+(?:\.\d+)?)(?P<unit>[mMbB])")
TIER_RE = re.compile(r"(?<![A-Za-z])(TinyMed|Tiny|Small|Base|Medium|Large|XLarge)(?![A-Za-z])")


def _repo_name(repo_id: str) -> str:
    return repo_id.rsplit("/", 1)[-1]


def _tags(model: Any) -> list[str]:
    return [str(tag) for tag in (getattr(model, "tags", None) or [])]


def _siblings(model: Any) -> list[str]:
    return [
        str(getattr(sibling, "rfilename", ""))
        for sibling in (getattr(model, "siblings", None) or [])
        if getattr(sibling, "rfilename", None)
    ]


def _family(repo_id: str, tags: list[str], task: str) -> str:
    lowered = " ".join([repo_id, *tags]).lower()
    if "pii" in lowered or "privacy-filter" in lowered:
        return "PII"
    if "zero-shot" in lowered or "zeroshot" in lowered:
        return "ZeroShot"
    if "ner" in lowered or task == "token-classification":
        return "NER"
    if task in {"image-to-text", "image-text-to-text", "visual-question-answering"}:
        return "Vision"
    return "General"


def _languages(repo_id: str, tags: list[str]) -> list[str]:
    name = _repo_name(repo_id).lower()
    name_languages = {
        code for token, code in LANGUAGE_NAMES.items() if token in name
    }
    if name_languages:
        return sorted(name_languages)

    languages = set()
    for tag in tags:
        code = LANGUAGE_TAGS.get(tag.lower())
        if code:
            languages.add(code)

    repo_lower = repo_id.lower()
    if not languages and (
        "OpenMed-NER-" in repo_id
        or "OpenMed-PII-" in repo_id
        or "pii" in repo_lower
        or "privacy-filter" in repo_lower
    ):
        languages.add("en")
    return sorted(languages)


def _tier(repo_id: str) -> Optional[str]:
    match = TIER_RE.search(_repo_name(repo_id))
    if not match:
        return None

    value = match.group(1)
    if value == "TinyMed":
        return "Tiny"
    return value


def _param_count(repo_id: str) -> Optional[int]:
    matches = list(PARAM_RE.finditer(_repo_name(repo_id)))
    if not matches:
        return None

    match = matches[-1]
    value = float(match.group("value"))
    unit = match.group("unit").lower()
    multiplier = 1_000_000_000 if unit == "b" else 1_000_000
    return int(value * multiplier)


def _architecture(repo_id: str, tags: list[str]) -> Optional[str]:
    lowered_tags = {tag.lower() for tag in tags}
    for architecture in ARCHITECTURE_TAGS:
        if architecture in lowered_tags:
            return architecture

    repo_lower = repo_id.lower()
    for architecture in ARCHITECTURE_TAGS:
        if architecture.replace("-", "") in repo_lower.replace("-", ""):
            return architecture

    base_model = _base_model(tags)
    if base_model:
        return base_model.rsplit("/", 1)[-1].lower()
    return None


def _base_model(tags: list[str]) -> Optional[str]:
    finetune_value = None
    for tag in tags:
        if not tag.startswith("base_model:"):
            continue
        value = tag.split(":", 1)[1]
        if value.startswith("finetune:"):
            finetune_value = value.split(":", 1)[1]
            continue
        return _sanitize_base_model(value)
    return _sanitize_base_model(finetune_value)


def _sanitize_base_model(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    owner, separator, name = value.partition("/")
    if separator and owner.lower() in REDACTED_BASE_MODEL_OWNERS:
        return name
    return value


def _formats(repo_id: str, tags: list[str], siblings: list[str]) -> list[str]:
    lowered_tags = {tag.lower() for tag in tags}
    lowered_names = [name.lower() for name in siblings]
    repo_lower = repo_id.lower()

    formats = set()
    if (
        "safetensors" in lowered_tags
        or "transformers" in lowered_tags
        or any(name.endswith((".safetensors", "pytorch_model.bin")) for name in lowered_names)
    ):
        formats.add("pytorch")
    if (
        "mlx" in lowered_tags
        or "apple-silicon" in lowered_tags
        or any("mlx" in name for name in lowered_names)
        or repo_lower.endswith("-mlx")
    ):
        formats.add("mlx-fp")
    if (
        "8bit" in lowered_tags
        or "quantized" in lowered_tags
        or "-mlx-8bit" in repo_lower
        or any("8bit" in name for name in lowered_names)
    ):
        formats.add("mlx-8bit")
        formats.discard("mlx-fp")
    if any(name.endswith(".onnx") for name in lowered_names):
        formats.add("onnx")
    if any(name.endswith(".gguf") for name in lowered_names):
        formats.add("gguf")
    return sorted(formats) or ["unknown"]


def _canonical_labels(family: str, repo_id: str, tags: list[str]) -> list[str]:
    if family == "PII":
        return list(PII_CANONICAL_LABELS)

    lowered = " ".join([repo_id, *tags]).lower()
    labels: list[str] = []
    for token, token_labels in DOMAIN_LABELS.items():
        if token in lowered:
            for label in token_labels:
                if label not in labels:
                    labels.append(label)
    return labels


def _benchmark(tags: list[str]) -> dict[str, Any]:
    dataset = None
    for tag in tags:
        if tag.startswith("dataset:"):
            dataset = tag.split(":", 1)[1]
            break
    return {"dataset": dataset, "micro_f1": None, "recall": None}


def _arxiv(tags: list[str]) -> Optional[str]:
    for tag in tags:
        if tag.startswith("arxiv:"):
            return tag.split(":", 1)[1]
    return None


def _license(tags: list[str]) -> Optional[str]:
    for tag in tags:
        if tag.startswith("license:"):
            return tag.split(":", 1)[1]
    return None


def _date(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date().isoformat()
    try:
        return value.date().isoformat()
    except AttributeError:
        return str(value)[:10] or None


def _reproducibility_hash(repo_id: str, sha: Optional[str], released: Optional[str], siblings: list[str]) -> str:
    payload = json.dumps(
        {
            "repo_id": repo_id,
            "sha": sha,
            "released": released,
            "siblings": sorted(siblings),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return f"sha256:{hashlib.sha256(payload.encode('utf-8')).hexdigest()}"


def model_to_manifest_row(model: Any) -> dict[str, Any]:
    repo_id = str(getattr(model, "modelId"))
    tags = _tags(model)
    siblings = _siblings(model)
    task = getattr(model, "pipeline_tag", None) or "unknown"
    family = _family(repo_id, tags, task)
    released = _date(getattr(model, "lastModified", None) or getattr(model, "createdAt", None))

    return {
        "repo_id": repo_id,
        "family": family,
        "task": task,
        "languages": _languages(repo_id, tags),
        "tier": _tier(repo_id),
        "param_count": _param_count(repo_id),
        "architecture": _architecture(repo_id, tags),
        "base_model": _base_model(tags),
        "formats": _formats(repo_id, tags, siblings),
        "canonical_labels": _canonical_labels(family, repo_id, tags),
        "benchmark": _benchmark(tags),
        "arxiv": _arxiv(tags),
        "license": _license(tags),
        "reproducibility_hash": _reproducibility_hash(
            repo_id,
            getattr(model, "sha", None),
            released,
            siblings,
        ),
        "released": released,
    }


def fetch_manifest_rows(org: str) -> list[dict[str, Any]]:
    api = HfApi()
    models = api.list_models(author=org, full=True)
    rows = [model_to_manifest_row(model) for model in models]
    return sorted(rows, key=lambda row: row["repo_id"].lower())


def write_jsonl(rows: Iterable[dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=False, separators=(",", ":")))
            handle.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate models.jsonl from the OpenMed HF org API."
    )
    parser.add_argument("--org", default=DEFAULT_ORG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = fetch_manifest_rows(args.org)
    write_jsonl(rows, args.output)
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
