"""Model registry derived from the committed OpenMed model manifest."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


@dataclass
class ModelInfo:
    """Information about an OpenMed model."""

    model_id: str
    display_name: str
    category: str
    specialization: str
    description: str
    entity_types: List[str]
    size_category: str
    recommended_confidence: float = 0.60
    family: str = "Unknown"
    task: str = "unknown"
    languages: List[str] = field(default_factory=list)
    tier: Optional[str] = None
    param_count: Optional[int] = None
    architecture: Optional[str] = None
    base_model: Optional[str] = None
    formats: List[str] = field(default_factory=list)
    benchmark: Dict[str, Any] | List[Dict[str, Any]] = field(default_factory=dict)
    latency_ms: Dict[str, float] = field(default_factory=dict)
    peak_ram_mb: Dict[str, float] = field(default_factory=dict)
    recommended_tier: Optional[str] = None
    arxiv: Optional[str] = None
    license: Optional[str] = None
    reproducibility_hash: Optional[str] = None
    released: Optional[str] = None

    @property
    def size_mb(self) -> Optional[int]:
        """Extract estimated size in millions of parameters from metadata/name."""
        if self.param_count:
            return round(self.param_count / 1_000_000)

        patterns = (
            ("33M", 33),
            ("44M", 44),
            ("60M", 60),
            ("65M", 65),
            ("66M", 66),
            ("82M", 82),
            ("108M", 108),
            ("109M", 109),
            ("110M", 110),
            ("125M", 125),
            ("135M", 135),
            ("141M", 141),
            ("149M", 149),
            ("166M", 166),
            ("184M", 184),
            ("209M", 209),
            ("210M", 210),
            ("212M", 212),
            ("220M", 220),
            ("278M", 278),
            ("279M", 279),
            ("335M", 335),
            ("340M", 340),
            ("355M", 355),
            ("395M", 395),
            ("434M", 434),
            ("459M", 459),
            ("560M", 560),
            ("568M", 568),
            ("600M", 600),
            ("770M", 770),
        )
        for token, value in patterns:
            if token in self.model_id:
                return value
        return None


MANIFEST_PATH = Path(__file__).resolve().parents[2] / "models.jsonl"

_SLUG_RE = re.compile(r"[^a-z0-9]+")
_PARAM_RE = re.compile(r"^\d+(?:\.\d+)?[mb]$", re.IGNORECASE)
_VERSION_RE = re.compile(r"^v\d+$", re.IGNORECASE)

_LANGUAGE_NAME_TO_CODE = {
    "arabic": "ar",
    "dutch": "nl",
    "english": "en",
    "french": "fr",
    "german": "de",
    "hebrew": "he",
    "hindi": "hi",
    "italian": "it",
    "japanese": "ja",
    "portuguese": "pt",
    "spanish": "es",
    "telugu": "te",
    "turkish": "tr",
}
_LOCALIZED_PII_LANGUAGE_KEYS = {
    code for code in _LANGUAGE_NAME_TO_CODE.values() if code != "en"
}
_LANGUAGE_CONFIG = {
    code: {"name": name.title(), "prefix": f"{name.title()}-"}
    for name, code in _LANGUAGE_NAME_TO_CODE.items()
    if code not in {"en", "nl", "hi", "te", "pt", "ar", "ja", "tr"}
}

_PII_ENTITY_TYPES = [
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

_CATEGORY_ENTITY_TYPES = {
    "Disease": ["DISEASE", "CONDITION", "PATHOLOGY"],
    "Pharmaceutical": ["CHEM", "DRUG", "MEDICATION"],
    "Oncology": ["CANCER", "CELL", "GENE_OR_GENE_PRODUCT"],
    "Anatomy": ["ORGAN", "TISSUE", "ANATOMY"],
    "Genomics": ["GENE_OR_GENE_PRODUCT", "GENE", "PROTEIN", "DNA", "RNA"],
    "Chemical": ["SIMPLE_CHEMICAL", "CHEM"],
    "Species": ["ORGANISM", "SPECIES"],
    "Microbiology": ["MICROORGANISM", "ANTIBIOTIC", "SUSCEPTIBILITY"],
    "Protein": ["GENE_OR_GENE_PRODUCT", "PROTEIN"],
    "Pathology": ["DISEASE", "PATHOLOGY"],
    "Hematology": ["CANCER", "DISEASE"],
    # Forward metadata for future Cardiology models; no Cardiology model is
    # registered today (see issue #317).
    "Cardiology": [
        "CARDIAC_FINDING",
        "ECG_FINDING",
        "EJECTION_FRACTION",
        "CARDIAC_PROCEDURE",
        "CARDIAC_DEVICE",
        "ANATOMY",
    ],
    # Forward metadata for future Dermatology/Ophthalmology models; no such
    # model is registered today (see issue #318).
    "Dermatology": ["SKIN_LESION", "MORPHOLOGY", "DISTRIBUTION", "ANATOMY"],
    "Ophthalmology": [
        "EYE_FINDING",
        "VISUAL_ACUITY",
        "INTRAOCULAR_PRESSURE",
        "ANATOMY",
    ],
    "Privacy": _PII_ENTITY_TYPES,
}

_LEGACY_MODEL_ALIASES = {
    "OpenMed/OpenMed-NER-DiseaseDetect-SuperClinical-434M": [
        "disease_detection_superclinical"
    ],
    "OpenMed/OpenMed-NER-DiseaseDetect-TinyMed-135M": ["disease_detection_tiny"],
    "OpenMed/OpenMed-NER-PharmaDetect-SuperClinical-434M": [
        "pharma_detection_superclinical"
    ],
    "OpenMed/OpenMed-NER-PharmaDetect-SuperMedical-125M": [
        "pharma_detection_supermedical"
    ],
    "OpenMed/OpenMed-NER-OncologyDetect-SuperClinical-434M": [
        "oncology_detection_superclinical"
    ],
    "OpenMed/OpenMed-NER-OncologyDetect-TinyMed-65M": ["oncology_detection_tiny"],
    "OpenMed/OpenMed-NER-AnatomyDetect-ElectraMed-109M": [
        "anatomy_detection_electramed"
    ],
    "OpenMed/OpenMed-NER-GenomeDetect-BioClinical-108M": [
        "genome_detection_bioclinical"
    ],
    "OpenMed/OpenMed-NER-ChemicalDetect-PubMed-335M": ["chemical_detection_pubmed"],
    "OpenMed/OpenMed-NER-SpeciesDetect-BioClinical-108M": [
        "species_detection_bioclinical"
    ],
    "OpenMed/OpenMed-NER-ProteinDetect-PubMed-109M": ["protein_detection_pubmed"],
    "OpenMed/OpenMed-NER-PathologyDetect-ModernClinical-395M": [
        "pathology_detection_modern"
    ],
    "OpenMed/OpenMed-NER-BloodCancerDetect-SuperClinical-434M": [
        "blood_cancer_detection"
    ],
    "OpenMed/OpenMed-NER-DNADetect-SuperMedical-125M": ["dna_detection_supermedical"],
    "OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1": ["pii_detection"],
}


def load_manifest_rows(path: Path = MANIFEST_PATH) -> List[Dict[str, Any]]:
    """Load model manifest rows from the committed JSONL snapshot."""
    if not path.exists():
        return []

    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:  # pragma: no cover - import guard
                raise ValueError(
                    f"Invalid JSON in {path} line {line_number}: {exc}"
                ) from exc
            rows.append(row)
    return rows


def _slug(value: str) -> str:
    slug = _SLUG_RE.sub("_", value.lower()).strip("_")
    return slug or "model"


def _repo_name(repo_id: str) -> str:
    return repo_id.rsplit("/", 1)[-1]


def _split_repo_tokens(repo_id: str) -> List[str]:
    return [token for token in re.split(r"[-_]+", _repo_name(repo_id)) if token]


def _clean_model_tokens(tokens: Iterable[str]) -> List[str]:
    cleaned: List[str] = []
    for token in tokens:
        lowered = token.lower()
        if lowered in {"openmed", "ner", "pii"}:
            continue
        if _VERSION_RE.match(lowered):
            continue
        cleaned.append(token)
    return cleaned


def _category_from_row(row: Dict[str, Any]) -> str:
    repo = row.get("repo_id", "").lower()
    family = str(row.get("family") or "").lower()

    if family == "pii" or "pii" in repo or "privacy-filter" in repo:
        return "Privacy"
    if "bloodcancerdetect" in repo or "hematology" in repo or "leukemia" in repo:
        return "Hematology"
    if "diseasedetect" in repo:
        return "Disease"
    if "pharmadetect" in repo or "drug" in repo:
        return "Pharmaceutical"
    if "oncologydetect" in repo or "cancer" in repo:
        return "Oncology"
    if "anatomydetect" in repo or "anatomy" in repo:
        return "Anatomy"
    if (
        "genomedetect" in repo
        or "genomicdetect" in repo
        or "dnadetect" in repo
        or "rna" in repo
    ):
        return "Genomics"
    if "chemicaldetect" in repo or "chem" in repo:
        return "Chemical"
    if "speciesdetect" in repo or "organismdetect" in repo:
        return "Species"
    if "proteindetect" in repo:
        return "Protein"
    if "pathologydetect" in repo:
        return "Pathology"
    return "Medical"


def _display_name_from_row(row: Dict[str, Any]) -> str:
    name = _repo_name(row["repo_id"])
    name = re.sub(r"^OpenMed-", "", name)
    name = name.replace("-", " ")
    return re.sub(r"\s+", " ", name).strip()


def _specialization_from_row(row: Dict[str, Any], category: str) -> str:
    languages = row.get("languages") or []
    language = ""
    if len(languages) == 1 and languages[0] != "en":
        language = f"{languages[0].upper()} "

    if category == "Privacy":
        return f"{language}PII detection".strip()
    if category == "Medical":
        return str(row.get("task") or "medical model")
    return f"{language}{category.lower()} entity detection".strip()


def _description_from_row(row: Dict[str, Any], category: str) -> str:
    task = row.get("task") or "model"
    tier = row.get("tier")
    param_count = row.get("param_count")
    parts = [f"{category} {task} model"]
    if tier:
        parts.append(f"tier={tier}")
    if isinstance(param_count, int):
        parts.append(f"params={param_count}")
    return "; ".join(parts)


def _entity_types_from_row(row: Dict[str, Any], category: str) -> List[str]:
    labels = row.get("canonical_labels")
    if isinstance(labels, list) and labels:
        return [str(label) for label in labels]
    return list(_CATEGORY_ENTITY_TYPES.get(category, []))


def _recommended_confidence(category: str) -> float:
    if category == "Privacy":
        return 0.50
    if category in {"Pharmaceutical", "Oncology", "Genomics", "Hematology"}:
        return 0.65
    return 0.60


def _size_category(row: Dict[str, Any]) -> str:
    tier = row.get("tier")
    if isinstance(tier, str) and tier:
        return "Medium" if tier == "Base" else tier

    param_count = row.get("param_count")
    if isinstance(param_count, int):
        if param_count < 60_000_000:
            return "Tiny"
        if param_count < 100_000_000:
            return "Small"
        if param_count < 250_000_000:
            return "Medium"
        if param_count < 500_000_000:
            return "Large"
        return "XLarge"
    return "Unknown"


def _model_info_from_row(row: Dict[str, Any]) -> ModelInfo:
    category = _category_from_row(row)
    return ModelInfo(
        model_id=row["repo_id"],
        display_name=_display_name_from_row(row),
        category=category,
        specialization=_specialization_from_row(row, category),
        description=_description_from_row(row, category),
        entity_types=_entity_types_from_row(row, category),
        size_category=_size_category(row),
        recommended_confidence=_recommended_confidence(category),
        family=str(row.get("family") or "Unknown"),
        task=str(row.get("task") or "unknown"),
        languages=list(row.get("languages") or []),
        tier=row.get("tier"),
        param_count=row.get("param_count"),
        architecture=row.get("architecture"),
        base_model=row.get("base_model"),
        formats=list(row.get("formats") or []),
        benchmark=_benchmark_from_row(row),
        latency_ms=_number_map_from_row(row, "latency_ms"),
        peak_ram_mb=_number_map_from_row(row, "peak_ram_mb"),
        recommended_tier=row.get("recommended_tier")
        if isinstance(row.get("recommended_tier"), str)
        else None,
        arxiv=row.get("arxiv"),
        license=row.get("license"),
        reproducibility_hash=row.get("reproducibility_hash"),
        released=row.get("released"),
    )


def _benchmark_from_row(row: Dict[str, Any]) -> Dict[str, Any] | List[Dict[str, Any]]:
    benchmark = row.get("benchmark")
    if isinstance(benchmark, list):
        return [dict(entry) for entry in benchmark if isinstance(entry, dict)]
    if isinstance(benchmark, dict):
        return dict(benchmark)
    return {}


def _number_map_from_row(row: Dict[str, Any], field_name: str) -> Dict[str, float]:
    value = row.get(field_name)
    if not isinstance(value, dict):
        return {}
    return {
        str(device): float(measurement)
        for device, measurement in value.items()
        if isinstance(measurement, (int, float)) and not isinstance(measurement, bool)
    }


def _language_prefix(row: Dict[str, Any]) -> str:
    languages = row.get("languages") or []
    if len(languages) == 1 and languages[0] != "en":
        return f"{languages[0]}_"
    return ""


def _strip_language_tokens(tokens: List[str], row: Dict[str, Any]) -> List[str]:
    languages = set(row.get("languages") or [])
    stripped: List[str] = []
    for token in tokens:
        lowered = token.lower()
        code = _LANGUAGE_NAME_TO_CODE.get(lowered)
        if code and code in languages:
            continue
        stripped.append(token)
    return stripped


def _pii_registry_key(row: Dict[str, Any]) -> str:
    tokens = _clean_model_tokens(_split_repo_tokens(row["repo_id"]))
    tokens = _strip_language_tokens(tokens, row)
    if tokens and tokens[0].lower() == "privacy":
        tokens = ["privacy_filter"] + tokens[2:] if len(tokens) > 1 else tokens
    suffix = _slug("_".join(tokens))
    return f"pii_{_language_prefix(row)}{suffix}"


def _ner_registry_key(row: Dict[str, Any]) -> str:
    tokens = _clean_model_tokens(_split_repo_tokens(row["repo_id"]))
    if tokens and tokens[0].lower().endswith("detect"):
        head = tokens[0][:-6]
        tail = tokens[1:]
        return _slug("_".join([head, "detection", *tail]))
    return _slug("_".join(tokens))


def _registry_key(row: Dict[str, Any]) -> str:
    category = _category_from_row(row)
    if category == "Privacy":
        return _pii_registry_key(row)
    if str(row.get("family") or "").upper() in {"NER", "ZEROSHOT"}:
        return _ner_registry_key(row)
    return _slug(_repo_name(row["repo_id"]))


def _unique_key(
    base_key: str, row: Dict[str, Any], registry: Dict[str, ModelInfo]
) -> str:
    if base_key not in registry:
        return base_key

    formats = row.get("formats") or []
    for format_name in formats:
        candidate = f"{base_key}_{_slug(format_name)}"
        if candidate not in registry:
            return candidate

    repo_suffix = _slug(_repo_name(row["repo_id"]))
    candidate = f"{base_key}_{repo_suffix}"
    if candidate not in registry:
        return candidate

    index = 2
    while f"{candidate}_{index}" in registry:
        index += 1
    return f"{candidate}_{index}"


def _pii_compatibility_aliases(row: Dict[str, Any]) -> List[str]:
    if _category_from_row(row) != "Privacy":
        return []
    repo_id = row["repo_id"]
    if "privacy-filter-multilingual" in repo_id:
        aliases = []
        tokens = _clean_model_tokens(_split_repo_tokens(repo_id))
        if tokens and tokens[0].lower() == "privacy":
            tokens = ["privacy_filter"] + tokens[2:] if len(tokens) > 1 else tokens
        suffix = _slug("_".join(tokens))
        for lang in row.get("languages") or []:
            if lang != "en":
                aliases.append(f"pii_{lang}_{suffix}")
        return aliases
    if "/OpenMed-PII-" not in repo_id:
        return []

    tokens = _clean_model_tokens(_split_repo_tokens(repo_id))
    tokens = _strip_language_tokens(tokens, row)
    no_param_tokens = [token for token in tokens if not _PARAM_RE.match(token)]
    aliases = []
    if no_param_tokens:
        aliases.append(f"pii_{_language_prefix(row)}{_slug('_'.join(no_param_tokens))}")
    return aliases


def _compatibility_aliases(row: Dict[str, Any]) -> List[str]:
    aliases = list(_LEGACY_MODEL_ALIASES.get(row["repo_id"], []))
    aliases.extend(_pii_compatibility_aliases(row))
    return aliases


def _build_registry(rows: Iterable[Dict[str, Any]]) -> Dict[str, ModelInfo]:
    registry: Dict[str, ModelInfo] = {}
    for row in rows:
        repo_id = row.get("repo_id")
        if not isinstance(repo_id, str) or not repo_id:
            continue

        info = _model_info_from_row(row)
        key = _unique_key(_registry_key(row), row, registry)
        registry[key] = info

        for alias in _compatibility_aliases(row):
            registry.setdefault(alias, info)
    return registry


_MANIFEST_ROWS = load_manifest_rows()
OPENMED_MODELS = _build_registry(_MANIFEST_ROWS)


def _models_by_language_from_manifest(languages: Iterable[str]) -> Dict[str, ModelInfo]:
    language_set = set(languages)
    return {
        key: info
        for key, info in OPENMED_MODELS.items()
        if key.startswith("pii_")
        and info.category == "Privacy"
        and language_set.intersection(info.languages or ["en"])
    }


def _generate_multilingual_pii_models() -> Dict[str, ModelInfo]:
    """Return French, German, Italian, and Spanish PII entries from manifest."""
    return _models_by_language_from_manifest({"fr", "de", "it", "es"})


def _build_portuguese_pii_models() -> Dict[str, ModelInfo]:
    """Return Portuguese PII entries from manifest."""
    return _models_by_language_from_manifest({"pt"})


def _build_new_language_pii_models() -> Dict[str, ModelInfo]:
    """Return Arabic, Japanese, and Turkish PII entries from manifest."""
    return _models_by_language_from_manifest({"ar", "ja", "tr"})


def _build_categories() -> Dict[str, List[str]]:
    categories: Dict[str, List[str]] = {}
    for key, model in OPENMED_MODELS.items():
        categories.setdefault(model.category, []).append(key)
    return {category: sorted(keys) for category, keys in categories.items()}


CATEGORIES = _build_categories()

SIZE_RECOMMENDATIONS = {
    "fast": [
        key
        for key in (
            "disease_detection_tiny",
            "oncology_detection_tiny",
            "pii_superclinical_small",
            "pii_liteclinical_small",
            "pii_fastclinical_small",
        )
        if key in OPENMED_MODELS
    ],
    "balanced": [
        key
        for key in (
            "pharma_detection_supermedical",
            "genome_detection_bioclinical",
            "anatomy_detection_electramed",
            "pii_superclinical_base",
            "pii_clinicale5_base",
        )
        if key in OPENMED_MODELS
    ],
    "accurate": [
        key
        for key in (
            "disease_detection_superclinical",
            "pharma_detection_superclinical",
            "oncology_detection_superclinical",
            "pii_superclinical_large",
            "pii_qwenmed_xlarge",
        )
        if key in OPENMED_MODELS
    ],
}


def get_model_info(model_key: str) -> Optional[ModelInfo]:
    """Get model information by registry key or repo id."""
    if model_key in OPENMED_MODELS:
        return OPENMED_MODELS[model_key]

    for model in OPENMED_MODELS.values():
        if model.model_id == model_key:
            return model
    return None


def get_models_by_category(category: str) -> List[ModelInfo]:
    """Get all models in a specific category."""
    model_keys = CATEGORIES.get(category, [])
    return [OPENMED_MODELS[key] for key in model_keys if key in OPENMED_MODELS]


def get_models_by_size(size_category: str) -> List[ModelInfo]:
    """Get models by size category (Tiny, Small, Medium, Large, XLarge)."""
    return [
        model
        for model in OPENMED_MODELS.values()
        if model.size_category == size_category
    ]


def get_recommended_models(use_case: str = "balanced") -> List[ModelInfo]:
    """Get recommended models for a specific use case."""
    model_keys = SIZE_RECOMMENDATIONS.get(
        use_case, SIZE_RECOMMENDATIONS.get("balanced", [])
    )
    return [OPENMED_MODELS[key] for key in model_keys if key in OPENMED_MODELS]


def find_models_by_entity_type(entity_type: str) -> List[ModelInfo]:
    """Find models that can detect a specific entity type."""
    matching_models = []
    for model in OPENMED_MODELS.values():
        if any(entity_type.upper() in et.upper() for et in model.entity_types):
            matching_models.append(model)
    return matching_models


def get_all_models() -> Dict[str, ModelInfo]:
    """Get all available OpenMed models."""
    return OPENMED_MODELS.copy()


_CATEGORY_KEYWORDS: Dict[str, Tuple[str, str]] = {
    "pii|deidentif|hipaa|phi|protected health|patient name|ssn|medical record|privacy|anonymiz": (
        "Privacy",
        "Contains PII/de-identification terms",
    ),
    "cancer|tumor|oncolog|malign|chemotherapy|radiation": (
        "Oncology",
        "Contains cancer/oncology terms",
    ),
    "drug|medication|pharma|dose|mg|pill|tablet|cisplatin": (
        "Pharmaceutical",
        "Contains pharmaceutical terms",
    ),
    "gene|dna|protein|mutation|chromosome": (
        "Genomics",
        "Contains genomic/genetic terms",
    ),
    "ecg|ekg|ejection fraction|arrhythmia|stent|pacemaker|murmur|st elevation|echocardiogram|cardiac|cardiolog": (
        "Cardiology",
        "Contains cardiology terms",
    ),
    "rash|lesion|macule|papule|erythema|pruritus|biopsy of skin|dermatolog|skin": (
        "Dermatology",
        "Contains dermatology terms",
    ),
    "visual acuity|intraocular pressure|retina|cornea|glaucoma|fundus|ophthalmolog": (
        "Ophthalmology",
        "Contains ophthalmology terms",
    ),
    "heart|lung|brain|liver|kidney|organ": (
        "Anatomy",
        "Contains anatomical terms",
    ),
    "culture|gram\\s*stain|susceptib|\\bmic\\b|resistant|sensitive|e\\.\\s*coli|mrsa|oxacillin": (
        "Microbiology",
        "Contains microbiology terms",
    ),
    "bacteria|virus|organism|species": (
        "Species",
        "Contains organism/species terms",
    ),
    "disease|condition|disorder|syndrome": (
        "Disease",
        "Contains disease/condition terms",
    ),
    "pathology|histology|biopsy": (
        "Pathology",
        "Contains pathological terms",
    ),
    "blood|lymph|leukemia|lymphoma": (
        "Hematology",
        "Contains hematological terms",
    ),
}


def _match_categories(text: str) -> List[Tuple[str, str]]:
    """Return ``(category, reason)`` pairs whose keywords match ``text``.

    This is the routing layer behind :func:`get_model_suggestions`. It reports
    a category whenever the text matches its keywords, independently of whether
    any model is registered for that category (e.g. ``Cardiology`` has keyword
    routing but no registered model yet).
    """

    text_lower = text.lower()
    return [
        (category, reason)
        for pattern, (category, reason) in _CATEGORY_KEYWORDS.items()
        if re.search(pattern, text_lower)
    ]


def get_model_suggestions(text: str) -> List[Tuple[str, ModelInfo, str]]:
    """Suggest appropriate models based on text content."""
    suggestions: List[Tuple[str, ModelInfo, str]] = []

    for category, reason in _match_categories(text):
        for key in CATEGORIES.get(category, [])[:3]:
            suggestions.append((key, OPENMED_MODELS[key], reason))

    if not suggestions:
        for key in SIZE_RECOMMENDATIONS.get("balanced", [])[:3]:
            suggestions.append((key, OPENMED_MODELS[key], "General medical text"))

    return suggestions[:3]


def list_model_categories() -> List[str]:
    """List all available model categories."""
    return list(CATEGORIES.keys())


def get_entity_types_by_category(category: str) -> List[str]:
    """Get all entity types supported by models in a category."""
    models = get_models_by_category(category)
    entity_types = set()
    for model in models:
        entity_types.update(model.entity_types)
    return sorted(entity_types)


def get_pii_models_by_language(lang: str) -> Dict[str, ModelInfo]:
    """Return all single-language PII models for a given language."""
    if lang == "en":
        localized_prefixes = _LOCALIZED_PII_LANGUAGE_KEYS
        return {
            key: info
            for key, info in OPENMED_MODELS.items()
            if key.startswith("pii_")
            and info.category == "Privacy"
            and "en" in (info.languages or ["en"])
            and not any(key.startswith(f"pii_{lc}_") for lc in localized_prefixes)
        }

    prefix = f"pii_{lang}_"
    return {
        key: info
        for key, info in OPENMED_MODELS.items()
        if key.startswith(prefix)
        and info.category == "Privacy"
        and lang in (info.languages or [])
    }


def get_default_pii_model(lang: str) -> Optional[str]:
    """Return the default (recommended) PII model_id for a language."""
    from .pii_i18n import DEFAULT_PII_MODELS

    return DEFAULT_PII_MODELS.get(lang)
