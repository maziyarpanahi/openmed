"""Reproducible, text-free provenance helpers for clinical NLI training.

Public biomedical binary labels remain partial labels: ``not_entailment``
must never be silently converted to contradiction. Restricted patient-note
corpora are not accepted by the bundled recipe.
"""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

LABELS = ("contradiction", "neutral", "entailment")
PARTIAL_LABEL = "not_entailment"
BASE_MODEL = "OpenMed/OpenMed-PII-ClinicalE5-Small-33M-v1"
BASE_REVISION = "2ea947130eb28d610ceb96472af0b694fce9c344"
MULTINLI_REVISION = "da70db2af9d09693783c3320c4249840212ee221"
BIONLI_FILES = {
    "train": "1pqXyU4E13-foNHdH8uBQ_YnY7i03nG1J",
    "development": "1qpTqcmSHoF3P89Vf57agEBdyOGQJQpy5",
}
BIONLI_LICENSE_SOURCE = "https://stonybrooknlp.github.io/BioNLI/"
RECIPE_PATH = Path(__file__).parent / "configs" / "clinical_nli_small.yaml"


def load_training_recipe(path: str | Path = RECIPE_PATH) -> dict:
    """Read the explicit local-only recipe; reject privacy/publication drift.

    Args:
        path: Local YAML configuration with the audited data and model pins.

    Returns:
        Validated training configuration; this does not start any workload.
    """

    import yaml

    recipe = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(recipe, dict) or (
        recipe.get("base_model", {}).get("model_ref") != BASE_MODEL
        or recipe.get("base_model", {}).get("revision") != BASE_REVISION
        or recipe.get("labels") != list(LABELS)
        or recipe.get("data", {}).get("patient_data_allowed") is not False
        or recipe.get("data", {}).get("mednli") != "eval_only_user_supplied"
        or recipe.get("data", {}).get("mednli_download_allowed") is not False
        or recipe.get("publication", {}).get("enabled") is not False
        or recipe.get("training", {}).get("paid_compute_authorized") is not False
        or recipe.get("training", {}).get("gradient_accumulation_steps") != 1
    ):
        raise ValueError("training recipe must preserve audited local-only provenance")
    return recipe


def digest(value: object) -> str:
    """Return a stable SHA-256 fingerprint without retaining source text."""

    payload = json.dumps(value, sort_keys=True, ensure_ascii=True).encode()
    return "sha256:" + hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class NLIPair:
    """One local training pair with an opaque source-group identifier."""

    premise: str
    hypothesis: str
    label: str
    source_group: str
    synthetic: bool = False
    phenomena: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.label not in (*LABELS, PARTIAL_LABEL):
            raise ValueError("unsupported training label")
        if not self.premise.strip() or not self.hypothesis.strip():
            raise ValueError("training text must be nonempty")
        if not self.source_group or type(self.synthetic) is not bool:
            raise ValueError("training provenance is required")

    @property
    def pair_digest(self) -> str:
        """Return the normalized pair fingerprint used for deduplication."""

        return digest(
            (" ".join(self.premise.split()), " ".join(self.hypothesis.split()))
        )


def grouped_split(
    pairs: Iterable[NLIPair], *, seed: int = 3236
) -> dict[str, list[NLIPair]]:
    """Deduplicate before deterministic 80/10/10 source-group partitioning.

    Conflicting labels or one pair appearing under different groups are
    rejected, rather than allowing hidden train/evaluation leakage.

    Args:
        pairs: Local pairs carrying opaque source groups.
        seed: Seed incorporated into stable group assignment.

    Returns:
        Disjoint train, validation and test partitions.
    """

    result: dict[str, list[NLIPair]] = {"train": [], "validation": [], "test": []}
    seen: dict[str, NLIPair] = {}
    for pair in pairs:
        previous = seen.get(pair.pair_digest)
        if previous is not None:
            if (
                previous.label != pair.label
                or previous.source_group != pair.source_group
            ):
                raise ValueError("duplicate pair has conflicting provenance")
            continue
        seen[pair.pair_digest] = pair
        bucket = int(digest((seed, pair.source_group))[-8:], 16) % 10
        split = "test" if bucket == 9 else "validation" if bucket == 8 else "train"
        result[split].append(pair)
    return result


def load_bionli_csv(path: str | Path) -> list[NLIPair]:
    """Read the author-published BioNLI CSV without guessing three-way labels.

    The public corpus is biomedical literature, not clinical patient notes.
    PMID groups keep all perturbations of one publication in one partition.

    Args:
        path: Local author-published balanced CSV.

    Returns:
        Pairs labeled entailment or the explicit binary partial label.
    """

    result = []
    with Path(path).open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"pmid", "supp_set", "conclusion", "label_cat"}
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError("unsupported public biomedical schema")
        for row in reader:
            if row["label_cat"] not in {
                "pos",
                "LPR",
                "SEP",
                "SEN",
                "SREO",
                "SRE",
                "posToNeg",
                "negToPos",
                "swap_number",
                "generation",
                "generation_nd_SEN",
                "generation_nd",
                "generation_nd_SRE",
            }:
                raise ValueError("unsupported public biomedical label category")
            result.append(
                NLIPair(
                    premise=row["supp_set"],
                    hypothesis=row["conclusion"],
                    label="entailment" if row["label_cat"] == "pos" else PARTIAL_LABEL,
                    source_group="bionli:" + digest(row["pmid"])[7:],
                )
            )
    return result


def remove_conflicting_public_groups(
    pairs: Iterable[NLIPair],
) -> tuple[list[NLIPair], dict[str, int]]:
    """Exclude entire groups with ambiguous duplicate pairs before partitioning.

    Conflicting labels cannot be repaired from a preferred model prediction.
    Dropping whole publication groups also avoids train/evaluation connections
    through identical pairs attributed to different publications.

    Args:
        pairs: Public pairs before splitting or sampling.

    Returns:
        Retained pairs and aggregate excluded-group/pair counts.
    """

    rows = list(pairs)
    first: dict[str, NLIPair] = {}
    excluded = set()
    for pair in rows:
        old = first.get(pair.pair_digest)
        if old is not None and (
            old.label != pair.label or old.source_group != pair.source_group
        ):
            excluded.update((old.source_group, pair.source_group))
        first[pair.pair_digest] = pair
    kept = [pair for pair in rows if pair.source_group not in excluded]
    return kept, {
        "excluded_groups": len(excluded),
        "excluded_pairs": len(rows) - len(kept),
    }


def synthetic_clinical_pairs() -> list[NLIPair]:
    """Generate authored synthetic relationships, not patient data or diagnoses.

    Conditions are source groups. Holding out conditions tests lexical transfer
    within these templates, not independent clinical generalization. The
    separate committed negation challenge is never a training input.
    """

    conditions = (
        "fever",
        "cough",
        "nausea",
        "headache",
        "fatigue",
        "dizziness",
        "rash",
        "wheezing",
        "edema",
        "diarrhea",
        "constipation",
        "anemia",
        "asthma",
        "arthritis",
        "migraine",
        "insomnia",
        "tremor",
        "itching",
        "chest pain",
        "back pain",
        "abdominal pain",
        "shortness of breath",
        "hypertension",
        "diabetes",
        "pneumonia",
        "bronchitis",
        "hypotension",
        "tachycardia",
        "bradycardia",
        "dehydration",
        "sinusitis",
        "otitis",
        "sore throat",
        "joint pain",
        "muscle pain",
        "palpitations",
        "vertigo",
        "neuropathy",
        "gastritis",
        "reflux",
        "dysphagia",
        "dysuria",
        "hematuria",
        "hypoglycemia",
        "hyperglycemia",
        "hypothermia",
        "hyperthermia",
        "weight loss",
        "hearing loss",
        "blurred vision",
        "leg pain",
        "neck pain",
        "weakness",
        "vomiting",
        "seizure",
        "syncope",
        "ear pain",
        "nasal congestion",
        "dry mouth",
        "chills",
        "urinary retention",
        "incontinence",
        "swelling",
        "tingling",
    )
    medicines = (
        "aspirin",
        "metformin",
        "lisinopril",
        "amlodipine",
        "losartan",
        "atorvastatin",
        "omeprazole",
        "albuterol",
        "gabapentin",
        "ibuprofen",
        "acetaminophen",
        "amoxicillin",
        "insulin",
        "warfarin",
        "heparin",
        "furosemide",
        "prednisone",
        "levothyroxine",
        "naproxen",
        "cetirizine",
    )
    result = []
    for condition, medicine, quantity in itertools.product(
        conditions, medicines, (2, 5, 10, 20)
    ):
        group = "synthetic:" + digest(condition)[7:]
        templates = (
            (
                "negation",
                f"The patient explicitly denies {condition}. No {condition} is present.",
                f"The patient has no {condition}.",
                f"The patient currently has {condition}.",
            ),
            (
                "temporality",
                f"The patient had {condition} last year. It resolved and is absent today.",
                f"The patient previously had {condition}.",
                f"The patient currently has {condition}.",
            ),
            (
                "experiencer",
                f"The patient's mother has {condition}. The patient does not have it.",
                f"The patient's mother has {condition}.",
                f"The patient has {condition}.",
            ),
            (
                "numbers",
                f"The patient takes exactly {quantity} mg of {medicine} for {condition}.",
                f"The dose of {medicine} is {quantity} mg.",
                f"The dose of {medicine} is {quantity + 1} mg.",
            ),
            (
                "medication_status",
                f"{medicine} for {condition} was discontinued. It is not taken today.",
                f"The patient stopped taking {medicine}.",
                f"The patient currently takes {medicine}.",
            ),
            (
                "negation",
                f"The patient has {condition}. The symptom is present today.",
                f"The patient currently has {condition}.",
                f"The patient has no {condition}.",
            ),
            (
                "medication_status",
                f"The patient continues taking {medicine} for {condition} today.",
                f"The patient currently takes {medicine}.",
                f"The patient stopped taking {medicine}.",
            ),
            (
                "negation",
                f"It is not true that the patient has no {condition}. It is present.",
                f"The patient has {condition}.",
                f"The patient does not have {condition}.",
            ),
        )
        for phenomenon, premise, supported, opposed in templates:
            for label, hypothesis in (
                ("entailment", supported),
                ("contradiction", opposed),
                ("neutral", "The patient has a documented allergy to penicillin."),
            ):
                result.append(
                    NLIPair(premise, hypothesis, label, group, True, (phenomenon,))
                )
    # Templates that do not vary by medication/dose intentionally collapse.
    unique = {pair.pair_digest: pair for pair in result}
    return list(unique.values())


def corpus_fingerprint(pairs: Iterable[NLIPair]) -> str:
    """Bind labels, groups, and exact pair digests without exporting text.

    Args:
        pairs: Local corpus partition to fingerprint.

    Returns:
        Order-independent SHA-256 digest of the partition's provenance.
    """

    return digest(
        sorted((pair.pair_digest, pair.label, pair.source_group) for pair in pairs)
    )


def pair_payload(pair: NLIPair) -> dict:
    """Return a training payload for explicitly local, non-audit storage."""

    return asdict(pair)
