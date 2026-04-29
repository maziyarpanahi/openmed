"""OpenMed Privacy Filter Studio — interactive de-identification demo.

A two-pane web app: paste text on the left, press Run, see the output on the
right with each detected PII span highlighted with a colorful label. Toggle
between **Mask** mode (replace with ``[PERSON]`` placeholders) and
**Randomize** mode (replace with locale-aware Faker surrogates, deterministic
within the call).

Defaults to MLX BF16 on Apple Silicon
(``OpenMed/privacy-filter-nemotron-mlx``). On non-Apple-Silicon hosts the
unified ``openmed.extract_pii`` / ``openmed.deidentify`` API automatically
falls back to the matching PyTorch checkpoint
(``OpenMed/privacy-filter-nemotron``).

Run from the repository root:

    uvicorn examples.privacy_filter_studio.app:app --reload --port 8770

Set ``OPENMED_PRIVACY_FILTER_DOWNLOAD=1`` (or pass ``download: true`` in the
request) to allow first-run model downloads. Otherwise the model must
already be cached locally.
"""

from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


ROOT = Path(__file__).resolve().parent
STATIC_DIR = ROOT / "static"

DEFAULT_MODEL_ID = os.getenv(
    "OPENMED_STUDIO_MODEL",
    "OpenMed/privacy-filter-nemotron-mlx",
)
ALLOW_DOWNLOAD_ENV = os.getenv("OPENMED_PRIVACY_FILTER_DOWNLOAD", "").lower() in {
    "1",
    "true",
    "yes",
}


# ---------------------------------------------------------------------------
# Pre-defined examples (medical-themed, varying difficulty)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StudioExample:
    id: str
    title: str
    blurb: str
    text: str

    def to_public_dict(self) -> dict[str, Any]:
        return {"id": self.id, "title": self.title, "blurb": self.blurb, "text": self.text}


EXAMPLES: tuple[StudioExample, ...] = (
    StudioExample(
        id="discharge",
        title="Discharge summary",
        blurb="Names, MRN, DOB, contact details, account info — the daily case.",
        text=(
            "Patient Sarah Johnson (DOB 03/15/1985), MRN 4872910, was discharged on "
            "April 22 2026 after a three-day admission for community-acquired pneumonia. "
            "Follow-up scheduled with Dr. Michael Chen on May 6 2026 at Cedars Medical "
            "Center, 8700 Beverly Blvd, Los Angeles CA 90048. Reach the patient at "
            "(415) 555-7012 or sarah.johnson@example.com. Insurance member ID "
            "BLU-2284-118-A; copay reference invoice 2026-04-22-008712."
        ),
    ),
    StudioExample(
        id="referral",
        title="Specialist referral",
        blurb="Two clinicians, an address, a phone, an SSN — and a sensitive history.",
        text=(
            "Referring Provider: Dr. Aisha Patel (NPI 1326744929), Internal Medicine.\n"
            "Specialist: Dr. Marcus Reilly, Endocrinology.\n\n"
            "RE: Robert Alvarez, SSN 412-55-9183, DOB 11/02/1958.\n\n"
            "Mr. Alvarez presents with poorly controlled type 2 diabetes (HbA1c 9.4%) "
            "and worsening peripheral neuropathy. Most recent labs drawn at Mercy "
            "Hospital on 04/10/2026. Patient prefers correspondence at "
            "ralvarez1958@example.org or his cell, +1 312-555-0144. Mailing address "
            "1428 W Roosevelt Rd, Apt 6B, Chicago IL 60608."
        ),
    ),
    StudioExample(
        id="enterprise",
        title="Enterprise IT incident",
        blurb="Mixed-domain PII: emails, IPs, MACs, an API key, financial accounts.",
        text=(
            "Incident IM-2026-04-2271 logged by jdoe@hospital.example at 22:14 UTC.\n\n"
            "Workstation 10.42.118.27 (MAC 4c:bb:58:9a:11:7e) issued repeated auth "
            "failures against billing-svc using API key sk_live_8ahWJD2lQ3pX9pTk1mZ. "
            "Affected employee: Linda Park, employee ID EMP-447128, hired 2019-08-12, "
            "currently in Finance.\n\n"
            "Card on file ending 4111-1111-1111-7392 was rotated; routing number "
            "021000089 confirmed. Forwarded to security@hospital.example and "
            "ciso.office@hospital.example for review. Vehicle plate IL-MED-2018 "
            "noted in lobby footage at 21:58."
        ),
    ),
    StudioExample(
        id="multilingual",
        title="Multilingual snippet",
        blurb="A multilingual paragraph mixing English, French, and Portuguese PII.",
        text=(
            "Patient: Pedro Almeida, CPF 123.456.789-09, telefone +351 912 345 678, "
            "email pedro.almeida@hospital.pt, morada Rua das Flores 25, 1200-195 "
            "Lisboa.\n\n"
            "Le médecin référent, Dr. Camille Lefèvre, peut être joint au "
            "+33 1 42 96 10 10 ou à camille.lefevre@hopital.fr.\n\n"
            "English follow-up: please confirm with Sarah Johnson "
            "(sarah.johnson@example.com, 415-555-7012) by April 30 2026."
        ),
    ),
    StudioExample(
        id="paragraph",
        title="Casual narrative",
        blurb="Free-form prose — looks innocuous but mentions multiple identifiers.",
        text=(
            "I called Dr. Hannah Wright last Tuesday around 4pm. She wanted to confirm "
            "my new address — 1742 Ocean View Ave, Apt 14, San Diego CA 92107 — "
            "before mailing the lab results. I gave her my mobile, 619-555-0287, and "
            "the new email address (hannah.wright.patient@example.net). Apparently "
            "the previous chart had my old SSN listed; I asked her to update it from "
            "the placeholder to the correct one ending in 5188. While we were on the "
            "phone she also confirmed my insurance group number HMO-PROD-44128."
        ),
    ),
)


# ---------------------------------------------------------------------------
# Pydantic request models
# ---------------------------------------------------------------------------

class RunRequest(BaseModel):
    text: str = Field(..., description="Free-form text to deidentify.")
    mode: Literal["mask", "randomize"] = Field(
        default="mask",
        description="``mask`` uses placeholder labels; ``randomize`` uses Faker surrogates.",
    )
    seed: int = Field(default=42, description="Seed for randomize/consistent surrogates.")
    locale: str | None = Field(
        default=None,
        description="Optional Faker locale override (e.g. ``pt_BR``, ``fr_FR``).",
    )
    download: bool = Field(
        default=False,
        description="Allow first-run model download from the Hub.",
    )


# ---------------------------------------------------------------------------
# Pipeline lifecycle
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _load_pipeline(download: bool) -> tuple[Any, float, str]:
    """Build the unified privacy-filter pipeline once and cache it.

    The cache key includes ``download`` so that flipping the toggle invalidates
    the cached pipeline (and lets a previously-failed download retry).
    """
    started = time.perf_counter()
    from openmed.core.backends import (
        create_privacy_filter_pipeline,
        select_privacy_filter_backend,
    )

    # Allow first-run downloads only when explicitly requested. Otherwise ask
    # both Hugging Face Hub and Transformers to stay cache-only.
    prior_hf_offline = os.environ.get("HF_HUB_OFFLINE")
    prior_transformers_offline = os.environ.get("TRANSFORMERS_OFFLINE")
    if download:
        os.environ.pop("HF_HUB_OFFLINE", None)
        os.environ.pop("TRANSFORMERS_OFFLINE", None)
    else:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    backend = select_privacy_filter_backend(DEFAULT_MODEL_ID)
    try:
        pipeline = create_privacy_filter_pipeline(DEFAULT_MODEL_ID)
    finally:
        if prior_hf_offline is None:
            os.environ.pop("HF_HUB_OFFLINE", None)
        else:
            os.environ["HF_HUB_OFFLINE"] = prior_hf_offline
        if prior_transformers_offline is None:
            os.environ.pop("TRANSFORMERS_OFFLINE", None)
        else:
            os.environ["TRANSFORMERS_OFFLINE"] = prior_transformers_offline

    return pipeline, time.perf_counter() - started, backend


# ---------------------------------------------------------------------------
# Inference + deidentification
# ---------------------------------------------------------------------------

def _entities_from_pipeline(pipeline: Any, text: str) -> list[dict[str, Any]]:
    raw = pipeline(text)
    entities: list[dict[str, Any]] = []
    for item in raw:
        label = item.get("entity_group") or item.get("entity") or ""
        start = int(item.get("start", 0))
        end = int(item.get("end", 0))
        if end <= start:
            continue
        entities.append(
            {
                "label": str(label),
                "start": start,
                "end": end,
                "text": text[start:end],
                "score": float(item.get("score", 0.0)),
            }
        )
    entities.sort(key=lambda e: e["start"])
    return entities


def _mask_text(text: str, entities: list[dict[str, Any]]) -> str:
    """Replace each entity span with ``[LABEL]``, preserving everything else."""
    parts: list[str] = []
    cursor = 0
    for ent in entities:
        if ent["start"] < cursor:
            continue  # skip overlapping entries (shouldn't happen with Viterbi)
        parts.append(text[cursor:ent["start"]])
        parts.append(f"[{ent['label'].upper()}]")
        cursor = ent["end"]
    parts.append(text[cursor:])
    return "".join(parts)


def _randomize_text(
    text: str,
    entities: list[dict[str, Any]],
    *,
    seed: int,
    locale: str | None,
) -> tuple[str, dict[str, str]]:
    """Replace each entity span with a deterministic Faker surrogate.

    Same ``(label, original)`` pair always produces the same surrogate within
    one call — repeat mentions of the same person resolve to one fake.
    """
    from openmed.core.anonymizer import Anonymizer

    anon = Anonymizer(lang="en", locale=locale, consistent=True, seed=seed)
    surrogate_map: dict[str, str] = {}

    def surrogate_for(label: str, original: str) -> str:
        key = f"{label}|{original}"
        if key not in surrogate_map:
            surrogate_map[key] = anon.surrogate(original, label)
        return surrogate_map[key]

    parts: list[str] = []
    cursor = 0
    for ent in entities:
        if ent["start"] < cursor:
            continue
        parts.append(text[cursor:ent["start"]])
        parts.append(surrogate_for(ent["label"], ent["text"]))
        cursor = ent["end"]
    parts.append(text[cursor:])
    return "".join(parts), surrogate_map


def _entities_with_surrogates(
    entities: list[dict[str, Any]],
    surrogate_map: dict[str, str],
) -> list[dict[str, Any]]:
    """Augment each entity with its surrogate value (for client-side rendering)."""
    annotated: list[dict[str, Any]] = []
    for ent in entities:
        key = f"{ent['label']}|{ent['text']}"
        annotated.append({**ent, "surrogate": surrogate_map.get(key, "")})
    return annotated


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="OpenMed Privacy Filter Studio",
    description=(
        "Interactive PII de-identification demo built on the OpenMed unified "
        "extract_pii / deidentify API. Defaults to the Nemotron-PII MLX BF16 "
        "checkpoint on Apple Silicon, with automatic PyTorch fallback elsewhere."
    ),
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/examples")
def api_examples() -> dict[str, Any]:
    return {
        "model": DEFAULT_MODEL_ID,
        "examples": [example.to_public_dict() for example in EXAMPLES],
    }


@app.post("/api/run")
def api_run(payload: RunRequest) -> dict[str, Any]:
    text = payload.text.strip()
    if not text:
        return {
            "status": "empty",
            "modelId": DEFAULT_MODEL_ID,
            "entities": [],
            "masked": "",
            "randomized": "",
            "stats": {"tokens": 0, "entities": 0, "inferenceMs": 0.0},
            "note": "Empty input.",
        }

    download = bool(payload.download or ALLOW_DOWNLOAD_ENV)
    try:
        pipeline, load_s, backend = _load_pipeline(download)
    except Exception as exc:  # noqa: BLE001 — surface load errors in the UI
        return {
            "status": "error",
            "modelId": DEFAULT_MODEL_ID,
            "entities": [],
            "masked": text,
            "randomized": text,
            "stats": {"tokens": 0, "entities": 0, "inferenceMs": 0.0},
            "note": f"Model load failed: {type(exc).__name__}: {exc}",
        }

    started = time.perf_counter()
    try:
        entities = _entities_from_pipeline(pipeline, text)
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "error",
            "modelId": DEFAULT_MODEL_ID,
            "entities": [],
            "masked": text,
            "randomized": text,
            "stats": {"tokens": 0, "entities": 0, "inferenceMs": 0.0},
            "note": f"Inference failed: {type(exc).__name__}: {exc}",
        }
    inference_s = time.perf_counter() - started

    masked = _mask_text(text, entities)
    randomized, surrogate_map = _randomize_text(
        text, entities, seed=payload.seed, locale=payload.locale,
    )
    annotated = _entities_with_surrogates(entities, surrogate_map)

    # Approximate token count from whitespace split — gives a useful UI metric
    # without paying for a second tokenization pass.
    token_estimate = len(re.findall(r"\S+", text))

    return {
        "status": "live",
        "modelId": DEFAULT_MODEL_ID,
        "backend": backend,
        "entities": annotated,
        "masked": masked,
        "randomized": randomized,
        "stats": {
            "tokens": token_estimate,
            "entities": len(entities),
            "inferenceMs": round(inference_s * 1000, 1),
            "loadMs": round(load_s * 1000, 1),
        },
        "note": (
            f"{backend.upper()} runtime — {len(entities)} entities in "
            f"{inference_s * 1000:.1f} ms."
        ),
    }
