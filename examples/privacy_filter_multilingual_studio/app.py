"""OpenMed Privacy Filter Multilingual Studio — side-by-side comparison demo.

Two MLX pipelines run on the same text simultaneously:

* **OpenMed/privacy-filter-multilingual-mlx** (left pane)
  — the OpenMed multilingual fine-tune, 54 PII categories, trained over 16
  languages.
* **mlx-community/openai-privacy-filter-bf16** (right pane)
  — the upstream OpenAI baseline, 8 coarse PII categories, English-leaning.

Pre-defined examples for each of the 16 supported languages let you click
through the scenarios and see how the two models differ on the same input.

Run from the repository root:

    uvicorn examples.privacy_filter_multilingual_studio.app:app --reload --port 8780

Set ``OPENMED_PRIVACY_FILTER_DOWNLOAD=1`` (or pass ``download: true`` in the
request) to allow first-run model downloads. Otherwise both models must
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

from .examples import LANGUAGE_EXAMPLES, LANGUAGE_META

logger = logging.getLogger(__name__)


ROOT = Path(__file__).resolve().parent
STATIC_DIR = ROOT / "static"

# Two-model setup. Left = our multilingual fine-tune; right = upstream baseline.
OPENMED_MODEL_ID = "OpenMed/privacy-filter-multilingual-mlx"

# Baseline = `openai/privacy-filter` converted into the OpenMed-MLX layout
# (same weights, different on-disk file naming). Shown to the user under the
# upstream display name. The OPENMED_BASELINE_PATH env var lets you point at
# a different local artifact directory if needed; otherwise we fall back to
# the locally-converted dir produced by ``scripts/export/convert-mlx``.
BASELINE_DISPLAY = "openai/privacy-filter"
DEFAULT_BASELINE_PATH = Path(
    os.path.expanduser(
        os.getenv(
            "OPENMED_BASELINE_PATH",
            "~/Developer/openmed-mlx-export/out/mlx/openai-privacy-filter-bf16",
        )
    )
)

ALLOW_DOWNLOAD_ENV = os.getenv("OPENMED_PRIVACY_FILTER_DOWNLOAD", "").lower() in {
    "1",
    "true",
    "yes",
}


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class RunRequest(BaseModel):
    text: str = Field(..., description="Free-form text to deidentify.")
    mode: Literal["mask", "randomize"] = Field(default="mask")
    seed: int = Field(default=42)
    locale: str | None = Field(default=None)
    download: bool = Field(default=False)


# ---------------------------------------------------------------------------
# Pipeline lifecycle — load both pipelines once
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LoadedPipeline:
    side: Literal["openmed", "baseline"]
    model_id: str
    display_name: str
    pipeline: Any
    load_seconds: float


@lru_cache(maxsize=1)
def _load_pipelines(download: bool) -> tuple[LoadedPipeline, LoadedPipeline]:
    """Build both pipelines once and cache. Cache key includes ``download`` so
    flipping the toggle invalidates and lets a previously-failed download
    retry."""
    from huggingface_hub import snapshot_download
    from openmed.mlx.inference import PrivacyFilterMLXPipeline

    prior_hf_offline = os.environ.get("HF_HUB_OFFLINE")
    prior_transformers_offline = os.environ.get("TRANSFORMERS_OFFLINE")
    if download:
        os.environ.pop("HF_HUB_OFFLINE", None)
        os.environ.pop("TRANSFORMERS_OFFLINE", None)
    else:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    def build(
        side: Literal["openmed", "baseline"],
        *,
        hub_id: str | None,
        local_path: Path | None,
        display: str,
    ) -> LoadedPipeline:
        started = time.perf_counter()
        if local_path is not None and local_path.exists():
            path = str(local_path)
        else:
            assert hub_id is not None
            path = snapshot_download(hub_id)
        pipeline = PrivacyFilterMLXPipeline(path)
        elapsed = time.perf_counter() - started
        return LoadedPipeline(
            side=side,
            model_id=hub_id or display,
            display_name=display,
            pipeline=pipeline,
            load_seconds=elapsed,
        )

    try:
        left = build(
            "openmed",
            hub_id=OPENMED_MODEL_ID,
            local_path=None,
            display=OPENMED_MODEL_ID,
        )
        # Baseline: prefer the local converted artifact (semantically identical
        # to openai/privacy-filter); fall through to the local DEFAULT path.
        right = build(
            "baseline",
            hub_id=None,
            local_path=DEFAULT_BASELINE_PATH,
            display=BASELINE_DISPLAY,
        )
        return left, right
    finally:
        if prior_hf_offline is None:
            os.environ.pop("HF_HUB_OFFLINE", None)
        else:
            os.environ["HF_HUB_OFFLINE"] = prior_hf_offline
        if prior_transformers_offline is None:
            os.environ.pop("TRANSFORMERS_OFFLINE", None)
        else:
            os.environ["TRANSFORMERS_OFFLINE"] = prior_transformers_offline


# ---------------------------------------------------------------------------
# Inference + de-identification helpers
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
    """Replace each entity span with ``[LABEL]``."""
    parts: list[str] = []
    cursor = 0
    for ent in entities:
        if ent["start"] < cursor:
            continue
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
    """Replace each entity span with a deterministic Faker surrogate."""
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
    annotated: list[dict[str, Any]] = []
    for ent in entities:
        key = f"{ent['label']}|{ent['text']}"
        annotated.append({**ent, "surrogate": surrogate_map.get(key, "")})
    return annotated


def _process_side(
    pipeline_record: LoadedPipeline,
    text: str,
    *,
    mode: str,
    seed: int,
    locale: str | None,
) -> dict[str, Any]:
    started = time.perf_counter()
    entities = _entities_from_pipeline(pipeline_record.pipeline, text)
    inference_s = time.perf_counter() - started

    masked = _mask_text(text, entities)
    randomized, surrogate_map = _randomize_text(text, entities, seed=seed, locale=locale)
    annotated = _entities_with_surrogates(entities, surrogate_map)

    return {
        "side": pipeline_record.side,
        "modelId": pipeline_record.model_id,
        "displayName": pipeline_record.display_name,
        "entities": annotated,
        "masked": masked,
        "randomized": randomized,
        "stats": {
            "entities": len(entities),
            "inferenceMs": round(inference_s * 1000, 1),
        },
    }


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="OpenMed Privacy Filter Multilingual Studio",
    description=(
        "Side-by-side comparison of OpenMed/privacy-filter-multilingual-mlx "
        "vs the upstream openai/privacy-filter baseline across 16 languages."
    ),
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/examples")
def api_examples() -> dict[str, Any]:
    return {
        "languages": LANGUAGE_META,
        "examples": {
            lang: [example.to_public_dict() for example in examples]
            for lang, examples in LANGUAGE_EXAMPLES.items()
        },
        "models": {
            "openmed": {
                "id": OPENMED_MODEL_ID,
                "label": OPENMED_MODEL_ID,
            },
            "baseline": {
                "id": BASELINE_DISPLAY,
                "label": BASELINE_DISPLAY,
            },
        },
    }


@app.post("/api/run")
def api_run(payload: RunRequest) -> dict[str, Any]:
    text = payload.text.strip()
    if not text:
        return {
            "status": "empty",
            "openmed": {"entities": [], "masked": "", "randomized": "", "stats": {}},
            "baseline": {"entities": [], "masked": "", "randomized": "", "stats": {}},
            "stats": {"tokens": 0},
            "note": "Empty input.",
        }

    download = bool(payload.download or ALLOW_DOWNLOAD_ENV)
    try:
        left, right = _load_pipelines(download)
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "error",
            "openmed": {"entities": [], "masked": text, "randomized": text, "stats": {}},
            "baseline": {"entities": [], "masked": text, "randomized": text, "stats": {}},
            "stats": {"tokens": len(re.findall(r"\S+", text))},
            "note": f"Model load failed: {type(exc).__name__}: {exc}",
        }

    try:
        openmed_result = _process_side(
            left, text,
            mode=payload.mode, seed=payload.seed, locale=payload.locale,
        )
        baseline_result = _process_side(
            right, text,
            mode=payload.mode, seed=payload.seed, locale=payload.locale,
        )
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "error",
            "openmed": {"entities": [], "masked": text, "randomized": text, "stats": {}},
            "baseline": {"entities": [], "masked": text, "randomized": text, "stats": {}},
            "stats": {"tokens": len(re.findall(r"\S+", text))},
            "note": f"Inference failed: {type(exc).__name__}: {exc}",
        }

    token_estimate = len(re.findall(r"\S+", text))

    return {
        "status": "live",
        "openmed": openmed_result,
        "baseline": baseline_result,
        "stats": {
            "tokens": token_estimate,
        },
        "note": (
            f"OpenMed {openmed_result['stats']['entities']} entities · "
            f"{openmed_result['stats']['inferenceMs']} ms vs "
            f"baseline {baseline_result['stats']['entities']} entities · "
            f"{baseline_result['stats']['inferenceMs']} ms"
        ),
    }
