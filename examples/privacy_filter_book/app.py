"""Interactive Privacy Filter MLX vs CPU book demo.

Run from the repository root with:

    uvicorn examples.privacy_filter_book.app:app --reload --port 8765

The live model path is intentionally cache-first so the UI previews quickly.
Pass {"download": true} to /api/run, or set OPENMED_PRIVACY_FILTER_DOWNLOAD=1,
to allow first-run Hugging Face downloads.
"""

from __future__ import annotations

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
from pydantic import BaseModel


ROOT = Path(__file__).resolve().parent
STATIC_DIR = ROOT / "static"

MLX_MODEL_ID = os.getenv(
    "OPENMED_PRIVACY_FILTER_MLX_MODEL",
    "OpenMed/privacy-filter-mlx-8bit",
)
CPU_MODEL_ID = os.getenv(
    "OPENMED_PRIVACY_FILTER_CPU_MODEL",
    "openai/privacy-filter",
)
ALLOW_DOWNLOAD_ENV = os.getenv("OPENMED_PRIVACY_FILTER_DOWNLOAD", "").lower() in {
    "1",
    "true",
    "yes",
}


@dataclass(frozen=True)
class DemoDocument:
    id: int
    title: str
    codename: str
    date: str
    class_name: str
    languages: str
    paragraphs: tuple[str, ...]

    @property
    def text(self) -> str:
        return "\n\n".join(self.paragraphs)

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "codename": self.codename,
            "date": self.date,
            "className": self.class_name,
            "languages": self.languages,
            "paragraphs": list(self.paragraphs),
            "text": self.text,
        }


DOCUMENTS: tuple[DemoDocument, ...] = (
    DemoDocument(
        id=0,
        title="Paper Lantern",
        codename="OPF-1944-001",
        date="17 Apr 1944",
        class_name="Confidential",
        languages="English / Deutsch / Francais",
        paragraphs=(
            "Berlin desk reported that Captain Ernst Vogel left a note at Unter den Linden 17, Berlin, before dawn. The note named Elise Moreau as courier and asked that replies go to elise.moreau@maquis.example before 21 April 1944.",
            "Der zweite Absatz war harmlos geschrieben: Bitte rufen Sie 01 44 55 19 40 an, falls die Lampe am Bahnhof flackert. The same line carried the passphrase VIOLET-WINDSOR-1944 and the account number Konto 4481-77-209.",
            "Le message final disait: rendez-vous au Cafe du Pont, 12 Rue de la Paix, Paris, le 21 avril 1944. Keep the weather sentence intact, but protect the people, address, date, account, phone, email, and secret.",
        ),
    ),
    DemoDocument(
        id=1,
        title="Station X Digest",
        codename="OPF-1944-002",
        date="03 May 1944",
        class_name="Most Secret",
        languages="English / Deutsch",
        paragraphs=(
            "Dr. Margaret Haines received a coastal intercept from Lt. Otto Klein on May 3, 1944. Her reply address was m.haines@stationx.example, and the dead-drop instructions were posted at https://cipher.example/ultra/drop?key=bravo-7.",
            "Der Funker schrieb: mein Ersatzcode ist KRONEN-29-ALPHA; bitte anrufen unter +49 30 5550 1944, wenn die Nordsee still bleibt. The public weather phrase should remain readable.",
            "A clerk added the ledger line ACCT-0091-7782 and the fallback address 221B Baker Street, London NW1. The digest should retain its wartime mood while hiding operational identifiers.",
        ),
    ),
    DemoDocument(
        id=2,
        title="Marseille Harbor",
        codename="OPF-1944-003",
        date="04 Jun 1944",
        class_name="Restricted",
        languages="English / Francais",
        paragraphs=(
            "Agent Clara Whitfield met Henri Duval near 22 Boulevard Longchamp, Marseille, on 4 juin 1944. Duval wrote that the harbor ledger should be sent to hduval@resistance.example.",
            "Le telephone de secours etait +33 4 91 55 01 18. The notebook also included https://docs.example.org/liberte/recover?token=82KlmQp9, which looked like logistics but behaved like a private recovery URL.",
            "The final password phrase was silver-maple-lune-88, filed beside ticket MRS-7741. Bitte nicht den historischen Kontext entfernen; nur Namen, Kontaktdaten, Adressen, Daten, URLs und Geheimnisse schuetzen.",
        ),
    ),
)


class RunRequest(BaseModel):
    page_id: int = 0
    download: bool = False


def _find_document(page_id: int) -> DemoDocument:
    for document in DOCUMENTS:
        if document.id == page_id:
            return document
    return DOCUMENTS[0]


def _count_tokens(text: str) -> int:
    try:
        import tiktoken

        encoding = tiktoken.get_encoding("o200k_base")
        return len(encoding.encode(text, allowed_special="all"))
    except (ImportError, ModuleNotFoundError):
        return max(1, len(re.findall(r"\S+", text)))
    except Exception as exc:
        import logging
        logging.getLogger(__name__).warning(
            "tiktoken token counting failed, falling back to word count: %s", exc
        )
        return max(1, len(re.findall(r"\S+", text)))


def _label_from_raw(raw_label: str) -> str:
    label = raw_label.strip()
    if len(label) > 2 and label[1] == "-" and label[0] in {"B", "I", "E", "S"}:
        label = label[2:]
    label = label.lower().replace(" ", "_").replace("-", "_")
    aliases = {
        "email": "private_email",
        "url": "private_url",
        "phone": "private_phone",
        "date": "private_date",
        "person": "private_person",
        "name": "private_person",
        "address": "private_address",
        "account": "account_number",
        "account_id": "account_number",
        "password": "secret",
    }
    return aliases.get(label, label)


def _normalize_entities(raw_entities: list[dict[str, Any]], text: str) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for item in raw_entities:
        start = item.get("start")
        end = item.get("end")
        if start is None or end is None:
            word = str(item.get("word", "")).strip()
            if not word:
                continue
            start = text.find(word)
            if start < 0:
                continue
            end = start + len(word)

        try:
            start_i = int(start)
            end_i = int(end)
        except (TypeError, ValueError):
            continue

        if start_i < 0 or end_i <= start_i or end_i > len(text):
            continue

        raw_label = str(item.get("entity_group") or item.get("entity") or item.get("label") or "private")
        score = item.get("score", item.get("confidence", 0.0))
        try:
            score_f = float(score)
        except (TypeError, ValueError):
            score_f = 0.0

        normalized.append(
            {
                "label": _label_from_raw(raw_label),
                "start": start_i,
                "end": end_i,
                "text": text[start_i:end_i],
                "score": round(score_f, 4),
            }
        )

    return _dedupe_entities(normalized)


def _dedupe_entities(entities: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for entity in sorted(
        entities,
        key=lambda item: (item["start"], -(item["end"] - item["start"]), item["label"]),
    ):
        overlaps = any(
            not (entity["end"] <= existing["start"] or entity["start"] >= existing["end"])
            for existing in selected
        )
        if not overlaps:
            selected.append(entity)
    return sorted(selected, key=lambda item: (item["start"], item["end"], item["label"]))


EXACT_ENTITY_HINTS: tuple[tuple[str, str], ...] = (
    ("private_person", "Captain Ernst Vogel"),
    ("private_person", "Elise Moreau"),
    ("private_address", "Unter den Linden 17, Berlin"),
    ("private_address", "Cafe du Pont"),
    ("private_address", "12 Rue de la Paix, Paris"),
    ("private_date", "21 April 1944"),
    ("private_date", "21 avril 1944"),
    ("account_number", "Konto 4481-77-209"),
    ("secret", "VIOLET-WINDSOR-1944"),
    ("private_person", "Dr. Margaret Haines"),
    ("private_person", "Lt. Otto Klein"),
    ("private_date", "May 3, 1944"),
    ("secret", "KRONEN-29-ALPHA"),
    ("account_number", "ACCT-0091-7782"),
    ("private_address", "221B Baker Street, London NW1"),
    ("private_person", "Agent Clara Whitfield"),
    ("private_person", "Henri Duval"),
    ("private_address", "22 Boulevard Longchamp, Marseille"),
    ("private_date", "4 juin 1944"),
    ("secret", "silver-maple-lune-88"),
    ("account_number", "MRS-7741"),
)

REGEX_ENTITY_HINTS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("private_email", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")),
    ("private_url", re.compile(r"\bhttps?://[^\s,;]+")),
    ("private_phone", re.compile(r"(?:\+\d{1,3}[\s.-]?)?(?:\(?\d{1,4}\)?[\s.-]?){3,6}\d{2,4}")),
    ("private_date", re.compile(r"\b\d{1,2}\s+(?:Apr|April|May|Jun|June|juin|avril)\s+\d{4}\b", re.IGNORECASE)),
)


def _fallback_entities(text: str) -> list[dict[str, Any]]:
    entities: list[dict[str, Any]] = []

    for label, literal in EXACT_ENTITY_HINTS:
        start = 0
        while True:
            found = text.find(literal, start)
            if found == -1:
                break
            entities.append(
                {
                    "label": label,
                    "start": found,
                    "end": found + len(literal),
                    "text": literal,
                    "score": 0.94,
                }
            )
            start = found + len(literal)

    for label, pattern in REGEX_ENTITY_HINTS:
        for match in pattern.finditer(text):
            candidate = match.group(0).rstrip(".")
            entities.append(
                {
                    "label": label,
                    "start": match.start(),
                    "end": match.start() + len(candidate),
                    "text": candidate,
                    "score": 0.91,
                }
            )

    return _dedupe_entities(entities)


def _fallback_result(
    *,
    engine: Literal["mlx", "cpu"],
    text: str,
    reason: str,
) -> dict[str, Any]:
    tokens = _count_tokens(text)
    base_tps = 1180.0 if engine == "mlx" else 182.0
    total_s = max(0.08, tokens / base_tps)
    jitter = (len(text) % 17) / 1000
    total_s += jitter
    entities = _fallback_entities(text)
    return {
        "engine": engine,
        "status": "fallback",
        "modelId": MLX_MODEL_ID if engine == "mlx" else CPU_MODEL_ID,
        "runtime": "Python + MLX" if engine == "mlx" else "Python + Torch CPU",
        "note": reason,
        "entities": entities,
        "stats": {
            "tokens": tokens,
            "entities": len(entities),
            "loadMs": 0.0,
            "inferenceMs": round(total_s * 1000, 1),
            "totalMs": round(total_s * 1000, 1),
            "tokensPerSecond": round(tokens / total_s, 1),
        },
    }


@lru_cache(maxsize=1)
def _load_mlx_pipeline(download: bool) -> tuple[Any, float]:
    started = time.perf_counter()
    from huggingface_hub import snapshot_download
    from openmed.mlx.inference import create_mlx_pipeline

    model_path = snapshot_download(
        repo_id=MLX_MODEL_ID,
        repo_type="model",
        local_files_only=not download,
    )
    pipeline = create_mlx_pipeline(model_path)
    return pipeline, time.perf_counter() - started


@lru_cache(maxsize=1)
def _load_cpu_pipeline(download: bool) -> tuple[Any, Any, float]:
    started = time.perf_counter()
    from transformers import AutoTokenizer

    from openmed.torch.privacy_filter import PrivacyFilterTorchPipeline

    classifier = PrivacyFilterTorchPipeline(
        CPU_MODEL_ID,
        device="cpu",
        aggregation_strategy="simple",
        local_files_only=not download,
    )
    # Expose the tokenizer separately for the token-count UI metric.
    tokenizer = classifier.tokenizer
    return classifier, tokenizer, time.perf_counter() - started


def _run_mlx(text: str, download: bool) -> dict[str, Any]:
    allow_download = bool(download or ALLOW_DOWNLOAD_ENV)
    try:
        pipeline, load_s = _load_mlx_pipeline(allow_download)
        tokens = _count_tokens(text)
        started = time.perf_counter()
        raw_entities = pipeline(text)
        inference_s = time.perf_counter() - started
        entities = _normalize_entities(raw_entities, text)
        total_s = load_s + inference_s
        return {
            "engine": "mlx",
            "status": "live",
            "modelId": MLX_MODEL_ID,
            "runtime": "Python + MLX",
            "note": "OpenAI privacy-filter 8-bit MLX artifact through OpenMed.",
            "entities": entities,
            "stats": {
                "tokens": tokens,
                "entities": len(entities),
                "loadMs": round(load_s * 1000, 1),
                "inferenceMs": round(inference_s * 1000, 1),
                "totalMs": round(total_s * 1000, 1),
                "tokensPerSecond": round(tokens / max(inference_s, 1e-6), 1),
            },
        }
    except Exception as exc:
        return _fallback_result(
            engine="mlx",
            text=text,
            reason=(
                "Live MLX model was not available from local cache. "
                f"{type(exc).__name__}: {exc}"
            ),
        )


def _run_cpu(text: str, download: bool) -> dict[str, Any]:
    allow_download = bool(download or ALLOW_DOWNLOAD_ENV)
    try:
        classifier, tokenizer, load_s = _load_cpu_pipeline(allow_download)
        tokens = len(tokenizer(text, return_tensors=None)["input_ids"])
        started = time.perf_counter()
        raw_entities = classifier(text)
        inference_s = time.perf_counter() - started
        entities = _normalize_entities(raw_entities, text)
        total_s = load_s + inference_s
        return {
            "engine": "cpu",
            "status": "live",
            "modelId": CPU_MODEL_ID,
            "runtime": "Python + Torch CPU",
            "note": "OpenAI privacy-filter through Transformers on CPU.",
            "entities": entities,
            "stats": {
                "tokens": tokens,
                "entities": len(entities),
                "loadMs": round(load_s * 1000, 1),
                "inferenceMs": round(inference_s * 1000, 1),
                "totalMs": round(total_s * 1000, 1),
                "tokensPerSecond": round(tokens / max(inference_s, 1e-6), 1),
            },
        }
    except Exception as exc:
        return _fallback_result(
            engine="cpu",
            text=text,
            reason=(
                "Live CPU model was not available from local cache. "
                f"{type(exc).__name__}: {exc}"
            ),
        )


app = FastAPI(
    title="OpenMed Privacy Filter Book",
    description="Side-by-side OpenAI privacy-filter MLX vs CPU comparison demo.",
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/favicon.ico")
def favicon() -> FileResponse:
    return FileResponse(STATIC_DIR / "assets" / "logo.svg", media_type="image/svg+xml")


@app.get("/api/documents")
def documents() -> dict[str, Any]:
    return {
        "documents": [document.to_public_dict() for document in DOCUMENTS],
        "models": {
            "mlx": MLX_MODEL_ID,
            "cpu": CPU_MODEL_ID,
        },
    }


@app.post("/api/run")
def run_privacy_filter(payload: RunRequest) -> dict[str, Any]:
    document = _find_document(payload.page_id)
    return {
        "document": document.to_public_dict(),
        "results": {
            "mlx": _run_mlx(document.text, payload.download),
            "cpu": _run_cpu(document.text, payload.download),
        },
    }
