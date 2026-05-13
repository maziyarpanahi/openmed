"""HealthAdvocate — FastAPI web application."""

from __future__ import annotations

import sys
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

# Ensure project root is on path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from healthadvocate.core.engine import HealthEngine
from healthadvocate.core import (
    symptom_assessor,
    document_decoder,
    bill_decoder,
    insurance_fighter,
    drug_checker,
    appointment_prep,
    discharge_translator,
    second_opinion,
    community_health,
    family_tracker,
    health_tracks,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

engine = HealthEngine()

_MAX_INPUT_LENGTH = 50_000  # 50KB max input text


def _validate_length(text: str, field: str = "input") -> None:
    if len(text) > _MAX_INPUT_LENGTH:
        raise HTTPException(
            status_code=422,
            detail=f"{field} exceeds maximum length of {_MAX_INPUT_LENGTH} characters.",
        )

# ---------------------------------------------------------------------------
# Lifespan: pre-warm OpenMed models at startup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app):
    logger.info("Pre-loading OpenMed models...")
    await run_in_threadpool(engine.preload)
    logger.info("Models loaded. HealthAdvocate ready.")
    yield

app = FastAPI(
    title="HealthAdvocate",
    version="1.0.0",
    description="Patient healthcare advocacy powered by OpenMed medical NLP",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal error occurred. Please try again."},
    )

# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class SymptomRequest(BaseModel):
    symptoms: str
    profile_id: Optional[str] = None

class DocumentRequest(BaseModel):
    text: str
    lang: str = "en"
    profile_id: Optional[str] = None

class BillRequest(BaseModel):
    bill_text: str
    profile_id: Optional[str] = None

class DenialRequest(BaseModel):
    denial_text: str
    patient_info: str = ""
    profile_id: Optional[str] = None

class DrugRequest(BaseModel):
    drug_name: str
    profile_id: Optional[str] = None

class AppointmentRequest(BaseModel):
    symptoms: str
    concern: str = ""
    profile_id: Optional[str] = None

class DischargeRequest(BaseModel):
    text: str
    lang: str = "en"
    profile_id: Optional[str] = None

class SecondOpinionRequest(BaseModel):
    records: str
    lang: str = "en"
    profile_id: Optional[str] = None

class CommunityRequest(BaseModel):
    text: str

class FamilyProfileRequest(BaseModel):
    name: str
    relationship: str = "self"

class FamilyConditionRequest(BaseModel):
    condition: str

class FamilyMedicationRequest(BaseModel):
    medication: str
    dosage: str = ""

class TrackRequest(BaseModel):
    concern: str
    category: str = "general"

class TrackUpdateRequest(BaseModel):
    status: str = ""
    note: str = ""

# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/api/health")
async def health_check():
    return {"status": "ok", "service": "HealthAdvocate", "version": "1.0.0"}

# ---------------------------------------------------------------------------
# Feature endpoints
# ---------------------------------------------------------------------------

@app.post("/api/symptoms/assess")
async def assess_symptoms(request: SymptomRequest):
    _validate_length(request.symptoms, "Symptoms")
    result = await run_in_threadpool(
        symptom_assessor.assess_symptoms, engine, request.symptoms, request.profile_id
    )
    return result

@app.post("/api/documents/decode")
async def decode_document(request: DocumentRequest):
    _validate_length(request.text, "Document text")
    result = await run_in_threadpool(
        document_decoder.decode_document, engine, request.text, request.lang, request.profile_id
    )
    return result

@app.post("/api/bills/decode")
async def decode_bill(request: BillRequest):
    _validate_length(request.bill_text, "Bill text")
    result = await run_in_threadpool(
        bill_decoder.decode_bill, engine, request.bill_text, request.profile_id
    )
    return result

@app.post("/api/insurance/fight")
async def fight_denial(request: DenialRequest):
    _validate_length(request.denial_text, "Denial text")
    result = await run_in_threadpool(
        insurance_fighter.fight_denial, engine, request.denial_text, request.patient_info, request.profile_id
    )
    return result

@app.post("/api/drugs/check")
async def check_drug(request: DrugRequest):
    result = await run_in_threadpool(
        drug_checker.check_drug, engine, request.drug_name, request.profile_id
    )
    return result

@app.post("/api/appointments/prepare")
async def prepare_appointment(request: AppointmentRequest):
    _validate_length(request.symptoms, "Symptoms")
    result = await run_in_threadpool(
        appointment_prep.prepare_appointment, engine, request.symptoms, request.concern, request.profile_id
    )
    return result

@app.post("/api/discharge/translate")
async def translate_discharge(request: DischargeRequest):
    _validate_length(request.text, "Discharge text")
    result = await run_in_threadpool(
        discharge_translator.translate_discharge, engine, request.text, request.lang, request.profile_id
    )
    return result

@app.post("/api/second-opinion/create")
async def create_second_opinion(request: SecondOpinionRequest):
    _validate_length(request.records, "Medical records")
    result = await run_in_threadpool(
        second_opinion.create_brief, engine, request.records, request.lang, request.profile_id
    )
    return result

@app.post("/api/community/scan")
async def scan_community(request: CommunityRequest):
    _validate_length(request.text, "Bulletin text")
    result = await run_in_threadpool(
        community_health.scan_bulletin, engine, request.text
    )
    return result

# ---------------------------------------------------------------------------
# Family tracker endpoints
# ---------------------------------------------------------------------------

@app.post("/api/family/profiles")
async def create_family_profile(request: FamilyProfileRequest):
    return family_tracker.create_profile(request.name, request.relationship)

@app.get("/api/family/profiles")
async def list_family_profiles():
    return family_tracker.list_profiles()

@app.get("/api/family/profiles/{profile_id}")
async def get_family_profile(profile_id: str):
    result = family_tracker.get_profile(profile_id)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result

@app.post("/api/family/profiles/{profile_id}/conditions")
async def add_family_condition(profile_id: str, request: FamilyConditionRequest):
    result = family_tracker.add_condition(profile_id, request.condition)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result

@app.post("/api/family/profiles/{profile_id}/medications")
async def add_family_medication(profile_id: str, request: FamilyMedicationRequest):
    result = family_tracker.add_medication(profile_id, request.medication, request.dosage)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result

# ---------------------------------------------------------------------------
# Health tracks endpoints
# ---------------------------------------------------------------------------

@app.post("/api/tracks")
async def create_track(request: TrackRequest):
    return health_tracks.create_track(request.concern, request.category)

@app.get("/api/tracks")
async def list_tracks(status: Optional[str] = None):
    return health_tracks.list_tracks(status)

@app.post("/api/tracks/{track_id}")
async def update_track(track_id: str, request: TrackUpdateRequest):
    result = health_tracks.update_track(track_id, request.status, request.note)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result

@app.get("/api/tracks/dashboard")
async def get_dashboard():
    return health_tracks.get_dashboard()

# ---------------------------------------------------------------------------
# Static files and SPA
# ---------------------------------------------------------------------------

_static_dir = Path(__file__).parent / "static"
if _static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")

@app.get("/")
async def index():
    return FileResponse(str(_static_dir / "index.html"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("healthadvocate.app:app", host="0.0.0.0", port=8080, reload=True)
