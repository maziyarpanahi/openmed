# HealthAdvocate

> **Patient-first healthcare advocacy powered by OpenMed medical NLP and local LLM reasoning**

Navigate the medical system, fight insurance denials, decode bills, and advocate for your family's health. Free, private, runs entirely on your machine.

---

## What It Does

HealthAdvocate is a patient health advocacy tool that combines **medical NER (Named Entity Recognition)** with **local LLM reasoning** to help patients understand, navigate, and fight back against the healthcare system.

It runs **100% locally** — your medical data never leaves your machine. No accounts, no cloud, no telemetry.

### 11 Features

| Feature | What It Does |
|---------|-------------|
| **Symptom Assessment** | Analyzes symptoms with NER entity extraction + LLM assessment. Cross-validates urgency between both AI layers. |
| **Document Decoder** | Translates confusing medical documents into plain language. Explains every medical term detected. |
| **Bill Decoder** | Breaks down medical bills line by line. Flags suspicious charges and explains your billing rights. |
| **Insurance Denial Fighter** | Analyzes denial letters, identifies the reason, drafts appeal arguments, and generates a ready-to-send appeal letter. |
| **Drug Checker** | Looks up drug information including side effects, warnings, interactions, and questions to ask your doctor. |
| **Appointment Prep** | Prepares talking points, questions to ask, and an advocacy script before your doctor visit. |
| **Discharge Translator** | Translates discharge instructions into clear medication schedules, warning signs, and follow-up steps. |
| **Second Opinion Brief** | Creates a structured brief from your medical records to bring to a specialist for a second opinion. |
| **Community Health Scanner** | Analyzes health bulletins and social media claims for credibility, scientific accuracy, and real risk. |
| **Family Health Tracker** | Manage health profiles for family members with conditions, medications, and allergies. Context flows into all other features. |
| **Health Tracks** | Track active health concerns over time with status updates and notes. |

---

## Architecture

```
User Input
    |
    v
[OpenMed NER]  -->  Entity extraction (diseases, drugs, anatomy, PII)
    |                 Confidence scoring
    |
    v
[PII Deidentification]  -->  Mask/remove personal data before LLM
    |
    v
[Local LLM (Meditron3-8B)]  -->  Structured JSON assessment
    |                              via LM Studio
    v
[Cross-Validation]  -->  Compare NER vs LLM output
    |                    Detect urgency disagreement
    v
Structured Response (JSON)
```

### Two-Layer AI System

1. **NER Layer (OpenMed)**: Extracts medical entities — diseases, drugs, anatomical terms, PII — with confidence scores. Runs via HuggingFace transformers.
2. **LLM Layer (Meditron3-8B)**: Generates structured assessments with urgency, action items, red flags, and plain-language explanations. Runs locally via LM Studio's OpenAI-compatible API.

These two layers **cross-validate** each other:
- Entities confirmed by both NER and LLM get high reliability
- Urgency disagreements between layers are escalated to "high" for safety
- NER-only or LLM-only findings are flagged separately

### PII Protection

All patient-facing modules deidentify text before sending to the LLM using OpenMed's HIPAA-compliant PII detection:
- Names, dates, SSNs, phone numbers, emails, addresses are masked
- The `pii_scrubbed` field in every response confirms when scrubbing occurred
- Drug names and public bulletin text are the only exceptions (no PII expected)

---

## Quick Start

### Prerequisites

- Python 3.10+
- [LM Studio](https://lmstudio.ai/) with a medical model loaded (e.g. Meditron3-8B)
- [OpenMed](https://github.com/maziyarpanahi/openmed) installed

### Install

```bash
cd healthadvocate/
pip install -r requirements.txt
pip install openmed[hf]
```

### Run

1. Start LM Studio and load your medical model (default port: 1234)

2. Set the LM Studio URL and start the server:

```bash
export LM_STUDIO_URL=http://localhost:1234/v1
uvicorn healthadvocate.app:app --host 0.0.0.0 --port 8080
```

3. Open http://localhost:8080 in your browser

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LM_STUDIO_URL` | `http://172.26.0.1:1234/v1` | LM Studio OpenAI-compatible API URL |
| `MEDICAL_LLM_MODEL` | `meditron3-8b` | Model name in LM Studio |

---

## API Reference

All endpoints accept JSON and return structured JSON responses.

### Health Check

```
GET /api/health
```

### Feature Endpoints

All feature endpoints follow the same pattern: POST with a text payload, receive structured analysis.

```
POST /api/symptoms/assess     {"symptoms": "...", "profile_id": "..."}
POST /api/documents/decode    {"text": "..."}
POST /api/bills/decode        {"bill_text": "..."}
POST /api/insurance/fight     {"denial_text": "...", "patient_info": "..."}
POST /api/drugs/check         {"drug_name": "..."}
POST /api/appointments/prepare {"symptoms": "...", "concern": "..."}
POST /api/discharge/translate {"text": "..."}
POST /api/second-opinion/create {"records": "..."}
POST /api/community/scan      {"text": "..."}
```

### Family Tracker

```
POST   /api/family/profiles                  {"name": "...", "relationship": "..."}
GET    /api/family/profiles
GET    /api/family/profiles/{id}
POST   /api/family/profiles/{id}/conditions   {"condition": "..."}
POST   /api/family/profiles/{id}/medications  {"medication": "...", "dosage": "..."}
```

### Health Tracks

```
POST   /api/tracks              {"concern": "...", "category": "..."}
GET    /api/tracks?status=...
POST   /api/tracks/{id}         {"status": "...", "note": "..."}
GET    /api/tracks/dashboard
```

### Response Format

Every feature endpoint returns:

```json
{
  "explanation": "Plain language summary",
  "urgency": "high|medium|low",
  "action_items": ["..."],
  "red_flags": ["..."],
  "pii_scrubbed": true,
  "validation": {
    "confirmed": ["entity confirmed by both NER and LLM"],
    "ner_only": ["entity found by NER only"],
    "llm_only": ["entity found by LLM only"],
    "reliability": "high|medium|low",
    "urgency_disagreement": false
  },
  "structured_output": { "...full LLM response..." }
}
```

Module-specific fields (e.g. `suspicious_charges`, `draft_appeal`, `medication_instructions`) are included alongside the common fields.

---

## Project Structure

```
healthadvocate/
  app.py                  FastAPI application, endpoints, request models
  requirements.txt        Python dependencies
  static/
    index.html            Single-page UI with WCAG 2.1 AA accessibility
    styles.css            Humanistic design system (light/dark themes)
    app.js                Frontend logic, API calls, result rendering
  core/
    engine.py             OpenMed model management and inference
    llm_client.py         LM Studio client with structured JSON parsing
    cross_validation.py   NER vs LLM cross-validation logic
    symptom_assessor.py   Symptom assessment module
    document_decoder.py   Medical document decoder
    bill_decoder.py       Medical bill analyzer with price extraction
    insurance_fighter.py  Insurance denial analyzer and appeal generator
    drug_checker.py       Drug information checker
    appointment_prep.py   Appointment preparation module
    discharge_translator.py  Discharge instruction translator
    second_opinion.py     Second opinion brief creator
    community_health.py   Community health bulletin scanner
    family_tracker.py     Family health profile management
    health_tracks.py      Health concern tracking
```

---

## Design Principles

- **Patient-first language**: Every response is written for the patient, not the clinician
- **Safety-first urgency**: If NER and LLM disagree on urgency, the system defaults to "high"
- **PII never leaves**: All text is deidentified before any LLM call
- **Two-layer validation**: NER confidence + LLM reasoning cross-checked for reliability
- **Zero telemetry**: No analytics, no tracking, no data collection

---

## Accessibility

The UI meets **WCAG 2.1 AA** standards:

- Skip-to-content link as first focusable element
- `aria-live` regions on all result containers for screen reader announcements
- Sr-only labels on all form inputs
- `:focus-visible` styles for keyboard navigation
- `prefers-reduced-motion` media query support
- Contrast ratios meet AA minimums (4.5:1 for text)
- `role="img"` on all decorative SVGs

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Backend | FastAPI, Pydantic, Uvicorn |
| Medical NER | [OpenMed](https://github.com/maziyarpanahi/openmed) (HuggingFace transformers) |
| LLM | [Meditron3-8B](https://huggingface.co/epfl-llm/meditron-3) via [LM Studio](https://lmstudio.ai/) |
| Frontend | Vanilla HTML/CSS/JS, no framework dependencies |
| Fonts | DM Sans (Google Fonts) |

---

## Known Limitations

- **In-memory storage**: Family profiles and health tracks are stored in memory and lost on server restart. This is by design for privacy — no persistent data means no data to breach.
- **LLM dependency**: Requires a running LM Studio instance with a loaded medical model. Without it, all feature endpoints will return errors.
- **JSON parsing**: The LLM generates structured JSON via prompt engineering. If the model produces malformed output, the system falls back to labeled-text parsing or raw text extraction.

---

## Credits

HealthAdvocate is built on:

- **[OpenMed](https://github.com/maziyarpanahi/openmed)** by Maziyar Panahi — Apache 2.0 licensed medical NLP toolkit with 12+ specialized NER models
- **[Meditron3-8B](https://huggingface.co/epfl-llm/meditron-3)** by EPFL LLM Team — open-weight medical LLM
- **[LM Studio](https://lmstudio.ai/)** — local LLM runtime with OpenAI-compatible API
- **[FastAPI](https://fastapi.tiangolo.com/)** — modern Python web framework
- **[HuggingFace Transformers](https://huggingface.co/docs/transformers)** — model inference backend for OpenMed NER

### OpenMed Citation

```bibtex
@misc{panahi2025openmedneropensourcedomainadapted,
      title={OpenMed NER: Open-Source, Domain-Adapted State-of-the-Art Transformers for Biomedical NER Across 12 Public Datasets},
      author={Maziyar Panahi},
      year={2025},
      eprint={2508.01630},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2508.01630},
}
```

---

## License

HealthAdvocate is part of the OpenMed project and released under the [Apache-2.0 License](../LICENSE).
