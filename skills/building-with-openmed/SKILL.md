---
name: building-with-openmed
description: "Orient and bootstrap any project that uses OpenMed, the on-device clinical and biomedical NLP library, for named-entity recognition, PHI de-identification, FHIR export, and evaluation. Use when the user mentions OpenMed, wants to install it, asks which OpenMed capability or model fits a task, or is starting to build a clinical/medical text pipeline and needs the right entry point."
license: Apache-2.0
metadata:
  project: OpenMed
  category: openmed-core
  pairs: adjacent
  version: "1.0"
---

# Building with OpenMed

OpenMed is an Apache-2.0, **local-first** Python library for clinical and
biomedical NLP. Models download once from the Hugging Face Hub and then run
**fully on-device** — no network calls, no telemetry, no raw PHI in logs,
caches, or temp files. This skill is the map: it tells you what OpenMed can do
and which focused skill (or API) to reach for next.

## When to use this skill

Use it to scope a task and pick an entry point. For the actual work, hand off to
the focused OpenMed skills (each is grounded in the real API):

| Task | Skill / API |
| --- | --- |
| Find and load a model | `loading-openmed-models`, `choosing-openmed-models` |
| Run clinical/biomedical NER | `extracting-clinical-entities` (`openmed.analyze_text`) |
| Zero-shot NER (no fine-tune) | `running-zeroshot-ner` (`openmed zero`) |
| Remove / mask PHI | `deidentifying-clinical-text` (`openmed.deidentify`) |
| Detect PHI spans only | `extracting-pii-entities` (`openmed.extract_pii`) |
| Restore masked PHI | `reidentifying-text` (`openmed.reidentify`) |
| Pick a privacy policy | `configuring-privacy-policies` (7 bundled profiles) |
| Non-English PHI | `deidentifying-multilingual-text` |
| Signed, no-PHI audit | `auditing-deidentification-runs` (`audit=True`) |
| Negation / temporality | `resolving-clinical-context` (`openmed.clinical`) |
| Evaluate with leakage gates | `evaluating-with-leakage-gates` (`openmed.eval`) |
| FHIR R4 export | `exporting-to-fhir` (`openmed.interop`) |
| Serve REST / MCP | `serving-openmed-rest-api`, `deploying-openmed-mcp` |
| Run on Apple Silicon / edge | `running-openmed-ondevice` (MLX / CoreML / ONNX) |

## Install

```bash
pip install openmed                 # core: NER + de-identification
pip install "openmed[hf]"           # add Hugging Face model downloads
pip install "openmed[mcp]"          # Model Context Protocol server
pip install "openmed[service]"      # FastAPI REST service
pip install "openmed[mlx]"          # Apple Silicon acceleration
pip install "openmed[presidio]"     # Microsoft Presidio bridge
```

Extras map to capabilities: `cli`, `mcp`, `service`, `presidio`, `spacy`,
`langchain`, `gliner` (zero-shot), `multimodal`/`ocr-paddle` (document intake),
`mlx`/`coreml`/`onnx` (on-device backends), `hf` (model hub), `dev` (tests/lint).

## The three core calls

```python
import openmed

# 1) Named-entity recognition (token classification)
result = openmed.analyze_text(
    "Patient prescribed 500 mg metformin for type 2 diabetes.",
    model_name="disease_detection_superclinical",  # registry key, HF id, or local path
    output_format="dict",                           # dict | json | html | csv
)

# 2) De-identify PHI (mask | remove | replace | hash | shift_dates)
deid = openmed.deidentify(
    "John Doe (MRN 12345) seen on 2024-03-02.",
    method="replace",
    policy="hipaa_safe_harbor",   # bundled policy profile
)
print(deid.deidentified_text)     # PHI removed; deid.pii_entities lists the spans

# 3) Detect PHI spans without changing the text
pii = openmed.extract_pii("Call Dr. Smith at 617-555-0123.")  # PredictionResult
spans = pii.entities                                          # the PHI spans
```

`analyze_text` and `deidentify` are the workhorses. Everything else
(multilingual, audit, policies, FHIR, eval) layers on top of these.

## Discover what is available at runtime

Never hardcode model lists or language counts — query them:

```python
import openmed
openmed.list_model_categories()          # e.g. Privacy, Disease, Oncology, Genomics ...
openmed.get_models_by_category("Disease")
openmed.get_pii_models_by_language("es")
from openmed.core.pii_i18n import SUPPORTED_LANGUAGES   # de-id language set
```

CLI equivalents: `openmed models list`, `openmed models info <key>`,
`openmed analyze --text "<text>" --model <key> --format json`. MCP/REST expose
the same surface as tools (`openmed_analyze_text`, `openmed_deidentify`,
`openmed_list_models`, …).

## Non-negotiable rules when building with OpenMed

- **Local-first.** Do not add cloud calls to PHI workflows. Models run on-device
  after a one-time download.
- **No raw PHI in artifacts.** Logs, caches, audit reports, and error messages
  must use offsets, hashes, and labels — never plaintext identifiers. Use
  `audit=True` for tamper-evident, no-PHI audit output.
- **Permissive licensing only.** Do not bundle UMLS, SNOMED CT, CPT, MIMIC,
  i2b2, or n2c2 assets. Call restricted terminologies out-of-process with the
  user's own credentials.
- **De-identification is verified, not assumed.** Gate on leakage with
  `openmed.eval`, not on F1 alone (see `evaluating-with-leakage-gates`).
- **Clinical safety.** OpenMed assists; it does not make autonomous clinical
  decisions. Surface disclaimers for any borderline medical-device behavior.

## A typical pipeline

```
ingest (HL7v2 / FHIR / C-CDA / OCR)
   → de-identify (openmed.deidentify, policy=…)
   → extract entities (openmed.analyze_text)
   → ground to terminology (out-of-process: RxNorm / LOINC / SNOMED)
   → assemble FHIR (openmed.interop)
   → evaluate (openmed.eval leakage gates)
```

Each stage has a companion skill in this directory. Start here, then jump to the
stage you need.
