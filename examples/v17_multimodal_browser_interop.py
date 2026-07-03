#!/usr/bin/env python3
"""v1.7 multimodal, interop, and browser-export walkthrough.

All inputs are synthetic. The example injects a tiny local redactor so it can
exercise the public APIs without model downloads or network calls.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import Any

logging.getLogger("openmed.core.models").setLevel(logging.ERROR)

from openmed.clinical.exporters.fhir import to_bundle, to_operation_outcome
from openmed.interop.fhir_bulk import deidentify_ndjson
from openmed.interop.fhir_operations import de_identify_resource
from openmed.interop.hl7v2 import redact_hl7v2
from openmed.multimodal import (
    FakeOcrEngine,
    OcrWord,
    available_ocr_engines,
    extract_asciidoc,
    redact_chatlog_jsonl,
    redact_source_text,
    redact_table,
)
from openmed.multimodal.ocr import ocr
from openmed.onnx.transformersjs import find_missing_bundle_files

ASCII_DOC_NOTE = """= Synthetic Handoff

Patient:: Casey Example
MRN:: MRN-12345
Assessment:: Diabetes follow-up, BP 128/76.
"""

CHAT_JSONL = """\
{"conversation_id":"demo-1","messages":[{"role":"user","name":"casey","content":"My name is Casey Example and my phone is 212-555-0198."}]}
"""

CSV_TEXT = """\
patient_name,mrn,dob,note
Casey Example,MRN-12345,1975-04-03,Call 212-555-0198 before discharge
"""

FHIR_PATIENT = {
    "resourceType": "Patient",
    "id": "synthetic-patient",
    "name": [{"text": "Casey Example"}],
    "telecom": [{"system": "phone", "value": "212-555-0198"}],
}

HL7_MESSAGE = (
    "\r".join(
        [
            "MSH|^~\\&|OPENMED|LAB|EHR|HOSP|202607010900||ADT^A01|MSG1|P|2.5",
            "PID|1||MRN-12345||Example^Casey||19750403|F|||1 Main St^^Boston^MA",
            "OBX|1|TX|NOTE||Call Casey Example at 212-555-0198 before discharge",
        ]
    )
    + "\r"
)

BROWSER_PIPELINE_SNIPPET = """\
import { pipeline } from "@huggingface/transformers";

const detector = await pipeline(
  "token-classification",
  "/models/openmed-pii/transformersjs",
  { device: "webgpu" },
);
const entities = await detector("Patient Casey Example called 212-555-0198.");
"""


def redact_text(text: str) -> str:
    """Small deterministic redactor for this synthetic example."""
    return (
        text.replace("Casey Example", "[PERSON]")
        .replace("Example^Casey", "[PERSON]")
        .replace("casey", "speaker_1")
        .replace("212-555-0198", "[PHONE]")
        .replace("MRN-12345", "ID_HASH_6b1d")
        .replace("1975-04-03", "1975-04-10")
    )


def deidentifier(text: str, **_: Any) -> SimpleNamespace:
    """FHIR/HL7-compatible deidentifier object."""
    return SimpleNamespace(deidentified_text=redact_text(text))


def run_markup_example() -> dict[str, Any]:
    """Extract AsciiDoc text and project normalized redaction to source text."""
    document = extract_asciidoc(ASCII_DOC_NOTE)
    start = document.text.index("Casey Example")
    redacted_source = redact_source_text(
        document,
        [(start, start + len("Casey Example"), "[PERSON]")],
    )
    return {
        "format": document.metadata["format"],
        "span_count": len(document.spans),
        "redacted_source": redacted_source,
    }


def run_ocr_example() -> dict[str, Any]:
    """Run the shared OCR contract through the deterministic fake engine."""
    engine = FakeOcrEngine(
        [
            OcrWord("Patient", (10.0, 20.0, 70.0, 35.0), 0.98),
            OcrWord("Casey", (80.0, 20.0, 125.0, 35.0), 0.97),
            OcrWord("Example", (130.0, 20.0, 190.0, 35.0), 0.97),
        ],
        source="synthetic-image",
    )
    result = ocr("synthetic-image-bytes", engine=engine, languages=["en", "fr"])
    document = result.to_document()
    return {
        "available_engines": list(available_ocr_engines()),
        "selected_languages": engine.last_languages,
        "text": result.text,
        "first_word_bbox": document.spans[0].bbox,
    }


def run_chat_and_table_examples() -> dict[str, Any]:
    """Redact chat JSONL and CSV/TSV-like tabular data."""
    chat = redact_chatlog_jsonl(
        CHAT_JSONL,
        text_redactor=redact_text,
        pseudonymize_speakers=True,
    )
    table = redact_table(
        CSV_TEXT,
        text_redactor=redact_text,
        date_shift_days=7,
    )
    return {
        "chat_summary": chat.summary.to_dict(),
        "chat_text": chat.text,
        "table_manifest": list(table.manifest),
        "table_text": table.text,
    }


def run_fhir_and_hl7_examples(tmp_dir: Path) -> dict[str, Any]:
    """Run FHIR resource, FHIR Bulk NDJSON, Bundle, and HL7 v2 paths."""
    resource = de_identify_resource(FHIR_PATIENT, deidentifier=deidentifier)
    ndjson_in = tmp_dir / "Patient.ndjson"
    ndjson_out = tmp_dir / "out" / "Patient.ndjson"
    ndjson_in.write_text(json.dumps(FHIR_PATIENT) + "\n", encoding="utf-8")
    summary = deidentify_ndjson(
        ndjson_in,
        ndjson_out,
        deidentifier=deidentifier,
    )
    bundle = to_bundle(
        [
            resource,
            {
                "resourceType": "Observation",
                "id": "bp-1",
                "status": "final",
                "subject": {"reference": "Patient/synthetic-patient"},
                "code": {"text": "Blood pressure"},
                "valueString": "BP 128/76",
            },
        ],
        doc_id="synthetic-fhir-example",
    )
    outcome = to_operation_outcome(
        [
            {
                "severity": "information",
                "code": "informational",
                "diagnostics": "Synthetic de-identification completed.",
                "expression": "Patient.name",
            }
        ]
    )
    hl7 = redact_hl7v2(
        HL7_MESSAGE,
        deidentifier=deidentifier,
        date_shift_days=7,
    )
    return {
        "resource": resource,
        "bulk_summary": asdict(summary),
        "bulk_text": ndjson_out.read_text(encoding="utf-8").strip(),
        "bundle_type": bundle["type"],
        "bundle_entry_count": len(bundle["entry"]),
        "operation_outcome": outcome,
        "hl7": hl7,
    }


def run_transformersjs_example(tmp_dir: Path) -> dict[str, Any]:
    """Show the browser bundle contract and the client-side load snippet."""
    bundle_dir = tmp_dir / "transformersjs"
    bundle_dir.mkdir()
    missing = find_missing_bundle_files(bundle_dir)
    return {
        "expected_bundle_files": missing,
        "browser_pipeline_snippet": BROWSER_PIPELINE_SNIPPET,
    }


def main() -> None:
    with TemporaryDirectory() as temp_dir:
        tmp_dir = Path(temp_dir)
        payload = {
            "markup": run_markup_example(),
            "ocr": run_ocr_example(),
            "chat_and_table": run_chat_and_table_examples(),
            "fhir_and_hl7": run_fhir_and_hl7_examples(tmp_dir),
            "transformersjs": run_transformersjs_example(tmp_dir),
        }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
