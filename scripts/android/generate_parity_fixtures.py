"""Generate Android span parity fixtures from an Android ONNX export.

The fixture pins the Python reference path for an exported token-classification
model: tokenizer IDs, character offsets, argmax labels, and decoded spans. The
Android module can load the JSON resource and assert that on-device tokenization
and ONNX Runtime Mobile inference produce the same contract.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from openmed.core.labels import normalize_label, policy_label_for
from openmed.core.schemas import CURRENT_SCHEMA_VERSION, hmac_text_hash
from openmed.onnx.android_profile import (
    ANDROID_ONNX_FORMAT,
    ANDROID_ONNX_OPSET,
    ANDROID_PROFILE_NAME,
    validate_android_profile,
)
from openmed.onnx.convert import DEFAULT_ONNX_FILENAME, MANIFEST_FILENAME

FIXTURE_FORMAT = "openmed-android-span-parity"
FIXTURE_FORMAT_VERSION = 1
PARITY_HASH_SECRET = "openmed-android-span-parity-v1"
DEFAULT_OUTPUT_PATH = Path(
    "android/openmedkit/src/test/resources/parity/android_span_parity.json"
)
DEFAULT_DOCS_PATH = "docs/android-parity.md"
DEFAULT_GENERATOR_PATH = "scripts/android/generate_parity_fixtures.py"


@dataclass(frozen=True)
class ParityCase:
    """Synthetic text used to pin Python and Android span parity."""

    id: str
    text: str


DEFAULT_SYNTHETIC_CASES: tuple[ParityCase, ...] = (
    ParityCase(
        id="synth-note-001",
        text=("SYNTH_PATIENT_ALPHA visited SYNTH_CLINIC_BETA on SYNTH_DATE_2099Q4D18."),
    ),
    ParityCase(
        id="synth-note-002",
        text=("Route results to SYNTH_PERSON_BRAVO through SYNTH_PHONE_TOKEN_42."),
    ),
)

_BIOES_RE = re.compile(r"^([BIES])-([A-Za-z0-9_ -]+)$")
_SUSPICIOUS_PHI_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    re.compile(r"\b(?:\+?1[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b"),
    re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    re.compile(r"\b(?:John|Jane)\s+Doe\b", re.IGNORECASE),
)


def build_fixture_payload(
    export_dir: str | Path,
    *,
    cases: Sequence[ParityCase] = DEFAULT_SYNTHETIC_CASES,
    validate_model: bool = True,
) -> dict[str, Any]:
    """Return a deterministic Android parity fixture payload.

    Args:
        export_dir: Directory produced by ``openmed.onnx.convert --profile android``.
        cases: Synthetic cases to run through the Python reference.
        validate_model: When true, validate the ONNX graph against the Android
            profile before generating fixture content.

    Returns:
        JSON-serializable fixture payload.
    """

    export_path = Path(export_dir)
    metadata, id2label = _load_android_export(export_path)
    model_path = export_path / metadata["model_path"]
    if validate_model:
        validate_android_profile(model_path)

    case_payloads = _run_reference(export_path, metadata, id2label, cases)
    payload = {
        "format": FIXTURE_FORMAT,
        "format_version": FIXTURE_FORMAT_VERSION,
        "openmed_span_schema_version": CURRENT_SCHEMA_VERSION,
        "android_profile": {
            "profile": ANDROID_PROFILE_NAME,
            "onnx_format": ANDROID_ONNX_FORMAT,
            "opset": ANDROID_ONNX_OPSET,
        },
        "contract": _contract_payload(),
        "paths": {
            "android_resource": DEFAULT_OUTPUT_PATH.as_posix(),
            "docs": DEFAULT_DOCS_PATH,
            "generator": DEFAULT_GENERATOR_PATH,
        },
        "model": metadata,
        "cases": case_payloads,
    }
    validate_fixture_payload(payload)
    return payload


def validate_fixture_payload(payload: Mapping[str, Any]) -> None:
    """Validate the committed fixture privacy and structural contract."""

    if payload.get("format") != FIXTURE_FORMAT:
        raise ValueError(f"fixture format must be {FIXTURE_FORMAT!r}")
    if payload.get("format_version") != FIXTURE_FORMAT_VERSION:
        raise ValueError(f"fixture format_version must be {FIXTURE_FORMAT_VERSION}")

    cases = payload.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("fixture must contain at least one parity case")

    seen_ids: set[str] = set()
    for case in cases:
        if not isinstance(case, Mapping):
            raise ValueError("each fixture case must be an object")
        case_id = str(case.get("id") or "")
        if not case_id or case_id in seen_ids:
            raise ValueError("fixture case ids must be unique and non-empty")
        seen_ids.add(case_id)

        text = str(case.get("text") or "")
        if not text or "SYNTH_" not in text:
            raise ValueError(f"{case_id} must use explicit SYNTH_ placeholders")
        if case.get("synthetic") is not True or case.get("phi_free") is not True:
            raise ValueError(f"{case_id} must be marked synthetic and phi_free")
        for pattern in _SUSPICIOUS_PHI_PATTERNS:
            if pattern.search(text):
                raise ValueError(
                    f"{case_id} contains PHI-shaped text: {pattern.pattern}"
                )

        _validate_tokens(case_id, text, case.get("tokens"))
        _validate_spans(case_id, text, case.get("spans"))


def write_fixture(payload: Mapping[str, Any], output_path: str | Path) -> Path:
    """Write a fixture payload as deterministic JSON."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def load_cases(path: str | Path | None) -> tuple[ParityCase, ...]:
    """Load synthetic cases from JSON, or return the default case set."""

    if path is None:
        return DEFAULT_SYNTHETIC_CASES
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("cases JSON must be a list of objects with id and text")
    return tuple(
        ParityCase(id=str(item["id"]), text=str(item["text"])) for item in payload
    )


def _load_android_export(export_dir: Path) -> tuple[dict[str, Any], dict[int, str]]:
    manifest_path = export_dir / MANIFEST_FILENAME
    manifest = _read_json_if_exists(manifest_path)
    if manifest and not _manifest_has_android_artifact(manifest):
        raise ValueError(
            f"{manifest_path} does not describe an {ANDROID_ONNX_FORMAT} artifact"
        )

    artifact = _android_artifact(manifest)
    model_rel = str(artifact.get("path") or DEFAULT_ONNX_FILENAME)
    label_rel = str(manifest.get("label_map_path") or "id2label.json")
    id2label_path = export_dir / label_rel
    if not id2label_path.exists():
        raise FileNotFoundError(f"missing Android label map: {id2label_path}")

    id2label_raw = json.loads(id2label_path.read_text(encoding="utf-8"))
    if not isinstance(id2label_raw, Mapping) or not id2label_raw:
        raise ValueError(f"{id2label_path} must contain a non-empty id2label object")
    id2label = {int(index): str(label) for index, label in id2label_raw.items()}

    metadata = {
        "source_model_id": str(manifest.get("source_model_id") or "unknown"),
        "model_path": model_rel,
        "manifest_path": MANIFEST_FILENAME if manifest else None,
        "label_map_path": label_rel,
        "tokenizer_path": str((manifest.get("tokenizer") or {}).get("path") or "."),
        "max_sequence_length": int(manifest.get("max_sequence_length") or 512),
    }
    return metadata, id2label


def _read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _manifest_has_android_artifact(manifest: Mapping[str, Any]) -> bool:
    if ANDROID_ONNX_FORMAT in set(manifest.get("formats") or ()):
        return True
    return bool(_android_artifact(manifest))


def _android_artifact(manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    artifacts = manifest.get("artifacts") or ()
    if isinstance(artifacts, Sequence):
        for artifact in artifacts:
            if (
                isinstance(artifact, Mapping)
                and artifact.get("format") == ANDROID_ONNX_FORMAT
            ):
                return artifact
    return {}


def _run_reference(
    export_dir: Path,
    metadata: Mapping[str, Any],
    id2label: Mapping[int, str],
    cases: Sequence[ParityCase],
) -> list[dict[str, Any]]:
    try:
        import onnxruntime as ort
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "onnxruntime and transformers are required to generate Android "
            "parity fixtures. Install with: pip install openmed[onnx]"
        ) from exc

    tokenizer_path = export_dir / str(metadata.get("tokenizer_path") or ".")
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), use_fast=True)
    session = ort.InferenceSession(
        str(export_dir / str(metadata["model_path"])),
        providers=["CPUExecutionProvider"],
    )
    session_input_names = {item.name for item in session.get_inputs()}
    max_length = int(metadata.get("max_sequence_length") or 512)

    payloads: list[dict[str, Any]] = []
    for case in cases:
        encoding = tokenizer(
            case.text,
            max_length=max_length,
            padding=False,
            truncation=True,
            return_offsets_mapping=True,
            return_tensors="np",
        )
        input_ids = [int(value) for value in _first_row(encoding["input_ids"])]
        offsets = [
            _offset_pair(value) for value in _first_row(encoding["offset_mapping"])
        ]
        attention_mask = [
            int(value)
            for value in _first_row(
                encoding.get("attention_mask", [1] * len(input_ids))
            )
        ]
        model_inputs = {
            name: encoding[name] for name in session_input_names if name in encoding
        }
        logits = _first_row(session.run(["logits"], model_inputs)[0])
        tokens = _decode_tokens(input_ids, offsets, attention_mask, logits, id2label)
        spans = _decode_spans(case.id, case.text, tokens)
        payloads.append(
            {
                "id": case.id,
                "synthetic": True,
                "phi_free": True,
                "text": case.text,
                "tokens": tokens,
                "spans": spans,
            }
        )
    return payloads


def _decode_tokens(
    input_ids: Sequence[int],
    offsets: Sequence[tuple[int, int]],
    attention_mask: Sequence[int],
    logits: Sequence[Sequence[float]],
    id2label: Mapping[int, str],
) -> list[dict[str, Any]]:
    if not (len(input_ids) == len(offsets) == len(attention_mask) == len(logits)):
        raise ValueError("token ids, offsets, attention mask, and logits must align")

    tokens: list[dict[str, Any]] = []
    for index, token_id in enumerate(input_ids):
        row = [float(value) for value in logits[index]]
        label_id = _argmax_lowest_id(row)
        start, end = offsets[index]
        tokens.append(
            {
                "index": index,
                "id": int(token_id),
                "offset": [start, end],
                "attention_mask": int(attention_mask[index]),
                "label_id": label_id,
                "label": id2label.get(label_id, f"LABEL_{label_id}"),
                "score": round(_softmax_score(row, label_id), 6),
                "special": _is_special_offset(start, end)
                or int(attention_mask[index]) == 0,
            }
        )
    return tokens


def _decode_spans(
    doc_id: str,
    text: str,
    tokens: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    spans: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None

    def flush() -> None:
        nonlocal current
        if current is None:
            return
        start = int(current["start"])
        end = int(current["end"])
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        if end > start:
            raw_label = str(current["raw_label"])
            canonical = normalize_label(raw_label)
            scores = [float(score) for score in current["scores"]]
            surface = text[start:end]
            spans.append(
                {
                    "schema_version": CURRENT_SCHEMA_VERSION,
                    "doc_id": doc_id,
                    "start": start,
                    "end": end,
                    "token_start": int(current["token_start"]),
                    "token_end": int(current["token_end"]),
                    "text_hash": hmac_text_hash(surface, PARITY_HASH_SECRET),
                    "entity_type": canonical.lower(),
                    "raw_label": raw_label,
                    "canonical_label": canonical,
                    "policy_label": policy_label_for(canonical),
                    "score": round(sum(scores) / len(scores), 6),
                    "detector": "android_onnx_parity",
                }
            )
        current = None

    for token in tokens:
        if bool(token.get("special")):
            continue
        label = str(token["label"])
        tag, raw_label = _split_label(label)
        if tag == "O":
            flush()
            continue

        start, end = _offset_pair(token["offset"])
        if end <= start:
            continue
        token_index = int(token["index"])
        score = float(token["score"])
        entity_key = re.sub(r"[^A-Za-z0-9]", "", raw_label).upper()
        continues = (
            tag in {"I", "E"}
            and current is not None
            and current.get("entity_key") == entity_key
        )
        if not continues:
            flush()
            current = {
                "entity_key": entity_key,
                "raw_label": raw_label,
                "start": start,
                "end": end,
                "token_start": token_index,
                "token_end": token_index + 1,
                "scores": [score],
            }
        else:
            current["end"] = end
            current["token_end"] = token_index + 1
            current["scores"].append(score)

        if tag in {"E", "S"}:
            flush()

    flush()
    return spans


def _split_label(label: str) -> tuple[str, str]:
    if label == "O":
        return "O", "O"
    match = _BIOES_RE.match(label)
    if match:
        return match.group(1), match.group(2)
    return "B", label


def _argmax_lowest_id(row: Sequence[float]) -> int:
    if not row:
        raise ValueError("logit row must contain at least one label")
    return max(range(len(row)), key=lambda index: (float(row[index]), -index))


def _softmax_score(row: Sequence[float], label_id: int) -> float:
    max_logit = max(float(value) for value in row)
    shifted = [math.exp(float(value) - max_logit) for value in row]
    denominator = sum(shifted)
    if denominator == 0:
        return 0.0
    return shifted[label_id] / denominator


def _first_row(value: Any) -> list[Any]:
    data = _plain_list(value)
    if data and isinstance(data[0], list):
        return data[0]
    return data


def _plain_list(value: Any) -> list[Any]:
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, tuple):
        return list(value)
    return value


def _offset_pair(value: Any) -> tuple[int, int]:
    pair = _plain_list(value)
    if len(pair) < 2:
        raise ValueError(f"offset must contain start and end: {value!r}")
    return int(pair[0]), int(pair[1])


def _is_special_offset(start: int, end: int) -> bool:
    return start == 0 and end == 0


def _contract_payload() -> dict[str, Any]:
    return {
        "token_ids": {
            "mode": "exact",
            "json_path": "cases[].tokens[].id",
        },
        "char_offsets": {
            "mode": "exact",
            "json_path": "cases[].tokens[].offset",
        },
        "span_labels": {
            "mode": "exact",
            "json_path": "cases[].spans[].canonical_label",
        },
        "span_boundaries": {
            "mode": "exact",
            "tolerance_chars": 0,
            "json_path": "cases[].spans[].start/end",
        },
        "logit_ties": {
            "tie_break": "lowest_label_id",
            "reason": "Python argmax and Android decoder must be deterministic.",
        },
    }


def _validate_tokens(case_id: str, text: str, tokens: Any) -> None:
    if not isinstance(tokens, list) or not tokens:
        raise ValueError(f"{case_id} must include token records")
    for token in tokens:
        if not isinstance(token, Mapping):
            raise ValueError(f"{case_id} token records must be objects")
        if not isinstance(token.get("id"), int):
            raise ValueError(f"{case_id} token id must be an integer")
        start, end = _offset_pair(token.get("offset"))
        if start < 0 or end < start:
            raise ValueError(f"{case_id} token offset must satisfy 0 <= start <= end")
        if not _is_special_offset(start, end) and end > len(text):
            raise ValueError(f"{case_id} token offset exceeds text length")
        if not isinstance(token.get("label_id"), int):
            raise ValueError(f"{case_id} token label_id must be an integer")
        if not isinstance(token.get("label"), str):
            raise ValueError(f"{case_id} token label must be a string")


def _validate_spans(case_id: str, text: str, spans: Any) -> None:
    if not isinstance(spans, list):
        raise ValueError(f"{case_id} spans must be a list")
    for span in spans:
        if not isinstance(span, Mapping):
            raise ValueError(f"{case_id} span records must be objects")
        forbidden = {"text", "word", "surface"} & set(span)
        if forbidden:
            raise ValueError(f"{case_id} span stores surface text: {sorted(forbidden)}")
        start = int(span.get("start"))
        end = int(span.get("end"))
        if start < 0 or end <= start or end > len(text):
            raise ValueError(f"{case_id} span has invalid offsets")
        token_start = int(span.get("token_start"))
        token_end = int(span.get("token_end"))
        if token_start < 0 or token_end <= token_start:
            raise ValueError(f"{case_id} span has invalid token range")
        canonical = str(span.get("canonical_label") or "")
        if normalize_label(canonical) != canonical:
            raise ValueError(f"{case_id} span canonical_label is not canonical")
        expected_hash = hmac_text_hash(text[start:end], PARITY_HASH_SECRET)
        if span.get("text_hash") != expected_hash:
            raise ValueError(f"{case_id} span text_hash does not match offsets")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate Android span parity fixtures from an ONNX export.",
    )
    parser.add_argument(
        "--export-dir",
        required=True,
        help="Directory produced by openmed.onnx.convert --profile android.",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_PATH,
        help="Output JSON fixture path.",
    )
    parser.add_argument(
        "--cases",
        default=None,
        help="Optional JSON list of synthetic {id, text} cases.",
    )
    parser.add_argument(
        "--skip-model-validation",
        action="store_true",
        help="Skip Android ONNX graph validation before running inference.",
    )
    args = parser.parse_args(argv)

    payload = build_fixture_payload(
        args.export_dir,
        cases=load_cases(args.cases),
        validate_model=not args.skip_model_validation,
    )
    output_path = write_fixture(payload, args.output)
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
