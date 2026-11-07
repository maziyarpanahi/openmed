"""Generate the synthetic model and reference spans for the Android emulator lane.

The OpenMedKit emulator lane (`android/openmedkit/src/androidTest`) loads a real
ONNX token classifier and Hugging Face tokenizer through ONNX Runtime Android and
the DJL tokenizer JNI library. This script builds that model offline:

1. A tiny BERT token classifier (two layers, 32 hidden units) is initialised
   from a fixed seed. Nothing is downloaded and no weights are trained.
2. The repository's Android ONNX exporter (`openmed.onnx.convert`) writes a
   single self-contained ``model.onnx`` (every weight inline, no external-data
   sidecar) plus ``tokenizer.json`` and ``id2label.json``.
3. Python ONNX Runtime on CPU computes reference token IDs, offsets, labels and
   scores for synthetic text, and the expected ``OpenMedKit.analyzeText`` spans
   are derived with the same aggregation rules as the Kotlin facade.

Seeds are tried in order until every token has a clear top-2 logit margin, no
span confidence sits near the confidence threshold, and every case yields at
least one span. That keeps the lane deterministic across ONNX Runtime builds.

The committed fixture contains only synthetic text, case identifiers, labels,
offsets, scores and file digests. Run ``check`` to recompute the reference from
the committed model and compare it with the committed fixture.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import sys
import tempfile
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
LANE_ROOT = ROOT / "android/openmedkit/src/androidTest/assets/emulator-lane"
MODEL_DIRNAME = "model"
FIXTURE_FILENAME = "expected.json"

FIXTURE_FORMAT = "openmed-android-emulator-lane"
FIXTURE_FORMAT_VERSION = 1
GENERATOR_PATH = "scripts/android/generate_emulator_lane_fixture.py"
SOURCE_MODEL_ID = "synthetic/openmed-emulator-lane-tiny-bert"

DEFAULT_SEED = 3938
MAX_SEED_ATTEMPTS = 64
CONFIDENCE_THRESHOLD = 0.5
MIN_TOP2_LOGIT_MARGIN = 0.05
MIN_THRESHOLD_MARGIN = 0.02
TOKEN_SCORE_TOLERANCE = 1e-4
SPAN_CONFIDENCE_TOLERANCE = 1e-3
MAX_SEQUENCE_LENGTH = 128
CLASSIFIER_INIT_STD = 0.6
OUTSIDE_LABEL_BIAS = 1.0

LABELS: tuple[str, ...] = (
    "O",
    "B-NAME",
    "I-NAME",
    "B-DATE",
    "I-DATE",
    "B-ID",
    "I-ID",
)
SPECIAL_TOKENS: tuple[str, ...] = ("[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]")
WHOLE_WORDS: tuple[str, ...] = (
    "synth",
    "patient",
    "person",
    "clinic",
    "date",
    "note",
    "id",
    "alpha",
    "beta",
    "bravo",
    "seen",
    "at",
    "on",
    "ref",
    "via",
    "follow",
    "up",
)
PUNCTUATION: tuple[str, ...] = ("_", "-", ".", ",", ":")

# Files copied to the device and opened with `OpenMedKit.fromDirectory`.
MODEL_FILES: tuple[str, ...] = (
    "model.onnx",
    "tokenizer.json",
    "tokenizer_config.json",
    "id2label.json",
    "config.json",
    "openmed-onnx.json",
)

_CASE_ID_RE = re.compile(r"^synth-lane-\d{3}$")
_TOKEN_PREFIXES = frozenset({"B", "I", "E", "L", "S", "U"})
_SUSPICIOUS_PHI_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    re.compile(r"\b(?:\+?1[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b"),
    re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    re.compile(r"\b(?:John|Jane)\s+(?:Doe|Roe)\b", re.IGNORECASE),
)


class LaneFixtureError(ValueError):
    """Raised when a lane fixture is inconsistent or unsafe to commit."""


@dataclass(frozen=True)
class LaneCase:
    """One synthetic input for the emulator lane."""

    id: str
    text: str


DEFAULT_CASES: tuple[LaneCase, ...] = (
    LaneCase(
        id="synth-lane-001",
        text="SYNTH_PATIENT_ALPHA seen at SYNTH_CLINIC_BETA on 2099-01-02.",
    ),
    LaneCase(
        id="synth-lane-002",
        text="Ref SYNTH_ID_7781, follow-up via SYNTH_NOTE_42.",
    ),
    LaneCase(
        # U+1F600 lies outside the BMP: Unicode scalar and UTF-16 offsets
        # diverge after it, which pins the JNI tokenizer offset contract.
        id="synth-lane-003",
        text="SYNTH_PERSON_BRAVO \U0001f600 seen on SYNTH_DATE_2099Q1.",
    ),
)


def synthetic_vocabulary() -> tuple[str, ...]:
    """Return the deterministic WordPiece vocabulary of the lane model."""

    letters = [chr(code) for code in range(ord("a"), ord("z") + 1)]
    digits = [str(digit) for digit in range(10)]
    return (
        *SPECIAL_TOKENS,
        *WHOLE_WORDS,
        *letters,
        *digits,
        *PUNCTUATION,
        *(f"##{piece}" for piece in (*letters, *digits)),
    )


def split_token_label(raw_label: str) -> tuple[str, str]:
    """Split a token label into its tagging prefix and entity label.

    Mirrors `splitTokenLabel` in OpenMedKit's `Runtime.kt`.
    """

    label = raw_label.strip()
    if len(label) > 2 and label[1] in "-_":
        prefix = label[0].upper()
        if prefix in _TOKEN_PREFIXES:
            return prefix, label[2:]
    return "", label


def aggregate_token_predictions(
    scalar_length: int,
    tokens: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Group token predictions into entity spans.

    Mirrors `aggregateTokenPredictions` in OpenMedKit's `Runtime.kt`: special
    tokens are skipped, ``O`` (any case) and zero-width tokens close the open
    span, ``B``/``S``/``U`` always start a new span, ``E``/``L``/``S``/``U``
    close it, and an unprefixed label starts a new span after a gap.

    Args:
        scalar_length: Length of the source text in Unicode scalar values.
        tokens: Token records with ``label``, ``score``, ``start``, ``end`` and
            ``special`` keys. Offsets are Unicode scalar offsets.

    Returns:
        Span records with ``label``, ``start``, ``end`` and ``confidence``.
    """

    spans: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None

    def flush() -> None:
        nonlocal current
        if current is None:
            return
        start = int(current["start"])
        end = int(current["end"])
        if 0 <= start < end <= scalar_length:
            scores = [float(score) for score in current["scores"]]
            spans.append(
                {
                    "label": str(current["label"]),
                    "start": start,
                    "end": end,
                    "confidence": _float32(sum(scores) / len(scores)),
                }
            )
        current = None

    for token in tokens:
        if token.get("special"):
            continue
        start = int(token["start"])
        end = int(token["end"])
        prefix, label = split_token_label(str(token["label"]))
        score = float(token["score"])
        if start == end or label.upper() == "O":
            flush()
            continue
        starts_new = (
            current is None
            or current["label"] != label
            or prefix in {"B", "S", "U"}
            or (prefix not in {"I", "E", "L"} and start > int(current["end"]))
        )
        if starts_new:
            flush()
            current = {"label": label, "start": start, "end": end, "scores": [score]}
        else:
            assert current is not None
            current["end"] = max(int(current["end"]), end)
            current["scores"].append(score)
        if prefix in {"E", "L", "S", "U"}:
            flush()
    flush()
    return spans


def expected_facade_spans(
    text: str,
    tokens: Sequence[Mapping[str, Any]],
    *,
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
) -> list[dict[str, Any]]:
    """Return the spans `OpenMedKit.analyzeText` reports for reference tokens.

    The Kotlin decoder also snaps spans to grapheme boundaries. Lane text is
    restricted to single-code-point graphemes (see `validate_lane_text`), so
    that step leaves every span unchanged; the JVM fixture-contract test runs
    the real Kotlin decoder over the same tokens to confirm it.

    Args:
        text: Synthetic source text.
        tokens: Reference token records.
        confidence_threshold: Minimum span confidence to keep.

    Returns:
        Kept spans sorted by start and end offset.
    """

    spans = aggregate_token_predictions(len(text), tokens)
    kept = [span for span in spans if span["confidence"] >= confidence_threshold]
    return sorted(kept, key=lambda span: (span["start"], span["end"]))


def validate_lane_text(case_id: str, text: str) -> None:
    """Reject lane text that is not explicitly synthetic and grapheme-simple.

    Args:
        case_id: Case identifier, used only in error messages.
        text: Candidate synthetic text.

    Raises:
        LaneFixtureError: If the text is unsafe or not grapheme-simple.
    """

    if not _CASE_ID_RE.match(case_id):
        raise LaneFixtureError(f"case id {case_id!r} must look like synth-lane-001")
    if "SYNTH_" not in text:
        raise LaneFixtureError(f"{case_id} must use explicit SYNTH_ placeholders")
    for pattern in _SUSPICIOUS_PHI_PATTERNS:
        if pattern.search(text):
            raise LaneFixtureError(f"{case_id} contains identifier-shaped text")
    for character in text:
        category = unicodedata.category(character)
        if category.startswith("M") or category in {"Cc", "Cf", "Cs", "Co", "Cn"}:
            raise LaneFixtureError(
                f"{case_id} must use single-code-point graphemes only"
            )


def validate_fixture(payload: Mapping[str, Any], lane_root: Path) -> None:
    """Validate the committed lane fixture without ONNX Runtime.

    Checks the structure, the privacy contract (synthetic text, value-free
    spans), file digests of the committed model directory, the absence of an
    external-data sidecar, and that the expected spans follow from the
    recorded token predictions.

    Args:
        payload: Parsed ``expected.json``.
        lane_root: Directory holding ``expected.json`` and ``model/``.

    Raises:
        LaneFixtureError: If any check fails.
    """

    if payload.get("format") != FIXTURE_FORMAT:
        raise LaneFixtureError(f"fixture format must be {FIXTURE_FORMAT!r}")
    if payload.get("format_version") != FIXTURE_FORMAT_VERSION:
        raise LaneFixtureError(f"format_version must be {FIXTURE_FORMAT_VERSION}")

    model = _require_mapping(payload.get("model"), "model")
    model_dir = lane_root / str(model.get("directory"))
    files = _require_mapping(model.get("files"), "model.files")
    if sorted(files) != sorted(MODEL_FILES):
        raise LaneFixtureError(f"model.files must list exactly {sorted(MODEL_FILES)}")
    present = sorted(path.name for path in model_dir.iterdir())
    if present != sorted(MODEL_FILES):
        raise LaneFixtureError(
            "model directory must contain only the listed files; "
            "an external-data sidecar or stray file is not allowed"
        )
    for name, digest in files.items():
        record = _require_mapping(digest, f"model.files.{name}")
        data = (model_dir / name).read_bytes()
        if record.get("bytes") != len(data) or record.get("sha256") != _sha256(data):
            raise LaneFixtureError(f"{name} does not match its recorded digest")
    if model.get("external_data") is not False:
        raise LaneFixtureError("the lane model must embed every weight inline")

    labels = [str(label) for label in model.get("labels") or ()]
    id2label = json.loads((model_dir / "id2label.json").read_text(encoding="utf-8"))
    if labels != [id2label[str(index)] for index in range(len(id2label))]:
        raise LaneFixtureError("model.labels must match id2label.json")
    entity_labels = {split_token_label(label)[1] for label in labels} - {"O"}

    contract = _require_mapping(payload.get("contract"), "contract")
    threshold = float(contract["confidence_threshold"])
    min_margin = float(contract["min_top2_logit_margin"])
    threshold_margin = float(contract["min_threshold_margin"])
    if contract.get("offset_unit") != "unicode_scalar":
        raise LaneFixtureError("contract.offset_unit must be unicode_scalar")

    cases = payload.get("cases")
    if not isinstance(cases, list) or not cases:
        raise LaneFixtureError("fixture must contain at least one case")
    seen: set[str] = set()
    for case in cases:
        case = _require_mapping(case, "case")
        case_id = str(case.get("id") or "")
        if case_id in seen:
            raise LaneFixtureError(f"duplicate case id {case_id!r}")
        seen.add(case_id)
        text = str(case.get("text") or "")
        validate_lane_text(case_id, text)
        if case.get("synthetic") is not True or case.get("phi_free") is not True:
            raise LaneFixtureError(f"{case_id} must be marked synthetic and phi_free")
        tokens = case.get("tokens")
        if not isinstance(tokens, list) or len(tokens) < 3:
            raise LaneFixtureError(f"{case_id} must record its token sequence")
        _validate_tokens(case_id, text, tokens, labels)
        if float(case.get("min_top2_logit_margin") or 0.0) < min_margin:
            raise LaneFixtureError(f"{case_id} has an ambiguous token decision")
        spans = case.get("spans")
        if not isinstance(spans, list) or not spans:
            raise LaneFixtureError(f"{case_id} must expect at least one span")
        for span in spans:
            span = _require_mapping(span, f"{case_id} span")
            if set(span) != {"label", "start", "end", "confidence"}:
                raise LaneFixtureError(f"{case_id} spans must be value-free")
            if span["label"] not in entity_labels:
                raise LaneFixtureError(f"{case_id} span label is not a model label")
            if not 0 <= int(span["start"]) < int(span["end"]) <= len(text):
                raise LaneFixtureError(f"{case_id} span offsets are out of range")
        derived = expected_facade_spans(text, tokens, confidence_threshold=threshold)
        if _span_signatures(derived) != _span_signatures(spans):
            raise LaneFixtureError(f"{case_id} spans do not follow from its tokens")
        for span in aggregate_token_predictions(len(text), tokens):
            if abs(float(span["confidence"]) - threshold) < threshold_margin:
                raise LaneFixtureError(
                    f"{case_id} has a span confidence too close to the threshold"
                )


def generate(
    lane_root: Path = LANE_ROOT,
    *,
    seed: int = DEFAULT_SEED,
    cases: Sequence[LaneCase] = DEFAULT_CASES,
) -> Path:
    """Build, export and record the lane model, replacing ``lane_root``.

    Args:
        lane_root: Destination directory for ``expected.json`` and ``model/``.
        seed: First seed to try.
        cases: Synthetic inputs.

    Returns:
        Path to the written ``expected.json``.

    Raises:
        LaneFixtureError: If no seed within the attempt budget yields a
            fixture with clear decision margins.
    """

    for case in cases:
        validate_lane_text(case.id, case.text)

    with tempfile.TemporaryDirectory(prefix="openmed-emulator-lane-") as scratch:
        scratch_root = Path(scratch)
        for attempt in range(MAX_SEED_ATTEMPTS):
            candidate_seed = seed + attempt
            work = scratch_root / f"seed-{candidate_seed}"
            checkpoint = _write_synthetic_checkpoint(
                work / "checkpoint", candidate_seed
            )
            model_dir = work / MODEL_DIRNAME
            _export_lane_model(checkpoint, model_dir)
            payload = _build_payload(model_dir, cases, seed=candidate_seed)
            if _decision_margins_ok(payload):
                lane_root.mkdir(parents=True, exist_ok=True)
                target = lane_root / MODEL_DIRNAME
                shutil.rmtree(target, ignore_errors=True)
                shutil.copytree(model_dir, target)
                fixture_path = lane_root / FIXTURE_FILENAME
                fixture_path.write_text(
                    json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8",
                )
                validate_fixture(payload, lane_root)
                return fixture_path
    raise LaneFixtureError(
        f"no seed in [{seed}, {seed + MAX_SEED_ATTEMPTS}) met the decision margins"
    )


def check(lane_root: Path = LANE_ROOT) -> None:
    """Recompute the Python reference from the committed model and compare.

    Args:
        lane_root: Directory holding ``expected.json`` and ``model/``.

    Raises:
        LaneFixtureError: If the committed fixture no longer matches the
            committed model under the installed ONNX Runtime.
    """

    payload = json.loads((lane_root / FIXTURE_FILENAME).read_text(encoding="utf-8"))
    validate_fixture(payload, lane_root)
    model_dir = lane_root / str(payload["model"]["directory"])
    cases = [LaneCase(id=case["id"], text=case["text"]) for case in payload["cases"]]
    recomputed = _reference_cases(model_dir, cases)
    tolerance = float(payload["contract"]["token_score_tolerance"])
    for committed, fresh in zip(payload["cases"], recomputed, strict=True):
        case_id = committed["id"]
        for index, (old, new) in enumerate(
            zip(committed["tokens"], fresh["tokens"], strict=True)
        ):
            for key in ("id", "start", "end", "special", "label_id", "label"):
                if old.get(key) != new.get(key):
                    raise LaneFixtureError(f"{case_id} token {index} {key} changed")
            if not old["special"] and abs(old["score"] - new["score"]) > tolerance:
                raise LaneFixtureError(f"{case_id} token {index} score drifted")
        if _span_signatures(committed["spans"]) != _span_signatures(fresh["spans"]):
            raise LaneFixtureError(f"{case_id} spans changed")


def _build_payload(
    model_dir: Path,
    cases: Sequence[LaneCase],
    *,
    seed: int,
) -> dict[str, Any]:
    import onnx
    import onnxruntime
    import tokenizers
    import torch
    import transformers

    model = onnx.load(str(model_dir / "model.onnx"), load_external_data=False)
    external = any(
        initializer.data_location == onnx.TensorProto.EXTERNAL
        for initializer in model.graph.initializer
    )
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    return {
        "format": FIXTURE_FORMAT,
        "format_version": FIXTURE_FORMAT_VERSION,
        "generator": GENERATOR_PATH,
        "seed": seed,
        "model": {
            "directory": MODEL_DIRNAME,
            "variant": "fp32",
            "source_model_id": SOURCE_MODEL_ID,
            "labels": list(LABELS),
            "architecture": {
                "family": str(config.get("model_type")),
                "hidden_size": int(config["hidden_size"]),
                "num_hidden_layers": int(config["num_hidden_layers"]),
                "num_attention_heads": int(config["num_attention_heads"]),
                "intermediate_size": int(config["intermediate_size"]),
                "vocab_size": int(config["vocab_size"]),
                "max_position_embeddings": int(config["max_position_embeddings"]),
            },
            "external_data": external,
            "files": {
                name: {
                    "bytes": (model_dir / name).stat().st_size,
                    "sha256": _sha256((model_dir / name).read_bytes()),
                }
                for name in MODEL_FILES
            },
        },
        "reference": {
            "runtime": "onnxruntime",
            "provider": "CPUExecutionProvider",
            "versions": {
                "onnx": onnx.__version__,
                "onnxruntime": onnxruntime.__version__,
                "tokenizers": tokenizers.__version__,
                "torch": torch.__version__,
                "transformers": transformers.__version__,
            },
        },
        "contract": {
            "offset_unit": "unicode_scalar",
            "confidence_threshold": CONFIDENCE_THRESHOLD,
            "token_ids": "exact",
            "token_offsets": "exact",
            "token_labels": "exact",
            "token_score_tolerance": TOKEN_SCORE_TOLERANCE,
            "span_labels": "exact",
            "span_boundaries": "exact",
            "span_confidence_tolerance": SPAN_CONFIDENCE_TOLERANCE,
            "min_top2_logit_margin": MIN_TOP2_LOGIT_MARGIN,
            "min_threshold_margin": MIN_THRESHOLD_MARGIN,
        },
        "cases": _reference_cases(model_dir, cases),
    }


def _reference_cases(
    model_dir: Path,
    cases: Sequence[LaneCase],
) -> list[dict[str, Any]]:
    import numpy as np
    import onnxruntime
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    session = onnxruntime.InferenceSession(
        str(model_dir / "model.onnx"),
        providers=["CPUExecutionProvider"],
    )
    input_names = {item.name for item in session.get_inputs()}
    id2label = {
        int(index): str(label)
        for index, label in json.loads(
            (model_dir / "id2label.json").read_text(encoding="utf-8")
        ).items()
    }

    records: list[dict[str, Any]] = []
    for case in cases:
        encoding = tokenizer.encode(case.text, add_special_tokens=True)
        ids = np.asarray([encoding.ids], dtype=np.int64)
        feeds = {
            "input_ids": ids,
            "attention_mask": np.ones_like(ids),
            "token_type_ids": np.zeros_like(ids),
        }
        feeds = {name: value for name, value in feeds.items() if name in input_names}
        logits = session.run(["logits"], feeds)[0][0].astype(np.float32)

        tokens: list[dict[str, Any]] = []
        margins: list[float] = []
        for index, token_id in enumerate(encoding.ids):
            start, end = (int(value) for value in encoding.offsets[index])
            special = start == 0 and end == 0
            record: dict[str, Any] = {
                "id": int(token_id),
                "start": start,
                "end": end,
                "special": special,
            }
            if not special:
                row = [float(value) for value in logits[index]]
                label_id = _argmax_lowest_id(row)
                ranked = sorted(row, reverse=True)
                margins.append(ranked[0] - ranked[1])
                record.update(
                    {
                        "label_id": label_id,
                        "label": id2label.get(label_id, "O"),
                        "score": round(_kotlin_softmax_score(row), 6),
                    }
                )
            tokens.append(record)
        records.append(
            {
                "id": case.id,
                "synthetic": True,
                "phi_free": True,
                "text": case.text,
                "min_top2_logit_margin": round(min(margins), 6),
                "tokens": tokens,
                "spans": [
                    {**span, "confidence": round(span["confidence"], 6)}
                    for span in expected_facade_spans(case.text, tokens)
                ],
            }
        )
    return records


def _decision_margins_ok(payload: Mapping[str, Any]) -> bool:
    contract = payload["contract"]
    threshold = float(contract["confidence_threshold"])
    for case in payload["cases"]:
        if float(case["min_top2_logit_margin"]) < MIN_TOP2_LOGIT_MARGIN:
            return False
        if not case["spans"]:
            return False
        for span in aggregate_token_predictions(len(case["text"]), case["tokens"]):
            if abs(float(span["confidence"]) - threshold) < MIN_THRESHOLD_MARGIN:
                return False
    return True


def _write_synthetic_checkpoint(directory: Path, seed: int) -> Path:
    import torch
    from transformers import BertConfig, BertForTokenClassification, BertTokenizerFast

    directory.mkdir(parents=True, exist_ok=True)
    vocab_path = directory / "vocab.txt"
    vocab_path.write_text("\n".join(synthetic_vocabulary()) + "\n", encoding="utf-8")
    tokenizer = BertTokenizerFast(
        vocab_file=str(vocab_path),
        do_lower_case=True,
        model_max_length=MAX_SEQUENCE_LENGTH,
    )
    tokenizer.save_pretrained(str(directory))

    config = BertConfig(
        vocab_size=len(synthetic_vocabulary()),
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=64,
        max_position_embeddings=MAX_SEQUENCE_LENGTH,
        type_vocab_size=2,
        id2label=dict(enumerate(LABELS)),
        label2id={label: index for index, label in enumerate(LABELS)},
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = BertForTokenClassification(config)
        generator = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            # A wider classifier head gives clear argmax margins; a small bias
            # towards "O" keeps a realistic mix of entity and outside tokens.
            model.classifier.weight.normal_(
                0.0, CLASSIFIER_INIT_STD, generator=generator
            )
            model.classifier.bias.zero_()
            model.classifier.bias[LABELS.index("O")] = OUTSIDE_LABEL_BIAS
    model.eval()
    model.save_pretrained(str(directory), safe_serialization=True)
    return directory


def _export_lane_model(checkpoint: Path, model_dir: Path) -> None:
    from openmed.onnx.android_profile import (
        ANDROID_ONNX_FORMAT,
        ANDROID_PROFILE_NAME,
        validate_android_profile,
    )
    from openmed.onnx.convert import (
        MINIMUM_TOKEN_CLASSIFICATION_OPSET,
        ExportArtifact,
        export_onnx,
        save_source_assets,
        write_export_manifest,
    )

    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = export_onnx(
        str(checkpoint),
        model_dir / "model.onnx",
        max_seq_length=MAX_SEQUENCE_LENGTH,
        profile=ANDROID_PROFILE_NAME,
    )
    config, _ = save_source_assets(
        str(checkpoint),
        model_dir,
        max_seq_length=MAX_SEQUENCE_LENGTH,
        require_id2label=True,
        require_tokenizer_json=True,
    )
    # Keep only the files the device needs; never commit a local path.
    config.pop("_name_or_path", None)
    (model_dir / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    tokenizer_config_path = model_dir / "tokenizer_config.json"
    tokenizer_config = json.loads(tokenizer_config_path.read_text(encoding="utf-8"))
    tokenizer_config.pop("name_or_path", None)
    tokenizer_config_path.write_text(
        json.dumps(tokenizer_config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    # The exporter traces with a padded batch, which leaves fixed-length padding
    # in the saved tokenizer.json. On-device inference encodes one unpadded
    # sequence, so the lane tokenizer pads nothing and truncates at the model's
    # position limit, which is also what DJL applies on Android.
    from tokenizers import Tokenizer

    tokenizer_path = model_dir / "tokenizer.json"
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    tokenizer.no_padding()
    tokenizer.enable_truncation(max_length=MAX_SEQUENCE_LENGTH)
    tokenizer.save(str(tokenizer_path), pretty=True)
    for path in model_dir.iterdir():
        if path.name not in MODEL_FILES:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()

    validation = validate_android_profile(model_path)
    write_export_manifest(
        model_dir,
        source_model_id=SOURCE_MODEL_ID,
        config=config,
        artifacts=[
            ExportArtifact(
                format=ANDROID_ONNX_FORMAT,
                path=model_path,
                precision="float32",
                metadata=validation.to_metadata(),
            )
        ],
        tokenizer_files=["tokenizer.json", "tokenizer_config.json"],
        minimum_opset=MINIMUM_TOKEN_CLASSIFICATION_OPSET,
        optimization={"enabled": False},
    )
    present = sorted(path.name for path in model_dir.iterdir())
    if present != sorted(MODEL_FILES):
        raise LaneFixtureError(f"unexpected exported files: {present}")
    # End every JSON file with a newline so the repository's end-of-file hook
    # leaves the committed bytes, and therefore their digests, unchanged.
    for name in MODEL_FILES:
        path = model_dir / name
        if path.suffix == ".json":
            text = path.read_text(encoding="utf-8")
            if not text.endswith("\n"):
                path.write_text(text + "\n", encoding="utf-8")


def _argmax_lowest_id(row: Sequence[float]) -> int:
    return max(range(len(row)), key=lambda index: (row[index], -index))


def _kotlin_softmax_score(row: Sequence[float]) -> float:
    """Top-1 softmax score computed exactly as `OnnxTokenClassifier.decodeToken`."""

    import numpy as np

    values = np.asarray(row, dtype=np.float32)
    max_logit = values.max()
    denominator = sum(math.exp(float(value - max_logit)) for value in values)
    return _float32(1.0 / denominator)


def _float32(value: float) -> float:
    import struct

    return struct.unpack("<f", struct.pack("<f", value))[0]


def _span_signatures(spans: Sequence[Mapping[str, Any]]) -> list[tuple[str, int, int]]:
    return [
        (str(span["label"]), int(span["start"]), int(span["end"])) for span in spans
    ]


def _validate_tokens(
    case_id: str,
    text: str,
    tokens: Sequence[Any],
    labels: Sequence[str],
) -> None:
    for index, token in enumerate(tokens):
        token = _require_mapping(token, f"{case_id} token")
        if not isinstance(token.get("id"), int) or not isinstance(
            token.get("special"), bool
        ):
            raise LaneFixtureError(f"{case_id} token {index} is malformed")
        start = int(token["start"])
        end = int(token["end"])
        if token["special"]:
            if (start, end) != (0, 0) or "label" in token:
                raise LaneFixtureError(f"{case_id} special token {index} is malformed")
            continue
        if not 0 <= start < end <= len(text):
            raise LaneFixtureError(f"{case_id} token {index} offsets are out of range")
        label_id = token.get("label_id")
        if not isinstance(label_id, int) or labels[label_id] != token.get("label"):
            raise LaneFixtureError(f"{case_id} token {index} label is inconsistent")
        if not 0.0 < float(token["score"]) <= 1.0:
            raise LaneFixtureError(f"{case_id} token {index} score is out of range")
    if not tokens[0]["special"] or not tokens[-1]["special"]:
        raise LaneFixtureError(f"{case_id} must keep the [CLS] and [SEP] tokens")


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise LaneFixtureError(f"{label} must be an object")
    return value


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    """Run the fixture generator or checker from the command line."""

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    subparsers = parser.add_subparsers(dest="command", required=True)
    generate_parser = subparsers.add_parser(
        "generate", help="Build the model and write expected.json."
    )
    generate_parser.add_argument("--lane-root", type=Path, default=LANE_ROOT)
    generate_parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    check_parser = subparsers.add_parser(
        "check", help="Recompute the reference and compare with expected.json."
    )
    check_parser.add_argument("--lane-root", type=Path, default=LANE_ROOT)
    args = parser.parse_args(argv)

    try:
        if args.command == "generate":
            fixture_path = generate(args.lane_root, seed=args.seed)
            print(f"wrote {fixture_path.name} and {MODEL_DIRNAME}/")
        else:
            check(args.lane_root)
            print("emulator lane fixture matches the committed model")
    except LaneFixtureError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
