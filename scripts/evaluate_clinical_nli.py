#!/usr/bin/env python3
"""Export and evaluate a local NLI candidate; never publish or promote it."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import socket
from pathlib import Path

from openmed.clinical.nli_backends import EncoderNLIBackend
from openmed.clinical.nli_gate import NLIThresholds
from openmed.eval.nli_calibration import calibrate_nli_thresholds
from openmed.eval.nli_error_slices import build_nli_error_slice_report
from openmed.eval.nli_gate import (
    NLIEvaluationCounts,
    NLIFormatParity,
    build_nli_candidate_report,
)
from openmed.eval.nli_negation_challenge import (
    default_nli_negation_cases,
    run_nli_negation_challenge,
)
from openmed.training.clinical_nli import (
    LABELS,
    PARTIAL_LABEL,
    NLIPair,
    corpus_fingerprint,
)

MODEL_ID = "OpenMed/OpenMed-NLI-ClinicalE5-Small-33M-v1"


def _write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _fingerprint(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def export(candidate: Path) -> None:
    """Export the exact local BERT classifier to ONNX int8 and MLX float32."""

    import mlx.core as mx
    import numpy as np
    import torch
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from transformers import AutoModelForSequenceClassification

    from openmed.mlx.convert import remap_key
    from openmed.mlx.models.bert_sc import BertForSequenceClassification

    model = AutoModelForSequenceClassification.from_pretrained(
        candidate, local_files_only=True, attn_implementation="eager"
    ).eval()
    if model.config.model_type != "bert":
        raise ValueError("this audited export recipe requires a BERT checkpoint")

    class Wrapper(torch.nn.Module):
        def __init__(self, classifier):
            super().__init__()
            self.classifier = classifier

        def forward(self, input_ids, attention_mask, token_type_ids):
            return self.classifier(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            ).logits

    sample = (
        torch.tensor([[101, 2001, 102, 2002, 102]], dtype=torch.long),
        torch.ones(1, 5, dtype=torch.long),
        torch.tensor([[0, 0, 0, 1, 1]], dtype=torch.long),
    )
    fp32 = candidate / "model.onnx"
    torch.onnx.export(
        Wrapper(model),
        sample,
        str(fp32),
        input_names=["input_ids", "attention_mask", "token_type_ids"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "token_type_ids": {0: "batch", 1: "sequence"},
            "logits": {0: "batch"},
        },
        opset_version=17,
        dynamo=False,
    )
    quantize_dynamic(
        str(fp32),
        str(candidate / "model_int8.onnx"),
        weight_type=QuantType.QInt8,
        op_types_to_quantize=["MatMul", "Gemm"],
    )
    config = model.config.to_dict() | {
        "num_labels": 3,
        "_mlx_task": "sequence-classification",
        "_mlx_weights_format": "safetensors",
    }
    weights = {}
    for key, tensor in model.state_dict().items():
        target = (
            key.replace("bert.pooler.dense.", "pooler.")
            if key.startswith("bert.pooler.dense.")
            else remap_key(key, "bert")
        )
        weights[target] = mx.array(np.asarray(tensor.cpu(), dtype=np.float32))
    mlx_model = BertForSequenceClassification(config)
    mlx_model.load_weights(list(weights.items()), strict=True)
    mlx_model.eval()
    mx.eval(mlx_model.parameters())
    directory = candidate / "mlx"
    directory.mkdir(exist_ok=True)
    _write(directory / "config.json", config)
    mx.save_safetensors(str(directory / "weights.safetensors"), weights)


def _read_pairs(path: Path) -> list[NLIPair]:
    result = []
    for value in json.loads(path.read_text()):
        value["phenomena"] = tuple(value.get("phenomena", ()))
        result.append(NLIPair(**value))
    return result


def evaluate(root: Path, *, skip_export: bool = False) -> None:
    """Produce real, aggregate-only held-out evidence under network denial."""

    import mlx.core as mx
    import numpy as np
    import onnxruntime as ort
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from openmed.mlx.models.bert_sc import load_model

    def deny_network(*_args, **_kwargs):
        raise RuntimeError("network is disabled during candidate evaluation")

    socket.create_connection = deny_network
    socket.socket.connect = deny_network
    candidate = root / "candidate"
    if not skip_export:
        export(candidate)
    model_digest = _fingerprint(candidate / "model.safetensors")
    tokenizer = AutoTokenizer.from_pretrained(candidate, local_files_only=True)
    torch_model = AutoModelForSequenceClassification.from_pretrained(
        candidate, local_files_only=True, attn_implementation="eager"
    ).eval()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    torch_model.to(device)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 8
    onnx_model = ort.InferenceSession(
        str(candidate / "model_int8.onnx"),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    mlx_model = load_model(candidate / "mlx")
    inputs = {item.name for item in onnx_model.get_inputs()}

    def tokenize(pair):
        tokens = tokenizer(
            pair.premise, pair.hypothesis, return_tensors="np", truncation=False
        )
        return None if len(tokens["input_ids"][0]) > 512 else tokens

    def probabilities(pair, runtime="onnx"):
        tokens = tokenize(pair)
        if tokens is None:
            return None
        if runtime == "onnx":
            logits = onnx_model.run(
                None, {k: v for k, v in tokens.items() if k in inputs}
            )[0][0]
        elif runtime == "torch":
            with torch.inference_mode():
                logits = (
                    torch_model(
                        **{k: torch.tensor(v, device=device) for k, v in tokens.items()}
                    )
                    .logits[0]
                    .cpu()
                    .numpy()
                )
        else:
            output = mlx_model(**{k: mx.array(v) for k, v in tokens.items()})
            mx.eval(output)
            logits = np.array(output[0])
        exponential = np.exp(logits.astype(np.float64) - float(np.max(logits)))
        return exponential / exponential.sum()

    validation = _read_pairs(root / "validation.json")
    calibration_fixtures = []
    contradiction_fixtures = []
    for index, pair in enumerate(validation):
        scores = probabilities(pair)
        if scores is None:
            continue
        calibration_fixtures.append(
            {
                "fixture_id": f"validation-{index}",
                "premise": pair.premise,
                "hypothesis": pair.hypothesis,
                "gold_label": "entailment"
                if pair.label == "entailment"
                else "not_entailment",
                "entailment_score": float(scores[2]),
            }
        )
        if pair.label != PARTIAL_LABEL:
            contradiction_fixtures.append(
                {
                    "fixture_id": f"validation-{index}",
                    "premise": pair.premise,
                    "hypothesis": pair.hypothesis,
                    "gold_label": "entailment"
                    if pair.label == "contradiction"
                    else "not_entailment",
                    "entailment_score": float(scores[0]),
                }
            )
    calibration = calibrate_nli_thresholds(
        calibration_fixtures,
        model_id=MODEL_ID,
        model_revision=model_digest,
        thresholds=[value / 1000 for value in range(500, 1001, 5)],
        precision_floor=0.99,
        recall_floor=0.25,
        false_positive_rate_ceiling=0.01,
    )
    contradiction_calibration = calibrate_nli_thresholds(
        contradiction_fixtures,
        model_id=MODEL_ID,
        model_revision=model_digest,
        thresholds=[value / 1000 for value in range(500, 1001, 5)],
        precision_floor=0.99,
        recall_floor=0.25,
        false_positive_rate_ceiling=0.01,
    )
    thresholds = NLIThresholds(
        entailment=calibration.recommended_threshold,
        contradiction=contradiction_calibration.recommended_threshold,
        margin=0.05,
        calibration_id=calibration.fixture_fingerprint,
        calibration_method="held-out-selective-precision-fpr",
    )
    backend = EncoderNLIBackend(
        candidate,
        runtime="onnx",
        label_mapping=dict(enumerate(LABELS)),
        thresholds=thresholds,
    )
    reports = root / "evidence"
    reports.mkdir(exist_ok=True)
    _write(reports / "calibration.json", calibration.to_dict())
    _write(
        reports / "contradiction-calibration.json", contradiction_calibration.to_dict()
    )
    _write(reports / "thresholds.json", thresholds.to_dict())
    print(
        json.dumps(
            {
                "calibration_selection": calibration.selection,
                "threshold": thresholds.entailment,
            }
        ),
        flush=True,
    )
    held_out = {}
    counts = {}
    skipped = {}
    slices = []
    entailment_support = entailment_accepted = 0
    for name in ("public_test", "biomedical_test", "synthetic_test"):
        rows = _read_pairs(root / f"{name}.json")
        held_out[name] = rows
        correct = overlength = 0
        for index, pair in enumerate(rows):
            scores = probabilities(pair)
            if scores is None:
                overlength += 1
                predicted = "abstention"
            else:
                predicted = LABELS[int(np.argmax(scores))]
                correct += int(
                    predicted != "entailment"
                    if pair.label == PARTIAL_LABEL
                    else predicted == pair.label
                )
            if pair.synthetic:
                selective = backend.predict(pair.premise, pair.hypothesis)["label"]
                slices.append(
                    {
                        "fixture_id": f"heldout-{index}",
                        "phenomena": pair.phenomena,
                        "gold_label": pair.label,
                        "predicted_label": selective,
                    }
                )
                if pair.label == "entailment":
                    entailment_support += 1
                    entailment_accepted += int(selective == "entailment")
        # Overlength inputs remain incorrect in the denominator, not omitted.
        counts[name] = NLIEvaluationCounts(len(rows), correct, corpus_fingerprint(rows))
        skipped[name] = overlength
        print(
            json.dumps(
                {
                    "split": name,
                    "count": len(rows),
                    "correct": correct,
                    "overlength": overlength,
                }
            ),
            flush=True,
        )
    error_slices = build_nli_error_slice_report(
        slices, fixture_set_id="authored-clinical-heldout-v1", model_digest=model_digest
    )
    negation = run_nli_negation_challenge(runner=backend.predict)
    _write(reports / "error-slices.json", error_slices.to_dict())
    _write(reports / "negation.json", negation.to_dict())
    parity_rows = []
    rng = random.Random(3236)
    for rows in held_out.values():
        eligible = [pair for pair in rows if tokenize(pair) is not None]
        parity_rows.extend(rng.sample(eligible, min(40, len(eligible))))
    parity_rows.extend(
        NLIPair(case.premise, case.hypothesis, case.gold_label, "challenge")
        for case in default_nli_negation_cases()
    )
    parity = []
    for runtime in ("onnx", "mlx"):
        agreement = 0
        delta = 0.0
        for pair in parity_rows:
            reference = probabilities(pair, "torch")
            measured = probabilities(pair, runtime)
            if reference is None or measured is None:
                raise ValueError("parity corpus exceeded the audited context limit")
            agreement += int(np.argmax(reference) == np.argmax(measured))
            delta = max(delta, float(np.max(np.abs(reference - measured))))
        parity.append(
            NLIFormatParity(
                "onnx_int8" if runtime == "onnx" else "mlx",
                len(parity_rows),
                agreement,
                delta,
                model_digest,
            )
        )
    report, checks = build_nli_candidate_report(
        model_id=MODEL_ID,
        artifact_digest=model_digest,
        public=counts["public_test"],
        biomedical=counts["biomedical_test"],
        synthetic=counts["synthetic_test"],
        error_slices=error_slices,
        calibration=calibration,
        contradiction_calibration=contradiction_calibration,
        negation=negation,
        synthetic_entailment_support=entailment_support,
        synthetic_entailment_accepted=entailment_accepted,
        parity=tuple(parity),
    )
    report.write_json(reports / "benchmark.json")
    report.write_markdown(reports / "benchmark.md")
    _write(
        reports / "export-manifest.json",
        {
            "model_id": MODEL_ID,
            "artifact_digest": model_digest,
            "files": {
                str(path.relative_to(candidate)): _fingerprint(path)
                for path in candidate.rglob("*")
                if path.is_file() and ".cache" not in path.parts
            },
            "overlength_heldout": skipped,
            "offline_socket_denial": True,
            "biomedical_protocol": "author-generated binary perturbations; not clinical patient-note validation",
            "synthetic_protocol": "held-out conditions within authored templates; not clinical generalization",
            "public_protocol": "source-grouped non-fiction MultiNLI; general-domain three-way NLI",
            "publication": "not performed",
        },
    )
    print(
        json.dumps(
            {
                "stage": report.metadata["stage"],
                "checks": [check.to_dict() for check in checks],
            }
        ),
        flush=True,
    )


def main() -> None:
    """Run explicit local export/evaluation with no remote inference."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--skip-export", action="store_true")
    args = parser.parse_args()
    evaluate(args.run, skip_export=args.skip_export)


if __name__ == "__main__":
    main()
