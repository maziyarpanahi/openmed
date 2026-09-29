# Clinical NLI: local training and release evidence

The local backend supports cached, pinned three-class sequence classifiers
through PyTorch, ONNX int8 and Apple MLX. Its public result remains four-state:
`entailment`, `contradiction`, `neutral` or `abstention`. This is assistive
grounding evidence for human review, not a clinical decision or a guarantee
of factual correctness. The default fails closed until a qualified checkpoint
is independently published, verified and registered at an immutable revision.

## Reproduce a local candidate

Use a separate training environment with `torch`, `transformers`,
`huggingface_hub`, `safetensors`, `pyarrow` and `PyYAML`. On Apple Silicon,
export/evaluation additionally require `mlx`, `onnx` and `onnxruntime`.
These are optional training tools, not new mandatory core dependencies.

```bash
python scripts/train_clinical_nli.py --output PLANS/nli-candidate
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python scripts/evaluate_clinical_nli.py --run PLANS/nli-candidate
```

The training download phase fetches the pinned public model and corpora.
Training uses the MPS device when available, otherwise the CPU; it never
starts a paid or hosted job. Evaluation denies sockets and loads artifacts
locally. Neither command publishes a model or changes registry defaults.
Use a fresh output directory for a new training run so earlier evidence is
preserved. `--max-steps` provides a bounded training smoke test, not a release.

The checked-in configuration is
`openmed/training/configs/clinical_nli_small.yaml`. The base is the Apache-2.0
ClinicalE5-Small encoder at revision
`2ea947130eb28d610ceb96472af0b694fce9c344`. Its PII token-classification head
is discarded; the sequence pooler and three-way classifier are trained anew.
The recipe uses three seeded epochs of AdamW and selects the candidate by
validation accuracy, never by final held-out performance. Seeds are recorded;
bitwise MPS determinism is not claimed.

## Data provenance and limits

- Three-way public NLI uses source-grouped non-fiction MultiNLI at revision
  `da70db2af9d09693783c3320c4249840212ee221`. Fiction is excluded because its
  source licenses differ; other selected genres use the permissive OANC
  terms recorded in the [publisher's dataset card](https://huggingface.co/datasets/nyu-mll/multi_nli/blob/da70db2af9d09693783c3320c4249840212ee221/README.md).
  The report describes this withheld training-corpus protocol, not an
  official matched/mismatched MultiNLI benchmark score.
- [BioNLI's authors](https://stonybrooknlp.github.io/BioNLI/) publish the
  biomedical-literature corpus under CC BY 4.0. Record downloaded file hashes
  and retain attribution to Bastan, Surdeanu and Balasubramanian (2022).
  Its generated perturbations have binary labels. A negative is trained
  using partial-label loss over contradiction plus neutral; it is never
  relabeled as an invented three-way contradiction. Author development
  publications are reserved for the biomedical final evaluation. Conflicting
  duplicates remove whole affected training source groups before splitting.
- Authored synthetic clinical relationships cover negation, temporality,
  experiencer, numbers and medication status. Conditions are source groups.
  Held-out conditions measure lexical transfer within these templates, not
  independent clinical generalization. The separate committed negation
  challenge is not a training input.

Split manifests retain counts, label distributions, source-group counts,
hashes and license references. Check normalized pair and source-group overlap
before training. Training discards overlength pairs without clipping; held-out
overlength pairs count as incorrect/abstained and are reported separately.
Do not commit downloaded text, weights or local training partitions. Generated
audit evidence contains only counts, rates, hashes and controlled metadata.

MedNLI, MIMIC, i2b2 and n2c2 are not downloaded or trained on. MedNLI remains
DUA-gated, user-supplied and evaluation-only. If lawful local access exists,
report its aggregate independently; no examples or derived training data
may enter a published artifact.

## Candidate gates and exports

`build_nli_candidate_report` composes real serving-export evaluation into a
`BenchmarkReport`. Promotion floors are fixed before inspecting the final
holdout: public three-way accuracy 0.65, biomedical binary accuracy 0.75,
synthetic three-way accuracy 0.90. Entailment and contradiction operating
points each require validation precision 0.99, recall 0.25 and FPR at most
0.01. Calibration fallback is a failure, not an acceptable operating point.
The synthetic final holdout also requires accepted-entailment recall 0.25,
preventing a vacuous all-abstention success. These are engineering gates,
not clinical validation thresholds.

The separate negation challenge requires zero false entailments. Error-slice
reports cover all five clinical phenomena and keep abstentions visible.
The exported int8 model supplies the actual safety and held-out predictions.
Parity includes at least 96 real held-out/challenge pairs: MLX float32 must
match PyTorch argmax exactly with maximum probability delta 0.001; ONNX int8
requires argmax agreement at least 0.98 and delta at most 0.05. Tokenization,
the complete pair and the BERT pooler are shared across runtimes.

The export directory contains PyTorch `model.safetensors`, dynamic-axis
`model.onnx`, `model_int8.onnx`, and `mlx/weights.safetensors` plus its explicit
sequence-classification configuration. The evidence directory contains
aggregate calibration, negation, error-slice and benchmark reports, an export
checksum manifest and offline-load evidence. The candidate digest is a weight
SHA-256, not a fabricated Hub commit. Publication remains a failing gate until
an actual immutable public revision has been independently resolved and its
checksums verified. No `models.jsonl` alias may be enabled on failed evidence.

The earlier `build_nli_preparation_report` remains available for report
composition. Its publication gate always fails; synthetic unit-test results
are never checkpoint quality evidence. Paid compute needs a separate budget
decision. Public checkpoint publication must stay within the named model
delivery task, after review of genuine quality and provenance evidence; these
local scripts contain no publication or visibility-changing operation.
