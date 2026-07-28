# GGUF Embedding Export

OpenMed exports local encoder backbones to GGUF for embedding inference in
llama.cpp-compatible runtimes. The export is intended for dense grounding and
retrieval models such as SapBERT. It produces both an F16 artifact and a Q8_0
artifact from the same local checkpoint.

Token-classification checkpoints are intentionally rejected. llama.cpp supports
the BERT-family encoder embedding path, but it does not expose the classifier
head required to turn token representations into OpenMed entity labels. Export
the underlying encoder backbone instead, for example a checkpoint whose
`config.json` names `BertModel` rather than `BertForTokenClassification`.

## Prerequisites

Keep the model and converter local. OpenMed does not clone llama.cpp or download
a model as a side effect of export.

1. Build or check out [llama.cpp](https://github.com/ggml-org/llama.cpp) and
   install the Python dependencies required by its `convert_hf_to_gguf.py`.
2. Download the embedding checkpoint into a local directory containing
   `config.json`, its weights, and tokenizer assets.

The converter accepts either a llama.cpp checkout or a direct path to the
conversion script. `LLAMA_CPP_DIR` can also point at the checkout.

## Export F16 and Q8_0

```bash
python -m openmed.gguf.convert \
  --model ./models/SapBERT-from-PubMedBERT-fulltext \
  --output ./artifacts/sapbert-gguf \
  --llama-cpp ../llama.cpp \
  --source-model-id cambridgeltl/SapBERT-from-PubMedBERT-fulltext
```

The command invokes the upstream converter once with `--outtype f16` and once
with `--outtype q8_0`. It writes:

```text
artifacts/sapbert-gguf/
├── config.json
├── model-f16.gguf
├── model-q8_0.gguf
└── openmed-gguf.json
```

Existing OpenMed GGUF files are not replaced unless `--overwrite` is supplied.
Each variant has a one-hour timeout by default; use `--timeout` to adjust it for
the checkpoint and machine.

## Manifest contract

`openmed-gguf.json` is the artifact manifest used by the later publishing path.
Its canonical `formats` value is `gguf`, matching `models.jsonl`, while each
artifact records its precision and quantization explicitly:

```json
{
  "format": "openmed-gguf",
  "format_version": 1,
  "formats": ["gguf"],
  "task": "feature-extraction",
  "artifacts": [
    {
      "format": "gguf",
      "path": "model-f16.gguf",
      "precision": "float16",
      "quantization": "F16"
    },
    {
      "format": "gguf",
      "path": "model-q8_0.gguf",
      "precision": "q8_0",
      "quantization": "Q8_0"
    }
  ]
}
```

Export only prepares local artifacts. It does not create or upload a model
repository.
