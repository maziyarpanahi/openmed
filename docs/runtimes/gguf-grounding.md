# GGUF Grounding Runtime

OpenMed can publish a certified `Q4_K_M` GGUF embedding artifact for local
grounding retrieval. The flow builds on the F16 and Q8_0 embedding export:

```text
local encoder checkpoint
        │
        ├─ OM-195 exporter ─ model-f16.gguf and model-q8_0.gguf
        └─ llama-quantize ─ model-q4_k_m.gguf
```

The checkpoint, converter, quantizer, and embedding executable must already be
available on local storage. OpenMed does not download a model, clone
llama.cpp, or bundle its binaries.

## Export and certify

Install the normal development dependencies, build llama.cpp locally, and
provide its converter, quantizer, and embedding executable paths. The
quantizer is invoked as an external process with the `Q4_K_M` output type:

```python
from openmed.onnx import export_gguf_int4

result = export_gguf_int4(
    "models/synthetic-grounding-encoder",
    "artifacts/grounding-gguf",
    llama_cpp_dir="../llama.cpp",
    embedding_binary="../llama.cpp/build/bin/llama-embedding",
    source_model_id="local/synthetic-grounding-encoder",
)

print(result.q4_k_m_path)
print(result.recall_gate.passed)
```

The export first stages the OM-195 F16 and Q8_0 outputs, quantizes the staged
F16 file, and then runs a synthetic retrieval certificate. For each synthetic
query, the certificate compares the Q4_K_M top-k passage indexes with the F16
parent:

```text
recall_delta = 1 - mean(fp16_top_k ∩ q4_k_m_top_k / top_k)
```

The default tolerance is the shared INT4 G4 limit (`0.01`). The candidate is
also run twice and must return identical normalized vectors. Missing,
malformed, nondeterministic, or over-tolerance evidence rejects the artifact;
the staged files are not published.

## Published bundle and evidence

A passing export contains:

```text
artifacts/grounding-gguf/
├── config.json
├── model-f16.gguf
├── model-q8_0.gguf
├── model-q4_k_m.gguf
├── openmed-gguf.json
└── gguf-grounding-benchmark.json
```

`openmed-gguf.json` records `Q4_K_M`, the G4 verdict, the recall delta, the
report path, and the certified artifact's size and SHA-256 digest. The
benchmark report binds the same artifact identity to synthetic top-k overlap,
determinism, F16 and Q4_K_M latency, and footprint. Loading recomputes the
digest and fails closed if the artifact or either evidence file has been
replaced. Reports contain no clinical terminology, patient text, or raw input
prompts.

Publication rechecks output conflicts after certification. With
`overwrite=True`, existing bundle files are held for rollback until every
staged file has been placed, so a failed replacement restores the previous
outputs.

## Run the local runtime

The runtime is a small subprocess bridge. It never imports a llama.cpp Python
binding. On POSIX systems it streams each prompt through the subprocess's
anonymous standard-input pipe and gives llama.cpp `/dev/stdin` as its prompt
file. Raw text is therefore absent from process arguments, temporary files,
stderr, and OpenMed logs:

```python
from openmed.onnx import load_gguf_grounding_embedder

embedder = load_gguf_grounding_embedder(
    "artifacts/grounding-gguf",
    executable="../llama.cpp/build/bin/llama-embedding",
)
vectors = embedder.encode(["synthetic mention", "synthetic concept label"])
```

Loading validates the local manifest and the passing G4 report before starting
the executable. A failed or missing certificate cannot be bypassed by the
runtime. Runtime requests are bounded to 256 texts, 32,768 characters per
text, and 65,536 vector dimensions; malformed or oversized executable output
is rejected. llama.cpp remains a user-built, out-of-process dependency; no
llama.cpp source or binary is included in OpenMed.

The bridge removes inherited `LLAMA_ARG_*` overrides and rejects RPC, prompt
logging, prompt-cache, remote-model, and model-replacement arguments. Runtime
configuration therefore cannot silently turn the local embedding call into a
networked or raw-prompt-persisting operation.

The direct `llama-embedding` bridge requires a POSIX-compatible `/dev/stdin`.
It fails closed on platforms without that private pipe transport instead of
placing patient text in command-line arguments or temporary files.

All examples use synthetic offline data. Grounding suggestions are assistive
and require human verification; this runtime does not make clinical decisions.
