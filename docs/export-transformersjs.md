# Transformers.js Export

OpenMed can package an ONNX token-classification export for browser inference
through Transformers.js. The bundle is built from the ONNX artifact created by
`openmed.onnx.convert` and contains the file layout expected by browser
pipelines:

```text
transformersjs/
  config.json
  tokenizer.json
  tokenizer_config.json
  quantize_config.json
  transformersjs-contract.json
  onnx/
    model.onnx
    model_quantized.onnx
```

The converter validates that the ONNX graph exposes the token-classification
contract used by browser pipelines:

- inputs: `input_ids` and `attention_mask`, plus optional `token_type_ids`
- output: `logits`
- dynamic axes: `[batch, sequence]` for inputs and
  `[batch, sequence, labels]` for logits

## From a fresh ONNX conversion

Use `--include-transformersjs` when creating ONNX artifacts:

```bash
.venv/bin/python -m openmed.onnx.convert \
  --model OpenMed/example-token-classifier \
  --output dist/example-onnx \
  --include-transformersjs
```

The OpenMed ONNX manifest then records all emitted runtime formats, including
`transformersjs`, so later publish steps can carry the format list forward.

## From an existing ONNX export

If `model.onnx`, `config.json`, `tokenizer.json`, and
`tokenizer_config.json` already exist in an ONNX export directory, build only
the browser bundle:

```bash
.venv/bin/python -m openmed.onnx.transformersjs \
  --onnx-export-dir dist/example-onnx
```

By default this writes `dist/example-onnx/transformersjs` and updates
`dist/example-onnx/openmed-onnx.json` if that manifest exists. Pass
`--no-manifest-update` to leave the source manifest unchanged.

## Validation

The Python validator checks required files, `config.json` label metadata, and
the ONNX graph contract:

```python
from openmed.onnx import validate_transformersjs_bundle

validate_transformersjs_bundle("dist/example-onnx/transformersjs")
```

The repository also includes a headless Node smoke fixture at
`tests/fixtures/onnx/transformersjs_smoke.mjs`. It verifies the same file
layout and tensor contract from the generated `transformersjs-contract.json`.

## Browser usage

Once the `transformersjs/` directory is served by your application or copied
into a static model asset path, load it with Transformers.js:

```javascript
import { pipeline } from "@huggingface/transformers";

const detector = await pipeline(
  "token-classification",
  "/models/openmed-pii/transformersjs",
  { device: "webgpu" },
);

const entities = await detector("Patient Casey Example called 212-555-0198.");
```

Use the browser bundle for local-first token classification where PHI should
stay inside the user's browser session. The export step only packages model
artifacts; it does not change Hugging Face repository visibility.

For an offline synthetic walkthrough that prints the expected bundle files and
browser load snippet, run:

```bash
uv run python examples/v17_multimodal_browser_interop.py
```
