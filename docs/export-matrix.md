# Export Format Support Matrix

This matrix is the quick reference for OpenMed model-export support across
architecture families and deployment tiers. It keeps the current backend limits
in one place so contributors can decide whether a model belongs on the MLX,
CoreML, browser, or embedding-only path before opening an export issue.

Status key:

- **Supported**: the repository has a current runtime or converter path for this
  combination.
- **Partial**: a path exists with the limitation named in the cell.
- **Unsupported**: no stable public path exists yet.

## Current Limitations

- GGUF is embedding-only in the current export story; it is not a
  token-classification or span-classification runtime.
- CoreML is token-classification-only today; zero-shot span, classification, and
  relation-extraction tasks should stay on the MLX or PyTorch path.
- The MLX rows track the current `_SUPPORTED_TOKEN_CLASSIFICATION_MODEL_TYPES`
  allowlist plus the experimental GLiNER-family artifacts.
- INT4 only-if-recall-holds: 4-bit MLX artifacts should be published only when
  recall remains acceptable for the target clinical or PII task.

## Architecture by Format

| Architecture family | MLX-fp | MLX-8bit | MLX-4bit | CoreML-fp16 | CoreML-int8 | ONNX | Transformers.js | GGUF |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `bert` | Supported - BERT token-classification runtime | Supported - quantized MLX weights | Partial - INT4 only-if-recall-holds | Partial - token-classification-only | Partial - token-classification-only and calibration-dependent | Partial - use standard token-classification export | Partial - browser runtime depends on tokenizer/assets | Unsupported - GGUF is embedding-only |
| `distilbert` | Supported - dispatches through the BERT MLX runtime | Supported - quantized MLX weights | Partial - INT4 only-if-recall-holds | Partial - token-classification-only | Partial - token-classification-only and calibration-dependent | Partial - use standard token-classification export | Partial - browser runtime depends on tokenizer/assets | Unsupported - GGUF is embedding-only |
| `electra` | Supported - dispatches through the BERT MLX runtime | Supported - quantized MLX weights | Partial - INT4 only-if-recall-holds | Partial - token-classification-only | Partial - token-classification-only and calibration-dependent | Partial - use standard token-classification export | Partial - browser runtime depends on tokenizer/assets | Unsupported - GGUF is embedding-only |
| `roberta` | Supported - dispatches through the BERT MLX runtime | Supported - quantized MLX weights | Partial - INT4 only-if-recall-holds | Partial - token-classification-only | Partial - token-classification-only and calibration-dependent | Partial - use standard token-classification export | Partial - browser runtime depends on tokenizer/assets | Unsupported - GGUF is embedding-only |
| `xlm-roberta` | Supported - multilingual BERT-family MLX path | Supported - quantized MLX weights | Partial - INT4 only-if-recall-holds | Partial - token-classification-only | Partial - token-classification-only and calibration-dependent | Partial - use standard token-classification export | Partial - browser runtime depends on tokenizer/assets | Unsupported - GGUF is embedding-only |
| `xlm_roberta` | Supported - alias for `xlm-roberta` | Supported - quantized MLX weights | Partial - INT4 only-if-recall-holds | Partial - token-classification-only | Partial - token-classification-only and calibration-dependent | Partial - use standard token-classification export | Partial - browser runtime depends on tokenizer/assets | Unsupported - GGUF is embedding-only |
| `deberta` | Supported - resolves to the DeBERTa-v2 MLX runtime | Supported - quantized MLX weights | Partial - INT4 only-if-recall-holds | Partial - token-classification-only; current arm64 packaging is still rollout work | Partial - token-classification-only and calibration-dependent | Partial - use standard token-classification export | Partial - browser runtime depends on tokenizer/assets | Unsupported - GGUF is embedding-only |
| `deberta-v2` | Supported - native DeBERTa-v2 MLX runtime | Supported - quantized MLX weights | Partial - INT4 only-if-recall-holds | Partial - token-classification-only; current arm64 packaging is still rollout work | Partial - token-classification-only and calibration-dependent | Partial - use standard token-classification export | Partial - browser runtime depends on tokenizer/assets | Unsupported - GGUF is embedding-only |
| `modernbert` | Unsupported - converter support is still in active rollout | Unsupported - no supported MLX runtime yet | Unsupported - no supported MLX runtime yet | Partial - target architecture is planned, not stable | Unsupported - no stable quantized CoreML path yet | Partial - backend-specific export work required | Partial - backend-specific export work required | Unsupported - GGUF is embedding-only |
| `longformer` | Unsupported - converter support is still in active rollout | Unsupported - no supported MLX runtime yet | Unsupported - no supported MLX runtime yet | Partial - target architecture is planned, not stable | Unsupported - no stable quantized CoreML path yet | Partial - backend-specific export work required | Partial - long-sequence browser limits apply | Unsupported - GGUF is embedding-only |
| `openai-privacy-filter` | Supported - OpenAI Privacy Filter MLX runtime | Supported - quantized MLX weights | Partial - INT4 only-if-recall-holds | Partial - token-classification-only | Partial - token-classification-only and calibration-dependent | Partial - preserve BIOES label schema | Partial - tokenizer and label assets must travel with the model | Unsupported - GGUF is embedding-only |
| `privacy-filter` | Supported - alias for `openai-privacy-filter` | Supported - quantized MLX weights | Partial - INT4 only-if-recall-holds | Partial - token-classification-only | Partial - token-classification-only and calibration-dependent | Partial - preserve BIOES label schema | Partial - tokenizer and label assets must travel with the model | Unsupported - GGUF is embedding-only |
| `privacy-filter-nemotron` | Supported - Nemotron weights on the Privacy Filter runtime | Supported - quantized MLX weights | Partial - INT4 only-if-recall-holds | Partial - token-classification-only | Partial - token-classification-only and calibration-dependent | Partial - preserve BIOES label schema | Partial - tokenizer and label assets must travel with the model | Unsupported - GGUF is embedding-only |
| `nemotron-privacy-filter` | Supported - alias for `privacy-filter-nemotron` | Supported - quantized MLX weights | Partial - INT4 only-if-recall-holds | Partial - token-classification-only | Partial - token-classification-only and calibration-dependent | Partial - preserve BIOES label schema | Partial - tokenizer and label assets must travel with the model | Unsupported - GGUF is embedding-only |
| `privacy-filter-multilingual` | Supported - multilingual Privacy Filter runtime | Supported - quantized MLX weights | Partial - INT4 only-if-recall-holds | Partial - token-classification-only | Partial - token-classification-only and calibration-dependent | Partial - preserve multilingual BIOES label schema | Partial - tokenizer and label assets must travel with the model | Unsupported - GGUF is embedding-only |
| `multilingual-privacy-filter` | Supported - alias for `privacy-filter-multilingual` | Supported - quantized MLX weights | Partial - INT4 only-if-recall-holds | Partial - token-classification-only | Partial - token-classification-only and calibration-dependent | Partial - preserve multilingual BIOES label schema | Partial - tokenizer and label assets must travel with the model | Unsupported - GGUF is embedding-only |
| `gliner-uni-encoder-span` | Partial - experimental GLiNER span MLX runtime | Partial - experimental; validate per artifact | Partial - INT4 only-if-recall-holds and span recall must be checked | Unsupported - CoreML is token-classification-only | Unsupported - CoreML is token-classification-only | Partial - task-specific export work required | Partial - task-specific export work required | Unsupported - GGUF is embedding-only |
| `gliclass-uni-encoder` | Partial - experimental GLiClass MLX runtime | Partial - experimental; validate per artifact | Partial - INT4 only-if-recall-holds and classification recall must be checked | Unsupported - CoreML is token-classification-only | Unsupported - CoreML is token-classification-only | Partial - task-specific export work required | Partial - task-specific export work required | Unsupported - GGUF is embedding-only |
| `gliner-uni-encoder-token-relex` | Partial - experimental GLiNER relation-extraction MLX runtime | Partial - experimental; validate per artifact | Partial - INT4 only-if-recall-holds and relation recall must be checked | Unsupported - CoreML is token-classification-only | Unsupported - CoreML is token-classification-only | Partial - task-specific export work required | Partial - task-specific export work required | Unsupported - GGUF is embedding-only |

## Reading the Table

For Python inference on Apple Silicon, prefer the MLX-fp row first and consider
MLX-8bit when memory or cold-start size matters. Use MLX-4bit only after a
task-specific recall check, especially for privacy or clinical extraction.

For Swift and Apple app packaging, use CoreML when you already have a compatible
token-classification package or need simulator/older-platform coverage. Use
Swift MLX for current Apple Silicon and real-device paths that consume OpenMed
MLX artifacts directly.

For browser and interchange formats, treat ONNX and Transformers.js as
backend-specific packaging work. Treat GGUF as an embedding export target only,
not as a replacement for token-classification, span, classification, or
relation-extraction runtimes.
