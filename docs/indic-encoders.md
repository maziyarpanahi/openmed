# Indic encoder backbones

OpenMed can expose a user-supplied MuRIL or IndicBERT backbone to an Indic
entity adapter. This is an opt-in embedding contract, not a replacement for the
complete Hindi and Telugu PII predictors used by `extract_pii` and
`deidentify`.

OpenMed does not bundle encoder weights and does not resolve a repository until
you configure one. Listing PII models never downloads weights. A missing
configuration, optional dependency, or checkpoint returns a skip reason so
offline and base-package imports remain deterministic.

## Supported families

| Family | License | Capability |
|---|---|---|
| MuRIL | Apache-2.0 | 17 Indian languages plus transliterated counterparts |
| IndicBERT | MIT | 12 languages |

MuRIL was trained with transliterated segment pairs. It is therefore the
preferred backbone for Hinglish and other code-mixed paths where an Indian
language is written in Latin script. A backbone only produces aligned hidden
states; an application still needs an entity adapter or task head evaluated on
its own synthetic and governed data.

## Configure and load

Install the optional runtime without changing the base package footprint:

```bash
pip install "openmed[hf]" torch
```

Then configure either a repository identifier that you have chosen or an
existing local directory:

```python
from openmed.core.model_registry import (
    configure_indic_encoder,
    get_pii_models_by_language,
    load_configured_indic_encoder,
)

configure_indic_encoder(
    "google/muril-base-cased",
    family="muril",
    languages=("hi", "te"),
    local_files_only=True,
)

# Existing dedicated predictors remain present; the configured backbone is
# added only when its optional runtime dependencies are importable.
models = get_pii_models_by_language("hi")

resolution = load_configured_indic_encoder()
if not resolution.available:
    print(resolution.skip_reason)
else:
    encoded = resolution.handle.encode("Patient Asha Verma ka record dekhiye.")
    print(encoded.last_hidden_state.shape, encoded.offset_mapping)
```

Set `local_files_only=False` only when you intentionally want the explicitly
configured repository to be resolved through the model host. Private or gated
repositories can use the `token=` argument. Tokens are held only in the
process-local configuration and are excluded from its representation.

The encoder output contains tokenizer tensors, an attention mask, character
offsets, final hidden states, and a SHA-256 input identifier. It contains no raw
input text. Encoding diagnostics log only that digest and aggregate lengths.

Use `clear_indic_encoder_config()` to remove the process-local configuration.

## Model-card provenance

When publishing metadata for an adapter that consumes one of these backbones,
add an `encoder_provenance` object to the manifest row passed to the model-card
renderer:

```json
{
  "family": "MuRIL",
  "source": "google/muril-base-cased",
  "license": "Apache-2.0",
  "provenance": "user-supplied",
  "weights": "user-supplied; not bundled",
  "supports_transliterated_text": true
}
```

The rendered model card includes these values in a dedicated Encoder
Provenance section.
