# OpenMed v1.4.0

OpenMed v1.4.0 is the multilingual Privacy Filter release.

This release brings the **OpenMed Multilingual Privacy Filter** into the main OpenMed ecosystem across Python, MLX, OpenMedKit, the iOS Scan Demo, and the web demo experience. The new family officially supports 16 languages and ships in PyTorch, MLX full-precision, and MLX 8-bit forms.

The headline: developers can now use the same `extract_pii()` / `deidentify()` API for the OpenAI baseline, OpenAI Nemotron Privacy Filter, and OpenMed Multilingual Privacy Filter, while Apple demos can showcase all three model choices without changing application code.

## Highlights

- Added the OpenMed Multilingual Privacy Filter model family:
  - `OpenMed/privacy-filter-multilingual`
  - `OpenMed/privacy-filter-multilingual-mlx`
  - `OpenMed/privacy-filter-multilingual-mlx-8bit`
- Added Python MLX routing for the multilingual full and 8-bit artifacts.
- Added family-aware fallback so multilingual MLX names resolve to the multilingual PyTorch checkpoint on non-MLX hosts.
- Added MLX family aliases for multilingual Privacy Filter artifacts that reuse the existing OpenAI Privacy Filter runtime and BIOES decoder.
- Updated the OpenMed Scan Demo with the 8-bit multilingual model, a clearer three-model picker, and EN/FR/AR sample buttons.
- Added French and Arabic scanned demo documents for screenshot-ready multilingual flows.
- Added a multilingual web studio that compares the OpenAI baseline, OpenAI Nemotron Privacy Filter, and OpenMed Multilingual Privacy Filter.
- Updated README, anonymization docs, MLX docs, Swift docs, CHANGELOG, and version surfaces for `1.4.0`.

## Privacy Filter Families

OpenMed now documents and routes three Privacy Filter families:

| Variant | PyTorch | MLX full | MLX 8-bit |
| --- | --- | --- | --- |
| OpenAI Privacy Filter | `openai/privacy-filter` | `OpenMed/privacy-filter-mlx` | `OpenMed/privacy-filter-mlx-8bit` |
| OpenAI Nemotron Privacy Filter | `OpenMed/privacy-filter-nemotron` | `OpenMed/privacy-filter-nemotron-mlx` | `OpenMed/privacy-filter-nemotron-mlx-8bit` |
| OpenMed Multilingual Privacy Filter | `OpenMed/privacy-filter-multilingual` | `OpenMed/privacy-filter-multilingual-mlx` | `OpenMed/privacy-filter-multilingual-mlx-8bit` |

All three families use the OpenAI Privacy Filter architecture. The multilingual family uses OpenMed multilingual PII training data and officially supports 16 languages.

## Python Usage

The public API stays the same:

```python
from openmed import extract_pii, deidentify

text = "Patient Marie Dubois, nee le 14/03/1982, email marie.dubois@example.fr."

entities = extract_pii(
    text,
    model_name="OpenMed/privacy-filter-multilingual-mlx-8bit",
)

safe = deidentify(
    text,
    model_name="OpenMed/privacy-filter-multilingual-mlx-8bit",
    method="replace",
    consistent=True,
    seed=42,
)
```

On Apple Silicon with MLX available, the MLX artifact runs through `PrivacyFilterMLXPipeline`. On other hosts, OpenMed substitutes the matching PyTorch checkpoint and emits a one-time warning:

- `OpenMed/privacy-filter-mlx*` -> `openai/privacy-filter`
- `OpenMed/privacy-filter-nemotron-mlx*` -> `OpenMed/privacy-filter-nemotron`
- `OpenMed/privacy-filter-multilingual-mlx*` -> `OpenMed/privacy-filter-multilingual`

## Apple And Demo Updates

The iOS Scan Demo now presents three privacy engines cleanly:

- OpenMed PII
- OpenAI Nemotron Privacy Filter
- OpenMed Multilingual Privacy Filter

The multilingual path uses `OpenMed/privacy-filter-multilingual-mlx-8bit` so the demo stays aligned with the 8-bit Apple artifact strategy. The sample controls now use compact `EN`, `FR`, and `AR` buttons, and switching language/sample clears previous annotations before the next run starts.

The multilingual web studio now uses a single top-to-bottom scan pass and redacts line by line during that pass, matching the original Privacy Filter Studio demo feel without looping the scan effect.

## Upgrade Notes

- The package version is now `1.4.0`.
- Swift demo marketing versions are now `1.4.0`.
- `OpenMed/privacy-filter-multilingual-mlx` and `OpenMed/privacy-filter-multilingual-mlx-8bit` are first-class model names in the MLX routing table.
- The multilingual MLX artifacts must include a valid `openmed-mlx.json`; stale cached HTTP error bodies are no longer treated as manifests by the scan demo downloader.

## Validation

This release adds targeted unit coverage for multilingual Privacy Filter routing, MLX family alias dispatch, and family-aware fallback behavior. The OpenMed Scan Demo was also rebuilt after the multilingual 8-bit integration.
