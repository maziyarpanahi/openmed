"""Demo: unified privacy-filter API across MLX and PyTorch.

Same code path on every platform:

    >>> from openmed import extract_pii, deidentify
    >>> r = extract_pii(text, model_name="OpenMed/privacy-filter-mlx-8bit")

On Apple Silicon with MLX installed, this routes through the MLX
pipeline (fast, quantized). On Linux/Windows or any host without MLX,
the same call substitutes the upstream ``openai/privacy-filter`` model
via Transformers and emits a one-time UserWarning explaining the swap.

Output entities have the same shape from either backend, so downstream
``deidentify(...)`` works identically.

This example uses live downloads. Set OPENMED_PRIVACY_FILTER_DOWNLOAD=1
to allow them, or run after the model is cached locally:

    huggingface-cli download openai/privacy-filter
    OPENMED_PRIVACY_FILTER_DOWNLOAD=1 python examples/privacy_filter_unified.py
"""

from __future__ import annotations

import os
import sys
import time

from openmed import deidentify, extract_pii
from openmed.core.backends import select_privacy_filter_backend


SAMPLE_TEXTS = [
    (
        "Discharge note",
        "Patient Sarah Connor (DOB: 03/15/1985) was admitted at "
        "Cedars Hospital under MRN 4471882. Contact: sarah.connor@example.com, "
        "phone (415) 555-7012, address 1640 Riverside Drive, Apt 14, "
        "Los Angeles CA 90039.",
    ),
    (
        "Multilingual short note",
        "Le patient Pierre Durand a téléphoné à Marie Dupont au +33 6 12 34 56 78. "
        "Email: pierre.durand@hopital.fr — RDV le 22 mars 2024 à Paris.",
    ),
    (
        "Portuguese note",
        "Paciente Pedro Almeida, CPF 123.456.789-09, telefone +351 912 345 678, "
        "morada Rua das Flores 25, 1200-195 Lisboa.",
    ),
]


def announce_backend(model_name: str) -> None:
    backend = select_privacy_filter_backend(model_name)
    print(f"  Requested model:   {model_name}")
    print(f"  Selected backend:  {backend}")
    if backend == "torch" and "mlx" in model_name.lower():
        print(f"  -> Running on a non-Apple-Silicon host or MLX is unavailable.")
        print(f"     The library substitutes openai/privacy-filter via Transformers.")
    print()


def run_extraction(text: str, model_name: str) -> None:
    started = time.perf_counter()
    result = extract_pii(
        text,
        model_name=model_name,
        confidence_threshold=0.5,
    )
    elapsed = time.perf_counter() - started
    print(f"  Detected {len(result.entities)} entities in {elapsed*1000:.1f} ms")
    for ent in result.entities[:8]:
        print(f"    {ent.label:24s} {ent.text!r}  conf={ent.confidence:.2f}")
    if len(result.entities) > 8:
        print(f"    ... ({len(result.entities) - 8} more)")
    print()


def run_deidentification(text: str, model_name: str) -> None:
    print("  Deidentified (mask):")
    masked = deidentify(text, method="mask", model_name=model_name,
                        confidence_threshold=0.5)
    print(f"    {masked.deidentified_text}")
    print()
    print("  Deidentified (replace, consistent=True, seed=42):")
    replaced = deidentify(text, method="replace", model_name=model_name,
                          consistent=True, seed=42, confidence_threshold=0.5)
    print(f"    {replaced.deidentified_text}")
    print()


MODEL_IDS = (
    ("OpenAI Privacy Filter (8-bit MLX)", "OpenMed/privacy-filter-mlx-8bit"),
    ("Nemotron Privacy Filter (8-bit MLX)", "OpenMed/privacy-filter-nemotron-mlx-8bit"),
)


def main() -> None:
    if os.environ.get("OPENMED_PRIVACY_FILTER_DOWNLOAD", "").lower() not in {"1", "true", "yes"}:
        print("WARNING: OPENMED_PRIVACY_FILTER_DOWNLOAD is not set.")
        print("If the privacy-filter model is not already cached, the call below will fail.")
        print("Set OPENMED_PRIVACY_FILTER_DOWNLOAD=1 to allow first-run downloads.")
        print()

    # The API accepts either the MLX artifact name (auto-substituted on
    # non-Mac) or the PyTorch model name directly. The fallback is
    # family-aware: a Nemotron MLX request on Linux substitutes
    # OpenMed/privacy-filter-nemotron, not openai/privacy-filter.
    for family_label, model_id in MODEL_IDS:
        print()
        print("#" * 76)
        print(f"# {family_label}")
        print("#" * 76)
        for title, text in SAMPLE_TEXTS:
            print("=" * 76)
            print(title)
            print("=" * 76)
            announce_backend(model_id)
            try:
                run_extraction(text, model_id)
                run_deidentification(text, model_id)
            except Exception as exc:  # noqa: BLE001 - demo runs in many environments
                print(f"  Skipped: {type(exc).__name__}: {exc}")
                print(f"  (Likely the model is not yet downloaded.)")
                print()


if __name__ == "__main__":
    sys.exit(main())
