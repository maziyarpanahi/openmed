"""Compare OpenMed outputs with medical remapping enabled vs disabled.

Run:
    .venv-openmed/bin/python examples/custom_tokenizer/compare_medical_remap.py
"""

from openmed import analyze_text
from openmed.core import OpenMedConfig


TEXT = (
    "62-year-old male with B-cell ALL day +5 post CAR-T (tisagenlecleucel) developed "
    "IL-6-mediated cytokine storm with Tmax 39.8C, tachycardia 128."
)


def summarize(result):
    return [(e.label, e.text, e.start, e.end, float(e.confidence)) for e in result.entities]


def main():
    cfg_off = OpenMedConfig(use_medical_tokenizer=False)
    cfg_on = OpenMedConfig(use_medical_tokenizer=True)

    off = analyze_text(
        TEXT,
        model_name="oncology_detection_superclinical",
        config=cfg_off,
        group_entities=True,
    )
    on = analyze_text(
        TEXT,
        model_name="oncology_detection_superclinical",
        config=cfg_on,
        group_entities=True,
    )

    print("=== Disabled (model spans only) ===")
    for row in summarize(off):
        print(row)

    print("\n=== Enabled (remapped to medical tokens) ===")
    for row in summarize(on):
        print(row)


if __name__ == "__main__":
    main()

