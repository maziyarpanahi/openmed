"""Assess an intended release against an explicit reference population.

The example is fully offline and synthetic. It demonstrates exact k-map and
delta-presence measurement over patient-level longitudinal QI profiles. A
qualified expert must still justify whether a real reference table represents
the anticipated attack population.
"""

from openmed.risk import assess_population_risk


def main() -> None:
    sample = [
        {"sample_unit": "s-1", "age_band": "40-49", "region": "north"},
        {"sample_unit": "s-2", "age_band": "50-59", "region": "south"},
    ]
    reference_population = [
        {
            "population_unit": f"p-{index}",
            "age_band": age_band,
            "region": region,
        }
        for index, (age_band, region) in enumerate(
            (
                ("40-49", "north"),
                ("40-49", "north"),
                ("40-49", "north"),
                ("50-59", "south"),
                ("50-59", "south"),
            ),
            start=1,
        )
    ]

    assessment = assess_population_risk(
        sample,
        reference_population,
        ("age_band", "region"),
        sample_privacy_unit="sample_unit",
        population_privacy_unit="population_unit",
        target_k_map=2,
        max_delta_presence=0.5,
    )

    print(assessment.to_json())
    print(f"assessment_digest={assessment.digest}")


if __name__ == "__main__":
    main()
