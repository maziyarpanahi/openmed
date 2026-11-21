"""Write invented temporal notes directly into a test's temporary directory."""

from xml.etree import ElementTree as ET


def write_temporal_fixture(path, relations=("BEFORE",)):
    text = "On 2031-06-01 a test before therapy. No rash. Visit showed a ward."
    root = ET.Element("ClinicalNarrativeTemporalAnnotation")
    ET.SubElement(root, "TEXT").text = text
    tags = ET.SubElement(root, "TAGS")

    def span(tag, surface, identifier, kind, **attrs):
        start = text.index(surface)
        ET.SubElement(
            tags,
            tag,
            id=identifier,
            text=surface,
            start=str(start),
            end=str(start + len(surface)),
            type=kind,
            **attrs,
        )

    span("TIMEX3", "2031-06-01", "T0", "DATE", val="2031-06-01", mod="NA")
    span("SECTIME", "2031-06-01", "S0", "ADMISSION", dvalue="2031-06-01")
    for i, (surface, kind) in enumerate(
        (
            ("test", "TEST"),
            ("therapy", "TREATMENT"),
            ("rash", "PROBLEM"),
            ("Visit", "OCCURRENCE"),
            ("showed", "EVIDENTIAL"),
            ("ward", "CLINICAL_DEPT"),
        )
    ):
        span(
            "EVENT",
            surface,
            f"E{i}",
            kind,
            modality="FACTUAL",
            polarity="NEG" if surface == "rash" else "POS",
        )
    for i, relation in enumerate(relations):
        ET.SubElement(tags, "TLINK", id=f"L{i}", fromID="E0", toID="E1", type=relation)
    ET.ElementTree(root).write(path, encoding="utf-8")
