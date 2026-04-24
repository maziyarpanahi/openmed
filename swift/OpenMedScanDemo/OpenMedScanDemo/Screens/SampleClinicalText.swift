import Foundation

/// Bundled sample clinical note so "Use Sample" returns instant text without
/// requiring OCR. Mirrors the style of the image asset (`SampleClinicalDocument`)
/// but can be fed straight into the PII + clinical pipelines.
public enum SampleClinicalText {
    public static let note: String = """
    PATIENT NOTE — Saint Mary Community Hospital

    Patient: Maria Garcia Lopez
    DOB: 15/03/1985
    MRN: 123456789
    Phone: +34 612 345 678
    Email: maria.garcia@example.com
    Address: 42 Calle del Sol, Madrid, Spain
    Encounter date: 03/04/2026
    Provider: Dr. Ana Fernandez Ruiz, MD — Department of Hematology

    HISTORY OF PRESENT ILLNESS
    The patient is a 41-year-old woman who presents for routine follow-up of chronic myeloid leukemia (CML). She reports three weeks of progressive fatigue, unintentional weight loss of approximately 4 kg, and occasional night sweats. Denies fevers or infection. Adherent to imatinib 400 mg daily; no new medications.

    ALLERGIES
    Penicillin — reported rash in childhood.

    MEDICATIONS
    Imatinib 400 mg PO once daily.
    Folic acid 1 mg PO once daily.

    ASSESSMENT
    Chronic myeloid leukemia, chronic phase — BCR-ABL1 transcript trending up on the last two quarterly PCRs.

    PLAN
    1. Repeat CBC with differential, peripheral smear, and quantitative BCR-ABL1 PCR today.
    2. Obtain bone marrow biopsy if transcript levels remain >0.1% IS on repeat testing.
    3. Continue imatinib 400 mg daily; counseled on adherence.
    4. Follow-up clinic visit in 6 weeks with Dr. Fernandez.
    5. Return precautions discussed — seek care for new fevers, bleeding, or severe fatigue.

    Signed electronically by Dr. Ana Fernandez Ruiz, MD on 03/04/2026.
    """
}
