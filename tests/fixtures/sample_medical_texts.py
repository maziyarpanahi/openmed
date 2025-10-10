"""Sample medical texts for testing purposes."""

# Clinical Notes
CLINICAL_NOTE_1 = """
Patient: John Doe, 65-year-old male
Date: 2024-01-15

Chief Complaint: Chest pain and shortness of breath

History of Present Illness:
Patient presents with acute onset chest pain radiating to left arm,
associated with shortness of breath and diaphoresis. Pain started
approximately 2 hours ago while at rest.

Past Medical History:
- Type 2 diabetes mellitus diagnosed 2015
- Hypertension diagnosed 2010
- Hyperlipidemia
- Former smoker (quit 2018)

Current Medications:
- Metformin 1000mg BID
- Lisinopril 10mg daily
- Atorvastatin 40mg HS
- Aspirin 81mg daily

Physical Examination:
Vital signs: BP 150/95, HR 98, RR 22, Temp 98.6°F, O2 sat 95% on RA
General: Mild distress, diaphoretic
Cardiovascular: Regular rate and rhythm, no murmurs

Assessment and Plan:
1. Acute coronary syndrome - likely STEMI
   - Serial EKGs and cardiac enzymes
   - Cardiology consultation urgent
   - Consider cardiac catheterization

2. Diabetes management
   - Continue metformin
   - Monitor glucose closely

3. Hypertension
   - May need to adjust antihypertensive regimen
"""

CLINICAL_NOTE_2 = """
Patient: Jane Smith, 45-year-old female
Date: 2024-01-20

Chief Complaint: Fatigue and weight gain

History:
Patient reports 6-month history of progressive fatigue, cold intolerance,
and 15-pound weight gain despite no change in diet. Also notes hair loss
and constipation.

Past Medical History:
- Depression (stable on sertraline)
- Seasonal allergies

Medications:
- Sertraline 100mg daily
- Multivitamin
- Claritin 10mg PRN

Physical Exam:
Vital signs: BP 110/70, HR 58, Temp 97.2°F
General: Appears tired, slow speech
Thyroid: Slightly enlarged, no nodules palpated

Labs:
TSH: 12.5 mIU/L (elevated)
Free T4: 0.8 ng/dL (low)

Assessment:
Primary hypothyroidism

Plan:
- Start levothyroxine 50 mcg daily
- Recheck TSH in 6 weeks
- Patient education on medication timing
"""

# Medication Lists
MEDICATION_LIST_1 = [
    "Metformin 1000mg twice daily",
    "Lisinopril 10mg once daily in morning",
    "Atorvastatin 40mg at bedtime",
    "Aspirin 81mg daily",
    "Insulin glargine 20 units subcutaneous at bedtime"
]

MEDICATION_LIST_2 = [
    "Levothyroxine 75 mcg daily on empty stomach",
    "Sertraline 100mg daily",
    "Omeprazole 20mg daily before breakfast",
    "Vitamin D3 2000 IU daily"
]

# Short Medical Texts
SHORT_TEXTS = [
    "Patient has diabetes and hypertension.",
    "Prescribed metformin 500mg twice daily.",
    "Blood pressure elevated at 160/95 mmHg.",
    "History of myocardial infarction in 2020.",
    "Allergic to penicillin and sulfa drugs.",
    "Underwent coronary artery bypass surgery.",
    "Laboratory results show elevated glucose.",
    "Patient complains of chest pain and dyspnea."
]

# Procedure Notes
PROCEDURE_NOTE = """
Procedure: Colonoscopy
Date: 2024-01-25
Physician: Dr. Smith

Indication: Screening colonoscopy, family history of colon cancer

Procedure:
Patient received conscious sedation with midazolam 2mg IV and
fentanyl 50mcg IV. Colonoscope was inserted and advanced to
the cecum. Withdrawal time was 8 minutes.

Findings:
1. Small polyp in ascending colon - removed with cold forceps
2. Internal hemorrhoids
3. Otherwise normal examination

Pathology:
Polyp sent for histopathologic examination

Recommendations:
- Follow up pathology results
- Repeat colonoscopy in 5 years if pathology benign
- High fiber diet and adequate hydration
"""

# Radiology Report
RADIOLOGY_REPORT = """
Examination: Chest X-ray PA and lateral
Date: 2024-01-22
Clinical indication: Cough and fever

Technique:
PA and lateral chest radiographs obtained.

Findings:
Heart size is normal. There is consolidation in the right lower lobe
consistent with pneumonia. No pleural effusion. No pneumothorax.
Osseous structures are intact.

Impression:
Right lower lobe pneumonia.

Recommendation:
Clinical correlation and appropriate antibiotic therapy. Follow-up
chest X-ray in 4-6 weeks to document resolution.
"""

# Test Cases with Expected Entities
TEST_CASES = [
    {
        "text": "Patient has diabetes and takes metformin 500mg daily.",
        "expected_entities": [
            {"text": "diabetes", "label": "CONDITION", "confidence": 0.95},
            {"text": "metformin", "label": "MEDICATION", "confidence": 0.98},
            {"text": "500mg", "label": "DOSAGE", "confidence": 0.89}
        ]
    },
    {
        "text": "Blood pressure is 140/90 mmHg, heart rate 85 bpm.",
        "expected_entities": [
            {"text": "140/90 mmHg", "label": "VITAL_SIGN", "confidence": 0.92},
            {"text": "85 bpm", "label": "VITAL_SIGN", "confidence": 0.88}
        ]
    },
    {
        "text": "History of myocardial infarction, currently on aspirin and lisinopril.",
        "expected_entities": [
            {"text": "myocardial infarction", "label": "CONDITION", "confidence": 0.97},
            {"text": "aspirin", "label": "MEDICATION", "confidence": 0.94},
            {"text": "lisinopril", "label": "MEDICATION", "confidence": 0.96}
        ]
    }
]

# Edge Cases for Testing
EDGE_CASES = [
    "",  # Empty string
    "   ",  # Whitespace only
    "Patient.",  # Very short text
    "a" * 1000,  # Very long text
    "Patient has 🏥 symptoms.",  # Unicode characters
    "Patient has diabetes; hypertension, and COPD!",  # Multiple punctuation
    "Metformin 500 mg BID x 30 days #30 disp",  # Prescription format
    "H/O DM, HTN, CAD",  # Medical abbreviations
]

# Batch Processing Test Data
BATCH_TEST_DATA = [
    "Patient diagnosed with type 2 diabetes mellitus.",
    "Prescribed metformin 1000mg twice daily with meals.",
    "Blood pressure reading of 150/95 mmHg noted.",
    "History significant for coronary artery disease.",
    "Currently taking lisinopril 10mg daily for hypertension.",
    "Laboratory shows elevated HbA1c at 8.2%.",
    "Patient scheduled for cardiology follow-up.",
    "Medication adherence discussed with patient."
]
