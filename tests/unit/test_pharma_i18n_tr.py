import unittest
from openmed.core.pharma_i18n import extract_turkish_pharma_entities

class TestPharmaTurkish(unittest.TestCase):
    def test_brand_extraction(self):
        text = "ADALAT CRONO 30 mg kullanıyor."
        entities = extract_turkish_pharma_entities(text)
        
        labels = [e["label"] for e in entities]
        texts = [e["text"] for e in entities]
        
        self.assertIn("DRUG_BRAND", labels)
        self.assertIn("ADALAT CRONO", texts)

    def test_substance_extraction(self):
        text = "Etken maddesi atorvastatin olan ilaç."
        entities = extract_turkish_pharma_entities(text)
        
        labels = [e["label"] for e in entities]
        texts = [e["text"] for e in entities]
        
        self.assertIn("DRUG_SUBSTANCE", labels)
        self.assertIn("atorvastatin", texts)

    def test_case_insensitivity(self):
        text = "adalat crono ve ATORVASTATIN."
        entities = extract_turkish_pharma_entities(text)
        self.assertEqual(len(entities), 2)

    def test_no_match(self):
        text = "Bu cümlede ilaç ismi yok."
        entities = extract_turkish_pharma_entities(text)
        self.assertEqual(len(entities), 0)

if __name__ == "__main__":
    unittest.main()
