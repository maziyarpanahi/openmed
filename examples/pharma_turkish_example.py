"""Example of Turkish Pharmaceutical NER using OpenMed.

This example demonstrates how to extract drug brands and active substances
from Turkish clinical text.
"""

from openmed.core.pharma_i18n import extract_turkish_pharma_entities

def main():
    # Sample Turkish clinical text
    text = "Hastada hipertansiyon mevcuttur. Tedavi olarak ADALAT CRONO başlandı. Ayrıca atorvastatin kalsiyum dozajı ayarlandı."
    
    print(f"Input Text: {text}\n")
    
    # Extract entities
    entities = extract_turkish_pharma_entities(text)
    
    print("Detected Entities:")
    print("-" * 30)
    for ent in entities:
        canonical = ent["metadata"].get("canonical", "")
        atc = ent["metadata"].get("atc", "")
        print(f"Text: {ent['text']}")
        print(f"Type: {ent['label']}")
        print(f"Canonical: {canonical}")
        if atc:
            print(f"ATC Code(s): {atc}")
        print("-" * 30)

if __name__ == "__main__":
    main()
