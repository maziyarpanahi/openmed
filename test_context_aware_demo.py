"""
Demo script showing context-aware PII detection improvements.

This demonstrates the Presidio-inspired context-aware scoring system.
"""

from openmed.core.pii import extract_pii

# Test 1: SSN with clear context
print("=" * 80)
print("Test 1: SSN with context 'SSN:'")
print("=" * 80)
text1 = "Patient SSN: 123-45-6789 and phone: 555-123-4567"
result1 = extract_pii(text1, use_smart_merging=True)
print(f"Text: {text1}")
print(f"\nDetected entities:")
for entity in result1.entities:
    print(f"  - {entity.word!r} => {entity.entity_type} (confidence: {entity.score:.3f})")

# Test 2: SSN without context
print("\n" + "=" * 80)
print("Test 2: SSN without context")
print("=" * 80)
text2 = "The number is 456-78-9123"
result2 = extract_pii(text2, use_smart_merging=True)
print(f"Text: {text2}")
print(f"\nDetected entities:")
for entity in result2.entities:
    print(f"  - {entity.word!r} => {entity.entity_type} (confidence: {entity.score:.3f})")

# Test 3: Differentiate phone vs NPI by context
print("\n" + "=" * 80)
print("Test 3: 10-digit number disambiguated by context")
print("=" * 80)
text3a = "Contact phone: 5551234567"
result3a = extract_pii(text3a, use_smart_merging=True)
print(f"Text: {text3a}")
for entity in result3a.entities:
    print(f"  - {entity.word!r} => {entity.entity_type} (confidence: {entity.score:.3f})")

text3b = "Provider NPI: 1234567893"
result3b = extract_pii(text3b, use_smart_merging=True)
print(f"\nText: {text3b}")
for entity in result3b.entities:
    print(f"  - {entity.word!r} => {entity.entity_type} (confidence: {entity.score:.3f})")

# Test 4: Email with and without context
print("\n" + "=" * 80)
print("Test 4: Email detection")
print("=" * 80)
text4 = "Contact email: patient@example.com for appointments"
result4 = extract_pii(text4, use_smart_merging=True)
print(f"Text: {text4}")
for entity in result4.entities:
    print(f"  - {entity.word!r} => {entity.entity_type} (confidence: {entity.score:.3f})")

# Test 5: Valid vs Invalid SSN (checksum)
print("\n" + "=" * 80)
print("Test 5: Valid SSN vs Invalid (area code 666 or 000)")
print("=" * 80)
text5a = "Valid SSN: 123-45-6789"
result5a = extract_pii(text5a, use_smart_merging=True)
print(f"Text: {text5a}")
for entity in result5a.entities:
    print(f"  - {entity.word!r} => {entity.entity_type} (confidence: {entity.score:.3f})")

text5b = "Invalid SSN: 666-45-6789"  # Area code 666 is invalid
result5b = extract_pii(text5b, use_smart_merging=True)
print(f"\nText: {text5b}")
for entity in result5b.entities:
    print(f"  - {entity.word!r} => {entity.entity_type} (confidence: {entity.score:.3f})")

# Test 6: Full medical record
print("\n" + "=" * 80)
print("Test 6: Complete medical record with multiple PII types")
print("=" * 80)
text6 = """
Patient: John Doe
DOB: 01/15/1970
SSN: 123-45-6789
Phone: (555) 234-5678
Email: john.doe@email.com
Address: 123 Main Street, Springfield
MRN: MRN123456789
Provider NPI: 1234567893
"""
result6 = extract_pii(text6, use_smart_merging=True)
print(f"Detected {len(result6.entities)} entities:")
for entity in sorted(result6.entities, key=lambda e: e.start):
    print(f"  - {entity.word!r:30s} => {entity.entity_type:25s} (score: {entity.score:.3f})")

print("\n" + "=" * 80)
print("Summary: Context-aware scoring boosts confidence when keywords are found nearby!")
print("=" * 80)
