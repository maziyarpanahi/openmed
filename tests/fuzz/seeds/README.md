# Synthetic format-parser seeds

Every value in this directory is fabricated for parser testing. The corpus
contains no real patient data or identifiers.

- `minimal.eml`: RFC 5322/MIME email seed for the EML/MSG parser family.
- `minimal.md`: Markdown seed.
- `minimal.hl7`: HL7 v2 ADT seed.
- `minimal.cda.xml`: CDA ClinicalDocument seed.
- `minimal.rtf`: Rich Text Format seed.
- `minimal.odt`: OpenDocument Text ZIP seed.
- `minimal.epub`: EPUB ZIP seed.
- `minimal.x12`: X12 837 interchange seed.

The fuzz harness starts with each valid seed, then explores truncations,
deletions, insertions, bit flips, and arbitrary replacement bytes.
