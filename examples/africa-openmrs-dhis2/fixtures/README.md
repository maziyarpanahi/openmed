# Synthetic fixture boundary

The fixture demo generates a deterministic, fictional OpenMRS FHIR2 recording
in memory and serves it only on an ephemeral `127.0.0.1` port. The committed
organisation-unit snapshot represents one fictional country, province,
district, and three facilities. No Java service, internet access, credentials,
or real patient corpus is used.

Regenerate an inspectable recording from the repository root:

```bash
python examples/africa-openmrs-dhis2/synthetic_data.py \
  --seed 875 \
  --patient-count 50 \
  --output /tmp/openmed-africa-recording.json
```

The resulting file intentionally contains synthetic PHI and must remain inside
the facility-side test boundary. It is not a DHIS2 export artifact.
