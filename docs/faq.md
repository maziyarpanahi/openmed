# Frequently Asked Questions

## Install and Run

### Can OpenMed run locally?

Yes. OpenMed can be run locally without sending your data to external services.

For offline/local-only usage, enable:

`OPENMED_OFFLINE`

Refer to the installation documentation for setup details.

---

## Models and Languages

### Which models are supported?

OpenMed supports different models depending on the language and use case.

Check the model documentation for the supported models and configurations.

---

### Which languages are supported?

OpenMed provides de-identification support for multiple languages.

The recommended model depends on the language being processed.

---

## Privacy and De-identification

### Is de-identification reversible?

De-identification removes sensitive information from text.

Whether the original information can be recovered depends on the method and configuration used. Always review results before using de-identified data.

---

### What happens to identifiers that do not match specific categories?

OpenMed maps certain identifiers into canonical labels.

For example, different types of identification numbers may be grouped under the `ID_NUM` label.

---

## Licensing

### What license does OpenMed use?

OpenMed is released under the Apache-2.0 license.

You can use, modify, and distribute the project according to the license terms.

---

## Performance

### Does OpenMed require a GPU?

No. OpenMed can run on CPU.

A GPU can improve performance, especially for larger workloads or when processing large amounts of text.

---

### Where can I find more documentation?

Check the project documentation pages for detailed guides, installation instructions, and usage examples.