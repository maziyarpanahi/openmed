# Presidio Recognizer

`openmed.interop.presidio_recognizer.OpenMedRecognizer` lets you add an OpenMed
PII model to a Presidio `AnalyzerEngine` as a regular recognizer. Presidio keeps
its own pattern recognizers and NLP engine; OpenMed adds clinical-grade name,
contact, date and identifier detection next to them.

```bash
pip install "openmed[presidio,hf]"
```

```python
from presidio_analyzer import AnalyzerEngine
from openmed.interop.presidio_recognizer import OpenMedRecognizer

analyzer = AnalyzerEngine()
analyzer.registry.add_recognizer(OpenMedRecognizer())

results = analyzer.analyze(text=note, language="en")
```

## Behavior

- **Entity names.** Results use names Presidio already defines (`PERSON`,
  `PHONE_NUMBER`, `EMAIL_ADDRESS`, `DATE_TIME`, `LOCATION`, `US_SSN`, and the
  rest of `SUPPORTED_ENTITIES`). OpenMed labels without a Presidio equivalent,
  such as age, are not reported. Passing an unknown name in
  `supported_entities` raises `ValueError`.
- **Local cache only.** The recognizer runs with `local_only=True`, so the model
  must already be cached (for example after one `openmed.extract_pii` call while
  online). Like any OpenMed local-only config, this sets the Hugging Face offline
  environment flags for the process. The model loads on the first `analyze`
  call, not when the recognizer is built.
- **No source text.** `analysis_explanation` and `recognition_metadata` carry the
  recognizer name and model id only, never the matched text.
- **Threshold.** `confidence_threshold` (default `0.5`) drops low-confidence
  OpenMed spans before Presidio sees them.

## Options and testing

`create_openmed_recognizer()` takes the same options plus injection points for
`extractor`, `base_cls`, `result_cls` and `explanation_cls`, so tests can run
without Presidio or a model:

```python
from openmed.interop.presidio_recognizer import create_openmed_recognizer

recognizer = create_openmed_recognizer(
    model_name="OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1",
    confidence_threshold=0.7,
    supported_entities=["PERSON", "PHONE_NUMBER"],
)
```

To convert existing Presidio results into OpenMed spans or merge both result
sets instead, use `openmed.interop.presidio` (`to_canonical`,
`from_canonical`, `merge_with_openmed`).
