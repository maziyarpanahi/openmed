# Diagnosis-to-treatment relation candidates

`generate_diagnosis_treatment_candidates()` requires explicit linking language
between existing diagnosis and treatment spans. Supported guarded patterns
include diagnosis-first forms such as “treated with” and treatment-first forms
such as “for” or “to treat.” Co-occurrence alone produces no candidate.

```python
from openmed.clinical import generate_diagnosis_treatment_candidates

text = "Plan: possible pneumonia treated with ceftriaxone."
candidates = generate_diagnosis_treatment_candidates(
    text,
    [
        {"label": "DIAGNOSIS", "start": 15, "end": 24},
        {"label": "MEDICATION", "start": 38, "end": 49},
    ],
    max_distance=80,
    allowed_sections={"plan"},
)
```

Both endpoints must share a deterministic sentence and section, satisfy
`max_distance`, and—when supplied—belong to `allowed_sections`. Results include
offset/hash cue evidence, assertion axes for both endpoints, and a controlled
`none`, `possible`, `conditional`, or `refuted` uncertainty state.

Every output has `review_required=True`, `treatment_recommendation=False`, and
`treatment_evaluated=False`. The API does not recommend or assess treatment,
copies no source surfaces into serialization, is deterministic, and performs no
network call.

