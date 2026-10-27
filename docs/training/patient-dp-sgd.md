# Patient-level DP-SGD

`openmed.training.patient_dp_sgd` runs one differentially private SGD step for
a single site that fine-tunes an adapter on its own clinical notes. The privacy
unit is the patient, not the note. A patient with many notes otherwise spends a
multiple of the declared epsilon, so notes are grouped by an opaque patient key
and each patient's contribution to a step is bounded as a whole.

The step is framework-neutral. You compute per-example gradients with any
framework and hand them to the step as nested sequences or as an object with
`tolist()`, such as a numpy array. Neither numpy nor torch is a core
dependency. The step does not include accountant math: you inject an accountant
and the step gives it the accountant inputs.

This module protects records inside one site's training run. Client-level
federated update mechanisms, trainer orchestration and accountant math are
separate components.

## Mechanism

Each call to `PatientDPSGD.step()` does the following:

1. Groups notes into units by patient key, in first-appearance order.
2. Draws one value from the sampling randomness source per unit. A unit is
   included when the value is below the declared sampling rate (Poisson
   sampling). The realized batch size varies from step to step.
3. Calls your gradient provider once with the ascending note positions of
   the sampled units. The provider receives positions only, never keys.
4. Averages each sampled patient's note gradients, then clips that mean to
   the L2 bound `clip_norm`. One patient moves the summed update by at most
   `clip_norm`, however many notes it has.
5. Sums the clipped contributions and adds Gaussian noise with standard
   deviation `noise_multiplier * clip_norm` to every coordinate, using a
   separate noise randomness source.
6. Divides by the declared `expected_batch_size`, a fixed and
   data-independent normalizer, rather than by the realized batch size.

The provider is not called when no unit is sampled. The step still releases
noise of the declared dimension, so an empty batch looks like any other batch.

## Declare the privacy unit with the recipe

`PatientDPSGDDeclaration` records the privacy unit and mechanism and refers to
the adapter training recipe by digest, for example the `recipe_hash` of an
`AdapterRecipeDryRun`. The recipe module does not change. The declaration has
its own `digest`, and every step record and privacy report carries it.

```python
import math
import random

from openmed.training.patient_dp_sgd import (
    PatientDPSGD,
    PatientDPSGDDeclaration,
)

# Synthetic recipe digest and gradients for this example only.
declaration = PatientDPSGDDeclaration(
    recipe_digest="sha256:" + "a" * 64,
    privacy_unit="patient",
    batch_sampling="poisson",
    amplification_claimed=True,
    sampling_rate=0.5,
    expected_batch_size=2.0,
    noise_multiplier=1.1,
    clip_norm=1.0,
    max_steps=3,
)

session = PatientDPSGD(
    declaration,
    gradient_dimension=2,
    sampling_rng=random.SystemRandom(),
    noise_rng=random.SystemRandom(),
)

note_patient_keys = ["patient-a", "patient-a", "patient-b", "patient-c"]
synthetic_rows = [[0.2, -0.1], [0.4, 0.3], [5.0, 5.0], [0.0, 0.1]]


def gradient_provider(positions):
    # Compute per-example gradients only for the sampled notes.
    return [synthetic_rows[position] for position in positions]


result = session.step(note_patient_keys, gradient_provider)
assert len(result.update) == 2
assert all(math.isfinite(value) for value in result.update)
assert result.record.to_dict()["guarantee_claimed"] is True
```

Apply `result.update` to the adapter parameters with your optimizer. The
result's `repr` shows only the step number and the dimension.

## Refused configurations

The declaration and the session fail closed. Each refusal raises
`PatientDPSGDError` whose message and only argument is a fixed reason code.

| Reason code | When it is raised |
| --- | --- |
| `dp_sampling_not_poisson` | Amplification is claimed with `fixed_size` or `shuffle` batches, or the batch sampling is unknown. |
| `dp_noise_missing` | `noise_multiplier` is `None` or zero. |
| `dp_config_invalid` | A clip norm, rate, batch size, step count, flag or dimension is invalid, or a rate below 1.0 is declared without amplification. |
| `dp_privacy_unit_note_level` | A note-level unit is declared while patient keys are available, or a note-level step receives patient keys. |
| `dp_privacy_unit_invalid` | The privacy unit is not `patient` or `note`. |
| `dp_patient_key_invalid` | A note has no opaque string key of 1 to 256 characters under the patient unit. |
| `dp_recipe_digest_invalid` | The recipe reference is not a lowercase `sha256:` digest. |
| `dp_rng_not_csprng` | Operational mode receives anything other than an exact `random.SystemRandom` instance. |
| `dp_rng_reused` | One source serves both sampling and noise, or a source is already bound to another session in this process. |
| `dp_rng_invalid` | A test-mode source returns an out-of-range or non-finite value. |
| `dp_mode_invalid` | The mode is not `operational` or `test`. |
| `dp_gradient_invalid` | Gradient rows have the wrong count or length, or contain non-finite or non-numeric values. |
| `dp_gradient_provider_failed` | The gradient provider raised. Its message and context are dropped. |
| `dp_step_budget_exhausted` | A step would exceed `max_steps`. |
| `dp_session_closed` | An earlier step failed. A failed step closes the session so it cannot be retried until it succeeds, and the attempt still counts as a step. |
| `dp_accountant_invalid` | The accountant, delta or accountant identifier is invalid, or the accountant returned a value that is not a finite, non-negative number. |
| `dp_accountant_failed` | The accountant raised. |
| `dp_test_mode_no_guarantee` | A privacy report is requested from a test session. |
| `dp_adapter_unavailable` | The optional torch adapter is used without torch installed. |

Without amplification (`amplification_claimed=False`) the sampling rate must
be `1.0`, so every unit is used every step. The accountant then sees
`sampling="full_batch"` and a rate of 1.0.

## Accountant binding and privacy reports

`PatientDPSGDDeclaration.accountant_inputs()` returns the inputs for the
declared schedule, and `PatientDPSGD.accountant_inputs()` returns them for the
steps attempted so far. A privacy report calls the injected accountant with
the keyword arguments `sampling_rate`, `noise_multiplier`, `steps` and `delta`.

```python
import random

from openmed.training.patient_dp_sgd import (
    PatientDPSGD,
    PatientDPSGDDeclaration,
)


def my_accountant(*, sampling_rate, noise_multiplier, steps, delta):
    # Replace with a reviewed Poisson-subsampled Gaussian RDP or PRV
    # accountant. This placeholder only shows the calling convention.
    return 0.5 * steps


declaration = PatientDPSGDDeclaration(
    recipe_digest="sha256:" + "a" * 64,
    sampling_rate=256 / 60_000,
    expected_batch_size=256.0,
    noise_multiplier=1.1,
    clip_norm=1.0,
    max_steps=14_062,
)
planned = declaration.accountant_inputs()
assert planned.steps == 14_062 and planned.sampling == "poisson"

session = PatientDPSGD(
    declaration,
    gradient_dimension=1,
    sampling_rng=random.SystemRandom(),
    noise_rng=random.SystemRandom(),
)
session.step(["patient-a"], lambda positions: [[0.1]] * len(positions))
report = session.privacy_report(
    my_accountant, delta=1e-5, accountant_id="example.placeholder"
)
assert report.accountant_inputs.steps == 1
assert report.privacy_unit == "patient"
```

The report names the privacy unit, the accountant identifier, the accountant
inputs, the recipe and declaration digests, and a fixed list of assumption
codes:

- `poisson_patient_sampling` (or `no_sampling_amplification`)
- `add_remove_one_unit_adjacency`
- `per_patient_l2_clipping`
- `gaussian_noise_scaled_to_clip_norm`
- `fixed_expected_batch_normalization`
- `os_csprng_randomness`
- `floating_point_gaussian_not_hardened`
- `accountant_injected_not_verified`

The unit tests bind the accountant inputs to an integer-order Renyi DP
accountant for the Poisson-subsampled Gaussian mechanism (Mironov, Talwar and
Zhang, 2019). For the TensorFlow Privacy MNIST tutorial schedule (60,000
examples, batch 256, noise multiplier 1.1, 60 epochs, delta 1e-5) it gives the
published epsilon of 3.01. That test checks the binding only. It is not a
statement about any OpenMed model.

## Randomness modes

Operational mode accepts only exact `random.SystemRandom` instances, which
read operating-system randomness. Sampling and noise need separate instances,
and an instance bound to one session cannot be bound to another.

`mode="test"` accepts deterministic sources such as `random.Random(seed)` for
synthetic tests. Every test-mode record has `test_only=True` and
`guarantee_claimed=False`, test-mode accountant inputs have `test_only=True`,
and `privacy_report()` refuses with `dp_test_mode_no_guarantee`. Seeded
output must never be used to make a privacy claim.

## What leaves the step

The step returns the noisy, normalized update and a step record. The record
holds the schema version, step index, declaration digest, mode, privacy unit,
gradient dimension, `noise_applied`, `test_only` and `guarantee_claimed`. It
does not hold per-example gradients, patient keys, note text, noise draws, or
the counts of sampled patients or notes, because the realized Poisson batch
size is not privatized. The module writes no logs.

## Optional torch adapter

`flatten_torch_per_example_gradients()` turns per-example gradient tensors
(example axis first, for example `grad_sample` tensors) into gradient rows.
`unflatten_torch_update()` splits a released update back into tensors shaped
like the parameters. Both import torch lazily and raise
`dp_adapter_unavailable` when it is not installed.

## Assumptions and limits

- Gradients come from a model trained only on this site's data under the
  declared recipe. Gradients that depend on other patients beyond the clipped
  per-patient mean, for example through batch normalization across patients,
  break the sensitivity bound.
- Each patient key must stand for one real patient. A patient split across
  several keys gets a multiple of the declared epsilon.
- Noise uses floating-point Gaussian draws. They are not hardened against
  floating-point side channels (Mironov, CCS 2012). Treat the report as a
  statement under that assumption.
- The accountant is injected and not verified here. A report is only as good
  as the accountant, the declared unit and the assumptions it lists.
- Synthetic utility results are not evidence of a formal guarantee. This
  module makes no clinical decisions and authorizes no model publication.
