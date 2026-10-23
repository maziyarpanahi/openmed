# Declared synthetic waveform records

`generate_waveform_record(WaveformConfig(...))` creates deterministic local
software stimuli with configurable duration, sample rate, canonical lead names,
pulse rate, regular/irregular/pause timing, bounded noise and optional event
annotations. A local seeded PRNG leaves global random state unchanged. Input lead
order does not affect output identities.

Fixed Gaussian pulse components and scaled lead copies are **not** a clinically
validated ECG simulator or physiological lead-projection model. Timing labels
describe generator decisions, not diagnoses. Events identify template pulses and
intentional pauses with exact relative times and sample indices. Synthetic device
metadata uses a fixed manufacturer code and seed-derived digest, never an actual
device, subject or institutional identifier.

All exports carry `synthetic=true`, source/version markers, millivolt units and a
mandatory non-diagnostic notice. `to_dict()` exports generated stimulus samples;
`report()` exports counts and a digest without samples. Do not substitute patient
signals into this generator's declared synthetic records. No network, models,
restricted datasets or clinical decisions are involved. Duration, rate, channels,
noise and numeric values are bounded before allocating the signal arrays.

Deterministic identities assume the pinned Python/libm environment. These records
exercise software handling; they do not establish real-world device accuracy,
clinical realism, privacy protection or downstream clinical performance.
