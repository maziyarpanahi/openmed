# Synthetic de-identification fuzz corpus

Every case is fabricated and contains no clinical or personal data. The corpus
uses a small ASCII envelope so invalid UTF-8 and large logical inputs remain
portable, reviewable, and compact in Git.

- `text` passes the remaining bytes through unchanged.
- `hex` decodes the remaining ASCII hexadecimal as input bytes.
- `repeat` expands a byte unit to a bounded target length.
- `custom-rule-count` creates a bounded synthetic custom-recognizer mapping.

Run coverage-guided fuzzing for five minutes when Atheris is installed:

```bash
python tests/fuzz/fuzz_deidentify.py tests/fuzz/corpus -max_total_time=300
```

Run deterministic corpus replay without a fuzzing engine:

```bash
python tests/fuzz/fuzz_deidentify.py --replay tests/fuzz/corpus
```

The harness is offline and logs only input length, SHA-256, and outcome.
