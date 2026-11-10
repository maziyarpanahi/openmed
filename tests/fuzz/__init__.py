"""Property-based fuzz harness for OpenMed's privacy-critical paths.

These modules use Hypothesis to assert structural and leakage invariants on the
de-identification surface (``deidentify`` / ``extract_pii`` / ``reidentify``) and
the date-shift helpers. All inputs are synthetically generated; no real PHI is
ever committed or logged.

Runtime is bounded so the default suite stays fast. A heavier example budget is
opted into via the ``HYPOTHESIS_PROFILE=fuzz-nightly`` environment variable,
which the scheduled CI job sets.
"""
