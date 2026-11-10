# Prompt-injection guard

OpenMed screens untrusted agent and MCP tool input with a deterministic,
local lexical guard before a registered MCP handler runs. It looks for four
reviewable cue families: instruction overrides, tool-name spoofing, delimiter
breakouts, and data-exfiltration requests.

The default policy is `strict`. A flagged MCP call returns a structured,
PHI-safe error containing only `pattern_id`, original-text `start`/`end`
offsets, and `severity`; the handler is not invoked. Findings and errors do
not contain the submitted text, and the server does not log the input.

For controlled integrations, `allow` mode can be selected with
`create_mcp_server(injection_guard_mode="allow")` or the
`OPENMED_MCP_INJECTION_GUARD_MODE=allow` environment variable. This does not
pass the matched instruction through: the guard replaces every flagged span
with an inert `[OPENMED_QUARANTINED_PROMPT_INJECTION:...]` marker before
dispatch. Any other value is rejected; strict mode remains the safe default.

## Bypass-resistance note

The scanner applies Unicode NFKC normalization, case folding, and removal of
Unicode format characters before matching, while mapping findings back to
original Python string offsets. Patterns also tolerate ordinary whitespace
variation. This is a deterministic defense-in-depth control, not a semantic
proof that a document is harmless: novel wording, multilingual attacks, and
application-specific encodings can evade a lexical list. Keep strict mode for
untrusted callers, extend the synthetic corpus when a bypass is found, and do
not treat allow mode as model-level adversarial robustness.

The guard is local-first and offline. It performs no network lookup, stores no
input text, and emits no raw PHI or raw injection payload in findings.
