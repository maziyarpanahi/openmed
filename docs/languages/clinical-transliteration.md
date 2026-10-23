# Clinical transliteration provenance

`openmed.clinical.languages.transliteration_graph` keeps caller-supplied local
transliteration alternatives without choosing one as clinically equivalent.
It bundles no restricted terminology, model, remote provider, or automatic
transliteration engine. Callers retain their spelling lookup locally.

Create `TransliterationNode.from_text(term, language=..., script=...)` to retain
an exact SHA-256 term identity without storing spelling in graph output. Nodes
bind language and script, so identical spellings in different scripts remain
separate. Supply `TransliterationRule` objects with source/target nodes, a
controlled rule-source code, evidence digest, confidence hint, and explicit
reversibility. A rule marked reversible may be traversed backward; a lossy rule
cannot. The declaration is caller evidence, not proof of linguistic correctness.

`TransliterationGraph.candidates(source, language="en", script="Latin")` returns
all simple routes within bounded depth. Routes to the same target retain distinct
provenance. Cycles terminate. Results order by descending product of declared
confidence hints, then deterministic metadata. Products are ranking hints, not
calibrated probabilities. Reviewers must resolve contradictory alternatives.

Use each path's `to_dict()` for fingerprints and controlled metadata only. No term
spellings enter exports or errors. Plain hashes are not anonymization guarantees;
do not publicly release graph artifacts derived from private patient identifiers.
Graph size is capped at 10,000 rules; depth at 8, path count at 1,000 and traversal
steps at 100,000. Exceeding a budget raises a controlled error rather than silently
returning an incomplete candidate set. No network calls or normalization alter
caller terminology. User-supplied vocabulary stays under its original license.
