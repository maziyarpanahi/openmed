# Synthetic de-identification evidence pack

Run the end-to-end fixture from the repository root.

```sh
.venv/bin/python -m pytest tests/unit/eval/test_evidence_pack_fixture.py -q
```

The test rebuilds the pack and compares its digest and its signed bytes to
`pack.json`. The public test HMAC key, replay clock and key identifier are
in the test. G2 uses exact-span recall from the committed synthetic golden
benchmark. The other lanes use the labeled span, mask and fixed scores in
`control.json`. G4 rounds those scores to signed 8-bit precision and compares
recall. These lane values exercise the pack format and verifier. They provide
no measured quantized-model or clinical performance claim.

Both source files are bound by their UTF-8, LF-normalized digests as synthetic inputs. Use site-controlled
keys, a current clock and independently measured reports for release review.
