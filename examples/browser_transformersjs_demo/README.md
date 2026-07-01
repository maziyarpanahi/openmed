# Browser Transformers.js Demo

Static browser demo for running token classification with Transformers.js.
It uses `@huggingface/transformers@4.2.0` and the public
`Xenova/bert-base-NER` ONNX model from the Hugging Face Hub.

## Run

From the repository root:

```bash
python3 -m http.server 8788 --directory examples/browser_transformersjs_demo
```

Open <http://127.0.0.1:8788>.

The first run downloads the JavaScript package from jsDelivr and model files
from Hugging Face. After that, browser caching is enabled. The text you enter
is not sent to a server by this demo; inference and deterministic regex matching
run in the page.

## Notes

- Samples are synthetic and safe to commit.
- The generic NER model detects people, organizations, locations, and
  miscellaneous entities. Local browser regexes add obvious synthetic PII
  matches for email, phone, dates, DOB, and MRN values.
- This demo is intentionally static: no Python server, no OpenMed package
  install, and no build step.
- To use an OpenMed model in the browser, export a compatible ONNX bundle with
  the workflow in `docs/export-transformersjs.md`, then replace the model id in
  `app.js` with the published or locally served Transformers.js artifact.
