# OpenMed v1.4.1

OpenMed v1.4.1 is a focused hotfix release for local model loading and offline deployments.

This release fixes a regression where existing filesystem paths could be treated like Hugging Face model names, which
caused local model directories to pick up the default org prefix or trigger unwanted Hub access. The result is a cleaner
local-first path for air-gapped, mirrored, and pre-downloaded model workflows.

## Highlights

- Fixed local model directory resolution in `ModelLoader` so existing paths are treated as local models before any org
  prefixing logic runs.
- Forced `local_files_only=True` for local-path config, tokenizer, model, pipeline, and max-length probing calls so
  offline environments stay fully local.
- Added `model_id` as a public alias for `model_name` in `openmed.analyze_text()`, including support for local
  directory paths.
- Updated README, docs, website, and Apple demo version surfaces for the `1.4.1` hotfix.

## Upgrade Notes

- The package version is now `1.4.1`.
- Swift package install snippets now reference `from: "1.4.1"`.
- Docker examples now tag images as `openmed:1.4.1`.

## Validation

- Release preflight checks pass for `v1.4.1`.
- Targeted unit coverage for the local-model-loading hotfix passes in the repo virtualenv.
- Package build validation completes successfully before tagging.
