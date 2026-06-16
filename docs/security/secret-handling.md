# Secret Handling Policy

## What Gets Scanned

OpenMed uses [gitleaks](https://github.com/gitleaks/gitleaks) to scan every commit and PR for accidentally committed credentials. The scanner covers:

- Hugging Face tokens (`hf_*`)
- PyPI tokens (`.pypirc` format)
- Generic API keys and high-entropy strings

## Local Setup

After cloning the repo, run once:

```bash
pip install pre-commit
pre-commit install
