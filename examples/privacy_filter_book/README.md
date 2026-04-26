# Privacy Filter Book Demo

Interactive web app comparing OpenAI's privacy-filter MLX artifact with the CPU Transformers path in a book-style redaction demo.

```bash
uvicorn examples.privacy_filter_book.app:app --reload --port 8765
```

Open <http://127.0.0.1:8765>.

By default the app tries cached local model files first and falls back to a deterministic local detector so the UI opens immediately. To allow first-run Hugging Face downloads for the live 8-bit MLX and CPU models, use the in-app "Allow Downloads" control or start the server with:

```bash
OPENMED_PRIVACY_FILTER_DOWNLOAD=1 uvicorn examples.privacy_filter_book.app:app --reload --port 8765
```

Model labels:

- MLX: `OpenMed/privacy-filter-mlx-8bit`
- CPU: `openai/privacy-filter`

