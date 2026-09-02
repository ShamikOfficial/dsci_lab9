# RAG Textbook Q&A

[![License: MIT](https://img.shields.io/badge/License-MIT-blue?style=flat-square)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python&logoColor=white)]()

> Retrieval-augmented generation chatbot that answers questions from Applied Data Science textbook PDFs using OpenAI or local GGUF models.

---

## Problem

Students and practitioners need quick, accurate answers from large textbook PDFs without manually searching hundreds of pages. A generic chatbot hallucinates; a RAG pipeline grounds answers in source material.

## Solution

CLI chatbot that ingests PDFs, chunks and embeds text into a FAISS index, and answers questions via OpenAI GPT (default) or local LlamaCpp models with MiniLM embeddings.

## Tech Stack

`Python` · `LangChain` · `FAISS` · `OpenAI` · `LlamaCpp` · `MiniLM`

---

## Setup

1. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   ```
   - Windows: `.\.venv\Scripts\activate`
   - macOS/Linux: `source .venv/bin/activate`

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Copy `.env.example` to `.env` and set variables as needed. Do not commit `.env`.

### Using OpenAI

- Set `OPENAI_API_KEY=your-key` in `.env`. This is the default.

### Using local models

- Install Python bindings (included in requirements):
  ```bash
  pip install llama-cpp-python
  ```
- Put GGUF (or `.bin`) model files in the `models/` folder. Each file is listed by its **filename** (e.g. `llama-2-7b-chat.Q4_K_M.gguf`).

## Environment variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENAI_API_KEY` | For OpenAI | — | OpenAI API key |
| `ADS_LLM_BACKEND` | No | (prompt at run) | `openai` or a model filename from `models/` |
| `ADS_PDF_DIR` | No | `data/pdfs` | Folder containing PDF files |
| `ADS_MODELS_DIR` | No | `models` | Folder containing GGUF/.bin model files |
| `ADS_FAISS_INDEX_DIR` | No | `faiss_index` | FAISS index for OpenAI embeddings |
| `ADS_FAISS_INDEX_DIR_LLAMA` | No | `faiss_index_llama` | FAISS index for local models (MiniLM embeddings) |

## Run

```bash
python App_p1.py
```

- Choose backend: **1) OpenAI** (default) or **2) …** one option per file in `models/`.
- First run: reads PDFs, chunks text, builds embeddings, saves FAISS index.
- Later runs: loads existing index and starts Q&A loop. Type `exit` to quit.

## Project Structure

```
rag-textbook-qa/
├── App_p1.py           # RAG pipeline + CLI
├── web_app.py          # Optional web frontend
├── data/pdfs/          # Textbook PDFs
├── models/             # Local GGUF models
├── faiss_index/        # OpenAI embeddings index (git-ignored)
└── requirements.txt
```

## How It Works

1. **Backend** — OpenAI default; local options discovered from `models/` by filename
2. **PDFs** → `load_pdfs()` extracts text per page
3. **Chunks** → `CharacterTextSplitter` (size 500, overlap 80)
4. **Index** — Embeddings stored in FAISS (OpenAI or MiniLM)
5. **Q&A** — Top-k retrieval + LLM answer grounded in context

## License

MIT — see [LICENSE](LICENSE).

---

*Originally developed as USC DSCI 560 coursework — refactored for clarity and portfolio presentation.*
