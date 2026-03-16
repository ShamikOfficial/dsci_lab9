# Lab 9: Q&A Chatbot (Backend)

CLI chatbot that answers questions from the ADS textbook PDFs. **Default is OpenAI** (GPT + OpenAI embeddings). You can also use **local models**: put any GGUF (or `.bin`) file in the `models/` folder and it appears as an option by its **filename** (LlamaCpp + MiniLM embeddings).

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
- Put GGUF (or `.bin`) model files in the `models/` folder. Each file is listed by its **filename** (e.g. `llama-2-7b-chat.Q4_K_M.gguf`). No env path needed.

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

- You are prompted to choose backend: **1) OpenAI** (default) or **2) …** one option per file in `models/` (by filename). Set `ADS_LLM_BACKEND=openai` or `ADS_LLM_BACKEND=your-model.gguf` to skip the prompt.
- First run for that backend: reads PDFs from `data/pdfs`, chunks text, builds embeddings (OpenAI or MiniLM for local), and saves a FAISS index.
- Later runs: load the existing index for the chosen backend and start the Q&A loop.
- Type a question and press Enter; type `exit` to quit.

## Project layout

- `App_p1.py` – Backend-agnostic pipeline: PDF load, chunking, FAISS build/load, conversation chain, CLI.
- `data/pdfs/` – Put your PDFs here.
- `models/` – Put GGUF (or `.bin`) model files here; each file is offered as an option by its filename.
- `faiss_index/` – FAISS index for OpenAI embeddings (git-ignored).
- `faiss_index_llama/` – FAISS index for local models / MiniLM embeddings (git-ignored).

## How it works

1. **Backend** – Default is OpenAI. Local options are discovered from `models/` by filename (no separate logic per model type).
2. **PDFs** → `load_pdfs()` reads all `.pdf` files and extracts text per page.
3. **Chunks** → `make_chunks()` splits text with `CharacterTextSplitter` (size 500, overlap 80).
4. **Index** – Chunks are embedded (OpenAI for `openai`, MiniLM for any local file) and stored in a FAISS index (one index for OpenAI, one for all local models).
5. **Q&A** – Each question is embedded, top-k chunks are retrieved, and the selected LLM (GPT or LlamaCpp) answers using only that context.
