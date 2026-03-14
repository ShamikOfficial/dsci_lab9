# Lab 9: Q&A Chatbot (Backend)

CLI chatbot that answers questions from the ADS textbook PDFs. You can use **OpenAI** (GPT + OpenAI embeddings) or **Llama 2** running locally (LlamaCpp + MiniLM embeddings).

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

- Set `OPENAI_API_KEY=your-key` in `.env`.

### Using Llama 2 locally

- Install Python bindings (included in requirements):
  ```bash
  pip install llama-cpp-python
  ```
- Download a Llama 2 GGML model (e.g. [TheBloke/LLaMa-7B-GGML](https://huggingface.co/TheBloke/LLaMa-7B-GGML/blob/main/llama-7b.ggmlv3.q4_1.bin)) and place the `.bin` file in the `models` folder.
- Default path: `models/llama-7b.ggmlv3.q4_1.bin`. Override with `ADS_LLAMA_MODEL_PATH` in `.env` if you put the file elsewhere.

## Environment variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENAI_API_KEY` | For OpenAI backend | — | OpenAI API key |
| `ADS_LLM_BACKEND` | No | (prompt at run) | `openai` or `llama` |
| `ADS_PDF_DIR` | No | `data/pdfs` | Folder containing PDF files |
| `ADS_FAISS_INDEX_DIR` | No | `faiss_index` | FAISS index dir for OpenAI embeddings |
| `ADS_FAISS_INDEX_DIR_LLAMA` | No | `faiss_index_llama` | FAISS index dir for MiniLM (Llama backend) |
| `ADS_LLAMA_MODEL_PATH` | No | `models/llama-7b.ggmlv3.q4_1.bin` | Path to GGML/GGUF Llama model file |

## Run

```bash
python App_p1.py
```

- You will be prompted to choose backend: **1) OpenAI** or **2) Llama (Open-Source)** (or set `ADS_LLM_BACKEND=openai` / `ADS_LLM_BACKEND=llama` to skip the prompt).
- First run for that backend: reads PDFs from `data/pdfs`, chunks text, builds embeddings (OpenAI or MiniLM for Llama), and saves a FAISS index.
- Later runs: load the existing index for the chosen backend and start the Q&A loop.
- Type a question and press Enter; type `exit` to quit.

## Project layout

- `App_p1.py` – Backend-agnostic pipeline: PDF load, chunking, FAISS build/load, conversation chain, CLI.
- `data/pdfs/` – Put your PDFs here.
- `models/` – Put downloaded Llama GGML/GGUF model files here (e.g. `llama-7b.ggmlv3.q4_1.bin`).
- `faiss_index/` – FAISS index for OpenAI embeddings (git-ignored).
- `faiss_index_llama/` – FAISS index for MiniLM embeddings when using Llama backend (git-ignored).

## How it works

1. **Backend** – At startup you choose `openai` or `llama`. New backends can be added in `get_embeddings()`, `get_llm()`, and `get_faiss_index_dir()`.
2. **PDFs** → `load_pdfs()` reads all `.pdf` files and extracts text per page.
3. **Chunks** → `make_chunks()` splits text with `CharacterTextSplitter` (size 500, overlap 80).
4. **Index** – Chunks are embedded (OpenAI for `openai`, MiniLM for `llama`) and stored in a FAISS index (one index per backend).
5. **Q&A** – Each question is embedded, top-k chunks are retrieved, and the selected LLM (GPT or LlamaCpp) answers using only that context.
