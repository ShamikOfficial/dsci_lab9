# Lab 9: Q&A Chatbot (Backend)

CLI chatbot that answers questions from the ADS textbook PDFs using OpenAI embeddings, FAISS, and GPT.

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

3. Copy `.env.example` to `.env` and set your OpenAI API key (or create `.env` with `OPENAI_API_KEY=your-key`). Do not commit `.env`.

## Environment variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENAI_API_KEY` | Yes | — | OpenAI API key |
| `ADS_PDF_DIR` | No | `data/pdfs` | Folder containing PDF files |
| `ADS_FAISS_INDEX_DIR` | No | `faiss_index` | Folder where the FAISS index is saved/loaded |

## Run

```bash
python App_p1.py
```

- First run: reads PDFs from `data/pdfs`, chunks text (500 chars, 80 overlap), builds embeddings, and saves a FAISS index under `faiss_index`.
- Later runs: load the existing index and start the Q&A loop.
- Type a question and press Enter; type `exit` to quit.

## Project layout

- `App_p1.py` – PDF load, chunking, FAISS index build/load, conversation chain, CLI loop.
- `data/pdfs/` – Put your PDFs here (e.g. ADS cookbook, installation guides).
- `faiss_index/` – Created automatically; holds the vector index (ignored by git).

## How it works

1. **PDFs** → `load_pdfs()` reads all `.pdf` files and extracts text per page.
2. **Chunks** → `make_chunks()` splits text with LangChain `CharacterTextSplitter` (size 500, overlap 80).
3. **Index** → Chunks are embedded with OpenAI and stored in a FAISS index (build once, then load from disk).
4. **Q&A** → Each question is embedded, top-k chunks are retrieved from FAISS, and GPT answers using only that context.
